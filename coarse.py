# -*- coding: utf-8 -*-
# engine/coarse.py
#
# MULTI-RESOLUTION PACKING  (sphere → few beads → atoms)
#
# A packing search moves molecules a long way at first and only fine-tunes
# at the end. The far moves do not need atoms: a molecule is first a single
# sphere carrying a few charge sites (its electrostatic "surface forces"),
# then a handful of beads that rough out its shape, and only the last stage
# sees every atom. The cost per molecule pair falls from n_atoms² (10 000
# for a 100-atom molecule) to ~40 at level 0.
#
# Each level, built once per molecule and conformer and cached:
#   vdW beads    k-means groups of atoms (mass weighted); radius from the
#                group's extent + its atoms' MMFF radii, well depth from its
#                atom count; two scale factors fitted to the atomistic
#                dimer energies (MMFF, crystal calibration) of ~600 random
#                near-contact dimers
#   charge sites k-means points whose charges reproduce the molecule's
#                electrostatic potential on a shell around it (ESP fit,
#                total charge fixed)
# Sites are linear combinations of the atoms (fixed weights), so they move
# and turn exactly with the molecule; after each coarse round the atomistic
# Crystal is updated with the same rigid motions.
#
# coarse_minimise(x) runs the levels on an atomistic lattice.Crystal and
# returns it pre-packed for lattice.minimise.

import math
import numpy as np
import torch
from dataclasses import replace
from functools import lru_cache

from lattice import (build_pairs, _RigidModel, _rodrigues, _reduce_and_wrap,
                     plane_spacing, MIN_SPACING, SKIN, COMPACT_PRESSURE)

LEVELS = ((1, 6), (4, 8), (10, 12))   # (vdW beads, charge sites) per level
COUL_CUT = 9.0          # Å: molecule-centre cutoff for the site charges (switched)
COUL_ON = 7.0
VDW_CUT = 5.0           # Å: bead-surface gap cutoff for van der Waals
CHARGE_WIDTH = 0.6      # Å: Gaussian smearing of site charges
N_FIT = 600             # dimers for the van der Waals fit
E_CAP = 80.0            # kJ/mol: repulsion fitted up to here (overlapping dimers)
ROUNDS = 6              # max L-BFGS rounds per level and phase (stops when converged)
TOL = 1e-4              # relative energy change that ends a phase


def _kmeans(X, w, k, rng, iters=60):
    """Weighted k-means → (labels, centres)."""
    k = min(k, len(X))
    c = X[rng.choice(len(X), k, replace=False, p=w / w.sum())]
    lab = np.zeros(len(X), int)
    for _ in range(iters):
        lab = np.argmin(((X[:, None] - c[None]) ** 2).sum(-1), axis=1)
        for j in range(k):
            m = lab == j
            if m.any():
                c[j] = (w[m, None] * X[m]).sum(0) / w[m].sum()
    used = np.unique(lab)
    return np.searchsorted(used, lab), c[used]


def _esp_charges(X, q, sites, rng):
    """Site charges reproducing the ESP of atom charges q on a shell."""
    from scipy.spatial import cKDTree
    r_mol = np.linalg.norm(X, axis=1).max()
    pts = rng.normal(size=(6000, 3))
    pts *= (rng.uniform(1.0, r_mol + 4.0, 6000) / np.linalg.norm(pts, axis=1))[:, None]
    d, _ = cKDTree(X).query(pts)
    pts = pts[(d > 2.2) & (d < 4.0)]                       # outside the vdW surface
    V = (q[None] / np.linalg.norm(pts[:, None] - X[None], axis=2)).sum(1)
    A = 1.0 / np.linalg.norm(pts[:, None] - sites[None], axis=2)
    k = len(sites)                                         # Lagrange: Σ q = total
    M = np.zeros((k + 1, k + 1))
    M[:k, :k] = A.T @ A + 1e-4 * np.eye(k)
    M[:k, k] = M[k, :k] = 1.0
    rhs = np.concatenate([A.T @ V, [q.sum()]])
    return np.linalg.solve(M, rhs)[:k]


def _dimer_samples(X, ffm, rng):
    """Random near-contact dimers: (relative rotation, shift, atomistic vdW E)."""
    from puzzle import _pair_tables
    from scipy.spatial.transform import Rotation
    R, e, _ = (t.numpy() for t in _pair_tables(ffm))
    out = []
    while len(out) < N_FIT:
        Q = Rotation.random(random_state=rng.integers(1 << 30)).as_matrix()
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        B = X @ Q.T
        # walk in from far until atoms first touch, then sample around contact
        lo = 0.0
        for d in np.arange(2 * np.linalg.norm(X, axis=1).max() + 4.0, 0.0, -0.25):
            if np.linalg.norm(X[:, None] - (B + d * u)[None], axis=2).min() < 2.6:
                lo = d
                break
        # a quarter of the samples overlap: the coarse bodies must learn that
        # molecules cannot pass through each other (fitted up to E_CAP)
        dist = lo + (rng.uniform(-1.5, -0.3) if rng.random() < 0.25 else rng.uniform(-0.3, 2.5))
        r = np.linalg.norm(X[:, None] - (B + dist * u)[None], axis=2)
        E = (e * (1.07 * R / (r + 0.07 * R)) ** 7 * (1.12 * R ** 7 / (r ** 7 + 0.12 * R ** 7) - 2.0)).sum()
        out.append((Q, dist * u, min(float(E), E_CAP)))
    return out


def _vdw_energy(pa, pb, Ra, Rb, ea, eb, sR, se):
    """Bead-bead buffered 14-7 (pa, pb: (n,3) bead positions)."""
    Rab = sR * (Ra[:, None] + Rb[None])
    eab = se * torch.sqrt(ea[:, None] * eb[None])
    r = torch.sqrt(((pa[:, None] - pb[None]) ** 2).sum(-1) + 1e-12)
    return (eab * (1.07 * Rab / (r + 0.07 * Rab)) ** 7 *
            (1.12 * Rab ** 7 / (r ** 7 + 0.12 * Rab ** 7) - 2.0)).sum()


@lru_cache(maxsize=None)
def coarse_levels(material: str, conformer: int = 0):
    """[(level dict) for LEVELS]: weights A (sites × atoms) and parameters."""
    from forcefield import get_ff_molecule
    ffm = get_ff_molecule(material)
    X = ffm.conformer(conformer)[0]
    X = X - X.mean(0)
    from ase.data import atomic_masses, atomic_numbers
    mass = np.array([atomic_masses[atomic_numbers[s]] for s in ffm.symbols])
    rad = np.diag(ffm.pa) / 2.0                 # MMFF R*_ii / 2: atom contact radius
    eps_at = np.diag(ffm.pb)
    rng = np.random.default_rng(5)
    samples = _dimer_samples(X, ffm, rng)
    levels = []
    for n_vdw, n_q in LEVELS:
        lab, cen = _kmeans(X, mass, n_vdw, rng)
        A_v = np.zeros((len(cen), len(X)))
        for j in range(len(cen)):
            A_v[j, lab == j] = mass[lab == j] / mass[lab == j].sum()
        Rb = np.array([np.linalg.norm(X[lab == j] - cen[j], axis=1).max(initial=0.0)
                       + rad[lab == j].mean() for j in range(len(cen))])
        eb = np.array([eps_at[lab == j].sum() for j in range(len(cen))])
        lq, cq = _kmeans(X, np.abs(ffm.charges) + 0.05, n_q, rng)
        A_q = np.zeros((len(cq), len(X)))
        for j in range(len(cq)):
            A_q[j, lq == j] = 1.0 / (lq == j).sum()
        sites = A_q @ X
        qs = _esp_charges(X, np.asarray(ffm.charges), sites, rng)
        # fit the two van der Waals scales to the atomistic dimers
        P = torch.as_tensor(A_v @ X)
        Rt, et = torch.as_tensor(Rb), torch.as_tensor(eb)
        E_ref = torch.as_tensor([s[2] for s in samples])
        w = torch.exp(-torch.clamp(E_ref, min=-30.0, max=15.0) / 15.0)   # contacts weigh most, overlaps still count
        partners = [(torch.as_tensor(P.numpy() @ Q.T + t)) for Q, t, _ in samples]
        lsR = torch.zeros((), requires_grad=True)
        lse = torch.zeros((), requires_grad=True)
        opt = torch.optim.LBFGS([lsR, lse], max_iter=200, line_search_fn='strong_wolfe')

        def closure():
            opt.zero_grad()
            pred = torch.stack([_vdw_energy(P, pb, Rt, Rt, et, et, torch.exp(lsR), torch.exp(lse))
                                for pb in partners])
            loss = (w * (torch.clamp(pred, max=E_CAP) - E_ref) ** 2).sum() / w.sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            pred = torch.stack([_vdw_energy(P, pb, Rt, Rt, et, et, torch.exp(lsR), torch.exp(lse))
                                for pb in partners])
            err = float(torch.sqrt((w * (torch.clamp(pred, max=E_CAP) - E_ref) ** 2).sum() / w.sum()))
        levels.append({'A_v': A_v, 'R': Rb, 'eps': eb, 'sR': float(torch.exp(lsR)),
                       'se': float(torch.exp(lse)), 'A_q': A_q, 'q': qs, 'fit_rmse': err})
    return levels


class _CoarseModel:
    """Energy (kJ/mol per cell) of the coarse sites; same (strain, translation,
    rotation) variables as lattice._RigidModel."""

    def __init__(self, xs, pairs, lev, n_v):
        from forcefield import COULOMB
        self.base = _RigidModel(xs, pairs)
        i, j, _ = pairs
        si, sj = xs.site[i], xs.site[j]
        is_v = xs.site < n_v                    # sites: vdW beads first, then charges
        vv = is_v[i] & is_v[j]
        qq = ~is_v[i] & ~is_v[j]
        self.vv, self.qq = torch.as_tensor(vv), torch.as_tensor(qq)
        R = np.concatenate([lev['R'], np.zeros(len(lev['q']))])
        e = np.concatenate([lev['eps'], np.zeros(len(lev['q']))])
        q = np.concatenate([np.zeros(len(lev['R'])), lev['q']])
        self.Rab = torch.as_tensor(lev['sR'] * (R[si] + R[sj]))
        self.eab = torch.as_tensor(lev['se'] * np.sqrt(e[si] * e[sj]))
        self.qij = torch.as_tensor(COULOMB * q[si] * q[sj])
        self.mi, self.mj = torch.as_tensor(xs.mol[i]), torch.as_tensor(xs.mol[j])

    def geometry(self, eps, dcom, rot):
        return self.base.geometry(eps, dcom, rot)

    def __call__(self, eps, dcom, rot):
        b = self.base
        cell, com, off = b.geometry(eps, dcom, rot)
        pos = com[b.mol] + off
        shift = b.n @ cell
        d = pos[b.j] + shift - pos[b.i]
        r = torch.sqrt((d * d).sum(-1) + 1e-12)
        R, e = self.Rab, self.eab
        gap = r - R
        sw_v = torch.clamp(1.0 - gap / VDW_CUT, 0.0, 1.0) ** 2
        e_v = e * (1.07 * R / (r + 0.07 * R)) ** 7 * (1.12 * R ** 7 / (r ** 7 + 0.12 * R ** 7) - 2.0)
        energy = torch.where(self.vv, e_v * sw_v, torch.zeros_like(r)).sum()
        # charges: switched on the molecule-centre distance (neutral groups)
        dc = com[self.mj] + shift - com[self.mi]
        rc = torch.sqrt((dc * dc).sum(-1) + 1e-12)
        t = torch.clamp((rc - COUL_ON) / (COUL_CUT - COUL_ON), 0.0, 1.0)
        sw_c = 1.0 - t * t * (3.0 - 2.0 * t)
        e_c = self.qij * torch.erf(r / CHARGE_WIDTH) / r * sw_c
        energy = energy + torch.where(self.qq, e_c, torch.zeros_like(r)).sum()
        return energy / 2.0


def _sites_crystal(x, lev):
    """Coarse-site Crystal (same cell, centres, molecules) of atomistic x."""
    n_at = len(lev['A_v'][0])
    A = np.vstack([lev['A_v'], lev['A_q']])
    off = np.concatenate([A @ x.offset[x.mol == m][np.argsort(x.site[x.mol == m])]
                          for m in range(x.z)])
    k = len(A)
    return replace(x, offset=off, mol=np.repeat(np.arange(x.z), k), site=np.tile(np.arange(k), x.z),
                   sym=['X'] * (k * x.z), radii=np.zeros(k * x.z), charge=np.zeros(k * x.z),
                   donor=np.zeros(k * x.z, bool), acceptor=np.zeros(k * x.z, bool)), len(lev['A_v'])


def _site_pairs(xs, k, mol_cut):
    """Site pairs (i, j, image) between molecules whose centres are within
    mol_cut — a molecule-level list: the cost scales with neighbouring
    molecules × sites², not with a site cutoff wide enough for the charges."""
    from lattice import image_range
    from scipy.spatial import cKDTree
    shifts = image_range(xs.cell, mol_cut)
    com = xs.com
    imgs = (com[None] + (shifts @ xs.cell)[:, None]).reshape(-1, 3)
    coo = cKDTree(com).sparse_distance_matrix(cKDTree(imgs), mol_cut, output_type='coo_matrix')
    a, kk = coo.row, coo.col
    b, m = kk % len(com), kk // len(com)
    keep = ~((a == b) & np.all(shifts[m] == 0, axis=1))
    a, b, m = a[keep], b[keep], m[keep]
    si, sj = np.meshgrid(np.arange(k), np.arange(k), indexing='ij')
    si, sj = si.ravel(), sj.ravel()
    i = (a[:, None] * k + si[None]).ravel()
    j = (b[:, None] * k + sj[None]).ravel()
    n = np.repeat(shifts[m], len(si), axis=0).astype(float)
    return i, j, n


def _round(x, lev, pressure, max_iter=40):
    xs, n_v = _sites_crystal(x, lev)
    k = len(lev['A_v']) + len(lev['A_q'])
    r_site = float(np.linalg.norm(xs.offset, axis=1).max())
    cut = max(COUL_CUT, VDW_CUT + 2 * float(lev['sR'] * lev['R'].max()) + 2 * r_site)
    pairs = _site_pairs(xs, k, cut + SKIN)
    model = _CoarseModel(xs, pairs, lev, n_v)
    budget = SKIN / 3.0
    s_max = budget / (cut + SKIN)
    t_max = budget / max(float(np.linalg.norm(x.offset, axis=1).max()), 0.5)
    eps_r = torch.zeros(6, requires_grad=True)
    dcom_r = torch.zeros(x.z, 3, requires_grad=True)
    rot_r = torch.zeros(x.z, 3, requires_grad=True)

    def squash():
        return (s_max * torch.tanh(eps_r / s_max), budget * torch.tanh(dcom_r / budget),
                t_max * torch.tanh(rot_r / t_max))

    opt = torch.optim.LBFGS([eps_r, dcom_r, rot_r], max_iter=max_iter, tolerance_grad=1e-6,
                            tolerance_change=1e-10, history_size=20, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        v = squash()
        e = model(*v)
        if pressure:
            e = e + pressure * torch.abs(torch.linalg.det(model.geometry(*v)[0]))
        e.backward()
        return e

    opt.step(closure)
    with torch.no_grad():
        v = squash()
        cell, com, _ = model.geometry(*v)
        R = _rodrigues(v[2])
        off = torch.einsum('nk,nlk->nl', torch.as_tensor(x.offset), R[torch.as_tensor(x.mol)])
        e = float(model(*v)) / x.z
    y = replace(x, cell=cell.numpy().copy(), com=com.numpy().copy(), offset=off.numpy().copy())
    return _reduce_and_wrap(y), e


def _anneal_segment(x, lev, pressure, kT, steps, rng, mobility):
    """Overdamped Langevin (Brownian) steps on a fixed pair list:
    Δp = -μ ∂H/∂p + sqrt(2 μ kT) ξ for every variable p (strain, centres,
    rotations), H = E + P·V. A shaken container of soft bodies that settles
    as kT is lowered (simulated annealing in the sphere picture)."""
    xs, n_v = _sites_crystal(x, lev)
    k = len(lev['A_v']) + len(lev['A_q'])
    r_site = float(np.linalg.norm(xs.offset, axis=1).max())
    cut = max(COUL_CUT, VDW_CUT + 2 * float(lev['sR'] * lev['R'].max()) + 2 * r_site)
    model = _CoarseModel(xs, _site_pairs(xs, k, cut + SKIN), lev, n_v)
    budget = SKIN / 3.0
    s_max = budget / (cut + SKIN)
    r_mol = max(float(np.linalg.norm(x.offset, axis=1).max()), 0.5)
    t_max = budget / r_mol
    L = float(np.cbrt(x.volume))
    mu = {'eps': mobility / (L * L), 'dcom': mobility, 'rot': mobility / (r_mol * r_mol)}
    eps_r = torch.zeros(6, requires_grad=True)
    dcom_r = torch.zeros(x.z, 3, requires_grad=True)
    rot_r = torch.zeros(x.z, 3, requires_grad=True)
    gen = torch.Generator().manual_seed(int(rng.integers(1 << 31)))

    def squash():
        return (s_max * torch.tanh(eps_r / s_max), budget * torch.tanh(dcom_r / budget),
                t_max * torch.tanh(rot_r / t_max))

    for _ in range(steps):
        v = squash()
        h = model(*v) + pressure * torch.abs(torch.linalg.det(model.geometry(*v)[0]))
        grads = torch.autograd.grad(h, [eps_r, dcom_r, rot_r])
        with torch.no_grad():
            for p, g, key in zip((eps_r, dcom_r, rot_r), grads, ('eps', 'dcom', 'rot')):
                m = mu[key]
                noise = torch.randn(p.shape, generator=gen, dtype=p.dtype)
                p -= m * torch.clamp(g, -1e3, 1e3) - math.sqrt(2 * m * kT) * noise
    with torch.no_grad():
        v = squash()
        cell, com, _ = model.geometry(*v)
        R = _rodrigues(v[2])
        off = torch.einsum('nk,nlk->nl', torch.as_tensor(x.offset), R[torch.as_tensor(x.mol)])
        e = float(model(*v)) / x.z
    y = replace(x, cell=cell.numpy().copy(), com=com.numpy().copy(), offset=off.numpy().copy())
    return _reduce_and_wrap(y), e


ANNEAL_KT = (5.0, 0.2)     # kJ/mol: start and end of the shaking (≈ 600 K → 25 K)
ANNEAL_SEGMENTS = 10       # pair list rebuilt per segment
ANNEAL_STEPS = 20          # Langevin steps per segment
MOBILITY = 2e-3            # Å² per (kJ/mol) per step for molecule centres


def shake(x, rng, level=0, pressure=COMPACT_PRESSURE):
    """Sphere-level Langevin annealing under a gentle confining pressure
    (the 'container'); returns (crystal or None, energy per molecule)."""
    lev = coarse_levels(x.ff, x.meta.get('conformer', 0))[level]
    v0 = x.volume
    e = float('nan')
    for kT in np.geomspace(*ANNEAL_KT, ANNEAL_SEGMENTS):
        x, e = _anneal_segment(x, lev, pressure, float(kT), ANNEAL_STEPS, rng, MOBILITY)
        if not np.isfinite(e) or plane_spacing(x.cell) < MIN_SPACING or x.volume > 6 * v0:
            return None, e
    return x, e


def coarse_minimise(x, levels=(0, 1, 2), rounds=ROUNDS):
    """Pre-pack an atomistic MMFF Crystal through the coarse levels: the
    first level compacts (as lattice.minimise does), every level then
    relaxes at P = 0 until converged. Returns (crystal, energies); the
    crystal is marked meta['prepacked'] so lattice.minimise skips its own
    compaction."""
    lev_all = coarse_levels(x.ff, x.meta.get('conformer', 0))
    energies = []
    v0 = x.volume
    for li in levels:
        lev = lev_all[li]
        for pressure in ((COMPACT_PRESSURE, 0.0) if li == levels[0] else (0.0,)):
            h_prev = None
            for _ in range(rounds):
                x, e = _round(x, lev, pressure)
                if (not np.isfinite(e) or plane_spacing(x.cell) < MIN_SPACING
                        or x.volume > 6 * v0):
                    return None, energies
                h = e + pressure * x.volume / x.z
                if h_prev is not None and abs(h - h_prev) <= TOL * max(1.0, abs(h)):
                    break
                h_prev = h
        energies.append(e)
    return replace(x, meta={**x.meta, 'prepacked': True}), energies


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    for key in sys.argv[1:]:
        for k, lev in enumerate(coarse_levels(key)):
            print(f'{key} level {k}: {len(lev["R"])} vdW beads, {len(lev["q"])} charge sites, '
                  f'scales R×{lev["sR"]:.2f} ε×{lev["se"]:.2f}, dimer fit rmse {lev["fit_rmse"]:.1f} kJ/mol')
