# -*- coding: utf-8 -*-
# engine/ellipsoid.py
#
# ELLIPSOID STAGE + SHAKING  (coarse start of a candidate)
#
# Each molecule is an ellipsoid (its principal axes and extents) carrying a
# few charge sites that reproduce its electrostatic potential. The candidate
# is shaken in a gently pressed "container" (the periodic cell under
# pressure): overdamped Langevin dynamics with decreasing temperature, so
# molecules can turn and slide out of poor arrangements before the atomistic
# minimisation (lattice.minimise) fits the real shapes together.
#
#   contact   σ(û) = s / sqrt(ûᵀ (G_A + G_B)⁻¹ û),  G = Σ a_k² e_k e_kᵀ
#             (Berne-Pechukas overlap of two ellipsoids, the basis of the
#             Gay-Berne model); s fitted
#   energy    ε0 (R_eff / R̄) [x¹² - 2x⁶],  x = σ0 / (r - σ + σ0)
#             Derjaguin approximation: the attraction of two curved bodies
#             scales with the effective radius of curvature at the contact,
#             R_eff = R_A R_B / (R_A + R_B), R = K^(-1/2) from the Gauss
#             curvature of the ellipsoid where its normal is ±û:
#             K = h(û)⁴ / (abc)², h(û) = sqrt(ûᵀ G û) — flat faces in
#             contact (large R) bind more strongly than edges
#   charges   ESP-fitted sites (coarse._esp_charges), smeared Coulomb
#   shaking   Δp = -μ ∂H/∂p + sqrt(2 μ kT) ξ for centres, rotations and cell;
#             rotational mobility ∝ inverse inertia tensor, so a long
#             molecule turns more easily about its long axis
#
# Parameters (s, ε0, σ0) are fitted once per molecule and conformer to
# atomistic MMFF dimer energies, including overlapping dimers.

import math
import numpy as np
import torch
from dataclasses import replace
from functools import lru_cache

from lattice import (_RigidModel, _rodrigues, _reduce_and_wrap, plane_spacing,
                     MIN_SPACING, SKIN, COMPACT_PRESSURE, image_range)
from coarse import _dimer_samples, _esp_charges, _kmeans, E_CAP

N_CHARGES = 6
GAP_CUT = 6.0            # Å: ellipsoid surface gap beyond which van der Waals is off
COUL_CUT = 10.0          # Å: molecule-centre cutoff for the charge sites
COUL_ON = 8.0
CHARGE_WIDTH = 0.6
KT_RANGE = (4.0, 0.1)    # kJ/mol: shaking temperature, start → end
SEGMENTS = 10            # pair list rebuilt per segment
STEPS = 25               # Langevin steps per segment
MOBILITY = 2e-3          # Å² per kJ/mol per step for molecule centres


def _contact(u, G):
    """σ(û) = 1 / sqrt(ûᵀ G⁻¹ û) for G = G_A + G_B (batched)."""
    return 1.0 / torch.sqrt(torch.einsum('pi,pij,pj->p', u, torch.linalg.inv(G), u))


def _curvature_radius(u, G, abc):
    """K^(-1/2) of the ellipsoid G at the surface point whose normal is u."""
    h2 = torch.einsum('pi,pij,pj->p', u, G, u)          # h(u)² (support function²)
    return abc / h2


def _pair_energy(r_vec, GA, GB, p):
    r = torch.sqrt((r_vec * r_vec).sum(-1) + 1e-12)
    u = r_vec / r[:, None]
    sig = p['s'] * _contact(u, GA + GB)
    s0 = p['sig0']
    x = s0 / torch.clamp(r - sig + s0, min=0.15 * s0)
    RA = _curvature_radius(u, GA, p['abc'])
    RB = _curvature_radius(u, GB, p['abc'])
    eps = p['eps0'] * (RA * RB / (RA + RB)) / p['R_mean']
    return eps * (x ** 12 - 2 * x ** 6), r - sig


@lru_cache(maxsize=None)
def ellipsoid_model(material: str, conformer: int = 0):
    from forcefield import get_ff_molecule
    ffm = get_ff_molecule(material)
    X = ffm.conformer(conformer)[0]
    X = X - X.mean(0)
    rad = np.diag(ffm.pa) / 2.0
    lam, V = np.linalg.eigh(X.T @ X / len(X))
    axes = np.sqrt(5.0 * np.maximum(lam, 0.0)) + rad.mean()      # semi-axes (Å)
    # axis directions as fixed linear combinations of the atoms (rotate with
    # the molecule); the third from the cross product (planar molecules)
    W = np.linalg.pinv(X.T) @ V[:, 1:]                             # (n, 2): two largest
    W = W.T
    rng = np.random.default_rng(5)
    lq, cq = _kmeans(X, np.abs(ffm.charges) + 0.05, N_CHARGES, rng)
    A_q = np.zeros((len(cq), len(X)))
    for j in range(len(cq)):
        A_q[j, lq == j] = 1.0 / (lq == j).sum()
    qs = _esp_charges(X, np.asarray(ffm.charges), A_q @ X, rng)

    G0 = torch.as_tensor((V * axes ** 2) @ V.T)
    samples = _dimer_samples(X, ffm, rng)
    Qs = torch.as_tensor(np.array([q for q, _, _ in samples]))
    ts = torch.as_tensor(np.array([t for _, t, _ in samples]))
    E_ref = torch.as_tensor([e for _, _, e in samples])
    GB = Qs @ G0 @ Qs.transpose(1, 2)
    GA = G0.expand_as(GB)
    # 1. scale s from geometry: the distance at which the atoms first touch
    #    (0.85 × contact radii) along random directions, vs the BP contact
    from scipy.spatial.transform import Rotation
    ratios = []
    for _ in range(150):
        Q = Rotation.random(random_state=rng.integers(1 << 30)).as_matrix()
        u = rng.normal(size=3)
        u /= np.linalg.norm(u)
        B = X @ Q.T
        lim = 0.85 * (rad[:, None] + rad[None])
        for d in np.arange(2 * np.linalg.norm(X, axis=1).max() + 6.0, 0.0, -0.1):
            if (np.linalg.norm(X[:, None] - (B + d * u)[None], axis=2) - lim).min() < 0:
                break
        G = G0.numpy() + Q @ G0.numpy() @ Q.T
        ratios.append(d * math.sqrt(u @ np.linalg.inv(G) @ u))
    s_geom = float(np.median(ratios))
    abc = float(np.prod(axes))
    uu = ts / ts.norm(dim=1, keepdim=True)
    RA, RB = _curvature_radius(uu, GA, abc), _curvature_radius(uu, GB, abc)
    R_mean = float((RA * RB / (RA + RB)).mean())
    w = torch.exp(-torch.clamp(E_ref, min=-30.0, max=15.0) / 15.0)
    # 2. energy parameters, bounded: s within ±25 % of s_geom, σ0 0.5-4 Å,
    #    ε0 0.5-100 kJ/mol
    theta = torch.zeros(3, requires_grad=True)
    span = lambda t, lo, hi: lo + (hi - lo) * torch.sigmoid(t)

    def params(t):
        return {'s': s_geom * span(t[0], 0.75, 1.25), 'eps0': torch.exp(span(t[1], math.log(0.5), math.log(100.0))),
                'sig0': span(t[2], 0.5, 4.0), 'abc': abc, 'R_mean': R_mean}

    opt = torch.optim.LBFGS([theta], max_iter=300, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        pred, _ = _pair_energy(ts, GA, GB, params(theta))
        loss = (w * (torch.clamp(pred, max=E_CAP) - E_ref) ** 2).sum() / w.sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        pred, _ = _pair_energy(ts, GA, GB, params(theta))
        rmse = float(torch.sqrt((w * (torch.clamp(pred, max=E_CAP) - E_ref) ** 2).sum() / w.sum()))
        corr = float(np.corrcoef(torch.clamp(pred, max=E_CAP).numpy(), E_ref.numpy())[0, 1])
    p = {k: float(v) if torch.is_tensor(v) else v for k, v in params(theta.detach()).items()}
    return {'axes': axes, 'V': V, 'W': W, 'A_q': A_q, 'q': qs, 'params': p,
            'fit_rmse': rmse, 'fit_corr': corr, 'n_atoms': len(X)}


class _EllipsoidModel:
    """Energy per cell of the ellipsoid crystal; variables as lattice._RigidModel."""

    def __init__(self, x, model, cut):
        from forcefield import COULOMB
        self.base = _RigidModel(x, (np.zeros(0, int), np.zeros(0, int), np.zeros((0, 3))))
        self.m = model
        self.idx = torch.as_tensor(np.array([np.flatnonzero(x.mol == k)[np.argsort(x.site[x.mol == k])]
                                             for k in range(x.z)]))
        # molecule pairs (a, b, image) with centres within cut
        from scipy.spatial import cKDTree
        shifts = image_range(x.cell, cut)
        imgs = (x.com[None] + (shifts @ x.cell)[:, None]).reshape(-1, 3)
        coo = cKDTree(x.com).sparse_distance_matrix(cKDTree(imgs), cut, output_type='coo_matrix')
        a, kk = coo.row, coo.col
        b, s = kk % x.z, kk // x.z
        keep = ~((a == b) & np.all(shifts[s] == 0, axis=1))
        self.a, self.b = torch.as_tensor(a[keep]), torch.as_tensor(b[keep])
        self.n = torch.as_tensor(shifts[s[keep]].astype(float))
        self.W = torch.as_tensor(model['W'])
        self.ax2 = torch.as_tensor(model['axes'] ** 2)
        self.Aq = torch.as_tensor(model['A_q'])
        q = model['q']
        self.qq = torch.as_tensor(COULOMB * q[:, None] * q[None, :])
        self.p = model['params']

    def geometry(self, eps, dcom, rot):
        return self.base.geometry(eps, dcom, rot)

    def frames(self, off):
        offs = off[self.idx]                                   # (Z, n, 3)
        e = torch.einsum('kn,znl->zkl', self.W, offs)          # (Z, 2, 3): axes 2, 3
        e = e / torch.linalg.norm(e, dim=-1, keepdim=True)
        e1 = torch.cross(e[:, 0], e[:, 1], dim=-1)
        E = torch.stack([e1, e[:, 0], e[:, 1]], dim=1)          # rows = axes, eigh order
        G = torch.einsum('k,zki,zkj->zij', self.ax2, E, E)
        sites = torch.einsum('sn,znl->zsl', self.Aq, offs)      # (Z, K, 3)
        return G, sites

    def __call__(self, eps, dcom, rot):
        cell, com, off = self.base.geometry(eps, dcom, rot)
        G, sites = self.frames(off)
        shift = self.n @ cell
        r_vec = com[self.b] + shift - com[self.a]
        e_v, gap = _pair_energy(r_vec, G[self.a], G[self.b], self.p)
        e_v = torch.where(gap < GAP_CUT, e_v, torch.zeros_like(e_v))
        rc = torch.sqrt((r_vec * r_vec).sum(-1) + 1e-12)
        t = torch.clamp((rc - COUL_ON) / (COUL_CUT - COUL_ON), 0.0, 1.0)
        sw = 1.0 - t * t * (3.0 - 2.0 * t)
        d = (sites[self.b] + (com[self.b] + shift)[:, None])[:, None, :, :] - \
            (sites[self.a] + com[self.a][:, None])[:, :, None, :]           # (P, K, K, 3)
        r = torch.sqrt((d * d).sum(-1) + 1e-12)
        e_c = (self.qq * torch.erf(r / CHARGE_WIDTH) / r).sum((1, 2)) * sw
        return (e_v + e_c).sum() / 2.0


def _segment(x, model, pressure, kT, rng):
    cut = max(COUL_CUT, 2 * float(np.sqrt(model['axes'].max() ** 2 * 2)) + GAP_CUT) + SKIN
    m = _EllipsoidModel(x, model, cut)
    budget = SKIN / 3.0
    s_max = budget / cut
    r_mol = max(float(np.linalg.norm(x.offset, axis=1).max()), 0.5)
    t_max = budget / r_mol
    L = float(np.cbrt(x.volume))
    # rotational mobility ∝ inverse inertia tensor (lab frame), normalised
    # so its mean equals MOBILITY / r_mol²
    from ase.data import atomic_masses, atomic_numbers
    mass = np.array([atomic_masses[atomic_numbers[s]] for s in x.sym])
    Ls = []
    for k in range(x.z):
        sel = x.mol == k
        o, mk = x.offset[sel], mass[sel]
        I = (mk[:, None, None] * ((o * o).sum(1)[:, None, None] * np.eye(3) - o[:, :, None] * o[:, None, :])).sum(0)
        Minv = np.linalg.inv(I)
        Minv *= (MOBILITY / r_mol ** 2) / np.trace(Minv) * 3
        Ls.append(np.linalg.cholesky(Minv))
    Lrot = torch.as_tensor(np.array(Ls))                        # (Z, 3, 3)
    mu_t, mu_e = MOBILITY, MOBILITY / (L * L)
    eps_r = torch.zeros(6, requires_grad=True)
    dcom_r = torch.zeros(x.z, 3, requires_grad=True)
    rot_r = torch.zeros(x.z, 3, requires_grad=True)
    gen = torch.Generator().manual_seed(int(rng.integers(1 << 31)))

    def squash():
        return (s_max * torch.tanh(eps_r / s_max), budget * torch.tanh(dcom_r / budget),
                t_max * torch.tanh(rot_r / t_max))

    for _ in range(STEPS):
        v = squash()
        h = m(*v) + pressure * torch.abs(torch.linalg.det(m.geometry(*v)[0]))
        g_e, g_c, g_r = torch.autograd.grad(h, [eps_r, dcom_r, rot_r])
        with torch.no_grad():
            noise = lambda shape: torch.randn(shape, generator=gen, dtype=torch.float64)
            eps_r -= mu_e * torch.clamp(g_e, -1e4, 1e4) - math.sqrt(2 * mu_e * kT) * noise(6)
            dcom_r -= mu_t * torch.clamp(g_c, -1e3, 1e3) - math.sqrt(2 * mu_t * kT) * noise((x.z, 3))
            # anisotropic rotation: Δ = -M g + L ξ sqrt(2kT), M = L Lᵀ
            Mg = torch.einsum('zij,zkj,zk->zi', Lrot, Lrot, torch.clamp(g_r, -1e3, 1e3))
            rot_r -= Mg - math.sqrt(2 * kT) * torch.einsum('zij,zj->zi', Lrot, noise((x.z, 3)))
    with torch.no_grad():
        v = squash()
        cell, com, off = m.geometry(*v)
        e = float(m(*v)) / x.z
    y = replace(x, cell=cell.numpy().copy(), com=com.numpy().copy(), offset=off.numpy().copy())
    return _reduce_and_wrap(y), e


def shake(x, rng, pressure=COMPACT_PRESSURE):
    """Shaken ellipsoid packing of an atomistic MMFF Crystal: (crystal or None,
    ellipsoid energy per molecule). The result is marked meta['prepacked']
    (lattice.minimise then skips its own compaction)."""
    model = ellipsoid_model(x.ff, x.meta.get('conformer', 0))
    v0 = x.volume
    e = float('nan')
    for kT in np.geomspace(*KT_RANGE, SEGMENTS):
        x, e = _segment(x, model, pressure, float(kT), rng)
        if not np.isfinite(e) or plane_spacing(x.cell) < MIN_SPACING or x.volume > 6 * v0:
            return None, e
    return replace(x, meta={**x.meta, 'prepacked': True}), e


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    for key in sys.argv[1:]:
        m = ellipsoid_model(key)
        p = m['params']
        print(f'{key}: semi-axes {np.round(m["axes"], 2).tolist()} Å, s {p["s"]:.2f}, ε0 {p["eps0"]:.1f} kJ/mol, '
              f'σ0 {p["sig0"]:.2f} Å; dimer fit rmse {m["fit_rmse"]:.1f} kJ/mol, r = {m["fit_corr"]:.2f}')
