# -*- coding: utf-8 -*-
# engine/lattice.py
#
# UNIT-CELL CRYSTAL + LATTICE-ENERGY MINIMISATION
#
# A crystal is one triclinic unit cell (rows of `cell` = a, b, c) holding
# Z rigid molecules, repeated by periodic images. This replaces the
# 18 Å cubic simulation box: periodicity is exact, the cell shape is free,
# and a candidate needs only Z molecules instead of ~30.
#
# Energy: the same pair physics as physics.pair_energy_array (contact
# 12-6 LJ, screened Coulomb, isotropic H-bond, magnetic), written in
# torch so gradients come from autodiff. Coulomb and magnetic terms are
# tapered to zero between 7 and 8 Å so the energy is smooth for L-BFGS.
#
# Minimiser: L-BFGS over cell strain (6) + molecule translation (3 per
# molecule) + molecule rotation vector (3 per molecule), at P = 0, in P1
# (symmetry is not enforced, so a structure may lower its symmetry if
# that lowers the energy). The pair list is rebuilt between outer rounds
# and the cell is Minkowski-reduced so images stay cheap to enumerate.

import math
import numpy as np
import torch
from dataclasses import dataclass, field, replace
from typing import List, Optional
from ase.geometry import minkowski_reduce
from scipy.spatial import cKDTree

from physics import CUTOFF, LJ_RC, HB_CONTACT, E_CAP, _MAGNETIC_MOMENTS

torch.set_default_dtype(torch.float64)

SKIN = 2.0          # Å added to the cutoff when building pair lists
TAPER_ON = 7.0      # Å — Coulomb/magnetic taper start


@dataclass
class Crystal:
    """One unit cell of rigid molecules (cartesian, Å)."""
    cell: np.ndarray            # (3,3) rows a, b, c
    com: np.ndarray             # (Z,3) molecule centres
    offset: np.ndarray          # (N,3) atom position - its molecule centre
    mol: np.ndarray             # (N,) molecule index of each atom
    sym: List[str]              # (N,) element symbols
    radii: np.ndarray           # (N,)
    charge: np.ndarray          # (N,)
    donor: np.ndarray           # (N,) bool
    acceptor: np.ndarray        # (N,) bool
    mass: float = 0.0           # g/mol per molecule
    space_group: str = 'P1'     # group it was generated in
    energy: float = 0.0         # lattice energy per molecule (after minimise)
    meta: dict = field(default_factory=dict)
    site: Optional[np.ndarray] = None  # (N,) atom index within its molecule
    ff: Optional[str] = None    # forcefield.py model key; None = toy physics

    @property
    def energy_unit(self) -> str:
        return 'kJ/mol' if self.ff else 'model units'

    @property
    def z(self) -> int:
        return len(self.com)

    @property
    def positions(self) -> np.ndarray:
        return self.com[self.mol] + self.offset

    @property
    def volume(self) -> float:
        return float(abs(np.linalg.det(self.cell)))

    @property
    def density(self) -> float:
        """g/cm³ (needs mass per molecule)."""
        return self.z * self.mass / (0.6022 * self.volume) if self.mass else 0.0

    def cell_params(self) -> tuple:
        a, b, c = self.cell
        la, lb, lc = (np.linalg.norm(v) for v in (a, b, c))
        ang = lambda u, v: math.degrees(math.acos(np.clip(
            u @ v / (np.linalg.norm(u) * np.linalg.norm(v)), -1, 1)))
        return la, lb, lc, ang(b, c), ang(a, c), ang(a, b)

    def to_ase(self):
        """ASE Atoms of the real atoms (massless 'M' charge sites dropped)."""
        from ase import Atoms
        keep = [k for k, s in enumerate(self.sym) if s != 'M']
        return Atoms(symbols=[self.sym[k] for k in keep],
                     positions=self.positions[keep],
                     cell=self.cell, pbc=True)


def cell_from_params(a, b, c, alpha, beta, gamma) -> np.ndarray:
    """Rows a, b, c from lengths (Å) and angles (degrees)."""
    al, be, ga = (math.radians(x) for x in (alpha, beta, gamma))
    ax = np.array([a, 0.0, 0.0])
    bx = np.array([b * math.cos(ga), b * math.sin(ga), 0.0])
    cx = c * math.cos(be)
    cy = c * (math.cos(al) - math.cos(be) * math.cos(ga)) / math.sin(ga)
    cz2 = c * c - cx * cx - cy * cy
    if cz2 <= 0:
        raise ValueError('invalid cell angles')
    return np.array([ax, bx, [cx, cy, math.sqrt(cz2)]])


# ─────────────────────────────────────────────────────────────
# PAIR LIST  (numpy, rebuilt between minimiser rounds)
# ─────────────────────────────────────────────────────────────

def image_range(cell: np.ndarray, cut: float) -> np.ndarray:
    """Integer image shifts n (M,3) covering every vector shorter than cut."""
    vol = abs(np.linalg.det(cell))
    k = []
    for d in range(3):
        u, v = cell[(d + 1) % 3], cell[(d + 2) % 3]
        spacing = vol / np.linalg.norm(np.cross(u, v))
        k.append(int(math.ceil(cut / spacing)))
    g = [np.arange(-kk, kk + 1) for kk in k]
    return np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)


def build_pairs(x: Crystal, cut: float):
    """(i, j, n) with |r_j + n·cell - r_i| < cut, excluding atoms of the
    same molecule in the same image. Every pair appears in both orders."""
    # k-d tree over all image atoms: memory scales with the number of
    # pairs found, not atoms² × images (which ran to GBs for Z=8 cells).
    pos = x.positions
    n = len(pos)
    shifts = image_range(x.cell, cut)
    images = (pos[None, :, :] + (shifts @ x.cell)[:, None, :]).reshape(-1, 3)
    coo = cKDTree(pos).sparse_distance_matrix(
        cKDTree(images), cut, output_type='coo_matrix')
    i, k = coo.row, coo.col
    j, m = k % n, k // n
    keep = ~((x.mol[i] == x.mol[j]) & np.all(shifts[m] == 0, axis=1))
    keep &= ~((i == j) & np.all(shifts[m] == 0, axis=1))
    return i[keep], j[keep], shifts[m[keep]].astype(float)


# ─────────────────────────────────────────────────────────────
# ENERGY  (torch)
# ─────────────────────────────────────────────────────────────

def _taper(r):
    t = torch.clamp((r - TAPER_ON) / (CUTOFF - TAPER_ON), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def pair_energy_torch(r, d0, qq, hb, mm):
    sigma = d0 / 2.0 ** (1.0 / 6.0)
    rc = torch.clamp(LJ_RC * sigma, max=CUTOFF)
    x = (sigma / r) ** 6
    xc = (sigma / rc) ** 6
    lj = torch.where(r < rc, 4.0 * (x * x - x) - 4.0 * (xc * xc - xc),
                     torch.zeros_like(r))
    lj = E_CAP * torch.tanh(lj / E_CAP)                # smooth repulsion cap
    taper = _taper(r)
    coul = 0.2 * qq / (4.0 * r) * taper
    ratio = 2.0 / r
    e_hb = torch.clamp(0.8 * (5 * ratio ** 12 - 6 * ratio ** 10), -1.5, 0.5)
    e_hb = torch.where(hb & (r >= 1.2) & (r <= 3.5), e_hb, torch.zeros_like(r))
    mag = -0.05 * mm / (r * r) * taper
    e = lj + coul + e_hb + mag
    return torch.where(r < CUTOFF, e, torch.zeros_like(r))


def _rodrigues(w):
    """Rotation matrices (Z,3,3) from rotation vectors (Z,3)."""
    th = torch.sqrt((w * w).sum(-1) + 1e-30)[:, None, None]
    k = w / th[:, :, 0]
    K = torch.zeros(w.shape[0], 3, 3)
    K[:, 0, 1], K[:, 0, 2] = -k[:, 2], k[:, 1]
    K[:, 1, 0], K[:, 1, 2] = k[:, 2], -k[:, 0]
    K[:, 2, 0], K[:, 2, 1] = -k[:, 1], k[:, 0]
    I = torch.eye(3).expand_as(K)
    return I + torch.sin(th) * K + (1 - torch.cos(th)) * (K @ K)


class _RigidModel:
    """Geometry of a rigid-molecule crystal as a function of
    (strain, translations, rotations) relative to a fixed reference
    geometry and pair list."""

    def __init__(self, x: Crystal, pairs):
        i, j, n = pairs
        self.i = torch.as_tensor(i)
        self.j = torch.as_tensor(j)
        self.n = torch.as_tensor(n)
        self.cell = torch.as_tensor(x.cell)
        self.com = torch.as_tensor(x.com)
        self.offset = torch.as_tensor(x.offset)
        self.mol = torch.as_tensor(x.mol)

    def geometry(self, eps, dcom, rot):
        E = torch.stack([
            torch.stack([eps[0], eps[5] / 2, eps[4] / 2]),
            torch.stack([eps[5] / 2, eps[1], eps[3] / 2]),
            torch.stack([eps[4] / 2, eps[3] / 2, eps[2]])])
        S = torch.eye(3) + E
        cell = self.cell @ S
        com = self.com @ S + dcom
        R = _rodrigues(rot)
        off = torch.einsum('nk,nlk->nl', self.offset, R[self.mol])
        return cell, com, off


class _EnergyModel(_RigidModel):
    """Toy pair physics (physics.pair_energy_array in torch)."""

    def __init__(self, x: Crystal, pairs):
        super().__init__(x, pairs)
        i, j, _ = pairs
        hb = (x.donor[i] & x.acceptor[j]) | (x.donor[j] & x.acceptor[i])
        self.hb = torch.as_tensor(hb)
        self.d0 = torch.as_tensor(np.where(hb, HB_CONTACT,
                                           x.radii[i] + x.radii[j]))
        self.qq = torch.as_tensor(x.charge[i] * x.charge[j]
                                  * ((np.abs(x.charge[i]) >= 1e-3)
                                     & (np.abs(x.charge[j]) >= 1e-3)))
        mom = np.array([_MAGNETIC_MOMENTS.get(s, 0.0) for s in x.sym])
        self.mm = torch.as_tensor(mom[i] * mom[j])

    def __call__(self, eps, dcom, rot):
        cell, com, off = self.geometry(eps, dcom, rot)
        pos = com[self.mol] + off
        d = pos[self.j] + self.n @ cell - pos[self.i]
        r = torch.sqrt((d * d).sum(-1))
        e = pair_energy_torch(r, self.d0, self.qq, self.hb, self.mm)
        return torch.clamp(e, min=-3.0).sum() / 2.0     # each pair counted twice


def model_cutoff(x: Crystal) -> float:
    if x.ff:
        from forcefield import get_ff_molecule
        level = x.meta.get('level', 'full')
        cut = get_ff_molecule(x.ff).cutoff(level)
        from forcefield import LEVELS
        cut = max(cut, LEVELS.get(level, {}).get('dsf_cut', 0.0))   # DSF Coulomb reaches further
        shape = x.meta.get('shape', 1.0)
        if shape < 1.0 and len(x.offset):       # morphed atoms sit closer than the real ones
            cut += 2.0 * (1.0 - shape) * float(np.linalg.norm(x.offset, axis=1).max())
        return cut
    return CUTOFF


def make_model(x: Crystal, pairs):
    """Physical force field when x.ff is set, else the toy pair model."""
    if x.ff:
        from forcefield import FFModel
        return FFModel(_RigidModel(x, pairs), x, pairs)
    return _EnergyModel(x, pairs)


def lattice_energy(x: Crystal) -> float:
    """Lattice energy per molecule of the current geometry."""
    model = make_model(x, build_pairs(x, model_cutoff(x)))
    z = torch.zeros
    with torch.no_grad():
        e = model(z(6), z(x.z, 3), z(x.z, 3))
    return float(e) / x.z


SYM_TOL = 0.05       # Å: a molecule is an exact symmetry image of molecule 0 within this


def symmetry_maps(x: Crystal):
    """[(A, t)] with fractional f_m = f_0 Aᵀ + t and offsets o_m = o_0 L⁻¹AᵀL
    for every molecule m (A integer, det ±1), read off the geometry — or
    None if the molecules are not all images of molecule 0."""
    if x.z < 2 or x.site is None or not x.ff:
        return None
    L = np.asarray(x.cell, float)
    Linv = np.linalg.inv(L)
    idx = [np.flatnonzero(x.mol == m)[np.argsort(x.site[x.mol == m])] for m in range(x.z)]
    if any(len(ix) != len(idx[0]) for ix in idx):
        return None
    o0 = x.offset[idx[0]]
    f0 = x.com[0] @ Linv
    maps = []
    for m in range(x.z):
        om = x.offset[idx[m]]
        U, _, Vt = np.linalg.svd(o0.T @ om)
        found = None
        # orthogonal Procrustes; a planar molecule leaves the sign of the
        # normal open, so both handednesses are tried
        for D in (np.eye(3), np.diag([1.0, 1.0, -1.0])):
            M = U @ D @ Vt
            A = np.rint((L @ M @ Linv).T)
            if abs(abs(np.linalg.det(A)) - 1) > 1e-6:
                continue
            if np.abs(o0 @ (Linv @ A.T @ L) - om).max() < SYM_TOL:
                found = A
                break
        if found is None:
            return None
        maps.append((found, x.com[m] @ Linv - f0 @ found.T))
    return maps


class _SymGeometry(_RigidModel):
    """Rigid crystal with all molecules generated from molecule 0 by its
    symmetry operations: 6 strain + 3 translation + 3 rotation variables,
    and pairs only from molecule 0 (pair_weight = Z) — Z× fewer pairs."""

    def __init__(self, x: Crystal, pairs, maps):
        super().__init__(x, pairs)
        self.A = torch.as_tensor(np.array([a for a, _ in maps]))
        self.t = torch.as_tensor(np.array([t for _, t in maps]))
        idx0 = np.flatnonzero(x.mol == 0)[np.argsort(x.site[x.mol == 0])]
        self.o0 = torch.as_tensor(x.offset[idx0])
        self.site = torch.as_tensor(x.site)
        self.pair_weight = float(x.z)

    def geometry(self, eps, dcom, rot):
        E = torch.stack([
            torch.stack([eps[0], eps[5] / 2, eps[4] / 2]),
            torch.stack([eps[5] / 2, eps[1], eps[3] / 2]),
            torch.stack([eps[4] / 2, eps[3] / 2, eps[2]])])
        S = torch.eye(3) + E
        cell = self.cell @ S
        Linv = torch.linalg.inv(cell)
        f0 = (self.com[0] @ S + dcom[0]) @ Linv
        com = (torch.einsum('k,zlk->zl', f0, self.A) + self.t) @ cell
        o0 = self.o0 @ _rodrigues(rot[:1])[0].T
        # Cartesian image operations; a strain that breaks the symmetry
        # would shear them, so only their orthogonal (polar) part is used
        M = torch.einsum('ij,zkj,kl->zil', Linv, self.A, cell)
        for _ in range(3):         # Newton polar iteration (SVD has no gradient for
            M = 0.5 * (M + torch.linalg.inv(M).transpose(1, 2))   # equal singular values)
        off = torch.einsum('ak,zkl->zal', o0, M)[self.mol, self.site]
        return cell, com, off


def _reduce_and_wrap(x: Crystal) -> Crystal:
    rcell, _ = minkowski_reduce(x.cell)
    rcell = np.asarray(rcell, float)
    frac = x.com @ np.linalg.inv(rcell)
    frac -= np.floor(frac)
    return replace(x, cell=rcell, com=frac @ rcell)


GPA = 0.60221        # kJ/mol per Å³ at 1 GPa
COMPACT_PRESSURE = 0.17 * GPA   # first phase: squeeze loose candidates
MIN_SPACING = 1.0    # Å — lattice-plane spacing below which a cell has collapsed
MORPH_STAGES = (0.0, 0.35, 0.7)   # shape λ of the sphere → molecule stages
MORPH_ROUNDS = 4     # L-BFGS rounds per morph stage


def plane_spacing(cell: np.ndarray) -> float:
    vol = abs(np.linalg.det(cell))
    return min(vol / np.linalg.norm(np.cross(cell[(d + 1) % 3], cell[(d + 2) % 3]))
               for d in range(3))


def _round(x: Crystal, pressure: float, max_iter: int):
    """
    One L-BFGS round on a fixed pair list. Returns (crystal, lattice energy
    per molecule at its final geometry, on the round's own pair list).
    A trust region keeps it valid: strain, translations and rotations are
    squashed (tanh) so that no pair distance changes by more than ~3 × budget ≈ 2 Å per round. Unbounded,
    one long line-search step could bring atoms from beyond the list into
    contact (missing repulsion) and the cell collapsed unopposed. 2 Å is
    safe: a pair must travel ~10 Å from the list edge to reach contact,
    and misses near the edge only touch the switched-off vdW tail.
    """
    budget = SKIN / 3.0                                # each of 3 motions
    s_max = budget / (model_cutoff(x) + SKIN)          # strain scales |r_ij|
    r_mol = np.linalg.norm(x.offset, axis=1).max() if len(x.offset) else 0.0
    t_max = budget / max(r_mol, 0.5)
    pairs = build_pairs(x, model_cutoff(x) + SKIN)
    maps = symmetry_maps(x) if x.meta.get('symmetric') else None
    if maps is not None:
        from forcefield import get_ff_molecule, FFModel
        if get_ff_molecule(x.ff).form not in ('mmff', 'lj'):
            maps = None                    # many-body metals need every atom's density
    if maps is not None:
        i, j, n = pairs
        keep = x.mol[i] == 0
        pairs = (i[keep], j[keep], n[keep])
        model = FFModel(_SymGeometry(x, pairs, maps), x, pairs)
    else:
        model = make_model(x, pairs)
    import fastmin
    if maps is None and fastmin.applicable(model):
        # compiled L-BFGS round (fastmin.py): same steps, no PyTorch overhead
        cell, com, off, e = fastmin.run_round(x, model, pressure, max_iter, s_max, budget, t_max)
        return _reduce_and_wrap(replace(x, cell=cell, com=com, offset=off)), e
    n_free = 1 if maps is not None else x.z
    eps_r = torch.zeros(6, requires_grad=True)
    dcom_r = torch.zeros(n_free, 3, requires_grad=True)
    rot_r = torch.zeros(n_free, 3, requires_grad=True)

    def squash():
        return (s_max * torch.tanh(eps_r / s_max),
                budget * torch.tanh(dcom_r / budget),
                t_max * torch.tanh(rot_r / t_max))

    opt = torch.optim.LBFGS([eps_r, dcom_r, rot_r], lr=1.0, max_iter=max_iter,
                            tolerance_grad=1e-6, tolerance_change=1e-10,
                            history_size=20, line_search_fn='strong_wolfe')

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
        e = float(model(*v)) / x.z
        cell, com, off = model.geometry(*v)
    return _reduce_and_wrap(replace(x, cell=cell.numpy().copy(),
                                    com=com.numpy().copy(),
                                    offset=off.numpy().copy())), e


def minimise(x: Crystal, max_rounds: int = 40, max_iter: int = 40,
             tol: float = 1e-6, max_volume_factor: float = 6.0) -> Crystal:
    """
    Lattice-energy minimisation; the final energy is at P = 0. Returns a
    new Crystal with .energy = lattice energy per molecule (NaN if the
    cell collapsed or exploded) and meta['converged'].

    With a physical force field (x.ff) a first phase minimises the
    enthalpy E + P·V at COMPACT_PRESSURE: random candidates start loose,
    and beyond the cutoff nothing pulls separated molecules back, so
    without it many drift apart instead of packing.
    """
    v0 = x.volume
    x = _reduce_and_wrap(x)
    compact = x.ff and not x.meta.get('prepacked')      # coarse.py already packed it
    phases = ([(COMPACT_PRESSURE, 1.0)] if compact else []) + [(0.0, 1.0)]
    if x.ff and x.meta.get('morph'):
        # sphere → real shape (forcefield.FFModel): a few rounds per stage
        phases = [(COMPACT_PRESSURE, lam) for lam in MORPH_STAGES] + phases
    converged = failed = False
    rounds = 0
    for pressure, shape in phases:
        h_prev = None
        x = replace(x, meta={**x.meta, 'shape': shape})
        stage_end = rounds + MORPH_ROUNDS if shape < 1.0 else max_rounds
        while rounds < min(stage_end, max_rounds):
            rounds += 1
            x, e_round = _round(x, pressure, max_iter)
            if (not np.isfinite(x.volume) or x.volume > max_volume_factor * v0
                    or plane_spacing(x.cell) < MIN_SPACING):
                failed = True
                break
            # the round's own energy: a fresh lattice_energy() here rebuilt the
            # pair list and model every round (convergence only needs differences)
            h = e_round + pressure * x.volume / x.z
            if h_prev is not None and abs(h - h_prev) <= tol * max(1.0, abs(h)):
                converged = pressure == 0.0
                break
            h_prev = h
        if failed:
            break
    e = float('nan') if failed else lattice_energy(x)
    return replace(x, energy=e, meta={**x.meta, 'morph': False,     # only the first pass morphs
                                      'converged': converged,
                                      'rounds': rounds,
                                      'volume_ratio': x.volume / v0})


def primitive_cell(x: Crystal, symprec: float = 0.1) -> np.ndarray:
    """Primitive cell (spglib) — a candidate generated with Z=4 can relax
    into a Z=2 structure whose own unit cell is half as big; comparing
    cells (not AMD) needs this. Falls back to the cell itself."""
    try:
        import spglib
        from ase.data import atomic_numbers
        nums = [atomic_numbers.get(s, 0) for s in x.sym]
        frac = (x.positions @ np.linalg.inv(x.cell)) % 1.0
        prim = spglib.standardize_cell((x.cell, frac, nums), to_primitive=True,
                                       no_idealize=True, symprec=symprec)
        if prim is not None:
            return np.asarray(prim[0], float)
    except Exception:
        pass
    return x.cell


def space_group_of(x: Crystal, symprec: float = 0.1) -> str:
    """Space group of the minimised structure (spglib), or '?'."""
    try:
        import spglib
        from ase.data import atomic_numbers
        nums = [atomic_numbers.get(s, 0) for s in x.sym]
        frac = x.positions @ np.linalg.inv(x.cell)
        sg = spglib.get_spacegroup((x.cell, frac % 1.0, nums), symprec=symprec)
        return sg or '?'
    except Exception:
        return '?'
