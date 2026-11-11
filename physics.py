# -*- coding: utf-8 -*-
# engine/physics.py  (v2.8 — performance-optimized)
#
# Physics engine — all inter-atomic and inter-molecular forces.
#
# Optimization notes (2026 perf pass):
#   * The scalar hot-path functions (lj_energy, coulomb_energy,
#     hbond_energy, magnetic_energy, pair_energy) now use the python
#     `math` module instead of numpy. Calling numpy on scalars carries
#     ~10-17x per-call dispatch overhead vs plain float math, and these
#     functions are called tens of thousands of times per MC run.
#   * assembly_energy_per_atom now has a fully vectorized fast path
#     (all pairwise distances at once) for large boxes.
#
# Crystal-stability pass (2026-11-09):
#   * Bulk energy is now a FULL sum over every pair within the cutoff
#     (cKDTree neighbour list with periodic boundaries). The old code
#     summed a strided subset of pairs and, in the MC loops, let each atom
#     feel only the first 25-40 atoms of the list instead of its real
#     neighbours — which melted even a perfect crystal into glass.
#   * LJ is 12-6 with its minimum at the contact distance r1+r2
#     (sigma = (r1+r2)/2^(1/6)), truncated + shifted at 2.5 sigma. The old
#     sigma=(r1+r2)/2 put the minimum at 56-71% of the contact distance,
#     and the 4-2 tail out to 8 Å made compression always lower the energy.
#   * Donor-H···acceptor pairs now get the H-bond term (pair_energy used
#     to give it to donor/non-acceptor pairs because of the XOR test).
#   * Atoms sharing a 'mol' id are one rigid molecule: intra-molecular
#     pairs are excluded and AtomSystem.mc moves molecules as rigid bodies.

import math
import numpy as np
from typing import Tuple
from scipy.spatial import cKDTree

CUTOFF = 8.0      # Å — pair energy = 0 beyond this
HARD_WALL = 0.5   # fraction of sigma below which = overlap penalty
LJ_RC = 2.5       # LJ truncated (and shifted to 0) at LJ_RC * sigma
HB_CONTACT = 2.0  # Å — LJ contact distance for donor-H···acceptor pairs
E_CAP = 1000.0    # cap on pair repulsion: overlaps stay steep but finite
_TWO_SIXTH = 2.0 ** (1.0 / 6.0)


def _clip(v, lo, hi):
    return lo if v < lo else (hi if v > hi else v)


def _rint(x):
    # nearest-integer, matches np.round for PBC minimum-image use
    return math.floor(x + 0.5) if x >= 0.0 else math.ceil(x - 0.5)


# ─────────────────────────────────────────────────────────────
# BASIC GEOMETRY
# ─────────────────────────────────────────────────────────────

def dist3(p1, p2) -> float:
    dx = p1[0]-p2[0]; dy = p1[1]-p2[1]; dz = p1[2]-p2[2]
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def dist3_pbc(p1, p2, box: float):
    """Distance with periodic boundary conditions.
    Returns (distance, displacement_vector as np.ndarray)."""
    dx = p2[0]-p1[0]; dy = p2[1]-p1[1]; dz = p2[2]-p1[2]
    dx -= box * _rint(dx/box)
    dy -= box * _rint(dy/box)
    dz -= box * _rint(dz/box)
    r = math.sqrt(dx*dx + dy*dy + dz*dz)
    return r, np.array((dx, dy, dz))


def dist_pbc_scalar(p1, p2, box: float) -> float:
    """PBC distance only — no displacement vector allocated. Hot path."""
    dx = p2[0]-p1[0]; dy = p2[1]-p1[1]; dz = p2[2]-p1[2]
    dx -= box * _rint(dx/box)
    dy -= box * _rint(dy/box)
    dz -= box * _rint(dz/box)
    return math.sqrt(dx*dx + dy*dy + dz*dz)


def dist_matrix_pbc(positions: np.ndarray, box: float) -> np.ndarray:
    """NxN distance matrix with PBC. positions: (N,3). Vectorized."""
    p = np.asarray(positions, dtype=float)
    d = p[:, None, :] - p[None, :, :]
    d -= box * np.round(d / box)
    return np.sqrt(np.einsum('ijk,ijk->ij', d, d))


# ─────────────────────────────────────────────────────────────
# LENNARD-JONES  (12-6 reduced to 4-2 for metals, speed)
# ─────────────────────────────────────────────────────────────

def lj_energy(r: float, sigma: float, epsilon: float = 1.0) -> float:
    if r < 0.01:
        return 5.0
    ratio = sigma / r
    if ratio > 2.0:
        return 5.0
    r2 = ratio*ratio
    lj = epsilon * (r2*r2 - r2)
    return _clip(lj, -2.0, 5.0)


def lj_energy_full(r: float, sigma: float, epsilon: float = 1.0) -> float:
    if r < 0.01:
        return 5.0
    ratio = sigma / r
    if ratio > 2.0:
        return 5.0
    r6 = ratio**6
    lj = 4*epsilon*(r6*r6 - r6)
    return _clip(lj, -2.0, 5.0)


def lj_contact(r: float, d0: float) -> float:
    """12-6 LJ (epsilon=1) with its minimum (-1) at the contact distance d0
    = sum of radii. Truncated and shifted to zero at LJ_RC*sigma."""
    sigma = d0 / _TWO_SIXTH
    rc = min(LJ_RC * sigma, CUTOFF)
    if r >= rc:
        return 0.0
    if r < sigma * HARD_WALL:
        return E_CAP
    x = (sigma / r) ** 6
    xc = (sigma / rc) ** 6
    return min(4.0*(x*x - x) - 4.0*(xc*xc - xc), E_CAP)


# ─────────────────────────────────────────────────────────────
# COULOMB
# ─────────────────────────────────────────────────────────────

def coulomb_energy(r: float, q1: float, q2: float,
                   dielectric: float = 4.0) -> float:
    if abs(q1) < 0.001 or abs(q2) < 0.001:
        return 0.0
    if r < 0.5:
        return 5.0 * (1.0 if q1*q2 > 0 else (-1.0 if q1*q2 < 0 else 0.0))
    return 0.2 * q1 * q2 / (dielectric * r)


# ─────────────────────────────────────────────────────────────
# HYDROGEN BOND  (directional, geometry-aware)
# ─────────────────────────────────────────────────────────────

def hbond_energy(r: float, donor: bool, acceptor: bool,
                 angle_cos: float = 1.0) -> float:
    if not (donor ^ acceptor):
        return 0.0
    if r < 1.2 or r > 3.5:
        return 0.0
    r_star = 2.0
    ratio = r_star / r
    e_dist = 0.8 * (5*ratio**12 - 6*ratio**10)
    af = angle_cos if angle_cos > 0.0 else 0.0
    angle_factor = af*af*af*af
    return _clip(e_dist * angle_factor, -1.5, 0.5)


def hbond_geometry(pos_donor, pos_H, pos_acceptor) -> float:
    v1 = pos_donor - pos_H
    v2 = pos_acceptor - pos_H
    n1 = math.sqrt(v1[0]*v1[0]+v1[1]*v1[1]+v1[2]*v1[2])
    n2 = math.sqrt(v2[0]*v2[0]+v2[1]*v2[1]+v2[2]*v2[2])
    if n1 < 0.01 or n2 < 0.01:
        return 0.0
    dot = v1[0]*v2[0]+v1[1]*v2[1]+v1[2]*v2[2]
    return -dot / (n1 * n2)


def molecule_hbond_energy(mol_a, mol_b,
                          use_directionality: bool = True) -> float:
    if not use_directionality:
        total = 0.0
        for a in mol_a.atoms:
            for b in mol_b.atoms:
                r = dist3(a.pos, b.pos)
                total += hbond_energy(r, a.hbond_donor, b.hbond_acceptor)
                total += hbond_energy(r, b.hbond_donor, a.hbond_acceptor)
        return total

    total = 0.0

    def find_parent(mol, h_idx):
        for bond in mol.bonds:
            if bond.i == h_idx and mol.atoms[bond.j].symbol != 'H':
                return bond.j
            if bond.j == h_idx and mol.atoms[bond.i].symbol != 'H':
                return bond.i
        return None

    for i, a in enumerate(mol_a.atoms):
        if not a.hbond_donor:
            continue
        parent_idx = find_parent(mol_a, i)
        if parent_idx is None:
            for b in mol_b.atoms:
                if not b.hbond_acceptor:
                    continue
                r = dist3(a.pos, b.pos)
                total += hbond_energy(r, True, False, 1.0)
            continue
        pos_D = mol_a.atoms[parent_idx].pos
        pos_H = a.pos
        for b in mol_b.atoms:
            if not b.hbond_acceptor:
                continue
            r = dist3(pos_H, b.pos)
            if r > 3.5:
                continue
            cos_angle = hbond_geometry(pos_D, pos_H, b.pos)
            total += hbond_energy(r, True, False, cos_angle)

    for i, b in enumerate(mol_b.atoms):
        if not b.hbond_donor:
            continue
        parent_idx = find_parent(mol_b, i)
        if parent_idx is None:
            for a in mol_a.atoms:
                if not a.hbond_acceptor:
                    continue
                r = dist3(b.pos, a.pos)
                total += hbond_energy(r, True, False, 1.0)
            continue
        pos_D = mol_b.atoms[parent_idx].pos
        pos_H = b.pos
        for a in mol_a.atoms:
            if not a.hbond_acceptor:
                continue
            r = dist3(pos_H, a.pos)
            if r > 3.5:
                continue
            cos_angle = hbond_geometry(pos_D, pos_H, a.pos)
            total += hbond_energy(r, True, False, cos_angle)

    return total


# ─────────────────────────────────────────────────────────────
# MAGNETIC
# ─────────────────────────────────────────────────────────────

_MAGNETIC_MOMENTS = {
    'Fe': 2.22, 'Co': 1.72, 'Ni': 0.60,
    'Mn': 3.00, 'Cr': 1.50, 'Gd': 7.00,
}

def magnetic_energy(r: float, sym1: str, sym2: str) -> float:
    m1 = _MAGNETIC_MOMENTS.get(sym1, 0.0)
    m2 = _MAGNETIC_MOMENTS.get(sym2, 0.0)
    if m1 == 0 or m2 == 0:
        return 0.0
    if r < 0.5 or r > CUTOFF:
        return 0.0
    return -0.05 * m1 * m2 / (r*r)


# ─────────────────────────────────────────────────────────────
# COMPOSITE PAIR ENERGY
# ─────────────────────────────────────────────────────────────

def pair_energy(r: float, r1: float, r2: float,
                q1: float = 0.0, q2: float = 0.0,
                hb_donor1: bool = False, hb_acc1: bool = False,
                hb_donor2: bool = False, hb_acc2: bool = False,
                sym1: str = 'X', sym2: str = 'X',
                use_full_lj: bool = False) -> float:
    """Scalar twin of pair_energy_array. use_full_lj is kept for call
    compatibility: the bulk potential is always the contact 12-6 form."""
    if r > CUTOFF:
        return 0.0

    hb = (hb_donor1 and hb_acc2) or (hb_donor2 and hb_acc1)
    total = lj_contact(r, HB_CONTACT if hb else r1 + r2)
    total += coulomb_energy(r, q1, q2)
    if hb:
        total += hbond_energy(r, True, False)
    total += magnetic_energy(r, sym1, sym2)
    return max(total, -3.0)


def pair_energy_array(r, r1, r2, q1, q2, hb, m1, m2) -> np.ndarray:
    """Vectorised pair_energy. hb: pair is donor-H···acceptor (isotropic
    H-bond term). m1, m2: magnetic moments."""
    rs = np.maximum(r, 1e-3)
    d0 = np.where(hb, HB_CONTACT, r1 + r2)
    sigma = d0 / _TWO_SIXTH
    rc = np.minimum(LJ_RC * sigma, CUTOFF)
    x = (sigma / rs) ** 6
    xc = (sigma / rc) ** 6
    e = np.where(rs < rc, 4.0*(x*x - x) - 4.0*(xc*xc - xc), 0.0)
    e = np.where(rs < sigma * HARD_WALL, E_CAP, np.minimum(e, E_CAP))

    qq = q1 * q2
    e += np.where((np.abs(q1) >= 0.001) & (np.abs(q2) >= 0.001),
                  0.2 * qq / (4.0 * rs), 0.0)

    ratio = 2.0 / rs
    e_hb = np.clip(0.8 * (5*ratio**12 - 6*ratio**10), -1.5, 0.5)
    e += np.where(hb & (rs >= 1.2) & (rs <= 3.5), e_hb, 0.0)

    mm = m1 * m2
    e += np.where((mm > 0) & (rs >= 0.5), -0.05 * mm / (rs*rs), 0.0)

    e = np.maximum(e, -3.0)
    return np.where(r <= CUTOFF, e, 0.0)


# ─────────────────────────────────────────────────────────────
# ATOM SYSTEM  (periodic box, neighbour lists, rigid-body MC)
# ─────────────────────────────────────────────────────────────

def wrap_positions(pos: np.ndarray, box: float) -> np.ndarray:
    """Wrap into [0, box) — strictly, as cKDTree(boxsize=...) requires."""
    p = pos - box * np.floor(pos / box)
    p[p >= box] = 0.0
    return p


def _small_rotation(rng, angle_sigma: float) -> np.ndarray:
    """Rotation about a random axis by a normal(0, angle_sigma) angle."""
    k = rng.normal(size=3)
    k /= np.linalg.norm(k)
    a = rng.normal(0.0, angle_sigma)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(a) * K + (1 - math.cos(a)) * (K @ K)


class AtomSystem:
    """
    Periodic cubic box of atoms with real neighbour lists.

    Atoms sharing a 'mol' id form one rigid molecule: their mutual pairs
    are excluded from the energy and mc() moves them together. Atoms
    without a 'mol' key are each their own body (metals).
    """

    def __init__(self, atoms: list, box: float, skin: float = 1.0):
        self.n = len(atoms)
        self.box = float(box)
        self.pos = wrap_positions(
            np.array([[a['x'], a['y'], a['z']] for a in atoms], float),
            self.box)
        # LJ-minimum radii (see molecules.lj_radius); bare 'r' = metallic
        self.radii = np.array([a.get('lj_r', a.get('r', 1.4)) for a in atoms],
                              float)
        self.charges = np.array([a.get('charge', 0.0) or 0.0 for a in atoms],
                                float)
        self.donor = np.array([bool(a.get('hbond_donor', False))
                               for a in atoms])
        self.acceptor = np.array([bool(a.get('hbond_acceptor', False))
                                  for a in atoms])
        self.moment = np.array([_MAGNETIC_MOMENTS.get(a.get('sym', 'X'), 0.0)
                                for a in atoms], float)

        lookup = {}
        self.mol = np.empty(self.n, int)
        for i, a in enumerate(atoms):
            key = a.get('mol')
            key = ('atom', i) if key is None else key
            self.mol[i] = lookup.setdefault(key, len(lookup))
        self.n_groups = len(lookup)
        order = np.argsort(self.mol, kind='stable')
        self.groups = np.split(order, np.cumsum(
            np.bincount(self.mol, minlength=self.n_groups))[:-1])
        self.first = np.array([g[0] for g in self.groups], int)

        self.skin = skin
        self.step = 0.10       # Å, adapted toward ~40% acceptance
        self.rot_step = 0.05   # rad

    # ── energy ────────────────────────────────────────────────
    def _pairs(self, pos, box, cut):
        """i<j pairs closer than cut, on different molecules."""
        if self.n < 2:
            return np.empty((0, 2), int)
        tree = cKDTree(wrap_positions(pos, box), boxsize=box)
        p = tree.query_pairs(min(cut, 0.4999 * box), output_type='ndarray')
        return p[self.mol[p[:, 0]] != self.mol[p[:, 1]]]

    def _pair_energy(self, i, j, pos, box):
        d = pos[i] - pos[j]
        d -= box * np.round(d / box)
        r = np.sqrt(np.einsum('ij,ij->i', d, d))
        hb = ((self.donor[i] & self.acceptor[j]) |
              (self.donor[j] & self.acceptor[i]))
        return pair_energy_array(r, self.radii[i], self.radii[j],
                                 self.charges[i], self.charges[j], hb,
                                 self.moment[i], self.moment[j])

    def energy(self, pos=None, box=None) -> float:
        """Total energy per atom (every pair within CUTOFF)."""
        pos = self.pos if pos is None else pos
        box = self.box if box is None else box
        p = self._pairs(pos, box, CUTOFF)
        if len(p) == 0:
            return 0.0
        return float(self._pair_energy(p[:, 0], p[:, 1], pos, box).sum()
                     / max(self.n, 1))

    def coms(self, pos=None, box=None) -> np.ndarray:
        """Centre of each molecule (unwrapped across the boundary)."""
        pos = self.pos if pos is None else pos
        box = self.box if box is None else box
        rel = pos - pos[self.first[self.mol]]
        rel -= box * np.round(rel / box)
        cnt = np.bincount(self.mol, minlength=self.n_groups)
        c = np.stack([np.bincount(self.mol, weights=rel[:, k],
                                  minlength=self.n_groups)
                      for k in range(3)], axis=1) / cnt[:, None]
        return pos[self.first] + c

    def to_atoms(self, atoms: list) -> list:
        p = wrap_positions(self.pos, self.box)
        return [{**a, 'x': float(p[i, 0]), 'y': float(p[i, 1]),
                 'z': float(p[i, 2])} for i, a in enumerate(atoms)]

    # ── Verlet list (per molecule) ────────────────────────────
    def _build_list(self, pos, box):
        p = self._pairs(pos, box, CUTOFF + self.skin)
        src = np.concatenate([p[:, 0], p[:, 1]])
        dst = np.concatenate([p[:, 1], p[:, 0]])
        g = self.mol[src]
        order = np.argsort(g, kind='stable')
        self._src, self._dst = src[order], dst[order]
        self._bounds = np.searchsorted(g[order], np.arange(self.n_groups + 1))
        self._ref = pos.copy()

    def _group_energy(self, g, pos, box) -> float:
        a, b = self._bounds[g], self._bounds[g + 1]
        if a == b:
            return 0.0
        return float(self._pair_energy(self._src[a:b], self._dst[a:b],
                                       pos, box).sum())

    # ── Monte Carlo ───────────────────────────────────────────
    def _trial_move(self, A, pos, box, rng):
        shift = rng.normal(0.0, self.step, 3)
        if len(A) == 1:
            return pos[A] + shift
        rel = pos[A] - pos[A[0]]
        rel -= box * np.round(rel / box)
        c = rel.mean(axis=0)
        R = _small_rotation(rng, self.rot_step)
        return pos[A[0]] + c + shift + (rel - c) @ R.T

    def _breathe(self, pos, box, T, rng, sigma=0.01):
        """Isotropic box move: molecule centres scale, molecules stay rigid."""
        s = math.exp(rng.normal(0.0, sigma))
        trial = pos + (s - 1.0) * self.coms(pos, box)[self.mol]
        dE = (self.energy(trial, box * s) - self.energy(pos, box)) * self.n
        if dE <= 0 or (T > 0 and rng.random() < math.exp(-dE / T)):
            pos[:] = trial
            box = box * s
            self._build_list(pos, box)
        return box

    def mc(self, T: float, n_sweeps: int, breathe: bool = False,
           rng=None) -> float:
        """
        Metropolis rigid-body MC at temperature T (energy units).
        One sweep = one trial move per molecule. Step sizes adapt toward
        ~40% acceptance. Updates self.pos / self.box; returns acceptance.
        """
        rng = rng or np.random.default_rng()
        pos, box = self.pos.copy(), self.box
        self._build_list(pos, box)
        half_skin2 = (0.5 * self.skin) ** 2
        accepted = 0
        for _ in range(n_sweeps):
            acc = 0
            for g in rng.integers(self.n_groups, size=self.n_groups):
                A = self.groups[g]
                old = pos[A].copy()
                e_old = self._group_energy(g, pos, box)
                pos[A] = self._trial_move(A, pos, box, rng)
                dE = self._group_energy(g, pos, box) - e_old
                if dE <= 0 or (T > 0 and rng.random() < math.exp(-dE / T)):
                    acc += 1
                    d = pos[A] - self._ref[A]
                    d -= box * np.round(d / box)
                    if np.einsum('ij,ij->i', d, d).max() > half_skin2:
                        self._build_list(pos, box)
                else:
                    pos[A] = old
            if breathe:
                box = self._breathe(pos, box, T, rng)
            rate = acc / self.n_groups
            if rate > 0.5:
                self.step = min(self.step * 1.15, 0.5)
                self.rot_step = min(self.rot_step * 1.15, 0.3)
            elif rate < 0.3:
                self.step = max(self.step * 0.85, 0.002)
                self.rot_step = max(self.rot_step * 0.85, 0.001)
            accepted += acc
        self.pos, self.box = wrap_positions(pos, box), box
        return accepted / max(n_sweeps * self.n_groups, 1)

    def relax(self, T: float, n_sweeps: int, breathe: bool = True,
              rng=None) -> float:
        """Anneal at T, then quench toward T=0. Returns energy per atom."""
        rng = rng or np.random.default_rng()
        self.mc(T, n_sweeps, breathe=breathe, rng=rng)
        for f in (0.3, 0.1, 0.03, 0.0):
            self.mc(T * f, max(2, n_sweeps // 4), breathe=breathe, rng=rng)
        return self.energy()


# ─────────────────────────────────────────────────────────────
# ASSEMBLY ENERGY  (whole box)
# ─────────────────────────────────────────────────────────────

def assembly_energy_per_atom(atoms_list: list, box: float,
                             max_pairs: int = 800,
                             use_full_lj: bool = False) -> float:
    """Full energy per atom: every inter-molecular pair within CUTOFF.
    max_pairs and use_full_lj are accepted for call compatibility only."""
    if len(atoms_list) < 2:
        return 0.0
    return AtomSystem(atoms_list, box).energy()


# ─────────────────────────────────────────────────────────────
# CONTACT ENERGY BETWEEN TWO MOLECULES
# ─────────────────────────────────────────────────────────────

def molecule_contact_energy(mol_a, mol_b,
                            use_full_lj: bool = True,
                            use_directional_hbond: bool = True) -> float:
    total = 0.0
    # VdW radii -> LJ-minimum radii (same rule as molecules.lj_radius)
    scale = 1.0 if mol_a.category == 'metal' else _TWO_SIXTH
    for a in mol_a.atoms:
        for b in mol_b.atoms:
            r = dist3(a.pos, b.pos)
            if r > CUTOFF:
                continue
            hb = ((a.hbond_donor and b.hbond_acceptor) or
                  (b.hbond_donor and a.hbond_acceptor))
            total += lj_contact(r, HB_CONTACT if hb
                                else scale * (a.vdw_radius + b.vdw_radius))
            total += coulomb_energy(r, a.charge, b.charge)
            total += magnetic_energy(r, a.symbol, b.symbol)

    total += molecule_hbond_energy(mol_a, mol_b,
                                   use_directionality=use_directional_hbond)
    return total
