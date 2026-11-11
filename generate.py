# -*- coding: utf-8 -*-
# engine/generate.py
#
# RANDOM SYMMETRIC CRYSTAL GENERATOR
#
# Replaces forge + bounding-box tiling. A candidate is drawn the way
# real molecular crystals are built:
#   1. pick a space group (weighted by CSD frequency)
#   2. pick a cell consistent with its crystal system, sized to a
#      sensible packing coefficient
#   3. place Z' molecules at general positions (random orientation),
#      or a centrosymmetric molecule on an inversion centre
#   4. generate all symmetry images (inversion, screw axes, glides …)
#   5. inflate the cell until no two molecules overlap
# The result goes to lattice.minimise().

import numpy as np
from typing import Optional

from lattice import Crystal, cell_from_params
from molecules import Molecule, lj_radius
from spacegroups import (operations, crystal_system, sampling_weights,
                         inversion_centres)
from physics import HB_CONTACT

MIN_CONTACT = 0.75   # reject/inflate below this fraction of contact distance


def random_rotation(rng) -> np.ndarray:
    q = rng.normal(size=4)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - z*w), 2*(x*z + y*w)],
        [2*(x*y + z*w), 1 - 2*(x*x + z*z), 2*(y*z - x*w)],
        [2*(x*z - y*w), 2*(y*z + x*w), 1 - 2*(x*x + y*y)]])


def molecular_volume(mol: Molecule, n_samples: int = 20000,
                     rng=None) -> float:
    """Volume of the union of VdW spheres (Monte Carlo), Å³."""
    rng = rng or np.random.default_rng(0)
    p = mol.centered().positions
    r = mol.radii
    lo, hi = (p - r[:, None]).min(0), (p + r[:, None]).max(0)
    pts = rng.uniform(lo, hi, (n_samples, 3))
    d2 = ((pts[:, None, :] - p[None]) ** 2).sum(-1)
    inside = (d2 < r[None] ** 2).any(1)
    return float(np.prod(hi - lo) * inside.mean())


def is_centrosymmetric(mol: Molecule, tol: float = 0.05) -> bool:
    p = mol.centered().positions
    syms = [a.symbol for a in mol.atoms]
    for k, q in enumerate(p):
        d = np.linalg.norm(p + q, axis=1)          # distance to -q
        m = int(np.argmin(d))
        if d[m] > tol or syms[m] != syms[k]:
            return False
    return True


def random_cell(system: str, volume: float, rng) -> np.ndarray:
    ratios = np.exp(rng.uniform(np.log(0.5), np.log(2.0), 3))
    if system == 'triclinic':
        while True:
            ang = rng.uniform(60, 120, 3)
            try:
                cell = cell_from_params(*ratios, *ang)
                if abs(np.linalg.det(cell)) > 0.4 * np.prod(ratios):
                    break
            except ValueError:
                pass
    elif system == 'monoclinic':                  # unique axis b
        cell = cell_from_params(*ratios, 90, rng.uniform(80, 125), 90)
    else:                                          # orthorhombic / cubic
        cell = cell_from_params(*ratios, 90, 90, 90)
    return cell * (volume / abs(np.linalg.det(cell))) ** (1 / 3)


def _min_contact_ratio(x: Crystal) -> float:
    """Smallest (distance / contact distance) between different molecules."""
    from lattice import build_pairs
    i, j, n = build_pairs(x, cut=6.0)
    if len(i) == 0:
        return np.inf
    pos = x.positions
    d = np.linalg.norm(pos[j] + n @ x.cell - pos[i], axis=1)
    hb = (x.donor[i] & x.acceptor[j]) | (x.donor[j] & x.acceptor[i])
    d0 = np.where(hb, HB_CONTACT, x.radii[i] + x.radii[j])
    return float((d / d0).min())


def random_crystal(mol: Molecule, rng, space_group: Optional[str] = None,
                   z_prime: int = 1, packing: Optional[float] = None,
                   v_mol: Optional[float] = None,
                   max_inflate: int = 8) -> Optional[Crystal]:
    """
    One random crystal of `mol` (None if it cannot be placed).
    z_prime independent molecules per asymmetric unit; a centrosymmetric
    molecule is put on an inversion centre with probability 0.5 when the
    group has one (Z' = 1/2 — common for CO2, benzene, naphthalene …).
    """
    if space_group is None:
        names, p = sampling_weights()
        space_group = str(rng.choice(names, p=p))
    ops = operations(space_group)
    body0 = mol.centered().positions

    centres = inversion_centres(space_group)
    special = (z_prime == 1 and centres and is_centrosymmetric(mol)
               and rng.random() < 0.5)

    # asymmetric unit: fractional centre + orientation
    asym = []
    for _ in range(z_prime):
        if special:
            t = rng.integers(0, 2, 3) / 2.0
            s = (centres[rng.integers(len(centres))] + t) % 1.0
        else:
            s = rng.random(3)
        asym.append((s, body0 @ random_rotation(rng).T))

    # symmetry images (fractional centre, cartesian-free body via W)
    mols = []                                  # (s, W, body)
    for s, body in asym:
        seen = []
        for W, w in ops:
            s2 = (W @ s + w) % 1.0
            if any(np.allclose((s2 - q + 0.5) % 1.0 - 0.5, 0, atol=1e-4)
                   for q in seen):
                continue                       # molecule on a special position
            seen.append(s2)
            mols.append((s2, W, body))
    z = len(mols)

    if v_mol is None:
        v_mol = molecular_volume(mol)
    pc = packing or rng.uniform(0.55, 0.72)
    cell = random_cell(crystal_system(space_group), z * v_mol / pc, rng)

    syms = [a.symbol for a in mol.atoms]
    n_at = len(syms)

    def assemble(cell):
        Linv = np.linalg.inv(cell)
        com, off = [], []
        for s, W, body in mols:
            com.append(s @ cell)
            # cartesian image of the body: x' = x · L⁻¹ Wᵀ L
            off.append(body @ (Linv @ W.T @ cell))
        return Crystal(
            cell=cell, com=np.array(com), offset=np.concatenate(off),
            mol=np.repeat(np.arange(z), n_at), sym=syms * z,
            radii=np.tile([lj_radius(a.vdw_radius, mol.category)
                           for a in mol.atoms], z),
            charge=np.tile([a.charge for a in mol.atoms], z),
            donor=np.tile([a.hbond_donor for a in mol.atoms], z),
            acceptor=np.tile([a.hbond_acceptor for a in mol.atoms], z),
            mass=mol.mass, space_group=space_group,
            meta={'z': z, 'special_position': bool(special)},
            site=np.tile(np.arange(n_at), z))

    # Inflate at most 1.06^8 ≈ 1.6× per axis: a molecule sitting near a
    # symmetry element overlaps its own image at any cell size, and
    # inflating that one 10× produced 25 000 Å³ cells for two aspirins.
    x = assemble(cell)
    for _ in range(max_inflate):
        if _min_contact_ratio(x) >= MIN_CONTACT:
            return x
        cell = cell * 1.06
        x = assemble(cell)
    return None
