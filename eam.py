# -*- coding: utf-8 -*-
# engine/eam.py
#
# EMBEDDED-ATOM METHOD (EAM) — tabulated alloy potentials (setfl / eam.alloy)
#
#   E = Σ_i F_{t_i}(ρ_i) + ½ Σ_{i≠j} φ_{t_i t_j}(r_ij),   ρ_i = Σ_{j≠i} ρ_{t_j}(r_ij)
#
# F: embedding energy (how deep atom i sits in the electron sea of its
# neighbours), ρ: electron density each neighbour contributes, φ: pair
# repulsion/attraction. Tables are read from the standard LAMMPS setfl
# file and interpolated exactly as LAMMPS does (pair_eam.cpp: 7-coefficient
# cubic spline on a uniform grid), so results match published values.
#
# Potentials (references/potentials/, see SOURCE.txt):
#   TiAl   Zope & Mishin, Phys. Rev. B 68, 024102 (2003), NIST IPR
#
# Validation against the NIST release notes:  python eam.py

import os
import math
import numpy as np
import torch
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
POT_DIR = os.path.join(HERE, 'references', 'potentials')
POTENTIALS = {'TiAl': 'Zope-Ti-Al-2003.eam.alloy'}
EV = 96.48533                               # kJ/mol per eV

torch.set_default_dtype(torch.float64)


def _spline(f):
    """LAMMPS pair_eam interpolate(): coefficients c3..c6 per grid point."""
    n = len(f)
    c6 = np.asarray(f, float).copy()
    c5 = np.zeros(n)
    c5[0] = f[1] - f[0]
    c5[1] = 0.5 * (f[2] - f[0])
    c5[n - 2] = 0.5 * (f[n - 1] - f[n - 3])
    c5[n - 1] = f[n - 1] - f[n - 2]
    c5[2:n - 2] = ((f[0:n - 4] - f[4:n]) + 8.0 * (f[3:n - 1] - f[1:n - 3])) / 12.0
    c4 = np.zeros(n)
    c3 = np.zeros(n)
    c4[:n - 1] = 3.0 * (f[1:] - f[:-1]) - 2.0 * c5[:-1] - c5[1:]
    c3[:n - 1] = c5[:-1] + c5[1:] - 2.0 * (f[1:] - f[:-1])
    return np.stack([c3, c4, c5, c6], axis=1)          # (n, 4)


class Table:
    """A function on a uniform grid x_k = k·dx, evaluated like LAMMPS."""

    def __init__(self, values, dx):
        self.n = len(values)
        self.inv = 1.0 / dx
        self.c = torch.as_tensor(_spline(np.asarray(values, float)))

    def __call__(self, x):
        p = x * self.inv
        m = torch.clamp(torch.floor(p).long(), 0, self.n - 2)
        t = torch.clamp(p - m, max=1.0)
        c = self.c[m]
        return ((c[:, 0] * t + c[:, 1]) * t + c[:, 2]) * t + c[:, 3]


class EAMPotential:
    def __init__(self, path):
        with open(path, encoding='utf-8', errors='replace') as f:
            lines = f.read().split('\n')
        head = lines[3].split()
        self.elements = head[1:1 + int(head[0])]
        n_rho, d_rho, n_r, d_r, cut = lines[4].split()
        n_rho, n_r = int(n_rho), int(n_r)
        self.d_rho, self.d_r, self.cutoff = float(d_rho), float(d_r), float(cut)
        nums = []
        pos = 5
        self.mass, self.F, self.rho = {}, {}, {}

        def take(count):
            nonlocal pos, nums
            while len(nums) < count:
                nums += [float(v.replace('D', 'E')) for v in lines[pos].split()]
                pos += 1
            out, nums = nums[:count], nums[count:]
            return np.array(out)

        for el in self.elements:
            info = lines[pos].split()
            pos += 1
            self.mass[el] = float(info[1])
            self.F[el] = Table(take(n_rho), self.d_rho)
            self.rho[el] = Table(take(n_r), self.d_r)
        self.z2r = {}
        for i, a in enumerate(self.elements):
            for j in range(i + 1):
                b = self.elements[j]
                t = Table(take(n_r), self.d_r)       # r·φ(r)
                self.z2r[(a, b)] = self.z2r[(b, a)] = t

    def prepare(self, i, j, types):
        """Element masks for a fixed pair list (reused across evaluations)."""
        t = np.array([self.elements.index(s) for s in types])
        ti, tj = t[np.asarray(i)], t[np.asarray(j)]
        return {
            'n': len(types), 'i': torch.as_tensor(np.asarray(i)),
            'rho': [(el, torch.as_tensor(tj == k)) for k, el in enumerate(self.elements)],
            'phi': [((a, b), torch.as_tensor((ti == ka) & (tj == kb)))
                    for ka, a in enumerate(self.elements)
                    for kb, b in enumerate(self.elements)],
            'emb': [(el, torch.as_tensor(t == k)) for k, el in enumerate(self.elements)],
        }

    def energy_prepared(self, r, prep):
        """Total energy (eV) for pair distances r matching prepare()."""
        mask = r < self.cutoff
        r = torch.where(mask, r, torch.full_like(r, self.cutoff))
        rho_j = torch.zeros_like(r)
        phi = torch.zeros_like(r)
        for el, sel in prep['rho']:
            if sel.any():
                rho_j = torch.where(sel, self.rho[el](r), rho_j)
        for ab, sel in prep['phi']:
            if sel.any():
                phi = torch.where(sel, self.z2r[ab](r) / r, phi)
        rho_j = torch.where(mask, rho_j, torch.zeros_like(r))
        phi = torch.where(mask, phi, torch.zeros_like(r))
        dens = torch.zeros(prep['n'], dtype=r.dtype).index_add(0, prep['i'], rho_j)
        emb = torch.zeros(prep['n'], dtype=r.dtype)
        for el, sel in prep['emb']:
            if sel.any():
                emb = torch.where(sel, self.F[el](dens), emb)
        return emb.sum() + 0.5 * phi.sum()

    def energy(self, r, i, j, types):
        """Total energy (eV) from pair distances r (both orders), atom
        indices i (centre), j (neighbour) and per-atom element names."""
        return self.energy_prepared(r, self.prepare(i, j, types))


@lru_cache(maxsize=None)
def get_potential(name='TiAl') -> EAMPotential:
    return EAMPotential(os.path.join(POT_DIR, POTENTIALS[name]))


# ─────────────────────────────────────────────────────────────
# ASE calculator (energy, forces, stress) — for phonons / QHA / relaxing
# ─────────────────────────────────────────────────────────────

def _pairs(pos, cell, cut):
    from scipy.spatial import cKDTree
    from lattice import image_range
    shifts = image_range(cell, cut)
    n = len(pos)
    images = (pos[None, :, :] + (shifts @ cell)[:, None, :]).reshape(-1, 3)
    coo = cKDTree(pos).sparse_distance_matrix(cKDTree(images), cut, output_type='coo_matrix')
    i, k = coo.row, coo.col
    j, m = k % n, k // n
    keep = ~((i == j) & np.all(shifts[m] == 0, axis=1))
    return i[keep], j[keep], shifts[m[keep]].astype(float)


def eam_energy_forces_stress(atoms, pot: EAMPotential):
    pos0 = np.array(atoms.get_positions())
    cell0 = np.array(atoms.cell)
    i, j, n = _pairs(pos0, cell0, pot.cutoff + 0.01)
    pos = torch.as_tensor(pos0).clone().requires_grad_(True)
    eps = torch.zeros(3, 3, requires_grad=True)
    S = torch.eye(3) + 0.5 * (eps + eps.T)
    cell = torch.as_tensor(cell0) @ S
    p = pos @ S
    d = p[torch.as_tensor(j)] + torch.as_tensor(n) @ cell - p[torch.as_tensor(i)]
    r = torch.sqrt((d * d).sum(-1))
    e = pot.energy(r, torch.as_tensor(i), torch.as_tensor(j), atoms.get_chemical_symbols())
    g_pos, g_eps = torch.autograd.grad(e, (pos, eps))
    vol = abs(np.linalg.det(cell0))
    stress = (g_eps.numpy() / vol)                       # eV/Å³ (ASE sign)
    voigt = np.array([stress[0, 0], stress[1, 1], stress[2, 2],
                      stress[1, 2], stress[0, 2], stress[0, 1]])
    return float(e), -g_pos.numpy(), voigt


def calculator(name='TiAl'):
    from ase.calculators.calculator import Calculator, all_changes

    class EAMCalculator(Calculator):
        implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            e, f, s = eam_energy_forces_stress(self.atoms, get_potential(name))
            self.results = {'energy': e, 'free_energy': e, 'forces': f, 'stress': s}

    return EAMCalculator()


# ─────────────────────────────────────────────────────────────
# Validation against the NIST release notes (Zope & Mishin 2003)
# ─────────────────────────────────────────────────────────────

NIST_CHECKS = [   # (name, builder args, eV/atom from SOLD/LAMMPS)
    ('Al fcc a=4.05', ('fcc', 'Al', 4.05, None), -3.3600000286),
    ('Ti hcp a=2.953 c=4.68', ('hcp', 'Ti', 2.953, 4.68), -4.8500111),
    ('Ti3Al L12 a=4.089', ('L12', ('Ti', 'Al'), 4.089, None), -4.765009),
    ('TiAl L10 a=4.001 c=4.181045', ('L10', ('Ti', 'Al'), 4.001, 4.181045), -4.5085371),
    ('TiAl B2 a=3.278', ('B2', ('Ti', 'Al'), 3.278, None), -4.390773865),
    ('TiAl3 L12 a=4.05', ('L12', ('Al', 'Ti'), 4.05, None), -4.0359496),
]


def build(kind, el, a, c=None):
    from ase import Atoms
    from ase.build import bulk
    if kind == 'fcc':
        return bulk(el, 'fcc', a=a, cubic=True)
    if kind == 'hcp':
        return bulk(el, 'hcp', a=a, c=c)
    if kind == 'L12':           # A3B: B at corner, A at face centres
        A, B = el
        return Atoms([B, A, A, A], scaled_positions=[(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)],
                     cell=[a, a, a], pbc=True)
    if kind == 'L10':           # AB: layers along c
        A, B = el
        return Atoms([A, A, B, B], scaled_positions=[(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)],
                     cell=[a, a, c], pbc=True)
    if kind == 'B2':
        A, B = el
        return Atoms([A, B], scaled_positions=[(0, 0, 0), (.5, .5, .5)], cell=[a, a, a], pbc=True)
    raise ValueError(kind)


if __name__ == '__main__':
    import sys
    sys.path.insert(0, HERE)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    pot = get_potential('TiAl')
    print(f'Zope-Mishin Ti-Al EAM: elements {pot.elements}, cutoff {pot.cutoff:.4f} Å')
    worst = 0.0
    for name, args, ref in NIST_CHECKS:
        at = build(*args)
        e, f, s = eam_energy_forces_stress(at, pot)
        per = e / len(at)
        worst = max(worst, abs(per - ref))
        print(f'  {name:<30} {per:14.9f} eV/atom   NIST {ref:14.9f}   Δ {per - ref:+.2e}')
    print(f'max deviation {worst:.2e} eV/atom')
