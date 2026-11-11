# -*- coding: utf-8 -*-
# engine/flex.py
#
# FLEXIBLE-MOLECULE RELAXATION  (final polish after the rigid search)
#
# The search keeps each molecule rigid in one of a few gas-phase conformers.
# Real crystals bend molecules slightly (twisted nitro groups, flattened
# amides, ...); a rigid conformer that cannot bend clashes in the true
# packing, and the true form then ranks far too high (BT3 X: +53 kJ/mol).
#
# Here every atom may also move within its molecule (displacements δ per
# atom; each molecule bends independently, so cells whose molecules hold
# different conformers are handled too). Energy per molecule:
#
#   E = [E_inter(crystal) + Σ_m (E_MMFF,intra(molecule m) - E_gas min)] / Z
#
# with the intermolecular part exactly as in the rigid model (forcefield.
# FFModel, incl. calibration and Ewald) and the intramolecular part from
# RDKit's MMFF94 (energy + analytic gradient), so the strain a crystal
# imposes is paid for. At δ = 0 it equals the rigid energy + e_conf.

import numpy as np
import torch
from dataclasses import replace
from functools import lru_cache

from lattice import (Crystal, build_pairs, model_cutoff, _rodrigues, _reduce_and_wrap,
                     plane_spacing, MIN_SPACING, SKIN)

DELTA_BUDGET = 0.25      # Å: max atom displacement within its molecule per round


@lru_cache(maxsize=None)
def _mmff(key):
    """RDKit MMFF force field (atom order of forcefield._from_smiles) and the
    gas-phase minimum energy (kJ/mol) that conformer energies refer to."""
    from rdkit import Chem
    from rdkit.Chem import AllChem
    from forcefield import SMILES, KCAL, get_ff_molecule
    mol = Chem.AddHs(Chem.MolFromSmiles(SMILES[key]))
    AllChem.EmbedMolecule(mol, randomSeed=7)
    ff = AllChem.MMFFGetMoleculeForceField(mol, AllChem.MMFFGetMoleculeProperties(mol))
    c0 = get_ff_molecule(key).conformer(0)[0]       # lowest conformer: e_conf = 0
    return ff, ff.CalcEnergy(c0.flatten().tolist()) * KCAL


class _Intra(torch.autograd.Function):
    """MMFF94 intramolecular energy (kJ/mol) of one molecule, RDKit gradient."""

    @staticmethod
    def forward(ctx, xyz, key):
        from forcefield import KCAL
        ff, _ = _mmff(key)
        flat = xyz.detach().numpy().astype(float).flatten().tolist()
        e = ff.CalcEnergy(flat) * KCAL
        g = np.asarray(ff.CalcGrad(flat)).reshape(-1, 3) * KCAL
        ctx.save_for_backward(torch.as_tensor(g, dtype=xyz.dtype))
        return torch.tensor(e, dtype=xyz.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        (g,) = ctx.saved_tensors
        return grad_out * g, None


class _FlexGeometry:
    """lattice._RigidModel geometry with per-atom displacements δ (lab frame,
    applied before the molecule's rotation)."""

    def __init__(self, x, pairs):
        i, j, n = pairs
        self.i, self.j, self.n = (torch.as_tensor(i), torch.as_tensor(j), torch.as_tensor(n))
        self.cell = torch.as_tensor(x.cell)
        self.com = torch.as_tensor(x.com)
        self.offset = torch.as_tensor(x.offset)
        self.mol = torch.as_tensor(x.mol)
        self.delta = torch.zeros_like(self.offset)

    def geometry(self, eps, dcom, rot):
        E = torch.stack([
            torch.stack([eps[0], eps[5] / 2, eps[4] / 2]),
            torch.stack([eps[5] / 2, eps[1], eps[3] / 2]),
            torch.stack([eps[4] / 2, eps[3] / 2, eps[2]])])
        S = torch.eye(3) + E
        off = torch.einsum('nk,nlk->nl', self.offset + self.delta, _rodrigues(rot)[self.mol])
        return self.cell @ S, self.com @ S + dcom, off


def _molecules(x):
    """Atom indices of each molecule, in the force field's site order."""
    return [np.flatnonzero(x.mol == m)[np.argsort(x.site[x.mol == m])] for m in range(x.z)]


def _round(x, max_iter):
    from forcefield import FFModel, COULOMB, get_ff_molecule
    cut = model_cutoff(x)
    pairs = build_pairs(x, cut + SKIN)
    geo = _FlexGeometry(x, pairs)
    model = FFModel(geo, x, pairs)
    mols = [torch.as_tensor(ix) for ix in _molecules(x)]
    q = torch.as_tensor(get_ff_molecule(x.ff).charges)
    iu = torch.triu_indices(len(q), len(q), 1)
    alpha = model.lv['alpha']

    def intra_ewald(xyz):           # Σ_{i<j} q_i q_j erf(αr)/r within one molecule
        d = xyz[iu[0]] - xyz[iu[1]]
        r = torch.sqrt((d * d).sum(-1))
        return (q[iu[0]] * q[iu[1]] * torch.erf(alpha * r) / r).sum()

    off0 = geo.offset
    with torch.no_grad():           # the value FFModel.e_const subtracted
        fixed = sum(float(intra_ewald(off0[ix])) for ix in mols) if model.charged else 0.0
    _, e_gas = _mmff(x.ff)
    budget = SKIN / 3.0
    s_max = budget / (cut + SKIN)
    t_max = budget / max(float(np.linalg.norm(x.offset, axis=1).max()), 0.5)
    eps_r = torch.zeros(6, requires_grad=True)
    dcom_r = torch.zeros(x.z, 3, requires_grad=True)
    rot_r = torch.zeros(x.z, 3, requires_grad=True)
    d_r = torch.zeros_like(off0, requires_grad=True)

    def squash():
        return (s_max * torch.tanh(eps_r / s_max), budget * torch.tanh(dcom_r / budget),
                t_max * torch.tanh(rot_r / t_max), DELTA_BUDGET * torch.tanh(d_r / DELTA_BUDGET))

    def energy(v):
        eps, dcom, rot, delta = v
        geo.delta = delta
        e = model(eps, dcom, rot)
        shape = off0 + delta        # rotation-free: intramolecular terms are invariant
        for ix in mols:
            e = e + _Intra.apply(shape[ix], x.ff) - e_gas
        if model.charged:           # the molecule's Ewald self-term follows its shape
            e = e + COULOMB * (fixed - sum(intra_ewald(shape[ix]) for ix in mols))
        return e / x.z

    opt = torch.optim.LBFGS([eps_r, dcom_r, rot_r, d_r], lr=1.0, max_iter=max(max_iter, 1),
                            tolerance_grad=1e-6, tolerance_change=1e-10,
                            history_size=20, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        e = energy(squash())
        e.backward()
        return e

    if max_iter:                     # max_iter=0: evaluate only
        opt.step(closure)
    with torch.no_grad():
        v = squash()
        e = float(energy(v))
        geo.delta = v[3]
        cell, com, off = geo.geometry(*v[:3])
    y = replace(x, cell=cell.numpy().copy(), com=com.numpy().copy(), offset=off.numpy().copy())
    return _reduce_and_wrap(y), e


def strain_energy(x: Crystal) -> float:
    """Mean MMFF intramolecular energy above the gas-phase minimum (kJ/mol)."""
    _, e_gas = _mmff(x.ff)
    return float(np.mean([float(_Intra.apply(torch.as_tensor(x.offset[ix]), x.ff)) - e_gas
                          for ix in _molecules(x)]))


def flex_minimise(x: Crystal, max_rounds: int = 30, max_iter: int = 40,
                  tol: float = 1e-6) -> Crystal:
    """Flexible relaxation of a rigid-minimised MMFF crystal. Returns a new
    Crystal with .energy = lattice energy + intramolecular strain per
    molecule (comparable with the rigid energy, which includes e_conf)."""
    from forcefield import get_ff_molecule
    if get_ff_molecule(x.ff).form != 'mmff':
        return x
    start = x.offset.copy()
    e_prev, rounds = None, 0
    for rounds in range(1, max_rounds + 1):
        x, e = _round(x, max_iter)
        if not np.isfinite(e) or plane_spacing(x.cell) < MIN_SPACING:
            return replace(x, energy=float('nan'), meta={**x.meta, 'flex': 'failed'})
        if e_prev is not None and abs(e - e_prev) <= tol * max(1.0, abs(e)):
            break
        e_prev = e
    # bending = shape change beyond a rigid rotation (Kabsch-aligned per molecule)
    bend = 0.0
    for ix in _molecules(x):
        P, Qm = start[ix], x.offset[ix]
        U, _, Vt = np.linalg.svd(P.T @ Qm)
        D = np.diag([1, 1, np.sign(np.linalg.det(U @ Vt))])
        bend = max(bend, float(np.linalg.norm(P @ U @ D @ Vt - Qm, axis=1).max()))
    return replace(x, energy=e, meta={**x.meta, 'flex': 'ok', 'flex_rounds': rounds,
                                      'e_conf': strain_energy(x), 'max_bend': bend})
