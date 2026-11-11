# -*- coding: utf-8 -*-
# engine/dft_rank.py
#
# QUANTUM-MECHANICAL RE-RANKING OF PREDICTED CRYSTALS  (periodic PBE-D3(BJ))
#
# The force field finds candidate packings well but ranks polymorphs only to
# ~5 kJ/mol, while real polymorphs differ by 1-4 kJ/mol. The final ranking
# is therefore redone with density-functional theory: PBE (no fitted crystal
# data) plus Grimme's D3(BJ) dispersion, the standard first-principles level
# in CSP (blind tests 5-7).
#
#   PySCF periodic DFT: GTH pseudopotentials, Gaussian basis, real-space
#     multigrid density; Γ point on a supercell ≥ MIN_LENGTH Å per direction
#   simple-DFT-D3: periodic D3(BJ) energy for the same cell
#
# Energies are per molecule of the primitive cell (kJ/mol); polymorphs of
# the same molecule are compared directly (conformer strain is included).
#
# Usage:  python dft_rank.py hydantoin [--window 15] [--basis gth-dzvp]

import os
import sys
import json
import time
import argparse
import pickle
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT_DIR = os.path.join(HERE, 'dft_results')
HARTREE_KJ = 2625.4996
BOHR = 0.52917721
PRESCALE = 0.99           # round-1 single point at this linear scale (DFT-D volumes ≈ 0.97 × FF)
N_SCAN = 5                # round 2: finalists with the full volume scan
MIN_LENGTH = 11.0         # Å: Γ supercell ≥ this per direction (7 Å shifted energies by up to 9 kJ/mol)
KE_CUTOFF = 100.0         # Hartree: real-space grid for the multigrid density


def to_atoms(x):
    """Crystal → ASE Atoms in the primitive cell (spglib, no idealisation)."""
    import spglib
    from ase import Atoms
    from ase.data import atomic_numbers
    nums = [atomic_numbers[s] for s in x.sym]
    frac = np.linalg.solve(np.asarray(x.cell).T, np.asarray(x.positions).T).T % 1.0
    out = spglib.standardize_cell((np.asarray(x.cell), frac, nums), to_primitive=True,
                                  no_idealize=True, symprec=1e-3)
    if out is None or len(out[2]) > len(nums):
        return Atoms(numbers=nums, scaled_positions=frac, cell=x.cell, pbc=True)
    return Atoms(numbers=out[2], scaled_positions=out[1], cell=out[0], pbc=True)


def d3_energy(at):
    """Periodic D3(BJ)-PBE dispersion energy (Hartree)."""
    from dftd3.interface import RationalDampingParam, DispersionModel
    model = DispersionModel(at.numbers, at.positions / BOHR,
                            lattice=np.array(at.cell) / BOHR, periodic=np.array([True] * 3))
    return model.get_dispersion(RationalDampingParam(method='pbe'), grad=False)['energy']


def pbe_energy(at, basis='gth-dzvp', ke_cutoff=KE_CUTOFF, verbose=0):
    """Periodic PBE total energy (Hartree, Γ point) and basis size."""
    from pyscf.pbc import gto, dft
    cell = gto.Cell()
    cell.atom = [(s, p) for s, p in zip(at.get_chemical_symbols(), at.positions)]
    cell.a = np.array(at.cell)
    cell.unit = 'Angstrom'
    cell.basis = basis
    cell.pseudo = 'gth-pbe'
    if ke_cutoff:
        cell.ke_cutoff = ke_cutoff
    cell.verbose = verbose
    cell.max_memory = 24000
    cell.build()
    mf = dft.RKS(cell)          # Γ only: the k-point multigrid path diverges for some cells
    mf.xc = 'pbe'
    mf.conv_tol = 1e-7
    mf.max_cycle = 100
    mf = mf.multigrid_numint()   # real-space multigrid (GPW-like): fast for GTH + GGA
    e = mf.kernel()
    if not mf.converged:         # small-gap / conjugated cases: second-order solver
        mf2 = mf.newton()
        e = mf2.kernel(mf.make_rdm1())
        if not mf2.converged:
            raise RuntimeError('SCF not converged')
    return float(e), cell.nao


def energy_per_molecule(x, basis='gth-dzvp', ke_cutoff=KE_CUTOFF, verbose=0):
    at = to_atoms(x)
    rep = [max(1, int(np.ceil(MIN_LENGTH / L))) for L in at.cell.lengths()]
    at = at.repeat(rep)            # supercell = Monkhorst-Pack mesh rep (Γ-folded)
    n_mol = len(at) // (len(x.sym) // (int(np.max(x.mol)) + 1))
    t0 = time.time()
    e_pbe, nao = pbe_energy(at, basis, ke_cutoff, verbose)
    e_d3 = float(d3_energy(at))
    return {'E_kJ_per_mol': (e_pbe + e_d3) * HARTREE_KJ / n_mol,
            'E_pbe': e_pbe, 'E_d3': e_d3, 'Z_prim': n_mol, 'atoms': len(at),
            'supercell': rep, 'nao': nao, 'seconds': round(time.time() - t0)}


def scaled(x, f):
    """Cell scaled by the linear factor f, molecules rigid (centres follow)."""
    from dataclasses import replace
    return replace(x, cell=np.asarray(x.cell) * f, com=np.asarray(x.com) * f)


def volume_scan(x, basis='gth-dzvp', ke_cutoff=KE_CUTOFF, step=0.02, max_points=7):
    """
    DFT-D energy along rigid-molecule cell scaling, walking downhill from
    the FF cell, then a parabola through the three lowest points. Corrects
    the FF density (which is off by up to ~20 % for S/halogen crystals)
    without a full DFT relaxation.
    """
    pts = {}
    def e(f):
        if f not in pts:
            pts[f] = energy_per_molecule(scaled(x, f), basis, ke_cutoff)
        return pts[f]['E_kJ_per_mol']
    d = -step if e(1.0 - step) < e(1.0) else step
    f = 1.0 + d if d < 0 else 1.0
    while len(pts) < max_points:
        nxt = round(f + d, 4)
        if e(nxt) >= e(f):
            break
        f = nxt
    fs = sorted(pts, key=lambda k: pts[k]['E_kJ_per_mol'])[:3]
    es = [pts[k]['E_kJ_per_mol'] for k in fs]
    a, b, c = np.polyfit(fs, es, 2)
    f_min = -b / (2 * a) if a > 0 else fs[0]
    f_min = float(np.clip(f_min, min(fs), max(fs)))
    best = dict(pts[fs[0]])
    best.update({'E_kJ_per_mol': float(np.polyval([a, b, c], f_min)) if a > 0 else es[0],
                 'scale': f_min, 'V_over_ff': f_min ** 3,
                 'scan': {str(k): v['E_kJ_per_mol'] for k, v in sorted(pts.items())},
                 'seconds': sum(v['seconds'] for v in pts.values())})
    return best


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from csp import _finish, _polish
    from lattice import minimise
    from references import COD_REFERENCES, experimental_crystal
    ap = argparse.ArgumentParser()
    ap.add_argument('material')
    ap.add_argument('--window', type=float, default=15.0,
                    help='re-rank every candidate within this many kJ/mol of the FF minimum')
    ap.add_argument('--max', type=int, default=40, help='cap on re-ranked candidates')
    ap.add_argument('--scan', type=int, default=N_SCAN, help='finalists given the full volume scan')
    ap.add_argument('--basis', default='gth-dzvp')
    ap.add_argument('--ke', type=float, default=KE_CUTOFF, help='kinetic cutoff (Hartree)')
    a = ap.parse_args()
    land = os.path.join(HERE, 'db', 'benchmark_landscapes', f'{a.material}.pkl')
    with open(land, 'rb') as f:
        _, unique = pickle.load(f)
    # blind rule: everything the FF puts within the window (the real form was
    # FF rank 28 at +14.5 kJ/mol for 5-hydroxyuracil); experimental forms are
    # added only as a check, labelled as such
    jobs = [(f'FF #{k + 1}', s.crystal, s.energy) for k, s in enumerate(unique[:a.max])
            if s.energy <= unique[0].energy + a.window]
    for label, cod_id, _ in COD_REFERENCES.get(a.material, []):
        y = _finish(minimise(experimental_crystal(a.material, cod_id)))
        if unique[0].crystal.meta.get('flex') == 'ok':
            y = _polish(y) or y            # same flexible polish as the landscape
        jobs.append((f'experimental {label} (COD {cod_id})', y, y.energy))
    os.makedirs(OUT_DIR, exist_ok=True)
    path = os.path.join(OUT_DIR, f'{a.material}.json')
    done = json.load(open(path, encoding='utf-8')) if os.path.isfile(path) else {}
    # results belong to one landscape ("FF #3" is another structure after a
    # new search): a changed landscape archives the old results
    stamp = os.path.getmtime(land)
    if done and done.get('_landscape_mtime') != stamp:
        os.replace(path, path.replace('.json', f'_{int(time.time())}.json'))
        done = {}
    done['_landscape_mtime'] = stamp
    def run(name, x, e_ff, key, fn):
        if key in done and 'error' not in done[key]:
            return done[key]
        try:
            r = fn(x)
        except Exception as e:
            r = {'error': repr(e)[:200]}
        r.update({'name': name, 'E_ff': e_ff, 'basis': a.basis})
        done[key] = r
        json.dump(done, open(path, 'w', encoding='utf-8'), indent=1)
        print(f'{name:<34} FF {e_ff:8.2f}   DFT {r.get("E_kJ_per_mol", float("nan")):14.2f} kJ/mol  '
              f'{"V/V_FF %.3f" % r["V_over_ff"] if "V_over_ff" in r else "single point"}  ({r.get("atoms")} atoms, '
              f'supercell {r.get("supercell")}, {r.get("seconds")} s) {r.get("error", "")}', flush=True)
        return r

    # round 1: one DFT-D point per candidate at a typical density;
    # round 2: full volume scan for the N_SCAN best of round 1 (+ the
    # experimental forms, as a check) — ~3× cheaper than scanning everything
    tag = f'{a.basis}|{a.ke}'
    sp = {}
    for name, x, e_ff in jobs:
        if name.startswith('experimental'):
            continue
        sp[name] = run(name, x, e_ff, f'{name}|{tag}|sp',
                       lambda x: energy_per_molecule(scaled(x, PRESCALE), a.basis, a.ke))
    ok = sorted((n for n in sp if 'E_kJ_per_mol' in sp[n]), key=lambda n: sp[n]['E_kJ_per_mol'])
    finalists = set(ok[:a.scan])
    for name, x, e_ff in jobs:
        if name in finalists or name.startswith('experimental'):
            run(name, x, e_ff, f'{name}|{tag}', lambda x: volume_scan(x, a.basis, a.ke))
    final = {}
    for v in done.values():
        if not isinstance(v, dict) or v.get('basis') != a.basis or 'E_kJ_per_mol' not in v:
            continue
        scanned = 'V_over_ff' in v
        if v['name'] not in final or scanned:
            final[v['name']] = (v, scanned)
    if final:
        e0 = min(v['E_kJ_per_mol'] for v, _ in final.values())
        f0 = min(v['E_ff'] for v, _ in final.values())
        print('\nre-ranked (kJ/mol relative to the lowest; * = single point only):')
        for v, scanned in sorted(final.values(), key=lambda t: t[0]['E_kJ_per_mol']):
            print(f'  DFT {v["E_kJ_per_mol"] - e0:6.2f}{" " if scanned else "*"}  FF {v["E_ff"] - f0:6.2f}   {v["name"]}')


if __name__ == '__main__':
    main()
