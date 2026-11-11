# -*- coding: utf-8 -*-
# engine/csp_benchmark.py
#
# BENCHMARK — does the CSP pipeline find the experimental structures?
#
# For each reference, run CSP and report:
#   rank      position of the first predicted structure matching the
#             experimental one (normalised AMD distance < MATCH_TOL);
#             1 = predicted as the most stable form, '-' = not found
#   dE        its energy above the predicted global minimum (kJ/mol per
#             molecule). Standard CSP success criterion: found with
#             dE <= SUCCESS_DE (force fields cannot resolve smaller gaps).
#   E / E_exp predicted lattice (cohesive) energy of the match vs the
#             experimental value (approximate, from sublimation enthalpy
#             or cohesive energy)
#   ρ ratio   predicted / experimental density of the match
#
# Re-run after every engine change; results are appended to
# benchmark_results.jsonl so progress can be tracked over time.
#
# References built exactly from crystallography (atom-level match):
#   Al, Cu (FCC), Fe (BCC), CO2 (Pa-3, a = 5.624 Å, O at x = 0.1185),
#   ice Ih (P6₃/mmc O sublattice, a 4.497, c 7.322 Å, z = 0.0625 —
#   protons are disordered, so only oxygen positions are compared)
#   Ni, Au (FCC)
# Organic molecules with a COD entry (references.py) are matched against
# the experimental structure rebuilt with the FF molecule and minimised;
# the row also reports where the FF ranks that real form (ff_rank, ff_dE).
# Otherwise cell-level check only: same space group
# and Minkowski-reduced cell lengths within CELL_TOL (setting-independent,
# so P2₁/a, P2₁/n and P2₁/c descriptions compare correctly). Approximate
# literature values:
#   aspirin I      P2₁/c   11.43  6.59 11.40  β 95.7   ρ 1.40
#   paracetamol I  P2₁/n    7.10  9.37 11.66  β 97.4   ρ 1.29
#   benzene        Pbca     7.39  9.42  6.81           ρ 1.09 (138 K)
#   naphthalene    P2₁/a    8.235 6.003 8.658 β 122.9  ρ 1.18
#   anthracene     P2₁/a    8.56  6.04 11.16  β 124.7  ρ 1.25
#   acetic acid    Pna2₁   13.15  3.92  5.76           ρ 1.34 (133 K)
# Experimental lattice energies ≈ -(sublimation enthalpy + 2RT).
#
# Usage:  python csp_benchmark.py              (all, default sizes)
#         python csp_benchmark.py Al CO2 --n 60
#         python csp_benchmark.py --toy        (old unit-less physics)
#         python csp_benchmark.py --rematch    (re-score saved landscapes)

import os
import sys
import json
import time
import pickle
import argparse
import datetime
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csp import run_csp, _finish, _polish, DEDUP_TOL
from lattice import minimise
from references import COD_REFERENCES, experimental_crystal
from spacegroups import operations
from lattice import cell_from_params, primitive_cell
from invariants import amd_normalised, amd_distance

MATCH_TOL = 0.05        # normalised AMD distance for "same structure"
CELL_TOL = 0.07         # reduced-cell length tolerance (force fields ±5% density)
SUCCESS_DE = 2.0        # kJ/mol — experimental form within this of #1
RESULTS_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            'benchmark_results.jsonl')
LANDSCAPES = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          'db', 'benchmark_landscapes')   # last landscape per material


def _expand(sg: str, cell: np.ndarray, sites) -> np.ndarray:
    """Cartesian positions of all symmetry images of fractional sites."""
    pts = []
    for s in sites:
        for W, w in operations(sg):
            f = (W @ np.asarray(s, float) + w) % 1.0
            if not any(np.allclose((f - q + 0.5) % 1.0 - 0.5, 0, atol=1e-4)
                       for q in pts):
                pts.append(f)
    return np.array(pts) @ cell


def _cubic(a, fracs):
    cell = np.eye(3) * a
    return cell, np.array(fracs, float) @ cell


def _ice_ih():
    cell = cell_from_params(4.497, 4.497, 7.322, 90, 90, 120)
    z = 0.0625
    fr = [(1/3, 2/3, z), (2/3, 1/3, z + .5), (2/3, 1/3, -z), (1/3, 2/3, .5 - z)]
    return cell, (np.array(fr) % 1.0) @ cell


def _co2_pa3():
    cell = np.eye(3) * 5.624
    return cell, np.vstack([_expand('Pa-3', cell, [(0, 0, 0)]),
                            _expand('Pa-3', cell, [(0.1185,) * 3])])


FCC = [(0, 0, 0), (.5, .5, 0), (.5, 0, .5), (0, .5, .5)]
BCC = [(0, 0, 0), (.5, .5, .5)]

REFERENCES = {
    'Al':  dict(name='FCC Al', ref=lambda: _cubic(4.05, FCC),
                density=2.70, e_exp=-327, n=40),
    'Cu':  dict(name='FCC Cu', ref=lambda: _cubic(3.615, FCC),
                density=8.96, e_exp=-336, n=40),
    'Ni':  dict(name='FCC Ni', ref=lambda: _cubic(3.524, FCC),
                density=8.91, e_exp=-428, n=40),
    'Au':  dict(name='FCC Au', ref=lambda: _cubic(4.078, FCC),
                density=19.3, e_exp=-368, n=40),
    'Fe':  dict(name='BCC Fe', ref=lambda: _cubic(2.866, BCC),
                density=7.87, e_exp=-413, n=40),
    'CO2': dict(name='CO2 Pa-3', ref=_co2_pa3,
                density=1.64, e_exp=-27, n=100),
    # Proton-ordered models of (disordered) ice Ih distort the O lattice:
    # a correct ordered Ih sits 0.072 from the ideal reference, cubic ice
    # Ic 0.342 — so 0.10 accepts Ih and still rejects its nearest rival.
    'H2O': dict(name='ice Ih (O sublattice)', ref=_ice_ih, only='O',
                density=0.93, e_exp=-59, n=200, match_tol=0.10),
    'benzene': dict(name='benzene (cell only)', ref=None,
                    cell=(7.39, 9.42, 6.81, 90, 90, 90), sg='Pbca',
                    density=1.09, e_exp=-49, n=300),
    'naphthalene': dict(name='naphthalene (cell only)', ref=None,
                        cell=(8.235, 6.003, 8.658, 90, 122.9, 90), sg='P2_1/c',
                        density=1.18, e_exp=-77, n=300),
    'anthracene': dict(name='anthracene (cell only)', ref=None,
                       cell=(8.56, 6.04, 11.16, 90, 124.7, 90), sg='P2_1/c',
                       density=1.25, e_exp=-105, n=300),
    'acetic_acid': dict(name='acetic acid (cell only)', ref=None,
                        cell=(13.15, 3.92, 5.76, 90, 90, 90), sg='Pna2_1',
                        density=1.34, e_exp=None, n=300),
    'paracetamol': dict(name='paracetamol I (cell only)', ref=None,
                        cell=(7.10, 9.37, 11.66, 90, 97.4, 90), sg='P2_1/c',
                        density=1.29, e_exp=-122, n=600),
    'aspirin': dict(name='aspirin I (cell only)', ref=None,
                    cell=(11.43, 6.59, 11.40, 90, 95.7, 90), sg='P2_1/c',
                    density=1.40, e_exp=-113, n=800),
    # CCDC blind-test targets and blind-predicted forms, matched to COD
    # (references.COD_REFERENCES) — the matching path ignores 'cell'
    **{k: dict(name=f'{k} (COD)', ref=None, cell=None, sg=None, density=rho,
               e_exp=None, n=n) for k, rho, n in [
        ('hydantoin', 1.669, 400), ('acrolein', 1.152, 400), ('dbcfb', 2.528, 400),
        ('bt6_xxii', 1.727, 400), ('hoat', 1.616, 400), ('oxauracil', 1.717, 400),
        ('hydroxyuracil', 1.803, 400), ('bt4_thione', 1.479, 800), ('bt3_x', 1.528, 800)]},
}


def _reduced_lengths(cell_matrix) -> np.ndarray:
    from ase.geometry import minkowski_reduce
    rc, _ = minkowski_reduce(np.asarray(cell_matrix, float))
    return np.sort(np.linalg.norm(np.asarray(rc, float), axis=1))


def _cell_match(structure, params, sg, tol=CELL_TOL) -> bool:
    """Same space group and Minkowski-reduced cell lengths within tol."""
    if not structure.space_group.startswith(sg + ' '):
        return False
    exp = _reduced_lengths(cell_from_params(*params))
    pred = _reduced_lengths(primitive_cell(structure.crystal))
    return bool(np.all(np.abs(pred / exp - 1) < tol))


def _fingerprint(structure, only=None):
    x = structure.crystal
    if only is None:
        return structure.amd
    keep = np.array([s == only for s in x.sym])
    return amd_normalised(x.cell, x.positions[keep])


def evaluate(key: str, n: int = None, workers: int = None,
             physical: bool = True, rematch: bool = False) -> dict:
    spec = REFERENCES[key]
    n = n or spec['n']
    t0 = time.time()
    path = os.path.join(LANDSCAPES, f'{key}.pkl')
    if rematch:                       # re-score a saved landscape, no search
        with open(path, 'rb') as f:
            n, unique = pickle.load(f)
    else:
        unique = run_csp(key, n_candidates=n, workers=workers, verbose=False,
                         physical=physical)
        if physical:
            os.makedirs(LANDSCAPES, exist_ok=True)
            with open(path, 'wb') as f:
                pickle.dump((n, unique), f)
    out = {'material': key, 'reference': spec['name'], 'candidates': n,
           'unique': len(unique), 'seconds': round(time.time() - t0, 1),
           'physics': 'forcefield' if physical else 'toy',
           'rank': None, 'dE': None, 'E_ratio': None, 'density_ratio': None,
           'top1': unique[0].space_group if unique else None,
           'top1_E': round(unique[0].energy, 2) if unique else None,
           'top1_density': round(unique[0].crystal.density, 3) if unique else None}
    if not unique:
        return out
    e_min = unique[0].energy
    if physical and key in COD_REFERENCES:
        # Experimental form (COD) rebuilt with the FF's own molecule and
        # minimised: matching two minima of the same model is unambiguous,
        # and its energy shows where the FF ranks it even if the search
        # missed it (ff_dE < 0: something better than anything found).
        label, cod_id, _ = COD_REFERENCES[key][0]
        y = _finish(minimise(experimental_crystal(key, cod_id)))
        if y is not None and unique[0].crystal.meta.get('flex') == 'ok':
            y = _polish(y) or y            # same flexible polish as the landscape
        if y is not None:
            fp = amd_normalised(y.cell, y.positions)
            dists = [amd_distance(fp, s.amd) for s in unique]
            out['closest_amd'] = round(min(dists), 4)
            out['ff_rank'] = 1 + sum(s.energy < y.energy - 0.05 for s in unique)
            out['ff_dE'] = round(y.energy - e_min, 3)
            out['ref_E'] = round(y.energy, 2)
            out['reference'] = f'{spec["name"].split(" (")[0]} {label} (COD {cod_id})'
            hits = [k for k, d in enumerate(dists) if d < DEDUP_TOL]
        else:
            hits = []
    elif spec['ref'] is not None:
        cell, pos = spec['ref']()
        fp = amd_normalised(cell, pos)
        dists = [amd_distance(fp, _fingerprint(s, spec.get('only')))
                 for s in unique]
        out['closest_amd'] = round(min(dists), 4)
        tol = spec.get('match_tol', MATCH_TOL)
        hits = [k for k, d in enumerate(dists) if d < tol]
    else:
        hits = [k for k, s in enumerate(unique)
                if _cell_match(s, spec['cell'], spec['sg'])]
    if hits:
        k = hits[0]
        out['rank'] = k + 1
        out['dE'] = round(unique[k].energy - e_min, 3)
        out['E'] = round(unique[k].energy, 2)
        if physical and spec.get('e_exp'):
            out['E_ratio'] = round(unique[k].energy / spec['e_exp'], 3)
        out['density_ratio'] = round(unique[k].crystal.density / spec['density'], 3)
    return out


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('materials', nargs='*', default=list(REFERENCES))
    ap.add_argument('--n', type=int, default=None)
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--toy', action='store_true', help='old unit-less physics')
    ap.add_argument('--rematch', action='store_true',
                    help='re-score the saved landscapes (no new search)')
    a = ap.parse_args()

    stamp = datetime.datetime.now().isoformat(timespec='seconds')
    fmt = lambda v, f: (format(v, f) if v is not None
                        else format('-', f.split('.')[0].rstrip('f')))
    print(f'{"material":<9} {"reference":<22} {"cand":>4} {"uniq":>4} '
          f'{"rank":>4} {"dE":>7} {"E/Eexp":>6} {"ρ ratio":>7} {"AMD":>6}  '
          f'{"predicted #1":<16} time')
    rows = []
    for key in a.materials:
        r = evaluate(key, a.n, a.workers, physical=not a.toy, rematch=a.rematch)
        if a.rematch:
            r['rematched'] = True
        r['timestamp'] = stamp
        rows.append(r)
        print(f'{key:<9} {r["reference"]:<22} {r["candidates"]:>4} {r["unique"]:>4} '
              f'{fmt(r["rank"], ">4")} {fmt(r["dE"], ">7.2f")} '
              f'{fmt(r["E_ratio"], ">6.2f")} {fmt(r["density_ratio"], ">7.3f")} '
              f'{fmt(r.get("closest_amd"), ">6.3f")}  '
              f'{str(r["top1"]):<16} {r["seconds"]:.0f}s'
              + (f'  | FF ranks real form {r["ff_rank"]} (ΔE {r["ff_dE"]:+.2f})'
                 if r.get('ff_rank') else ''), flush=True)
    with open(RESULTS_FILE, 'a', encoding='utf-8') as f:
        for r in rows:
            f.write(json.dumps(r) + '\n')
    found = sum(r['rank'] is not None for r in rows)
    top1 = sum(r['rank'] == 1 for r in rows)
    ok = sum(r['dE'] is not None and r['dE'] <= SUCCESS_DE for r in rows)
    print(f'\nfound {found}/{len(rows)}, ranked #1 {top1}/{len(rows)}, '
          f'within {SUCCESS_DE:g} kJ/mol of #1 {ok}/{len(rows)}  → {RESULTS_FILE}')


if __name__ == '__main__':
    main()
