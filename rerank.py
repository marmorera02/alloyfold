# -*- coding: utf-8 -*-
# engine/rerank.py
#
# INDEPENDENT RE-RANKING OF HUNTER FINDS  (MACE-MP-0 + D3, flexible)
#
# The search uses rigid molecules and a calibrated classical force field
# (MMFF94). This re-examines every low-energy find with an independent,
# higher-level model: the MACE-MP-0 machine-learning potential (trained on
# DFT/PBE, MIT licence) plus Grimme D3 dispersion. Everything is relaxed:
# cell, molecular positions/orientations AND the molecules' internal
# geometry (no rigid-body approximation).
#
# For each material:
#   - the known experimental forms (COD) are relaxed the same way, so the
#     method's ability to rank the real forms is visible next to the finds
#   - lattice energy = E(crystal)/Z - E(isolated molecule), kJ/mol
#   - a find "survives" if it is still the same structure after relaxation
#     (pymatgen StructureMatcher, heavy atoms) — otherwise it was an
#     artefact of the rigid model and falls into another basin
#   - powder XRD pattern (Cu Kα) of every relaxed structure → CSV + PNG
#
# Usage:  python rerank.py [materials] [--workers 8] [--max-per-material 12]

import os
import io
import sys
import json
import time
import sqlite3
import argparse
import warnings
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, 'db', 'alloyfold.db')
OUT_DIR = os.path.join(HERE, 'hunt_results')
EV = 96.48533                    # kJ/mol per eV
FMAX = 0.02                      # eV/Å convergence
MAX_STEPS = 600

_CALC = None


def _calc():
    global _CALC
    if _CALC is None:
        warnings.filterwarnings('ignore')
        import torch
        torch.set_num_threads(2)
        from mace.calculators import mace_mp
        _CALC = mace_mp(model='medium', dispersion=True,
                        default_dtype='float64', device='cpu')
    return _CALC


def _relax(task):
    """Relax one crystal (cell + all atoms). task = (key, cif_text)."""
    key, cif = task
    from ase.io import read
    from ase.optimize import FIRE
    from ase.filters import FrechetCellFilter
    warnings.filterwarnings('ignore')
    try:
        at = read(io.BytesIO(cif.encode('utf-8')), format='cif')
        at.calc = _calc()
        opt = FIRE(FrechetCellFilter(at), logfile=None)
        converged = opt.run(fmax=FMAX, steps=MAX_STEPS)
        energy = float(at.get_potential_energy())
        out = io.BytesIO()
        at.calc = None
        from ase.io import write
        write(out, at, format='cif')
        return key, {'E_total_eV': energy, 'cif': out.getvalue().decode('utf-8'),
                     'steps': opt.nsteps, 'converged': bool(converged)}
    except Exception as e:
        return key, {'error': str(e)[:200]}


def _molecule_energy(material):
    """Lowest MACE energy of the isolated molecule (relaxed, from each
    force-field conformer)."""
    from ase import Atoms
    from ase.optimize import BFGS
    from forcefield import get_ff_molecule
    ff = get_ff_molecule(material)
    best = None
    for c in range(ff.n_conformers):
        xyz = ff.conformer(c)[0]
        keep = [k for k, s in enumerate(ff.symbols) if s != 'M']
        mol = Atoms([ff.symbols[k] for k in keep], positions=xyz[keep],
                    cell=[30, 30, 30], pbc=False)
        mol.center()
        mol.calc = _calc()
        BFGS(mol, logfile=None).run(fmax=0.01, steps=500)
        e = float(mol.get_potential_energy())
        best = e if best is None else min(best, e)
    return best


def _xrd(cif, path_base, title):
    """Simulated Cu Kα powder pattern: CSV of peaks + PNG plot."""
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    s = Structure.from_str(cif, fmt='cif')
    pat = XRDCalculator(wavelength='CuKa').get_pattern(s, two_theta_range=(4, 40))
    with open(path_base + '.csv', 'w', encoding='utf-8') as f:
        f.write('two_theta,intensity,d_spacing\n')
        for x, y, d in zip(pat.x, pat.y, pat.d_hkls):
            f.write(f'{x:.3f},{y:.2f},{d:.4f}\n')
    tt = np.linspace(4, 40, 3600)
    prof = sum(y * np.exp(-0.5 * ((tt - x) / 0.06) ** 2) for x, y in zip(pat.x, pat.y))
    fig, ax = plt.subplots(figsize=(7, 2.4), dpi=120)
    ax.plot(tt, prof, lw=0.9, color='#3b5bdb')
    ax.set_xlabel('2θ (°, Cu Kα)')
    ax.set_ylabel('intensity')
    ax.set_title(title, fontsize=9)
    ax.set_yticks([])
    fig.tight_layout()
    fig.savefig(path_base + '.png')
    plt.close(fig)
    top = sorted(zip(pat.y, pat.x), reverse=True)[:5]
    return [round(x, 2) for _, x in sorted(top, key=lambda t: t[1])]


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    warnings.filterwarnings('ignore')
    ap = argparse.ArgumentParser()
    ap.add_argument('materials', nargs='*')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--max-per-material', type=int, default=12)
    a = ap.parse_args()

    from references import COD_REFERENCES, fetch_cif
    from lattice import space_group_of
    from pymatgen.core import Structure
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, scale=True,
                          attempt_supercell=True)

    con = sqlite3.connect(DB_PATH)
    mats = a.materials or [m for (m,) in con.execute(
        'SELECT DISTINCT material FROM hunt_structures WHERE cif IS NOT NULL')]
    tasks, meta = [], {}
    for m in mats:
        rows = con.execute('SELECT id, energy, space_group, density, label, cif FROM '
                           'hunt_structures WHERE material=? AND cif IS NOT NULL '
                           'ORDER BY energy LIMIT ?', (m, a.max_per_material)).fetchall()
        for fid, E, sg, rho, label, cif in rows:
            key = f'{m}|find {fid}'
            tasks.append((key, cif))
            meta[key] = {'material': m, 'kind': 'find', 'id': fid, 'mmff_E': E,
                         'mmff_sg': sg, 'mmff_density': rho, 'label': label, 'cif_in': cif}
        for label, cod_id, T in COD_REFERENCES.get(m, []):
            key = f'{m}|exp {label} (COD {cod_id})'
            with open(fetch_cif(cod_id), encoding='utf-8', errors='replace') as f:
                tasks.append((key, f.read()))
            meta[key] = {'material': m, 'kind': 'experimental', 'cod': cod_id,
                         'label': f'{label} (COD {cod_id}, {T} K)'}
    print(f'{len(tasks)} structures to relax with MACE-MP-0 + D3 '
          f'({a.workers} workers)', flush=True)

    t0 = time.time()
    results = {}
    with Pool(a.workers) as pool:
        mol_E = dict(zip(mats, pool.map(_molecule_energy, mats)))
        for k, (key, r) in enumerate(pool.imap_unordered(_relax, tasks), start=1):
            results[key] = r
            print(f'  [{k}/{len(tasks)}] {key}: '
                  + (r.get('error') or f'{r["steps"]} steps'), flush=True)

    xrd_dir = os.path.join(OUT_DIR, 'xrd')
    os.makedirs(xrd_dir, exist_ok=True)
    report = {}
    for m in mats:
        rows = []
        for key, info in meta.items():
            if info['material'] != m or 'error' in results.get(key, {'error': 1}):
                continue
            r = results[key]
            s_out = Structure.from_str(r['cif'], fmt='cif')
            from forcefield import get_ff_molecule
            n_mol = len([x for x in get_ff_molecule(m).symbols if x != 'M'])
            z = round(len(s_out) / n_mol)
            e_latt = (r['E_total_eV'] / z - mol_E[m]) * EV
            rho = s_out.density
            same = None
            if info['kind'] == 'find':
                s_in = Structure.from_str(info['cif_in'], fmt='cif')
                a_in = s_in.copy(); a_in.remove_species(['H'])
                a_out = s_out.copy(); a_out.remove_species(['H'])
                same = bool(sm.fit(a_in, a_out))
            from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
            try:
                sg = SpacegroupAnalyzer(s_out, symprec=0.1).get_space_group_symbol()
            except Exception:
                sg = '?'
            tag = (f'find{info["id"]}' if info['kind'] == 'find'
                   else f'exp_COD{info["cod"]}')
            peaks = _xrd(r['cif'], os.path.join(xrd_dir, f'{m}_{tag}'),
                         f'{m} {tag} — {sg}, ρ {rho:.3f}, E_latt {e_latt:.1f} kJ/mol (MACE-MP-0+D3)')
            with open(os.path.join(xrd_dir, f'{m}_{tag}_mace.cif'), 'w', encoding='utf-8') as f:
                f.write(r['cif'])
            rows.append({**{k: v for k, v in info.items() if k != 'cif_in'},
                         'mace_E_latt': round(e_latt, 2), 'mace_density': round(rho, 3),
                         'mace_space_group': sg, 'survives_relaxation': same,
                         'z': z, 'steps': r['steps'], 'converged': r['converged'],
                         'xrd_top_peaks_2theta': peaks})
        rows.sort(key=lambda x: x['mace_E_latt'])
        e0 = rows[0]['mace_E_latt'] if rows else 0
        for k, x in enumerate(rows, start=1):
            x['mace_rank'] = k
            x['mace_dE'] = round(x['mace_E_latt'] - e0, 2)
        report[m] = {'molecule_E_eV': mol_E[m], 'structures': rows}
        print(f'\n=== {m} (MACE-MP-0 + D3, lattice energies kJ/mol)')
        for x in rows:
            name = (f'EXPERIMENTAL {x["label"]}' if x['kind'] == 'experimental'
                    else f'find #{x["id"]} [{x["label"]}] (MMFF {x["mmff_E"]:.1f})')
            surv = ('' if x['survives_relaxation'] is None else
                    ('  survives' if x['survives_relaxation'] else '  → changed basin'))
            print(f'  {x["mace_rank"]:>2}. {x["mace_E_latt"]:8.2f} (+{x["mace_dE"]:5.2f})  '
                  f'{x["mace_space_group"]:<10} ρ {x["mace_density"]:.3f}  {name}{surv}')
    with open(os.path.join(OUT_DIR, 'rerank_mace.json'), 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1)
    print(f'\n{time.time() - t0:.0f}s → {os.path.join(OUT_DIR, "rerank_mace.json")}, '
          f'XRD patterns in {xrd_dir}')


if __name__ == '__main__':
    main()
