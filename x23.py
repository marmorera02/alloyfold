# -*- coding: utf-8 -*-
# engine/x23.py
#
# X23 BENCHMARK — lattice energies and cell volumes of 23 molecular crystals
#
# The standard yardstick for molecular-crystal energy models (Reilly &
# Tkatchenko 2013; revised by Dolgonos, Hoja & Boese 2019).
#
# Reference lattice energies: diffusion Monte Carlo (DMC), Della Pia et al.,
# arXiv:2402.13059 (Phys. Rev. Lett. 2024), Table 1 — total uncertainty
# ~2 kJ/mol, more accurate than back-corrected sublimation enthalpies.
#
# Structures: the standard X23 crystal structures (2×2×2 supercells of the
# experimental cells) from the MolCrys-MACE repository (references/x23/,
# see SOURCE.txt), cut back to one cell; --cod uses COD entries instead
# (incomplete: several X23 crystals are not in COD). Each is rebuilt with
# AlloyFold's rigid force-field molecule and minimised.
#
# Reported: lattice-energy errors vs DMC and vs the X23b experiment-derived
# values, and volume error (%) vs the X23b 0 K electronic reference volumes
# (experimental volumes back-corrected for thermal expansion, Table 2).
# Benzene, naphthalene, anthracene and acetic acid were used to fit the
# MMFF crystal correction → reported separately; the other 19 are held out.
#
# Usage:  python x23.py [--uncalibrated] [--workers 8]

import os
import io
import csv
import sys
import json
import time
import argparse
import urllib.parse
import urllib.request
import numpy as np
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, 'references')

# key: (name, SMILES, COD text query, COD formula, name keyword, space group
#       requirement or None, DMC lattice energy kJ/mol)
X23 = {
    'cyclohexanedione': ('1,4-cyclohexanedione', 'O=C1CCC(=O)CC1', 'cyclohexane-1,4-dione', 'C6 H8 O2', '', None, -88.3),
    'acetic_acid':  ('acetic acid', 'CC(=O)O', 'acetic acid', 'C2 H4 O2', '', None, -71.7),
    'adamantane':   ('adamantane', 'C1C2CC3CC1CC(C2)C3', 'adamantane', 'C10 H16', 'adamant|tricyclo', None, -61.0),
    'ammonia':      ('ammonia', 'N', 'ammonia', 'H3 N', '', None, -38.2),
    'anthracene':   ('anthracene', 'c1ccc2cc3ccccc3cc2c1', 'anthracene', 'C14 H10', '', None, -100.2),
    'benzene':      ('benzene', 'c1ccccc1', 'benzene', 'C6 H6', '', None, -49.8),
    'CO2':          ('carbon dioxide', None, 'carbon dioxide', 'C O2', '', None, -29.4),
    'cyanamide':    ('cyanamide', 'NC#N', 'cyanamide', 'C H2 N2', '', None, -83.6),
    'cytosine':     ('cytosine', 'Nc1cc[nH]c(=O)n1', 'cytosine', 'C4 H5 N3 O', '', None, -156.2),
    'ethylcarbamate': ('ethyl carbamate', 'CCOC(N)=O', 'carbamate', 'C3 H7 N O2', 'carbamat', None, -84.2),
    'formamide':    ('formamide', 'NC=O', 'formamide', 'C H3 N O', '', None, -81.0),
    'imidazole':    ('imidazole', 'c1c[nH]cn1', 'imidazole', 'C3 H4 N2', 'imidazol', None, -88.2),
    'naphthalene':  ('naphthalene', 'c1ccc2ccccc2c1', 'naphthalene', 'C10 H8', '', None, -75.5),
    'oxalic_alpha': ('oxalic acid α', 'OC(=O)C(=O)O', 'oxalic acid', 'C2 H2 O4', '', None, -102.6),
    'oxalic_beta':  ('oxalic acid β', 'OC(=O)C(=O)O', 'oxalic acid', 'C2 H2 O4', '', None, -102.3),
    'pyrazine':     ('pyrazine', 'c1cnccn1', 'pyrazine', 'C4 H4 N2', '', None, -61.1),
    'pyrazole':     ('pyrazole', 'c1cn[nH]c1', 'pyrazole', 'C3 H4 N2', 'pyrazol', None, -77.3),
    'triazine':     ('s-triazine', 'c1ncncn1', 'triazine', 'C3 H3 N3', '', None, -60.5),
    'trioxane':     ('trioxane', 'C1OCOCO1', 'trioxane', 'C3 H6 O3', '', None, -62.1),
    # name filter: same formula as pyrazine N,N'-dioxide and maleic hydrazide (COD has no uracil itself)
    'uracil':       ('uracil', 'O=c1cc[nH]c(=O)[nH]1', 'uracil', 'C4 H4 N2 O2', 'uracil|pyrimidine-2,4', None, -134.3),
    'urea':         ('urea', 'NC(N)=O', 'urea', 'C H4 N2 O', '', None, -108.5),
    'hexamine':     ('hexamine', 'C1N2CN3CN1CN(C2)C3', 'hexamethylenetetramine', 'C6 H12 N4', '', None, -86.2),
    'succinic_acid': ('succinic acid', 'OC(=O)CCC(=O)O', 'succinic acid', 'C4 H6 O4', 'succin|butanedioic', None, -125.2),
}
CALIBRATION_SET = {'benzene', 'naphthalene', 'anthracene', 'acetic_acid'}

# X23b (Dolgonos, Hoja & Boese, PCCP 2019, 21, 24333 — references/c9cp04488d.pdf)
#   E_exp   recommended reference lattice energy E_latt^ref,exp (Table 5), kJ/mol
#   V_exp   experimental cell volume at T_l (Table 2), Å³ per cell
#   V_ref   electronic (0 K, vibration-free) reference volume (Table 2), Å³ per cell
X23B = {
    'cyclohexanedione': (-90.0, 279.6, 262.5), 'acetic_acid': (-73.6, 297.3, 288.8),
    'adamantane': (-71.8, 393.1, 357.6), 'ammonia': (-38.7, 128.6, 121.5),
    'anthracene': (-110.4, 455.2, 441.2), 'benzene': (-54.8, 461.8, 444.3),
    'CO2': (-29.4, 171.3, 164.8), 'cyanamide': (-81.5, 415.7, 407.9),
    'cytosine': (-163.5, 472.4, 440.3), 'ethylcarbamate': (-88.2, 248.8, 231.2),
    'formamide': (-81.1, 224.1, 211.9), 'hexamine': (-84.1, 332.4, 321.6),
    'imidazole': (-90.4, 348.8, 336.4), 'naphthalene': (-81.3, 340.8, 329.7),
    'oxalic_alpha': (-98.8, 312.6, 293.2), 'oxalic_beta': (-96.8, 156.9, 150.5),
    'pyrazine': (-64.3, 203.6, 189.6), 'pyrazole': (-78.8, 698.3, 662.5),
    'triazine': (-62.6, 586.8, 528.0), 'trioxane': (-64.6, 616.5, 580.7),
    'succinic_acid': (-130.1, 239.3, 233.3), 'uracil': (-136.2, 463.4, 442.0),
    'urea': (-102.1, 145.1, 140.8),
}


def score(res, sel, label):
    """Errors vs DMC and X23b energies; volume vs the X23b 0 K reference
    (per molecule: the X23b cell's Z is inferred from our experimental cell)."""
    print(f'\nX23 — {label}')
    print(f'  {"crystal":<18} {"COD":>8} {"E_AF":>8} {"ΔE DMC":>7} {"ΔE X23b":>8} {"ΔV %":>7}  set')
    rows = []
    for key in X23:
        r, s = res.get(key), sel.get(key)
        if not s or not r or 'error' in r:
            print(f'  {key:<18} {"-":>8}  skipped: {(r or {}).get("error", "no COD structure")}')
            continue
        e_x23b, v_exp_cell, v_ref_cell = X23B[key]
        z_ref = max(1, round(v_exp_cell / r['V_exp']))
        dE_dmc = r['E'] - X23[key][6]
        dE_b = r['E'] - e_x23b
        dV = 100 * (r['V_min'] / (v_ref_cell / z_ref) - 1)
        held = key not in CALIBRATION_SET
        rows.append((key, dE_dmc, dE_b, dV, held))
        print(f'  {key:<18} {str(s["cod"]):>8} {r["E"]:8.1f} {dE_dmc:+7.1f} {dE_b:+8.1f} {dV:+7.1f}  '
              f'{"held-out" if held else "calibration"}')
    for name, sub in (('all', rows), ('held-out', [r for r in rows if r[4]])):
        if not sub:
            continue
        a = np.array([[r[1], r[2], r[3]] for r in sub])
        print(f'  {name:<9} n={len(sub):>2}  MAE energy: vs DMC {np.abs(a[:, 0]).mean():5.1f} '
              f'(signed {a[:, 0].mean():+5.1f}),  vs X23b {np.abs(a[:, 1]).mean():5.1f} '
              f'(signed {a[:, 1].mean():+5.1f}) kJ/mol   '
              f'MAE volume {np.abs(a[:, 2]).mean():4.1f}% (signed {a[:, 2].mean():+4.1f}%)')
    return rows


# Space groups of the X23 crystal forms (COD spellings; alternatives |)
X23_SG = {
    'cyclohexanedione': 'P 1 21/c 1|P 1 21/n 1|P 1 21/a 1', 'acetic_acid': 'P n a 21',
    'adamantane': 'P -4 21 c', 'ammonia': 'P 21 3', 'anthracene': 'P 1 21/a 1|P 1 21/c 1',
    'benzene': 'P b c a', 'CO2': 'P a -3', 'cyanamide': 'P b c a',
    'cytosine': 'P 21 21 21', 'ethylcarbamate': 'P -1', 'formamide': 'P 1 21/n 1|P 1 21/c 1',
    'hexamine': 'I -4 3 m', 'imidazole': 'P 1 21/c 1|P 1 21/n 1|P 1 21/a 1',
    'naphthalene': 'P 1 21/a 1|P 1 21/c 1', 'oxalic_alpha': 'P c a b|P b c a',
    'oxalic_beta': 'P 1 21/c 1|P 1 21/n 1|P 1 21/a 1', 'pyrazine': 'P m n n|P n n m',
    'pyrazole': 'P n a 21|P 1 21/c 1|P 1 21/n 1', 'triazine': 'R -3 c :H|R -3 c',
    'trioxane': 'R 3 c :H|R 3 c', 'uracil': 'P 1 21/a 1|P 1 21/c 1',
    'urea': 'P -4 21 m', 'succinic_acid': 'P 1 21/c 1|P 1 21/n 1|P 1 21/a 1',
}


def _cod_search(formula):
    url = ('https://www.crystallography.net/cod/result?format=csv&formula='
           + urllib.parse.quote(formula))
    for attempt in range(4):
        try:
            time.sleep(4 + 6 * attempt)
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read().decode('utf-8', 'replace')
            return list(csv.DictReader(io.StringIO('\n'.join(
                l for l in data.splitlines() if not l.startswith('#')))))
        except OSError:
            pass
    return []


def _num(x, default):
    try:
        return float(x)
    except (TypeError, ValueError):
        return default


def select_structures():
    """
    COD entry per X23 crystal: searched by formula, then
      - name keyword (only to separate same-formula compounds)
      - the X23 space group (X23_SG)
      - volume check: COD cell = X23 cell × {1/2, 1, 2, 4} within 6%
      - ambient pressure; lowest temperature wins
    Cached in references/x23_selection.json (failed lookups are retried).
    """
    path = os.path.join(REF_DIR, 'x23_selection.json')
    sel = {}
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            sel = json.load(f)
    for key, (name, smi, text, formula, kw, _, _) in X23.items():
        if sel.get(key) and sel[key].get('method') == 'formula+sg+volume':
            continue
        rows = _cod_search(formula)
        v_x23 = X23B[key][1]
        ok, rejected = [], []
        for r in rows:
            names = (r.get('chemname', '') + ' ' + r.get('commonname', '')).lower()
            if kw and not any(k in names for k in kw.split('|')):
                rejected.append((r['file'], 'name'))
                continue
            if r.get('sg', '').strip() not in X23_SG[key].split('|'):
                rejected.append((r['file'], r.get('sg', '').strip()))
                continue
            if _num(r.get('cellpressure') or r.get('diffrpressure'), 101) > 1000:
                rejected.append((r['file'], 'pressure'))
                continue
            ratio = _num(r.get('vol'), 0) / v_x23
            if not any(abs(ratio / k - 1) < 0.06 for k in (0.5, 1, 2, 4)):
                rejected.append((r['file'], f'volume ×{ratio:.2f}'))
                continue
            ok.append(r)
        if not ok:
            sel[key] = None
            print(f'  {key}: no suitable COD entry ({len(rows)} with formula; rejected: '
                  f'{rejected[:6]})', flush=True)
        else:
            best = min(ok, key=lambda r: (_num(r.get('celltemp') or r.get('diffrtemp'), 295),
                                          -_num(r.get('year'), 0)))
            sel[key] = {'cod': int(best['file']), 'sg': best['sg'],
                        'T': best.get('celltemp') or best.get('diffrtemp'),
                        'year': best.get('year'), 'name': best.get('chemname'),
                        'method': 'formula+sg+volume'}
            print(f'  {key}: COD {best["file"]} {best["sg"]} T {sel[key]["T"] or "?"} '
                  f'({len(ok)} of {len(rows)} pass)', flush=True)
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(sel, f, indent=1)
    return sel


X23_DIR = os.path.join(REF_DIR, 'x23')
# the MolCrys-MACE repository has no licence file, so its structures are not
# shipped with AlloyFold: they are downloaded on the first benchmark run
X23_URL = ('https://raw.githubusercontent.com/water-ice-group/MolCrys-MACE/main/'
           'CMD_QMD_INPUT_OUTPUT_FILES/{}/solid/classic/POSCAR')
X23_FOLDERS = dict(zip(
    ['cyclohexanedione', 'acetic_acid', 'adamantane', 'ammonia', 'anthracene', 'benzene', 'CO2',
     'cyanamide', 'cytosine', 'ethylcarbamate', 'formamide', 'imidazole', 'naphthalene',
     'oxalic_alpha', 'oxalic_beta', 'pyrazine', 'pyrazole', 'triazine', 'trioxane', 'uracil',
     'urea', 'hexamine', 'succinic_acid'],
    ['01_cyclohexanedione', '02_acetic_acid', '03_adamantane', '04_ammonia', '05_anthracene',
     '06_benzene', '07_co2', '08_cyanamide', '09_cytosine', '10_ethylcarbamate', '11_formamide',
     '12_imidazole', '13_naphthalene', '14_oxalic_acid_alpha', '15_oxalic_acid_beta',
     '16_pyrazine', '17_pyrazole', '18_triazine', '19_trioxane', '20_uracil', '21_urea',
     '22_hexamine', '23_succinic_acid']))


def x23_file(key):
    """Path of the X23 structure for key; downloaded once if missing (None if unavailable)."""
    path = os.path.join(X23_DIR, f'{key}.vasp')
    if not os.path.isfile(path) and key in X23_FOLDERS:
        try:
            with urllib.request.urlopen(X23_URL.format(X23_FOLDERS[key]), timeout=30) as r:
                data = r.read()
            os.makedirs(X23_DIR, exist_ok=True)
            with open(path, 'wb') as f:
                f.write(data)
        except OSError as e:
            print(f'  {key}: X23 structure not available ({e})', flush=True)
            return None
    return path


def _repeats(sc, tol=0.05):
    """Supercell repeat factor along each axis: the largest n ≤ 4 for which
    a shift by 1/n of that axis maps every atom onto an atom (< tol Å)."""
    f = sc.get_scaled_positions(wrap=True)
    cell = np.array(sc.cell)
    out = []
    for d in range(3):
        for n in (4, 3, 2, 1):
            g = f.copy()
            g[:, d] = (g[:, d] + 1.0 / n) % 1.0
            diff = (f[None, :, :] - g[:, None, :] + 0.5) % 1.0 - 0.5
            if np.max(np.min(np.linalg.norm(diff @ cell, axis=2), axis=1)) < tol:
                out.append(n)
                break
    return out


def x23_unit_cell(key):
    """The X23 structure (references/x23/<key>.vasp, MolCrys-MACE) is a
    supercell of the X23 experimental cell (repeats differ per crystal,
    e.g. acetic acid 1×4×2): detect the repeats and cut back to one cell."""
    from ase.io import read
    sc = read(x23_file(key), format='vasp')
    n = np.array(_repeats(sc))
    # small uniform shift: atoms on a cut plane fall consistently on one side
    frac = (sc.get_scaled_positions(wrap=True) + 1.37e-3) % 1.0
    keep = np.all(frac < 1.0 / n, axis=1)
    cell = sc.copy()[keep]
    cell.set_cell(np.array(sc.cell) / n[:, None], scale_atoms=False)
    if int(np.prod(n)) * len(cell) != len(sc):
        raise ValueError(f'{key}: repeats {n.tolist()} do not divide the supercell')
    if _volume_ok(key, cell):
        return cell
    # Some files are skewed supercells (axes are combinations of the cell's
    # axes, e.g. uracil: 8 cells, no repeat along any file axis), so the cut
    # above keeps the whole supercell. spglib finds the true lattice instead.
    cell = _conventional_cell(sc)
    if not _volume_ok(key, cell):
        raise ValueError(f'{key}: no unit cell with the X23b volume {X23B[key][1]} Å³ found '
                         f'(got {cell.get_volume():.1f} Å³)')
    return cell


def _volume_ok(key, atoms, tol=0.06):
    """Cell volume matches the X23b experimental cell (Dolgonos et al. 2019)."""
    return abs(atoms.get_volume() / X23B[key][1] - 1) < tol


def _conventional_cell(sc, symprec=0.05):
    """Conventional unit cell (spglib), atoms kept where they are (no idealising)."""
    import spglib
    from ase import Atoms
    lat, frac, nums = spglib.standardize_cell(
        (np.array(sc.cell), sc.get_scaled_positions(), sc.get_atomic_numbers()),
        to_primitive=False, no_idealize=True, symprec=symprec)
    return Atoms(numbers=nums, scaled_positions=frac, cell=lat, pbc=True)


def _evaluate(args):
    key, cod_id, uncalibrated = args
    import warnings
    warnings.filterwarnings('ignore')
    import torch
    torch.set_num_threads(1)
    import forcefield
    if uncalibrated:
        forcefield.MMFF_SCALE.update({'eps': 1.0, 'r': 1.0, 'eps_x': 1.0, 'r_x': 1.0})
    from references import experimental_crystal
    from csp import _finish
    from lattice import minimise, space_group_of
    try:
        x = (experimental_crystal(key, atoms=x23_unit_cell(key)) if cod_id == 'x23'
             else experimental_crystal(key, cod_id))
        y = _finish(minimise(x))
        if y is None:
            return key, {'error': 'minimisation failed'}
        return key, {'E': y.energy, 'V_exp': x.volume / x.z, 'V_min': y.volume / y.z,
                     'z': x.z, 'sg': space_group_of(y)}
    except Exception as e:
        return key, {'error': str(e)[:160]}


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser()
    ap.add_argument('--uncalibrated', action='store_true')
    ap.add_argument('--workers', type=int, default=8)
    ap.add_argument('--cod', action='store_true',
                    help='use COD structures instead of the standard X23 files')
    ap.add_argument('--rescore', default=None,
                    help='re-score a saved results JSON (no minimisation)')
    a = ap.parse_args()
    if a.rescore:
        with open(a.rescore, encoding='utf-8') as f:
            d = json.load(f)
        score(d['results'], d['selection'], d['label'])
        return
    if a.cod:
        print('selecting experimental structures from COD…', flush=True)
        sel = select_structures()
    else:   # the standard X23 structures (references/x23, MolCrys-MACE repository)
        sel = {k: {'cod': 'x23', 'source': 'MolCrys-MACE X23 POSCAR'} for k in X23 if x23_file(k)}
    jobs = [(k, v['cod'], a.uncalibrated) for k, v in sel.items() if v]
    t0 = time.time()
    with Pool(a.workers) as pool:
        res = dict(pool.map(_evaluate, jobs))
    label = 'uncalibrated MMFF94' if a.uncalibrated else 'AlloyFold (calibrated MMFF94)'
    print(f'({time.time() - t0:.0f}s)')
    score(res, sel, label)
    out = os.path.join(HERE, 'benchmark_x23' + ('_uncalibrated' if a.uncalibrated else '') + '.json')
    with open(out, 'w', encoding='utf-8') as f:
        json.dump({'label': label, 'reference': 'DMC, Della Pia et al. arXiv:2402.13059',
                   'selection': sel, 'results': res,
                   'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S')}, f, indent=1, default=str)
    print(f'→ {out}')


if __name__ == '__main__':
    main()
