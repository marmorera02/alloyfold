# -*- coding: utf-8 -*-
# engine/verify_finds.py
#
# VERIFY HUNTER FINDS AGAINST ALL KNOWN EXPERIMENTAL STRUCTURES
#
# For each material in the hunt landscape:
#   1. list every COD entry with the same formula (all polymorphs,
#      temperatures, pressures) → references/cod_index_<material>.json
#   2. compare every labelled find (low-energy, CIF stored) with each entry:
#      (a) pymatgen StructureMatcher, heavy atoms, ltol 0.2, stol 0.3,
#          angle_tol 5°, volumes scaled (absorbs force-field density errors)
#      (b) force-field basin: COD structure re-minimised with the CSP model,
#          normalised AMD distance to the find < DEDUP_TOL
#   3. write hunt_results/verification.json and print a table
#
# Usage:  python verify_finds.py [materials]

import os
import io
import csv
import sys
import json
import time
import sqlite3
import urllib.parse
import urllib.request
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csp import _finish, DEDUP_TOL
from lattice import minimise
from invariants import amd_normalised, amd_distance
from references import fetch_cif, experimental_crystal, REF_DIR

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, 'db', 'alloyfold.db')
OUT = os.path.join(HERE, 'hunt_results', 'verification.json')

COD_QUERY = {      # material: (COD text search, formula as COD writes it)
    'aspirin': ('acetylsalicylic', 'C9 H8 O4'),
    'paracetamol': ('paracetamol', 'C8 H9 N O2'),
    'benzene': ('benzene', 'C6 H6'),
    'naphthalene': ('naphthalene', 'C10 H8'),
    'anthracene': ('anthracene', 'C14 H10'),
    'acetic_acid': ('acetic acid', 'C2 H4 O2'),
    'caffeine': ('caffeine', 'C8 H10 N4 O2'),
    'urea': ('urea', 'C H4 N2 O'),
    'pyrazine': ('pyrazine', 'C4 H4 N2'),
}


def cod_entries(material):
    path = os.path.join(REF_DIR, f'cod_index_{material}.json')
    if os.path.isfile(path):
        with open(path, encoding='utf-8') as f:
            return json.load(f)
    text, formula = COD_QUERY[material]
    url = ('https://www.crystallography.net/cod/result?format=csv&text='
           + urllib.parse.quote(text))
    data = None
    for attempt in range(4):
        try:
            time.sleep(4 + 6 * attempt)
            with urllib.request.urlopen(url, timeout=60) as r:
                data = r.read().decode('utf-8', 'replace')
            break
        except OSError:
            pass
    if data is None:
        return []
    rows = list(csv.DictReader(io.StringIO('\n'.join(
        l for l in data.splitlines() if not l.startswith('#')))))
    keep = []
    for r in rows:
        if r.get('formula', '').strip('- ').strip() != formula:
            continue
        keep.append({k: r.get(k, '') for k in (
            'file', 'sg', 'a', 'b', 'c', 'alpha', 'beta', 'gamma', 'celltemp',
            'diffrtemp', 'cellpressure', 'diffrpressure', 'Z', 'Zprime',
            'commonname', 'title', 'authors', 'year', 'doi')})
    os.makedirs(REF_DIR, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(keep, f, indent=1)
    return keep


def heavy_structure(cif_text_or_path, is_path=True):
    from pymatgen.core import Structure
    try:
        s = (Structure.from_file(cif_text_or_path) if is_path
             else Structure.from_str(cif_text_or_path, fmt='cif'))
    except Exception:
        # older CIFs pymatgen rejects: read with ASE (applies symmetry) instead
        from ase.io import read
        from pymatgen.io.ase import AseAtomsAdaptor
        src = cif_text_or_path if is_path else io.BytesIO(cif_text_or_path.encode())
        s = AseAtomsAdaptor.get_structure(read(src, format='cif'))
    s.remove_species([e for e in ('H', 'D') if e in [str(x) for x in s.composition.elements]])
    return s


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    import warnings
    warnings.filterwarnings('ignore')
    from pymatgen.analysis.structure_matcher import StructureMatcher
    sm = StructureMatcher(ltol=0.2, stol=0.3, angle_tol=5, primitive_cell=True,
                          scale=True, attempt_supercell=True)
    con = sqlite3.connect(DB_PATH)
    mats = sys.argv[1:] or [m for (m,) in con.execute(
        'SELECT DISTINCT material FROM hunt_structures')]
    report = {}
    for m in mats:
        finds = con.execute(
            "SELECT id, energy, space_group, z, density, label, cif, amd FROM hunt_structures "
            "WHERE material=? AND cif IS NOT NULL ORDER BY energy", (m,)).fetchall()
        entries = cod_entries(m) if m in COD_QUERY else []
        print(f'\n=== {m}: {len(finds)} low-energy finds, {len(entries)} COD entries', flush=True)
        refs = []
        for e in entries:
            cid = int(e['file'])
            info = {'cod': cid, 'sg': e['sg'], 'T': e['celltemp'] or e['diffrtemp'],
                    'P': e['cellpressure'] or e['diffrpressure'], 'year': e['year'],
                    'name': e['commonname'], 'Zprime': e['Zprime'], 'doi': e['doi']}
            try:
                info['raw'] = heavy_structure(fetch_cif(cid))
            except Exception as ex:
                info['error'] = f'CIF: {ex}'
                refs.append(info)
                continue
            try:
                y = _finish(minimise(experimental_crystal(m, cid)))
                info['ff_amd'] = amd_normalised(y.cell, y.positions) if y is not None else None
                info['ff_E'] = round(y.energy, 2) if y is not None else None
            except Exception as ex:
                info['ff_amd'], info['ff_error'] = None, str(ex)[:80]
            refs.append(info)
            print(f'  COD {cid} {e["sg"]:<14} T {info["T"] or "?":>6} P {info["P"] or "-":>7} '
                  f'Z\' {e["Zprime"] or "?"} FF-min E {info.get("ff_E")}  '
                  f'{info.get("ff_error", "") or info.get("error", "")}', flush=True)
        rows = []
        for fid, E, sg, z, rho, label, cif, amd in finds:
            amd = np.array(json.loads(amd))
            try:
                fs = heavy_structure(cif, is_path=False)
            except Exception:
                fs = None
            hits = []
            for r in refs:
                if 'raw' not in r:
                    continue
                a = bool(fs is not None and sm.fit(fs, r['raw']))
                b = r.get('ff_amd') is not None and amd_distance(amd, r['ff_amd']) < DEDUP_TOL
                if a or b:
                    hits.append({'cod': r['cod'], 'sg': r['sg'], 'T': r['T'], 'P': r['P'],
                                 'structure_matcher': a, 'ff_basin': b})
            rows.append({'id': fid, 'energy': round(E, 2), 'space_group': sg, 'z': z,
                         'density': round(rho, 3), 'hunter_label': label, 'matches': hits})
            tag = ('KNOWN  ' + ', '.join(f'COD {h["cod"]} ({h["sg"].strip()}; '
                                          f'{"SM" if h["structure_matcher"] else ""}'
                                          f'{"+" if h["structure_matcher"] and h["ff_basin"] else ""}'
                                          f'{"FF" if h["ff_basin"] else ""})' for h in hits)
                   if hits else 'no match in COD')
            print(f'  find #{fid:<4} E {E:8.2f} {sg:<16} ρ {rho:.3f}  [{label}]  → {tag}', flush=True)
        report[m] = {'cod_entries': [{k: v for k, v in r.items() if k not in ('raw', 'ff_amd')}
                                     for r in refs], 'finds': rows}
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=1, default=str)
    print(f'\n→ {OUT}')


if __name__ == '__main__':
    main()
