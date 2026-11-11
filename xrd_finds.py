# -*- coding: utf-8 -*-
# engine/xrd_finds.py
#
# SIMULATED POWDER X-RAY PATTERNS of hunter finds (storage + findability)
#
# For every stored low-energy structure (hunt_structures with a CIF):
#   hunt_results/xrd/<material>_<id>.csv   peak list (2θ, intensity, d)
#   hunt_results/xrd/<material>_<id>.png   pattern (Cu Kα, 4-40° 2θ)
#   hunt_structures.xrd_peaks              JSON of the 10 strongest peaks
# A lab pattern can then be matched against the predicted structures.
#
# Patterns are computed directly from the force-field structures; those
# are ~2-5% off in density, so peak positions shift by ~1-2% in 2θ —
# match on the pattern, not on exact angles.
#
# Usage:  python xrd_finds.py [materials]

import os
import io
import sys
import json
import sqlite3
import warnings
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, 'db', 'alloyfold.db')
XRD_DIR = os.path.join(HERE, 'hunt_results', 'xrd')


def pattern(cif: str):
    from pymatgen.core import Structure
    from pymatgen.analysis.diffraction.xrd import XRDCalculator
    try:
        s = Structure.from_str(cif, fmt='cif')
    except Exception:
        from ase.io import read
        from pymatgen.io.ase import AseAtomsAdaptor
        s = AseAtomsAdaptor.get_structure(read(io.BytesIO(cif.encode()), format='cif'))
    return XRDCalculator(wavelength='CuKa').get_pattern(s, two_theta_range=(4, 40))


def save(pat, base, title):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    with open(base + '.csv', 'w', encoding='utf-8') as f:
        f.write('two_theta,intensity,d_spacing\n')
        for x, y, d in zip(pat.x, pat.y, pat.d_hkls):
            f.write(f'{x:.3f},{y:.2f},{d:.4f}\n')
    tt = np.linspace(4, 40, 3600)
    prof = sum(y * np.exp(-0.5 * ((tt - x) / 0.06) ** 2) for x, y in zip(pat.x, pat.y))
    fig, ax = plt.subplots(figsize=(7, 2.3), dpi=110)
    ax.plot(tt, prof, lw=0.9, color='#3b5bdb')
    ax.set_xlabel('2θ (°, Cu Kα)')
    ax.set_yticks([])
    ax.set_title(title, fontsize=8)
    fig.tight_layout()
    fig.savefig(base + '.png')
    plt.close(fig)


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    warnings.filterwarnings('ignore')
    os.makedirs(XRD_DIR, exist_ok=True)
    con = sqlite3.connect(DB_PATH)
    try:
        con.execute('ALTER TABLE hunt_structures ADD COLUMN xrd_peaks TEXT')
    except sqlite3.OperationalError:
        pass                                         # column exists
    mats = sys.argv[1:]
    q = ('SELECT id, material, energy, space_group, density, label, cif FROM '
         'hunt_structures WHERE cif IS NOT NULL')
    rows = con.execute(q).fetchall()
    n = 0
    for sid, m, E, sg, rho, label, cif in rows:
        if mats and m not in mats:
            continue
        try:
            pat = pattern(cif)
        except Exception as e:
            print(f'  #{sid} {m}: {e}')
            continue
        top = sorted(zip(pat.y, pat.x, pat.d_hkls), reverse=True)[:10]
        peaks = [{'two_theta': round(x, 2), 'intensity': round(y, 1), 'd': round(d, 3)}
                 for y, x, d in sorted(top, key=lambda t: t[1])]
        save(pat, os.path.join(XRD_DIR, f'{m}_{sid}'),
             f'{m} #{sid} {sg} ρ {rho:.3f} E {E:.1f} kJ/mol [{label}] (simulated, Cu Kα)')
        con.execute('UPDATE hunt_structures SET xrd_peaks=? WHERE id=?',
                    (json.dumps(peaks), sid))
        n += 1
    con.commit()
    print(f'{n} patterns → {XRD_DIR} (peak lists stored in hunt_structures.xrd_peaks)')


if __name__ == '__main__':
    main()
