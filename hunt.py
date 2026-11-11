# -*- coding: utf-8 -*-
# engine/hunt.py
#
# POLYMORPH HUNTER — open-ended crystal structure search
#
# Cycles through materials, runs a fresh-seeded CSP batch for each and
# merges the results into a persistent energy landscape per material
# (SQLite table hunt_structures in db/alloyfold.db, CIFs in hunt_results/).
#
# Each new structure within NOVEL_WINDOW kJ/mol of the material's lowest
# energy is
#   1. compared with known experimental forms (KNOWN_FORMS, reduced cell
#      + space group) → labelled 'known: <form>' or 'candidate'
#   2. checked for robustness: rattled (atoms ±0.15 Å via molecule moves,
#      cell ±3%), re-minimised; it counts as a real minimum only if it
#      falls back into the same structure (AMD).
#
# 'candidate' means: low-energy, robust, and not one of the forms listed
# here. Whether it is truly unknown needs a check against the Cambridge
# Structural Database, which this tool cannot access.
#
# Prints one line per event (CYCLE / NEW / CANDIDATE / KNOWN); runs until
# stopped (Ctrl+C or killing the process). Safe to restart: the landscape
# and the seed counter persist.
#
# Usage:  python hunt.py                       (default material rotation)
#         python hunt.py aspirin paracetamol --batch 300 --workers 16

import os
import sys
import json
import time
import sqlite3
import argparse
import numpy as np
from dataclasses import replace

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from csp import run_csp, DEDUP_TOL, _finish
from lattice import minimise, cell_from_params, space_group_of, primitive_cell
from invariants import amd_normalised, amd_distance

HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(HERE, 'db', 'alloyfold.db')
CIF_DIR = os.path.join(HERE, 'hunt_results')
NOVEL_WINDOW = 5.0      # kJ/mol above the landscape minimum worth examining
CELL_TOL = 0.07

DEFAULT_MATERIALS = ['aspirin', 'paracetamol', 'acetic_acid', 'benzene',
                     'naphthalene', 'anthracene', 'caffeine', 'urea', 'pyrazine']

# Known experimental forms: (label, space group, a, b, c, alpha, beta, gamma)
# Approximate literature cells; compared via Minkowski-reduced lengths.
KNOWN_FORMS = {
    'aspirin': [('form I', 'P2_1/c', 11.43, 6.59, 11.40, 90, 95.7, 90),
                ('form II', 'P2_1/c', 12.10, 6.49, 11.34, 90, 111.6, 90)],
    'paracetamol': [('form I', 'P2_1/c', 7.10, 9.37, 11.66, 90, 97.4, 90),
                    ('form II', 'Pbca', 17.17, 11.78, 7.21, 90, 90, 90)],
    'benzene': [('form I', 'Pbca', 7.39, 9.42, 6.81, 90, 90, 90)],
    'naphthalene': [('form I', 'P2_1/c', 8.235, 6.003, 8.658, 90, 122.9, 90)],
    'anthracene': [('form I', 'P2_1/c', 8.56, 6.04, 11.16, 90, 124.7, 90)],
    'acetic_acid': [('form I', 'Pna2_1', 13.15, 3.92, 5.76, 90, 90, 90)],
    'caffeine': [],          # anhydrous forms are disordered (no ordered reference)
    'urea': [('form I', 'P-42_1m', 5.57, 5.57, 4.69, 90, 90, 90)],
    'pyrazine': [('form I', 'Pnnm', 9.31, 5.85, 3.73, 90, 90, 90)],  # = Pmnn
}


def _db():
    con = sqlite3.connect(DB_PATH)
    con.executescript('''
        CREATE TABLE IF NOT EXISTS hunt_structures (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            material    TEXT NOT NULL,
            energy      REAL NOT NULL,
            density     REAL,
            space_group TEXT,
            z           INTEGER,
            cell        TEXT,
            amd         TEXT NOT NULL,
            hits        INTEGER DEFAULT 1,
            label       TEXT,
            robust      INTEGER,
            cif         TEXT,
            found_at    REAL,
            cycle       INTEGER
        );
        CREATE TABLE IF NOT EXISTS hunt_state (
            material    TEXT PRIMARY KEY,
            cycles      INTEGER DEFAULT 0,
            candidates  INTEGER DEFAULT 0
        );
        CREATE TABLE IF NOT EXISTS hunt_progress (
            id          INTEGER PRIMARY KEY CHECK (id = 1),
            material    TEXT,
            cycle       INTEGER,
            done        INTEGER,
            total       INTEGER,
            stage       TEXT,
            started_at  REAL,
            updated_at  REAL
        );
        CREATE TABLE IF NOT EXISTS hunt_log (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            t           REAL,
            line        TEXT
        );
    ''')
    return con


def _progress(con, material, cycle, done, total, stage, started):
    """Live progress row for the dashboard (single row, id = 1)."""
    con.execute('INSERT INTO hunt_progress (id, material, cycle, done, total, stage, '
                'started_at, updated_at) VALUES (1,?,?,?,?,?,?,?) '
                'ON CONFLICT(id) DO UPDATE SET material=?, cycle=?, done=?, total=?, '
                'stage=?, started_at=?, updated_at=?',
                (material, cycle, done, total, stage, started, time.time()) * 2)
    con.commit()


def _event(con, line):
    """Print an event and keep it in the dashboard's live log."""
    print(line, flush=True)
    con.execute('INSERT INTO hunt_log (t, line) VALUES (?,?)', (time.time(), line))
    con.commit()


def _reduced_lengths(cell) -> np.ndarray:
    from ase.geometry import minkowski_reduce
    rc, _ = minkowski_reduce(np.asarray(cell, float))
    return np.sort(np.linalg.norm(np.asarray(rc, float), axis=1))


_EXP_MINIMA = {}


def _experimental_minima(material: str):
    """[(label, AMD, energy)] of the COD forms re-minimised with the current FF."""
    if material not in _EXP_MINIMA:
        from references import COD_REFERENCES, experimental_crystal
        out = []
        for label, cod_id, _ in COD_REFERENCES.get(material, []):
            try:
                y = _finish(minimise(experimental_crystal(material, cod_id)))
            except Exception:
                y = None
            if y is not None:
                out.append((f'{label} (COD {cod_id})',
                            amd_normalised(y.cell, y.positions), y.energy))
        _EXP_MINIMA[material] = out
    return _EXP_MINIMA[material]


def known_form(material: str, structure) -> str:
    # rigorous: same basin as an experimental form minimised with the same FF
    for label, fp, _ in _experimental_minima(material):
        if amd_distance(fp, structure.amd) < DEDUP_TOL:
            return label
    pred = _reduced_lengths(primitive_cell(structure.crystal))
    for label, sg, *params in KNOWN_FORMS.get(material, []):
        if not structure.space_group.startswith(sg + ' '):
            continue
        exp = _reduced_lengths(cell_from_params(*params))
        if np.all(np.abs(pred / exp - 1) < CELL_TOL):
            # cell + space group only — verify_finds.py showed this can be a
            # different packing, so it is reported as unconfirmed
            return f'{label}? (cell-like, unconfirmed)'
    return ''


def is_robust(structure, trials: int = 2, rng=None) -> bool:
    """Rattle + strain, re-minimise; robust if it returns to itself."""
    rng = rng or np.random.default_rng()
    x = structure.crystal
    for _ in range(trials):
        strain = np.eye(3) + rng.normal(0, 0.03, (3, 3))
        cell = x.cell @ strain
        frac = x.com @ np.linalg.inv(x.cell)
        com = frac @ cell + rng.normal(0, 0.15, x.com.shape)
        y = _finish(minimise(replace(x, cell=cell, com=com,
                                     meta={**x.meta, 'level': 'full'})))
        if y is None:
            return False
        if amd_distance(amd_normalised(y.cell, y.positions), structure.amd) > DEDUP_TOL:
            return False
    return True


def _cif(crystal) -> str:
    import io
    from ase.io import write
    buf = io.BytesIO()
    write(buf, crystal.to_ase(), format='cif')
    return buf.getvalue().decode('utf-8')


def hunt_cycle(con, material: str, batch: int, workers: int) -> None:
    row = con.execute('SELECT cycles, candidates FROM hunt_state WHERE material=?',
                      (material,)).fetchone()
    cycle = (row[0] if row else 0) + 1
    # fresh, reproducible seeds per material and cycle (str hash() is salted)
    seed = 7919 * cycle + sum(map(ord, material)) * 101
    t0 = time.time()
    last = [0.0]

    def progress(done, total):
        if time.time() - last[0] > 2 or done == total:      # throttle DB writes
            last[0] = time.time()
            stage = 'screening' if done <= batch else 'refining'
            _progress(con, material, cycle, done, total, stage, t0)

    _progress(con, material, cycle, 0, batch, 'starting', t0)
    unique = run_csp(material, n_candidates=batch, workers=workers, seed=seed,
                     verbose=False, progress=progress)
    _progress(con, material, cycle, batch, batch, 'merging', t0)
    con.execute('INSERT INTO hunt_state (material, cycles, candidates) VALUES (?,?,?) '
                'ON CONFLICT(material) DO UPDATE SET cycles=?, candidates=candidates+?',
                (material, cycle, batch, cycle, batch))

    stored = con.execute('SELECT id, energy, amd FROM hunt_structures WHERE material=?',
                         (material,)).fetchall()
    stored = [(i, e, np.array(json.loads(a))) for i, e, a in stored]
    # baseline: the lowest of everything found AND the known experimental
    # forms (minimised with the same FF) — a structure is only 'low' if it
    # is close to the true minimum, even when the search missed that form
    known_e = [e for _, _, e in _experimental_minima(material)]
    e_min = min([e for _, e, _ in stored] + [s.energy for s in unique] + known_e or [0.0])
    n_new = 0
    for s in unique:
        match = next((i for i, _, a in stored if amd_distance(a, s.amd) < DEDUP_TOL), None)
        if match is not None:
            con.execute('UPDATE hunt_structures SET hits = hits + ? WHERE id=?',
                        (s.count, match))
            continue
        n_new += 1
        dE = s.energy - e_min
        label, robust, cif = '', None, None
        if dE <= NOVEL_WINDOW:
            label = known_form(material, s)
            robust = is_robust(s)
            label = f'known: {label}' if label else ('candidate' if robust else 'shallow')
            cif = _cif(s.crystal)
            os.makedirs(os.path.join(CIF_DIR, material), exist_ok=True)
            with open(os.path.join(CIF_DIR, material,
                                   f'{material}_E{s.energy:.2f}_{s.space_group.split()[0].replace("/", "_")}.cif'),
                      'w', encoding='utf-8') as f:
                f.write(cif)
        a, b, c, al, be, ga = s.crystal.cell_params()
        con.execute('INSERT INTO hunt_structures (material, energy, density, space_group, z, '
                    'cell, amd, hits, label, robust, cif, found_at, cycle) '
                    'VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)',
                    (material, s.energy, s.crystal.density, s.space_group, s.crystal.z,
                     json.dumps([round(v, 3) for v in (a, b, c, al, be, ga)]),
                     json.dumps(s.amd.tolist()), s.count, label,
                     None if robust is None else int(robust), cif, time.time(), cycle))
        stored.append((None, s.energy, s.amd))
        if dE <= NOVEL_WINDOW:
            tag = 'CANDIDATE' if label == 'candidate' else ('KNOWN' if label.startswith('known') else 'NEW')
            _event(con, f'{tag} {material} E={s.energy:.2f} kJ/mol dE={dE:.2f} '
                        f'{s.space_group} Z={s.crystal.z} rho={s.crystal.density:.3f} '
                        f'cell={a:.2f},{b:.2f},{c:.2f},{al:.0f},{be:.0f},{ga:.0f} [{label}]')
    con.commit()
    total = con.execute('SELECT COUNT(*) FROM hunt_structures WHERE material=?',
                        (material,)).fetchone()[0]
    _event(con, f'CYCLE {material} #{cycle}: {batch} candidates, {len(unique)} unique, '
                f'{n_new} new to landscape (total {total}), {time.time() - t0:.0f}s')


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='AlloyFold polymorph hunter (runs until stopped)')
    ap.add_argument('materials', nargs='*', default=DEFAULT_MATERIALS)
    ap.add_argument('--batch', type=int, default=200)
    ap.add_argument('--workers', type=int, default=None)
    a = ap.parse_args()
    con = _db()
    _event(con, f'HUNT start: {", ".join(a.materials)} (batch {a.batch})')
    while True:
        for m in a.materials:
            try:
                hunt_cycle(con, m, a.batch, a.workers)
            except Exception as e:
                _event(con, f'ERROR {m}: {e}')


if __name__ == '__main__':
    main()
