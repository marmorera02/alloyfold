# -*- coding: utf-8 -*-
# engine/calibrate.py
#
# CRYSTAL CORRECTION FOR MMFF94 — fitted on experimental crystal structures
#
# MMFF94 is a gas-phase force field: on hydrocarbon crystals it gives ~78%
# of the lattice energy at ~90% of the density. This fits four scale
# factors on its buffered 14-7 van der Waals term:
#     eps, r        C/H-C/H pairs
#     eps_x, r_x    pairs involving O or N (H-bond donor-acceptor pairs keep
#                   MMFF's own parameters)
# (one shared pair over-compressed acetic acid by 8%; C/H-only left
# aspirin 10% too loose — the two pair classes need their own factors).
#
# Data: experimental structures from the Crystallography Open Database
# (references.py), each rebuilt with the force field's own rigid molecule
# and minimised; the minimised density and lattice energy are compared to
# experiment.
#   training    benzene, naphthalene, anthracene, acetic acid, aspirin
#   validation  paracetamol forms I, II, III — never used in the fit;
#               experiment: form I is the most stable at 0 K
#
# Density targets: the minimisation is at 0 K; references measured at
# temperature T expand by THERMAL(T).
#
# Writes mmff_crystal_calibration.json, which forcefield.py loads.
#
# Usage:  python calibrate.py              (~30-60 min)

import os
import sys
import json
import time
import datetime
import numpy as np
from dataclasses import replace
from multiprocessing import Pool

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import forcefield
from csp import _init_worker, _finish
from lattice import minimise, space_group_of
from references import COD_REFERENCES, experimental_crystal

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, 'mmff_crystal_calibration.json')

# material: experimental lattice energy kJ/mol (≈ -(ΔH_sub + 2RT)) or None
TRAIN = {'benzene': -49, 'naphthalene': -77, 'anthracene': -105,
         'acetic_acid': None, 'aspirin': -113}
VALIDATE = 'paracetamol'
KEYS = ('eps', 'r', 'eps_x', 'r_x')


def thermal(T) -> float:
    """0 K density / measured density (organic crystals expand ~2% to RT)."""
    if T is None or T >= 250:
        return 1.02
    return 1.0 + 0.02 * T / 295.0


def _relax(args):
    """Minimise one experimental crystal with the given scales (worker)."""
    crystal, scale = args
    forcefield.MMFF_SCALE.update(scale)
    return _finish(minimise(replace(crystal, meta={**crystal.meta, 'level': 'full'})))


def load(materials):
    out = {}
    for m in materials:
        for label, cod_id, T in COD_REFERENCES[m]:
            x = experimental_crystal(m, cod_id)
            out[(m, label)] = (x, x.density * thermal(T))
    return out


def evaluate(pool, crystals, scale):
    keys = list(crystals)
    ys = pool.map(_relax, [(crystals[k][0], scale) for k in keys])
    return dict(zip(keys, ys))


def main():
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    train = load(TRAIN)
    train = {k: v for k, v in train.items() if k[1] == 'form I'}
    valid = load([VALIDATE])
    print('training:', ', '.join(m for m, _ in train), '| validation:',
          ', '.join(f'{m} {l}' for m, l in valid), flush=True)

    from scipy.optimize import minimize as nm
    history = []
    t0 = time.time()
    with Pool(len(train), initializer=_init_worker) as pool:
        def report(scale, ys, crystals):
            rows = {}
            for k, y in ys.items():
                x, rho_target = crystals[k]
                if y is None:
                    rows[k] = (None, None, None)
                    continue
                e_exp = TRAIN.get(k[0])
                rows[k] = (y.density / rho_target,
                           y.energy / e_exp if e_exp else None, y.energy)
            return rows

        def loss(v):
            scale = dict(zip(KEYS, map(float, v)))
            ys = evaluate(pool, train, scale)
            rows = report(scale, ys, train)
            L = 0.0
            for k, (dr, er, _) in rows.items():
                if dr is None:
                    return 10.0                      # minimisation failed
                L += (dr - 1) ** 2 + ((er - 1) ** 2 if er else 0.0)
            for k, y in ys.items():                 # warm start
                if y is not None:
                    train[k] = (y, train[k][1])
            history.append((scale, rows, L))
            print('  ' + ' '.join(f'{k}×{v:.3f}' for k, v in scale.items()) +
                  f'  loss {L:.4f}  ' + '  '.join(
                      f'{m}: ρ {dr:.3f}' + (f' E {er:.3f}' if er else '')
                      for (m, _), (dr, er, _) in rows.items()), flush=True)
            return L

        x0 = [forcefield.MMFF_SCALE[k] for k in KEYS]
        simplex = [x0] + [[x0[j] * (1.15 if j == i and i % 2 == 0 else
                                     0.97 if j == i else 1.0) for j in range(4)]
                          for i in range(4)]
        nm(loss, x0=x0, method='Nelder-Mead',
           options={'initial_simplex': simplex, 'xatol': 0.005,
                    'fatol': 2e-4, 'maxiter': 80})
        best = min(history, key=lambda h: h[2])

        # validation: paracetamol polymorphs, never seen by the fit
        ys = evaluate(pool, valid, best[0])
    val = {}
    for (m, label), y in ys.items():
        x, rho_target = valid[(m, label)]
        val[label] = {'density_ratio': round(y.density / rho_target, 3) if y else None,
                      'energy': round(y.energy, 2) if y else None,
                      'space_group': space_group_of(y) if y else None}
    e_val = {l: v['energy'] for l, v in val.items() if v['energy'] is not None}
    order = sorted(e_val, key=e_val.get)

    result = {
        'scale': best[0], 'loss': round(best[2], 5),
        'training': {f'{m} {l}': {'density_ratio': round(dr, 3),
                                  'energy_ratio': round(er, 3) if er else None,
                                  'energy': round(e, 2)}
                     for (m, l), (dr, er, e) in best[1].items()},
        'validation': val,
        'validation_energy_order': order,
        'validation_note': 'experiment: paracetamol form I most stable',
        'method': 'COD experimental structures, rigid MMFF94 conformers, '
                  'density (thermal-corrected) + lattice energy',
        'fitted': datetime.datetime.now().isoformat(timespec='seconds'),
    }
    with open(OUT, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print('\nbest: ' + ' '.join(f'{k}×{v:.3f}' for k, v in best[0].items()) +
          f'  (loss {best[2]:.4f}, {time.time() - t0:.0f}s)')
    print('validation (paracetamol): ' + '  '.join(
        f'{l}: ρ {v["density_ratio"]} E {v["energy"]} {v["space_group"]}'
        for l, v in val.items()))
    print('validation energy order (lowest first):', ' < '.join(order),
          '| experiment: form I most stable')


if __name__ == '__main__':
    main()
