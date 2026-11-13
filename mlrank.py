# -*- coding: utf-8 -*-
# mlrank.py
#
# MACHINE-LEARNED RE-RANKING  (middle step between force field and DFT)
#
#   force field  → every candidate (thousands, ~1 ms per energy)
#   ML potential → the ML_TOP lowest unique structures (~0.5–2 s per energy)
#   DFT-D        → the few best by ML (dft_rank.py --ml MODEL, minutes to hours each)
#
# Each structure keeps its force-field geometry; only the cell is scaled
# (rigid molecules, same downhill walk + parabola as dft_rank.volume_scan),
# because force-field densities are off by a few per cent.
#
# Test against the saved PBE-D3 results (4 blind-test targets, 80 structures,
# 13 November 2026), rank agreement with DFT (Spearman, 1 = same order):
#   MMFF94 force field 0.34 · mace-mpa-0 + D3 0.65 · MACE-OFF23 0.62 · MACE-OMOL 0.74
# DFT's best structure was always within MACE-OMOL's top 6.
#
# Licences differ: 'mpa0' (MIT) may be used commercially; 'omol' and 'off23'
# are under the Academic Software License (non-commercial use only).
#
# Needs:  pip install mace-torch   (models are downloaded on first use)

import os
import json
import time
from typing import List, Optional

import numpy as np

EV = 96.48533                    # kJ/mol per eV
ML_TOP = 250                     # structures re-ranked by default
HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'db', 'ml_rank')

MODELS = {
    'mpa0':  dict(loader='mp', model='medium-mpa-0', dispersion=True, license='MIT',
                  about='MACE-MPA-0 medium + D3(BJ) dispersion, trained on PBE data'),
    'omol':  dict(loader='omol', model='extra_large', license='ASL (non-commercial)',
                  about='MACE-OMOL extra large, trained on wB97M-V molecular data'),
    'off23': dict(loader='off', model='medium', license='ASL (non-commercial)',
                  about='MACE-OFF23 medium, organic molecules'),
}
DEFAULT_MODEL = 'mpa0'

_calcs = {}


def available() -> bool:
    try:
        import mace.calculators  # noqa: F401
        return True
    except ImportError:
        return False


def calculator(name: str):
    """MACE calculator for a model key (loaded once per process)."""
    if name not in MODELS:
        raise ValueError(f'unknown ML model {name!r}; choose from {", ".join(MODELS)}')
    if name not in _calcs:
        import warnings
        warnings.filterwarnings('ignore', module='e3nn|mace|torch')
        from mace.calculators import mace_mp, mace_off, mace_omol
        spec = MODELS[name]
        kw = dict(model=spec['model'], device='cpu', default_dtype='float64')
        if spec['loader'] == 'mp':
            _calcs[name] = mace_mp(dispersion=spec.get('dispersion', False), **kw)
        elif spec['loader'] == 'omol':
            _calcs[name] = mace_omol(**kw)
        else:
            _calcs[name] = mace_off(**kw)
    return _calcs[name]


def energy(x, model: str = DEFAULT_MODEL, scale: float = 1.0) -> float:
    """ML energy per molecule (kJ/mol) of crystal x with its cell scaled by `scale`."""
    from dft_rank import to_atoms, scaled
    at = to_atoms(scaled(x, scale) if scale != 1.0 else x)
    n_mol = len(at) // (len(x.sym) // x.z)
    at.calc = calculator(model)
    if MODELS[model]['loader'] == 'omol':
        at.info.update(charge=0, spin=1)
    return float(at.get_potential_energy()) * EV / n_mol


def relaxed_energy(x, model: str = DEFAULT_MODEL) -> tuple:
    """(lowest ML energy per molecule over the cell scale, that scale)."""
    from dft_rank import scan_scale
    pts = {}
    def e(f):
        if f not in pts:
            pts[f] = energy(x, model, f)
        return pts[f]
    e_min, f_min, _ = scan_scale(e)
    return e_min, f_min


def rerank(unique: list, model: str = DEFAULT_MODEL, top: int = ML_TOP,
           cache: Optional[str] = None, progress=None, verbose: bool = True) -> list:
    """
    Annotate the `top` lowest force-field structures with s.ml_energy,
    s.ml_scale and s.ml_model (kJ/mol per molecule). The list keeps its
    force-field order; use ml_order() for the ML ranking. `cache` is a JSON
    path (results per index, valid while the landscape does not change).
    """
    import torch
    torch.set_num_threads(os.cpu_count() or 1)
    done = {}
    stamp = [round(s.energy, 6) for s in unique[:top]]
    if cache and os.path.isfile(cache):
        c = json.load(open(cache, encoding='utf-8'))
        if c.get('model') == model and c.get('ff_energies') == stamp:
            done = {int(k): v for k, v in c['results'].items()}
    t0 = time.time()
    n = min(top, len(unique))
    for k in range(n):
        if k not in done:
            done[k] = relaxed_energy(unique[k].crystal, model)
            if cache and (k % 10 == 9 or k == n - 1):
                os.makedirs(os.path.dirname(cache), exist_ok=True)
                json.dump({'model': model, 'ff_energies': stamp,
                           'results': {str(i): v for i, v in done.items()}},
                          open(cache, 'w', encoding='utf-8'))
        unique[k].ml_energy, unique[k].ml_scale = done[k]
        unique[k].ml_model = model
        if progress:
            progress(k + 1, n)
    if verbose:
        print(f'ML re-ranking ({model}: {MODELS[model]["about"]}): {n} structures, '
              f'{time.time() - t0:.0f}s', flush=True)
    return unique


def ml_order(unique: list) -> list:
    """Structures with an ML energy, lowest first, then the rest in force-field order."""
    scored = [s for s in unique if getattr(s, 'ml_energy', None) is not None]
    rest = [s for s in unique if getattr(s, 'ml_energy', None) is None]
    return sorted(scored, key=lambda s: s.ml_energy) + rest


def cache_path(material: str, model: str) -> str:
    return os.path.join(CACHE_DIR, f'{material}_{model}.json')
