# -*- coding: utf-8 -*-
# engine/csp.py
#
# CRYSTAL STRUCTURE PREDICTION RUN
#
#   generate.random_crystal  → symmetric candidate in a real unit cell
#                              (random conformer for flexible molecules)
#   puzzle.puzzle_crystal    → candidate assembled from the molecule's best
#                              pair contacts (generator='puzzle' | 'mixed')
#   lattice.minimise         → L-BFGS lattice-energy minimum (P = 0)
#   invariants.amd           → deduplicate into unique structures
#   rank by lattice energy per molecule → energy landscape
#
# Funnel (molecules with a physical force field):
#   1. screen  every candidate at the cheap 'screen' level (9 Å cutoff,
#              looser convergence)
#   2. refine  the unique structures within REFINE_WINDOW of the screened
#              minimum at the 'full' level
#   3. polish  the lowest unique structures (FLEX_TOP, or all within
#              FLEX_WINDOW, at most FLEX_MAX) with flexible
#              molecules (flex.py: atoms move within the molecule, MMFF
#              intramolecular strain included), then deduplicate again
# Energies include the conformer's strain energy above the gas-phase
# minimum, so structures with different conformers rank fairly.
#
# Candidates are independent, so they run in a process pool.
#
# Usage:
#   python csp.py CO2 --n 100
#   python csp.py aspirin --n 200 --workers 12 --cif out_dir
#   python csp.py caffeine --smiles "CN1C=NC2=C1C(=O)N(C(=O)N2C)C" --n 300

import os
import sys
import time
import argparse
import numpy as np
from dataclasses import dataclass, field, replace
from typing import List, Optional
from multiprocessing import get_context

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from molecules import get_molecule
from generate import random_crystal, molecular_volume, _min_contact_ratio
from lattice import Crystal, minimise, space_group_of
from forcefield import has_forcefield, get_ff_molecule
from invariants import amd_normalised, amd_distance

DEDUP_TOL = 0.02        # normalised AMD distance below which = same structure
MAX_EXPANSION = 2.0     # candidates generate loose; real minima shrink, not grow
MIN_FINAL_CONTACT = 0.5 # reject minima with atoms closer than 0.5 × contact
REFINE_WINDOW = 15.0    # kJ/mol above the screened minimum → refined at 'full'
REFINE_MIN = 10         # always refine at least this many (if available)
SCREEN_TOL = 1e-4       # screening convergence (full level uses 1e-6)
SCREEN_LEVEL = os.environ.get('ALLOYFOLD_SCREEN', 'fast')   # 'fast' = DSF Coulomb, 'screen' = Ewald
WORKER_RAM_GB = 2.5     # budget per worker (observed peak 2.4 GB in a benchmark)
FLEX_TOP = 30           # at least this many lowest unique structures re-relaxed flexibly,
FLEX_WINDOW = 12.0      # ... and every one within this many kJ/mol of the minimum
FLEX_MAX = 100          # ... up to this many (flexing lowers energies by 1-10 kJ/mol, so a
                        # cut by count left near-lying rigid copies of real forms ranked too high)


def load_molecule(material_key: str, physical: bool = True, conformer: int = 0):
    """(Molecule for the generator, force-field key or None)."""
    if physical and has_forcefield(material_key):
        return get_ff_molecule(material_key).to_molecule(conformer), material_key
    return get_molecule(material_key), None


@dataclass
class Structure:
    crystal: Crystal
    energy: float                 # lattice energy (+ conformer strain) per molecule
    amd: np.ndarray
    space_group: str              # found by spglib after minimisation
    count: int = 1                # how many candidates fell into it
    generated_in: List[str] = field(default_factory=list)

    def summary(self, e_min: float) -> dict:
        a, b, c, al, be, ga = self.crystal.cell_params()
        return {
            'energy': round(self.energy, 4),
            'dE': round(self.energy - e_min, 4),
            'space_group': self.space_group,
            'z': self.crystal.z,
            'density': round(self.crystal.density, 3),
            'cell': [round(v, 3) for v in (a, b, c, al, be, ga)],
            'count': self.count,
            'conformer': self.crystal.meta.get('conformer', 0),
            'generated_in': sorted(set(self.generated_in)),
        }


def default_workers() -> int:
    """CPU cores minus 2, capped so workers use at most half the free RAM."""
    n = max(1, (os.cpu_count() or 2) - 2)
    try:
        import psutil
        free_gb = psutil.virtual_memory().available / 1e9
        n = min(n, max(1, int(0.5 * free_gb / WORKER_RAM_GB)))
    except ImportError:
        n = min(n, 4)
    return n


def _init_worker():
    import torch
    torch.set_num_threads(1)


def _finish(y: Crystal) -> Optional[Crystal]:
    """Reject failed minima; add the conformer strain energy."""
    if not np.isfinite(y.energy) or y.energy >= 0:
        return None                   # stuck in overlap / collapsed
    if y.meta['volume_ratio'] > MAX_EXPANSION:
        return None                   # flew apart into layers + vacuum
    if _min_contact_ratio(y) < MIN_FINAL_CONTACT:
        return None                   # atoms fused: numerical artefact
    return replace(y, energy=y.energy + y.meta.get('e_conf', 0.0))


def _polish(x: Crystal) -> Optional[Crystal]:
    """Flexible-molecule relaxation of a refined minimum (flex.py)."""
    from flex import flex_minimise
    try:
        y = flex_minimise(x)
    except Exception:
        return None
    if not np.isfinite(y.energy) or _min_contact_ratio(y) < MIN_FINAL_CONTACT:
        return None
    return y


def _candidate(args) -> Optional[Crystal]:
    material_key, seed, v_mol, space_groups, physical, level = args[:6]
    generator = args[6] if len(args) > 6 else 'random'
    morph = args[7] if len(args) > 7 else False
    coarse = args[8] if len(args) > 8 else False
    rng = np.random.default_rng(seed)
    conf, e_conf = 0, 0.0
    if physical and has_forcefield(material_key):
        ffm = get_ff_molecule(material_key)
        conf = int(rng.integers(ffm.n_conformers))
        e_conf = ffm.conformer(conf)[1]
    mol, ff = load_molecule(material_key, physical, conf)
    sg = str(rng.choice(space_groups)) if space_groups else None
    z_prime = int(rng.integers(1, 5)) if mol.category == 'metal' else 1
    x = None
    made_by = 'random'
    if generator == 'puzzle' or (generator == 'mixed' and rng.random() < 0.5):
        from puzzle import puzzle_crystal, contact_library
        x = puzzle_crystal(mol, contact_library(material_key)[conf], rng,
                           space_group=sg, v_mol=v_mol)
        made_by = 'puzzle'
    if x is None:                     # no puzzle solution (or random generator)
        made_by = 'random'
        for _ in range(20):           # redraw until it fits without heavy inflation
            x = random_crystal(mol, rng, space_group=sg, z_prime=z_prime, v_mol=v_mol)
            if x is not None:
                break
    if x is None:
        return None
    x = replace(x, ff=ff, meta={**x.meta, 'level': level, 'generator': made_by,
                                'morph': bool(morph and made_by == 'random'),
                                'conformer': conf, 'e_conf': e_conf})
    try:
        if coarse:                    # pre-pack: sphere → beads (coarse.py)
            from coarse import coarse_minimise
            x, _ = coarse_minimise(x)
            if x is None:
                return None
        y = minimise(x, tol=SCREEN_TOL if level != 'full' else 1e-6)
    except Exception:
        return None
    return _finish(y)


def _refine(x: Crystal) -> Optional[Crystal]:
    try:
        y = minimise(replace(x, meta={**x.meta, 'level': 'full'}))
    except Exception:
        return None
    return _finish(y)


def deduplicate(crystals: List[Crystal], tol: float = DEDUP_TOL) -> List[Structure]:
    """Group minimised crystals into unique structures, lowest energy first.
    A crystal carrying meta['hits'] (screening duplicates) counts that many."""
    unique: List[Structure] = []
    for x in sorted(crystals, key=lambda c: c.energy):
        fp = amd_normalised(x.cell, x.positions)
        hits = x.meta.get('hits', 1)
        for s in unique:
            if amd_distance(fp, s.amd) < tol:
                s.count += hits
                s.generated_in.append(x.space_group)
                break
        else:
            unique.append(Structure(x, x.energy, fp, space_group_of(x), count=hits,
                                    generated_in=[x.space_group]))
    return unique


def run_csp(material_key: str, n_candidates: int = 100,
            space_groups: Optional[List[str]] = None,
            workers: Optional[int] = None, seed: int = 0,
            verbose: bool = True, physical: bool = True,
            progress=None, funnel: Optional[bool] = None,
            flex: Optional[bool] = None, generator: str = 'random',
            morph: bool = False, coarse: bool = False) -> List[Structure]:
    """
    physical=True uses the forcefield.py model (kJ/mol) when one exists
    for the material, else the toy pair physics. funnel (default: on for
    molecules with a force field) screens cheaply, then refines; flex
    (default: on for MMFF molecules) polishes the best with flexible molecules.
    generator: 'random' (generate.py), 'puzzle' (puzzle.py, contact assembly;
    falls back to random when no combination fits) or 'mixed' (half each).
    morph: random candidates start as spheres that grow into the molecule's
    shape while electrostatics already orients them (lattice.minimise).
    coarse: candidates are pre-packed as a sphere with charge sites, then
    4 and 10 beads (coarse.py), before the atomistic minimisation.
    progress(done, total) is called as candidates finish.
    """
    mol, ff = load_molecule(material_key, physical)
    if funnel is None:
        funnel = ff is not None and mol.category != 'metal'
    if flex is None:
        flex = ff is not None and get_ff_molecule(ff).form == 'mmff'
    v_mol = molecular_volume(mol)
    workers = workers or default_workers()
    level = SCREEN_LEVEL if funnel else 'full'
    if generator != 'random':
        if ff is None or get_ff_molecule(ff).form not in ('mmff', 'lj'):
            generator = 'random'      # contacts need a molecular force field
        else:
            from puzzle import contact_library
            contact_library(material_key)      # build once; workers load the cache
    if coarse and ff is not None and get_ff_molecule(ff).form == 'mmff':
        from coarse import coarse_levels
        coarse_levels(material_key)            # fitted once here (workers refit per conformer)
    else:
        coarse = False
    jobs = [(material_key, seed * 1_000_003 + i, v_mol, space_groups, physical, level, generator, morph, coarse)
            for i in range(n_candidates)]

    t0 = time.time()
    # spawn on every OS (Windows has no fork; forking the threaded web server
    # on Linux could copy held locks into the workers)
    with get_context('spawn').Pool(workers, initializer=_init_worker) as pool:
        results = []
        for r in pool.imap_unordered(_candidate, jobs, chunksize=1):
            results.append(r)
            if progress:
                progress(len(results), n_candidates)
        crystals = [r for r in results if r is not None]
        n_screened = len(crystals)

        if funnel and crystals:
            pre = deduplicate(crystals)
            e0 = pre[0].energy
            chosen = [s for k, s in enumerate(pre)
                      if k < REFINE_MIN or s.energy <= e0 + REFINE_WINDOW]
            todo = [replace(s.crystal, meta={**s.crystal.meta, 'hits': s.count})
                    for s in chosen]
            refined = []
            for r in pool.imap_unordered(_refine, todo, chunksize=1):
                refined.append(r)
                if progress:
                    progress(n_candidates + len(refined), n_candidates + len(todo))
            crystals = [r for r in refined if r is not None]
        unique = deduplicate(crystals)
        if flex and unique:
            k = max(FLEX_TOP, sum(s.energy <= unique[0].energy + FLEX_WINDOW for s in unique))
            k = min(k, FLEX_MAX)
            top = [replace(s.crystal, meta={**s.crystal.meta, 'hits': s.count})
                   for s in unique[:k]]
            polished = [y if y is not None else x for x, y in
                        zip(top, pool.map(_polish, top, chunksize=1))]
            unique = deduplicate(polished + [s.crystal for s in unique[k:]])

    if verbose:
        unit = 'kJ/mol' if ff else 'model units'
        stage = (f', {len(crystals)} refined at full accuracy' if funnel else '') + \
            (', best relaxed with flexible molecules' if flex else '')
        print(f'{mol.name}: {n_screened}/{n_candidates} candidates minimised{stage}, '
              f'{len(unique)} unique structures, {time.time() - t0:.0f}s '
              f'({workers} workers, energies in {unit})')
        print_landscape(unique)
    return unique


def print_landscape(unique: List[Structure], top: int = 15):
    if not unique:
        return
    e_min = unique[0].energy
    print(f'  {"#":>3} {"E/mol":>10} {"ΔE":>8} {"found as":<16} {"Z":>2} '
          f'{"ρ g/cm³":>7} {"hits":>4}  cell a b c α β γ')
    for k, s in enumerate(unique[:top]):
        d = s.summary(e_min)
        cell = ' '.join(f'{v:.2f}' if i < 3 else f'{v:.0f}'
                        for i, v in enumerate(d['cell']))
        print(f'  {k + 1:>3} {d["energy"]:>10.3f} {d["dE"]:>8.3f} '
              f'{d["space_group"]:<16} {d["z"]:>2} {d["density"]:>7.3f} '
              f'{d["count"]:>4}  {cell}')


def write_cifs(unique: List[Structure], out_dir: str, top: int = 10):
    from ase.io import write
    os.makedirs(out_dir, exist_ok=True)
    for k, s in enumerate(unique[:top]):
        write(os.path.join(out_dir, f'rank{k + 1:02d}_E{s.energy:.3f}.cif'),
              s.crystal.to_ase(), format='cif')


if __name__ == '__main__':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    ap = argparse.ArgumentParser(description='AlloyFold crystal structure prediction')
    ap.add_argument('material', help='a built-in key (CO2, aspirin, …) or, with --smiles, a name for your molecule')
    ap.add_argument('--smiles', default=None, help='your own molecule (neutral, one fragment, MMFF94 elements)')
    ap.add_argument('--n', type=int, default=100, help='candidates')
    ap.add_argument('--workers', type=int, default=None)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--sg', nargs='*', default=None, help='restrict space groups')
    ap.add_argument('--cif', default=None, help='write top CIFs to this folder')
    ap.add_argument('--toy', action='store_true', help='use the toy pair physics')
    ap.add_argument('--no-funnel', action='store_true', help='minimise all at full accuracy')
    ap.add_argument('--rigid', action='store_true', help='skip the flexible-molecule polish')
    ap.add_argument('--generator', default='random', choices=['random', 'puzzle', 'mixed'])
    ap.add_argument('--morph', action='store_true', help='sphere-to-shape start for random candidates')
    ap.add_argument('--coarse', action='store_true', help='pre-pack through sphere / bead levels')
    a = ap.parse_args()
    if a.smiles:
        from forcefield import register_user_molecule
        try:
            a.material = register_user_molecule(a.material, a.smiles)
        except ValueError as e:
            sys.exit(str(e))
        print(f'molecule registered as {a.material}', flush=True)
    res = run_csp(a.material, a.n, a.sg, a.workers, a.seed, physical=not a.toy,
                  funnel=False if a.no_funnel else None, flex=False if a.rigid else None,
                  generator=a.generator, morph=a.morph, coarse=a.coarse)
    if a.cif:
        write_cifs(res, a.cif)
