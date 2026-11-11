# -*- coding: utf-8 -*-
# engine/puzzle.py
#
# PUZZLE ASSEMBLY — crystals built from the molecule's best pair contacts
#
# The random generator (generate.py) draws cell and pose blindly; most
# candidates start far from any good packing. Here the molecule's own
# "edges" guide the build:
#
#   1. CONTACT LIBRARY  For each conformer, the strongest two-molecule
#      arrangements in which the partner is a symmetry image of the first:
#        inv    partner = inversion through a point        (-I)
#        rot2   partner = 2-fold rotation / 2₁ screw       (2nnᵀ - I)
#        mir    partner = mirror / glide reflection        (I - 2nnᵀ)
#        trans  partner = translation                      (I)
#      Each contact is a rigid motion x → x Qᵀ + q in the molecule frame,
#      a local minimum of the dimer energy (same force field as the
#      crystal). These are the puzzle pieces' matching edges: H-bonded
#      dimers, stacks, chains (the "synthons" of crystal engineering).
#
#   2. ASSEMBLY  Pick a space group and 1-3 contacts, assign each to a
#      symmetry operation of the same kind, and solve for the cell and the
#      molecule's pose in which every operation reproduces its contact
#      (least squares on atom positions). A small residual means the
#      contacts fit together in that group — a solved puzzle; anything
#      the contacts leave open stays at its random start value.
#
# Output: a lattice.Crystal like generate.random_crystal, for the same
# minimiser. csp.run_csp(generator='puzzle' | 'mixed').

import os
import math
import pickle
import numpy as np
import torch
from functools import lru_cache
from typing import Optional

from lattice import Crystal
from molecules import lj_radius
from spacegroups import operations, crystal_system, sampling_weights
from generate import random_cell, random_rotation, _min_contact_ratio, MIN_CONTACT

HERE = os.path.dirname(os.path.abspath(__file__))
CACHE_DIR = os.path.join(HERE, 'db', 'contacts')

N_STARTS = 20           # dimer minimisations per contact kind and conformer
KEEP_PER_KIND = 6       # contacts kept per kind
KEEP_FRACTION = 0.35    # ... and only those with at least this share of the best energy
SAME_CONTACT = 0.35     # Å: image RMSD below which two contacts are the same
INTRINSIC = 0.6         # Å: screw / glide shift above which an operation is not "pure"
FIT_TOL = 0.35          # Å: residual (rms over atoms) of an accepted puzzle solution
MAX_TRIES = 40          # assembly attempts per candidate
KT_PICK = 8.0           # kJ/mol: contacts are picked with weight exp(-ΔE / KT_PICK)


# ─────────────────────────────────────────────────────────────
# 1. CONTACT LIBRARY
# ─────────────────────────────────────────────────────────────

def _pair_tables(ffm):
    """(R*, eps, qq) matrices between two copies of the molecule, with the
    crystal calibration of forcefield.FFModel."""
    from forcefield import MMFF_SCALE
    m = len(ffm.symbols)
    r_s, e_s = np.ones((m, m)), np.ones((m, m))
    if ffm.form == 'mmff':
        ch = np.array([s in ('C', 'H') for s in ffm.symbols])
        free = ~ffm.hbond_pairs
        both = ch[:, None] & ch[None, :]
        r_s[free & both], e_s[free & both] = MMFF_SCALE['r'], MMFF_SCALE['eps']
        r_s[free & ~both], e_s[free & ~both] = MMFF_SCALE['r_x'], MMFF_SCALE['eps_x']
    q = np.asarray(ffm.charges)
    return (torch.as_tensor(ffm.pa * r_s), torch.as_tensor(ffm.pb * e_s),
            torch.as_tensor(q[:, None] * q[None, :]))


def _dimer_energy(a, b, tables, form):
    """Interaction energy (kJ/mol) of two isolated molecules a, b (m,3)."""
    from forcefield import COULOMB, CHARGE_WIDTH
    R, e, qq = tables
    d = a[:, None, :] - b[None, :, :]
    r = torch.sqrt((d * d).sum(-1) + 1e-12)
    if form == 'lj':
        sr6 = torch.where(e > 0, (R / r) ** 6, torch.zeros_like(r))
        vdw = 4 * e * (sr6 * sr6 - sr6)
    else:                                   # MMFF94 buffered 14-7
        vdw = e * (1.07 * R / (r + 0.07 * R)) ** 7 * \
            (1.12 * R ** 7 / (r ** 7 + 0.12 * R ** 7) - 2.0)
    return vdw.sum() + COULOMB * (qq * torch.erf(r / CHARGE_WIDTH) / r).sum()


def _motion(kind, v, q):
    """Q (3,3) of a contact kind with axis / normal direction v."""
    eye = torch.eye(3, dtype=q.dtype)
    if kind == 'inv':
        return -eye
    if kind == 'trans':
        return eye
    n = v / torch.sqrt((v * v).sum() + 1e-30)
    nn = n[:, None] * n[None, :]
    return 2 * nn - eye if kind == 'rot2' else eye - 2 * nn


def _same(img_a, img_b, sym):
    """Two partner images describe the same dimer (atoms matched per element)."""
    from scipy.spatial import cKDTree
    worst = 0.0
    for el in set(sym):
        k = [i for i, s in enumerate(sym) if s == el]
        d, _ = cKDTree(img_b[k]).query(img_a[k])
        worst = max(worst, float(d.max()))
    return worst < SAME_CONTACT


def _contacts_for(body, ffm, rng):
    tables = _pair_tables(ffm)
    a = torch.as_tensor(body)
    r_mol = float(np.linalg.norm(body, axis=1).max())
    found = []
    for kind in ('inv', 'rot2', 'mir', 'trans'):
        for _ in range(N_STARTS):
            u = rng.normal(size=3)
            u /= np.linalg.norm(u)
            v = torch.tensor(rng.normal(size=3), requires_grad=kind in ('rot2', 'mir'))
            q = torch.tensor(u * (2 * r_mol + 2.0), requires_grad=True)   # apart, then pulled in
            params = [p for p in (v, q) if p.requires_grad]
            opt = torch.optim.LBFGS(params, max_iter=150, history_size=15,
                                    tolerance_grad=1e-5, line_search_fn='strong_wolfe')

            def closure():
                opt.zero_grad()
                e = _dimer_energy(a, a @ _motion(kind, v, q).T + q, tables, ffm.form)
                e.backward()
                return e

            try:
                opt.step(closure)
            except Exception:
                continue
            with torch.no_grad():
                Q = _motion(kind, v, q).numpy().copy()
                e = float(_dimer_energy(a, a @ _motion(kind, v, q).T + q, tables, ffm.form))
            qv = q.detach().numpy().copy()
            if not np.isfinite(e) or e >= 0 or np.linalg.norm(qv) > 4 * r_mol + 8:
                continue
            n = None
            if kind in ('rot2', 'mir'):
                n = v.detach().numpy() / np.linalg.norm(v.detach().numpy())
            shift = {'inv': 0.0, 'trans': float(np.linalg.norm(qv)),
                     'rot2': abs(float(qv @ n)) if n is not None else 0.0,
                     'mir': float(np.linalg.norm(qv - (qv @ n) * n)) if n is not None else 0.0}[kind]
            found.append({'kind': kind, 'Q': Q, 'q': qv, 'n': n, 'energy': e,
                          'intrinsic': shift > INTRINSIC, 'image': body @ Q.T + qv})
    if not found:
        return []
    e_best = min(c['energy'] for c in found)
    out = []
    for c in sorted(found, key=lambda c: c['energy']):
        if c['energy'] > KEEP_FRACTION * e_best:
            break
        inverse = (body - c['q']) @ c['Q']              # the dimer seen from the partner
        if any(_same(c['image'], o['image'], ffm.symbols) or
               _same(inverse, o['image'], ffm.symbols) for o in out if o['kind'] == c['kind']):
            continue
        if sum(o['kind'] == c['kind'] for o in out) < KEEP_PER_KIND:
            out.append(c)
    return out


@lru_cache(maxsize=None)
def contact_library(material: str):
    """[contacts of conformer 0, of conformer 1, ...] — cached in db/contacts."""
    from forcefield import get_ff_molecule, MMFF_SCALE
    ffm = get_ff_molecule(material)
    tag = '_'.join(f'{MMFF_SCALE[k]:.3f}' for k in sorted(MMFF_SCALE))
    path = os.path.join(CACHE_DIR, f'{material}.pkl')
    if os.path.isfile(path):
        with open(path, 'rb') as f:
            saved = pickle.load(f)
        if saved.get('tag') == tag:
            return saved['contacts']
    rng = np.random.default_rng(11)
    lib = []
    for c in range(ffm.n_conformers):
        body = ffm.to_molecule(c).centered().positions
        lib.append(_contacts_for(body, ffm, rng))
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(path, 'wb') as f:
        pickle.dump({'tag': tag, 'contacts': lib}, f)
    return lib


# ─────────────────────────────────────────────────────────────
# 2. ASSEMBLY
# ─────────────────────────────────────────────────────────────

@lru_cache(maxsize=None)
def _group_ops(space_group: str):
    """Non-identity operations as (kind, W, w, has intrinsic shift, axis index)."""
    out = []
    for W, w in operations(space_group):
        det, tr = round(float(np.linalg.det(W))), int(np.trace(W))
        if np.array_equal(W, np.eye(3, dtype=int)):
            if np.allclose(w % 1.0, 0):
                continue
            kind = 'trans'                               # centring translation
        elif (det, tr) == (-1, -3):
            kind = 'inv'
        elif (det, tr) == (1, -1):
            kind = 'rot2'
        elif (det, tr) == (-1, 1):
            kind = 'mir'
        else:
            continue                                     # 3-, 4-, 6-fold: not used
        wi = ((W @ w + w) / 2.0) % 1.0                   # screw / glide part
        intrinsic = kind == 'trans' or bool(np.any(np.minimum(wi, 1 - wi) > 1e-6))
        axis = None
        if kind in ('rot2', 'mir'):                      # direction kept (rot2) / flipped (mir)
            axis = int(np.argmax(np.diag(W) == (1 if kind == 'rot2' else -1)))
        out.append((kind, W, w, intrinsic, axis))
    return out


def _cell_torch(p, system):
    """Cell rows from raw parameters (3 log lengths, 3 angles in rad)."""
    a, b, c = torch.exp(p[0]), torch.exp(p[1]), torch.exp(p[2])
    half = torch.tensor(math.pi / 2, dtype=p.dtype)
    al, be, ga = (p[3], p[4], p[5]) if system == 'triclinic' else \
        (half, p[4], half) if system == 'monoclinic' else (half, half, half)
    cx = c * torch.cos(be)
    cy = c * (torch.cos(al) - torch.cos(be) * torch.cos(ga)) / torch.sin(ga)
    cz = torch.sqrt(torch.clamp(c * c - cx * cx - cy * cy, min=1e-6))
    zero = torch.zeros((), dtype=p.dtype)
    return torch.stack([torch.stack([a, zero, zero]),
                        torch.stack([b * torch.cos(ga), b * torch.sin(ga), zero]),
                        torch.stack([cx, cy, cz])])


def _align(n, target, rng):
    """Rotation taking unit vector n to ±target, with a random spin about it."""
    from scipy.spatial.transform import Rotation
    target = target / np.linalg.norm(target) * rng.choice([-1.0, 1.0])
    axis = np.cross(n, target)
    s, c = np.linalg.norm(axis), float(n @ target)
    R0 = Rotation.from_rotvec(axis / s * math.atan2(s, c)).as_matrix() if s > 1e-9 else \
        (np.eye(3) if c > 0 else Rotation.from_rotvec(
            math.pi * np.linalg.svd(np.eye(3) - np.outer(n, n))[0][:, 0]).as_matrix())
    return Rotation.from_rotvec(target * rng.uniform(0, 2 * math.pi)).as_matrix() @ R0


def _solve(body, picks, system, cell0, R0, s0):
    """Least-squares (cell, R, s) so each picked operation reproduces its
    contact. picks: [(contact, W, w + n)]. Returns (cell, R, s, rms Å)."""
    from lattice import _rodrigues
    b = torch.as_tensor(body)
    lengths = np.linalg.norm(cell0, axis=1)
    ang = lambda u, v: math.acos(np.clip(u @ v / np.linalg.norm(u) / np.linalg.norm(v), -1, 1))
    p = torch.tensor([*np.log(lengths), ang(cell0[1], cell0[2]), ang(cell0[0], cell0[2]),
                      ang(cell0[0], cell0[1])], requires_grad=True)
    rot = torch.zeros(1, 3, requires_grad=True)
    s = torch.tensor(s0, requires_grad=True)
    Rb = torch.as_tensor(R0)
    data = [(torch.as_tensor(c['Q']), torch.as_tensor(c['q']),
             torch.as_tensor(W.astype(float)), torch.as_tensor(wn)) for c, W, wn in picks]

    def state():
        cell = _cell_torch(p, system)
        R = _rodrigues(rot)[0] @ Rb
        return cell, R, s @ cell

    def loss():
        cell, R, x0 = state()
        Linv = torch.linalg.inv(cell)
        first = b @ R.T + x0
        total = 0.0
        for Q, q, W, wn in data:
            image = first @ (Linv @ W.T @ cell) + wn @ cell
            partner = (b @ Q.T + q) @ R.T + x0
            total = total + ((image - partner) ** 2).sum(-1).mean()
        return total / len(data)

    opt = torch.optim.LBFGS([p, rot, s], max_iter=80, history_size=20,
                            tolerance_grad=1e-7, line_search_fn='strong_wolfe')

    def closure():
        opt.zero_grad()
        e = loss()
        e.backward()
        return e

    opt.step(closure)
    with torch.no_grad():
        cell, R, _ = state()
        return (cell.numpy().copy(), R.numpy().copy(), s.detach().numpy() % 1.0,
                math.sqrt(max(float(loss()), 0.0)))


def _build(mol, space_group, cell, R, s):
    """Crystal from cell, orientation and fractional position (general position)."""
    body = mol.centered().positions @ R.T
    mols = []
    for W, w in operations(space_group):
        s2 = (W @ s + w) % 1.0
        if any(np.allclose((s2 - q + 0.5) % 1.0 - 0.5, 0, atol=1e-3) for q, _ in mols):
            return None                    # landed on a special position
        mols.append((s2, W))
    z, n_at = len(mols), len(mol.atoms)
    Linv = np.linalg.inv(cell)
    syms = [a.symbol for a in mol.atoms]
    return Crystal(
        cell=cell, com=np.array([q @ cell for q, _ in mols]),
        offset=np.concatenate([body @ (Linv @ W.T @ cell) for _, W in mols]),
        mol=np.repeat(np.arange(z), n_at), sym=syms * z,
        radii=np.tile([lj_radius(a.vdw_radius, mol.category) for a in mol.atoms], z),
        charge=np.tile([a.charge for a in mol.atoms], z),
        donor=np.tile([a.hbond_donor for a in mol.atoms], z),
        acceptor=np.tile([a.hbond_acceptor for a in mol.atoms], z),
        mass=mol.mass, space_group=space_group,
        meta={'z': z, 'special_position': False}, site=np.tile(np.arange(n_at), z))


def puzzle_crystal(mol, contacts, rng, space_group: Optional[str] = None,
                   v_mol: float = None) -> Optional[Crystal]:
    """One crystal assembled from the contact library (None if no
    combination tried fits). meta['puzzle'] lists the contacts used."""
    if not contacts:
        return None
    body = mol.centered().positions
    e = np.array([c['energy'] for c in contacts])
    weight = np.exp(-(e - e.min()) / KT_PICK)
    names, p_sg = sampling_weights()
    for _ in range(MAX_TRIES):
        sg = space_group or str(rng.choice(names, p=p_sg))
        ops = _group_ops(sg)
        system = crystal_system(sg)
        k = int(rng.choice([1, 2, 3], p=[0.3, 0.45, 0.25]))
        order = rng.choice(len(contacts), size=min(k, len(contacts)), replace=False,
                           p=weight / weight.sum())
        picks, used = [], set()
        for ci in order:
            c = contacts[ci]
            if c['kind'] == 'trans' and rng.random() < 0.7:        # a lattice translation
                n = np.zeros(3)
                n[rng.integers(3)] = rng.choice([-1.0, 1.0])
                if tuple(n) not in used:
                    used.add(tuple(n))
                    picks.append((c, np.eye(3, dtype=int), n))
                continue
            fits = [j for j, (kind, W, w, intr, ax) in enumerate(ops)
                    if kind == c['kind'] and j not in used and
                    (kind in ('inv', 'trans') or intr == c['intrinsic'])]
            if not fits:
                continue
            j = int(rng.choice(fits))
            used.add(j)
            n = rng.choice([-1.0, 0.0, 1.0], size=3, p=[0.25, 0.5, 0.25])
            picks.append((c, ops[j][1], ops[j][2] + n))
        if not picks:
            continue
        z = len(operations(sg))
        cell0 = random_cell(system, z * v_mol / rng.uniform(0.55, 0.72), rng)
        R0 = random_rotation(rng)
        for c, W, _ in picks:              # orient the molecule by the first axis contact
            if c['n'] is not None:
                ax = next(a for kind, W2, _, _, a in ops if W2 is W)
                R0 = _align(c['n'], cell0[ax] if c['kind'] == 'rot2' else
                            np.cross(cell0[(ax + 1) % 3], cell0[(ax + 2) % 3]), rng)
                break
        try:
            cell, R, s, rms = _solve(body, picks, system, cell0, R0, rng.random(3))
        except Exception:
            continue
        vol = abs(np.linalg.det(cell))
        if rms > FIT_TOL or not np.isfinite(vol) or not 0.9 < vol / (z * v_mol) < 3.0:
            continue
        x = _build(mol, sg, cell, R, s)
        for _ in range(4):                 # a little room if images just touch
            if x is None or _min_contact_ratio(x) >= MIN_CONTACT:
                break
            cell = cell * 1.04
            x = _build(mol, sg, cell, R, s)
        if x is None or _min_contact_ratio(x) < MIN_CONTACT:
            continue
        x.meta['puzzle'] = [f"{c['kind']}:{c['energy']:.0f}" for c, _, _ in picks]
        x.meta['puzzle_rms'] = rms
        return x
    return None


if __name__ == '__main__':
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    for key in sys.argv[1:]:
        for c, lib in enumerate(contact_library(key)):
            print(f'{key} conformer {c}: {len(lib)} contacts')
            for k in lib:
                print(f'   {k["kind"]:<5} {"screw/glide/trans" if k["intrinsic"] else "pure":<17} '
                      f'E {k["energy"]:7.1f} kJ/mol   centre distance {np.linalg.norm(k["image"].mean(0)):.2f} Å')
