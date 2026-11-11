# -*- coding: utf-8 -*-
# engine/references.py
#
# EXPERIMENTAL REFERENCE STRUCTURES from the Crystallography Open Database
# (COD, https://www.crystallography.net — open data, CC0).
#
# CIFs are downloaded once and cached in references/cod_<id>.cif.
# Low-temperature determinations are preferred: lattice-energy minima are
# 0 K structures, so they compare best with cold data.
#
# Matching (csp_benchmark, hunt) uses heavy atoms only: X-ray diffraction
# places H atoms ~0.1 Å too close to their parent atom.

import os
import time
import urllib.request
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REF_DIR = os.path.join(HERE, 'references')

# material: [(label, COD id, temperature K)]
COD_REFERENCES = {
    'aspirin':     [('form I', 7050900, 20)],
    'paracetamol': [('form I', 2104364, 100), ('form II', 4510931, 100),
                    ('form III', 2021811, 100)],
    'benzene':     [('form I', 7238223, 150)],
    'naphthalene': [('form I', 2311090, 80)],
    'anthracene':  [('form I', 5000168, None)],
    'acetic_acid': [('form I', 2008187, 40)],
    # blind-test targets (Lommerse 2000 … Reilly 2016) and blind-predicted forms
    'hydantoin':   [('BT3 VIII', 2100163, None)],
    'bt3_x':       [('BT3 X', 2100165, None)],
    'acrolein':    [('BT4', 2105702, None)],
    'dbcfb':       [('BT4', 2105703, None)],
    'bt4_thione':  [('BT4', 2105704, None)],
    'bt6_xxii':    [('BT6 XXII', 2108026, None)],
    'hoat':        [('form I', 2100524, None), ('form II', 2100525, None)],
    'oxauracil':   [('form I', 7216063, None)],
    'hydroxyuracil': [('form I', 7216065, None)],
    'progesterone': [('form I', 4511761, 110), ('racemic, CSP-predicted', 7101588, 150)],
    'artemisinin': [('form I', 1563559, 100)],
}


def fetch_cif(cod_id: int) -> str:
    """Path of the cached CIF, downloading it from COD if needed."""
    os.makedirs(REF_DIR, exist_ok=True)
    path = os.path.join(REF_DIR, f'cod_{cod_id}.cif')
    if not os.path.isfile(path):
        url = f'https://www.crystallography.net/cod/{cod_id}.cif'
        for attempt in range(4):
            try:
                time.sleep(2 + 5 * attempt)      # be polite to the COD server
                with urllib.request.urlopen(url, timeout=60) as r:
                    data = r.read()
                break
            except OSError:
                if attempt == 3:
                    raise
        with open(path, 'wb') as f:
            f.write(data)
    return path


def load_reference(cod_id: int, heavy_only: bool = True):
    """(cell 3×3, cartesian positions, symbols) of the full unit cell."""
    from ase.io import read
    if atoms is None:                    # else: an ASE Atoms (e.g. X23 file)
        atoms = read(fetch_cif(cod_id), format='cif')
    sym = atoms.get_chemical_symbols()
    pos = atoms.get_positions()
    if heavy_only:
        keep = np.array([s not in ('H', 'D') for s in sym])
        pos = pos[keep]
        sym = [s for s, k in zip(sym, keep) if k]
    return np.array(atoms.cell), pos, sym


def _split_molecules(atoms):
    """Molecule index per atom + coordinates unwrapped across the cell."""
    from ase.neighborlist import neighbor_list, natural_cutoffs
    i, j, S = neighbor_list('ijS', atoms, natural_cutoffs(atoms, mult=1.15))
    n = len(atoms)
    pos = atoms.get_positions()
    cell = np.array(atoms.cell)
    nbrs = [[] for _ in range(n)]
    for a, b, s in zip(i, j, S):
        nbrs[a].append((b, s))
    mol = -np.ones(n, int)
    unwrapped = pos.copy()
    m = 0
    for start in range(n):
        if mol[start] >= 0:
            continue
        mol[start] = m
        stack = [start]
        while stack:
            a = stack.pop()
            for b, s in nbrs[a]:
                if mol[b] < 0:
                    mol[b] = m
                    unwrapped[b] = unwrapped[a] + (pos[b] + s @ cell - pos[a])
                    stack.append(b)
        m += 1
    return mol, unwrapped


def _heavy_mapping(ff_key: str, exp_sym, exp_pos):
    """FF heavy-atom index → experimental atom index (graph isomorphism on
    heavy atoms, bond orders ignored), or None."""
    from rdkit import Chem
    from rdkit.Chem import rdDetermineBonds
    from forcefield import SMILES

    def plain(mol):
        rw = Chem.RWMol(mol)
        for b in rw.GetBonds():
            b.SetBondType(Chem.BondType.SINGLE)
            b.SetIsAromatic(False)
        for a in rw.GetAtoms():
            a.SetIsAromatic(False)
            a.SetNoImplicit(True)
            a.SetNumExplicitHs(0)
            a.SetFormalCharge(0)         # nitro N+/O- etc.: graph only
        return rw.GetMol()

    smiles = {**SMILES, 'CO2': 'O=C=O'}.get(ff_key)   # CO2 is a TraPPE model
    ff_mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    ff_heavy = [a.GetIdx() for a in ff_mol.GetAtoms() if a.GetSymbol() != 'H']
    heavy = [k for k, s in enumerate(exp_sym) if s not in ('H', 'D')]
    xyz = f'{len(heavy)}\n\n' + '\n'.join(
        f'{exp_sym[k]} {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}' for k, p in
        ((k, exp_pos[k]) for k in heavy))
    exp_mol = Chem.MolFromXYZBlock(xyz)
    rdDetermineBonds.DetermineConnectivity(exp_mol)
    query = plain(Chem.RemoveHs(ff_mol, sanitize=False))
    # ALL matches: without bond orders, e.g. a carboxylic acid's C=O and O-H
    # oxygens are equivalent — the hydrogens decide which mapping is right
    matches = plain(exp_mol).GetSubstructMatches(query, uniquify=False, maxMatches=500)
    return [{ff_heavy[q]: heavy[t] for q, t in enumerate(m)}
            for m in matches if len(m) == len(ff_heavy)]


def _kabsch(P, Q):
    """Rotation R minimising |P R - Q| for centred P, Q (rows = points)."""
    H = P.T @ Q
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.diag([1, 1, d])
    return U @ D @ Vt


def experimental_crystal(material: str, cod_id: int = None, atoms=None):
    """
    The experimental structure as a force-field Crystal: each experimental
    molecule is replaced by the best-fitting FF conformer (heavy atoms
    mapped by graph isomorphism, optimally rotated onto it), so it can be
    minimised with exactly the model the search uses.
    """
    from ase.io import read
    from forcefield import get_ff_molecule
    from molecules import lj_radius, VDW_RADII, DEFAULT_VDW
    from lattice import Crystal
    if atoms is None:                    # else: an ASE Atoms (e.g. an X23 file)
        atoms = read(fetch_cif(cod_id), format='cif')
    sym = atoms.get_chemical_symbols()
    mol_id, pos = _split_molecules(atoms)
    ff = get_ff_molecule(material)
    heavy_ff = [k for k, s in enumerate(ff.symbols) if s != 'H']
    coms, offsets, chosen = [], [], []
    for m in range(mol_id.max() + 1):
        idx = np.flatnonzero(mol_id == m)
        msym = [sym[k] for k in idx]
        if sum(s not in ('H', 'D') for s in msym) != len(heavy_ff):
            continue                                   # solvent / fragment
        mappings = _heavy_mapping(material, msym, pos[idx])
        if not mappings:
            raise ValueError(f'COD {cod_id}: molecule does not match {material}')
        # polar H (O-H / N-H) decide H-bonding: each placed polar H is scored
        # by its distance to the nearest experimental H (when the file has
        # H atoms) — this picks both the right atom mapping and conformer
        exp_h = np.array([pos[t] for t in idx if sym[t] in ('H', 'D')])
        polar_ff = [k for k, s in enumerate(ff.symbols)
                    if s == 'H' and ff.donor is not None and ff.donor[k]]
        best = None
        for mapping in mappings:
            target = np.array([pos[idx[mapping[k]]] for k in heavy_ff])
            Q = target - target.mean(0)
            for c in range(ff.n_conformers):
                body, e_rel = ff.conformer(c)
                P = body[heavy_ff] - body[heavy_ff].mean(0)
                R = _kabsch(P, Q)
                placed = (body - body[heavy_ff].mean(0)) @ R + target.mean(0)
                dev = list(((placed[heavy_ff] - target) ** 2).sum(1))
                if len(exp_h):
                    dev += [np.min(((exp_h - placed[k]) ** 2).sum(1)) for k in polar_ff]
                rmsd = float(np.sqrt(np.mean(dev)))
                if best is None or rmsd + 0.02 * e_rel < best[0]:
                    best = (rmsd + 0.02 * e_rel, c, e_rel, placed)
        _, c, e_rel, placed = best
        com = placed.mean(0)
        coms.append(com)
        offsets.append(placed - com)
        chosen.append((c, e_rel))
    z = len(coms)
    n_at = len(ff.symbols)
    radii = [lj_radius(VDW_RADII.get(s, DEFAULT_VDW), 'pharma') for s in ff.symbols]
    # one conformer per crystal in the rigid model: use the most common fit
    c_main = max(set(c for c, _ in chosen), key=[c for c, _ in chosen].count)
    return Crystal(
        cell=np.array(atoms.cell), com=np.array(coms),
        offset=np.concatenate(offsets), mol=np.repeat(np.arange(z), n_at),
        sym=list(ff.symbols) * z, radii=np.tile(radii, z),
        charge=np.tile(ff.charges, z),
        donor=np.tile(ff.donor if ff.donor is not None else np.zeros(n_at, bool), z),
        acceptor=np.tile(ff.acceptor if ff.acceptor is not None else np.zeros(n_at, bool), z),
        mass=ff.mass,
        space_group='experimental', site=np.tile(np.arange(n_at), z),
        ff=material, meta={'level': 'full', 'conformer': c_main,
                           'e_conf': ff.conformer(c_main)[1],
                           'cod_id': cod_id})


def reference_forms(material: str):
    """[(label, cod_id, (cell, heavy positions, symbols))] for a material."""
    out = []
    for label, cod_id, _ in COD_REFERENCES.get(material, []):
        out.append((label, cod_id, load_reference(cod_id)))
    return out


if __name__ == '__main__':
    for m, forms in COD_REFERENCES.items():
        for label, cod_id, T in forms:
            cell, pos, sym = load_reference(cod_id)
            z_heavy = len(pos)
            print(f'{m:<12} {label:<9} COD {cod_id}  T {T}  heavy atoms/cell {z_heavy}  '
                  f'V {abs(np.linalg.det(cell)):.1f} Å³')
