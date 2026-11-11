# -*- coding: utf-8 -*-
# engine/forcefield.py
#
# PHYSICAL FORCE FIELDS FOR CRYSTAL STRUCTURE PREDICTION  (kJ/mol)
#
# Replaces the unit-less toy pair potential for the CSP pipeline.
# Energies are real (kJ/mol per molecule / per atom), so lattice
# energies can be checked against experimental sublimation enthalpies
# and cohesive energies.
#
#   organic molecules  RDKit: 3-D conformer from SMILES (ETKDG + MMFF94
#                      minimised), MMFF94 partial charges (neutral) and
#                      MMFF94 buffered 14-7 van der Waals parameters
#   water              TIP4P/Ice (rigid, massless M site)
#   CO2                TraPPE (rigid, quadrupole from 3 charges)
#   FCC metals         Sutton-Chen many-body potential (Al Cu Ni Au Ag Pt)
#   Fe                 Finnis-Sinclair many-body potential (BCC)
#
# Electrostatics: Ewald summation (real space + reciprocal space − self
# − intramolecular), the standard for periodic crystals. Van der Waals
# terms are smoothly switched off between 10 and 12 Å.
#
# FFModel plugs into lattice.minimise() in place of the toy model:
# same variables (cell strain, molecule translations and rotations).

import os
import json
import math
import numpy as np
import torch
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

from molecules import Molecule, Atom, VDW_RADII, DEFAULT_VDW

KCAL = 4.184                     # kJ per kcal
EV = 96.48533                    # kJ/mol per eV
K_B = 0.0083144626               # kJ/mol per K
COULOMB = 332.0637 * KCAL        # kJ/mol · Å / e²

# Accuracy levels. 'screen' is the cheap first pass of the CSP funnel
# (about half the pairs); survivors are re-minimised at 'full'.
#   vdw_on/vdw_cut  Å   van der Waals switching region
#   alpha           Å⁻¹ Ewald split: erfc(α·vdw_cut) ≈ 1e-5
#   kmax            Å⁻¹ reciprocal cutoff: exp(-k²/4α²) ≈ 1e-7
#   coulomb         'ewald' | 'dsf': damped shifted force (Fennell & Gezelter,
#                   J. Chem. Phys. 124, 234104, 2006), real space only with
#                   cutoff dsf_cut — no reciprocal sum, so with the symmetric
#                   model (pairs of one molecule) the cost falls ~Z×
LEVELS = {
    'full':   dict(vdw_on=10.0, vdw_cut=12.0, alpha=0.25, kmax=2.0, coulomb='ewald'),
    'screen': dict(vdw_on=7.5,  vdw_cut=9.0,  alpha=0.34, kmax=2.7, coulomb='ewald'),
    'fast':   dict(vdw_on=7.5,  vdw_cut=9.0,  alpha=0.20, kmax=0.0, coulomb='dsf',
                   dsf_cut=11.0),
}
VDW_ON, VDW_CUT = LEVELS['full']['vdw_on'], LEVELS['full']['vdw_cut']
EWALD_ALPHA = LEVELS['full']['alpha']
EWALD_KMAX = LEVELS['full']['kmax']
CONF_WINDOW = 12.0               # kJ/mol — conformers kept above the minimum

# Crystal correction for MMFF94 van der Waals (fitted by calibrate.py on
# hydrocarbon crystals; MMFF is a gas-phase force field and under-binds
# crystals: ~78% of the lattice energy at ~91% of the density).
_CALIB_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                           'mmff_crystal_calibration.json')
# 'eps'/'r'     C/H-C/H pairs
# 'eps_x'/'r_x' pairs involving O or N (except H-bond donor-acceptor pairs,
#               which keep MMFF's own geometry)
MMFF_SCALE = {'eps': 1.0, 'r': 1.0, 'eps_x': 1.0, 'r_x': 1.0}
if os.path.isfile(_CALIB_FILE):
    with open(_CALIB_FILE, encoding='utf-8') as _f:
        MMFF_SCALE.update(json.load(_f).get('scale', {}))
MAX_CONFORMERS = 6
USE_KERNEL = os.environ.get('ALLOYFOLD_KERNEL', '1') != '0'   # compiled pair kernel (kernels.py)
CHARGE_WIDTH = 0.3               # Å — Gaussian smearing of point charges:
                                 # removes the 1/r singularity (a line search
                                 # once dropped an H onto a neighbour's charge
                                 # site: -4.5e12 kJ/mol); unchanged > 1.5 Å


@dataclass
class FFMolecule:
    key: str
    name: str
    symbols: list                 # per site ('M' = massless charge site)
    coords: np.ndarray            # (m,3) centred, Å
    charges: np.ndarray           # (m,) e
    form: str                     # 'lj' | 'mmff' | 'sutton-chen' | 'finnis-sinclair'
    pa: Optional[np.ndarray] = None   # (m,m) LJ sigma or MMFF R*
    pb: Optional[np.ndarray] = None   # (m,m) LJ / MMFF epsilon (kJ/mol)
    hbond_pairs: Optional[np.ndarray] = None  # (m,m) MMFF donor-acceptor pairs
    metal: dict = field(default_factory=dict)
    donor: Optional[np.ndarray] = None
    acceptor: Optional[np.ndarray] = None
    mass: float = 0.0
    category: str = 'molecular'
    reference: str = ''
    # [(coords (m,3) centred, conformational energy above the minimum kJ/mol)]
    conformers: list = field(default_factory=list)

    def cutoff(self, level: str = 'full') -> float:
        if self.form == 'eam':
            import eam
            return eam.get_potential(self.metal['potential']).cutoff
        if self.form == 'meam':
            import meam
            return meam.get_potential(self.metal['potential']).rc
        if self.form == 'sutton-chen':
            return self.metal['rc']
        if self.form == 'finnis-sinclair':
            return max(self.metal['c'], self.metal['d'])
        return LEVELS[level]['vdw_cut']

    def conformer(self, k: int):
        """(coords, energy penalty) of conformer k (the base geometry if none)."""
        return self.conformers[k] if self.conformers else (self.coords, 0.0)

    @property
    def n_conformers(self) -> int:
        return max(1, len(self.conformers))

    def to_molecule(self, conformer: int = 0) -> Molecule:
        """molecules.Molecule view, so generate.random_crystal can use it."""
        coords = self.conformer(conformer)[0]
        m = len(self.symbols)
        donor = self.donor if self.donor is not None else np.zeros(m, bool)
        acc = self.acceptor if self.acceptor is not None else np.zeros(m, bool)
        atoms = []
        for k, s in enumerate(self.symbols):
            if s == 'M':
                r = 0.2
            elif self.category == 'metal':
                r = self.metal['r_metallic']
            else:
                r = VDW_RADII.get(s, DEFAULT_VDW)
            atoms.append(Atom(s, *coords[k], vdw_radius=r,
                              charge=float(self.charges[k]),
                              hbond_donor=bool(donor[k]),
                              hbond_acceptor=bool(acc[k])))
        return Molecule(self.name, self.name, atoms, mass=self.mass,
                        category=self.category)


# ─────────────────────────────────────────────────────────────
# MOLECULE MODELS
# ─────────────────────────────────────────────────────────────

SMILES = {
    'aspirin':     'CC(=O)Oc1ccccc1C(=O)O',
    'paracetamol': 'CC(=O)Nc1ccc(O)cc1',
    'caffeine':    'Cn1cnc2c1c(=O)n(C)c(=O)n2C',
    'benzene':     'c1ccccc1',
    'naphthalene': 'c1ccc2ccccc2c1',
    'anthracene':  'c1ccc2cc3ccccc3cc2c1',
    'urea':        'NC(N)=O',
    'acetic_acid': 'CC(=O)O',
    'pyrazine':    'c1cnccn1',
    # X23 benchmark set (x23.py)
    'cyclohexanedione': 'O=C1CCC(=O)CC1',
    'adamantane':  'C1C2CC3CC1CC(C2)C3',
    'ammonia':     'N',
    'cyanamide':   'NC#N',
    'cytosine':    'Nc1cc[nH]c(=O)n1',
    'ethylcarbamate': 'CCOC(N)=O',
    'formamide':   'NC=O',
    'imidazole':   'c1c[nH]cn1',
    'oxalic_alpha': 'OC(=O)C(=O)O',
    'oxalic_beta': 'OC(=O)C(=O)O',
    'pyrazole':    'c1cn[nH]c1',
    'triazine':    'c1ncncn1',
    'trioxane':    'C1OCOCO1',
    'uracil':      'O=c1cc[nH]c(=O)[nH]1',
    'hexamine':    'C1N2CN3CN1CN(C2)C3',
    'succinic_acid': 'OC(=O)CCC(=O)O',
    # CCDC blind-test targets and blind-predicted forms (COD-deposited)
    'hydantoin':   'O=C1CNC(=O)N1',                               # BT3 VIII
    'bt3_x':       'CC(=O)Nc1cc([N+](=O)[O-])c([N+](=O)[O-])cc1C',  # BT3 X
    'acrolein':    'C=CC=O',                                      # BT4
    'dbcfb':       'Fc1cc(Br)c(Cl)c(Br)c1',                       # BT4
    'bt4_thione':  'CN(C)C(=S)N1C(=S)Sc2ccccc21',                 # BT4
    'bt6_xxii':    'N#CC1=C(C#N)Sc2c(nsc2C#N)S1',                 # BT6 XXII
    'hoat':        'On1nnc2cccnc21',
    'oxauracil':   'O=c1cc[nH]c(=O)o1',
    'hydroxyuracil': 'O=c1[nH]cc(O)c(=O)[nH]1',
    # larger molecules (multi-resolution tests); stereo from the COD files
    'progesterone': 'CC(=O)[C@H]1CC[C@H]2[C@@H]3CCC4=CC(=O)CC[C@]4(C)[C@H]3CC[C@]12C',
    'artemisinin': 'C[C@@H]1CC[C@H]2[C@@H](C)C(=O)O[C@@H]3O[C@@]4(C)CC[C@@H]1[C@]32OO4',
}


def _from_smiles(key: str, smiles: str, n_conf: int = 60,
                 seed: int = 7) -> FFMolecule:
    from rdkit import Chem
    from rdkit.Chem import AllChem, Descriptors
    mol = Chem.AddHs(Chem.MolFromSmiles(smiles))
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    cids = list(AllChem.EmbedMultipleConfs(mol, n_conf, params))
    res = AllChem.MMFFOptimizeMoleculeConfs(mol, maxIters=2000)
    energies = np.array([e for _, e in res]) * KCAL
    order = np.argsort(energies)

    # Distinct low-energy conformers: the crystal often holds a conformer
    # other than the gas-phase minimum, paid for by its strain energy.
    # Compared on heavy atoms + polar H (O-H / N-H direction decides the
    # H-bonding), ignoring C-H (methyl rotations are not distinct forms).
    from rdkit.Chem import rdMolAlign
    sel = [a.GetIdx() for a in mol.GetAtoms()
           if a.GetSymbol() != 'H' or a.GetNeighbors()[0].GetSymbol() in ('N', 'O')]
    amap = [(i, i) for i in sel]
    conformers, kept = [], []
    for k in order:
        e_rel = float(energies[k] - energies[order[0]])
        if e_rel > CONF_WINDOW or len(kept) >= MAX_CONFORMERS:
            break
        if any(rdMolAlign.AlignMol(mol, mol, int(cids[k]), int(cids[j]),
                                   atomMap=amap) < 0.3
               for j in kept):
            continue
        kept.append(k)
        xyz = mol.GetConformer(int(cids[k])).GetPositions()
        conformers.append((xyz - xyz.mean(axis=0), e_rel))
    xyz = conformers[0][0]

    props = AllChem.MMFFGetMoleculeProperties(mol)
    m = mol.GetNumAtoms()
    q = np.array([props.GetMMFFPartialCharge(i) for i in range(m)])
    R = np.zeros((m, m))
    eps = np.zeros((m, m))
    hbond = np.zeros((m, m), bool)
    for i in range(m):
        for j in range(m):
            R0, e0, R[i, j], eps[i, j] = props.GetMMFFVdWParams(i, j)
            # MMFF shortens donor-acceptor pairs (R*×0.8, eps×0.5):
            # scaled != unscaled marks an H-bond pair
            hbond[i, j] = abs(R0 - R[i, j]) > 1e-9 or abs(e0 - eps[i, j]) > 1e-9

    sym = [a.GetSymbol() for a in mol.GetAtoms()]
    donor = np.array([s == 'H' and a.GetNeighbors()[0].GetSymbol() in ('N', 'O')
                      for s, a in zip(sym, mol.GetAtoms())])
    acceptor = np.array([s in ('O', 'N') and a.GetTotalNumHs() == 0
                         for s, a in zip(sym, mol.GetAtoms())]) | \
        np.array([s == 'O' for s in sym])
    return FFMolecule(key, key, sym, xyz, q, 'mmff', R, eps * KCAL,
                      hbond_pairs=hbond, donor=donor, acceptor=acceptor,
                      mass=Descriptors.MolWt(mol),
                      category='pharma', conformers=conformers,
                      reference=f'MMFF94 (RDKit), {len(conformers)} conformer(s) '
                                f'within {CONF_WINDOW:g} kJ/mol')


def _lj_mixed(sig, eps):
    """Lorentz-Berthelot mixing matrices from per-site sigma/epsilon."""
    s = np.asarray(sig, float)
    e = np.asarray(eps, float)
    return (s[:, None] + s[None]) / 2, np.sqrt(e[:, None] * e[None])


def _water_tip4p_ice() -> FFMolecule:
    a = math.radians(104.52 / 2)
    oh, om = 0.9572, 0.1577
    xyz = np.array([[0, 0, 0],
                    [oh * math.sin(a), oh * math.cos(a), 0],
                    [-oh * math.sin(a), oh * math.cos(a), 0],
                    [0, om, 0]], float)
    com = (15.999 * xyz[0] + 1.008 * (xyz[1] + xyz[2])) / 18.015
    pa, pb = _lj_mixed([3.1668, 0, 0, 0], [106.1 * K_B, 0, 0, 0])
    return FFMolecule('H2O', 'water', ['O', 'H', 'H', 'M'], xyz - com,
                      np.array([0.0, 0.5897, 0.5897, -1.1794]), 'lj', pa, pb,
                      donor=np.array([False, True, True, False]),
                      acceptor=np.array([True, False, False, False]),
                      mass=18.015, reference='TIP4P/Ice (Abascal 2005)')


def _co2_trappe() -> FFMolecule:
    xyz = np.array([[-1.16, 0, 0], [0, 0, 0], [1.16, 0, 0]])
    pa, pb = _lj_mixed([3.05, 2.80, 3.05], [79.0 * K_B, 27.0 * K_B, 79.0 * K_B])
    return FFMolecule('CO2', 'CO2', ['O', 'C', 'O'], xyz,
                      np.array([-0.35, 0.70, -0.35]), 'lj', pa, pb,
                      acceptor=np.array([True, False, True]),
                      mass=44.01, reference='TraPPE (Potoff & Siepmann 2001)')


# Sutton & Chen, Phil. Mag. Lett. 61 (1990) 139: n, m, epsilon (eV), c, a (Å)
SUTTON_CHEN = {
    'Al': (7, 6, 3.3147e-2, 16.399, 4.05),
    'Cu': (9, 6, 1.2382e-2, 39.432, 3.61),
    'Ni': (9, 6, 1.5707e-2, 39.432, 3.52),
    'Au': (10, 8, 1.2793e-2, 34.408, 4.08),
    'Ag': (12, 6, 2.5415e-3, 144.41, 4.09),
    'Pt': (10, 8, 1.9833e-2, 34.408, 3.92),
}
MASS = {'Al': 26.98, 'Cu': 63.55, 'Ni': 58.69, 'Au': 196.97, 'Ag': 107.87,
        'Pt': 195.08, 'Fe': 55.85}


def _metal(sym: str) -> FFMolecule:
    one = dict(symbols=[sym], coords=np.zeros((1, 3)), charges=np.zeros(1),
               mass=MASS[sym], category='metal')
    if sym == 'Fe':
        # Finnis & Sinclair, Phil. Mag. A 50 (1984) 45 — alpha-Fe
        p = dict(d=3.569745, A=1.828905, beta=1.8, c=3.40,
                 c0=1.2371147, c1=-0.3592185, c2=-0.0385607,
                 r_metallic=1.241)
        return FFMolecule('Fe', 'Fe', form='finnis-sinclair', metal=p,
                          reference='Finnis-Sinclair 1984', **one)
    n, m, e, c, a = SUTTON_CHEN[sym]
    p = dict(n=n, m=m, eps=e, c=c, a=a, rc=min(3.0 * a, 14.0),
             r_metallic=a / (2 * math.sqrt(2)))
    return FFMolecule(sym, sym, form='sutton-chen', metal=p,
                      reference='Sutton-Chen 1990', **one)


def _alloy_meam(potential: str, elements, reference) -> FFMolecule:
    """Like _alloy_eam, with the 2NN-MEAM engine (meam.py)."""
    m = len(elements)
    return FFMolecule(f'{potential}_MEAM', f'{"-".join(elements)} (2NN-MEAM)', list(elements),
                      np.zeros((m, 3)), np.zeros(m), 'meam',
                      metal={'potential': potential}, category='metal', reference=reference)


def _alloy_eam(potential: str, elements) -> FFMolecule:
    """Multi-element alloy: every atom is its own body; species come from
    Crystal.sym (alloy_csp.py). Energies from the tabulated EAM (eam.py)."""
    m = len(elements)
    return FFMolecule(f'{potential}_EAM', f'{"-".join(elements)} (EAM)', list(elements),
                      np.zeros((m, 3)), np.zeros(m), 'eam',
                      metal={'potential': potential}, category='metal',
                      reference='Zope & Mishin, Phys. Rev. B 68, 024102 (2003), NIST IPR')


# Molecules added by the user (web interface): {key: {name, smiles, added}}.
# Kept in a file so that worker processes (spawned, re-importing this
# module) build the same force field.
USER_FILE = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'db', 'user_molecules.json')
MMFF_ELEMENTS = {'H', 'C', 'N', 'O', 'F', 'P', 'S', 'Cl', 'Br', 'I'}
MAX_HEAVY_ATOMS = 60


def _load_user_molecules() -> dict:
    try:
        with open(USER_FILE, encoding='utf-8') as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


USER_MOLECULES = _load_user_molecules()

BUILDERS = {
    'H2O': _water_tip4p_ice,
    'CO2': _co2_trappe,
    'TiAl_EAM': lambda: _alloy_eam('TiAl', ('Ti', 'Al')),
    'AlZr_MEAM': lambda: _alloy_meam('AlZr', ('Al', 'Zr'),
                                     'Fereidonnejad et al., Comput. Mater. Sci. 213, 111685 (2022), NIST IPR'),
    **{k: (lambda k=k: _from_smiles(k, SMILES[k])) for k in SMILES},
    **{s: (lambda s=s: _metal(s)) for s in list(SUTTON_CHEN) + ['Fe']},
    **{k: (lambda k=k: _from_smiles(k, USER_MOLECULES[k]['smiles'])) for k in USER_MOLECULES},
}


def check_smiles(smiles: str) -> str:
    """Canonical SMILES if MMFF94 can model the molecule, else ValueError (plain words)."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog('rdApp.*')
    mol = Chem.MolFromSmiles(smiles.strip()) if smiles and smiles.strip() else None
    if mol is None:
        raise ValueError('This is not a valid SMILES code. Copy it exactly from PubChem '
                         '("Canonical SMILES" or "Isomeric SMILES").')
    if len(Chem.GetMolFrags(mol)) > 1:
        raise ValueError('The SMILES contains several separate pieces (a "." in the code). '
                         'Salts, hydrates and co-crystals are not supported yet: enter one molecule.')
    bad = sorted({a.GetSymbol() for a in mol.GetAtoms()} - MMFF_ELEMENTS)
    if bad:
        raise ValueError(f'The force field cannot handle {", ".join(bad)}. '
                         f'Allowed elements: {", ".join(sorted(MMFF_ELEMENTS))}.')
    if Chem.GetFormalCharge(mol):
        raise ValueError('The molecule carries a net charge. Only neutral molecules are supported.')
    if mol.GetNumHeavyAtoms() > MAX_HEAVY_ATOMS:
        raise ValueError(f'The molecule has {mol.GetNumHeavyAtoms()} non-hydrogen atoms; '
                         f'the limit is {MAX_HEAVY_ATOMS} (larger ones take too long on one PC).')
    if not AllChem.MMFFHasAllMoleculeParams(Chem.AddHs(mol)):
        raise ValueError('The MMFF94 force field has no parameters for part of this molecule.')
    return Chem.MolToSmiles(mol)


def register_user_molecule(name: str, smiles: str) -> str:
    """Add (or replace) a user molecule; returns its key. Builds it once to be sure it embeds in 3-D."""
    import re
    import time
    smiles = check_smiles(smiles)
    name = (name or '').strip()[:60] or smiles
    key = 'user_' + (re.sub(r'[^a-z0-9]+', '_', name.lower()).strip('_') or 'molecule')
    try:
        _from_smiles(key, smiles)
    except Exception as e:
        raise ValueError(f'RDKit could not build a 3-D shape for this molecule ({e}).')
    USER_MOLECULES[key] = {'name': name, 'smiles': smiles, 'added': time.time()}
    BUILDERS[key] = lambda k=key: _from_smiles(k, USER_MOLECULES[k]['smiles'])
    get_ff_molecule.cache_clear()
    os.makedirs(os.path.dirname(USER_FILE), exist_ok=True)
    with open(USER_FILE, 'w', encoding='utf-8') as f:
        json.dump(USER_MOLECULES, f, indent=1)
    return key


def remove_user_molecule(key: str) -> None:
    if key in USER_MOLECULES:
        del USER_MOLECULES[key]
        BUILDERS.pop(key, None)
        get_ff_molecule.cache_clear()
        with open(USER_FILE, 'w', encoding='utf-8') as f:
            json.dump(USER_MOLECULES, f, indent=1)


def has_forcefield(key: str) -> bool:
    return key in BUILDERS


@lru_cache(maxsize=None)
def get_ff_molecule(key: str) -> FFMolecule:
    return BUILDERS[key]()


# ─────────────────────────────────────────────────────────────
# ENERGY MODEL (torch)
# ─────────────────────────────────────────────────────────────

def _switch(r, r_on, r_off):
    t = torch.clamp((r - r_on) / (r_off - r_on), 0.0, 1.0)
    return 1.0 - t * t * (3.0 - 2.0 * t)


def _kvectors(cell: np.ndarray, kmax: float) -> np.ndarray:
    """Integer reciprocal-lattice indices m (≠0) with |k| < kmax."""
    B = 2 * math.pi * np.linalg.inv(cell).T          # rows: b1, b2, b3
    nmax = [int(kmax / np.linalg.norm(b)) + 1 for b in B]
    g = [np.arange(-n, n + 1) for n in nmax]
    m = np.stack(np.meshgrid(*g, indexing='ij'), -1).reshape(-1, 3)
    m = m[np.any(m != 0, axis=1)]
    k = m @ B
    return m[np.einsum('ij,ij->i', k, k) < kmax * kmax].astype(float)


def _intramolecular_ewald(q: np.ndarray, p: np.ndarray, alpha: float) -> float:
    """Σ_{i<j same molecule} q_i q_j erf(α r)/r — included by the
    reciprocal sum but not part of the lattice energy (constant for a
    rigid molecule; p is that molecule's actual geometry)."""
    total = 0.0
    for i in range(len(q)):
        for j in range(i + 1, len(q)):
            r = float(np.linalg.norm(p[i] - p[j]))
            total += q[i] * q[j] * math.erf(alpha * r) / r
    return total


class FFModel:
    """Physical energy (kJ/mol per cell) of a rigid-molecule crystal as a
    function of (strain, translations, rotations); see lattice._RigidModel."""

    def __init__(self, base, x, pairs):
        self.base = base                       # lattice._RigidModel geometry
        self.ff = ff = get_ff_molecule(x.ff)
        self.lv = lv = LEVELS[x.meta.get('level', 'full')]
        i, j, _ = pairs
        self.form = ff.form
        self.n_atoms = len(x.mol)
        self.shape = 1.0
        # pairs from one molecule only (lattice._SymGeometry): each counts Z×
        self.pair_weight = getattr(base, 'pair_weight', 1.0)
        if ff.form == 'eam':                    # alloys: species per atom
            import eam
            self.pot = eam.get_potential(ff.metal['potential'])
            self.prep = self.pot.prepare(i, j, list(x.sym))
            self.charged = False
            return
        if ff.form == 'meam':                   # own pair + screening-triplet lists
            import meam
            self.pot = meam.get_potential(ff.metal['potential'])
            self.prep = self.pot.prepare(x.positions, x.cell, list(x.sym), skin=1.0)
            self.charged = False
            return
        si, sj = x.site[i], x.site[j]
        if ff.form in ('lj', 'mmff'):
            # crystal correction (dispersion), fitted by calibrate.py on
            # experimental structures: one scale pair for C/H-C/H, one for
            # pairs with O/N (a shared one over-compressed acetic acid by 8%
            # or left aspirin 10% too loose). H-bond pairs are not scaled.
            r_s = np.ones(len(si))
            e_s = np.ones(len(si))
            if ff.form == 'mmff':
                ch = np.array([s in ('C', 'H') for s in ff.symbols])
                free = ~ff.hbond_pairs[si, sj]
                both_ch = ch[si] & ch[sj]
                r_s[free & both_ch] = MMFF_SCALE['r']
                e_s[free & both_ch] = MMFF_SCALE['eps']
                r_s[free & ~both_ch] = MMFF_SCALE['r_x']
                e_s[free & ~both_ch] = MMFF_SCALE['eps_x']
            self.pa = torch.as_tensor(ff.pa[si, sj] * r_s)
            self.pb = torch.as_tensor(ff.pb[si, sj] * e_s)
            # shape morphing (lattice.minimise, meta['shape'] = λ < 1): for the
            # van der Waals part every atom is pulled towards its molecule's
            # centre by (1-λ) and its contact distance grown by the same
            # length — at λ = 0 the molecule is a sphere of its own size, at
            # λ = 1 the real shape. Charges stay in place, so electrostatics
            # orients the molecules before the shapes have to fit.
            self.shape = float(x.meta.get('shape', 1.0))
            if self.shape < 1.0:
                rho = np.linalg.norm(x.offset, axis=1)
                self.pa = self.pa + torch.as_tensor((1.0 - self.shape) * (rho[i] + rho[j]))
                self.pb = self.pb * (0.25 + 0.75 * self.shape)
        q = np.tile(ff.charges, x.z)
        self.charged = bool(np.any(np.abs(q) > 1e-9))
        self.dsf = self.charged and lv.get('coulomb') == 'dsf'
        if self.dsf:
            # pair term q_i q_j [erfc(αr)/r - erfc(αRc)/Rc + F(Rc)(r - Rc)] for
            # r < Rc (point-charge form, smeared at short range like the Ewald
            # real part); the self term and the intramolecular pairs the sum
            # leaves out are constants for rigid molecules
            a, Rc = lv['alpha'], lv['dsf_cut']
            self.dsf_shift = math.erfc(a * Rc) / Rc
            self.dsf_force = math.erfc(a * Rc) / Rc ** 2 +                 2 * a / math.sqrt(math.pi) * math.exp(-a * a * Rc * Rc) / Rc
            self.qq = torch.as_tensor(q[i] * q[j])
            p0 = x.offset[x.mol == 0][np.argsort(x.site[x.mol == 0])]
            # DSF approximates the sum over ALL pairs (incl. intramolecular)
            # plus a self term; the lattice energy wants the intermolecular
            # part, so each intramolecular pair's DSF value is swapped for its
            # bare 1/r: correction -q q [1/r - dsf(r)] (pairs beyond Rc: -q q/r)
            intra = 0.0
            for u in range(len(p0)):
                for v in range(u + 1, len(p0)):
                    r = float(np.linalg.norm(p0[u] - p0[v]))
                    qq = ff.charges[u] * ff.charges[v]
                    if r < Rc:
                        intra += qq * (math.erf(a * r) / r + self.dsf_shift - self.dsf_force * (r - Rc))
                    else:
                        intra += qq / r
            self.e_const = -COULOMB * x.z * (
                (self.dsf_shift / 2 + a / math.sqrt(math.pi)) * float((ff.charges ** 2).sum())
                + intra)
        elif self.charged:
            self.q = torch.as_tensor(q)
            self.qq = torch.as_tensor(q[i] * q[j])
            self.mvec = torch.as_tensor(_kvectors(x.cell, lv['kmax'] * 1.15))
            first = x.mol == 0
            self.e_const = -COULOMB * (
                lv['alpha'] / math.sqrt(math.pi) * float((q * q).sum())
                + x.z * _intramolecular_ewald(ff.charges, x.offset[first],
                                              lv['alpha']))
        self.n_atoms = len(x.mol)
        # compiled pair kernel (kernels.py): one pass for energy + forces
        self.use_kernel = USE_KERNEL and ff.form in ('lj', 'mmff') and self.shape >= 1.0
        if self.use_kernel:
            coul = 2 if self.dsf else (1 if self.charged else 0)
            self.kernel_spec = (
                np.ascontiguousarray(i, dtype=np.int64), np.ascontiguousarray(j, dtype=np.int64),
                np.ascontiguousarray(pairs[2], dtype=np.float64),
                self.pa.numpy().astype(np.float64), self.pb.numpy().astype(np.float64),
                self.qq.numpy().astype(np.float64) if self.charged else np.zeros(len(i)),
                ff.form == 'lj', lv['vdw_on'], lv['vdw_cut'], coul, lv['alpha'], CHARGE_WIDTH,
                lv.get('dsf_cut', 0.0), getattr(self, 'dsf_shift', 0.0), getattr(self, 'dsf_force', 0.0),
                COULOMB, float(self.pair_weight))

    def geometry(self, eps, dcom, rot):
        return self.base.geometry(eps, dcom, rot)

    def _reciprocal(self, pos, cell):
        lv = self.lv
        vol = torch.abs(torch.linalg.det(cell))
        B = 2 * math.pi * torch.linalg.inv(cell).T
        k = self.mvec @ B
        k2 = (k * k).sum(-1)
        phase = pos @ k.T                                   # (N, K)
        S2 = (self.q @ torch.cos(phase)) ** 2 + (self.q @ torch.sin(phase)) ** 2
        return COULOMB * (2 * math.pi / vol) * (
            torch.exp(-k2 / (4 * lv['alpha'] ** 2)) / k2 * S2).sum()

    def __call__(self, eps, dcom, rot):
        b = self.base
        cell, com, off = b.geometry(eps, dcom, rot)
        pos = com[b.mol] + off
        if self.use_kernel:
            from kernels import PairEnergy
            energy = PairEnergy.apply(pos, cell, self.kernel_spec)
            if self.dsf:
                return energy + self.e_const
            if self.charged:
                return energy + self._reciprocal(pos, cell) + self.e_const
            return energy
        d = pos[b.j] + b.n @ cell - pos[b.i]
        r = torch.sqrt((d * d).sum(-1))
        if self.form == 'eam':
            return EV * self.pot.energy_prepared(r, self.prep)
        if self.form == 'meam':
            return EV * self.pot.energy(pos, cell, self.prep)
        if self.form == 'sutton-chen':
            return self._sutton_chen(r)
        if self.form == 'finnis-sinclair':
            return self._finnis_sinclair(r)

        lv = self.lv
        rv = r                                  # distances seen by van der Waals
        if self.shape < 1.0:
            pv = com[b.mol] + self.shape * off
            dv = pv[b.j] + b.n @ cell - pv[b.i]
            rv = torch.sqrt((dv * dv).sum(-1) + 1e-12)
        sw = _switch(rv, lv['vdw_on'], lv['vdw_cut'])
        if self.form == 'lj':
            sr6 = torch.where(self.pb > 0, (self.pa / rv) ** 6, torch.zeros_like(rv))
            e_vdw = 4 * self.pb * (sr6 * sr6 - sr6)
        else:                                   # MMFF94 buffered 14-7
            R, e = self.pa, self.pb
            e_vdw = e * (1.07 * R / (rv + 0.07 * R)) ** 7 * \
                (1.12 * R ** 7 / (rv ** 7 + 0.12 * R ** 7) - 2.0)
        energy = self.pair_weight * (e_vdw * sw).sum() / 2.0

        if self.dsf:
            Rc = lv['dsf_cut']
            pair = (torch.erf(r / CHARGE_WIDTH) - torch.erf(lv['alpha'] * r)) / r \
                - self.dsf_shift + self.dsf_force * (r - Rc)
            e_dsf = torch.where(r < Rc, self.qq * pair, torch.zeros_like(r)).sum()
            return energy + self.pair_weight * COULOMB * e_dsf / 2 + self.e_const
        if self.charged:
            # erf(r/w) - erf(αr) = erfc(αr) - erfc(r/w): finite as r → 0
            e_real = self.pair_weight * COULOMB * (self.qq * (torch.erf(r / CHARGE_WIDTH)
                                           - torch.erf(lv['alpha'] * r)) / r).sum() / 2
            energy = energy + e_real + self._reciprocal(pos, cell) + self.e_const
        return energy

    def _sutton_chen(self, r):
        p = self.ff.metal
        sw = _switch(r, p['rc'] - 1.0, p['rc'])
        ar = p['a'] / r
        rep = (ar ** p['n'] * sw).sum() / 2.0
        rho = torch.zeros(self.n_atoms).index_add(0, self.base.i, ar ** p['m'] * sw)
        return p['eps'] * EV * (rep - p['c'] * torch.sqrt(rho + 1e-12).sum())

    def _finnis_sinclair(self, r):
        p = self.ff.metal
        V = torch.where(r < p['c'],
                        (r - p['c']) ** 2 * (p['c0'] + p['c1'] * r + p['c2'] * r * r),
                        torch.zeros_like(r))
        phi = torch.where(r < p['d'],
                          (r - p['d']) ** 2 + p['beta'] * (r - p['d']) ** 3 / p['d'],
                          torch.zeros_like(r))
        rho = torch.zeros(self.n_atoms).index_add(0, self.base.i, phi)
        return EV * (V.sum() / 2.0 - p['A'] * torch.sqrt(rho + 1e-12).sum())
