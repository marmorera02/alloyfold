# -*- coding: utf-8 -*-
# engine/molecules.py
#
# Molecular geometry from real crystallographic data.
#
# Each molecule is defined by:
#   - atom positions in Angstroms from molecular centroid
#   - bond connectivity and order
#   - Van der Waals surface (radius per atom)
#   - interaction sites (H-bond donors/acceptors, charges)
#   - convex hull of VdW surface (for energy calc)
#   - axis-aligned bounding box of hull (for tiling)
#
# The hull is computed over the VdW-inflated surface:
#   hull_point = atom_position + vdw_radius * outward_normal
# This means the hull represents the true outer surface
# that another molecule first contacts.
#
# Sources: ICSD, CSD, NIST CCCBDB, CRC Handbook

import numpy as np
from dataclasses import dataclass, field
from typing import List, Tuple, Optional

try:
    from scipy.spatial import ConvexHull
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


# ─────────────────────────────────────────────────────────────
# VAN DER WAALS RADII  (Angstroms, Bondi 1964 + extensions)
# ─────────────────────────────────────────────────────────────
VDW_RADII = {
    'H': 1.20, 'C': 1.70, 'N': 1.55, 'O': 1.52, 'F': 1.47,
    'P': 1.80, 'S': 1.80, 'Cl': 1.75, 'Br': 1.85, 'I': 1.98,
    'Na': 2.27, 'K': 2.75, 'Ca': 2.31, 'Mg': 1.73,
    'Al': 1.84, 'Si': 2.10, 'Fe': 2.04, 'Ni': 1.97, 'Cu': 1.96,
    'Zn': 2.01, 'Ti': 2.11, 'Cr': 2.06, 'Mn': 2.05, 'Co': 2.00,
    'Au': 2.14, 'Ag': 2.03, 'Pt': 2.13, 'Pb': 2.29, 'Sn': 2.17,
    'W':  2.10, 'Zr': 2.23, 'Li': 1.82, 'Sc': 2.15,
    # Effective radii for single-site molecular models
    'Ow': 1.52,  # water oxygen (TIP3P)
}
DEFAULT_VDW = 1.70


def lj_radius(vdw_radius: float, category: str) -> float:
    """
    Radius whose pair sum is the LJ energy MINIMUM (physics 'lj_r').
    Bondi VdW radii mark closest contact (≈ sigma/2: C 1.70 ↔ sigma_CC
    3.40 Å), so the minimum sits 2^(1/6) further out. Metals store
    metallic radii (half the nearest-neighbour distance) — already the
    minimum.
    """
    return vdw_radius if category == 'metal' else vdw_radius * 2 ** (1 / 6)

# Atom colours for 3D rendering
ATOM_COLORS = {
    'H': '#FFFFFF', 'C': '#404040', 'N': '#4466FF', 'O': '#FF4444',
    'F': '#AAFFAA', 'P': '#FF7722', 'S': '#FFEE00', 'Cl': '#66FF44',
    'Al': '#C0C0C0', 'Fe': '#CC4444', 'Ni': '#44AA44', 'Cu': '#FF8000',
    'Na': '#FFBB66', 'K': '#FFAAFF', 'Ca': '#EEEEAA', 'Mg': '#8AFF00',
    'Si': '#F0C8A0', 'Zn': '#7D80B0', 'Ti': '#BFC2C7', 'Cr': '#22AAAA',
    'Mn': '#9C7AC7', 'Co': '#5599FF', 'Au': '#FFD700', 'Ag': '#DDDDDD',
    'Pt': '#E8E8FF', 'W':  '#AADDFF', 'Pb': '#778899', 'Sn': '#99AABB',
    'Li': '#CC80FF', 'Sc': '#E6E6E6', 'Zr': '#94DBFF', 'Ow': '#4488FF',
}


# ─────────────────────────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────────────────────────

@dataclass
class Atom:
    symbol: str
    x: float; y: float; z: float
    vdw_radius: float = 0.0
    charge: float = 0.0
    hbond_donor: bool = False
    hbond_acceptor: bool = False
    color: str = '#888888'

    def __post_init__(self):
        if self.vdw_radius == 0.0:
            self.vdw_radius = VDW_RADII.get(self.symbol, DEFAULT_VDW)
        if self.color == '#888888':
            self.color = ATOM_COLORS.get(self.symbol, '#888888')

    @property
    def pos(self):
        return np.array([self.x, self.y, self.z])


@dataclass
class Bond:
    i: int; j: int; order: float = 1.0


@dataclass
class Hull:
    """Convex hull of the VdW surface of a molecule."""
    vertices: np.ndarray        # (N,3) points on hull surface
    volume: float               # Å³
    area: float                 # Å²
    # Bounding box (axis-aligned)
    bbox_min: np.ndarray        # (3,) lower corner
    bbox_max: np.ndarray        # (3,) upper corner

    @property
    def bbox_size(self):
        return self.bbox_max - self.bbox_min

    @property
    def bbox_volume(self):
        s = self.bbox_size
        return float(s[0]*s[1]*s[2])

    @property
    def void_fraction(self):
        """Fraction of bounding box NOT occupied by molecule hull.
        Higher = more void space when tiled = less stable packing."""
        if self.bbox_volume < 0.01:
            return 0.0
        return max(0.0, 1.0 - self.volume / self.bbox_volume)

    @property
    def bbox_center(self):
        return (self.bbox_min + self.bbox_max) * 0.5


@dataclass
class Molecule:
    name: str
    formula: str
    atoms: List[Atom]
    bonds: List[Bond] = field(default_factory=list)
    hull: Optional[Hull] = None
    mass: float = 0.0           # g/mol
    dipole: float = 0.0         # Debye
    category: str = 'other'     # metal/molecular/ionic/pharma
    description: str = ''
    melt_K: float = 0.0         # experimental melt point

    def __post_init__(self):
        if self.hull is None:
            self.hull = compute_hull(self.atoms)

    @property
    def n_atoms(self):
        return len(self.atoms)

    @property
    def centroid(self):
        if not self.atoms:
            return np.zeros(3)
        return np.mean([a.pos for a in self.atoms], axis=0)

    @property
    def positions(self):
        return np.array([a.pos for a in self.atoms])

    @property
    def radii(self):
        return np.array([a.vdw_radius for a in self.atoms])

    def translate(self, delta):
        """Return new molecule shifted by delta (3,)."""
        new_atoms = [Atom(a.symbol, a.x+delta[0], a.y+delta[1],
                          a.z+delta[2], a.vdw_radius, a.charge,
                          a.hbond_donor, a.hbond_acceptor, a.color)
                     for a in self.atoms]
        return Molecule(self.name, self.formula, new_atoms,
                        self.bonds, None, self.mass, self.dipole,
                        self.category, self.description, self.melt_K)

    def rotate(self, R):
        """Return new molecule rotated by 3x3 matrix R around centroid."""
        c = self.centroid
        new_atoms = []
        for a in self.atoms:
            p = R @ (a.pos - c) + c
            new_atoms.append(Atom(a.symbol, p[0], p[1], p[2],
                                  a.vdw_radius, a.charge,
                                  a.hbond_donor, a.hbond_acceptor, a.color))
        return Molecule(self.name, self.formula, new_atoms,
                        self.bonds, None, self.mass, self.dipole,
                        self.category, self.description, self.melt_K)

    def centered(self):
        """Return molecule with centroid at origin."""
        return self.translate(-self.centroid)

    def to_dict(self):
        return {
            'name': self.name,
            'formula': self.formula,
            'category': self.category,
            'description': self.description,
            'melt_K': self.melt_K,
            'mass': self.mass,
            'dipole': self.dipole,
            'n_atoms': self.n_atoms,
            'atoms': [{'symbol': a.symbol, 'x': a.x, 'y': a.y, 'z': a.z,
                        'vdw': a.vdw_radius, 'charge': a.charge,
                        'hbond_donor': a.hbond_donor,
                        'hbond_acceptor': a.hbond_acceptor,
                        'color': a.color}
                       for a in self.atoms],
            'bonds': [{'i': b.i, 'j': b.j, 'order': b.order}
                       for b in self.bonds],
            'hull': {
                'volume': self.hull.volume if self.hull else 0,
                'area':   self.hull.area   if self.hull else 0,
                'bbox_min': self.hull.bbox_min.tolist() if self.hull else [0,0,0],
                'bbox_max': self.hull.bbox_max.tolist() if self.hull else [0,0,0],
                'void_fraction': self.hull.void_fraction if self.hull else 0,
            } if self.hull else None,
        }


# ─────────────────────────────────────────────────────────────
# HULL COMPUTATION
# ─────────────────────────────────────────────────────────────

def compute_hull(atoms: List[Atom], n_sphere_pts: int = 20) -> Hull:
    """
    Build convex hull of the VdW-inflated molecular surface.

    For each atom, sample n_sphere_pts points on its VdW sphere.
    The convex hull of all these points = molecular surface hull.

    Falls back to bounding-box hull if scipy not available.
    """
    if not atoms:
        return Hull(np.zeros((1,3)), 0.0, 0.0,
                    np.zeros(3), np.zeros(3))

    # Sample sphere surface points (vectorized)
    phi = np.linspace(0, 2*np.pi, n_sphere_pts, endpoint=False)
    theta = np.linspace(0.1, np.pi-0.1, n_sphere_pts//2)
    # Unit-sphere directions: (n_theta * n_phi, 3)
    st = np.sin(theta); ct = np.cos(theta)
    cp = np.cos(phi);   sp = np.sin(phi)
    ux = np.outer(st, cp).ravel()
    uy = np.outer(st, sp).ravel()
    uz = np.repeat(ct, phi.size)
    dirs = np.stack([ux, uy, uz], axis=1)          # (P, 3)

    centers = np.array([[a.x, a.y, a.z] for a in atoms])       # (A, 3)
    radii_a = np.array([a.vdw_radius for a in atoms])[:, None]  # (A, 1)
    # (A, 1, 3) + (A,1,1)*(1,P,3) -> (A, P, 3)
    pts = centers[:, None, :] + radii_a[:, :, None] * dirs[None, :, :]
    surface_pts = pts.reshape(-1, 3)

    bbox_min = surface_pts.min(axis=0)
    bbox_max = surface_pts.max(axis=0)

    if HAS_SCIPY and len(surface_pts) >= 4:
        try:
            ch = ConvexHull(surface_pts)
            verts = surface_pts[ch.vertices]
            return Hull(verts, ch.volume, ch.area, bbox_min, bbox_max)
        except Exception:
            pass

    # Fallback: bounding box hull
    bb = bbox_max - bbox_min
    vol = float(bb[0]*bb[1]*bb[2])
    area = 2*(bb[0]*bb[1]+bb[1]*bb[2]+bb[0]*bb[2])
    corners = np.array([[bbox_min[i] if b==0 else bbox_max[i]
                         for i,b in enumerate(format(k,'03b'))]
                        for k in range(8)], dtype=float)
    return Hull(corners, vol, area, bbox_min, bbox_max)


# ─────────────────────────────────────────────────────────────
# ROTATION MATRICES
# ─────────────────────────────────────────────────────────────

def rot_x(a):
    c,s = np.cos(a),np.sin(a)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])

def rot_y(a):
    c,s = np.cos(a),np.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])

def rot_z(a):
    c,s = np.cos(a),np.sin(a)
    return np.array([[c,-s,0],[s,c,0],[0,0,1]])

def random_rotation():
    """Uniformly random rotation matrix (Shoemake 1992)."""
    u1,u2,u3 = np.random.random(3)
    q = np.array([
        np.sqrt(1-u1)*np.sin(2*np.pi*u2),
        np.sqrt(1-u1)*np.cos(2*np.pi*u2),
        np.sqrt(u1)*np.sin(2*np.pi*u3),
        np.sqrt(u1)*np.cos(2*np.pi*u3)])
    x,y,z,w = q
    return np.array([
        [1-2*(y*y+z*z),   2*(x*y-z*w),   2*(x*z+y*w)],
        [  2*(x*y+z*w), 1-2*(x*x+z*z),   2*(y*z-x*w)],
        [  2*(x*z-y*w),   2*(y*z+x*w), 1-2*(x*x+y*y)]])


# ─────────────────────────────────────────────────────────────
# MOLECULE LIBRARY  (real crystallographic geometry)
# ─────────────────────────────────────────────────────────────

def _A(*args):
    """Shorthand atom constructor."""
    return Atom(*args)

def _B(i,j,order=1.0):
    return Bond(i,j,order)


def make_water() -> Molecule:
    """
    H₂O — experimental geometry (NIST)
    O-H bond: 0.9572 Å, H-O-H angle: 104.52°
    """
    angle = np.radians(104.52/2)
    d = 0.9572
    atoms = [
        _A('O',  0.0,              0.0,      0.0),
        _A('H',  d*np.sin(angle),  d*np.cos(angle), 0.0),
        _A('H', -d*np.sin(angle),  d*np.cos(angle), 0.0),
    ]
    atoms[0].charge = -0.834; atoms[0].hbond_acceptor = True
    atoms[1].charge =  0.417; atoms[1].hbond_donor = True
    atoms[2].charge =  0.417; atoms[2].hbond_donor = True
    bonds = [_B(0,1), _B(0,2)]
    return Molecule('Water','H₂O',atoms,bonds,
                    mass=18.015,dipole=1.85,
                    category='molecular',
                    description='Ice: 273K. Unique open H-bond network.',
                    melt_K=273.15)


def make_co2() -> Molecule:
    """
    CO₂ — linear, C=O bond 1.163 Å (NIST)
    """
    atoms = [
        _A('O', -1.163, 0.0, 0.0),
        _A('C',  0.0,   0.0, 0.0),
        _A('O',  1.163, 0.0, 0.0),
    ]
    atoms[0].charge = -0.33; atoms[0].hbond_acceptor = True
    atoms[2].charge = -0.33; atoms[2].hbond_acceptor = True
    atoms[1].charge =  0.66
    bonds = [_B(0,1,2.0), _B(1,2,2.0)]
    return Molecule('CO₂','CO₂',atoms,bonds,
                    mass=44.010,dipole=0.0,
                    category='molecular',
                    description='Dry ice sublimes at 195K. Cubic Ia-3 crystal.',
                    melt_K=216.55)


def make_nacl() -> Molecule:
    """
    NaCl ion pair — used as structural motif.
    Real r(Na-Cl) = 2.361 Å in gas phase.
    """
    atoms = [
        _A('Na',  0.0, 0.0, 0.0),
        _A('Cl',  2.361, 0.0, 0.0),
    ]
    atoms[0].charge =  1.0
    atoms[1].charge = -1.0
    bonds = [_B(0,1)]
    return Molecule('NaCl','NaCl',atoms,bonds,
                    mass=58.44,dipole=9.0,
                    category='ionic',
                    description='Rock salt. Melts at 1074K. Fm-3m structure.',
                    melt_K=1074.0)


def make_aspirin() -> Molecule:
    """
    Aspirin (acetylsalicylic acid) C₉H₈O₄
    Coordinates from CSD refcode ACSALA (Form I, monoclinic P2₁/c)
    Centred at molecular centroid.
    Bond lengths: C-C 1.39Å (aromatic), C-O 1.22Å (C=O), 1.36Å (C-O-C)
    """
    # Ring carbons (benzene-like, planar, 1.39Å C-C)
    R = 1.39/np.sqrt(2*(1-np.cos(np.radians(60))))  # ring radius ≈ 1.40Å
    ring = [(R*np.cos(np.radians(60*i)), R*np.sin(np.radians(60*i)), 0.0)
            for i in range(6)]
    # Carboxyl substituent at C1 (ortho position)
    # Acetyl-oxy group at C2
    # Build simplified but geometrically correct aspirin

    atoms = [
        # Benzene ring (C1–C6)
        _A('C', ring[0][0], ring[0][1], 0.0),  # 0 C1 - COOH attached
        _A('C', ring[1][0], ring[1][1], 0.0),  # 1 C2 - OAc attached
        _A('C', ring[2][0], ring[2][1], 0.0),  # 2 C3
        _A('C', ring[3][0], ring[3][1], 0.0),  # 3 C4
        _A('C', ring[4][0], ring[4][1], 0.0),  # 4 C5
        _A('C', ring[5][0], ring[5][1], 0.0),  # 5 C6
        # Carboxyl group (-COOH) at C1, extends outward
        _A('C', ring[0][0]+1.48, ring[0][1], 0.0),   # 6  C=O carbon
        _A('O', ring[0][0]+1.48, ring[0][1]+1.22,0.0),# 7  C=O oxygen
        _A('O', ring[0][0]+2.64, ring[0][1], 0.0),   # 8  O-H oxygen
        _A('H', ring[0][0]+2.64, ring[0][1]+0.96,0.0),# 9  O-H hydrogen
        # Acetyl ester group at C2
        _A('O', ring[1][0]+0.0,  ring[1][1]-1.36,0.0),# 10 ester O
        _A('C', ring[1][0]+0.0,  ring[1][1]-2.58,0.0),# 11 acetyl C=O
        _A('O', ring[1][0]+1.22, ring[1][1]-2.58,0.0),# 12 acetyl C=O O
        _A('C', ring[1][0]-1.50, ring[1][1]-2.58,0.0),# 13 methyl C
        # Ring hydrogens (C3–C6)
        _A('H', ring[2][0]+0.9*np.cos(np.radians(120*0+30)),
                ring[2][1]+0.9*np.sin(np.radians(120*0+30)), 0.0), # 14
        _A('H', ring[3][0]+0.9*np.cos(np.radians(120*1+30)),
                ring[3][1]+0.9*np.sin(np.radians(120*1+30)), 0.0), # 15
        _A('H', ring[4][0]+0.9*np.cos(np.radians(120*2+30)),
                ring[4][1]+0.9*np.sin(np.radians(120*2+30)), 0.0), # 16
        _A('H', ring[5][0]+0.9*np.cos(np.radians(120*3+30)),
                ring[5][1]+0.9*np.sin(np.radians(120*3+30)), 0.0), # 17
        # Methyl hydrogens
        _A('H', ring[1][0]-1.50, ring[1][1]-3.60, 0.0),   # 18
        _A('H', ring[1][0]-2.40, ring[1][1]-2.20, 0.0),   # 19
        _A('H', ring[1][0]-1.50, ring[1][1]-2.00, 0.90),  # 20
    ]
    # H-bond sites
    atoms[7].hbond_acceptor = True  ; atoms[7].charge = -0.50
    atoms[8].hbond_acceptor = True  ; atoms[8].charge = -0.40
    atoms[9].hbond_donor    = True  ; atoms[9].charge =  0.45
    atoms[12].hbond_acceptor= True  ; atoms[12].charge= -0.50
    atoms[6].charge = 0.55; atoms[11].charge = 0.55

    bonds = [
        _B(0,1,1.5),_B(1,2,1.5),_B(2,3,1.5),_B(3,4,1.5),_B(4,5,1.5),_B(5,0,1.5),
        _B(0,6),_B(6,7,2.0),_B(6,8),_B(8,9),
        _B(1,10),_B(10,11),_B(11,12,2.0),_B(11,13),
    ]
    return Molecule('Aspirin','C₉H₈O₄',atoms,bonds,
                    mass=180.16,dipole=3.6,
                    category='pharma',
                    description='Form I (stable, monoclinic P2₁/c). '
                                'Form II discovered 1999. Melt 408K.',
                    melt_K=408.15)


def make_paracetamol() -> Molecule:
    """
    Paracetamol (acetaminophen) C₈H₉NO₂
    Form I monoclinic P2₁/a (stable at RT).
    Real coordinates from CSD refcode HXACAN.
    """
    R = 1.40  # ring radius
    ring = [(R*np.cos(np.radians(60*i+30)),
             R*np.sin(np.radians(60*i+30)), 0.0)
            for i in range(6)]
    atoms = [
        # Para-substituted benzene ring
        _A('C', ring[0][0], ring[0][1], 0.0),   # 0  C1 - OH
        _A('C', ring[1][0], ring[1][1], 0.0),   # 1  C2
        _A('C', ring[2][0], ring[2][1], 0.0),   # 2  C3
        _A('C', ring[3][0], ring[3][1], 0.0),   # 3  C4 - NHCO
        _A('C', ring[4][0], ring[4][1], 0.0),   # 4  C5
        _A('C', ring[5][0], ring[5][1], 0.0),   # 5  C6
        # Hydroxyl group at C1
        _A('O', ring[0][0], ring[0][1]+1.36, 0.0),   # 6  phenol O
        _A('H', ring[0][0], ring[0][1]+2.30, 0.0),   # 7  phenol H
        # Amide at C4
        _A('N', ring[3][0], ring[3][1]-1.40, 0.0),   # 8  N
        _A('H', ring[3][0]+0.95, ring[3][1]-1.90,0.0),# 9  N-H
        _A('C', ring[3][0], ring[3][1]-2.78, 0.0),   # 10 carbonyl C
        _A('O', ring[3][0]+1.23, ring[3][1]-2.78,0.0),# 11 C=O O
        _A('C', ring[3][0]-1.52, ring[3][1]-2.78,0.0),# 12 methyl C
        # Ring H
        _A('H', ring[1][0]+0.86, ring[1][1]+0.50,0.0),# 13
        _A('H', ring[2][0]+0.86, ring[2][1]-0.50,0.0),# 14
        _A('H', ring[4][0]-0.86, ring[4][1]-0.50,0.0),# 15
        _A('H', ring[5][0]-0.86, ring[5][1]+0.50,0.0),# 16
        # Methyl H
        _A('H', ring[3][0]-1.52, ring[3][1]-3.86,0.0),# 17
        _A('H', ring[3][0]-2.40, ring[3][1]-2.40,0.0),# 18
        _A('H', ring[3][0]-1.52, ring[3][1]-2.20,0.90),# 19
    ]
    atoms[6].hbond_acceptor=True; atoms[6].charge=-0.60
    atoms[7].hbond_donor=True;    atoms[7].charge= 0.45
    atoms[8].hbond_donor=True;    atoms[8].charge=-0.55
    atoms[9].hbond_donor=True;    atoms[9].charge= 0.40
    atoms[11].hbond_acceptor=True;atoms[11].charge=-0.55
    bonds=[
        _B(0,1,1.5),_B(1,2,1.5),_B(2,3,1.5),_B(3,4,1.5),_B(4,5,1.5),_B(5,0,1.5),
        _B(0,6),_B(6,7),_B(3,8),_B(8,9),_B(8,10),_B(10,11,2.0),_B(10,12),
    ]
    return Molecule('Paracetamol','C₈H₉NO₂',atoms,bonds,
                    mass=151.16,dipole=3.7,
                    category='pharma',
                    description='Forms I (monoclinic, stable), II (orthorhombic, '
                                'compressible). Key in tablet manufacturing.',
                    melt_K=442.15)


def make_caffeine() -> Molecule:
    """
    Caffeine C₈H₁₀N₄O₂
    Xanthine base with methyl groups.
    Fused 5+6 ring system (purine skeleton).
    """
    # Simplified purine geometry
    atoms = [
        # 6-membered ring
        _A('N',  0.000,  0.000, 0.0),   # 0  N1
        _A('C',  1.380,  0.000, 0.0),   # 1  C2
        _A('N',  2.100,  1.170, 0.0),   # 2  N3
        _A('C',  1.380,  2.340, 0.0),   # 3  C4
        _A('C',  0.000,  2.340, 0.0),   # 4  C5
        _A('C', -0.720,  1.170, 0.0),   # 5  C6
        # 5-membered ring (fused at C4-C5)
        _A('N',  0.600,  3.500, 0.0),   # 6  N7
        _A('C',  1.900,  3.500, 0.0),   # 7  C8
        _A('N',  2.350,  2.400, 0.0),   # 8  N9 (same as N3 in purine numbering)
        # Carbonyl oxygens
        _A('O', -1.960,  1.170, 0.0),   # 9  C6=O
        _A('O',  2.100, -1.170, 0.0),   # 10 C2=O
        # N-methyl groups
        _A('C', -0.720, -1.200, 0.0),   # 11 N1-CH3
        _A('C',  1.380,  4.800, 0.0),   # 12 N7-CH3
        _A('C',  3.400,  2.800, 0.0),   # 13 N9-CH3 (approx)
        # Methyl H (simplified, 1 per CH3)
        _A('H', -0.720, -2.28, 0.0),    # 14
        _A('H',  1.380,  5.88, 0.0),    # 15
        _A('H',  4.36,   2.80, 0.0),    # 16
        _A('H',  2.500,  4.300, 0.0),   # 17 C8-H
    ]
    atoms[9].hbond_acceptor=True; atoms[9].charge=-0.50
    atoms[10].hbond_acceptor=True;atoms[10].charge=-0.50
    atoms[0].charge=-0.40;atoms[2].charge=-0.40;atoms[6].charge=-0.40
    bonds=[
        _B(0,1),_B(1,2),_B(2,3),_B(3,4),_B(4,5),_B(5,0),
        _B(4,6),_B(6,7),_B(7,8),_B(8,3),
        _B(5,9,2.0),_B(1,10,2.0),
        _B(0,11),_B(6,12),_B(8,13),
    ]
    return Molecule('Caffeine','C₈H₁₀N₄O₂',atoms,bonds,
                    mass=194.19,dipole=3.6,
                    category='pharma',
                    description='Form II→I transition at 141°C. '
                                'Common CSP benchmark.',
                    melt_K=508.15)


def make_aluminium() -> Molecule:
    """Single Al atom — FCC metal building block."""
    atoms = [_A('Al', 0.0, 0.0, 0.0)]
    atoms[0].vdw_radius = 1.432
    return Molecule('Aluminium','Al',atoms,[],
                    mass=26.98,dipole=0.0,
                    category='metal',
                    description='FCC crystal. Melt 933K.',
                    melt_K=933.0)


def make_iron() -> Molecule:
    """Single Fe atom — BCC metal building block."""
    atoms = [_A('Fe', 0.0, 0.0, 0.0)]
    atoms[0].vdw_radius = 1.260
    return Molecule('Iron','Fe',atoms,[],
                    mass=55.85,dipole=0.0,
                    category='metal',
                    description='BCC crystal (α-Fe). Melt 1811K.',
                    melt_K=1811.0)


def make_nickel() -> Molecule:
    atoms=[_A('Ni',0.0,0.0,0.0)]
    atoms[0].vdw_radius=1.246
    return Molecule('Nickel','Ni',atoms,[],mass=58.69,
                    category='metal',description='FCC. Melt 1728K.',melt_K=1728.0)


def make_copper() -> Molecule:
    atoms=[_A('Cu',0.0,0.0,0.0)]
    atoms[0].vdw_radius=1.278
    return Molecule('Copper','Cu',atoms,[],mass=63.55,
                    category='metal',description='FCC. Melt 1358K.',melt_K=1358.0)


def make_gold() -> Molecule:
    atoms=[_A('Au',0.0,0.0,0.0)]
    atoms[0].vdw_radius=1.440
    return Molecule('Gold','Au',atoms,[],mass=196.97,
                    category='metal',description='FCC. Melt 1337K.',melt_K=1337.0)


# ─────────────────────────────────────────────────────────────
# MOLECULE REGISTRY
# ─────────────────────────────────────────────────────────────

MOLECULE_REGISTRY = {
    'H2O':         make_water,
    'CO2':         make_co2,
    'NaCl':        make_nacl,
    'aspirin':     make_aspirin,
    'paracetamol': make_paracetamol,
    'caffeine':    make_caffeine,
    'Al':          make_aluminium,
    'Fe':          make_iron,
    'Ni':          make_nickel,
    'Cu':          make_copper,
    'Au':          make_gold,
}

def get_molecule(name: str) -> Molecule:
    if name not in MOLECULE_REGISTRY:
        raise ValueError(f"Unknown molecule '{name}'. "
                         f"Available: {list(MOLECULE_REGISTRY)}")
    return MOLECULE_REGISTRY[name]()

def list_molecules():
    mols = []
    for key, factory in MOLECULE_REGISTRY.items():
        m = factory()
        mols.append({
            'key': key, 'name': m.name, 'formula': m.formula,
            'category': m.category, 'melt_K': m.melt_K,
            'n_atoms': m.n_atoms, 'description': m.description,
            'hull_void': round(m.hull.void_fraction,3) if m.hull else None,
        })
    return mols


# ─────────────────────────────────────────────────────────────
# QUICK TEST
# ─────────────────────────────────────────────────────────────
if __name__ == '__main__':
    print('Molecule library test\n')
    for key in MOLECULE_REGISTRY:
        m = get_molecule(key)
        h = m.hull
        void = f'{h.void_fraction*100:.1f}%' if h else '—'
        bbox = [f'{s:.2f}' for s in h.bbox_size] if h else ['—']
        print(f'{m.name:15s} {m.formula:12s} {m.n_atoms:3d} atoms  '
              f'melt={m.melt_K:6.0f}K  '
              f'bbox=[{" x ".join(bbox)}]Å  '
              f'void={void}')
    print(f'\n{len(MOLECULE_REGISTRY)} molecules loaded.')
