# -*- coding: utf-8 -*-
# engine/meam.py
#
# SECOND-NEAREST-NEIGHBOUR MODIFIED EMBEDDED-ATOM METHOD (2NN-MEAM)
#
# MEAM extends EAM with angular electron densities (needed for elements
# like Zr and Sc) and many-body screening:
#   ρ_i^(0..3)   spherical, dipole, quadrupole, octupole partial densities
#   Γ_i = Σ_l t_l (ρ^(l)/ρ^(0))²,  ρ̄_i = ρ^(0) G(Γ),  G = 2/(1+e^-Γ) (ibar 3)
#   E = Σ_i A E_c (ρ̄_i/ρ_ref) ln(ρ̄_i/ρ_ref) + ½ Σ_ij S_ij φ_ij(r_ij)
#   S_ij: screening by third atoms k (Cmin/Cmax ellipse) × radial cutoff
#   φ: derived from the Rose equation of state of each reference lattice,
#      with the 2NN series / B2 cross-term construction
# Follows LAMMPS pair_style meam (meam_setup_done, meam_dens_*, meam_force)
# for the options the published Al-Zr potential uses (ialloy 2,
# erose_form 2, augt1 0, ibar 3, no ZBL). Validated against NIST's
# LAMMPS-computed energies:  python meam.py
#
# Potential: Fereidonnejad, Moghaddam, Moaddeli, Comput. Mater. Sci. 213,
# 111685 (2022), Al-Zr 2NN-MEAM, NIST IPR (references/potentials/).

import os
import re
import math
import numpy as np
import torch
from functools import lru_cache

HERE = os.path.dirname(os.path.abspath(__file__))
POT_DIR = os.path.join(HERE, 'references', 'potentials')
POTENTIALS = {'AlZr': ('Fereidonnejad-2022-AlZr.library.meam',
                       'Fereidonnejad-2022-AlZr.meam', ('Al', 'Zr'))}
torch.set_default_dtype(torch.float64)

NN1 = {'fcc': 12, 'hcp': 12, 'bcc': 8, 'b2': 8, 'b1': 6, 'dia': 4, 'l12': 12}
NN2 = {   # (Z2, arat, numscr) — LAMMPS get_Zij2
    'fcc': (6, math.sqrt(2.0), 4), 'hcp': (6, math.sqrt(2.0), 4),
    'bcc': (6, 2.0 / math.sqrt(3.0), 4), 'b2': (6, 2.0 / math.sqrt(3.0), 4),
    'b1': (12, math.sqrt(2.0), 2), 'l12': (6, math.sqrt(2.0), 4),
}


def fcut(x):
    """LAMMPS MEAM cutoff: 0 (x≤0) … (1-(1-x)^4)^2 … 1 (x≥1)."""
    if isinstance(x, torch.Tensor):
        y = torch.clamp(x, 0.0, 1.0)
        return (1.0 - (1.0 - y) ** 4) ** 2
    x = min(max(x, 0.0), 1.0)
    return (1.0 - (1.0 - x) ** 4) ** 2


def _spline(f):
    """LAMMPS interpolate_meams(): cubic coefficients on a uniform grid."""
    n = len(f)
    c1 = np.zeros(n)
    c1[0] = f[1] - f[0]
    c1[1] = 0.5 * (f[2] - f[0])
    c1[n - 2] = 0.5 * (f[n - 1] - f[n - 3])
    c1[n - 1] = 0.0
    c1[2:n - 2] = ((f[0:n - 4] - f[4:n]) + 8.0 * (f[3:n - 1] - f[1:n - 3])) / 12.0
    c2 = np.zeros(n)
    c3 = np.zeros(n)
    c2[:n - 1] = 3.0 * (f[1:] - f[:-1]) - 2.0 * c1[:-1] - c1[1:]
    c3[:n - 1] = c1[:-1] + c1[1:] - 2.0 * (f[1:] - f[:-1])
    return np.stack([c3, c2, c1, np.asarray(f, float)], axis=1)


class MEAM:
    def __init__(self, library, param, elements):
        self.el = list(elements)
        n = len(self.el)
        lib = self._read_library(library)
        P = {k: lib[e][k] for e in self.el for k in ()}  # noqa: placeholder
        self.lat = [lib[e]['lat'] for e in self.el]
        self.alpha = np.zeros((n, n)); self.re = np.zeros((n, n)); self.Ec = np.zeros((n, n))
        self.attrac = np.zeros((n, n)); self.repuls = np.zeros((n, n))
        self.nn2 = np.zeros((n, n), int)
        self.lattce = [[None] * n for _ in range(n)]
        self.Cmin = np.full((n, n, n), 2.0); self.Cmax = np.full((n, n, n), 2.8)
        self.beta = np.zeros((n, 4)); self.t = np.zeros((n, 4)); self.A = np.zeros(n)
        self.rho0 = np.ones(n); self.ibar = np.zeros(n, int)
        for a, e in enumerate(self.el):
            L = lib[e]
            self.alpha[a, a] = L['alpha']; self.Ec[a, a] = L['esub']; self.A[a] = L['asub']
            self.beta[a] = [L['b0'], L['b1'], L['b2'], L['b3']]
            self.t[a] = [L['t0'], L['t1'], L['t2'], L['t3']]
            self.rho0[a] = L['rozero']; self.ibar[a] = L['ibar']
            self.lattce[a][a] = L['lat']
            self.re[a, a] = L['alat'] / math.sqrt(2.0) if L['lat'] == 'fcc' else \
                (L['alat'] * math.sqrt(3.0) / 2.0 if L['lat'] == 'bcc' else L['alat'])
        self.rc, self.delr = 4.0, 0.1
        self._read_param(param)
        # LAMMPS alloyparams(): symmetric pair parameters, Cmin/Cmax i>j ← j<i
        for a in range(n):
            for b in range(a):
                for arr in (self.alpha, self.re, self.Ec, self.attrac, self.repuls, self.nn2):
                    arr[a, b] = arr[b, a]
                self.lattce[a][b] = self.lattce[b][a]
                for k in range(n):
                    self.Cmin[a, b, k] = self.Cmin[b, a, k]
                    self.Cmax[a, b, k] = self.Cmax[b, a, k]
        self.rho_ref = np.array([self._rho_ref(a) for a in range(n)])
        self._tabulate_phi()

    # ── input ─────────────────────────────────────────────────
    @staticmethod
    def _read_library(path):
        toks = []
        for line in open(path, encoding='utf-8', errors='replace'):
            line = line.split('#')[0].strip()
            if line:
                toks += line.replace("'", ' ').split()
        out, k = {}, 0
        while k < len(toks):
            name, lat = toks[k], toks[k + 1].lower()
            v = list(map(float, toks[k + 2:k + 19]))
            out[name] = dict(lat=lat, z=v[0], atwt=v[2], alpha=v[3], b0=v[4], b1=v[5],
                             b2=v[6], b3=v[7], alat=v[8], esub=v[9], asub=v[10],
                             t0=v[11], t1=v[12], t2=v[13], t3=v[14], rozero=v[15], ibar=int(v[16]))
            k += 19
        return out

    def _read_param(self, path):
        for line in open(path, encoding='utf-8', errors='replace'):
            line = line.split('#')[0].strip()
            m = re.match(r"(\w+)\s*(?:\(([\d,\s]+)\))?\s*=\s*'?([\w.+\-]+)'?", line)
            if not m:
                continue
            key, idx, val = m.group(1), m.group(2), m.group(3)
            ix = [int(s) - 1 for s in idx.split(',')] if idx else []
            if key == 'rc':
                self.rc = float(val)
            elif key == 'delr':
                self.delr = float(val)
            elif key in ('Ec', 're', 'alpha', 'attrac', 'repuls'):
                getattr(self, key)[ix[0], ix[1]] = float(val)
            elif key == 'nn2':
                self.nn2[ix[0], ix[1]] = int(float(val))
            elif key == 'lattce':
                self.lattce[ix[0]][ix[1]] = val.lower()
            elif key in ('Cmin', 'Cmax'):
                getattr(self, key)[ix[0], ix[1], ix[2]] = float(val)
            elif key == 'rho0':
                self.rho0[ix[0]] = float(val)
            elif key in ('ialloy', 'erose_form', 'augt1', 'zbl', 'emb_lin_neg', 'bkgd_dyn'):
                if key in ('ialloy', 'erose_form') and int(float(val)) != 2:
                    raise NotImplementedError(f'{key} = {val}')
                if key == 'augt1' and int(float(val)) != 0:
                    raise NotImplementedError('augt1')
                if key == 'zbl' and int(float(val)) != 0:
                    raise NotImplementedError('zbl')

    # ── reference-structure functions (numpy) ─────────────────
    def _scrn2(self, lat, cmin, cmax):
        z2, arat, numscr = NN2[lat]
        C = 4.0 / (arat * arat) - 1.0
        return z2, arat, fcut((C - cmin) / (cmax - cmin)) ** numscr

    def _G(self, gam, a):
        if self.ibar[a] == 3:
            return 2.0 / (1.0 + math.exp(-gam))
        return math.sqrt(max(1.0 + gam, 0.0))

    def _rho_ref(self, a):
        lat = self.lattce[a][a]
        Z = NN1[lat]
        s = {'hcp': (0.0, 0.0, 1.0 / 3.0)}.get(lat, (0.0, 0.0, 0.0))
        gam = (self.t[a, 1] * s[0] + self.t[a, 2] * s[1] + self.t[a, 3] * s[2]) / (Z * Z)
        Gbar = self._G(gam, a) if self.ibar[a] > 0 else 1.0
        rho0 = self.rho0[a] * Z
        if self.nn2[a, a]:
            z2, arat, scrn = self._scrn2(lat, self.Cmin[a, a, a], self.Cmax[a, a, a])
            rho0 += z2 * scrn * self.rho0[a] * math.exp(-self.beta[a, 0] * (arat - 1.0))
        return rho0 * Gbar

    def _densref(self, r, a, b):
        """Partial densities of atom a (…1) and b (…2) in the a-b reference
        lattice at nearest-neighbour distance r (LAMMPS get_densref)."""
        lat = self.lattce[a][b]
        a1, a2 = r / self.re[a, a] - 1.0, r / self.re[b, b] - 1.0
        ra = [self.rho0[a] * math.exp(-self.beta[a, l] * a1) for l in range(4)]
        rb = [self.rho0[b] * math.exp(-self.beta[b, l] * a2) for l in range(4)]
        Z = NN1[lat]
        rho01, rho02 = Z * rb[0], Z * ra[0]
        r31 = r32 = 0.0
        if lat == 'hcp':
            r31, r32 = rb[3] ** 2 / 3.0, ra[3] ** 2 / 3.0
        elif lat not in ('fcc', 'bcc', 'b2', 'b1', 'l12'):
            raise NotImplementedError(lat)
        if self.nn2[a, b]:
            z2, arat, scrn = self._scrn2(lat, self.Cmin[a, a, b], self.Cmax[a, a, b])
            rho01 += z2 * scrn * self.rho0[a] * math.exp(-self.beta[a, 0] * (arat * r / self.re[a, a] - 1.0))
            z2, arat, scrn = self._scrn2(lat, self.Cmin[b, b, a], self.Cmax[b, b, a])
            rho02 += z2 * scrn * self.rho0[b] * math.exp(-self.beta[b, 0] * (arat * r / self.re[b, b] - 1.0))
        return rho01, r31, rho02, r32

    def _erose(self, r, a, b):
        re_, al = self.re[a, b], self.alpha[a, b]
        astar = al * (r / re_ - 1.0)
        a3 = self.attrac[a, b] if astar >= 0 else self.repuls[a, b]
        return -self.Ec[a, b] * (1.0 + astar + a3 * astar ** 3) * math.exp(-astar)

    def _embed(self, a, rhobar):
        return self.A[a] * self.Ec[a, a] * rhobar * math.log(rhobar) if rhobar > 0 else 0.0

    def _phi1(self, r, a, b):
        """Pair term from the a-b reference lattice (LAMMPS phi_meam)."""
        if r < 1e-12:
            return 0.0
        rho01, r31, rho02, r32 = self._densref(r, a, b)
        if rho01 <= 1e-14 and rho02 <= 1e-14:
            return 0.0
        g1 = 0.0 if rho01 < 1e-14 else self.t[a, 3] * r31 / rho01 ** 2
        g2 = 0.0 if rho02 < 1e-14 else self.t[b, 3] * r32 / rho02 ** 2
        F1 = self._embed(a, rho01 / self.rho_ref[a] * self._G(g1, a))
        F2 = self._embed(b, rho02 / self.rho_ref[b] * self._G(g2, b))
        Eu = self._erose(r, a, b)
        return (2.0 * Eu - F1 - F2) / NN1[self.lattce[a][b]]

    def _phi_same(self, r, a):
        """φ_aa with the 2NN series (LAMMPS phi_meam + phi_meam_series)."""
        lat = self.lattce[a][a]
        phi = self._phi1(r, a, a)
        if self.nn2[a, a]:
            z2, arat, scrn = self._scrn2(lat, self.Cmin[a, a, a], self.Cmax[a, a, a])
            z1 = NN1[lat]
            if scrn > 0:
                for k in range(1, 11):
                    v = (-z2 * scrn / z1) ** k * self._phi1(r * arat ** k, a, a)
                    if abs(v) < 1e-20:
                        break
                    phi += v
        return phi

    def _phi_pair(self, r, a, b):
        if a == b:
            return self._phi_same(r, a)
        lat = self.lattce[a][b]
        phi = self._phi1(r, a, b)
        if self.nn2[a, b]:
            if lat not in ('b2', 'b1'):
                raise NotImplementedError(f'cross reference lattice {lat}')
            z2, arat, scrn = self._scrn2(lat, self.Cmin[a, a, b], self.Cmax[a, a, b])
            _, _, scrn2 = self._scrn2(lat, self.Cmin[b, b, a], self.Cmax[b, b, a])
            phiaa = self._phi_same(r * arat, a)
            phibb = self._phi_same(r * arat, b)
            z1 = NN1[lat]
            phi -= z2 * scrn / (2 * z1) * phiaa + z2 * scrn2 / (2 * z1) * phibb
        return phi

    def _tabulate_phi(self):
        self.nr = 1000
        self.dr = 1.1 * self.rc / self.nr
        n = len(self.el)
        self.phi = {}
        for a in range(n):
            for b in range(a, n):
                vals = np.array([self._phi_pair(k * self.dr, a, b) for k in range(self.nr)])
                t = torch.as_tensor(_spline(vals))
                self.phi[(a, b)] = self.phi[(b, a)] = t

    def _phi_eval(self, r, a, b):
        p = r / self.dr
        m = torch.clamp(torch.floor(p).long(), 0, self.nr - 2)
        x = torch.clamp(p - m, max=1.0)
        c = self.phi[(a, b)][m]
        return ((c[:, 0] * x + c[:, 1]) * x + c[:, 2]) * x + c[:, 3]

    # ── crystal energy (torch) ────────────────────────────────
    def prepare(self, pos, cell, types, skin=0.0):
        """Pair (i, j, shift) and screening-triplet (pair, k, shift) lists."""
        from scipy.spatial import cKDTree
        from lattice import image_range
        t = np.array([self.el.index(s) for s in types])
        n = len(pos)
        cut_k = 1.1 * (self.rc + skin)
        shifts = image_range(cell, cut_k)
        images = (pos[None, :, :] + (shifts @ cell)[:, None, :]).reshape(-1, 3)
        tree_img = cKDTree(images)
        coo = cKDTree(pos).sparse_distance_matrix(tree_img, cut_k, output_type='coo_matrix')
        ii, kk = coo.row, coo.col
        jj, mm = kk % n, kk // n
        keep = ~((ii == jj) & np.all(shifts[mm] == 0, axis=1))
        ii, jj, mm, dd = ii[keep], jj[keep], mm[keep], coo.data[keep]
        nb = [[] for _ in range(n)]
        for a, b, s, d in zip(ii, jj, mm, dd):
            nb[a].append((b, s, d))
        pi, pj, ps = [], [], []
        tri_p, tri_k, tri_s = [], [], []
        zero = int(np.flatnonzero(np.all(shifts == 0, axis=1))[0])
        for a in range(n):
            for b, s, d in nb[a]:
                if d >= self.rc + skin:
                    continue
                p = len(pi)
                pi.append(a); pj.append(b); ps.append(s)
                for k, sk, dk in nb[a]:
                    if k == b and sk == s:
                        continue
                    tri_p.append(p); tri_k.append(k); tri_s.append(sk)
        sh = torch.as_tensor(shifts.astype(float))
        return {'t': t, 'n': n, 'i': torch.as_tensor(pi), 'j': torch.as_tensor(pj),
                'sj': sh[torch.as_tensor(ps)], 'tp': torch.as_tensor(tri_p),
                'tk': torch.as_tensor(tri_k), 'sk': sh[torch.as_tensor(tri_s)],
                'cmin': torch.as_tensor(self.Cmin[t[pi]][np.arange(len(pi))][:, t[pj]].diagonal()
                                        if False else np.zeros(0)),
                'pair_types': (t[np.array(pi)], t[np.array(pj)]) if pi else (np.zeros(0, int),) * 2}

    def energy(self, pos, cell, prep):
        """Total energy (eV); pos (N,3) and cell (3,3) are torch tensors."""
        t = prep['t']
        i, j = prep['i'], prep['j']
        ti, tj = prep['pair_types']
        dij = pos[j] + prep['sj'] @ cell - pos[i]
        rij = torch.sqrt((dij * dij).sum(-1))
        # screening S_ij = Π_k s_ikj × fcut((rc - r)/delr)
        p, k = prep['tp'], prep['tk']
        rk = pos[k] + prep['sk'] @ cell
        dik = rk - pos[i[p]]
        djk = rk - (pos[j[p]] + prep['sj'][p] @ cell)
        r2ij = rij[p] ** 2
        xik = (dik * dik).sum(-1) / r2ij
        xjk = (djk * djk).sum(-1) / r2ij
        aa = 1.0 - (xik - xjk) ** 2
        tk = t[k.numpy()]
        cmin = torch.as_tensor(self.Cmin[ti[p.numpy()], tj[p.numpy()], tk])
        cmax = torch.as_tensor(self.Cmax[ti[p.numpy()], tj[p.numpy()], tk])
        safe = torch.where(aa > 0, aa, torch.ones_like(aa))
        C = (2.0 * (xik + xjk) + aa - 2.0) / safe
        s = fcut((C - cmin) / (cmax - cmin))
        s = torch.where(aa > 0, s, torch.ones_like(s))
        logS = torch.zeros(len(rij)).index_add(0, p, torch.log(torch.clamp(s, min=1e-300)))
        S = torch.exp(logS) * fcut((self.rc - rij) / self.delr)
        # partial densities from neighbours j
        beta = torch.as_tensor(self.beta[tj])                    # (P,4)
        re_j = torch.as_tensor(self.re[tj, tj])
        rho0j = torch.as_tensor(self.rho0[tj])
        fa = rho0j[:, None] * torch.exp(-beta * (rij / re_j - 1.0)[:, None]) * S[:, None]
        u = dij / rij[:, None]
        n = prep['n']
        rho0 = torch.zeros(n).index_add(0, i, fa[:, 0])
        a1 = torch.zeros(n, 3).index_add(0, i, fa[:, 1:2] * u)
        uu = u[:, :, None] * u[:, None, :]
        a2 = torch.zeros(n, 3, 3).index_add(0, i, fa[:, 2, None, None] * uu)
        a2b = torch.zeros(n).index_add(0, i, fa[:, 2])
        uuu = uu[:, :, :, None] * u[:, None, None, :]
        a3 = torch.zeros(n, 3, 3, 3).index_add(0, i, fa[:, 3, None, None, None] * uuu)
        a3b = torch.zeros(n, 3).index_add(0, i, fa[:, 3:4] * u)
        rho1 = (a1 * a1).sum(-1)
        rho2 = (a2 * a2).sum((-1, -2)) - a2b ** 2 / 3.0
        rho3 = (a3 * a3).sum((-1, -2, -3)) - 0.6 * (a3b * a3b).sum(-1)
        tt = torch.as_tensor(self.t[t])                          # ialloy 2: own t
        gam = (tt[:, 1] * rho1 + tt[:, 2] * rho2 + tt[:, 3] * rho3) / \
            torch.clamp(rho0, min=1e-300) ** 2
        gam = torch.where(rho0 > 0, gam, torch.zeros_like(gam))
        G = 2.0 / (1.0 + torch.exp(-gam))
        rhob = rho0 * G / torch.as_tensor(self.rho_ref[t])
        A = torch.as_tensor(self.A[t] * self.Ec[t, t])
        emb = torch.where(rhob > 0, A * rhob * torch.log(torch.clamp(rhob, min=1e-300)),
                          torch.zeros_like(rhob))
        phi = torch.zeros_like(rij)
        for a in range(len(self.el)):
            for b in range(len(self.el)):
                sel = torch.as_tensor((ti == a) & (tj == b))
                if sel.any():
                    phi = torch.where(sel, self._phi_eval(rij, a, b), phi)
        return emb.sum() + 0.5 * (phi * S).sum()


@lru_cache(maxsize=None)
def get_potential(name='AlZr') -> MEAM:
    lib, par, els = POTENTIALS[name]
    return MEAM(os.path.join(POT_DIR, lib), os.path.join(POT_DIR, par), els)


def energy_forces_stress(atoms, pot: MEAM):
    pos0 = np.array(atoms.get_positions())
    cell0 = np.array(atoms.cell)
    prep = pot.prepare(pos0, cell0, atoms.get_chemical_symbols())
    pos = torch.as_tensor(pos0).clone().requires_grad_(True)
    eps = torch.zeros(3, 3, requires_grad=True)
    S = torch.eye(3) + 0.5 * (eps + eps.T)
    e = pot.energy(pos @ S, torch.as_tensor(cell0) @ S, prep)
    g_pos, g_eps = torch.autograd.grad(e, (pos, eps))
    st = g_eps.numpy() / abs(np.linalg.det(cell0))
    return float(e), -g_pos.numpy(), np.array([st[0, 0], st[1, 1], st[2, 2],
                                               st[1, 2], st[0, 2], st[0, 1]])


def calculator(name='AlZr'):
    from ase.calculators.calculator import Calculator, all_changes

    class MEAMCalculator(Calculator):
        implemented_properties = ['energy', 'free_energy', 'forces', 'stress']

        def calculate(self, atoms=None, properties=('energy',), system_changes=all_changes):
            super().calculate(atoms, properties, system_changes)
            e, f, s = energy_forces_stress(self.atoms, get_potential(name))
            self.results = {'energy': e, 'free_energy': e, 'forces': f, 'stress': s}

    return MEAMCalculator()


if __name__ == '__main__':
    import sys
    sys.path.insert(0, HERE)
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    from ase import Atoms
    from ase.build import bulk
    pot = get_potential('AlZr')
    print(f'Al-Zr 2NN-MEAM: rc {pot.rc}, rho_ref {np.round(pot.rho_ref, 6).tolist()}')
    checks = [   # NIST LAMMPS-computed (static), eV/atom
        ('Al fcc', bulk('Al', 'fcc', a=4.04465078, cubic=True), -3.36),
        ('Al hcp', bulk('Al', 'hcp', a=2.83255636, c=4.79463956), -3.330481449),
        ('Al bcc', bulk('Al', 'bcc', a=3.22691665, cubic=True), -3.241667759),
        ('Zr hcp', bulk('Zr', 'hcp', a=3.23080672, c=5.12522463), -6.363869474),
        ('Zr fcc', bulk('Zr', 'fcc', a=4.53281088, cubic=True), -6.309229463),
        ('Zr bcc', bulk('Zr', 'bcc', a=3.58009491, cubic=True), -6.344960614),
        ('Al3Zr L12', Atoms(['Zr', 'Al', 'Al', 'Al'],
                            scaled_positions=[(0, 0, 0), (0, .5, .5), (.5, 0, .5), (.5, .5, 0)],
                            cell=[4.113452231497525] * 3, pbc=True), -4.597078820968999),
        ('AlZr B2', Atoms(['Al', 'Zr'], scaled_positions=[(0, 0, 0), (.5, .5, .5)],
                          cell=[3.3401] * 3, pbc=True), -5.5112),
    ]
    worst = 0.0
    for name, at, ref in checks:
        e, f, s = energy_forces_stress(at, pot)
        per = e / len(at)
        worst = max(worst, abs(per - ref))
        print(f'  {name:<10} {per:14.9f} eV/atom   NIST {ref:14.9f}   Δ {per - ref:+.2e}')
    print(f'max deviation {worst:.2e} eV/atom')
