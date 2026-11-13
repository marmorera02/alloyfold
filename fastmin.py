# -*- coding: utf-8 -*-
# engine/fastmin.py
#
# COMPILED RIGID-MOLECULE MINIMISER  (Numba)  for lattice._round
#
# Profiling the screening stage (13 Nov 2026) showed that only ~14 % of the
# time went into the force-field pair kernel (kernels._pair_kernel); the
# rest was PyTorch bookkeeping on tiny tensors: autograd (~26 %), the
# rigid-body geometry (~30 %) and the L-BFGS loop (~30 %). This module does
# one whole L-BFGS round in compiled code:
#
#   parameters  (strain 6, translation 3/molecule, rotation 3/molecule),
#   squashed by tanh as in lattice._round
#     → cell, molecule centres, rotated atom offsets      (geometry)
#     → energy + dE/dpos + dE/dcell                        (kernels._pair_kernel,
#                                                           + Ewald reciprocal sum)
#     → dE/dparameters                                     (chain rule, by hand)
#
# The optimiser is a line-by-line port of torch.optim.LBFGS with the
# strong-Wolfe line search, so it takes the same steps as before (up to
# floating-point rounding). Set ALLOYFOLD_TORCH_MIN=1 to use the old path.

import math
import os
import numpy as np
from numba import njit

from kernels import _pair_kernel

ENABLED = os.environ.get('ALLOYFOLD_TORCH_MIN', '') != '1'
TWO_PI = 2.0 * math.pi


# ─────────────────────────────────────────────────────────────
# GEOMETRY + GRADIENT
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def _skew(v):
    K = np.zeros((3, 3))
    K[0, 1], K[0, 2] = -v[2], v[1]
    K[1, 0], K[1, 2] = v[2], -v[0]
    K[2, 0], K[2, 1] = -v[1], v[0]
    return K


@njit(cache=True)
def _rodrigues(w):
    """Rotation matrix from a rotation vector (same formula as lattice._rodrigues)."""
    th = math.sqrt(w[0] * w[0] + w[1] * w[1] + w[2] * w[2] + 1e-30)
    K = _skew(w / th)
    return np.eye(3) + math.sin(th) * K + (1.0 - math.cos(th)) * (K @ K)


@njit(cache=True)
def _rodrigues_grad(w, R):
    """dR/dw_i (3,3,3), Gallego & Yezzi (2015): (w_i [w]x + [w x (I-R) e_i]x) R / |w|^2."""
    out = np.zeros((3, 3, 3))
    th2 = w[0] * w[0] + w[1] * w[1] + w[2] * w[2]
    if th2 < 1e-24:
        for i in range(3):
            e = np.zeros(3)
            e[i] = 1.0
            out[i] = _skew(e)
        return out
    W = _skew(w)
    IR = np.eye(3) - R
    for i in range(3):
        c = np.cross(w, IR[:, i])
        out[i] = (w[i] * W + _skew(c)) @ R / th2
    return out


@njit(cache=True)
def _reciprocal(pos, cell, q, mvec, alpha, coulomb):
    """Ewald reciprocal-space energy (as forcefield.FFModel._reciprocal) with
    dE/dpos and dE/dcell."""
    n = pos.shape[0]
    gpos = np.zeros((n, 3))
    vol = abs(np.linalg.det(cell))
    Linv = np.linalg.inv(cell)
    B = TWO_PI * Linv.T
    pref = coulomb * TWO_PI / vol
    a4 = 1.0 / (4.0 * alpha * alpha)
    energy = 0.0
    M = np.zeros((3, 3))
    cs = np.empty(n)
    sn = np.empty(n)
    for kk in range(mvec.shape[0]):
        k = mvec[kk] @ B
        k2 = k[0] * k[0] + k[1] * k[1] + k[2] * k[2]
        A = math.exp(-k2 * a4) / k2
        Cr = 0.0
        Si = 0.0
        for a in range(n):
            ph = k[0] * pos[a, 0] + k[1] * pos[a, 1] + k[2] * pos[a, 2]
            cs[a] = math.cos(ph)
            sn[a] = math.sin(ph)
            Cr += q[a] * cs[a]
            Si += q[a] * sn[a]
        S2 = Cr * Cr + Si * Si
        energy += pref * A * S2
        dS2dk = np.zeros(3)
        for a in range(n):
            c = 2.0 * q[a] * (Si * cs[a] - Cr * sn[a])
            if c != 0.0:
                for l in range(3):
                    gpos[a, l] += pref * A * c * k[l]
                    dS2dk[l] += c * pos[a, l]
        dA = A * (-a4 - 1.0 / k2)           # dA/d(k²)
        for l in range(3):
            G = pref * (dA * 2.0 * k[l] * S2 + A * dS2dk[l])
            for m in range(3):
                M[l, m] += G * k[m]
    gcell = -energy * Linv.T - Linv.T @ M
    return energy, gpos, gcell


@njit(cache=True)
def _geometry(p, cell0, com0, off0, mol, s_max, budget, t_max):
    z = com0.shape[0]
    eps = np.empty(6)
    for a in range(6):
        eps[a] = s_max * math.tanh(p[a] / s_max)
    S = np.eye(3)
    S[0, 0] += eps[0]
    S[1, 1] += eps[1]
    S[2, 2] += eps[2]
    S[1, 2] += eps[3] / 2
    S[2, 1] += eps[3] / 2
    S[0, 2] += eps[4] / 2
    S[2, 0] += eps[4] / 2
    S[0, 1] += eps[5] / 2
    S[1, 0] += eps[5] / 2
    cell = cell0 @ S
    com = com0 @ S
    Rs = np.empty((z, 3, 3))
    for m in range(z):
        w = np.empty(3)
        for l in range(3):
            com[m, l] += budget * math.tanh(p[6 + 3 * m + l] / budget)
            w[l] = t_max * math.tanh(p[6 + 3 * z + 3 * m + l] / t_max)
        Rs[m] = _rodrigues(w)
    off = np.empty_like(off0)
    for a in range(off0.shape[0]):
        off[a] = Rs[mol[a]] @ off0[a]
    return cell, com, off, Rs


@njit(cache=True)
def _fg(p, cell0, com0, off0, mol, s_max, budget, t_max, pressure, e_const, ks, ewald, q, mvec, alpha_e):
    """Energy (+ pressure·volume) per cell and its gradient w.r.t. the raw parameters p."""
    z = com0.shape[0]
    cell, com, off, Rs = _geometry(p, cell0, com0, off0, mol, s_max, budget, t_max)
    pos = np.empty_like(off)
    for a in range(off.shape[0]):
        pos[a] = com[mol[a]] + off[a]
    e, gpos, gcell = _pair_kernel(pos, cell, ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], ks[6], ks[7],
                                  ks[8], ks[9], ks[10], ks[11], ks[12], ks[13], ks[14], ks[15], ks[16])
    if ewald:
        er, gpr, gcr = _reciprocal(pos, cell, q, mvec, alpha_e, ks[15])
        e += er
        gpos += gpr
        gcell += gcr
    e += e_const
    if pressure != 0.0:
        det = np.linalg.det(cell)
        e += pressure * abs(det)
        gcell += pressure * abs(det) * np.linalg.inv(cell).T
    g = np.zeros(p.shape[0])
    gcom = np.zeros((z, 3))
    GR = np.zeros((z, 3, 3))                 # dE/dR per molecule
    for a in range(pos.shape[0]):
        m = mol[a]
        for l in range(3):
            gcom[m, l] += gpos[a, l]
            for k in range(3):
                GR[m, l, k] += gpos[a, l] * off0[a, k]
    GS = cell0.T @ gcell + com0.T @ gcom
    ge = np.array([GS[0, 0], GS[1, 1], GS[2, 2],
                   (GS[1, 2] + GS[2, 1]) / 2, (GS[0, 2] + GS[2, 0]) / 2, (GS[0, 1] + GS[1, 0]) / 2])
    for a in range(6):
        th = math.tanh(p[a] / s_max)
        g[a] = ge[a] * (1.0 - th * th)
    for m in range(z):
        w = np.empty(3)
        for l in range(3):
            w[l] = t_max * math.tanh(p[6 + 3 * z + 3 * m + l] / t_max)
        dR = _rodrigues_grad(w, Rs[m])
        for l in range(3):
            th = math.tanh(p[6 + 3 * m + l] / budget)
            g[6 + 3 * m + l] = gcom[m, l] * (1.0 - th * th)
            th = math.tanh(p[6 + 3 * z + 3 * m + l] / t_max)
            gw = 0.0
            for u in range(3):
                for v in range(3):
                    gw += GR[m, u, v] * dR[l, u, v]
            g[6 + 3 * z + 3 * m + l] = gw * (1.0 - th * th)
    return e, g


# ─────────────────────────────────────────────────────────────
# L-BFGS  (port of torch.optim.LBFGS, line_search_fn='strong_wolfe', lr=1)
# ─────────────────────────────────────────────────────────────

@njit(cache=True)
def _cubic_interpolate(x1, f1, g1, x2, f2, g2, has_bounds, lo, hi):
    if has_bounds:
        xmin_bound, xmax_bound = lo, hi
    elif x1 <= x2:
        xmin_bound, xmax_bound = x1, x2
    else:
        xmin_bound, xmax_bound = x2, x1
    d1 = g1 + g2 - 3 * (f1 - f2) / (x1 - x2)
    d2_square = d1 * d1 - g1 * g2
    if d2_square >= 0:
        d2 = math.sqrt(d2_square)
        if x1 <= x2:
            min_pos = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2 * d2))
        else:
            min_pos = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2 * d2))
        return min(max(min_pos, xmin_bound), xmax_bound)
    return (xmin_bound + xmax_bound) / 2.0


@njit(cache=True)
def _round(p, max_iter, tol_grad, tol_change, history,
           cell0, com0, off0, mol, s_max, budget, t_max, pressure, e_const, ks, ewald, q, mvec, alpha_e):
    """One L-BFGS run from p (as lattice._round's opt.step). Returns the final p."""
    n = p.shape[0]
    max_eval = max_iter * 5 // 4
    c1, c2, ls_tol, max_ls = 1e-4, 0.9, 1e-9, 25

    f, g = _fg(p, cell0, com0, off0, mol, s_max, budget, t_max, pressure, e_const, ks, ewald, q, mvec, alpha_e)
    evals = 1
    if np.max(np.abs(g)) <= tol_grad:
        return p
    old_dirs = np.zeros((history, n))
    old_stps = np.zeros((history, n))
    ro = np.zeros(history)
    al = np.zeros(history)
    n_old = 0
    H_diag = 1.0
    d = np.zeros(n)
    t = 0.0
    prev_g = np.zeros(n)
    prev_f = 0.0
    it = 0
    while it < max_iter:
        it += 1
        if it == 1:
            d = -g
            n_old = 0
            H_diag = 1.0
        else:
            y = g - prev_g
            s = d * t
            ys = y @ s
            if ys > 1e-10:
                if n_old == history:
                    for h in range(history - 1):
                        old_dirs[h] = old_dirs[h + 1]
                        old_stps[h] = old_stps[h + 1]
                        ro[h] = ro[h + 1]
                    n_old -= 1
                old_dirs[n_old] = y
                old_stps[n_old] = s
                ro[n_old] = 1.0 / ys
                n_old += 1
                H_diag = ys / (y @ y)
            qv = -g
            for h in range(n_old - 1, -1, -1):
                al[h] = (old_stps[h] @ qv) * ro[h]
                qv = qv - al[h] * old_dirs[h]
            r = qv * H_diag
            for h in range(n_old):
                be = (old_dirs[h] @ r) * ro[h]
                r = r + (al[h] - be) * old_stps[h]
            d = r
        prev_g[:] = g
        prev_f = f
        if it == 1:
            t = min(1.0, 1.0 / np.sum(np.abs(g)))
        else:
            t = 1.0
        gtd = g @ d
        if gtd > -tol_change:
            break

        # ── strong Wolfe line search (torch.optim.lbfgs._strong_wolfe) ──
        d_norm = np.max(np.abs(d))
        f_new, g_new = _fg(p + t * d, cell0, com0, off0, mol, s_max, budget, t_max, pressure, e_const,
                           ks, ewald, q, mvec, alpha_e)
        ls_evals = 1
        gtd_new = g_new @ d
        t_prev, f_prev, g_prev, gtd_prev = 0.0, f, g.copy(), gtd
        done = False
        ls_iter = 0
        b_t = np.zeros(2)
        b_f = np.zeros(2)
        b_gtd = np.zeros(2)
        b_g = np.zeros((2, n))
        b_len = 0
        while ls_iter < max_ls:
            if f_new > (f + c1 * t * gtd) or (ls_iter > 1 and f_new >= f_prev):
                b_t[0], b_t[1] = t_prev, t
                b_f[0], b_f[1] = f_prev, f_new
                b_g[0], b_g[1] = g_prev, g_new
                b_gtd[0], b_gtd[1] = gtd_prev, gtd_new
                b_len = 2
                break
            if abs(gtd_new) <= -c2 * gtd:
                b_t[0], b_f[0], b_g[0] = t, f_new, g_new
                b_len = 1
                done = True
                break
            if gtd_new >= 0:
                b_t[0], b_t[1] = t_prev, t
                b_f[0], b_f[1] = f_prev, f_new
                b_g[0], b_g[1] = g_prev, g_new
                b_gtd[0], b_gtd[1] = gtd_prev, gtd_new
                b_len = 2
                break
            min_step = t + 0.01 * (t - t_prev)
            max_step = t * 10
            tmp = t
            t = _cubic_interpolate(t_prev, f_prev, gtd_prev, t, f_new, gtd_new, True, min_step, max_step)
            t_prev = tmp
            f_prev = f_new
            g_prev = g_new.copy()
            gtd_prev = gtd_new
            f_new, g_new = _fg(p + t * d, cell0, com0, off0, mol, s_max, budget, t_max, pressure, e_const,
                               ks, ewald, q, mvec, alpha_e)
            ls_evals += 1
            gtd_new = g_new @ d
            ls_iter += 1
        if ls_iter == max_ls:
            b_t[0], b_t[1] = 0.0, t
            b_f[0], b_f[1] = f, f_new
            b_g[0], b_g[1] = g, g_new
            b_len = 2
        if b_len == 1 or b_f[0] <= b_f[1]:
            low, high = 0, 1
        else:
            low, high = 1, 0
        insuf = False
        while not done and ls_iter < max_ls:
            if abs(b_t[1] - b_t[0]) * d_norm < ls_tol:
                break
            t = _cubic_interpolate(b_t[0], b_f[0], b_gtd[0], b_t[1], b_f[1], b_gtd[1], False, 0.0, 0.0)
            bmax = max(b_t[0], b_t[1])
            bmin = min(b_t[0], b_t[1])
            eps = 0.1 * (bmax - bmin)
            if min(bmax - t, t - bmin) < eps:
                if insuf or t >= bmax or t <= bmin:
                    if abs(t - bmax) < abs(t - bmin):
                        t = bmax - eps
                    else:
                        t = bmin + eps
                    insuf = False
                else:
                    insuf = True
            else:
                insuf = False
            f_new, g_new = _fg(p + t * d, cell0, com0, off0, mol, s_max, budget, t_max, pressure, e_const,
                               ks, ewald, q, mvec, alpha_e)
            ls_evals += 1
            gtd_new = g_new @ d
            ls_iter += 1
            if f_new > (f + c1 * t * gtd) or f_new >= b_f[low]:
                b_t[high], b_f[high], b_g[high], b_gtd[high] = t, f_new, g_new, gtd_new
                if b_f[0] <= b_f[1]:
                    low, high = 0, 1
                else:
                    low, high = 1, 0
            else:
                if abs(gtd_new) <= -c2 * gtd:
                    done = True
                elif gtd_new * (b_t[high] - b_t[low]) >= 0:
                    b_t[high], b_f[high], b_gtd[high] = b_t[low], b_f[low], b_gtd[low]
                    b_g[high] = b_g[low]
                b_t[low], b_f[low], b_g[low], b_gtd[low] = t, f_new, g_new, gtd_new
        t = b_t[low]
        f = b_f[low]
        g = b_g[low].copy()
        # ── end line search ──

        p = p + t * d
        opt_cond = np.max(np.abs(g)) <= tol_grad
        evals += ls_evals
        if it == max_iter:
            break
        if evals >= max_eval:
            break
        if opt_cond:
            break
        if np.max(np.abs(d * t)) <= tol_change:
            break
        if abs(f - prev_f) < tol_change:
            break
    return p


# ─────────────────────────────────────────────────────────────
# ENTRY POINT  (called by lattice._round)
# ─────────────────────────────────────────────────────────────

def applicable(model) -> bool:
    """True for the common case: MMFF/LJ force field through the compiled pair
    kernel, plain rigid geometry (no symmetric mode, no shape morphing)."""
    from lattice import _RigidModel
    return (ENABLED and getattr(model, 'use_kernel', False)
            and type(getattr(model, 'base', None)) is _RigidModel)


def run_round(x, model, pressure, max_iter, s_max, budget, t_max):
    """(cell, com, offset, energy per molecule at P = 0) after one L-BFGS round."""
    ks = model.kernel_spec
    ks = (ks[0], ks[1], ks[2], ks[3], ks[4], ks[5], bool(ks[6]), float(ks[7]), float(ks[8]), int(ks[9]),
          float(ks[10]), float(ks[11]), float(ks[12]), float(ks[13]), float(ks[14]), float(ks[15]),
          float(ks[16]))
    ewald = bool(model.charged and not model.dsf)
    if ewald:
        q = np.ascontiguousarray(model.q.numpy(), dtype=np.float64)
        mvec = np.ascontiguousarray(model.mvec.numpy(), dtype=np.float64)
        alpha_e = float(model.lv['alpha'])
    else:
        q, mvec, alpha_e = np.zeros(1), np.zeros((0, 3)), 1.0
    e_const = float(model.e_const) if model.charged else 0.0
    cell0 = np.ascontiguousarray(x.cell, dtype=np.float64)
    com0 = np.ascontiguousarray(x.com, dtype=np.float64)
    off0 = np.ascontiguousarray(x.offset, dtype=np.float64)
    mol = np.ascontiguousarray(x.mol, dtype=np.int64)
    s_max, budget, t_max = float(s_max), float(budget), float(t_max)
    p0 = np.zeros(6 + 6 * x.z)
    p = _round(p0, int(max_iter), 1e-6, 1e-10, 20, cell0, com0, off0, mol, s_max, budget, t_max,
               float(pressure), e_const, ks, ewald, q, mvec, alpha_e)
    e, _ = _fg(p, cell0, com0, off0, mol, s_max, budget, t_max, 0.0, e_const, ks, ewald, q, mvec, alpha_e)
    cell, com, off, _ = _geometry(p, cell0, com0, off0, mol, s_max, budget, t_max)
    return cell, com, off, e / x.z
