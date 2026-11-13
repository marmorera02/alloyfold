# -*- coding: utf-8 -*-
# engine/kernels.py
#
# COMPILED PAIR KERNEL  (Numba)  for forcefield.FFModel
#
# For molecular crystals the tensors are small (10⁴-10⁵ pairs), and in
# PyTorch each of the ~300 elementwise operations of the energy and its
# gradient carries a fixed dispatch cost: an evaluation took ~10 ms almost
# independently of the number of pairs. This kernel computes the van der
# Waals (MMFF94 buffered 14-7 or LJ, switched) and real-space Coulomb
# (Ewald real part or damped shifted force) energy together with its
# analytic derivatives in ONE compiled loop over the pairs, and hands
# dE/d(positions) and dE/d(cell) back to PyTorch, which only does the cheap
# chain rule through the rigid-body geometry.

import math
import numpy as np
import torch
from numba import njit

SQRT_PI = math.sqrt(math.pi)


# Floating-point relaxations that cannot change results beyond rounding.
# 'nnan'/'ninf' are left out on purpose: a collapsed cell must still
# produce inf/NaN so that lattice.minimise() can reject it.
FASTMATH = {'contract', 'arcp', 'reassoc', 'nsz', 'afn'}


@njit(cache=True, fastmath=FASTMATH)
def _pair_kernel(pos, cell, I, J, N, R, EPS, QQ, form_lj, vdw_on, vdw_cut,
                 coul, alpha, width, rc, shift, force, coulomb, weight):
    """Energy, dE/dpos (n,3), dE/dcell (3,3). coul: 0 none, 1 Ewald real, 2 DSF.
    Each listed pair contributes 0.5 * weight; FFModel passes a half list
    (each unordered pair once) with the weight doubled."""
    n_at = pos.shape[0]
    gpos = np.zeros((n_at, 3))
    gcell = np.zeros((3, 3))
    energy = 0.0
    span = vdw_cut - vdw_on
    r_smear = 6.0 * width       # erf(r/width) == 1.0 exactly in float64 beyond this
    for p in range(I.shape[0]):
        i, j = I[p], J[p]
        n0, n1, n2 = N[p, 0], N[p, 1], N[p, 2]
        dx = pos[j, 0] - pos[i, 0] + n0 * cell[0, 0] + n1 * cell[1, 0] + n2 * cell[2, 0]
        dy = pos[j, 1] - pos[i, 1] + n0 * cell[0, 1] + n1 * cell[1, 1] + n2 * cell[2, 1]
        dz = pos[j, 2] - pos[i, 2] + n0 * cell[0, 2] + n1 * cell[1, 2] + n2 * cell[2, 2]
        r2 = dx * dx + dy * dy + dz * dz
        r = math.sqrt(r2)
        inv_r = 1.0 / r
        e = 0.0
        de = 0.0
        if r < vdw_cut and EPS[p] != 0.0:
            Rs = R[p]
            if form_lj:
                s2 = Rs * Rs / r2
                sr6 = s2 * s2 * s2
                ev = 4.0 * EPS[p] * (sr6 * sr6 - sr6)
                dev = 4.0 * EPS[p] * (-12.0 * sr6 * sr6 + 6.0 * sr6) * inv_r
            else:
                ib = 1.0 / (r + 0.07 * Rs)
                a = 1.07 * Rs * ib
                a2 = a * a
                A = a2 * a2 * a2 * a
                dA = -7.0 * A * ib
                R2 = Rs * Rs
                R7 = R2 * R2 * R2 * Rs
                r6 = r2 * r2 * r2
                iden = 1.0 / (r6 * r + 0.12 * R7)
                B = 1.12 * R7 * iden - 2.0
                dB = -1.12 * R7 * 7.0 * r6 * iden * iden
                ev = EPS[p] * A * B
                dev = EPS[p] * (dA * B + A * dB)
            t = (r - vdw_on) / span
            if t <= 0.0:
                e += ev
                de += dev
            elif t < 1.0:
                sw = 1.0 - t * t * (3.0 - 2.0 * t)
                dsw = -6.0 * t * (1.0 - t) / span
                e += ev * sw
                de += dev * sw + ev * dsw
        if coul > 0 and QQ[p] != 0.0 and (coul == 1 or r < rc):
            ar = alpha * r
            if r < r_smear:
                f = (math.erf(r / width) - math.erf(ar)) * inv_r
                df = (2.0 / SQRT_PI) * (math.exp(-(r / width) ** 2) / width
                                        - alpha * math.exp(-ar * ar)) * inv_r - f * inv_r
            else:               # smearing term is exactly 1 (value) and 0 (slope)
                f = math.erfc(ar) * inv_r
                df = -(2.0 / SQRT_PI) * alpha * math.exp(-ar * ar) * inv_r - f * inv_r
            if coul == 2:
                f += -shift + force * (r - rc)
                df += force
            e += coulomb * QQ[p] * f
            de += coulomb * QQ[p] * df
        energy += 0.5 * weight * e
        g = 0.5 * weight * de * inv_r
        gx, gy, gz = g * dx, g * dy, g * dz
        gpos[j, 0] += gx
        gpos[j, 1] += gy
        gpos[j, 2] += gz
        gpos[i, 0] -= gx
        gpos[i, 1] -= gy
        gpos[i, 2] -= gz
        gcell[0, 0] += n0 * gx
        gcell[0, 1] += n0 * gy
        gcell[0, 2] += n0 * gz
        gcell[1, 0] += n1 * gx
        gcell[1, 1] += n1 * gy
        gcell[1, 2] += n1 * gz
        gcell[2, 0] += n2 * gx
        gcell[2, 1] += n2 * gy
        gcell[2, 2] += n2 * gz
    return energy, gpos, gcell


def half_pairs(i, j, n):
    """Indices of a list that holds every pair in both orders, (i, j, n) and
    (j, i, -n), keeping each unordered pair once: i < j, or for an atom and
    its own image (i == j) the shift that is lexicographically positive."""
    pos_shift = (n[:, 0] > 0) | ((n[:, 0] == 0) & ((n[:, 1] > 0) | ((n[:, 1] == 0) & (n[:, 2] > 0))))
    return np.flatnonzero((i < j) | ((i == j) & pos_shift))


class PairEnergy(torch.autograd.Function):
    """E(pos, cell) from the compiled kernel; gradients by its analytic forces."""

    @staticmethod
    def forward(ctx, pos, cell, spec):
        e, gp, gc = _pair_kernel(pos.detach().numpy(), cell.detach().numpy(), *spec)
        ctx.save_for_backward(torch.as_tensor(gp), torch.as_tensor(gc))
        return torch.tensor(e, dtype=pos.dtype)

    @staticmethod
    def backward(ctx, grad_out):
        gp, gc = ctx.saved_tensors
        return grad_out * gp, grad_out * gc, None
