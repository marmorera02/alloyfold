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


@njit(cache=True, fastmath=False)
def _pair_kernel(pos, cell, I, J, N, R, EPS, QQ, form_lj, vdw_on, vdw_cut,
                 coul, alpha, width, rc, shift, force, coulomb, weight):
    """Energy, dE/dpos (n,3), dE/dcell (3,3). coul: 0 none, 1 Ewald real, 2 DSF."""
    n_at = pos.shape[0]
    gpos = np.zeros((n_at, 3))
    gcell = np.zeros((3, 3))
    energy = 0.0
    span = vdw_cut - vdw_on
    for p in range(I.shape[0]):
        i, j = I[p], J[p]
        dx = pos[j, 0] - pos[i, 0] + N[p, 0] * cell[0, 0] + N[p, 1] * cell[1, 0] + N[p, 2] * cell[2, 0]
        dy = pos[j, 1] - pos[i, 1] + N[p, 0] * cell[0, 1] + N[p, 1] * cell[1, 1] + N[p, 2] * cell[2, 1]
        dz = pos[j, 2] - pos[i, 2] + N[p, 0] * cell[0, 2] + N[p, 1] * cell[1, 2] + N[p, 2] * cell[2, 2]
        r = math.sqrt(dx * dx + dy * dy + dz * dz)
        e = 0.0
        de = 0.0
        if r < vdw_cut and EPS[p] != 0.0:
            Rs = R[p]
            if form_lj:
                sr6 = (Rs / r) ** 6
                ev = 4.0 * EPS[p] * (sr6 * sr6 - sr6)
                dev = 4.0 * EPS[p] * (-12.0 * sr6 * sr6 + 6.0 * sr6) / r
            else:
                a = 1.07 * Rs / (r + 0.07 * Rs)
                A = a ** 7
                dA = -7.0 * A / (r + 0.07 * Rs)
                R7 = Rs ** 7
                r7 = r ** 7
                den = r7 + 0.12 * R7
                B = 1.12 * R7 / den - 2.0
                dB = -1.12 * R7 * 7.0 * r ** 6 / (den * den)
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
            f = (math.erf(r / width) - math.erf(alpha * r)) / r
            df = (2.0 / SQRT_PI) * (math.exp(-(r / width) ** 2) / width
                                    - alpha * math.exp(-(alpha * r) ** 2)) / r - f / r
            if coul == 2:
                f += -shift + force * (r - rc)
                df += force
            e += coulomb * QQ[p] * f
            de += coulomb * QQ[p] * df
        energy += 0.5 * weight * e
        g = 0.5 * weight * de / r
        gx, gy, gz = g * dx, g * dy, g * dz
        gpos[j, 0] += gx
        gpos[j, 1] += gy
        gpos[j, 2] += gz
        gpos[i, 0] -= gx
        gpos[i, 1] -= gy
        gpos[i, 2] -= gz
        for a_ in range(3):
            gcell[a_, 0] += N[p, a_] * gx
            gcell[a_, 1] += N[p, a_] * gy
            gcell[a_, 2] += N[p, a_] * gz
    return energy, gpos, gcell


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
