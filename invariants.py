# -*- coding: utf-8 -*-
# engine/invariants.py
#
# CRYSTAL FINGERPRINTS — Average Minimum Distance (AMD)
#
# AMD_k = mean over the atoms of one unit cell of the distance to their
# k-th nearest neighbour in the infinite crystal (Widdowson & Kurlin,
# 2022). It is an isometry invariant: independent of the unit cell
# chosen, the origin and the orientation, and it changes continuously
# when atoms move. That makes it a proper distance between crystals,
# replacing the threshold-sensitive RDF cosine similarity.
#
# amd_normalised divides by (V/N)^(1/3), so structures that differ only
# by a uniform scale (e.g. a force field 3% too dense) still match.

import math
import numpy as np

from lattice import image_range

K_DEFAULT = 100


def amd(cell: np.ndarray, positions: np.ndarray, k: int = K_DEFAULT) -> np.ndarray:
    """AMD vector (k,) in Å."""
    pos = np.asarray(positions, float)
    n = len(pos)
    vol = abs(np.linalg.det(cell))
    r = (3 * (k + 1) / (4 * math.pi * n / vol)) ** (1 / 3) * 1.5 + 1.0
    T = image_range(cell, r) @ cell                       # (M,3)
    images = (pos[None, :, :] + T[:, None, :]).reshape(-1, 3)
    out = np.zeros(k)
    for p in pos:
        d = np.linalg.norm(images - p, axis=1)
        d = d[d > 1e-8]                                   # drop the atom itself
        out += np.sort(np.partition(d, k - 1)[:k])       # k nearest, in order
    return out / n


def amd_normalised(cell, positions, k: int = K_DEFAULT) -> np.ndarray:
    n = len(positions)
    scale = (abs(np.linalg.det(cell)) / n) ** (1 / 3)
    return amd(cell, positions, k) / scale


def amd_distance(a: np.ndarray, b: np.ndarray) -> float:
    """L-infinity distance between two AMD vectors."""
    m = min(len(a), len(b))
    return float(np.max(np.abs(a[:m] - b[:m])))
