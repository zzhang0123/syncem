"""Default energy range of Spec Sec. 4.6 rule 1, measured on the averaged kernel.

The range [gamma_lo, gamma_hi] is the narrowest one for which gamma^-2.6 electrons below
gamma_lo, and separately above gamma_hi, contribute at most ``tol`` of the band-integrated
I_j in every channel j. It is measured with the same averaged kernel and bump channels as
the basis, so it is the exact window that the closed-form EnergyWindow bounds from outside.
The cells follow rule 2: equal in ln gamma, n = ceil(ln(hi/lo) / 0.484), order 2.
"""
from __future__ import annotations

import numpy as np

import core as C

from . import ledger as L


def shares(ch, pop=None):
    """Panel breaks g_k and the shares of I_j below and above each break, (K,), (K, J), (K, J)."""
    pop = L.power_law() if pop is None else pop
    g, w0, I, _ = L.full_quadrature(pop)
    w = L._weights(pop, g, w0)
    tI, _ = L.tail(pop)
    contrib = w[:, None] * I[:, ch]
    order = np.argsort(g)
    g, contrib = g[order], contrib[order]
    total = contrib.sum(axis=0) + tI[ch]
    below = np.cumsum(contrib, axis=0) / total
    above = 1.0 - below
    return g, below, above


def default_range(ch, tol=1e-3) -> tuple[float, float]:
    g, below, above = shares(ch)
    lo = []
    hi = []
    for j in range(below.shape[1]):
        k = np.searchsorted(below[:, j], tol)  # first node with share below >= tol
        lo.append(np.exp(np.interp(np.log(tol), np.log(np.maximum(below[k - 1:k + 1, j], 1e-300)),
                                   np.log(g[k - 1:k + 1]))))
        a = above[:, j]
        k = np.flatnonzero(a > tol)[-1]  # last node with share above > tol
        hi.append(np.exp(np.interp(-np.log(tol), -np.log(a[k:k + 2]), np.log(g[k:k + 2]))))
    return float(min(lo)), float(max(hi))


def default_cells(ch, tol=1e-3, order=2) -> C.Cells:
    lo, hi = default_range(ch, tol)
    n = int(np.ceil(np.log(hi / lo) / 0.484))
    return C.Cells.log(lo, hi, n, order=order)
