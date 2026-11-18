"""Measures as nodes and weights, reference weights, and moments on the lattice rows.

a[row] = (1/w_c) sum over nodes i in product cell c of weight_i * prod_v u_v^r_v
         * prod_l P_l(x_l) (Legendre axes) [* exp(2i phi) for the spin-2 block].
w_c is the product over axes of the cell mass of the reference density: gamma^-slope
normalised on the table's gamma range for the variable ``gamma``, 1 otherwise (Spec Sec. 4.5).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

from .lattice import ProductTable, to_u


@dataclass(frozen=True, eq=False)
class Measure:
    """int f dN ~ sum_i weights_i f(nodes_i). ``nodes[v]`` has shape (M,)."""

    nodes: dict
    weights: np.ndarray

    def subset(self, mask) -> "Measure":
        m = np.asarray(mask)
        return Measure({v: np.asarray(x)[m] for v, x in self.nodes.items()}, self.weights[m])


def panel_measure(variable, density, breaks, n=24, chart="log") -> Measure:
    """Gauss-Legendre panels between consecutive breaks, in ln x (log) or x (affine)."""
    t, w = leggauss(n)
    b = np.unique(np.asarray(breaks, float))
    s = np.log(b) if chart == "log" else b
    xs, ws = [], []
    for a, c in zip(s[:-1], s[1:]):
        u = 0.5 * (c - a) * t + 0.5 * (c + a)
        x = np.exp(u) if chart == "log" else u
        jac = x if chart == "log" else 1.0
        xs.append(x)
        ws.append(0.5 * (c - a) * w * jac * density(x))
    return Measure({variable: np.concatenate(xs)}, np.concatenate(ws))


def product(*measures: Measure) -> Measure:
    """Tensor product of independent measures on disjoint variables."""
    nodes, weights = {}, np.ones(1)
    for m in measures:
        k = m.weights.size
        nodes = {v: np.repeat(x, k) for v, x in nodes.items()}
        nodes.update({v: np.tile(x, weights.size) for v, x in m.nodes.items()})
        weights = np.outer(weights, m.weights).ravel()
    return Measure(nodes, weights)


def reference_weights(table: ProductTable, slope=2.6, variable="gamma") -> np.ndarray:
    """(n_cells,) cell masses of gamma^-slope on the table's gamma range, 1 for other axes."""
    if variable not in table.variables:
        return np.ones(table.n_cells)
    v = table.variables.index(variable)
    lo, hi = table.lo[:, v], table.hi[:, v]
    if not np.all(np.isfinite(hi)) or np.any(lo <= 0):
        return np.ones(table.n_cells)
    k = 1.0 - slope
    mass = (hi**k - lo**k) / k
    return mass / ((hi.max() ** k - lo.min() ** k) / k)


def pair(table: ProductTable, measure: Measure, w_cells, *, legendre=None, harmonic=False) -> dict:
    """{"M0": (n_rows, L...) real, "M2": complex or None}. Nodes outside every cell are dropped.

    ``legendre``: {variable: L}. ``harmonic``: also return M2 = <exp(2i phi) ...> from
    ``measure.nodes["phi"]``.
    """
    legendre = dict(legendre or {})
    cell = table.locate(measure.nodes)
    shape = tuple(L + 1 for L in legendre.values())
    M0 = np.zeros((table.n_rows,) + shape)
    M2 = np.zeros((table.n_rows,) + shape, complex) if harmonic else None
    offsets = table.offsets
    for c in np.unique(cell[cell >= 0]):
        sel = cell == c
        w = measure.weights[sel]
        U = [np.asarray(to_u(table.chart[c][i], measure.nodes[v][sel], table.reference[c, i],
                             table.half_width[c, i])) for i, v in enumerate(table.variables)]
        fac = np.ones((w.size,) + shape)
        for k, (v, L) in enumerate(legendre.items()):
            Pl = np.stack([eval_legendre(l, measure.nodes[v][sel]) for l in range(L + 1)], axis=1)
            fac = fac * Pl.reshape((w.size,) + (1,) * k + (L + 1,) + (1,) * (len(shape) - k - 1))
        for m, r in enumerate(table.exponents(c)):
            mono = w * np.prod([U[i] ** r[i] for i in range(len(U))], axis=0)
            M0[offsets[c] + m] = np.tensordot(mono, fac, axes=1) / w_cells[c]
            if harmonic:
                M2[offsets[c] + m] = np.tensordot(mono * np.exp(2j * measure.nodes["phi"][sel]), fac,
                                                  axes=1) / w_cells[c]
    return {"M0": M0, "M2": M2}
