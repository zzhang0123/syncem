"""Field-scaling relation columns for product cells over (gamma, B) (Spec Sec. 8.1, 8.3).

The continuum identity Kbar(gamma, L B) = L Kbar(L^(1/2) gamma, B) (PAPER eq: field scaling)
gives every Taylor coefficient in (u_gamma, u_B) of a product cell from the energy Taylor
coefficients of one fixed-field kernel at B_0, with the energy reference shifted by
(B_b/B_0)^(1/2) and the weight B_b/B_0 (Spec Sec. 8.3, decisions).

Write lam(v) = B(v)/B_b for the field chart variable v of the cell, and
sigma(v) = ln(lam(v)) / (2 h_gamma). With G_n = (1/n!) d^n/du^n Kbar(gamma_s e^{h u}, B_0) at
the shifted reference gamma_s = gamma_b (B_b/B_0)^(1/2),

    Kbar(gamma_b e^{h u}, B(v)) = L lam(v) sum_n G_n (u + sigma(v))^n,   L = B_b / B_0,

so the coefficient of u^r v^s is L sum_n G_n C(n, r) [v^s] lam(v) sigma(v)^(n - r). Only
n <= r + s contributes, because sigma(v) starts at v^1. Affine chart: lam = 1 + (h_B/B_b) v.
Log chart: lam = exp(h_B v).
"""
from __future__ import annotations

import math

import numpy as np

import core as C
from core.lattice import ProductTable


def _series_mul(a, b, n):
    return np.convolve(a, b)[: n + 1]


def chart_series(chart: str, h_B: float, B_b: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Power series in v to order n of lam(v) = B/B_b and of ln(lam(v)) (both length n+1)."""
    lam, log_lam = np.zeros(n + 1), np.zeros(n + 1)
    if chart == "affine":
        e = h_B / B_b
        lam[0] = 1.0
        if n >= 1:
            lam[1] = e
        for k in range(1, n + 1):
            log_lam[k] = (-1) ** (k + 1) * e**k / k
    elif chart == "log":
        for k in range(n + 1):
            lam[k] = h_B**k / math.factorial(k)
        if n >= 1:
            log_lam[1] = h_B
    else:
        raise ValueError(f"unknown chart {chart}")
    return lam, log_lam


def scaling_coefficients(exponents, chart_B: str, h_B: float, B_b: float, h_g: float) -> np.ndarray:
    """Matrix T (n_rows, n_max+1): column(r, s) = L * sum_n T[row, n] G_n.

    ``exponents`` are (r, s) pairs, r in gamma and s in B.
    """
    exps = [tuple(int(v) for v in e) for e in exponents]
    smax = max(s for _, s in exps)
    nmax = max(r + s for r, s in exps)
    lam, log_lam = chart_series(chart_B, h_B, B_b, smax)
    sigma = log_lam / (2.0 * h_g)
    powers = [np.eye(1, smax + 1, 0).ravel()]  # sigma^0
    for _ in range(nmax):
        powers.append(_series_mul(powers[-1], sigma, smax))
    T = np.zeros((len(exps), nmax + 1))
    for k, (r, s) in enumerate(exps):
        for n in range(r, r + s + 1):
            T[k, n] = math.comb(n, r) * _series_mul(lam, powers[n - r], smax)[s]
    return T


def shifted_energy_table(table: ProductTable, B0, nmax: int) -> ProductTable:
    """Table over (gamma, B) at (gamma_b (B_b/B_0)^(1/2), B_0), order nmax in gamma and 0 in B."""
    B0 = np.broadcast_to(np.asarray(B0, float), (table.n_cells,))
    ref = np.c_[table.reference[:, 0] * np.sqrt(table.reference[:, 1] / B0), B0]
    n = table.n_cells
    hw = np.c_[table.half_width[:, 0], np.ones(n)]
    return ProductTable(("gamma", "B"), np.zeros((n, 2), int), np.zeros((n, 2)), np.full((n, 2), np.inf),
                        ref, hw, tuple(("log", "log") for _ in range(n)), tuple((nmax, 0) for _ in range(n)), nmax)


def scaling_columns(avg, table: ProductTable, B0) -> dict:
    """Columns of a (gamma, B) table from energy derivatives of ``avg`` at the field B_0.

    ``avg`` takes a point with gamma and B. ``B0`` is a scalar or one value per product cell.
    Only energy derivatives (B exponent 0) are evaluated, which is the cost of the route.
    Returns {"I", "P", "energy"} with I and P in the lattice rows of ``table``.
    """
    if table.variables[:2] != ("gamma", "B") or len(table.variables) != 2:
        raise ValueError("table variables must be (gamma, B)")
    if any(ch[0] != "log" for ch in table.chart):
        raise ValueError("the scaling relation needs the log chart in gamma")
    B0 = np.broadcast_to(np.asarray(B0, float), (table.n_cells,))
    nmax = max(sum(e) for c in range(table.n_cells) for e in table.exponents(c))
    etab = shifted_energy_table(table, B0, nmax)
    E = C.build_columns(avg, etab)
    o, oe = table.offsets, etab.offsets
    out_I = np.zeros((table.n_rows,) + E["I"].shape[1:])
    out_P = np.zeros((table.n_rows,) + E["P"].shape[1:], complex)
    for c in range(table.n_cells):
        L = table.reference[c, 1] / B0[c]
        T = scaling_coefficients(table.exponents(c), table.chart[c][1], table.half_width[c, 1],
                                 table.reference[c, 1], table.half_width[c, 0])
        out_I[o[c]:o[c + 1]] = L * np.tensordot(T, E["I"][oe[c]:oe[c + 1]], axes=1)
        out_P[o[c]:o[c + 1]] = L * np.tensordot(T, E["P"][oe[c]:oe[c + 1]], axes=1)
    return {"I": out_I, "P": out_P, "energy": E}


def cauchy_factor(cols: dict, table: ProductTable, factor_coefs: np.ndarray) -> np.ndarray:
    """P columns times a resolved factor c(x): Cauchy product over the lower set of each cell.

    ``factor_coefs`` (n_rows,) complex holds the Taylor coefficients of c in the same rows.
    The product of two truncated series is exact on a lower set (a lower set is closed under
    the decomposition alpha = beta + gamma with beta, gamma <= alpha).
    """
    o = table.offsets
    out = np.zeros_like(cols["P"])
    for c in range(table.n_cells):
        exps = [tuple(e) for e in table.exponents(c)]
        pos = {e: k for k, e in enumerate(exps)}
        for k, a in enumerate(exps):
            acc = 0.0
            for b in exps:
                g = tuple(x - y for x, y in zip(a, b))
                if min(g) >= 0:
                    acc = acc + factor_coefs[o[c] + pos[b]] * cols["P"][o[c] + pos[g]]
            out[o[c] + k] = acc
    return out
