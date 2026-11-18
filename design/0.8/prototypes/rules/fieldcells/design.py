"""Design matrices of (gamma, B) lattices and the structural tie of the field-scaling relation.

The design maps lattice rows to the real data vector [I_j, Q_j, U_j]. With a constant azimuth
factor c, Q + iU = c P. The field-scaling relation (PAPER eq: field scaling) gives, in each
product cell, column(r, s) as L sum_n T[(r, s), n] G_n over the energy Taylor coefficients
G_0..G_{r+s} (rules/fieldcells/scaling.py). The columns of a product cell therefore lie in
the span of n_max + 1 vectors, and every vector in the null space of T^T is a combination of
moments that no data constrain. ``predicted_null`` returns these vectors per product cell.
"""
from __future__ import annotations

import numpy as np
import scipy.linalg as sla

from core.lattice import ProductTable

from .scaling import scaling_coefficients


def design(cols: dict, azimuth=1.0, w_rows=None) -> np.ndarray:
    """(3 n_ch, n_rows) real design. ``w_rows`` scales the columns (reference weights)."""
    I, P = np.asarray(cols["I"]), azimuth * np.asarray(cols["P"])
    D = np.concatenate([I.T, P.real.T, P.imag.T], axis=0)
    return D if w_rows is None else D * np.asarray(w_rows)[None, :]


def predicted_null(table: ProductTable) -> np.ndarray:
    """(n_rows, k) null vectors of the design implied by the scaling relation, cell by cell."""
    o, vecs = table.offsets, []
    for c in range(table.n_cells):
        T = scaling_coefficients(table.exponents(c), table.chart[c][1], table.half_width[c, 1],
                                 table.reference[c, 1], table.half_width[c, 0])
        Z = sla.null_space(T.T)  # moment combinations a with T^T a = 0
        for z in Z.T:
            v = np.zeros(table.n_rows)
            v[o[c]:o[c + 1]] = z
            vecs.append(v)
    return np.array(vecs).T if vecs else np.zeros((table.n_rows, 0))


def structural_rank(table: ProductTable) -> int:
    """Rank of the column relations alone: sum over cells of rank(T)."""
    return int(sum(np.linalg.matrix_rank(scaling_coefficients(
        table.exponents(c), table.chart[c][1], table.half_width[c, 1], table.reference[c, 1],
        table.half_width[c, 0])) for c in range(table.n_cells)))


def singular_values(D: np.ndarray) -> np.ndarray:
    """Singular values of D with unit-norm columns, divided by the largest."""
    n = np.linalg.norm(D, axis=0)
    s = np.linalg.svd(D / np.where(n > 0, n, 1.0), compute_uv=False)
    return s / s[0]


def numerical_rank(D: np.ndarray, rtol=1e-10) -> int:
    return int(np.sum(singular_values(D) > rtol))


def field_residual(D: np.ndarray, table: ProductTable) -> float:
    """Largest relative residual of a field column (s > 0) after projection onto the energy
    columns (s = 0) of the same product cell. Zero when the scaling relation holds."""
    o, worst = table.offsets, 0.0
    for c in range(table.n_cells):
        exps = table.exponents(c)
        e = [o[c] + k for k, r in enumerate(exps) if r[1] == 0]
        f = [o[c] + k for k, r in enumerate(exps) if r[1] > 0]
        if not f:
            continue
        Q, _ = np.linalg.qr(D[:, e])
        for j in f:
            d = D[:, j]
            worst = max(worst, float(np.linalg.norm(d - Q @ (Q.T @ d)) / np.linalg.norm(d)))
    return worst


def null_residual(D: np.ndarray, V: np.ndarray) -> float:
    """max over predicted null vectors of |D v| / (|D| |v|), columns of D normalised as in sigma."""
    n = np.linalg.norm(D, axis=0)
    Dn = D / np.where(n > 0, n, 1.0)
    Vn = V * n[:, None]
    Vn = Vn / np.linalg.norm(Vn, axis=0)
    return float(np.max(np.linalg.norm(Dn @ Vn, axis=0)) / np.linalg.norm(Dn, 2))
