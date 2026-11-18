"""Exact linear maps between lattices: ``recentre`` and ``coarsen`` (SPEC Sec. 4.5).

LABEL ``eq: energy cell response``. In one chart the coordinate of a target cell is affine in the coordinate of a
source cell, ``u_t = alpha u_s + beta`` with ``alpha = h_s / h_t`` and ``beta = (c_s - c_t) / h_t`` (``c`` the
reference point in the chart). A target monomial is then a polynomial in the source monomials of lower or equal
exponents, ``prod_v (alpha_v u_v + beta_v)^r_v``, which the source lower set contains. ``recentre`` maps the
moments of the same cells to other reference points. ``coarsen`` maps the moments of cells to the moments of
their unions, ``a_t = (1 / w_t) sum_{s in t} w_s T_s a_s``. Both are exact up to rounding. Refinement from
moments alone is not defined.

Units: none of their own. Shapes: values of shape ``(..., n_source)`` map to ``(..., n_target)``. Not certified: rounding of the
binomial sums, validated to 1e-12 on densities with a cut-off inside a cell (tests/v080/test_v080_maps.py).
"""

from __future__ import annotations

from math import comb
from typing import Any

import jax.numpy as jnp
import numpy as np

from .index import MomentIndex

__all__ = ["coarsen_values", "recentre_values"]


def _chart_coords(index: MomentIndex, c: int) -> tuple[np.ndarray, np.ndarray]:
    t = index.table
    ref = np.array(
        [
            np.log(t.reference[c, i]) if t.chart[c][i] == "log" else t.reference[c, i]
            for i in range(len(t.variables))
        ]
    )
    return ref, t.half_width[c].astype(float)


def _transfer(
    target: tuple, source: tuple, alpha: np.ndarray, beta: np.ndarray
) -> np.ndarray:
    """``T[i, j]``: coefficient of source monomial ``j`` in target monomial ``i``."""
    where = {e: j for j, e in enumerate(source)}
    out = np.zeros((len(target), len(source)))
    for i, r in enumerate(target):
        for k in np.ndindex(*[x + 1 for x in r]):
            if k not in where:
                raise ValueError(
                    f"source cell lacks power {k} needed for target power {r}: lower the target order"
                )
            coef = 1.0
            for rv, kv, a, b in zip(r, k, alpha, beta):
                coef *= comb(rv, kv) * a**kv * b ** (rv - kv)
            out[i, where[k]] += coef
    return out


def _check_structure(src: MomentIndex, dst: MomentIndex) -> None:
    if (
        src.blocks != dst.blocks
        or src.legendre != dst.legendre
        or src.table.variables != dst.table.variables
    ):
        raise ValueError(
            "the lattices differ in blocks, Legendre axes or celled variables"
        )


def _check_order(src: MomentIndex, dst: MomentIndex, s: int, t: int) -> None:
    ps, pt = (
        src.patterns[int(src.cell_pattern[s])],
        dst.patterns[int(dst.cell_pattern[t])],
    )
    if not set(pt) <= set(ps):
        raise ValueError(
            f"target cell {t} has powers outside source cell {s}: its order is too high"
        )


def _apply(
    values: Any,
    src: MomentIndex,
    dst: MomentIndex,
    groups: list[list[tuple[int, float]]],
    masked: tuple[int, ...] = (),
) -> tuple[Any, tuple[int, ...]]:
    """Target rows from source rows, and the target rows that depend on a masked source row.

    ``groups[t]`` lists ``(source cell, weight factor)`` for target cell ``t``.
    """
    values = jnp.asarray(values)
    lead = values.shape[:-1]
    masked_in, masked_out = set(masked), set()
    pieces, start = [], 0
    for b in range(len(dst.blocks)):
        for t, members in enumerate(groups):
            pt = dst.patterns[int(dst.cell_pattern[t])]
            ref_t, h_t = _chart_coords(dst, t)
            acc = jnp.zeros(lead + (dst.n_pairs, len(pt)))
            for s, factor in members:
                ps = src.patterns[int(src.cell_pattern[s])]
                ref_s, h_s = _chart_coords(src, s)
                transfer = _transfer(pt, ps, h_s / h_t, (ref_s - ref_t) / h_t)
                rows = src.cell_rows(s, b)
                block = values[..., rows.start : rows.stop].reshape(
                    lead + (src.n_pairs, len(ps))
                )
                acc = acc + factor * jnp.einsum("...pj,ij->...pi", block, transfer)
                if masked_in:
                    for p in range(src.n_pairs):
                        for i, j in zip(*np.nonzero(transfer)):
                            if rows.start + p * len(ps) + j in masked_in:
                                masked_out.add(start + p * len(pt) + int(i))
            pieces.append(acc.reshape(lead + (dst.n_pairs * len(pt),)))
            start += dst.n_pairs * len(pt)
    return jnp.concatenate(pieces, axis=-1), tuple(sorted(masked_out))


def _same(a: Any, b: Any) -> bool:
    """Equal boxes up to 1e-12 of each box's extent in every variable (infinite ends must match)."""
    if a.lo.shape != b.lo.shape:
        return False
    finite = (
        np.isfinite(a.lo) & np.isfinite(a.hi) & np.isfinite(b.lo) & np.isfinite(b.hi)
    )
    scale = np.where(finite, np.maximum(np.abs(a.lo), np.abs(a.hi)), 0.0)
    for x, y in ((a.lo, b.lo), (a.hi, b.hi)):
        both = np.isfinite(x) & np.isfinite(y)
        if np.any(~both & (x != y)):
            return False
        if np.any(
            np.abs(np.subtract(x, y, out=np.zeros_like(x, dtype=float), where=both))
            > 1e-12 * scale
        ):
            return False
    return True


def recentre_values(
    values: Any,
    src: MomentIndex,
    dst: MomentIndex,
    w_src: np.ndarray,
    w_dst: np.ndarray,
    masked: tuple = (),
) -> tuple[Any, tuple[int, ...]]:
    """Moments of the same cells at the reference points (and half-widths) of ``dst``."""
    _check_structure(src, dst)
    a, b = src.table, dst.table
    same = a.n_cells == b.n_cells and _same(a, b)
    if not same or a.chart != b.chart:
        raise ValueError(
            "recentre needs the same cells in the same chart. Use coarsen for unions of cells"
        )
    for c in range(a.n_cells):
        _check_order(src, dst, c, c)
    groups = [[(c, float(w_src[c] / w_dst[c]))] for c in range(a.n_cells)]
    return _apply(values, src, dst, groups, masked)


def _volume(table: Any, c: int) -> float:
    lo, hi = table.lo[c], table.hi[c]
    return float(
        np.prod(
            [
                np.log(h / l) if ch == "log" else h - l
                for l, h, ch in zip(lo, hi, table.chart[c])
            ]
        )
    )


def coarsen_values(
    values: Any,
    src: MomentIndex,
    dst: MomentIndex,
    w_src: np.ndarray,
    w_dst: np.ndarray,
    masked: tuple = (),
) -> tuple[Any, tuple[int, ...]]:
    """Moments of unions of cells: every source cell lies in one target cell, which the sources fill."""
    _check_structure(src, dst)
    a, b = src.table, dst.table
    if not (
        np.all(np.isfinite(a.lo))
        and np.all(np.isfinite(a.hi))
        and np.all(np.isfinite(b.lo))
        and np.all(np.isfinite(b.hi))
    ):
        raise ValueError(
            "coarsen needs cells with edges. Cells without edges can only be recentred"
        )
    slack = 1e-12 * np.maximum(np.abs(b.lo), np.abs(b.hi))
    groups: list[list[tuple[int, float]]] = [[] for _ in range(b.n_cells)]
    for s in range(a.n_cells):
        inside = np.all((a.lo[s] >= b.lo - slack) & (a.hi[s] <= b.hi + slack), axis=1)
        hits = np.nonzero(inside)[0]
        if hits.size != 1:
            raise ValueError(
                f"source cell {s} lies in {hits.size} target cells: the targets must be unions of cells"
            )
        t = int(hits[0])
        if a.chart[s] != b.chart[t]:
            raise ValueError(
                f"source cell {s} and target cell {t} use different charts"
            )
        _check_order(src, dst, s, t)
        groups[t].append((s, float(w_src[s] / w_dst[t])))
    for t, members in enumerate(groups):
        filled = sum(_volume(a, s) for s, _ in members)
        if not np.isclose(filled, _volume(b, t), rtol=1e-10, atol=0.0):
            raise ValueError(f"target cell {t} is not a union of source cells")
    return _apply(values, src, dst, groups, masked)
