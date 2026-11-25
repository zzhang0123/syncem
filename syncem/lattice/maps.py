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

from ._cell_values import log_displacement
from .index import MomentIndex

__all__ = ["coarsen_values", "recentre_values"]


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
            ref_t, h_t = dst.table.reference[t], dst.table.half_width[t]
            acc = jnp.zeros(lead + (dst.n_pairs, len(pt)))
            for s, factor in members:
                ps = src.patterns[int(src.cell_pattern[s])]
                ref_s, h_s = src.table.reference[s], src.table.half_width[s]
                delta = np.array(
                    [
                        log_displacement(a, b) if chart == "log" else a - b
                        for a, b, chart in zip(ref_s, ref_t, dst.table.chart[t])
                    ]
                )
                transfer = _transfer(pt, ps, h_s / h_t, delta / h_t)
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
    same = np.array_equal(a.lo, b.lo) and np.array_equal(a.hi, b.hi)
    if not same or a.chart != b.chart:
        raise ValueError(
            "recentre needs the same cells in the same chart. Use coarsen for unions of cells"
        )
    for c in range(a.n_cells):
        _check_order(src, dst, c, c)
    groups = [[(c, float(w_src[c] / w_dst[c]))] for c in range(a.n_cells)]
    return _apply(values, src, dst, groups, masked)


def _fills_box(source, members, target, cell):
    """Count elementary boxes between distinct faces in each coordinate."""
    if not members:
        return False
    ids = [index for index, _ in members]
    lo, hi = source.lo[ids], source.hi[ids]
    # The sources already have disjoint interiors. Replacing each face by its
    # sorted rank preserves coverage, including gaps smaller than a volume ulp.
    counts = np.ones(len(ids), dtype=object)
    target_count = 1
    for axis in range(lo.shape[1]):
        lower, upper = lo[:, axis], hi[:, axis]
        faces = np.unique(np.r_[lower, upper])
        if faces[0] != target.lo[cell, axis] or faces[-1] != target.hi[cell, axis]:
            return False
        counts *= np.searchsorted(faces, upper) - np.searchsorted(faces, lower)
        target_count *= len(faces) - 1
    return sum(counts) == target_count


def _covered(lo, hi, box_lo, box_hi):
    """Test coverage of a closed face, including lower-dimensional intersections."""
    touched = np.all((box_hi >= lo) & (box_lo <= hi), axis=1)
    left, right = box_lo[touched], box_hi[touched]
    if not len(left):
        return False
    if np.any(np.all((left <= lo) & (right >= hi), axis=1)):
        return True
    # Split the face at the first nontrivial coordinate. The finite union
    # is closed, so coverage of every subinterval includes its endpoints.
    axis = np.flatnonzero(lo < hi)[0]
    cuts = np.unique(
        np.clip(
            np.r_[lo[axis], left[:, axis], right[:, axis], hi[axis]], lo[axis], hi[axis]
        )
    )
    keep = np.arange(len(lo)) != axis
    for a, b in zip(cuts[:-1], cuts[1:]):
        covers = (left[:, axis] <= a) & (right[:, axis] >= b)
        if not _covered(
            lo[keep], hi[keep], left[covers][:, keep], right[covers][:, keep]
        ):
            return False
    return True


def _check_face_ownership(source, target, owners):
    """A point belongs to the first closed cell containing it in each table."""
    if all(a <= b for a, b in zip(owners[:-1], owners[1:])):
        return
    for s, t in enumerate(owners):
        # Earlier target cells can overlap this member on faces and corners.
        lower = np.maximum(source.lo[s], target.lo[:t])
        upper = np.minimum(source.hi[s], target.hi[:t])
        for i in np.flatnonzero(np.all(lower <= upper, axis=1)):
            if not _covered(lower[i], upper[i], source.lo[:s], source.hi[:s]):
                raise ValueError(
                    "coarsen would change ownership of a shared cell boundary"
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
    groups: list[list[tuple[int, float]]] = [[] for _ in range(b.n_cells)]
    owners = []
    for s in range(a.n_cells):
        inside = np.all((a.lo[s] >= b.lo) & (a.hi[s] <= b.hi), axis=1)
        hits = np.nonzero(inside)[0]
        if hits.size != 1:
            raise ValueError(
                f"source cell {s} lies in {hits.size} target cells: the targets must be unions of cells"
            )
        t = int(hits[0])
        owners.append(t)
        if a.chart[s] != b.chart[t]:
            raise ValueError(
                f"source cell {s} and target cell {t} use different charts"
            )
        _check_order(src, dst, s, t)
        groups[t].append((s, float(w_src[s] / w_dst[t])))
    for t, members in enumerate(groups):
        if not _fills_box(a, members, b, t):
            raise ValueError(f"target cell {t} is not a union of source cells")
    _check_face_ownership(a, b, owners)
    return _apply(values, src, dst, groups, masked)
