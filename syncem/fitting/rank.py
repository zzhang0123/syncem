"""Numerical rank and whitened SVD policies (SPEC Sec. 6.2 and decision 174).

LABEL ``sec: linear fit``. Singular values use the design in scaled lattice
moments, without column equilibration. The rank cutoff and pair test are
conventions for a finite-precision reduction.

Not certified. The reduction does not constrain discarded moment directions.

The singular directions refer to the scaled moment coordinates. Retained
values and errors have shape (n_mode,) on each sightline and use the moment
units.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral
from typing import NamedTuple
import warnings

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..values.errors import DataError

__all__ = [
    "FullRank",
    "PAIR_RTOL",
    "RANK_RTOL",
    "Rank",
    "RankCut",
    "SVDReduction",
    "SVDSolution",
    "UPRE",
    "rank_cut",
    "reduce_svd",
    "solve_svd",
    "warn_rank_margin",
]

RANK_RTOL = 1e-13
PAIR_RTOL = 1e-9
_NEAR_CUT_FACTOR = 1.1


def _rank_value(value, label):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 0
    ):
        raise ValueError(f"{label} must be a nonnegative integer.")
    return int(value)


@dataclass(frozen=True)
class UPRE:
    """Minimise the paper's residual and complexity score up to an optional cap."""

    cap: int | None = None

    def __post_init__(self):
        if self.cap is not None:
            object.__setattr__(self, "cap", _rank_value(self.cap, "UPRE cap"))


@dataclass(frozen=True)
class Rank:
    """Keep a fixed number of estimable directions."""

    R: int

    def __post_init__(self):
        object.__setattr__(self, "R", _rank_value(self.R, "rank"))


@dataclass(frozen=True)
class FullRank:
    """Keep every estimable direction."""


class RankCut(NamedTuple):
    """Cutoff and nearest singular value. Margin is signed relative to the cutoff."""

    numerical_rank: jax.Array
    cutoff: jax.Array
    index: jax.Array
    margin: jax.Array
    near: jax.Array


class SVDReduction(NamedTuple):
    """Thin SVD with left columns and right directions in rows."""

    left: jax.Array
    singular: jax.Array
    directions: jax.Array
    cut: RankCut
    paired: jax.Array
    pair_rounding_limited: jax.Array


class SVDSolution(NamedTuple):
    """Minimum-norm moments and fixed-width retained statistics."""

    moments: jax.Array
    beta: jax.Array
    sigma: jax.Array
    rank: jax.Array
    chi2: jax.Array


def rank_cut(singular, rtol: float = RANK_RTOL) -> RankCut:
    """Apply a strict relative cutoff to descending nonnegative singular values."""
    if not np.isfinite(rtol) or not 0 < rtol < 1:
        raise ValueError("rank tolerance must lie between zero and one.")
    singular = jnp.asarray(singular)
    if singular.ndim == 0 or singular.shape[-1] == 0:
        raise ValueError("provide at least one singular value.")
    scale = singular[..., 0]
    cutoff = rtol * scale
    safe = jnp.where(scale > 0, scale, 1.0)
    ratio = (singular / safe[..., None]) / rtol
    margin = ratio - 1.0
    index = jnp.argmin(jnp.abs(margin), axis=-1)
    nearest = jnp.take_along_axis(ratio, index[..., None], axis=-1)[..., 0]
    near = (nearest >= 1 / _NEAR_CUT_FACTOR) & (nearest <= _NEAR_CUT_FACTOR)
    return RankCut(
        jnp.sum(singular > cutoff[..., None], axis=-1),
        cutoff,
        jnp.where(scale > 0, index, -1),
        jnp.where(scale > 0, nearest - 1.0, jnp.inf),
        (scale > 0) & near,
    )


def warn_rank_margin(cut: RankCut) -> None:
    """Report the closest borderline case on the host after a reduction."""
    near, margins = np.asarray(cut.near), np.asarray(cut.margin)
    if np.any(near):
        idx = int(np.argmin(np.where(near, np.abs(margins), np.inf)))
        margin = float(margins.flat[idx])
        singular_index = int(np.asarray(cut.index).flat[idx])
        warnings.warn(
            f"{np.count_nonzero(near)} numerical rank cut(s) have a nearby "
            f"singular value. The closest is index {singular_index} at "
            f"{margin:+.3%} of the cut from the threshold.",
            RuntimeWarning,
            stacklevel=2,
        )


def _pairs(singular, cut: RankCut):
    scale = jnp.where(singular[0] > 0, singular[0], 1.0)
    left, right = singular[:-1] / scale, singular[1:] / scale
    representable = jnp.finfo(singular.dtype).eps <= PAIR_RTOL * right
    both_kept = singular[1:] > cut.cutoff
    close = left - right <= PAIR_RTOL * right
    return both_kept & representable & close, both_kept & ~representable


def _indices(columns, size):
    try:
        x, y = (np.asarray(c) for c in columns)
    except (TypeError, ValueError):
        raise ValueError(
            "complex columns must provide two equal index vectors."
        ) from None
    if (
        x.ndim != 1
        or x.size == 0
        or x.shape != y.shape
        or not np.issubdtype(x.dtype, np.integer)
        or not np.issubdtype(y.dtype, np.integer)
        or np.any(x < 0)
        or np.any(y < 0)
        or np.any(x >= size)
        or np.any(y >= size)
        or np.unique(np.concatenate([x, y])).size != 2 * x.size
    ):
        raise ValueError("complex column indices must be distinct and in bounds.")
    return jnp.asarray(x), jnp.asarray(y)


def _canonical_pairs(design, left, singular, directions, paired, columns):
    """Rotate isolated two-planes of the realified complex design."""
    x, y = _indices(columns, design.shape[1])
    if singular.size < 2:
        return left, directions
    isolated = paired & ~jnp.pad(paired, (1, 0))[:-1] & ~jnp.pad(paired, (0, 1))[1:]

    def rotate(i, vectors):
        u, vt = vectors
        v1, v2 = vt[i], vt[i + 1]
        z = v1[x] + 1j * v1[y]
        anchor = z[jnp.argmax(jnp.abs(z))]
        norm = jnp.linalg.norm(z)
        z = z * jnp.conj(anchor) / jnp.where(norm > 0, norm * jnp.abs(anchor), 1.0)
        a = jnp.zeros_like(v1).at[x].set(z.real).at[y].set(z.imag)
        b = jnp.zeros_like(v1).at[x].set(-z.imag).at[y].set(z.real)
        plane = jnp.stack([v1, v2])
        block = jnp.stack([a, b])
        residual = block - (block @ plane.T) @ plane
        tolerance = jnp.maximum(
            PAIR_RTOL, 16 * jnp.finfo(singular.dtype).eps * (singular[0] / singular[i])
        )
        valid = (norm > 0) & (jnp.max(jnp.linalg.norm(residual, axis=1)) <= tolerance)

        def accepted(_):
            u1, u2 = (design @ a) / singular[i], (design @ b) / singular[i + 1]
            return u.at[:, i].set(u1).at[:, i + 1].set(u2), vt.at[i].set(a).at[
                i + 1
            ].set(b)

        return jax.lax.cond(valid, accepted, lambda _: vectors, operand=None)

    def step(i, vectors):
        return jax.lax.cond(isolated[i], lambda v: rotate(i, v), lambda v: v, vectors)

    return jax.lax.fori_loop(0, singular.size - 1, step, (left, directions))


def reduce_svd(
    design, *, complex_columns=None, rtol: float = RANK_RTOL
) -> SVDReduction:
    """Reduce one whitened matrix. Use vmap for independent pixel matrices."""
    design = jnp.asarray(design)
    if design.ndim != 2 or min(design.shape) == 0 or jnp.iscomplexobj(design):
        raise DataError("the whitened design must be a real nonempty matrix.")
    design = design.astype(jnp.result_type(design, 1.0))
    design = eqx.error_if(
        design, jnp.any(~jnp.isfinite(design)), "the whitened design must be finite"
    )
    left, singular, directions = jnp.linalg.svd(design, full_matrices=False)
    singular = eqx.error_if(
        singular, jnp.any(~jnp.isfinite(singular)), "the whitened SVD did not converge"
    )
    pivots = jnp.argmax(jnp.abs(directions), axis=1)
    signs = jnp.where(directions[jnp.arange(singular.size), pivots] < 0, -1.0, 1.0)
    left, directions = left * signs, directions * signs[:, None]
    cut = rank_cut(singular, rtol)
    paired, limited = _pairs(singular, cut)
    if complex_columns is None:
        paired, limited = jnp.zeros_like(paired), jnp.zeros_like(limited)
    else:
        left, directions = _canonical_pairs(
            design, left, singular, directions, paired, complex_columns
        )
    return SVDReduction(left, singular, directions, cut, paired, limited)


def solve_svd(reduction: SVDReduction, data, rank=None) -> SVDSolution:
    """Apply one rank policy to one or more data vectors in the same whitened space."""
    if rank is None:
        rank = UPRE()
    if not isinstance(rank, (UPRE, Rank, FullRank)):
        raise TypeError("rank must be UPRE(), Rank(R) or FullRank().")
    data = jnp.asarray(data)
    left, s, vt, cut, paired, _ = reduction
    if data.ndim == 0 or data.shape[-1] != left.shape[0] or jnp.iscomplexobj(data):
        raise DataError("data must end in the same whitened rows as the design.")
    proj = data @ left
    ranks = jnp.arange(1, s.size + 1)
    allowed = (ranks <= cut.numerical_rank) & jnp.concatenate(
        [~paired, jnp.array([True])]
    )
    if isinstance(rank, FullRank):
        selected = cut.numerical_rank
    elif isinstance(rank, Rank):
        if rank.R > s.size:
            raise DataError("fixed rank exceeds the number of singular directions.")
        selected = jnp.asarray(rank.R)
        if rank.R:
            selected = eqx.error_if(
                selected,
                ~allowed[rank.R - 1],
                "fixed rank exceeds the numerical rank or splits a singular pair",
            )
    else:
        if rank.cap is not None:
            allowed &= ranks <= rank.cap
        # The squared norm of the data is a common additive constant.
        score = 2 * ranks - jnp.cumsum(proj**2, axis=-1)
        selected = jnp.where(
            jnp.any(allowed),
            jnp.argmin(jnp.where(allowed, score, jnp.inf), axis=-1) + 1,
            0,
        )

    selected = jnp.broadcast_to(selected, data.shape[:-1])
    active = ranks <= selected[..., None]
    inv = jnp.where(active, 1.0, 0.0) / jnp.where(active, s, 1.0)
    beta = proj * inv
    residual = data - jnp.where(active, proj, 0.0) @ left.T
    return SVDSolution(beta @ vt, beta, inv, selected, jnp.sum(residual**2, axis=-1))
