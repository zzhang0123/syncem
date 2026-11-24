"""Conditional cell integration and shifted correlations (SPEC Sec. 5.2, R3a).

LABEL ``eq: energy cell response``. Conditional densities use canonical
child units. The integrated child monomials are dimensionless. Taylor
coefficients have shape ``(n_shifts, n_child_moments)``. A sampled
truncation residual does not certify the error between samples.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..model._harmonic_cells import static_int
from ..values.errors import RoleError
from . import _jet_routine
from .provider_jet import _point, level_plan, taylor_coefficients

__all__ = [
    "ChildCellRule",
    "ConditionalContraction",
    "TruncationEstimate",
    "conditional_coefficients",
    "conditional_contraction",
]

_CHUNK_BYTES = 32 * 2**20
_OPERATOR_BYTES = 128 * 2**20


def _powers(rows, dimensions, name, *, distinct=True):
    result = tuple(tuple(row) for row in rows)
    if not result or any(
        len(row) != dimensions
        or any(
            not isinstance(p, (int, np.integer))
            or isinstance(p, (bool, np.bool_))
            or p < 0
            for p in row
        )
        for row in result
    ):
        raise ValueError(f"{name} needs nonempty exponent rows of length {dimensions}")
    if distinct and len(set(result)) != len(result):
        raise ValueError(f"{name} needs distinct exponent rows")
    return tuple(tuple(int(p) for p in row) for row in result)


def _chart(chart, reference, half, name):
    ref, hw = float(reference), float(half)
    if chart not in ("affine", "log") or not (
        np.isfinite(ref) and np.isfinite(hw) and hw > 0 and (chart != "log" or ref > 0)
    ):
        raise ValueError(
            f"{name} needs an affine or log chart with finite reference and positive scale"
        )
    return chart, ref, hw


def _check_density(density):
    if not callable(density):
        raise RoleError("a resolved child needs a conditional density or average=")


class ChildCellRule:
    """One fixed Gauss rule per axis of a bounded child cell.

    Axes give ``(chart, lo, hi, reference, half_width)``. Density support
    on the panel is fixed. Split cells at fixed jumps in the density.
    """

    def __init__(self, axes: Mapping, powers, *, nodes: int = 24):
        if not axes:
            raise ValueError("R3a needs at least one bounded child variable")
        node_count = static_int(nodes, "nodes", 1)
        if node_count**2 * 6 * 8 > _CHUNK_BYTES:
            raise MemoryError(
                "the requested one-axis Gauss rule exceeds the R3a work budget"
            )
        z, w = np.polynomial.legendre.leggauss(node_count)
        rules = []
        for name, (chart, lo, hi, ref, hw) in axes.items():
            chart, ref, hw = _chart(chart, ref, hw, name)
            lo, hi = float(lo), float(hi)
            if not (
                np.isfinite(lo)
                and np.isfinite(hi)
                and lo < hi
                and lo <= ref <= hi
                and (chart != "log" or lo > 0)
            ):
                raise ValueError(
                    f"child cell {name} needs two finite faces containing its reference"
                )
            low, high = (np.log(lo), np.log(hi)) if chart == "log" else (lo, hi)
            scale = 0.5 * high - 0.5 * low
            center = 0.5 * low + 0.5 * high
            u = center + scale * z
            if chart == "log":
                physical = np.exp(u)
                weight = scale * w * physical
                local = (u - np.log(ref)) / hw
            else:
                physical, weight, local = u, scale * w, (u - ref) / hw
            if scale <= 0 or any(
                np.any(~np.isfinite(a)) for a in (physical, weight, local)
            ):
                raise ValueError(f"child cell {name} has an unresolved chart interval")
            rules.append((physical, weight, local))
        self.names = tuple(axes)
        self.powers = _powers(powers, len(axes), "child powers")
        self.nodes, self._rules = node_count, tuple(rules)

    def _chunks(self, chunk_bytes, amplification=1):
        dims, moments = len(self.names), len(self.powers)
        per_node = 8 * (3 * dims + 2 * moments + amplification * (moments + 1))
        chunk = chunk_bytes // per_node
        if chunk < 1:
            raise MemoryError("R3a byte budget cannot hold one child node")
        count = self.nodes**dims
        if count > np.iinfo(np.intp).max:
            raise MemoryError("too many child quadrature nodes to index")
        shape = (self.nodes,) * dims
        exponents = np.asarray(self.powers)
        for start in range(0, count, chunk):
            ids = np.unravel_index(np.arange(start, min(start + chunk, count)), shape)
            points = {}
            weights = np.ones(ids[0].size)
            monomials = np.ones((moments, ids[0].size))
            for i, (name, (physical, weight, local)) in enumerate(
                zip(self.names, self._rules)
            ):
                selected = ids[i]
                points[name] = jnp.asarray(physical[selected])
                weights *= weight[selected]
                monomials *= local[selected] ** exponents[:, i, None]
            yield points, jnp.asarray(monomials * weights)

    def moments(self, density: Callable, parents: Mapping, *, chunk_bytes=_CHUNK_BYTES):
        """Integrate the child monomials at concrete or traced parent values."""
        _check_density(density)
        budget = static_int(chunk_bytes, "chunk_bytes", 1)
        return sum(
            _integrated(density, parents, points, weights)
            for points, weights in self._chunks(budget)
        )


def _integrated(density, parents, points, weighted_monomials):
    weight = jnp.asarray(density(parents, points))
    if jnp.iscomplexobj(weight):
        raise TypeError("conditional quadrature needs a real density")
    if weight.shape not in ((), (1,), (weighted_monomials.shape[1],)):
        raise ValueError("conditional density needs one value per child node")
    return jnp.sum(weighted_monomials * weight, axis=1)


def _derivatives(order):
    if order <= getattr(_jet_routine, "NESTED_MAX_ORDER", 5):
        return taylor_coefficients
    from ._taylor_jet import taylor_mode_coefficients

    return taylor_mode_coefficients


def conditional_coefficients(
    density: Callable,
    parent_axes: Mapping,
    child: ChildCellRule,
    shifts,
    *,
    fixed: Mapping | None = None,
    chunk_bytes: int = _CHUNK_BYTES,
):
    """Taylor coefficients of the integrated child monomials in parent chart coordinates.

    Parent axes give ``(chart, reference, half_width)``. The density takes
    the parent point and the fixed child node arrays. The caller validates
    support and parameters. The byte budget sizes the quadrature chunks.
    """
    _check_density(density)
    budget = static_int(chunk_bytes, "chunk_bytes", 1)
    names = tuple(parent_axes)
    indices = _powers(shifts, len(names), "Taylor shifts")
    fixed = {} if fixed is None else dict(fixed)
    clash = set(names).intersection(fixed)
    if clash or (set(names) | set(fixed)).intersection(child.names):
        raise ValueError(
            "the moving parents, fixed parents and child variables must be disjoint"
        )
    if 8 * len(indices) * len(child.powers) > budget:
        raise MemoryError("R3a coefficient output exceeds its byte budget")
    if not names:
        return child.moments(density, fixed, chunk_bytes=budget)[None, :]
    charts, reference, half = zip(
        *(_chart(*spec, name) for name, spec in parent_axes.items())
    )
    order = _jet_routine.check_order(names, indices)
    coefficient_rule = _derivatives(order)
    amplification = (
        math.prod(1 + len(level) for level in level_plan(indices))
        if coefficient_rule is taylor_coefficients
        else (order + 1) * len(indices)
    )

    @eqx.filter_jit
    def evaluate(points, weighted):
        def fn(z):
            parents = _point(names, charts, reference, half, z)
            return _integrated(density, {**fixed, **parents}, points, weighted)

        return coefficient_rule(fn, len(names), indices)

    return sum(
        evaluate(points, weighted)
        for points, weighted in child._chunks(budget, amplification)
    )


@dataclass(frozen=True)
class TruncationEstimate:
    """Largest sampled absolute residual per unit parent column."""

    value: np.ndarray
    samples: int
    kind: ClassVar[str] = "estimate"
    note: ClassVar[str] = "Sampled conditional residual per unit parent column."


@dataclass(frozen=True)
class ConditionalContraction:
    """Sparse linear node and Taylor shifts omitted by the parent lower set."""

    parent_powers: tuple[tuple[int, ...], ...]
    output_rows: tuple
    parent_weight: float
    joint_weight: float
    retained_rows: np.ndarray
    retained_parents: np.ndarray
    retained_coefficients: np.ndarray
    dropped_rows: np.ndarray
    dropped_powers: np.ndarray
    dropped_coefficients: np.ndarray

    def apply(self, moments: Any):
        """Map parent moments with arbitrary leading axes to scaled joint rows."""
        values = jnp.asarray(moments)
        if values.shape[-1:] != (len(self.parent_powers),):
            raise ValueError("parent moments must follow the declared exponent order")
        dtype = jnp.result_type(values.dtype, self.retained_coefficients.dtype)
        out = jnp.zeros(values.shape[:-1] + (len(self.output_rows),), dtype=dtype)
        return out.at[..., self.retained_rows].add(
            values[..., self.retained_parents] * self.retained_coefficients
        )

    def estimate(
        self, coordinates, child_moments, *, design=None, chunk_bytes=_CHUNK_BYTES
    ):
        """Compare the retained node to direct child moments on a parent grid.

        Coordinates use parent chart units. A design has scaled joint
        rows on its last axis. The maximum estimates the supremum.
        """
        budget = static_int(chunk_bytes, "chunk_bytes", 1)
        z, actual = np.asarray(coordinates, float), np.asarray(child_moments, float)
        dims = len(self.parent_powers[0])
        if z.ndim != 2 or not z.shape[0] or z.shape[1] != dims:
            raise ValueError("coordinates must have shape (n_samples, n_parents)")
        if actual.ndim != 2 or actual.shape[0] != z.shape[0]:
            raise ValueError(
                "child_moments must have shape (n_samples, n_child_moments)"
            )
        bases, children = zip(*self.output_rows)
        if actual.shape[1] <= max(children):
            raise ValueError("a joint row names an absent child moment")
        matrix = None if design is None else np.asarray(design)
        if matrix is not None and matrix.shape[-1:] != (len(bases),):
            raise ValueError("the last design axis must name the joint rows")
        result_shape = (len(bases),) if matrix is None else matrix.shape[:-1]
        per_point = 16 * (
            dims * (len(bases) + len(self.parent_powers))
            + 2 * len(bases)
            + len(self.retained_rows)
            + math.prod(result_shape)
        )
        chunk = budget // max(1, per_point)
        if chunk < 1:
            raise MemoryError("R3a byte budget cannot hold one diagnostic point")
        maximum = np.zeros(result_shape)
        p, r = np.asarray(self.parent_powers), np.asarray(bases)
        for start in range(0, len(z), chunk):
            point = z[start : start + chunk, None, :]
            parent = np.prod(point**p, axis=-1) / self.parent_weight
            reference = (
                np.prod(point**r, axis=-1)
                * actual[start : start + chunk, children]
                / self.joint_weight
            )
            residual = np.asarray(self.apply(parent)) - reference
            if matrix is not None:
                residual = np.tensordot(residual, matrix, axes=(-1, -1))
            if not np.all(np.isfinite(residual)):
                raise ValueError("sampled conditional residual must be finite")
            maximum = np.maximum(maximum, np.abs(residual).max(axis=0))
        return TruncationEstimate(maximum, len(z))


def conditional_contraction(
    coefficients,
    shifts,
    parent_powers,
    output_rows,
    *,
    parent_weight: float,
    joint_weight: float,
    max_bytes: int = _OPERATOR_BYTES,
) -> ConditionalContraction:
    """Pair each base parent exponent with a child row and record missing shifts."""

    budget = static_int(max_bytes, "max_bytes", 1)
    parent_weight, joint_weight = float(parent_weight), float(joint_weight)
    if not all(math.isfinite(w) and w > 0 for w in (parent_weight, joint_weight)):
        raise ValueError("parent and joint weights must be finite and positive")
    ratio = parent_weight / joint_weight
    if not math.isfinite(ratio) or ratio <= 0:
        raise ValueError("parent and joint weights must have a finite positive ratio")
    powers = tuple(tuple(p) for p in parent_powers)
    if not powers:
        raise ValueError("parent_powers needs at least the order-zero row")
    dims = len(powers[0])
    powers = _powers(powers, dims, "parent powers")
    if (0,) * dims not in powers:
        raise ValueError("parent_powers needs an order-zero row")
    shifts = _powers(shifts, dims, "Taylor shifts")
    rows = tuple(output_rows)
    bases = _powers((base for base, _ in rows), dims, "joint powers", distinct=False)
    h = np.asarray(coefficients)
    if h.ndim != 2 or h.shape[0] != len(shifts) or np.iscomplexobj(h):
        raise ValueError(
            "coefficients must have shape (n_shifts, n_child_moments) and be real"
        )
    children = tuple(c for _, c in rows)
    if any(
        not isinstance(c, (int, np.integer))
        or isinstance(c, (bool, np.bool_))
        or not 0 <= c < h.shape[1]
        for c in children
    ):
        raise ValueError("joint rows need valid child moment indices")
    if not np.all(np.isfinite(h)):
        raise ValueError("conditional coefficients must be finite")
    capacity = len(rows) * len(shifts)
    # Allow for both working arrays and compact copies of populated entries.
    if 16 * (5 + dims) * capacity > budget:
        raise MemoryError(
            "R3a contraction exceeds its byte budget. Split the requested joint rows."
        )
    retained = np.empty((capacity, 2), dtype=np.int64)
    dropped = np.empty((capacity, dims + 1), dtype=np.int64)
    kept_values, dropped_values = np.empty(capacity), np.empty(capacity)
    lookup = {power: k for k, power in enumerate(powers)}
    nk = nd = 0
    for row, (base, child) in enumerate(zip(bases, children)):
        for shift, value in zip(shifts, h[:, child]):
            if value == 0:
                continue
            shifted = tuple(r + k for r, k in zip(base, shift))
            factor = ratio * float(value)
            if not np.isfinite(factor):
                raise ValueError("scaled conditional coefficient overflowed")
            col = lookup.get(shifted)
            if col is None:
                dropped[nd] = (row, *shifted)
                dropped_values[nd] = factor
                nd += 1
            else:
                retained[nk] = (row, col)
                kept_values[nk] = factor
                nk += 1
    retained, dropped = retained[:nk].copy(), dropped[:nd].copy()
    return ConditionalContraction(
        powers,
        tuple(zip(bases, children)),
        parent_weight,
        joint_weight,
        retained[:, 0],
        retained[:, 1],
        kept_values[:nk].copy(),
        dropped[:, 0],
        dropped[:, 1:],
        dropped_values[:nd].copy(),
    )
