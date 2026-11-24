"""Cell tables for direct references and coefficient checks (SPEC Sec. 7.2 and 8.3).

The tensor grid has 17 Chebyshev-Lobatto nodes per variable, on the
physical cell faces in their respective charts. The interpolants are a
separate cached product. Their coefficient check measures agreement
with derivative columns before the table is used as a reference.

LABEL ``eq: channel basis expansion``. The coefficient arrays have shape
``(17, ...)`` or ``(17, 17, ...)`` followed by the kernel's mode axes, in its
channel spectral units. Interpolation error between nodes is not certified.
"""

from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev as cheb

from ..base._fingerprint import NotCacheable, digest
from ..base.kernel import Modes
from ..base.phase import phase_fingerprints
from ._keys import ENGINE_SOURCE, averaged_record, kernel_record
from ._rule_fingerprint import _context, _grid
from .average import AveragedKernel
from ._scaling_reference import ScalingReference, TableRefused, scaling_reference
from .cache import default_cache
from .direct import node_modes
from .provider_jet import jet_columns

__all__ = [
    "CellKernelTable",
    "ScalingReference",
    "TableAgreement",
    "TableRefused",
    "build_cell_table",
    "check_tables",
    "scaling_reference",
]

TABLE_NODES = 17
TABLE_TOL = 1e-10
_LOBATTO = np.cos(np.pi * np.arange(TABLE_NODES) / (TABLE_NODES - 1))
_INTERPOLATE = np.linalg.inv(cheb.chebvander(_LOBATTO, TABLE_NODES - 1))


@dataclass(frozen=True)
class _Axis:
    name: str
    chart: str
    lo: float
    hi: float
    reference: float
    half_width: float

    def charted(self, x):
        return np.log(x) if self.chart == "log" else x

    @property
    def centre_radius(self):
        lo, hi = self.charted(self.lo), self.charted(self.hi)
        return lo + 0.5 * (hi - lo), 0.5 * (hi - lo)

    def nodes(self):
        centre, radius = self.centre_radius
        charted = centre + radius * _LOBATTO
        nodes = np.exp(charted) if self.chart == "log" else charted
        nodes[0], nodes[-1] = self.hi, self.lo
        return nodes

    def coordinate(self, x):
        centre, radius = self.centre_radius
        return (self.charted(x) - centre) / radius

    def derivative_weights(self, order):
        _, radius = self.centre_radius
        ref = self.coordinate(self.reference)
        scale = self.half_width / radius
        polynomials = cheb.chebder(np.eye(TABLE_NODES), m=order, axis=0)
        return cheb.chebval(ref, polynomials) * (scale**order / math.factorial(order))


def _fit(values: np.ndarray, dimensions: int) -> np.ndarray:
    grid = values.reshape((TABLE_NODES,) * dimensions + values.shape[1:])
    for i in range(dimensions):
        fitted = np.tensordot(_INTERPOLATE, grid, axes=(1, i))
        grid = np.moveaxis(fitted, 0, i)
    return grid


def _contract(vectors, coefficients):
    ans = coefficients
    for vector in vectors:
        ans = np.tensordot(vector, ans, axes=1)
    return ans


@dataclass(frozen=True, eq=False)
class CellKernelTable:
    """One cell's interpolant of its natural-basis channel modes."""

    cell: int
    axes: tuple[_Axis, ...]
    coefficients: Modes
    averaged: AveragedKernel
    channel_keys: tuple[str, ...]

    def evaluate(self, points: Mapping[str, Any]) -> Modes:
        """Interpolate at matching one-dimensional coordinate arrays."""
        if set(points) != {a.name for a in self.axes}:
            raise ValueError("interpolation points must name exactly the table axes")
        vectors = []
        for axis in self.axes:
            x = np.asarray(points[axis.name], dtype=float)
            if (
                x.ndim != 1
                or np.any(~np.isfinite(x))
                or np.any((x < axis.lo) | (x > axis.hi))
            ):
                raise TableRefused(
                    f"interpolation needs nodes inside the {axis.name} cell"
                )
            vectors.append(cheb.chebvander(axis.coordinate(x), TABLE_NODES - 1))
        if len({v.shape[0] for v in vectors}) != 1:
            raise ValueError("interpolation points must have matching lengths")
        if len(vectors) == 1:
            return Modes(
                *(np.tensordot(vectors[0], a, axes=1) for a in self.coefficients)
            )
        return Modes(
            *(
                np.einsum("nr,ns,rs...->n...", *vectors, a, optimize=True)
                for a in self.coefficients
            )
        )

    def columns(self, exponents: Iterable[tuple[int, ...]]) -> Modes:
        """Taylor coefficients in the lattice coordinates, in requested order."""
        outputs = []
        for ex in exponents:
            if len(ex) != len(self.axes) or any(
                not isinstance(n, (int, np.integer))
                or isinstance(n, (bool, np.bool_))
                or n < 0
                or n >= TABLE_NODES
                for n in ex
            ):
                raise TableRefused(
                    "Taylor table requires an order from 0 to 16 per axis"
                )
            vectors = [
                a.derivative_weights(n) for a, n in zip(self.axes, ex, strict=True)
            ]
            outputs.append(Modes(*(_contract(vectors, a) for a in self.coefficients)))
        if not outputs:
            raise ValueError("give at least one Taylor exponent")
        return Modes(*(np.stack(a) for a in zip(*outputs, strict=True)))

    def sampled_derivative(self, order):
        """Sample one Taylor coefficient on the original single-axis grid."""
        if len(self.axes) != 1 or not 0 <= order < TABLE_NODES:
            raise TableRefused("sampled derivatives need one axis and order below 17")
        axis = self.axes[0]
        _, radius = axis.centre_radius
        factor = (axis.half_width / radius) ** order / math.factorial(order)
        return Modes(
            *(
                np.moveaxis(
                    cheb.chebval(_LOBATTO, cheb.chebder(values, m=order, axis=0)),
                    -1,
                    0,
                )
                * factor
                for values in self.coefficients
            )
        )


def _cell_axes(lattice, cell):
    cells = lattice.cells
    lo, hi, ref, hw = (
        getattr(cells, name)[cell] for name in ("lo", "hi", "reference", "half_width")
    )
    if len(cells.variables) not in (1, 2):
        raise TableRefused("a cell table takes one or two bounded variables")
    result = []
    for i, name in enumerate(cells.variables):
        chart = cells.chart[cell][i]
        if not (
            np.isfinite(lo[i])
            and np.isfinite(hi[i])
            and lo[i] < hi[i]
            and (chart != "log" or lo[i] > 0)
        ):
            raise TableRefused(f"the {name} cell needs two finite faces in its chart")
        result.append(
            _Axis(name, chart, float(lo[i]), float(hi[i]), float(ref[i]), float(hw[i]))
        )
    return tuple(result)


def _rules_at_nodes(averaged, points):
    names = tuple(points)
    fixed = dict(averaged.fixed)
    arguments = {n: jnp.asarray(x) for n, x in points.items()}
    tabulation = []
    for name, rule in averaged.averaged:
        context = _context(rule, name, averaged, names)

        def evaluate(resolved):
            nodes, weights, _ = _grid(
                context.levels, rule, {**fixed, **resolved}, context.hidden
            )
            return nodes, weights

        nodes, weights = jax.vmap(evaluate)(arguments)
        tabulation.append((name, np.asarray(nodes), np.asarray(weights)))
    return tuple(tabulation)


def _table_key(averaged, channels, axes, points):
    if averaged.leaf_factors:
        return None
    try:
        kernel = kernel_record(averaged.kernel)
        if kernel is None:
            return None
        guards = tuple(
            (name, value)
            for name, value in averaged.kernel.describe()
            if name in getattr(type(averaged.kernel), "guards", ())
        )
        return digest(
            (
                "Chebyshev cell table",
                ENGINE_SOURCE,
                kernel,
                guards,
                phase_fingerprints(averaged.phase, channels),
                channels.row_keys(),
                TABLE_NODES,
                axes,
                averaged_record(averaged),
                _rules_at_nodes(averaged, points),
            )
        )
    except (NotCacheable, KeyError, TypeError):
        return None


def build_cell_table(
    averaged: AveragedKernel,
    channels: Any,
    lattice: Any,
    cell: int,
    *,
    cache: bool = True,
    batch: int = 4,
) -> CellKernelTable:
    """Build or reuse the cell-table interpolant on the actual cell faces.

    The cache key tabulates moving averaged laws at the interpolation
    nodes. The tables count toward the existing process cache's memory
    bound and neither read nor write disk entries.
    """
    axes = _cell_axes(lattice, cell)
    grid = np.meshgrid(*(a.nodes() for a in axes), indexing="ij")
    points = {a.name: row.ravel() for a, row in zip(axes, grid, strict=True)}
    key = _table_key(averaged, channels, axes, points) if cache else None
    store = default_cache() if key is not None else None
    coefficients = None if store is None else store.get("memo", key)
    if coefficients is None:
        values = node_modes(
            averaged, channels, points, apply_azimuth=False, batch=batch
        )
        coefficients = Modes(*(_fit(a, len(axes)) for a in values))
        if store is not None:
            coefficients = store.put("memo", key, coefficients)
    return CellKernelTable(cell, axes, coefficients, averaged, channels.row_keys())


@dataclass(frozen=True)
class TableAgreement:
    """Largest measured difference relative to the reference channel maxima."""

    relative: float
    tolerance: float

    @property
    def passed(self) -> bool:
        return self.relative <= self.tolerance


def _channel_max(modes, channel_axis):
    return np.max(
        np.abs(modes), axis=tuple(i for i in range(modes.ndim) if i != channel_axis)
    )


def check_tables(
    averaged,
    channels,
    lattice,
    tables: Iterable[CellKernelTable],
    *,
    tolerance=TABLE_TOL,
) -> TableAgreement:
    """Check degree 0 to 2 on the built lower sets using jet columns.

    Each Stokes block uses its own channel maximum over the probe cells
    and tested degrees. Both sides use the given channel quadrature.
    """
    if not math.isfinite(tolerance) or tolerance < 0:
        raise ValueError("table tolerance must be finite and nonnegative")
    differences = []
    maxima = []
    channel_keys = channels.row_keys()
    for table in tables:
        if (
            table.averaged is not averaged
            or table.channel_keys != channel_keys
            or table.axes != _cell_axes(lattice, table.cell)
        ):
            raise TableRefused(
                "table and derivative checks need matching statistics, channels and cells"
            )
        pattern = lattice.index.patterns[int(lattice.index.cell_pattern[table.cell])]
        exponents = [ex for ex in pattern if sum(ex) <= 2]
        axes = {a.name: (a.chart, a.reference, a.half_width) for a in table.axes}
        i, p, v = jet_columns(averaged, channels, axes, exponents, with_v=True)
        native = table.columns(exponents)
        actual = Modes(native.I, native.V, native.P[:, 0])
        ref = Modes(i, v, p)
        differences.append(
            np.stack(
                [
                    _channel_max(a - b, dim)
                    for a, b, dim in zip(actual, ref, (1, 1, 1), strict=True)
                ]
            )
        )
        maxima.append(
            np.stack(
                [_channel_max(a, dim) for a, dim in zip(ref, (1, 1, 1), strict=True)]
            )
        )
    if not maxima:
        raise ValueError("table check requires at least one probe cell")
    difference, scale = np.max(differences, axis=0), np.max(maxima, axis=0)
    ratio = np.divide(difference, scale, out=np.zeros_like(difference), where=scale > 0)
    ratio[(scale == 0) & (difference != 0)] = np.inf
    return TableAgreement(float(ratio.max()), float(tolerance))
