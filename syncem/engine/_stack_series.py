"""Stack columns of a product cell: cached-S1 mode, per-node mode and field cells (design/0.8/SPEC.md Sec. 8.3).

LABEL ``eq: illustrated derivative spectra``. The averaged kernel's levels are walked with Taylor series in the
energy chart ``u``. At the leaf the stack gives the series of the kernel at fixed ``B`` and ``eta``
(``provider_stack.point_series``) from a one-node S1 table. An averaged level sums its children with the
weights of its rule. A rule whose parents include the celled ``gamma`` keeps its nodes at the cell's reference
point and reweights them, ``w_q(u) = w_q(ref) rho(x_q | gamma(u)) / rho(x_q | ref)``, whose Taylor coefficients
enter by a Cauchy product (frozen-node reweighting). A field law with no parents that no other law depends on is
averaged once by its S1 table (cached-S1 mode). A ``(gamma, B)`` cell takes the field-scaling relation from the
energy series at ``B_b`` (``expand.field_scaling_columns``) when no law depends on ``gamma`` or ``B``.

Cases the stack cannot take raise ``StackRefused`` with the reason, and the caller takes the jet: a kernel other than
``ContinuumKernel`` itself (S0 holds its ``F`` and ``G``), unknown parents, a law that reads a celled variable it does
not declare (an order-1 probe per rule and cell), a celled parent without a density or with one that JAX cannot trace,
reweighted weights that do not reproduce the jet of the rule itself (decision 139: node by node for fixed nodes, on
``t^m``, ``m = 0..3``, for moving ones), weights amplified beyond ``_MAX_AMPLIFICATION`` (a law that moves by many
widths across the cell), orders above the S1 table order, energies above the S1 grid for the lowest channel, and celled
variables other than ``gamma`` and ``B``. Nodes at ``|eta| = 1`` or ``B = 0`` add zero. Each S1 table reads the S0
table of its quantised ``ln x`` span, held in the memo once per span (decision 158). The memo is a dict, or a
``StageMemo`` that reads and stores the tables through the process cache (decisions 159 to 162). The refusal
checks and the frozen-node weights are in ``_stack_checks``, and the field cells and the row layout in
``_stack_fields`` (decision 145). A walk computes the channel nodes and weights once per depth value, and
``stack_cell`` reports the mode it took. ``stack_stage`` walks dry: it reads the same S1 tables and runs the
refusal checks that do not read the channels, and gives the mode without a channel series (decision 159). Units: those of the kernel. Shapes: series ``(order + 1, J, *L)``, cell rows
``(n_rows, J)`` in the lattice order. Not certified: the reweighted quadrature, validated against the jet. The check
bounds moments, so a law whose nodes do not resolve the kernel gives a quadrature error that the jet shares.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from ..base.capabilities import Similarity
from ..base.continuum import ContinuumKernel
from ._stack_checks import (
    StackRefused,
    _call,
    _check_lattice,
    _check_undeclared,
    _weights,
)
from ._stack_fields import _field_cell, _to_rows
from ._stage_memo import StageMemo
from ._stage_memo import s1_table as _s1_table
from .average import AveragedKernel, NodeRule
from .field_average import TABLE_ORDER, FieldAverage, OutsideTable
from .provider_stack import _channel_arrays, point_series

__all__ = ["StackRefused", "stack_cell", "stack_lattice", "stack_stage"]

# the continuum kernel is the isotropic pitch average and carries no azimuth factor
_KERNEL_FREE = {"mu", "phi"}


def _mul(a: np.ndarray, b: np.ndarray, order: int) -> np.ndarray:
    """Cauchy product along axis 0, truncated at ``order``. ``a`` is ``(order + 1,)`` or broadcasts."""
    out = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.result_type(a, b))
    for n in range(order + 1):
        for k in range(n + 1):
            out[n] = out[n] + a[k] * b[n - k]
    return out


class _Context:
    def __init__(
        self,
        averaged: AveragedKernel,
        channels: Any,
        chart: str,
        ref: float,
        h: float,
        order: int,
        s1: str,
        memo: dict,
        reweight: bool,
        probe: tuple[str, ...] = ("gamma",),
        dry: bool = False,
    ):
        self.dry = dry  # read the tables and evaluate the laws, no channel series
        self.averaged, self.channels = averaged, channels
        self.chart, self.ref, self.h = chart, ref, h
        self.order, self.s1, self.memo, self.reweight = order, s1, memo, reweight
        self.probe = probe  # celled variables a law must declare when it reads them
        self.probed: set[int] = set()
        self.cached: FieldAverage | None = None
        self._arrays: dict[float, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}

    def channel_arrays(self, depth: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Channel nodes, response weights and phase weights, computed once per depth value (decision 145)."""
        key = float(depth)
        if key not in self._arrays:
            self._arrays[key] = _channel_arrays(self.channels, self.averaged.phase, key)
        return self._arrays[key]

    def table(self, b_gauss: float) -> FieldAverage:
        node = np.array([b_gauss * 1e6])
        if isinstance(self.memo, StageMemo):
            return _s1_table(self.memo, self.s1, node, np.ones(1), kind="node")
        key = (self.s1, float(b_gauss))
        if key not in self.memo:
            self.memo[key] = _s1_table(
                self.memo, self.s1, np.array([b_gauss * 1e6]), np.ones(1)
            )
        return self.memo[key]

    def gamma_at(self, z: Any) -> Any:
        return (
            self.ref * jnp.exp(self.h * z)
            if self.chart == "log"
            else self.ref + self.h * z
        )


def _zeros(ctx: _Context, probs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    shape = (ctx.order + 1, np.asarray(ctx.channels.nodes).shape[0]) + probs.shape[1:]
    return np.zeros(shape), np.zeros(shape, complex)


def _leaf(
    point: dict, ctx: _Context, eta: tuple[np.ndarray, np.ndarray] | None = None
) -> tuple[Any, Any]:
    """The series at fixed ``B`` with ``eta`` fixed or averaged. Nodes at ``|eta| = 1`` or ``B = 0`` add zero."""
    nodes, probs = (
        eta if eta is not None else (np.array([point.get("eta", 0.0)]), np.array([1.0]))
    )
    if not np.all(np.isfinite(nodes) & (np.abs(nodes) <= 1.0)):
        raise StackRefused(
            f"inclination nodes {nodes} outside [-1, 1]: the jet takes the cell"
        )
    keep = np.abs(nodes) < 1.0  # s = sqrt(1 - eta^2) = 0 gives no emission
    if ctx.cached is None and float(point["B"]) < 0:
        raise StackRefused(
            f"a negative field node B = {point['B']}: the jet takes the cell"
        )
    if not np.any(keep) or (ctx.cached is None and float(point["B"]) == 0.0):
        return _zeros(ctx, probs)
    nodes, probs = nodes[keep], probs[keep]
    table = ctx.cached if ctx.cached is not None else ctx.table(float(point["B"]))
    if ctx.dry:
        return _zeros(ctx, probs)
    depth = float(point.get("depth", 0.0))
    return point_series(
        table,
        ctx.channels,
        ctx.averaged.phase,
        ctx.ref,
        ctx.chart,
        ctx.h,
        np.sqrt(1 - nodes**2),
        probs,
        ctx.order,
        depth=depth,
        arrays=ctx.channel_arrays(depth),
    )


def _walk(levels: list, point: dict, ctx: _Context) -> tuple[Any, Any]:
    if not levels:
        return _leaf(point, ctx)
    kind, name, spec = levels[0]
    if kind == "legendre":
        degree, n = spec
        x, w = np.polynomial.legendre.leggauss(n)
        table = np.stack(
            [np.polynomial.legendre.Legendre.basis(l)(x) for l in range(degree + 1)],
            axis=1,
        )
        weights = w[:, None] * table * (2 * np.arange(degree + 1) + 1) / 2.0
        parts = [_walk(levels[1:], {**point, name: float(v)}, ctx) for v in x]
        return tuple(
            np.einsum("nrj...,nl->rjl...", np.stack([p[k] for p in parts]), weights)
            for k in range(2)
        )
    rule = spec
    if (
        len(levels) == 1
        and name == "eta"
        and rule.parents is not None
        and "gamma" not in rule.parents
    ):
        _check_undeclared(rule, name, point, ctx)
        return _leaf(point, ctx, eta=_call(rule, name, point))
    nodes, series = _weights(rule, name, point, ctx)
    total_i, total_p = 0.0, 0.0
    for x, w in zip(nodes, series):
        ci, cp = _walk(levels[1:], {**point, name: float(x)}, ctx)
        shape = (ctx.order + 1,) + (1,) * (ci.ndim - 1)
        total_i = total_i + _mul(w.reshape(shape), ci, ctx.order)
        total_p = total_p + _mul(w.reshape(shape), cp, ctx.order)
    return total_i, total_p


def _levels(averaged: AveragedKernel) -> list:
    levels = [("legendre", n, (L, k)) for n, L, k in averaged.legendre]
    for name, rule in averaged.averaged:
        if not isinstance(rule, NodeRule):
            raise StackRefused(
                f"the law of {name} is a bare function: declare it as a NodeRule with its parents"
            )
        levels.append(("avg", name, rule))
    for name, _, _ in averaged.legendre:
        if name != "eta":
            raise StackRefused(
                f"a Legendre axis on {name}: the stack projects only eta"
            )
    return levels


def _cached(levels: list, point: dict, ctx: _Context) -> list:
    """Average the field law once by its S1 table when it has no parents and no other law depends on B."""
    field = [lv for lv in levels if lv[0] == "avg" and lv[1] == "B"]
    if not field or field[0][2].parents != ():
        return levels
    if any(lv[0] == "avg" and lv[2].parents and "B" in lv[2].parents for lv in levels):
        return levels
    _check_undeclared(field[0][2], "B", point, ctx)
    nodes, probs = _call(field[0][2], "B", point)
    if isinstance(ctx.memo, StageMemo):
        ctx.cached = _s1_table(ctx.memo, ctx.s1, nodes * 1e6, probs)
    else:
        key = (ctx.s1, "law", tuple(nodes), tuple(probs))
        if key not in ctx.memo:
            ctx.memo[key] = _s1_table(ctx.memo, ctx.s1, nodes * 1e6, probs)
        ctx.cached = ctx.memo[key]
    return [lv for lv in levels if lv is not field[0]]


def _gamma_series(
    averaged: AveragedKernel,
    channels: Any,
    chart: str,
    ref: float,
    h: float,
    order: int,
    s1: str,
    memo: dict,
    fixed: dict,
    reweight: bool,
    probe: tuple[str, ...] = ("gamma",),
    dry: bool = False,
) -> tuple[Any, Any, str]:
    """Energy series ``(I, P)`` of the walk and its mode, "cached-S1" or "per-node"."""
    if type(
        averaged.kernel
    ) is not ContinuumKernel or not any(  # a subclass may change the kernel values
        isinstance(c, Similarity) and "gamma" in c.variables
        for c in averaged.kernel.capabilities
    ):
        raise StackRefused(
            f"the stack holds the continuum F and G only, not {type(averaged.kernel).__name__}: the jet takes it"
        )
    ctx = _Context(
        averaged, channels, chart, ref, h, order, s1, memo, reweight, probe, dry
    )
    levels = [
        lv
        for lv in _levels(averaged)
        if not (
            lv[0] == "avg" and lv[1] in _KERNEL_FREE and not _feeds(lv[1], averaged)
        )
    ]
    point = {**dict(averaged.fixed), **fixed, "gamma": ref}
    if "B" not in point:
        levels = _cached(levels, point, ctx)
    if (
        ctx.cached is None
        and "B" not in point
        and not any(lv[0] == "avg" and lv[1] == "B" for lv in levels)
    ):
        raise StackRefused("B is neither celled nor averaged nor fixed")
    si, sp = _walk(levels, point, ctx)
    return si, sp, "per-node" if ctx.cached is None else "cached-S1"


def _feeds(name: str, averaged: AveragedKernel) -> bool:
    return any(
        isinstance(r, NodeRule) and (r.parents is None or name in r.parents)
        for _, r in averaged.averaged
    )


def stack_cell(
    averaged: AveragedKernel,
    channels: Any,
    lattice: Any,
    cell: int,
    *,
    s1: str = "lookup",
    memo: dict | None = None,
    reweight: bool = True,
    record: list | None = None,
) -> tuple[Any, Any]:
    """``(I, P)`` rows of one product cell in lattice order, or ``StackRefused``.

    ``record``, when given, receives the mode of the cell: "cached-S1", "per-node" or "field-scaling" (decision
    145). A refused cell appends nothing.
    """
    try:
        rows, mode = _stack_cell(
            averaged, channels, lattice, cell, s1, memo, reweight, False
        )
    except OutsideTable as err:
        raise StackRefused(f"{err}. The jet takes the cell") from err
    if record is not None:
        record.append(mode)
    return rows


def stack_stage(
    averaged: AveragedKernel,
    channels: Any,
    lattice: Any,
    cell: int,
    *,
    s1: str = "lookup",
    memo: Any = None,
    reweight: bool = True,
) -> str:
    """The mode of ``stack_cell`` on ``cell`` from a dry walk, or ``StackRefused``.

    The dry walk evaluates the laws, runs the refusal checks that do not read the channels and reads the S1
    tables of the walk through ``memo``, so a ``StageMemo`` then holds the S1 keys that ``stack_cell`` reads
    (the S2 stage of decision 159). It computes no channel series. The grid reach of the channels refuses in
    ``stack_cell`` only.
    """
    try:
        _, mode = _stack_cell(
            averaged, channels, lattice, cell, s1, memo, reweight, True
        )
    except OutsideTable as err:
        raise StackRefused(f"{err}. The jet takes the cell") from err
    return mode


def _stack_cell(
    averaged: AveragedKernel,
    channels: Any,
    lattice: Any,
    cell: int,
    s1: str,
    memo: dict | None,
    reweight: bool,
    dry: bool,
) -> tuple[tuple[Any, Any], str]:
    if s1 not in ("lookup", "quadrature"):
        raise ValueError("s1 must be 'lookup' or 'quadrature'")
    memo = {} if memo is None else memo
    table, index = lattice.cells, lattice.index
    names = table.variables
    _check_lattice(averaged, lattice)
    g = names.index("gamma")
    chart, ref, h = (
        table.chart[cell][g],
        float(table.reference[cell, g]),
        float(table.half_width[cell, g]),
    )
    pattern = list(index.patterns[int(index.cell_pattern[cell])])
    if max(sum(e) for e in pattern) > TABLE_ORDER:
        raise StackRefused(
            f"order {max(sum(e) for e in pattern)} exceeds the S1 table order {TABLE_ORDER}: the jet takes the cell"
        )
    if names == ("gamma",):
        order = max(e[0] for e in pattern)
        si, sp, mode = _gamma_series(
            averaged, channels, chart, ref, h, order, s1, memo, {}, reweight, dry=dry
        )
        rows = _to_rows(si, pattern, lambda e: si[e[0]]), _to_rows(
            sp, pattern, lambda e: sp[e[0]]
        )
        return rows, mode

    def series(need: int, fixed: dict, probe: tuple[str, ...]) -> tuple[Any, Any]:
        si, sp, _ = _gamma_series(
            averaged, channels, "log", ref, h, need, s1, memo, fixed, True, probe, dry
        )
        return si, sp

    rows = _field_cell(averaged, table, cell, pattern, names, (chart, ref, h), series)
    return rows, "field-scaling"


def stack_lattice(
    averaged: AveragedKernel, channels: Any, lattice: Any, *, s1: str = "lookup"
) -> tuple[Any, Any]:
    """Rows of every product cell in the rows of one block. One memo of S1 tables serves all cells."""
    memo: dict = {}
    parts = [
        stack_cell(averaged, channels, lattice, c, s1=s1, memo=memo)
        for c in range(lattice.n_cells)
    ]
    return np.concatenate([p[0] for p in parts]), np.concatenate([p[1] for p in parts])
