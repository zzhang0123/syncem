"""Stack columns of a product cell: cached-S1 mode, per-node mode and field cells (design/0.8/SPEC.md Sec. 8.3).

LABEL ``eq: illustrated derivative spectra``. The averaged kernel's levels are walked with Taylor series in the
energy chart ``u``. At the leaf the stack gives the series of the kernel at fixed ``B`` and ``eta``
(``provider_stack.point_series``) from a one-node S1 table. An averaged level sums its children with the
weights of its rule. A rule whose parents include the celled ``gamma`` keeps its nodes at the cell's reference
point and reweights them, ``w_q(u) = w_q(ref) rho(x_q | gamma(u)) / rho(x_q | ref)``, whose Taylor coefficients
enter by a Cauchy product (frozen-node reweighting). A field law with no parents that no other law depends on is
averaged once by its S1 table (cached-S1 mode). A ``(gamma, B)`` cell takes the field-scaling relation from the
energy series at ``B_b`` (``expand.field_scaling_columns``) when no law depends on ``gamma`` or ``B``.

Cases the stack cannot take raise ``StackRefused`` with the reason, and the caller takes the jet: a kernel other
than ``ContinuumKernel`` itself (S0 holds its ``F`` and ``G``), unknown parents, a law that reads a celled variable
it does not declare (an order-1 probe per rule and cell), a celled parent without a density or with one that JAX
cannot trace, reweighted weights that do not reproduce the jet of the rule itself (decision 139: node by node for
fixed nodes, on ``t^m``, ``m = 0..3``, for moving ones), weights amplified beyond ``_MAX_AMPLIFICATION`` (a law that
moves by many widths across the cell), orders above the S1 table order, energies above the S1 grid for the lowest
channel, and celled variables other than ``gamma`` and ``B``. Nodes at ``|eta| = 1`` or ``B = 0`` add zero. All S1
tables of one memo read one S0 table. Units: those of the kernel. Shapes: series ``(order + 1, J, *L)``, cell rows
``(n_rows, J)`` in the lattice order. Not certified: the reweighted quadrature, validated against the jet. The
check bounds moments, so a law whose nodes do not resolve the kernel gives a quadrature error that the jet shares.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from ..base.capabilities import FieldScalingRelation, Similarity
from ..base.continuum import ContinuumKernel
from .average import AveragedKernel, NodeRule
from .expand import field_scaling_columns
from .field_average import TABLE_ORDER, FieldAverage, OutsideTable
from .primitive import PrimitiveTable
from .provider_jet import taylor_coefficients
from .provider_stack import point_series

__all__ = ["StackRefused", "stack_cell", "stack_lattice"]

_KERNEL_FREE = {
    "mu",
    "phi",
}  # the continuum kernel is the isotropic pitch average and carries no azimuth factor


class StackRefused(ValueError):
    """The stack provider does not apply to this product cell. The reason names the condition."""


def _mul(a: np.ndarray, b: np.ndarray, order: int) -> np.ndarray:
    """Cauchy product along axis 0, truncated at ``order``. ``a`` is ``(order + 1,)`` or broadcasts."""
    out = np.zeros(np.broadcast_shapes(a.shape, b.shape), dtype=np.result_type(a, b))
    for n in range(order + 1):
        for k in range(n + 1):
            out[n] = out[n] + a[k] * b[n - k]
    return out


_S0_SPAN = (
    float(np.log(1e-12)),
    float(np.log(1e7)),
)  # the first shared S0 covers at least this ln x range
_S0_HEADROOM = 10.0  # e-folds added beyond a short end on a rebuild, so a monotone field sequence rebuilds rarely


def _primitive(memo: dict, nodes_ug: np.ndarray) -> PrimitiveTable:
    """The memo's S0 table, rebuilt when a law needs more, with ``_S0_HEADROOM`` beyond each short end (decision 144)."""
    lo, hi = FieldAverage.needed_lnx(nodes_ug)
    s0 = memo.get("S0")
    cur_lo, cur_hi = _S0_SPAN if s0 is None else (s0.lnx0, s0.lnx_hi)
    if s0 is not None and cur_lo <= lo and cur_hi >= hi:
        return s0
    new_lo = cur_lo if lo >= cur_lo else lo - _S0_HEADROOM
    new_hi = cur_hi if hi <= cur_hi else hi + _S0_HEADROOM
    memo["S0"] = PrimitiveTable.build(new_lo, new_hi, order=TABLE_ORDER)
    return memo["S0"]


def _s1_table(
    memo: dict, method: str, nodes_ug: np.ndarray, probs: np.ndarray
) -> FieldAverage:
    """An S1 table of the field law. Nodes at ``B = 0`` and zero weights are dropped: ``B W(y / B) -> 0``."""
    nodes_ug, probs = np.asarray(nodes_ug, float), np.asarray(probs, float)
    if (
        nodes_ug.shape != probs.shape
        or not np.all(np.isfinite(nodes_ug) & (nodes_ug >= 0))
        or not np.all(np.isfinite(probs) & (probs >= 0))
    ):
        raise StackRefused(
            "the field law needs finite non-negative nodes and weights of equal length: the jet takes the cell"
        )
    keep = (nodes_ug > 0) & (probs > 0)
    if not np.any(keep):
        raise StackRefused(
            "the field law puts all its weight at B = 0: the jet takes the cell"
        )
    nodes_ug, probs = nodes_ug[keep], probs[keep]
    if method == "quadrature":
        return FieldAverage.quadrature(nodes_ug, probs)
    return FieldAverage.lookups(nodes_ug, probs, primitive=_primitive(memo, nodes_ug))


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
    ):
        self.averaged, self.channels, self.chart, self.ref, self.h = (
            averaged,
            channels,
            chart,
            ref,
            h,
        )
        self.order, self.s1, self.memo, self.reweight = order, s1, memo, reweight
        self.probe = probe  # celled variables a law must declare when it reads them
        self.probed: set[int] = set()
        self.cached: FieldAverage | None = None

    def table(self, b_gauss: float) -> FieldAverage:
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


def _call(rule: NodeRule, name: str, point: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(nodes, probs)`` of a rule at a concrete point. A read of a variable the point lacks refuses."""
    try:
        nodes, probs = rule(point)
    except KeyError as err:
        raise StackRefused(
            f"the law of {name} reads {err} outside its declared parents {rule.parents}: the jet takes the cell"
        ) from err
    return np.asarray(nodes, float), np.asarray(probs, float)


def _check_undeclared(rule: NodeRule, name: str, point: dict, ctx: _Context) -> None:
    """A law must declare every celled variable it reads. Probed once per rule and cell by an order-1 jet."""
    if id(rule) in ctx.probed:
        return
    ctx.probed.add(id(rule))
    for var in ctx.probe:
        if var in rule.parents or var not in point:
            continue

        def moved(z: Any, var: str = var) -> Any:
            x, p = rule({**point, var: point[var] * jnp.exp(z[0])})
            return {"x": jnp.ravel(jnp.asarray(x)), "p": jnp.ravel(jnp.asarray(p))}

        try:
            jet = taylor_coefficients(moved, 1, [(0,), (1,)])
        except TypeError as err:  # a read through Python control flow or float()
            raise StackRefused(
                f"the law of {name} cannot be traced in {var}, which it does not declare: the jet takes the cell"
            ) from err
        for key in ("x", "p"):
            a = np.asarray(jet[key])
            scale = max(float(np.max(np.abs(a[0]), initial=0.0)), 1e-300)
            if not np.all(np.isfinite(a)) or np.any(np.abs(a[1]) > 1e-10 * scale):
                raise StackRefused(
                    f"the law of {name} depends on {var} but declares parents {rule.parents}: the jet takes the cell"
                )


_MAX_AMPLIFICATION = 1e4  # decision 139: the cancellation error is 1.5e-14 to 5.4e-14 of the amplification


def _weights(
    rule: NodeRule, name: str, point: dict, ctx: _Context
) -> tuple[Any, np.ndarray]:
    """Nodes and weight series ``(n_q, order + 1)`` of one averaged level."""
    if rule.parents is None:
        raise StackRefused(
            "a law with undeclared parents: only the jet differentiates it"
        )
    _check_undeclared(rule, name, point, ctx)
    nodes, probs = _call(rule, name, point)
    series = np.zeros((nodes.size, ctx.order + 1))
    series[:, 0] = probs
    if "gamma" not in rule.parents:
        return nodes, series
    if not ctx.reweight:
        return nodes, series  # the naive frozen weights, kept for the regression test
    if rule.density is None:
        raise StackRefused(
            "a law conditional on the celled gamma without a density: frozen-node reweighting needs it"
        )

    def ratio(z: Any) -> Any:
        moved = {**point, "gamma": ctx.gamma_at(z[0])}
        return rule.density(jnp.asarray(nodes), moved) / rule.density(
            jnp.asarray(nodes), point
        )

    try:
        dens = np.asarray(rule.density(jnp.asarray(nodes), point), float)
        if np.any((probs > 0) & ~(np.isfinite(dens) & (dens > 0))):
            raise StackRefused(
                f"the density of a law conditional on gamma is not positive and finite at its own nodes {nodes}: "
                "the jet takes the cell"
            )
        coefs = np.asarray(
            taylor_coefficients(ratio, 1, [(n,) for n in range(ctx.order + 1)])
        )  # (order+1, n_q)
    except TypeError as err:  # a density that JAX cannot trace in gamma
        raise StackRefused(
            f"the density of the law of {name} cannot be differentiated in gamma: {err}"
        ) from err
    series = probs[:, None] * coefs.T
    _check_frozen(rule, point, ctx, nodes, series)
    amplification = float(
        np.max(np.sum(np.abs(series), axis=0)) / np.sum(np.abs(probs))
    )
    if amplification > _MAX_AMPLIFICATION:
        raise StackRefused(
            f"frozen-node reweighting amplifies the weights by {amplification:.3g} at {point}: the law moves by "
            "many widths across the cell and the columns lose digits by cancellation. The jet takes the cell"
        )
    return nodes, series


_TEST_POWERS = 3


def _check_frozen(
    rule: NodeRule,
    point: dict,
    ctx: _Context,
    nodes: np.ndarray,
    series: np.ndarray,
    tol: float = 1e-8,
) -> None:
    """Decision 139: the reweighted weights reproduce the jet of the rule itself.

    With nodes that do not move with ``gamma`` the reference is the rule's own weight jet, node by node (an exact
    identity for a density that matches the rule). With moving nodes it is the jet of ``sum_q p_q(gamma)
    t(x_q(gamma))^m``, ``m = 0..3``, ``t = (x - centre) / spread`` from the reference nodes, against
    ``sum_q w_q^(k) t_q^m``. They agree to ``tol`` of ``max(1, sum_q |w_q^(k)| |t_q|^m)``. A support that moves with
    ``gamma``, a density whose ``gamma`` dependence differs from the rule's in these moments, or non-finite weights
    fail. The check bounds moments, not the integrand: a law whose nodes do not resolve the kernel is a quadrature
    error that the jet shares (decision 139). The columns are Taylor coefficients at the reference point, so node
    shifts across the cell do not enter.
    """
    order = series.shape[1] - 1
    probs = series[:, 0]
    centre = float(np.sum(probs * nodes))
    spread = float(np.sqrt(np.sum(probs * (nodes - centre) ** 2))) or 1.0
    powers = np.arange(_TEST_POWERS + 1)

    def moments(z: Any) -> Any:
        x, p = rule({**point, "gamma": ctx.gamma_at(z[0])})
        x, p = jnp.asarray(x), jnp.asarray(p)
        t = (x - centre) / spread
        return {"x": x, "p": p, "m": jnp.stack([jnp.sum(p * t**m) for m in powers])}

    try:
        jet = taylor_coefficients(moments, 1, [(n,) for n in range(order + 1)])
    except TypeError as err:  # a rule that JAX cannot trace in gamma
        raise StackRefused(
            f"the rule of a law conditional on gamma cannot be differentiated: {err}"
        ) from err
    jx, jp, jm = (np.asarray(jet[k]) for k in ("x", "p", "m"))
    if np.all(np.abs(jx[1:]) <= 1e-12 * (float(np.max(np.abs(nodes))) or 1.0)):
        scale = np.maximum(1.0, np.sum(np.abs(series), axis=0))[None, :]
        bad = (
            ~np.isfinite(series)
            | ~np.isfinite(jp.T)
            | (np.abs(series - jp.T) > tol * scale)
        )
        if np.any(bad):
            q, k = (int(i) for i in np.argwhere(bad)[0])
            raise StackRefused(
                f"frozen-node reweighting does not reproduce the rule's weights at its fixed nodes at {point}: node "
                f"{q}, order {k}, {series[q, k]:.6g} against {jp[k, q]:.6g}. The density does not match the rule. "
                "The jet takes the cell"
            )
        return
    t = (nodes - centre) / spread
    frozen = series.T @ (t[:, None] ** powers)  # (order + 1, m)
    scale = np.maximum(1.0, np.abs(series).T @ (np.abs(t)[:, None] ** powers))
    bad = ~np.isfinite(frozen) | ~np.isfinite(jm) | (np.abs(frozen - jm) > tol * scale)
    if np.any(bad):
        k, m = (int(i) for i in np.argwhere(bad)[0])
        raise StackRefused(
            f"frozen-node reweighting does not reproduce the moving nodes at {point}: order {k}, test power {m}, "
            f"{frozen[k, m]:.6g} against {jm[k, m]:.6g}. The support moves with gamma, the density does not "
            "match the rule, or the nodes do not resolve these moments. The jet takes the cell"
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
        depth=float(point.get("depth", 0.0)),
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
) -> tuple[Any, Any]:
    if type(
        averaged.kernel
    ) is not ContinuumKernel or not any(  # a subclass may change the kernel values
        isinstance(c, Similarity) and "gamma" in c.variables
        for c in averaged.kernel.capabilities
    ):
        raise StackRefused(
            f"the stack holds the continuum F and G only, not {type(averaged.kernel).__name__}: the jet takes it"
        )
    ctx = _Context(averaged, channels, chart, ref, h, order, s1, memo, reweight, probe)
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
    return _walk(levels, point, ctx)


def _feeds(name: str, averaged: AveragedKernel) -> bool:
    return any(
        isinstance(r, NodeRule) and (r.parents is None or name in r.parents)
        for _, r in averaged.averaged
    )


def _to_rows(series: Any, pattern: list[tuple[int, ...]], picker: Any) -> Any:
    """Rows in lattice order (Legendre pair, then monomial) from per-exponent arrays ``(J, *L)``."""
    cols = np.stack([picker(e) for e in pattern])  # (n_m, J, *L)
    n_l = cols.ndim - 2
    if n_l == 0:
        return cols
    moved = np.moveaxis(cols, tuple(range(2, cols.ndim)), tuple(range(n_l)))
    return moved.reshape((-1, cols.shape[1]))


def stack_cell(
    averaged: AveragedKernel,
    channels: Any,
    lattice: Any,
    cell: int,
    *,
    s1: str = "lookup",
    memo: dict | None = None,
    reweight: bool = True,
) -> tuple[Any, Any]:
    """``(I, P)`` rows of one product cell in lattice order, or ``StackRefused``."""
    try:
        return _stack_cell(averaged, channels, lattice, cell, s1, memo, reweight)
    except OutsideTable as err:
        raise StackRefused(f"{err}. The jet takes the cell") from err


def _stack_cell(
    averaged: AveragedKernel,
    channels: Any,
    lattice: Any,
    cell: int,
    s1: str,
    memo: dict | None,
    reweight: bool,
) -> tuple[Any, Any]:
    if s1 not in ("lookup", "quadrature"):
        raise ValueError("s1 must be 'lookup' or 'quadrature'")
    memo = {} if memo is None else memo
    table, index = lattice.cells, lattice.index
    names = table.variables
    if "gamma" not in names or not set(names) <= {"gamma", "B"}:
        raise StackRefused(
            f"celled variables {names}: the stack takes gamma and optionally B. The jet takes the cell"
        )
    taken = {n for n, _ in averaged.averaged} | {n for n, _ in averaged.fixed}
    if set(names) & taken:
        raise ValueError(
            f"{sorted(set(names) & taken)} are celled and also averaged or fixed in the averaged kernel"
        )
    if index.legendre != tuple((n, L) for n, L, _ in averaged.legendre):
        raise ValueError(
            "the lattice's Legendre axes differ from the averaged kernel's"
        )
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
        si, sp = _gamma_series(
            averaged, channels, chart, ref, h, order, s1, memo, {}, reweight
        )
        return _to_rows(si, pattern, lambda e: si[e[0]]), _to_rows(
            sp, pattern, lambda e: sp[e[0]]
        )
    return _field_cell(
        averaged, channels, table, cell, pattern, names, (chart, ref, h), s1, memo
    )


def _field_cell(
    averaged: AveragedKernel,
    channels: Any,
    table: Any,
    cell: int,
    pattern: list,
    names: tuple,
    energy: tuple,
    s1: str,
    memo: dict,
) -> tuple[Any, Any]:
    chart, ref, h = energy
    if chart != "log":
        raise StackRefused(
            "field cells with an affine energy chart: the scaling relation needs the log chart"
        )
    if not any(
        isinstance(c, FieldScalingRelation) for c in averaged.kernel.capabilities
    ):
        raise StackRefused("the kernel declares no FieldScalingRelation")
    for name, rule in averaged.averaged:
        if not isinstance(rule, NodeRule):
            raise StackRefused(
                f"the law of {name} is a bare function: the field-scaling precondition needs its parents"
            )
        if rule.parents is None:
            raise StackRefused(
                f"the law of {name} has undeclared parents: the field-scaling precondition needs them"
            )
        if {"gamma", "B"} & set(rule.parents):
            raise StackRefused(
                f"the law of {name} depends on gamma or B: the field-scaling precondition fails"
            )
    b = names.index("B")
    b_chart, b_ref, b_h = (
        table.chart[cell][b],
        float(table.reference[cell, b]),
        float(table.half_width[cell, b]),
    )
    need = max(sum(e) for e in pattern)
    si, sp = _gamma_series(
        averaged,
        channels,
        "log",
        ref,
        h,
        need,
        s1,
        memo,
        {"B": b_ref},
        True,
        ("gamma", "B"),
    )
    exps = [(e[names.index("gamma")], e[b]) for e in pattern]
    kw = dict(
        h_gamma=h,
        field_chart=b_chart,
        field_ref=b_ref,
        field_half_width=b_h,
        field_basis=b_ref,
    )
    ci = field_scaling_columns({n: si[n] for n in range(need + 1)}, exps, **kw)
    cp = field_scaling_columns({n: sp[n] for n in range(need + 1)}, exps, **kw)
    return (
        _to_rows(ci, exps, lambda e: np.asarray(ci[e])),
        _to_rows(cp, exps, lambda e: np.asarray(cp[e])),
    )


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
