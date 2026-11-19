"""Digest chains of averaged rules and the trace identity of a rule (design/0.8/SPEC.md decisions 157, 160).

LABEL ``[extension]``. ``rule_chain(rule, name, averaged, lattice, cell, order)`` gives the digests ``d_0, ...,
d_order`` of the ``NodeRule`` of the averaged variable ``name`` at one product cell. The chain reads the rule
through its values:

* the context is the declared parents, closed over the parents of averaged parents. A celled parent is
  resolved, a fixed value enters the point, and a Legendre or earlier averaged parent is a level of a nested
  grid over that parent's nodes (Gauss nodes for a Legendre variable), in the order of the averaged kernel,
* ``d_0`` hashes the name, parents, celled names, the cell's charts, reference point and half-widths, and the
  nodes, probabilities and density values ``density(nodes, point)`` on the nested grid at the point sets: the
  reference point, and both edges and 8 Gauss-Legendre points in the chart of each resolved parent's cell,
  one variable at a time with the others at the reference. An edge outside the chart's domain is taken at
  chart coordinate -1 or 1,
* ``d_n = H(d_(n-1), c_n)``. ``c_n`` holds the order-``n`` Taylor coefficients at the reference point in the
  chart coordinates of every celled variable, declared or not. The nodes and probabilities are differentiated
  as the jet moves them, with the nodes of averaged parents moving with the point. The density is
  differentiated at the rule's nodes and the parents' nodes frozen at the reference, as the stack reweights.
  One eager ``taylor_coefficients`` call per rule and cell gives every ``c_n``.

``order`` is the largest total degree, over every celled variable, of an exponent the build reads. A rule that
is not a ``NodeRule``, has ``parents=None``, declares a parent that is not celled, fixed, projected or averaged
before it, reads an undeclared Legendre or averaged variable (by ``[]``, ``get`` or ``in``), or cannot be
traced by JAX has no chain.
``rule_chain`` then returns ``None`` and ``chain_with_reason`` gives a reason starting "not cached: ".
``TabulatedRule``, ``with_tabulated_rules`` and ``fixed_fingerprint`` live in ``engine/_tabulated_rule.py``
and are imported here.

Cost per reference point of ``B`` given ``eta`` (16 by 12 nodes, median a power of each celled variable), first
call in a fresh process and warm mean over 3 to 8 other cells, 1-minute load 17.3 to 17.6 (2026-11-19, loaded).
At ``N = 2``: 1.40 s and 0.012 s (``d = 1``), 1.82 s and 0.020 s (``d = 2``), 1.86 s and 0.020 s (``d = 3``).
At ``N = 4``: 1.78 s and 0.020 s (``d = 1``), 2.47 s and 0.039 s (``d = 2``), 2.47 s and 0.049 s (``d = 3``).

Units: those of the rule's nodes, the density per unit of the node, dimensionless chart coordinates. Shapes:
a tuple of ``order + 1`` sha256 hex strings. Not certified: the chain identifies a rule by finitely many values
and coefficients, so two rules equal on them share keys. S1 and S2 read the rule through these only.
"""

from __future__ import annotations

import dataclasses
import itertools
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ..base._fingerprint import NotCacheable, digest
from .average import AveragedKernel, NodeRule
from ._tabulated_rule import TabulatedRule, fixed_fingerprint, with_tabulated_rules
from .provider_jet import _point, taylor_coefficients

__all__ = [
    "TabulatedRule",
    "chain_with_reason",
    "fixed_fingerprint",
    "rule_chain",
    "with_tabulated_rules",
]

CHAIN_FORMAT = "syncem-rule-chain-1"
CELL_POINTS = 8  # Gauss-Legendre points per resolved parent cell (decision 157)


@dataclasses.dataclass(frozen=True)
class _Level:
    """One level of the nested grid: a Legendre variable (fixed ``values``) or an averaged ``rule``."""

    name: str
    values: Any = None
    rule: Any = None


@dataclasses.dataclass(frozen=True)
class _Context:
    """What the chain of one rule reads: grid levels, celled names, resolved names and fixed values."""

    levels: tuple[_Level, ...]
    celled: tuple[str, ...]
    resolved: tuple[str, ...]
    fixed: tuple[tuple[str, float], ...]
    hidden: frozenset[str] = frozenset()


class _Point(dict):
    """A point of the chain. A read of a hidden name by ``[]``, ``get`` or ``in`` raises ``NotCacheable``.

    The hidden names are the Legendre and earlier averaged variables that the rule does not declare. The jet
    gives a rule their values, and the chain does not, so a read of one would key a law the chain never saw.
    """

    __slots__ = ("hidden",)

    def __init__(self, values: dict, hidden: frozenset[str]) -> None:
        super().__init__(values)
        self.hidden = hidden

    def _check(self, key: Any) -> None:
        if key in self.hidden and not dict.__contains__(self, key):
            raise NotCacheable(
                f"a law reads {key!r}, which it does not declare as a parent"
            )

    def __getitem__(self, key: Any) -> Any:
        self._check(key)
        return dict.__getitem__(self, key)

    def get(self, key: Any, default: Any = None) -> Any:
        self._check(key)
        return dict.get(self, key, default)

    def __contains__(self, key: Any) -> bool:
        self._check(key)
        return dict.__contains__(self, key)


# ------------------------------------------------------------------ context of a rule


def _declared(rule: Any, name: str) -> tuple[str, ...]:
    if not isinstance(rule, NodeRule) or rule.parents is None:
        what = "parents=None" if isinstance(rule, NodeRule) else "a bare callable"
        raise NotCacheable(
            f"the law of {name} has {what}, so its tabulation points are undefined"
        )
    return tuple(rule.parents)


def _context(rule: Any, name: str, averaged: AveragedKernel, celled: tuple) -> _Context:
    """The nested grid and the resolved parents of ``rule``, closed over the parents of its averaged parents."""
    names = [n for n, _ in averaged.averaged]
    earlier = dict(averaged.averaged[: names.index(name)])
    legendre = {n: k for n, _, k in averaged.legendre}
    fixed = dict(averaged.fixed)
    used, resolved, stack = set(), set(), [(name, rule)]
    while stack:
        owner, law = stack.pop()
        for parent in _declared(law, owner):
            if parent in celled:
                resolved.add(parent)
            elif parent in legendre or parent in fixed:
                used.add(parent)
            elif parent in earlier:
                if parent not in used:
                    used.add(parent)
                    stack.append((parent, earlier[parent]))
            else:
                raise NotCacheable(
                    f"the law of {owner} declares the parent {parent}, which is not celled, fixed, "
                    "projected or averaged before it"
                )
    levels = [
        _Level(n, values=jnp.asarray(np.polynomial.legendre.leggauss(k)[0]))
        for n, k in legendre.items()
        if n in used
    ]
    levels += [_Level(n, rule=r) for n, r in earlier.items() if n in used]
    return _Context(
        tuple(levels),
        celled,
        tuple(v for v in celled if v in resolved),
        tuple((n, v) for n, v in averaged.fixed),
        frozenset((set(legendre) | set(earlier)) - used),
    )


# ------------------------------------------------------------------ the nested grid


def _grid(
    levels: tuple[_Level, ...], rule: Any, point: dict, hidden: frozenset[str]
) -> tuple[Any, Any, tuple]:
    """Nodes and probabilities ``(*grid, m)`` of ``rule`` and the level values ``(*grid,)`` of each level."""
    if not levels:
        nodes, probs = rule(_Point(point, hidden))
        return jnp.asarray(nodes, dtype=float), jnp.asarray(probs, dtype=float), ()
    level, rest = levels[0], levels[1:]
    values = (
        level.values if level.rule is None else level.rule(_Point(point, hidden))[0]
    )

    def one(value: Any) -> tuple[Any, Any, tuple]:
        nodes, probs, inner = _grid(rest, rule, {**point, level.name: value}, hidden)
        return nodes, probs, (jnp.broadcast_to(value, nodes.shape[:-1]), *inner)

    return jax.vmap(one)(jnp.asarray(values, dtype=float))


def _frozen_density(
    rule: NodeRule, ctx: _Context, nodes: Any, values: tuple, point: dict
) -> Any:
    """``density(nodes, point)`` at given nodes and level values, the point free to move."""
    if not ctx.levels:
        return jnp.asarray(rule.density(nodes, _Point(point, ctx.hidden)), dtype=float)
    grid = nodes.shape[:-1]
    flat = [jnp.reshape(v, (-1,)) for v in values]

    def one(q_nodes: Any, *q_values: Any) -> Any:
        at = {**point, **{lv.name: v for lv, v in zip(ctx.levels, q_values)}}
        return jnp.asarray(rule.density(q_nodes, _Point(at, ctx.hidden)), dtype=float)

    out = jax.vmap(one)(jnp.reshape(nodes, (-1, nodes.shape[-1])), *flat)
    return jnp.reshape(out, (*grid, -1))


def _tabulate(rule: NodeRule, ctx: _Context, point: dict) -> tuple[Any, ...]:
    """Nodes, probabilities, density at the own nodes (or ``None``) and level values at one point."""
    nodes, probs, values = _grid(ctx.levels, rule, point, ctx.hidden)
    if rule.density is None:
        return nodes, probs, None, values
    return nodes, probs, _frozen_density(rule, ctx, nodes, values, point), values


# ------------------------------------------------------------------ point sets and coefficients


def _chart_point(chart: str, ref: float, hw: float, u: Any) -> Any:
    return ref * np.exp(hw * u) if chart == "log" else ref + hw * u


def _edge(
    chart: str, ref: float, hw: float, x: float, side: float
) -> tuple[float, float]:
    """An edge and its chart coordinate. An edge outside the chart's domain is taken at ``side``."""
    if not np.isfinite(x) or (chart == "log" and x <= 0):
        return float(_chart_point(chart, ref, hw, side)), side
    return x, float(np.log(x / ref) / hw if chart == "log" else (x - ref) / hw)


def _cell_values(chart: str, ref: float, hw: float, lo: float, hi: float) -> np.ndarray:
    """Both edges and ``CELL_POINTS`` Gauss-Legendre points of one cell, in canonical units."""
    lo, a = _edge(chart, ref, hw, lo, -1.0)
    hi, b = _edge(chart, ref, hw, hi, 1.0)
    t = np.polynomial.legendre.leggauss(CELL_POINTS)[0]
    inner = _chart_point(chart, ref, hw, 0.5 * (b - a) * t + 0.5 * (a + b))
    return np.concatenate([np.asarray([lo, hi], dtype=float), inner])


def _point_set(ctx: _Context, table: Any, cell: int) -> np.ndarray:
    """``(n_points, n_celled)``: the reference, then the cell points of each resolved variable."""
    ref = np.asarray(table.reference[cell], dtype=float)
    rows = [ref]
    for v in ctx.resolved:
        i = ctx.celled.index(v)
        values = _cell_values(
            table.chart[cell][i],
            float(ref[i]),
            float(table.half_width[cell, i]),
            float(table.lo[cell, i]),
            float(table.hi[cell, i]),
        )
        block = np.repeat(ref[None, :], values.size, axis=0)
        block[:, i] = values
        rows.extend(block)
    return np.stack(rows)


def _exponents(d: int, order: int) -> tuple[tuple[int, ...], ...]:
    """Every exponent of total degree 1 to ``order`` in ``d`` variables, by degree then descending."""
    every = itertools.product(range(order + 1), repeat=d)
    chosen = [e for e in every if 1 <= sum(e) <= order]
    return tuple(sorted(chosen, key=lambda e: (sum(e), tuple(-x for x in e))))


def _coefficients(
    rule: NodeRule, ctx: _Context, table: Any, cell: int, frozen: tuple, exps: tuple
) -> tuple[Any, ...]:
    """Taylor coefficients ``(n_exp, ...)`` of nodes, probabilities and frozen-node density at the reference."""
    charts = tuple(table.chart[cell])
    ref = jnp.asarray(table.reference[cell], dtype=float)
    hw = jnp.asarray(table.half_width[cell], dtype=float)
    nodes0, _, _, values0 = frozen

    def f(z: Any) -> tuple[Any, ...]:
        point = {**dict(ctx.fixed), **_point(ctx.celled, charts, ref, hw, z)}
        nodes, probs, _ = _grid(ctx.levels, rule, point, ctx.hidden)
        if rule.density is None:
            return nodes, probs
        return nodes, probs, _frozen_density(rule, ctx, nodes0, values0, point)

    return tuple(taylor_coefficients(f, len(ctx.celled), exps))


# ------------------------------------------------------------------ the chain


def _first(
    name: str, rule: NodeRule, ctx: _Context, geometry: tuple, tab: tuple
) -> str:
    """``d_0``: name, parents, celled names, cell geometry, point sets and the values there."""
    nodes, probs, dens, _ = tab
    values = tuple(np.asarray(a) for a in (nodes, probs))
    density = "no density" if dens is None else np.asarray(dens)
    levels = tuple(lv.name for lv in ctx.levels)
    head = (CHAIN_FORMAT, name, tuple(rule.parents), ctx.celled, levels)
    return digest((*head, geometry, values, density))


def _chain(
    rule: Any, name: str, averaged: AveragedKernel, lattice: Any, cell: int, order: int
) -> tuple[str, ...]:
    table = lattice.cells
    ctx = _context(rule, name, averaged, tuple(table.variables))
    points = _point_set(ctx, table, cell)
    fixed = dict(ctx.fixed)

    def at(row: Any) -> tuple[Any, ...]:
        return _tabulate(rule, ctx, {**fixed, **dict(zip(ctx.celled, row))})

    tab = jax.vmap(at)(jnp.asarray(points))
    reference = jax.tree.map(lambda a: a[0], tab)
    geometry = (
        tuple(table.chart[cell]),
        np.asarray(table.reference[cell], dtype=float),
        np.asarray(table.half_width[cell], dtype=float),
        points,
    )
    chain = [_first(name, rule, ctx, geometry, tab)]
    exps = _exponents(len(ctx.celled), order)
    coefs = _coefficients(rule, ctx, table, cell, reference, exps) if exps else ()
    degrees = np.asarray([sum(e) for e in exps], dtype=int)
    for n in range(1, order + 1):
        pick = np.flatnonzero(degrees == n)
        c_n = (tuple(exps[k] for k in pick), *(np.asarray(a)[pick] for a in coefs))
        chain.append(digest((chain[-1], n, c_n)))
    return tuple(chain)


def _count(value: Any, upper: int | None = None) -> bool:
    ok = isinstance(value, (int, np.integer)) and not isinstance(value, bool)
    return ok and value >= 0 and (upper is None or value < upper)


def _check(
    name: str, averaged: AveragedKernel, lattice: Any, cell: Any, order: Any
) -> None:
    if name not in [n for n, _ in averaged.averaged]:
        raise ValueError(f"{name} is not an averaged variable of the averaged kernel")
    n_cells = lattice.cells.n_cells
    if not _count(cell, n_cells):
        raise ValueError(
            f"cell must be a product cell of 0..{n_cells - 1}, not {cell!r}"
        )
    if not _count(order):
        raise ValueError(f"order must be a non-negative int, not {order!r}")


def chain_with_reason(
    rule: Any, name: str, averaged: AveragedKernel, lattice: Any, cell: int, order: int
) -> tuple[tuple[str, ...] | None, str | None]:
    """``(chain, None)``, or ``(None, "not cached: <reason>")`` for a rule without a chain."""
    _check(name, averaged, lattice, cell, order)
    try:
        return _chain(rule, name, averaged, lattice, int(cell), int(order)), None
    except NotCacheable as err:
        return None, f"not cached: {err.reason}"
    except KeyError as err:
        return None, (
            f"not cached: the law of {name} or of a parent reads {err} outside its declared parents"
        )
    except TypeError as err:  # JAX refuses to trace or differentiate the rule
        return None, (
            f"not cached: the law of {name} cannot be tabulated and differentiated by JAX: {err}"
        )


def rule_chain(
    rule: Any, name: str, averaged: AveragedKernel, lattice: Any, cell: int, order: int
) -> tuple[str, ...] | None:
    """The digests ``d_0, ..., d_order`` of the law of ``name`` at ``cell``, or ``None`` without a chain."""
    return chain_with_reason(rule, name, averaged, lattice, cell, order)[0]
