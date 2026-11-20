"""The jet provider: Taylor columns of the averaged kernel in any number of celled variables (SPEC Sec. 8.3).

LABEL ``eq: channel basis expansion``. A product cell with celled variables ``v_1 .. v_d`` maps the chart
coordinates ``z`` to the point ``x_v = x_b e^(h z_v)`` (log) or ``x_b + h z_v`` (affine) and evaluates the
averaged kernel there. The columns are ``(1/alpha!) d^alpha f(0)`` for every exponent ``alpha`` of the cell's
lower set, by nested forward mode. Each nesting level differentiates only the directions that some exponent
needs at that depth (cap-aware nesting, the ``level_plan`` of ``syncem.model._harmonic_taylor`` generalised to
``d`` directions): an exponent's derivatives are taken innermost for the variables with the smallest caps, so a
variable of cap ``c`` appears only in the levels it can reach.

The functions loop in Python over cells, channels and node chunks, with one call of a compiled program per (cell,
channel, chunk) (decision 165). A channel of ``n_nu`` nodes runs in ``ceil(n_nu / 64)`` chunks of equal size,
the last padded with copies of its last node at weight 0, and the host adds the chunk columns in chunk order in
float64 (complex128 for P). This needs ``channel()`` to be the sum over the node axis of its per-node values
(Sec. 8.1). Each chunk is a ``ResponseView`` with the channel's response weights evaluated on the host, so every
channel family shares the program of a chunk size (decision 171). A channel's columns therefore depend on its
nodes and response weights, the cell, the exponent set and the averaged kernel, and not on the other channels or
cells of the call. Channels without nodes (``None``) run whole.

``make_routine(names, charts, exponents)`` makes a ``JetRoutine`` (``engine/_jet_routine.py``): programs compiled
ahead of time, at most 4 per routine, a memory budget checked before the first execution and an order limit
checked before tracing (decisions 166 and 169). The routines live in the process ``Cache`` (``engine/cache.py``,
decision 160) and not in module state. A call without ``routine=`` takes the routine of the signature from
``default_cache()`` with the token "direct". The basis build passes its own ``routine`` and wraps the rules as
``TabulatedRule``, so an edited rule retraces.

The cost per cell is about ``prod_i (1 + |D_i|)`` primal evaluations over the levels ``i``, with ``D_i`` the
directions of level ``i``. The plan prints it (SPEC Sec. 8.3, decision (b)). The columns carry the units of the kernel.
Shapes: host arrays, ``I`` and ``P`` ``(n_rows, n_ch, *L)``, one row per exponent (``P`` complex, one phase-weight
row). Not certified: nothing beyond the kernel and its averages, since forward mode gives the derivatives to
rounding, and the chunk sums differ from one sum over the channel nodes by rounding.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from ._jet_routine import JetRoutine, check_order, chunk_views
from .average import AveragedKernel
from .cache import default_cache

__all__ = [
    "DIRECT",
    "JET_EVENTS",
    "cost",
    "jet_cell",
    "jet_columns",
    "jet_lattice",
    "level_plan",
    "make_routine",
    "taylor_coefficients",
]

DIRECT = "direct"  # the routine token of a provider call outside a basis build
JET_EVENTS = ("jet groups", "jet calls", "jet compiles", "jet memory unchecked")


def _order(exponents: tuple[tuple[int, ...], ...]) -> list[int]:
    """Directions sorted by cap, smallest first (ties by position)."""
    d = len(exponents[0]) if exponents else 0
    caps = [max(e[v] for e in exponents) for v in range(d)]
    return sorted(range(d), key=lambda v: (caps[v], v))


def _sequence(alpha: tuple[int, ...], order: list[int]) -> list[int]:
    """Direction of each nesting level for ``alpha``, innermost first."""
    return [v for v in order for _ in range(alpha[v])]


def level_plan(exponents: Any) -> tuple[tuple[int, ...], ...]:
    """Directions differentiated at each nesting level (innermost first) to reach every exponent."""
    exponents = tuple(tuple(int(x) for x in e) for e in exponents)
    order = _order(exponents)
    levels = max((sum(e) for e in exponents), default=0)
    plan: list[set[int]] = [set() for _ in range(levels)]
    for e in exponents:
        for i, v in enumerate(_sequence(e, order)):
            plan[i].add(v)
    return tuple(tuple(sorted(s)) for s in plan)


def _nest(prev: Any, dirs: tuple[int, ...], d: int) -> Any:
    """One forward pass of ``prev`` along the basis vectors ``dirs``. The new axis comes first."""
    basis = jnp.eye(d)[jnp.asarray(dirs)]

    def level(y: Any) -> Any:
        def push(t: Any) -> Any:
            _, tangent, history = jax.jvp(prev, (y,), (t,), has_aux=True)
            return tangent, history

        tangents, history = jax.vmap(push, out_axes=(0, None))(basis.astype(y.dtype))
        return tangents, history + (tangents,)

    return level


def taylor_coefficients(f: Any, d: int, exponents: Any) -> Any:
    """``(1/alpha!) d^alpha f(0)`` stacked along a leading axis, ``f: (d,) -> pytree``."""
    exponents = tuple(tuple(int(x) for x in e) for e in exponents)
    plan = level_plan(exponents)
    order = _order(exponents)

    def level0(y: Any) -> Any:
        v = f(y)
        return v, (v,)

    fn = level0
    for dirs in plan:
        fn = _nest(fn, dirs, d)
    tensors = fn(jnp.zeros(d))[1]
    picks = []
    for e in exponents:
        seq = _sequence(e, order)
        index = tuple(plan[i].index(v) for i, v in reversed(list(enumerate(seq))))
        norm = math.prod(math.factorial(r) for r in e)
        picks.append(
            jax.tree.map(lambda a, i=index, s=norm: a[i] / s, tensors[len(seq)])
        )
    return jax.tree.map(lambda *xs: jnp.stack(xs), *picks)


def _point(
    names: tuple[str, ...], charts: tuple[str, ...], ref: Any, hw: Any, z: Any
) -> dict:
    return {
        v: (
            ref[i] * jnp.exp(hw[i] * z[i])
            if charts[i] == "log"
            else ref[i] + hw[i] * z[i]
        )
        for i, v in enumerate(names)
    }


def make_routine(
    names: tuple[str, ...],
    charts: tuple[str, ...],
    exponents: tuple[tuple[int, ...], ...],
) -> JetRoutine:
    """The ``JetRoutine`` of the Taylor columns of one cell on one channel view (decisions 165 and 166).

    Raises ``JetMemoryError`` before tracing when the nesting depth exceeds the order limit.
    """

    def one(averaged: AveragedKernel, channels: Any, ref: Any, hw: Any) -> Any:
        def f(z: Any) -> Any:
            modes = averaged(channels, _point(names, charts, ref, hw, z))
            return modes.I, modes.P[0], modes.V  # one phase-weight row (decision 132)

        return taylor_coefficients(f, len(names), exponents)

    return JetRoutine(names, charts, exponents, one)


def _direct(names: tuple, charts: tuple, exponents: tuple) -> Any:
    """The routine of a provider call outside a basis build, from the process cache."""
    return default_cache().routine(names, charts, exponents, DIRECT, make=make_routine)


def _check(averaged: AveragedKernel, names: tuple[str, ...]) -> None:
    taken = (
        {n for n, _ in averaged.averaged}
        | {n for n, _, _ in averaged.legendre}
        | {n for n, _ in averaged.fixed}
    )
    clash = set(names) & taken
    if clash:
        raise ValueError(
            f"{sorted(clash)} are celled and also averaged, projected or fixed in the averaged kernel"
        )


def _views(channels: Any) -> list[tuple[Any, ...]]:
    """The node-chunk views of each channel, or the channels whole when they have no nodes."""
    if getattr(channels, "nodes", None) is None:
        return [(channels,)]
    return [chunk_views(channels, j) for j in range(int(channels.n_ch))]


def _host(a: Any) -> np.ndarray:
    a = np.asarray(a)
    return a.astype(np.result_type(a.dtype, np.float64), copy=False)


def _cell(
    run: Any, averaged: Any, views: list, ref: Any, hw: Any, calls: dict
) -> tuple:
    """``(I, P, V)`` of one cell, ``(n_rows, n_ch, *L)``: one call per channel chunk, chunks added in order."""
    per_channel = []
    for chunks in views:
        total = None
        for view in chunks:
            out = run(
                averaged,
                view,
                ref,
                hw,
                events=calls["events"],
                request=calls["request"],
            )
            out = tuple(_host(a) for a in out)
            total = out if total is None else tuple(t + o for t, o in zip(total, out))
            calls["n"] += 1
        per_channel.append(total)
    return tuple(np.concatenate([c[k] for c in per_channel], axis=1) for k in range(3))


def _calls(channels: Any, n_cells: int, events: dict | None) -> dict:
    n_ch = int(channels.n_ch) if hasattr(channels, "n_ch") else "?"
    return {"n": 0, "events": events, "request": (n_ch, n_cells)}


def _count(events: dict | None, groups: int, calls: dict) -> None:
    if events is None:
        return
    for name, n in (("jet groups", groups), ("jet calls", calls["n"])):
        events[name] = events.get(name, 0) + n
    for name in JET_EVENTS:
        events.setdefault(name, 0)


def jet_columns(
    averaged: AveragedKernel,
    channels: Any,
    axes: dict[str, tuple[str, float, float]],
    exponents: list[tuple[int, ...]],
    *,
    with_v: bool = False,
    routine: Callable | None = None,
) -> tuple[Any, ...]:
    """``(I, P)`` columns, or ``(I, P, V)`` with ``with_v``, one row per exponent over ``axes``.

    ``axes`` maps each celled variable to ``(chart, reference, half_width)``. Every exponent has one non-negative
    integer per axis.
    """
    names = tuple(axes)
    _check(averaged, names)

    def valid(e: Any) -> bool:
        return (
            isinstance(e, (tuple, list))
            and len(e) == len(names)
            and all(
                isinstance(x, (int, np.integer)) and not isinstance(x, bool) and x >= 0
                for x in e
            )
        )

    if not exponents or not all(valid(e) for e in exponents):
        raise ValueError(
            f"exponents must be a non-empty list of {len(names)} non-negative ints each, one per axis {names}"
        )
    charts = tuple(axes[v][0] for v in names)
    ref = np.asarray([axes[v][1] for v in names], dtype=float)
    hw = np.asarray([axes[v][2] for v in names], dtype=float)
    exps = tuple(tuple(int(x) for x in e) for e in exponents)
    check_order(names, exps)
    run = (routine or _direct)(names, charts, exps)
    out = _cell(run, averaged, _views(channels), ref, hw, _calls(channels, 1, None))
    return out if with_v else out[:2]


def _legendre_order(averaged: AveragedKernel, lattice: Any) -> None:
    wanted = tuple((n, L) for n, L, _ in averaged.legendre)
    if lattice.index.legendre != wanted:
        raise ValueError(
            f"the lattice's Legendre axes {lattice.index.legendre} differ from the averaged kernel's "
            f"{wanted}"
        )


def _rows(columns: Any, n_lead: int) -> np.ndarray:
    """``(n_m, ..., n_ch, *L)`` to lattice order: Legendre pair, then monomial."""
    columns = np.asarray(columns)
    n_l = columns.ndim - n_lead - 1
    if n_l == 0:
        return columns
    moved = np.moveaxis(
        columns, tuple(range(columns.ndim - n_l, columns.ndim)), tuple(range(n_l))
    )
    return moved.reshape((-1,) + moved.shape[n_l + 1 : n_l + 1 + n_lead])


def jet_cell(
    averaged: AveragedKernel,
    channels: Any,
    lattice: Any,
    cell: int,
    *,
    with_v: bool = False,
) -> tuple[Any, ...]:
    """``(I, P)`` columns of one product cell in lattice row order (Legendre pair, then monomial)."""
    _legendre_order(averaged, lattice)
    table, index = lattice.cells, lattice.index
    axes = {
        v: (
            table.chart[cell][i],
            float(table.reference[cell, i]),
            float(table.half_width[cell, i]),
        )
        for i, v in enumerate(table.variables)
    }
    exponents = list(index.patterns[int(index.cell_pattern[cell])])
    out = jet_columns(averaged, channels, axes, exponents, with_v=with_v)
    return tuple(_rows(a, 1) for a in out)


def _selection(table: Any, cells: Any) -> list[int]:
    if cells is None:
        return list(range(table.n_cells))
    chosen = [int(c) for c in cells]
    if not chosen or any(c < 0 or c >= table.n_cells for c in chosen):
        raise ValueError(
            f"cells must name product cells of 0..{table.n_cells - 1}, got {cells!r}"
        )
    return chosen


def _fixed(exponents: Any, names: tuple) -> tuple | None:
    fixed = (
        None
        if exponents is None
        else tuple(tuple(int(x) for x in e) for e in exponents)
    )
    if fixed is not None and (
        not fixed or any(len(e) != len(names) or min(e) < 0 for e in fixed)
    ):
        raise ValueError(
            f"exponents must be a non-empty tuple of {len(names)} non-negative ints each, one per axis {names}"
        )
    return fixed


def jet_lattice(
    averaged: AveragedKernel,
    channels: Any,
    lattice: Any,
    *,
    with_v: bool = False,
    cells: Any = None,
    exponents: Any = None,
    routine: Callable | None = None,
    events: dict | None = None,
) -> tuple[Any, ...]:
    """Columns of product cells in the rows of one block, ``(block_size, n_ch)`` each.

    ``cells`` selects product cells, and the rows of each follow in the order given (repeats allowed, each
    distinct cell computed once). ``exponents`` replaces every selected cell's lower set, so a cell gives
    ``n_pairs * len(exponents)`` rows. Cells of equal (charts, exponents) share one routine,
    ``routine(names, charts, exponents)``, by default the one of ``default_cache()`` with the token "direct".
    Every order is checked before a routine is made. ``events``, when given, gains "jet groups", "jet calls",
    "jet compiles" and "jet memory unchecked".
    """
    _legendre_order(averaged, lattice)
    table, index = lattice.cells, lattice.index
    names = table.variables
    _check(averaged, names)
    chosen = _selection(table, cells)
    fixed = _fixed(exponents, names)
    distinct = list(dict.fromkeys(chosen))
    patterns = {
        c: fixed if fixed is not None else index.patterns[int(index.cell_pattern[c])]
        for c in distinct
    }
    for c in distinct:
        check_order(names, patterns[c], cell=c)
    runs: dict[tuple, Any] = {}
    for c in distinct:
        key = (tuple(table.chart[c]), tuple(patterns[c]))
        if key not in runs:
            runs[key] = (routine or _direct)(names, *key)
    views, calls = _views(channels), _calls(channels, len(distinct), events)
    pieces: dict[int, tuple] = {}
    for c in distinct:
        run = runs[(tuple(table.chart[c]), tuple(patterns[c]))]
        ref = np.asarray(table.reference[c], dtype=float)
        hw = np.asarray(table.half_width[c], dtype=float)
        pieces[c] = tuple(
            _rows(a, 1) for a in _cell(run, averaged, views, ref, hw, calls)
        )
    _count(events, len(runs), calls)
    out = tuple(np.concatenate([pieces[c][k] for c in chosen]) for k in range(3))
    return out if with_v else out[:2]


def cost(exponents: Any) -> int:
    """Primal evaluations per cell of the nested plan, ``prod_i (1 + |D_i|)``."""
    return int(np.prod([1 + len(dirs) for dirs in level_plan(exponents)]))
