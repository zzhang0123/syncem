"""The jet provider: Taylor columns of the averaged kernel in any number of celled variables (SPEC Sec. 8.3).

LABEL ``eq: channel basis expansion``. A product cell with celled variables ``v_1 .. v_d`` maps the chart
coordinates ``z`` to the point ``x_v = x_b e^(h z_v)`` (log) or ``x_b + h z_v`` (affine) and evaluates the
averaged kernel there. The columns are ``(1/alpha!) d^alpha f(0)`` for every exponent ``alpha`` of the cell's
lower set, by nested forward mode. Each nesting level differentiates only the directions that some exponent
needs at that depth (cap-aware nesting, the ``level_plan`` of ``syncem.model._harmonic_taylor`` generalised to
``d`` directions): an exponent's derivatives are taken innermost for the variables with the smallest caps, so a
variable of cap ``c`` appears only in the levels it can reach. Cells with equal charts and lower sets share one
compiled routine, mapped over the cells with ``lax.map``.

The cost per cell is about ``prod_i (1 + |D_i|)`` primal evaluations over the levels ``i``, with ``D_i`` the
directions of level ``i``. The plan prints it (SPEC Sec. 8.3, decision (b)). The columns carry the units of the kernel.
Shapes: ``I`` and ``P`` ``(n_rows, n_ch, *L)``, one row per exponent (``P`` complex, one phase-weight row). Not certified:
nothing beyond the kernel and its averages, since forward mode gives the derivatives to rounding.
"""

from __future__ import annotations

import functools
import math
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .average import AveragedKernel

__all__ = [
    "cost",
    "jet_cell",
    "jet_columns",
    "jet_lattice",
    "level_plan",
    "taylor_coefficients",
]


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


@functools.lru_cache(maxsize=64)
def _routine(
    names: tuple[str, ...],
    charts: tuple[str, ...],
    exponents: tuple[tuple[int, ...], ...],
) -> Any:
    @eqx.filter_jit
    def run(averaged: AveragedKernel, channels: Any, refs: Any, hws: Any) -> Any:
        def one(args: Any) -> Any:
            ref, hw = args

            def f(z: Any) -> Any:
                modes = averaged(channels, _point(names, charts, ref, hw, z))
                return (
                    modes.I,
                    modes.P[0],
                    modes.V,
                )  # one phase-weight row (decision 132)

            return taylor_coefficients(f, len(names), exponents)

        return jax.lax.map(one, (refs, hws))

    return run


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


def jet_columns(
    averaged: AveragedKernel,
    channels: Any,
    axes: dict[str, tuple[str, float, float]],
    exponents: list[tuple[int, ...]],
    *,
    with_v: bool = False,
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
    refs = jnp.asarray([[axes[v][1] for v in names]], dtype=float)
    hws = jnp.asarray([[axes[v][2] for v in names]], dtype=float)
    exps = tuple(tuple(int(x) for x in e) for e in exponents)
    i, p, v = _routine(names, charts, exps)(averaged, channels, refs, hws)
    return (i[0], p[0], v[0]) if with_v else (i[0], p[0])


def _legendre_order(averaged: AveragedKernel, lattice: Any) -> None:
    wanted = tuple((n, L) for n, L, _ in averaged.legendre)
    if lattice.index.legendre != wanted:
        raise ValueError(
            f"the lattice's Legendre axes {lattice.index.legendre} differ from the averaged kernel's "
            f"{wanted}"
        )


def _rows(columns: Any, n_lead: int) -> Any:
    """``(n_m, ..., n_ch, *L)`` to lattice order: Legendre pair, then monomial."""
    n_l = columns.ndim - n_lead - 1
    if n_l == 0:
        return columns
    moved = jnp.moveaxis(
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


def jet_lattice(
    averaged: AveragedKernel, channels: Any, lattice: Any, *, with_v: bool = False
) -> tuple[Any, ...]:
    """Columns of every product cell in the rows of one block, ``(block_size, n_ch)`` each."""
    _legendre_order(averaged, lattice)
    table, index = lattice.cells, lattice.index
    names = table.variables
    _check(averaged, names)
    groups: dict[tuple, list[int]] = {}
    for c in range(table.n_cells):
        key = (table.chart[c], index.patterns[int(index.cell_pattern[c])])
        groups.setdefault(key, []).append(c)
    pieces_i: list[Any] = [None] * table.n_cells
    pieces_p: list[Any] = [None] * table.n_cells
    pieces_v: list[Any] = [None] * table.n_cells
    for (charts, pattern), cells in groups.items():
        run = _routine(names, tuple(charts), tuple(pattern))
        refs = jnp.asarray(table.reference[cells], dtype=float)
        hws = jnp.asarray(table.half_width[cells], dtype=float)
        i, p, v = run(averaged, channels, refs, hws)
        for k, c in enumerate(cells):
            pieces_i[c], pieces_p[c], pieces_v[c] = (
                _rows(i[k], 1),
                _rows(p[k], 1),
                _rows(v[k], 1),
            )
    out = (jnp.concatenate(pieces_i), jnp.concatenate(pieces_p))
    return out + (jnp.concatenate(pieces_v),) if with_v else out


def cost(exponents: Any) -> int:
    """Primal evaluations per cell of the nested plan, ``prod_i (1 + |D_i|)``."""
    return int(np.prod([1 + len(dirs) for dirs in level_plan(exponents)]))
