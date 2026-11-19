"""Per-cell routing of the step-5 basis build between the stack, the jet and the core (design/0.8/SPEC.md decision 150).

LABEL ``eq: channel basis expansion``. ``routes="auto"`` asks ``stack_precondition`` first. A reason sends the cell to
the jet with that reason. Otherwise the stack runs with one memo shared across cells, and a ``StackRefused`` sends the
cell to the jet with ``str(err)``. A plain ``ValueError`` is a user error and propagates. A provider name, or a tuple
or mapping of names per cell, routes explicitly and raises on refusal. The core provider is explicit only. Jet cells
run in one ``jet_lattice`` call, batched by (charts, pattern) and padded to powers of two.

``stack_precondition`` is a necessary static check: the kernel type, the celled set, declared ``NodeRule`` parents,
the Legendre axes and an order of at most the S1 table order. ``None`` does not promise that the stack accepts the
cell. Grid reach at the lowest channel, an undeclared read, a missing or non-positive density, the amplification
bound, the frozen-node check and a field law with all its weight at ``B = 0`` refuse at evaluation time.

Units: those of the kernel. Shapes: ``I`` and ``V`` ``(block_size, n_ch)`` float, ``P`` ``(block_size, n_ch)`` complex,
rows in the lattice order of one block, zero for cells not routed. Not certified: the providers, which are compared
with each other by the tests of decision 150 and by check (a) of Sec. 8.3.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np

from ..base.capabilities import Similarity
from ..base.continuum import ContinuumKernel
from ..records.records import RouteRecord
from ._stack_checks import StackRefused
from ._stack_series import stack_cell
from .average import AveragedKernel, NodeRule
from .field_average import TABLE_ORDER
from .provider_core import core_cell
from .provider_jet import _padded, jet_lattice

__all__ = ["PROVIDERS", "check_stokes", "route_cells", "stack_precondition"]

PROVIDERS = ("stack", "jet", "core")
_STACK_SET = (
    "gamma",
    "B",
)  # the celled variables the stack takes (_stack_checks._check_lattice)
_BLOCKS = ("I", "P", "V")
_BLOCK_OF = {"I": "I", "P": "P", "Q": "P", "U": "P", "V": "V"}
_EVENTS = ("stack cells", "jet cells", "core cells", "jet groups", "jet padded cells")


def check_stokes(kernel: Any, stokes: Any) -> tuple[str, ...]:
    """The Stokes blocks of a request in "IPV" order. A block outside ``kernel.stokes`` raises ``ValueError``.

    ``stokes`` is a string such as "IQU" or a sequence of distinct names of I, P, Q, U and V. Q and U come
    together from the P block.
    """
    names = tuple(stokes)
    if not names or len(set(names)) != len(names) or not set(names) <= set(_BLOCK_OF):
        raise ValueError(
            f"stokes must hold distinct names of I, P, Q, U, V, got {stokes!r}"
        )
    if ("Q" in names) != ("U" in names):
        raise ValueError("Q and U come together from the P block. Request both, or P")
    wanted = {_BLOCK_OF[n] for n in names}
    blocks = tuple(b for b in _BLOCKS if b in wanted)
    have = tuple(getattr(kernel, "stokes", ()))
    missing = [b for b in blocks if b not in have]
    if missing:
        name = getattr(kernel, "name", type(kernel).__name__)
        raise ValueError(
            f"the kernel {name} has the Stokes blocks {have}, and {missing} were requested"
        )
    return blocks


def _check_clash(averaged: AveragedKernel, names: tuple[str, ...]) -> None:
    taken = (
        {n for n, _ in averaged.averaged}
        | {n for n, _, _ in averaged.legendre}
        | {n for n, _ in averaged.fixed}
    )
    if set(names) & taken:
        raise ValueError(
            f"{sorted(set(names) & taken)} are celled and also averaged, projected or fixed in the averaged kernel"
        )


def _similarity(kernel: Any) -> tuple[str, ...]:
    found = [
        c.variables
        for c in getattr(kernel, "capabilities", ())
        if isinstance(c, Similarity)
    ]
    return tuple(v for vs in found for v in vs)


def _law_reason(averaged: AveragedKernel) -> str | None:
    for name, rule in averaged.averaged:
        if not isinstance(rule, NodeRule):
            return f"the law of {name} is a bare function: declare it as a NodeRule with its parents"
        if rule.parents is None:
            return f"the law of {name} has undeclared parents: only the jet differentiates it"
    for name, _, _ in averaged.legendre:
        if name != "eta":
            return f"a Legendre axis on {name}: the stack projects only eta"
    return None


def stack_precondition(averaged: AveragedKernel, lattice: Any, cell: int) -> str | None:
    """The reason the stack cannot take ``cell``, or ``None``. A necessary static check only (decision 150)."""
    names = lattice.cells.variables
    _check_clash(averaged, names)
    kernel, similar = averaged.kernel, _similarity(averaged.kernel)
    if type(kernel) is not ContinuumKernel or _STACK_SET[0] not in similar:
        return f"the stack holds the continuum F and G only, not {type(kernel).__name__}: the jet takes it"
    averaged_names = {n for n, _ in averaged.averaged}
    loose = [v for v in similar if v not in names and v in averaged_names]
    if loose:
        return f"{loose[0]} averaged: Similarity variable not celled -> jet"
    if _STACK_SET[0] not in names or not set(names) <= set(_STACK_SET):
        return f"celled variables {names}: the stack takes gamma and optionally B. The jet takes the cell"
    reason = _law_reason(averaged)
    if reason is not None:
        return reason
    pattern = lattice.index.patterns[int(lattice.index.cell_pattern[cell])]
    order = max(sum(e) for e in pattern)
    if order > TABLE_ORDER:
        return f"order {order} exceeds the S1 table order {TABLE_ORDER}: the jet takes the cell"
    return None


def _chosen(n_cells: int, cells: Any) -> tuple[int, ...]:
    if cells is None:
        return tuple(range(n_cells))
    chosen = tuple(int(c) for c in cells)
    if (
        not chosen
        or len(set(chosen)) != len(chosen)
        or not all(0 <= c < n_cells for c in chosen)
    ):
        raise ValueError(
            f"cells must be distinct product cells of 0..{n_cells - 1}, got {cells!r}"
        )
    return chosen


def _plan(routes: Any, n_cells: int, chosen: tuple[int, ...]) -> dict[int, str | None]:
    """Provider per chosen cell, ``None`` for auto."""
    usage = f"routes must be 'auto', one of {PROVIDERS}, or a tuple or mapping of them per cell"
    if isinstance(routes, str):
        if routes != "auto" and routes not in PROVIDERS:
            raise ValueError(f"{usage}, got {routes!r}")
        return {c: None if routes == "auto" else routes for c in chosen}
    if isinstance(routes, Mapping):
        keys = list(routes)
        if any(
            isinstance(k, bool) or not isinstance(k, (int, np.integer)) for k in keys
        ):
            raise ValueError(f"{usage}. Mapping keys are product cells, got {keys!r}")
        table = {int(k): v for k, v in routes.items()}
        if not set(table) <= set(range(n_cells)) or not set(chosen) <= set(table):
            raise ValueError(
                f"{usage}. The mapping must cover cells {chosen} within 0..{n_cells - 1}"
            )
    elif isinstance(routes, (tuple, list)):
        if len(routes) != n_cells:
            raise ValueError(
                f"{usage}. A tuple needs one entry per cell ({n_cells}), got {len(routes)}"
            )
        table = dict(enumerate(routes))
    else:
        raise ValueError(f"{usage}, got {routes!r}")
    if not all(isinstance(table[c], str) and table[c] in PROVIDERS for c in chosen):
        raise ValueError(f"{usage}, got {routes!r}")
    return {c: table[c] for c in chosen}


def _jet_groups(lattice: Any, cells: list[int]) -> tuple[int, int]:
    """Number of jet groups and padded cells of ``jet_lattice`` on ``cells``."""
    groups: dict[tuple, int] = {}
    for c in cells:
        key = (
            lattice.cells.chart[c],
            lattice.index.patterns[int(lattice.index.cell_pattern[c])],
        )
        groups[key] = groups.get(key, 0) + 1
    return len(groups), sum(len(_padded([0] * n)) for n in groups.values())


def _assemble(index: Any, pieces: dict, with_v: bool) -> tuple[Any, Any, Any]:
    n_ch = next(iter(pieces.values()))[0].shape[1]
    out_i = np.zeros((index.block_size, n_ch))
    out_p = np.zeros((index.block_size, n_ch), complex)
    out_v = np.zeros((index.block_size, n_ch)) if with_v else None
    for c, (i, p, v) in pieces.items():
        rows = index.cell_rows(c)
        span = slice(rows.start, rows.stop)
        out_i[span], out_p[span] = i, p
        if with_v:
            out_v[span] = v
    return out_i, out_p, out_v


def _run_jet(
    averaged, channels, lattice, jet: dict, with_v: bool, pieces: dict
) -> None:
    cells = list(jet)
    out = [
        np.asarray(a)
        for a in jet_lattice(averaged, channels, lattice, with_v=with_v, cells=cells)
    ]
    start = 0
    for c in cells:
        n = len(lattice.index.cell_rows(c))
        rows = slice(start, start + n)
        pieces[c] = (out[0][rows], out[1][rows], out[2][rows] if with_v else None)
        start += n


def _count(events: dict | None, lattice: Any, records: dict, jet: dict) -> None:
    """Add the route counts of one call to ``events``."""
    if events is None:
        return
    providers = [r.provider for r in records.values()]
    groups, padded = _jet_groups(lattice, list(jet)) if jet else (0, 0)
    counts = (*(providers.count(p) for p in PROVIDERS), groups, padded)
    for key, n in zip(_EVENTS, counts):
        events[key] = events.get(key, 0) + n


def route_cells(
    averaged: AveragedKernel,
    channels: Any,
    lattice: Any,
    routes: Any = "auto",
    *,
    s1: str = "lookup",
    with_v: bool = False,
    cells: Any = None,
    memo: dict | None = None,
    events: dict | None = None,
) -> tuple[Any, Any, Any, tuple[RouteRecord, ...]]:
    """``(I, P, V or None, records)`` of the chosen cells, one ``RouteRecord`` per cell in the order of ``cells``.

    ``memo`` is shared by the stack cells (S1 tables). ``events``, when given, gains the counts "stack cells",
    "jet cells", "core cells", "jet groups" and "jet padded cells".
    """
    if s1 not in ("lookup", "quadrature"):
        raise ValueError(f"s1 must be 'lookup' or 'quadrature', got {s1!r}")
    _check_clash(averaged, lattice.cells.variables)
    if with_v:
        check_stokes(averaged.kernel, "V")
    chosen = _chosen(lattice.n_cells, cells)
    plan = _plan(routes, lattice.n_cells, chosen)
    if with_v and {"stack", "core"} & set(plan.values()):
        raise ValueError("the stack and the core give no V. Route the cells to the jet")
    memo = {} if memo is None else memo
    pieces: dict[int, tuple] = {}
    records: dict[int, RouteRecord] = {}
    jet: dict[int, str | None] = {}
    for c in chosen:
        _route_one(
            averaged, channels, lattice, c, plan[c], s1, memo, pieces, records, jet
        )
    if jet:
        _run_jet(averaged, channels, lattice, jet, with_v, pieces)
        records.update({c: RouteRecord(c, "jet", reason=r) for c, r in jet.items()})
    _count(events, lattice, records, jet)
    return (
        *_assemble(lattice.index, pieces, with_v),
        tuple(records[c] for c in chosen),
    )


def _route_one(
    averaged, channels, lattice, c, provider, s1, memo, pieces, records, jet
) -> None:
    """Route one cell. Jet cells are collected in ``jet`` with their reason and run together."""
    if provider == "core":
        i, p = core_cell(averaged, channels, lattice, c)
        pieces[c], records[c] = (np.asarray(i), np.asarray(p), None), RouteRecord(
            c, "core"
        )
        return
    if provider == "jet":
        jet[c] = None
        return
    reason = None if provider == "stack" else stack_precondition(averaged, lattice, c)
    if reason is None:
        mode: list = []
        try:
            i, p = stack_cell(
                averaged, channels, lattice, c, s1=s1, memo=memo, record=mode
            )
        except StackRefused as err:
            if provider == "stack":
                raise
            reason = str(err)
        else:
            pieces[c] = (np.asarray(i), np.asarray(p), None)
            records[c] = RouteRecord(c, "stack", mode[0], s1)
            return
    jet[c] = reason
