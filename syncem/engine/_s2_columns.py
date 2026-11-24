"""Per-column reuse of S2 entries in a basis build (design/0.8/SPEC.md decision 159 part 3, Sec. 8.2).

LABEL ``eq: channel basis expansion``. ``reuse_cells`` is the routing of a build with keys. It routes each
product cell as ``route_cells`` does, reads the S2 store per (cell, block, channel) context key and calls the
providers for the columns that the store lacks. A stack cell is routed by a dry walk (``stack_stage``), which
reads the S1 tables of the stack walk and gives its mode and S1 keys, the stage of the context key, without a
channel series. A jet or core cell has the mode "none" and the kernel record as its stage. A column is one
(Legendre degrees, exponent) of a cell. It is reused when the store holds it under the context key and the
stored rule chain agrees with the cell chain up to its total degree (``Cache.columns``).

The providers run on the channels that lack a column, once per distinct context key. The jet takes the
exponents that lack a column, in the order of the cell's lower set, and gives all Legendre degrees of them. The
stack and the core give the whole cell. The build keeps from a provider call only the values that the store
lacks, so a reused value equals the stored one bit for bit, and it stores them with the cell chain. "S2 columns
computed" counts those values, one per (row, context key, block), and "S2 columns reused" counts the values
read. A stack cell that refuses the channels that lack a column takes the jet on every channel, as in ``at``.
A dry walk that reads other S1 tables than the stack walk raises ``RuntimeError``.

The averaged-kernel record of the context key is ``_keys.column_record``, which holds each Legendre axis by
name and node count. ``L`` enters through the degrees of the column id.

Units: those of the kernel per unit scaled moment. Shapes: ``I`` and ``V`` ``(block_size, n_ch)`` float and
``P`` ``(block_size, n_ch)`` complex, as ``route_cells``. Not certified: a key is as complete as the
fingerprints it hashes (decision 157), and the providers, which the tests compare with each other.
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..base._fingerprint import digest
from ..records.records import RouteRecord
from . import _routes
from ._keys import column_record, s2_context_keys
from ._stack_checks import StackRefused
from ._jet_routine import NESTED_MAX_ORDER, JetMemoryError, check_order

__all__ = ["Reuse", "cell_chain", "column_ids", "reuse_cells"]

_DTYPE = {"I": float, "P": complex, "V": float}
_TALLY = ("S2 columns computed", "S2 columns reused", "S2 columns stored")


@dataclasses.dataclass(frozen=True)
class Reuse:
    """What a build with keys gives ``reuse_cells``: the cache, the ``BuildKeys``, the ``StageMemo`` and the jet."""

    cache: Any
    keys: Any
    memo: Any
    jet: Any


@dataclasses.dataclass
class _Cell:
    """One routed cell: its route, context keys per block, found masks ``(n_ch, n_rows)`` and values."""

    cell: int
    provider: str
    mode: str | None
    reason: str | None
    stage: Any
    context: dict = dataclasses.field(default_factory=dict)
    found: dict = dataclasses.field(default_factory=dict)
    values: dict = dataclasses.field(default_factory=dict)

    def missing(self) -> np.ndarray:
        """``(n_ch, n_rows)``: a value of some block is not stored."""
        return np.any([~f for f in self.found.values()], axis=0)

    def first(self) -> tuple[np.ndarray, np.ndarray]:
        """The first channel of each distinct context key, and that channel for every channel."""
        keys = next(iter(self.context.values()))
        _, first, inverse = np.unique(keys, return_index=True, return_inverse=True)
        return np.sort(first), first[inverse.reshape(-1)]

    def record(self, s1: str) -> RouteRecord:
        if self.provider == "stack":
            return RouteRecord(self.cell, "stack", self.mode, s1)
        if self.provider == "jet":
            return RouteRecord(self.cell, "jet", reason=self.reason)
        return RouteRecord(self.cell, "core")


def column_ids(lattice: Any) -> list[tuple]:
    """``(Legendre degrees, exponent)`` of every row of one block."""
    a = lattice.index.arrays
    return [
        (tuple(int(x) for x in a.degrees[k]), tuple(int(x) for x in a.powers[k]))
        for k in range(lattice.index.block_size)
    ]


def cell_chain(keys: Any, cell: int) -> tuple[str, ...]:
    """The digest per degree of the chains of every rule on ``cell``."""
    names = sorted(keys.chains)
    return tuple(
        digest(("cell chain", n, tuple((m, keys.chains[m][cell][n]) for m in names)))
        for n in range(keys.orders[cell] + 1)
    )


def _subset(channels: Any, rows: np.ndarray) -> Any:
    return (
        channels if np.array_equal(rows, np.arange(channels.n_ch)) else channels[rows]
    )


# ------------------------------------------------------------------ routing and lookup


def _route(reuse, averaged, channels, lattice, c, provider, s1) -> tuple:
    """``(provider, mode, reason, stage)`` of cell ``c``, with no channel series."""
    if provider in ("jet", "core"):
        return provider, None, None, reuse.keys.kernel
    reason = (
        None
        if provider == "stack"
        else _routes.stack_precondition(averaged, lattice, c)
    )
    if reason is None:
        reuse.memo.take_reads()
        try:
            mode = _routes.stack_stage(
                averaged, channels, lattice, c, s1=s1, memo=reuse.memo
            )
        except StackRefused as err:
            if provider == "stack":
                raise
            reason = str(err)
        else:
            return "stack", mode, None, reuse.memo.take_reads()
    return "jet", None, reason, reuse.keys.kernel


def _keyed(reuse, lattice, c, route, blocks, shared) -> _Cell:
    """The cell with its context keys, the stored values and their found masks."""
    ids, record, n_ch = shared
    keys, span = reuse.keys, lattice.index.cell_rows(c)
    cell, mine = _Cell(c, *route), ids[span.start : span.stop]
    stage = cell.stage
    if cell.provider == "jet":
        index = lattice.index
        pattern = tuple(index.patterns[int(index.cell_pattern[c])])
        try:
            order = check_order(lattice.cells.variables, pattern, c)
        except JetMemoryError as err:
            raise _routes.with_reason(err, lattice, {c: cell.reason}) from None
        if order > NESTED_MAX_ORDER:
            stage = ("Taylor mode", stage, pattern)
    chain = cell_chain(keys, c)
    for block in blocks:
        context = s2_context_keys(
            provider=cell.provider,
            mode=cell.mode or "none",
            stage=stage,
            block=block,
            row_keys=keys.row_keys,
            phase=keys.phase,
            averaged=record,
            d0={m: keys.chains[m][c][0] for m in keys.chains},
            geometry=keys.geometry[c],
            rules=keys.rules,
        )
        if context is None:
            raise RuntimeError(f"cell {c} has no S2 context key in a build with keys")
        found, stored = reuse.cache.columns(context, mine, [chain] * n_ch)
        values = np.zeros((len(mine), n_ch), _DTYPE[block])
        if stored is not None:
            values[found.T] = stored.transpose(1, 0)[found.T]
        cell.context[block], cell.found[block], cell.values[block] = (
            context,
            found,
            values,
        )
    return cell


# ------------------------------------------------------------------ providers


def _reps(cell: _Cell) -> np.ndarray:
    """The first channel of each distinct context key that lacks a value."""
    first, _ = cell.first()
    return first[cell.missing()[first].any(axis=1)]


def _fill(cell: _Cell, reps: np.ndarray, out: dict, positions: np.ndarray) -> None:
    """Write the provider values ``out[block]`` ``(len(positions), len(reps))`` that the store lacks."""
    for block, part in out.items():
        lack = ~cell.found[block][
            np.ix_(reps, positions)
        ]  # (len(reps), len(positions))
        target = cell.values[block]
        for m, j in enumerate(reps):
            target[positions[lack[m]], j] = np.asarray(part)[lack[m], m]


def _stack(reuse, averaged, channels, lattice, cell: _Cell, s1, plan) -> str | None:
    """Compute the stack values of ``cell`` that the store lacks. The reason when the stack refuses them."""
    reps = _reps(cell)
    if reps.size == 0:
        return None
    mode: list = []
    reuse.memo.take_reads()
    try:
        i, p = _routes.stack_cell(
            averaged,
            _subset(channels, reps),
            lattice,
            cell.cell,
            s1=s1,
            memo=reuse.memo,
            record=mode,
        )
    except StackRefused as err:
        if plan == "stack":
            raise
        return str(err)
    if mode != [cell.mode] or reuse.memo.take_reads() != cell.stage:
        raise RuntimeError(
            f"the stack walk of cell {cell.cell} read other S1 tables than its dry walk"
        )
    n_rows = next(iter(cell.values.values())).shape[0]
    _fill(cell, reps, {"I": i, "P": p}, np.arange(n_rows))
    return None


def _core(averaged, channels, lattice, cell: _Cell) -> None:
    reps = _reps(cell)
    if reps.size:
        i, p = _routes.core_cell(averaged, _subset(channels, reps), lattice, cell.cell)
        _fill(cell, reps, {"I": i, "P": p}, np.arange(cell.values["I"].shape[0]))


def _exponents(lattice: Any, cell: _Cell, reps: np.ndarray) -> tuple | None:
    """The exponents of the lower set with a missing value, or ``None`` when that is all of them."""
    index = lattice.index
    pattern = tuple(index.patterns[int(index.cell_pattern[cell.cell])])
    if max(map(sum, pattern)) > NESTED_MAX_ORDER:
        return None  # keep the reconstruction independent of which rows are missing
    lack = cell.missing()[reps].any(axis=0)
    rows = index.cell_rows(cell.cell)
    powers = {
        tuple(int(x) for x in index.arrays.powers[rows.start + k])
        for k in np.flatnonzero(lack)
    }
    chosen = tuple(e for e in pattern if e in powers)
    return None if chosen == pattern else chosen


def _positions(lattice: Any, c: int, exponents: tuple | None) -> np.ndarray:
    """Cell rows of the jet rows of ``exponents`` (Legendre pair, then monomial)."""
    index = lattice.index
    pattern = list(index.patterns[int(index.cell_pattern[c])])
    if exponents is None:
        return np.arange(index.n_pairs * len(pattern))
    return np.array(
        [
            p * len(pattern) + pattern.index(e)
            for p in range(index.n_pairs)
            for e in exponents
        ],
        dtype=np.int64,
    )


def _jet(reuse, channels, lattice, cells: list, with_v: bool, counts: dict) -> None:
    """Compute the jet values the store lacks, one ``jet_lattice`` call per (channels, exponents) group."""
    groups: dict[tuple, list[_Cell]] = {}
    for cell in cells:
        reps = _reps(cell)
        if reps.size:
            key = (tuple(int(j) for j in reps), _exponents(lattice, cell, reps))
            groups.setdefault(key, []).append(cell)
    for (reps, exponents), members in groups.items():
        rows = np.array(reps, dtype=np.int64)
        try:
            out = _routes.jet_lattice(
                reuse.jet.averaged,
                _subset(channels, rows),
                lattice,
                with_v=with_v,
                cells=[m.cell for m in members],
                exponents=exponents,
                routine=reuse.jet.routine,
                events=counts,
            )
        except JetMemoryError as err:
            reasons = {m.cell: m.reason for m in members}
            raise _routes.with_reason(err, lattice, reasons) from None
        arrays, start = [np.asarray(a) for a in out], 0
        for m in members:
            positions = _positions(lattice, m.cell, exponents)
            span = slice(start, start + positions.size)
            part = dict(zip(("I", "P", "V"), (a[span] for a in arrays)))
            _fill(m, rows, part, positions)
            start += positions.size


# ------------------------------------------------------------------ storing and the entry


def _store(reuse, lattice, cell: _Cell, ids: list) -> tuple[int, int]:
    """Put the values computed for ``cell``. Returns (values computed, values reused)."""
    first, again = cell.first()
    span, chain = lattice.index.cell_rows(cell.cell), cell_chain(reuse.keys, cell.cell)
    mine, computed, reused = ids[span.start : span.stop], 0, 0
    for block, context in cell.context.items():
        lack = ~cell.found[block][first]  # (n_first, n_rows)
        reused += int((~lack).sum())
        by_mask: dict[bytes, list[int]] = {}
        for m in np.flatnonzero(lack.any(axis=1)):
            by_mask.setdefault(lack[m].tobytes(), []).append(int(m))
        for members in by_mask.values():
            where = np.flatnonzero(lack[members[0]])
            chans = first[members]
            values = cell.values[block][np.ix_(where, chans)].T
            reuse.cache.put_columns(
                context[chans], [mine[k] for k in where], values, [chain] * len(chans)
            )
            computed += where.size * len(chans)
        cell.values[block] = cell.values[block][:, again]
    return computed, reused


def reuse_cells(
    reuse: Reuse,
    averaged: Any,
    channels: Any,
    lattice: Any,
    routes: Any = "auto",
    *,
    s1: str = "lookup",
    with_v: bool = False,
    cells: Any = None,
    events: dict | None = None,
) -> tuple[Any, Any, Any, tuple[RouteRecord, ...]]:
    """``route_cells`` through the S2 store of ``reuse``. ``events`` gains the S2 and route counts."""
    chosen, plan = _routes.prepare(averaged, lattice, routes, s1, with_v, cells)
    blocks = ("I", "P", "V") if with_v else ("I", "P")
    ids = column_ids(lattice)
    shared = (ids, column_record(averaged), channels.n_ch)
    found = {}
    for c in chosen:
        route = _route(reuse, averaged, channels, lattice, c, plan[c], s1)
        found[c] = _keyed(reuse, lattice, c, route, blocks, shared)
    counts = dict.fromkeys(_TALLY, 0) if events is None else events
    for c, cell in found.items():
        if cell.provider == "core":
            _core(averaged, channels, lattice, cell)
        elif cell.provider == "stack":
            why = _stack(reuse, averaged, channels, lattice, cell, s1, plan[c])
            if why is not None:
                route = ("jet", None, why, reuse.keys.kernel)
                found[c] = _keyed(reuse, lattice, c, route, blocks, shared)
    jets = [cell for cell in found.values() if cell.provider == "jet"]
    _jet(reuse, channels, lattice, jets, with_v, counts)
    _tally(reuse, lattice, found, ids, counts)
    pieces = {
        c: tuple(cell.values.get(b) for b in ("I", "P", "V"))
        for c, cell in found.items()
    }
    return (
        *_routes._assemble(lattice.index, pieces, with_v),
        tuple(found[c].record(s1) for c in chosen),
    )


def _tally(reuse, lattice, found: dict, ids: list, counts: dict) -> None:
    """Store the computed values and add the S2 and route counts."""
    for name in _TALLY:
        counts.setdefault(name, 0)
    for cell in found.values():
        computed, reused = _store(reuse, lattice, cell, ids)
        counts["S2 columns computed"] += computed
        counts["S2 columns stored"] += computed
        counts["S2 columns reused"] += reused
    providers = [cell.provider for cell in found.values()]
    for p in _routes.PROVIDERS:
        counts[f"{p} cells"] = counts.get(f"{p} cells", 0) + providers.count(p)
    for name in _routes.JET_EVENTS:
        counts.setdefault(name, 0)
