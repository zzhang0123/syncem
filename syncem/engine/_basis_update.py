"""Channel and order updates of a step-5 spectral basis: the helpers of ``at`` and ``with_order`` (decision 153).

LABEL ``eq: channel basis expansion``. Private helpers of ``syncem.engine.basis``. ``match_channels`` pairs new
channels with stored rows by ``Channels.row_keys``. ``fill_new`` computes the rows of new channels with the
recorded route of each cell, and a stack cell that the new channels refuse takes the jet on every channel. A
kernel guard failure on new channels becomes a ``ValueError`` naming ``ContinuumKernel.resolved``.
``capped_lattice`` caps each per-cell order of the variables inside the total at ``N'`` and sets the total to
``N'``, so a lattice with uneven axis or cell orders keeps a subset of its rows. ``order_indices`` maps the stored
spectra of a lower-order column map to those of the built basis through the (cell, pair, powers) of each row,
without a provider call. The new rows of a cached parent go through the S2 store (``reuse_cells`` with the keys
of ``_basis_cache.derived_reuse``), and those of any other parent through ``route_cells`` with fresh jet
routines. ``raised_routes`` gives ``with_order`` above the built order the parent's routes. The build argument
checks ``check_arguments``, ``stokes_blocks`` and ``check_factor`` live here too. A basis from ``at`` or
from ``with_order`` at or below the built order has the key ``None``: a fresh build need not equal it bitwise
(decision 153), so it is not a memo entry. ``build_record`` and ``derived_record`` write the
records of a build and of an update.

Units: those of the kernel per unit scaled moment. Shapes: ``s0`` ``(n_ch, n_s0, n0)``, ``s2`` ``(n_ch, n2)``,
provider rows ``(block_size, n_ch)``, index arrays ``(n0,)`` and ``(n2,)``. Not certified: the providers. The
updates are exact selection of stored spectra and provider calls on the new channels.
"""

from __future__ import annotations

import dataclasses
import functools
import time
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..lattice._lattice_parts import table_with
from ..lattice.lattice import Lattice
from ..records.records import BuildRecord, RouteRecord, load_average
from ..lattice.representations import AzimuthHarmonic
from ._basis_cache import derived_reuse, fresh_jet
from ._routes import check_stokes, route_cells
from ._s2_columns import reuse_cells
from ._stack_checks import StackRefused
from ._stage_memo import close_memos, local_memo
from .assemble import gather_spectra, stokes_axes
from .columns import ColumnRules

__all__ = [
    "DERIVED",
    "S2_EVENTS",
    "build_record",
    "capped_lattice",
    "check_arguments",
    "check_factor",
    "derived_record",
    "fill_new",
    "match_channels",
    "order_indices",
    "raised_routes",
    "record_keys",
    "record_sizes",
    "stokes_blocks",
    "uncached_record",
]

DERIVED = "not cached: a basis from at() or with_order() is not a memo entry"
S2_EVENTS = ("S2 columns computed", "S2 columns reused", "S2 columns stored")

_RESOLVE = (
    "the new channels reach below the kernel's x_min guard. Resolve the kernel for these channels with "
    "ContinuumKernel.resolved and build again"
)


# ------------------------------------------------------------------ at


def match_channels(
    stored: tuple[str, ...], wanted: tuple[str, ...]
) -> tuple[np.ndarray, dict]:
    """Stored channel per wanted channel (-1 when new), and the first position of each new key."""
    first: dict[str, int] = {}
    for j, k in enumerate(stored):
        first.setdefault(k, j)
    found = np.array([first.get(k, -1) for k in wanted], dtype=np.int64)
    compute: dict[str, int] = {}
    for j in np.flatnonzero(found < 0):
        compute.setdefault(wanted[j], int(j))
    return found, compute


def _guarded(provide: Callable, *args: Any, **kwargs: Any) -> tuple:
    """``provide`` with a kernel guard failure turned into a ``ValueError`` naming ``resolved``."""
    try:
        return provide(*args, **kwargs)
    except RuntimeError as err:
        if "x_min" not in str(err):
            raise
        raise ValueError(_RESOLVE) from err


def _summed(parts: list, shape: tuple, with_v: bool) -> tuple:
    """Block-sized ``(I, P, V)`` from ``route_cells`` outputs on disjoint cells."""
    i, p = np.zeros(shape), np.zeros(shape, complex)
    v = np.zeros(shape) if with_v else None
    for part in parts:
        i, p = i + part[0], p + part[1]
        if with_v:
            v = v + part[2]
    return i, p, v


def _plain(*args: Any, memo: Any, jet: Any, events: dict, **kwargs: Any) -> tuple:
    """``route_cells`` with every value of its cells counted as computed."""
    out = route_cells(*args, memo=memo, jet=jet, events=events, **kwargs)
    lattice, n_ch = args[2], args[1].n_ch
    rows = sum(len(lattice.index.cell_rows(r.cell)) for r in out[3])
    blocks = sum(x is not None for x in out[:3])
    events[S2_EVENTS[0]] = events.get(S2_EVENTS[0], 0) + rows * n_ch * blocks
    return out


def _providers(basis: Any) -> tuple[Callable, list]:
    """``make(channels)``: the provider call for new rows on ``channels``, and the stage memos to close.

    A cached parent reads the S2 store (``reuse_cells``). Any other parent computes every value with fresh
    jet routines and one ``local_memo()`` for its stack tables.
    """
    prints = {r.variable: r.fingerprint for r in basis.specified}
    shared = local_memo()  # the stack tables of an uncached parent (decision 167)
    memos: list = [shared]

    def make(channels: Any) -> Callable:
        found = derived_reuse(basis, channels)
        if found is None:
            return functools.partial(
                _plain, memo=shared, jet=fresh_jet(basis.averaged, prints)
            )
        reuse, stage = found
        memos.append(stage)
        return functools.partial(reuse_cells, reuse)

    return make, memos


def fill_new(
    basis: Any, channels: Any, compute: dict, s0: Any, s2: Any, counts: dict
) -> tuple:
    """Write the new rows into ``s0`` and ``s2``. The routes and memo notes, with the memo counts in ``counts``."""
    rows = np.array(sorted(compute.values()), dtype=np.int64)
    common = dict(s1=basis.record.s1, with_v="V" in basis.columns.stokes, events=counts)
    make, memos = _providers(basis)
    try:
        routes = _fill(basis, channels, rows, make, common, s0, s2, counts)
    finally:
        notes = close_memos(memos, counts)
    return routes, notes


def _fill(basis, channels, rows, make, common, s0, s2, counts) -> tuple:
    plan = {r.cell: r.provider for r in basis.record.routes}
    sub, lattice, with_v = channels[rows], basis.lattice, common["with_v"]
    provide, parts, refused = make(sub), [], {}
    for c in (c for c, p in plan.items() if p == "stack"):
        try:
            parts.append(
                _guarded(
                    provide,
                    basis.averaged,
                    sub,
                    lattice,
                    {c: "stack"},
                    cells=(c,),
                    **common,
                )
            )
        except StackRefused as err:
            refused[c] = str(err)
    others = {c: p for c, p in plan.items() if p != "stack"}
    if others:
        parts.append(
            _guarded(
                provide,
                basis.averaged,
                sub,
                lattice,
                others,
                cells=tuple(others),
                **common,
            )
        )
    shape = (lattice.index.block_size, rows.size)
    g0, g2 = gather_spectra(*_summed(parts, shape, with_v), basis.columns)
    s0[rows] = g0
    if s2 is not None:
        s2[rows] = g2
    if refused:
        _jet_on_all(basis, channels, tuple(refused), s0, s2, make(channels), common)
    counts["cells moved to the jet"] = len(refused)
    return tuple(
        RouteRecord(r.cell, "jet", reason=refused[r.cell]) if r.cell in refused else r
        for r in basis.record.routes
    )


def _jet_on_all(
    basis: Any,
    channels: Any,
    cells: tuple,
    s0: Any,
    s2: Any,
    provide: Callable,
    common: dict,
) -> None:
    """Recompute ``cells`` by the jet on every channel and overwrite their stored spectra."""
    jet = {c: "jet" for c in cells}
    out = _guarded(
        provide, basis.averaged, channels, basis.lattice, jet, cells=cells, **common
    )
    g0, g2 = gather_spectra(*out[:3], basis.columns)
    cell_of = basis.lattice.index.arrays.cell
    m0 = np.isin(cell_of[basis.columns.spin0_base], cells)
    s0[:, :, m0] = g0[:, :, m0]
    if s2 is not None:
        m2 = np.isin(cell_of[basis.columns.spin2_base], cells)
        s2[:, m2] = g2[:, m2]


# ------------------------------------------------------------------ with_order


def capped_lattice(lattice: Any, order: int) -> Any:
    """The lattice with each per-cell order of the variables inside the total capped at ``order``, total ``order``.

    Variables outside the total (depth, unless ``depth_in_total``) keep their orders. The rows are those of
    ``lattice`` whose powers obey the caps, so they are a subset of the built rows (decision 153).
    """
    t = lattice.cells
    weights = lattice.weights if lattice.explicit_weights else None
    common = dict(
        order=order,
        weights=weights,
        depth_in_total=lattice.depth_in_total,
        **lattice._config(),
    )
    if lattice.cell_axes is not None:
        cells = {
            n: (
                c
                if n in t.excluded
                else c.with_order(tuple(min(o, order) for o in c.order))
            )
            for n, c in lattice.cell_axes
        }
        return Lattice(**common, **cells, **lattice.representations)
    rows = tuple(
        tuple(o if v in t.excluded else min(o, order) for v, o in zip(t.variables, r))
        for r in t.order
    )
    return Lattice(
        **common, _table=table_with(t, order=rows), _reps=lattice.representations
    )


def _base_keys(index: Any) -> np.ndarray:
    a, n = index.arrays, index.block_size
    return np.column_stack([a.cell[:n], a.pair[:n], a.powers[:n]])


def _row_map(old: Any, new: Any, cells=None) -> np.ndarray:
    """The row of one ``old`` block with the (cell, pair, powers) of each row of one ``new`` block."""
    ko, kn = _base_keys(old), _base_keys(new)
    if cells is not None:
        kn[:, 0] = cells[kn[:, 0]]
    _, inverse = np.unique(np.concatenate([ko, kn]), axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    position = np.full(int(inverse.max()) + 1, -1, dtype=np.int64)
    position[inverse[: len(ko)]] = np.arange(len(ko))
    out = position[inverse[len(ko) :]]
    if np.any(out < 0):
        raise ValueError(
            "the lower-order lattice has rows that the built lattice does not hold"
        )
    return out


def order_indices(basis: Any, lattice: Any, columns: Any, *, cells=None) -> tuple:
    """Stored spin-0 and spin-2 spectrum of the parent for each stored spectrum of the lower-order map."""
    rows, old = _row_map(basis.lattice.index, lattice.index, cells), basis.columns
    col = old.col_of_row[rows[columns.spin0_base]]
    idx0 = np.where(col >= 0, old.spin0_from[np.maximum(col, 0)], -1)
    position = np.full(basis.lattice.index.block_size, -1, dtype=np.int64)
    position[old.spin2_base] = np.arange(old.n2)
    idx2 = position[rows[columns.spin2_base]]
    if np.any(idx0 < 0) or np.any(idx2 < 0):
        raise ValueError(
            "the built basis stores no spectrum for a column of the lower-order map"
        )
    return idx0, idx2


# ------------------------------------------------------------------ records


def record_sizes(columns: Any) -> dict:
    return {
        "n_rows": columns.n_rows,
        "n": columns.n,
        "n0": columns.n0,
        "n2": columns.n2,
    }


def record_keys(columns: Any, lattice: Any) -> dict:
    return {
        "column map": columns.key,
        "index": lattice.index.key,
        "lattice": lattice.key,
    }


def build_record(
    routes: tuple, blocks: tuple, s1: str, columns: Any, lattice: Any, **fields: Any
) -> BuildRecord:
    """The record of a fresh build. ``fields`` gives ``events``, ``timings``, ``keys``, ``defaults`` and ``notes``."""
    keys = {**record_keys(columns, lattice), **fields.pop("keys", {})}
    return BuildRecord(
        routes=routes,
        stokes=stokes_axes(blocks),
        s1=s1,
        sizes=record_sizes(columns),
        keys=keys,
        **fields,
    )


def derived_record(
    record: BuildRecord, routes: tuple, counts: dict, start: float
) -> BuildRecord:
    """The record of an update or a memo hit: new routes, events, timing and load."""
    providers = [("stack cells", 0), ("jet cells", 0), ("core cells", 0)]
    return dataclasses.replace(
        record,
        routes=routes,
        events={**dict(providers), **counts},
        timings={"build": time.perf_counter() - start},
        load=load_average(),
    )


def uncached_record(record: BuildRecord, keys: dict | None = None) -> BuildRecord:
    """``record`` without memo and design keys, with the note of a basis from ``at`` or ``with_order``."""
    found = {**dict(record.keys), **(keys or {}), "memo": None, "design": None}
    notes = tuple(n for n in record.notes if n != DERIVED) + (DERIVED,)
    return dataclasses.replace(record, keys=found, notes=notes)


# ------------------------------------------------------------------ build arguments


def check_arguments(rules: Any, cache: Any) -> None:
    if not isinstance(cache, bool):
        raise ValueError(f"cache must be a bool, not {cache!r}")
    if not isinstance(rules, ColumnRules):
        raise TypeError(f"rules must be ColumnRules, not {type(rules).__name__}")


def stokes_blocks(kernel: Any, rules: ColumnRules, stokes: Any) -> tuple[str, ...]:
    """The Stokes blocks of ``stokes`` or ``rules.stokes``, which must agree when both are given."""
    if stokes is not None and rules.stokes is not None:
        if check_stokes(kernel, stokes) != check_stokes(kernel, rules.stokes):
            raise ValueError(
                f"stokes={stokes!r} differs from rules.stokes={rules.stokes!r}"
            )
    request = stokes if stokes is not None else rules.stokes
    return check_stokes(kernel, kernel.stokes if request is None else request)


def check_factor(
    kernel: Any, lattice: Any, blocks: tuple[str, ...], factor: Any
) -> Any:
    """The R5 (i) stand-in: required only for P with an azimuth that no ``AzimuthHarmonic`` resolves."""
    harmonic = any(isinstance(r, AzimuthHarmonic) for _, r in lattice.reps)
    if factor is None:
        if "P" in blocks and kernel.azimuth is not None and not harmonic:
            raise ValueError(
                f"P needs the factor of the azimuth {kernel.azimuth}: pass azimuth_factor (route R5 (i)) or "
                f"resolve {kernel.azimuth} with AzimuthHarmonic"
            )
        return None
    if "P" not in blocks or kernel.azimuth is None or harmonic:
        raise ValueError(
            "azimuth_factor multiplies P for an azimuth the lattice leaves unresolved. Here P is not requested, "
            "the kernel has no azimuth, or AzimuthHarmonic resolves it"
        )
    value = jnp.asarray(factor, dtype=jnp.complex128)
    if value.shape != ():
        raise ValueError(
            f"azimuth_factor must be a complex scalar, not shape {value.shape}"
        )
    return value


def raised_routes(routes: tuple) -> dict:
    """The routes of a parent for a higher order: "auto" where the parent took the stack, the provider elsewhere."""
    return {r.cell: "auto" if r.provider == "stack" else r.provider for r in routes}
