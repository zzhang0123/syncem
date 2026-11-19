"""Channel and order updates of a step-5 spectral basis: the helpers of ``at`` and ``with_order`` (decision 153).

LABEL ``eq: channel basis expansion``. Private helpers of ``syncem.engine.basis``. ``match_channels`` pairs new
channels with stored rows by ``Channels.row_keys``. ``fill_new`` computes the rows of new channels with the
recorded route of each cell, and a stack cell that the new channels refuse takes the jet on every channel. A
kernel guard failure on new channels becomes a ``ValueError`` naming ``ContinuumKernel.resolved``.
``capped_lattice`` caps each per-cell order of the variables inside the total at ``N'`` and sets the total to
``N'``, so a lattice with uneven axis or cell orders keeps a subset of its rows. ``order_indices`` maps the stored spectra of a lower-order column map to those of the built basis through the
(cell, pair, powers) of each row, without a provider call.

Units: those of the kernel per unit scaled moment. Shapes: ``s0`` ``(n_ch, n_s0, n0)``, ``s2`` ``(n_ch, n2)``,
provider rows ``(block_size, n_ch)``, index arrays ``(n0,)`` and ``(n2,)``. Not certified: the providers. The
updates are exact selection of stored spectra and provider calls on the new channels.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..lattice._lattice_parts import table_with
from ..lattice.lattice import Lattice
from ..records.records import RouteRecord
from ._routes import route_cells
from ._stack_checks import StackRefused
from .assemble import gather_spectra

__all__ = ["capped_lattice", "fill_new", "match_channels", "order_indices"]

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


def _guarded(*args: Any, **kwargs: Any) -> tuple:
    """``route_cells`` with a kernel guard failure turned into a ``ValueError`` naming ``resolved``."""
    try:
        return route_cells(*args, **kwargs)
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


def fill_new(
    basis: Any, channels: Any, compute: dict, s0: Any, s2: Any, counts: dict
) -> tuple:
    """Write the new channel rows into ``s0`` and ``s2`` and return the realised routes."""
    plan = {r.cell: r.provider for r in basis.record.routes}
    rows = np.array(sorted(compute.values()), dtype=np.int64)
    sub, lattice, s1 = channels[rows], basis.lattice, basis.record.s1
    with_v = "V" in basis.columns.stokes
    common = dict(s1=s1, with_v=with_v, events=counts)
    memo: dict = {}
    parts, refused = [], {}
    for c in (c for c, p in plan.items() if p == "stack"):
        try:
            parts.append(
                _guarded(
                    basis.averaged,
                    sub,
                    lattice,
                    {c: "stack"},
                    cells=(c,),
                    memo=memo,
                    **common,
                )
            )
        except StackRefused as err:
            refused[c] = str(err)
    others = {c: p for c, p in plan.items() if p != "stack"}
    if others:
        parts.append(
            _guarded(
                basis.averaged, sub, lattice, others, cells=tuple(others), **common
            )
        )
    shape = (lattice.index.block_size, rows.size)
    g0, g2 = gather_spectra(*_summed(parts, shape, with_v), basis.columns)
    s0[rows] = g0
    if s2 is not None:
        s2[rows] = g2
    if refused:
        _jet_on_all(basis, channels, tuple(refused), s0, s2, common)
    counts["cells moved to the jet"] = len(refused)
    return tuple(
        RouteRecord(r.cell, "jet", reason=refused[r.cell]) if r.cell in refused else r
        for r in basis.record.routes
    )


def _jet_on_all(
    basis: Any, channels: Any, cells: tuple, s0: Any, s2: Any, common: dict
) -> None:
    """Recompute ``cells`` by the jet on every channel and overwrite their stored spectra."""
    out = _guarded(
        basis.averaged,
        channels,
        basis.lattice,
        {c: "jet" for c in cells},
        cells=cells,
        **common,
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
    common = dict(order=order, weights=weights, depth_in_total=lattice.depth_in_total)
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


def _row_map(old: Any, new: Any) -> np.ndarray:
    """The row of one ``old`` block with the (cell, pair, powers) of each row of one ``new`` block."""
    ko, kn = _base_keys(old), _base_keys(new)
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


def order_indices(basis: Any, lattice: Any, columns: Any) -> tuple:
    """Stored spin-0 and spin-2 spectrum of the parent for each stored spectrum of the lower-order map."""
    rows, old = _row_map(basis.lattice.index, lattice.index), basis.columns
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
