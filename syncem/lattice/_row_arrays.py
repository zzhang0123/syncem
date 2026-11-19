"""Vectorised row attributes of a moment index, one entry per lattice row (SPEC Sec. 3.4, decision 147).

LABEL ``eq: channel basis expansion``. ``row_arrays(index)`` computes per row the block number, the flat
product cell, the Legendre pair number, the monomial number inside the cell's lower set, the powers per
celled variable and the Legendre degrees per axis. It works per pattern from ``patterns``, ``cell_pattern``,
``offsets`` and ``n_pairs`` and creates no ``Row`` object, so G15 ``lat8`` (2 737 152 rows) costs integer
arrays only. Row order is that of ``MomentIndex.row``: block, then cell, then Legendre pair, then monomial.
``pair`` is the Legendre pair number (``np.ravel_multi_index`` of the degrees), not the Re and Im partner
of ``Row.pair``. The partner of row ``k`` in an ``Re M2`` or ``Im M2`` block is ``k`` plus or minus
``block_size``.

Units: dimensionless integers. Shapes: ``block``, ``cell``, ``pair`` and ``monomial`` ``(n_rows,)``,
``powers`` ``(n_rows, n_celled)`` in the order of ``table.variables``, ``degrees`` ``(n_rows, n_legendre)``
in the order of ``index.legendre``. All arrays are int64 and read-only. Not certified: the arrays are exact
integer arithmetic on the index and have no error of their own.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from .index import MomentIndex

__all__ = ["RowArrays", "row_arrays"]


@dataclass(frozen=True, eq=False)
class RowArrays:
    """Row attributes of a moment index as read-only int64 arrays."""

    block: np.ndarray
    cell: np.ndarray
    pair: np.ndarray
    monomial: np.ndarray
    powers: np.ndarray
    degrees: np.ndarray

    @property
    def n(self) -> int:
        return int(self.cell.shape[0])


def _pattern_tables(index: "MomentIndex") -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Monomial count and first position of each pattern, and the stacked exponents of all patterns."""
    n_celled = len(index.table.variables)
    counts = np.array([len(p) for p in index.patterns], dtype=np.int64)
    starts = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
    exponents = np.array(
        [e for p in index.patterns for e in p], dtype=np.int64
    ).reshape(-1, n_celled)
    return counts, starts, exponents


def _one_block(index: "MomentIndex") -> dict[str, np.ndarray]:
    """Cell, pair, monomial, powers and degrees of the rows of one block."""
    counts, starts, exponents = _pattern_tables(index)
    pattern = np.asarray(index.cell_pattern, dtype=np.int64)
    per_cell = counts[pattern] * index.n_pairs
    n_cells = index.table.n_cells
    cell = np.repeat(np.arange(n_cells, dtype=np.int64), per_cell)
    offsets = np.asarray(index.offsets, dtype=np.int64)
    within = np.arange(index.block_size, dtype=np.int64) - index.n_pairs * offsets[cell]
    pair, monomial = np.divmod(within, counts[pattern][cell])
    powers = exponents[starts[pattern][cell] + monomial]
    dims = tuple(d + 1 for _, d in index.legendre)
    if dims:
        degrees = np.stack(np.unravel_index(pair, dims), axis=1).astype(np.int64)
    else:
        degrees = np.zeros((index.block_size, 0), dtype=np.int64)
    return dict(cell=cell, pair=pair, monomial=monomial, powers=powers, degrees=degrees)


def _frozen(array: np.ndarray) -> np.ndarray:
    out = np.ascontiguousarray(array, dtype=np.int64)
    out.setflags(write=False)
    return out


def row_arrays(index: "MomentIndex") -> RowArrays:
    """The row arrays of ``index``, built per pattern and tiled over the blocks."""
    one = _one_block(index)
    n_blocks = len(index.blocks)
    block = np.repeat(np.arange(n_blocks, dtype=np.int64), index.block_size)
    tiled = {
        k: np.tile(v, (n_blocks, 1)) if v.ndim == 2 else np.tile(v, n_blocks)
        for k, v in one.items()
    }
    return RowArrays(block=_frozen(block), **{k: _frozen(v) for k, v in tiled.items()})
