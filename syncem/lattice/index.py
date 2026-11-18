"""The moment index: one row per lattice row, by arithmetic, never stored row by row (SPEC Sec. 3.3, 4.4).

LABEL ``eq: channel basis expansion``. Rows are ordered by block, then product cell, then Legendre pair, then
powers in graded lexicographic order (total degree first, then the larger first exponent first). The powers of
a product cell form the lower set ``r_v <= N_v`` for every celled variable with ``sum r_v <= order`` over the
variables that are not excluded. The powers of an excluded variable (depth) run over ``0..N_d`` on every row of
the others (SPEC Sec. 4.3). For one celled variable the position inside a block is ``b (N + 1) + r``.

Positions use integers per variable on a tensor table, or the flat cell number. Selections accept integers,
slices, intervals ``(lo, hi)`` (cells inside the interval) and predicates ``f(lo, hi) -> bool``. ``rows`` is a
lazy view. Units: interval bounds in canonical units. Shapes: ``offsets`` ``(n_cells + 1,)``. Not certified:
the index is exact arithmetic and has no error of its own.
"""

from __future__ import annotations

import hashlib
import itertools
from dataclasses import dataclass, field
from typing import Any, Callable

import numpy as np

from .table import CellTable

__all__ = ["MomentIndex", "Row", "lower_set"]


def lower_set(
    caps: tuple[int, ...], total: int | None, excluded: tuple[bool, ...]
) -> tuple[tuple[int, ...], ...]:
    """Exponents with ``r_v <= caps_v`` and ``sum r_v <= total`` over the variables not excluded, graded lex."""
    out = [
        r
        for r in itertools.product(*[range(c + 1) for c in caps])
        if total is None or sum(x for x, e in zip(r, excluded) if not e) <= total
    ]
    return tuple(sorted(out, key=lambda r: (sum(r), tuple(-x for x in r))))


@dataclass(frozen=True)
class Row:
    """One lattice row: block, flat cell, intervals and powers per celled variable, Legendre degrees, partner."""

    block: str
    cell: int
    intervals: dict[str, tuple[float, float]]
    powers: dict[str, int]
    degrees: dict[str, int]
    pair: int | None


class _Rows:
    def __init__(self, index: "MomentIndex"):
        self._index = index

    def __len__(self) -> int:
        return self._index.n

    def __getitem__(self, k: int | slice) -> Row | list[Row]:
        if isinstance(k, slice):
            return [self._index.row(i) for i in range(*k.indices(self._index.n))]
        return self._index.row(k if k >= 0 else self._index.n + k)


@dataclass(frozen=True, eq=False)
class MomentIndex:
    """Arithmetic index of the lattice rows. Equality and hash follow ``key``."""

    blocks: tuple[str, ...]
    table: CellTable = field(repr=False)
    legendre: tuple[tuple[str, int], ...]
    total: int | None
    patterns: tuple[tuple[tuple[int, ...], ...], ...]
    cell_pattern: np.ndarray = field(repr=False)
    offsets: np.ndarray = field(repr=False)
    key: str = ""

    @classmethod
    def build(
        cls,
        table: CellTable,
        blocks: tuple[str, ...],
        legendre: tuple[tuple[str, int], ...],
        total: int | None,
    ) -> "MomentIndex":
        excluded = tuple(v in table.excluded for v in table.variables)
        lookup: dict[tuple[int, ...], int] = {}
        patterns: list[tuple[tuple[int, ...], ...]] = []
        ids = np.empty(table.n_cells, dtype=np.int64)
        for c, caps in enumerate(table.order):
            if caps not in lookup:
                lookup[caps] = len(patterns)
                patterns.append(lower_set(caps, total, excluded))
            ids[c] = lookup[caps]
        counts = np.array([len(patterns[i]) for i in ids], dtype=np.int64)
        offsets = np.concatenate([[0], np.cumsum(counts)])
        digest = hashlib.sha256(
            b"|".join(
                [
                    repr((blocks, legendre, total, tuple(patterns))).encode(),
                    ids.tobytes(),
                    table.definition(),
                ]
            )
        ).hexdigest()
        return cls(
            blocks, table, legendre, total, tuple(patterns), ids, offsets, digest
        )

    def __eq__(self, other: object) -> bool:
        return isinstance(other, MomentIndex) and other.key == self.key

    def __hash__(self) -> int:
        return hash(self.key)

    # ------------------------------------------------------------ sizes
    @property
    def n_pairs(self) -> int:
        return (
            int(np.prod([d + 1 for _, d in self.legendre], dtype=np.int64))
            if self.legendre
            else 1
        )

    @property
    def block_size(self) -> int:
        return int(self.offsets[-1]) * self.n_pairs

    @property
    def n(self) -> int:
        return len(self.blocks) * self.block_size

    @property
    def rows(self) -> _Rows:
        return _Rows(self)

    def _count(self, c: int) -> int:
        return int(self.offsets[c + 1] - self.offsets[c])

    def cell_rows(self, c: int, block: int = 0) -> range:
        """Positions of the rows of product cell ``c`` in one block."""
        if not 0 <= c < self.table.n_cells or not 0 <= block < len(self.blocks):
            raise IndexError(f"cell {c} or block {block} outside the lattice")
        start = block * self.block_size + self.n_pairs * int(self.offsets[c])
        return range(start, start + self.n_pairs * self._count(c))

    # ------------------------------------------------------------ rows and positions
    def row(self, k: int) -> Row:
        if not 0 <= k < self.n:
            raise IndexError(f"row {k} outside 0..{self.n - 1}")
        b, rest = divmod(k, self.block_size)
        c = int(np.searchsorted(self.offsets, rest // self.n_pairs, side="right") - 1)
        within = rest - self.n_pairs * int(self.offsets[c])
        pair, m = divmod(within, self._count(c))
        powers = dict(
            zip(self.table.variables, self.patterns[int(self.cell_pattern[c])][m])
        )
        degrees = self._degrees(pair)
        intervals = {
            v: (float(self.table.lo[c, i]), float(self.table.hi[c, i]))
            for i, v in enumerate(self.table.variables)
        }
        block = self.blocks[b]
        partner = {"Re M2": 1, "Im M2": -1}.get(block)
        return Row(
            block,
            c,
            intervals,
            powers,
            degrees,
            None if partner is None else k + partner * self.block_size,
        )

    def _degrees(self, pair: int) -> dict[str, int]:
        if not self.legendre:
            return {}
        values = np.unravel_index(pair, tuple(d + 1 for _, d in self.legendre))
        return {name: int(v) for (name, _), v in zip(self.legendre, values)}

    def _cell(self, selector: dict[str, Any]) -> int:
        """Flat cell number from ``cell=`` or from per-variable numbers, intervals or predicates (one cell)."""
        extra = set(selector) - set(self.table.variables) - {"cell"}
        if extra:
            raise ValueError(
                f"unknown selector {sorted(extra)}. Celled variables: {self.table.variables}"
            )
        per = {v: selector[v] for v in self.table.variables if v in selector}
        if "cell" in selector:
            if per:
                raise ValueError("pass cell= or per-variable selectors, not both")
            c = selector["cell"]
            if (
                isinstance(c, bool)
                or not isinstance(c, (int, np.integer))
                or not 0 <= c < self.table.n_cells
            ):
                raise ValueError(f"cell {c} outside 0..{self.table.n_cells - 1}")
            return c
        if not self.table.variables:
            return 0
        numbers = all(
            isinstance(s, (int, np.integer)) and not isinstance(s, bool)
            for s in per.values()
        )
        if (
            numbers
            and self.table.tensor is not None
            and set(per) == set(self.table.variables)
        ):
            for v, size in zip(self.table.variables, self.table.tensor):
                if not -size <= per[v] < size:
                    raise ValueError(f"{v}={per[v]} outside the {size} cells of {v}")
                per[v] = per[v] % size
            return int(
                np.ravel_multi_index(
                    tuple(int(per[v]) for v in self.table.variables), self.table.tensor
                )
            )
        chosen = self.cells_where(**per)
        if chosen.size != 1:
            raise ValueError(
                f"the selectors {per} pick {chosen.size} product cells, and a position needs one cell. "
                f"Pass cell= or select every celled variable {self.table.variables}"
            )
        return int(chosen[0])

    def _check_names(self, powers: dict | None, degrees: dict | None) -> None:
        unknown = set(powers or {}) - set(self.table.variables)
        if unknown:
            raise ValueError(
                f"no cells for powers of {sorted(unknown)}. Celled variables: {self.table.variables}"
            )
        names = {n for n, _ in self.legendre}
        unknown = set(degrees or {}) - names
        if unknown:
            raise ValueError(f"no Legendre axis for {sorted(unknown)}")

    def position(
        self,
        *,
        block: str | None = None,
        powers: dict | None = None,
        degrees: dict | None = None,
        **cell: Any,
    ) -> int:
        """Position of one row. Missing powers and degrees are zero, a missing block is the first."""
        self._check_names(powers, degrees)
        b = self.blocks.index(block) if block is not None else 0
        c = self._cell(dict(cell))
        exps = tuple(int((powers or {}).get(v, 0)) for v in self.table.variables)
        pattern = self.patterns[int(self.cell_pattern[c])]
        if exps not in pattern:
            raise ValueError(
                f"powers {dict(zip(self.table.variables, exps))} are not rows of cell {c}"
            )
        deg = tuple(int((degrees or {}).get(n, 0)) for n, _ in self.legendre)
        for (n, top), d in zip(self.legendre, deg):
            if not 0 <= d <= top:
                raise ValueError(f"degree {d} of {n} outside 0..{top}")
        pair = (
            int(np.ravel_multi_index(deg, tuple(d + 1 for _, d in self.legendre)))
            if self.legendre
            else 0
        )
        return self.cell_rows(c, b)[pair * self._count(c) + pattern.index(exps)]

    def cells_where(self, **selector: Any) -> np.ndarray:
        """Flat numbers of the product cells that every per-variable selector admits."""
        mask = np.ones(self.table.n_cells, dtype=bool)
        for v, sel in selector.items():
            if v not in self.table.variables:
                raise ValueError(f"{v} has no cells on this lattice")
            i = self.table.variables.index(v)
            lo, hi = self.table.lo[:, i], self.table.hi[:, i]
            mask &= _admit(sel, i, lo, hi, self.table)
        return np.nonzero(mask)[0]

    def select(
        self,
        *,
        block: str | None = None,
        powers: dict | None = None,
        degrees: dict | None = None,
        **cells: Any,
    ) -> tuple[int, ...]:
        """Row positions of the selected cells, optionally restricted to one block, powers or degrees."""
        self._check_names(powers, degrees)
        chosen = self.cells_where(**cells)
        blocks = (
            range(len(self.blocks)) if block is None else [self.blocks.index(block)]
        )
        out = []
        for b in blocks:
            for c in chosen:
                for k in self.cell_rows(int(c), b):
                    if powers is None and degrees is None:
                        out.append(k)
                        continue
                    row = self.row(k)
                    if powers is not None and any(
                        row.powers.get(v, 0) != p for v, p in powers.items()
                    ):
                        continue
                    if degrees is not None and any(
                        row.degrees.get(v, 0) != d for v, d in degrees.items()
                    ):
                        continue
                    out.append(k)
        return tuple(sorted(out))

    # ------------------------------------------------------------ labels
    def label(self, k: int) -> str:
        row = self.row(k)
        numbers = self.table.cell_numbers(row.cell)
        cell = (
            ", ".join(f"{v}={i}" for v, i in numbers.items())
            if numbers
            else f"cell={row.cell}"
        )
        if len(self.table.variables) == 1:
            powers = [f"u^{row.powers[self.table.variables[0]]}"]
        else:
            powers = [f"u_{v}^{r}" for v, r in row.powers.items()]
        powers += [f"P_{d}({v})" for v, d in row.degrees.items()]
        return (
            f"{row.block}[{cell} | {' '.join(powers)}]"
            if cell
            else f"{row.block}[{' '.join(powers)}]"
        )

    def labels(self) -> tuple[str, ...]:
        return tuple(self.label(k) for k in range(self.n))

    def summary(self) -> str:
        legendre = ", ".join(f"{v}: Legendre({d})" for v, d in self.legendre) or "none"
        sizes = sorted({len(p) for p in self.patterns})
        return (
            f"{self.n} rows = {len(self.blocks)} blocks ({', '.join(self.blocks)}) x {self.table.n_cells} product "
            f"cells x {self.n_pairs} Legendre pairs ({legendre}) x powers per cell {sizes}, total degree "
            f"{self.total} over {[v for v in self.table.variables if v not in self.table.excluded]}"
        )


def _admit(
    sel: Any, i: int, lo: np.ndarray, hi: np.ndarray, table: CellTable
) -> np.ndarray:
    if callable(sel):
        return np.array([bool(sel(a, b)) for a, b in zip(lo, hi)], dtype=bool)
    if isinstance(sel, tuple) and len(sel) == 2:
        a, b = (float(x) for x in sel)
        finite = [abs(x) for x in (a, b) if np.isfinite(x)]
        tol = 1e-12 * max(finite + [1.0])
        return (lo >= a - tol) & (hi <= b + tol)
    if table.tensor is None:
        raise ValueError(
            "integer and slice selectors need a tensor table. Use an interval (lo, hi) or a predicate"
        )
    numbers = np.unravel_index(np.arange(table.n_cells), table.tensor)[i]
    wanted = np.arange(table.tensor[i])[sel]
    return np.isin(numbers, np.atleast_1d(wanted))


Predicate = Callable[[float, float], bool]
