"""The table of product cells of a lattice, one row per product cell (SPEC Sec. 3.3).

LABEL ``eq: energy cell response``. Each row holds, per celled variable, the interval ``[lo, hi]``, the reference
point, the half-width in the chart, the chart and the order. ``tensor`` holds the cell counts per variable when
the table is a tensor product (flat cell number in C order) and is ``None`` otherwise. ``excluded`` names the
variables whose powers lie outside the total degree (depth, SPEC Sec. 4.3).

Units: canonical units of each variable. Shapes: ``lo``, ``hi``, ``reference``, ``half_width``
``(n_cells, n_celled)``. Boxes of a valid table share at most faces (``check_overlaps``). Not certified:
coverage of the population's support, which the ``outside_lattice`` term measures.
"""

from __future__ import annotations

import itertools
from typing import Any

import equinox as eqx
import numpy as np

__all__ = ["CellTable"]


class CellTable(eqx.Module):
    """Stored product cells: arrays per row and variable, charts and orders as static tuples."""

    variables: tuple[str, ...] = eqx.field(static=True)
    lo: np.ndarray
    hi: np.ndarray
    reference: np.ndarray
    half_width: np.ndarray
    chart: tuple[tuple[str, ...], ...] = eqx.field(static=True)
    order: tuple[tuple[int, ...], ...] = eqx.field(static=True)
    tensor: tuple[int, ...] | None = eqx.field(static=True)
    excluded: tuple[str, ...] = eqx.field(static=True, default=())

    def __check_init__(self) -> None:
        shape = (len(self.chart), len(self.variables))
        for name in ("lo", "hi", "reference", "half_width"):
            if np.shape(getattr(self, name)) != shape:
                raise ValueError(f"CellTable.{name} must have shape {shape}")
        if len(self.order) != shape[0] or any(len(o) != shape[1] for o in self.order):
            raise ValueError("CellTable.order needs one order per row and variable")
        if self.tensor is not None and int(np.prod(self.tensor)) != shape[0]:
            raise ValueError("CellTable.tensor does not match the number of rows")
        if not set(self.excluded) <= set(self.variables):
            raise ValueError("CellTable.excluded names a variable without cells")

    @classmethod
    def tensor_product(
        cls, axes: dict[str, Any], kinds: dict[str, str], excluded: tuple[str, ...] = ()
    ) -> "CellTable":
        """The product of ``Cells`` axes, cells numbered in C order over the axes as given."""
        names = tuple(axes)
        if not names:
            empty = np.zeros((1, 0))
            return cls(names, empty, empty, empty, empty, ((),), ((),), (), ())
        counts = tuple(axes[v].n for v in names)
        grids = np.meshgrid(*[np.arange(c) for c in counts], indexing="ij")
        flat = [g.ravel() for g in grids]
        cols = {k: [] for k in ("lo", "hi", "reference", "half_width")}
        for v, idx in zip(names, flat):
            lo, hi = axes[v].bounds(kinds.get(v, "real"))
            cols["lo"].append(lo[idx])
            cols["hi"].append(hi[idx])
            cols["reference"].append(axes[v].reference[idx])
            cols["half_width"].append(axes[v].half_width[idx])
        arrays = {k: np.stack(v, axis=1).astype(float) for k, v in cols.items()}
        chart = tuple(tuple(axes[v].chart for v in names) for _ in range(len(flat[0])))
        order = tuple(itertools.product(*[axes[v].order for v in names]))
        return cls(
            names,
            arrays["lo"],
            arrays["hi"],
            arrays["reference"],
            arrays["half_width"],
            chart,
            order,
            counts,
            tuple(v for v in excluded if v in names),
        )

    @property
    def n_cells(self) -> int:
        return len(self.chart)

    def rows(self, indices: Any) -> "CellTable":
        """The sub-table of the given rows, as a non-tensor table."""
        idx = np.atleast_1d(np.arange(self.n_cells)[indices])
        return CellTable(
            self.variables,
            self.lo[idx],
            self.hi[idx],
            self.reference[idx],
            self.half_width[idx],
            tuple(self.chart[i] for i in idx),
            tuple(self.order[i] for i in idx),
            None,
            self.excluded,
        )

    def with_bounds(self, lo: Any = None, hi: Any = None) -> "CellTable":
        """A copy with other bounds, as a non-tensor table (``Lattice.from_cells`` checks the result)."""
        return CellTable(
            self.variables,
            self.lo if lo is None else np.asarray(lo, float),
            self.hi if hi is None else np.asarray(hi, float),
            self.reference,
            self.half_width,
            self.chart,
            self.order,
            None,
            self.excluded,
        )

    def check_overlaps(self, chunk: int = 512) -> None:
        """Raise when two boxes share more than a face (interval test per variable)."""
        n = self.n_cells
        for start in range(0, n, chunk):
            lo, hi = (
                self.lo[start : start + chunk, None, :],
                self.hi[start : start + chunk, None, :],
            )
            inside = np.all(
                (np.maximum(lo, self.lo[None]) < np.minimum(hi, self.hi[None])), axis=-1
            )
            rows, cols = np.nonzero(inside)
            rows = rows + start
            clash = rows != cols
            if np.any(clash):
                a, b = int(rows[clash][0]), int(cols[clash][0])
                raise ValueError(
                    f"product cells {a} and {b} overlap: boxes may share faces only"
                )

    def validate(self, kinds: dict[str, str]) -> None:
        """Raise unless every box has ``lo < hi`` with no NaN, contains its reference point, and fits the kind."""
        lo, hi, ref = self.lo, self.hi, self.reference
        if np.any(np.isnan(lo)) or np.any(np.isnan(hi)) or np.any(~(lo < hi)):
            raise ValueError("every box needs lo < hi without NaN")
        slack = 1e-12 * np.where(
            np.isfinite(lo) & np.isfinite(hi), np.maximum(np.abs(lo), np.abs(hi)), 0.0
        )
        if (
            np.any(ref < lo - slack)
            or np.any(ref > hi + slack)
            or np.any(~np.isfinite(ref))
            or np.any(~(self.half_width > 0))
        ):
            raise ValueError(
                "every box needs a finite reference point inside it and a positive half-width"
            )
        for i, v in enumerate(self.variables):
            kind = kinds.get(v, "real")
            log = np.array([c[i] == "log" for c in self.chart])
            if kind == "angle":
                raise ValueError(f"{v} is an azimuth and takes no cells")
            if kind == "cosine" and (np.any(lo[:, i] < -1) or np.any(hi[:, i] > 1)):
                raise ValueError(f"{v} is a cosine: its cells must lie in [-1, 1]")
            if (kind == "positive" or np.any(log)) and np.any(lo[:, i] < 0):
                raise ValueError(
                    f"{v} is positive: its cells must lie at or above zero"
                )
            if np.any(log & (ref[:, i] <= 0)):
                raise ValueError(f"{v}: the log chart needs positive reference points")

    def cell_numbers(self, c: int) -> dict[str, int] | None:
        """Cell number per variable of flat cell ``c`` on a tensor table, else ``None``."""
        if self.tensor is None:
            return None
        return dict(
            zip(self.variables, (int(i) for i in np.unravel_index(c, self.tensor)))
        )

    def definition(self) -> bytes:
        """Bytes that define the table, for the lattice key."""
        parts = [
            repr(
                (self.variables, self.chart, self.order, self.tensor, self.excluded)
            ).encode()
        ]
        parts += [
            (np.ascontiguousarray(getattr(self, k), dtype=float) + 0.0).tobytes()
            for k in ("lo", "hi", "reference", "half_width")
        ]
        return b"|".join(parts)
