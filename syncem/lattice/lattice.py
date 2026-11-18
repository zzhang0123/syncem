"""The lattice: product cells, representations of the other resolved variables, the moment index (SPEC Sec. 3.3).

LABEL ``eq: energy cell response``. ``Lattice(**axes)`` is the tensor product of its ``Cells`` axes with the
``Legendre`` and ``AzimuthHarmonic`` representations. ``Lattice.from_cells`` takes any table of boxes (overlaps
refused, no coverage required), and ``refine(where=)`` and ``select`` give such tables. ``order`` caps the total
Taylor degree over the celled variables other than depth. Depth powers lie outside it (SPEC Sec. 4.3) unless
``depth_in_total=True``, the index set of ``Lattice.single``. The reference weights ``w_c`` are the cell masses
of ``gamma^-2.6`` normalised on the lattice's ``gamma`` range, times 1 for the other axes (SPEC Sec. 4.5).
``key`` hashes the cell table, the index definition and the weights, computed eagerly without per-row objects.

Units: canonical units. Shapes: ``reference_weights`` ``(n_cells,)``, moment values ``(..., n)``. Not
certified: the lattice is a choice of representation, and its truncation is measured by the basis.
"""

from __future__ import annotations

import hashlib
from typing import Any

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..values.errors import SizeError
from .cells import Cells
from .index import MomentIndex
from .maps import coarsen_values, recentre_values
from .moments import Moments
from .representations import AzimuthHarmonic, Legendre
from .table import CellTable
from ._lattice_parts import (
    REFERENCE_SLOPE,
    _check_axis,
    _default_weights,
    _grouped,
    _kind,
    refine_table,
    shift_table,
    table_with,
)

__all__ = ["Lattice", "REFERENCE_SLOPE"]


class Lattice(eqx.Module):
    """Product cells, representations, reference weights, index and key."""

    cells: CellTable
    weights: np.ndarray
    reps: tuple[tuple[str, Any], ...] = eqx.field(static=True)
    order: int | None = eqx.field(static=True)
    index: MomentIndex = eqx.field(static=True)
    key: str | None = eqx.field(static=True)
    depth_in_total: bool = eqx.field(static=True)
    explicit_weights: bool = eqx.field(static=True)
    explicit_order: bool = eqx.field(static=True)
    cell_axes: tuple[tuple[str, Cells], ...] | None

    def __init__(
        self,
        *,
        order: int | None = None,
        weights: Any = None,
        depth_in_total: bool = False,
        _table: CellTable | None = None,
        _reps: dict | None = None,
        **axes: Any,
    ):
        if _table is None:
            for name, axis in axes.items():
                _check_axis(name, axis)
            cell_axes = {n: a for n, a in axes.items() if isinstance(a, Cells)}
            reps = {n: a for n, a in axes.items() if not isinstance(a, Cells)}
            kinds = {n: _kind(n, a) for n, a in cell_axes.items()}
            _table = CellTable.tensor_product(
                cell_axes, kinds, () if depth_in_total else ("depth",)
            )
            self.cell_axes = tuple(cell_axes.items())
        else:
            reps = dict(_reps or {})
            for name, axis in reps.items():
                if isinstance(axis, Cells) or name in _table.variables:
                    raise ValueError(
                        f"{name}: cells belong to the table, and a celled variable takes no other "
                        "representation"
                    )
                _check_axis(name, axis)
            self.cell_axes = None
        if sum(isinstance(a, AzimuthHarmonic) for a in reps.values()) > 1:
            raise ValueError("a lattice takes one AzimuthHarmonic axis")
        self.cells = _table
        self.reps = tuple(reps.items())
        self.order = self._total(order)
        self.explicit_order = order is not None
        self.depth_in_total = depth_in_total
        harmonic = any(isinstance(a, AzimuthHarmonic) for a in reps.values())
        blocks = ("M0", "Re M2", "Im M2") if harmonic else ("M0",)
        legendre = tuple(
            (n, a.degree) for n, a in reps.items() if isinstance(a, Legendre)
        )
        self.index = MomentIndex.build(_table, blocks, legendre, self.order)
        self.explicit_weights = weights is not None
        self.weights = self._weights(weights)
        concrete = not any(
            hasattr(x, "aval") for x in (self.weights, _table.lo, _table.reference)
        )
        digest = hashlib.sha256(
            self.index.key.encode() + (np.asarray(self.weights, float) + 0.0).tobytes()
        )
        self.key = digest.hexdigest() if concrete else None

    def _total(self, order: int | None) -> int | None:
        table = self.cells
        free = [i for i, v in enumerate(table.variables) if v not in table.excluded]
        if order is not None and (
            isinstance(order, bool)
            or not isinstance(order, (int, np.integer))
            or order < 0
        ):
            raise ValueError(f"order must be a non-negative int or None, not {order!r}")
        if not free:
            if order is not None:
                raise ValueError(
                    f"order={order} caps the total over celled variables other than depth, "
                    "and this lattice has none"
                )
            return None
        maxima = {table.variables[i]: max(o[i] for o in table.order) for i in free}
        if order is None:
            return max(maxima.values())
        if order > sum(maxima.values()):
            raise ValueError(
                f"order={order} is above the sum of the axis orders {maxima}, so no row reaches it. "
                "Raise the axis orders with Cells.with_order, or lower order"
            )
        return int(order)

    def _weights(self, weights: Any) -> np.ndarray:
        if weights is None:
            return _default_weights(self.cells)
        w = np.asarray(weights, dtype=float)
        if (
            w.shape != (self.cells.n_cells,)
            or np.any(~np.isfinite(w))
            or np.any(w <= 0)
        ):
            raise ValueError(
                f"weights must be {self.cells.n_cells} positive finite numbers, one per product cell"
            )
        return w

    # ------------------------------------------------------------ constructors and derived lattices
    @classmethod
    def from_cells(
        cls,
        table: CellTable,
        *,
        representations: dict | None = None,
        order: int | None = None,
        weights: Any = None,
        depth_in_total: bool = False,
    ) -> "Lattice":
        """Any table of axis-aligned boxes. Invalid boxes and overlaps raise. Coverage is not required."""
        excluded = (
            () if depth_in_total else tuple(v for v in table.variables if v == "depth")
        )
        table = table_with(table, tensor=None, excluded=excluded)
        table.validate({v: _kind(v, None) for v in table.variables})
        table.check_overlaps()
        return cls(
            order=order,
            weights=weights,
            depth_in_total=depth_in_total,
            _table=table,
            _reps=representations,
        )

    def _derived(self, table: CellTable, weights: Any = None) -> "Lattice":
        return Lattice(
            order=self.order if self.explicit_order else None,
            weights=weights,
            depth_in_total=self.depth_in_total,
            _table=table,
            _reps=self.representations,
        )

    @property
    def representations(self) -> dict[str, Any]:
        return dict(self.reps)

    @property
    def axes(self) -> dict[str, Any]:
        """One ``Cells`` per celled variable and the other representations, on a tensor table only."""
        if self.cell_axes is None:
            raise AttributeError(
                "this lattice has no tensor axes. Use lattice.cells, the table of product cells"
            )
        return {**dict(self.cell_axes), **self.representations}

    def _tensor(self, order: Any = "keep", **changes: Any) -> "Lattice":
        axes = self.axes
        unknown = set(changes) - set(axes)
        if unknown:
            raise ValueError(
                f"{sorted(unknown)} are not axes of this lattice: {sorted(axes)}"
            )
        axes.update(changes)
        if order == "keep":
            order = self.order if self.explicit_order else None
        weights = self.weights if self.explicit_weights else None
        return Lattice(
            order=order, weights=weights, depth_in_total=self.depth_in_total, **axes
        )

    def replace(self, **axes: Any) -> "Lattice":
        """The tensor lattice with some axes replaced. An implicit total is recomputed."""
        return self._tensor(**axes)

    def with_order(self, order: int) -> "Lattice":
        """Order ``order`` on every non-depth celled variable and as the total."""
        t = self.cells
        if self.cell_axes is not None:
            cells = {
                n: (c.with_order(order) if n not in t.excluded else c)
                for n, c in self.cell_axes
            }
            return self._tensor(order=order, **cells)
        rows = tuple(
            tuple(order if v not in t.excluded else o for v, o in zip(t.variables, r))
            for r in t.order
        )
        return Lattice(
            order=order,
            weights=self.weights if self.explicit_weights else None,
            depth_in_total=self.depth_in_total,
            _table=table_with(t, order=rows),
            _reps=self.representations,
        )

    def with_key(self, key: str) -> "Lattice":
        """The same lattice with a key computed before a transformation (SPEC Sec. 8.6)."""
        out = object.__new__(Lattice)
        for name in Lattice.__dataclass_fields__:
            object.__setattr__(out, name, key if name == "key" else getattr(self, name))
        return out

    def select(self, **cells: Any) -> "Lattice":
        """The sub-lattice of the selected product cells (a non-tensor table)."""
        chosen = self.index.cells_where(**cells)
        if chosen.size == 0:
            raise ValueError("the selection contains no product cell")
        return self._derived(
            self.cells.rows(chosen),
            self.weights[chosen] if self.explicit_weights else None,
        )

    def shift(self, **fractions: float) -> "Lattice":
        """Reference points moved by a fraction of the half-width in the chart, per variable."""
        if self.cell_axes is not None:
            unknown = set(fractions) - set(dict(self.cell_axes))
            if unknown:
                raise ValueError(
                    f"shift: {sorted(unknown)} have no cells on this lattice"
                )
            return self._tensor(
                **{n: dict(self.cell_axes)[n].shift(f) for n, f in fractions.items()}
            )
        return self._derived(
            shift_table(self.cells, fractions),
            self.weights if self.explicit_weights else None,
        )

    def refine(self, *, where: dict | None = None, **factors: int) -> "Lattice":
        """Split the selected product cells (all when ``where`` is None) into equal parts in the chart."""
        if where is None and self.cell_axes is not None:
            unknown = set(factors) - set(dict(self.cell_axes))
            if unknown:
                raise ValueError(
                    f"refine: {sorted(unknown)} have no cells on this lattice"
                )
            for v, k in factors.items():
                if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
                    raise ValueError(
                        f"refine factor for {v} must be a positive int, not {k!r}"
                    )
            return self._tensor(
                **{n: dict(self.cell_axes)[n].refine(k) for n, k in factors.items()}
            )
        t = self.cells
        selected = set(
            range(t.n_cells)
            if where is None
            else self.index.cells_where(**where).tolist()
        )
        return self._derived(refine_table(t, selected, factors))

    # ------------------------------------------------------------ sizes and labels
    @property
    def n(self) -> int:
        return self.index.n

    @property
    def n_cells(self) -> int:
        return self.cells.n_cells

    @property
    def reference_weights(self) -> np.ndarray:
        return self.weights

    def labels(self) -> tuple[str, ...]:
        return self.index.labels()

    def check_size(self, n_columns: int, *, max_columns: int = 20_000) -> None:
        """Raise ``SizeError`` when the basis would have more than ``max_columns`` columns (SPEC Sec. 4.6)."""
        if n_columns <= max_columns:
            return
        digits = len(str(n_columns)) - 2
        accept = -(-n_columns // 10**digits) * 10**digits
        raise SizeError(
            f"the lattice has {_grouped(self.n)} rows and {_grouped(n_columns)} basis columns "
            f"({self.n * 8 / 1e6:.1f} MB of moments per sightline), above max_columns={_grouped(max_columns)}. "
            f"Pass max_columns={accept:_} to accept, or use fewer cells or a lower order."
        )

    # ------------------------------------------------------------ moments
    def _own(self, moments: Moments) -> None:
        if not isinstance(moments, Moments):
            raise TypeError(f"expected Moments, not {type(moments).__name__}")
        if moments.key != self.key or moments.index != self.index:
            raise ValueError(
                f"the moments belong to lattice key {str(moments.key)[:12]}, not to this lattice "
                f"(key {str(self.key)[:12]})"
            )

    def _row_weights(self) -> np.ndarray:
        counts = np.diff(self.index.offsets) * self.index.n_pairs
        return np.tile(np.repeat(self.weights, counts), len(self.index.blocks))

    def column(self, moments: Moments) -> Any:
        """``N_src = sum_c w_c a_c`` over the order-0 spin-0 rows (pair 0)."""
        self._own(moments)
        first = self.index.n_pairs * self.index.offsets[:-1]
        return jnp.asarray(moments.values)[..., first] @ jnp.asarray(self.weights)

    def raw(self, moments: Moments) -> Any:
        """``w a``, the unscaled integrals in cm^-2, per row."""
        self._own(moments)
        return jnp.asarray(moments.values) * jnp.asarray(self._row_weights())

    def _mapped(self, moments: Moments, to: Any, function: Any) -> Moments:
        self._own(moments)
        if not isinstance(to, Lattice):
            raise TypeError(f"to= must be a Lattice, not {type(to).__name__}")
        values, masked = function(
            moments.values,
            self.index,
            to.index,
            self.weights,
            to.weights,
            moments.masked,
        )
        return Moments(
            to,
            values,
            origin=moments.origin,
            masked=masked,
            retained=moments.retained,
            component=moments.component,
        )

    def recentre(self, moments: Moments, to: "Lattice") -> Moments:
        """Exact moments of the same cells at the reference points of ``to``. Masked rows propagate."""
        return self._mapped(moments, to, recentre_values)

    def coarsen(self, moments: Moments, to: "Lattice") -> Moments:
        """Exact moments of the unions of cells of ``to``. Masked rows propagate."""
        return self._mapped(moments, to, coarsen_values)
