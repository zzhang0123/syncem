"""Cells: a partition of one variable's range with a reference point and an order per cell (SPEC Sec. 3.3, 4.1).

LABEL ``eq: energy cell response``. A cell has an interval, a chart, a reference point ``x_b``, a half-width
``h_b`` in the chart and a local polynomial order ``N_b``. The chart coordinate is ``u = ln(x / x_b) / h_b``
(``"log"``) or ``u = (x - x_b) / h_b`` (``"affine"``), so ``|u| <= 1`` inside a bounded cell with its reference
point at the centre. The default reference point is the centre in the chart. ``Cells.single`` and
``Cells.at_points`` give cells without edges, whose bounds are the ends of the variable's kind (set by the
lattice) or the declared ``support``.

Units: values are converted once to the canonical unit of their dimension (``syncem.units``). Strings and
quantities carry their unit, and numbers take ``unit=``. Bare numbers keep ``given_unit = None`` and are checked
when the cells are placed on a variable (SPEC Sec. 2.4). Log-chart half-widths are dimensionless. Shapes:
``edges`` ``(n + 1,)`` or ``None``, ``reference`` and ``half_width`` ``(n,)``, ``order`` a tuple of ``n`` ints.
Not certified: the accuracy of a cell's Taylor expansion, which the basis measures.
"""

from __future__ import annotations

import math
from typing import Any, Literal

import equinox as eqx
import numpy as np

from ..units import UnitError
from ._cell_values import (
    _canonical,
    _count,
    _is_carrier,
    _items,
    _like,
    _orders,
    _reference,
    _spelling,
)

__all__ = ["CHARTS", "Cells"]

CHARTS = ("log", "affine")


class Cells(eqx.Module):
    """A partition of one variable's range. Each cell has its own reference point and order."""

    chart: Literal["log", "affine"] = eqx.field(static=True)
    edges: np.ndarray | None
    reference: np.ndarray
    half_width: np.ndarray
    order: tuple[int, ...] = eqx.field(static=True)
    given_unit: str | None = eqx.field(static=True)
    support: tuple[float, float] | None = eqx.field(static=True, default=None)
    spelling: tuple = eqx.field(static=True, default=("Cells", (), (), ""))
    is_point: bool = eqx.field(static=True, default=False)
    noun: str = eqx.field(static=True, default="set of cells")

    def __check_init__(self) -> None:
        if self.chart not in CHARTS:
            raise ValueError(f"chart must be one of {CHARTS}, not {self.chart!r}")
        if np.any(~np.isfinite(self.reference)) or np.any(~(self.half_width > 0)):
            raise ValueError("reference points must be finite and half-widths positive")
        if self.chart == "log" and np.any(self.reference <= 0):
            raise ValueError("the log chart needs positive reference points")
        slack = (
            1e-12 * np.maximum(np.abs(self.edges[:-1]), np.abs(self.edges[1:]))
            if self.edges is not None
            else 0
        )
        if self.edges is not None and (
            np.any(self.reference < self.edges[:-1] - slack)
            or np.any(self.reference > self.edges[1:] + slack)
        ):
            raise ValueError("reference points must lie inside their cells")
        if self.support is not None:
            lo, hi = self.support
            if (
                not lo < hi
                or np.any(self.reference < lo)
                or np.any(self.reference > hi)
            ):
                raise ValueError(
                    f"support {self.support} must satisfy lo < hi and contain the reference points"
                )

    # ------------------------------------------------------------ constructors
    @classmethod
    def from_edges(
        cls,
        edges: Any,
        *,
        chart: str = "log",
        order: Any = 2,
        unit: str | None = None,
        reference: Any = None,
    ) -> "Cells":
        """Cells between consecutive edges (canonical unit after conversion), one order per cell or one for all."""
        spelling = _spelling(
            "from_edges",
            (
                (
                    edges
                    if _is_carrier(edges) or isinstance(edges, np.ndarray)
                    else list(edges)
                ),
            ),
            dict(chart=chart, order=order, unit=unit, reference=reference),
            dict(chart="log", order=2, unit=None, reference=None),
        )
        e, given = _canonical(edges, unit, "Cells.from_edges")
        return cls._from_canonical_edges(
            e, chart, order, given, _reference(reference, unit, given), spelling
        )

    @classmethod
    def log(
        cls,
        lo: Any,
        hi: Any,
        n: int,
        *,
        order: Any = 2,
        unit: str | None = None,
        reference: Any = None,
    ) -> "Cells":
        """``n`` cells equal in ``ln x`` on ``[lo, hi]``, log chart."""
        spelling = _spelling(
            "log",
            (lo, hi, n),
            dict(order=order, unit=unit, reference=reference),
            dict(order=2, unit=None, reference=None),
        )
        (a, b), given = _canonical([lo, hi], unit, "Cells.log")
        n = _count(n)
        if not 0 < a < b:
            raise ValueError(f"Cells.log needs 0 < lo < hi, not {a!r}, {b!r}")
        edges = np.exp(np.linspace(math.log(a), math.log(b), n + 1))
        edges[0], edges[-1] = a, b
        return cls._from_canonical_edges(
            edges, "log", order, given, _reference(reference, unit, given), spelling
        )

    @classmethod
    def linear(
        cls,
        lo: Any,
        hi: Any,
        n: int,
        *,
        order: Any = 2,
        unit: str | None = None,
        reference: Any = None,
    ) -> "Cells":
        """``n`` equal cells on ``[lo, hi]``, affine chart."""
        spelling = _spelling(
            "linear",
            (lo, hi, n),
            dict(order=order, unit=unit, reference=reference),
            dict(order=2, unit=None, reference=None),
        )
        (a, b), given = _canonical([lo, hi], unit, "Cells.linear")
        n = _count(n)
        if not a < b:
            raise ValueError(f"Cells.linear needs lo < hi, not {a!r}, {b!r}")
        edges = np.linspace(a, b, n + 1)
        return cls._from_canonical_edges(
            edges, "affine", order, given, _reference(reference, unit, given), spelling
        )

    @classmethod
    def single(
        cls,
        reference: Any,
        scale: Any,
        *,
        order: int = 2,
        chart: str = "affine",
        unit: str | None = None,
        support: Any = None,
    ) -> "Cells":
        """One cell without edges: ``u = (x - x_b) / scale`` (affine) or ``ln(x / x_b) / scale`` (log)."""
        spelling = _spelling(
            "single",
            (reference, scale),
            dict(order=order, chart=chart, unit=unit),
            dict(order=2, chart="affine", unit=None),
        )
        cells = cls.at_points(
            [reference], [scale], order=order, chart=chart, unit=unit, support=support
        )
        return cells._replace(spelling=spelling)

    @classmethod
    def at_points(
        cls,
        references: Any,
        half_widths: Any,
        *,
        order: Any = 2,
        chart: str = "affine",
        unit: str | None = None,
        support: Any = None,
    ) -> "Cells":
        """Cells without edges at arbitrary reference points, for example the probe cells of a check."""
        refs, scales = _items(references), _items(half_widths)
        x, given = _canonical(refs, unit, "Cells reference")
        if chart == "log":
            if any(_is_carrier(s) for s in scales):
                raise UnitError(
                    "a log-chart half-width is dimensionless. Pass a number such as 0.24."
                )
            h = np.asarray(scales, dtype=float).ravel()
        else:
            h = _like(scales, unit, given, "Cells scale")
        if h.shape != x.shape or x.size == 0:
            raise ValueError("at_points needs one half-width per reference point")
        bounds = None
        if support is not None:
            lo_hi = _like(list(support), unit, given, "Cells support")
            if lo_hi.shape != (2,):
                raise ValueError("support must be a pair (lo, hi)")
            bounds = (float(lo_hi[0]), float(lo_hi[1]))
        spelling = _spelling(
            "at_points",
            (references, half_widths),
            dict(order=order, chart=chart),
            dict(order=2, chart="affine"),
        )
        return cls(
            chart=chart,
            edges=None,
            reference=x,
            half_width=h,
            order=_orders(order, len(refs)),
            given_unit=given,
            support=bounds,
            spelling=spelling,
        )

    @classmethod
    def _from_canonical_edges(
        cls,
        edges: np.ndarray,
        chart: str,
        order: Any,
        given: str | None,
        reference: np.ndarray | None,
        spelling: tuple,
    ) -> "Cells":
        if chart not in CHARTS:
            raise ValueError(f"chart must be one of {CHARTS}, not {chart!r}")
        if (
            edges.ndim != 1
            or edges.size < 2
            or np.any(np.diff(edges) <= 0)
            or np.any(~np.isfinite(edges))
        ):
            raise ValueError(
                "edges must be finite and strictly increasing, at least two of them"
            )
        if chart == "log" and edges[0] <= 0:
            raise ValueError("the log chart needs positive edges")
        t = np.log(edges) if chart == "log" else edges
        half = 0.5 * np.diff(t)
        centre = 0.5 * (t[1:] + t[:-1])
        if reference is None:
            ref = np.exp(centre) if chart == "log" else centre
        else:
            ref = reference
            if (
                ref.shape != centre.shape
                or np.any(ref < edges[:-1])
                or np.any(ref > edges[1:])
            ):
                raise ValueError(
                    "reference points must lie inside their cells, one per cell"
                )
        return cls(
            chart=chart,
            edges=edges,
            reference=ref,
            half_width=half,
            order=_orders(order, half.size),
            given_unit=given,
            spelling=spelling,
        )

    # ------------------------------------------------------------ derived cells
    @property
    def n(self) -> int:
        return int(self.reference.shape[0])

    def _replace(self, **changes: Any) -> "Cells":
        fields = {
            f: getattr(self, f)
            for f in (
                "chart",
                "edges",
                "reference",
                "half_width",
                "order",
                "given_unit",
                "support",
                "spelling",
            )
        }
        fields.update(changes)
        return Cells(**fields)

    def _derived(self, suffix: str) -> tuple:
        name, shown, args, previous = self.spelling
        return (name, shown, args, previous + suffix)

    def with_order(self, order: Any) -> "Cells":
        return self._replace(
            order=_orders(order, self.n),
            spelling=self._derived(f".with_order({order!r})"),
        )

    def shift(self, fraction: float) -> "Cells":
        """Reference points moved by ``fraction`` of the half-width in the chart (PAPER:741)."""
        spelling = self._derived(f".shift({fraction!r})")
        if self.chart == "log":
            return self._replace(
                reference=self.reference * np.exp(fraction * self.half_width),
                spelling=spelling,
            )
        return self._replace(
            reference=self.reference + fraction * self.half_width, spelling=spelling
        )

    def refine(self, k: int) -> "Cells":
        """Every cell split into ``k`` cells equal in the chart, reference points at the new centres."""
        k = _count(k)
        if self.edges is None:
            raise ValueError("cells without edges cannot be refined")
        t = np.log(self.edges) if self.chart == "log" else self.edges
        fine = np.concatenate(
            [np.linspace(a, b, k + 1)[:-1] for a, b in zip(t[:-1], t[1:])] + [t[-1:]]
        )
        edges = np.exp(fine) if self.chart == "log" else fine
        edges[0], edges[-1] = self.edges[0], self.edges[-1]
        return Cells._from_canonical_edges(
            edges,
            self.chart,
            tuple(o for o in self.order for _ in range(k)),
            self.given_unit,
            None,
            self._derived(f".refine({k!r})"),
        )

    def bounds(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        """``(lo, hi)`` per cell: the edges, the declared support, or the ends of the variable's kind."""
        if self.edges is not None:
            return self.edges[:-1], self.edges[1:]
        if self.support is not None:
            lo, hi = self.support
        else:
            lo, hi = {
                "positive": (0.0, math.inf),
                "cosine": (-1.0, 1.0),
                "angle": (-math.inf, math.inf),
            }.get(kind, (-math.inf, math.inf))
            if self.chart == "log":
                lo = max(lo, 0.0)
        return np.full(self.n, lo), np.full(self.n, hi)

    # ------------------------------------------------------------ messages
    def __repr__(self) -> str:
        name, shown, _, suffix = self.spelling
        return f"se.Cells.{name}({', '.join(shown)}){suffix}"

    def unit_remedy(self, unit: str) -> str:
        name, shown, args, suffix = self.spelling
        if (
            name in ("log", "linear")
            and len(args) == 3
            and not any(_is_carrier(a) for a in args)
        ):
            lo, hi, n = args
            extra = "".join(f", {s}" for s in shown[3:])
            call = f'se.Cells.{name}("{float(lo):g} {unit}", "{float(hi):g} {unit}", {n}{extra}){suffix}'
            return f'Write {call}, or pass unit="{unit}".'
        return f'Pass unit="{unit}".'
