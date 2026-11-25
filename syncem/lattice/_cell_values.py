"""Value conversion and argument checks for ``Cells`` constructors (private helpers of ``cells.py``).

LABEL ``[extension]``. A value with a unit is a string (``"5 uG"``), a ``Quantity``, a pair ``(array, "uG")`` or an
astropy quantity. A sequence of values may mix scalars of these kinds, all with units or all without. Units:
values are converted to the canonical unit of their dimension. Shapes: 1-D arrays. Not certified: these helpers
check arguments and convert units only.
"""

from __future__ import annotations

import sys
from typing import Any

import numpy as np

from ..units import CANONICAL, Quantity, UnitError, convert, dimension_of, parse


def log_displacement(value, reference, *, offset=None):
    """Compute log((value + offset)/reference) without rounding away the offset."""
    delta = np.asarray(value) - reference
    near = np.abs(delta) <= 0.5 * reference
    fraction = np.divide(
        delta, reference, where=near, out=np.zeros_like(delta, dtype=float)
    )
    result = np.where(near, np.log1p(fraction), np.log(value) - np.log(reference))
    return result if offset is None else result + np.log1p(offset / value)


def panel_coordinates(
    lower, upper, chart, nodes, weights, *, offsets=False, edge_offsets=None
):
    """Map Gauss nodes to physical x and dx with optional rounding residuals."""
    t = (1 + nodes) / 2
    width = upper - lower
    dl, du = (0.0, 0.0) if edge_offsets is None else edge_offsets
    if chart == "log" and np.all(lower > 0):
        span = log_displacement(upper, lower)
        if edge_offsets is not None:
            span += np.log1p(du / upper) - np.log1p(dl / lower)
        near = width <= 0.5 * lower
        # For wide ratios the local exponential can overflow although x is finite.
        growth = np.expm1(np.where(near, span, 0) * t)
        local = lower * growth
        if edge_offsets is not None:
            local += dl * (1 + growth)
        x = np.where(near, lower + local, np.exp(np.log(lower) + span * t))
        residual = np.where(near, local - (x - lower), 0) if offsets else None
        w = 0.5 * span * x * weights
    else:
        if edge_offsets is not None:
            width = width + (du - dl)
        local = dl + width * t
        x = lower + local
        residual = local - (x - lower) if offsets else None
        w = 0.5 * width * weights
    return x, w, residual


def interpolation_nodes(lower, upper, chart, fractions):
    """Map fractions of the cell span to physical floats with exact endpoints."""
    if chart == "log":
        if upper - lower <= 0.5 * lower:
            span = np.log1p((upper - lower) / lower)
            points = lower + lower * np.expm1(fractions * span)
        else:
            points = np.exp((1 - fractions) * np.log(lower) + fractions * np.log(upper))
    elif lower < 0 < upper:
        points = (1 - fractions) * lower + fractions * upper
    else:
        points = lower + (upper - lower) * fractions
    points = np.clip(points, lower, upper)
    return np.where(fractions == 0, lower, np.where(fractions == 1, upper, points))


def partition_edges(lo, hi, count, chart):
    """Subdivide a positive log interval or affine interval with exact endpoints."""
    if chart != "log":
        return np.linspace(lo, hi, count + 1)
    if hi - lo <= 0.5 * lo:
        offsets = np.linspace(0, np.log1p((hi - lo) / lo), count + 1)
        edges = lo + lo * np.expm1(offsets)
    else:
        edges = np.exp(np.linspace(np.log(lo), np.log(hi), count + 1))
    edges[0], edges[-1] = lo, hi
    return edges


def centers_and_widths(edges, chart):
    lower, upper = edges[:-1], edges[1:]
    if chart != "log":
        return 0.5 * lower + 0.5 * upper, 0.5 * upper - 0.5 * lower
    width = 0.5 * log_displacement(upper, lower)
    near = upper - lower <= 0.5 * lower
    centers = np.exp(0.5 * (np.log(lower) + np.log(upper)))
    centers[near] = lower[near] + lower[near] * np.expm1(width[near])
    return np.clip(centers, lower, upper), width


def edge_tolerance(lo, hi):
    """Boundary tolerance in units of the bounded cell width."""
    finite = np.isfinite(lo) & np.isfinite(hi)
    # Scaling first also handles intervals spanning the floating-point range.
    return np.subtract(
        1e-12 * hi, 1e-12 * lo, where=finite, out=np.zeros(np.shape(finite))
    )


def _astropy(value: Any) -> bool:
    module = sys.modules.get("astropy.units")
    return module is not None and isinstance(value, module.Quantity)


def _pair(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[1], str)
        and not isinstance(value[0], str)
    )


def _is_carrier(value: Any) -> bool:
    return isinstance(value, (str, Quantity)) or _pair(value) or _astropy(value)


def _split(value: Any) -> tuple[np.ndarray, str]:
    """Numbers and unit of one value with a unit."""
    if isinstance(value, str):
        number, unit = parse(value)
        return np.array([number]), unit
    if isinstance(value, Quantity):
        return np.atleast_1d(np.asarray(value.value, dtype=float)), value.unit
    if _pair(value):
        return np.atleast_1d(np.asarray(value[0], dtype=float)), value[1]
    return np.atleast_1d(np.asarray(value.value, dtype=float)), value.unit.to_string()


def _items(values: Any) -> list:
    if _is_carrier(values) and not isinstance(values, str):
        return [values]
    if isinstance(values, str):
        return [values]
    return (
        list(values)
        if isinstance(values, (list, tuple))
        else list(np.atleast_1d(np.asarray(values, dtype=object)))
    )


def _canonical(
    values: Any, unit: str | None, what: str, dimension: str | None = None
) -> tuple[np.ndarray, str | None]:
    """Values in the canonical unit of their dimension, and the unit given (``None`` for bare numbers).

    ``dimension``, when given, is the dimension the values must have.
    """
    items = _items(values)
    carriers = [_is_carrier(v) for v in items]
    if unit is not None:
        if any(carriers):
            raise UnitError(
                f"{what} gives units in the values and unit={unit!r}. Drop unit=, or pass plain numbers."
            )
        found = dimension_of(unit)
        if found is None or (dimension is not None and found != dimension):
            raise UnitError(
                f"{what}: unit {unit!r} is not a {dimension or 'named dimension'}"
            )
        return (
            np.asarray(
                convert(np.asarray(items, dtype=float), unit, CANONICAL[found]),
                dtype=float,
            ),
            unit,
        )
    if not any(carriers):
        return np.asarray(items, dtype=float).ravel(), None
    if not all(carriers):
        raise UnitError(
            f"{what} mixes values with and without units. Give every value with its unit."
        )
    parsed = [_split(v) for v in items]
    dims = {dimension_of(u) for _, u in parsed}
    if len(dims) != 1 or None in dims:
        raise UnitError(
            f"{what} mixes units of different dimensions: {sorted({u for _, u in parsed})}"
        )
    found = dims.pop()
    if dimension is not None and found != dimension:
        raise UnitError(
            f"{what} has unit {parsed[0][1]} ({found}), and it needs a {dimension}"
        )
    out = np.concatenate(
        [np.asarray(convert(x, u, CANONICAL[found]), dtype=float) for x, u in parsed]
    )
    return out, parsed[0][1]


def _like(values: Any, unit: str | None, given: str | None, what: str) -> np.ndarray:
    """Values that share the dimension of the reference points (affine half-widths, supports)."""
    if unit is not None:
        return _canonical(values, unit, what)[0]
    items = _items(values)
    if given is None:
        if any(_is_carrier(v) for v in items):
            raise UnitError(
                f"{what} has a unit, and the reference points are bare numbers. Give both with units."
            )
        return np.asarray(items, dtype=float).ravel()
    if not all(_is_carrier(v) for v in items):
        raise UnitError(
            f'{what} needs a unit like the reference points, for example "1 {given}".'
        )
    return _canonical(items, None, what, dimension=dimension_of(given))[0]


def _orders(order: Any, n: int) -> tuple[int, ...]:
    single = isinstance(order, (int, np.integer)) and not isinstance(order, bool)
    orders = (order,) * n if single else tuple(order)
    bad = any(
        not isinstance(o, (int, np.integer)) or isinstance(o, bool) or o < 0
        for o in orders
    )
    if len(orders) != n or bad:
        raise ValueError(
            f"order must be a non-negative int or {n} non-negative ints, not {order!r}"
        )
    return tuple(int(o) for o in orders)


def _count(n: Any, what: str = "the number of cells") -> int:
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)) or n < 1:
        raise ValueError(f"{what} must be a positive int, not {n!r}")
    return int(n)


def _differs(value: Any, default: Any) -> bool:
    if default is None:
        return value is not None
    try:
        return bool(value != default)
    except (TypeError, ValueError):
        return True


def _spelling(name: str, args: tuple, kwargs: dict, defaults: dict) -> tuple:
    """``(constructor, shown arguments, raw positional arguments, suffix of derived calls)``."""
    shown = tuple(repr(a) for a in args)
    extra = tuple(
        f"{k}={v!r}"
        for k, v in kwargs.items()
        if k in defaults and _differs(v, defaults[k])
    )
    scalars = all(
        isinstance(a, (int, float, str, np.integer, np.floating)) for a in args
    )
    raw = []
    if scalars:
        for arg in args:
            if isinstance(arg, np.integer):
                arg = int(arg)
            elif isinstance(arg, np.floating):
                arg = float(arg)
            raw.append(arg)
    return (name, shown + extra, tuple(raw), "")


def _reference(
    reference: Any, unit: str | None, given: str | None = None
) -> np.ndarray | None:
    """Reference points in the unit of the edges: bare numbers with bare edges or ``unit=``, else with units."""
    if reference is None:
        return None
    return _like(reference, unit, given, "reference")
