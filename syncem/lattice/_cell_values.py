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
