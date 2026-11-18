"""Boundary unit conversion and the parent value ``Quantity`` (design/0.8/SPEC.md Sec. 2.4).

LABEL ``[extension]``: the manuscript states its quantities in physical units and has no unit layer.

A dimensional scalar is a string such as ``"5 uG"``, ``"19 Myr"`` or ``"60 deg"``. A dimensional
array is a pair ``(array, "uG")``. An astropy ``Quantity`` is accepted through an ``isinstance``
check, without importing astropy. Values convert once, at the boundary, to the canonical unit of
their dimension (``CANONICAL``), and kernels see floats. Bare numbers are read in the canonical
unit only for the dimensions of ``BARE`` (Lorentz factors and cosines, Faraday depths in
rad m^-2, columns in cm^-2). Any other bare number raises ``UnitError`` with the spelling to use.

Units are products of powers of four base units, cm, s, G and rad. A unit expression is a
sequence of names with optional exponents (``"rad m^-2"``, ``"rad/m^2"``, ``"cm-3"``,
``"G s"``). ``Quantity(value, unit)`` is a pytree whose value is an array leaf and whose unit is
static, so it passes through ``jit``, ``vmap`` and ``grad``. Sums need equal dimensions, and
products and powers combine the units. A plain number in a dimensional sum is refused.

Shapes: conversions keep the shape of the input. Not certified: the unit table covers the
units listed in ``UNITS`` only, and ``yr`` is the Julian year.
"""

from __future__ import annotations

import math
import re
import sys
from fractions import Fraction
from typing import Any

import equinox as eqx
import numpy as np

__all__ = [
    "BARE",
    "CANONICAL",
    "EXAMPLE",
    "Quantity",
    "SyncEMError",
    "UNITS",
    "UnitError",
    "convert",
    "dimension_of",
    "parse",
    "to_canonical",
]


class SyncEMError(ValueError):
    """Base class of the errors of the 0.8 layer. Each message names the remedy."""


class UnitError(SyncEMError):
    """A value without a unit where one is needed, or with a unit of the wrong dimension."""


# fmt: off
_PC_CM = 3.0856775814913673e18
_YR_S = 365.25 * 86400.0
_L, _T, _F, _A = (1, 0, 0, 0), (0, 1, 0, 0), (0, 0, 1, 0), (0, 0, 0, 1)
_HZ = (0, -1, 0, 0)

# name -> (factor to the base units cm, s, G, rad; exponents over (cm, s, G, rad))
UNITS: dict[str, tuple[float, tuple[int, int, int, int]]] = {
    "Hz": (1.0, _HZ), "kHz": (1e3, _HZ), "MHz": (1e6, _HZ), "GHz": (1e9, _HZ), "THz": (1e12, _HZ),
    "G": (1.0, _F), "mG": (1e-3, _F), "uG": (1e-6, _F), "µG": (1e-6, _F), "nG": (1e-9, _F),
    "T": (1e4, _F), "mT": (10.0, _F), "uT": (1e-2, _F), "µT": (1e-2, _F), "nT": (1e-5, _F),
    "rad": (1.0, _A), "deg": (math.pi / 180, _A), "arcmin": (math.pi / 10800, _A), "arcsec": (math.pi / 648000, _A),
    "cm": (1.0, _L), "m": (100.0, _L), "km": (1e5, _L), "mm": (0.1, _L), "um": (1e-4, _L), "µm": (1e-4, _L),
    "nm": (1e-7, _L), "au": (1.495978707e13, _L), "pc": (_PC_CM, _L), "kpc": (1e3 * _PC_CM, _L),
    "Mpc": (1e6 * _PC_CM, _L),
    "s": (1.0, _T), "yr": (_YR_S, _T), "kyr": (1e3 * _YR_S, _T), "Myr": (1e6 * _YR_S, _T), "Gyr": (1e9 * _YR_S, _T),
}
_DIMENSIONS: dict[tuple, str] = {
    (0, 0, 0, 0): "dimensionless", _HZ: "frequency", _F: "field", _A: "angle", (-2, 0, 0, 1): "depth",
    (-2, 0, 0, 0): "column", (-3, 0, 0, 0): "density", _L: "length", _T: "time",
}
CANONICAL = {
    "frequency": "Hz", "field": "G", "angle": "rad", "depth": "rad m^-2", "column": "cm^-2",
    "density": "cm^-3", "length": "cm", "time": "s", "dimensionless": "",
}
EXAMPLE = {
    "frequency": "GHz", "field": "uG", "angle": "deg", "depth": "rad m^-2", "column": "cm^-2",
    "density": "cm^-3", "length": "pc", "time": "Myr", "dimensionless": "",
}
BARE = frozenset({"dimensionless", "depth", "column"})
_ASTROPY = {
    "frequency": "Hz", "field": "G", "angle": "rad", "depth": "rad / m2", "column": "1 / cm2",
    "density": "1 / cm3", "length": "cm", "time": "s", "dimensionless": "",
}
# fmt: on
_BASE_NAMES = ("cm", "s", "G", "rad")

_NUMBER = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
_SCALAR = re.compile(rf"^\s*({_NUMBER})\s*(.*?)\s*$")
_TOKEN = re.compile(r"^([A-Za-zµ]+)(?:\^?([-+]?\d+(?:\.\d+)?))?$")


def _unit(text: str) -> tuple[float, tuple[Fraction, ...]]:
    """Factor to the base units and exponents of a unit expression."""
    factor, exps = 1.0, [Fraction(0)] * 4
    sign = 1
    for part in re.split(r"(/)|\s+|\*", text.strip().replace("μ", "µ")):
        if not part or part == "1":
            continue
        if part == "/":
            sign = -1
            continue
        match = _TOKEN.match(part)
        if match is None or match.group(1) not in UNITS:
            known = ", ".join(sorted({k for k in UNITS if k.isascii()}))
            raise UnitError(f"unknown unit {part!r} in {text!r}. Known units: {known}.")
        f, e = UNITS[match.group(1)]
        power = sign * Fraction(match.group(2) or "1").limit_denominator(1000)
        factor *= f ** float(power)
        exps = [a + power * b for a, b in zip(exps, e)]
    return factor, tuple(exps)


def dimension_of(unit: str) -> str | None:
    """Named dimension of a unit, or ``None`` for a product without a name (``"G^2"``)."""
    return _DIMENSIONS.get(_unit(unit)[1])


def _rescale(value: Any, source: str, target: str) -> Any:
    """``value * f_source / f_target``: the identity for equal factors, else one multiplication by the ratio
    when it is at least 1 and one division by its inverse otherwise, so that powers of ten round-trip.
    """
    fs, es = _unit(source)
    ft, et = _unit(target)
    if es != et:
        raise UnitError(
            f"cannot convert {source!r} to {target!r}: the dimensions differ"
        )
    if fs == ft:
        return value
    return value * (fs / ft) if fs >= ft else value / (ft / fs)


def convert(value: Any, source: str, target: str) -> Any:
    """``value`` in ``source`` expressed in ``target`` (same dimension)."""
    return _rescale(value, source, target)


def parse(text: str) -> tuple[float, str]:
    """``"5 uG"`` -> ``(5.0, "uG")``. A number alone gives the empty unit."""
    match = _SCALAR.match(text) if isinstance(text, str) else None
    if match is None:
        raise UnitError(f'{text!r} is not a number followed by a unit, such as "5 uG".')
    unit = match.group(2)
    if unit:
        _unit(unit)
    return float(match.group(1)), unit


def _format(exps: tuple[Fraction, ...]) -> str:
    parts = []
    for name, x in zip(_BASE_NAMES, exps):
        if x == 1:
            parts.append(name)
        elif x != 0:
            parts.append(
                f"{name}^{x.numerator if x.denominator == 1 else repr(float(x))}"
            )
    return " ".join(parts)


def _astropy_quantity(value: Any) -> bool:
    astropy_units = sys.modules.get("astropy.units")
    return astropy_units is not None and isinstance(value, astropy_units.Quantity)


def _traced(value: Any) -> bool:
    return hasattr(value, "aval")


def _numbers(value: Any) -> Any:
    """A float, a NumPy array, or a JAX array left as it is."""
    if _traced(value):
        return value
    return float(value) if np.ndim(value) == 0 else np.asarray(value, dtype=float)


def _bare_error(value: Any, dimension: str, name: str) -> UnitError:
    example = EXAMPLE[dimension]
    if isinstance(value, (int, float, np.integer, np.floating)) and not isinstance(
        value, bool
    ):
        return UnitError(
            f'{name}={value!r} has no unit. Write {name}="{float(value):g} {example}".'
        )
    if np.ndim(value) > 0 and not isinstance(value, (str, dict)):
        return UnitError(
            f'{name} is an array without a unit. Write {name}=(array, "{example}").'
        )
    return UnitError(
        f'{name}={value!r} is not a value with a unit. Write {name}="1 {example}" or (array, "{example}").'
    )


def to_canonical(value: Any, dimension: str, *, name: str) -> Any:
    """``value`` (string, pair, ``Quantity``, astropy quantity or bare number) in ``CANONICAL[dimension]``."""
    canonical = CANONICAL[dimension]
    if isinstance(value, Quantity):
        unit, number = value.unit, value.value
    elif _astropy_quantity(value):
        astropy_units = sys.modules["astropy.units"]
        try:
            out = value.to_value(astropy_units.Unit(_ASTROPY[dimension]))
        except astropy_units.UnitConversionError as err:
            raise UnitError(
                f"{name}={value!r} cannot be read as a {dimension}: {err}"
            ) from err
        return _numbers(out)
    elif isinstance(value, str):
        number, unit = parse(value)
    elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        number, unit = _numbers(value[0]), value[1]
    else:
        if dimension not in BARE or not _is_number(value):
            raise _bare_error(value, dimension, name)
        return _numbers(value)
    if unit == "":
        if dimension not in BARE:
            raise _bare_error(number, dimension, name)
        return number
    found = dimension_of(unit)
    if found != dimension:
        shown = f'"{value}"' if isinstance(value, str) else repr(value)
        raise UnitError(
            f"{name}={shown} has unit {unit} ({found or 'a product of units'}). "
            f"{name} needs a {dimension}, for example in {EXAMPLE[dimension] or 'bare numbers'}."
        )
    return _rescale(number, unit, canonical)


def _is_number(value: Any) -> bool:
    if _traced(value):
        return True
    try:
        return np.asarray(value).dtype.kind in "iuf"
    except (TypeError, ValueError):
        return False


class Quantity(eqx.Module):
    """Parent value inside a conditional: ``value`` (array leaf) in ``unit`` (static)."""

    value: Any
    unit: str = eqx.field(static=True)
    __array_ufunc__ = None  # NumPy defers to the reflected operators of Quantity

    def __init__(self, value: Any, unit: str):
        if isinstance(value, str):
            raise UnitError(
                'Quantity takes a number or an array, not a string. Use se.Quantity(5.0, "uG").'
            )
        _unit(unit)
        self.value = value
        self.unit = unit

    @property
    def _exps(self) -> tuple[Fraction, ...]:
        return _unit(self.unit)[1]

    def to(self, unit: str) -> "Quantity":
        return Quantity(_rescale(self.value, self.unit, unit), unit)

    def _other(self, other: Any, operation: str) -> "Quantity":
        if isinstance(other, Quantity):
            return other
        if isinstance(other, str):
            return Quantity(*parse(other))
        if any(self._exps):
            raise UnitError(
                f"{operation} of a plain number {other!r} and a parent value in {self.unit}: plain floats are "
                f'refused in a dimensional slot. Use the parent value, or a string such as "1 {self.unit}".'
            )
        return Quantity(other, "")

    def _same(self, other: Any, operation: str) -> Any:
        o = self._other(other, operation)
        if o._exps != self._exps:
            raise UnitError(
                f"{operation} of {o.unit} and {self.unit}: the dimensions differ"
            )
        return _rescale(o.value, o.unit, self.unit)

    def __add__(self, other: Any) -> "Quantity":
        return Quantity(self.value + self._same(other, "sum"), self.unit)

    __radd__ = __add__

    def __sub__(self, other: Any) -> "Quantity":
        return Quantity(self.value - self._same(other, "difference"), self.unit)

    def __rsub__(self, other: Any) -> "Quantity":
        return Quantity(self._same(other, "difference") - self.value, self.unit)

    def __neg__(self) -> "Quantity":
        return Quantity(-self.value, self.unit)

    def __abs__(self) -> "Quantity":
        return Quantity(abs(self.value), self.unit)

    def _product(self, other: Any, power: int) -> "Quantity":
        other = other if isinstance(other, Quantity) else Quantity(*parse(other))
        fs, es = _unit(self.unit)
        fo, eo = _unit(other.unit)
        value = self.value * fs * (other.value * fo) ** power
        return Quantity(value, _format(tuple(a + power * b for a, b in zip(es, eo))))

    def __mul__(self, other: Any) -> "Quantity":
        if isinstance(other, (Quantity, str)):
            return self._product(other, 1)
        return Quantity(self.value * other, self.unit)

    __rmul__ = __mul__

    def __truediv__(self, other: Any) -> "Quantity":
        if isinstance(other, (Quantity, str)):
            return self._product(other, -1)
        return Quantity(self.value / other, self.unit)

    def __rtruediv__(self, other: Any) -> "Quantity":
        numerator = (
            Quantity(*parse(other)) if isinstance(other, str) else Quantity(other, "")
        )
        return numerator / self

    def __pow__(self, power: Any) -> "Quantity":
        factor, exps = _unit(self.unit)
        if not any(exps):
            return Quantity((self.value * factor) ** power, "")
        if _traced(power) or not isinstance(
            power, (int, float, np.integer, np.floating)
        ):
            raise UnitError(
                f"the power of a parent value in {self.unit} must be a concrete number, because it sets the unit"
            )
        p = Fraction(float(power)).limit_denominator(1000)
        return Quantity(
            (self.value * factor) ** power, _format(tuple(x * p for x in exps))
        )

    def __rpow__(self, base: Any) -> "Quantity":
        if any(self._exps):
            raise UnitError(
                f"an exponent in {self.unit} is not a number. Divide by a reference value first."
            )
        return Quantity(base ** (self.value * _unit(self.unit)[0]), "")

    def __lt__(self, other: Any) -> Any:
        return self.value < self._same(other, "comparison")

    def __le__(self, other: Any) -> Any:
        return self.value <= self._same(other, "comparison")

    def __gt__(self, other: Any) -> Any:
        return self.value > self._same(other, "comparison")

    def __ge__(self, other: Any) -> Any:
        return self.value >= self._same(other, "comparison")

    def __float__(self) -> float:
        if any(self._exps):
            raise UnitError(
                f"a parent value in {self.unit} is not a plain number. Use .to(unit).value."
            )
        return float(self.value * _unit(self.unit)[0])

    def __repr__(self) -> str:
        return f"Quantity({self.value!r}, {self.unit!r})"
