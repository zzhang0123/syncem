"""Variables of a base model and the placement check of declared values (design/0.8/SPEC.md Sec. 2.4, 3.2).

LABEL ``[extension]``. A ``Variable`` names one argument of a kernel: its kind (``positive``, ``real``,
``cosine``, ``angle``), its canonical unit, whether a bare number is read in that unit, its usual
range and the name of the neutral declaration used when it is undeclared. ``SYNCHROTRON_VARIABLES``
holds the six variables of the synchrotron kernels, the names of ``VARS`` in
``syncem.model._factor``.

``Variable.place(value)`` is the check at placement in ``Population(**factors)`` or
``Lattice(**axes)``. A point value (a number, a string such as ``"5 uG"``, a pair, a ``Quantity`` or
an astropy quantity) is converted to the canonical unit and checked against the kind. A
declaration object (a distribution or ``Cells``) is checked through the placement protocol of
``Placeable`` and returned unchanged, because it converts its own values in its constructor. An
angle unit on a cosine is accepted on a point value only, read as its cosine.

The undeclared default is a name (``"Isotropic"``), not an instance, because distributions are a
later layer. The planning layer resolves it. Units: canonical units of ``syncem.units``. Shapes:
a point value keeps its shape (a scalar gives a float). Not certified: the usual ranges are
conventions for a warning line, not physical limits.
"""

from __future__ import annotations

from typing import Any, Literal, Protocol, runtime_checkable

import equinox as eqx
import numpy as np

from ..units import Quantity, UnitError, dimension_of, parse, to_canonical

__all__ = ["KINDS", "Placeable", "SYNCHROTRON_VARIABLES", "Variable"]

KINDS = ("positive", "real", "cosine", "angle")
DIMENSION_NAMES = {
    "frequency": "frequency",
    "field": "field strength",
    "angle": "angle",
    "depth": "Faraday depth",
    "column": "column density",
    "density": "number density",
    "length": "length",
    "time": "time",
    "dimensionless": "a dimensionless quantity",
}


@runtime_checkable
class Placeable(Protocol):
    """Placement protocol of declaration objects (distributions, ``Cells``).

    ``given_unit`` is the unit the object was given, ``None`` for bare numbers. ``is_point`` is true
    for a point value such as a Dirac distribution. Optional members: ``carries_values`` (default
    true) is false for a declaration that holds no values of the variable (``AzimuthFactor``,
    ``UniformAzimuth``, ``FreeMoments``), which needs no unit. ``as_cosine()`` returns a point
    given as an angle converted to its cosine. ``unit_remedy(unit)`` returns the sentence that
    tells how to give the unit, ``angle_remedy()`` the sentence for a range of angles on a cosine,
    and ``noun`` names the object in messages (default ``"distribution"``).
    """

    given_unit: str | None
    is_point: bool


class Variable(eqx.Module):
    """One variable of a kernel. All fields are static, so a ``Variable`` is hashable."""

    name: str = eqx.field(static=True)
    meaning: str = eqx.field(static=True)
    kind: Literal["positive", "real", "cosine", "angle"] = eqx.field(static=True)
    unit: str = eqx.field(static=True)
    bare: bool = eqx.field(static=True)
    usual: tuple[float, float] | None = eqx.field(static=True)
    default: str | None = eqx.field(static=True)
    display_unit: str = eqx.field(static=True)

    def __check_init__(self) -> None:
        if self.kind not in KINDS:
            raise ValueError(
                f"Variable {self.name!r}: kind must be one of {KINDS}, not {self.kind!r}"
            )
        if dimension_of(self.unit) is None:
            raise ValueError(
                f"Variable {self.name!r}: unit {self.unit!r} has no named dimension"
            )

    @property
    def dimension(self) -> str:
        return dimension_of(self.unit)

    def place(self, value: Any) -> Any:
        """Canonical float for a point value, the object itself for a declaration object."""
        if isinstance(value, Placeable) and not isinstance(value, Quantity):
            return self._place_object(value)
        if isinstance(value, str):
            number, unit = parse(value)
            shown = f'"{value}"'
        elif isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
            number, unit, shown = value[0], value[1], repr(value)
        elif isinstance(value, Quantity):
            number, unit, shown = value.value, value.unit, repr(value)
        else:
            number, unit, shown = value, None, repr(value)
        if unit == "" or (
            unit is None and np.ndim(number) == 0 and not hasattr(value, "unit")
        ):
            return self._check_kind(self._bare(number, shown), shown)
        if unit is not None and self.kind == "cosine" and dimension_of(unit) == "angle":
            return self._check_kind(
                np.cos(to_canonical(value, "angle", name=self.name)), shown
            )
        if unit is not None and dimension_of(unit) != self.dimension:
            raise self._dimension_error(shown, unit)
        return self._check_kind(
            to_canonical(value, self.dimension, name=self.name), shown
        )

    def outside_usual(self, value: float) -> str | None:
        """One warning line when ``value`` (canonical unit) lies outside the usual range, else ``None``."""
        if self.usual is None or self.usual[0] <= value <= self.usual[1]:
            return None
        unit = f" {self.unit}" if self.unit else ""
        lo, hi = self.usual
        return f"{self.name} = {value:g}{unit} is outside the usual range {lo:g} to {hi:g}{unit}"

    # ------------------------------------------------------------ helpers
    def _bare(self, number: Any, shown: str) -> float:
        if self.bare:
            return float(number)
        x, unit = float(number), self.display_unit
        raise UnitError(
            f'{self.name}={shown} has no unit. Write {self.name}="{x:g} {unit}", '
            f'or se.Uniform({0.8 * x:g}, {1.2 * x:g}, unit="{unit}") for a range.'
        )

    def _check_kind(self, x: Any, shown: str) -> Any:
        values = np.asarray(x, dtype=float)
        if not np.all(np.isfinite(values)):
            raise ValueError(f"{self.name}={shown} is not finite")
        if self.kind == "positive" and np.any(values <= 0):
            raise ValueError(f"{self.name}={shown} must be positive ({self.meaning})")
        if self.kind == "cosine" and np.any(np.abs(values) > 1):
            raise ValueError(f"{self.name}={shown} is a cosine and must lie in [-1, 1]")
        return x

    def _place_object(self, value: Any) -> Any:
        given = value.given_unit
        if not getattr(value, "carries_values", True):
            return value
        if given is None:
            if self.bare:
                return value
            remedy = getattr(value, "unit_remedy", None)
            text = (
                remedy(self.display_unit)
                if remedy
                else f'Pass unit="{self.display_unit}".'
            )
            raise UnitError(f"{self.name}={value!r} has no unit. {text}")
        found = dimension_of(given)
        if found == self.dimension:
            return value
        if self.kind == "cosine" and found == "angle":
            convert = getattr(value, "as_cosine", None)
            if value.is_point and convert is not None:
                return convert()
            if value.is_point:
                raise UnitError(
                    f"{self.name}={value!r} is a point given as an angle on a cosine, and it cannot convert "
                    'itself to its cosine. Give the point as a cosine, or as a string such as "60 deg".'
                )
            remedy = getattr(value, "angle_remedy", None)
            text = (
                remedy() if remedy else "Give the range in the cosine, without a unit."
            )
            raise UnitError(
                f"{self.name}={value!r} is a range of angles on a cosine. {text}"
            )
        raise self._dimension_error(
            repr(value), given, noun=getattr(value, "noun", "distribution")
        )

    def _dimension_error(
        self, shown: str, unit: str, noun: str | None = None
    ) -> UnitError:
        found = dimension_of(unit)
        other = next(
            (
                v
                for v in SYNCHROTRON_VARIABLES.values()
                if v.dimension == found
                and v.name != self.name
                and found != "dimensionless"
            ),
            None,
        )
        if noun is not None and other is not None:
            return UnitError(
                f"{self.name}={shown} is a {noun} of {DIMENSION_NAMES[found]}. "
                f"{self.name} is the {self.meaning}. Write {other.name}=."
            )
        wanted = (
            f"in {self.unit} or {self.display_unit}" if self.unit else "without a unit"
        )
        return UnitError(
            f"{self.name}={shown} has unit {unit} ({found or 'a product of units'}). "
            f"{self.name} is the {self.meaning}, {wanted}."
        )


SYNCHROTRON_VARIABLES: dict[str, Variable] = {
    v.name: v
    for v in (
        Variable(
            "gamma",
            "electron Lorentz factor",
            "positive",
            "",
            True,
            (1.0, 1e8),
            None,
            "",
        ),
        Variable(
            "B", "field strength", "positive", "G", False, (1e-9, 1e-3), None, "uG"
        ),
        Variable("mu", "pitch cosine", "cosine", "", True, None, "Isotropic", ""),
        Variable(
            "eta", "field-inclination cosine", "cosine", "", True, None, "Isotropic", ""
        ),
        Variable(
            "phi",
            "projected-field azimuth",
            "angle",
            "rad",
            False,
            None,
            "UniformAzimuth",
            "deg",
        ),
        Variable(
            "depth",
            "Faraday depth from emitter to observer",
            "real",
            "rad m^-2",
            True,
            (-1e4, 1e4),
            "NoRotation",
            "rad m^-2",
        ),
    )
}
