"""Parameters and declared parents of a distribution (SPEC Sec. 3.5).

LABEL [extension].

A Free declaration marks a coordinate of scalar shape with optional units. A
conditional accepts the named scalar parents and returns one law. These
records have no certified physical range beyond the declared bounds.
"""

from __future__ import annotations

import inspect
from dataclasses import fields, is_dataclass
from typing import Any, Callable

import equinox as eqx
import numpy as np

from ..base._fingerprint import NotCacheable, digest
from ..base._function_fingerprint import function_fingerprint
from ..model._provenance import strict_json
from ..units import CANONICAL, Quantity, dimension_of, parse, to_canonical

__all__ = ["Conditional", "Free"]


class Free(eqx.Module):
    """A scalar parameter to be supplied by a fit."""

    start: Any = eqx.field(static=True, default=None)
    bounds: tuple | None = eqx.field(static=True, default=None)
    prior: Any = eqx.field(static=True, default=None)
    log: bool = eqx.field(static=True, default=False)
    shared: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        if self.bounds is not None and len(self.bounds) != 2:
            raise ValueError("Free bounds must give a lower and an upper limit")
        if not isinstance(self.log, bool) or not isinstance(self.shared, bool):
            raise TypeError("Free log and shared must be bools")

    def record(self) -> dict:
        return {
            "kind": "free",
            "start": strict_json(self.start),
            "bounds": strict_json(self.bounds),
            "prior": None if self.prior is None else self.prior.record(),
            "log": self.log,
            "shared": self.shared,
        }


def free_fields(declaration):
    """Free instance fields of a law, excluding class constants and properties."""
    if not is_dataclass(declaration):
        return {}
    return {
        field.name: value
        for field in fields(declaration)
        if isinstance(value := getattr(declaration, field.name), Free)
    }


def parameter_unit(marker, value=None):
    """Canonical unit when a parameter declaration uses dimensional values."""
    for item in (marker.start, *(marker.bounds or ()), value):
        if isinstance(item, str):
            _, unit = parse(item)
        elif isinstance(item, Quantity):
            unit = item.unit
        elif isinstance(item, tuple) and len(item) == 2 and isinstance(item[1], str):
            unit = item[1]
        else:
            continue
        return CANONICAL[dimension_of(unit)]
    return None


def parameter_number(value, unit, name):
    """Read an optimizer coordinate in the parameter's canonical units."""
    if unit is None or not isinstance(value, (str, Quantity, tuple)):
        return float(value)
    return float(to_canonical(value, dimension_of(unit), name=name))


def parameter_argument(marker, value):
    """Restore the unit when a canonical coordinate is passed to a declaration factory."""
    if not isinstance(marker, Free) or isinstance(value, (str, tuple, Quantity, Free)):
        return value
    unit = parameter_unit(marker)
    return value if unit is None else Quantity(value, unit)


class Conditional(eqx.Module):
    """A law factory with explicit parameter values and declared parent names.

    Names in params are scalar parameters. Other named arguments are parents.
    A compiler can supply canonical units through evaluate(). Explicit units
    on the declaration override the canonical units.
    """

    fn: Callable = eqx.field(static=True)
    params: dict
    units: tuple = eqx.field(static=True)
    parity: tuple = eqx.field(static=True)
    name: str = eqx.field(static=True)
    parents: tuple[str, ...] = eqx.field(static=True)

    def __init__(self, fn, *, params=None, units=None, parity=None, name=None):
        self.fn = fn
        self.params = dict(params or {})
        self.units = tuple((units or {}).items())
        self.parity = tuple((parity or {}).items())
        self.name = name or getattr(fn, "__name__", type(fn).__name__)
        args = inspect.signature(fn).parameters
        if any(
            p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            for p in args.values()
        ):
            raise TypeError(
                "Conditional needs named arguments for every parent and parameter"
            )
        unknown = set(self.params) - set(args)
        if unknown:
            raise TypeError(
                f"Conditional parameters are absent from the function: {sorted(unknown)}"
            )
        self.parents = tuple(key for key in args if key not in self.params)
        if set(dict(self.units)) - set(self.parents) or set(dict(self.parity)) - set(
            self.parents
        ):
            raise ValueError("Conditional units and parity may name only parents")
        if any(mode not in ("even", "odd") for _, mode in self.parity):
            raise ValueError("Conditional parity must be even or odd")

    def evaluate(self, parents, *, params=None, canonical_units=None):
        parameters = {**self.params, **(params or {})}
        if any(isinstance(value, Free) for value in parameters.values()):
            raise ValueError(
                "resolve the Free parameters before evaluating the conditional"
            )
        units = {**(canonical_units or {}), **dict(self.units)}
        args = {}
        for name in self.parents:
            value = parents[name]
            if name in units and isinstance(value, Quantity):
                value = value.to(units[name])
            elif name in units:
                source = (canonical_units or {}).get(name, units[name])
                value = Quantity(value, source).to(units[name])
            args[name] = value
        return self.fn(**args, **parameters)

    def record(self) -> dict:
        parameters = {
            key: val.record() if isinstance(val, Free) else _parameter_value(val)
            for key, val in self.params.items()
        }
        try:
            fingerprint = digest(
                (
                    function_fingerprint(self.fn),
                    self.parents,
                    self.units,
                    self.parity,
                    parameters,
                )
            )
        except NotCacheable:
            fingerprint = None
        return {
            "kind": "conditional",
            "name": self.name,
            "parents": list(self.parents),
            "units": dict(self.units),
            "parity": dict(self.parity),
            "params": parameters,
            "fingerprint": fingerprint,
        }


def _parameter_value(value):
    if isinstance(value, Quantity):
        return {"value": _parameter_value(value.value), "unit": value.unit}
    if np.iscomplexobj(value):
        return {
            "real": strict_json(np.real(value)),
            "imag": strict_json(np.imag(value)),
        }
    return strict_json(value)
