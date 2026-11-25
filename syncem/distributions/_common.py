"""Shared declaration checks and quadrature for the 0.8 laws."""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar, Protocol

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..base._fingerprint import digest
from ..model._provenance import strict_json
from ..units import CANONICAL, Quantity, UnitError, dimension_of, parse, to_canonical
from .parameters import Free


class Distribution(Protocol):
    """A law on a variable or a group of variables (SPEC Sec. 3.5).

    The slot supplies the variable name for an unnamed law. A density law
    offers rule(n=None, parents=None) and density(x, parents=None).
    A point law has a rule and an independent screen can have only a
    characteristic function.
    """

    variables: tuple[str, ...]
    support: tuple
    given_unit: str | None

    def record(self) -> dict: ...


class Law(eqx.Module):
    """Declaration metadata of the built-in laws."""

    variables: ClassVar[tuple[str, ...]] = ()
    given_unit: ClassVar[str | None] = None
    is_point: ClassVar[bool] = False
    carries_values: ClassVar[bool] = True
    fixed_nodes: ClassVar[bool] = False
    is_even: ClassVar[bool] = False

    def record(self) -> dict:
        """Strict JSON with the parameters, support and their content fingerprint."""
        parameters = {
            f.name: _record_value(getattr(self, f.name))
            for f in dataclasses.fields(self)
        }
        fields = jax.tree_util.tree_leaves(
            self, is_leaf=lambda value: isinstance(value, Free)
        )
        support = (
            None if any(isinstance(value, Free) for value in fields) else self.support
        )
        data = {
            "family": type(self).__name__,
            "variables": list(self.variables),
            "given_unit": self.given_unit,
            "support": _record_value(support),
            "parameters": parameters,
        }
        return {**data, "fingerprint": digest(data)}


def _record_value(value):
    if hasattr(value, "record"):
        return value.record()
    if np.iscomplexobj(value):
        return {
            "real": strict_json(jnp.real(value)),
            "imag": strict_json(jnp.imag(value)),
        }
    return strict_json(value)


def _unit_of(value: Any) -> str | None:
    if isinstance(value, Free):
        values = (value.start, *(value.bounds or ()))
        units = tuple(unit for bound in values if (unit := _unit_of(bound)) is not None)
        if len({dimension_of(unit) for unit in units}) > 1:
            raise UnitError("Free start and bounds must have one dimension")
        return units[0] if units else None
    if isinstance(value, Quantity):
        return value.unit
    if isinstance(value, str):
        return parse(value)[1] or None
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        return value[1]
    if hasattr(value, "unit") and hasattr(value, "to_value"):
        return str(value.unit)
    return None


def canonical(value: Any, *, unit: str | None = None, name: str = "value"):
    """One numeric argument and the unit supplied with it."""
    if isinstance(value, Free):
        return value, unit if unit is not None else _unit_of(value)
    found = _unit_of(value)
    given = unit if unit is not None else found
    if given is None:
        return jnp.asarray(value, dtype=float), None
    dimension = dimension_of(given)
    if dimension not in CANONICAL:
        raise UnitError(f"{name} needs a named dimension, got {given!r}")
    source = value if found is not None else (value, given)
    result = to_canonical(source, dimension, name=name)
    return jnp.asarray(result, dtype=float), given


def pair_values(a, b, *, unit=None, names=("lower bound", "upper bound")):
    """Two arguments in one dimension with a shared optional unit."""
    if unit is None:
        units = (_unit_of(a), _unit_of(b))
        if (units[0] is None) != (units[1] is None):
            dimension = dimension_of(units[0] or units[1])
            if dimension not in ("depth", "column", "dimensionless"):
                raise UnitError(
                    f"give units for both {names[0]} and {names[1]}, or pass unit="
                )
        unit = units[0] or units[1]
    first, given = canonical(a, unit=unit, name=names[0])
    second, _ = canonical(b, unit=unit, name=names[1])
    return scalar(first, names[0]), scalar(second, names[1]), given


def scalar(value: Any, name: str, *, allow_inf: bool = False):
    if isinstance(value, Free):
        return value
    x = jnp.asarray(value, dtype=float)
    if x.ndim:
        raise ValueError(f"{name} must be a scalar")
    require(~jnp.isnan(x) if allow_inf else jnp.isfinite(x), f"{name} must be finite")
    return x


def require(condition: Any, message: str) -> None:
    """Validate concrete parameters. The plan validates conditional tabulations before tracing."""
    if not isinstance(condition, jax.core.Tracer) and not np.all(condition):
        raise ValueError(message)


def count(value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"nodes must be a positive integer, got {value!r}")
    return int(value)


def node_count(override: int | None, default: int) -> int:
    return default if override is None else count(override)


def _legendre(n):
    x, w = np.polynomial.legendre.leggauss(n)
    return x, w


def interval(lo, hi, n):
    """Gauss–Legendre nodes and unnormalised weights on an interval."""
    t, w = _legendre(n)
    width = (hi - lo) / 2
    return lo + (t + 1) * width, w * width


def log_span(lo, hi):
    """Logarithmic separation of nearby or widely spaced positive bounds."""
    near = hi - lo <= 0.5 * lo
    relative = (jnp.where(near, hi, lo) - lo) / lo
    return jnp.where(near, jnp.log1p(relative), jnp.log(hi) - jnp.log(lo))


def log_interval(lo, hi, n):
    """Gauss nodes in log x with physical dx weights on a positive interval."""
    t, w = _legendre(n)
    near = hi - lo <= 0.5 * lo
    span = log_span(lo, hi)
    step = 0.5 * span * (1 + t)
    local = lo * jnp.expm1(jnp.where(near, step, 0.0))
    x = jnp.where(near, lo + local, jnp.exp(jnp.log(lo) + step))
    return x, 0.5 * span * w * x


def probabilities(nodes, weights):
    total = jnp.sum(weights)
    require(jnp.isfinite(total) & (total > 0), "rule must have positive finite mass")
    return nodes, weights / total


def bounded_density(x, lo, hi, f):
    """Evaluate a smooth density on its support, with a safe argument outside."""
    x = jnp.asarray(x, dtype=float)
    inside = (x >= lo) & (x <= hi)
    return jnp.where(inside, f(jnp.where(inside, x, 0.5 * lo + 0.5 * hi)), 0.0)


def concrete(*values) -> bool:
    return not any(isinstance(value, (Free, jax.core.Tracer)) for value in values)


def resolved(*values) -> None:
    if any(isinstance(value, Free) for value in values):
        raise ValueError("resolve the Free parameters before evaluating this law")
