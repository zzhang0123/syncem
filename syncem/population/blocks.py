"""Joint population declarations for discrete samples and unknown moments.

LABEL ``eq: energy cell response``. A sample block holds one row per joint
draw, with shape ``(n_samples,)`` for each variable and the weights. After
placement, coordinates use the declared variables' canonical units. A
sample column is in cm^-2. Unknown moment blocks declare their variables,
cell hints and support. The declarations do not certify the population
outside the supplied draws or support.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..lattice.cells import Cells
from ..model._provenance import strict_json
from ..units import CANONICAL, Quantity, UnitError, dimension_of, parse, to_canonical
from ..values.errors import RoleError

__all__ = ["FreeMoments", "NonNegativeCells", "Samples"]


def _names(variables):
    if any(not isinstance(name, str) or not name.strip() for name in variables):
        raise ValueError("block variable names must be nonempty strings")
    if len(set(variables)) != len(variables):
        raise ValueError("a joint block needs distinct variables")
    return tuple(variables)


def _definitions(variables):
    if isinstance(variables, Mapping):
        return variables
    return {variable.name: variable for variable in variables}


def _real(value, name):
    x = jnp.asarray(value)
    if x.dtype.kind not in "fiu":
        raise ValueError(f"{name} must contain real numbers")
    return x.astype(float)


def _checked(value, bad, message):
    if not isinstance(bad, jax.core.Tracer):
        if np.any(np.asarray(bad)):
            raise ValueError(message)
        return value
    return eqx.error_if(value, jnp.any(bad), message)


def _finite(value, name):
    return _checked(value, ~jnp.isfinite(value), f"{name} must be finite")


def _quantity_parts(value):
    if isinstance(value, str):
        out, unit = parse(value)
        return out, unit or None
    if isinstance(value, Quantity) or (
        hasattr(value, "unit") and hasattr(value, "to_value")
    ):
        return value.value, str(value.unit)
    if isinstance(value, tuple) and len(value) == 2 and isinstance(value[1], str):
        return value
    return value, None


def _coordinate(value, name):
    number, given = _quantity_parts(value)
    x = _real(number, name)
    if given is None:
        return _finite(x, name), None, None
    dimension = dimension_of(given)
    if dimension not in CANONICAL:
        raise UnitError(f"{name} needs a named unit, got {given!r}")
    canonical = _real(to_canonical((x, given), dimension, name=name), name)
    return _finite(canonical, name), given, CANONICAL[dimension]


def _placed_value(x, unit, definition, *, support=False):
    name = definition.name
    if not unit and not definition.bare:
        raise UnitError(
            f'{name} requires a unit. Give a value such as (values, "{definition.display_unit}").'
        )
    if unit is not None:
        dimension = dimension_of(unit)
        if definition.kind == "cosine" and dimension == "angle" and not support:
            x = jnp.cos(x)
        elif dimension != definition.dimension:
            raise UnitError(
                f"{name} takes {definition.unit or 'dimensionless values'}, got {unit}"
            )
    if definition.kind == "positive":
        bad = (x < 0) if support else (x <= 0)
        x = _checked(x, bad, f"{name} must be positive")
    if definition.kind == "cosine":
        x = _checked(x, jnp.abs(x) > 1, f"{name} is a cosine and must lie in [-1, 1]")
    return x


def _probabilities(weights):
    """Scale relative weights before summing so a finite total is representable."""
    if not isinstance(weights, jax.core.Tracer):
        host = np.asarray(weights)
        if np.any(~np.isfinite(host) | (host < 0)) or np.max(host) <= 0:
            raise ValueError(
                "sample weights must be finite, nonnegative and have positive mass"
            )
        scaled = host / host.max()
        return jnp.asarray(scaled / scaled.sum())
    valid = jnp.isfinite(weights) & (weights >= 0)
    weights = _checked(
        weights,
        (~jnp.all(valid)) | (jnp.max(weights) <= 0),
        "sample weights must be finite, nonnegative and have positive mass",
    )
    scaled = jax.lax.optimization_barrier(weights / jnp.max(weights))
    return scaled / jnp.sum(scaled)


class Samples(eqx.Module):
    """A joint set of weighted samples. The optional column carries the mass."""

    variables: tuple[str, ...] = eqx.field(static=True)
    given_units: tuple[tuple[str, str | None], ...] = eqx.field(static=True)
    units: tuple[tuple[str, str | None], ...] = eqx.field(static=True)
    nodes: dict[str, Any]
    weights: Any
    column: Any = None

    def __init__(self, *, weights=None, column=None, **arrays):
        names = _names(arrays)
        if not names:
            raise ValueError("Samples needs at least one named variable")
        nodes = {}
        given_units, units = [], []
        for name, data in arrays.items():
            x, given, unit = _coordinate(data, name)
            nodes[name] = x
            given_units.append((name, given))
            units.append((name, unit))
        shape = nodes[names[0]].shape
        if (
            len(shape) != 1
            or shape[0] == 0
            or any(value.shape != shape for value in nodes.values())
        ):
            raise ValueError(
                "sample arrays must be nonempty one-dimensional arrays of equal length"
            )
        relative = jnp.ones(shape) if weights is None else _real(weights, "weights")
        if relative.shape != shape:
            raise ValueError(
                "sample weights must have the same shape as the sample arrays"
            )
        self.variables = names
        self.nodes = nodes
        self.given_units = tuple(given_units)
        self.units = tuple(units)
        self.weights = _probabilities(relative)
        if column is not None:
            raw, unit = _quantity_parts(column)
            column_input = raw if unit is None else (raw, unit)
            column = _real(
                to_canonical(column_input, "column", name="sample column"),
                "sample column",
            )
            if column.ndim:
                raise ValueError("sample column must be a scalar")
            column = _checked(
                column,
                (column < 0) | ~jnp.isfinite(column),
                "sample column must be finite and nonnegative",
            )
        self.column = column

    @property
    def size(self) -> int:
        """Number of rows in the joint measure."""
        return self.weights.shape[0]

    @property
    def support(self) -> dict:
        """Coordinate extrema among nodes of positive weight, including point supports."""
        use = self.weights > 0
        return {
            name: (
                jnp.min(jnp.where(use, value, jnp.inf)),
                jnp.max(jnp.where(use, value, -jnp.inf)),
            )
            for name, value in self.nodes.items()
        }

    def placed(self, variables) -> Samples:
        """Check sample coordinates and convert pointwise angle inputs to cosines."""
        defs = _definitions(variables)
        unknown = sorted(set(self.variables) - set(defs))
        if unknown:
            raise RoleError(f"sample variables {unknown} are absent from the kernel")
        nodes = {
            name: _placed_value(self.nodes[name], unit, defs[name])
            for name, unit in self.units
        }
        result = object.__new__(type(self))
        for attr in ("variables", "given_units", "weights", "column"):
            object.__setattr__(result, attr, getattr(self, attr))
        object.__setattr__(result, "nodes", nodes)
        object.__setattr__(
            result,
            "units",
            tuple((name, defs[name].unit) for name in self.variables),
        )
        return result

    def rule(self):
        """Coordinate arrays and relative weights normalised to unit mass."""
        return dict(self.nodes), self.weights

    def mass_rule(self):
        """Coordinate arrays and column weights, with unit mass when no column is set."""
        nodes, weights = self.rule()
        return nodes, weights if self.column is None else weights * self.column

    def record(self) -> dict:
        """Strict JSON of the weighted joint measure."""
        return strict_json(
            {
                "family": "Samples",
                "variables": self.variables,
                "given_units": dict(self.given_units),
                "units": dict(self.units),
                "nodes": self.nodes,
                "weights": self.weights,
                "column": self.column,
            }
        )


def _hint_map(hint, variables, name):
    if hint is None:
        return None
    if isinstance(hint, Mapping):
        absent = set(hint) - set(variables)
        if variables and absent:
            raise RoleError(
                f"{name} declares variables outside the block {sorted(absent)}"
            )
        return dict(hint)
    if len(variables) > 1:
        raise RoleError(f"{name} for a joint block needs a variable-to-value mapping")
    return hint


def _hint_for(hint, name):
    return hint.get(name) if isinstance(hint, Mapping) else hint


def _record_hint(value):
    if isinstance(value, slice):
        return {"slice": (value.start, value.stop, value.step)}
    if isinstance(value, Cells):
        return {
            "chart": value.chart,
            "given_unit": value.given_unit,
            "edges": value.edges,
            "reference": value.reference,
            "half_width": value.half_width,
            "order": value.order,
        }
    if isinstance(value, Mapping):
        return {key: _record_hint(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return tuple(_record_hint(item) for item in value)
    if isinstance(value, Quantity):
        return value.value, value.unit
    if hasattr(value, "record"):
        return value.record()
    if hasattr(value, "to_dict"):
        return value.to_dict()
    return value


class _MomentBlock(eqx.Module):
    variables: tuple[str, ...] = eqx.field(static=True)
    cells: Any
    support: Any

    given_unit: ClassVar[None] = None
    is_point: ClassVar[bool] = False
    carries_values: ClassVar[bool] = False
    allows_selection: ClassVar[bool] = False

    def _set_hints(self, names, cells, support):
        self.variables = _names(names)
        self.cells = _hint_map(cells, self.variables, "cells")
        self.support = _hint_map(support, self.variables, "support")
        axes = self.cells.values() if isinstance(self.cells, Mapping) else (self.cells,)
        kinds = (Cells, slice) if self.allows_selection else (Cells,)
        if any(axis is not None and not isinstance(axis, kinds) for axis in axes):
            raise TypeError(
                "cells expects a Cells object or a mapping of variables to Cells"
            )

    def bind(self, name):
        """Resolve an unnamed marker in a keyword slot."""
        _names((name,))
        if self.variables == (name,):
            return self
        if self.variables:
            raise RoleError(
                f"the block on {self.variables} cannot occupy keyword {name!r}"
            )
        _hint_map(self.cells, (name,), "cells")
        _hint_map(self.support, (name,), "support")
        result = object.__new__(type(self))
        for attr, value in vars(self).items():
            object.__setattr__(result, attr, value)
        object.__setattr__(result, "variables", (name,))
        return result

    def selections(self):
        """Cell selections per variable, as slices over that variable's cells."""
        return {
            name: hint
            for name in self.variables
            if isinstance(hint := _hint_for(self.cells, name), slice)
        }

    def cells_for(self, variable):
        """Declared cell hint for one member of the block."""
        if variable.name not in self.variables:
            raise RoleError(f"variable {variable.name!r} is absent from the block")
        hint = _hint_for(self.cells, variable.name)
        return hint if isinstance(hint, Cells) else None

    def support_for(self, variable):
        """A finite increasing support interval in the variable's canonical units."""
        if variable.name not in self.variables:
            raise RoleError(f"variable {variable.name!r} is absent from the block")
        hint = _hint_for(self.support, variable.name)
        if hint is None:
            return None
        if not isinstance(hint, (list, tuple)) or len(hint) != 2:
            raise RoleError(f"support for {variable.name} needs a pair (lo, hi)")
        values = []
        for bound in hint:
            x, _, unit = _coordinate(bound, variable.name)
            if x.ndim:
                raise RoleError(f"support bounds for {variable.name} must be scalar")
            values.append(float(_placed_value(x, unit, variable, support=True)))
        lo, hi = values
        if not lo < hi:
            raise RoleError(f"support for {variable.name} must have lo < hi")
        return lo, hi

    def record(self) -> dict:
        """Strict JSON of the constraints and their declared variable order."""
        return strict_json(
            {
                "family": type(self).__name__,
                **{name: _record_hint(value) for name, value in vars(self).items()},
            }
        )


class FreeMoments(_MomentBlock):
    """Unknown joint moments over the block's variables.

    ``cells=slice(...)`` (or a mapping of variables to slices) fits the selected
    cells only. The other cells carry no mass, and the fit and its moments are
    on the selected lattice, as with ``basis.select(...)``.
    """

    prior: Any
    allows_selection: ClassVar[bool] = True

    def __init__(self, *variables, cells=None, support=None, prior=None):
        self._set_hints(variables, cells, support)
        self.prior = prior


class NonNegativeCells(_MomentBlock):
    """An unknown measure represented by nonnegative masses in each cell."""

    point_masses: int = eqx.field(static=True)
    allows_selection: ClassVar[bool] = True

    def __init__(self, *variables, point_masses=17, cells=None, support=None):
        if (
            isinstance(point_masses, (bool, np.bool_))
            or not isinstance(point_masses, (int, np.integer))
            or point_masses < 1
        ):
            raise ValueError("point_masses must be a positive integer")
        self._set_hints(variables, cells, support)
        self.point_masses = int(point_masses)
