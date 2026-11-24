"""Column measures assembled from fixed points and declared laws.

LABEL [extension].

Population factors have the units and shapes of their declared variables or
sample blocks. Dependency bookkeeping has no certified information about the
adequacy of the assumed laws.
"""

from __future__ import annotations

import inspect
from collections.abc import Mapping
from typing import Any

import equinox as eqx
import jax.numpy as jnp

from ..base.variables import SYNCHROTRON_VARIABLES
from ..distributions import AzimuthFactor, Delta
from ..distributions.parameters import (
    Conditional,
    Free,
    free_fields,
    parameter_argument,
)
from ..distributions.user import UserFamily
from ..model._provenance import strict_json
from ..units import Quantity
from ..values.errors import RoleError
from .blocks import Samples

__all__ = ["Population"]


def parents(law: Any) -> tuple[str, ...]:
    """Variable dependencies of a conditional declaration."""
    if isinstance(law, Conditional):
        return law.parents
    if isinstance(law, AzimuthFactor) and isinstance(law.c, Conditional):
        return law.c.parents
    if callable(law) and not hasattr(law, "rule"):
        return tuple(inspect.signature(law).parameters)
    return ()


def law_at(law: Any, point: Mapping, units: Mapping | None = None) -> Any:
    """Resolve a conditional law at its current parent values."""
    if isinstance(law, Conditional):
        selected = {
            name: units[name] for name in law.parents if units and name in units
        }
        return law.evaluate(point, canonical_units=selected)
    if callable(law) and not hasattr(law, "rule"):
        return law(
            **{
                name: (
                    Quantity(point[name], units[name])
                    if units and units.get(name)
                    else point[name]
                )
                for name in parents(law)
            }
        )
    return law


def placed_law(law, variable):
    """Place a conditional result without forcing traced numerical values to the host."""
    if hasattr(law, "characteristic") and not hasattr(law, "rule"):
        raise RoleError(f"{variable.name} needs quadrature nodes in this role")
    declaration = law if hasattr(law, "rule") else Delta(law)
    return variable.place(declaration)


def ordered(
    factors: Mapping, available: tuple = (), aliases=None, requires=None
) -> tuple:
    """Factor order with each law after its parents."""
    result, pending = [], dict(factors)
    supplied = set(available)
    aliases = aliases or {}
    needed = {
        name: {aliases.get(parent, parent) for parent in parents(law)}
        for name, law in pending.items()
    }
    for name, dependencies in (requires or {}).items():
        needed[name].update(dependencies)
    for name, law in pending.items():
        missing = needed[name] - set(pending) - supplied
        if missing:
            raise RoleError(f"{name} depends on undeclared variables {sorted(missing)}")
    while pending:
        ready = [name for name in pending if supplied.issuperset(needed[name])]
        if not ready:
            raise RoleError(f"cyclic population dependencies among {tuple(pending)}")
        for name in ready:
            result.append((name, pending.pop(name)))
            supplied.add(name)
    return tuple(result)


class Population(eqx.Module):
    """A product of factors and joint blocks, or a sum of such measures."""

    factors: dict
    blocks: tuple
    components: tuple
    symmetries: tuple[str, ...] = eqx.field(static=True)
    column_factor: Any

    def __init__(self, *blocks, symmetries=(), **factors):
        self.factors = dict(factors)
        self.blocks = tuple(blocks)
        self.components = ()
        self.symmetries = tuple(symmetries)
        self.column_factor = 1.0

    def replace(self, **factors) -> Population:
        """Return the population with the named factors replaced."""
        if self.components:
            raise RoleError("replace factors on a component of a population sum")
        after = {**self.factors, **factors}
        return eqx.tree_at(lambda p: p.factors, self, after)

    def __add__(self, other):
        if not isinstance(other, Population):
            return NotImplemented
        parts = (*self.parts, *other.parts)
        return eqx.tree_at(lambda p: p.components, Population(), parts)

    def __rmul__(self, factor):
        factor = jnp.asarray(factor)
        if factor.ndim:
            raise ValueError("a population scale must be scalar")
        factor = eqx.error_if(
            factor,
            ~jnp.isfinite(factor) | (factor < 0),
            "population scale must be finite and nonnegative",
        )
        return eqx.tree_at(lambda p: p.column_factor, self, factor * self.column_factor)

    @property
    def parts(self):
        """Leaf components carrying their total population scale."""
        if not self.components:
            return (self,)
        return tuple(
            self.column_factor * part
            for child in self.components
            for part in child.parts
        )

    def placed(self, variables) -> Population:
        """Place dimensional values using the kernel's variable declarations."""
        definitions = {v.name: v for v in variables}
        names = [
            *self.factors,
            *(
                name
                for block in self.blocks
                for name in getattr(block, "variables", ())
            ),
        ]
        unique = set(names)
        if len(unique) != len(names):
            raise RoleError("a population variable has multiple declarations")
        unknown = unique - set(definitions)
        if unknown:
            raise RoleError(
                f"population variables {sorted(unknown)} are absent from the kernel"
            )
        factors = {
            name: (
                law
                if isinstance(law, (Free, Conditional))
                or (callable(law) and not hasattr(law, "rule"))
                else definitions[name].place(law)
            )
            for name, law in self.factors.items()
        }
        blocks = tuple(
            block.placed(definitions) if isinstance(block, Samples) else block
            for block in self.blocks
        )
        return eqx.tree_at(lambda p: (p.factors, p.blocks), self, (factors, blocks))

    def to_distributions(self, lattice=None) -> dict:
        """Factors with no representation on the given lattice."""
        if self.components:
            raise RoleError("resolve the factors of each population component")
        resolved = (
            set(lattice.cells.variables) | set(lattice.representations)
            if lattice is not None
            else set()
        )
        return {name: law for name, law in self.factors.items() if name not in resolved}

    def at(self, params) -> Population:
        """Resolve free keyword factors and the named parameters of conditional laws."""
        if self.components:
            return eqx.tree_at(
                lambda p: p.components,
                self,
                tuple(p.at(params) for p in self.components),
            )
        factors = {}
        for name, value in self.factors.items():
            if isinstance(value, Free):
                factors[name] = parameter_argument(value, params.get(name, value))
            elif isinstance(value, (Conditional, UserFamily)):
                updates = {
                    key: (
                        parameter_argument(item, params.get(f"{name}.{key}", item))
                        if isinstance(value, Conditional)
                        else params.get(f"{name}.{key}", item)
                    )
                    for key, item in value.params.items()
                }
                factors[name] = eqx.tree_at(lambda law: law.params, value, updates)
            else:
                factors[name] = value
                for field in free_fields(value):
                    key = f"{name}.{field}"
                    if key in params:
                        factors[name] = eqx.tree_at(
                            lambda law: getattr(law, field), factors[name], params[key]
                        )
        return eqx.tree_at(lambda p: p.factors, self, factors)

    def to_dict(self) -> dict:
        """Strict-JSON declaration record for concrete population parameters."""

        def record(item):
            method = getattr(item, "record", None) or getattr(item, "to_dict", None)
            if method is not None:
                return method()
            if callable(item):
                return {
                    "conditional": getattr(item, "__qualname__", type(item).__name__)
                }
            return item

        return strict_json(
            {
                "factors": {
                    name: record(value) for name, value in self.factors.items()
                },
                "blocks": [record(block) for block in self.blocks],
                "components": [item.to_dict() for item in self.components],
                "symmetries": self.symmetries,
                "column_factor": self.column_factor,
            }
        )


def conventional_units(factors) -> dict:
    """Canonical units available when pairing synchrotron declarations directly."""
    return {
        name: SYNCHROTRON_VARIABLES[name].unit
        for name in factors
        if name in SYNCHROTRON_VARIABLES
    }
