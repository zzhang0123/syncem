"""Read population declarations as an ordered graph of factor sources."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import jax

from ..distributions import (
    AzimuthFactor,
    Conditional,
    Free,
    Isotropic,
    NoRotation,
    UniformAzimuth,
)
from ..distributions._statistics import factor_key
from ..lattice.moments import Moments
from ..population.blocks import FreeMoments, NonNegativeCells, Samples
from ..population.declaration import parents
from ..population.ray import Ray
from ..values.errors import RoleError

_DEFAULTS = {
    "Isotropic": Isotropic,
    "UniformAzimuth": UniformAzimuth,
    "NoRotation": NoRotation,
}


@dataclass(frozen=True, eq=False)
class Source:
    """One factor or joint block. The declaration and its parents stay together."""

    variables: tuple[str, ...]
    declaration: Any
    kind: str
    parents: tuple[str, ...] = ()
    default: bool = False

    def support_for(self, variable):
        """Support or support hint in canonical units, if the source declares it."""
        if variable.name not in self.variables:
            raise RoleError(f"source does not supply {variable.name}")
        query = getattr(self.declaration, "support_for", None)
        if query is not None:
            return query(variable)
        support = getattr(self.declaration, "support", None)
        if isinstance(support, Mapping):
            return support.get(variable.name)
        if self.kind == "point":
            return self.declaration, self.declaration
        return support


def _neutral(variable):
    try:
        factory = _DEFAULTS[variable.default]
    except KeyError:
        raise RoleError(
            f"{variable.name} has unknown kernel default {variable.default!r}"
        ) from None
    return factory()


def _is_conditional(declaration):
    return isinstance(declaration, Conditional) or (
        callable(declaration) and not hasattr(declaration, "rule")
    )


def _free(declaration):
    return any(
        isinstance(leaf, Free)
        for leaf in jax.tree.leaves(declaration, is_leaf=lambda x: isinstance(x, Free))
    )


def _keyword(name, declaration, definitions):
    if name not in definitions:
        raise RoleError(f"population variable {name!r} is absent from the kernel")
    if isinstance(declaration, (FreeMoments, NonNegativeCells)):
        bound = declaration.bind(name)
        kind = (
            "NonNegativeCells" if isinstance(bound, NonNegativeCells) else "FreeMoments"
        )
        return Source((name,), bound, kind)
    names = tuple(getattr(declaration, "variables", ()))
    if names and names != (name,):
        raise RoleError(f"the distribution on {names} cannot occupy keyword {name!r}")
    if _is_conditional(declaration):
        return Source((name,), declaration, "conditional", parents(declaration))
    if isinstance(declaration, Free):
        return Source((name,), declaration, "free parameter")
    law = definitions[name].place(declaration)
    kind = (
        "distribution"
        if hasattr(law, "rule")
        or hasattr(law, "basis_density")
        or hasattr(law, "characteristic")
        or isinstance(law, AzimuthFactor)
        else "point"
    )
    dependencies = (
        law.c.parents
        if isinstance(law, AzimuthFactor) and isinstance(law.c, Conditional)
        else tuple(getattr(law, "parents", ()))
    )
    return Source((name,), law, kind, dependencies)


def _moment_variables(moments, kernel):
    index = moments.index
    names = (*index.table.variables, *(name for name, _ in index.legendre))
    if "Re M2" in index.blocks:
        if kernel.azimuth is None:
            raise RoleError(
                "supplied Moments have an azimuth block but the kernel has no azimuth"
            )
        names += (kernel.azimuth,)
    return names


def _block(block, kernel, definitions):
    if isinstance(block, Moments):
        names = _moment_variables(block, kernel)
        kind = "Moments"
    else:
        names = tuple(getattr(block, "variables", ()))
        if not names:
            raise RoleError("a positional population block must name its variables")
        if isinstance(block, Samples):
            kind = "Samples"
        elif isinstance(block, Ray):
            kind = "Ray"
        elif isinstance(block, FreeMoments):
            kind = "FreeMoments"
        elif isinstance(block, NonNegativeCells):
            kind = "NonNegativeCells"
        elif hasattr(block, "rule"):
            kind = "joint distribution"
        else:
            raise RoleError(f"{type(block).__name__} supplies no joint rule")
    if any(not isinstance(name, str) for name in names) or len(set(names)) != len(
        names
    ):
        raise RoleError("a block must name distinct kernel variables")
    unknown = set(names) - definitions.keys()
    if unknown:
        raise RoleError(
            f"population block variables {sorted(unknown)} are absent from the kernel"
        )
    if isinstance(block, Samples):
        block = block.placed(definitions)
    return Source(names, block, kind, tuple(getattr(block, "parents", ())))


def _sources(pop, kernel, definitions):
    found = [_keyword(name, law, definitions) for name, law in pop.factors.items()]
    found.extend(_block(block, kernel, definitions) for block in pop.blocks)
    supplied = set()
    for source in found:
        duplicate = supplied.intersection(source.variables)
        if duplicate:
            raise RoleError(
                f"variables {sorted(duplicate)} are declared more than once"
            )
        supplied.update(source.variables)
    for name, variable in definitions.items():
        if name in supplied:
            continue
        if variable.default is None:
            raise RoleError(f"declare the value or law of {name}")
        found.append(Source((name,), _neutral(variable), "distribution", default=True))
        supplied.add(name)
    # Check the original graph before explicitly dropping any dependence.
    for source in found:
        missing = set(source.parents) - supplied
        if missing:
            raise RoleError(
                f"{', '.join(source.variables)} depends on undeclared variables {sorted(missing)}"
            )
    ordered, ready = [], set()
    while found:
        eligible = [source for source in found if ready.issuperset(source.parents)]
        if not eligible:
            cycle = tuple(name for source in found for name in source.variables)
            raise RoleError(f"cyclic population dependencies among {cycle}")
        ordered.extend(eligible)
        ready.update(name for source in eligible for name in source.variables)
        found = [source for source in found if source not in eligible]
    return tuple(ordered)


def _by_variable(sources):
    return {name: source for source in sources for name in source.variables}


def _children(sources, parent, average):
    if parent is None or parent in average:
        return ()
    return tuple(source for source in sources if parent in source.parents)


def _statistics_key(source):
    declaration = source.declaration
    if source.kind == "point":
        return ("point", float(declaration))
    if _free(declaration):
        return ("free", id(declaration))
    return factor_key(declaration)
