"""Structural role compilation before any kernel or basis evaluation (SPEC Sec. 5.2).

LABEL [extension].

The role table has the shape of one entry per named variable of a component.
Variable units follow the kernel. A role assignment carries no certified
statistical independence.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from ..population.declaration import Population
from ..values.errors import RoleError
from ._role_routes import (
    Role,
    _assign_axes,
    _averaged_role,
    _dependencies,
    _lattice_representations,
    _phi_must_resolve,
    _resolved_rule,
    _seed,
    _validate_component,
)
from ._role_sources import Source, _by_variable, _sources, _statistics_key

__all__ = ["CompiledRoles", "ComponentRoles", "Role", "Source", "compile_roles"]


@dataclass(frozen=True, eq=False)
class ComponentRoles:
    """Sources in dependency order and the roles of one population component."""

    population: Population
    sources: tuple[Source, ...]
    roles: Mapping[str, Role]
    dropped: tuple[tuple[str, str], ...]
    direct: bool

    def source_for(self, variable: str) -> Source:
        """The unique declaration responsible for a kernel variable."""
        return next(source for source in self.sources if variable in source.variables)

    @property
    def defaults(self) -> tuple[tuple[str, str], ...]:
        return tuple(
            (role.variable, role.declared)
            for role in self.roles.values()
            if role.default
        )


@dataclass(frozen=True, eq=False)
class CompiledRoles:
    """Eager structural plan. Numerical rules consume the original declarations."""

    resolved: tuple[str, ...]
    representations: Mapping[str, str]
    components: tuple[ComponentRoles, ...]
    basis_groups: tuple[tuple[int, ...], ...]
    assumptions: tuple[str, ...]
    caveats: tuple[str, ...]

    @property
    def direct(self) -> bool:
        return any(component.direct for component in self.components)


def _options(names, label, definitions):
    names = (names,) if isinstance(names, str) else tuple(names)
    if any(not isinstance(name, str) for name in names):
        raise RoleError(f"{label}= must name kernel variables")
    unknown = set(names) - definitions.keys()
    if unknown:
        raise RoleError(f"{label}= names unknown variables {sorted(unknown)}")
    return set(names)


def _component(pop, sources, kernel, representations, promoted, average, lattice):
    by_name = _by_variable(sources)
    resolved = representations.keys()
    direct = lattice is None and any(source.kind == "Ray" for source in sources)
    roles, dropped = {}, []
    for variable in kernel.variables:
        name = variable.name
        source = by_name[name]
        dependencies = tuple(
            parent for parent in source.parents if parent not in average
        )
        dropped.extend((name, parent) for parent in source.parents if parent in average)
        representation = representations.get(name)
        if source.kind == "Ray" and direct:
            role, how, rule = "direct", "ray integration", "R8"
            representation = None
        elif name in resolved:
            role = "resolved"
            how, rule = _resolved_rule(
                source,
                name,
                dependencies,
                by_name,
                representation,
                promoted,
                lattice is not None,
            )
        else:
            role, how, rule = _averaged_role(
                source, name, kernel, sources, dependencies, representations, average
            )
        declared = (
            type(source.declaration).__name__
            if source.kind in ("distribution", "joint distribution")
            else source.kind
        )
        if source.default and (role == "averaged" and rule == "R4"):
            rule = "R0"
        roles[name] = Role(
            name, declared, role, representation, how, rule, source.default
        )
    return ComponentRoles(pop, sources, roles, tuple(dropped), direct)


def _groups(components):
    groups = {}
    for index, component in enumerate(components):
        if component.direct:
            key = ("direct", index)
        else:
            key = tuple(
                (source.variables, _statistics_key(source))
                for source in sorted(
                    component.sources, key=lambda source: source.variables
                )
                if any(
                    component.roles[name].role
                    in ("averaged", "basis parameter", "azimuth parameter")
                    for name in source.variables
                )
            )
        groups.setdefault(key, []).append(index)
    return tuple(tuple(group) for group in groups.values())


def compile_roles(
    population, kernel, lattice=None, resolve=(), average=()
) -> CompiledRoles:
    """Resolve the population graph without evaluating quadrature or constructing a basis."""
    definitions = {variable.name: variable for variable in kernel.variables}
    requested = _options(resolve, "resolve", definitions)
    suppressed = _options(average, "average", definitions)
    if lattice is not None and requested:
        raise RoleError(
            "resolve= is an error with lattice=. The lattice declares the axes."
        )
    if requested & suppressed:
        raise RoleError("resolve= and average= cannot name the same variable")
    pop = population if isinstance(population, Population) else Population(population)
    parts = pop.parts
    specs = tuple(_sources(part, kernel, definitions) for part in parts)
    basis_specs = tuple(
        sources
        for sources in specs
        if lattice is not None or not any(source.kind == "Ray" for source in sources)
    )
    explicit = lattice is not None
    representations = _lattice_representations(lattice, kernel) if explicit else {}
    if unknown := set(representations) - definitions.keys():
        raise RoleError(
            f"lattice variables {sorted(unknown)} are absent from the kernel"
        )

    resolved = set(representations) | requested
    if kernel.primary is not None and basis_specs and not explicit:
        resolved.add(kernel.primary)
    if not explicit:
        for sources in basis_specs:
            resolved.update(_seed(sources, kernel, suppressed))
    promoted, caveats = set(), []
    while True:
        added = set().union(
            *(
                _dependencies(sources, resolved, suppressed, explicit=explicit)
                for sources in basis_specs
            )
        )
        if _phi_must_resolve(
            basis_specs, resolved, kernel, suppressed, explicit=explicit
        ):
            phi = kernel.azimuth
            added.add(phi)
            note = (
                f"R7: resolved {phi} because the components have different azimuth laws"
            )
            if note not in caveats:
                caveats.append(note)
        added -= resolved
        if not added:
            break
        resolved.update(added)
        promoted.update(added)
    _assign_axes(kernel, basis_specs, resolved, suppressed, representations, explicit)
    for sources in specs:
        _validate_component(
            sources, resolved, representations, kernel, suppressed, lattice
        )
    components = tuple(
        _component(
            part, sources, kernel, representations, promoted, suppressed, lattice
        )
        for part, sources in zip(parts, specs, strict=True)
    )
    groups = _groups(components)
    if len(components) > 1:
        caveats.append(
            f"R7: {len(groups)} groups of specified statistics across {len(components)} components"
        )
    assumptions = tuple(
        dict.fromkeys(
            f"independence:{child}|{parent}"
            for component in components
            for child, parent in component.dropped
        )
    )
    return CompiledRoles(
        tuple(name for name in definitions if name in resolved),
        representations,
        components,
        groups,
        assumptions,
        tuple(caveats),
    )
