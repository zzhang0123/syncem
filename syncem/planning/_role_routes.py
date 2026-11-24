"""Apply structural constraints and classify the role of each axis."""

from __future__ import annotations

from dataclasses import dataclass

from ..base.capabilities import Beaming
from ..distributions import AzimuthFactor, Free
from ..population.blocks import FreeMoments, NonNegativeCells
from ..values.errors import RoleError
from ._role_sources import _by_variable, _children, _free, _statistics_key

_UNKNOWN = ("FreeMoments", "NonNegativeCells")
_RESOLVED_BLOCKS = (*_UNKNOWN, "Samples", "Moments")


@dataclass(frozen=True)
class Role:
    """One variable's structural representation and deciding rule."""

    variable: str
    declared: str
    role: str
    representation: str | None
    how: str
    rule: str
    default: bool


def _independent_phi(source, kernel, sources, average):
    return (
        source.kind == "FreeMoments"
        and source.variables == (kernel.azimuth,)
        and not _children(sources, kernel.azimuth, average)
    )


def _lattice_representations(lattice, kernel):
    representations = dict.fromkeys(lattice.cells.variables, "Cells")
    for name, axis in lattice.representations.items():
        representation = axis.signature()[0]
        representations[name] = representation
        if representation == "FieldScaling" and getattr(axis, "inclination", False):
            for capability in kernel.capabilities:
                if isinstance(capability, Beaming):
                    representations[capability.partner] = representation
    return representations


def _seed(sources, kernel, average):
    seed = set()
    for source in sources:
        if source.kind in _RESOLVED_BLOCKS and not _independent_phi(
            source, kernel, sources, average
        ):
            seed.update(source.variables)
    return seed


def _dependencies(sources, resolved, average, *, explicit):
    """Close one component under its resolved parents and joint source variables."""
    promoted = set()
    for source in sources:
        resolved_outputs = set(source.variables) & resolved
        if not resolved_outputs or source.kind == "Ray":
            continue
        missing_outputs = set(source.variables) - resolved
        missing_parents = set(source.parents) - (resolved | average)
        if explicit and missing_outputs:
            raise RoleError(
                f"joint block {source.variables} spans resolved and averaged variables. "
                "Give all of its variables axes, or average the entire block."
            )
        if explicit and missing_parents:
            child = ", ".join(sorted(resolved_outputs))
            raise RoleError(
                f"{child} depends on {sorted(missing_parents)}, which lattice= leaves averaged. "
                "Add axes for the parents, or name them in average= to drop the dependence."
            )
        promoted.update(missing_outputs | missing_parents)
    return promoted


def _phi_must_resolve(component_sources, resolved, kernel, average, *, explicit):
    phi = kernel.azimuth
    if phi is None or phi in resolved or len(component_sources) < 2:
        return False
    keys = {
        _statistics_key(_by_variable(sources)[phi]) for sources in component_sources
    }
    if len(keys) < 2:
        return False
    if any(
        set(child.variables) - resolved
        for sources in component_sources
        for child in _children(sources, phi, average)
    ):
        return False  # each distinct nested law gets its own basis
    if explicit:
        raise RoleError(
            f"components differ in averaged {phi}. Resolve {phi} with AzimuthHarmonic on the lattice."
        )
    return True


def _needs_fourier(sources, resolved, phi, average):
    source = _by_variable(sources)[phi]
    return any(
        set(child.variables) - resolved or source.kind in _UNKNOWN
        for child in _children(sources, phi, average)
    )


def _assign_axes(
    kernel, component_sources, resolved, average, representations, explicit
):
    for definition in kernel.variables:
        name = definition.name
        if name not in resolved:
            continue
        if name == kernel.azimuth:
            fourier = any(
                _needs_fourier(sources, resolved, name, average)
                for sources in component_sources
            )
            if explicit and fourier and representations[name] != "AzimuthFourier":
                raise RoleError(
                    f"a conditional on {name} needs AzimuthFourier(m_max) on the lattice. "
                    "Use Fourier rows, resolve a specified child, or name the parent in average=."
                )
            if not explicit:
                representations[name] = (
                    "AzimuthFourier" if fourier else "AzimuthHarmonic"
                )
        elif not explicit:
            laws = [
                _by_variable(sources)[name].declaration for sources in component_sources
            ]
            needs_cells = any(
                isinstance(law, NonNegativeCells)
                or (
                    isinstance(law, FreeMoments)
                    and law.cells_for(definition) is not None
                )
                for law in laws
            )
            representations[name] = (
                "Legendre"
                if definition.kind == "cosine" and not needs_cells
                else "Cells"
            )


def _validate_component(sources, resolved, representations, kernel, average, lattice):
    by_name = _by_variable(sources)
    for source in sources:
        missing = set(source.variables) - resolved
        if lattice is not None and source.kind == "Ray":
            missing.discard(kernel.primary)
            if missing:
                raise RoleError(
                    f"Ray needs axes for {sorted(missing)}. Omit lattice= to use direct integration."
                )
        if lattice is not None and source.kind in _RESOLVED_BLOCKS and missing:
            if not _independent_phi(source, kernel, sources, average):
                raise RoleError(
                    f"{source.kind} needs axes for {sorted(missing)} in lattice="
                )
        if lattice is not None and source.kind == "Moments":
            lattice._own(source.declaration)

    depth = kernel.depth
    if depth in by_name:
        law = by_name[depth].declaration
        if (
            hasattr(law, "characteristic")
            and not hasattr(law, "rule")
            and (depth in resolved or _children(sources, depth, average))
        ):
            raise RoleError(
                f"{depth} has a characteristic function only. Give a screen with quadrature nodes "
                "when depth is resolved or a conditional parent."
            )
    phi = kernel.azimuth
    if (
        phi in by_name
        and isinstance(by_name[phi].declaration, AzimuthFactor)
        and (phi in resolved or _children(sources, phi, average))
    ):
        raise RoleError(
            f"{phi} needs angular nodes. Use AzimuthDensity, Tabulated or a joint distribution "
            "instead of AzimuthFactor."
        )
    for capability in kernel.capabilities:
        if not isinstance(capability, Beaming):
            continue
        name, partner = capability.variable, capability.partner
        if representations.get(name) != "Cells":
            continue
        other = by_name[partner]
        if partner in resolved and representations.get(partner) == "Legendre":
            continue
        if (
            partner not in resolved
            and not getattr(other.declaration, "is_point", False)
            and (hasattr(other.declaration, "density") or other.kind == "conditional")
        ):
            continue
        raise RoleError(
            f"{name} has Cells under Beaming but {partner} has no averaged density or Legendre axis. "
            f"Declare a density for {partner}, or use {name}=Legendre(8)."
        )


def _resolved_rule(
    source, name, dependencies, by_name, representation, promoted, explicit
):
    if source.kind == "Ray":
        return "ray moments", "R8"
    if representation == "AzimuthFourier":
        return "azimuth Fourier rows", "R4a"
    if any(by_name[parent].kind in _UNKNOWN for parent in dependencies):
        return "conditional linear moment map", "R3a"
    if dependencies:
        return "conditional quadrature moments", "R3"
    if source.kind in _RESOLVED_BLOCKS:
        return source.kind, "R1" if explicit else "R2"
    rule = "R1" if explicit else "R3" if name in promoted else "R2"
    return "point moments" if source.kind == "point" else "quadrature moments", rule


def _averaged_role(
    source, name, kernel, sources, dependencies, representations, average
):
    resolved = representations.keys()
    free_factor = (
        name == kernel.azimuth
        and isinstance(source.declaration, AzimuthFactor)
        and isinstance(source.declaration.c, Free)
    )
    if free_factor or _independent_phi(source, kernel, sources, average):
        return "azimuth parameter", "free complex azimuth factor", "R5"
    if _free(source.declaration):
        return "basis parameter", "rebuild for averaged parameters", "R6"
    if name == kernel.azimuth:
        if (
            len(source.variables) > 1
            or any(parent not in resolved for parent in dependencies)
            or _children(sources, name, average)
        ):
            return (
                "averaged",
                "azimuth in nested quadrature",
                "R4a" if (_children(sources, name, average)) else "R5(iii)",
            )
        if not dependencies:
            return "averaged", "constant azimuth factor", "R5(i)"
        if all(representations[parent] == "Cells" for parent in dependencies):
            return "averaged", "azimuth factor on cells", "R5(ii)"
        if any(representations[parent] == "Legendre" for parent in dependencies):
            return "averaged", "azimuth factor in Legendre projection", "R5(iv)"
        raise RoleError(
            f"no azimuth factor route for {dependencies}. Use Cells or average=."
        )
    if (
        name == kernel.depth
        and not _children(sources, name, average)
        and (
            len(source.variables) == 1 and hasattr(source.declaration, "characteristic")
        )
    ):
        return "averaged", "screen characteristic", "R4"
    if kernel.azimuth in dependencies and kernel.azimuth not in resolved:
        return "averaged", "nested quadrature on azimuth nodes", "R4a"
    for capability in kernel.capabilities:
        if isinstance(capability, Beaming) and capability.variable == name:
            return "averaged", f"beaming at {capability.partner}", "R4"
    if source.kind == "point":
        return "averaged", "fixed kernel input", "R4"
    how = (
        "joint quadrature"
        if len(source.variables) > 1
        else (
            "nested quadrature"
            if any(p not in resolved for p in dependencies)
            else "conditional quadrature" if dependencies else "quadrature"
        )
    )
    return "averaged", how, "R4"
