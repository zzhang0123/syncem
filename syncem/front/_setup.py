"""Compile a specified population into resolved factors and engine inputs."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..base.capabilities import (
    FieldScalingRelation,
    ReferenceDensity,
)
from ..base.continuum import ContinuumKernel
from ..base.phase import FaradayRotation, PhaseLaw
from ..distributions import (
    AzimuthFactor,
    CumulantScreen,
    Isotropic,
    NoRotation,
    UniformAzimuth,
)
from ..engine.average import AveragedKernel, NodeRule
from ..engine.columns import ColumnRules
from ..lattice.representations import AzimuthHarmonic
from ..planning.defaults import beaming_knots, primary_lattice, support_for
from ..population import Samples
from ..population.declaration import Population, law_at, ordered, parents, placed_law
from ..population.quadrature import cell_boxes
from ..records.ledger import ErrorTerm
from ..values.errors import RegimeError, RoleError
from ._beaming_layout import diagonal_reference, merge_beaming, resolved_beaming
from ._leaf import (
    beam,
    conditional_spin,
    projection_panels,
    spin_factor,
    split_beaming_rule,
)


@dataclass(frozen=True)
class ForwardSetup:
    population: Population
    averaged: AveragedKernel
    resolved: dict
    units: dict
    azimuth: complex | None
    b_max: float | None
    defaults: tuple


def _cap(kernel, type_):
    return next((item for item in kernel.capabilities if isinstance(item, type_)), None)


def bind_lattice(lattice, kernel):
    """Apply declared depth, units and reference densities before building the index."""
    references = tuple(
        (c.variable, c.density)
        for c in kernel.capabilities
        if isinstance(c, ReferenceDensity)
    )
    names = set(lattice.cells.variables) | set(lattice.representations)
    unknown = names - {v.name for v in kernel.variables}
    if unknown:
        raise RoleError(
            f"lattice variables {sorted(unknown)} are absent from the kernel"
        )
    return lattice._for_kernel(kernel.variables, kernel.depth, references)


def _default(name):
    return {
        "Isotropic": Isotropic,
        "UniformAzimuth": UniformAzimuth,
        "NoRotation": NoRotation,
    }[name]()


def placed(population, kernel):
    if not isinstance(population, Population):
        population = Population(population)
    if any(not isinstance(block, Samples) for block in population.blocks):
        raise RoleError(
            "a specified forward prediction needs concrete laws or sample blocks"
        )
    pop = population.placed(kernel.variables)
    supplied = {name for block in pop.blocks for name in block.variables} | set(
        pop.factors
    )
    column_count = sum(block.column is not None for block in pop.blocks)
    if column_count > 1 or (
        column_count == 1
        and any(hasattr(law, "panel_rule") for law in pop.factors.values())
    ):
        raise RoleError("use one electron column across factors and sample blocks")
    defaults = {
        v.name: v.default
        for v in kernel.variables
        if v.default is not None and v.name not in supplied
    }
    if defaults:
        pop = pop.replace(**{name: _default(law) for name, law in defaults.items()})
    missing = {v.name for v in kernel.variables if v.default is None} - supplied
    if missing:
        raise RoleError(f"declare the values or laws of {sorted(missing)}")
    return pop, tuple((name, law) for name, law in defaults.items())


def default_lattice(population, kernel, channels):
    pop, _ = placed(population, kernel)
    return primary_lattice(pop, kernel, channels)


def _resolved_bound(kernel, lattice, pop):
    scaling = _cap(kernel, FieldScalingRelation)
    if scaling is None:
        return None
    if scaling.variable in lattice.cells.variables:
        i = lattice.cells.variables.index(scaling.variable)
        return float(np.max(lattice.cells.hi[:, i]))
    return support_for(pop, scaling.variable)[1]


def _quadrature(law, factor):
    if not hasattr(law, "rule"):
        raise RoleError(
            "a depth on the lattice or used as a parent needs quadrature nodes"
        )
    if factor == 1 or getattr(law, "is_point", False):
        return law.rule()
    count = getattr(law, "nodes", None)
    if not isinstance(count, int):
        count = len(law.rule()[0])
    return law.rule(n=factor * count)


def _rule(law, units, variable, factor=1):
    dependencies = parents(law)

    def evaluate(point):
        result = law_at(law, point, units)
        if dependencies:
            if hasattr(result, "characteristic") and not hasattr(result, "rule"):
                raise RoleError("a depth used as a parent needs quadrature nodes")
            result = placed_law(result, variable)
        return _quadrature(result, factor)

    def density(x, point):
        return law_at(law, point, units).density(x)

    if dependencies:
        return NodeRule(evaluate, parents=dependencies)
    return NodeRule(evaluate, (), density if hasattr(law, "density") else None)


def _conditional_depth(law, units, variable, factor):
    def at(point):
        result = law_at(law, point, units)
        if hasattr(result, "characteristic"):
            return variable.place(result).characteristic
        return _quadrature(placed_law(result, variable), factor)

    return at


def screen_truncation(population, phase, conditional=False):
    """Report that the omitted terms of an averaged screen closure are unbounded."""
    if phase is False:
        return None
    law = population.factors.get("depth")
    foreground = phase.screens if isinstance(phase, PhaseLaw) else ()
    screens = (law, *(getattr(factor, "__self__", factor) for factor in foreground))
    if conditional or any(
        isinstance(screen, CumulantScreen) and screen.truncates for screen in screens
    ):
        return ErrorTerm.unbounded(
            "No reference bound on omitted conditional or higher screen cumulants",
            name="depth_cumulant_truncation",
        )
    return None


def _average(kernel, pop, lattice, phase, units, direct=False, diagonal=(), refine=()):
    factors = pop.factors
    variables = {v.name: v for v in kernel.variables}
    resolved = set(lattice.cells.variables) | set(lattice.representations)
    parent_names = {name for law in factors.values() for name in parents(law)}
    angular = lattice.representations.get(kernel.azimuth)
    if (
        not direct
        and isinstance(angular, AzimuthHarmonic)
        and any(
            kernel.azimuth in parents(law)
            for name, law in factors.items()
            if name not in resolved
        )
    ):
        raise RoleError(
            f"an averaged child of {kernel.azimuth} needs AzimuthFourier(m_max) on the lattice. "
            "Resolve the child or average the azimuth with a density."
        )
    if direct and diagonal:
        ties, beam_factors = diagonal_reference(pop, lattice, diagonal, units)
        resolved.discard(ties[0][0])
    else:
        ties, beam_factors = beam(kernel, factors, resolved, units)
    aliases = dict(ties)
    beam_knots = {target: beaming_knots(factors.get(source)) for source, target in ties}
    for name, knots in beam_knots.items():
        if name in lattice.cells.variables and not (direct and diagonal):
            i = lattice.cells.variables.index(name)
            cuts = [
                x
                for x in knots
                if np.any((lattice.cells.lo[:, i] < x) & (x < lattice.cells.hi[:, i]))
            ]
            if cuts:
                raise RoleError(
                    f"Beaming density changes inside a {name} cell at {cuts}. Split the cell there."
                )
    leaf_factors = list(beam_factors)
    fixed, rules = {}, []
    legendre = (
        {}
        if direct
        else {
            name: (degree, max(12, 2 * degree + 1))
            for name, degree in lattice.index.legendre
        }
    )
    spin = None
    blocked = tuple(name for block in pop.blocks for name in block.variables)
    ordered(factors, available=blocked)
    sources = {name: law for name, law in factors.items() if name not in aliases}
    for name, law in ordered(sources, available=blocked, aliases=aliases):
        factor = 2 if name in refine else 1
        dependency = {aliases.get(p, p) for p in parents(law)}
        if name in resolved:
            if dependency - resolved:
                raise RoleError(
                    f"{name} is resolved and depends on averaged {sorted(dependency - resolved)}"
                )
            continue
        if name == kernel.azimuth:
            if name in parent_names:
                if not hasattr(law, "rule") and not dependency:
                    raise RoleError(
                        "a law that depends on the azimuth needs an azimuth density"
                    )
                rules.append((name, _rule(law, units, variables[name], factor)))
                leaf_factors.append(
                    ("P", lambda point: jnp.exp(2j * point[kernel.azimuth]))
                )
                spin = 1.0
            elif dependency:
                leaf_factors.append(("P", conditional_spin(law, units)))
                spin = 1.0
            else:
                spin = complex(spin_factor(law))
        elif (
            name == kernel.depth
            and name not in parent_names
            and (dependency or hasattr(law, "characteristic"))
        ):
            if isinstance(phase, PhaseLaw):
                if dependency:
                    leaf_factors.append(
                        (
                            "depth",
                            _conditional_depth(law, units, variables[name], factor),
                        )
                    )
                else:
                    phase = phase.with_screen(law.characteristic)
            fixed[name] = 0.0
        elif (
            type(kernel) is ContinuumKernel
            and name == "mu"
            and isinstance(law, Isotropic)
        ):
            # The continuum kernel already includes this pitch average.
            fixed[name] = 0.0
        elif not hasattr(law, "rule") and not dependency and not callable(law):
            fixed[name] = float(law)
        elif getattr(law, "is_point", False) and not dependency:
            fixed[name] = float(law.rule()[0][0])
        elif beam_knots.get(name) and not dependency and hasattr(law, "density"):
            nodes, weights = split_beaming_rule(law, beam_knots[name], factor=factor)
            rules.append((name, NodeRule(lambda _, x=nodes, w=weights: (x, w))))
        else:
            rules.append((name, _rule(law, units, variables[name], factor)))
    return (
        AveragedKernel(
            kernel,
            rules,
            legendre=legendre,
            fixed=fixed,
            phase=phase,
            ties=ties,
            leaf_factors=leaf_factors,
            diagonal=() if direct else diagonal,
            legendre_panels=projection_panels(factors, ties, beam_knots, legendre),
        ),
        spin,
    )


def configure(
    population,
    kernel,
    channels,
    lattice,
    phase,
    *,
    direct=False,
    strict=False,
    refine=(),
):
    pop, defaults = placed(population, kernel)
    units = {v.name: v.unit for v in kernel.variables}
    names = set(lattice.cells.variables) | set(lattice.representations)
    joint = {name for block in pop.blocks for name in block.variables}
    if joint - names:
        raise RoleError(
            f"the lattice must resolve sample variables {sorted(joint - names)}"
        )
    resolved = {name: factor for name, factor in pop.factors.items() if name in names}
    if any(isinstance(law, AzimuthFactor) for law in resolved.values()):
        raise RoleError("resolved azimuth needs a point or density")
    phase = FaradayRotation() if phase is None and kernel.depth is not None else phase
    bound = _resolved_bound(kernel, lattice, pop)
    scaling = _cap(kernel, FieldScalingRelation)
    if direct and scaling is not None:
        bound = support_for(pop, scaling.variable)[1]
    if type(kernel) is ContinuumKernel:
        if not direct:
            for cell, box in enumerate(cell_boxes(lattice.cells)):
                declared = box
                if kernel.primary not in declared:
                    declared = {
                        **box,
                        kernel.primary: support_for(pop, kernel.primary),
                    }
                result = kernel.regime(declared, channels)
                if not result.ok:
                    raise RegimeError(f"cell {cell}: {result.reason}")
        if kernel.x_min is None:
            top = (
                float(
                    lattice.cells.hi[
                        :, lattice.cells.variables.index(kernel.primary)
                    ].max()
                )
                if kernel.primary in lattice.cells.variables
                else support_for(pop, kernel.primary)[1]
            )
            if bound and np.isfinite(bound) and np.isfinite(top):
                kernel = kernel.resolved(channels, top, bound)
    elif not direct:
        for cell, box in enumerate(cell_boxes(lattice.cells)):
            result = kernel.regime(box, channels)
            if not result.ok:
                raise RegimeError(f"cell {cell}: {result.reason}")
    if strict and defaults:
        raise RoleError(f"strict prediction requires explicit defaults: {defaults}")
    diagonal = resolved_beaming(kernel, pop, lattice)
    averaged, spin = _average(
        kernel, pop, lattice, phase, units, direct, diagonal, refine
    )
    if direct and diagonal:
        resolved.pop(averaged.ties[0][0], None)
    return ForwardSetup(pop, averaged, resolved, units, spin, bound, defaults)


def column_rules(setup, lattice):
    kernel = setup.averaged.kernel
    return ColumnRules(
        merge=merge_beaming(kernel, lattice),
        symmetries=tuple(
            name for name in setup.population.symmetries if name == "pitch_reversal"
        ),
        depth_free_spin0=type(kernel) is ContinuumKernel
        and all(
            kernel.depth not in (rule.parents or ())
            for _, rule in setup.averaged.averaged
        ),
    )
