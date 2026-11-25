"""Compile a specified population into resolved factors and engine inputs."""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np

from ..base.capabilities import (
    FieldScalingRelation,
    ReferenceDensity,
)
from ..base.continuum import ContinuumKernel
from ..base.hybrid import HybridKernel
from ..base.phase import FaradayRotation, PhaseLaw
from ..distributions import (
    AzimuthFactor,
    Conditional,
    CumulantScreen,
    Isotropic,
    NoRotation,
    UniformAzimuth,
)
from ..distributions.electrons import ElectronFamily
from ..engine.average import AveragedKernel, NodeRule
from ..engine.direct import continuum_children
from ..engine._beaming_diagonal import diagonal_weight
from ..engine.columns import ColumnRules
from ..lattice.representations import AzimuthHarmonic
from ..planning.defaults import beaming_knots, primary_lattice, support_for
from ..population import Samples
from ..population._diagonal import (
    fixed_angular_knots,
    host_diagonal_knots,
    knot_parents,
    knot_paths,
    uses_panel_nodes,
)
from ..population.declaration import (
    Population,
    ancestors,
    law_at,
    ordered,
    parents,
    placed_law,
)
from ..population.quadrature import cell_boxes
from ..records.ledger import ErrorTerm
from ..values.errors import RegimeError, RoleError
from ._beaming_layout import diagonal_reference, merge_beaming, resolved_beaming
from ._sample_beaming import pitch_measure, sample_beaming
from ._leaf import (
    beam,
    conditional_spin,
    moving_beaming_rule,
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
    beaming_breaks: dict
    beaming_geometry: dict


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
    return pitch_measure(pop, kernel), tuple(
        (name, law) for name, law in defaults.items()
    )


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


def _rule(law, units, variable, factor=1, *, physical_column=False):
    dependencies = parents(law)

    def evaluate(point):
        result = law_at(law, point, units)
        if dependencies:
            if hasattr(result, "characteristic") and not hasattr(result, "rule"):
                raise RoleError("a depth used as a parent needs quadrature nodes")
            result = placed_law(result, variable)
        nodes, weights = _quadrature(result, factor)
        if physical_column and isinstance(result, ElectronFamily):
            weights = weights * result.integrated_column()
        return nodes, weights

    def density(x, point):
        return law_at(law, point, units).density(x)

    if dependencies:
        return NodeRule(evaluate, parents=dependencies)
    return NodeRule(evaluate, (), density if hasattr(law, "density") else None)


def _column_reweighting(primary, factors, resolved, units, variables, refine):
    """Disintegrate the electron column over latent parents at fixed lattice coordinates."""
    if primary in resolved:
        return ()
    hidden = ancestors(factors, primary, given=resolved)
    if not hidden:
        return ()
    law = factors[primary]
    rules = [
        (name, _rule(parent, units, variables[name], 2 if name in refine else 1))
        for name, parent in ordered(
            {name: factors[name] for name in hidden}, available=resolved
        )
    ]

    def mass(point):
        shape = law_at(law, point, units)
        return shape.integrated_column() if isinstance(shape, ElectronFamily) else 1.0

    def marginal(level, point):
        if level == len(rules):
            return mass(point)
        name, rule = rules[level]
        # Placed fixed scalars carry unit probability.
        parent = factors[name]
        if not parents(parent) and not hasattr(parent, "rule"):
            return marginal(level + 1, {**point, name: parent})
        nodes, weights = rule(point)
        values = jax.vmap(lambda x: marginal(level + 1, {**point, name: x}))(nodes)
        return jnp.dot(weights, values)

    def relative_mass(point):
        total = marginal(0, point)
        return mass(point) / jnp.where(total > 0, total, 1.0)

    return (("all", relative_mass),)


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
    sampled = sample_beaming(kernel, pop, lattice, projected=True)
    emission_ties = ()
    point_projection = (
        sampled is not None
        and sampled.smooth
        and not direct
        and not diagonal
        and sampled.pitch in dict(lattice.index.legendre)
        and sampled.pitch not in ancestors(factors, sampled.look)
    )
    if point_projection:
        # The viewing node selects the physical pitch of every averaged child.
        ties, beam_factors = (), (("all", diagonal_weight),)
    elif sampled is not None and (direct or not diagonal):
        empirical = sample_beaming(kernel, pop, lattice) is not None
        if direct and not empirical:
            # A continuous pitch law has an exact density even when its basis
            # projects pitch against a fixed line of sight.
            ties, beam_factors = diagonal_reference(
                pop, lattice, (sampled.pitch, sampled.look), units
            )
        else:
            ties, beam_factors, emission_ties = sampled.routes(factors, units)
        if direct and empirical and not sampled.smooth:
            resolved.add(sampled.pitch)
        for source, _ in ties:
            resolved.discard(source)
    elif direct and diagonal:
        ties, beam_factors = diagonal_reference(pop, lattice, diagonal, units)
        resolved.discard(ties[0][0])
    else:
        ties, beam_factors = beam(kernel, factors, resolved, units)
    aliases = dict(ties)
    beam_knots = {}
    moving = {}
    geometry = {}

    def rule_nodes(name, local, point):
        placed = placed_law(local, variables[name])
        return _quadrature(placed, 2 if name in refine else 1)[0]

    for source, target in ties:
        folded = factors.get(source)
        if not parents(folded):
            beam_knots[target] = beaming_knots(folded)
        else:
            known = fixed_angular_knots(folded, units)
            if known is None:
                geometry[target] = knot_paths(
                    factors,
                    source,
                    target,
                    rule_nodes,
                    given=() if direct else resolved.difference((source, target)),
                )
                external = tuple(
                    name for name in knot_parents(geometry[target]) if name != target
                )
            if known is not None:
                beam_knots[target] = tuple(x for x in known if -1 < x < 1)
            elif (
                direct
                and target in resolved
                and uses_panel_nodes(geometry[target], resolved)
            ):
                moving[target] = (geometry[target], external)
            elif not external:
                driver = factors.get(target)
                span = (
                    (-1.0, 1.0)
                    if target in lattice.representations and not direct
                    else getattr(driver, "support", (-1.0, 1.0))
                )
                beam_knots[target] = host_diagonal_knots(
                    geometry[target], target, {}, units, span
                )
            else:
                moving[target] = (geometry[target], external)
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
    leaf_factors = []
    if not direct:
        leaf_factors.extend(
            _column_reweighting(
                kernel.primary, factors, resolved, units, variables, refine
            )
        )
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
    # A moving angular panel is built at the external parents of its folded
    # density. When a parent itself descends from the driver, its own nested
    # conditional must remain inside the angular integral.
    requires = {}
    promoted = set()
    for name, (_, external) in moving.items():
        earlier = set(external)
        for parent in external:
            earlier.update(ancestors(factors, parent))
        if name not in {aliases.get(parent, parent) for parent in earlier}:
            requires[name] = external
            if direct and name in resolved:
                promoted.update(earlier.intersection(factors) - resolved)
    resolved.update(promoted)
    ordering_aliases = aliases
    order_parents = dict(requires)
    if point_projection:
        ordering_aliases = {**aliases, sampled.pitch: sampled.look}
    elif emission_ties:
        # A known child of the pitch is evaluated on the physical line of sight.
        # Ancestors that generate that sightline retain their declared conditioning.
        view_parents = ancestors(factors, sampled.look)
        for name, law in sources.items():
            if name not in (sampled.look, *view_parents) and sampled.pitch in parents(
                law
            ):
                order_parents[name] = (*order_parents.get(name, ()), sampled.look)
    for name, law in ordered(
        sources, available=blocked, aliases=ordering_aliases, requires=order_parents
    ):
        factor = 2 if name in refine else 1
        dependency = {aliases.get(p, p) for p in parents(law)}
        if name in resolved:
            if dependency - resolved:
                raise RoleError(
                    f"{name} is resolved and depends on averaged {sorted(dependency - resolved)}"
                )
            continue
        if not dependency and (
            isinstance(law, Conditional) or (callable(law) and not hasattr(law, "rule"))
        ):
            law = variables[name].place(law_at(law, {}, units))
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
            and name not in aliases.values()
            and name not in parent_names
        ):
            # The continuum kernel already includes this pitch average.
            fixed[name] = 0.0
        elif not hasattr(law, "rule") and not dependency and not callable(law):
            fixed[name] = float(law)
        elif getattr(law, "is_point", False) and not dependency:
            fixed[name] = float(law.rule()[0][0])
        elif name in requires or (dependency and beam_knots.get(name)):
            source = next(source for source, target in ties if target == name)
            folded = geometry.get(name, factors[source])
            external = requires.get(name, ())

            def evaluate(
                point,
                law=law,
                folded=folded,
                name=name,
                factor=factor,
                breaks=beam_knots.get(name),
            ):
                return moving_beaming_rule(
                    law, folded, name, units, variables[name], factor, point, breaks
                )

            rules.append(
                (name, NodeRule(evaluate, tuple(sorted(dependency.union(external)))))
            )
        elif beam_knots.get(name) and not dependency and hasattr(law, "density"):
            nodes, weights = split_beaming_rule(law, beam_knots[name], factor=factor)
            rules.append((name, NodeRule(lambda _, x=nodes, w=weights: (x, w))))
        else:
            rule = _rule(
                law,
                units,
                variables[name],
                factor,
                physical_column=direct and name == kernel.primary,
            )
            rules.append((name, rule))
    panels = projection_panels(
        factors, ties, beam_knots, legendre, requires, units, geometry
    )
    if point_projection:
        panels = (*panels, (sampled.pitch, sampled.projection_rule()))
    return (
        AveragedKernel(
            kernel,
            rules,
            legendre=legendre,
            fixed=fixed,
            phase=phase,
            ties=ties,
            leaf_factors=(*beam_factors, *leaf_factors),
            diagonal=() if direct else diagonal,
            emission_ties=emission_ties,
            legendre_panels=panels,
        ),
        spin,
        promoted,
        beam_knots,
        geometry,
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
    continua = continuum_children(kernel)
    if continua:
        # A mixed kernel need not share the continuum child's field-scaling
        # capability. The continuation still depends on the physical field.
        if bound is None:
            if not direct and "B" in lattice.cells.variables:
                bound = float(
                    lattice.cells.hi[:, lattice.cells.variables.index("B")].max()
                )
            else:
                bound = support_for(pop, "B")[1]
        if any(child.x_min is None for child in continua):
            top = (
                float(lattice.cells.hi[:, lattice.cells.variables.index("gamma")].max())
                if "gamma" in lattice.cells.variables
                else support_for(pop, "gamma")[1]
            )
            if bound and np.isfinite(bound) and np.isfinite(top):
                if isinstance(kernel, HybridKernel):
                    kernel = kernel._with_evaluators(
                        (
                            child.resolved(channels, top, bound)
                            if type(child) is ContinuumKernel and child.x_min is None
                            else child
                        )
                        for child in kernel.kernels
                    )
                else:
                    kernel = kernel.resolved(channels, top, bound)
    if not direct:
        if isinstance(kernel, HybridKernel):
            # Assignment sees the geometric cell. Each child's validity check
            # also sees the declared support of the variables averaged inside.
            intervals = {
                name: support_for(pop, name)
                for name, law in pop.factors.items()
                if name not in names
                and (
                    hasattr(law, "support")
                    or np.isscalar(law)
                    or getattr(law, "shape", None) == ()
                )
            }
            kernel = kernel._with_regime_support(intervals)
        for cell, box in enumerate(cell_boxes(lattice.cells)):
            if type(kernel) is ContinuumKernel and kernel.primary not in box:
                box = {**box, kernel.primary: support_for(pop, kernel.primary)}
            result = kernel.regime(box, channels)
            if not result.ok:
                raise RegimeError(f"cell {cell}: {result.reason}")
    if strict and defaults:
        raise RoleError(f"strict prediction requires explicit defaults: {defaults}")
    diagonal = resolved_beaming(kernel, pop, lattice)
    averaged, spin, promoted, beam_knots, geometry = _average(
        kernel, pop, lattice, phase, units, direct, diagonal, refine
    )
    resolved.update((name, pop.factors[name]) for name in promoted)
    if direct:
        sampled = sample_beaming(kernel, pop, lattice)
        if sampled is not None and not sampled.smooth and sampled.pitch in pop.factors:
            resolved[sampled.pitch] = pop.factors[sampled.pitch]
        for source, _ in averaged.ties:
            resolved.pop(source, None)
    return ForwardSetup(
        pop, averaged, resolved, units, spin, bound, defaults, beam_knots, geometry
    )


def column_rules(setup, lattice):
    kernel = setup.averaged.kernel
    resolved = set(lattice.cells.variables) | set(lattice.representations)
    return ColumnRules(
        merge=merge_beaming(kernel, lattice),
        symmetries=tuple(
            name for name in setup.population.symmetries if name == "pitch_reversal"
        ),
        depth_free_spin0=type(kernel) is ContinuumKernel
        and all(
            kernel.depth not in parents(law)
            for name, law in setup.population.factors.items()
            if name not in resolved
        ),
    )
