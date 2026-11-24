"""Replace selected conditional laws by fixed parent quadrature mixtures."""

from __future__ import annotations

import inspect
import itertools
import math
from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..distributions import (
    AzimuthFactor,
    Conditional,
    Free,
    Isotropic,
    NoRotation,
    UniformAzimuth,
)
from ..units import Quantity
from ..values.errors import RoleError
from .blocks import FreeMoments, NonNegativeCells
from .declaration import Population, parents

_DEFAULTS = {
    "Isotropic": Isotropic,
    "NoRotation": NoRotation,
    "UniformAzimuth": UniformAzimuth,
}


def _contains_free(law):
    leaves = jax.tree.leaves(law, is_leaf=lambda item: isinstance(item, Free))
    return any(isinstance(leaf, Free) for leaf in leaves)


def _parent_rule(part, variable):
    name = variable.name
    message = f"average= needs a specified independent marginal with a rule for {name}"
    if name in part.factors:
        source = part.factors[name]
    elif any(name in getattr(block, "variables", ()) for block in part.blocks):
        raise RoleError(f"{message}. Its source is a joint block.")
    elif variable.default in _DEFAULTS:
        source = _DEFAULTS[variable.default]()
    else:
        raise RoleError(message)
    if (
        parents(source)
        or isinstance(source, (Conditional, FreeMoments, NonNegativeCells))
        or _contains_free(source)
    ):
        raise RoleError(message)
    law = variable.place(source)
    if hasattr(law, "rule"):
        nodes, weights = (np.asarray(x) for x in law.rule())
    elif isinstance(law, (int, float, np.number)):
        nodes, weights = np.asarray([law]), np.ones(1)
    else:
        raise RoleError(message)
    if (
        nodes.ndim != 1
        or weights.shape != nodes.shape
        or np.iscomplexobj(nodes)
        or np.iscomplexobj(weights)
        or not np.all(np.isfinite(nodes))
        or not np.all(np.isfinite(weights) & (weights >= 0))
    ):
        raise RoleError(f"{message}. Its quadrature is invalid.")
    active = weights > 0
    mass = np.sum(weights)
    if not np.isfinite(mass) or mass <= 0:
        raise RoleError(f"{message}. Its quadrature has no finite positive mass.")
    return tuple(zip(nodes[active], weights[active] / mass, strict=True))


def _draws(names, rules):
    for choices in itertools.product(*(rules[name] for name in names)):
        yield (
            dict(zip(names, (node for node, _ in choices), strict=True)),
            math.prod(weight for _, weight in choices),
        )


def _evaluate(source, point, params, units):
    if isinstance(source, Conditional):
        return source.evaluate(point, params=params, canonical_units=units)
    arguments = {
        name: Quantity(value, units[name])
        if units.get(name) and not isinstance(value, Quantity)
        else value
        for name, value in point.items()
    }
    return source(**arguments)


def _renamed(source, removed, draws, units, *, spin=False):
    remaining = tuple(name for name in parents(source) if name not in removed)
    parameters = source.params if isinstance(source, Conditional) else {}

    def at(**values):
        point = {name: values[name] for name in remaining}
        params = {name: values[name] for name in parameters}
        if not spin:
            fixed, _ = draws[0]
            return _evaluate(source, {**point, **fixed}, params, units)
        coefficient = 0j
        for fixed, weight in draws:
            law = _evaluate(source, {**point, **fixed}, params, units)
            factor = law if hasattr(law, "circular") else AzimuthFactor(law)
            coefficient += weight * factor.circular(2)
        return coefficient / jnp.maximum(1, jnp.abs(coefficient))

    if not remaining:
        if _contains_free(parameters):
            raise RoleError(
                "evaluate the conditional's free parameters before averaging all its parents"
            )
        result = at(**parameters)
        return AzimuthFactor(result) if spin else result
    at.__signature__ = inspect.Signature(
        [
            inspect.Parameter(name, kind=inspect.Parameter.KEYWORD_ONLY)
            for name in (*remaining, *parameters)
        ]
    )
    src_units = dict(source.units) if isinstance(source, Conditional) else {}
    src_parity = dict(source.parity) if isinstance(source, Conditional) else {}
    conditional = Conditional(
        at,
        params=parameters,
        units={name: src_units[name] for name in remaining if name in src_units},
        parity={name: src_parity[name] for name in remaining if name in src_parity},
        name=getattr(source, "name", getattr(source, "__name__", "marginal")),
    )
    return AzimuthFactor(conditional) if spin else conditional


def _part_plan(part, requested, definitions):
    edges = {
        name: dropped
        for name, law in part.factors.items()
        if (dropped := tuple(parent for parent in parents(law) if parent in requested))
    }
    for block in part.blocks:
        if requested.intersection(getattr(block, "parents", ())):
            raise RoleError("average= cannot marginalize a conditional joint block")
    rules = {
        name: _parent_rule(part, definitions[name])
        for name in dict.fromkeys(name for group in edges.values() for name in group)
    }
    n = math.prod(
        len(rules[parent])
        for child, group in edges.items()
        if not isinstance(part.factors[child], AzimuthFactor)
        for parent in group
    )
    spin_draws = max(
        (
            math.prod(len(rules[parent]) for parent in group)
            for child, group in edges.items()
            if isinstance(part.factors[child], AzimuthFactor)
        ),
        default=1,
    )
    return edges, rules, n, spin_draws


def _parts(part, edges, rules, units):
    if not edges:
        return (part,)
    children, choices = [], []
    for child, dropped in edges.items():
        law = part.factors[child]
        children.append(child)
        units_in = {name: units[name] for name in parents(law) if name in units}
        if isinstance(law, AzimuthFactor):
            draws = tuple(_draws(dropped, rules))
            choices.append(
                ((_renamed(law.c, dropped, draws, units_in, spin=True), 1.0),)
            )
        else:
            choices.append(
                tuple(
                    (_renamed(law, dropped, ((draw, weight),), units_in), weight)
                    for draw, weight in _draws(dropped, rules)
                )
            )
    output = []
    for selected in itertools.product(*choices):
        changes = dict(zip(children, (law for law, _ in selected), strict=True))
        weight = math.prod(weight for _, weight in selected)
        output.append(weight * part.replace(**changes))
    return tuple(output)


def marginalize(population, kernel, average, *, max_components=256):
    """Return a population and its declared independence assumptions.

    A dropped edge uses the unconditional parent probability rule even when
    the parent factor remains on the lattice. Each child receives its own
    draw, with the surviving dependencies and the original child
    normalisation. Joint sources need an explicit marginal before calling.
    """
    if isinstance(average, str):
        raise RoleError("average= must be a sequence of variable names")
    requested = tuple(average)
    if any(not isinstance(name, str) for name in requested):
        raise RoleError("average= must name kernel variables")
    requested = set(requested)
    if not requested:
        return population, ()
    definitions = {variable.name: variable for variable in kernel.variables}
    unknown = requested - set(definitions)
    if unknown:
        raise RoleError(
            f"average= names variables absent from the kernel: {sorted(unknown)}"
        )
    if not isinstance(population, Population):
        population = Population(population)
    if (
        not isinstance(max_components, Integral)
        or isinstance(max_components, (bool, np.bool_))
        or max_components < 1
    ):
        raise ValueError("max_components must be a positive integer")
    original = population.parts
    plans = tuple(_part_plan(part, requested, definitions) for part in original)
    assumptions = tuple(
        dict.fromkeys(
            f"independence:{child}|{parent}"
            for edges, _, _, _ in plans
            for child, names in edges.items()
            for parent in names
        )
    )
    if not assumptions:
        return population, assumptions
    total = sum(n for _, _, n, _ in plans)
    maximum = max(draws for _, _, _, draws in plans)
    if total > max_components or maximum > max_components:
        raise RoleError(
            f"average= needs {total} components and up to {maximum} spin-factor draws. "
            f"The marginalization limit is {max_components}. Use fewer parent quadrature nodes."
        )
    units = {
        name: variable.unit for name, variable in definitions.items() if variable.unit
    }
    result = tuple(
        member
        for part, (edges, rules, _, _) in zip(original, plans, strict=True)
        for member in _parts(part, edges, rules, units)
    )
    transformed = (
        result[0]
        if len(result) == 1
        else eqx.tree_at(lambda pop: pop.components, Population(), result)
    )
    return transformed, assumptions
