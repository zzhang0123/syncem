"""Node-doubling diagnostics on a bounded subset of the direct panels."""

import jax.numpy as jnp

from ..base.hybrid import HybridKernel
from ..engine.direct import integrate
from ..population.declaration import parents
from ..records.ledger import ErrorTerm
from ..values.errors import RoleError
from ._direct import _panel_geometry, _stokes_array
from ._setup import configure


def _sample(basis, setup, channels, cells, nodes, tail):
    modes = integrate(
        setup.averaged,
        channels,
        _panel_geometry(
            setup, basis.lattice, channels, "cells", nodes=nodes, cells=cells
        ),
        b_perp_max=setup.b_max,
        azimuth_factor=setup.azimuth,
    )
    tail.record(setup, basis.lattice.cells, channels, "cells", cells)
    return _stokes_array(basis, modes)


def quadrature_terms(basis, checks, populations, budget, direct_cost, tail):
    """Compare the aggregate spectrum of the first, middle and last cells."""
    lat, ch = basis.lattice, basis.channels
    kernel = basis.averaged.kernel
    count = lat.n_cells
    cells = tuple(sorted({0, count // 2, count - 1}))
    setups = tuple(setup for _, setup in checks)
    depth = kernel.depth
    coupled_depth = tuple(
        depth is not None
        and any(name == depth for name, _ in setup.averaged.averaged)
        and any(depth in parents(law) for law in setup.population.factors.values())
        for setup in setups
    )
    leaf_depth = tuple(
        depth is not None
        and any(name == "depth" for name, _ in setup.averaged.leaf_factors)
        for setup in setups
    )
    unavailable = ["quadrature"]
    if any(coupled_depth):
        unavailable.append("quadrature:depth")
    route_cost = sum(
        (
            1
            + 2
            ** (
                len(setup.resolved)
                + len(setup.averaged.averaged)
                + leaf_check
                + (ch.family != "monochromatic")
            )
        )
        * (2 if depth_check else 1)
        for setup, depth_check, leaf_check in zip(
            setups, coupled_depth, leaf_depth, strict=True
        )
    )
    # Limited probes and a conservative estimate of the added work. Responses
    # above 512 nodes already carry a large quadrature graph.
    estimate = direct_cost * len(cells) / count * route_cost
    if estimate > budget or ch.n_nu > 512 or isinstance(kernel, HybridKernel):
        return tuple(
            ErrorTerm.unbounded(
                "node doubling was unavailable within the check budget", name=name
            )
            for name in unavailable
        )
    raw, doubled, high_depth = 0, 0, 0
    try:
        high = ch._double_nodes()
        for setup, population, depth_check, leaf_check in zip(
            setups, populations, coupled_depth, leaf_depth, strict=True
        ):
            rules = tuple(
                name
                for name, _ in setup.averaged.averaged
                if not (depth_check and name == depth)
            )
            if leaf_check:
                rules = (*rules, depth)
            raw_part = _sample(basis, setup, ch, cells, 24, tail)
            refined = configure(
                population,
                kernel,
                high,
                lat,
                basis.forward_phase,
                direct=True,
                refine=rules,
            )
            raw += raw_part
            doubled += _sample(basis, refined, high, cells, 48, tail)
            if depth_check:
                refined = configure(
                    population,
                    kernel,
                    ch,
                    lat,
                    basis.forward_phase,
                    direct=True,
                    refine=(depth,),
                )
                high_depth += _sample(basis, refined, ch, cells, 24, tail) - raw_part
    except (RoleError, TypeError) as exc:
        return tuple(
            ErrorTerm.unbounded(f"node doubling unavailable. {exc}", name=name)
            for name in unavailable
        )
    terms = [
        ErrorTerm(
            jnp.abs(doubled - raw),
            "estimate",
            f"Node doubling in probe cells {cells}. The other cells were unsampled.",
            name="quadrature",
        )
    ]
    if any(coupled_depth):
        terms.append(
            ErrorTerm(
                jnp.abs(high_depth),
                "estimate",
                f"Depth node doubling in probe cells {cells} with all other rules fixed.",
                name="quadrature:depth",
            )
        )
    return tuple(terms)
