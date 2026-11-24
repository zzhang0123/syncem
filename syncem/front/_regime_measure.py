"""Physical measure panels and moment pairing in the two response coordinates."""

import numpy as np

from ..base.variables import SYNCHROTRON_VARIABLES
from ..base.window import A1_HZ_PER_G
from ..lattice.moments import Moments
from ..population.declaration import law_at, parents
from ..population.pairing import _basis_values
from ..population.quadrature import _axis, ownership, panels
from ..values.errors import RegimeError, RoleError
from ._leaf import conditional_spin, spin_factor

_UNITS = {name: var.unit for name, var in SYNCHROTRON_VARIABLES.items()}


def dependencies(factors, name):
    """Return the transitive parents of one physical variable."""
    found, pending = set(), list(parents(factors.get(name)))
    while pending:
        parent = pending.pop()
        if parent not in found:
            found.add(parent)
            pending.extend(parents(factors.get(parent)))
    return found


def _has_density(law):
    return hasattr(law, "density") and not getattr(law, "is_point", False)


def _diagonal_routes(factors):
    """Use one eligible continuous cosine law at each physical parent node."""
    routes = []
    for tied, driver in (("mu", "eta"), ("eta", "mu")):
        law = factors.get(tied)
        if tied not in dependencies(factors, driver) and (
            parents(law) or _has_density(law_at(law, {}, _UNITS))
        ):
            routes.append((tied, driver))
    if not routes:
        raise RoleError(
            "the continuum measure needs a cosine density on the beaming diagonal"
        )
    preferred = factors[routes[0][0]]
    needed = parents(preferred)
    if len(routes) == 1 or not needed:
        return ((routes[0], None),)

    def use_preferred(law, _):
        return _has_density(law)

    def use_reverse(_, current):
        values = [current[name] for name in needed]
        return np.asarray(
            [
                not _has_density(
                    law_at(preferred, dict(zip(needed, row, strict=True)), _UNITS)
                )
                for row in zip(*values, strict=True)
            ]
        )

    return ((routes[0], use_preferred), (routes[1], use_reverse))


def _diagonal_bounds(factors, diagonal):
    tied, driver = diagonal
    support = getattr(factors.get(tied), "support", None)
    return {} if support is None else {driver: support}


def _spin_law(factors):
    if any("phi" in parents(law) for law in factors.values()):
        return None
    law = factors.pop("phi", None)
    if law is None:
        return None
    if parents(law):
        evaluate = conditional_spin(law, _UNITS)

        def spin(points):
            args = tuple(points[name] for name in parents(law))
            names = parents(law)
            return np.asarray(
                [
                    evaluate(dict(zip(names, v, strict=True)))
                    for v in zip(*args, strict=True)
                ]
            )

        return spin
    value = complex(spin_factor(law))
    return lambda _: value


def _transverse(B, eta):
    return B * np.sqrt(np.maximum(0, (1 - eta) * (1 + eta)))


def harmonic_spacing(gamma, B, eta):
    """Frequency spacing on the beaming diagonal, evaluated without pole cancellation."""
    beta = np.sqrt(1 - gamma**-2)
    doppler = (1 - eta) * (1 + eta) + eta**2 / (gamma**2 * (1 + beta))
    return (2 / 3) * A1_HZ_PER_G * B / (gamma * doppler)


def _below_switch(pop, switch):
    law = pop.factors.get("gamma")
    if law is not None:
        support = getattr(law, "support", None)
        if support is not None:
            return float(support[1]) < switch or (
                float(support[1]) == switch
                and hasattr(law, "density")
                and not getattr(law, "is_point", False)
            )
        if not parents(law) and not hasattr(law, "rule"):
            return float(law) < switch
    for block in pop.blocks:
        if "gamma" in block.variables:
            return bool(np.all(np.asarray(block.nodes["gamma"]) < switch))
    return False


def physical_panels(
    pop,
    lattice,
    branch,
    switch,
    field,
    averaged,
    *,
    nodes,
    min_width=None,
    retain_phi=False,
):
    """Yield coordinates, spin weights and both original and projected mass."""
    continuum = branch == "continuum"
    if continuum and _below_switch(pop, switch):
        return
    factors = {name: law for name, law in pop.factors.items() if name not in averaged}
    spin = None if retain_phi else _spin_law(factors)
    for point, weight in _branch_panels(
        pop, factors, lattice, continuum, switch, nodes
    ):
        energy = point["gamma"]
        keep = (energy >= switch) if continuum else (energy < switch)
        keep &= np.asarray(weight) != 0
        if not np.any(keep):
            continue
        point = {name: values[keep] for name, values in point.items()}
        mass = np.asarray(weight)[keep] * float(pop.column_factor)
        azimuth = np.exp(2j * point["phi"]) if spin is None else spin(point)
        if continuum:
            bp = _transverse(point["B"], point["eta"])
            if min_width is not None:
                spacing = harmonic_spacing(point["gamma"], point["B"], point["eta"])
                if np.any((bp > 0) & (min_width < 32 * spacing)):
                    raise RegimeError(
                        "the narrowest channel contains fewer than 32 harmonic spacings on the sampled continuum measure"
                    )
            nu_c = A1_HZ_PER_G * bp * point["gamma"] ** 2
            point = {**point, "nu_c": nu_c}
            mass = 2 * bp / field * mass
        yield point, azimuth, mass


def _gamma_splits(factors, lattice, switch, driver):
    """Split a continuous energy law on the effective cell faces when possible."""
    splits = {}
    extra = {}
    gamma_law = factors.get("gamma")
    # A mass point has no interior panel to split. Keep it vectorised for the
    # study's physical-cell projection.
    splittable_gamma = (
        hasattr(gamma_law, "density") and not getattr(gamma_law, "is_point", False)
    ) or bool(parents(gamma_law))
    if splittable_gamma:
        axis = lattice.cells.variables.index("nu_c")
        faces = np.unique(np.r_[lattice.cells.lo[:, axis], lattice.cells.hi[:, axis]])
        geometry = {"B", driver}
        if not any("gamma" in dependencies(factors, name) for name in geometry):

            def gamma_faces(point):
                bp = _transverse(point["B"], point[driver])
                return np.sqrt(faces / (A1_HZ_PER_G * bp)) if bp > 0 else ()

            splits["gamma"] = (tuple(sorted(geometry)), gamma_faces)
        else:
            support = getattr(gamma_law, "support", None)
            if support is not None and np.isfinite(support[1]) and support[1] > switch:
                extra["gamma"] = np.geomspace(switch, support[1], 64)[1:-1]
    return splits, extra


def _branch_panels(pop, factors, lattice, continuum, switch, nodes):
    routes = _diagonal_routes(factors) if continuum else ((None, None),)
    for diagonal, use_route in routes:
        if continuum:
            bounds = {"gamma": (switch, np.inf), **_diagonal_bounds(factors, diagonal)}
            splits, extra = _gamma_splits(factors, lattice, switch, diagonal[1])
        else:
            bounds, splits, extra = {"gamma": (1.0, switch)}, {}, {}
        for axis, name in enumerate(lattice.cells.variables):
            if name != "nu_c":
                extra[name] = np.unique(
                    np.r_[lattice.cells.lo[:, axis], lattice.cells.hi[:, axis]]
                )
        yield from panels(
            factors,
            bounds,
            blocks=pop.blocks,
            units=_UNITS,
            variables=SYNCHROTRON_VARIABLES,
            charts={"gamma": "log", "B": "log"} if continuum else {"gamma": "log"},
            breaks=extra,
            n=nodes,
            diagonal=diagonal,
            diagonal_filter=use_route,
            split_points=splits,
        )


def pair_branch(parts, basis, branch, switch, field, averaged, *, nodes, channels=None):
    """Accumulate basis rows with shared-face ownership and signed spin two."""
    lattice, total, outside = basis.lattice, 0.0, 0.0
    index = lattice.index
    rows = np.zeros((len(index.blocks), index.block_size))
    width = (
        None
        if channels is None
        else float(np.min(np.diff(np.asarray(channels.support), axis=1)))
    )
    for pop in parts:
        for points, spin, mass in physical_panels(
            pop, lattice, branch, switch, field, averaged, nodes=nodes, min_width=width
        ):
            cells = ownership(points, lattice.cells, mass.size)
            total += mass.sum()
            outside += mass[cells < 0].sum()
            for cell in np.unique(cells[cells >= 0]):
                keep = cells == cell
                selected = {name: values[keep] for name, values in points.items()}
                values = _basis_values(index, cell, selected, int(keep.sum()))
                weighted = values * mass[keep, None] / lattice.weights[cell]
                span = index.cell_rows(cell)
                rows[0, span] += weighted.sum(axis=0)
                coefficient = np.broadcast_to(spin, mass.shape)[keep] @ weighted
                rows[1, span] += coefficient.real
                rows[2, span] += coefficient.imag
    moment = Moments(lattice, rows.ravel(), component=branch)
    return moment, float(total), float(outside)


def source_columns(parts, switch, *, nodes):
    """Measure the physical electron column on either side of the switch."""
    low, high = 0.0, 0.0
    for pop in parts:
        needed = {"gamma", *dependencies(pop.factors, "gamma")}
        factors = {name: law for name, law in pop.factors.items() if name in needed}
        for point, weight in panels(
            factors,
            {"gamma": (1.0, np.inf)},
            blocks=pop.blocks,
            units=_UNITS,
            variables=SYNCHROTRON_VARIABLES,
            charts={"gamma": "log"},
            breaks={"gamma": (switch,)},
            n=nodes,
        ):
            mass = np.asarray(weight) * float(pop.column_factor)
            lower = point["gamma"] < switch
            low += float(mass[lower].sum())
            high += float(mass[~lower].sum())
    return low, high


def projected_continuum_width(factors, lattice, switch, *, nodes):
    """Required channel width over fit cells and sampled fixed geometry."""
    table = lattice.cells
    energy = table.variables.index("gamma")
    upper = table.lo[:, energy] >= switch
    if not np.any(upper):
        return 0.0
    lo, hi = table.lo[upper, energy], table.hi[upper, energy]
    angular = {name: factors[name] for name in ("mu", "eta")}
    max_eta = -1.0
    for diagonal, use_route in _diagonal_routes(angular):
        for point, mass in panels(
            angular,
            _diagonal_bounds(angular, diagonal),
            units=_UNITS,
            variables=SYNCHROTRON_VARIABLES,
            n=nodes,
            diagonal=diagonal,
            diagonal_filter=use_route,
        ):
            eta = np.abs(point["eta"])
            emitting = (eta < 1) & (mass != 0)
            max_eta = max(max_eta, np.max(eta[emitting], initial=-1.0))
    if max_eta < 0:
        return 0.0
    if "B" in table.variables:
        field = table.hi[upper, table.variables.index("B")]
    else:
        field = max(
            (
                np.max(values[np.asarray(mass) != 0], initial=0.0)
                for values, mass in _axis(
                    factors["B"], (-np.inf, np.inf), "log", nodes, ()
                )
            ),
            default=0.0,
        )
    # For fixed eta, gamma - eta^2 * sqrt(gamma^2 - 1) has its minimum
    # where beta = eta^2. The interval maximum spacing is at that point
    # or the nearest edge, including the regime-switch face.
    square = max_eta**2
    gamma = np.clip(1 / np.sqrt((1 - square) * (1 + square)), lo, hi)
    return float(32 * np.max(harmonic_spacing(gamma, field, max_eta)))


def diagonal_energy_moments(parts, lattice, *, nodes):
    """Pair the beaming-diagonal energy law before any field reweighting."""
    if lattice.cells.variables != ("gamma",) or lattice.representations:
        raise RoleError("the reference diagonal energy law needs a gamma-only lattice")
    table, index = lattice.cells, lattice.index
    values = np.zeros(index.block_size)
    faces = np.unique(np.r_[table.lo[:, 0], table.hi[:, 0]])
    for pop in parts:
        needed = {"gamma", "eta", "mu"}
        for name in tuple(needed):
            needed |= dependencies(pop.factors, name)
        factors = {key: law for key, law in pop.factors.items() if key in needed}
        for tie, use_route in _diagonal_routes(factors):
            for point, mass in panels(
                factors,
                {"gamma": (faces[0], faces[-1])},
                blocks=pop.blocks,
                units=_UNITS,
                variables=SYNCHROTRON_VARIABLES,
                charts={"gamma": "log"},
                breaks={"gamma": faces},
                diagonal=tie,
                diagonal_filter=use_route,
                n=nodes,
            ):
                cells = ownership(point, table, len(mass))
                for cell in np.unique(cells[cells >= 0]):
                    keep = cells == cell
                    powers = _basis_values(
                        index, cell, {"gamma": point["gamma"][keep]}, int(keep.sum())
                    )
                    scale = 2 * float(pop.column_factor) / lattice.weights[cell]
                    values[index.cell_rows(cell)] += (
                        scale * np.asarray(mass)[keep] @ powers
                    )
    return Moments(lattice, values)
