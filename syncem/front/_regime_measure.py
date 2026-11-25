"""Physical measure panels and moment pairing in the two response coordinates."""

from functools import lru_cache

import numpy as np

from ..base.variables import SYNCHROTRON_VARIABLES
from ..base.window import A1_HZ_PER_G
from ..lattice._cell_values import log_displacement
from ..lattice.moments import Moments
from ..population._diagonal import fixed_angular_knots, knot_parents, knot_paths
from ..population._node_offsets import point_displacement, sum_with_residual
from ..population._panel_cuts import PanelCuts
from ..population.declaration import ancestors, law_at, parents, placed_law
from ..population.pairing import _basis_values
from ..population.quadrature import _axis, ownership, panels
from ..values.errors import RegimeError, RoleError
from ._leaf import conditional_spin, spin_factor
from ._regime_frequency import (
    inverse_frequency_faces,
    projected_frequency,
)

_UNITS = {name: var.unit for name, var in SYNCHROTRON_VARIABLES.items()}


def _has_density(law):
    return hasattr(law, "density") and not getattr(law, "is_point", False)


def _diagonal_routes(factors):
    """Use one eligible continuous cosine law at each physical parent node."""
    routes = []
    for tied, driver in (("mu", "eta"), ("eta", "mu")):
        law = factors.get(tied)
        if tied not in ancestors(factors, driver) and (
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


def harmonic_spacing(gamma, B, eta, *, sine=None):
    """Frequency spacing on the beaming diagonal, evaluated without pole cancellation."""
    beta = np.sqrt(1 - gamma**-2)
    sine_squared = (1 - eta) * (1 + eta) if sine is None else sine**2
    doppler = sine_squared + eta**2 / (gamma**2 * (1 + beta))
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


def _below_switch_mask(energy, switch, shift):
    """Assign rounded continuous nodes below the switch while keeping switch atoms above."""
    if shift is None:
        return energy < switch
    return (energy < switch) | ((energy == switch) & (shift < 0))


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
    return_offsets=False,
):
    """Yield coordinates, spin weights and both original and projected mass."""
    continuum = branch == "continuum"
    if continuum and _below_switch(pop, switch):
        return
    factors = {name: law for name, law in pop.factors.items() if name not in averaged}
    spin = None if retain_phi else _spin_law(factors)
    offset_axes = set(lattice.cells.variables) | {"gamma"}
    if continuum:
        offset_axes.update(("B", "eta", "mu"))
    for point, weight, shifts in _branch_panels(
        pop,
        factors,
        lattice,
        continuum,
        switch,
        nodes,
        offset_axes=offset_axes,
    ):
        energy = point["gamma"]
        below = _below_switch_mask(energy, switch, shifts.get("gamma"))
        keep = (~below) if continuum else below
        keep &= np.asarray(weight) != 0
        if not np.any(keep):
            continue
        point = {name: values[keep] for name, values in point.items()}
        shifts = {name: values[keep] for name, values in shifts.items()}
        mass = np.asarray(weight)[keep] * float(pop.column_factor)
        azimuth = np.exp(2j * point["phi"]) if spin is None else spin(point)
        if continuum:
            sine, bp, nu_c, residual = projected_frequency(point, shifts)
            if residual is not None:
                shifts["nu_c"] = residual
            if min_width is not None:
                spacing = harmonic_spacing(
                    point["gamma"], point["B"], point["eta"], sine=sine
                )
                if np.any((bp > 0) & (min_width < 32 * spacing)):
                    raise RegimeError(
                        "the narrowest channel contains fewer than 32 harmonic spacings on the sampled continuum measure"
                    )
            point = {**point, "nu_c": nu_c}
            mass = 2 * bp / field * mass
        if return_offsets:
            yield point, azimuth, mass, shifts
        else:
            yield point, azimuth, mass


def _measure_kind(law):
    if _has_density(law):
        return "density"
    if getattr(law, "is_point", False) or not hasattr(law, "rule"):
        return "point"
    return "nodes"


def _declared_kind(law):
    names = parents(law)
    if not names:
        return _measure_kind(law)
    import jax

    found = []

    def inspect(*values):
        found.append(_measure_kind(law_at(law, dict(zip(names, values)), _UNITS)))
        return ()

    try:
        jax.eval_shape(inspect, *(jax.ShapeDtypeStruct((), float) for _ in names))
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        # A Python law factory may choose between points and densities at run time.
        return "unknown"
    return found[0]


def _point_value(law, name):
    placed = placed_law(law, SYNCHROTRON_VARIABLES[name])
    return float(placed.rule()[0][0])


def _face_dependencies(factors, tied, driver, splits):
    deps = {name: set(parents(law)) for name, law in factors.items()}
    for name, (required, _) in splits.items():
        deps[name].update(required)
    if tied is not None:
        deps[tied].add(driver)
        # The driver follows the parents that determine its folded pitch panels.
        folded = factors[tied]
        if parents(folded) and fixed_angular_knots(folded, _UNITS) is None:
            paths = knot_paths(factors, tied, driver, None)
            deps.setdefault(driver, set()).update(set(knot_parents(paths)) - {driver})

    def closure(name):
        found, pending = set(), list(deps.get(name, ()))
        while pending:
            parent = pending.pop()
            if parent not in found:
                found.add(parent)
                pending.extend(deps.get(parent, ()))
        return found

    return {name: closure(name) for name in deps}


def _resolved_root(distance, candidate, limit):
    """Interpolate the crossing inside its final adjacent physical floats."""
    value = distance(candidate)
    if value == 0:
        return candidate, 0.0
    direction = np.inf if limit > candidate else -np.inf
    step = np.nextafter(candidate, direction) - candidate
    outer = candidate + step
    while abs(outer - candidate) < abs(limit - candidate):
        other = distance(outer)
        if np.sign(other) != np.sign(value):
            break
        step *= 2
        outer = candidate + step
    else:
        outer = limit
        other = distance(outer)
    while outer != candidate:
        mid = candidate + (outer - candidate) / 2
        if mid == candidate or mid == outer:
            break
        probe = distance(mid)
        if np.sign(probe) == np.sign(value):
            candidate, value = mid, probe
        else:
            outer, other = mid, probe
    if other == 0:
        return outer, 0.0
    displacement = (outer - candidate) * (value / (value - other))
    return sum_with_residual(candidate, displacement)


def _derived_faces(frequency, support, faces, *, logarithmic):
    """Find bracketed cell faces on a deterministic conditional trajectory."""
    from scipy.optimize import brentq, minimize_scalar

    low, high = map(float, support)
    if not low < high:
        return ()

    # Compactify infinite tails. A finite search of an arbitrary conditional
    # function has no bound on unresolved oscillations.
    def coordinate(t):
        if np.isfinite(low) and np.isfinite(high):
            if logarithmic and low > 0:
                span = float(log_displacement(high, low))
                if high - low <= 0.5 * low:
                    return low + low * np.expm1(t * span)
                return np.exp(np.log(low) + t * span)
            return low + t * (high - low)
        if np.isfinite(low):
            return low + max(1.0, abs(low)) * t / (1 - t)
        if np.isfinite(high):
            return high - max(1.0, abs(high)) * (1 - t) / t
        return np.tan(np.pi * (t - 0.5))

    projected = lru_cache(maxsize=256)(frequency)

    def evaluate(t):
        return projected(coordinate(t))[0]

    grid = np.linspace(
        0.0 if np.isfinite(low) else np.finfo(float).eps,
        1.0 if np.isfinite(high) else 1.0 - np.finfo(float).eps,
        65,
    )
    values = np.array([evaluate(t) for t in grid])
    changes = np.diff(values)
    boundaries = list(zip(grid, values))
    for i in np.flatnonzero(changes[:-1] * changes[1:] < 0) + 1:
        sign = np.sign(changes[i])
        extremum = minimize_scalar(
            lambda t: sign * evaluate(t),
            bounds=(grid[i - 1], grid[i + 1]),
            method="bounded",
            options={"xatol": 1e-14},
        )
        # A second sample of the same rounded extremum can manufacture a
        # finite interval at a single tangency with a regime face.
        if extremum.success and sign * extremum.fun != values[i]:
            boundaries.append((extremum.x, sign * extremum.fun))
    boundaries.sort()
    roots = []
    for (a, fa), (b, fb) in zip(boundaries, boundaries[1:]):
        if not (np.isfinite(fa) and np.isfinite(fb)):
            continue
        start = np.searchsorted(faces, min(fa, fb))
        stop = np.searchsorted(faces, max(fa, fb), side="right")
        for face in faces[start:stop]:

            def distance(x):
                value, shift = projected(x)
                return (value - face) + shift

            left, right = distance(coordinate(a)), distance(coordinate(b))
            if left == right or np.sign(left) == np.sign(right):
                continue
            t = brentq(
                lambda x: distance(coordinate(x)), a, b, xtol=np.finfo(float).tiny
            )
            candidate = coordinate(t)
            toward = b if np.sign(distance(candidate)) == np.sign(left) else a
            roots.append(_resolved_root(distance, candidate, coordinate(toward)))
    if not roots:
        return ()
    values, offsets = np.array(roots).T
    return (PanelCuts(values, offsets),)


def _face_splits(factors, target, faces, *, diagonal=None, switch=None, prior=None):
    """Pull energy or critical-frequency faces back onto continuous parent panels."""
    prior = prior or {}
    if target == "gamma" and not parents(factors.get("gamma")):
        return prior
    tied, driver = diagonal if diagonal else (None, None)
    coordinates = ("gamma", "B", driver) if target == "nu_c" else ("gamma",)
    upstream = _face_dependencies(factors, tied, driver, prior)
    relevant = set(coordinates).union(
        *(ancestors(factors, name) for name in coordinates)
    )
    if tied is not None and tied in relevant:
        relevant.update({driver, *ancestors(factors, driver)})
        relevant.remove(tied)
    kinds = {
        name: _declared_kind(factors[name]) for name in relevant if name in factors
    }
    fixed = {
        name: float(
            factors[name].rule()[0][0]
            if hasattr(factors[name], "rule")
            else factors[name]
        )
        for name, kind in kinds.items()
        if kind == "point" and not parents(factors[name])
    }

    selected = {}

    def path(axis):
        external, derived = set(), []

        def add(name):
            if name == tied:
                name = driver
            if name in fixed or name == axis or name in derived:
                return True
            if axis not in upstream.get(name, ()):
                external.add(name)
                return True
            can_be_point = kinds.get(name) == "point" or (
                name in selected and kinds.get(name) == "unknown"
            )
            if not can_be_point or not all(add(p) for p in parents(factors[name])):
                return False
            derived.append(name)
            return True

        if all(add(name) for name in coordinates):
            return external, derived
        return None

    def cuts(point, axis, derived):
        context = {**fixed, **point}
        if not derived:
            return (
                faces
                if target == "gamma"
                else inverse_frequency_faces(axis, driver, context, faces)
            )

        charts = {}

        def projected(value):
            values = {**context, axis: value}
            values.pop((axis,), None)
            for name in derived:
                if driver is not None and driver in values:
                    values[tied] = values[driver]
                    if (driver,) in values:
                        values[(tied,)] = values[(driver,)]
                law = law_at(factors[name], values, _UNITS)
                if _measure_kind(law) != "point":
                    # The inner continuous axis owns the cut on this branch.
                    return np.nan, 0.0
                node = _point_value(law, name)
                shift = point_displacement(
                    factors[name],
                    SYNCHROTRON_VARIABLES[name],
                    _UNITS,
                    values,
                    node,
                    faces if name == target else (),
                    charts,
                )
                values[name] = node
                if shift is not None:
                    values[name], values[(name,)] = sum_with_residual(node, shift)
            if target == "gamma":
                return values["gamma"], values.get(("gamma",), 0.0)
            # Every cut is a boundary of a continuous integral, even when
            # the other physical variables are specified by conditional atoms.
            values["eta"] = values[driver]
            shifts = {
                name: values.get((source,), 0.0)
                for name, source in (("B", "B"), ("gamma", "gamma"), ("eta", driver))
            }
            return projected_frequency(values, shifts)[2:]

        law = placed_law(
            law_at(factors[axis], context, _UNITS), SYNCHROTRON_VARIABLES[axis]
        )
        lo, hi = getattr(law, "quadrature_support", law.support)
        return _derived_faces(
            projected,
            (max(lo, switch) if axis == "gamma" and switch is not None else lo, hi),
            faces,
            logarithmic=axis in ("gamma", "B"),
        )

    candidates = [
        name
        for name in dict.fromkeys((*coordinates, *sorted(relevant)))
        if kinds.get(name) in ("density", "unknown")
    ]
    while candidates:
        routes = []
        for axis in candidates:
            plan = path(axis)
            if plan is not None:
                routes.append((axis, *plan))
        if not routes:
            break
        axis, deps, derived = min(
            routes, key=lambda p: (kinds[p[0]] == "unknown", len(p[2]))
        )
        selected[axis] = (
            tuple(sorted(deps)),
            lambda point, a=axis, d=derived: cuts(point, a, d),
        )
        if kinds[axis] == "density":
            break
        # A factory may return a point at some parents. On this branch a
        # preceding continuous variable must split the derived trajectory.
        previous = set(deps).union(*(upstream.get(name, ()) for name in deps))
        for name, earlier in upstream.items():
            if name == axis or axis in earlier:
                earlier.update(previous)
        candidates.remove(axis)
    combined = dict(prior)
    for axis, (deps, cut) in selected.items():
        if axis in prior:
            previous, existing = prior[axis]
            combined[axis] = (
                set(previous) | set(deps),
                lambda point, old=existing, new=cut: (*old(point), *new(point)),
            )
        else:
            combined[axis] = deps, cut
    return combined


def _branch_panels(
    pop, factors, lattice, continuum, switch, nodes, *, offset_axes=None
):
    routes = _diagonal_routes(factors) if continuum else ((None, None),)
    for diagonal, use_route in routes:
        table = lattice.cells
        faces = {
            name: np.unique(np.r_[table.lo[:, i], table.hi[:, i]])
            for i, name in enumerate(table.variables)
        }
        if continuum:
            bounds = {"gamma": (switch, np.inf), **_diagonal_bounds(factors, diagonal)}
            splits = _face_splits(
                factors, "nu_c", faces.pop("nu_c"), diagonal=diagonal, switch=switch
            )
        else:
            bounds, splits = {"gamma": (1.0, switch)}, {}
        energy_faces = np.union1d(faces.get("gamma", ()), [switch])
        splits = _face_splits(
            factors, "gamma", energy_faces, diagonal=diagonal, prior=splits
        )
        yield from panels(
            factors,
            bounds,
            blocks=pop.blocks,
            units=_UNITS,
            variables=SYNCHROTRON_VARIABLES,
            charts={"gamma": "log", "B": "log"} if continuum else {"gamma": "log"},
            breaks=faces,
            n=nodes,
            diagonal=diagonal,
            diagonal_filter=use_route,
            split_points=splits,
            offset_axes=offset_axes,
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
        for points, spin, mass, shifts in physical_panels(
            pop,
            lattice,
            branch,
            switch,
            field,
            averaged,
            nodes=nodes,
            min_width=width,
            return_offsets=True,
        ):
            cells = ownership(points, lattice.cells, mass.size, offsets=shifts)
            total += mass.sum()
            outside += mass[cells < 0].sum()
            for cell in np.unique(cells[cells >= 0]):
                keep = cells == cell
                selected = {name: values[keep] for name, values in points.items()}
                local = {name: values[keep] for name, values in shifts.items()}
                values = _basis_values(
                    index, cell, selected, int(keep.sum()), offsets=local
                )
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
        needed = {"gamma", *ancestors(pop.factors, "gamma")}
        factors = {name: law for name, law in pop.factors.items() if name in needed}
        for point, weight, shifts in panels(
            factors,
            {"gamma": (1.0, np.inf)},
            blocks=pop.blocks,
            units=_UNITS,
            variables=SYNCHROTRON_VARIABLES,
            charts={"gamma": "log"},
            breaks={"gamma": (switch,)},
            split_points=_face_splits(factors, "gamma", np.array([switch])),
            n=nodes,
            offset_axes=("gamma",),
        ):
            mass = np.asarray(weight) * float(pop.column_factor)
            lower = _below_switch_mask(point["gamma"], switch, shifts.get("gamma"))
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
            needed |= ancestors(pop.factors, name)
        factors = {key: law for key, law in pop.factors.items() if key in needed}
        for tie, use_route in _diagonal_routes(factors):
            for point, mass, shifts in panels(
                factors,
                {"gamma": (faces[0], faces[-1])},
                blocks=pop.blocks,
                units=_UNITS,
                variables=SYNCHROTRON_VARIABLES,
                charts={"gamma": "log"},
                breaks={"gamma": faces},
                split_points=_face_splits(factors, "gamma", faces, diagonal=tie),
                diagonal=tie,
                diagonal_filter=use_route,
                n=nodes,
                offset_axes=("gamma",),
            ):
                cells = ownership(point, table, len(mass), offsets=shifts)
                for cell in np.unique(cells[cells >= 0]):
                    keep = cells == cell
                    local = {name: values[keep] for name, values in shifts.items()}
                    powers = _basis_values(
                        index,
                        cell,
                        {"gamma": point["gamma"][keep]},
                        int(keep.sum()),
                        offsets=local,
                    )
                    scale = 2 * float(pop.column_factor) / lattice.weights[cell]
                    values[index.cell_rows(cell)] += (
                        scale * np.asarray(mass)[keep] @ powers
                    )
    return Moments(lattice, values)
