"""Panels for specified population factors on product boxes.

Weights are column weights for an electron family and probabilities for
the other factors. A continuous axis is integrated on its own support
intersected with the box. Point masses on shared faces belong to the
first cell that contains them.

LABEL [extension].

Nodes have shape (n_node,) per variable in canonical units. Weights
represent masses or columns as declared by the measure. Finite panels have
no certified integration error.
"""

from __future__ import annotations

import math
from functools import lru_cache

import numpy as np

from ..distributions.aged import AgedPowerLaw
from ..distributions.electrons import ElectronFamily
from ..distributions.user import UserFamily
from ..lattice._cell_values import panel_coordinates
from ..values.errors import RoleError
from ._diagonal import (
    fixed_angular_knots,
    host_diagonal_knots,
    knot_parents,
    knot_paths,
    on_panel_nodes,
)
from ._node_offsets import point_displacement, sum_with_residual
from ._panel_cuts import panel_knots, rounded_cuts
from .declaration import ancestors, law_at, ordered, parents, placed_law

_INTERVAL = (-np.inf, np.inf)


@lru_cache(maxsize=32)
def _legendre(n):
    return np.polynomial.legendre.leggauss(n)


def _interval(lo, hi, n):
    x, w = _legendre(n)
    half = (hi - lo) / 2
    return lo + half * (1 + x), half * w


def _density_panels(law, knots, chart, n, offsets, shifts=None):
    """Integrate a density in local panel coordinates and retain node rounding."""
    lower, upper = knots[:-1, None], knots[1:, None]
    nodes, weights = _legendre(n)
    x, w, residual = panel_coordinates(
        lower,
        upper,
        chart,
        nodes,
        weights,
        offsets=offsets,
        edge_offsets=None if shifts is None else (shifts[:-1, None], shifts[1:, None]),
    )
    x = x.ravel()
    # The density at an excluded upper support endpoint can be zero. A Gauss
    # node which rounds to that endpoint samples the left-hand limit.
    evaluate = np.minimum(x.reshape(lower.size, n), np.nextafter(upper, lower))
    if shifts is not None:
        # Two distinct cuts can round to the upper endpoint of the law.
        a, b = law.support
        evaluate = np.clip(evaluate, a, np.nextafter(float(b), float(a)))
    mass = w.ravel() * law.density(evaluate.ravel())
    return (x, mass, residual.ravel()) if offsets else (x, mass)


def _custom_panel(law, low, high, n, offsets):
    nodes, weights = law.panel_rule(low, high, n=n)
    if not offsets:
        return nodes, weights
    # This is a continuous integral. Even when a custom rule returns an
    # endpoint, its weight represents this panel's interior.
    nodes = np.asarray(nodes)
    step = np.nextafter(0.0, 1.0)
    residual = np.where(nodes == low, step, np.where(nodes == high, -step, 0.0))
    return nodes, weights, residual


def _finite_rule(law, lo, hi, chart, n, extra, *, offsets=False):
    knots, residual = panel_knots(law, lo, hi, extra)
    for i, (low, high) in enumerate(zip(knots[:-1], knots[1:])):
        edges = knots[i : i + 2]
        shifts = None if residual is None else residual[i : i + 2]
        # Evaluate a smooth law in its own log chart when its absolute
        # log endpoints lose precision. Preserve specialised cutoff rules.
        narrow = low > 0 and high - low <= 1e-6 * low
        smooth_log = (isinstance(law, UserFamily) and law.cutoff is None) or (
            isinstance(law, ElectronFamily)
            and type(law) is not AgedPowerLaw
            and _batchable_density(law)
        )
        if narrow and smooth_log:
            yield _density_panels(law, edges, "log", n, offsets, shifts)
        elif hasattr(law, "panel_rule"):
            yield from _custom_interval(law, edges, shifts, chart, n, offsets)
        else:
            yield _density_panels(law, edges, chart, n, offsets, shifts)


def _custom_interval(law, edges, shifts, chart, n, offsets):
    """Preserve the custom rule on the representable interior of a cut panel."""
    low, high = edges
    if shifts is None or not np.any(shifts):
        yield _custom_panel(law, low, high, n, offsets)
        return
    a = np.nextafter(low, np.inf) if shifts[0] > 0 else low
    b = np.nextafter(high, -np.inf) if shifts[1] < 0 else high
    if a >= b:
        yield _density_panels(law, edges, chart, n, offsets, shifts)
        return
    if shifts[0]:
        yield _density_panels(
            law, np.array([low, a]), chart, n, offsets, np.array([shifts[0], 0.0])
        )
    yield _custom_panel(law, a, b, n, offsets)
    if shifts[1]:
        yield _density_panels(
            law, np.array([b, high]), chart, n, offsets, np.array([0.0, shifts[1]])
        )


def _batched_finite_rule(law, lo, hi, chart, n, extra, *, offsets=False):
    knots, shifts = panel_knots(law, lo, hi, extra)
    if isinstance(law, AgedPowerLaw) and knots[-1] == float(law.gamma_cut):
        yield from _custom_interval(
            law, knots[-2:], None if shifts is None else shifts[-2:], chart, n, offsets
        )
        knots = knots[:-1]
        if shifts is not None:
            shifts = shifts[:-1]
        if knots.size == 1:
            return
    yield _density_panels(law, knots, chart, n, offsets, shifts)


def _batchable_density(law):
    if not hasattr(law, "panel_rule"):
        return True
    return type(law) is AgedPowerLaw or (
        getattr(type(law), "_shape_panel", None) is ElectronFamily._shape_panel
        and type(law).panel_rule is ElectronFamily.panel_rule
    )


def _unbounded(law, lo, hi, chart, n, extra, *, offsets):
    rounded = tuple(rounded_cuts(extra))
    scales = [abs(x) for x in (lo, hi, *rounded) if np.isfinite(x)]
    scale = max([1.0, *scales])
    if np.isfinite(lo) and np.isinf(hi) and chart == "log":
        # Integrate in log panels to a finite anchor, then compactify the tail.
        last = max(100 * scale, 1e8) if hasattr(law, "panel_rule") else 100 * scale
        n_panels = math.ceil(math.log(last / max(lo, 1e-12)) / 0.7)
        log_grid = np.geomspace(max(lo, 1e-12), last, max(1, n_panels) + 1)
        yield from _finite_rule(
            law, lo, last, chart, n, (*extra, *log_grid), offsets=offsets
        )
        t, w = _interval(0.0, 1.0, max(96, n))
        x = last / t
        yield x, w * last / t**2 * law.density(x)
        return
    centre = getattr(law, "mean", 0.0)
    width = getattr(law, "sigma", None)
    if width is not None and float(width) > 0:
        scale = float(width)
        profile = float(centre) + scale * np.arange(-8, 9, 2)
    else:
        profile = (0.0,)
    edges, _ = panel_knots(law, lo, hi, (*rounded, *profile))
    for a, b in zip(edges[:-1], edges[1:]):
        if np.isfinite(a) and np.isfinite(b):
            yield from _finite_rule(law, a, b, chart, n, extra, offsets=offsets)
        else:
            lower = np.isneginf(a)
            t, w = _interval(0.0, 1.0, max(96, n))
            if np.isinf(a) and np.isinf(b):
                ordinate = np.pi * (t - 0.5)
                x = scale * np.tan(ordinate)
                yield x, w * np.pi * scale / np.cos(ordinate) ** 2 * law.density(x)
            else:
                x = (b if lower else a) + (-1 if lower else 1) * scale * t / (1 - t)
                yield x, w * scale / (1 - t) ** 2 * law.density(x)


def _axis(law, interval, chart, n, extra, *, batch_panels=False, offsets=False):
    lo, hi = interval
    if not hasattr(law, "rule"):
        try:
            point = float(law)
        except (ValueError, TypeError) as err:
            raise RoleError(
                f"{law!r} supplies neither a point value nor a quadrature"
            ) from err
        if lo <= point <= hi:
            yield np.asarray([point]), np.ones(1)
        return
    if getattr(law, "is_point", False) or not hasattr(law, "density"):
        points, weights = map(np.asarray, law.rule())
        match = (lo <= points) & (points <= hi)
        if np.any(match):
            yield points[match], weights[match]
        return
    low, high = getattr(law, "quadrature_support", law.support)
    lo, hi = max(float(low), lo), min(float(high), hi)
    if lo >= hi:
        return
    if np.isfinite(lo) and np.isfinite(hi):
        rule = (
            _batched_finite_rule
            if batch_panels and _batchable_density(law)
            else _finite_rule
        )
        yield from rule(law, lo, hi, chart, n, extra, offsets=offsets)
        return
    yield from _unbounded(law, lo, hi, chart, n, extra, offsets=offsets)


def _expanded(nodes, weights, size):
    count, n = weights.shape[0], nodes.shape[0]
    for start in range(0, count * n, size):
        flat = np.arange(start, min(start + size, count * n))
        yield flat // n, flat % n


def panels(
    factors,
    box,
    *,
    blocks=(),
    units=None,
    variables=None,
    charts=None,
    breaks=None,
    n=24,
    axis_nodes=None,
    batch=4096,
    diagonal=None,
    diagonal_filter=None,
    split_points=None,
    aliases=None,
    diagonal_paths=None,
    fixed=None,
    offset_axes=None,
):
    """Yield batches of flattened quadrature nodes and weights over a box.

    A diagonal maps one integrated axis to a density evaluated at that
    axis. A diagonal filter can restrict that route to particular parent
    nodes. Dynamic panel splits are evaluated at their declared parents.
    Supplying offset_axes returns a third mapping of the sub-ulp displacement
    of continuous nodes on the requested axes for physical face ownership.
    """
    supplied = tuple(name for block in blocks for name in block.variables)
    tied, driver = diagonal if diagonal is not None else (None, None)
    split_points = dict(split_points or {})
    charts, breaks = charts or {}, breaks or {}
    axis_nodes = axis_nodes or {}
    aliases = aliases or {}
    fixed = fixed or {}
    paths = dict(diagonal_paths or {})
    return_offsets = offset_axes is not None
    offset_axes = frozenset(offset_axes or ())
    # A conditional point inherits continuous displacements through its parents.
    for axis in tuple(offset_axes):
        offset_axes |= ancestors(factors, axis)
    ties = {**aliases, **({tied: driver} if diagonal is not None else {})}
    for source, target in ties.items():
        if source in offset_axes:
            offset_axes |= {target, *ancestors(factors, target)}

    point_charts = {}

    def axis_rule(name, law, point):
        if name in offset_axes and point and getattr(law, "is_point", False):
            nodes, weights = map(np.asarray, law.rule())
            variable = variables.get(name) if variables else None
            lo, hi = box.get(name, _INTERVAL)
            faces = (*breaks.get(name, ()), lo, hi)
            shift = point_displacement(
                factors[name], variable, units, point, nodes[0], faces, point_charts
            )
            if shift is not None:
                nodes, shifts = sum_with_residual(nodes, shift)
                keep = _in_interval(nodes, lo, hi, shifts)
                return (
                    ((nodes[keep], weights[keep], shifts[keep]),)
                    if np.any(keep)
                    else ()
                )
        extra = breaks.get(name, ())
        if (
            name in split_points
            and hasattr(law, "density")
            and not getattr(law, "is_point", False)
        ):
            extra = (*extra, *split_points[name][1](point))
        return _axis(
            law,
            box.get(name, _INTERVAL),
            charts.get(name, "affine"),
            axis_nodes.get(name, n),
            extra,
            batch_panels=name in split_points,
            offsets=name in offset_axes,
        )

    def rule_nodes(name, law, point):
        if variables and name in variables:
            law = placed_law(law, variables[name])
        rows = [np.asarray(rule[0]) for rule in axis_rule(name, law, point)]
        return np.concatenate(rows) if rows else np.empty(0)

    if tied is not None:
        folded = factors[tied]

        knots = (
            fixed_angular_knots(folded, units)
            if parents(folded)
            else (*getattr(folded, "support", ()), *getattr(folded, "kinks", ()))
        )
        if knots is not None:
            breaks = {**breaks, driver: (*breaks.get(driver, ()), *knots)}
        elif driver in factors:
            paths[driver] = knot_paths(factors, tied, driver, rule_nodes, host=True)

    for axis, geom in paths.items():
        geometry = on_panel_nodes(geom, factors, rule_nodes)
        deps = set(knot_parents(geometry)) - {axis} - fixed.keys()
        prior_deps, prior_knots = split_points.get(axis, ((), None))

        def folded_knots(point, axis=axis, geometry=geometry, existing=prior_knots):
            lower, upper = box.get(axis, (-1.0, 1.0))
            interval = max(lower, -1.0), min(upper, 1.0)
            roots = host_diagonal_knots(
                geometry, axis, {**fixed, **point}, units, interval
            )
            return roots if existing is None else (*existing(point), *roots)

        split_points[axis] = (set(prior_deps) | deps, folded_knots)
    requires = {name: set(deps) for name, (deps, _) in split_points.items()}
    if tied is not None:
        requires.setdefault(tied, set()).add(driver)
    spec = ordered(factors, available=supplied, requires=requires, aliases=aliases)

    def walk(level, current, mass):
        for source, target in aliases.items():
            if target in current and source not in current:
                current = {**current, source: current[target]}
                if source in offset_axes and (target,) in current:
                    current[(source,)] = current[(target,)]
        if level == len(spec):
            if return_offsets:
                # Tuple keys carry sub-ulp displacements alongside their axes
                # while the product rule expands and slices parent nodes.
                shifts = {k[0]: v for k, v in current.items() if isinstance(k, tuple)}
                yield {
                    k: v for k, v in current.items() if isinstance(k, str)
                }, mass, shifts
            else:
                yield current, mass
            return
        name, declaration = spec[level]
        dependencies = parents(declaration)
        if name == tied and not dependencies:
            law = law_at(declaration, {}, units)
            yield from collapse(level, name, law, current, mass)
        elif dependencies or name in split_points:
            for at in range(mass.size):
                point = {key: values[at] for key, values in current.items()}
                law = law_at(declaration, point, units)
                if variables and name in variables:
                    law = placed_law(law, variables[name])
                now = {key: values[at : at + 1] for key, values in current.items()}
                weight = mass[at : at + 1]
                if name == tied:
                    yield from collapse(level, name, law, now, weight)
                else:
                    yield from extend(level, name, law, now, weight, point)
        else:
            yield from extend(
                level, name, law_at(declaration, {}, units), current, mass
            )

    def collapse(level, name, law, current, mass):
        if diagonal_filter is not None:
            keep = np.broadcast_to(diagonal_filter(law, current), mass.shape)
            if not np.any(keep):
                return
            current = {key: values[keep] for key, values in current.items()}
            mass = mass[keep]
        if getattr(law, "is_point", False) or not hasattr(law, "density"):
            raise RoleError(
                f"the beaming diagonal needs a continuous density for {name}"
            )
        nodes = current[driver]
        if name in offset_axes and (driver,) in current:
            current = {**current, (name,): current[(driver,)]}
        yield from walk(level + 1, {**current, name: nodes}, mass * law.density(nodes))

    def extend(level, name, law, current, mass, point=None):
        for rule in axis_rule(name, law, point):
            nodes, weights = map(np.asarray, rule[:2])
            if nodes.ndim != 1 or weights.shape != nodes.shape:
                raise RoleError(f"{name} must return one-dimensional nodes and weights")
            for i, j in _expanded(nodes, mass, batch):
                points = {key: values[i] for key, values in current.items()}
                points[name] = nodes[j]
                if len(rule) == 3:
                    points[(name,)] = rule[2][j]
                yield from walk(level + 1, points, mass[i] * weights[j])

    def seed(level, current, mass):
        if level == len(blocks):
            yield from walk(0, current, mass)
            return
        nodes, weights = blocks[level].mass_rule()
        point = {name: np.asarray(x) for name, x in nodes.items()}
        weights = np.asarray(weights)
        use = weights != 0
        for name, x in point.items():
            lo, hi = box.get(name, _INTERVAL)
            use &= (lo <= x) & (x <= hi)
        selected = np.flatnonzero(use)
        for i, j in _expanded(selected, mass, batch):
            take = selected[j]
            again = {name: x[i] for name, x in current.items()}
            again.update((name, x[take]) for name, x in point.items())
            yield from seed(level + 1, again, mass[i] * weights[take])

    yield from seed(0, {}, np.ones(1))


def cell_boxes(table):
    """The lattice boxes as per-variable interval mappings."""
    return tuple(
        dict(zip(table.variables, zip(lo, hi))) for lo, hi in zip(table.lo, table.hi)
    )


def _in_interval(values, lo, hi, shift, *, upper_open=False):
    if shift is None:
        return (lo <= values) & ((values < hi) if upper_open else (values <= hi))
    lower = (lo < values) | ((lo == values) & (shift >= 0))
    upper = (values < hi) | (
        (values == hi) & ((shift < 0) if upper_open else (shift <= 0))
    )
    return lower & upper


def ownership(points, table, count, *, open_upper=None, offsets=None):
    """Index the first containing box, retaining the sides of rounded nodes."""
    offsets = offsets or {}
    found = np.full(count, -1, dtype=int)
    for cell, (lo, hi) in enumerate(zip(table.lo, table.hi)):
        match = found < 0
        for i, name in enumerate(table.variables):
            match &= _in_interval(
                points[name],
                lo[i],
                hi[i],
                offsets.get(name),
                upper_open=open_upper and open_upper.get(name) == hi[i],
            )
        found[match] = cell
    return found


def box_owner(points, boxes, count, *, offsets=None):
    """Index the first closed complement box containing each node."""
    offsets = offsets or {}
    found = np.full(count, -1, dtype=int)
    for k, box in enumerate(boxes):
        match = found < 0
        for name, (lo, hi) in box.items():
            match &= _in_interval(points[name], lo, hi, offsets.get(name))
        found[match] = k
    return found


def _subtract(box, cutter):
    overlap = {
        name: (max(lo, cutter[name][0]), min(hi, cutter[name][1]))
        for name, (lo, hi) in box.items()
    }
    if any(lo >= hi for lo, hi in overlap.values()):
        yield box
        return
    core = box.copy()
    for name, (lower, upper) in overlap.items():
        lo, hi = core[name]
        if lo < lower:
            yield {**core, name: (lo, lower)}
        if upper < hi:
            yield {**core, name: (upper, hi)}
        core[name] = lower, upper


def complement(table):
    """Disjoint boxes covering the complement of a finite union up to its faces."""
    keep = ({name: _INTERVAL for name in table.variables},)
    for cell in cell_boxes(table):
        keep = tuple(piece for box in keep for piece in _subtract(box, cell))
    return keep
