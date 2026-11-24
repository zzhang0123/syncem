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
from ..values.errors import RoleError
from .declaration import law_at, ordered, parents, placed_law

_INTERVAL = (-np.inf, np.inf)


@lru_cache(maxsize=32)
def _legendre(n):
    return np.polynomial.legendre.leggauss(n)


def _interval(lo, hi, n):
    x, w = _legendre(n)
    half = (hi - lo) / 2
    return lo + half * (1 + x), half * w


def _breaks(law, lo, hi, extra):
    anchors = [lo, hi, *(x for x in extra if lo < x < hi)]
    anchors.extend(x for x in getattr(law, "kinks", ()) if lo < x < hi)
    return np.unique(anchors)


def _finite_rule(law, lo, hi, chart, n, extra):
    knots = _breaks(law, lo, hi, extra)
    for low, high in zip(knots[:-1], knots[1:]):
        if hasattr(law, "panel_rule"):
            yield law.panel_rule(low, high, n=n)
        elif chart == "log" and low > 0:
            z, w = _interval(np.log(low), np.log(high), n)
            x = np.exp(z)
            yield x, x * w * law.density(x)
        else:
            x, w = _interval(low, high, n)
            yield x, w * law.density(x)


def _batched_finite_rule(law, lo, hi, chart, n, extra):
    knots = _breaks(law, lo, hi, extra)
    if isinstance(law, AgedPowerLaw) and knots[-1] == float(law.gamma_cut):
        yield law.panel_rule(knots[-2], knots[-1], n=n)
        knots = knots[:-1]
        if knots.size == 1:
            return
    log = chart == "log" and lo > 0
    edges = np.log(knots) if log else knots
    nodes, weights = _legendre(n)
    widths = np.diff(edges)[:, None] / 2
    x = (edges[:-1, None] + widths * (1 + nodes)).ravel()
    w = (widths * weights).ravel()
    if log:
        x = np.exp(x)
        w = w * x
    yield x, w * law.density(x)


def _batchable_density(law):
    if not hasattr(law, "panel_rule"):
        return True
    return type(law) is AgedPowerLaw or (
        getattr(type(law), "_shape_panel", None) is ElectronFamily._shape_panel
        and type(law).panel_rule is ElectronFamily.panel_rule
    )


def _unbounded(law, lo, hi, chart, n, extra):
    scales = [abs(x) for x in (lo, hi, *extra) if np.isfinite(x)]
    scale = max([1.0, *scales])
    if np.isfinite(lo) and np.isinf(hi) and chart == "log":
        # Integrate in log panels to a finite anchor, then compactify the tail.
        last = max(100 * scale, 1e8) if hasattr(law, "panel_rule") else 100 * scale
        n_panels = math.ceil(math.log(last / max(lo, 1e-12)) / 0.7)
        log_grid = np.geomspace(max(lo, 1e-12), last, max(1, n_panels) + 1)
        yield from _finite_rule(law, lo, last, chart, n, (*extra, *log_grid))
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
    edges = _breaks(law, lo, hi, (*extra, *profile))
    for a, b in zip(edges[:-1], edges[1:]):
        if np.isfinite(a) and np.isfinite(b):
            yield from _finite_rule(law, a, b, chart, n, ())
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


def _axis(law, interval, chart, n, extra, *, batch_panels=False):
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
        yield from rule(law, lo, hi, chart, n, extra)
        return
    yield from _unbounded(law, lo, hi, chart, n, extra)


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
):
    """Yield batches of flattened quadrature nodes and weights over a box.

    A diagonal maps one integrated axis to a density evaluated at that
    axis. A diagonal filter can restrict that route to particular parent
    nodes. Dynamic panel splits are evaluated at their declared parents.
    """
    supplied = tuple(name for block in blocks for name in block.variables)
    tied, driver = diagonal if diagonal is not None else (None, None)
    split_points = split_points or {}
    requires = {name: set(deps) for name, (deps, _) in split_points.items()}
    if tied is not None:
        requires.setdefault(tied, set()).add(driver)
    spec = ordered(factors, available=supplied, requires=requires)
    charts, breaks = charts or {}, breaks or {}
    axis_nodes = axis_nodes or {}

    def walk(level, current, mass):
        if level == len(spec):
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
        yield from walk(level + 1, {**current, name: nodes}, mass * law.density(nodes))

    def extend(level, name, law, current, mass, point=None):
        chart = charts.get(name, "affine")
        setting = box.get(name, _INTERVAL)
        count = axis_nodes.get(name, n)
        extra = breaks.get(name, ())
        if name in split_points:
            extra = (*extra, *split_points[name][1](point))
        for nodes, weights in _axis(
            law, setting, chart, count, extra, batch_panels=name in split_points
        ):
            nodes, weights = np.asarray(nodes), np.asarray(weights)
            if nodes.ndim != 1 or weights.shape != nodes.shape:
                raise RoleError(f"{name} must return one-dimensional nodes and weights")
            for i, j in _expanded(nodes, mass, batch):
                points = {key: values[i] for key, values in current.items()}
                points[name] = nodes[j]
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


def ownership(points, table, count, *, open_upper=None):
    """Index of the first containing box, or -1, for each column of points."""
    found = np.full(count, -1, dtype=int)
    for cell, (lo, hi) in enumerate(zip(table.lo, table.hi)):
        match = found < 0
        for i, name in enumerate(table.variables):
            upper = (
                points[name] < hi[i]
                if open_upper and open_upper.get(name) == hi[i]
                else points[name] <= hi[i]
            )
            match &= (lo[i] <= points[name]) & upper
        found[match] = cell
    return found


def box_owner(points, boxes, count):
    """Index of the first closed box containing each column of points, or -1.

    Complement boxes share faces, so a point on a shared face belongs to the
    first box only.
    """
    found = np.full(count, -1, dtype=int)
    for k, box in enumerate(boxes):
        match = found < 0
        for name, (lo, hi) in box.items():
            match &= (lo <= points[name]) & (points[name] <= hi)
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
