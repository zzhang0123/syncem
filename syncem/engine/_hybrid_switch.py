"""Direct comparisons on shared faces and extremal common-valid cells."""

from __future__ import annotations

from itertools import combinations
import warnings

import equinox as eqx
import numpy as np

from ..records.ledger import ErrorTerm
from ..values.errors import RegimeError
from .assemble import stokes_axes

_HARD_SWITCH_LIMIT = 1e-1
_REFERENCE = "the larger absolute intensity of the two child kernels on each probe"


def boxes(table):
    return tuple(
        {n: (float(a), float(b)) for n, a, b in zip(table.variables, lo, hi)}
        for lo, hi in zip(table.lo, table.hi)
    )


def _faces(table, assignments):
    lo, hi = np.asarray(table.lo), np.asarray(table.hi)
    for dimension in range(lo.shape[1]):
        lower, upper = {}, {}
        for cell, (a, b) in enumerate(zip(lo[:, dimension], hi[:, dimension])):
            lower.setdefault(a, []).append(cell)
            upper.setdefault(b, []).append(cell)
        rest = np.arange(lo.shape[1]) != dimension
        for edge in sorted(lower.keys() & upper.keys()):
            for left in upper[edge]:
                for right in lower[edge]:
                    if assignments[left] == assignments[right]:
                        continue
                    a = np.maximum(lo[left], lo[right])
                    b = np.minimum(hi[left], hi[right])
                    if np.all(a[rest] < b[rest]):
                        point = (a + b) / 2
                        point[dimension] = edge
                        yield tuple(
                            sorted((assignments[left], assignments[right]))
                        ), tuple(point)


def _regime(hybrid, label, box, channels):
    try:
        return hybrid.child_regime(label, box, channels).ok
    except KeyError:
        return False


def _comparisons(hybrid, averaged, lattice, channels, cell_boxes):
    table = lattice.cells
    names = table.variables
    cases = dict.fromkeys(_faces(table, hybrid.assignment))
    anchor = hybrid.switch[0] if hybrid.switch is not None else hybrid.primary
    direction = names.index(anchor) if anchor in names else 0
    for pair in combinations(range(len(hybrid.kernels)), 2):
        if not all(i in hybrid.assignment for i in pair):
            continue
        valid = [
            cell
            for cell, box in enumerate(cell_boxes)
            if all(_regime(hybrid, i, box, channels) for i in pair)
        ]
        if valid:
            ends = (
                min(valid, key=lambda c: table.lo[c, direction]),
                max(valid, key=lambda c: table.hi[c, direction]),
            )
            for cell, edge in zip(ends, (table.lo, table.hi)):
                point = np.asarray(table.reference[cell]).copy()
                point[direction] = edge[cell, direction]
                cases[(pair, tuple(point))] = None
    fixed = {n: (v, v) for n, v in averaged.fixed}
    for pair, point in cases:
        physical = dict(zip(names, point))
        box = {**fixed, **{n: (x, x) for n, x in physical.items()}}
        if not all(_regime(hybrid, i, box, channels) for i in pair):
            raise RegimeError(
                f"hybrid kernels {pair} do not both admit their comparison point {physical}"
            )
        yield pair, physical


def _ratios(a, b, blocks):
    intensity = np.maximum(np.abs(a.I), np.abs(b.I))
    reference = intensity[(slice(None), *(0,) * (intensity.ndim - 1))]
    changes = {"I": a.I - b.I}
    if "P" in blocks:
        difference = a.P[0] - b.P[0]
        changes["Q"], changes["U"] = difference.real, difference.imag
    if "V" in blocks:
        changes["V"] = a.V - b.V
    differences = np.column_stack(
        [
            np.max(np.abs(changes[axis]).reshape(reference.size, -1), axis=-1)
            for axis in stokes_axes(blocks)
        ]
    )
    reference = np.asarray(reference)[:, None]
    result = np.divide(
        differences, reference, out=np.zeros_like(differences), where=reference > 0
    )
    return np.where((reference == 0) & (differences > 0), np.inf, result)


def switch_measurement(averaged, lattice, channels, blocks):
    """Maximum measured relative switch difference at the declared probes."""
    hybrid = averaged.kernel
    children = tuple(
        eqx.tree_at(lambda a: a.kernel, averaged, k) for k in hybrid.kernels
    )
    fixed = {n: (v, v) for n, v in averaged.fixed}
    cell_boxes = tuple({**fixed, **box} for box in boxes(lattice.cells))
    hybrid.check_regime(cell_boxes, channels)
    values = np.zeros((channels.n_ch, len(stokes_axes(blocks))))
    tested = 0
    for (i, j), point in _comparisons(hybrid, averaged, lattice, channels, cell_boxes):
        a, b = children[i](channels, point), children[j](channels, point)
        ratios = _ratios(a, b, blocks)
        if not np.all(np.isfinite(ratios)):
            raise RegimeError(
                "hybrid switch comparison is nonfinite or has zero reference intensity"
            )
        values = np.maximum(values, ratios)
        tested += 1
    if not tested:
        return (
            ErrorTerm.unbounded(
                "there is no comparison between two active hybrid kernels in their common regime",
                name="kernel_switch",
            ),
            tested,
        )
    return switch_term(values, tested, hybrid.switch_tol), tested


def switch_term(values, tested, tolerance):
    largest = float(np.max(values))
    if largest > _HARD_SWITCH_LIMIT:
        raise RegimeError(
            f"hybrid kernel switch changes Stokes by {largest:.3g} of I, above {_HARD_SWITCH_LIMIT:g}"
        )
    if largest > tolerance:
        warnings.warn(
            f"hybrid kernel switch changes Stokes by {largest:.3g} of I, above switch_tol={tolerance:g}",
            RuntimeWarning,
            stacklevel=3,
        )
    return ErrorTerm(
        values,
        "measured",
        f"maximum over {tested} common-valid comparison points including shared faces and extremes",
        name="kernel_switch",
        relative_to=_REFERENCE,
    )
