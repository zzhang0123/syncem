"""A shared continuum table for measuring the specified energy-field edge."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from ..base.capabilities import FieldScalingRelation
from ..base.continuum import ContinuumKernel
from ..base.kernel import Modes
from ..engine.direct import continuum_tail_threshold, node_modes, sum_modes
from ..engine.provider_table import (
    TABLE_NODES,
    CellKernelTable,
    TableRefused,
    _cell_axes,
    _fit,
    check_tables,
    scaling_reference,
)
from ..lattice.cells import Cells
from ..lattice.lattice import Lattice
from ..population import Population
from ..population._marginal import _parent_rule, marginalize
from ..population.declaration import law_at, parents, placed_law
from ..population.quadrature import panels
from ..values.errors import RoleError, SizeError
from ._direct import _stokes_array
from ._setup import configure

_LOG_WIDTH = 0.4
_PANEL_BATCH = 4096


@dataclass(frozen=True)
class _Measure:
    law: object
    fields: np.ndarray
    weights: np.ndarray
    column: float
    support: tuple[float, float]


def _measure(law, variable, fields, weights, column, gamma_min):
    law = placed_law(law, variable)
    if not hasattr(law, "support"):
        raise TableRefused("the energy law needs declared support")
    support = tuple(float(x) for x in law.support)
    if not all(map(math.isfinite, support)) or support[0] < gamma_min:
        raise TableRefused(
            "the energy law needs finite support in the continuum regime"
        )
    return _Measure(law, fields, weights, float(column), support)


def _measures(basis, original, relation, definitions):
    field, energy = relation.variable, relation.energy
    kernel = basis.averaged.kernel
    components = original.parts
    if len(components) != 1 or components[0].blocks:
        raise TableRefused("the scaling check expects one population of concrete laws")
    original = components[0]
    law = original.factors.get(energy)
    if parents(law) != (field,):
        raise TableRefused("the energy law must depend on the field alone")
    if any(
        {field, energy}.intersection(parents(item))
        for name, item in original.factors.items()
        if name != energy
    ) or parents(original.factors.get(kernel.azimuth)):
        raise TableRefused("the scaling check needs independent kernel statistics")

    fields, weights = (
        np.asarray(x) for x in zip(*_parent_rule(original, definitions[field]))
    )
    keep = fields > 0
    if not np.any(keep):
        raise TableRefused("the scaling check needs a positive emitting field")
    fields, weights = fields[keep], weights[keep]
    units = {field: definitions[field].unit}
    direct = tuple(
        _measure(
            law_at(law, {field: b}, units),
            definitions[energy],
            np.asarray([b]),
            np.asarray([w]),
            original.column_factor,
            kernel.gamma_min,
        )
        for b, w in zip(fields, weights, strict=True)
    )
    marginal, _ = marginalize(original, kernel, (field,))
    independent = tuple(
        _measure(
            part.factors[energy],
            definitions[energy],
            fields,
            weights,
            part.column_factor,
            kernel.gamma_min,
        )
        for part in marginal.parts
    )
    return original, fields.min(), direct, independent


def _average(basis, original, relation, definitions, field):
    energy = relation.energy
    # Remove the unknown edge only when compiling the held kernel statistics.
    # Both input measures retain their actual density and normalization.
    population = original.replace(
        **{
            energy: basis.averaged.kernel.gamma_min,
            relation.variable: (field, definitions[relation.variable].unit),
        }
    )
    lower = basis.averaged.kernel.gamma_min
    layout = Lattice(**{energy: Cells.log(lower, 2 * lower, 1, order=0)})
    setup = configure(
        population,
        basis.averaged.kernel,
        basis.channels,
        layout,
        basis.forward_phase,
        direct=True,
    )
    if setup.azimuth is None:
        raise TableRefused("the scaling check needs a scalar azimuth factor")
    return setup.averaged, setup.azimuth


def _bounds(measures, b0):
    lo = min(row.support[0] * np.sqrt(row.fields.min() / b0) for row in measures)
    hi = max(row.support[1] * np.sqrt(row.fields.max() / b0) for row in measures)
    return np.nextafter(lo, 0), np.nextafter(max(hi, lo * (1 + 1e-6)), np.inf)


def _tabulate(averaged, channels, energy, bounds, b0, budget):
    lo, hi = bounds
    tail = continuum_tail_threshold(averaged.kernel, channels, b0)
    if tail is not None and hi > tail:
        raise TableRefused("the scaled support reaches the continuum tail")
    count = max(1, math.ceil((math.log(hi) - math.log(lo)) / _LOG_WIDTH))
    point = {**dict(averaged.fixed), energy: lo}
    product = 1
    for name, rule in averaged.averaged:
        nodes, _ = rule(point)
        product *= len(nodes)
        point[name] = nodes[len(nodes) // 2]
    estimate = (count * TABLE_NODES + 3 * min(count, 3)) * product
    if estimate * channels.n_ch * channels.n_nu * 1e-8 > budget:
        raise TableRefused("the fixed-field table exceeds the check budget")
    lattice = Lattice(**{energy: Cells.log(lo, hi, count, order=2)})
    axes = tuple(_cell_axes(lattice, cell) for cell in range(count))
    coordinates = np.concatenate([axis[0].nodes() for axis in axes])
    values = node_modes(averaged, channels, {energy: coordinates}, apply_azimuth=False)
    channels_key = channels.row_keys()
    tables = tuple(
        CellKernelTable(
            cell,
            axis,
            Modes(
                *(
                    _fit(a[cell * TABLE_NODES : (cell + 1) * TABLE_NODES], 1)
                    for a in values
                )
            ),
            averaged,
            channels_key,
        )
        for cell, axis in enumerate(axes)
    )
    probe = tuple(tables[i] for i in sorted({0, count // 2, count - 1}))
    if not check_tables(averaged, channels, lattice, probe).passed:
        raise TableRefused(
            "the fixed-field table did not match the jet on its probe cells"
        )
    return scaling_reference(averaged, tables)


def _interpolation_cost(measures, channel_count, breaks):
    # Count the Cartesian field draws for each energy panel. Interpolation
    # reads the same number of polynomial coefficients on each channel.
    pairs = 0
    for row in measures:
        lo, hi = row.support
        knots = (*breaks, *getattr(row.law, "kinks", ()))
        panel_count = 1 + sum(lo < x < hi for x in knots)
        nodes = 1 if getattr(row.law, "is_point", False) else 24 * panel_count
        pairs += len(row.fields) * nodes
    return pairs * channel_count * TABLE_NODES * 1e-8


def _integral(reference, measures, breaks):
    field, energy = reference.relation.variable, reference.relation.energy
    total = None
    for row in measures:
        if row.column == 0:
            continue
        pairs = tuple(
            (point[energy], mass)
            for point, mass in panels(
                {energy: row.law}, {}, charts={energy: "log"}, breaks=breaks
            )
        )
        if not pairs:
            continue
        energies, mass = (np.concatenate(a) for a in zip(*pairs, strict=True))
        count = len(energies) * len(row.fields)
        for start in range(0, count, _PANEL_BATCH):
            flat = np.arange(start, min(count, start + _PANEL_BATCH))
            i, j = np.divmod(flat, len(energies))
            values = reference.evaluate({energy: energies[j], field: row.fields[i]})
            term = sum_modes(values, row.column * row.weights[i] * mass[j])
            total = (
                term
                if total is None
                else Modes(*(a + b for a, b in zip(total, term, strict=True)))
            )
    return total


def scaling_difference(basis, original, name, budget):
    """Return the direct marginal-minus-joint spectrum from a checked table."""
    kernel = basis.averaged.kernel
    if (
        type(kernel) is not ContinuumKernel
        or budget <= 0
        or not isinstance(original, Population)
    ):
        return None
    (relation,) = (
        c for c in kernel.capabilities if isinstance(c, FieldScalingRelation)
    )
    field, energy = relation.variable, relation.energy
    if name != f"independence:{energy}|{field}":
        return None
    definitions = {var.name: var for var in kernel.variables}
    channels = basis.channels
    try:
        pop, b0, joint, marginal = _measures(basis, original, relation, definitions)
        averaged, spin = _average(basis, pop, relation, definitions, b0)
        cell = basis.lattice.cells
        breaks = {}
        if energy in cell.variables:
            i = cell.variables.index(energy)
            breaks[energy] = np.unique(np.r_[cell.lo[:, i], cell.hi[:, i]])
        measures = (*joint, *marginal)
        budget -= _interpolation_cost(measures, channels.n_ch, breaks.get(energy, ()))
        if budget <= 0:
            return None
        reference = _tabulate(
            averaged, channels, energy, _bounds(measures, b0), b0, budget
        )
        direct_joint = _integral(reference, joint, breaks)
        direct_marginal = _integral(reference, marginal, breaks)
    except (RoleError, SizeError, TableRefused):
        return None
    if direct_joint is None or direct_marginal is None:
        return None
    diff = Modes(*(a - b for a, b in zip(direct_marginal, direct_joint, strict=True)))
    return _stokes_array(basis, Modes(diff.I, diff.V, spin * diff.P))
