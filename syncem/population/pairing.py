"""Pair declared column measures with the lattice's unpruned moment rows.

LABEL [extension].

Moments have shape (n_row,) for one sightline and inherit the measure's
column units divided by the reference weights. Panel sums have no certified
integration error.
"""

from __future__ import annotations

import numpy as np

from ..lattice.moments import Moments
from ..lattice.representations import AzimuthFourier, AzimuthHarmonic
from ..values.errors import RoleError
from .declaration import Population, conventional_units
from .quadrature import cell_boxes, ownership, panels


def _basis_values(index, cell, points, count):
    row = index.cell_rows(cell)
    powers, degrees = index.arrays.powers[row], index.arrays.degrees[row]
    table = index.table
    coordinates = []
    for v, name in enumerate(table.variables):
        values = points[name]
        ref = table.reference[cell, v]
        z = np.log(values / ref) if table.chart[cell][v] == "log" else values - ref
        coordinates.append(z / table.half_width[cell, v])
    rows = (
        np.prod(np.asarray(coordinates).T[:, None, :] ** powers, axis=-1)
        if coordinates
        else np.ones((count, powers.shape[0]))
    )
    for v, (name, degree) in enumerate(index.legendre):
        values = np.polynomial.legendre.legvander(points[name], degree)
        rows *= values[:, degrees[:, v]]
    return rows


def _component(population, lattice, units, variables, n, open_upper):
    factors = population.factors
    azimuth, angular = next(
        (
            (name, rep)
            for name, rep in lattice.reps
            if isinstance(rep, (AzimuthHarmonic, AzimuthFourier))
        ),
        (None, None),
    )
    needed = set(lattice.cells.variables) | set(lattice.representations)
    missing = (
        needed
        - set(factors)
        - {name for block in population.blocks for name in block.variables}
    )
    if missing:
        raise RoleError(f"no specified measure for lattice variables {sorted(missing)}")
    axis_nodes = (
        {azimuth: max(n, angular.moment_nodes)}
        if isinstance(angular, AzimuthFourier)
        else None
    )
    index = lattice.index
    rows = np.zeros((len(index.blocks), index.block_size))
    for cell, box in enumerate(cell_boxes(lattice.cells)):
        charts = dict(zip(index.table.variables, index.table.chart[cell]))
        span = index.cell_rows(cell)
        for points, weights in panels(
            factors,
            box,
            blocks=population.blocks,
            charts=charts,
            units=units,
            variables=variables,
            n=n,
            axis_nodes=axis_nodes,
        ):
            keep = (
                ownership(points, lattice.cells, weights.size, open_upper=open_upper)
                == cell
            )
            if not np.any(keep):
                continue
            point = {name: values[keep] for name, values in points.items()}
            mass = weights[keep]
            weighted = _basis_values(index, cell, point, mass.size) * mass[:, None]
            rows[0, span] += weighted.sum(axis=0)
            if azimuth is not None:
                for block, mode in enumerate(angular.modes):
                    value = np.exp(1j * mode * point[azimuth]) @ weighted
                    rows[2 * block + 1, span] += value.real
                    rows[2 * block + 2, span] += value.imag
        rows[:, span] /= lattice.weights[cell]
    return population.column_factor * rows.reshape(-1)


def pair(measure, lattice, *, units=None, variables=None, n=24, open_upper=None):
    """Scaled moments of a specified population or matching supplied moments."""
    if isinstance(measure, Moments):
        lattice._own(measure)
        return measure
    if not isinstance(measure, Population):
        raise TypeError("pair needs a Population or Moments")
    parts = measure.parts
    values = [
        _component(
            p,
            lattice,
            conventional_units(p.factors) if units is None else units,
            variables,
            n,
            open_upper,
        )
        for p in parts
    ]
    return Moments(lattice, sum(values))
