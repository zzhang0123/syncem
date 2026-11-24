"""Lobatto masses for a prebuilt all-cell lattice."""

from __future__ import annotations

import math
from typing import NamedTuple

import numpy as np

from ..population.blocks import NonNegativeCells
from ..values.errors import RoleError, SizeError


class PointMassMap(NamedTuple):
    """Moment rows per atom, canonical coordinates, and cell identity at faces."""

    moments: np.ndarray
    coordinates: dict[str, np.ndarray]
    cells: np.ndarray


def _box(table, cell, hints):
    lower, upper = table.lo[cell].copy(), table.hi[cell].copy()
    for i, hint in enumerate(hints):
        if hint is not None:
            lower[i] = max(lower[i], hint[0])
            upper[i] = min(upper[i], hint[1])
    if np.any(lower >= upper):
        return None
    if not np.all(np.isfinite(lower) & np.isfinite(upper)):
        raise RoleError("nonnegative point masses need finite cell bounds or support=.")
    for i, chart in enumerate(table.chart[cell]):
        if chart == "log" and lower[i] <= 0:
            raise RoleError(
                f"a nonnegative point mass on log axis {table.variables[i]} needs positive support."
            )
    return lower, upper


def point_mass_map(basis, marker, budget):
    """Tensor Lobatto masses scaled by the same cell weights as `Moments`."""
    if not isinstance(marker, NonNegativeCells):
        raise TypeError("fit_nonnegative expects a NonNegativeCells marker.")
    lattice = basis.lattice
    table = lattice.cells
    names = table.variables
    if not marker.variables and len(names) == 1:
        marker = marker.bind(names[0])
    if not names or set(marker.variables) != set(names) or lattice.representations:
        raise RoleError(
            "a prebuilt nonnegative fit needs one joint marker for all resolved axes, "
            "with Cells on each variable."
        )
    definitions = {v.name: v for v in basis.averaged.kernel.variables}
    hints = tuple(marker.support_for(definitions[name]) for name in names)
    selected = lattice.index.cells_where(**marker.selections())
    boxes = {int(cell): _box(table, int(cell), hints) for cell in selected}
    active_cells = [cell for cell, box in boxes.items() if box is not None]
    if not active_cells:
        raise RoleError("the marker support does not overlap any cell.")
    n = marker.point_masses
    per_cell = n ** len(names)
    n_points = per_cell * len(active_cells)
    # Account for the dense map, its temporary copy in a stacked NNLS and
    # the node coordinates. The caller separately budgets observation rows.
    need = 8 * n_points * (2 * lattice.index.n + 2 * len(names) + 1)
    if need >= budget:
        raise SizeError(
            f"the {n_points:,} nonnegative masses need about {need:,} bytes of map workspace. "
            f"chunk_bytes={budget:,}. Use fewer masses or a larger budget."
        )

    lobatto = -np.cos(np.linspace(0, np.pi, n)) if n > 1 else np.zeros(1)
    masses = np.empty((len(names), n_points))
    moment_map = np.zeros((lattice.index.n, n_points))
    cells = np.repeat(active_cells, per_cell)
    for offset, cell in enumerate(active_cells):
        lower, upper = boxes[cell]
        nodes, z = [], []
        for i, chart in enumerate(table.chart[cell]):
            ref, hw = table.reference[cell, i], table.half_width[cell, i]
            t = 0.5 * (lobatto + 1)
            if chart == "log":
                x = np.exp(np.log(lower[i]) * (1 - t) + np.log(upper[i]) * t)
                normalized = np.log(x / ref) / hw
            else:
                x = lower[i] * (1 - t) + upper[i] * t
                normalized = (x - ref) / hw
            nodes.append(x)
            z.append(normalized)
        start = offset * per_cell
        columns = slice(start, start + per_cell)
        rows = lattice.index.cell_rows(cell)
        physical = np.meshgrid(*nodes, indexing="ij")
        normalized = np.meshgrid(*z, indexing="ij")
        masses[:, columns] = np.asarray(physical).reshape(len(names), per_cell)
        powers = lattice.index.arrays.powers[rows]
        values = np.ones((powers.shape[0], per_cell))
        for index, axis in enumerate(normalized):
            values *= axis.ravel()[None, :] ** powers[:, index, None]
        moment_map[rows, columns] = values / lattice.weights[cell]
    edge = getattr(basis, "excluded_upper_gamma", None)
    if edge is not None:
        axis = names.index("gamma")
        at_edge = np.abs(masses[axis] - edge) <= 4 * np.spacing(edge)
        keep = ~(at_edge & (table.hi[cells, axis] == edge))
        masses[axis, at_edge] = edge
        masses, moment_map, cells = masses[:, keep], moment_map[:, keep], cells[keep]
    return PointMassMap(moment_map, dict(zip(names, masses, strict=True)), cells)


def host_point_chunk(budget, data, points):
    """A bounded host batch. The eager SciPy call solves one pixel at a time."""
    n, p = points.shape
    fit_workspace = 8 * (2 * n * p + 3 * data.n_data * p + 2 * p * p)
    if fit_workspace >= budget:
        raise SizeError(
            f"one nonnegative solve needs about {fit_workspace:,} bytes. "
            f"chunk_bytes={budget:,}. Use fewer masses or a larger budget."
        )
    per = 8 * (data.n_data * n + data.n_data**2 + n + p + data.n_data)
    size = (budget - fit_workspace) // per
    if size < 1:
        raise SizeError("chunk_bytes cannot hold one nonnegative fit pixel.")
    return min(math.prod(data.batch_shape), size)
