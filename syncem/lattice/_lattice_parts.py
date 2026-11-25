"""Axis checks, reference weights and cell splitting for ``Lattice`` (private helpers of ``lattice.py``).

LABEL ``eq: energy cell response``. ``_default_weights`` gives the cell masses of ``gamma^-2.6`` normalised on
the lattice's ``gamma`` range (SPEC Sec. 4.5). ``_split`` cuts a product cell into equal parts in the chart.
Units: canonical units. Shapes: weights ``(n_cells,)``. Not certified: these helpers are exact arithmetic.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ..base.variables import SYNCHROTRON_PRIMARY, SYNCHROTRON_VARIABLES
from ._cell_values import (
    centers_and_widths,
    edge_tolerance,
    partition_edges,
    panel_coordinates,
    log_displacement,
)
from .cells import Cells
from .representations import AzimuthFourier, AzimuthHarmonic, Legendre
from .table import CellTable

REFERENCE_SLOPE = 2.6


def _kind(name: str, axis: Any, variables=SYNCHROTRON_VARIABLES) -> str:
    variable = variables.get(name)
    if variable is not None:
        return variable.kind
    return "positive" if isinstance(axis, Cells) and axis.chart == "log" else "real"


def _check_axis(name: str, axis: Any, variables=SYNCHROTRON_VARIABLES) -> None:
    variable = variables.get(name)
    if variable is not None:
        variable.place(axis)
    kind = _kind(name, axis, variables)
    if isinstance(axis, Cells):
        lo, hi = axis.bounds(kind)
        if kind == "angle":
            raise ValueError(
                f"{name} is an azimuth and takes no cells. Use se.AzimuthHarmonic()"
            )
        if kind == "cosine" and (np.any(lo < -1) or np.any(hi > 1)):
            raise ValueError(f"{name} is a cosine: its cells must lie in [-1, 1]")
        if kind == "positive" and (np.any(lo < 0) or np.any(axis.reference <= 0)):
            raise ValueError(
                f"{name} is positive: its cells must lie at or above zero and its reference points above zero"
            )
        if kind == "cosine" and np.any(np.abs(axis.reference) > 1):
            raise ValueError(
                f"{name} is a cosine: its reference points must lie in [-1, 1]"
            )
    elif isinstance(axis, Legendre):
        if variable is not None and kind != "cosine":
            raise ValueError(
                f"Legendre degrees resolve a cosine, and {name} is the {variable.meaning}"
            )
    elif isinstance(axis, (AzimuthHarmonic, AzimuthFourier)):
        if variable is not None and kind != "angle":
            raise ValueError(
                f"{type(axis).__name__} resolves the azimuth, and {name} is the {variable.meaning}"
            )
    else:
        raise TypeError(
            f"{name}={axis!r} is not a lattice representation (Cells, Legendre, AzimuthHarmonic, AzimuthFourier)"
        )


def _density_masses(density, bounds, chart):
    a, b = bounds
    x, w = np.polynomial.legendre.leggauss(32)
    z, jac, _ = panel_coordinates(a, b, chart, x, w)
    return np.sum(jac * np.asarray(density(z), dtype=float))


def _default_weights(table: CellTable, references=()) -> np.ndarray:
    weights = np.ones(table.n_cells)
    densities = dict(references)
    for i, name in enumerate(table.variables):
        lo, hi = table.lo[:, i], table.hi[:, i]
        if name in densities:
            if not np.all(np.isfinite(lo) & np.isfinite(hi)):
                raise ValueError(
                    f"ReferenceDensity for {name} needs finite cell bounds"
                )
            density = densities[name]
            domain = (lo.min(), hi.max())
            chart = "log" if all(row[i] == "log" for row in table.chart) else "affine"
            masses = np.array(
                [
                    _density_masses(density, (a, b), row[i])
                    for a, b, row in zip(lo, hi, table.chart)
                ]
            )
            norm = _density_masses(density, domain, chart)
            if (
                not np.isfinite(norm)
                or norm <= 0
                or np.any(~np.isfinite(masses) | (masses <= 0))
            ):
                raise ValueError(
                    f"ReferenceDensity for {name} needs finite positive cell masses"
                )
            weights *= masses / norm
        elif name == SYNCHROTRON_PRIMARY and np.all(np.isfinite(hi) & (lo > 0)):
            k = 1.0 - REFERENCE_SLOPE
            if np.any(hi - lo <= 1e-6 * lo) or lo.min() ** k == hi.max() ** k:
                span = log_displacement(hi, lo)
                interval_mass = np.exp(k * log_displacement(lo, lo.min())) * np.expm1(
                    k * span
                )
                weights *= interval_mass / np.expm1(
                    k * log_displacement(hi.max(), lo.min())
                )
            else:
                weights *= (hi**k - lo**k) / (hi.max() ** k - lo.min() ** k)
    return weights


def _grouped(n: int) -> str:
    return f"{n:,}".replace(",", " ")


def _box(t: CellTable, c: int) -> dict:
    return dict(
        lo=t.lo[c],
        hi=t.hi[c],
        reference=t.reference[c],
        half_width=t.half_width[c],
        chart=t.chart[c],
        order=t.order[c],
    )


def _split(t: CellTable, c: int, factors: dict[str, int]) -> list[dict]:
    """Sub-boxes of cell ``c`` in C order over the variables, equal in the chart, centred references."""
    pieces = []
    for i, v in enumerate(t.variables):
        k = int(factors.get(v, 1))
        if k < 1:
            raise ValueError(f"refine factor for {v} must be a positive int")
        lo, hi = t.lo[c, i], t.hi[c, i]
        if k > 1 and not (np.isfinite(lo) and np.isfinite(hi)):
            raise ValueError(f"cell {c} has no edges in {v} and cannot be refined")
        if k == 1:
            edges = np.array([lo, hi])
            refs, halves = np.array([t.reference[c, i]]), np.array([t.half_width[c, i]])
        else:
            edges = partition_edges(lo, hi, k, t.chart[c][i])
            refs, halves = centers_and_widths(edges, t.chart[c][i])
        pieces.append(
            [(edges[j], edges[j + 1], refs[j], halves[j]) for j in range(len(refs))]
        )
    out = []
    for combo in np.ndindex(*[len(p) for p in pieces]):
        parts = [pieces[i][j] for i, j in enumerate(combo)]
        out.append(
            dict(
                lo=np.array([p[0] for p in parts]),
                hi=np.array([p[1] for p in parts]),
                reference=np.array([p[2] for p in parts]),
                half_width=np.array([p[3] for p in parts]),
                chart=t.chart[c],
                order=t.order[c],
            )
        )
    return out


def table_with(t: CellTable, **changes: Any) -> CellTable:
    """A copy of a cell table with some fields replaced."""
    fields = dict(
        variables=t.variables,
        lo=t.lo,
        hi=t.hi,
        reference=t.reference,
        half_width=t.half_width,
        chart=t.chart,
        order=t.order,
        tensor=t.tensor,
        excluded=t.excluded,
    )
    fields.update(changes)
    return CellTable(**fields)


def _names(t: CellTable, names: Any, what: str) -> None:
    unknown = set(names) - set(t.variables)
    if unknown:
        raise ValueError(
            f"{what}: {sorted(unknown)} have no cells on this lattice. Celled variables: {t.variables}"
        )


def shift_table(t: CellTable, fractions: dict[str, float]) -> CellTable:
    """Reference points moved by a fraction of the half-width in the chart, per variable."""
    _names(t, fractions, "shift")
    ref = t.reference.copy()
    for name, f in fractions.items():
        i = t.variables.index(name)
        log = np.array([c[i] == "log" for c in t.chart])
        ref[:, i] = np.where(
            log,
            ref[:, i] * np.exp(f * t.half_width[:, i]),
            ref[:, i] + f * t.half_width[:, i],
        )
    slack = edge_tolerance(t.lo, t.hi)
    if np.any(ref < t.lo - slack) or np.any(ref > t.hi + slack):
        raise ValueError("the shift moves reference points outside their cells")
    return table_with(t, reference=ref, tensor=None)


def refine_table(
    t: CellTable, selected: set[int], factors: dict[str, int]
) -> CellTable:
    """The selected product cells split into equal parts in the chart, the others kept."""
    _names(t, factors, "refine")
    for v, k in factors.items():
        if isinstance(k, bool) or not isinstance(k, (int, np.integer)) or k < 1:
            raise ValueError(f"refine factor for {v} must be a positive int, not {k!r}")
    rows = {k: [] for k in ("lo", "hi", "reference", "half_width", "chart", "order")}
    for c in range(t.n_cells):
        for box in _split(t, c, factors) if c in selected else [_box(t, c)]:
            for k, v in box.items():
                rows[k].append(v)
    arrays = {k: np.array(rows[k]) for k in ("lo", "hi", "reference", "half_width")}
    return table_with(
        t, **arrays, chart=tuple(rows["chart"]), order=tuple(rows["order"]), tensor=None
    )
