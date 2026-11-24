"""Direct Stokes integration on independent support and cell panels."""

from __future__ import annotations


import jax.numpy as jnp
import numpy as np

from ..base.continuum import ContinuumKernel
from ..base.hybrid import HybridKernel
from ..engine._fourier_reference import projected_integral
from ..engine.direct import continuum_tail_threshold, integrate
from ..engine.provider_fourier import fourier_axis
from ..planning.defaults import beaming_knots
from ..population import Population
from ..population.declaration import parents
from ..population.quadrature import (
    box_owner,
    cell_boxes,
    complement,
    ownership,
    panels,
)
from ..records.ledger import ErrorTerm, Ledger
from ..values.spectrum import Spectrum
from ..values.errors import RoleError
from ._setup import (
    configure,
    support_for,
)


def _charts(setup, lattice):
    charts = {
        name: lattice.cells.chart[0][i]
        for i, name in enumerate(lattice.cells.variables)
    }
    primary = setup.averaged.kernel.primary
    if primary in setup.resolved:
        charts.setdefault(primary, "log")
    return charts


def _face_breaks(setup, lattice, channels):
    table = lattice.cells
    breaks = {
        name: tuple(np.unique(np.r_[table.lo[:, i], table.hi[:, i]]))
        for i, name in enumerate(table.variables)
    }
    for source, target in setup.averaged.ties:
        knots = beaming_knots(setup.population.factors.get(source))
        breaks[target] = (*breaks.get(target, ()), *knots)
    threshold = continuum_tail_threshold(setup.averaged.kernel, channels, setup.b_max)
    if threshold is not None:
        primary = setup.averaged.kernel.primary
        breaks[primary] = (*breaks.get(primary, ()), threshold)
    return breaks


def _panel_geometry(setup, lattice, channels, within, *, nodes=24, cells=None):
    table = lattice.cells
    if within == "cells":
        sequence = cell_boxes(table)
        boxes = (
            enumerate(sequence) if cells is None else ((i, sequence[i]) for i in cells)
        )
    elif within == "outside":
        pieces = complement(table)
        boxes = ((-1, box) for box in pieces)
    elif within == "support":
        boxes = ((None, {}),)
    else:
        raise ValueError("within must be 'support' or 'cells'")
    charts = _charts(setup, lattice)
    breaks = _face_breaks(setup, lattice, channels)
    kernel = setup.averaged.kernel
    variables = {v.name: v for v in kernel.variables}
    fourier = fourier_axis(lattice, kernel.azimuth)
    axis_nodes = (
        {kernel.azimuth: fourier.moment_nodes * nodes // 24}
        if fourier is not None
        else None
    )
    folded = {}
    for source, target in setup.averaged.ties:
        law = setup.population.factors.get(source)
        # A static folded density contributes zero outside its own support.
        # Conditional edges are evaluated at the parent nodes as before.
        if not parents(law) and hasattr(law, "support"):
            folded[target] = tuple(map(float, law.support))
    for piece, (expected, box) in enumerate(boxes):
        clipped = dict(box)
        for target, (lower, upper) in folded.items():
            lo, hi = box.get(target, (-np.inf, np.inf))
            clipped[target] = max(lo, lower), min(hi, upper)
        if any(lo > hi for lo, hi in clipped.values()):
            continue
        for point, weights in panels(
            setup.resolved,
            clipped,
            blocks=setup.population.blocks,
            units=setup.units,
            variables=variables,
            charts=charts,
            breaks=breaks,
            n=nodes,
            axis_nodes=axis_nodes,
        ):
            if expected is not None:
                match = ownership(point, table, weights.size) == expected
                if within == "outside":
                    match &= box_owner(point, pieces, weights.size) == piece
                if not np.any(match):
                    continue
                point, weights = (
                    {name: x[match] for name, x in point.items()},
                    weights[match],
                )
            yield point, weights * setup.population.column_factor


class UnassignedSupport(RoleError):
    """The hybrid cell assignment has no declaration on uncovered support."""


def _direct_values(basis, population, within):
    lat, ch = basis.lattice, basis.channels
    setup = configure(
        population, basis.averaged.kernel, ch, lat, basis.forward_phase, direct=True
    )
    kernel = setup.averaged.kernel
    if isinstance(kernel, HybridKernel) and kernel.switch is None and within != "cells":
        if next(_panel_geometry(setup, lat, ch, "outside"), None) is not None:
            raise UnassignedSupport(
                "hybrid assign= leaves emitting support outside its cells"
            )
    modes = integrate(
        setup.averaged,
        ch,
        _panel_geometry(setup, lat, ch, within),
        b_perp_max=setup.b_max,
        azimuth_factor=setup.azimuth,
    )
    return _stokes_array(basis, modes), setup


def _stokes_array(basis, modes):
    p = modes.P[0]
    values = {"I": modes.I, "Q": p.real, "U": p.imag, "V": modes.V}
    return jnp.stack(tuple(values[axis] for axis in basis.record.stokes), axis=-1)


def _projected_cell_values(basis, setup):
    channels, lattice = basis.channels, basis.lattice
    modes = projected_integral(
        setup.averaged,
        channels,
        _panel_geometry(setup, lattice, channels, "cells"),
        basis.m_max,
        b_perp_max=setup.b_max,
    )
    return _stokes_array(basis, modes)


def _spectrum(basis, values, *, ledger=None, moments=None):
    return Spectrum(
        values,
        basis.channels,
        components=tuple(basis.record.stokes),
        ledger=basis.ledger if ledger is None else ledger,
        plan=basis.plan,
        moments=moments,
        defaults=basis.forward_defaults,
    )


def direct(basis, population, *, within="support"):
    """Evaluate a specified population by kernel calls on split panels."""
    if within not in ("support", "cells"):
        raise ValueError("within must be 'support' or 'cells'")
    parts = population.parts if isinstance(population, Population) else (population,)
    values = []
    notes, tails = [], []
    for part in parts:
        val, setup = _direct_values(basis, part, within)
        values.append(val)
        kernel = setup.averaged.kernel
        if type(kernel) is not ContinuumKernel:
            continue
        lo, hi = support_for(setup.population, kernel.primary)
        if within == "support" and lo < kernel.gamma_min:
            notes.append("declared support extends below the continuum kernel regime")
        table = basis.lattice.cells
        if within == "cells" and kernel.primary in table.variables:
            column = table.variables.index(kernel.primary)
            hi = min(hi, float(np.max(table.hi[:, column])))
        cutoff = continuum_tail_threshold(kernel, basis.channels, setup.b_max)
        if cutoff is not None and hi > cutoff:
            tails.append(
                "the integrated support can reach the small-x continuation "
                f"above gamma = {cutoff:.6g}"
            )
    ledger = basis.ledger
    extra = []
    if notes:
        extra.append(ErrorTerm.unbounded(notes[0], name="regime"))
    if tails:
        extra.append(ErrorTerm.unbounded(tails[0], name="small_x:tail"))
    if extra:
        ledger = Ledger((*ledger.terms, *extra), assumptions=ledger.assumptions)
    return _spectrum(basis, sum(values), ledger=ledger)
