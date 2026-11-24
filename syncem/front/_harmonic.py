"""One affine product cell through the legacy analytic harmonic builder."""

from __future__ import annotations

import dataclasses

from ..base.variables import SYNCHROTRON_VARIABLES
from ..engine.single_reference import SingleReferenceBasis, from_single_reference
from ..lattice.lattice import Lattice
from ..lattice.representations import AzimuthHarmonic, Legendre
from ..model._direct import direct_channel_average
from ..model._population import PopulationSamples as NativeSamples, Reference, Support
from ..model.basis import build_basis
from ..model.index import Truncation
from ..population import Population, Samples, pair
from ..records.ledger import ErrorTerm, Ledger
from ..values.errors import RoleError
from ..values.spectrum import Spectrum
from ._predictions import channels_from

_AXES = ("gamma", "B", "depth")


def _scaled_samples(source):
    """The one joint Samples block and the population scale that multiplies it."""
    if isinstance(source, Samples):
        return source.placed(SYNCHROTRON_VARIABLES), 1.0
    parts = source.parts if isinstance(source, Population) else ()
    if (
        len(parts) != 1
        or len(parts[0].blocks) != 1
        or not isinstance(parts[0].blocks[0], Samples)
        or parts[0].factors
    ):
        raise RoleError(
            "the harmonic reference route needs one joint Samples declaration"
        )
    return parts[0].blocks[0].placed(SYNCHROTRON_VARIABLES), parts[0].column_factor


def _samples(source):
    return _scaled_samples(source)[0]


def build_harmonic(population, channels, lattice, kernel, *, phase=None, check=True):
    """Build the frozen harmonic response for one cell and convert its conventions."""
    if phase is not None:
        raise RoleError(
            "the harmonic reference route currently needs the sample-depth phase"
        )
    samples = _samples(population)
    lattice = Lattice.single(about=samples) if lattice is None else lattice
    table = lattice.cells
    if (
        table.n_cells != 1
        or set(table.variables) != set(_AXES)
        or not lattice.depth_in_total
        or lattice.order is None
        or any(chart != "affine" for chart in table.chart[0])
    ):
        raise RoleError("the harmonic bridge needs one affine gamma, B and depth cell")
    reps = lattice.representations
    if (
        not isinstance(reps.get("mu"), Legendre)
        or not isinstance(reps.get("eta"), Legendre)
        or not isinstance(reps.get("phi"), AzimuthHarmonic)
    ):
        raise RoleError(
            "the harmonic bridge needs pitch and inclination Legendre axes and AzimuthHarmonic"
        )
    columns = [table.variables.index(name) for name in _AXES]
    ranges = [(float(table.lo[0, i]), float(table.hi[0, i])) for i in columns]
    support = Support(*ranges)
    references = [float(table.reference[0, i]) for i in columns]
    widths = tuple(float(table.half_width[0, i]) for i in columns)
    ref = Reference(*references, scales=widths)
    degree = reps["mu"].degree, reps["eta"].degree
    tr = Truncation(*degree, lattice.order)
    ch = channels_from(channels)
    native = kernel.legacy(support, ch)
    legacy = build_basis(native, ch, tr, ref, support=support, convergence=False)
    return dataclasses.replace(
        from_single_reference(legacy), legacy_kernel=native, source_kernel=kernel
    )


def predict_harmonic(basis, population, *, check=True, stokes=None):
    """Pair samples on the imported geometry and optionally compare the native direct sum."""
    samples, scale = _scaled_samples(population)
    moments = pair(scale * Population(samples), basis.lattice)
    predicted = basis.predict(moments, check=False)
    ledger = predicted.ledger
    if check and basis.legacy_kernel is not None:
        points, w = samples.rule()
        source = NativeSamples(**points, weights=w)
        column = scale * (1 if samples.column is None else samples.column)
        reference = direct_channel_average(
            source, basis.legacy_kernel, basis.channels, amplitude=column, batch_size=4
        )
        keep = ["IQUV".index(axis) for axis in predicted.components]
        difference = predicted.stokes - reference.stokes[:, keep]
        ledger = Ledger(
            (*ledger.terms, ErrorTerm(difference, "measured", name="representation"))
        )
    result = Spectrum(
        predicted.stokes,
        basis.channels,
        components=predicted.components,
        ledger=ledger,
        moments=moments,
    )
    return result.select(stokes)


def is_harmonic_basis(basis):
    return isinstance(basis, SingleReferenceBasis) and basis.legacy_kernel is not None
