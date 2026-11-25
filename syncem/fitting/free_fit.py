"""A whitened free-moment fit on a prebuilt spectral basis (SPEC Sec. 6.1–6.5).

LABEL sec: linear fit. The design is lifted into the scaled lattice rows.
The result stores minimum-norm moments and noise-only retained coordinates.
A free fit provides no inferred population in the discarded subspace.

Not certified. Errors and predictive covariance describe the supplied noise
model at the selected rank. They do not bound representation error.

The lifted basis has shape (n_ch, n_stokes, n_row). Observations and Stokes
errors use the units declared in StokesData.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import replace
from numbers import Integral
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..engine.basis import SpectralBasis
from ..lattice.moments import Moments
from ..lattice.representations import AzimuthFourier
from ..records.ledger import ErrorTerm, Ledger
from ..results.printing import fit_summary
from ..values.errors import DataError, SizeError
from ..values.observable import Observable
from ..values.spectrum import Spectrum
from ._free_fit_arrays import (
    Withheld,
    factor_chunk,
    memory_chunk,
    on_pixels,
    pixel_chunks,
    project,
    shared_noise,
    solve_chunk,
    withheld_statistics,
)
from ._fit_records import RetainedCombinations, _Problem
from .data import StokesData
from ._response_bias import bias_elements, bias_terms, response_bias
from .rank import RankCut, warn_rank_margin

__all__ = ["Fit", "RetainedCombinations", "Withheld", "fit_free"]


def _design(basis):
    return basis.columns.lift(basis.design())


def _white(data, basis, design, *, held=False):
    return data.whitened(design, components=basis.record.stokes, withheld=held)


def _pairs(basis):
    names = basis.lattice.index.blocks
    if "Re M2" not in names or any(
        isinstance(rep, AzimuthFourier) for _, rep in basis.lattice.reps
    ):
        return None
    width = basis.lattice.index.block_size
    return tuple(
        tuple(range(names.index(name) * width, (names.index(name) + 1) * width))
        for name in ("Re M2", "Im M2")
    )


def _ledger(basis, data=None, sigma=None, *, moments=None, chunk_bytes=2**27):
    """Record missing model information without attributing it to noise."""
    old = {term.name: term for term in basis.ledger.terms if term.name}
    if "physical_kernel" not in old:
        old["physical_kernel"] = ErrorTerm.unbounded(
            "the source discrepancy from the kernel has no supplied bound",
            name="physical_kernel",
        )
    old.update(
        {
            "representation": ErrorTerm.unbounded(
                "the free moments specify no population for direct integration",
                name="representation",
            ),
            "remainder": ErrorTerm.unbounded(
                "the remainder needs the true even moments in each cell",
                name="remainder",
            ),
            "unconstrained": ErrorTerm.unbounded(
                "the row fraction outside the retained space is not a flux error",
                name="unconstrained",
            ),
            "outside_lattice": ErrorTerm.unbounded(
                "the source column outside the lattice has no supplied bound",
                name="outside_lattice",
            ),
        }
    )
    if sigma is not None:
        old["noise"] = ErrorTerm(
            sigma, "estimate", "standard deviation from observation noise", name="noise"
        )
    if data is not None and data.discrepancy is not None:
        old["discrepancy"] = data.discrepancy
    conditional_error = getattr(basis, "conditional_error", None)
    if moments is not None and conditional_error is not None:
        term = conditional_error(moments, chunk_bytes=chunk_bytes)
        if term.kind == "estimate":
            name = f"{term.name}:fitted_mass"
            old[name] = replace(term, name=name)
    return Ledger(tuple(old.values()), assumptions=basis.ledger.assumptions)


def _predict_basis(base, channels):
    if channels is None or channels is base.channels:
        return base
    return base.at(channels)


class Fit(eqx.Module):
    """Minimum-norm moment fit and predictions with noise-only uncertainties."""

    LABEL: ClassVar[str] = "sec: linear fit"
    kind: ClassVar[str] = "linear"
    estimator: ClassVar[str] = "noise-whitened SVD in the reference weights"
    population: ClassVar[None] = None
    params: ClassVar[None] = None
    weights: ClassVar[None] = None
    penalty: ClassVar[None] = None
    effective_parameters: ClassVar[None] = None

    rank: jax.Array
    chi2: jax.Array
    dof: jax.Array
    p_value: jax.Array
    retained: RetainedCombinations
    moments: Moments
    withheld: Withheld | None
    rank_cut: RankCut
    ledger: Ledger
    problem: _Problem
    chunk_bytes: int = eqx.field(static=True)
    chunk_pixels: int = eqx.field(static=True)

    def __str__(self):
        return fit_summary(self)

    @property
    def plan(self):
        """The basis plan or the build record when no plan has been attached."""
        basis = self.problem.basis
        return getattr(basis, "plan", basis.record)

    @property
    def defaults(self):
        """Processing choices used by this fit."""
        return (
            ("chunk bytes", self.chunk_bytes),
            ("pixels per fit chunk", self.chunk_pixels),
            (
                "retained directions",
                "stored" if self.retained.directions is not None else "recomputed",
            ),
        )

    def _project(self, rows, *, cov=False):
        data, basis = self.problem.data, self.problem.fit_basis
        fit_design = _design(basis)
        batch = data.batch_shape
        n = self.moments.values.shape[-1]
        pairs = _pairs(basis)
        shared = shared_noise(data)
        size = memory_chunk(
            data,
            n,
            self.chunk_bytes,
            rows.dtype.itemsize,
            n_out=rows.shape[0],
            cov=cov,
            shared=shared,
        )
        kept = self.retained.sigma.reshape(math.prod(batch), -1)
        directions = self.retained.directions
        per_pixel = directions is not None and directions.ndim > 2
        if per_pixel:
            directions = directions.reshape(math.prod(batch), -1, n)

        def chunk(part, start):
            stop = start + math.prod(part.batch_shape)
            spectrum = kept[start:stop] if batch else self.retained.sigma
            if per_pixel:
                vectors = directions[start:stop]
            elif directions is None:
                vectors = factor_chunk(_white(part, basis, fit_design).design, pairs)
            else:
                vectors = directions
            return project(rows, vectors, spectrum, cov)

        return on_pixels(chunk, data, size)

    def predict(self, channels=None, *, cov=False) -> Spectrum:
        """Predict channel Stokes and noise errors in the observation unit.

        Prediction without an explicit channel selection uses the original
        basis channels. An observing response changes the likelihood and
        leaves the returned spectrum in source Stokes space.
        """
        original = self.problem.basis
        basis = _predict_basis(original, channels)
        design = _design(basis)
        pixel_shape = self.problem.data.batch_shape
        row_count = math.prod(design.shape[:2])
        rows = design.reshape(row_count, design.shape[-1])
        if cov:
            n_bytes = math.prod(pixel_shape) * row_count**2 * rows.dtype.itemsize
            print(f"Spectrum covariance size {n_bytes:,} bytes")
        sigma, fraction, covariance = self._project(rows, cov=cov)
        shape = (*pixel_shape, *design.shape[:2])
        values = jnp.einsum("...k,csk->...cs", self.moments.values, design)
        sigma = sigma.reshape(shape)
        record = basis.record
        data = self.problem.data
        if np.any(
            np.asarray(basis.channels.support)
            < np.min(np.asarray(data.channels.support))
        ) or np.any(
            np.asarray(basis.channels.support)
            > np.max(np.asarray(data.channels.support))
        ):
            record = record.with_notes(
                "prediction channels extend outside the observed band"
            )
        result = Spectrum(
            values,
            basis.channels,
            ledger=_ledger(
                basis, sigma=sigma, moments=self.moments, chunk_bytes=self.chunk_bytes
            ),
            plan=basis.result_plan(record),
            components=basis.record.stokes,
            sigma=sigma,
            cov=covariance,
            unconstrained=fraction.reshape(shape),
            moments=self.moments,
        )
        return result.to(data.unit)

    def observable(self, c) -> Observable:
        """Predict any real linear functional in the scaled lattice rows."""
        rows = jnp.asarray(c)
        n = self.moments.values.shape[-1]
        if rows.shape[-1:] != (n,) or jnp.iscomplexobj(rows):
            raise DataError(
                f"a linear functional must end in {n} real lattice coefficients."
            )
        output = rows.shape[:-1]
        shape = (*self.problem.data.batch_shape, *output)
        flat = rows.reshape(math.prod(output), n)
        noise, fraction, _ = self._project(flat)
        value = self.moments.values @ flat.T
        return Observable(
            value.reshape(shape),
            noise.reshape(shape),
            fraction.reshape(shape),
            "minimum_norm",
            "linear functional of the scaled lattice moments",
        )

    @property
    def column(self) -> Observable:
        """Sum the zeroth spin-0 cell moments after restoring their weights."""
        lattice = self.problem.basis.lattice
        first = lattice.index.n_pairs * lattice.index.offsets[:-1]
        rows = jnp.zeros(lattice.index.n).at[first].set(lattice.weights)
        return self.observable(rows)._replace(label="column in cm^-2")

    @property
    def source_column(self) -> Observable:
        """The column on a standard inverse-moment lattice."""
        return self.column

    def electron_spectrum(self, gamma=None):
        """A free-moment fit does not define an electron density."""
        raise DataError(
            "use an electron family or nonnegative cells to fit an electron spectrum."
        )

    def save(self, path):
        """Save fitted numeric outputs and provenance."""
        from ._fit_io import save_fit

        save_fit(self, path)


def _fit_basis(basis, channels):
    return basis if basis.channels is channels else basis.at(channels)


def family_combinations(retained, basis, n_parameters):
    """Use genuine observed singular coordinates for a parametric likelihood."""

    def whiten(obs):
        observed = _fit_basis(basis, obs.channels)
        return obs.whitened(_design(observed), components=observed.record.stokes)

    return retained.for_parameters(n_parameters, whiten, pairs=_pairs(basis))


def fit_free(
    basis: SpectralBasis,
    data: StokesData,
    *,
    rank=None,
    keep_directions: bool = False,
    chunk_bytes: int = 2**31,
) -> Fit:
    """Fit free moments on a prebuilt basis. Batched reductions run in bounded chunks."""
    if not isinstance(basis, SpectralBasis) or not isinstance(data, StokesData):
        raise TypeError("fit_free expects a SpectralBasis and StokesData.")
    if (
        isinstance(chunk_bytes, bool)
        or not isinstance(chunk_bytes, Integral)
        or chunk_bytes <= 0
    ):
        raise ValueError("chunk_bytes must be a positive integer.")
    if not isinstance(keep_directions, bool):
        raise ValueError("keep_directions must be a bool.")
    if math.prod(data.batch_shape) == 0:
        raise DataError("there are no sightlines to fit.")
    if basis.lattice.key is None:
        raise DataError(
            "fit_free needs a concrete lattice key. Build the basis outside the transformation."
        )
    fitted_basis = _fit_basis(basis, data.channels)
    matrix = _design(fitted_basis)
    n = matrix.shape[-1]
    shared = shared_noise(data)
    held = bool(np.any(np.asarray(data.withheld)))
    pairs = _pairs(fitted_basis)
    fixed, pixel = bias_elements(data, n, shared=shared)
    scratch = math.ceil(1.5 * fixed * matrix.dtype.itemsize)
    if scratch >= chunk_bytes:
        raise SizeError(
            f"response diagnostics need {scratch:,} bytes of shared workspace. Increase chunk_bytes."
        )
    size = memory_chunk(
        data,
        n,
        int(chunk_bytes) - scratch,
        matrix.dtype.itemsize,
        held=held,
        shared=shared,
        extra_elements=pixel,
    )

    reduction = None
    if shared:
        sample = next(pixel_chunks(data, 1)) if data.batch_shape else data
        white = _white(sample, fitted_basis, matrix)
        _, reduction = solve_chunk(white, rank, pairs)

    def chunk(part, _):
        white = _white(part, fitted_basis, matrix)
        solution, svd = solve_chunk(white, rank, pairs, reduction)
        test = None
        if held:
            validation = _white(part, fitted_basis, matrix, held=True)
            test = withheld_statistics(part, validation, solution, svd)
        cut = None if shared else svd.cut
        stored = svd.directions if keep_directions and not shared else None
        bias = response_bias(part, matrix, fitted_basis.record.stokes, solution, svd)
        return solution, white.count, cut, stored, test, bias

    solution, count, cut, directions, validation, bias = on_pixels(chunk, data, size)
    if shared:
        directions, cut = reduction.directions, reduction.cut
    warn_rank_margin(cut)

    key_parts = f"{basis.lattice.key}:{basis.columns.key}"
    key = hashlib.sha256(key_parts.encode()).hexdigest()
    retained = RetainedCombinations(
        solution.beta,
        solution.sigma,
        directions,
        solution.rank,
        key,
        basis=basis,
        unit=data.unit,
        observations=data,
    )
    moment = Moments(
        basis.lattice,
        solution.moments,
        origin="minimum_norm",
        masked=tuple(map(int, np.flatnonzero(basis.columns.col_of_row < 0))),
        retained=key,
    )
    dof = count - solution.rank
    p = jax.scipy.special.gammaincc(jnp.maximum(dof, 1) / 2, solution.chi2 / 2)
    p = jnp.where(dof > 0, p, jnp.nan)
    reference = _ledger(fitted_basis, moments=moment, chunk_bytes=int(chunk_bytes))
    terms = tuple(
        term for term in reference.terms if term.name not in ("response", "discrepancy")
    )
    ledger = Ledger(
        (*terms, *bias_terms(data, bias)), assumptions=reference.assumptions
    )
    return Fit(
        solution.rank,
        solution.chi2,
        dof,
        p,
        retained,
        moment,
        validation,
        cut,
        ledger,
        _Problem(basis, fitted_basis, data),
        int(chunk_bytes),
        size,
    )
