"""Predictions from penalised and nonnegative prebuilt-basis fits."""

from __future__ import annotations

import math
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..engine._remainder import remainder_term
from ..lattice.moments import Moments
from ..population.blocks import Samples
from ..records.ledger import ErrorTerm, Ledger
from ..results.printing import fit_summary
from ..values.errors import DataError
from ..values.observable import Observable
from ..values.spectrum import Spectrum, channel_conversion
from ._free_fit_arrays import (
    Withheld,
    factor_chunk,
    memory_chunk,
    on_pixels,
    shared_noise,
)
from ._penalized_arrays import host_pixels, ridge_project
from ._point_mass_fit import PointMassMap
from ._resample import redraw
from .free_fit import (
    RetainedCombinations,
    _Problem,
    _design,
    _ledger,
    _pairs,
    _predict_basis,
    _white,
)
from .penalties import Penalty, profile_max
from .rank import RankCut


def _fit_ledger(
    basis, data=None, sigma=None, *, point_mass=False, moments=None, chunk_bytes=2**27
):
    ledger = _ledger(
        basis, data=data, sigma=sigma, moments=moments, chunk_bytes=chunk_bytes
    )
    terms = list(ledger.terms)
    if point_mass:
        terms = [
            term
            for term in terms
            if term.name not in ("representation", "unconstrained")
        ]
        terms.append(
            ErrorTerm.unbounded(
                "the fitted masses do not measure representation error of the source",
                name="representation",
            )
        )
    if not any(term.name == "remainder" and term.declared for term in terms):
        terms = [term for term in terms if term.name != "remainder"]
        terms.append(
            ErrorTerm.unbounded(
                "no truncation envelope for the source was supplied", name="remainder"
            )
        )
    return Ledger(tuple(terms), assumptions=ledger.assumptions)


class _Fitted(eqx.Module):
    """Common predictions from scaled fitted moments."""

    LABEL: ClassVar[str] = "sec: linear fit"
    params: ClassVar[None] = None
    problem: _Problem
    moments: Moments
    alpha: jax.Array
    effective_parameters: jax.Array
    chi2: jax.Array
    dof: jax.Array
    p_value: jax.Array
    withheld: Withheld | None
    ledger: Ledger
    penalty: Penalty = eqx.field(static=True)
    chunk_bytes: int = eqx.field(static=True)
    chunk_pixels: int = eqx.field(static=True)
    _point_mass: ClassVar[bool] = False

    def __str__(self):
        return fit_summary(self)

    @property
    def plan(self):
        basis = self.problem.basis
        return getattr(basis, "plan", basis.record)

    @property
    def defaults(self):
        return (
            ("penalty", "UPRE" if self.penalty.alpha is None else self.penalty.alpha),
            ("chunk bytes", self.chunk_bytes),
            ("pixels per fit chunk", self.chunk_pixels),
        )

    def predict(self, channels=None, *, cov=False) -> Spectrum:
        """Predict source Stokes in the observation unit."""
        original = self.problem.basis
        basis = _predict_basis(original, channels)
        design = _design(basis)
        values = jnp.einsum("...k,csk->...cs", self.moments.values, design)
        rows = design.reshape(-1, design.shape[-1])
        sigma, unconstrained, matrix = self._project(rows, cov=cov)
        if sigma is not None:
            sigma, unconstrained = (
                sigma.reshape(values.shape),
                unconstrained.reshape(values.shape),
            )
        record = basis.record
        data = self.problem.data
        observed, target = (
            np.asarray(data.channels.support),
            np.asarray(basis.channels.support),
        )
        if target.min() < observed.min() or target.max() > observed.max():
            record = record.with_notes(
                "prediction channels extend outside the observed band"
            )
        ledger = _fit_ledger(
            basis,
            sigma=sigma,
            point_mass=self._point_mass,
            moments=self.moments,
            chunk_bytes=self.chunk_bytes,
        )
        if self._point_mass and not ledger["remainder"].declared:
            term = remainder_term(basis, self.moments)
            ledger = Ledger(
                tuple(term if old.name == term.name else old for old in ledger.terms),
                assumptions=ledger.assumptions,
            )
        result = Spectrum(
            values,
            basis.channels,
            ledger=ledger,
            components=record.stokes,
            plan=basis.result_plan(record),
            sigma=sigma,
            cov=matrix,
            unconstrained=unconstrained,
            moments=self.moments,
            defaults=self.defaults,
        )
        return result.to(data.unit)

    def observable(self, c) -> Observable:
        """Report a real linear functional of the scaled lattice moments."""
        rows = jnp.asarray(c)
        n = self.moments.values.shape[-1]
        if rows.shape[-1:] != (n,) or jnp.iscomplexobj(rows):
            raise DataError(f"a functional must end in {n} real lattice coefficients.")
        shape = (*self.problem.data.batch_shape, *rows.shape[:-1])
        flat = rows.reshape(-1, n)
        sigma, unconstrained, _ = self._project(flat)
        return Observable(
            (self.moments.values @ flat.T).reshape(shape),
            None if sigma is None else sigma.reshape(shape),
            None if unconstrained is None else unconstrained.reshape(shape),
            self.moments.origin,
            "linear functional of the scaled lattice moments",
        )

    @property
    def column(self) -> Observable:
        lattice = self.problem.basis.lattice
        rows = jnp.zeros(lattice.index.n)
        first = lattice.index.n_pairs * lattice.index.offsets[:-1]
        return self.observable(rows.at[first].set(lattice.weights))._replace(
            label="column in cm^-2"
        )

    @property
    def source_column(self) -> Observable:
        return self.column

    def _resampled_data(self, n, key):
        basis, data = self.problem.fit_basis, self.problem.data
        matrix = data._model_matrix(_design(basis), basis.record.stokes, "cgs")
        mean = jnp.einsum("rk,...k->...r", matrix, self.moments.values)
        return redraw(data, mean, n, key, self.chunk_bytes)

    def save(self, path):
        """Save fitted numeric outputs and provenance."""
        from ._fit_io import save_fit

        save_fit(self, path)


class PenalizedFit(_Fitted):
    """Ridge moments and noise-only statistics at the selected penalty."""

    kind: ClassVar[str] = "linear"
    estimator: ClassVar[str] = "noise-whitened ridge in the reference weights"
    population: ClassVar[None] = None
    weights: ClassVar[None] = None
    retained: RetainedCombinations
    factors: jax.Array
    rank_cut: RankCut

    @property
    def rank(self):
        return self.retained.rank

    def resample(self, n, *, key):
        """Refit Gaussian noise realizations of the fitted observation model."""
        from .penalized_fit import fit_penalized

        return fit_penalized(
            self.problem.basis,
            self._resampled_data(n, key),
            self.penalty,
            keep_directions=self.retained.directions is not None,
            chunk_bytes=self.chunk_bytes,
        )

    def _project(self, rows, *, cov=False):
        data, basis = self.problem.data, self.problem.fit_basis
        batch = data.batch_shape
        n = rows.shape[1]
        directions = self.retained.directions
        per_pixel = directions is not None and directions.ndim > 2
        if per_pixel:
            directions = directions.reshape(math.prod(batch), -1, n)
        size = memory_chunk(
            data,
            n,
            self.chunk_bytes,
            rows.dtype.itemsize,
            n_out=rows.shape[0],
            cov=cov,
            shared=shared_noise(data),
        )
        sigmas = self.retained.sigma.reshape(math.prod(batch), -1)
        filters = self.factors.reshape(math.prod(batch), -1)
        design = _design(basis) if directions is None else None
        pairs = _pairs(basis)

        def project(part, start):
            stop = start + math.prod(part.batch_shape)
            noise = sigmas[start:stop] if batch else self.retained.sigma
            factor = filters[start:stop] if batch else self.factors
            if per_pixel:
                vectors = directions[start:stop]
            elif directions is None:
                vectors = factor_chunk(_white(part, basis, design).design, pairs)
            else:
                vectors = directions
            return ridge_project(rows, vectors, noise, factor, cov)

        return on_pixels(project, data, size)

    def electron_spectrum(self, gamma=None):
        raise DataError(
            "ridge moments do not define an electron density. Fit a family or nonnegative cells."
        )


class NonnegativeFit(_Fitted):
    """Positive masses on canonical Lobatto coordinates."""

    kind: ClassVar[str] = "non-negative linear"
    estimator: ClassVar[str] = (
        "noise-whitened nonnegative ridge in the reference weights"
    )
    retained: ClassVar[None] = None
    _point_mass: ClassVar[bool] = True
    weights: jax.Array
    points: PointMassMap
    negative_intensity_columns: int = eqx.field(static=True)

    @property
    def rank(self):
        """Number of active masses. Effective parameters use their moment span."""
        return jnp.count_nonzero(self.weights, axis=-1)

    def resample(self, n, *, key):
        """Refit realizations using the same point masses and penalty policy."""
        from .penalized_fit import _point_fit, _setup

        problem, design = _setup(
            self.problem.basis,
            self._resampled_data(n, key),
            self.penalty,
            self.chunk_bytes,
        )
        return _point_fit(problem, design, self.points, self.penalty, self.chunk_bytes)

    @property
    def population(self):
        """The single-pixel fitted measure, when its total column is positive."""
        if self.problem.data.batch_shape:
            return None
        column = self.weights.sum()
        if float(column) <= 0:
            return None
        units = {v.name: v.unit for v in self.problem.basis.averaged.kernel.variables}
        coordinates = {
            name: (value, units[name]) if units[name] else value
            for name, value in self.points.coordinates.items()
        }
        return Samples(weights=self.weights, column=column, **coordinates)

    def _project(self, rows, *, cov=False):
        if cov:
            raise DataError(
                "a nonnegative fit has no linear noise covariance at its active constraints."
            )
        return None, None, None

    def electron_spectrum(self, gamma=None):
        """Coordinates on the primary axis and absolute weights of the atoms."""
        if gamma is not None:
            raise DataError(
                "the fitted measure holds point masses rather than a density on gamma."
            )
        name = self.problem.basis.averaged.kernel.primary
        if name not in self.points.coordinates:
            raise DataError(
                "the fitted measure has no point masses on the kernel's primary axis."
            )
        return self.points.coordinates[name], self.weights

    def profile_max(self, channels=None, *, stokes="I", delta_chi2=1.0) -> Observable:
        """Noise-only maximum from unpenalised refits of the fitted observations."""
        original, fit_basis, data = (
            self.problem.basis,
            self.problem.fit_basis,
            self.problem.data,
        )
        basis = _predict_basis(original, channels)
        design = _design(fit_basis)
        if stokes not in basis.record.stokes:
            raise DataError(f"the basis has no {stokes!r} Stokes rows.")
        rows = _design(basis)[:, basis.record.stokes.index(stokes), :]
        rows = np.asarray(
            rows * channel_conversion(basis.channels, "cgs", data.unit)[:, None]
        )

        def run(part, _):
            white = _white(part, fit_basis, design)

            def value(matrix, obs):
                return profile_max(
                    matrix, obs, self.points.moments, rows, delta_chi2=delta_chi2
                ).value

            return host_pixels(white, value)

        return Observable(
            on_pixels(run, data, self.chunk_pixels),
            origin="nonnegative_profile",
            label=f"noise-only profile maximum of {stokes} in {data.unit}",
        )
