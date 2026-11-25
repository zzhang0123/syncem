"""Results for fixed and fitted electron families."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any

import numpy as np


from ..engine.assemble import predict_values
from ..population import Population
from ..results.printing import fit_summary
from ..values.errors import DataError
from ..values.observable import Observable
from ..values.spectrum import Spectrum, channel_conversion
from ._family_noise import physical_jacobian
from ._family_derived import family_quantity
from ._free_fit_arrays import Withheld


@dataclass(frozen=True)
class Parameter:
    """A physical parameter at the solution and its observation-noise uncertainty."""

    value: float
    sigma: float
    bounds: tuple[float, float]
    at_bound: bool
    identified: bool
    unit: str | None = None


@dataclass(frozen=True)
class FamilyFit:
    """One sightline fitted by profiling the linear normalization."""

    basis: Any
    data: Any
    population: Population
    params: dict
    moments: Any
    chi2: float
    dof: int
    p_value: float
    rank: int
    withheld_details: Withheld | None = field(kw_only=True, default=None)
    parameter_covariance: Any = field(kw_only=True, default=None)
    allow_mismatch: bool = field(kw_only=True, default=False)

    noise: Any = field(kw_only=True, default=None, repr=False)
    trial_population: Any = field(kw_only=True, default=None, repr=False)
    moment_jacobian: Any = field(kw_only=True, default=None, repr=False)
    spectrum_jacobian: Any = field(kw_only=True, default=None, repr=False)
    n_starts: int = field(kw_only=True, default=1)
    converged_starts: int = field(kw_only=True, default=1)

    kind = "separable"
    estimator = "noise-whitened trust-region fit with profiled normalization"
    retained = None

    @property
    def withheld(self):
        return self.withheld_details

    def __str__(self):
        return fit_summary(self)

    @property
    def ledger(self):
        return self._fit_spectrum.ledger

    @property
    def plan(self):
        return self.basis.plan

    @property
    def defaults(self):
        return (
            ("starts reaching the minimum", (self.converged_starts, self.n_starts)),
        )

    def observable(self, c):
        """Evaluate a real linear functional and propagate its local noise response."""
        blocks = self.moments if isinstance(self.moments, tuple) else (self.moments,)
        moments = np.concatenate([np.asarray(m.values) for m in blocks])
        rows = np.asarray(c)
        if rows.shape[-1:] != moments.shape or np.iscomplexobj(rows):
            raise DataError(
                f"a linear functional must end in {len(moments)} real lattice coefficients"
            )
        value = rows @ moments
        if self.params:
            response = rows.reshape(-1, len(moments)) @ self.moment_jacobian()
            sigma, _ = self.noise.propagate(response, value.shape)
        else:
            sigma = np.zeros(value.shape)
        return Observable(
            value,
            sigma,
            origin="family",
            label="linear functional of the scaled lattice moments",
        )

    @property
    def column(self):
        lattice = self.basis.lattice
        rows = np.zeros(lattice.index.n)
        rows[lattice.index.n_pairs * lattice.index.offsets[:-1]] = lattice.weights
        return self.observable(rows)._replace(label="column in cm^-2")

    @property
    def source_column(self):
        return self.column

    def electron_spectrum(self, gamma=None):
        law = self.population.factors[self.basis.averaged.kernel.primary]
        return law if gamma is None else law.density(gamma)

    def predict(self, channels=None, *, cov=False):
        basis = self.basis if channels is None else self.basis.at(channels)
        value = self._fit_spectrum if channels is None else self._predict(basis)
        if self.noise is None and self.params:
            if cov:
                raise DataError(
                    "the fitted estimator supplies no prediction covariance"
                )
            return value
        shape = value.stokes.shape
        if cov:
            print(f"Spectrum covariance size {8 * int(np.prod(shape))**2:,} bytes")
        if not self.params:
            sigma = np.zeros(shape)
            matrix = np.zeros((sigma.size, sigma.size)) if cov else None
        else:
            jac = self.spectrum_jacobian(basis)
            conversion = np.asarray(
                channel_conversion(basis.channels, "cgs", value.unit)
            )
            jac = jac * np.repeat(conversion, shape[-1])[:, None]
            sigma, matrix = self.noise.propagate(jac, shape, cov=cov)
        return replace(value, sigma=sigma, cov=matrix)

    def derived(self, name):
        """Return a paper invariant with its local observation-noise uncertainty."""
        value, label = family_quantity(self.population, name)
        sigma = 0.0
        if self.params:
            if self.noise is None or self.trial_population is None:
                raise DataError("the fitted estimator supplies no derived response")
            jac = physical_jacobian(
                lambda x: family_quantity(self.trial_population(x), name)[0],
                self.params,
            )
            sigma, _ = self.noise.propagate(jac, ())
        return Observable(value=value, sigma=sigma, origin="family", label=label)

    @cached_property
    def _fit_spectrum(self):
        return self._predict(self.basis)

    def _predict(self, basis):
        if basis.forward_accuracy is not None:
            checked = basis.predict(
                self.population,
                check=basis.forward_accuracy,
                allow_mismatch=self.allow_mismatch,
            )
            return replace(checked, moments=self.moments).to(self.data.unit)
        value = predict_values(basis, self.moments)
        return Spectrum(
            value,
            basis.channels,
            components=basis.record.stokes,
            ledger=basis.ledger,
            moments=self.moments,
            plan=basis.plan,
        ).to(self.data.unit)

    def save(self, path):
        """Save fitted numeric outputs and provenance."""
        from ._fit_io import save_fit

        save_fit(self, path)


class FixedFit(FamilyFit):
    """Goodness of fit of the supplied column measure."""

    kind = "fixed"
    estimator = "declared population in the observation likelihood"


class BasisFamilyFit(FamilyFit):
    """A fitted family whose averaged parameters also enter the spectral basis."""

    kind = "basis-parametric"
    estimator = "bounded trust-region fit with profiled normalization and rebuilt averaged statistics"
