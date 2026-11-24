"""Results for fixed and fitted electron families."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from functools import cached_property
from typing import Any


from ..engine.assemble import predict_values
from ..population import Population
from ..results.printing import fit_summary
from ..values.errors import DataError
from ..values.observable import Observable
from ..values.spectrum import Spectrum
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
        return ()

    @property
    def column(self):
        value = self.basis.lattice.column(self.moments)
        if hasattr(value, "value"):
            return value
        return Observable(value=value, origin="family", label="column in cm^-2")

    @property
    def source_column(self):
        return self.column

    def electron_spectrum(self, gamma=None):
        law = self.population.factors[self.basis.averaged.kernel.primary]
        return law if gamma is None else law.density(gamma)

    def predict(self, channels=None, *, cov=False):
        if cov:
            raise DataError(
                "full spectral covariance needs the fitted family response Jacobian"
            )
        return (
            self._fit_spectrum
            if channels is None
            else self._predict(self.basis.at(channels))
        )

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
