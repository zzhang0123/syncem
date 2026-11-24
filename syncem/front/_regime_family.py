"""Physical electron family fits sharing harmonic and continuum responses."""

from dataclasses import dataclass

import numpy as np

from ..distributions.electrons import ElectronFamily
from ..distributions.parameters import parameter_number, parameter_unit
from ..fitting._combined import combined_columns
from ..fitting._family_result import FamilyFit, Parameter
from ..fitting._finite_difference import _jacobian
from ..fitting._free_fit_arrays import Withheld
from ..fitting.data import StokesData
from ..fitting.family_fit import (
    _amplitude,
    _coordinate,
    _parameter_errors,
    _parameter_values,
    _population_at,
    free_parameters,
)
from ..fitting.sum_fit import SumFit
from ..lattice.moments import Moments
from ..population import Population
from ..values.errors import DataError, RoleError
from ..values.observable import Observable


@dataclass(frozen=True)
class RegimeFamilyFit(FamilyFit):
    """A fitted physical family and its two projected moment vectors."""

    @property
    def plan(self):
        return self.basis.plan

    @property
    def column(self):
        law = self.electron_spectrum()
        mass = (
            law._coefficient() * law._unnormalised_mass()
            if law.column is None
            else law.column
        )
        return Observable(
            value=mass * self.population.column_factor,
            origin="family",
            label="physical electron column in cm^-2",
        )

    def electron_spectrum(self, gamma=None):
        law = self.population.factors["gamma"]
        return law if gamma is None else law.density(gamma)

    def _predict(self, basis):
        moments = (
            self.moments if basis is self.basis else basis.moments(self.population)
        )
        return basis.predict(moments, check=False).to(self.data.unit)


class FixedRegimeFamilyFit(RegimeFamilyFit):
    """Goodness of fit of a supplied physical electron family."""

    kind = "fixed"
    estimator = "declared population in the two-regime observation likelihood"


def _fit_design(basis, data):
    """Map the full moment coordinates to whitened observations once."""
    if isinstance(data, StokesData):
        if data.batch_shape:
            raise DataError("the host family fit needs one sightline at a time")
        fitted = basis.at(data.channels)
        matrix = combined_columns(fitted._independent(), data.channels, data.components)
        white = data.whitened(matrix, components=data.components)
        return (
            fitted,
            np.asarray(white.data),
            np.asarray(white.design),
            int(white.count),
        )
    if not isinstance(data, SumFit):
        raise TypeError(
            "the two-regime family fit needs StokesData or a fitted moment sum"
        )
    retained = data.retained
    if retained.batch_shape:
        raise DataError("the host family fit needs one sightline at a time")
    if retained.directions is None:
        raise DataError(
            "fit the moment sum with keep_directions=True to use its likelihood"
        )
    supplied = data.problem.basis
    observed_channels = data.problem.data.channels
    expected = basis.at(observed_channels)._independent()
    keys = tuple((part.lattice.key, part.columns.key) for part in expected.parts)
    if (
        supplied.components != expected.components
        or keys
        != tuple((part.lattice.key, part.columns.key) for part in supplied.parts)
        or any(key is None for key, _ in keys)
        or any(
            not np.array_equal(
                original.at(observed_channels).design(), candidate.design()
            )
            for original, candidate in zip(supplied.parts, expected.parts, strict=True)
        )
    ):
        raise DataError(
            "the compressed likelihood needs the original branch moment coordinates"
        )
    count = int(retained.rank)
    if not count:
        raise DataError("the compressed likelihood has no retained coordinates")
    noise = np.asarray(retained.sigma[:count])
    return (
        basis,
        np.asarray(retained.beta[:count]) / noise,
        np.asarray(retained.directions[:count]) / noise[:, None],
        count,
    )


def _amplitude_bounds(marker, name):
    unit = parameter_unit(marker) if marker is not None else None
    lower, upper = (
        (None, None) if marker is None or marker.bounds is None else marker.bounds
    )
    lo = 0.0 if lower is None else parameter_number(lower, unit, name)
    hi = np.inf if upper is None else parameter_number(upper, unit, name)
    if not 0 <= lo < hi:
        raise RoleError(
            "the profiled normalization needs nonnegative increasing bounds"
        )
    return (lo, hi), unit


def _withheld(data, basis, moments):
    if not isinstance(data, StokesData) or not np.any(data.withheld & data.mask):
        return None
    spectrum = basis.predict(moments, check=False)
    white = data.whitened_spectrum(spectrum, selection=data.withheld)
    residual = np.asarray(white.data - white.design[:, 0])
    return Withheld(
        float(residual @ residual),
        int(white.count),
        None,
        float(np.max(np.abs(residual))),
    )


def fit_regime_family(basis, population, data, *, starts=None):
    """Fit physical parameters of one electron family with both bases fixed."""
    from scipy.optimize import least_squares
    from scipy.stats import chi2 as chi2_distribution

    if (
        not isinstance(population, Population)
        or population.components
        or not isinstance(population.factors.get("gamma"), ElectronFamily)
    ):
        raise RoleError(
            "a two-regime family fit needs one declared electron family in gamma"
        )
    fitted, target, design, count = _fit_design(basis, data)
    source = free_parameters(population)
    averaged = dict(basis.averaged)
    for name in source:
        if name.split(".", 1)[0] in averaged:
            raise RoleError(f"{name} changes specified basis statistics")
    linear = _amplitude(population, fitted, "gamma", source, starts)
    shape = {name: marker for name, marker in source.items() if name != linear}
    names, x0, lo, hi, log = _parameter_values(shape, starts or {})
    amp_bounds, amp_unit = _amplitude_bounds(source.get(linear), linear)

    def signal(x):
        trial = _population_at(population, names, _coordinate(x, log), linear)
        moments = fitted.moments(trial)
        values = np.concatenate([np.asarray(part.values) for part in moments])
        return design @ values

    def profile(x):
        model = signal(x)
        amplitude = 1.0
        if linear is not None:
            power = model @ model
            if not np.isfinite(power) or power == 0:
                raise DataError(
                    "the family emits no finite nonzero signal on the fitted data"
                )
            amplitude = float(np.clip((target @ model) / power, *amp_bounds))
        return target - amplitude * model, amplitude, model

    x = x0
    if names:
        solution = least_squares(
            lambda trial: profile(trial)[0],
            x0,
            bounds=(lo, hi),
            method="trf",
            max_nfev=60,
        )
        if not solution.success:
            raise DataError(
                f"two-regime family fit did not converge. {solution.message}"
            )
        x = solution.x
    residual, amplitude, unit_model = profile(x)
    physical = _coordinate(x, log)
    result = _population_at(population, names, physical, linear, amplitude)
    moments = tuple(
        Moments(
            part.lattice,
            m.values,
            origin="family",
            masked=m.masked,
            component=m.component,
        )
        for part, m in zip(
            fitted._independent().parts, fitted.moments(result), strict=True
        )
    )
    covariance, parameter_rank, parameters = None, 0, {}
    if names or linear is not None:
        jac = np.empty((target.size, 0))
        if names:
            jac, _ = _jacobian(
                lambda trial: amplitude * signal(trial),
                x,
                lo,
                hi,
                np.cbrt(np.finfo(float).eps),
            )
        scale = np.where(log, physical, 1.0)
        if linear is not None:
            jac = np.column_stack((jac, unit_model))
            scale = np.append(scale, 1.0)
        covariance, visible, parameter_rank = _parameter_errors(jac, scale)
        lower, upper = _coordinate(lo, log), _coordinate(hi, log)
        errors = np.sqrt(np.diag(covariance))
        for i, name in enumerate(names):
            parameters[name] = Parameter(
                float(physical[i]),
                float(errors[i]),
                (float(lower[i]), float(upper[i])),
                bool(min(x[i] - lo[i], hi[i] - x[i]) <= 1e-5),
                bool(visible[i]),
                parameter_unit(shape[name]),
            )
        if linear is not None:
            parameters[linear] = Parameter(
                amplitude,
                float(errors[-1]),
                amp_bounds,
                amplitude in amp_bounds,
                bool(visible[-1]),
                amp_unit,
            )
    chi2 = float(residual @ residual)
    dof = count - parameter_rank
    probability = float(chi2_distribution.sf(chi2, dof)) if dof > 0 else np.nan
    result_type = RegimeFamilyFit if parameters else FixedRegimeFamilyFit
    return result_type(
        basis,
        data.problem.data if isinstance(data, SumFit) else data,
        result,
        parameters,
        moments,
        chi2,
        dof,
        probability,
        parameter_rank,
        withheld_details=_withheld(data, fitted, moments),
        parameter_covariance=covariance,
    )
