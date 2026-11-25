"""Physical electron family fits sharing harmonic and continuum responses."""

from dataclasses import dataclass, replace

import numpy as np

from ..distributions.electrons import ElectronFamily
from ..distributions.parameters import parameter_number, parameter_unit
from ..fitting._family_batch import FamilyBatch, family_pixel, fit_pixels
from ..fitting._family_noise import physical_jacobian
from ..fitting._family_starts import solve_starts
from ..fitting._combined import combined_columns
from ..fitting._family_result import FamilyFit
from ..fitting._free_fit_arrays import Withheld
from ..fitting.data import StokesData
from ..fitting.family_fit import (
    _amplitude,
    _coordinate,
    _measure_parameters,
    _parameter_values,
    _physical_parameters,
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
        def column(pop):
            return pop.factors["gamma"].integrated_column() * pop.column_factor

        value = column(self.population)
        sigma = 0.0
        if self.params:
            jac = physical_jacobian(
                lambda x: column(self.trial_population(x)), self.params
            )
            sigma, _ = self.noise.propagate(jac, ())
        return Observable(
            value=value,
            sigma=sigma,
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


def _fit_design(basis, data, n_parameters):
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
    retained = retained.for_parameters(
        n_parameters,
        lambda obs: obs.whitened(
            combined_columns(supplied, obs.channels, obs.components),
            components=obs.components,
        ),
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
    from scipy.stats import chi2 as chi2_distribution

    if (
        not isinstance(population, Population)
        or population.components
        or not isinstance(population.factors.get("gamma"), ElectronFamily)
    ):
        raise RoleError(
            "a two-regime family fit needs one declared electron family in gamma"
        )
    if isinstance(data, StokesData) and data.batch_shape:
        return fit_pixels(
            data,
            lambda pixel: fit_regime_family(basis, population, pixel, starts=starts),
        )
    if isinstance(data, SumFit) and data.retained.batch_shape:
        observations = data.problem.data

        def fit_pixel(index):
            part = replace(
                data,
                retained=family_pixel(data.retained, index),
                problem=replace(data.problem, data=family_pixel(observations, index)),
            )
            return fit_regime_family(basis, population, part, starts=starts)

        return FamilyBatch(
            observations,
            tuple(fit_pixel(index) for index in np.ndindex(data.retained.batch_shape)),
        )

    source = free_parameters(population)
    averaged = dict(basis.averaged)
    for name in source:
        if name.split(".", 1)[0] in averaged:
            raise RoleError(f"{name} changes specified basis statistics")
    linear = _amplitude(population, basis, "gamma", source, starts)
    n_parameters = len(source) + int(linear is not None and linear not in source)
    fitted, target, design, count = _fit_design(basis, data, n_parameters)
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

    x, attempts, reached = x0, 1, 1
    if names:
        x, attempts, reached = solve_starts(
            lambda trial: profile(trial)[0], "2-point", x0, lo, hi, 3
        )
    residual, amplitude, _ = profile(x)
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
    parameters = _physical_parameters(
        names, x, lo, hi, log, shape, linear, amplitude, amp_bounds, amp_unit
    )

    def trial_population(values):
        norm = values[-1] if linear is not None else 1.0
        return _population_at(population, names, values[: len(names)], linear, norm)

    def project(values):
        return np.concatenate(
            [
                np.asarray(part.values)
                for part in fitted.moments(trial_population(values))
            ]
        )

    jacobian = noise = covariance = None
    parameter_rank = 0
    if parameters:
        jacobian = physical_jacobian(project, parameters)
        parameters, noise, covariance, parameter_rank = _measure_parameters(
            design @ jacobian, parameters
        )

    def spectrum_jacobian(output):
        axes = output.harmonic.record.stokes
        columns = combined_columns(output._independent(), output.channels, axes)
        return (np.asarray(columns) @ jacobian).reshape(-1, len(parameters))

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
        noise=noise,
        trial_population=trial_population,
        spectrum_jacobian=spectrum_jacobian,
        moment_jacobian=lambda: jacobian,
        n_starts=attempts,
        converged_starts=reached,
    )
