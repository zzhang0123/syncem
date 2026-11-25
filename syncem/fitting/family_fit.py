"""Host separable fits of specified lattice densities with free scalar parameters.

LABEL [extension].

The host solver uses a whitened residual of shape (n_data,). Parameter
values and their noise covariance use physical units. A local fit has no
certified global minimum.
"""

from __future__ import annotations


from dataclasses import replace
from functools import cache

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..distributions.parameters import (
    Conditional,
    Free,
    free_fields,
    parameter_number,
    parameter_unit,
)
from ..distributions.user import UserFamily
from ..distributions.electrons import ElectronFamily
from ..engine.assemble import predict_values
from ..population.declaration import law_at, parents
from ..values.errors import DataError, RoleError
from ._family_curve import compiled_curve
from ._family_noise import FamilyNoise, physical_jacobian
from ._primary_rows import primary_geometry, primary_moment_jacobian
from ._family_starts import solve_starts
from ._family_result import BasisFamilyFit, FamilyFit, FixedFit, Parameter
from .free_fit import RetainedCombinations, _fit_basis, family_combinations
from ._free_fit_arrays import Withheld


def free_parameters(population):
    """Walk scalar family and conditional fields without rewriting declarations."""
    found = {}
    for name, law in population.factors.items():
        if isinstance(law, Free):
            found[name] = law
        elif isinstance(law, (Conditional, UserFamily)):
            found.update(
                (f"{name}.{field}", value)
                for field, value in law.params.items()
                if isinstance(value, Free)
            )
        else:
            found.update(
                (f"{name}.{field}", v) for field, v in free_fields(law).items()
            )
    return found


def _parameter_values(declaration, guesses):
    names, initial, lower, upper, logarithmic = [], [], [], [], []
    for name, marker in declaration.items():
        value = guesses.get(name, marker.start)
        if value is None:
            raise RoleError(f"{name} needs a starting value")
        unit = parameter_unit(marker, value)
        x = parameter_number(value, unit, name)
        lo, hi = marker.bounds or (None, None)
        low = (
            (0.0 if marker.log else -np.inf)
            if lo is None
            else parameter_number(lo, unit, name)
        )
        high = np.inf if hi is None else parameter_number(hi, unit, name)
        if not low < high or not low <= x <= high or not np.isfinite(x):
            raise RoleError(f"{name} has an invalid starting value or bounds")
        if marker.log:
            if low < 0 or x <= 0:
                raise RoleError(
                    f"{name} needs a positive start and nonnegative lower bound"
                )
            low = np.log(low) if low > 0 else -np.inf
            high, x = np.log(high), np.log(x)
        names.append(name)
        initial.append(x)
        lower.append(low)
        upper.append(high)
        logarithmic.append(marker.log)
    return (
        tuple(names),
        np.asarray(initial),
        np.asarray(lower),
        np.asarray(upper),
        np.asarray(logarithmic, bool),
    )


def _coordinate(x, logarithmic):
    values = np.asarray(x).copy()
    values[logarithmic] = np.exp(values[logarithmic])
    return values


def start_values(population, starts=None):
    """Resolve starting values for the initial coverage and column build."""
    declarations = free_parameters(population)
    available = dict(starts or {})
    for key, param in declarations.items():
        if param.start is None and key.endswith((".amplitude", ".column")):
            available.setdefault(key, 1.0)
    names, x, _, _, log = _parameter_values(declarations, available)
    return dict(zip(names, _coordinate(x, log)))


def _amplitude(population, basis, primary, declarations, guesses):
    law = population.factors.get(primary)
    if isinstance(law, Conditional):
        if basis.forward_population is None:
            raise RoleError(
                "conditional amplitude needs the original population declaration"
            )
        initial = population.at(start_values(population, guesses))
        definitions = {v.name: v for v in basis.averaged.kernel.variables}
        reference = {}
        table = basis.lattice.cells
        for name in parents(law):
            if name in table.variables:
                reference[name] = table.reference[0, table.variables.index(name)]
            elif name in basis.lattice.representations:
                reference[name] = 0.0
            else:
                variable = definitions[name]
                parent = variable.place(initial.factors[name])
                reference[name] = (
                    parent.rule()[0][0] if hasattr(parent, "rule") else parent
                )
        units = {
            name: definitions[name].unit for name in reference if definitions[name].unit
        }
        local = law_at(initial.factors[primary], reference, units)
        if (
            hasattr(local, "amplitude")
            and hasattr(local, "column")
            and local.amplitude is None
            and local.column is None
            and f"{primary}.amplitude" not in declarations
        ):
            return f"{primary}.amplitude"
        return None
    if law is None or not hasattr(law, "amplitude") or not hasattr(law, "column"):
        return None
    for field in ("column", "amplitude"):
        key = f"{primary}.{field}"
        if key in declarations:
            return key
    if law.column is None and law.amplitude is None:
        return f"{primary}.amplitude"
    return None


def _population_at(source, names, x, linear=None, amplitude=1.0):
    values = dict(zip(names, x))
    if linear is not None:
        values[linear] = amplitude
    population = source.at(values)
    if linear is not None:
        variable, field = linear.split(".")
        law = population.factors[variable]
        if isinstance(law, Conditional):
            return population if field in law.params else amplitude * population
        if getattr(law, field) is None:
            law = eqx.tree_at(
                lambda item: getattr(item, field),
                law,
                amplitude,
                is_leaf=lambda x: x is None,
            )
            population = population.replace(**{variable: law})
    return population


def _white(data, basis, population, *, allow_mismatch=False):
    moments = basis.moments(population, allow_mismatch=allow_mismatch)
    if isinstance(data, RetainedCombinations):
        target, model, count = data.whitened_moments(moments)
        return np.asarray(target), np.asarray(model), int(count)
    spectrum = predict_values(basis, moments)
    result = data.whitened(spectrum[..., None], components=basis.record.stokes)
    return np.asarray(result.data), np.asarray(result.design[:, 0]), int(result.count)


def _withheld(data, basis, moments):
    if isinstance(data, RetainedCombinations) or not np.any(data.withheld):
        return None
    values = predict_values(basis, moments)
    white = data.whitened(
        values[..., None], components=basis.record.stokes, withheld=True
    )
    residual = white.data - white.design[..., 0]
    return Withheld(jnp.sum(residual**2), white.count, None, jnp.max(jnp.abs(residual)))


def _physical_parameters(
    names, x, lo, hi, log, declarations, linear, amplitude, bounds, unit
):
    """Parameter values and admissible directions in physical coordinates."""
    table = {}
    values = _coordinate(x, log)
    lower, upper = _coordinate(lo, log), _coordinate(hi, log)
    for i, name in enumerate(names):
        table[name] = Parameter(
            float(values[i]),
            np.nan,
            (float(lower[i]), float(upper[i])),
            bool(min(x[i] - lo[i], hi[i] - x[i]) <= 1e-5),
            False,
            parameter_unit(declarations[name]),
        )
    if linear is not None:
        table[linear] = Parameter(
            amplitude, np.nan, bounds, amplitude in bounds, False, unit
        )
    return table


def _measure_parameters(response, table):
    """Fit the rank and covariance in the same physical coordinates as each output."""
    noise = FamilyNoise.from_response(response, np.ones(len(table)))
    covariance, visible, rank = noise.covariance()
    measured = {
        name: replace(
            param, sigma=float(np.sqrt(covariance[i, i])), identified=bool(visible[i])
        )
        for i, (name, param) in enumerate(table.items())
    }
    return measured, noise, covariance, rank


def fit_family(
    basis, data, population, *, starts=None, rebuild=None, allow_mismatch=False
):
    """Profile the linear normalization and fit a fixed-basis family with SciPy."""
    from scipy.stats import chi2 as chi2_dist

    if data.batch_shape:
        from ._family_batch import fit_pixels

        return fit_pixels(
            data,
            lambda pixel: fit_family(
                basis,
                pixel,
                population,
                starts=starts,
                rebuild=rebuild,
                allow_mismatch=allow_mismatch,
            ),
        )
    if rebuild is not None and isinstance(data, RetainedCombinations):
        raise RoleError(
            "a compressed moment likelihood requires a fixed observational basis"
        )
    prediction_basis = basis
    if not isinstance(data, RetainedCombinations):
        basis = _fit_basis(basis, data.channels)
    source = free_parameters(population)
    primary = basis.averaged.kernel.primary
    linear = _amplitude(population, basis, primary, source, starts)
    if isinstance(data, RetainedCombinations):
        count = len(source) + int(linear is not None and linear not in source)
        data = family_combinations(data, basis, count)
    if not source and linear is None:
        target, model, count = _white(
            data, basis, population, allow_mismatch=allow_mismatch
        )
        chi2 = float(np.sum((target - model) ** 2))
        moments = basis.moments(population, allow_mismatch=allow_mismatch)
        p = float(chi2_dist.sf(chi2, count)) if count > 0 else np.nan
        return FixedFit(
            prediction_basis,
            data,
            population,
            {},
            moments,
            chi2,
            count,
            p,
            0,
            withheld_details=_withheld(data, basis, moments),
            allow_mismatch=allow_mismatch,
        )
    shape = {k: v for k, v in source.items() if k != linear}
    names, x0, lo, hi, log = _parameter_values(shape, starts or {})
    if rebuild is None:
        resolved = set(basis.lattice.cells.variables) | set(
            basis.lattice.representations
        )
        for name in names:
            if name.split(".", 1)[0] not in resolved:
                raise RoleError(f"{name} changes the specified basis statistics")
    amplitude_bounds = (0.0, np.inf)
    amplitude_unit = parameter_unit(source[linear]) if linear in source else None
    if linear in source and source[linear].bounds is not None:
        lower, upper = source[linear].bounds
        amplitude_bounds = (
            0.0 if lower is None else parameter_number(lower, amplitude_unit, linear),
            (
                np.inf
                if upper is None
                else parameter_number(upper, amplitude_unit, linear)
            ),
        )
        if not 0 <= amplitude_bounds[0] < amplitude_bounds[1]:
            raise RoleError(
                "the profiled normalization needs nonnegative increasing bounds"
            )

    def unit_model(x):
        actual = _coordinate(x, log)
        pop = _population_at(population, names, actual, linear)
        active = basis if rebuild is None else rebuild(pop, dict(zip(names, actual)))
        y, s, count = _white(data, active, pop, allow_mismatch=allow_mismatch)
        return y, s, count, active

    analytic, observation = None, None
    if names and rebuild is None:
        start_population = _population_at(
            population, names, _coordinate(x0, log), linear
        )
        y, s, count, _ = unit_model(x0)
        analytic = compiled_curve(basis, data, start_population, names, log, x0, s)
        observation = y, count

    def evaluate(x):
        if analytic is None:
            y, s, count, active = unit_model(x)
        else:
            y, count = observation
            s, _ = analytic(x)
            active = basis
        a = 1.0
        if linear is not None:
            sound = np.dot(s, s)
            if sound == 0:
                raise DataError("the family emits no modeled signal on the fitted data")
            a = float(np.clip(np.dot(s, y) / sound, *amplitude_bounds))
        return y - a * s, a, s, count, active

    if x0.size:

        def residual_jacobian(x):
            residual, a, s, _, _ = evaluate(x)
            _, ds = analytic(x)
            if linear is None or a in amplitude_bounds:
                return -a * ds
            da = (residual - a * s) @ ds / np.dot(s, s)
            return -a * ds - s[:, None] * da

        count_starts = (
            4
            if any(isinstance(law, UserFamily) for law in population.factors.values())
            else 3
        )
        x, n_starts, converged_starts = solve_starts(
            lambda trial: evaluate(trial)[0],
            "2-point" if analytic is None else residual_jacobian,
            x0,
            lo,
            hi,
            count_starts,
        )
    else:
        x, n_starts, converged_starts = x0, 1, 1
    residual, amplitude, _, count, fitted_basis = evaluate(x)
    actual = _coordinate(x, log)
    fitted = _population_at(population, names, actual, linear, amplitude)
    table = _physical_parameters(
        names,
        x,
        lo,
        hi,
        log,
        shape,
        linear,
        amplitude,
        amplitude_bounds,
        amplitude_unit,
    )
    chi2 = float(np.dot(residual, residual))
    moments = fitted_basis.moments(fitted, allow_mismatch=allow_mismatch)
    moments = type(moments)(
        fitted_basis.lattice, jnp.asarray(moments.values), origin="family"
    )

    def trial_population(values):
        norm = values[-1] if linear is not None else 1.0
        return _population_at(population, names, values[: len(names)], linear, norm)

    def trial_basis(values, trial):
        return (
            fitted_basis
            if rebuild is None
            else rebuild(trial, dict(zip(names, values)))
        )

    @cache
    def moment_jacobian():
        # Fit coordinates enter the compiled derivative through values. The
        # shared declaration keeps the executable independent of each sightline.
        law = population.factors.get(primary)
        geometry = primary_geometry(fitted_basis.lattice, primary)
        if (
            geometry is not None
            and isinstance(law, (UserFamily, ElectronFamily))
            and all(key.startswith(primary + ".") for key in table)
            and (analytic is not None or not names)
        ):
            values = jnp.asarray([parameter.value for parameter in table.values()])
            fields = tuple(key.split(".", 1)[1] for key in table)
            derivative, check = primary_moment_jacobian(law, geometry, fields, values)
            scale = fitted.column_factor
            expected = np.asarray(moments.values)
            if np.max(np.abs(check * scale - expected)) <= 2e-11 * np.max(
                np.abs(expected)
            ):
                return np.asarray(derivative * scale)

        def project(values):
            trial = trial_population(values)
            return (
                trial_basis(values, trial)
                .moments(trial, allow_mismatch=allow_mismatch)
                .values
            )

        return physical_jacobian(project, table)

    def stokes_response(output):
        design = output.columns.lift(output.design())
        response = np.asarray(design) @ moment_jacobian()
        if rebuild is not None:
            # d(A m) = A dm + (dA) m. Hold the best-fit moments fixed when
            # differentiating the basis, whose averaged statistics may move.
            def column_effect(values):
                trial = trial_population(values)
                active = trial_basis(values, trial)
                if active.channels.row_keys() != output.channels.row_keys():
                    active = active.at(output.channels)
                return predict_values(active, moments)

            response += physical_jacobian(column_effect, table).reshape(response.shape)
        return response

    fitted_response = (
        stokes_response(fitted_basis)
        if not isinstance(data, RetainedCombinations)
        else None
    )
    if fitted_response is None:
        _, operator, _ = data.whitened_operator()
        response = np.asarray(operator) @ moment_jacobian()
    else:
        response = data.whitened(
            fitted_response, components=fitted_basis.record.stokes
        ).design
    table, noise, covariance, n_parameters = _measure_parameters(
        np.asarray(response), table
    )

    def spectrum_jacobian(output):
        if fitted_response is not None and output is fitted_basis:
            return fitted_response.reshape(-1, len(table))
        if rebuild is None:
            return stokes_response(output).reshape(-1, len(table))
        # Rebuilding identical bands in a differently grouped grid can perturb
        # finite differences by roundoff. Reuse every fitted physical band row.
        original = dict(zip(fitted_basis.channels.row_keys(), fitted_response))
        keys = output.channels.row_keys()
        new = [i for i, key in enumerate(keys) if key not in original]
        if new:
            destination = (
                output if len(new) == len(keys) else output.at(output.channels[new])
            )
            original.update(zip((keys[i] for i in new), stokes_response(destination)))
        return np.stack([original[key] for key in keys]).reshape(-1, len(table))

    dof = count - n_parameters
    probability = float(chi2_dist.sf(chi2, dof)) if dof > 0 else float("nan")
    cls = FamilyFit if rebuild is None else BasisFamilyFit

    return cls(
        prediction_basis if rebuild is None else fitted_basis,
        data,
        fitted,
        table,
        moments,
        chi2,
        dof,
        probability,
        n_parameters,
        withheld_details=_withheld(data, fitted_basis, moments),
        parameter_covariance=covariance,
        noise=noise,
        trial_population=trial_population,
        spectrum_jacobian=spectrum_jacobian,
        moment_jacobian=moment_jacobian,
        n_starts=n_starts,
        converged_starts=converged_starts,
        allow_mismatch=allow_mismatch,
    )
