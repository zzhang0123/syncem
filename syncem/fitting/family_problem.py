"""Differentiable fixed-basis likelihood for a primary electron family.

LABEL [extension].

Physical parameters use their declared canonical units. Shape parameters
appear as scalars in a named PyTree. The sampling density has no certified
discrepancy outside the declared likelihood.
"""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp

from ..distributions import Conditional, Free
from ..distributions.electrons import ElectronFamily
from ..distributions.parameters import (
    parameter_argument,
    parameter_number,
    parameter_unit,
)
from ..population.declaration import law_at, parents
from ..values.errors import DataError, ImproperPosterior, RoleError
from .family_fit import fit_family, free_parameters, start_values
from .free_fit import _design, _fit_basis
from ._primary_rows import primary_geometry, primary_rows
from .problem import GaussianPrior, _log_normaliser


def family_template(population, kernel, starts):
    """Resolve the starting law, including conditional factories of fixed parents."""
    primary = kernel.primary
    law = population.factors.get(primary)
    if law is None or not isinstance(law, (Conditional, ElectronFamily)):
        raise RoleError("a differentiable family problem needs a primary electron law")
    declared = free_parameters(population)
    if set(declared) - {k for k in declared if k.startswith(f"{primary}.")}:
        raise RoleError("the family likelihood requires a fixed spectral basis")
    starts = start_values(population, starts)
    definition = {v.name: v for v in kernel.variables}
    point, units = {}, {}
    for name in parents(law):
        variable = definition[name]
        if name not in population.factors:
            raise RoleError(f"the differentiable family needs a fixed parent {name}")
        value = variable.place(population.factors[name])
        if hasattr(value, "rule"):
            if not getattr(value, "is_point", False):
                raise RoleError(
                    f"the differentiable family needs a fixed parent {name}"
                )
            value = value.rule()[0][0]
        point[name] = float(value)
        if variable.unit:
            units[name] = variable.unit
    initial = population.at(starts)
    initial_law = law_at(initial.factors[primary], point, units)
    if not isinstance(initial_law, ElectronFamily):
        raise RoleError(
            "the conditional must produce an electron family with panel_rule"
        )
    if initial_law.amplitude is None and initial_law.column is None:
        name = f"{primary}.amplitude"
        if name not in declared:
            declared[name] = Free(1.0, bounds=(0, None))
            starts[name] = 1.0
            initial_law = initial_law._at({"amplitude": jnp.asarray(1.0)})
    return initial.replace(**{primary: initial_law}), declared, starts, point, units


class FamilyProblem(eqx.Module):
    """Pure JAX likelihood in named physical parameters and their unconstrained coordinates."""

    basis: object
    data: object
    population: object
    params: dict
    declaration: object
    priors: tuple
    joint_prior: object
    bounds: tuple = eqx.field(static=True)
    point: dict
    units: tuple = eqx.field(static=True)
    primary: str = eqx.field(static=True)
    geometry: tuple = eqx.field(static=True)
    design: jax.Array
    target: jax.Array
    normaliser: jax.Array

    def __init__(self, basis, data, population, template, prior=None):
        initial, markers, starts, point, units = template
        primary = basis.averaged.kernel.primary
        lattice = basis.lattice
        if lattice.cells.variables != (primary,) or lattice.representations:
            raise RoleError(
                "the differentiable family likelihood needs only primary cells"
            )
        if data.batch_shape:
            raise DataError(
                "the differentiable family likelihood accepts one sightline"
            )
        if prior is not None and not isinstance(prior, (dict, GaussianPrior)):
            raise TypeError(
                "family prior must be GaussianPrior or a mapping by parameter"
            )
        self.joint_prior = prior if isinstance(prior, GaussianPrior) else None
        supplied = {} if self.joint_prior is not None or prior is None else prior
        if set(supplied) - markers.keys():
            raise RoleError("family priors must name fitted parameters")
        self.params = {key: jnp.asarray(starts[key], dtype=float) for key in markers}
        self.priors = tuple(
            supplied.get(name, marker.prior) for name, marker in markers.items()
        )
        self.bounds = tuple(
            (
                key,
                (
                    parameter_number(lo, parameter_unit(marker), key)
                    if lo is not None
                    else (0.0 if marker.log else -math.inf)
                ),
                (
                    parameter_number(hi, parameter_unit(marker), key)
                    if hi is not None
                    else math.inf
                ),
            )
            for key, marker in markers.items()
            for lo, hi in (marker.bounds or (None, None),)
        )
        if any(not lo < float(self.params[key]) < hi for key, lo, hi in self.bounds):
            raise RoleError(
                "family sampling starts must lie strictly inside their bounds"
            )
        if not bool(initial.factors[primary]._parameter_domain()):
            raise RoleError(
                "family sampling starts must define a valid electron density"
            )
        self.declaration = population.factors[primary]
        self.basis, self.data, self.population = basis, data, initial
        self.point, self.units, self.primary = point, tuple(units.items()), primary
        self.geometry = primary_geometry(lattice, primary)
        if self.geometry is None:
            raise RoleError(
                "the differentiable family likelihood needs finite cell faces"
            )
        fitted = _fit_basis(basis, data.channels)
        white = data.whitened(_design(fitted), components=fitted.record.stokes)
        self.design, self.target = white.design, white.data
        self.normaliser = _log_normaliser(data, white.count)

    @property
    def record(self):
        return self.basis.plan

    def _law(self, params):
        values = {name.split(".", 1)[1]: value for name, value in params.items()}
        if isinstance(self.declaration, Conditional):
            conditional = {
                k: parameter_argument(self.declaration.params[k], v)
                for k, v in values.items()
                if k in self.declaration.params
            }
            law = self.declaration.evaluate(
                self.point, params=conditional, canonical_units=dict(self.units)
            )
            fields = {k: v for k, v in values.items() if k not in conditional}
            return law if not fields else law._at(fields)
        return self.declaration._at(values) if values else self.declaration

    def moments(self, params):
        """Scaled moments on fixed cells with traced family parameters."""
        return (
            primary_rows(self._law(params), self.geometry)
            * self.population.column_factor
        )

    def _transform(self, z):
        z = jnp.asarray(z, dtype=float)
        if z.shape != (len(self.bounds),):
            raise DataError(
                "unconstrained family coordinates must match the named parameters"
            )
        params, log_jacobian = {}, 0.0
        for value, (name, lo, hi) in zip(z, self.bounds):
            if math.isfinite(lo) and math.isfinite(hi):
                x = lo + (hi - lo) * jax.nn.sigmoid(value)
                log_jacobian += (
                    math.log(hi - lo)
                    + jax.nn.log_sigmoid(value)
                    + jax.nn.log_sigmoid(-value)
                )
            elif math.isfinite(lo):
                x = lo + jnp.exp(value)
                log_jacobian += value
            elif math.isfinite(hi):
                x = hi - jnp.exp(value)
                log_jacobian += value
            else:
                x = value
            params[name] = x
        return params, log_jacobian

    def constrain(self, z):
        return self._transform(z)[0]

    def unconstrain(self, params):
        values = []
        for name, lo, hi in self.bounds:
            x = params[name]
            if math.isfinite(lo) and math.isfinite(hi):
                x = jnp.log(x - lo) - jnp.log(hi - x)
            elif math.isfinite(lo):
                x = jnp.log(x - lo)
            elif math.isfinite(hi):
                x = jnp.log(hi - x)
            values.append(x)
        return jnp.asarray(values)

    @eqx.filter_jit
    def _safe(self, params):
        if set(params) != set(self.params):
            raise DataError("family parameters must match the compiled names")
        valid = jnp.all(
            jnp.asarray(
                [
                    jnp.isfinite(params[key])
                    & (lo <= params[key])
                    & (params[key] <= hi)
                    for key, lo, hi in self.bounds
                ]
            )
        )
        candidate = {k: jnp.where(valid, v, self.params[k]) for k, v in params.items()}
        valid &= self._law(candidate)._parameter_domain()
        return {
            k: jnp.where(valid, v, self.params[k]) for k, v in candidate.items()
        }, valid

    def log_likelihood(self, params):
        """Normalized likelihood in the selected channel noise model."""
        if not isinstance(params, dict):
            params = self.constrain(params)
        safe, valid = self._safe(params)
        residual = self.design @ self.moments(safe) - self.target
        return jnp.where(valid, -0.5 * jnp.sum(residual**2) - self.normaliser, -jnp.inf)

    def log_prob(self, params):
        """Likelihood and declared priors, with Jacobian for unconstrained input."""
        jac = 0.0
        if not isinstance(params, dict):
            params, jac = self._transform(params)
        params, valid = self._safe(params)
        if self.joint_prior is not None:
            log_prior = self.joint_prior.log_prob(
                jnp.asarray([params[k] for k, _, _ in self.bounds])
            )
        else:
            log_prior = 0.0
            for (name, lo, hi), prior in zip(self.bounds, self.priors):
                if prior is not None and hasattr(prior, "log_prob"):
                    log_prior += prior.log_prob(params[name])
                elif prior is not None and hasattr(prior, "density"):
                    log_prior += jnp.log(prior.density(params[name]))
                elif prior is None and math.isfinite(lo) and math.isfinite(hi):
                    log_prior -= math.log(hi - lo)
                else:
                    raise ImproperPosterior(f"give a proper prior for {name}")
        return jnp.where(valid, self.log_likelihood(params) + log_prior + jac, -jnp.inf)

    def profiled(self):
        """Return chi-square as a function of the named shape parameters."""
        normalisation = {f"{self.primary}.amplitude", f"{self.primary}.column"}
        if isinstance(self.declaration, Conditional):
            normalisation.difference_update(
                f"{self.primary}.{name}" for name in self.declaration.params
            )
        linear = next((name for name in self.params if name in normalisation), None)
        shape_names = self.params.keys() - {linear}
        if linear is not None:
            lo, hi = next((lo, hi) for name, lo, hi in self.bounds if name == linear)
            lo = max(lo, 0.0)
            start = lo + min(1.0, (hi - lo) / 2)

        def evaluate(shape=None):
            shape = {} if shape is None else shape
            if not isinstance(shape, dict) or shape.keys() != shape_names:
                raise DataError(
                    "profiled family coordinates must name the shape parameters"
                )
            candidate = self.params | shape
            if linear is not None:
                candidate[linear] = start
            safe, valid = self._safe(candidate)
            model = self.design @ self.moments(safe)
            if linear is not None:
                power = jnp.dot(model, model)
                fraction = jnp.dot(model, self.target) / jnp.where(power > 0, power, 1)
                amplitude = jnp.clip(start * fraction, lo, hi)
                model = model * (amplitude / start)
            return jnp.where(valid, jnp.sum((self.target - model) ** 2), jnp.inf)

        return evaluate

    def solve(self, *, starts=None):
        """Fit the declared primary family in the observation likelihood."""
        if isinstance(self.declaration, Conditional):
            raise RoleError(
                "conditional family log likelihood is available through the sampler interface"
            )
        source = self.population.replace(**{self.primary: self.declaration})
        return fit_family(
            _fit_basis(self.basis, self.data.channels),
            self.data,
            source,
            starts=self.params if starts is None else starts,
        )
