"""The fixed-basis linear sampler interface (SPEC Sec. 3.7 and 6.6).

LABEL sec: linear fit. A Problem caches the whitened design, data and
Gaussian normalisation from selected observation rows. Real free moments
already use unconstrained coordinates, with identity Jacobian.

Not certified. This density uses the declared noise and prior. The model
error ledger does not assign a distribution to discrepancy.

The whitened design and target are dimensionless. Coefficient and observation
vectors have shapes (..., n_row) and (..., n_data).
"""

from __future__ import annotations

import math
from typing import ClassVar, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..distributions.densities import Normal
from ..engine.basis import SpectralBasis
from ..engine.basis_sum import BasisSum
from ..units import UnitError, dimension_of
from ..values.errors import DataError, ImproperPosterior, RoleError
from ..values.spectrum import _array
from ._combined import IndependentBases, combined_columns, parts_of
from ._free_fit_arrays import _marginal_factor
from .data import StokesData, _checked
from .free_fit import fit_free
from .penalties import Penalty
from .rank import RANK_RTOL

__all__ = ["GaussianPrior", "Params", "Problem"]
_LOG_2PI = math.log(2.0 * math.pi)


class Params(NamedTuple):
    """Free coefficients in the scaled rows of the component lattices."""

    a: jax.Array


class GaussianPrior(eqx.Module):
    """Proper independent Gaussian priors in the scaled moment coordinates."""

    mean: jax.Array
    sigma: jax.Array

    def __init__(self, mean=0.0, sigma=1.0):
        mean = _array(mean, "prior mean")
        sigma = _array(sigma, "prior sigma")
        self.mean = _checked(
            mean, jnp.any(~jnp.isfinite(mean)), "prior means must be finite."
        )
        self.sigma = _checked(
            sigma,
            jnp.any((sigma <= 0) | ~jnp.isfinite(sigma)),
            "prior standard deviations must be finite and positive.",
        )

    def log_prob(self, values):
        """Sum of the normalised densities over independent coefficients."""
        z = (values - self.mean) / self.sigma
        return -jnp.sum(0.5 * z**2 + 0.5 * _LOG_2PI + jnp.log(self.sigma))


def _gaussian_prior(value):
    if value is None:
        return None
    if isinstance(value, Normal):
        unit = value.given_unit
        if unit is not None and dimension_of(unit) != "column":
            raise UnitError("a scaled moment prior needs a column density unit")
        # Normal already stores its numerical parameters in canonical units.
        value = GaussianPrior(value.mean, value.sigma)
    if not isinstance(value, GaussianPrior):
        raise TypeError("linear Problem prior must be Normal or GaussianPrior.")
    return value


def _prior(value, shape):
    value = _gaussian_prior(value)
    if value is None:
        return None
    for array in (value.mean, value.sigma):
        try:
            matches = jnp.broadcast_shapes(array.shape, shape) == shape
        except ValueError:
            matches = False
        if not matches:
            raise DataError(
                f"prior shape {array.shape} cannot broadcast to coefficients {shape}."
            )
    return value


def _solve_penalty(prior, *, rank, penalty):
    if prior is None:
        if rank is not None and penalty is not None:
            raise RoleError("choose a rank policy or a ridge penalty")
        return penalty
    prior = _gaussian_prior(prior)
    if prior.mean.ndim or prior.sigma.ndim or float(prior.mean) != 0.0:
        raise RoleError(
            "deterministic solve needs a zero-mean scalar Gaussian prior. "
            "Use log_prob for the declared Gaussian density."
        )
    if rank is not None or penalty is not None:
        raise RoleError(
            "the Gaussian prior sets the ridge penalty. Omit rank= and penalty=."
        )
    with np.errstate(over="ignore", under="ignore"):
        alpha = float(np.square(np.reciprocal(float(prior.sigma))))
    if not math.isfinite(alpha) or alpha <= 0:
        raise RoleError("the Gaussian prior precision is outside the numeric range")
    return Penalty(alpha)


def _values(params, shape):
    values = _array(
        params.a if isinstance(params, Params) else params, "free coefficients"
    )
    if values.shape != shape:
        raise DataError(
            f"expected free coefficients of shape {shape}, got {values.shape}."
        )
    return values


def _whitened_model(basis, data):
    columns = combined_columns(basis, data.channels, data.components)
    return data.whitened(columns, components=data.components)


def _log_normaliser(data, count):
    keep = data.mask & ~data.withheld
    if data.sigma is not None:
        scales = data.sigma.reshape(
            *data.sigma.shape[: -len(data._layout)], data.n_data
        )
        log_det_factor = jnp.sum(jnp.log(jnp.where(keep, scales, 1.0)), axis=-1)
    else:
        factor = _marginal_factor(data.cov, keep)
        log_det_factor = jnp.sum(
            jnp.log(jnp.diagonal(factor, axis1=-2, axis2=-1)), axis=-1
        )
    return log_det_factor + 0.5 * count * _LOG_2PI


def _join_prior(a, b):
    if a.prior is None or b.prior is None:
        return None

    def parameter(name):
        return jnp.concatenate(
            tuple(
                jnp.broadcast_to(getattr(p.prior, name), p.params.a.shape)
                for p in (a, b)
            ),
            axis=-1,
        )

    return GaussianPrior(parameter("mean"), parameter("sigma"))


class Problem(eqx.Module):
    """Pure JAX Gaussian log density on a prebuilt linear spectral basis."""

    LABEL: ClassVar[str] = "sec: linear fit"
    record: object = eqx.field(static=True)
    basis: SpectralBasis | BasisSum | IndependentBases
    data: StokesData
    params: Params
    prior: GaussianPrior | None
    _declared_prior: bool = eqx.field(static=True)
    design: jax.Array
    target: jax.Array
    log_normaliser: jax.Array

    def __init__(self, basis, data, *, prior=None, params=None):
        if not isinstance(
            basis, (SpectralBasis, BasisSum, IndependentBases)
        ) or not isinstance(data, StokesData):
            raise TypeError(
                "linear Problem expects a prebuilt spectral basis and StokesData."
            )
        first = parts_of(basis)[0]
        self.record = getattr(first, "plan", first.record)
        white = _whitened_model(basis, data)
        shape = (*data.batch_shape, white.design.shape[-1])
        self.basis, self.data = basis, data
        self.prior = _prior(prior, shape)
        self._declared_prior = self.prior is not None
        start = (
            jnp.zeros(shape, dtype=white.design.dtype)
            if params is None
            else _values(params, shape)
        )
        self.params = Params(start)
        self.design, self.target = white.design, white.data
        self.log_normaliser = _log_normaliser(data, white.count)

    @property
    def lattice(self):
        """The lattice or the tuple of independent component lattices."""
        if isinstance(self.basis, IndependentBases):
            return tuple(part.lattice for part in self.basis.parts)
        return parts_of(self.basis)[0].lattice

    def constrain(self, z) -> Params:
        """Name a real vector in its existing coordinates."""
        return Params(_values(z, self.params.a.shape))

    def unconstrain(self, params) -> jax.Array:
        """Extract the existing real coordinates."""
        return _values(params, self.params.a.shape)

    def _log_likelihood(self, values):
        residual = (self.design @ values[..., None])[..., 0] - self.target
        return -0.5 * jnp.sum(residual**2) - jnp.sum(self.log_normaliser)

    def log_likelihood(self, params) -> jax.Array:
        """Joint normalised likelihood across all fitted sightlines."""
        return self._log_likelihood(self.unconstrain(params))

    def log_prob(self, params) -> jax.Array:
        """Joint log density in the real free-moment coordinates."""
        if self.prior is None:
            raise ImproperPosterior(
                "free moments have no declared prior. Pass prior=Normal(...) or "
                "GaussianPrior(...), or use log_likelihood or the retained combinations."
            )
        values = self.unconstrain(params)
        return self._log_likelihood(values) + self.prior.log_prob(values)

    def profiled(self):
        """Return the least-squares chi-square function with an empty nonlinear input."""
        pseudoinverse = jnp.linalg.pinv(self.design, rtol=RANK_RTOL)
        coefficients = pseudoinverse @ self.target[..., None]
        residual = (self.design @ coefficients)[..., 0] - self.target
        chi2 = jnp.sum(residual**2)

        def evaluate(theta=None):
            if theta is not None and jnp.asarray(theta).size:
                raise DataError(
                    "the fixed linear problem has no nonlinear coordinates."
                )
            return chi2

        return evaluate

    def solve(
        self, *, rank=None, penalty=None, keep_directions=False, chunk_bytes=2**31
    ):
        """Fit the moments using a rank policy, ridge or the declared scalar prior."""
        if isinstance(self.basis, (BasisSum, IndependentBases)):
            if self._declared_prior or penalty is not None:
                raise RoleError(
                    "joined deterministic fits do not support priors or ridge penalties"
                )
            from .sum_fit import solve_sum

            return solve_sum(self, rank, chunk_bytes, keep_directions=keep_directions)
        penalty = _solve_penalty(self.prior, rank=rank, penalty=penalty)
        if penalty is not None:
            from .penalized_fit import fit_penalized

            return fit_penalized(
                self.basis,
                self.data,
                penalty,
                keep_directions=keep_directions,
                chunk_bytes=chunk_bytes,
            )
        return fit_free(
            self.basis,
            self.data,
            rank=rank,
            keep_directions=keep_directions,
            chunk_bytes=chunk_bytes,
        )

    def __add__(self, other) -> Problem:
        """Concatenate independent coefficient groups on the very same observations."""
        if not isinstance(other, Problem):
            return NotImplemented
        if self.data is not other.data and not bool(
            eqx.tree_equal(self.data, other.data, typematch=True)
        ):
            raise DataError(
                "combined problems must share observations, covariance, masks, and response."
            )
        parts = (*parts_of(self.basis), *parts_of(other.basis))
        basis = IndependentBases(
            parts, tuple(f"component {i}" for i in range(len(parts)))
        )
        initial = jnp.concatenate((self.params.a, other.params.a), axis=-1)
        combined = Problem(
            basis, self.data, params=initial, prior=_join_prior(self, other)
        )
        # A partially specified joined prior still leaves an improper score.
        # Preserve its declaration for the deterministic solver's route check.
        object.__setattr__(
            combined, "_declared_prior", self._declared_prior or other._declared_prior
        )
        return combined
