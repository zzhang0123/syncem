"""Independent Faraday screens with characteristic functions (SPEC Sec. 3.5).

LABEL [extension].

Faraday depths use rad/m^2. Characteristic functions have the shape of the
supplied phase factor arguments. The specified screen law has no certified
discrepancy with the source.
"""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from ..units import to_canonical
from ._common import (
    Law,
    concrete,
    count,
    interval,
    node_count,
    pair_values,
    probabilities,
    require,
    resolved,
    scalar,
)
from .densities import Delta

_DEPTH = "rad m^-2"


class NoRotation(Delta):
    """A point mass at depth zero."""

    def __init__(self):
        super().__init__(0.0, unit=_DEPTH)

    def characteristic(self, tau):
        return jnp.ones_like(tau, dtype=complex)


class _Screen(Law):
    mean: Any
    sigma: Any
    nodes: int = eqx.field(static=True)
    given_unit: ClassVar[str] = _DEPTH
    support: ClassVar[tuple] = (-np.inf, np.inf)

    @property
    def is_point(self):
        return concrete(self.sigma) and bool(self.sigma == 0)

    def _point_rule(self):
        return self.mean[None], jnp.ones(1)


def _mean_sigma(mean, sigma):
    mean, sigma, _ = pair_values(mean, sigma, unit=_DEPTH, names=("mean", "sigma"))
    if concrete(sigma):
        require(sigma >= 0, "screen sigma must be nonnegative")
    return mean, sigma


class GaussianScreen(_Screen):
    """Gaussian depth law. A zero standard deviation gives a point mass."""

    def __init__(self, mean, sigma, *, nodes=32):
        self.mean, self.sigma = _mean_sigma(mean, sigma)
        self.nodes = count(nodes)

    def density(self, x, *, parents=None):
        resolved(self.mean, self.sigma)
        if self.is_point:
            raise ValueError("a point screen has no density")
        z = (x - self.mean) / self.sigma
        return jnp.exp(-0.5 * z**2) / (np.sqrt(2 * np.pi) * self.sigma)

    def rule(self, n=None, *, parents=None):
        resolved(self.mean, self.sigma)
        if self.is_point:
            return self._point_rule()
        x, w = np.polynomial.hermite.hermgauss(node_count(n, self.nodes))
        return self.mean + np.sqrt(2) * self.sigma * x, jnp.asarray(w / np.sqrt(np.pi))

    def characteristic(self, tau):
        resolved(self.mean, self.sigma)
        return jnp.exp(1j * tau * self.mean - 0.5 * (tau * self.sigma) ** 2)


class LaplaceScreen(_Screen):
    """Laplace depth law parameterised by its standard deviation."""

    def __init__(self, mean, sigma, *, nodes=32):
        self.mean, self.sigma = _mean_sigma(mean, sigma)
        self.nodes = count(nodes)

    def density(self, x, *, parents=None):
        resolved(self.mean, self.sigma)
        if self.is_point:
            raise ValueError("a point screen has no density")
        b = self.sigma / np.sqrt(2)
        return jnp.exp(-jnp.abs(x - self.mean) / b) / (2 * b)

    def rule(self, n=None, *, parents=None):
        resolved(self.mean, self.sigma)
        if self.is_point:
            return self._point_rule()
        n = node_count(n, self.nodes)
        if n == 1:
            return self._point_rule()
        # Independent Gauss–Laguerre rules on both exponential tails.
        from scipy.special import roots_laguerre

        xl, wl = roots_laguerre(n // 2)
        xr, wr = roots_laguerre(n - n // 2)
        x = jnp.asarray(np.r_[-xl[::-1], xr])
        w = jnp.asarray(np.r_[wl[::-1], wr] / 2)
        return self.mean + self.sigma / np.sqrt(2) * x, w

    def characteristic(self, tau):
        resolved(self.mean, self.sigma)
        return jnp.exp(1j * tau * self.mean) / (1 + 0.5 * (tau * self.sigma) ** 2)


class GammaScreen(_Screen):
    """Shifted gamma depth law, with mean and standard deviation in rad m^-2."""

    shape: Any

    def __init__(self, mean, sigma, shape=4.0, *, nodes=64):
        self.mean, self.sigma = _mean_sigma(mean, sigma)
        self.shape = scalar(shape, "GammaScreen shape")
        if concrete(self.shape):
            require(self.shape > 0, "GammaScreen shape must be positive")
        self.nodes = count(nodes)

    @property
    def support(self):
        resolved(self.mean, self.sigma, self.shape)
        return self.mean - self.sigma * jnp.sqrt(self.shape), np.inf

    def density(self, x, *, parents=None):
        resolved(self.mean, self.sigma, self.shape)
        if self.is_point:
            raise ValueError("a point screen has no density")
        scale = self.sigma / jnp.sqrt(self.shape)
        y = (x - self.mean) / scale + self.shape
        positive = y > 0
        z = jnp.where(positive, y, 1.0)
        ld = (self.shape - 1) * jnp.log(z) - z - jsp.gammaln(self.shape)
        return jnp.where(positive, jnp.exp(ld) / scale, 0.0)

    def rule(self, n=None, *, parents=None):
        resolved(self.mean, self.sigma, self.shape)
        if self.is_point:
            return self._point_rule()
        n = node_count(n, self.nodes)
        if n == 1:
            return self._point_rule()
        # Cancelling the y**(shape-1) factor at zero gives one smooth
        # panel. Use a second panel in log y to resolve the upper tail.
        # The fixed node counts keep the rule traceable in shape.
        n_low = n // 2
        t, wt = interval(0.0, 1.0, n_low)
        a = jnp.minimum(self.shape, 0.25)
        sigma_y = jnp.sqrt(self.shape)
        split = jnp.maximum(1.0, self.shape - 9 * sigma_y)
        upper = jnp.maximum(64.0, self.shape + 10 * sigma_y)
        log_split = jnp.log(split)
        log_norm = jsp.gammaln(self.shape)
        log_t = jnp.log(t)
        y_low = split * jnp.exp(log_t / a)
        ell_low = (
            self.shape * log_split + (self.shape / a - 1) * log_t - y_low - log_norm
        )
        w_low = wt * jnp.exp(ell_low) / a
        v, wv = interval(log_split, jnp.log(upper), n - n_low)
        y_high = jnp.exp(v)
        w_high = wv * jnp.exp(self.shape * v - y_high - log_norm)
        y = jnp.concatenate((y_low, y_high))
        w = jnp.concatenate((w_low, w_high))
        x = self.mean + self.sigma / jnp.sqrt(self.shape) * (y - self.shape)
        return probabilities(x, w)

    def characteristic(self, tau):
        resolved(self.mean, self.sigma, self.shape)
        t = tau * self.sigma / jnp.sqrt(self.shape)
        return jnp.exp(
            1j * (tau * self.mean - t * self.shape) - self.shape * jnp.log1p(-1j * t)
        )


class EmpiricalScreen(Law):
    """A finite depth distribution on specified points and probability weights."""

    depths: Any
    weights: Any
    given_unit: ClassVar[str] = _DEPTH
    fixed_nodes: ClassVar[bool] = True

    def __init__(self, depths, weights=None):
        points = jnp.asarray(to_canonical(depths, "depth", name="depths"), dtype=float)
        w = (
            jnp.ones_like(points)
            if weights is None
            else jnp.asarray(weights, dtype=float)
        )
        if points.ndim != 1 or not points.size or w.shape != points.shape:
            raise ValueError(
                "EmpiricalScreen needs nonempty 1D depths and matching weights"
            )
        require(jnp.all(jnp.isfinite(points)), "EmpiricalScreen depths must be finite")
        require(
            jnp.all(jnp.isfinite(w) & (w >= 0)),
            "EmpiricalScreen weights must be finite and nonnegative",
        )
        self.depths, self.weights = probabilities(points, w)

    @property
    def support(self):
        return jnp.min(self.depths), jnp.max(self.depths)

    def rule(self, n=None, *, parents=None):
        return self.depths, self.weights

    def characteristic(self, tau):
        tau = jnp.asarray(tau)

        def step(total, item):
            depth, weight = item
            return total + weight * jnp.exp(1j * tau * depth), None

        initial = jnp.zeros_like(tau, dtype=complex)
        return jax.lax.scan(step, initial, (self.depths, self.weights))[0]


class CumulantScreen(Law):
    """Characteristic-only depth model from the first four cumulants.

    Nonzero third or fourth cumulants require a finite `tau_max` in m².
    It is the caller's approximation domain. The transmission cannot
    amplify within this interval. The omitted cumulants have no bound.
    """

    cumulants: Any
    tau_max: float | None = eqx.field(static=True)
    given_unit: ClassVar[str] = _DEPTH
    support: ClassVar[tuple] = (-np.inf, np.inf)

    def __init__(self, cumulants, *, tau_max=None):
        self.cumulants = jnp.asarray(cumulants, dtype=float)
        if self.cumulants.shape != (4,):
            raise ValueError("CumulantScreen needs the first four cumulants")
        if tau_max is not None and (
            isinstance(tau_max, bool) or not np.isfinite(tau_max) or tau_max <= 0
        ):
            raise ValueError("CumulantScreen tau_max must be finite and positive")
        self.tau_max = None if tau_max is None else float(tau_max)
        _validate_cumulants(self.cumulants, self.tau_max)

    @property
    def truncates(self):
        """Whether the law uses higher cumulants in a finite-domain model."""
        return bool(np.any(np.asarray(self.cumulants[2:]) != 0))

    def characteristic(self, tau):
        argument = jnp.asarray(tau)
        if jnp.iscomplexobj(argument):
            raise ValueError("CumulantScreen needs real phase arguments")
        cumulants = _validate_cumulants(self.cumulants, self.tau_max)
        in_domain = jnp.isfinite(argument)
        if self.tau_max is not None:
            in_domain &= jnp.abs(argument) <= self.tau_max
        a, b, c, d = cumulants
        real = argument * (argument * (-b / 2 + argument * (argument * (d / 24))))
        imag = argument * (a - argument * (argument * (c / 6)))
        exponent = real + 1j * imag
        message = "CumulantScreen phase must be within its domain and cannot amplify"
        valid = jnp.all(in_domain & jnp.isfinite(exponent) & (real <= 0))
        exponent = _screen_check(exponent, valid, message)
        return jnp.exp(exponent)


def _screen_check(value, valid, message):
    if isinstance(valid, jax.core.Tracer):
        return eqx.error_if(value, ~valid, message)
    require(valid, message)
    return value


def _validate_cumulants(cumulants, tau_max):
    _, variance, skew, kurtosis = cumulants
    finite = jnp.all(jnp.isfinite(cumulants)) & (variance >= 0)
    message = "cumulants must be finite and variance nonnegative"
    if tau_max is None:
        finite &= (skew == 0) & (kurtosis == 0)
        message += ". Nonzero higher cumulants require tau_max"
    else:
        # The bracket in the real exponent is maximal at the endpoint
        # when the fourth cumulant is positive.
        endpoint = -variance / 2 + tau_max * (tau_max * (kurtosis / 24))
        finite &= (kurtosis <= 0) | ((endpoint <= 0) & jnp.isfinite(endpoint))
        message += ". The domain must not admit amplification"
    return _screen_check(cumulants, finite, message)
