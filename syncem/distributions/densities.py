"""Point, continuous and tabulated probability laws (SPEC Sec. 3.5).

LABEL [extension].

Scalar distributions retain the supplied unit and use canonical units
internally. Rules return nodes and probability weights of shape (n_node,).
The quadrature has no certified enclosure.
"""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
import jax.scipy.special as jsp
import numpy as np

from ..units import UnitError, dimension_of
from ._common import (
    Law,
    bounded_density,
    canonical,
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

_ROOT_2PI = np.sqrt(2 * np.pi)


class Delta(Law):
    """A unit point mass."""

    value: Any
    given_unit: str | None = eqx.field(static=True)
    is_point: ClassVar[bool] = True

    def __init__(self, value, *, unit=None, nodes=1):
        count(nodes)
        value, self.given_unit = canonical(value, unit=unit)
        self.value = scalar(value, "point")

    @property
    def support(self):
        return self.value, self.value

    @property
    def is_even(self):
        return concrete(self.value) and bool(self.value == 0)

    def as_cosine(self):
        """Convert an angle-valued point to a point in the cosine."""
        resolved(self.value)
        return Delta(jnp.cos(self.value))

    def rule(self, n=None, *, parents=None):
        resolved(self.value)
        return self.value[None], jnp.ones(1)


class Uniform(Law):
    """Constant probability density between two finite bounds."""

    lo: Any
    hi: Any
    given_unit: str | None = eqx.field(static=True)
    nodes: int = eqx.field(static=True)

    def __init__(self, lo, hi, *, unit=None, nodes=16):
        self.lo, self.hi, self.given_unit = pair_values(lo, hi, unit=unit)
        if concrete(self.lo, self.hi):
            require(
                self.hi > self.lo, "Uniform upper bound must exceed its lower bound"
            )
        self.nodes = count(nodes)

    @property
    def support(self):
        return self.lo, self.hi

    @property
    def is_even(self):
        return concrete(*self.support) and bool(self.lo == -self.hi)

    def rule(self, n=None, *, parents=None):
        resolved(*self.support)
        x, w = interval(self.lo, self.hi, node_count(n, self.nodes))
        return x, w / (self.hi - self.lo)

    def density(self, x, *, parents=None):
        resolved(*self.support)
        return bounded_density(x, *self.support, lambda _: 1 / (self.hi - self.lo))


class Normal(Law):
    """Normal probability density, with quadrature from mean minus six sigma to mean plus six sigma."""

    mean: Any
    sigma: Any
    given_unit: str | None = eqx.field(static=True)
    nodes: int = eqx.field(static=True)
    support: ClassVar[tuple[float, float]] = (-np.inf, np.inf)

    def __init__(self, mean, sigma, *, unit=None, nodes=32):
        self.mean, self.sigma, self.given_unit = pair_values(
            mean, sigma, unit=unit, names=("mean", "sigma")
        )
        if concrete(self.sigma):
            require(self.sigma > 0, "Normal sigma must be positive")
        self.nodes = count(nodes)

    @property
    def quadrature_support(self):
        resolved(self.mean, self.sigma)
        return self.mean - 6 * self.sigma, self.mean + 6 * self.sigma

    @property
    def is_even(self):
        return concrete(self.mean) and bool(self.mean == 0)

    def rule(self, n=None, *, parents=None):
        resolved(self.mean, self.sigma)
        z, w = interval(-6.0, 6.0, node_count(n, self.nodes))
        return probabilities(self.mean + self.sigma * z, w * jnp.exp(-(z**2) / 2))

    def density(self, x, *, parents=None):
        resolved(self.mean, self.sigma)
        z = (jnp.asarray(x) - self.mean) / self.sigma
        return jnp.exp(-(z**2) / 2) / (_ROOT_2PI * self.sigma)


class LogNormal(Law):
    """Log-normal probability density. Width is the standard deviation in natural log."""

    median: Any
    width: Any
    given_unit: str | None = eqx.field(static=True)
    nodes: int = eqx.field(static=True)
    support: ClassVar[tuple[float, float]] = (0.0, np.inf)

    def __init__(self, median, width, *, unit=None, nodes=32):
        value, self.given_unit = canonical(median, unit=unit, name="median")
        self.median = scalar(value, "median")
        self.width = scalar(width, "log-normal width")
        if concrete(self.median):
            require(self.median > 0, "LogNormal median must be positive")
        if concrete(self.width):
            require(self.width > 0, "LogNormal width must be positive")
        self.nodes = count(nodes)

    @property
    def quadrature_support(self):
        resolved(self.median, self.width)
        return self.median * jnp.exp(-6 * self.width), self.median * jnp.exp(
            6 * self.width
        )

    def rule(self, n=None, *, parents=None):
        resolved(self.median, self.width)
        z, w = interval(-6.0, 6.0, node_count(n, self.nodes))
        return probabilities(
            self.median * jnp.exp(self.width * z), w * jnp.exp(-(z**2) / 2)
        )

    def density(self, x, *, parents=None):
        resolved(self.median, self.width)
        x = jnp.asarray(x)
        safe = jnp.where(x > 0, x, self.median)
        z = jnp.log(safe / self.median) / self.width
        return jnp.where(
            x > 0, jnp.exp(-(z**2) / 2) / (_ROOT_2PI * self.width * safe), 0.0
        )


class Tabulated(Law):
    """Probability density defined by linear interpolation of nonnegative sample values."""

    x: Any
    values: Any
    cdf: Any
    given_unit: str | None = eqx.field(static=True)
    nodes: int = eqx.field(static=True)

    def __init__(self, x, density, *, unit=None, nodes=64):
        self.x, self.given_unit = canonical(x, unit=unit)
        values = jnp.asarray(density, dtype=float)
        if self.x.ndim != 1 or self.x.size < 2 or values.shape != self.x.shape:
            raise ValueError(
                "Tabulated needs at least two points and matching 1D density values"
            )
        dx = jnp.diff(self.x)
        require(
            jnp.all(jnp.isfinite(self.x)) & jnp.all(dx > 0),
            "Tabulated coordinates must be finite and strictly increasing",
        )
        require(
            jnp.all(jnp.isfinite(values) & (values >= 0)),
            "Tabulated density must be finite and nonnegative",
        )
        cumulative = jnp.cumsum(dx * (values[:-1] + values[1:]) / 2)
        mass = cumulative[-1]
        require(
            jnp.isfinite(mass) & (mass > 0),
            "Tabulated density must have positive finite mass",
        )
        self.values = values / mass
        self.cdf = jnp.concatenate((jnp.zeros(1), cumulative / mass))
        self.nodes = count(nodes)

    @property
    def support(self):
        return self.x[0], self.x[-1]

    @property
    def kinks(self):
        return self.x[1:-1]

    def density(self, x, *, parents=None):
        return jnp.interp(jnp.asarray(x), self.x, self.values, left=0.0, right=0.0)

    def rule(self, n=None, *, parents=None):
        # Integrate in the probability coordinate, solving one quadratic per node.
        t, w = interval(0.0, 1.0, node_count(n, self.nodes))
        i = jnp.searchsorted(self.cdf, t, side="right") - 1
        width = self.x[i + 1] - self.x[i]
        left, right = self.values[i], self.values[i + 1]
        elapsed = (t - self.cdf[i]) / width
        final = jnp.sqrt(jnp.maximum(left**2 + 2 * (right - left) * elapsed, 0.0))
        denominator = jnp.where(elapsed > 0, left + final, 1.0)
        fraction = 2 * elapsed / denominator
        return self.x[i] + fraction * width, jnp.asarray(w)


class Maxwellian(Law):
    """Strength of a Gaussian random field, truncated at cut times the untruncated RMS."""

    rms: Any
    cut: Any
    given_unit: str | None = eqx.field(static=True)
    nodes: int = eqx.field(static=True)

    def __init__(self, rms, *, cut=8.0, nodes=96):
        rms, self.given_unit = canonical(rms, name="rms")
        self.rms = scalar(rms, "rms")
        self.cut = scalar(cut, "Maxwellian cut")
        if concrete(self.rms):
            require(self.rms > 0, "Maxwellian rms must be positive")
        if concrete(self.cut):
            require(self.cut > 0, "Maxwellian cut must be positive")
        self.nodes = count(nodes)

    @property
    def support(self):
        resolved(self.rms, self.cut)
        return 0.0, self.rms * self.cut

    def density(self, x, *, parents=None):
        lo, hi = self.support
        sigma = self.rms / np.sqrt(3)
        mass = jsp.gammainc(1.5, 1.5 * self.cut**2)
        return bounded_density(
            x,
            lo,
            hi,
            lambda z: np.sqrt(2 / np.pi)
            * (z / sigma) ** 2
            * jnp.exp(-0.5 * (z / sigma) ** 2)
            / (sigma * mass),
        )

    def rule(self, n=None, *, parents=None):
        x, w = interval(*self.support, node_count(n, self.nodes))
        return probabilities(x, w * self.density(x))


class Isotropic(Uniform):
    """Uniform cosine with a polar-angle quadrature at the poles."""

    def __init__(self, *, nodes=12):
        super().__init__(-1.0, 1.0, nodes=nodes)

    def rule(self, n=None, *, parents=None):
        return cosine_rule(self, node_count(n, self.nodes))


class UniformAngle(Law):
    """Uniform polar angle, represented as a probability law in its cosine."""

    lo: Any
    hi: Any
    angle_unit: str = eqx.field(static=True)
    nodes: int = eqx.field(static=True)

    def __init__(self, lo, hi, *, unit, nodes=12):
        self.lo, self.hi, given = pair_values(lo, hi, unit=unit)
        if given is None:
            raise UnitError("UniformAngle needs unit= in radians or degrees")
        if dimension_of(given) != "angle":
            raise UnitError("UniformAngle needs unit= in radians or degrees")
        self.angle_unit = given
        if concrete(self.lo, self.hi):
            require(
                (self.lo >= 0) & (self.hi <= np.pi) & (self.lo < self.hi),
                "UniformAngle bounds must increase in [0, pi]",
            )
        self.nodes = count(nodes)

    @property
    def support(self):
        resolved(self.lo, self.hi)
        return jnp.cos(self.hi), jnp.cos(self.lo)

    @property
    def is_even(self):
        return concrete(self.lo, self.hi) and bool(self.lo + self.hi == np.pi)

    def rule(self, n=None, *, parents=None):
        resolved(self.lo, self.hi)
        theta, w = interval(self.lo, self.hi, node_count(n, self.nodes))
        return jnp.cos(theta[::-1]), w[::-1] / (self.hi - self.lo)

    def density(self, x, *, parents=None):
        return bounded_density(
            x,
            *self.support,
            lambda z: 1 / ((self.hi - self.lo) * jnp.sqrt((1 - z) * (1 + z))),
        )


def cosine_rule(law, n, *, half_range=False):
    """Uniform cosine quadrature. Use polar angle for an interval reaching a pole.

    Folding requires a checked even integrand and even law. The call site
    supplies the first condition after its joint-reversal check.
    """
    lo, hi = law.support
    resolved(lo, hi)
    if half_range:
        if not law.is_even:
            raise ValueError("a half-range rule needs an even cosine law")
        lo = 0.0
    require((lo >= -1) & (hi <= 1), "cosine support must lie in [-1, 1]")
    x, w = interval(lo, hi, n)
    t, wt = interval(jnp.arccos(hi), jnp.arccos(lo), n)
    polar = (lo == -1) | (hi == 1)
    return probabilities(
        jnp.where(polar, jnp.cos(t[::-1]), x),
        jnp.where(polar, (wt * jnp.sin(t))[::-1], w),
    )
