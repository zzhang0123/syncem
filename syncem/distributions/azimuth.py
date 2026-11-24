"""Probability laws and the spin-two factor of an azimuth (SPEC Sec. 3.5).

LABEL [extension].

Angles use radian units and circular moments are dimensionless complex
scalars. Rules return arrays of shape (n_node,). The model has no certified
error for unobserved azimuth correlations.
"""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..units import UnitError, dimension_of
from ._common import (
    Law,
    canonical,
    concrete,
    count,
    node_count,
    probabilities,
    require,
    resolved,
    scalar,
)
from .parameters import Conditional, Free

_PERIOD = 2 * np.pi


def _azimuth_nodes(n):
    return _PERIOD * (jnp.arange(n) + 0.5) / n


class UniformAzimuth(Law):
    """Uniform projected azimuth on one period."""

    support: ClassVar[tuple] = (0.0, _PERIOD)
    carries_values: ClassVar[bool] = False
    nodes: int = eqx.field(static=True)

    def __init__(self, *, nodes=16):
        self.nodes = count(nodes)

    def rule(self, n=None, *, parents=None):
        n = node_count(n, self.nodes)
        return _azimuth_nodes(n), jnp.full(n, 1 / n)

    def density(self, x, *, parents=None):
        x = jnp.asarray(x)
        return jnp.where((x >= 0) & (x <= _PERIOD), 1 / _PERIOD, 0.0)

    def circular(self, s):
        return jnp.where(s == 0, 1.0 + 0j, 0j)


class AzimuthFactor(Law):
    """The complex spin-two moment, with no specification of a full density."""

    c: Any
    carries_values: ClassVar[bool] = False
    support: ClassVar[tuple] = (0.0, _PERIOD)

    def __init__(self, c):
        if isinstance(c, (Conditional, Free)):
            self.c = c
            return
        self.c = jnp.asarray(c, dtype=complex)
        if self.c.ndim:
            raise ValueError("AzimuthFactor needs one complex scalar")
        require(
            jnp.isfinite(self.c) & (jnp.abs(self.c) <= 1),
            "AzimuthFactor modulus must not exceed one",
        )

    def circular(self, s=2, *, parents=None):
        resolved(self.c)
        if s not in (-2, 0, 2):
            raise ValueError("AzimuthFactor declares only the spin-two moment")
        coefficient = (
            self.c.evaluate(parents) if isinstance(self.c, Conditional) else self.c
        )
        if isinstance(coefficient, AzimuthFactor):
            coefficient = coefficient.c
        coefficient = AzimuthFactor(coefficient).c
        if s == 0:
            return jnp.asarray(1.0 + 0j)
        return coefficient if s == 2 else jnp.conj(coefficient)


class AzimuthDensity(UniformAzimuth):
    """Density (1 + amplitude cos(2 (phi - angle))) / (2 pi)."""

    amplitude: Any
    angle: Any
    given_unit: str = eqx.field(static=True)
    carries_values: ClassVar[bool] = True

    def __init__(self, amplitude, angle, *, unit=None, nodes=16):
        self.amplitude = scalar(amplitude, "azimuth amplitude")
        if concrete(self.amplitude):
            require(
                jnp.abs(self.amplitude) <= 1,
                "azimuth amplitude modulus must not exceed one",
            )
        angle, given = canonical(angle, unit=unit, name="azimuth angle")
        if not isinstance(angle, Free) and (
            given is None or dimension_of(given) != "angle"
        ):
            raise UnitError('give angle in radians or degrees, such as "15 deg"')
        self.angle = scalar(angle, "azimuth angle")
        self.given_unit = given or "rad"
        self.nodes = count(nodes)

    def density(self, x, *, parents=None):
        resolved(self.amplitude, self.angle)
        x = jnp.asarray(x)
        return super().density(x) * (1 + self.amplitude * jnp.cos(2 * (x - self.angle)))

    def rule(self, n=None, *, parents=None):
        n = node_count(n, self.nodes)
        if n < 3:
            raise ValueError(
                "AzimuthDensity quadrature needs at least three nodes to integrate its density"
            )
        x = _azimuth_nodes(n)
        return probabilities(x, self.density(x))

    def circular(self, s):
        resolved(self.amplitude, self.angle)
        return jnp.where(
            s == 0,
            1.0 + 0j,
            jnp.where(
                jnp.abs(s) == 2, self.amplitude / 2 * jnp.exp(1j * s * self.angle), 0j
            ),
        )
