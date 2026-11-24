"""Electron densities, amplitude conventions and cell quadratures (SPEC Sec. 3.5, 5.4).

LABEL [extension].

Electron energy coordinates are dimensionless Lorentz factors. Quadrature
weights represent columns in cm^-2 and have shape (n_node,). Analytic family
choices carry no certified error for the source density.
"""

from __future__ import annotations

from math import factorial
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..units import to_canonical
from ..values.errors import RoleError
from ..values.observable import Observable
from ._common import (
    Law,
    concrete,
    count,
    interval,
    node_count,
    probabilities,
    require,
    resolved,
    scalar,
)
from .parameters import Free

_LABEL = "paper coefficient"
_PAPER_PIVOT = 1e4


def _quantity(value, dimension, name):
    if isinstance(value, Free):
        return value
    return scalar(to_canonical(value, dimension, name=name), name)


def _bounds(lo, hi):
    lo = scalar(1.0 if lo is None else lo, "electron lower bound")
    hi = scalar(np.inf if hi is None else hi, "electron upper bound", allow_inf=True)
    if concrete(lo, hi):
        require((lo >= 1) & (hi > lo), "electron support must increase from gamma >= 1")
    return lo, hi


def _positive(value, name):
    x = scalar(value, name)
    if concrete(x):
        require(x > 0, f"{name} must be positive")
    return x


def _nonnegative(value, name):
    x = scalar(value, name)
    if concrete(x):
        require(x >= 0, f"{name} must be nonnegative")
    return x


def _log_panel(lo, hi, n):
    logs, w = interval(jnp.log(lo), jnp.log(hi), n)
    gamma = jnp.exp(logs)
    return gamma, w * gamma


def _tail_panel(lo, n):
    # Only used to normalise finite-mass smooth electron families on
    # an unbounded support. Kernel integration chooses its own tail route.
    t, w = interval(0.0, 1.0, n)
    d = -jnp.log1p(-t)
    gamma = lo * jnp.exp(d)
    return gamma, w * gamma / (1 - t)


class ElectronFamily(Law):
    """A nonnegative electron density in gamma carrying a column in cm^-2."""

    lo: Any
    hi: Any
    column: Any
    amplitude: Any
    pivot: Any
    nodes: int = eqx.field(static=True)
    variables: ClassVar[tuple[str, ...]] = ("gamma",)
    kinks: ClassVar[tuple] = ()
    cutoff: ClassVar[None] = None
    fit_nodes: ClassVar[int] = 32

    def _set_normalisation(self, lo, hi, column, amplitude, pivot, nodes):
        if column is not None and amplitude is not None:
            raise ValueError("give either column or amplitude")
        self.lo, self.hi = _bounds(lo, hi)
        self.pivot = _positive(pivot, "pivot")
        self.column = (
            None
            if column is None
            else _nonnegative(_quantity(column, "column", "column"), "column")
        )
        self.amplitude = (
            None
            if amplitude is None
            else _nonnegative(_quantity(amplitude, "column", "amplitude"), "amplitude")
        )
        self.nodes = count(nodes)

    @property
    def support(self):
        return self.lo, self.hi

    def _parameter_domain(self):
        lo, hi = self.support
        valid = jnp.isfinite(lo) & (lo >= 1) & (hi > lo)
        valid &= jnp.isfinite(self.pivot) & (self.pivot > 0)
        for scale in (self.column, self.amplitude):
            if scale is not None:
                valid &= jnp.isfinite(scale) & (scale >= 0)
        return valid

    def _shape(self, gamma):
        raise NotImplementedError

    def _shape_density(self, gamma):
        lo, hi = self.support
        resolved(lo, hi, self.pivot)
        gamma = jnp.asarray(gamma, dtype=float)
        inside = (gamma >= lo) & (gamma <= hi)
        safe = jnp.where(inside, gamma, lo)
        return jnp.where(inside, self._shape(safe), 0.0)

    def _shape_panel(self, lo, hi, n, *, fitted):
        x, w = _log_panel(lo, hi, n)
        return x, w * self._shape(x)

    def _unnormalised_mass(self):
        lo, hi = self.support
        resolved(lo, hi)
        safe_hi = jnp.where(jnp.isfinite(hi), hi, lo)
        _, w = self._shape_panel(lo, safe_hi, 128, fitted=True)
        if concrete(hi) and np.isfinite(hi):
            return jnp.sum(w)
        t, tw = _tail_panel(lo, 128)
        return jnp.where(jnp.isfinite(hi), jnp.sum(w), jnp.sum(tw * self._shape(t)))

    def _coefficient(self):
        if self.column is None:
            resolved(self.amplitude)
            return 1.0 if self.amplitude is None else self.amplitude
        resolved(self.column)
        mass = self._unnormalised_mass()
        require(
            jnp.isfinite(mass) & (mass > 0),
            "electron density must have finite positive mass for column=",
        )
        return jnp.where(jnp.isfinite(mass) & (mass > 0), self.column / mass, jnp.nan)

    def density(self, x, *, parents=None):
        """Column density dN/dgamma, including its normalisation."""
        return self._coefficient() * self._shape_density(x)

    def panel_rule(self, lo, hi, n=None, *, fitted=False):
        """Nodes and column weights on the intersection of a finite panel and the family support."""
        support_lo, support_hi = self.support
        resolved(support_lo, support_hi)
        lo = jnp.maximum(lo, support_lo)
        hi = jnp.minimum(hi, support_hi)
        valid = hi > lo
        require(jnp.isfinite(hi) | ~valid, "electron panels need a finite upper bound")
        a = jnp.where(valid, lo, support_lo)
        b = jnp.where(valid, hi, support_lo)
        n = node_count(n, self.fit_nodes if fitted else 24)
        x, w = self._shape_panel(a, b, n, fitted=fitted)
        return x, jnp.where(valid, w * self._coefficient(), 0.0)

    def rule(self, n=None, *, parents=None):
        """Probability rule. Forward cell pairing uses panel_rule to keep the column."""
        lo, hi = self.support
        resolved(lo, hi)
        if concrete(hi) and not np.isfinite(hi):
            raise RoleError(
                "an averaged electron family needs a bounded support or a kernel integration window"
            )
        x, w = self._shape_panel(lo, hi, node_count(n, self.nodes), fitted=True)
        return probabilities(x, w)

    def _paper_scale(self):
        return _PAPER_PIVOT * self._shape(_PAPER_PIVOT)

    def paper_coefficient(self, params=None):
        """Return the paper coefficient in its study normalisation (SPEC Sec. 3.5)."""
        law = self if params is None else self._at(params)
        return Observable(law._coefficient() * law._paper_scale(), label=_LABEL)

    def _at(self, params):
        fields = dict(params)
        names = tuple(fields)
        return eqx.tree_at(
            lambda law: tuple(getattr(law, name) for name in names),
            self,
            tuple(fields.values()),
            is_leaf=lambda value: value is None,
        )

    @classmethod
    def from_paper_coefficient(cls, A, **params):
        law = cls(**params)
        if law.column is not None or law.amplitude is not None:
            raise ValueError("from_paper_coefficient already sets the amplitude")
        value = _nonnegative(
            _quantity(A, "column", "paper coefficient"), "paper coefficient"
        )
        resolved(value)
        return law._at({"amplitude": value / law._paper_scale()})


class PowerLaw(ElectronFamily):
    """Density proportional to (gamma / pivot)**(-index)."""

    index: Any

    def __init__(
        self,
        index,
        *,
        lo=None,
        hi=None,
        column=None,
        amplitude=None,
        pivot=1e4,
        nodes=32,
    ):
        self.index = scalar(index, "power-law index")
        self._set_normalisation(lo, hi, column, amplitude, pivot, nodes)

    def _shape(self, gamma):
        resolved(self.index, self.pivot)
        return jnp.exp(-self.index * jnp.log(gamma / self.pivot))

    def _parameter_domain(self):
        valid = super()._parameter_domain() & jnp.isfinite(self.index)
        if self.column is not None:
            valid &= jnp.isfinite(self.hi) | (self.index > 1)
        return valid

    def _unnormalised_mass(self):
        lo, hi = self.support
        resolved(lo, hi, self.index)
        if concrete(hi, self.index) and not np.isfinite(hi) and self.index <= 1:
            raise ValueError("column= for an unbounded PowerLaw needs index > 1")
        finite = jnp.isfinite(hi)
        width = jnp.log(jnp.where(finite, hi, lo) / lo)
        q = 1 - self.index
        y = q * width
        denominator = jnp.where(y == 0, 1.0, y)
        exprel = jnp.where(
            jnp.abs(y) < 1e-5,
            1 + y * (1 / 2 + y * (1 / 6 + y / 24)),
            jnp.expm1(y) / denominator,
        )
        norm = lo * self._shape(lo)
        tail = jnp.where(q < 0, -1 / jnp.where(q < 0, q, -1.0), jnp.inf)
        return norm * jnp.where(finite, width * exprel, tail)


class BrokenPowerLaw(ElectronFamily):
    """Smoothly broken power law, normalised at the pivot."""

    index1: Any
    index2: Any
    gamma_break: Any
    smoothness: Any

    def __init__(
        self,
        index1,
        index2,
        gamma_break,
        *,
        smoothness=2.0,
        lo=None,
        hi=None,
        column=None,
        amplitude=None,
        pivot=1e4,
        nodes=32,
    ):
        self.index1 = scalar(index1, "lower index")
        self.index2 = scalar(index2, "upper index")
        self.gamma_break = _positive(gamma_break, "gamma_break")
        self.smoothness = _positive(smoothness, "smoothness")
        self._set_normalisation(lo, hi, column, amplitude, pivot, nodes)

    @property
    def kinks(self):
        return (self.gamma_break,)

    def _log_unscaled(self, gamma):
        resolved(self.index1, self.index2, self.gamma_break, self.smoothness)
        u = jnp.log(gamma / self.gamma_break)
        return -self.index1 * u - (
            self.index2 - self.index1
        ) / self.smoothness * jnp.logaddexp(0.0, self.smoothness * u)

    def _shape(self, gamma):
        resolved(self.pivot)
        return jnp.exp(self._log_unscaled(gamma) - self._log_unscaled(self.pivot))

    def _paper_scale(self):
        resolved(self.pivot)
        return _PAPER_PIVOT * jnp.exp(-self._log_unscaled(self.pivot))

    def _parameter_domain(self):
        valid = super()._parameter_domain()
        valid &= jnp.isfinite(self.index1) & jnp.isfinite(self.index2)
        valid &= jnp.isfinite(self.gamma_break) & (self.gamma_break > 0)
        valid &= jnp.isfinite(self.smoothness) & (self.smoothness > 0)
        if self.column is not None:
            valid &= jnp.isfinite(self.hi) | (self.index2 > 1)
        return valid

    def _unnormalised_mass(self):
        lo, hi = self.support
        resolved(lo, hi, self.index2)
        if concrete(hi, self.index2) and not np.isfinite(hi) and self.index2 <= 1:
            raise ValueError("column= for an unbounded BrokenPowerLaw needs index2 > 1")
        middle = jnp.clip(self.gamma_break, lo, hi)
        x, w = _log_panel(lo, middle, 64)
        left = jnp.sum(w * self._shape(x))
        finite = jnp.isfinite(hi)
        x, w = _log_panel(middle, jnp.where(finite, hi, middle), 64)
        bounded = jnp.sum(w * self._shape(x))
        # Cancel the leading tail power through t = (gamma/middle)**(1-index2).
        q = jnp.where(self.index2 > 1, self.index2 - 1, 1.0)
        t, wt = interval(0.0, 1.0, 128)
        u = jnp.log(middle / self.gamma_break) - jnp.log(t) / q
        log_scale = -self._log_unscaled(self.pivot) - self.index2 * jnp.log(
            middle / self.gamma_break
        )
        correction = (
            -(self.index2 - self.index1)
            / self.smoothness
            * jnp.logaddexp(0.0, -self.smoothness * u)
        )
        tail = middle / q * jnp.sum(wt * jnp.exp(log_scale + correction))
        return left + jnp.where(
            finite, bounded, jnp.where(self.index2 > 1, tail, jnp.inf)
        )


class _CumulantMixture(ElectronFamily):
    k1: Any
    k2: Any

    def __init__(self, k1, k2, *, lo, hi, amplitude, pivot, nodes):
        self.k1 = scalar(k1, "index mean")
        self.k2 = scalar(k2, "second index cumulant")
        self._set_normalisation(lo, hi, None, amplitude, pivot, nodes)

    def _shape(self, gamma):
        resolved(self.k1, self.k2, self.pivot)
        u = jnp.log(gamma / self.pivot)
        return jnp.exp(-self.k1 * u + 0.5 * self.k2 * u**2)


class _MomentMixture(Law):
    """Linear expansion in A times the moments of the index offsets."""

    variables: ClassVar[tuple[str, ...]] = ("gamma",)
    order: int = eqx.field(static=True)
    index0: Any
    pivot: Any
    lo: Any
    hi: Any

    def __init__(self, order, index0, *, pivot, lo, hi):
        if (
            isinstance(order, bool)
            or not isinstance(order, (int, np.integer))
            or order < 0
        ):
            raise ValueError("mixture order must be a nonnegative integer")
        self.order = int(order)
        self.index0 = scalar(index0, "reference index")
        self.pivot = _positive(pivot, "pivot")
        self.lo, self.hi = _bounds(lo, hi)

    @property
    def support(self):
        return self.lo, self.hi

    def basis_density(self, gamma):
        """One linear density column for every index moment, the last axis."""
        resolved(self.index0, self.pivot, self.lo, self.hi)
        gamma = jnp.asarray(gamma)
        keep = (gamma >= self.lo) & (gamma <= self.hi)
        u = jnp.log(jnp.where(keep, gamma, self.pivot) / self.pivot)
        powers = jnp.stack(
            [(-u) ** k / factorial(k) for k in range(self.order + 1)], axis=-1
        )
        return jnp.where(
            keep[..., None], jnp.exp(-self.index0 * u)[..., None] * powers, 0.0
        )


class PowerLawMixture:
    """Index mixture declarations in moment and cumulant form."""

    @staticmethod
    def moment_form(order, index0, *, pivot=1e4, lo=None, hi=None):
        return _MomentMixture(order, index0, pivot=pivot, lo=lo, hi=hi)

    @staticmethod
    def cumulant_form(
        k1, k2=0.0, *, pivot=1e4, lo=None, hi=None, amplitude=None, nodes=32
    ):
        return _CumulantMixture(
            k1, k2, lo=lo, hi=hi, amplitude=amplitude, pivot=pivot, nodes=nodes
        )

    @staticmethod
    def from_paper_coefficient(A, *, k1, k2=0.0, pivot=1e4, lo=None, hi=None, nodes=32):
        law = PowerLawMixture.cumulant_form(
            k1, k2, pivot=pivot, lo=lo, hi=hi, nodes=nodes
        )
        coefficient = _nonnegative(
            _quantity(A, "column", "paper coefficient"), "paper coefficient"
        )
        resolved(coefficient)
        return law._at({"amplitude": coefficient / law._paper_scale()})
