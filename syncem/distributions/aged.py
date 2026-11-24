"""Aged power laws and algebraic cutoff quadrature (SPEC Sec. 3.5, 5.4).

LABEL [extension].

Lorentz factors are dimensionless. Ages use seconds and fields use gauss
internally. Panel nodes and weights have shape (n_node,) with weights in
cm^-2. The model has no certified error for a physical electron population.
"""

from __future__ import annotations

from typing import Any, ClassVar

import jax.numpy as jnp
import numpy as np

from ..constants import C_CGS, M_E
from ..units import UnitError
from ._common import concrete, interval, require, resolved, scalar
from .electrons import (
    ElectronFamily,
    _PAPER_PIVOT,
    _log_panel,
    _nonnegative,
    _positive,
    _quantity,
)
from .parameters import Free

_LOSS_COEFF = 4 * 6.6524587321e-25 / (3 * M_E * C_CGS)
_EV_ERG = 1.602176634e-12


def _radiation(value):
    if isinstance(value, Free):
        return value
    if hasattr(value, "to_value") and hasattr(value, "unit"):
        return _nonnegative(value.to_value("erg / cm3"), "radiation energy density")
    if isinstance(value, str):
        number, unit = value.split(maxsplit=1)
        value = (float(number), unit)
    if isinstance(value, tuple) and len(value) == 2:
        amount, unit = value
        units = {
            "erg cm-3": 1.0,
            "erg cm^-3": 1.0,
            "erg/cm3": 1.0,
            "eV cm-3": _EV_ERG,
            "eV cm^-3": _EV_ERG,
            "eV/cm3": _EV_ERG,
        }
        if unit in units:
            return _nonnegative(
                jnp.asarray(amount) * units[unit], "radiation energy density"
            )
    raise UnitError('radiation energy density needs units, such as "1 eV cm-3"')


class AgedPowerLaw(ElectronFamily):
    """A power law with radiative losses and an algebraic upper cutoff.

    The pivot amplitude is the coefficient of the uncut density, even
    when gamma_cut is below the pivot. With pitch absent the loss is
    isotropic. A supplied cosine selects the pitch-dependent loss.
    """

    fit_nodes: ClassVar[int] = 64
    index: Any
    explicit_cut: Any
    age: Any
    loss_field: Any
    radiation: Any
    pitch: Any

    def __init__(
        self,
        index,
        *,
        gamma_cut=None,
        age=None,
        loss_field=None,
        radiation=None,
        pitch=None,
        lo=None,
        column=None,
        amplitude=None,
        pivot=1e4,
        nodes=64,
    ):
        self.index = scalar(index, "aged power-law index")
        if concrete(self.index):
            require(
                self.index > 1,
                "AgedPowerLaw needs index > 1 for its cutoff to be integrable",
            )
        self.explicit_cut = (
            None if gamma_cut is None else _positive(gamma_cut, "gamma_cut")
        )
        if gamma_cut is not None and any(
            v is not None for v in (age, loss_field, radiation)
        ):
            raise ValueError("give gamma_cut or the parameters of radiative losses")
        if gamma_cut is None and (
            age is None or (loss_field is None and radiation is None)
        ):
            raise ValueError("give gamma_cut or age and a loss_field or radiation")
        self.age = (
            None if age is None else _positive(_quantity(age, "time", "age"), "age")
        )
        self.loss_field = (
            0.0
            if loss_field is None
            else _nonnegative(
                _quantity(loss_field, "field", "loss_field"), "loss_field"
            )
        )
        self.radiation = 0.0 if radiation is None else _radiation(radiation)
        self.pitch = (
            None if pitch is None else _quantity(pitch, "dimensionless", "pitch")
        )
        if self.pitch is not None and concrete(self.pitch):
            require(jnp.abs(self.pitch) <= 1, "pitch is a cosine in [-1, 1]")
        self._set_normalisation(lo, None, column, amplitude, pivot, nodes)
        if concrete(
            self.lo,
            self.explicit_cut,
            self.age,
            self.loss_field,
            self.radiation,
            self.pitch,
        ):
            require(self.gamma_cut > self.lo, "gamma_cut must exceed the lower support")

    def _inverse_cut(self):
        resolved(
            self.explicit_cut, self.age, self.loss_field, self.radiation, self.pitch
        )
        if self.explicit_cut is not None:
            return 1 / self.explicit_cut
        magnetic = self.loss_field**2 / (8 * np.pi)
        if self.pitch is not None:
            magnetic = magnetic * (1.5 * (1 - self.pitch**2))
        return self.age * _LOSS_COEFF * (magnetic + self.radiation)

    @property
    def gamma_cut(self):
        inverse = self._inverse_cut()
        positive = inverse > 0
        return jnp.where(positive, 1 / jnp.where(positive, inverse, 1.0), jnp.inf)

    @property
    def support(self):
        return self.lo, self.gamma_cut

    @property
    def cutoff(self):
        return self.gamma_cut, self.index - 2

    def _parameter_domain(self):
        valid = (
            super()._parameter_domain() & jnp.isfinite(self.index) & (self.index > 1)
        )
        if self.explicit_cut is not None:
            return valid & jnp.isfinite(self.explicit_cut) & (self.explicit_cut > 0)
        valid &= jnp.isfinite(self.age) & (self.age > 0)
        valid &= jnp.isfinite(self.loss_field) & (self.loss_field >= 0)
        valid &= jnp.isfinite(self.radiation) & (self.radiation >= 0)
        if self.pitch is not None:
            valid &= jnp.abs(self.pitch) <= 1
        return valid

    def _shape(self, gamma):
        resolved(self.index, self.pivot)
        fraction = gamma * self._inverse_cut()
        valid = fraction < 1
        log_factor = jnp.log1p(-jnp.where(valid, fraction, 0.0))
        density = (gamma / self.pivot) ** (-self.index) * jnp.exp(
            (self.index - 2) * log_factor
        )
        return jnp.where(valid, density, 0.0)

    def _paper_scale(self):
        resolved(self.index, self.pivot)
        return _PAPER_PIVOT * (_PAPER_PIVOT / self.pivot) ** (-self.index)

    def _shape_panel(self, lo, hi, n, *, fitted):
        cut = self.gamma_cut
        resolved(self.index, self.pivot)
        if concrete(lo, hi, cut, self.index) and not fitted:
            if float(hi) == float(cut) and hi > lo:
                return self._jacobi_panel(lo, hi, n)
        return self._cutoff_panel(lo, hi, n)

    def _jacobi_panel(self, lo, hi, n):
        from scipy.special import roots_jacobi

        alpha = self.index - 2
        t, w = roots_jacobi(n, float(alpha), 0.0)
        h = jnp.log(hi / lo) / 2
        distance = h * (1 - t)
        gamma = hi * jnp.exp(-distance)
        smooth = (-jnp.expm1(-distance) / distance) ** alpha
        return (
            gamma,
            w
            * h ** (alpha + 1)
            * gamma
            * (gamma / self.pivot) ** (-self.index)
            * smooth,
        )

    def _cutoff_panel(self, lo, hi, n):
        cut = self.gamma_cut
        bounded = jnp.isfinite(cut)
        safe_cut = jnp.where(bounded, cut, 2 * hi)
        top = jnp.log(safe_cut / lo)
        bottom = jnp.log(safe_cut / hi)
        top = jnp.maximum(top, jnp.finfo(float).tiny)
        alpha = self.index - 2
        vmax = jnp.log(top)
        # Use a stable relative lower bound for partial cells and a finite
        # truncation only when the rule reaches the cutoff singularity.
        log_lo = jnp.log(jnp.where(bottom > 0, bottom, 1.0))
        vmin = jnp.maximum(
            jnp.where(bottom > 0, log_lo, -jnp.inf), vmax - 40 / (1 + alpha)
        )
        v, w = interval(vmin, vmax, n)
        d = jnp.exp(v)
        gamma = safe_cut * jnp.exp(-d)
        ratio = -jnp.expm1(-d) / jnp.where(d > 0, d, 1.0)
        ratio = jnp.where(d > 0, ratio, 1.0)
        density = (gamma / self.pivot) ** (-self.index) * ratio**alpha
        cut_w = w * jnp.exp((alpha + 1) * v) * gamma * density
        plain_x, plain_w = _log_panel(lo, hi, n)
        plain_w = plain_w * self._shape(plain_x)
        return jnp.where(bounded, gamma, plain_x), jnp.where(bounded, cut_w, plain_w)

    def _unnormalised_mass(self):
        resolved(self.lo, self.index, self.pivot)
        # The substitution y = 1/gamma - 1/gamma_cut leaves y**(index-2) dy.
        # This normaliser also gives the derivative at zero radiative loss.
        inverse_cut = self._inverse_cut()
        log_factor = (self.index - 1) * jnp.log1p(-self.lo * inverse_cut)
        return (
            self.lo
            * (self.lo / self.pivot) ** (-self.index)
            * jnp.exp(log_factor)
            / (self.index - 1)
        )
