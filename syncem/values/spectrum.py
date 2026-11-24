"""Channel Stokes values in the 0.8 interface (SPEC Sec. 2.4 and 3.8).

LABEL ``eq: channel basis expansion``. Values, errors and covariance use the
same channel-major Stokes layout. Temperature units refer to band-integrated
reference spectra. Conversion uses the response weights of the channels that
produced the values. The CMB temperature scale uses 2.7255 K.

Not certified. Conversion and response propagation retain the input ledger's
error kinds and do not supply bounds on the input model.

The trailing shape is (n_ch, n_stokes), with a leading batch shape for
multiple sightlines.
"""

from __future__ import annotations

from dataclasses import replace
from typing import TYPE_CHECKING, Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..units import UnitError, convert
from .errors import DataError

if TYPE_CHECKING:
    from ..base.channels import Channels
    from ..records.ledger import Ledger

__all__ = ["Spectrum", "channel_conversion"]

_UNITS = ("cgs", "Jy/sr", "MJy/sr", "K_RJ", "K_CMB")
_AXES = ("I", "Q", "U", "V")
_K_B = 1.380649e-16  # erg/K
_H = 6.62607015e-27  # erg s
_C_CM = 2.99792458e10  # cm/s
_T_CMB = 2.7255  # K


def _check_unit(unit: str) -> None:
    if unit not in _UNITS:
        raise UnitError(f"Choose a spectrum unit from {_UNITS}, got {unit!r}.")


def _reference(nu: jax.Array, unit: str) -> jax.Array | float:
    if unit == "Jy/sr":
        return 1e-23
    if unit == "MJy/sr":
        return 1e-17
    rj = 2 * _K_B * (nu / _C_CM) ** 2
    if unit == "K_RJ":
        return rj
    x = (_H / (_K_B * _T_CMB)) * nu
    small = x == 0
    safe_x = jnp.where(small, 1.0, x)
    correction = jnp.where(
        small, 1.0, safe_x * jnp.exp(-0.5 * safe_x) / (-jnp.expm1(-safe_x))
    )
    return rj * correction**2


def _band_reference(channels: Channels, unit: str) -> jax.Array:
    if unit == "cgs":
        # These values already represent the channel-integrated cgs inputs.
        return jnp.ones_like(channels.centres_hz)
    nu = channels.nodes
    reference = jnp.sum(channels.response_weights() * _reference(nu, unit), axis=-1)
    return eqx.error_if(
        reference,
        jnp.any((reference <= 0) | ~jnp.isfinite(reference)),
        f"The band reference for {unit!r} must be finite and positive.",
    )


def channel_conversion(channels: Channels, source: str, target: str) -> jax.Array:
    """The ``(n_ch,)`` row factors from one spectrum unit to another."""
    _check_unit(source)
    _check_unit(target)
    if source == target:
        return jnp.ones_like(channels.centres_hz)
    return _band_reference(channels, source) / _band_reference(channels, target)


def _select_terms(ledger, keep, count):
    def one(term):
        value = term.value
        if (
            value is None
            or term.space != "stokes"
            or term.relative_to is not None
            or value.shape[-1:] != (count,)
        ):
            return term
        return replace(term, value=jnp.asarray(value)[..., keep])

    return type(ledger)(tuple(map(one, ledger.terms)), assumptions=ledger.assumptions)


def _components(components, count: int) -> tuple[str, ...]:
    if components is None:
        components = _AXES[:count]
    labels = tuple(components)
    if (
        len(labels) != count
        or not labels
        or len(set(labels)) != count
        or any(axis not in _AXES for axis in labels)
    ):
        raise DataError(
            f"Expected {count} distinct Stokes labels from {_AXES}, got {labels}."
        )
    return labels


def _array(value, label):
    arr = jnp.asarray(value)
    if jnp.iscomplexobj(arr):
        raise DataError(f"{label} must be real.")
    return arr.astype(jnp.result_type(arr, 1.0))


def _optional_array(value, label, shape):
    if value is None:
        return None
    arr = _array(value, label)
    try:
        broadcast = jnp.broadcast_shapes(arr.shape, shape)
    except ValueError:
        broadcast = None
    if broadcast != shape:
        raise DataError(f"{label} shape {arr.shape} cannot broadcast to {shape}.")
    return arr


class Spectrum(eqx.Module):
    """Channel values with optional noise, moment estimates and a typed ledger."""

    LABEL: ClassVar[str] = "eq: channel basis expansion"
    stokes: jax.Array
    channels: Channels
    ledger: Ledger
    plan: Any = eqx.field(static=True)
    unit: str = eqx.field(static=True)
    components: tuple[str, ...] = eqx.field(static=True)
    sigma: jax.Array | None
    cov: jax.Array | None
    unconstrained: jax.Array | None
    moments: Any
    defaults: tuple = eqx.field(static=True)

    def __init__(
        self,
        stokes,
        channels: Channels,
        *,
        ledger: Ledger,
        plan=None,
        unit: str = "cgs",
        components=None,
        sigma=None,
        cov=None,
        unconstrained=None,
        moments=None,
        defaults=(),
    ):
        """Place Stokes values in ``(..., n_ch, n_stokes)`` layout.

        ``cov`` is a dense noise covariance over the flattened channel and
        Stokes axes. With pixel axes, a shared covariance can broadcast over
        pixels. Omitted component labels use the first columns of I, Q, U, V.
        """
        _check_unit(unit)
        stokes = _array(stokes, "stokes")
        if stokes.ndim < 2 or stokes.shape[-2] != channels.n_ch:
            raise DataError(
                f"Expected stokes shape (..., {channels.n_ch}, n_stokes), got {stokes.shape}."
            )
        self.components = _components(components, stokes.shape[-1])
        self.stokes, self.channels, self.unit = stokes, channels, unit
        # Low-level kernel errors may still carry all four canonical components.
        if len(self.components) < len(_AXES):
            ledger = _select_terms(
                ledger, [_AXES.index(a) for a in self.components], len(_AXES)
            )
        self.ledger, self.plan, self.moments = ledger, plan, moments
        self.defaults = tuple(defaults)
        self.sigma = _optional_array(sigma, "sigma", stokes.shape)
        n = stokes.shape[-2] * stokes.shape[-1]
        if cov is not None:
            cov = jnp.asarray(cov)
            if cov.shape[-2:] != (n, n):
                raise DataError(f"Expected a covariance of shape (..., {n}, {n}).")
        self.cov = _optional_array(cov, "cov", (*stokes.shape[:-2], n, n))
        self.unconstrained = _optional_array(
            unconstrained, "unconstrained", stokes.shape
        )

    def select(self, stokes) -> Spectrum:
        """Select and order components by name, with P for Q and U.

        Absolute Stokes errors and noise follow the selected components.
        Scalar and singleton axes retain their broadcast shape.
        """
        if stokes is None:
            return self
        names = tuple(stokes.replace("P", "QU") if isinstance(stokes, str) else stokes)
        names = tuple(a for n in names for a in (("Q", "U") if n == "P" else (n,)))
        if not names or len(set(names)) != len(names) or not set(names) <= set(
            self.components
        ):
            raise DataError(
                f"stokes={stokes!r} must name distinct components among {self.components}"
            )
        if names == self.components:
            return self
        keep = [self.components.index(a) for a in names]
        count, n_ch = len(self.components), self.channels.n_ch

        def pick(value):
            if value is None or value.shape[-1:] != (count,):
                return value
            return value[..., keep]

        cov = None
        if self.cov is not None:
            flat = np.asarray([c * count + k for c in range(n_ch) for k in keep])
            cov = jnp.asarray(self.cov)[..., flat[:, None], flat[None, :]]
        return replace(
            self,
            stokes=pick(self.stokes),
            components=names,
            sigma=pick(self.sigma),
            cov=cov,
            unconstrained=pick(self.unconstrained),
            ledger=_select_terms(self.ledger, keep, count),
        )

    @property
    def nu(self) -> jax.Array:
        """Channel centres in Hz."""
        return self.channels.centres_hz

    def _stokes_axis(self, label: str) -> jax.Array | None:
        if label in self.components:
            return self.stokes[..., self.components.index(label)]
        return None

    @property
    def I(self) -> jax.Array | None:  # noqa: E743
        """Stokes intensity when present."""
        return self._stokes_axis("I")

    @property
    def Q(self) -> jax.Array | None:
        """First linear polarisation component when present."""
        return self._stokes_axis("Q")

    @property
    def U(self) -> jax.Array | None:
        """Second linear polarisation component when present."""
        return self._stokes_axis("U")

    @property
    def V(self) -> jax.Array | None:
        """Circular polarisation when present."""
        return self._stokes_axis("V")

    @property
    def P(self) -> jax.Array | None:
        """Complex linear polarisation when both linear components are present."""
        q, u = self.Q, self.U
        return None if q is None or u is None else q + 1j * u

    def to(self, unit: str) -> Spectrum:
        """Express each channel and its noise or error values in the requested unit."""
        if self.unit == unit:
            return self
        factor = channel_conversion(self.channels, self.unit, unit)
        per_channel = factor[:, None]
        cov = self.cov
        if cov is not None:
            flat_factor = jnp.repeat(factor, len(self.components))
            cov = cov * flat_factor[:, None] * flat_factor[None, :]
        return replace(
            self,
            stokes=self.stokes * per_channel,
            sigma=None if self.sigma is None else self.sigma * per_channel,
            cov=cov,
            ledger=self.ledger.scaled(per_channel),
            unit=unit,
        )

    def through(
        self, response, *, channels: Channels | None = None, components=None
    ) -> Spectrum:
        """Apply a real linear response in the current unit.

        The response maps flattened channel-major Stokes vectors. Pass a target
        layout if the response changes the number or meaning of the rows.
        Noise propagation requires the full covariance. A row's unconstrained
        norm fraction cannot be propagated without the discarded directions.
        """
        if self.unconstrained is not None:
            raise DataError(
                "Apply the response to the prediction design when it has discarded directions."
            )
        if self.sigma is not None and self.cov is None:
            raise DataError(
                "Provide full covariance to propagate noise through a response."
            )
        target = self.channels if channels is None else channels
        labels = self.components if components is None else tuple(components)
        response = _array(response, "response")
        response = eqx.error_if(
            response, jnp.any(~jnp.isfinite(response)), "response must be finite"
        )
        source_shape = self.stokes.shape[-2:]
        target_shape = (target.n_ch, len(labels))
        expected = (int(np.prod(target_shape)), int(np.prod(source_shape)))
        if response.shape != expected:
            raise DataError(
                f"Expected response shape {expected} for these Stokes layouts, got {response.shape}."
            )
        shape = (*self.stokes.shape[:-2], *target_shape)
        values = (self.stokes.reshape(*shape[:-2], expected[1]) @ response.T).reshape(
            shape
        )
        cov, sigma = None, None
        if self.cov is not None:
            cov = response @ self.cov @ response.T
            diagonal = jnp.maximum(jnp.diagonal(cov, axis1=-2, axis2=-1), 0.0)
            sigma = jnp.sqrt(diagonal).reshape(*diagonal.shape[:-1], *target_shape)
        return replace(
            self,
            stokes=values,
            channels=target,
            components=labels,
            sigma=sigma,
            cov=cov,
            ledger=self.ledger.through(response, source_shape, target_shape),
        )

    def table(self) -> str:
        """Print the channel centres and Stokes values of one spectrum."""
        if self.stokes.ndim != 2:
            raise DataError("Choose one sightline to tabulate a batched spectrum.")
        freq_unit = self.channels.unit
        freq = convert(np.asarray(self.nu), "Hz", freq_unit)
        rows = [
            f"nu ({freq_unit}) | "
            + " | ".join(f"{s} ({self.unit})" for s in self.components)
        ]
        for nu, values in zip(freq, np.asarray(self.stokes)):
            rows.append(" | ".join(f"{v:.8g}" for v in (nu, *values)))
        return "\n".join(rows)
