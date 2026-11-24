"""Unit-aware channel observations and whitening (SPEC Sec. 3.7 and 6.4).

LABEL sec: linear fit. The response acts after channel unit conversion on
flattened channel-major Stokes. Excluded data rows are zero after whitening
and do not count toward the degrees of freedom. Keeping their shape permits
JIT and per-pixel masks.

Not certified. Noise and response inputs describe the caller's observing
model. Their accuracy is not inferred from the data.
"""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..base.channels import Channels
from ..model.errors import ErrorTerm as ModelErrorTerm
from ..records.ledger import ErrorTerm
from ..units import UnitError
from ..values.errors import DataError
from ..values.spectrum import _array, _check_unit, _components, channel_conversion

__all__ = ["StokesData", "Whitened"]
_AXES = ("I", "Q", "U", "V")


class Whitened(NamedTuple):
    """Data vector, design matrix and number of rows used, with optional pixel axes."""

    data: jax.Array
    design: jax.Array
    count: jax.Array


def _broadcast(value, shape, label):
    try:
        return jnp.broadcast_to(value, shape)
    except ValueError as exc:
        raise DataError(
            f"{label} shape {value.shape} cannot broadcast to {shape}."
        ) from exc


def _checked(value, predicate, message):
    if isinstance(predicate, jax.core.Tracer):
        return eqx.error_if(value, predicate, message)
    if bool(predicate):
        raise DataError(message)
    return value


def _rows(rows, size):
    if isinstance(rows, tuple):
        rows = list(rows)
    try:
        result = np.atleast_1d(np.arange(size)[rows])
    except (IndexError, TypeError) as exc:
        raise DataError(f"invalid selection for {size} rows.") from exc
    if result.ndim != 1 or result.size == 0:
        raise DataError("row selection must be one-dimensional and nonempty.")
    return result


def _from_stokes(stokes, fields, components):
    named = tuple((k, v) for k, v in zip(_AXES, fields) if v is not None)
    if stokes is None:
        if not named:
            raise DataError("pass stokes or at least one of I, Q, U, V.")
        if components is not None:
            raise DataError("named Stokes inputs already specify their labels.")
        components, values = zip(*named)
        stokes = jnp.stack(jnp.broadcast_arrays(*map(jnp.asarray, values)), axis=-1)
    elif named:
        raise DataError("pass stokes or named Stokes inputs.")
    return _array(stokes, "stokes"), components


def _response(response, uncertainty):
    if response is None:
        if uncertainty is not None:
            raise DataError("response_uncertainty requires a response.")
        return None, None
    response = _array(response, "response")
    if response.ndim != 2 or min(response.shape) < 1:
        raise DataError("response must be a nonempty data-by-model matrix.")
    response = _checked(
        response, jnp.any(~jnp.isfinite(response)), "response must be finite."
    )
    if uncertainty is not None:
        uncertainty = _array(uncertainty, "response_uncertainty")
        if uncertainty.shape != response.shape:
            raise DataError("response uncertainty must have the response matrix shape.")
        bad = jnp.any(~jnp.isfinite(uncertainty) | (uncertainty < 0))
        uncertainty = _checked(
            uncertainty, bad, "response uncertainties must be finite and nonnegative."
        )
    return response, uncertainty


def _geometry(channels, stokes, response, components):
    n_ch = channels.n_ch
    if response is None:
        if stokes.ndim < 2 or stokes.shape[-2] != n_ch:
            raise DataError(f"stokes requires shape (..., {n_ch}, n_stokes).")
        return _components(components, stokes.shape[-1]), stokes.shape[-2:]

    n_data, n_model = response.shape
    n_s, excess = divmod(n_model, n_ch)
    if excess:
        raise DataError("response columns must contain complete model channels.")
    labels = _components(components, n_s)
    if stokes.shape[-2:] == (n_ch, n_s) and n_data == n_model:
        return labels, stokes.shape[-2:]
    if stokes.ndim == 0 or stokes.shape[-1] != n_data:
        raise DataError(
            f"stokes in the response row space must end in {n_data} values."
        )
    return labels, (n_data,)


def _observed(value, layout, batch, label):
    """Broadcast rows while retaining any shared pixel axes."""
    value = jnp.asarray(value)
    n = int(np.prod(layout))
    if len(layout) == 2 and value.shape[-1:] == (n,) and value.shape[-2:] != layout:
        layout = (n,)
    lead = value.shape[: -len(layout)]
    _broadcast(value, (*batch, *layout), label)
    return _broadcast(value, (*lead, *layout), label).reshape(*lead, n)


def _discrepancy(term, model_shape, n_data):
    if term is None:
        return None
    if isinstance(term, ModelErrorTerm):
        term = ErrorTerm.from_model("discrepancy", term)
    if not isinstance(term, ErrorTerm):
        raise DataError("discrepancy must be an ErrorTerm.")
    if term.value is None:
        return replace(term, name="discrepancy")
    if term.value.shape != (n_data,):
        _broadcast(term.value, model_shape, "discrepancy")
    return replace(term, name="discrepancy", declared=True, signed=False)


class StokesData(eqx.Module):
    """Channel observations and their noise, or flattened rows of a response."""

    stokes: jax.Array
    channels: Channels
    sigma: jax.Array | None
    cov: jax.Array | None
    mask: jax.Array
    withheld: jax.Array
    response: jax.Array | None
    response_uncertainty: jax.Array | None
    discrepancy: ErrorTerm | None
    unit: str = eqx.field(static=True)
    components: tuple[str, ...] = eqx.field(static=True)
    _layout: tuple[int, ...] = eqx.field(static=True)

    def __init__(
        self,
        channels,
        *,
        unit=None,
        I=None,
        Q=None,
        U=None,
        V=None,
        stokes=None,
        sigma=None,
        cov=None,
        mask=None,
        response=None,
        response_uncertainty=None,
        discrepancy=None,
        components=None,
    ):
        if unit is None:
            raise UnitError(
                'StokesData has no unit. Pass unit="Jy/sr", "MJy/sr", "K_RJ", "K_CMB", or "cgs".'
            )
        _check_unit(unit)
        if (sigma is None) == (cov is None):
            raise DataError(
                "provide either standard deviations via sigma or a covariance via cov."
            )
        stokes, components = _from_stokes(stokes, (I, Q, U, V), components)
        self.response, self.response_uncertainty = _response(
            response, response_uncertainty
        )
        self.components, self._layout = _geometry(
            channels, stokes, self.response, components
        )
        self.stokes, self.channels, self.unit = stokes, channels, unit
        n = self.n_data
        if mask is None:
            flags = jnp.ones((n,), dtype=bool)
        else:
            flags = _observed(mask, self._layout, self.batch_shape, "mask")
            flags = _checked(
                flags,
                jnp.any((flags != 0) & (flags != 1)),
                "mask entries must be True or False.",
            )
        self.mask = flags.astype(bool)
        self.withheld = jnp.zeros((n,), dtype=bool)
        self.discrepancy = _discrepancy(
            discrepancy, (channels.n_ch, len(self.components)), n
        )
        self.sigma, self.cov = None, None
        if sigma is not None:
            raw = _observed(
                _array(sigma, "sigma"), self._layout, self.batch_shape, "sigma"
            )
            self.sigma = raw.reshape(*raw.shape[:-1], *self._layout)
        else:
            cov = _array(cov, "covariance")
            if cov.shape[-2:] != (n, n):
                raise DataError(f"cov must end in a square matrix of shape ({n}, {n}).")
            _broadcast(cov, (*self.batch_shape, n, n), "covariance")
            self.cov = cov

    @property
    def n_data(self) -> int:
        """Rows per sightline before masking."""
        return int(np.prod(self._layout))

    @property
    def batch_shape(self) -> tuple[int, ...]:
        """Leading sightline axes."""
        return self.stokes.shape[: -len(self._layout)]

    def data_vector(self) -> jax.Array:
        """Observations with the final row axis flattened."""
        return self.stokes.reshape(*self.batch_shape, self.n_data)

    @classmethod
    def simulate(cls, spectrum, *, noise, key) -> StokesData:
        """Draw Gaussian noise with sigma equal to the supplied fraction of channel I."""
        if spectrum.I is None:
            raise DataError(
                "simulation needs the intensity component to set fractional noise."
            )
        sigma = _array(noise, "noise") * spectrum.I
        bad = jnp.any((sigma <= 0) | ~jnp.isfinite(sigma))
        sigma = _checked(
            sigma, bad, "simulated standard deviations must be positive and finite."
        )
        values = spectrum.stokes + sigma[..., None] * jax.random.normal(
            key, spectrum.stokes.shape, dtype=spectrum.stokes.dtype
        )
        return cls(
            spectrum.channels,
            stokes=values,
            components=spectrum.components,
            sigma=sigma[..., None],
            unit=spectrum.unit,
        )

    def withhold(self, rows) -> StokesData:
        """Keep selected channels for validation. A flat layout selects response rows."""
        selected = _rows(rows, self._layout[0])
        flag = jnp.zeros(self._layout, dtype=bool).at[selected].set(True).reshape(-1)
        return eqx.tree_at(lambda data: data.withheld, self, self.withheld | flag)

    def _model_matrix(self, columns, components, unit):
        labels = self.components if components is None else tuple(components)
        missing = set(self.components) - set(labels)
        if missing:
            raise DataError(f"the design lacks Stokes components {sorted(missing)}.")
        n_ch = self.channels.n_ch
        columns = _array(columns, "design")
        if columns.ndim == 2 and columns.shape[0] == n_ch * len(labels):
            columns = columns.reshape(n_ch, len(labels), columns.shape[-1])
        if columns.ndim != 3 or columns.shape[:2] != (n_ch, len(labels)):
            raise DataError("design must have channel, Stokes and coefficient axes.")
        which = jnp.asarray([labels.index(label) for label in self.components])
        row_scale = channel_conversion(self.channels, unit, self.unit)
        scaled = columns[:, which, :] * row_scale[:, None, None]
        matrix = scaled.reshape(n_ch * len(self.components), columns.shape[-1])
        return matrix if self.response is None else self.response @ matrix

    def whitened(
        self, columns, *, components=None, unit="cgs", withheld=False
    ) -> Whitened:
        """Whiten data and design with the marginal covariance of selected rows."""
        matrix = self._model_matrix(columns, components, unit)
        keep = self.mask & (self.withheld if withheld else ~self.withheld)
        return self._whiten(matrix, keep)

    def whitened_spectrum(self, spectrum, *, selection=None) -> Whitened:
        """Whiten a supplied model spectrum and the matching observations."""
        if self.channels is not spectrum.channels:
            from ..engine._regime_columns import regime_channel_keys

            message = "the spectrum and observations must share response bands"
            if self.channels.row_keys() != spectrum.channels.row_keys():
                raise DataError(message)
            observed = regime_channel_keys(self.channels, continuum=False)
            model = regime_channel_keys(spectrum.channels, continuum=False)
            if None in observed or observed != model:
                raise DataError(message)
        missing = set(self.components) - set(spectrum.components)
        if missing:
            raise DataError(f"the spectrum lacks Stokes components {sorted(missing)}")
        if spectrum.stokes.shape[:-2] != self.batch_shape:
            raise DataError("the spectrum and observations must share sightlines")
        which = jnp.asarray(
            [spectrum.components.index(axis) for axis in self.components]
        )
        stokes = spectrum.to(self.unit).stokes[..., which]
        model = stokes.reshape(*self.batch_shape, -1, 1)
        if self.response is not None:
            model = self.response @ model
        keep = self.mask if selection is None else self.mask & selection
        return self._whiten(model, keep)

    def _whiten(self, matrix, keep):
        values = self.data_vector()
        values = _checked(
            values,
            jnp.any(keep & ~jnp.isfinite(values)),
            "selected observations must be finite.",
        )
        white_data = jnp.where(keep, values, 0.0)
        white_design = jnp.where(keep[..., :, None], matrix, 0.0)
        if self.sigma is not None:
            noise = self.sigma.reshape(
                *self.sigma.shape[: -len(self._layout)], self.n_data
            )
            bad = jnp.any(keep & ((noise <= 0) | ~jnp.isfinite(noise)))
            noise = _checked(
                noise, bad, "selected standard deviations must be finite and positive."
            )
            scale = jnp.where(keep, noise, 1.0)
            white_data, white_design = (
                white_data / scale,
                white_design / scale[..., :, None],
            )
        else:
            block = jnp.where(keep[..., :, None] & keep[..., None, :], self.cov, 0.0)
            bt = jnp.swapaxes(block, -1, -2)
            size = jnp.max(jnp.abs(block), axis=(-2, -1))
            bad = jnp.any(jnp.max(jnp.abs(block - bt), axis=(-2, -1)) > 1e-12 * size)
            block = _checked(block, bad, "selected noise covariance must be symmetric.")
            full = block + (~keep[..., :, None]) * jnp.eye(self.n_data)
            lower = jnp.linalg.cholesky(full)
            lower = _checked(
                lower,
                jnp.any(~jnp.isfinite(lower)),
                "selected noise covariance must be positive definite.",
            )
            white_data = jax.scipy.linalg.solve_triangular(
                lower, white_data[..., None], lower=True
            )[..., 0]
            white_design = jax.scipy.linalg.solve_triangular(
                lower, white_design, lower=True
            )
        count = jnp.broadcast_to(jnp.sum(keep, axis=-1), self.batch_shape)
        return Whitened(white_data, white_design, count)

    def __getitem__(self, rows) -> StokesData:
        """Select channels and the corresponding observation geometry."""
        if len(self._layout) != 2:
            raise DataError(
                "channel selection needs Stokes-shaped observations. Use withhold for response rows."
            )
        selected = _rows(rows, self.channels.n_ch)
        flat = np.arange(self.n_data).reshape(self._layout)[selected].reshape(-1)
        response, uncertainty = self.response, self.response_uncertainty
        if response is not None:
            removed = np.setdiff1d(np.arange(self.n_data), flat)
            depends = jnp.any(response[flat[:, None], removed] != 0)
            if uncertainty is not None:
                depends |= jnp.any(uncertainty[flat[:, None], removed] != 0)
                uncertainty = uncertainty[flat[:, None], flat]
            response = _checked(
                response,
                depends,
                "selected observations depend on excluded model channels via the response. Use withhold.",
            )[flat[:, None], flat]
        discrepancy = self.discrepancy
        if discrepancy is not None and discrepancy.value is not None:
            value = discrepancy.value
            value = (
                value[flat]
                if value.shape == (self.n_data,)
                else jnp.broadcast_to(value, self._layout)[selected]
            )
            discrepancy = replace(discrepancy, value=value)
        out = StokesData(
            self.channels[selected],
            stokes=self.stokes[..., selected, :],
            unit=self.unit,
            components=self.components,
            mask=self.mask[..., flat],
            sigma=None if self.sigma is None else self.sigma[..., selected, :],
            cov=None if self.cov is None else self.cov[..., flat[:, None], flat],
            response=response,
            response_uncertainty=uncertainty,
            discrepancy=discrepancy,
        )
        return eqx.tree_at(lambda data: data.withheld, out, self.withheld[flat])

    def __add__(self, other) -> StokesData:
        """Join independent observations of the same sightlines."""
        from ._data_join import join_data

        return join_data(self, other)
