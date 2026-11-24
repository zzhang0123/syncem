"""Bias envelopes in the retained estimator coordinates (SPEC Sec. 7.2)."""

from __future__ import annotations

from dataclasses import replace
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..records.ledger import ErrorTerm
from ..values.errors import DataError
from ..values.spectrum import channel_conversion
from ._free_fit_arrays import _marginal_factor

__all__ = ["ResponseBias", "bias_elements", "bias_terms", "response_bias"]


class ResponseBias(NamedTuple):
    """Optional envelopes with a trailing, zero-padded retained axis."""

    response: jax.Array | None
    discrepancy: jax.Array | None


def _has_discrepancy(data):
    term = data.discrepancy
    if term is None or term.value is None:
        return False
    if term.space != "stokes" or term.relative_to is not None:
        raise DataError(
            "discrepancy needs an absolute envelope in the observation unit"
        )
    return True


def bias_elements(data, n_columns, *, shared=False):
    """Conservative diagnostics scratch in elements, fixed and per pixel."""
    response = data.response_uncertainty is not None
    discrepancy = _has_discrepancy(data)
    if not (response or discrepancy):
        return 0, 0
    m = data.n_data
    k = min(m, n_columns)
    n_model = data.channels.n_ch * len(data.components)
    map_work = (4 * m * m if data.cov is not None else 0) + 2 * m * k
    per_pixel = (3 * m + 2 * k) * (response + discrepancy)
    # The Stokes contraction precedes component selection and conversion.
    per_pixel += (4 * data.channels.n_ch + 2 * n_model) if response else 0
    model_discrepancy = discrepancy and data.discrepancy.value.shape != (m,)
    fixed = m * n_model if model_discrepancy and data.response is not None else 0
    return (fixed + map_work, per_pixel) if shared else (fixed, per_pixel + map_work)


def _source_stokes(data, design, components, moments, unit):
    indices = jnp.asarray([components.index(axis) for axis in data.components])
    values = jnp.einsum("csk,...k->...cs", design, moments)[..., indices]
    scale = channel_conversion(data.channels, unit, data.unit)
    return (values * scale[:, None]).reshape(*moments.shape[:-1], -1)


def _model_discrepancy(data, magnitude):
    if magnitude.shape == (data.n_data,):
        return magnitude
    shape = (data.channels.n_ch, len(data.components))
    envelope = jnp.broadcast_to(magnitude, shape).reshape(-1)
    if data.response is None:
        return envelope
    output = jnp.abs(data.response) @ envelope
    if data.response_uncertainty is not None:
        output = output + data.response_uncertainty @ envelope
    return output


def _project_envelopes(data, envelopes, solution, reduction):
    # L is the Cholesky factor of the selected marginal noise covariance.
    # The transpose solve computes L^-T U. Taking abs before that solve
    # would lose the cancellations between correlated observations.
    active = data.mask & ~data.withheld
    use = jnp.where(active[..., :, None], envelopes, 0.0)
    left = reduction.left
    if data.sigma is not None:
        noise = data.sigma.reshape(*data.sigma.shape[: -len(data._layout)], data.n_data)
        use = use / jnp.where(active, noise, 1.0)[..., :, None]
    else:
        lower = _marginal_factor(data.cov, active)
        left = jax.scipy.linalg.solve_triangular(lower, left, lower=True, trans="T")
    projected = jnp.einsum("...mr,...me->...re", jnp.abs(left), use)
    # The estimator uses only kept directions. The fixed-width representation
    # leaves discarded entries exactly zero and permits chunk concatenation.
    return jnp.where(
        solution.sigma[..., None] > 0, projected * solution.sigma[..., None], 0.0
    )


@eqx.filter_jit
def response_bias(data, design, components, solution, reduction, *, unit="cgs"):
    """Return the responses of the retained estimator for one fit chunk.

    The unwhitened design is in channel, Stokes and coefficient order.
    Convert the fitted channel Stokes to observation units before applying
    the error in the response. The arrays in the result share the shape
    of the corresponding retained combinations.
    """
    response = data.response_uncertainty is not None
    discrepancy = _has_discrepancy(data)
    if not (response or discrepancy):
        return ResponseBias(None, None)
    envelopes = []
    if response:
        source = _source_stokes(data, design, components, solution.moments, unit)
        propagated = jnp.einsum(
            "ij,...j->...i", data.response_uncertainty, jnp.abs(source)
        )
        envelopes.append(propagated)
    if discrepancy:
        observed = _model_discrepancy(data, data.discrepancy.value)
        envelopes.append(
            jnp.broadcast_to(observed, (*solution.sigma.shape[:-1], data.n_data))
        )
    biased = _project_envelopes(
        data, jnp.stack(envelopes, axis=-1), solution, reduction
    )
    return ResponseBias(
        biased[..., 0] if response else None,
        biased[..., int(response)] if discrepancy else None,
    )


def bias_terms(data, bias):
    """Construct the fit's retained-coordinate ledger terms."""
    if data.response is None:
        response = ErrorTerm(
            None,
            "not_applicable",
            "no observing response was supplied",
            name="response",
            space="retained",
        )
    elif data.response_uncertainty is None:
        response = ErrorTerm(
            None,
            "unbounded",
            "the observing response has no declared uncertainty",
            name="response",
            space="retained",
        )
    elif np.any(np.asarray(data.response_uncertainty)):
        response = ErrorTerm(
            bias.response,
            "estimate",
            "Retained-combination bias from the declared response uncertainty and fitted Stokes.",
            name="response",
            declared=True,
            space="retained",
        )
    else:
        response = ErrorTerm(
            jnp.zeros_like(bias.response),
            "bound",
            "the response uncertainty was declared zero",
            name="response",
            declared=True,
            space="retained",
        )

    declared = data.discrepancy
    if declared is None:
        discrepancy = ErrorTerm(
            None,
            "unbounded",
            "no discrepancy envelope was declared",
            name="discrepancy",
            space="retained",
        )
    elif declared.value is None:
        discrepancy = replace(declared, space="retained")
    else:
        note = "Retained-combination bias from the declared model discrepancy."
        if (
            declared.value.shape != (data.n_data,)
            and data.response is not None
            and data.response_uncertainty is None
        ):
            note += " This term uses the nominal response."
        if declared.note:
            note += f" {declared.note}"
        discrepancy = ErrorTerm(
            bias.discrepancy,
            declared.kind,
            note,
            manuscript_term=declared.manuscript_term,
            name="discrepancy",
            declared=True,
            space="retained",
        )
    return response, discrepancy
