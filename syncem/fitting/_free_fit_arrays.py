"""Batched array operations for the 0.8 free-moment fit (SPEC Sec. 6.3 and 6.4).

LABEL sec: linear fit. Inputs are whitened observations or rows in the scaled
lattice moments. Fit covariance includes only observation noise.

Not certified. The retained subspace depends on the design and rank policy.
"""

from __future__ import annotations

import math
from typing import NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..values.errors import SizeError
from .rank import reduce_svd, solve_svd


class Withheld(NamedTuple):
    """Marginal noise chi-square, count, predictive chi-square and peak whitened residual."""

    chi2: jax.Array
    n: jax.Array
    chi2_predictive: jax.Array
    max_residual: jax.Array


def shared_noise(data):
    """Whether all pixels use the same noise matrix and masks."""
    return all(
        array is None or not array.shape[:-tail]
        for array, tail in (
            (data.sigma, len(data._layout)),
            (data.cov, 2),
            (data.mask, 1),
        )
    )


def pixel_chunks(data, count):
    """Yield contiguous pixel groups, gathering just the requested heterogeneity."""
    shape = data.batch_shape
    for start in range(0, math.prod(shape), count):
        indices = np.unravel_index(
            np.arange(start, min(start + count, math.prod(shape))), shape
        )

        def take(array, tail):
            if array is None or not array.shape[:-tail]:
                return array
            lead = array.shape[:-tail]
            group = indices[len(shape) - len(lead) :]
            return array[
                tuple(
                    i if size != 1 else np.zeros_like(i) for i, size in zip(group, lead)
                )
            ]

        yield eqx.tree_at(
            lambda d: (d.stokes, d.sigma, d.cov, d.mask),
            data,
            (
                take(data.stokes, len(data._layout)),
                take(data.sigma, len(data._layout)),
                take(data.cov, 2),
                take(data.mask, 1),
            ),
            is_leaf=lambda x: x is None,
        )


def memory_chunk(
    data,
    n_columns,
    chunk_bytes,
    itemsize,
    *,
    held=False,
    n_out=0,
    cov=False,
    shared=False,
    extra_elements=0,
):
    """Bound the working set per pixel by the stated chunk size model."""
    m, n = data.n_data, n_columns
    k = min(m, n)
    factorisation = 0 if shared else m * n + m * k + k * n
    held_space = 5 * m * m + 2 * m * k if held else 0
    output = 2 * n_out * n + 2 * n_out * k + (n_out * n_out if cov else 0)
    per_pixel = math.ceil(
        1.5 * itemsize * (factorisation + held_space + output + m + n + extra_elements)
    )
    count = chunk_bytes // per_pixel
    if count < 1:
        raise SizeError(
            f"chunk_bytes={chunk_bytes} cannot hold one fit or prediction pixel. Pass at least {per_pixel}."
        )
    if shared:
        count = min(100_000, count)
    return min(math.prod(data.batch_shape), count)


def on_pixels(function, data, pixels):
    """Concatenate compact chunk outputs and restore the original pixel axes."""
    if not data.batch_shape:
        return function(data, 0)
    chunks = []
    for index, part in enumerate(pixel_chunks(data, pixels)):
        output = function(part, index * pixels)
        # Complete the chunk before launching another workspace.
        if not isinstance(part.stokes, jax.core.Tracer):
            jax.block_until_ready(output)
        chunks.append(output)
    return jax.tree.map(
        lambda *a: jnp.concatenate(a, axis=0).reshape(
            *data.batch_shape, *a[0].shape[1:]
        ),
        *chunks,
    )


@eqx.filter_jit
def solve_chunk(white, rank, pairs, reduction=None):
    """Compute independent SVDs or reuse a shared one across the pixel chunk."""
    if reduction is None and white.design.ndim == 2:
        reduction = reduce_svd(white.design, complex_columns=pairs)
    if reduction is not None:
        return solve_svd(reduction, white.data, rank), reduction
    reduced = jax.vmap(lambda matrix: reduce_svd(matrix, complex_columns=pairs))(
        white.design
    )
    return jax.vmap(lambda r, d: solve_svd(r, d, rank))(reduced, white.data), reduced


@eqx.filter_jit
def factor_chunk(design, pairs):
    """Right directions of a shared or per-pixel whitened design."""
    if design.ndim == 2:
        return reduce_svd(design, complex_columns=pairs).directions
    return jax.vmap(
        lambda matrix: reduce_svd(matrix, complex_columns=pairs).directions
    )(design)


def _transpose(array):
    return jnp.swapaxes(array, -1, -2)


def _marginal_factor(cov, flags):
    block = jnp.where(flags[..., :, None] & flags[..., None, :], cov, 0.0)
    return jnp.linalg.cholesky(block + (~flags[..., :, None]) * jnp.eye(cov.shape[-1]))


def _cross_covariance(data):
    fitted, held = data.mask & ~data.withheld, data.mask & data.withheld
    cross = jnp.where(fitted[..., :, None] & held[..., None, :], data.cov, 0.0)
    chol_f = _marginal_factor(data.cov, fitted)
    chol_h = _marginal_factor(data.cov, held)
    left = jax.scipy.linalg.solve_triangular(chol_f, cross, lower=True)
    return _transpose(
        jax.scipy.linalg.solve_triangular(chol_h, _transpose(left), lower=True)
    )


@eqx.filter_jit
def withheld_statistics(data, white, solution, reduction):
    """Validate the fitted predictor on rows excluded from its estimator."""
    residual = (white.design @ solution.moments[..., None])[..., 0] - white.data
    prediction = (white.design @ _transpose(reduction.directions)) * solution.sigma[
        ..., None, :
    ]
    covariance = prediction @ _transpose(prediction) + jnp.eye(white.data.shape[-1])
    if data.cov is not None:
        estimator = prediction @ _transpose(reduction.left)
        cross = estimator @ _cross_covariance(data)
        covariance = covariance - cross - _transpose(cross)
    chol = jnp.linalg.cholesky(covariance)
    chol = eqx.error_if(
        chol,
        jnp.any(~jnp.isfinite(chol)),
        "withheld predictive covariance must be positive definite",
    )
    chi = jax.scipy.linalg.solve_triangular(chol, residual[..., None], lower=True)[
        ..., 0
    ]
    return Withheld(
        jnp.sum(residual**2, axis=-1),
        white.count,
        jnp.sum(chi**2, axis=-1),
        jnp.max(jnp.abs(residual), axis=-1),
    )


@eqx.filter_jit
def project(rows, directions, solution_sigma, with_covariance):
    """Noise and missing row-norm fraction for a set of linear functionals."""
    coordinates = rows @ _transpose(directions)
    noise = coordinates * solution_sigma[..., None, :]
    retained = jnp.where(solution_sigma[..., None, :] > 0, coordinates, 0.0)
    remainder = rows - retained @ directions
    row_norm = jnp.linalg.norm(rows, axis=-1)
    missing = jnp.linalg.norm(remainder, axis=-1) / jnp.where(
        row_norm > 0, row_norm, 1.0
    )
    uncertainty = jnp.linalg.norm(noise, axis=-1)
    covariance = noise @ _transpose(noise) if with_covariance else None
    return uncertainty, missing, covariance
