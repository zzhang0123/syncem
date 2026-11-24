"""Bounded array work for prebuilt ridge and nonnegative estimators."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..values.errors import SizeError
from .penalties import solve_nonnegative, solve_ridge
from .rank import reduce_svd


@eqx.filter_jit
def solve_ridge_chunk(white, penalty, pairs, reduction):
    """Reuse the shared SVD or solve per pixel."""
    if reduction is None and white.design.ndim == 2:
        reduction = reduce_svd(white.design, complex_columns=pairs)
    if reduction is not None:
        return solve_ridge(reduction, white.data, penalty), reduction
    split = jax.vmap(lambda matrix: reduce_svd(matrix, complex_columns=pairs))(
        white.design
    )
    return (
        jax.vmap(lambda one, data: solve_ridge(one, data, penalty))(split, white.data),
        split,
    )


def ridge_chunk(data, n, itemsize, budget, penalty, *, held=False, shared=False):
    """Estimate SVD, retained arrays, UPRE grid and diagnostics workspace."""
    m = data.n_data
    k = min(m, n)
    fixed = m * n + m * k + k * n if shared else m * n
    svd = 0 if shared else m * n + m * k + k * n
    memory = m + n + 4 * k + svd
    if penalty.alpha is None:
        if shared:
            fixed += 3 * 321 * k
            memory += 2 * 321
        else:
            memory += 3 * 321 * k
    if held:
        memory += 5 * m * m + 2 * m * k
    fixed = math.ceil(1.5 * itemsize * fixed)
    per = math.ceil(1.5 * itemsize * memory)
    size = (budget - fixed) // per
    if size < 1:
        raise SizeError(
            f"chunk_bytes={budget:,} cannot hold a ridge pixel. Pass at least {fixed + per:,}."
        )
    return min(math.prod(data.batch_shape), size, 100_000)


@eqx.filter_jit
def ridge_project(rows, vectors, sigma, factors, cov):
    """Apply the filter to prediction rows and propagate observation noise."""
    projection = rows @ jnp.swapaxes(vectors, -1, -2)
    gain = projection * sigma[..., None, :]
    residual = rows - (projection * factors[..., None, :]) @ vectors
    norm = jnp.linalg.norm(rows, axis=-1)
    return (
        jnp.linalg.norm(gain, axis=-1),
        jnp.linalg.norm(residual, axis=-1) / jnp.where(norm > 0, norm, 1.0),
        gain @ jnp.swapaxes(gain, -1, -2) if cov else None,
    )


def host_pixels(white, solve):
    """Apply a host estimator to each whitened vector with no broadcast copy."""
    design, values = np.asarray(white.design), np.asarray(white.data)
    if values.ndim == 1:
        return solve(design, values)
    if design.ndim == 2:
        outputs = [solve(design, value) for value in values]
    else:
        outputs = [
            solve(matrix, value) for matrix, value in zip(design, values, strict=True)
        ]
    return jax.tree.map(lambda *leaf: jnp.asarray(np.stack(leaf)), *outputs)


def solve_point_chunk(white, points, penalty):
    def solve(matrix, values):
        return solve_nonnegative(matrix, values, points, penalty)

    return host_pixels(white, solve)
