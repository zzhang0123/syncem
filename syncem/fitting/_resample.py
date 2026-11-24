"""Gaussian noise refits around the fitted observation vector."""

from numbers import Integral

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..values.errors import SizeError
from ._free_fit_arrays import _marginal_factor


def redraw(data, mean, n, key, chunk_bytes):
    """Preserve covariance and selection while prepending the realization axis."""
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive number of realizations")
    need = 4 * int(n) * int(np.prod(mean.shape)) * mean.dtype.itemsize
    if need >= chunk_bytes:
        raise SizeError(
            f"{n} realizations need about {need:,} bytes of input workspace. "
            f"Reduce n or increase chunk_bytes={chunk_bytes:,}."
        )
    z = jax.random.normal(key, (n, *mean.shape), dtype=mean.dtype)
    z = jnp.where(data.mask, z, 0.0)
    if data.sigma is not None:
        flat_sigma = data.sigma.reshape(
            *data.sigma.shape[: -len(data._layout)], data.n_data
        )
        perturbation = z * jnp.where(data.mask, flat_sigma, 0.0)
    else:
        lower = _marginal_factor(data.cov, data.mask)
        perturbation = jnp.einsum("...ij,n...j->n...i", lower, z)
    measured = (mean[None] + perturbation).reshape(n, *data.stokes.shape)
    return eqx.tree_at(lambda obj: obj.stokes, data, measured)
