"""Observed geometry and retained coordinates of a free-moment fit."""

import equinox as eqx
import jax
import jax.numpy as jnp

from ..engine.basis import SpectralBasis
from ..values.errors import DataError
from .data import StokesData


class RetainedCombinations(eqx.Module):
    """Noise-only combinations, zero-padded after rank on each sightline."""

    beta: jax.Array
    sigma: jax.Array
    directions: jax.Array | None
    rank: jax.Array
    key: str = eqx.field(static=True)
    basis: SpectralBasis | None = None
    unit: str = eqx.field(static=True, default="cgs")

    @property
    def batch_shape(self):
        return self.beta.shape[:-1]

    def whitened_moments(self, moments):
        """Retained observations and model coefficients in noise units."""
        if self.basis is None:
            raise DataError(
                "compressed likelihood needs unpenalized retained combinations"
            )
        if self.directions is None:
            raise DataError(
                "fit with keep_directions=True to use the compressed likelihood"
            )
        if moments.key != self.basis.lattice.key:
            raise DataError(
                "retained combinations need the original scaled moment coordinates"
            )
        used = jnp.arange(self.beta.shape[-1]) < self.rank[..., None]
        scale = jnp.where(used, self.sigma, 1.0)
        model = jnp.einsum("...kn,...n->...k", self.directions, moments.values)
        return (
            jnp.where(used, self.beta / scale, 0.0),
            jnp.where(used, model / scale, 0.0),
            self.rank,
        )


class _Problem(eqx.Module):
    """The prebuilt prediction basis, its observed rows and the original data."""

    basis: SpectralBasis
    fit_basis: SpectralBasis
    data: StokesData
