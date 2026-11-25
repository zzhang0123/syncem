"""Observed geometry and retained coordinates of a free-moment fit."""

from dataclasses import replace

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

    observations: StokesData | None = None

    def for_parameters(self, count, whiten, *, pairs=None):
        """Extend a scalar sightline's likelihood to the available parameter rank."""
        if (
            count <= int(self.rank)
            or self.observations is None
            or self.directions is None
        ):
            return self
        from .rank import Rank, reduce_svd, solve_svd

        white = whiten(self.observations)
        svd = reduce_svd(white.design, complex_columns=pairs)
        limit = int(svd.cut.numerical_rank)
        k = min(count, limit)
        while k < limit and k > 0 and bool(svd.paired[k - 1]):
            k += 1
        if k <= int(self.rank):
            return self
        extended = solve_svd(svd, white.data, Rank(k))
        return replace(
            self,
            beta=extended.beta,
            sigma=extended.sigma,
            rank=extended.rank,
            directions=svd.directions,
        )

    @property
    def batch_shape(self):
        return self.beta.shape[:-1]

    def whitened_operator(self):
        """Return the observation and the linear operator on scaled lattice moments."""
        if self.basis is None or self.directions is None:
            raise DataError(
                "compressed likelihood needs unpenalized retained combinations "
                "from a fit with keep_directions=True"
            )
        used = jnp.arange(self.beta.shape[-1]) < self.rank[..., None]
        scale = jnp.where(used, self.sigma, 1.0)
        return (
            jnp.where(used, self.beta / scale, 0.0),
            jnp.where(used[..., None], self.directions / scale[..., None], 0.0),
            self.rank,
        )

    def whitened_moments(self, moments):
        """Retained observations and model coefficients in noise units."""
        target, matrix, count = self.whitened_operator()
        if moments.key != self.basis.lattice.key:
            raise DataError(
                "retained combinations need the original scaled moment coordinates"
            )
        return target, jnp.einsum("...kn,...n->...k", matrix, moments.values), count


class _Problem(eqx.Module):
    """The prebuilt prediction basis, its observed rows and the original data."""

    basis: SpectralBasis
    fit_basis: SpectralBasis
    data: StokesData
