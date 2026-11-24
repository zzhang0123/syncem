"""Penalised fits on prebuilt bases (SPEC Sec. 6.2 and 6.3).

Both estimators fit the lifted design in the scaled lattice moments.
Ridge propagates observation noise through the filter. The nonnegative
result gives the fitted point-mass measure and noise-only profiles.
Neither fit bounds the representation error of the source.

LABEL [extension].

The scaled moment vector has shape (..., n_row) in the lattice units.
Ridge noise errors and active-set diagnostics have no certified source error
envelope.
"""

from __future__ import annotations

import hashlib
import math
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..engine.basis import SpectralBasis
from ..lattice.moments import Moments
from ..population.declaration import parents
from ..values.errors import DataError
from ._conditioned_points import conditioned_point_mass_map, conditioned_result
from ._free_fit_arrays import (
    Withheld,
    on_pixels,
    pixel_chunks,
    shared_noise,
    withheld_statistics,
)
from ._penalized_arrays import ridge_chunk, solve_point_chunk, solve_ridge_chunk
from ._penalized_result import NonnegativeFit, PenalizedFit, _fit_ledger
from ._point_mass_fit import host_point_chunk, point_mass_map
from .data import StokesData
from .free_fit import (
    RetainedCombinations,
    _Problem,
    _design,
    _fit_basis,
    _pairs,
    _white,
)
from .penalties import Penalty
from .rank import warn_rank_margin

__all__ = ["NonnegativeFit", "PenalizedFit", "fit_nonnegative", "fit_penalized"]


def _setup(basis, data, penalty, budget):
    if not isinstance(basis, SpectralBasis) or not isinstance(data, StokesData):
        raise TypeError("fit needs a SpectralBasis and StokesData.")
    if not isinstance(penalty, Penalty):
        raise TypeError("penalty must be Penalty() or Penalty(alpha).")
    if isinstance(budget, bool) or not isinstance(budget, Integral) or budget <= 0:
        raise ValueError("chunk_bytes must be a positive integer.")
    if math.prod(data.batch_shape) == 0:
        raise DataError("there are no sightlines to fit.")
    if basis.lattice.key is None:
        raise DataError("fit needs a prebuilt basis with a concrete lattice key.")
    fitted = _fit_basis(basis, data.channels)
    return _Problem(basis, fitted, data), _design(fitted)


def _result_moments(basis, values, origin, *, retained=None):
    masked = tuple(map(int, np.flatnonzero(basis.columns.col_of_row < 0)))
    return Moments(
        basis.lattice, values, origin=origin, masked=masked, retained=retained
    )


def _goodness(count, effective, chi2):
    dof = count - effective
    p = jax.scipy.special.gammaincc(jnp.where(dof > 0, dof, 1) / 2, chi2 / 2)
    return dof, jnp.where(dof > 0, p, jnp.nan)


def fit_penalized(
    basis, data, penalty=Penalty(), *, keep_directions=False, chunk_bytes=2**31
):
    """Fit ridge moments and retain enough statistics for later covariance."""
    problem, design = _setup(basis, data, penalty, chunk_bytes)
    if not isinstance(keep_directions, bool):
        raise ValueError("keep_directions must be a bool.")
    fit_basis = problem.fit_basis
    shared = shared_noise(data)
    held = bool(np.any(np.asarray(data.withheld)))
    pairs = _pairs(fit_basis)
    size = ridge_chunk(
        data,
        design.shape[-1],
        design.dtype.itemsize,
        chunk_bytes,
        penalty,
        held=held,
        shared=shared,
    )
    reduction = None
    if shared:
        sample = next(pixel_chunks(data, 1)) if data.batch_shape else data
        _, reduction = solve_ridge_chunk(
            _white(sample, fit_basis, design), penalty, pairs, None
        )

    def run(part, _):
        white = _white(part, fit_basis, design)
        solution, svd = solve_ridge_chunk(white, penalty, pairs, reduction)
        withheld = (
            withheld_statistics(
                part, _white(part, fit_basis, design, held=True), solution, svd
            )
            if held
            else None
        )
        stored = svd.directions if keep_directions and not shared else None
        rank = jnp.broadcast_to(svd.cut.numerical_rank, part.batch_shape)
        return (
            solution,
            white.count,
            rank,
            None if shared else svd.cut,
            stored,
            withheld,
        )

    solution, count, rank, cut, directions, held_stats = on_pixels(run, data, size)
    if shared:
        directions, cut = reduction.directions, reduction.cut
    warn_rank_margin(cut)
    key = hashlib.sha256(
        f"{basis.lattice.key}:{basis.columns.key}".encode()
    ).hexdigest()
    retained = RetainedCombinations(
        solution.beta, solution.sigma, directions, rank, key
    )
    moments = _result_moments(basis, solution.moments, "penalised", retained=key)
    dof, p_value = _goodness(count, solution.effective, solution.chi2)
    return PenalizedFit(
        problem,
        moments,
        solution.alpha,
        solution.effective,
        solution.chi2,
        dof,
        p_value,
        held_stats,
        _fit_ledger(fit_basis, data=data, moments=moments, chunk_bytes=int(chunk_bytes)),
        penalty,
        int(chunk_bytes),
        size,
        retained,
        solution.factors,
        cut,
    )


def fit_nonnegative(
    basis, data, marker, penalty=Penalty(), *, chunk_bytes=2**31, conditional=None
):
    """Fit a joint nonnegative measure on every cell axis on the prebuilt basis."""
    problem, design = _setup(basis, data, penalty, chunk_bytes)
    if conditional is not None:
        child, declaration = conditional
        points = conditioned_point_mass_map(
            basis, marker, child, declaration, chunk_bytes
        )
        result = _point_fit(problem, design, points, penalty, chunk_bytes)
        return conditioned_result(result, child, dependencies=parents(declaration))
    points = point_mass_map(basis, marker, chunk_bytes)
    return _point_fit(problem, design, points, penalty, chunk_bytes)


def _point_fit(problem, design, points, penalty, chunk_bytes):
    basis, fit_basis, data = problem.basis, problem.fit_basis, problem.data
    size = host_point_chunk(chunk_bytes, data, points.moments)
    hold = bool(np.any(np.asarray(data.withheld)))

    def run(part, _):
        white = _white(part, fit_basis, design)
        fit = solve_point_chunk(white, points.moments, penalty)
        held = None
        if hold:
            held_white = _white(part, fit_basis, design, held=True)
            residual = (held_white.design @ jnp.asarray(fit.moments)[..., None])[
                ..., 0
            ] - held_white.data
            held = Withheld(
                jnp.sum(residual**2, axis=-1),
                held_white.count,
                None,
                jnp.max(jnp.abs(residual), axis=-1),
            )
        return fit, white.count, held

    solution, count, held = on_pixels(run, data, size)
    negative = 0
    if "I" in fit_basis.record.stokes:
        i = fit_basis.record.stokes.index("I")
        negative = int(
            np.count_nonzero(
                np.any((np.asarray(design[:, i, :]) @ points.moments) < 0, axis=0)
            )
        )
    moments = _result_moments(
        basis, jnp.asarray(solution.moments), "nonnegative_penalised"
    )
    dof, p_value = _goodness(count, solution.effective, solution.chi2)
    return NonnegativeFit(
        problem,
        moments,
        jnp.asarray(solution.alpha),
        jnp.asarray(solution.effective),
        jnp.asarray(solution.chi2),
        dof,
        p_value,
        held,
        _fit_ledger(fit_basis, data=data, point_mass=True),
        penalty,
        int(chunk_bytes),
        size,
        jnp.asarray(solution.weights),
        points,
        negative,
    )
