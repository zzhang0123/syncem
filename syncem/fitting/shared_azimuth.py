"""Fits with one shared second azimuth moment (SPEC Sec. 6.1).

The scaled lattice moments are linear at fixed c2. The fit profiles their
retained directions over the physical disk |c2| <= 1.

LABEL [extension].

The fitted azimuth factor is one dimensionless complex scalar. Observation
arrays carry the declared Stokes units and have shape (n_ch, n_stokes). A
finite factor search has no certified global minimum.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..engine._keys import design_key
from ..engine.basis import SpectralBasis
from ..values.errors import DataError, RoleError
from .data import StokesData
from .free_fit import Fit, _design, _fit_basis, _white, fit_free
from .rank import PAIR_RTOL, RANK_RTOL, Rank, UPRE

__all__ = ["SharedAzimuthFit", "fit_shared_azimuth"]

_RADII = (0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)
_PHASES = 24
_FIXED_STARTS = 3
_MAXITER = 600
_MAXFEV = 800
_BOUND_TOL = 1e-5
_CAVEAT = "Linear-block uncertainty is conditional on the fitted azimuth factor."


class SharedAzimuthFit(Fit):
    """Minimum-norm retained moments conditional on a profiled complex factor."""

    kind: ClassVar[str] = "bilinear"
    estimator: ClassVar[str] = "SVD profiled over one complex azimuth factor"
    factor: complex = eqx.field(static=True)
    factor_name: str = eqx.field(static=True)
    profile_curvature: jax.Array | None
    curvature_step: float | None = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    rank_policy: Rank | UPRE = eqx.field(static=True)
    upre_curve: tuple[tuple[int, float], ...] = eqx.field(static=True)

    @property
    def params(self):
        """The gamma-independent second azimuth moment."""
        return {self.factor_name: self.factor}

    @property
    def at_bound(self):
        return abs(self.factor) >= 1.0 - _BOUND_TOL

    @property
    def effective_parameters(self):
        return self.rank + 2

    @property
    def defaults(self):
        return (
            *super().defaults,
            ("azimuth profile grid size", 1 + _PHASES * len(_RADII)),
            ("rank policy", str(self.rank_policy)),
            ("azimuth profile evaluations per refinement", _MAXFEV),
            ("azimuth profile converged", self.converged),
            ("uncertainty", _CAVEAT),
        )


def _with_factor(basis, factor, *, result=False):
    # The stored spectra and memo stay valid when only the final assembly changes.
    keys = dict(basis.record.keys)
    factor = jnp.asarray(factor, dtype=jnp.complex128)
    key = design_key(keys.get("memo"), basis.lattice.key, factor)
    record = replace(basis.record, keys={**keys, "design": key})
    if result:
        record = record.with_notes(_CAVEAT)
    return replace(basis, azimuth_factor=factor, key=key, record=record)


def _profile_design(basis, data):
    fitted = _fit_basis(basis, data.channels)
    constant = _design(replace(fitted, azimuth_factor=0j))
    real = _design(replace(fitted, azimuth_factor=1 + 0j)) - constant
    imag = _design(replace(fitted, azimuth_factor=1j)) - constant
    white = _white(data, fitted, jnp.concatenate((constant, real, imag), axis=-1))
    keep = np.asarray(data.mask & ~data.withheld)
    if not keep.any():
        raise DataError("there are no observations in the fitted rows")
    matrix = np.asarray(white.design)[keep]
    if np.any(~np.isfinite(matrix)) or np.any(~np.isfinite(np.asarray(white.data))):
        raise DataError("the whitened azimuth profile must be finite")
    constant, real, imag = np.split(matrix, 3, axis=-1)
    perturbations = np.stack((real.ravel(), imag.ravel()))
    if np.linalg.matrix_rank(perturbations) < 2:
        raise DataError(
            "the fitted rows must distinguish both azimuth factor coordinates"
        )
    if not np.any(constant):
        raise DataError(
            "without fitted spin-zero signal the azimuth magnitude and moment scale are tied"
        )
    return _AzimuthProfile(np.asarray(white.data)[keep], constant, real, imag)


@dataclass(frozen=True)
class _AzimuthProfile:
    data: np.ndarray
    constant: np.ndarray
    real: np.ndarray
    imag: np.ndarray

    def decompose(self, factor):
        matrix = self.constant + factor.real * self.real + factor.imag * self.imag
        try:
            left, values, _ = np.linalg.svd(matrix, full_matrices=False)
        except np.linalg.LinAlgError:
            from scipy.linalg import svd

            left, values, _ = svd(matrix, full_matrices=False, lapack_driver="gesvd")
        valid = values > RANK_RTOL * values[0]
        threshold = PAIR_RTOL * values[1:]
        # The shared-factor design inherits the free-fit gap policy.
        pairs = (values[:-1] - values[1:] <= threshold) & (
            np.finfo(values.dtype).eps * values[0] <= threshold
        )
        valid[:-1] &= ~(valid[1:] & pairs)
        return left, valid

    def chi2(self, factor, rank):
        left, valid = self.decompose(factor)
        if not valid[rank - 1]:
            return np.inf
        active = left[:, :rank]
        residual = self.data - active @ (active.T @ self.data)
        return float(residual @ residual)

    def curve(self, factor):
        left, valid = self.decompose(factor)
        projection = left.T @ self.data
        residual = self.data - left @ projection
        tail = np.r_[np.cumsum(projection[:0:-1] ** 2)[::-1], 0.0]
        return np.where(valid, residual @ residual + tail, np.inf)


def _disk_factor(point):
    radius, angle = point
    return complex(radius * np.exp(1j * angle))


@dataclass(frozen=True)
class _ProfileMinimum:
    point: tuple[float, float]
    chi2: float
    converged: bool

    @property
    def factor(self):
        return _disk_factor(self.point)


def _refine(profile, rank, starts):
    from scipy.optimize import minimize

    best = _ProfileMinimum(
        starts[0], profile.chi2(_disk_factor(starts[0]), rank), False
    )
    for radius, angle in starts:
        if not np.isfinite(profile.chi2(_disk_factor((radius, angle)), rank)):
            continue
        step = -0.03 if radius > 0.97 else 0.03
        refined = minimize(
            lambda point: profile.chi2(_disk_factor(point), rank),
            (radius, angle),
            method="Nelder-Mead",
            bounds=((0.0, 1.0), (None, None)),
            options={
                "initial_simplex": (
                    (radius, angle),
                    (radius + step, angle),
                    (radius, angle + 0.12),
                ),
                "xatol": 1e-8,
                "fatol": 1e-8,
                "maxiter": _MAXITER,
                "maxfev": _MAXFEV,
            },
        )
        if np.isfinite(refined.fun) and (
            refined.fun < best.chi2 or (refined.fun == best.chi2 and refined.success)
        ):
            best = _ProfileMinimum(
                tuple(refined.x), float(refined.fun), bool(refined.success)
            )
    return best


def _fit_profile(profile, policy):
    width = min(profile.constant.shape)
    if isinstance(policy, Rank):
        if policy.R > width:
            raise DataError("fixed azimuth fit rank exceeds the number of directions")
        ranks = (policy.R,)
    else:
        ranks = range(1, (width if policy.cap is None else min(width, policy.cap)) + 1)
    grid = [(0.0, 0.0)] + [
        (radius, angle)
        for radius in _RADII
        for angle in np.linspace(-np.pi, np.pi, _PHASES, endpoint=False)
    ]
    grid_scores = np.stack([profile.curve(_disk_factor(point)) for point in grid])
    score, best, previous, chosen = np.inf, None, None, None
    curve = []
    for rank in ranks:
        # The residual is nonnegative, so later ranks cannot beat this bound.
        complexity = 2 * (rank + 2)
        if complexity >= score:
            break
        costs = grid_scores[:, rank - 1]
        order = np.argsort(costs, kind="stable")
        finite_starts = [grid[i] for i in order if np.isfinite(costs[i])]
        if not finite_starts:
            continue
        if isinstance(policy, Rank):
            starts = finite_starts[:_FIXED_STARTS]
        else:
            starts = finite_starts[:1]
            if previous is not None:
                starts.append(previous.point)
        previous = _refine(profile, rank, starts)
        criterion = previous.chi2 + complexity
        curve.append((rank, criterion))
        if criterion < score:
            score, best, chosen = criterion, previous, rank
    if best is None:
        raise DataError(
            "the azimuth grid provides no estimable rank under the requested policy"
        )
    return Rank(chosen), best, tuple(curve)


def _curvature(objective, c, value):
    # A Cartesian stencil stays inside the disk for interior solutions.
    h = min(1e-3, (1.0 - abs(c)) / 3)
    if h < _BOUND_TOL:
        return None, None
    shifts = np.array([1, -1, 1j, -1j, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j])
    v = np.array([objective(c + h * step) for step in shifts])
    if not np.all(np.isfinite(v)):
        return None, None
    cross = (v[4] - v[5] - v[6] + v[7]) / 4
    matrix = np.array(
        [[v[0] + v[1] - 2 * value, cross], [cross, v[2] + v[3] - 2 * value]]
    )
    return jnp.asarray(matrix / h**2), h


def fit_shared_azimuth(
    basis: SpectralBasis,
    data: StokesData,
    *,
    rank: Rank | UPRE | int | None = None,
    keep_directions: bool = False,
    chunk_bytes: int = 2**31,
) -> SharedAzimuthFit:
    """Fit one sightline with free moments and a shared second azimuth moment.

    The basis needs P columns and an unresolved, independent azimuth. The
    radius and phase search uses the study's polar grid and local refinement
    on |c2| <= 1. UPRE profiles each candidate rank and scores the result
    with the complexity of the retained moments and two factor coordinates.
    The profile curvature is the Hessian of chi squared in
    (real c2, imaginary c2), evaluated by central differences in the interior.
    """
    if not isinstance(basis, SpectralBasis) or not isinstance(data, StokesData):
        raise TypeError("fit_shared_azimuth expects a SpectralBasis and StokesData")
    if data.batch_shape:
        raise DataError("the shared azimuth fit accepts one sightline")
    policy = UPRE() if rank is None else rank
    if not isinstance(policy, (Rank, UPRE)):
        policy = Rank(policy)
    if policy == Rank(0) or policy == UPRE(cap=0):
        raise DataError("the azimuth factor requires at least one retained moment")
    source = basis.averaged
    azimuth = source.kernel.azimuth
    if (
        azimuth is None
        or basis.azimuth_factor is None
        or basis.spectra2 is None
        or basis.lattice.index.blocks != ("M0",)
        or azimuth in basis.lattice.cells.variables
        or azimuth in dict(source.averaged)
        or any(block == "P" for block, _ in source.leaf_factors)
    ):
        raise RoleError(
            "the shared azimuth fit needs an independent unresolved azimuth and P columns"
        )
    profile = _profile_design(basis, data)
    selected, optimum, curve = _fit_profile(profile, policy)
    curvature, step = _curvature(
        lambda c: profile.chi2(c, selected.R), optimum.factor, optimum.chi2
    )
    fitted = fit_free(
        _with_factor(basis, optimum.factor, result=True),
        data,
        rank=selected,
        keep_directions=keep_directions,
        chunk_bytes=chunk_bytes,
    )
    dof = fitted.dof - 2
    p = jax.scipy.special.gammaincc(jnp.maximum(dof, 1) / 2, fitted.chi2 / 2)
    original = {field.name: getattr(fitted, field.name) for field in fields(Fit)}
    original.update(dof=dof, p_value=jnp.where(dof > 0, p, jnp.nan))
    return SharedAzimuthFit(
        **original,
        factor=optimum.factor,
        factor_name=f"{azimuth}.factor",
        profile_curvature=curvature,
        curvature_step=step,
        converged=optimum.converged,
        rank_policy=policy,
        upre_curve=curve if isinstance(policy, UPRE) else (),
    )
