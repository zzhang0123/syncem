"""Penalised linear estimators for noise-whitened data (SPEC Sec. 6.2 and 6.3).

Ridge penalises the scaled lattice moments, including when those moments
come from nonnegative point masses. The reported covariance propagates
noise through the linear ridge estimator. It excludes shrinkage bias.
The nonnegative solver reports the active-space effective dimension.

LABEL [extension].

Ridge acts on the squared norm of scaled moments in lattice units. The whitened design has
shape (n_data, n_row). The noise-only uncertainties have no certified
estimate of shrinkage or source discrepancy.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from numbers import Real
from typing import NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..values.errors import DataError
from .rank import RANK_RTOL, SVDReduction, reduce_svd

__all__ = [
    "Penalty",
    "RidgeSolution",
    "RidgePrediction",
    "NonnegativeSolution",
    "ProfileMaximum",
    "solve_ridge",
    "ridge",
    "ridge_prediction",
    "solve_nonnegative",
    "profile_max",
]

_ALPHA_GRID = tuple(10.0 ** np.linspace(-30.0, 2.0, 321))
_ACTIVE_SPACE_RTOL = 1e-12


def _nonnegative_scalar(value, label):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not math.isfinite(value)
        or value < 0
    ):
        raise ValueError(f"{label} must be a finite nonnegative scalar.")
    return float(value)


@dataclass(frozen=True)
class Penalty:
    """Strength of the squared norm of the scaled moments. None selects UPRE."""

    alpha: float | None = None

    def __post_init__(self):
        if self.alpha is not None:
            object.__setattr__(self, "alpha", _nonnegative_scalar(self.alpha, "alpha"))


class RidgeSolution(NamedTuple):
    """Scaled moments, singular-coordinate statistics and fit diagnostics."""

    moments: object
    beta: object
    sigma: object
    factors: object
    alpha: object
    effective: object
    chi2: object


class RidgePrediction(NamedTuple):
    """Noise-only statistics for linear prediction rows."""

    values: object
    sigma: object
    unconstrained: object
    covariance: object | None


class NonnegativeSolution(NamedTuple):
    """Scaled moments and point masses with the active-space effective count."""

    moments: np.ndarray
    weights: np.ndarray
    alpha: float
    effective: float
    chi2: float


class ProfileMaximum(NamedTuple):
    """Maximum and optimizer in the unpenalised noise-only profile domain."""

    value: np.ndarray | float
    weights: np.ndarray
    chi2: np.ndarray | float
    minimum_chi2: float


def _policy(penalty):
    if not isinstance(penalty, Penalty):
        raise TypeError("penalty must be Penalty() or Penalty(alpha).")
    return penalty.alpha


def _filters(scaled_squared, active, alpha_over_scale):
    denominator = scaled_squared + alpha_over_scale[..., None]
    return jnp.where(active, scaled_squared, 0.0) / jnp.where(active, denominator, 1.0)


def solve_ridge(
    reduction: SVDReduction, data, penalty: Penalty = Penalty()
) -> RidgeSolution:
    """Solve one or more data vectors with a shared noise-whitened SVD."""
    alpha = _policy(penalty)
    left, s, vt = reduction.left, reduction.singular, reduction.directions
    data = jnp.asarray(data)
    if data.ndim == 0 or data.shape[-1] != left.shape[0] or jnp.iscomplexobj(data):
        raise DataError("data must end in the same real whitened rows as the design.")
    data = eqx.error_if(
        data, jnp.any(~jnp.isfinite(data)), "the whitened data must be finite"
    )
    proj = data @ left
    active = s > reduction.cut.cutoff
    scale = jnp.where(s[0] > 0, s[0], 1.0)
    scaled_squared = (s / scale) ** 2
    if alpha is None:
        grid = jnp.asarray(_ALPHA_GRID, dtype=s.dtype)
        filters = _filters(scaled_squared, active, grid)
        # The squared norm of the residual perpendicular to U cancels in UPRE.
        score = proj**2 @ ((1 - filters) ** 2).T + 2 * jnp.sum(filters, axis=-1)
        best = jnp.argmin(score, axis=-1)
        factors = filters[best]
        alpha = (s[0] * jnp.sqrt(grid[best])) ** 2
    else:
        scaled_alpha = (jnp.sqrt(jnp.asarray(alpha, dtype=s.dtype)) / scale) ** 2
        factors = jnp.broadcast_to(
            _filters(scaled_squared, active, scaled_alpha), proj.shape
        )
        alpha = jnp.full(data.shape[:-1], alpha, dtype=s.dtype)

    sigma = factors / jnp.where(active, s, 1.0)
    beta = proj * sigma
    residual = data - (proj * factors) @ left.T
    return RidgeSolution(
        beta @ vt,
        beta,
        sigma,
        factors,
        alpha,
        jnp.sum(factors, axis=-1),
        jnp.sum(residual**2, axis=-1),
    )


def ridge(design, data, penalty: Penalty = Penalty()) -> RidgeSolution:
    """Solve a ridge problem from the real noise-whitened design."""
    return solve_ridge(reduce_svd(design), data, penalty)


def ridge_prediction(rows, reduction, solution, *, cov=False) -> RidgePrediction:
    """Propagate white noise and measure shrinkage on rows in moment coordinates."""
    vt = reduction.directions
    rows = jnp.atleast_2d(jnp.asarray(rows))
    if rows.ndim != 2 or rows.shape[1] != vt.shape[1] or jnp.iscomplexobj(rows):
        raise DataError(
            "prediction must have real rows on the ridge moment coordinates."
        )
    projection = rows @ vt.T
    gain = projection * solution.sigma[..., None, :]
    retained = (projection * solution.factors[..., None, :]) @ vt
    norm = jnp.linalg.norm(rows, axis=-1)
    outside = jnp.linalg.norm(rows - retained, axis=-1)
    return RidgePrediction(
        (rows @ solution.moments[..., None])[..., 0],
        jnp.linalg.norm(gain, axis=-1),
        outside / jnp.where(norm > 0, norm, 1.0),
        gain @ jnp.swapaxes(gain, -1, -2) if cov else None,
    )


def _finite_host(value, label):
    value = np.asarray(value)
    if np.iscomplexobj(value) or not np.all(np.isfinite(value)):
        raise DataError(f"{label} must be real and finite.")
    return value.astype(float, copy=False)


def _host_problem(design, data, points):
    design = _finite_host(design, "design")
    data = _finite_host(data, "data")
    points = _finite_host(points, "point-mass map")
    if design.ndim != 2 or min(design.shape) == 0:
        raise DataError("design must be a nonempty matrix.")
    n, p = design.shape
    if data.shape != (n,):
        raise DataError("nonnegative fits need one whitened data vector.")
    if points.ndim != 2 or points.shape[0] != p or not points.shape[1]:
        raise DataError(
            "points must map point masses into the design's moment columns."
        )
    return design, data, points


def _nnls(design, data):
    from scipy.optimize import nnls

    scales = np.max(np.abs(design), axis=0)
    keep = scales > 0
    weights = np.zeros(design.shape[1])
    if np.any(keep):
        matrix = design[:, keep] / scales[keep]
        lengths = np.linalg.norm(matrix, axis=0)
        solution, _ = nnls(matrix / lengths, data, maxiter=200 * design.shape[1])
        weights[keep] = (solution / lengths) / scales[keep]
    return weights


def _active_effective(design, points, weights, alpha):
    active = weights > 0.0
    if alpha == 0 or not np.any(active):
        return float(np.count_nonzero(active))
    u, s, _ = np.linalg.svd(points[:, active], full_matrices=False)
    span = u[:, s > _ACTIVE_SPACE_RTOL * s[0]]
    t = np.linalg.svd(design @ span, compute_uv=False)
    denominator = np.hypot(t, math.sqrt(alpha))
    return float(np.sum((t / denominator) ** 2))


def solve_nonnegative(
    design, data, points, penalty: Penalty = Penalty()
) -> NonnegativeSolution:
    """Fit nonnegative masses while penalising their scaled moments."""
    alpha = _policy(penalty)
    design, data, points = _host_problem(design, data, points)
    if alpha is None:
        alpha = float(ridge(design, data).alpha)
    matrix = design @ points
    if alpha > 0:
        weights = _nnls(
            np.vstack((matrix, math.sqrt(alpha) * points)),
            np.concatenate((data, np.zeros(points.shape[0]))),
        )
    else:
        weights = _nnls(matrix, data)
    moments = points @ weights
    residual = matrix @ weights - data
    return NonnegativeSolution(
        moments,
        weights,
        alpha,
        _active_effective(design, points, weights, alpha),
        float(residual @ residual),
    )


def _unbounded(value, size):
    return value, np.full(size, np.nan), np.nan


def _linprog_max(matrix, c, fitted):
    from scipy.optimize import linprog

    solved = linprog(-c, A_eq=matrix, b_eq=fitted, bounds=(0.0, None), method="highs")
    if solved.status == 3:
        if _increasing_recession(matrix, c):
            return None
        raise RuntimeError(
            "profile linear optimisation could not verify an unbounded direction."
        )
    if not solved.success:
        raise RuntimeError(f"profile linear optimisation failed. {solved.message}")
    return solved.x


def _increasing_recession(matrix, c):
    from scipy.optimize import linprog

    if np.any((c > 0) & np.all(matrix == 0, axis=0)):
        return True
    solved = linprog(
        -c,
        A_eq=matrix,
        b_eq=np.zeros(matrix.shape[0]),
        A_ub=np.ones((1, c.size)),
        b_ub=np.ones(1),
        bounds=(0.0, None),
        method="highs",
    )
    if not solved.success:
        raise RuntimeError(f"profile recession check failed. {solved.message}")
    eps = np.finfo(float).eps
    if -solved.fun <= 128 * eps:
        return False
    # HiGHS's equality tolerance can admit a very weakly constrained but
    # finite direction. Require a null witness at floating-point precision.
    residual = np.abs(matrix @ solved.x)
    roundoff = 32 * eps * (np.abs(matrix) @ np.abs(solved.x))
    if np.any(residual > roundoff):
        raise RuntimeError(
            "profile recession direction is unresolved at the linear solver tolerance."
        )
    return True


def _polish_profile_support(matrix, c, baseline, residual, delta, candidate, rtol):
    """Use the analytic ellipsoid maximiser on the numerical support."""
    active = candidate > np.max(candidate) * rtol
    if not np.any(active):
        return None
    left, singular, right = np.linalg.svd(matrix[:, active], full_matrices=False)
    keep = singular > singular[0] * RANK_RTOL
    left, singular, right = left[:, keep], singular[keep], right[keep]
    goals = right @ c[active] / singular
    slope = np.linalg.norm(goals)
    if not slope > 0:
        return None
    data = matrix @ baseline - residual
    centre = left.T @ data
    outside = left @ centre - data
    available = delta + (residual - outside) @ (residual + outside)
    if not available > 0:
        return None
    coefficients = (centre + np.sqrt(available) / slope * goals) / singular
    result = np.zeros_like(candidate)
    # Keep the original null-space coordinates. Several positive Lobatto
    # atoms in a cell can describe the same resolved direction.
    result[active] = candidate[active] + right.T @ (
        coefficients - right @ candidate[active]
    )
    return result


def _recover_noise_ball(matrix, c, baseline, residual, delta, candidate):
    """Return feasible weights whose objective is within the checked dual gap."""
    if not np.all(np.isfinite(candidate)):
        return None
    weights = np.maximum(candidate, 0.0)
    difference = matrix @ (weights - baseline)
    quadratic = difference @ difference
    linear = difference @ residual
    if quadratic + 2 * linear > delta:
        fraction = delta / (np.sqrt(linear**2 + quadratic * delta) + linear)
        fraction = np.nextafter(min(1.0, fraction), 0.0)
        weights = (1 - fraction) * baseline + fraction * weights
        difference = matrix @ (weights - baseline)
    increase = difference @ (difference + 2 * residual)
    if increase > delta * (1 + 1e-10):
        return None

    trial_residual = residual + difference
    projected_residual = matrix.T @ trial_residual
    active = weights > np.max(weights) * 1e-12
    gradient = projected_residual[active]
    norm2 = gradient @ gradient
    if norm2 <= 0:
        return None
    multiplier = max(0.0, (c[active] @ gradient) / norm2)
    dual = multiplier * trial_residual

    # The baseline observation supplies a separating direction for
    # nonnegative intensity models. A dual-feasible vector certifies
    # c @ w <= dual @ data + radius * ||dual|| over the noise ball.
    direction = matrix @ baseline
    forward = matrix.T @ direction
    present = np.any(matrix != 0, axis=0)
    if np.any(forward[present] <= 0) or np.any(c[~present] > 0):
        return None
    eps = 8 * (matrix.shape[0] + 1) * np.finfo(float).eps
    guard = eps * (np.abs(matrix).T @ np.abs(dual) + np.abs(c))
    deficit = (c - matrix.T @ dual + 2 * guard)[present]
    dual += max(0.0, np.max(deficit / forward[present])) * direction
    guard = eps * (np.abs(matrix).T @ np.abs(dual) + np.abs(c))
    slack = matrix.T @ dual - c
    if not np.all(np.isfinite(dual)) or np.any(slack[present] < guard[present]):
        return None

    # Evaluate the gap relative to the feasible point to avoid subtracting
    # large terms when the observations have an irreducible residual.
    dual_norm = np.linalg.norm(dual)
    trial_norm = np.linalg.norm(trial_residual)
    radius = np.hypot(np.linalg.norm(residual), np.sqrt(delta))
    gap = (slack + guard) @ weights
    gap += dual_norm * max(delta - increase, 0.0) / (radius + trial_norm)
    if dual_norm > 0 and trial_norm > 0:
        direction = dual / dual_norm - trial_residual / trial_norm
        gap += 0.5 * dual_norm * trial_norm * (direction @ direction)
    upper = c @ weights + gap
    if not np.all(np.isfinite([upper, gap])) or not (0 <= gap <= 5e-6 * abs(upper)):
        return None
    return weights


def _noise_ball(matrix, c, baseline, residual, delta):
    from scipy.optimize import minimize

    def slack(weights):
        shift = matrix @ (weights - baseline)
        increase = shift @ (shift + 2 * residual)
        return 1 - increase / delta

    def gradient(weights):
        shift = matrix @ (weights - baseline)
        return -2 * (matrix.T @ (shift + residual)) / delta

    solved = minimize(
        lambda weights: -c @ weights,
        baseline,
        jac=lambda _: -c,
        bounds=[(0.0, None)] * baseline.size,
        constraints=[{"type": "ineq", "fun": slack, "jac": gradient}],
        method="SLSQP",
        options={"maxiter": 1000, "ftol": 1e-11},
    )
    if solved.success and slack(solved.x) >= -1e-8 and np.all(solved.x >= 0):
        return solved.x
    recovered = _recover_noise_ball(matrix, c, baseline, residual, delta, solved.x)
    if recovered is not None:
        return recovered
    if np.all(np.isfinite(solved.x)):
        # A stalled solve can leave numerical residue on inactive weights.
        # Try a stricter support only when polishing the first one did not
        # produce a feasible point with a checked dual gap.
        for rtol in (_ACTIVE_SPACE_RTOL, np.sqrt(np.finfo(float).eps)):
            polished = _polish_profile_support(
                matrix, c, baseline, residual, delta, solved.x, rtol
            )
            if polished is not None:
                recovered = _recover_noise_ball(
                    matrix, c, baseline, residual, delta, polished
                )
                if recovered is not None:
                    return recovered
    raise RuntimeError(
        f"profile maximum did not converge to a feasible point. {solved.message}"
    )


def _profile_row(matrix, c, baseline, residual, delta):
    scale = np.max(np.abs(c))
    if scale == 0:
        return baseline
    c = c / scale
    if delta == 0:
        # Every minimiser has the same fitted observations.
        return _linprog_max(matrix, c, matrix @ baseline)
    if _increasing_recession(matrix, c):
        return None
    return _noise_ball(matrix, c, baseline, residual, delta)


def profile_max(design, data, points, rows, *, delta_chi2=1.0) -> ProfileMaximum:
    """Maximise each prediction row under the unpenalised NNLS residual limit.

    Recovery after a failed numerical solve returns a feasible value within
    five parts per million of the checked dual upper value.
    A direction of recession with positive prediction gives an infinite
    profile. There is no finite maximiser in that case. The returned
    weights and achieved chi2 for that row are NaN.
    """
    delta = _nonnegative_scalar(delta_chi2, "delta_chi2")
    design, data, points = _host_problem(design, data, points)
    rows = _finite_host(rows, "prediction rows")
    scalar = rows.ndim == 1
    rows = np.atleast_2d(rows)
    if rows.ndim != 2 or rows.shape[1] != design.shape[1] or not rows.shape[0]:
        raise DataError(
            "predictions must have rows in the design's moment coordinates."
        )
    matrix = design @ points
    weights = _nnls(matrix, data)
    residual = matrix @ weights - data
    minimum = float(residual @ residual)
    scale = np.max(np.abs(matrix), axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    matrix = matrix / scale
    baseline = weights * scale
    target = (rows @ points) / scale

    found = []
    for row in target:
        best = _profile_row(matrix, row, baseline, residual, delta)
        if best is None:
            found.append(_unbounded(np.inf, points.shape[1]))
        else:
            difference = matrix @ (best - baseline) + residual
            found.append(
                (float(row @ best), best / scale, float(difference @ difference))
            )
    values, optimizers, scores = map(np.asarray, zip(*found))
    return ProfileMaximum(
        float(values[0]) if scalar else values,
        optimizers[0] if scalar else optimizers,
        float(scores[0]) if scalar else scores,
        minimum,
    )
