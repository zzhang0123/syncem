"""Host variable projection for fitted basis statistics (SPEC Sec. 6.1).

LABEL [extension].

The inner design has shape (n_data, n_row) after whitening. Outer
coordinates report values in canonical units. Trial grids and differences
carry no certified global optimum.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import fields, replace
from itertools import product
from numbers import Integral, Real
from typing import ClassVar, NamedTuple

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..distributions.parameters import Free
from ..engine.basis import SpectralBasis
from ..values.errors import DataError, RoleError
from ._basis_trials import _Trials
from ._finite_difference import _coordinate_scale, _jacobian
from .data import StokesData
from .family_fit import Parameter, _coordinate, _parameter_values
from .free_fit import Fit, fit_free
from .rank import RANK_RTOL, FullRank, Rank, UPRE

__all__ = ["BasisParametricFit", "BasisTrial", "fit_basis_parametric"]

_BOUND_TOL = 1e-5
_CAVEAT = (
    "Retained linear uncertainties are conditional on the fitted basis parameters."
)


class BasisTrial(NamedTuple):
    """Physical grid values, score at the fixed rank and best grid policy rank."""

    values: tuple[float, ...]
    chi2: float
    rank: int | None


class BasisParametricFit(Fit):
    """Noise-only variable-projection result with fixed inner rank."""

    kind: ClassVar[str] = "basis-parametric"
    estimator: ClassVar[str] = "bounded trust-region fit with profiled SVD moments"
    parameter_table: tuple[tuple[str, Parameter], ...] = eqx.field(static=True)
    parameter_covariance: jax.Array
    grid: tuple[BasisTrial, ...] = eqx.field(static=True)
    grid_rank_policy: Rank | FullRank | UPRE = eqx.field(static=True)
    finite_difference_steps: tuple[tuple[str, float], ...] = eqx.field(static=True)
    basis_evaluations: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    converged_starts: int = eqx.field(static=True)

    @property
    def params(self):
        return dict(self.parameter_table)

    @property
    def effective_parameters(self):
        return self.rank + len(self.parameter_table)

    @property
    def profiles(self):
        """Minimum fixed-rank chi squared at each gridded parameter value."""
        output = {}
        for i, (name, _) in enumerate(self.parameter_table):
            scores = {}
            for row in self.grid:
                if np.isfinite(row.chi2):
                    value = row.values[i]
                    scores[value] = min(scores.get(value, np.inf), row.chi2)
            output[name] = tuple(sorted(scores.items()))
        return output

    @property
    def defaults(self):
        return (
            *super().defaults,
            ("grid rank policy", str(self.grid_rank_policy)),
            ("grid points", len(self.grid)),
            ("basis evaluations", self.basis_evaluations),
            (
                "finite difference steps in fit coordinates",
                self.finite_difference_steps,
            ),
            ("uncertainty", _CAVEAT),
        )


def _positive_int(value, name, *, minimum=1):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer of at least {minimum}")
    return int(value)


def _grid(x0, lo, hi, budget):
    """Use a complete small grid or a deterministic low-discrepancy grid."""
    scale = np.maximum(abs(x0), 1.0)
    lower = np.where(np.isfinite(lo), lo, x0 - scale)
    upper = np.where(np.isfinite(hi), hi, x0 + scale)
    # A one-sided infinite bound may lie beyond the start-relative window.
    lower = np.minimum(lower, x0)
    upper = np.maximum(upper, x0)
    if budget == 1:
        return x0[None, :]
    n = min(5, int((budget - 1) ** (1.0 / len(x0))))
    if n >= 3:
        axes = [np.linspace(a, b, n) for a, b in zip(lower, upper)]
        candidates = [x0, *product(*axes)]
    else:
        from scipy.stats import qmc

        points = qmc.Halton(len(x0), scramble=False).random(budget - 2)
        candidates = [x0, upper, *(lower + points * (upper - lower))]
    unique = dict.fromkeys(tuple(row) for row in candidates)
    return np.array(tuple(unique))[:budget]


def _select_rank(scores, policy, dim):
    width = scores.shape[1] - 1
    if isinstance(policy, Rank):
        if policy.R > width:
            raise DataError("the fixed rank exceeds the number of moment directions")
        chosen = policy.R
        per_trial = [chosen if np.isfinite(x) else None for x in scores[:, chosen]]
    elif isinstance(policy, FullRank):
        per_trial = [int(np.flatnonzero(np.isfinite(row))[-1]) for row in scores]
        chosen = max(per_trial)
    else:
        width = width if policy.cap is None else min(width, policy.cap)
        ranks = np.arange(width + 1)
        criterion = scores[:, : width + 1] + 2 * (ranks + dim)
        positive = np.isfinite(criterion[:, 1:]).any(axis=1)
        criterion[positive, 0] = np.inf
        per_trial = np.argmin(criterion, axis=1).tolist()
        if np.any(positive):
            criterion[:, 0] = np.inf
        _, chosen = np.unravel_index(np.argmin(criterion), criterion.shape)
    return int(chosen), per_trial


def _refine(trials, rank, starts, lo, hi, fd_step, max_nfev):
    from scipy.optimize import least_squares

    def evaluate(x):
        return trials.residual(np.clip(x, lo, hi), rank)

    # Unavailable fixed-rank trials cannot improve an admissible starting point.
    penalty = np.full_like(trials.y, max(1.0, np.linalg.norm(trials.y)) * 100)

    def objective(x):
        residual = evaluate(x)
        return penalty if residual is None else residual

    def jacobian(x):
        if evaluate(x) is None:
            return np.zeros((trials.y.size, x.size))
        return _jacobian(evaluate, x, lo, hi, fd_step)[0]

    best, best_score = starts[0], np.inf
    solutions = []
    for start in starts:
        residual = evaluate(start)
        score = float(residual @ residual)
        result = least_squares(
            objective,
            start,
            jac=jacobian,
            x_scale=_coordinate_scale(start, lo, hi),
            bounds=(lo, hi),
            method="trf",
            max_nfev=max_nfev,
        )
        local = evaluate(result.x)
        if local is not None:
            local_score = float(local @ local)
            if local_score < score or (result.success and local_score == score):
                start, score = result.x, local_score
            if result.success:
                solutions.append(local_score)
        if score < best_score:
            best, best_score = start, score
    count = sum(
        np.isclose(value, best_score, atol=1e-8, rtol=1e-8) for value in solutions
    )
    jac, steps = _jacobian(evaluate, best, lo, hi, fd_step)
    return best, jac, steps, count > 0 and all(steps), int(count)


def _diagnostics(trials, x, lo, hi, names, log, jac, steps):
    actual = _coordinate(x, log)
    lower, upper = _coordinate(lo, log), _coordinate(hi, log)
    fit_scale = _coordinate_scale(x, lo, hi)
    _, singular, vectors = np.linalg.svd(jac * fit_scale, full_matrices=False)
    used_steps = [abs(h) / scale for h, scale in zip(steps, fit_scale) if h != 0]
    fd_floor = (
        16 * np.finfo(float).eps * max(np.linalg.norm(trials.y), 1.0) / min(used_steps)
        if used_steps
        else np.inf
    )
    keep = singular > max(RANK_RTOL * singular[0], fd_floor)
    identified = np.sum(vectors[keep] ** 2, axis=0) >= 1 - 1e-8
    covariance = (vectors[keep].T / singular[keep] ** 2) @ vectors[keep]
    scale = fit_scale * np.where(log, actual, 1.0)
    covariance = covariance * scale[:, None] * scale
    covariance = np.where(identified[:, None] & identified, covariance, np.nan)
    distance = np.minimum(x - lo, hi - x)
    at_bound = distance <= _BOUND_TOL * fit_scale
    table = tuple(
        (
            name,
            Parameter(
                float(actual[i]),
                float(np.sqrt(covariance[i, i])),
                (float(lower[i]), float(upper[i])),
                bool(at_bound[i]),
                bool(identified[i]),
            ),
        )
        for i, name in enumerate(names)
    )
    return table, jnp.asarray(covariance)


def fit_basis_parametric(
    build_basis: Callable[[dict[str, float]], SpectralBasis],
    data: StokesData,
    parameters: Mapping[str, Free],
    *,
    starts=None,
    rank: Rank | FullRank | UPRE | int | None = None,
    fd_step: float = 1e-4,
    max_grid: int = 27,
    max_starts: int = 3,
    max_nfev: int = 40,
    keep_directions: bool = False,
    chunk_bytes: int = 2**31,
) -> BasisParametricFit:
    """Fit free moments and declared basis parameters on one sightline.

    ``build_basis`` receives physical values and manages trial engine caching.
    UPRE counts the outer coordinates and selects rank once on the coarse grid.
    FullRank picks the largest estimable grid rank. Refinement holds that rank
    fixed. Finite-difference steps are relative to fit coordinates or the
    narrower bound span. Fit coordinates are logarithmic when requested. Positive recorded steps are central or forward.
    Negative steps are backward. Zero marks an unavailable derivative.
    """
    if not callable(build_basis) or not isinstance(data, StokesData):
        raise TypeError("fit_basis_parametric expects a basis builder and StokesData")
    if data.batch_shape:
        raise DataError("the basis-parametric fit accepts one sightline")
    if not parameters or not all(
        isinstance(k, str) and k and isinstance(v, Free) for k, v in parameters.items()
    ):
        raise RoleError("supply named Free declarations for the basis parameters")
    if (
        not isinstance(fd_step, Real)
        or isinstance(fd_step, bool)
        or not 0 < fd_step < 1
    ):
        raise ValueError("fd_step must be a relative step between zero and one")
    max_grid = _positive_int(max_grid, "max_grid")
    max_starts = _positive_int(max_starts, "max_starts")
    max_nfev = _positive_int(max_nfev, "max_nfev")
    _positive_int(chunk_bytes, "chunk_bytes")
    if not isinstance(keep_directions, bool):
        raise ValueError("keep_directions must be a bool")
    policy = UPRE() if rank is None else rank
    if not isinstance(policy, (Rank, FullRank, UPRE)):
        policy = Rank(policy)
    starts = {} if starts is None else dict(starts)
    if unknown := set(starts) - set(parameters):
        raise RoleError(f"unknown basis parameter starts: {sorted(unknown)}")
    names, x0, lo, hi, log = _parameter_values(parameters, starts)
    trials = _Trials(build_basis, data, names, log)
    grid = _grid(x0, lo, hi, max_grid)
    scores = np.stack([trials.get(x).scores for x in grid])
    fixed, chosen = _select_rank(scores, policy, len(names))
    objective = scores[:, fixed]
    order = np.argsort(objective, kind="stable")
    candidates = [grid[i] for i in order[:max_starts] if np.isfinite(objective[i])]
    if not candidates:
        raise DataError("the fixed rank is unavailable at every basis grid point")

    x, jac, steps, converged, count = _refine(
        trials, fixed, candidates, lo, hi, float(fd_step), max_nfev
    )
    diagnostics, covariance = _diagnostics(trials, x, lo, hi, names, log, jac, steps)
    basis = trials.basis(x)
    basis = replace(basis, record=basis.record.with_notes(_CAVEAT))
    fitted = fit_free(
        basis,
        data,
        rank=Rank(fixed),
        keep_directions=keep_directions,
        chunk_bytes=chunk_bytes,
    )
    dof = fitted.dof - len(names)
    p = jax.scipy.special.gammaincc(jnp.maximum(dof, 1) / 2, fitted.chi2 / 2)
    original = {field.name: getattr(fitted, field.name) for field in fields(Fit)}
    original.update(dof=dof, p_value=jnp.where(dof > 0, p, jnp.nan))
    record = tuple(
        BasisTrial(tuple(map(float, _coordinate(point, log))), float(chi2), rank)
        for point, chi2, rank in zip(grid, objective, chosen)
    )
    return BasisParametricFit(
        **original,
        parameter_table=diagnostics,
        parameter_covariance=covariance,
        grid=record,
        grid_rank_policy=policy,
        finite_difference_steps=tuple(zip(names, steps)),
        basis_evaluations=trials.evaluations,
        converged=converged,
        converged_starts=count,
    )
