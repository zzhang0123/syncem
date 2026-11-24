"""Independent free moment factors on a tensor lattice (SPEC Sec. 6.1).

LABEL [extension].

Each factor carries scaled moments in its lattice units. The outer product
of shape (n_a, n_b) forms the joint coefficients. A finite set of starts
carries no certified global minimum.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace
from numbers import Integral
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..base.capabilities import FieldScalingRelation
from ..engine.basis import SpectralBasis
from ..lattice.moments import Moments
from ..values.errors import DataError
from ._free_fit_arrays import memory_chunk, withheld_statistics
from ._factor_coordinates import _groups, _initial, _Lift, _unit
from .data import StokesData
from .free_fit import (
    Fit,
    RetainedCombinations,
    _design,
    _fit_basis,
    _ledger,
    _Problem,
    _white,
)
from .rank import (
    RANK_RTOL,
    FullRank,
    Rank,
    UPRE,
    reduce_svd,
    solve_svd,
    warn_rank_margin,
)

__all__ = ["FactorFit", "MomentFactor", "fit_factors"]

_CONDITIONAL = (
    "Prediction uncertainty is conditional on the fitted outer moment factors."
)
_GAUGE = "Each outer factor has unit raw-coefficient norm and a positive largest-magnitude coefficient."


class MomentFactor(eqx.Module):
    """Raw moment coefficients for one group and their cell/power coordinates."""

    variables: tuple[str, ...] = eqx.field(static=True)
    coordinates: tuple[tuple[tuple[int, ...], tuple[int, ...]], ...] = eqx.field(
        static=True
    )
    values: jax.Array


class FactorFit(Fit):
    """A conditional retained fit with independent factors and a scaling gauge."""

    kind: ClassVar[str] = "bilinear"
    estimator: ClassVar[str] = "alternating SVD of independent moment factors"
    factors: tuple[MomentFactor, ...]
    outer_rank: int = eqx.field(static=True)
    outer_dimension: int = eqx.field(static=True)
    converged: bool = eqx.field(static=True)
    iterations: int = eqx.field(static=True)
    starts: int = eqx.field(static=True)
    initial_joint_rank: int = eqx.field(static=True)
    rank_policy: Rank | FullRank | UPRE = eqx.field(static=True)

    @property
    def effective_parameters(self):
        return self.rank + self.outer_dimension

    @property
    def outer_identified(self):
        """Whether the linearized data distinguish every outer shape direction."""
        return self.outer_rank == self.outer_dimension

    @property
    def defaults(self):
        return (
            *super().defaults,
            ("factor normalization", _GAUGE),
            ("factor rank policy", str(self.rank_policy)),
            ("factor starts", self.starts),
            ("factor iterations", self.iterations),
            ("factor convergence", self.converged),
            ("outer linearized rank", self.outer_rank),
            ("outer dimension", self.outer_dimension),
            ("uncertainty", _CONDITIONAL),
        )


@eqx.filter_jit
def _reduce(design):
    return reduce_svd(design)


@eqx.filter_jit
def _solve(reduction, data, policy):
    return solve_svd(reduction, data, policy)


def _fit_lifted(matrix, y, lift, policy):
    reduced = _reduce(jnp.asarray(lift.design(matrix)))
    if isinstance(policy, Rank) and policy.R:
        if policy.R > int(reduced.cut.numerical_rank):
            return None, reduced
        if policy.R < reduced.singular.size and bool(reduced.paired[policy.R - 1]):
            return None, reduced
    return _solve(reduced, y, policy), reduced


@dataclass(frozen=True)
class _State:
    factors: tuple[np.ndarray, ...]
    score: float
    lift: _Lift
    solution: object
    reduction: object
    iterations: int = 0
    converged: bool = False


def _profile(matrix, y, groups, factors, weights, policy):
    lift = _Lift.make(groups, factors, weights, 0)
    solution, reduction = _fit_lifted(matrix, y, lift, policy)
    if solution is None:
        return None
    values = lift.raw(np.asarray(solution.moments))
    score = float(solution.chi2)
    if isinstance(policy, UPRE):
        score += 2 * int(solution.rank)
    return _State((values, *factors[1:]), score, lift, solution, reduction)


def _sweep(matrix, y, groups, factors, weights):
    updated = list(factors)
    for i, group in enumerate(groups[1:], 1):
        lift = _Lift.make(groups, updated, weights, i)
        solution, _ = _fit_lifted(matrix, y, lift, FullRank())
        value = lift.raw(np.asarray(solution.moments))
        updated[i] = _unit(value, group.mass_start())
    return updated


def _interpolate(old, new, fraction):
    result = [old[0]]
    for prior, proposed in zip(old[1:], new[1:]):
        proposed = np.copysign(1.0, prior @ proposed) * proposed
        result.append(_unit(prior + fraction * (proposed - prior), prior))
    return tuple(result)


def _shape_change(old, new):
    return max(
        min(np.linalg.norm(a - b), np.linalg.norm(a + b))
        for a, b in zip(old[1:], new[1:])
    )


def _alternating(matrix, y, groups, factors, weights, policy, max_iterations):
    current = _profile(matrix, y, groups, factors, weights, policy)
    if current is None:
        return None
    for it in range(1, max_iterations + 1):
        updated = _sweep(matrix, y, groups, current.factors, weights)
        for fraction in (1.0, 0.5, 0.25, 0.125, 0.0625):
            attempt = _interpolate(current.factors, updated, fraction)
            candidate = _profile(matrix, y, groups, attempt, weights, policy)
            if candidate is not None and candidate.score <= current.score:
                break
        else:
            unchanged = _shape_change(updated, current.factors) < 1e-7
            return replace(current, iterations=it, converged=bool(unchanged))
        change = current.score - candidate.score
        shift = _shape_change(candidate.factors, current.factors)
        current = candidate
        if change <= 1e-11 * max(1.0, abs(current.score)) and shift < 1e-7:
            return replace(current, iterations=it, converged=True)
    return replace(current, iterations=max_iterations)


def _outer_rank(matrix, groups, factors, weights, state):
    directions = np.asarray(state.reduction.left)[:, : int(state.solution.rank)]
    jacobian = []
    for i, factor in enumerate(factors):
        if i == 0:
            continue
        lift = _Lift.make(groups, factors, weights, i)
        derivative = lift.design(matrix) * lift.norms
        # Rotate the factor's radial direction onto the pivot and drop it.
        pivot = int(np.argmax(abs(factor)))
        normal = factor.copy()
        normal[pivot] += np.copysign(1.0, normal[pivot])
        remaining = np.arange(factor.size) != pivot
        derivative = derivative[:, remaining] - np.outer(
            2 * (derivative @ normal) / (normal @ normal), normal[remaining]
        )
        derivative -= directions @ (directions.T @ derivative)
        jacobian.append(derivative)
    matrix_j = np.concatenate(jacobian, axis=1)
    if matrix_j.shape[1] == 0:
        return 0
    singular = np.linalg.svd(matrix_j, compute_uv=False)
    signal = np.linalg.norm(
        matrix @ state.lift.joint(np.asarray(state.solution.moments))
    )
    cutoff = max(RANK_RTOL * singular[0], 16 * np.finfo(float).eps * signal)
    return int(np.count_nonzero(singular > cutoff))


def _notes(basis, groups):
    notes = [_GAUGE, _CONDITIONAL]
    axes = [set(g.names) for g in groups]
    source = basis.averaged
    for capability in source.kernel.capabilities:
        if isinstance(capability, FieldScalingRelation):
            pair = {capability.variable, capability.energy}
            if all(any(name in group for group in axes) for name in pair) and not any(
                pair <= group for group in axes
            ):
                notes.append(
                    f"The kernel's {capability.variable} and {capability.energy} scaling relation can tie their fitted shapes."
                )
    return notes


def fit_factors(
    basis: SpectralBasis,
    data: StokesData,
    groups,
    *,
    rank: Rank | FullRank | UPRE | int | None = None,
    starts: int = 3,
    max_iterations: int = 40,
    seed: int = 0,
    chunk_bytes: int = 2**31,
) -> FactorFit:
    """Fit one independent moment factor per declared set of celled variables.

    The first group carries the scale and the selected SVD rank. Remaining
    groups have unit raw-coefficient norm and a fixed pivot sign. Their shape
    dimensions enter the UPRE complexity and the reported degrees of freedom.
    The inner norm and retained directions are in scaled joint lattice rows.
    The solver compares a singular-pair start and deterministic random starts.
    Retained combinations describe the conditional linear fit. They cannot be
    reused as a compressed likelihood over a different factor subspace.
    """
    if not isinstance(basis, SpectralBasis) or not isinstance(data, StokesData):
        raise TypeError("fit_factors expects a SpectralBasis and StokesData")
    if data.batch_shape:
        raise DataError("independent moment factors accept one sightline")
    for name, value in (
        ("starts", starts),
        ("max_iterations", max_iterations),
        ("chunk_bytes", chunk_bytes),
    ):
        if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
            raise ValueError(f"{name} must be a positive integer")
    if basis.lattice.key is None:
        raise DataError("independent factors need a concrete lattice key")
    groups = _groups(basis.lattice, groups)
    policy = UPRE() if rank is None else rank
    if not isinstance(policy, (UPRE, Rank, FullRank)):
        policy = Rank(policy)
    if isinstance(policy, Rank) and policy.R > groups[0].size:
        raise DataError("fixed rank exceeds the size of the primary moment group")
    rng = np.random.default_rng(seed)
    fitted = _fit_basis(basis, data.channels)
    held = bool(np.any(np.asarray(data.withheld)))
    memory_chunk(data, fitted.lattice.n, chunk_bytes, 8, held=held)
    design = _design(fitted)
    white = _white(data, fitted, design)
    if int(white.count) == 0:
        raise DataError("there are no observations in the fitted rows")
    matrix, y = np.asarray(white.design), white.data
    weights = np.asarray(fitted.lattice.weights)[fitted.lattice.index.arrays.cell]
    joint = _solve(_reduce(white.design), y, UPRE())
    first = _initial(groups, np.asarray(joint.moments) * weights, rng)
    best = None
    for start in range(starts):
        factors = (
            first
            if start == 0
            else tuple(
                _unit(rng.standard_normal(group.size), group.mass_start())
                for group in groups
            )
        )
        result = _alternating(
            matrix, y, groups, factors, weights, policy, max_iterations
        )
        if result is not None and (best is None or result.score < best.score):
            best = result
    if best is None:
        raise DataError("fixed rank is unavailable for the independent factor starts")
    solution, reduction = best.solution, best.reduction
    warn_rank_margin(reduction.cut)
    factors = tuple(
        MomentFactor(group.names, group.coordinates, jnp.asarray(value))
        for group, value in zip(groups, best.factors)
    )
    directions = jnp.asarray(best.lift.joint(np.asarray(reduction.directions)))
    values = jnp.asarray(best.lift.joint(np.asarray(solution.moments)))
    key = hashlib.sha256(f"{basis.lattice.key}:{basis.columns.key}".encode())
    for factor in best.factors[1:]:
        key.update(factor.tobytes())
    fingerprint = key.hexdigest()
    retained = RetainedCombinations(
        solution.beta, solution.sigma, directions, solution.rank, fingerprint
    )
    moments = Moments(
        basis.lattice,
        values,
        origin="minimum_norm",
        masked=tuple(map(int, np.flatnonzero(basis.columns.col_of_row < 0))),
        retained=fingerprint,
    )
    withheld = None
    if held:
        test = _white(data, fitted, design, held=True)
        reduced = test._replace(
            design=jnp.asarray(best.lift.design(np.asarray(test.design)))
        )
        withheld = withheld_statistics(data, reduced, solution, reduction)
    dimension = sum(group.size - 1 for group in groups[1:])
    outer_rank = _outer_rank(matrix, groups, best.factors, weights, best)
    dof = white.count - solution.rank - dimension
    p = jax.scipy.special.gammaincc(jnp.maximum(dof, 1) / 2, solution.chi2 / 2)
    notes = _notes(basis, groups)
    basis = replace(basis, record=basis.record.with_notes(*notes))
    fitted = replace(fitted, record=fitted.record.with_notes(*notes))
    return FactorFit(
        solution.rank,
        solution.chi2,
        dof,
        jnp.where(dof > 0, p, jnp.nan),
        retained,
        moments,
        withheld,
        reduction.cut,
        _ledger(fitted, data=data),
        _Problem(basis, fitted, data),
        int(chunk_bytes),
        1,
        factors,
        outer_rank,
        dimension,
        best.converged,
        best.iterations,
        int(starts),
        int(joint.rank),
        policy,
    )
