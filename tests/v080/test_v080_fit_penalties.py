"""Ridge, nonnegative ridge and noise-only profiles in whitened units."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syncem.fitting.penalties import (
    Penalty,
    profile_max,
    ridge,
    ridge_prediction,
    solve_nonnegative,
    solve_ridge,
)
from syncem.fitting.rank import RANK_RTOL, reduce_svd


def test_ridge_batched_covariance_and_unconstrained_against_normal_equations():
    design = np.array([[4.0, 2.0], [1.0, -2.0], [0.5, 1.0]])
    data = np.array([[5.0, 1.0, 2.0], [1.0, 2.0, -1.0]])
    alpha = 0.7
    penalty = Penalty(alpha)
    reduction = reduce_svd(design)
    found = jax.jit(lambda d: solve_ridge(reduction, d, penalty))(data)
    repeated = jax.vmap(lambda d: solve_ridge(reduction, d, penalty))(data)
    for calculated, expected in zip(found, repeated, strict=True):
        np.testing.assert_allclose(calculated, expected, rtol=5e-14)
    gram = design.T @ design
    estimator = np.linalg.solve(gram + alpha * np.eye(2), design.T)
    moments = data @ estimator.T
    residual = data - moments @ design.T
    np.testing.assert_allclose(found.moments, moments, rtol=2e-14)
    np.testing.assert_allclose(found.chi2, np.sum(residual**2, axis=-1))
    np.testing.assert_allclose(found.effective, np.trace(estimator @ design))

    rows = np.array([[2.0, -0.5], [1.0, 3.0], [0.0, 0.0]])
    prediction = ridge_prediction(rows, reduction, found, cov=True)
    gain = rows @ estimator
    covariance = gain @ gain.T
    fraction = np.linalg.norm(rows - rows @ estimator @ design, axis=1)
    fraction /= np.where(
        np.linalg.norm(rows, axis=1) > 0, np.linalg.norm(rows, axis=1), 1
    )
    np.testing.assert_allclose(prediction.values, moments @ rows.T)
    np.testing.assert_allclose(
        prediction.covariance, np.broadcast_to(covariance, (2, 3, 3))
    )
    np.testing.assert_allclose(
        prediction.sigma, np.broadcast_to(np.sqrt(np.diag(covariance)), (2, 3))
    )
    np.testing.assert_allclose(
        prediction.unconstrained, np.broadcast_to(fraction, (2, 3))
    )


def test_upre_minimises_the_archived_grid_against_direct_ridge_solves():
    design = np.diag([9.0, 2.0, 0.5])
    data = np.array([[9, -1, 0.1], [0.02, 0.3, 5], [0.2, -0.1, 0.9]])
    grid = 81 * 10.0 ** np.linspace(-30.0, 2.0, 321)
    gram = design.T @ design
    solvers = np.stack(
        [np.linalg.solve(gram + alpha * np.eye(3), design.T) for alpha in grid]
    )
    moments = np.einsum("gnk,dk->dgn", solvers, data)
    residual = data[:, None, :] - moments @ design.T
    edf = np.trace(design @ solvers, axis1=-2, axis2=-1)
    winners = np.argmin(np.sum(residual**2, axis=-1) + 2 * edf[None, :], axis=1)
    indices = np.arange(len(data))
    found = ridge(design, data)
    np.testing.assert_allclose(found.alpha, grid[winners], rtol=2e-14)
    np.testing.assert_allclose(found.moments, moments[indices, winners], rtol=2e-14)
    np.testing.assert_allclose(found.effective, edf[winners], rtol=2e-14)


def test_no_penalty_respects_the_strict_numerical_cut_and_null_rows():
    design = np.diag([1.0, 2 * RANK_RTOL, RANK_RTOL, 0.0])
    data = [3.0, 4 * RANK_RTOL, 2 * RANK_RTOL, 9.0]
    reduction = reduce_svd(design)
    solution = solve_ridge(reduction, data, Penalty(0))
    np.testing.assert_array_equal(solution.factors, [1, 1, 0, 0])
    np.testing.assert_allclose(solution.moments, [3, 2, 0, 0])
    prediction = ridge_prediction(np.eye(4), reduction, solution)
    np.testing.assert_array_equal(prediction.unconstrained, [0, 0, 1, 1])
    assert solution.effective == 2
    zero = ridge(jnp.zeros((3, 2)), data[:3])
    np.testing.assert_array_equal(zero.moments, [0, 0])
    assert zero.alpha == zero.effective == 0


@pytest.mark.parametrize("alpha", (-1, np.inf, np.nan, True))
def test_invalid_penalties_are_refused(alpha):
    with pytest.raises(ValueError, match="alpha"):
        Penalty(alpha)


def test_nonnegative_ridge_penalises_moments_instead_of_point_masses():
    design = np.array([[4.0, 0.0], [0.0, 3.0]])
    points = np.array([[1.0, 1.0], [-0.25, 2.0]])
    known = np.array([1.2, 2.8])
    data = design @ points @ known
    alpha = 0.9
    gram = design.T @ design
    expected = np.linalg.solve(gram + alpha * np.eye(2), design.T @ data)
    out = solve_nonnegative(design, data, points, Penalty(alpha))
    np.testing.assert_allclose(out.moments, expected, rtol=1e-14)
    np.testing.assert_allclose(out.weights, np.linalg.solve(points, expected))
    np.testing.assert_allclose(
        out.effective, np.trace(np.linalg.solve(gram + alpha * np.eye(2), gram))
    )
    assert out.chi2 == pytest.approx(np.sum((design @ expected - data) ** 2))
    exact = solve_nonnegative(design, data, points, Penalty(0.0))
    np.testing.assert_allclose(exact.weights, known)
    assert exact.effective == 2


def test_nonnegative_active_space_and_unused_masses():
    design = np.diag([2.0, 3.0])
    data = [6.0, -2.0]
    points = np.column_stack((np.eye(2), np.zeros(2)))
    out = solve_nonnegative(design, data, points, Penalty(2.0))
    np.testing.assert_array_equal(out.weights, [2, 0, 0])
    assert out.effective == pytest.approx(4 / 6)
    assert out.chi2 == pytest.approx(8.0)
    selected = solve_nonnegative(design, data, points)
    assert selected.alpha == float(ridge(design, data).alpha)
    assert np.all(selected.weights >= 0)


def test_profile_matches_analytic_chi2_ball_with_irreducible_residual():
    design = [[2.0], [0.0]]
    data = [6.0, 2.0]
    points = [[1.0]]
    out = profile_max(design, data, points, [[1.0], [-1.0]], delta_chi2=1.0)
    assert out.minimum_chi2 == pytest.approx(4.0)
    np.testing.assert_allclose(out.value, [3.5, -2.5], atol=1e-9)
    np.testing.assert_allclose(out.chi2, [5, 5], atol=1e-9)
    clipped = profile_max(design, data, points, [-1], delta_chi2=49)
    assert clipped.value == pytest.approx(0.0, abs=1e-10)
    assert clipped.weights.shape == (1,)
    assert clipped.chi2 <= clipped.minimum_chi2 + 49
    boundary = profile_max(design, data, points, [1.0], delta_chi2=0.0)
    assert boundary.value == pytest.approx(3.0)
    assert boundary.chi2 == boundary.minimum_chi2


@pytest.mark.parametrize("delta", (0.0, 1.0))
def test_profile_detects_a_positive_null_direction_but_profiles_bounded_rows(delta):
    out = profile_max(
        [[1.0, -1.0]], [2.0], np.eye(2), [[1.0, 1.0], [1.0, -1.0]], delta_chi2=delta
    )
    assert np.isposinf(out.value[0])
    assert np.isnan(out.weights[0]).all()
    assert out.value[1] == pytest.approx(2.0 + np.sqrt(delta), abs=1e-9)
    zero = profile_max([[1.0, 0.0]], [2.0], np.eye(2), [0.0, 1.0], delta_chi2=delta)
    assert np.isposinf(zero.value)
