"""Conditional cell coefficients, retained shifts and sampled residuals."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.polynomial import Polynomial

from syncem.engine import _jet_routine
from syncem.engine.conditional_columns import (
    ChildCellRule,
    conditional_coefficients,
    conditional_contraction,
)
from syncem.engine.provider_jet import taylor_coefficients
from syncem.values.errors import RoleError


def test_child_cell_quadrature_keeps_the_chart_jacobian_and_actual_faces():
    powers = ((0, 0), (1, 0), (0, 2))
    child = ChildCellRule(
        {"x": ("log", 2.0, 5.0, 3.1, 0.4), "mu": ("affine", -1, 1, 0.0, 1.0)},
        powers,
        nodes=12,
    )
    reference_log_moment = (5 * (np.log(5 / 3.1) - 1) - 2 * (np.log(2 / 3.1) - 1)) / (
        3 * 0.4
    )

    def density(point, nodes):
        return 1 / 6

    full = child.moments(density, {})
    chunked = child.moments(density, {}, chunk_bytes=2048)
    for values in (full, chunked):
        np.testing.assert_allclose(values, [1, reference_log_moment, 1 / 3], rtol=1e-14)


def test_taylor_h_for_mixed_parent_and_child_charts_matches_the_exact_integrals():
    axes = {"a": ("affine", 3.0, 0.4), "b": ("log", 2.0, 0.5)}
    child = ChildCellRule({"x": ("affine", -0.25, 0.75, 0.1, 0.3)}, ((0,), (1,), (2,)))
    shifts = ((0, 0), (1, 0), (0, 2), (1, 1), (0, 1))

    def density(point, nodes):
        u = (point["a"] - 3) / 0.4
        v = jnp.log(point["b"] / 2) / 0.5
        x = nodes["x"]
        return (
            1
            + 0.2 * u * x
            + 0.1 * v**2 * (3 * x**2 - 1) / 2
            + point["scale"] * u * v * x**3
        ) / 2

    polynomials = [
        Polynomial([0.5]),
        Polynomial([0, 0.1]),
        Polynomial([-0.025, 0, 0.075]),
        Polynomial([0, 0, 0, 0.15]),
        Polynomial([0]),
    ]
    monomial = Polynomial([-0.1, 1]) / 0.3
    expected = np.empty((len(shifts), len(child.powers)))
    for i, poly in enumerate(polynomials):
        for j, (power,) in enumerate(child.powers):
            integral = (poly * (monomial**power)).integ()
            expected[i, j] = integral(0.75) - integral(-0.25)

    coeff = conditional_coefficients(density, axes, child, shifts, fixed={"scale": 0.3})
    split = conditional_coefficients(
        density, axes, child, shifts, fixed={"scale": 0.3}, chunk_bytes=1100
    )
    for actual in (coeff, split):
        np.testing.assert_allclose(actual, expected, atol=4e-16)

    point = {"a": 3.2, "b": 2 * np.exp(-0.225), "scale": 0.3}
    terms = np.prod(np.asarray([0.5, -0.45]) ** np.asarray(shifts), axis=1)
    np.testing.assert_allclose(terms @ coeff, child.moments(density, point), atol=4e-16)


def test_shifted_correlation_keeps_only_parent_rows_and_respects_weight_ratio():
    shifts = ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1))
    parent = ((0, 0), (1, 0), (0, 1), (1, 1))
    rows = (((1, 0), 0), ((0, 0), 1), ((0, 1), 1))
    h = np.array([[1, 0.25], [0.2, 0], [0, 0.3], [0.1, 0], [0, 0.4]])
    node = conditional_contraction(
        h, shifts, parent, rows, parent_weight=0.4, joint_weight=0.1
    )
    p = jnp.array([2, 0.5, -0.2, 0.1])
    expected = np.array([2.0, 1.92, -0.2])
    np.testing.assert_allclose(node.apply(p), expected, atol=2e-15)
    np.testing.assert_allclose(
        jax.jit(node.apply)(jnp.stack([p, 2 * p])),
        np.stack([expected, 2 * expected]),
        atol=2e-15,
    )
    np.testing.assert_array_equal(node.dropped_rows, [0, 0, 2, 2])
    np.testing.assert_array_equal(node.dropped_powers, [[2, 0], [3, 0], [0, 2], [1, 2]])
    np.testing.assert_array_equal(node.dropped_coefficients, [0.8, 0.4, 1.2, 1.6])


def test_sampled_truncation_residual_preserves_shifted_face_and_uses_estimate_kind():
    node = conditional_contraction(
        [[1], [1], [1]],
        ((0,), (1,), (2,)),
        ((0,), (1,)),
        (((0,), 0),),
        parent_weight=2,
        joint_weight=0.5,
    )
    points = np.array([[-0.4], [0.2], [1.1]])
    exact = 1 + points + points**2
    estimate = node.estimate(points, exact, chunk_bytes=128)
    assert estimate.kind == "estimate" and estimate.samples == 3
    np.testing.assert_allclose(estimate.value, [2.42], atol=8e-16)
    report = node.estimate(points, exact, design=np.array([[0.5], [2.5]]))
    np.testing.assert_allclose(report.value, [1.21, 6.05], atol=2e-15)


def test_unbounded_faces_missing_density_and_small_memory_budgets_refuse():
    args = {"x": ("affine", 0, 1, 0.5, 0.5)}, ((0,), (1,))
    child = ChildCellRule(*args)
    with pytest.raises(RoleError, match="density or average="):
        child.moments(None, {})
    with pytest.raises(MemoryError, match="one child node"):
        child.moments(lambda p, x: 1, {}, chunk_bytes=2)
    with pytest.raises(ValueError, match="finite faces"):
        ChildCellRule({"x": ("log", 1, np.inf, 5, 1)}, ((0,),))
    with pytest.raises(MemoryError, match="byte budget"):
        conditional_contraction(
            [[1]],
            ((0,),),
            ((0,),),
            (((0,), 0),),
            parent_weight=1,
            joint_weight=1,
            max_bytes=1,
        )


@pytest.mark.skipif(
    not hasattr(_jet_routine, "NESTED_MAX_ORDER"),
    reason="needs the step-5c Taylor engine",
)
def test_order_five_comparison_and_order_six_taylor_dispatch():
    from syncem.engine._taylor_jet import taylor_mode_coefficients

    child = ChildCellRule({"x": ("affine", -1, 1, 0, 1)}, ((0,), (1,)), nodes=8)
    axes = {"p": ("affine", 0, 1)}

    def density(point, nodes):
        p, x = point["p"], nodes["x"]
        return (1 + 0.25 * x * jnp.exp(p)) / 2

    def f(z):
        return child.moments(density, {"p": z[0]})

    order_five = tuple((k,) for k in range(6))
    nested = taylor_coefficients(f, 1, order_five)
    taylor = taylor_mode_coefficients(f, 1, order_five)
    np.testing.assert_allclose(taylor, nested, atol=3e-16)
    order_six = (*order_five, (6,))
    produced = conditional_coefficients(density, axes, child, order_six)
    np.testing.assert_allclose(produced[:-1], nested, atol=3e-16)
    np.testing.assert_allclose(produced[6], [0, 1 / 8640], atol=1e-18)
