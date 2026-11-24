"""Tests of declared laws, canonical units and their quadrature (SPEC T4)."""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from syncem.base.variables import SYNCHROTRON_VARIABLES as VARS
from syncem.distributions import (
    AzimuthDensity,
    AzimuthFactor,
    Conditional,
    CumulantScreen,
    Delta,
    EmpiricalScreen,
    Free,
    GammaScreen,
    GaussianScreen,
    Isotropic,
    LaplaceScreen,
    LogNormal,
    Maxwellian,
    Normal,
    Tabulated,
    Uniform,
    UniformAngle,
    UniformAzimuth,
    family,
)
from syncem.distributions.densities import cosine_rule
from syncem.units import Quantity, UnitError


def test_placement_checks_units_and_converts_point_angles():
    law = VARS["B"].place(Uniform(4.0, 6.0, unit="uG"))
    np.testing.assert_array_equal(law.support, [4e-6, 6e-6])
    assert law.given_unit == "uG"
    with pytest.raises(UnitError, match="no unit"):
        VARS["B"].place(Uniform(4, 6))
    with pytest.raises(UnitError, match="range of angles"):
        VARS["eta"].place(Uniform(10, 50, unit="deg"))
    assert VARS["eta"].place(Delta("60 deg")).value == pytest.approx(0.5)
    angular = VARS["eta"].place(UniformAngle(30, 60, unit="deg"))
    np.testing.assert_allclose(angular.support, [0.5, np.sqrt(3) / 2])
    assert angular.angle_unit == "deg"
    with pytest.raises(UnitError, match="degrees"):
        AzimuthDensity(0.8, 15)


@pytest.mark.parametrize("rms", [1e-7, 1e-3])
def test_maxwellian_rule_matches_adaptive_field_average(rms):
    law = Maxwellian((rms, "G"))
    field, weights = law.rule()
    assert field.shape == (96,)
    assert np.sum(weights) == pytest.approx(1, abs=5e-15)
    assert np.dot(weights, (field / rms) ** 2) == pytest.approx(1, rel=2e-14)
    # A non-polynomial field observable with support at small B.
    expected = quad(
        lambda t: t * np.exp(-0.1 / t) * rms * float(law.density(t * rms)),
        0.0,
        8.0,
        epsabs=1e-13,
    )[0]
    assert weights @ (field / rms * np.exp(-0.1 * rms / field)) == pytest.approx(
        expected, rel=3e-12
    )


@pytest.mark.parametrize("width", [0.01, 0.05, 1.5])
def test_lognormal_rule_matches_normal_reference_on_its_quadrature_range(width):
    law = LogNormal("5 uG", width, nodes=64)
    x, p = law.rule()
    # Compare with the same six-width truncation in the standard Normal variable.
    baseline = quad(lambda z: np.exp(-0.5 * z**2), -6, 6, epsabs=1e-13)[0]
    expected = (
        quad(
            lambda z: np.exp(-0.5 * z**2) * np.logaddexp(0, width * z),
            -6,
            6,
            epsabs=1e-13,
        )[0]
        / baseline
    )
    assert p @ np.logaddexp(0, np.log(x / 5e-6)) == pytest.approx(expected, rel=2e-14)


def test_normal_and_tabulated_rules_are_probabilities():
    law = Normal(5, 0.5, unit="uG")
    x, w = law.rule()
    assert w @ x == pytest.approx(5e-6)
    law = Tabulated([0, 1, 2, 3, 4], [0, 0, 1, 0, 0], unit="uG")
    x, w = law.rule(128)
    assert w @ x == pytest.approx(2e-6, abs=1e-14)
    assert np.all((x > 1e-6) & (x < 3e-6))
    assert law.density(2e-6) == pytest.approx(1e6)
    np.testing.assert_array_equal(law.density(jnp.array([0.0, 4e-6])), [0.0, 0.0])


def test_tabulated_cdf_plateau_and_uniform_angle_outside_support():
    law = Tabulated([0, 1, 2, 3, 4], [1, 0, 0, 0, 1])
    x, w = law.rule(3)
    assert x[1] == 3.0
    assert np.all(np.isfinite(x)) and np.sum(w) == pytest.approx(1.0)
    angular = UniformAngle(30, 180, unit="deg")
    assert jax.grad(angular.density)(0.99) == 0.0


def test_even_cosine_quadratures_and_pole_branch():
    interior = Uniform(-0.5, 0.5, nodes=12)
    for law in (interior, Isotropic()):
        full = cosine_rule(law, 24)
        folded = cosine_rule(law, 24, half_range=True)

        def f(x):
            return np.sqrt(1 - x**2)

        expected = quad(f, *map(float, law.support), epsabs=1e-13)[0] / (
            law.hi - law.lo
        )
        for nodes, probs in (full, folded):
            assert probs @ f(nodes) == pytest.approx(expected, abs=2e-14)
    with pytest.raises(ValueError, match="even"):
        cosine_rule(Uniform(0, 1), 12, half_range=True)


def test_azimuth_factors_match_independent_rule():
    law = AzimuthDensity(0.8, "15 deg")
    x, w = law.rule()
    for s in (-3, -2, 0, 2, 3):
        np.testing.assert_allclose(w @ np.exp(1j * s * x), law.circular(s), atol=4e-15)
    assert law.circular(2) == pytest.approx(0.4 * np.exp(1j * np.pi / 6))
    assert UniformAzimuth().circular(2) == 0
    with pytest.raises(ValueError, match="modulus"):
        AzimuthFactor(1.1j)


@pytest.mark.parametrize(
    "law",
    [
        GaussianScreen(8, 0.5),
        LaplaceScreen(8, 0.5),
        GammaScreen(8, 0.5),
        GammaScreen(8, 0.5, 0.1),
        GammaScreen(8, 0.5, 1000),
    ],
)
def test_screen_characters_against_quadrature(law):
    x, w = law.rule(96)
    tau = np.array([0.0, 0.2, 0.75])
    np.testing.assert_allclose(
        law.characteristic(tau), np.exp(1j * tau[:, None] * x) @ w, atol=3e-12, rtol=0
    )
    assert w @ x == pytest.approx(8.0, abs=1e-12)
    assert w @ (x - 8) ** 2 == pytest.approx(0.25, abs=3e-12)


def test_empirical_cumulant_and_zero_width_screens():
    x = np.array([0, -2, 3])
    w = np.array([0.1, 0.4, 0.5])
    tau = np.array([[0.0, 1.0], [-3.0, 24.0]])
    empirical = EmpiricalScreen(x, w)
    np.testing.assert_allclose(
        empirical.characteristic(tau), np.exp(1j * tau[..., None] * x) @ w, atol=1e-15
    )
    gaussian = GaussianScreen(4, 0.8)
    np.testing.assert_allclose(
        CumulantScreen([4, 0.8**2, 0, 0]).characteristic(tau),
        gaussian.characteristic(tau),
    )
    assert not hasattr(CumulantScreen([4, 0, 0, 0]), "rule")
    np.testing.assert_array_equal(GaussianScreen(4, 0).rule(), [[4.0], [1.0]])


def test_conditional_parent_values_are_traceable_and_carry_units():
    law = Conditional(
        lambda B, *, width: LogNormal(B, width),
        params={"width": 0.2},
        units={"B": "uG"},
    )

    def first_field(B):
        return law.evaluate({"B": B}, canonical_units={"B": "G"}).rule()[0][0]

    B = 5e-6
    value = jax.jit(first_field)(B)
    assert value == pytest.approx(first_field(B))
    assert jax.grad(first_field)(B) == pytest.approx(value / B)
    assert law.parents == ("B",)
    altered = Conditional(law.fn, params={"width": 0.3}, units=dict(law.units))
    assert law.record()["fingerprint"] != altered.record()["fingerprint"]
    with pytest.raises(ValueError, match="Free"):
        Conditional(lambda x: Uniform(0, x), params={"x": Free()}).evaluate({})


def test_user_family_rule_passes_coordinates_in_the_declared_unit():
    @family(variable="B", support=(0.5, 50), unit="uG")
    def field(B, width=0.05):
        return jnp.exp(-0.5 * (jnp.log(B / 5) / width) ** 2) / B

    law = field(nodes=192)
    assert law.fixed_nodes
    x, w = law.rule()
    assert w @ x == pytest.approx(5e-6 * np.exp(0.05**2 / 2), rel=4e-12)
    assert jax.grad(lambda width: field(width=width, nodes=192).rule()[1] @ x)(0.05) > 0
    assert law.record()["fingerprint"] != field(width=0.1).record()["fingerprint"]
    json.dumps(field(width=Free(0.05)).record(), allow_nan=False)


def test_user_family_cutoff_moving_and_linear_effects():
    @family(variable="x", support=(0, 1), cutoff=lambda cutoff: (cutoff, -0.5))
    def rise(x, cutoff):
        return 1 / jnp.sqrt(cutoff - x)

    law = rise(cutoff=0.6)
    assert not law.fixed_nodes
    x, w = law.rule(32)
    assert w @ x == pytest.approx(0.4, abs=3e-13)
    assert law.panel_rule(0.0, 0.45)[1].sum() == pytest.approx(0.5, abs=2e-13)
    assert jax.grad(
        lambda cut: rise(cutoff=cut).rule(32)[1] @ rise(cutoff=cut).rule(32)[0]
    )(0.6) == pytest.approx(2 / 3)

    @family(variable="x", support=(0, 1), linear=("a", "b"))
    def polynomial(x, a, b):
        return a + b * x

    np.testing.assert_allclose(
        polynomial().linear_density(jnp.array([0.1, 0.2])), [[1, 0.1], [1, 0.2]]
    )


def test_invalid_declarations_and_records():
    with pytest.raises(ValueError, match="strictly increasing"):
        Tabulated([1, 1], [1, 1])
    with pytest.raises(ValueError, match="positive"):
        LogNormal("5 uG", 0)
    with pytest.raises(ValueError, match="positive integer"):
        Uniform(1, 2, nodes=False)
    law = VARS["B"].place(Maxwellian(Free("5 uG"), nodes=96))
    json.dumps(law.record(), allow_nan=False)
    with pytest.raises(ValueError, match="Free"):
        law.rule()
    assert (
        Conditional(lambda x: Uniform(0, x)).evaluate({"x": Quantity(1.0, "")}).hi == 1
    )
