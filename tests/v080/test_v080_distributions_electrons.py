"""Electron-family support, normalisation and the T7 cutoff boundary."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad

from syncem.distributions import (
    AgedPowerLaw,
    BrokenPowerLaw,
    Free,
    PowerLaw,
    PowerLawMixture,
)
from syncem.units import Quantity, UnitError
from syncem.values.errors import RoleError

_PAPER_BOUNDS = {"lo": 300.0, "hi": 2962642.632683605}


@pytest.mark.parametrize(
    "law, expected",
    [
        (PowerLaw(2.6, column=1, **_PAPER_BOUNDS), -5.14049),
        (BrokenPowerLaw(2, 3.2, 7800, column=1, **_PAPER_BOUNDS), -2.96604),
        (
            AgedPowerLaw(
                2.6,
                age="19 Myr",
                loss_field=(np.sqrt(76 / 3), "uG"),
                radiation="1 eV cm-3",
                lo=300,
                column=1,
            ),
            -5.11590,
        ),
    ],
)
def test_paper_coefficient_pins(law, expected):
    assert np.log(law.paper_coefficient().value) == pytest.approx(expected, abs=1e-5)


def test_e3_paper_coefficient_from_unit_column():
    law = PowerLawMixture.cumulant_form(2.6, 0.04, **_PAPER_BOUNDS)
    x, w = law.panel_rule(_PAPER_BOUNDS["lo"], _PAPER_BOUNDS["hi"], 96)
    normalised = PowerLawMixture.cumulant_form(
        2.6, 0.04, amplitude=1 / w.sum(), **_PAPER_BOUNDS
    )
    assert np.log(normalised.paper_coefficient().value) == pytest.approx(
        -5.31605, abs=1e-5
    )
    recovered = PowerLawMixture.from_paper_coefficient(
        normalised.paper_coefficient().value, k1=2.6, k2=0.04, **_PAPER_BOUNDS
    )
    np.testing.assert_allclose(recovered.density(x), normalised.density(x), rtol=3e-15)


@pytest.mark.parametrize("p", [0.8, 1.0, 1.000001, 2.6])
def test_column_normalisation_and_inverse_coefficient(p):
    law = PowerLaw(p, lo=2, hi=300, column=3)
    x, w = law.panel_rule(2, 300, 64)
    assert w.sum() == pytest.approx(3, rel=2e-13)
    a = law.paper_coefficient().value
    restored = PowerLaw.from_paper_coefficient(a, index=p, lo=2, hi=300)
    np.testing.assert_allclose(restored.density(x), law.density(x), rtol=3e-15)
    assert law.rule(48)[1].sum() == pytest.approx(1)
    assert law.density(301.0) == 0.0


def test_paper_coefficient_rejects_negative_density():
    with pytest.raises(ValueError, match="nonnegative"):
        PowerLaw.from_paper_coefficient(-1, index=2.6)


def test_unbounded_power_law_column_and_rule_require_different_support():
    law = PowerLaw(2.6, lo=300, column=3)
    assert quad(lambda x: float(law.density(x)), 300, np.inf)[0] == pytest.approx(
        3, rel=3e-10
    )
    with pytest.raises(RoleError, match="bounded support"):
        law.rule()
    with pytest.raises(ValueError, match="index > 1"):
        PowerLaw(0.8, column=1).density(300.0)


def _reference_cutoff(index, cut, a, b, power):
    upper = min(b, cut)
    if upper <= a:
        return 0.0
    t0, t1 = np.log(a / cut), np.log(upper / cut)

    def smooth(t):
        gamma = cut * np.exp(t)
        ratio = np.expm1(t) / t if t != 0 else 1.0
        return (
            gamma
            * (gamma / 1e4) ** (-index)
            * ratio ** (index - 2)
            * (gamma / b) ** power
        )

    if upper == cut:
        return quad(
            smooth, t0, 0, epsabs=0, epsrel=1e-12, weight="alg", wvar=(0.0, index - 2)
        )[0]
    return quad(
        lambda t: smooth(t) * (-t) ** (index - 2), t0, t1, epsabs=0, epsrel=1e-12
    )[0]


@pytest.mark.parametrize("index", [1.3, 2.6])
@pytest.mark.parametrize("relative_distance", [-1e-3, -1e-6, 0.0, 1e-6, 1e-3])
def test_cutoff_rules_at_and_near_cell_faces(index, relative_distance):
    a, b = 6000.0, 9000.0
    cut = b * (1 + relative_distance)
    law = AgedPowerLaw(index, gamma_cut=cut, lo=300)
    expected = _reference_cutoff(index, cut, a, b, 2)
    for fitting_rule in (False, True):
        x, w = law.panel_rule(a, b, 64, fitted=fitting_rule)
        assert w @ (x / b) ** 2 == pytest.approx(expected, rel=3e-9)
        assert np.all(x <= b)
    assert law.paper_coefficient().value == 1e4
    assert law.density(1e4) == 0.0


def test_cutoff_subthreshold_after_upper_face_and_traceable_fit_rule():
    def panel_integral(cut):
        family = AgedPowerLaw(2.6, gamma_cut=cut, lo=300)
        return family.panel_rule(400.0, 600.0, 64, fitted=True)[1].sum()

    val = jax.jit(panel_integral)(610.0)
    assert val == pytest.approx(
        _reference_cutoff(2.6, 610.0, 400.0, 600.0, 0), rel=3e-12
    )
    slope = (panel_integral(610.001) - panel_integral(609.999)) / 0.002
    assert jax.grad(panel_integral)(610.0) == pytest.approx(slope, rel=3e-9)
    assert panel_integral(310.0) == 0.0
    assert np.isfinite(jax.grad(panel_integral)(310.0))


def test_field_dependent_loss_and_uncut_coefficient_below_pivot():
    kwargs = {"index": 2.6, "age": "19 Myr", "radiation": "1 eV cm-3", "lo": 300}
    with pytest.raises(UnitError, match="unit"):
        AgedPowerLaw(loss_field=5e-6, **kwargs)
    normal = AgedPowerLaw(loss_field="5 uG", **kwargs)
    strong = AgedPowerLaw(loss_field=Quantity(12e-6, "G"), **kwargs)
    perpendicular = AgedPowerLaw(loss_field="5 uG", pitch=0.0, **kwargs)
    parallel = AgedPowerLaw(loss_field="5 uG", pitch=1.0, **kwargs)
    assert perpendicular.gamma_cut < normal.gamma_cut < parallel.gamma_cut
    assert 300 < strong.gamma_cut < 1e4 < normal.gamma_cut
    assert strong.paper_coefficient().value == normal.paper_coefficient().value == 1e4


def test_free_families_serialize_without_tabulation():
    law = AgedPowerLaw(Free(2.6), gamma_cut=Free(1e4), lo=300)
    assert law.record()["parameters"]["explicit_cut"]["kind"] == "free"
    with pytest.raises(ValueError, match="Free"):
        law.rule()
    mix = PowerLawMixture.moment_form(2, 2.6, lo=300, hi=1e6)
    np.testing.assert_allclose(
        mix.basis_density(jnp.array([10.0, 10000.0])), [[0, 0, 0], [1, 0, 0]]
    )


@pytest.mark.parametrize(
    "standard, shifted",
    [
        (PowerLaw(2.6, amplitude=1), PowerLaw(2.6, pivot=1e3, amplitude=10**2.6)),
        (
            AgedPowerLaw(2.6, amplitude=1, gamma_cut=8000.0),
            AgedPowerLaw(2.6, pivot=1e3, amplitude=10**2.6, gamma_cut=8000.0),
        ),
        (
            PowerLawMixture.cumulant_form(2.6, 0.0, amplitude=1),
            PowerLawMixture.cumulant_form(2.6, 0.0, pivot=1e3, amplitude=10**2.6),
        ),
    ],
)
def test_paper_coefficient_independent_of_the_declared_pivot(standard, shifted):
    assert standard.paper_coefficient().value == pytest.approx(
        shifted.paper_coefficient().value
    )
    np.testing.assert_allclose(
        standard.density(jnp.array([300.0, 3000.0])),
        shifted.density(jnp.array([300.0, 3000.0])),
        rtol=2e-14,
    )


def test_zero_radiative_loss_matches_the_uncut_density_and_its_limit():
    def mass(mu):
        model = AgedPowerLaw(
            2.6, age="19 Myr", loss_field="5 uG", pitch=mu, lo=300, column=1
        )
        return model.panel_rule(300.0, 1e4, fitted=True)[1].sum()

    expected = PowerLaw(2.6, lo=300, column=1).panel_rule(300.0, 1e4, 64)[1].sum()
    assert mass(1.0) == pytest.approx(expected, rel=2e-14)
    step = 1e-5
    slope = (3 * mass(1.0) - 4 * mass(1 - step) + mass(1 - 2 * step)) / (2 * step)
    assert jax.grad(mass)(1.0) == pytest.approx(slope, rel=2e-7)
