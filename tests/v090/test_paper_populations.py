"""Paper extensions preserve the common densities and their physical conditionals."""

import numpy as np
import pytest

from syncem import paper
from syncem.front._regime_paper import population
from syncem.population.declaration import law_at


@pytest.mark.parametrize(
    "name,legacy,points",
    [
        ("E1", paper.e1, ({},)),
        ("E1M", paper.e1m, ({},)),
        ("E2", paper.e2, ({},)),
        ("E3", paper.e3, ({},)),
        ("E4-JP", paper.e4, ({},)),
        ("E4-KP", paper.e4kp, ({"mu": -0.6}, {"mu": 0.0}, {"mu": 0.9})),
        ("E5", paper.e5, ({},)),
        ("E6", paper.e6, ({"B": 4e-6}, {"B": 5e-6}, {"B": 6e-6})),
    ],
)
def test_extension_preserves_in_band_density(name, legacy, points):
    original = legacy()
    extended = population(name)
    assert len(extended.parts) == len(original.parts)
    for new_part, old_part in zip(extended.parts, original.parts, strict=True):
        assert new_part.symmetries == old_part.symmetries
        assert new_part.column_factor == old_part.column_factor
        for key in old_part.factors.keys() - {"gamma"}:
            assert new_part.factors[key].record() == old_part.factors[key].record()
        new_gamma = new_part.factors["gamma"]
        old_gamma = old_part.factors["gamma"]
        for point in points:
            new = law_at(new_gamma, point, {"B": "G"})
            old = law_at(old_gamma, point, {"B": "G"})
            assert new.support == (100.0, old.support[1])
            sample = np.geomspace(300.0, float(old.support[1]) * 0.95, 11)
            np.testing.assert_array_equal(new.density(sample), old.density(sample))
            assert new.paper_coefficient().value == old.paper_coefficient().value
            assert old.density(150.0) == new.density(99.0) == 0.0
            assert new.density(100.0) > 0
            assert new.density(150.0) > 0


def test_original_column_is_retained_above_the_switch():
    pop = population("e1", gamma_min=100)
    gamma = pop.factors["gamma"]
    _, lower = gamma.panel_rule(100.0, 300.0)
    _, upper = gamma.panel_rule(300.0, gamma.hi)
    expected = (100**-1.6 - 300**-1.6) / (300**-1.6 - float(gamma.hi) ** -1.6)
    assert np.sum(upper) == pytest.approx(1.0, rel=1e-13)
    assert np.sum(lower) == pytest.approx(expected, rel=1e-13)


def test_conditional_cutoff_configuration_retains_original_dependencies():
    for name, legacy in (("E4-KP", paper.e4kp), ("E6", paper.e6)):
        original = legacy().factors["gamma"]
        law = population(name, gamma_min=220).factors["gamma"]
        assert law.parents == original.parents
        assert law.units == original.units
        assert law.parity == original.parity
        assert law.params["gamma_min"] == 220
        assert law.record()["fingerprint"] is not None
        assert law.record() != population(name).factors["gamma"].record()


def test_original_lower_limit_selects_original_preset():
    original = paper.e5()
    same = population("E5", gamma_min=300)
    assert same.to_dict() == original.to_dict()


@pytest.mark.parametrize("gamma_min", [True, 0, 300.1, np.inf, np.nan])
def test_invalid_extension_limit_is_rejected(gamma_min):
    with pytest.raises(ValueError, match="gamma_min"):
        population("E1", gamma_min=gamma_min)


def test_unknown_ensemble_is_rejected():
    with pytest.raises(ValueError, match="unknown paper ensemble"):
        population("E7")
