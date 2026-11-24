"""Reference preset constructors checked against the frozen classical study."""

import json

import numpy as np
import pytest

from syncem import paper
from syncem.distributions import Maxwellian
from syncem.population.declaration import law_at
from tests.v080 import _study

_REL = 3e-13


def test_channels_and_cells_match_the_study():
    cfg = _study.paper()
    channels, cells = paper.channels(), paper.cells(order=4)
    np.testing.assert_allclose(channels.nodes, cfg.chan_nu, rtol=1e-14)
    np.testing.assert_allclose(
        channels.response_weights(),
        np.broadcast_to(cfg.chan_w, channels.nodes.shape),
        atol=1e-14 * np.max(cfg.chan_w),
    )
    np.testing.assert_allclose(cells.edges, cfg.edges, rtol=1e-14)
    np.testing.assert_allclose(cells.reference, cfg.gamma_b, rtol=1e-14)
    assert all(order == 4 for order in np.asarray(cells.order))


def test_extrapolation_channels_follow_the_gallery_grids():
    band, dense = paper.extrapolation_channels(n_nu=64)
    centres = np.asarray(band.centres_hz)
    assert band.nodes.shape == (32, 64)
    assert dense.nodes.shape == (121, 64)
    np.testing.assert_allclose(centres[:21], np.geomspace(1e9, 4e10, 21), rtol=1e-14)
    np.testing.assert_allclose(
        dense.centres_hz, np.geomspace(1e9, 3e11, 121), rtol=1e-14
    )


def test_common_specified_factors_match_the_study():
    cfg = _study.paper()
    factors = paper.specified()
    f, w = factors["B"].rule()
    b, wb = cfg.b_nodes_weights
    np.testing.assert_allclose(f, b * 1e-6, rtol=1e-14)
    np.testing.assert_allclose(w, wb, rtol=1e-14)
    assert factors["eta"].support == (-cfg.eta_max, cfg.eta_max)
    assert factors["mu"].support == (-1.0, 1.0)
    assert factors["phi"].circular(2) == pytest.approx(cfg.azimuth_factor, rel=1e-15)
    assert factors["depth"].mean == cfg.screen_mean
    assert factors["depth"].sigma == cfg.screen_sd


@pytest.mark.parametrize(
    "factory, ensemble",
    [(paper.e1, "E1"), (paper.e2, "E2"), (paper.e3, "E3"), (paper.e4, "E4-JP")],
)
def test_separable_family_densities_match_study(factory, ensemble):
    reference = _study.study().ensembles.reference_ensembles()[ensemble]
    pop = factory()
    law = pop.factors["gamma"]
    assert law.paper_coefficient().value == pytest.approx(reference.norm, rel=_REL)
    g = np.geomspace(reference.lo, 0.99 * reference.hi, 21)
    np.testing.assert_allclose(law.density(g), reference.density(g), rtol=_REL)
    json.dumps(pop.to_dict(), allow_nan=False)


def test_maxwellian_field_rule():
    law = paper.e1m().factors["B"]
    b, w = _study.paper_maxwell().b_nodes_weights
    x, p = law.rule()
    assert isinstance(law, Maxwellian)
    # The law converts the support to gauss before mapping the nodes.
    np.testing.assert_allclose(
        x, b * 1e-6, rtol=1e-14, atol=np.finfo(float).eps * law.support[1]
    )
    np.testing.assert_allclose(p, w, atol=2e-15)


def test_kp_preset_has_the_studys_common_amplitude():
    reference = _study.study().ensembles.reference_ensembles()["E4-KP"]
    population = paper.e4kp()
    assert population.symmetries == ("pitch_reversal",)
    law = population.factors["gamma"]
    assert law.parents == ("mu",)
    for mu in (-0.5, 0.0, 0.7, 1.0):
        local = law_at(law, {"mu": mu})
        cut = reference.ageing.gamma_c_kp(mu)
        assert local.gamma_cut == pytest.approx(cut, rel=_REL)
        assert local.paper_coefficient().value == pytest.approx(
            reference.norm_unscaled_inv, rel=_REL
        )
        g = np.geomspace(reference.lo, 0.99 * cut, 7)
        expected = 2 * reference.norm_unscaled_inv * reference._shape(g, cut)
        np.testing.assert_allclose(local.density(g), expected, rtol=_REL)


def test_two_zones_reproduce_the_study_crossing_fractions():
    a, b = paper.e5().parts
    reference = _study.study().ensembles.reference_ensembles()["E5"]
    # The study stores the fractions of its equal-intensity channel in the
    # forward result. Rebuilding its kernel table is unnecessary here.
    source = _study.STUDY / "forward_results.json"
    expected_hash = _study.read_sums(_study.STUDY / "SHA256SUMS")[source.name]
    assert _study.sha256_file(source) == expected_hash
    result = json.loads(source.read_text())
    fa, fb, _ = result["ensembles"]["E5"]["fractions_A_B_crossing_channel"]
    for part, expected, fraction, spin in zip(
        (a, b), reference.zones, (fa, fb), reference.factors()
    ):
        gamma, phi = part.factors["gamma"], part.factors["phi"]
        assert gamma.index == expected.p
        assert gamma.column == fraction
        assert phi.circular(2) == pytest.approx(spin, rel=1e-15)


def test_e6_common_amplitude_and_local_field_dependence():
    study = _study.study()
    law = paper.e6().factors["gamma"]
    assert law.parents == ("B",)
    a = study.e6.a_unit()
    for b in (4.0, 5.0, 6.0):
        local = law_at(law, {"B": b * 1e-6}, {"B": "G"})
        cut = study.cost.gamma_c_local(b)
        assert local.gamma_cut == pytest.approx(cut, rel=_REL)
        assert local.paper_coefficient().value == pytest.approx(a, rel=2e-11)
