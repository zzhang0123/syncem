"""Full-paper forward errors for the specified separable distributions."""

import json
import numpy as np
import pytest

import syncem as se
from tests.v080 import _study

EXPECTED = {
    "E1": (2.56e-2, 8.58e-4, 8.76e-5),
    "E2": (3.83e-2, 9.07e-4, 1.16e-4),
    "E3": (2.74e-2, 1.02e-3, 9.64e-5),
    "E4-JP": (1.85e-1, 3.64e-3, 4.47e-4),
}


@pytest.fixture(scope="module")
def paper():
    cfg = _study.paper()
    truths = {}
    path = _study.ACCEPTANCE_DATA / "forward_spectra.npz"
    _study.verify_sums(_study.ACCEPTANCE_DATA / "SHA256SUMS")
    with np.load(path) as source:
        for name in (*EXPECTED, "E1M"):
            truths[name] = np.asarray(source[name.replace("-", "_")])
    ch = se.Channels.log(
        cfg.nu_lo,
        cfg.nu_hi,
        cfg.n_channels,
        unit="Hz",
        half_width=cfg.chan_half,
        n_nu=cfg.n_chan_nodes,
    )
    lat = se.Lattice(
        gamma=se.Cells.log(cfg.gamma_lo, cfg.gamma_hi, cfg.n_bins, order=4)
    )
    bounds = {"lo": cfg.gamma_lo, "hi": cfg.gamma_hi}
    references = _study.load().ensembles.reference_ensembles()
    laws = {
        "E1": se.PowerLaw(2.6, column=1, **bounds),
        "E2": se.BrokenPowerLaw(2, 3.2, 7800, column=1, **bounds),
        "E3": se.PowerLawMixture.from_paper_coefficient(
            references["E3"].norm, k1=2.6, k2=0.04, **bounds
        ),
        "E4-JP": se.AgedPowerLaw(
            2.6, gamma_cut=references["E4-JP"].gamma_c, lo=cfg.gamma_lo, column=1
        ),
    }
    b = se.Uniform(cfg.b_lo, cfg.b_hi, unit="uG", nodes=cfg.n_b)
    # The independent continuum integrands are even in eta on this setup.
    eta = se.Uniform(0, cfg.eta_max, nodes=cfg.n_eta)
    pop = se.Population(
        gamma=laws["E1"],
        B=b,
        eta=eta,
        depth=se.GaussianScreen(cfg.screen_mean, cfg.screen_sd),
        phi=se.AzimuthFactor(cfg.azimuth_factor),
    )
    return truths, ch, lat, laws, pop


def _error(out, direct):
    return np.max(np.abs(out - direct) / direct[:, :1])


def test_specified_separable_forward_errors(paper):
    truths, ch, lat, laws, pop = paper
    basis = se.spectral_basis(pop, ch, lattice=lat, check=False)
    for order_index, order in enumerate((0, 2, 4)):
        routed = basis.with_order(order)
        for case, coefficients in EXPECTED.items():
            predicted = routed.predict(pop.replace(gamma=laws[case]), check=False)
            assert _error(predicted.stokes, truths[case]) == pytest.approx(
                coefficients[order_index], rel=0.01
            )


def test_maxwellian_field_forward_errors(paper):
    truths, ch, lat, _, pop = paper
    maxwellian = pop.replace(B=se.Maxwellian("5 uG", nodes=96))
    uniform = se.spectral_basis(pop, ch, lattice=lat, check=False)
    matched = se.spectral_basis(maxwellian, ch, lattice=lat.with_order(2), check=False)
    direct = truths["E1M"]
    fitted = matched.predict(maxwellian, check=False)
    assert _error(fitted.stokes, direct) == pytest.approx(6.18e-4, rel=0.01)
    for order, expected in ((2, 3.84e-2), (4, 3.76e-2)):
        out = uniform.with_order(order).predict(
            maxwellian, check=False, allow_mismatch=True
        )
        assert _error(out.stokes, direct) == pytest.approx(expected, rel=0.01)


def test_quarter_cell_reference_shifts(paper):
    truths, _, lat, _, _ = paper
    data = _study.ACCEPTANCE_DATA
    sweep = json.loads((data / "sensitivity_results.json").read_text())
    config = sweep["config"]
    channels = se.Channels.log(
        config["nu_lo"],
        config["nu_hi"],
        config["n_channels"],
        unit="Hz",
        n_nu=config["n_chan_nodes"],
        half_width=config["chan_half"],
    )
    cases = {"E1": se.paper.e1(), "E2": se.paper.e2(), "E4-JP": se.paper.e4()}
    for shift in (-0.5, 0.5):
        shifted = lat.shift(gamma=shift)
        basis = se.spectral_basis(
            cases["E1"], channels, lattice=shifted, stokes="I", check=False
        )
        for row in sweep["sensitivity"]:
            name, order = row["name"], row["order"]
            if not (name in cases and order in (2, 4) and row["offset"] == shift):
                continue
            out = basis.with_order(order).predict(cases[name], check=False)
            truth = truths[name][:, :1]
            assert _error(out.stokes, truth) == pytest.approx(
                row["forward_max"], rel=0.01
            )
