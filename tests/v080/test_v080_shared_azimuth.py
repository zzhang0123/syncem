"""Whitened fixed-rank profiling of a shared azimuth factor (SPEC Sec. 6.1)."""

import json

import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import build_from_averaged
from syncem.fitting.data import StokesData
from syncem.fitting.rank import Rank, UPRE
from syncem.fitting.shared_azimuth import fit_shared_azimuth
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.values.errors import DataError
from tests.acceptance.test_acceptance_ranks import build
from tests.v080 import _study


@kernel(
    variables=(Variable("x"), Variable("angle", kind="angle", unit="rad")),
    stokes=("I", "P"),
    azimuth="angle",
)
def polarized_template(nu, x):
    intensity = x + nu / 1e9
    return {"I": intensity, "P": 0.5 * intensity}


@pytest.fixture(scope="module")
def template_basis():
    channels = Channels.at([1.0, 2.0, 3.0, 5.0], unit="GHz")
    lattice = Lattice(x=Cells.linear(2.0, 4.0, 1, order=0))
    return build_from_averaged(
        AveragedKernel(polarized_template), channels, lattice, azimuth_factor=1j
    )


def test_factor_profile_with_correlated_noise_and_withholding(template_basis):
    b = template_basis
    intensity = 1.25 * np.asarray(b.design())[:, 0, 0]
    factor = 0.31 - 0.27j
    stokes = intensity[:, None] * [1, 0.5 * factor.real, 0.5 * factor.imag]
    cov = 0.2 * np.eye(stokes.size) + 0.03 * np.ones((stokes.size, stokes.size))
    data = StokesData(b.channels, stokes=stokes, cov=cov, unit="cgs").withhold([1])

    fit = fit_shared_azimuth(b, data, rank=1)

    assert fit.factor == pytest.approx(factor, abs=1e-7)
    assert fit.params == {"angle.factor": fit.factor}
    assert fit.rank == 1 and fit.effective_parameters == 3 and fit.dof == 6
    assert fit.chi2 < 1e-12
    assert fit.withheld.chi2 < 1e-12
    assert fit.converged and not fit.at_bound
    assert fit.curvature_step == pytest.approx(1e-3)
    assert np.min(np.linalg.eigvalsh(fit.profile_curvature)) > 0
    assert fit.problem.basis.key != b.key
    assert dict(fit.problem.basis.record.keys)["design"] == fit.problem.basis.key
    assert np.asarray(b.azimuth_factor) == 1j
    prediction = fit.predict(cov=True)
    assert any("conditional on" in note for note in prediction.plan.notes)
    np.testing.assert_allclose(prediction.stokes, stokes, atol=1e-7)
    trial = Channels.at([1.5, 6.0], unit="GHz")
    np.testing.assert_allclose(
        fit.predict(trial).P / fit.predict(trial).I, 0.5 * factor, atol=1e-7
    )


def test_profile_respects_physical_boundary(template_basis):
    b = template_basis
    intensity = np.asarray(b.design())[:, 0, 0]
    factor = 1.4 * np.exp(0.56j)
    stokes = intensity[:, None] * [1, 0.5 * factor.real, 0.5 * factor.imag]
    data = StokesData(b.channels, stokes=stokes, sigma=1.0, unit="cgs")
    fit = fit_shared_azimuth(b, data, rank=Rank(1))
    assert fit.factor == pytest.approx(factor / abs(factor), abs=1e-7)
    assert abs(fit.factor) <= 1 + 1e-15
    assert fit.at_bound
    assert fit.profile_curvature is None
    assert fit.chi2 > 0


def test_fit_reports_missing_azimuth_information(template_basis):
    b = template_basis
    data = StokesData(b.channels, I=np.ones(4), sigma=1.0, unit="cgs")
    with pytest.raises(DataError, match="distinguish both"):
        fit_shared_azimuth(b, data, rank=1)


def test_polarization_only_cannot_fix_factor_magnitude(template_basis):
    b = template_basis
    data = StokesData(b.channels, Q=np.ones(4), U=np.ones(4), sigma=1.0, unit="cgs")
    with pytest.raises(DataError, match="azimuth magnitude and moment scale are tied"):
        fit_shared_azimuth(b, data, rank=1)


@pytest.mark.parametrize("policy", [None, Rank(7)])
def test_e5_shared_azimuth_matches_the_study(policy):
    cfg = _study.paper()
    reference = json.loads((_study.STUDY / "fits_results.json").read_text())
    archived = reference["E5"]["separable_free_factor"]["seeded"]
    lattice = Lattice(gamma=Cells.log(300, cfg.gamma_hi, cfg.n_bins, order=2))
    basis = build(cfg, lattice, "lookup", 0j)
    with np.load(_study.ACCEPTANCE_DATA / "forward_spectra.npz") as arrays:
        truth = arrays["E5"].copy()
    fitted = cfg.fit_idx
    held = np.setdiff1d(np.arange(cfg.n_channels), fitted)
    sigma = 0.01 * truth[:, :1]
    observed = truth.copy()
    observed[fitted] += sigma[fitted] * cfg.draw_noise(len(fitted))
    observed[held] += sigma[held] * cfg.draw_noise(len(held), seed=cfg.seed + 1)
    data = StokesData(basis.channels, stokes=observed, sigma=sigma, unit="cgs")

    result = fit_shared_azimuth(basis, data.withhold(held), rank=policy)

    assert result.rank == 7 and result.dof == 54
    assert result.chi2 == pytest.approx(archived["chi2_fit"], abs=1e-3)
    assert result.factor == pytest.approx(complex(*archived["c"]), abs=1e-3)
    assert result.withheld.chi2 == pytest.approx(archived["chi2_held"], abs=1e-3)
    assert result.withheld.chi2_predictive == pytest.approx(
        archived["chi2_held_predictive_fixed_c"], abs=1e-3
    )
    err = np.max(np.abs(result.predict().stokes - truth) / truth[:, :1])
    assert err == pytest.approx(
        archived["max_fit_minus_direct_over_I"]["value"], abs=1e-3
    )
    if policy is None:
        assert result.rank_policy == UPRE()
        assert min(result.upre_curve, key=lambda item: item[1])[0] == archived["R_star"]
        assert dict(result.upre_curve)[archived["R_star"]] == pytest.approx(
            archived["chi2_fit"] + 2 * (archived["R_star"] + 2), abs=1e-3
        )
    else:
        assert not result.upre_curve
