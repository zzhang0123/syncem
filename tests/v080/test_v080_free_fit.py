"""Free-moment estimation and predictions on prebuilt bases (SPEC Sec. 6)."""

import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import build_from_averaged
from syncem.fitting.data import StokesData
from syncem.fitting.free_fit import fit_free
from syncem.fitting.rank import FullRank, Rank
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import AzimuthHarmonic
from tests.acceptance.test_acceptance_ranks import build
from tests.v080 import _study


@kernel(variables=(Variable("x", kind="real"),), stokes=("I",))
def _linear_kernel(nu, x):
    return {"I": 2.0 + (nu / 1e9) * x}


@pytest.fixture(scope="module")
def basis():
    channels = Channels.at([1.0, 2.0, 4.0], unit="GHz")
    lattice = Lattice(x=Cells.linear(1.0, 3.0, 1, order=1), weights=[0.3])
    return build_from_averaged(
        AveragedKernel(_linear_kernel), channels, lattice, cache=False
    )


def test_covariance_and_unconstrained_rows_against_explicit_estimator(basis):
    design = np.asarray(basis.design())[:, 0, :]
    y = np.array([2.0, 2.5, 5.0])
    cov = np.array([[1.0, 0.4, -0.1], [0.4, 4.0, 0.5], [-0.1, 0.5, 2.0]])
    data = StokesData(basis.channels, I=y, cov=cov, unit="cgs").withhold([1])
    fit = fit_free(basis, data, rank=Rank(1))
    fitted = [0, 2]
    c = cov[np.ix_(fitted, fitted)]
    linv = np.linalg.inv(np.linalg.cholesky(c))
    u, s, vt = np.linalg.svd(linv @ design[fitted], full_matrices=False)
    k = vt[:1].T @ ((u[:, :1].T @ linv) / s[0])
    moment = k @ y[fitted]
    p = design @ k
    expected_cov = p @ c @ p.T
    fitted_residual = linv @ (design[fitted] @ moment - y[fitted])
    expected_chi2 = float(fitted_residual @ fitted_residual)
    row_remainder = design - (design @ vt[:1].T) @ vt[:1]
    missing = np.linalg.norm(row_remainder, axis=-1) / np.linalg.norm(design, axis=-1)

    assert fit.moments.origin == "minimum_norm"
    assert fit.rank == 1 and fit.dof == 1
    np.testing.assert_allclose(fit.moments.values, moment)
    np.testing.assert_allclose(fit.chi2, expected_chi2)
    assert 0.0 < float(fit.p_value) <= 1.0
    predicted = fit.predict(cov=True)
    np.testing.assert_allclose(predicted.I, design @ moment)
    np.testing.assert_allclose(predicted.cov, expected_cov)
    np.testing.assert_allclose(predicted.sigma[:, 0], np.sqrt(np.diag(expected_cov)))
    np.testing.assert_allclose(predicted.unconstrained[:, 0], missing)
    assert predicted.ledger["unconstrained"].kind == "unbounded"
    r = design[1] @ moment - y[1]
    predictive_variance = cov[1, 1] + expected_cov[1, 1] - 2 * p[1] @ cov[fitted, 1]
    assert fit.withheld.n == 1
    np.testing.assert_allclose(fit.withheld.chi2, r**2 / cov[1, 1])
    np.testing.assert_allclose(fit.withheld.chi2_predictive, r**2 / predictive_variance)
    np.testing.assert_allclose(fit.column.value, 0.3 * moment[0])


def test_per_pixel_noise_and_mask_chunks_match_pointwise_fit(basis):
    matrix = np.asarray(basis.design())[:, 0, :]
    truth = np.arange(12, dtype=float).reshape(2, 3, 2) / 10.0
    obs = np.einsum("...k,ck->...c", truth, matrix)
    sigma = np.array([[0.2, 1.0, 2.0], [2.0, 0.4, 1.0], [0.5, 2.0, 1.0]])
    mask = np.ones((2, 3, 3, 1), dtype=bool)
    mask[0, 1, 0, 0] = False
    mask[1, 2, 2, 0] = False
    data = StokesData(
        basis.channels, I=obs, sigma=sigma[..., None], mask=mask, unit="cgs"
    )
    fit = fit_free(basis, data, rank=FullRank(), chunk_bytes=800)
    stored = fit_free(
        basis, data, rank=FullRank(), keep_directions=True, chunk_bytes=800
    )
    assert fit.chunk_pixels == 3
    assert fit.retained.directions is None
    assert stored.retained.directions.shape == (2, 3, 2, 2)
    np.testing.assert_allclose(fit.moments.values, truth, atol=5e-15)
    p, q = fit.predict(cov=True), stored.predict(cov=True)
    np.testing.assert_allclose(p.cov, q.cov, rtol=1e-14)
    np.testing.assert_allclose(p.unconstrained, q.unconstrained, atol=5e-15)
    for i, j in np.ndindex(2, 3):
        y = StokesData(
            basis.channels, I=obs[i, j], sigma=sigma[j], mask=mask[i, j], unit="cgs"
        )
        local = fit_free(basis, y, rank=FullRank())
        np.testing.assert_allclose(p.sigma[i, j], local.predict().sigma, rtol=2e-14)


def test_aligned_training_channels_retain_original_prediction_basis(basis):
    coefficients = np.array([2.0, -0.7])
    design = np.asarray(basis.design())[:, 0, :]
    observed = (design @ coefficients) * 1e23
    data = StokesData(
        basis.channels[[0, 2]], I=observed[[0, 2]], sigma=0.5e23, unit="Jy/sr"
    )
    fit = fit_free(basis, data, rank=FullRank())
    assert fit.problem.fit_basis.channels.n_ch == 2
    np.testing.assert_allclose(fit.predict().I, observed)
    new = Channels.at([5.0, 7.0], unit="GHz")
    prediction = fit.predict(new)
    np.testing.assert_allclose(
        prediction.I, np.asarray(basis.at(new).design())[:, 0] @ coefficients * 1e23
    )
    assert (
        "prediction channels extend outside the observed band" in prediction.plan.notes
    )


@pytest.mark.parametrize(
    "ensemble, expected_rank, expected_chi2",
    [
        ("E1", 7, 52.13245518691838),
        ("E5", 17, 43.16593433381058),
    ],
)
def test_prebuilt_fit_accepts_the_seeded_study(ensemble, expected_rank, expected_chi2):
    cfg = _study.paper()
    gamma = Cells.log(300, cfg.gamma_hi, cfg.n_bins, order=2)
    azimuth = {} if ensemble == "E1" else {"phi": AzimuthHarmonic()}
    lattice = Lattice(gamma=gamma, **azimuth)
    basis = build(
        cfg, lattice, "lookup", cfg.azimuth_factor if ensemble == "E1" else None
    )
    with np.load(_study.ACCEPTANCE_DATA / "forward_spectra.npz") as arrays:
        truth = arrays[ensemble].copy()
    n_fit = len(cfg.fit_idx)
    sigma = 0.01 * truth[:, :1]
    observed = truth.copy()
    observed[cfg.fit_idx] += sigma[cfg.fit_idx] * cfg.draw_noise(n_fit)
    held = np.setdiff1d(np.arange(cfg.n_channels), cfg.fit_idx)
    fit = fit_free(
        basis,
        StokesData(basis.channels, stokes=observed, sigma=sigma, unit="cgs").withhold(
            held
        ),
    )
    assert fit.rank == expected_rank
    assert fit.dof == 3 * n_fit - expected_rank
    assert float(fit.chi2) == pytest.approx(expected_chi2, abs=1e-6)
    assert fit.withheld.n == 3 * len(held)
