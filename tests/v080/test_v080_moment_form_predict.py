"""Predictions and withheld diagnostics of a linear density-form fit (review P2-1)."""

import numpy as np

import syncem as se
from syncem.distributions import PowerLawMixture


def test_moment_form_fit_predicts_and_reports_withheld_rows():
    channels = se.Channels.log(0.5, 4.0, 5, unit="GHz", half_width=0.2, n_nu=8)
    population = se.paper.e1()
    lattice = se.Lattice(gamma=se.Cells.log(1e4, 4e4, 2, order=1))
    basis = se.spectral_basis(population, channels, lattice=lattice, check=False)
    truth = basis.predict(population, check=False)
    sigma = 0.01 * float(np.max(np.abs(truth.I)))
    observed = np.asarray(truth.I) * np.array([1.0, 1.02, 0.97, 1.01, 1.0])
    data = se.StokesData(channels, I=observed, unit=truth.unit, sigma=sigma)
    data = data.withhold([2])
    form = population.replace(
        gamma=PowerLawMixture.moment_form(1, 2.6, pivot=2e4, lo=1e4, hi=4e4)
    )
    fit = se.fit(form, data, basis=basis)
    predicted = fit.predict()
    assert np.all(np.isfinite(predicted.I))
    assert predicted.ledger["representation"].kind == "unbounded"
    withheld = fit.withheld
    assert withheld is not None and int(withheld.n) == 1
    residual = (observed[2] - float(np.asarray(predicted.I)[2])) / sigma
    np.testing.assert_allclose(float(withheld.chi2), residual**2, rtol=1e-8)
