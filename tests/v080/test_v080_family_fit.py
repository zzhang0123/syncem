"""A separable physical fit profiles its normalization on the whitened data."""

import numpy as np
import pytest

import syncem as se


def test_power_law_profiles_amplitude_and_recovers_shape():
    ch = se.Channels.log(0.2, 8, 8, unit="GHz", n_nu=8)
    lat = se.Lattice(gamma=se.Cells.log(300, 1e6, 15, order=2))
    args = {"lo": 300.0, "hi": 1e6}
    pop = se.Population(gamma=se.PowerLaw(2.65, amplitude=0.7, **args), B="5 uG")
    basis = se.spectral_basis(pop, ch, lattice=lat, stokes="I", check=False)
    truth = basis.predict(pop, check=False)
    data = se.StokesData(ch, I=truth.I, sigma=0.01 * truth.I, unit="cgs")
    guess = pop.replace(
        gamma=se.PowerLaw(se.Free(start=2.2, bounds=(1.5, 4.0)), **args)
    )
    result = se.fit(guess, data, lattice=lat, check=False)
    assert result.params["gamma.index"].value == pytest.approx(2.65, abs=1e-7)
    assert result.params["gamma.amplitude"].value == pytest.approx(0.7, abs=1e-7)
    assert result.chi2 < 1e-12
    np.testing.assert_allclose(result.predict().I, truth.I, rtol=1e-9)
    step = 2e-5
    around = [
        basis.predict(
            pop.replace(gamma=se.PowerLaw(2.65 + dx, amplitude=0.7, **args)),
            check=False,
        ).I
        for dx in (-step, step)
    ]
    jac = np.column_stack(((around[1] - around[0]) / (2 * step), truth.I / 0.7))
    white = jac / (0.01 * np.asarray(truth.I)[:, None])
    np.testing.assert_allclose(
        result.parameter_covariance, np.linalg.inv(white.T @ white), rtol=2e-7
    )
    for i, name in enumerate(result.params):
        assert result.params[name].sigma ** 2 == pytest.approx(
            result.parameter_covariance[i, i]
        )
    subset = se.StokesData(
        ch[1::2], I=truth.I[1::2], sigma=0.01 * truth.I[1::2], unit="cgs"
    )
    fit_subset = se.fit(guess, subset, basis=basis, check=False)
    assert fit_subset.basis.channels.row_keys() == ch.row_keys()
    np.testing.assert_allclose(fit_subset.predict().I, truth.I, rtol=1e-9)
    direct = se.fit(
        se.Population(se.FreeMoments("gamma"), B="5 uG"),
        data,
        basis=basis,
        rank=se.FullRank(),
    )
    compressed = se.fit(guess, direct.retained, lattice=basis.lattice)
    assert compressed.params["gamma.index"].value == pytest.approx(2.65, abs=1e-7)
    assert compressed.params["gamma.amplitude"].value == pytest.approx(0.7, abs=1e-7)


@pytest.mark.parametrize("fit_shape", [False, True])
def test_family_fit_whitens_withheld_rows(fit_shape):
    channels = se.Channels.log(0.3, 8, 6, unit="GHz", n_nu=8)
    lattice = se.Lattice(gamma=se.Cells.log(300, 1e6, 8, order=2))
    limits = {"lo": 300.0, "hi": 1e6, "amplitude": 0.7}
    source = se.Population(gamma=se.PowerLaw(2.65, **limits), B="5 uG")
    basis = se.spectral_basis(source, channels, lattice=lattice, check=False)
    truth = np.asarray(basis.predict(source, check=False).to("Jy/sr").I)
    sigma = 0.01 * truth
    response = np.eye(truth.size)
    response[-1, -2] = 0.5
    correlation = 0.4 ** np.abs(np.arange(truth.size)[:, None] - np.arange(truth.size))
    covariance = correlation * np.outer(sigma, sigma)
    observed = response @ truth
    observed[-1] += 2 * sigma[-1]
    data = se.StokesData(
        channels, I=observed, cov=covariance, response=response, unit="Jy/sr"
    ).withhold([-1])
    model = (
        source.replace(gamma=se.PowerLaw(se.Free(2.4, bounds=(1.5, 4.0)), **limits))
        if fit_shape
        else source
    )

    fit = se.fit(model, data, basis=basis, check=False)

    assert fit.chi2 < 1e-10
    assert fit.withheld.n == 1
    assert fit.withheld.chi2 == pytest.approx(4.0)
    assert fit.withheld.max_residual == pytest.approx(2.0)


def test_linear_index_mixture_on_a_polynomial_in_log_energy():
    import jax.numpy as jnp

    @se.kernel(variables=(se.Variable("gamma", kind="positive"),), stokes=("I",))
    def model(nu, gamma):
        t = nu / 1e9
        u = jnp.log(gamma / 1000)
        return {"I": 2 + t * u + t**2 * u**2}

    ch = se.Channels.at([0.5, 1.0, 1.5, 2.0, 3.0], unit="GHz")
    lat = se.Lattice(gamma=se.Cells.log(500, 2000, 3, order=2))
    law = se.PowerLawMixture.moment_form(2, 2.6, pivot=1000, lo=500, hi=2000)
    x, w = np.polynomial.legendre.leggauss(128)
    points, w = 1250 + 750 * x, 750 * w
    coefficients = np.array([1.0, 0.2, 0.01])
    values = model.fn(ch.centres_hz[:, None], points)["I"]
    expected = values @ (w * (law.basis_density(points) @ coefficients))
    data = se.StokesData(ch, I=expected, unit="cgs", sigma=1.0)
    result = se.fit(
        se.Population(gamma=law), data, lattice=lat, kernel=model, check=False
    )
    np.testing.assert_allclose(result.coefficients, coefficients, rtol=1e-7, atol=1e-10)
    np.testing.assert_allclose(
        result.electron_spectrum(points), law.basis_density(points) @ coefficients
    )
