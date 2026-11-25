"""Independent reference values for the physical family likelihood and output map."""

import numpy as np
import pytest

import syncem as se
from syncem.constants import C_CGS, E_ESU, M_E


@se.family(variable="x", support=(-1.0, 1.0))
def skew_density(x, theta):
    return 1 + theta**3 * x


@se.kernel(variables=(se.Variable("x"),), stokes=("I",))
def linear_probe(nu, x):
    return {"I": 2 + (nu / 1e9) * x}


@pytest.mark.parametrize("initial", [0.0, 0.4])
def test_user_density_explores_starts_and_reports_identification(initial):
    nu = np.array([0.5, 1.0, 2.0])
    ch = se.Channels.at(nu, unit="GHz")
    cells = se.Lattice(x=se.Cells.linear(-1, 1, 1, order=1), weights=[1])
    data = se.StokesData(ch, I=2 + nu * 0.8**3 / 3, sigma=0.01, unit="cgs")
    source = se.Population(x=skew_density(theta=se.Free(initial, bounds=(-0.95, 0.95))))
    fit = se.fit(source, data, lattice=cells, kernel=linear_probe, check=False)
    assert fit.n_starts == 4
    assert 0 < fit.converged_starts <= fit.n_starts
    if initial == 0:
        assert fit.converged_starts < fit.n_starts
    assert fit.params["x.theta"].identified
    assert fit.params["x.theta"].value == pytest.approx(0.8, abs=2e-8)
    assert fit.chi2 < 1e-13
    assert "Starts reaching the minimum" in str(fit)
    sigma = 0.01 / (0.8**2 * np.linalg.norm(nu))
    np.testing.assert_allclose(fit.params["x.theta"].sigma, sigma, rtol=2e-7)
    np.testing.assert_allclose(fit.predict().sigma[:, 0], nu * 0.8**2 * sigma)


@se.kernel(variables=(se.Variable("gamma", kind="positive"),), stokes=("I",))
def column_kernel(nu, gamma):
    return {"I": (1 + nu / 1e9) + 0 * gamma}


@pytest.mark.parametrize("compressed", [False, True])
def test_family_batch_propagates_dense_pixel_covariance(compressed):
    nu = np.array([0.5, 1.0, 2.0, 3.0])
    ch = se.Channels.at(nu, unit="GHz")
    forecast = se.Channels.at([1.5, 4.0], unit="GHz")
    row = 1 + nu
    output = np.array([2.5, 5.0])
    lattice = se.Lattice(gamma=se.Cells.linear(2.0, 4.0, 1, order=1), weights=[1.0])
    source = se.Population(gamma=se.PowerLaw(2.0, lo=2, hi=4, column=se.Free(2.0)))
    basis = se.spectral_basis(
        source.at({"gamma.column": 2.0}),
        ch,
        lattice=lattice,
        kernel=column_kernel,
        check=False,
    )
    amplitudes = np.array([[0.4, 1.0], [2.0, 7.0]])
    values = amplitudes[..., None] * row
    covariance = (
        0.02**2
        * np.outer(row, row)
        * (0.2 ** np.abs(np.subtract.outer(np.arange(4), np.arange(4))))
    )
    covariances = np.stack((covariance, 4 * covariance))
    mask = np.ones((2, 1, 4), bool)
    mask[1, 0, 1] = False
    data = se.StokesData(
        ch, I=values, cov=covariances, mask=mask[..., None], unit="cgs"
    ).withhold([2])
    target = data
    if compressed:
        free = se.Population(se.FreeMoments("gamma"))
        target = se.fit(
            free,
            data,
            basis=basis,
            rank=se.FullRank(),
            keep_directions=True,
            check=False,
        ).retained
    result = se.fit(source, target, basis=basis, check=False)
    pred = result.predict(forecast, cov=True)
    np.testing.assert_allclose(result.params["gamma.column"].value, amplitudes)
    np.testing.assert_allclose(pred.I, amplitudes[..., None] * output)
    for i, j in np.ndindex(amplitudes.shape):
        used = mask[i, 0].copy()
        used[2] = False
        r = row[used]
        variance = 1 / (r @ np.linalg.solve(covariances[j][np.ix_(used, used)], r))
        np.testing.assert_allclose(result.parameter_covariance[i, j], [[variance]])
        np.testing.assert_allclose(pred.sigma[i, j, :, 0], np.sqrt(variance) * output)
        np.testing.assert_allclose(pred.cov[i, j], variance * np.outer(output, output))
        assert result.chi2[i, j] < 1e-18
    if not compressed:
        np.testing.assert_array_equal(result.withheld.n, np.ones_like(amplitudes))
        assert np.max(result.withheld.chi2) < 1e-18


@pytest.mark.parametrize("compressed", [False, True])
def test_aged_family_derived_physical_quantities(compressed):
    ch = se.Channels.log(0.3, 8, 8, unit="GHz", n_nu=8)
    lat = se.Lattice(gamma=se.Cells.log(300, 1e6, 10, order=2))
    g, a, p, pivot, field = 20000.0, 0.7, 2.4, 5000.0, 5e-6
    source = se.Population(
        gamma=se.AgedPowerLaw(p, gamma_cut=g, amplitude=a, lo=300, pivot=pivot),
        B=(field, "G"),
    )
    basis = se.spectral_basis(source, ch, lattice=lat, stokes="I", check=False)
    y = basis.predict(source, check=False).I
    data = se.StokesData(ch, I=y, sigma=0.01 * y, unit="cgs")
    model = source.replace(
        gamma=se.AgedPowerLaw(
            p,
            gamma_cut=se.Free(18000, bounds=(10000, 30000), log=True),
            lo=300,
            pivot=pivot,
        )
    )
    target = data
    if compressed:
        target = se.fit(
            source.replace(gamma=se.FreeMoments()),
            data,
            basis=basis,
            keep_directions=True,
            rank=se.FullRank(),
            check=False,
        ).retained
    fit = se.fit(model, target, basis=basis, check=False)
    assert tuple(fit.params) == ("gamma.explicit_cut", "gamma.amplitude")
    coefficient = fit.derived("paper_coefficient")
    paper_factor = 1e4 * (1e4 / pivot) ** -p
    np.testing.assert_allclose(coefficient.value, a * paper_factor)
    np.testing.assert_allclose(
        coefficient.sigma, fit.params["gamma.amplitude"].sigma * paper_factor
    )
    frequency = fit.derived("break_frequency")
    critical = 3 * E_ESU / (4 * np.pi * M_E * C_CGS)
    mean_field = field * np.pi / 4
    np.testing.assert_allclose(frequency.value, critical * mean_field * g**2)
    np.testing.assert_allclose(
        frequency.sigma,
        2 * critical * mean_field * g * fit.params["gamma.explicit_cut"].sigma,
    )
    bands = se.Channels.at([9.0, 15.0], unit="GHz")
    pred = fit.predict(bands, cov=True)
    other = basis.at(bands)

    def intensity(cut):
        pop = source.replace(
            gamma=se.AgedPowerLaw(p, gamma_cut=cut, amplitude=a, lo=300, pivot=pivot)
        )
        return np.asarray(other.predict(pop, check=False).I)

    jac = np.column_stack(((intensity(g + 2) - intensity(g - 2)) / 4, intensity(g) / a))
    cov = jac @ fit.parameter_covariance @ jac.T
    np.testing.assert_allclose(pred.cov, cov, rtol=2e-6, atol=0)
    np.testing.assert_allclose(pred.sigma[:, 0] ** 2, np.diag(cov), rtol=2e-6, atol=0)


@se.kernel(variables=(se.Variable("gamma", kind="positive"),), stokes=("I",))
def affine_energy(nu, gamma):
    return {"I": 1 + nu / 1e9 * (gamma - 3) / 3}


def test_compressed_rank_floor_uses_observed_directions_and_propagates_functionals():
    channels = se.Channels.at([0.4, 0.9, 1.7, 2.5], unit="GHz")
    lattice = se.Lattice(gamma=se.Cells.linear(2, 4, 2, order=1), weights=[1, 1])
    source = se.Population(gamma=se.PowerLaw(1.3, lo=2, hi=4, column=1.9))
    basis = se.spectral_basis(
        source, channels, lattice=lattice, kernel=affine_energy, check=False
    )
    y = basis.predict(source, check=False).I
    data = se.StokesData(channels, I=y, sigma=0.3, unit="cgs")
    retained = se.fit(
        se.Population(se.FreeMoments("gamma")),
        data,
        basis=basis,
        keep_directions=True,
        check=False,
    ).retained
    assert retained.rank == 1
    amplitude = source.replace(gamma=se.PowerLaw(1.3, lo=2, hi=4, column=se.Free(1.1)))
    slope = source.replace(
        gamma=se.PowerLaw(
            se.Free(1.0, bounds=(0.5, 4)), lo=2, hi=4, column=se.Free(1.1)
        )
    )
    mixture = source.replace(
        gamma=se.PowerLawMixture.moment_form(1, 1.3, pivot=3, lo=2, hi=4)
    )
    for family in (slope, mixture):
        direct = se.fit(family, data, basis=basis, check=False)
        reduced = se.fit(family, retained, basis=basis, check=False)
        assert direct.rank == reduced.rank == 2
        for name, param in direct.params.items():
            np.testing.assert_allclose(
                reduced.params[name].value, param.value, atol=2e-11
            )
        np.testing.assert_allclose(
            reduced.parameter_covariance, direct.parameter_covariance, rtol=1e-10
        )
        np.testing.assert_allclose(
            reduced.column.sigma, direct.column.sigma, rtol=1e-10
        )
    fit = se.fit(amplitude, data, basis=basis, check=False)
    sigma = 0.3 / np.linalg.norm(y / 1.9)
    np.testing.assert_allclose(fit.column.sigma, sigma)
    rows = np.array([[1, 0, 1, 0], [2, 0, 2, 0]])
    np.testing.assert_allclose(fit.observable(rows).sigma, [sigma, 2 * sigma])


@se.family(variable="x", support=(-1.0, 1.0))
def quadratic_shape(x, a, b):
    return 1 + a * x + b * (3 * x * x - 1) / 2


@se.kernel(variables=(se.Variable("x"),), stokes=("I",))
def collinear_probe(nu, x):
    return {"I": 2 + 3 * x + 7.5 * (nu / 1e9) * (x * x - 1 / 3)}


def test_weak_null_space_component_has_no_finite_prediction_error():
    # The training channel measures a+b. A different frequency measures a+t*b.
    source = se.Population(
        x=quadratic_shape(
            a=se.Free(0.1, bounds=(-0.4, 0.4)), b=se.Free(0.1, bounds=(-0.4, 0.4))
        )
    )
    channels = se.Channels.at([1], unit="GHz")
    survey = se.StokesData(channels, I=[2.3], sigma=0.03, unit="cgs")
    lattice = se.Lattice(x=se.Cells.linear(-1, 1, 1, order=2), weights=[1])
    fit = se.fit(source, survey, kernel=collinear_probe, lattice=lattice, check=False)
    nu = np.array([1, 1.00000005, 1.002])
    pred = fit.predict(se.Channels.at(nu, unit="GHz"), cov=True)
    rows = np.column_stack([-2.5 * nu, np.full(3, 3), 7.5 * nu])
    for sigma in [pred.sigma.ravel(), fit.observable(rows).sigma]:
        np.testing.assert_allclose(sigma[0], 0.03, rtol=1e-11)
        assert np.isnan(sigma[1:]).all()
    assert np.isnan(pred.cov[1:]).all() and np.isnan(pred.cov[:, 1:]).all()
