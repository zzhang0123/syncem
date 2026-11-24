"""Normalised linear likelihoods and gradient checks for the sampler hook."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se
from syncem.distributions.densities import Normal
from syncem.fitting.data import StokesData
from syncem.fitting.problem import GaussianPrior, Params, Problem
from syncem.fitting.rank import FullRank
from syncem.values.errors import DataError, ImproperPosterior, RoleError
from tests.v080.test_v080_free_fit import basis as basis


def test_normalised_correlated_likelihood_and_proper_prior_gradient(basis):
    channels = basis.channels
    observations = np.array([2.0, 7.0, 3.0])
    covariance = np.array([[2.0, 0.1, 0.5], [0.1, 3.0, 0.4], [0.5, 0.4, 1.0]])
    data = StokesData(channels, I=observations, cov=covariance, unit="cgs").withhold(
        [1]
    )
    mean, std = np.array([0.5, 0.25]), np.array([1.0, 2.0])
    problem = Problem(basis, data, prior=GaussianPrior(mean, std))
    parameters = np.array([0.4, -0.1])
    selected = [0, 2]
    design = np.asarray(basis.design())[selected, 0, :]
    covariance = covariance[np.ix_(selected, selected)]
    residual = observations[selected] - design @ parameters
    precision = np.linalg.inv(covariance)
    loglike = -0.5 * (
        residual @ precision @ residual
        + np.linalg.slogdet(covariance)[1]
        + 2 * math.log(2 * math.pi)
    )
    ref_prior = float(
        np.sum(
            -0.5 * ((parameters - mean) / std) ** 2 - np.log(std * np.sqrt(2 * np.pi))
        )
    )
    model = eqx.filter_jit(problem.log_prob)
    np.testing.assert_allclose(
        problem.log_likelihood(Params(parameters)), loglike, rtol=1e-14
    )
    np.testing.assert_allclose(
        model(Params(parameters)), loglike + ref_prior, rtol=1e-14
    )
    exact_grad = design.T @ precision @ residual - (parameters - mean) / std**2
    np.testing.assert_allclose(
        jax.grad(model)(Params(parameters)).a, exact_grad, rtol=1e-14
    )
    step = 1e-5
    finite = np.array(
        [
            (model(parameters + d) - model(parameters - d)) / (2 * step)
            for d in step * np.eye(2)
        ]
    )
    np.testing.assert_allclose(finite, exact_grad, rtol=1e-9, atol=1e-10)
    np.testing.assert_array_equal(
        problem.unconstrain(problem.constrain(parameters)), parameters
    )
    assert problem.profiled()() == pytest.approx(
        fit_chi2 := float(Problem(basis, data).solve(rank=FullRank()).chi2), abs=1e-14
    )
    assert fit_chi2 < 1e-14
    with pytest.raises(RoleError, match="zero-mean scalar Gaussian"):
        problem.solve()
    with pytest.raises(DataError, match="no nonlinear coordinates"):
        problem.profiled()(parameters)


def test_missing_prior_refuses_log_prob_but_allows_likelihood(basis):
    data = StokesData(
        basis.channels, I=[1.0, 2.0, 3.0], sigma=1.0, unit="cgs"
    ).withhold([0, 1])
    problem = Problem(basis, data)
    assert np.isfinite(problem.log_likelihood(problem.params))
    with pytest.raises(ImproperPosterior, match="prior=Normal"):
        problem.log_prob(problem.params)
    with pytest.raises(DataError, match="prior shape"):
        Problem(basis, data, prior=GaussianPrior(np.zeros(3), 1.0))
    with pytest.raises(DataError, match="positive"):
        GaussianPrior(sigma=0.0)


def test_batch_noise_and_parameter_updates_share_one_jit_trace(basis):
    values = np.array([[1.0, 2.0, 3.0], [1.4, 2.3, 3.8]])
    sigmas = np.array([[0.2, 1.1, 1.8], [1.2, 0.6, 1.1]])
    traces = []

    @eqx.filter_jit
    def log_density(problem):
        traces.append(True)
        return problem.log_prob(problem.params)

    fits = []
    for shift in (0.0, 0.2):
        data = StokesData(
            basis.channels,
            I=values + shift,
            sigma=sigmas[..., None],
            unit="cgs",
        )
        problem = Problem(
            basis, data, prior=Normal(shift, 1.0), params=np.full((2, 2), shift)
        )
        fits.append(problem)
    assert len({float(log_density(p)) for p in fits}) == 2
    assert len(traces) == 1
    start = fits[0].params.a
    candidates = jnp.stack((start, start + 0.1))
    got = jax.vmap(fits[0].log_prob)(candidates)
    expected = np.array([fits[0].log_prob(c) for c in candidates])
    np.testing.assert_allclose(got, expected)


def test_joined_problem_sums_designs_and_keeps_one_observation_likelihood(basis):
    data = StokesData(basis.channels, I=[4.0, 6.0, 7.0], sigma=0.5, unit="cgs")
    a = Problem(basis, data, prior=GaussianPrior(0.0, 1.0))
    b = Problem(basis, data, prior=GaussianPrior(0.5, 2.0))
    joined = a + b
    x, y = np.array([0.3, 0.2]), np.array([-0.2, 1.2])
    combined = np.concatenate((x, y))
    np.testing.assert_allclose(joined.log_likelihood(combined), a.log_likelihood(x + y))
    expected = a.log_likelihood(x + y) + a.prior.log_prob(x) + b.prior.log_prob(y)
    np.testing.assert_allclose(eqx.filter_jit(joined.log_prob)(combined), expected)
    np.testing.assert_allclose(joined.profiled()(), a.profiled()())
    with pytest.raises(RoleError, match="joined deterministic fits"):
        joined.solve()
    plain = Problem(basis, data)
    partial = a + plain
    with pytest.raises(ImproperPosterior):
        partial.log_prob(combined)
    with pytest.raises(RoleError, match="joined deterministic fits"):
        partial.solve()
    joined_plain = plain + plain
    fit = joined_plain.solve(rank=FullRank())
    assert len(fit.moments) == 2
    np.testing.assert_allclose(
        fit.predict().I, plain.solve(rank=FullRank()).predict().I, atol=1e-14
    )
    np.testing.assert_allclose(fit.chi2, joined.profiled()(), atol=1e-14)
    changed = StokesData(basis.channels, I=[4.0, 6.0, 7.0], sigma=0.6, unit="cgs")
    with pytest.raises(DataError, match="share observations, covariance"):
        _ = a + Problem(basis, changed)


def test_joined_mechanisms_use_their_own_lattice_and_common_stokes(basis):
    @se.kernel(variables=(se.Variable("grain"),), stokes=("I", "P"))
    def grains(nu, grain):
        return {"I": grain * (nu / 1e9) ** 2, "P": 1j * grain}

    channels = basis.channels
    lattice = se.Lattice(grain=se.Cells.linear(1.0, 2.0, 1, order=0))
    second = se.SpectralBasis.build(channels, lattice, kernel=grains, check=False)
    a = np.array([0.4, 0.2])
    b = np.array([1.1])
    signal = (
        np.asarray(basis.design(stokes="I"))[:, 0] @ a
        + second.design(stokes="I")[:, 0] @ b
    )
    data = StokesData(channels, I=signal, sigma=0.5, unit="cgs")
    first_problem = Problem(basis, data)
    second_problem = Problem(second, data)
    joined = first_problem + second_problem
    assert tuple(lat.key for lat in joined.lattice) == (basis.lattice.key, lattice.key)
    np.testing.assert_allclose(
        joined.log_likelihood(np.r_[a, b]), -joined.log_normaliser, atol=1e-14
    )
    fit = joined.solve(rank=FullRank())
    assert tuple(item.key for item in fit.moments) == (basis.lattice.key, lattice.key)
    out = fit.predict()
    assert out.components == ("I",)
    np.testing.assert_allclose(out.I, signal, rtol=1e-13)


@pytest.mark.parametrize("units", [None, "m^-2"])
def test_scalar_moment_prior_is_fixed_ridge_from_both_entry_points(basis, units):
    standard_deviation = 2.0 if units is None else 2e4
    prior = Normal(0.0, standard_deviation, unit=units)
    population = se.Population(x=se.FreeMoments(prior=prior))
    data = StokesData(
        basis.channels, I=[2.0, 7.0, 3.0], sigma=0.4, unit="cgs"
    ).withhold([1])
    problem = se.problem(population, data, basis=basis)
    fixed = se.Penalty(0.25)
    explicit = Problem(basis, data).solve(penalty=fixed)
    from_prior = problem.solve()
    from_fit = se.fit(population, data, basis=basis)
    w, y = np.asarray(problem.design), np.asarray(problem.target)
    expected = np.linalg.solve(w.T @ w + 0.25 * np.eye(w.shape[-1]), w.T @ y)
    for fit in (explicit, from_prior, from_fit):
        assert fit.moments.origin == "penalised"
        assert fit.alpha == pytest.approx(0.25)
        np.testing.assert_allclose(fit.moments.values, expected, rtol=1e-12)
    np.testing.assert_allclose(jax.grad(problem.log_prob)(expected), 0, atol=1e-13)
    with pytest.raises(RoleError, match="prior sets the ridge penalty"):
        problem.solve(rank=FullRank())
    with pytest.raises(RoleError, match="prior sets the ridge penalty"):
        se.fit(population, data, basis=basis, penalty=fixed)
    with pytest.raises(RoleError, match="rank policy or a ridge penalty"):
        Problem(basis, data).solve(rank=FullRank(), penalty=fixed)
