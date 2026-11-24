"""Traced parameters in a fixed-basis conditional electron likelihood."""

import equinox as eqx
import jax
import numpy as np

import syncem as se


def test_conditional_family_likelihood_gradient_and_coordinates():
    def electron(B, index, amplitude):
        exponent = index + 0.01 * B.to("uG").value
        return se.PowerLaw(exponent, lo=300, hi=1e5, amplitude=amplitude)

    model = se.Conditional(
        electron,
        params={
            "index": se.Free(2.5, bounds=(1.5, 3.5)),
            "amplitude": se.Free(0.5, bounds=(0.1, 1.0)),
        },
    )
    pop = se.Population(gamma=model, B="5 uG")
    ch = se.Channels.at([0.5, 1.0, 3.0], unit="GHz")
    lat = se.Lattice(gamma=se.Cells.log(300, 1e5, 6, order=2))
    truth = se.Population(
        gamma=se.PowerLaw(2.7, lo=300, hi=1e5, amplitude=0.6), B="5 uG"
    )
    spec = se.predict(truth, ch, lattice=lat, stokes="I", check=False)
    data = se.StokesData(ch, I=spec.I, sigma=0.02 * spec.I, unit="cgs")
    problem = se.problem(pop, data, lattice=lat, check=False)
    logp = eqx.filter_jit(problem.log_prob)
    z = problem.unconstrain(problem.params)
    np.testing.assert_allclose(
        jax.tree.leaves(problem.constrain(z)), jax.tree.leaves(problem.params)
    )
    gradient = jax.grad(logp)(z)
    h = 1e-5
    expected = [(logp(z + h * e) - logp(z - h * e)) / (2 * h) for e in np.eye(2)]
    np.testing.assert_allclose(gradient, expected, rtol=1e-7)
    optimum = {"gamma.index": 2.65, "gamma.amplitude": 0.6}
    np.testing.assert_allclose(
        problem.log_likelihood(optimum), -problem.normaliser, atol=1e-8
    )
    profiled = eqx.filter_jit(problem.profiled())
    assert profiled(optimum) < 1e-20
    assert profiled({**optimum, "gamma.amplitude": 0.7}) > 1e-3

    def unit_electron(B, index):
        return se.PowerLaw(index + 0.01 * B.to("uG").value, lo=300, hi=1e5)

    unit_population = pop.replace(
        gamma=se.Conditional(
            unit_electron, params={"index": se.Free(2.5, bounds=(1.5, 3.5))}
        )
    )
    with_profile = se.problem(unit_population, data, lattice=lat, check=False)
    assert with_profile.profiled()({"gamma.index": 2.65}) < 1e-20


def test_unbounded_normalised_power_law_rejects_nonintegrable_proposals():
    def electron(index):
        return se.PowerLaw(index, lo=300, column=1.0)

    ch = se.Channels.at([1.0, 4.0], unit="GHz")
    lat = se.Lattice(gamma=se.Cells.log(300, 1e5, 6, order=2))
    data = se.StokesData(ch, I=[1.0, 1.0], sigma=1.0, unit="cgs")
    pop = se.Population(
        gamma=se.Conditional(electron, params={"index": se.Free(2.0, bounds=(0, 4))}),
        B="5 uG",
    )
    problem = se.problem(pop, data, lattice=lat, check=False)
    log_prob = eqx.filter_jit(problem.log_prob)
    assert np.isfinite(log_prob({"gamma.index": 1.01}))
    for index in (1.0, 0.5):
        outside = {"gamma.index": index}
        assert np.isneginf(log_prob(outside))
        assert np.isneginf(problem.log_likelihood(outside))
        assert jax.grad(log_prob)(outside)["gamma.index"] == 0.0
