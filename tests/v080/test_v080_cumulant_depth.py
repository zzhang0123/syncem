"""Regression checks for finite-domain and conditional Faraday cumulants."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se


def test_higher_cumulants_have_a_checked_nonamplifying_domain():
    with pytest.raises(ValueError, match="require tau_max"):
        se.CumulantScreen([0, 1, 0, 3])
    with pytest.raises(ValueError, match="amplification"):
        se.CumulantScreen([0, 1, 0, 3], tau_max=2.1)
    law = se.CumulantScreen([0, 1, 0, 3], tau_max=2)
    np.testing.assert_allclose(law.characteristic([0, 1, 2]), [1, np.exp(-0.375), 1])
    with pytest.raises(ValueError, match="within its domain"):
        law.characteristic([2.000001])


def test_composed_phase_keeps_screen_parameters_in_the_jax_tree():
    first = se.CumulantScreen([2.0, 0.3**2, 0, 0])
    second = se.GaussianScreen(1.1, 0.2)
    phase = se.FaradayRotation(screen=first).with_screen(second)
    compiled = eqx.filter_jit(lambda law, nu: law.weights(nu))
    frequency = jnp.array([0.55e9, 1.2e9])
    tau = phase.tau(frequency)
    expected = first.characteristic(tau) * second.characteristic(tau)
    np.testing.assert_allclose(compiled(phase, frequency), expected, rtol=1e-12)
    assert any(leaf is first.cumulants for leaf in jax.tree.leaves(phase))


def test_conditioned_cumulants_compose_with_foreground_in_08():
    frequency = se.Channels.at([0.65, 1.5], unit="GHz", half_width=0.08, n_nu=16)
    gamma = 400.0
    foreground = se.GaussianScreen(1.0, 0.4)
    phase = se.FaradayRotation(screen=foreground.characteristic)

    def depth(gamma):
        gamma = getattr(gamma, "value", gamma)
        return se.CumulantScreen(jnp.array([gamma / 100, 0.5, 0.2, 0.1]), tau_max=1)

    pop = se.Population(
        gamma=se.Delta(gamma),
        B="5 uG",
        eta=0.25,
        phi=se.AzimuthFactor(0.8),
        depth=se.Conditional(depth),
    )
    lattice = se.Lattice(gamma=se.Cells.log(350, 450, 3, order=2))
    model = se.ContinuumKernel()
    baseline = se.predict(
        pop.replace(depth=se.NoRotation()), frequency, lattice=lattice, kernel=model
    )
    conditional = se.spectral_basis(
        pop, frequency, lattice=lattice, kernel=model, phase=phase
    ).predict(pop)
    independent = se.predict(
        pop.replace(depth=depth(gamma)),
        frequency,
        lattice=lattice,
        kernel=model,
        phase=phase,
    )
    assert np.max(np.abs(conditional.P)) < np.max(np.abs(baseline.P))
    np.testing.assert_allclose(
        conditional.stokes, independent.stokes, rtol=1e-10, atol=1e-40
    )
    omitted = [
        term
        for term in conditional.ledger.terms
        if term.name == "depth_cumulant_truncation"
    ]
    assert len(omitted) == 1 and omitted[0].kind == "unbounded"
    for declaration in (pop, pop.replace(depth=depth(gamma))):
        disabled = se.predict(
            declaration,
            frequency,
            lattice=lattice,
            kernel=model,
            phase=False,
        )
        np.testing.assert_allclose(
            disabled.stokes, baseline.stokes, rtol=1e-12, atol=1e-40
        )
        assert "depth_cumulant_truncation" not in disabled.ledger.unbounded()
    composed = se.predict(
        pop.replace(depth=se.NoRotation()),
        frequency,
        lattice=lattice,
        kernel=model,
        phase=phase.with_screen(depth(gamma).characteristic),
    )
    np.testing.assert_allclose(
        composed.stokes, independent.stokes, rtol=1e-12, atol=1e-40
    )
    assert "depth_cumulant_truncation" in composed.ledger.unbounded()


def test_quadrature_fallback_for_conditional_depth_is_refined():
    channels = se.Channels.at([0.5, 0.7], unit="GHz")
    lattice = se.Lattice(gamma=se.Cells.log(350, 450, 1, order=0))
    source = se.Population(
        gamma=se.Delta(float(lattice.cells.reference[0, 0])),
        B="1 mG",
        eta=0.25,
        phi=se.AzimuthFactor(0.8),
        depth=se.Conditional(
            lambda gamma: se.Uniform(-20, 20, unit="rad m^-2", nodes=1)
        ),
    )
    result = se.predict(source, channels, lattice=lattice, check=True)
    quadrature = result.ledger["quadrature"]
    assert quadrature.kind == "estimate"
    assert np.max(quadrature.value[..., 1:] / result.I[:, None]) > 0.1
