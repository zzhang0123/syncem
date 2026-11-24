"""Verify conditional Faraday phase at the physical points of both branches."""

import numpy as np
import pytest

import syncem as se
from syncem.base.window import A1_HZ_PER_G
from syncem.constants import C_SI_M
from syncem.engine._regime_harmonic import harmonic_channel
from syncem.fitting.free_fit import fit_free
from syncem.values.errors import RoleError

B = 0.005
ETA = 0.5
SPIN = 0.25 + 0.05j


def _depth(gamma, mu):
    gamma = getattr(gamma, "value", gamma)
    mu = getattr(mu, "value", mu)
    return se.CumulantScreen(
        [2 + 0.02 * (gamma - 300) + mu, 0.4, 0.15, 0.05], tau_max=2
    )


@pytest.fixture(scope="module")
def setup():
    scale = A1_HZ_PER_G * B * np.sqrt(1 - ETA**2)
    channels = se.Channels.at([0.50, 0.80, 1.20], unit="GHz", half_width=0.12, n_nu=64)
    source = se.Population(
        gamma=se.PowerLaw(2.6, lo=280, hi=321, column=1),
        B=(B, "G"),
        eta=ETA,
        mu=se.Isotropic(),
        phi=se.AzimuthFactor(SPIN),
        depth=se.Conditional(_depth),
    )
    phase = se.FaradayRotation(screen=se.GaussianScreen(1.2, 0.1))
    lattice = se.Lattice(
        gamma=se.Cells.log(280, 300, 3, order=2), phi=se.AzimuthHarmonic()
    )
    high = se.Lattice(
        nu_c=se.Cells.log(scale * 300**2, scale * 321**2, 3, order=2),
        phi=se.AzimuthHarmonic(),
    )
    basis = se.RegimeBasis.build(
        source,
        channels,
        phase=phase,
        harmonic=lattice,
        continuum=high,
        angular_nodes=8,
        frequency_nodes=64,
        check=False,
    )
    return basis, source, phase


def _conditional_transmission(gamma, frequency, pitch):
    tau = 2 * (C_SI_M / frequency) ** 2
    mean = 2 + 0.02 * (gamma - 300) + np.asarray(pitch)[..., None]
    return np.exp(
        1j * (tau * mean - 0.15 * tau**3 / 6) - 0.4 * tau**2 / 2 + 0.05 * tau**4 / 24
    )


def test_both_branches_evaluate_phase_before_reducing_physical_parents(setup):
    basis, source, phase = setup
    channels = basis.channels
    gh = 289.0
    low = basis.predict(source.replace(gamma=se.Delta(gh)))
    expected_low = harmonic_channel(
        channels,
        gh,
        B,
        ETA,
        source.factors["mu"].density,
        phase=lambda nu: phase.weights(nu, 0),
        angular_nodes=8,
        phase_at_pitch=lambda nu, mu: _conditional_transmission(gh, nu, mu),
    )
    np.testing.assert_allclose(low.I, expected_low[:, 0].real, rtol=2e-7)
    np.testing.assert_allclose(low.P, expected_low[:, 1] * SPIN, rtol=5e-7)
    gc = 305.0
    high = basis.predict(source.replace(gamma=se.Delta(gc)))
    nodes, weights = channels.nodes, channels.response_weights()
    i, q = se.ContinuumKernel().model.kernels(nodes, gc, B, ETA)
    phased = phase.weights(nodes, 0) * _conditional_transmission(gc, nodes, ETA)
    np.testing.assert_allclose(high.I, (i * weights).sum(-1), rtol=3e-10)
    np.testing.assert_allclose(
        high.P, SPIN * (q * phased * weights).sum(-1), rtol=3e-10
    )
    assert "depth_cumulant_truncation" in high.ledger.unbounded()
    with pytest.raises(
        RoleError, match="conditional screen needs a physical Population"
    ):
        basis.moments(source)


def test_projection_and_free_fit_with_conditional_screen(setup):
    basis, source, _ = setup
    cells = se.Cells.from_edges([280, 300, 321], order=3)
    projected = se.paper.energy_basis(basis, source, cells)
    truth = basis.predict(source)
    spectrum = projected.predict(source)
    error = np.max(np.abs(spectrum.stokes - truth.stokes) / truth.I[:, None])
    assert error < 3e-5
    assert spectrum.ledger["physical_energy_difference"].kind == "estimate"
    assert "depth_cumulant_truncation" in spectrum.ledger.unbounded()
    data = se.StokesData(
        basis.channels,
        stokes=spectrum.stokes,
        sigma=spectrum.I[:, None] * 0.01,
        unit="cgs",
    )
    assert float(fit_free(projected, data, rank=se.FullRank()).chi2) < 1e-12
    points, values = projected.point_columns(se.NonNegativeCells(point_masses=5))
    boundary = np.flatnonzero(points.coordinates["gamma"] == 300)
    assert boundary.size == 1
    edge = basis.predict(source.replace(gamma=se.Delta(300)))
    np.testing.assert_allclose(values[..., boundary[0]], edge.stokes, rtol=1e-12)


@pytest.mark.parametrize("foreground", [False, True])
def test_independent_higher_screen_is_unbounded_in_regime_ledger(setup, foreground):
    conditional, source, phase = setup
    screen = _depth(300, ETA)
    if foreground:
        source = source.replace(depth=se.NoRotation())
        phase = phase.with_screen(screen.characteristic)
    else:
        source = source.replace(depth=screen)
    basis = se.RegimeBasis.build(
        source,
        conditional.channels,
        phase=phase,
        harmonic=conditional.harmonic.lattice,
        continuum=conditional.continuum.lattice,
        angular_nodes=8,
        frequency_nodes=64,
        check=False,
    )
    assert basis.predict(source).ledger["depth_cumulant_truncation"].kind == "unbounded"
