"""Small-x continuation at each frequency node (SPEC Sec. 8.3, decision 198).

Both methods are evaluated directly, bypassing the dispatcher: the continuum
kernel with a lowered ``x_min`` and the asymptote probed at the threshold.
"""

import numpy as np
import pytest

import syncem as se
from syncem.base.window import A1_HZ_PER_G
from syncem.distributions import Delta, Isotropic
from syncem.engine.direct import TAIL_X, _ContinuumTail

CHANNELS = se.Channels.at([0.01, 1.4, 300.0], unit="GHz", half_width=0.1, n_nu=8)
REFERENCE = se.ContinuumKernel(x_min=1e-11)
TAIL = _ContinuumTail(se.ContinuumKernel(x_min=TAIL_X), TAIL_X)


def _both(gamma, b, eta, j):
    """Channel j by the kernel and by the tail evaluated on all channels."""
    point = {"gamma": gamma, "B": b, "eta": eta}
    kernel = REFERENCE.channel(CHANNELS[j], point, phase=None)
    tail = TAIL.channel(CHANNELS, point, phase=None)
    return (kernel.I[0], kernel.P[0, 0].real), tail


def _rel(a, b):
    a, b = np.asarray(a), np.asarray(b)
    return np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), 1e-300)


@pytest.mark.parametrize("b", [1e-7, 5e-6, 1e-3])
@pytest.mark.parametrize("eta", [0.0, 0.9])
@pytest.mark.parametrize("factor", [1 - 1e-3, 1.0, 1 + 1e-3, 3.0])
def test_kernel_and_asymptote_agree_at_and_above_each_channel_threshold(b, eta, factor):
    point = {"B": b, "eta": eta}
    lows = np.min(np.asarray(CHANNELS.nodes), axis=-1)
    for j, low in enumerate(lows):
        threshold = float(TAIL._threshold(float(low), point))
        (i, q), tail = _both(factor * threshold, b, eta, j)
        assert np.all(np.isfinite(tail.I)) and np.all(np.isfinite(tail.P))
        # Below the threshold the tail is the kernel. Above it the asymptote
        # errs by about 0.844 x_tail^(2/3) = 8.4e-7.
        assert _rel(i, tail.I[j]) < 2e-6
        assert _rel(q, tail.P[0, j].real) < 2e-6


@pytest.mark.parametrize("gamma", [300.0, 1e4, 1e6])
def test_tail_is_the_kernel_far_below_every_threshold(gamma):
    for j in range(CHANNELS.n_ch):
        (i, q), tail = _both(gamma, 5e-6, 0.0, j)
        np.testing.assert_allclose(tail.I[j], i, rtol=1e-12, atol=0)
        np.testing.assert_allclose(tail.P[0, j].real, q, rtol=1e-12, atol=0)


def _review_case(gamma):
    channels = se.Channels.at([0.01, 1.4], unit="GHz", half_width=0.1, n_nu=8)
    population = se.Population(gamma=Delta(gamma), B="5 uG", eta=0, mu=Isotropic())
    lattice = se.Lattice(gamma=se.Cells.log(300, 2000, 2, order=1))
    basis = se.spectral_basis(population, channels, lattice=lattice, check=False)
    return channels, population, basis


def test_direct_outside_the_lattice_matches_kernel_calls():
    """Review P1-4: before the fix the 1.4 GHz channel gave 3.96e-32 and 0.01 GHz was 11% low."""
    channels, population, basis = _review_case(1e4)
    got = basis.direct(population)
    point = {"gamma": 1e4, "B": 5e-6, "eta": 0.0}
    want = REFERENCE.channel(channels, point, phase=None).I
    np.testing.assert_allclose(got.I, want, rtol=1e-10)
    np.testing.assert_allclose(
        got.I, [3.2932774187532353e-29, 7.488000615647577e-29], rtol=1e-8
    )
    assert "small_x:tail" not in [term.name for term in got.ledger.terms]


def test_direct_into_the_tail_carries_the_unbounded_term():
    channels, population, basis = _review_case(1e8)
    got = basis.direct(population)
    point = {"gamma": 1e8, "B": 5e-6, "eta": 0.0}
    want = REFERENCE.channel(channels, point, phase=None).I
    assert np.all(_rel(got.I, want) < 2e-6)
    assert got.ledger["small_x:tail"].kind == "unbounded"


def test_wide_channel_continues_only_its_small_x_nodes():
    channels = se.Channels.from_bandpass(
        [([1e7, 1.1e7, 9.99e10, 1e11], [0.0, 1.0, 1.0, 0.0])],
        unit="Hz",
        n_nu=128,
    )
    point = {"gamma": 93660271.26705244, "B": 5e-6, "eta": 0.4, "depth": 1.23}
    phase = se.FaradayRotation()
    reference = se.ContinuumKernel(x_min=1e-12)
    expected = reference.channel(channels, point, phase=phase, per_node=True)
    got = TAIL.channel(channels, point, phase=phase, per_node=True)

    scale = (
        A1_HZ_PER_G * point["B"] * np.sqrt(1 - point["eta"] ** 2) * point["gamma"] ** 2
    )
    above_guard = np.asarray(channels.nodes[0]) / scale > TAIL_X
    assert np.any(above_guard) and np.any(~above_guard)
    for actual, target in ((got.I, expected.I), (got.P, expected.P)):
        np.testing.assert_allclose(
            actual[..., above_guard], target[..., above_guard], rtol=1e-13, atol=0
        )
        np.testing.assert_allclose(actual, target, rtol=1e-6, atol=0)
    integrated = TAIL.channel(channels, point, phase=phase)
    np.testing.assert_allclose(integrated.P, np.sum(got.P, axis=-1), rtol=1e-14, atol=0)
    population = se.Population(
        gamma=Delta(point["gamma"]), B="5 uG", eta=0.4, mu=Isotropic()
    )
    lattice = se.Lattice(gamma=se.Cells.log(300, 2000, 2, order=1))
    basis = se.spectral_basis(population, channels, lattice=lattice, check=False)
    direct = basis.direct(population)
    np.testing.assert_allclose(direct.I, [1.15745735e-30], rtol=1e-6, atol=0)
    assert direct.ledger["small_x:tail"].kind == "unbounded"


def test_near_axis_tail_and_budgeted_check():
    channels = se.Channels.at([0.01], unit="GHz")
    point = {"gamma": 1e9, "B": 5e-6, "eta": 0.999}
    population = se.Population(
        gamma=Delta(1e9), B="5 uG", eta=point["eta"], mu=Isotropic()
    )
    lattice = se.Lattice(gamma=se.Cells.log(300, 2000, 2, order=1))
    basis = se.spectral_basis(population, channels, lattice=lattice, check=False)
    reference = se.ContinuumKernel(x_min=1e-12).channel(channels, point, phase=None)
    np.testing.assert_allclose(reference.I, [1.97255957e-33], rtol=1e-8, atol=0)
    np.testing.assert_allclose(
        basis.direct(population).I, reference.I, rtol=1e-6, atol=0
    )
    assert (
        basis.predict(population, check=True).ledger["small_x:tail"].kind == "unbounded"
    )
    skipped = basis.predict(population, check=se.AccuracyCheck(max_seconds=0))
    assert skipped.ledger["representation"].kind == "unbounded"
    assert "small_x:tail" not in {term.name for term in skipped.ledger.terms}


@pytest.mark.parametrize("b,eta", [(0.0, 0.0), (5e-6, -1.0), (5e-6, 1.0)])
def test_zero_transverse_field_emits_zero(b, eta):
    point = {"gamma": 1e9, "B": b, "eta": eta}
    for mode in TAIL.channel(CHANNELS, point, phase=None, per_node=True):
        np.testing.assert_array_equal(mode, np.zeros_like(mode))


def test_unbounded_field_bound_enables_local_tail_and_rejects_invalid_bounds():
    from syncem.engine.average import AveragedKernel
    from syncem.engine.direct import (
        continuum_tail_threshold,
        continuum_tail_thresholds,
        node_modes,
    )

    kernel = se.ContinuumKernel(x_min=1e-7)
    channels = se.Channels.at([0.01, 1.4], unit="GHz", half_width=0.1, n_nu=8)
    assert continuum_tail_thresholds(kernel, channels, np.inf) is None
    assert continuum_tail_threshold(kernel, channels, np.inf) is None
    for bad in (-1.0, -np.inf, np.nan):
        with pytest.raises(ValueError, match="field in G"):
            continuum_tail_thresholds(kernel, channels, bad)
    avg = AveragedKernel(kernel, fixed={"B": 5e-6, "eta": 0.0})
    gamma = np.array([1e4, 1e8])
    got = node_modes(avg, channels, {"gamma": gamma}, b_perp_max=np.inf)
    want = node_modes(
        AveragedKernel(REFERENCE, fixed={"B": 5e-6, "eta": 0.0}),
        channels,
        {"gamma": gamma},
    )
    np.testing.assert_allclose(got.I[0], want.I[0], rtol=1e-12, atol=0)
    assert np.all(_rel(got.I[1], want.I[1]) < 2e-6)
    assert np.all(_rel(got.P[1], want.P[1]) < 2e-6)


def test_conditional_field_predict_and_direct_with_continuum_kernel():
    import jax.numpy as jnp

    channels = se.Channels.log(0.5, 5.0, 4, unit="GHz", half_width=0.1, n_nu=8)
    lattice = se.Lattice(gamma=se.Cells.log(1e3, 1e4, 6, order=2))

    pop_eta = se.Population(
        gamma=se.PowerLaw(2.5, column=1.0, lo=1e3, hi=1e4),
        B=lambda eta: se.Delta(5.0 + 1.0 * eta, unit="uG"),
        eta=se.Uniform(-0.5, 0.5),
        mu=Isotropic(),
        phi=se.AzimuthFactor(0.4),
        depth=se.NoRotation(),
    )
    basis_eta = se.spectral_basis(pop_eta, channels, lattice=lattice, check=False)
    pred_eta = basis_eta.predict(pop_eta, check=True)
    direct_eta = basis_eta.direct(pop_eta)
    assert pred_eta.ledger["small_x:tail"].kind == "unbounded"
    assert direct_eta.ledger["small_x:tail"].kind == "unbounded"
    np.testing.assert_allclose(
        pred_eta.ledger["representation"].value,
        pred_eta.stokes - direct_eta.stokes,
        rtol=1e-12,
        atol=1e-42,
    )
    rel_eta = np.max(np.abs(pred_eta.stokes - direct_eta.stokes)) / np.max(direct_eta.I)
    assert rel_eta < 1e-3

    # Conditional B(phi) with AzimuthFourier must also support checked prediction.
    lat_fourier = se.Lattice(
        gamma=se.Cells.log(1e3, 1e4, 6, order=2), phi=se.AzimuthFourier(4)
    )
    pop_phi = se.Population(
        gamma=se.PowerLaw(2.5, column=1.0, lo=1e3, hi=1e4),
        B=lambda phi: se.Delta(5.0 + 1.0 * jnp.cos(phi.to("rad").value), unit="uG"),
        eta=0.0,
        mu=Isotropic(),
        phi=se.Uniform(0.0, 2 * np.pi, unit="rad"),
        depth=se.NoRotation(),
    )
    basis_phi = se.spectral_basis(pop_phi, channels, lattice=lat_fourier, check=False)
    pred_phi = basis_phi.predict(pop_phi, check=True)
    direct_phi = basis_phi.direct(pop_phi)
    assert pred_phi.ledger["small_x:tail"].kind == "unbounded"
    rel_phi = np.max(np.abs(pred_phi.stokes - direct_phi.stokes)) / np.max(direct_phi.I)
    assert rel_phi < 1e-3

    # Conditional B resolved on finite B cells uses the cell upper bound for within="cells".
    pop_full_eta = pop_eta.replace(eta=Isotropic())
    lat_cells = se.Lattice(
        gamma=se.Cells.log(1e3, 1e4, 6, order=2),
        B=se.Cells.linear("3 uG", "7 uG", 2, order=2),
        eta=se.Legendre(2),
    )
    basis_cells = se.spectral_basis(
        pop_full_eta, channels, lattice=lat_cells, check=False
    )
    pred_cells = basis_cells.predict(pop_full_eta, check=True)
    direct_in = basis_cells.direct(pop_full_eta, within="cells")
    assert "small_x:tail" not in {term.name for term in direct_in.ledger.terms}
    np.testing.assert_allclose(
        pred_cells.ledger["representation"].value,
        pred_cells.stokes - direct_in.stokes,
        rtol=1e-12,
        atol=1e-42,
    )
    rel_cells = np.max(np.abs(pred_cells.stokes - direct_in.stokes)) / np.max(
        direct_in.I
    )
    assert rel_cells < 1e-2
