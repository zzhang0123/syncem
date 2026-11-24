"""Harmonic channel values against direct sums and the frozen host route."""

from types import SimpleNamespace

import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.constants import C_CGS, C_SI_M, E_ESU, M_E
from syncem.engine import _regime_harmonic as harmonic
from syncem.model.high_order_harmonic import high_order_channel_modes
from syncem.model.phase import TaylorPhase

_GYRO = E_ESU / (2 * np.pi * M_E * C_CGS)


def _legacy(channels, gamma, field, mu, eta, **settings):
    result = high_order_channel_modes(channels, gamma, field, mu, eta, **settings)
    return np.column_stack((result.modes.I, result.modes.P[0])), result.routes


def _rotation(nu, depth):
    return np.exp(1j * 2 * (C_SI_M / nu) ** 2 * depth)


@pytest.mark.parametrize(
    ("centre", "width", "route"),
    [
        (10_000, 2047.8, "direct"),
        (10_000.5, 2047.8, "direct"),
        (10_000, 2048.2, "corrected_integral"),
    ],
)
def test_both_routes_against_integer_sum_at_threshold(
    monkeypatch, centre, width, route
):
    gamma = 20.0
    spacing = _GYRO / gamma
    channels = Channels.bump(
        [centre * spacing], [width * spacing], normalisation="unit_integral"
    )
    depth = 90.0

    def phase(nu):
        return _rotation(nu, depth)

    default, used = _legacy(
        channels, gamma, 1.0, 0.0, 0.0, phase=TaylorPhase(0), depth_ref=depth
    )
    assert used == (route,)
    ours = harmonic.harmonic_channel(
        channels, gamma, 1.0, 0.0, 0.0, phase=phase, frequency_nodes=128
    )
    np.testing.assert_allclose(ours, default, rtol=9e-12, atol=0)
    direct, _ = _legacy(
        channels,
        gamma,
        1.0,
        0.0,
        0.0,
        direct_limit=5000,
        phase=TaylorPhase(0),
        depth_ref=depth,
    )
    monkeypatch.setattr(harmonic, "_DIRECT_LIMIT", 1)
    dense = harmonic.harmonic_channel(
        channels, gamma, 1.0, 0.0, 0.0, phase=phase, frequency_nodes=128
    )
    np.testing.assert_allclose(dense, direct, rtol=9e-12, atol=0)


def test_pinned_sparse_line_in_tapered_channel():
    gamma, field, mu, eta = 10.0, 1.0, 0.23, 0.2
    spacing_at_zero_mu = _GYRO / gamma
    channels = Channels.bump(
        [300 * spacing_at_zero_mu],
        [20 * spacing_at_zero_mu],
        normalisation="unit_integral",
    )
    result = harmonic.harmonic_channel(channels, gamma, field, eta, mu)
    np.testing.assert_allclose(
        result[0], [9.37279847e-23, -8.78720446e-23], rtol=1e-8, atol=0
    )


@pytest.mark.parametrize(
    ("gamma", "centre", "width", "eta", "expected_i", "rtol"),
    [
        (300.0, 1769982.0602225773, 2e5, 0.35, 2.24194617e-26, 5e-8),
        (300.0, 1e8, 2e7, 0.0, 4.18497533e-45, 4e-7),
        (100.0, 1e10, 2e9, 0.0, 0.0, 0),
    ],
)
def test_high_order_bright_and_faint_extremes(
    gamma, centre, width, eta, expected_i, rtol
):
    channels = Channels.bump([centre], [width], normalisation="unit_integral")
    legacy, used = _legacy(
        channels,
        gamma,
        5e-6,
        eta,
        eta,
        rtol=1e-8,
        max_quad_order=512,
        allow_unconverged=True,
    )
    assert used == ("corrected_integral",)
    result = harmonic.harmonic_channel(
        channels, gamma, 5e-6, eta, eta, frequency_nodes=128
    )
    np.testing.assert_allclose(result, legacy, rtol=rtol, atol=0)
    np.testing.assert_allclose(result[0, 0], expected_i, rtol=rtol, atol=0)


def test_pitch_density_matches_degree_zero_and_refines_the_narrow_ridge():
    channels = Channels.at([1e6, 1e8], unit="Hz", half_width=0.2)
    prepared = harmonic.prepare_harmonic_channels(channels, frequency_nodes=128)
    params = channels, 300.0, 5e-6, 0.33
    coarse = harmonic.harmonic_channel(
        *params, lambda mu: 0.5, angular_nodes=16, prepared=prepared
    )
    fine = harmonic.harmonic_channel(
        *params, lambda mu: 0.5, angular_nodes=32, prepared=prepared
    )
    polynomial = harmonic.harmonic_channel(
        *params, None, degrees=(0, 2), angular_nodes=64, prepared=prepared
    )
    expected = [
        [7.45888123e-29, -5.23988199e-29],
        [9.93527680e-50, -9.80303569e-50],
    ]
    np.testing.assert_allclose(polynomial[0], expected, rtol=1e-6, atol=0)
    np.testing.assert_allclose(fine, polynomial[0], rtol=1e-6, atol=0)
    np.testing.assert_allclose(coarse, polynomial[0], rtol=3e-5, atol=0)
    assert np.max(np.abs(coarse - polynomial[0])) > 4 * np.max(
        np.abs(fine - polynomial[0])
    )


def test_dense_underflow_screen_preserves_mixed_pitch_and_channel_values(monkeypatch):
    channels = Channels.at([2e5, 2e6, 1e8, 1e10], unit="Hz", half_width=0.2)
    args = (channels, 300.0, 5e-6, 0.3, lambda mu: np.full_like(mu, 0.5))
    screened = harmonic.harmonic_channel(*args, angular_nodes=32)
    monkeypatch.setattr(harmonic, "_BESSEL_UNDERFLOW_EXPONENT", np.inf)
    full = harmonic.harmonic_channel(*args, angular_nodes=32)
    np.testing.assert_array_equal(screened, full)
    assert np.all(screened[:3, 0].real > 0)
    assert screened[3, 0] == 0


def test_dense_underflow_screen_keeps_the_cutoff_node(monkeypatch):
    special = harmonic._scipy_special()
    calls = []

    def capture(order, x):
        calls.append(order)
        return special.jv(order, x)

    monkeypatch.setattr(harmonic, "_scipy_special", lambda: SimpleNamespace(jv=capture))
    cutoff = 3 * harmonic._BESSEL_UNDERFLOW_EXPONENT
    prepared = SimpleNamespace(
        frequency=np.asarray([[cutoff, cutoff + 1, cutoff + 2]]),
        weight=np.ones((1, 3)),
        phased_weight=np.ones((1, 3)),
    )
    zero, one = np.zeros(1), np.ones(1)
    result = harmonic._dense_sum(
        prepared, np.array([0]), 100.0, 5e-6, one, one, one, zero, zero
    )
    assert len(calls) == 2
    np.testing.assert_array_equal(calls[0], [cutoff - 1, cutoff])
    np.testing.assert_array_equal(calls[1], [cutoff + 1, cutoff + 2])
    np.testing.assert_array_equal(result, np.zeros((1, 2)))


def test_restricted_density_and_axis_fundamental():
    gamma, field, eta = 20.0, 1.0, 1.0
    channels = Channels.at([_GYRO / gamma], unit="Hz", half_width=0.2)

    def pitch(mu):
        assert np.all((mu >= -0.1) & (mu <= 0.1))
        return 5 * np.ones_like(mu)

    result = harmonic.harmonic_channel(
        channels, gamma, field, eta, pitch, pitch_support=(-0.1, 0.1), angular_nodes=24
    )
    nodes, weights = np.polynomial.legendre.leggauss(64)
    reference = [_legacy(channels, gamma, field, mu, eta)[0][0] for mu in 0.1 * nodes]
    expected = 0.5 * weights @ reference
    np.testing.assert_allclose(result[0, 0], expected[0], rtol=3e-13, atol=0)
    assert abs(expected[1]) <= np.finfo(float).eps * expected[0].real
    assert result[0, 1] == 0
    assert result[0, 0].real > 0


def test_sparse_table_and_dense_table_refusal():
    gyro = _GYRO / 10
    sparse = Channels.from_bandpass(
        [([10 * gyro, 11 * gyro, 12 * gyro], [0, 1, 0])], unit="Hz"
    )
    expected, _ = _legacy(sparse, 10.0, 1.0, 0.0, 0.0)
    actual = harmonic.harmonic_channel(sparse, 10.0, 1.0, 0.0, 0.0)
    np.testing.assert_array_equal(actual, expected)
    dense = Channels.from_bandpass(
        [([10_000 * gyro, 15_000 * gyro, 20_000 * gyro], [0, 1, 0])], unit="Hz"
    )
    with pytest.raises(ValueError, match="bump with smooth support"):
        harmonic.harmonic_channel(dense, 10.0, 1.0, 0.0, 0.0)


def test_monochromatic_harmonics_are_refused():
    with pytest.raises(ValueError, match="nonzero width"):
        harmonic.harmonic_channel(Channels.at([1.0], unit="GHz"), 20.0, 1.0, 0.0, 0.0)
