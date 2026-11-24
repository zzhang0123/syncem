"""Compare a phased two-branch response to separate direct quadratures."""

import numpy as np

import syncem as se
from syncem.base.window import A1_HZ_PER_G
from syncem.constants import C_CGS, C_SI_M, E_ESU, M_E
from syncem.engine.assemble import predict_values
from syncem.model.high_order_harmonic import high_order_line_powers

B_GAUSS = 0.005
ETA = 0.5
GAMMA_SWITCH = 300.0
GAMMA_MIN = 280.0
GAMMA_MAX = 321.0
AZIMUTH = 0.25 + 0.05j
DEPTH, DEPTH_SIGMA = 3.0, 0.4
SLOPE = 2.6

_X, _W = np.polynomial.legendre.leggauss(12)


def gauss(lo, hi):
    half = (hi - lo) / 2
    return lo + half * (1 + _X), half * _W


def critical_frequency(gamma):
    return A1_HZ_PER_G * B_GAUSS * np.sqrt(1 - ETA**2) * gamma**2


def direct_harmonic(gamma, channels, response, phase):
    gyro = E_ESU * B_GAUSS / (2 * np.pi * gamma * M_E * C_CGS)
    beta = np.sqrt(1 - gamma**-2)
    theta = np.arccos(ETA)
    frequencies = np.asarray(channels.nodes)
    total = np.zeros((channels.n_ch, 2), dtype=complex)
    edges = [-12, -8, -3, -1, 0, 1, 3, 8, 12]
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        u, w = gauss(a, b)
        alpha = theta + u / gamma
        for mu, weight in zip(
            np.cos(alpha), np.sin(alpha) * w / (2 * gamma), strict=True
        ):
            spacing = gyro / (1 - beta * mu * ETA)
            orders = frequencies / spacing
            line_i, line_q, _, _ = high_order_line_powers(
                orders, gamma, B_GAUSS, mu, ETA
            )
            total[:, 0] += weight * np.sum(line_i * response, axis=1) / spacing
            total[:, 1] += weight * np.sum(line_q * response * phase, axis=1) / spacing
    return total


def direct_continuum(gamma, channels, response, phase):
    kernel = se.ContinuumKernel().model
    i, q = kernel.kernels(channels.nodes, gamma, B_GAUSS, ETA)
    return np.column_stack(
        (np.sum(i * response, axis=1), np.sum(q * response * phase, axis=1))
    )


def direct_population(lo, hi, channels, kernel):
    phase_arg = 2 * (C_SI_M / np.asarray(channels.nodes)) ** 2
    phase = np.exp(1j * DEPTH * phase_arg - 0.5 * (DEPTH_SIGMA * phase_arg) ** 2)
    response = np.asarray(channels.response_weights())
    norm = (SLOPE - 1) / (GAMMA_MIN ** (1 - SLOPE) - GAMMA_MAX ** (1 - SLOPE))
    gamma, weights = gauss(lo, hi)
    total = sum(
        weight * g ** (-SLOPE) * kernel(g, channels, response, phase)
        for g, weight in zip(gamma, weights, strict=True)
    )
    total[:, 1] *= AZIMUTH
    return norm * total


def test_two_bright_branches_with_screen():
    observed = se.Channels.at(
        [0.4, 0.8, 1.6, 3.0], unit="GHz", half_width=0.2, n_nu=128
    )
    pop = se.Population(
        gamma=se.PowerLaw(SLOPE, lo=GAMMA_MIN, hi=GAMMA_MAX, column=1.0),
        B=(B_GAUSS, "G"),
        eta=ETA,
        mu=se.Isotropic(),
        phi=se.AzimuthFactor(AZIMUTH),
        depth=se.GaussianScreen(DEPTH, DEPTH_SIGMA),
    )
    harmonic = se.Lattice(
        gamma=se.Cells.log(GAMMA_MIN, GAMMA_SWITCH, 4, order=2),
        phi=se.AzimuthHarmonic(),
    )
    continuum = se.Lattice(
        nu_c=se.Cells.log(
            critical_frequency(GAMMA_SWITCH), critical_frequency(GAMMA_MAX), 4, order=2
        ),
        phi=se.AzimuthHarmonic(),
    )
    basis = se.RegimeBasis.build(
        pop,
        observed,
        harmonic=harmonic,
        continuum=continuum,
        gamma_switch=GAMMA_SWITCH,
        angular_nodes=12,
        frequency_nodes=64,
        check=True,
    )
    result = basis.predict(pop)
    expected_low = direct_population(GAMMA_MIN, GAMMA_SWITCH, observed, direct_harmonic)
    expected_high = direct_population(
        GAMMA_SWITCH, GAMMA_MAX, observed, direct_continuum
    )
    expected = expected_low + expected_high
    fraction = expected_low[:, 0].real / expected[:, 0].real
    np.testing.assert_allclose(
        expected[0],
        [7.38132793e-26, 8.28513362e-27 + 4.02675674e-27j],
        rtol=5e-8,
        atol=0,
    )
    np.testing.assert_allclose(
        fraction,
        [0.53352855, 0.52544525, 0.50957753, 0.48188114],
        rtol=5e-8,
        atol=0,
    )

    def check(measured, reference):
        error = np.max(np.abs(measured - reference) / expected[:, :1].real)
        assert error < 3e-6

    check(np.column_stack((result.I, result.P)), expected)
    for part, moments, reference in (
        (basis.harmonic, result.moments[0], expected_low),
        (basis.continuum, result.moments[1], expected_high),
    ):
        i, q, u = predict_values(part, moments).T
        check(np.column_stack((i, q + 1j * u)), reference)

    low, high = basis.moments(pop.replace(gamma=se.Delta(GAMMA_SWITCH)))
    assert not np.any(low.values)
    assert np.any(high.values)
