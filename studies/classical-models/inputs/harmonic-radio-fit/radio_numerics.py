"""Harmonic radio channels and piecewise log-energy moment coefficients."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
import sys

import numpy as np
from numpy.polynomial.legendre import leggauss

PACKAGE = Path(__file__).resolve().parents[4]
PIN = "d51f9bef4fa744a536a60c3da821b1033818a232"
PACKAGE_SHA = "a5ed224ceaff820992b042ca28b37cbd7be0732ff97d855c980e5de8ac138b52"
sys.path.insert(0, str(PACKAGE))
from syncmoments.constants import C_CGS, C_SI_M, E_ESU, M_E  # noqa: E402
from stable_lines import high_order_line_powers, omitted_tail_envelope  # noqa: E402

B0 = 5e-6
GYRO0 = E_ESU * B0 / (2 * np.pi * M_E * C_CGS)
CENTRES = np.geomspace(1e8, 1e10, 41)
HALF = 0.20
ENERGY_EDGES = np.geomspace(300.0, 1e5, 13)
LOG_EDGES = np.log(ENERGY_EDGES)
LOG_MIDS = (LOG_EDGES[1:] + LOG_EDGES[:-1]) / 2
LOG_HALF = np.diff(LOG_EDGES) / 2
GAMMA_MIDS = np.exp(LOG_MIDS)
SLOPE = 2.6
SCREEN_MEAN = 8.0
SCREEN_SD = 0.5
SKY = 0.4 * np.exp(2j * 0.25)
REF_BIN_MASS = np.diff(ENERGY_EDGES ** (1 - SLOPE))
REF_BIN_MASS /= REF_BIN_MASS.sum()
_bump_x, _bump_w = leggauss(256)
BUMP_AREA = float(np.dot(_bump_w, np.exp(1 - 1 / (1 - _bump_x**2))))


def package_digest():
    digest = hashlib.sha256()
    for path in sorted((PACKAGE / "syncmoments").rglob("*.py")):
        digest.update(path.relative_to(PACKAGE / "syncmoments").as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def model():
    return {
        "gamma_bounds": ENERGY_EDGES[[0, -1]].tolist(),
        "energy_density": "normalised gamma^-2.6 with respect to dgamma",
        "log_bin_edges": LOG_EDGES.tolist(),
        "coefficient_coordinate": "x_b=(ln(gamma)-ln(gamma_b))/h_b, within bin b",
        "reference_bin_masses": REF_BIN_MASS.tolist(),
        "moment_metric": "a_b,r=E[1_bin(b) x_b^r]/reference_bin_mass_b; Euclidean metric",
        "degree": 2,
        "B_gauss_uniform": [4e-6, 6e-6],
        "mu_density": "uniform[-1,1]",
        "eta_density": "uniform[-0.5,0.5]",
        "azimuth_density": "[1+0.8 cos(2(phi-0.25))]/(2pi)",
        "screen_gaussian_mean_sd_rad_m2": [SCREEN_MEAN, SCREEN_SD],
        "independence": "energy, B, mu, eta, phi and depth independent",
        "channel_centres_Hz": CENTRES.tolist(),
        "channel_response": "exp(1-1/(1-t^2))/(halfwidth*BUMP_AREA) for |t|<1; zero otherwise",
        "channel_halfwidth_fraction": HALF,
        "channel_normalisation": "unit integral over frequency in Hz",
        "radiation": "finite-gamma helical harmonic line powers, stable cubic Airy evaluation at high order; Euler-Maclaurin channel sum quadrature",
    }


def uniform(lo, hi, nodes):
    x, w = leggauss(nodes)
    return (lo + hi) / 2 + (hi - lo) * x / 2, w / 2


def pitch_nodes(gamma, eta, nodes):
    x, w = leggauss(nodes)
    mus, masses = [], []
    for sign, extent in ((-1, eta + 1), (1, 1 - eta)):
        maximum = math.asinh(extent * gamma)
        cuts = np.r_[np.arange(0.0, maximum, 2.0), maximum]
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            t = (lo + hi) / 2 + (hi - lo) * x / 2
            mus.extend(eta + sign / gamma * np.sinh(t))
            masses.extend((hi - lo) * w / gamma * np.cosh(t) / 4)
    return np.asarray(mus), np.asarray(masses)


def dense_channels(gamma, mu, eta, fields, centres, nq):
    beta = math.sqrt(1 - gamma**-2)
    d = 1 - beta * mu * eta
    spacing = GYRO0 * (fields[:, None] / B0) / (gamma * d)
    lo, hi = (1 - HALF) * centres, (1 + HALF) * centres
    lower = np.maximum(1.0, np.ceil(lo / spacing))
    upper = np.floor(hi / spacing)
    if np.any(upper <= lower):
        raise ValueError("channel interval is not dense")
    x, w = leggauss(nq)
    orders = (lower + upper)[..., None] / 2 + (upper - lower)[..., None] * x / 2
    orders = np.concatenate([orders, lower[..., None], upper[..., None]], axis=-1)
    li, lq, _, _ = high_order_line_powers(orders, gamma, B0, mu, eta)
    tail = omitted_tail_envelope(orders, gamma, B0, mu, eta)
    frequency = orders * spacing[..., None]
    t = (frequency - centres[None, :, None]) / (HALF * centres[None, :, None])
    band = np.zeros_like(t)
    inside = abs(t) < 1
    band[inside] = np.exp(1 - 1 / (1 - t[inside] ** 2))
    band /= HALF * centres[None, :, None] * BUMP_AREA
    weights = np.concatenate(
        [(upper - lower)[..., None] * w / 2, np.full((*lower.shape, 2), 0.5)], axis=-1
    )
    lam2 = (C_SI_M / frequency) ** 2
    screen = np.exp(2j * SCREEN_MEAN * lam2 - 2 * SCREEN_SD**2 * lam2**2)
    field_scale = (fields / B0)[:, None, None] ** 2
    weight = weights * band * field_scale
    return (
        np.sum(weight * li, axis=-1),
        SKY * np.sum(weight * lq * screen, axis=-1),
        np.sum(weight * tail, axis=-1),
    )


def kernel(gamma, settings, centres=CENTRES):
    fields, fw = uniform(4e-6, 6e-6, settings["field"])
    etas, ew = uniform(-0.5, 0.5, settings["eta"])
    result = np.zeros((len(centres), 3))
    tail = np.zeros(len(centres))
    for eta, we in zip(etas, ew):
        mus, mw = pitch_nodes(gamma, eta, settings["pitch"])
        for mu, wm in zip(mus, mw):
            intensity, polarization, envelope = dense_channels(
                gamma, float(mu), float(eta), fields, centres, settings["channel"]
            )
            i = fw @ intensity
            p = fw @ polarization
            result[:, 0] += we * wm * i
            result[:, 1] += we * wm * p.real
            result[:, 2] += we * wm * p.imag
            tail += we * wm * (fw @ envelope)
    if not np.all(np.isfinite(result)) or np.min(result[:, 0]) < 0:
        raise ValueError("invalid integrated harmonic response")
    return result, tail


def basis(settings):
    step = settings["step"]
    result = np.zeros((len(CENTRES), 3, 36))
    pure = np.zeros((12, 3, len(CENTRES), 3))
    for b, (mid, half) in enumerate(zip(LOG_MIDS, LOG_HALF)):
        values = np.stack(
            [
                kernel(float(np.exp(mid + half * step * offset)), settings)[0]
                for offset in range(-2, 3)
            ]
        )
        pure[b, 0] = values[2]
        pure[b, 1] = (values[0] - 8 * values[1] + 8 * values[3] - values[4]) / (
            12 * step
        )
        pure[b, 2] = (
            -values[0] + 16 * values[1] - 30 * values[2] + 16 * values[3] - values[4]
        ) / (24 * step**2)
        for r in range(3):
            result[:, :, 3 * b + r] = REF_BIN_MASS[b] * pure[b, r]
        print(f"basis bin {b+1}/12 gamma={np.exp(mid):.3f}", flush=True)
    return {"basis": result, "pure_coefficients": pure}


def truth(settings):
    result = np.zeros((len(CENTRES), 3))
    tail = np.zeros(len(CENTRES))
    moments = np.zeros(36)
    x, w = leggauss(settings["energy"])
    normalisation = (
        ENERGY_EDGES[-1] ** (1 - SLOPE) - ENERGY_EDGES[0] ** (1 - SLOPE)
    ) / (1 - SLOPE)
    for b, (mid, half) in enumerate(zip(LOG_MIDS, LOG_HALF)):
        gammas = np.exp(mid + half * x)
        weights = half * w * gammas ** (1 - SLOPE) / normalisation
        for gamma, weight in zip(gammas, weights):
            values, envelope = kernel(float(gamma), settings)
            result += weight * values
            tail += weight * envelope
        for r in range(3):
            moments[3 * b + r] = np.sum(weights * x**r) / REF_BIN_MASS[b]
        print(f"truth bin {b+1}/12", flush=True)
    return {"truth": result, "moments": moments, "omitted_tail": tail}
