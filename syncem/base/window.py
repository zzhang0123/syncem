"""Energy window of the eta-averaged continuum kernel (design/0.8/SPEC.md Sec. 4.6, rule 1).

LABEL ``eq: directional continuum``. For electrons with density ``gamma^-p`` the emission at one frequency,
written in ``y = nu / (a_1 B gamma^2)`` with ``a_1 = 3 e / (4 pi m_e c)`` and ``B`` the total field, has the
weight ``y^((p - 3)/2) Fbar(y) dy`` with ``Fbar(y) = <s F(y / s)>_eta`` and ``s = sqrt(1 - eta^2)``. The window
``[y_lo, y_hi]`` leaves the fraction ``eps`` of that weight on each side (PAPER:1059). ``energy_range`` maps it
to ``gamma`` with the channel-support edges and the field support, a conservative bound because every response
vanishes outside its support: ``gamma_lo = sqrt(nu_lo / (a_1 B_hi y_hi))`` and
``gamma_hi = sqrt(nu_hi / (a_1 B_lo y_lo))``. It needs no S0 or S1 table. The grid and the cumulative trapezoid
follow ``design/0.8/prototypes/synth/window.py``, with the closed-form tail below ``y = 1e-8``.

Units: ``y`` dimensionless, frequencies in Hz, fields in G. Shapes: the inputs and outputs are scalars, and the
grid has shape ``(6000,)``. Not certified: the trapezoid on 6000
log-spaced points, validated against the reference script to 1e-4 in ``y_hi`` and 1.1e-3 in ``y_lo`` (the
reference tabulates ``F`` coarsely at small ``x``), and to 1.3e-5 against a 40000-point grid down to ``y = 1e-20``.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from ..constants import C_CGS, E_ESU, M_E
from ..ultrarel import F

__all__ = [
    "A1_HZ_PER_G",
    "continuum_window",
    "continuum_mixture_window",
    "energy_range",
    "continuum_reference_outside",
]

A1_HZ_PER_G = 3.0 * E_ESU / (4.0 * math.pi * M_E * C_CGS)
_LOG_Y = np.linspace(math.log(1e-8), math.log(200.0), 6000)
_REFERENCE_BLOCK = 16_384


def _isotropic() -> tuple[np.ndarray, np.ndarray]:
    t, w = np.polynomial.legendre.leggauss(48)
    return 0.5 * t + 0.5, 0.5 * w


def _average_F(y, sine, mass):
    # F expands each input into a modified Bessel quadrature. Bound the
    # number of inputs as well as the output to keep that work bounded.
    result = np.zeros_like(y)
    for start in range(0, sine.size, _REFERENCE_BLOCK):
        stop = min(sine.size, start + _REFERENCE_BLOCK)
        s = sine[start:stop]
        weights = jnp.asarray(mass[start:stop] * s)
        step = _REFERENCE_BLOCK // s.size
        for i in range(0, y.size, step):
            keep = slice(i, i + step)
            x = jnp.asarray(y[keep, None] / s[None, :])
            result[keep] += np.asarray(
                jnp.sum(weights[None, :] * F(jnp.minimum(x, 1e4)), axis=1)
            )
    return result


def _continuum_cumulative(averaged, density):
    nodes, probs = (
        np.asarray(a, dtype=float) for a in averaged.get("eta", _isotropic())
    )
    if (
        nodes.ndim != 1
        or nodes.shape != probs.shape
        or not np.all(np.isfinite(nodes))
        or np.any(~np.isfinite(probs) | (probs < 0))
        or not np.isclose(probs.sum(), 1.0, rtol=1e-10)
        or np.any(np.abs(nodes) >= 1)
    ):
        raise ValueError(
            "the eta law needs nodes inside (-1, 1) and probabilities summing to one"
        )
    slope = (density - 1.0) / 2.0 + 1.0 / 3.0
    if not slope > 0:
        raise ValueError(
            f"density={density!r}: the weight diverges at small y for density <= 1/3"
        )
    s = np.sqrt(1.0 - nodes**2)
    y = np.exp(_LOG_Y)
    fbar = _average_F(y, s, probs)
    weight = fbar * y ** ((density - 3.0) / 2.0) * y
    tail = weight[0] / slope
    cum = tail + np.concatenate(
        [[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(_LOG_Y))]
    )
    return cum, tail, slope


def continuum_window(
    averaged: dict, channels: Any = None, *, density: float = 2.6, eps: float = 1e-3
) -> tuple[float, float]:
    """``(y_lo, y_hi)`` for the averaged laws ``{name: (nodes, probabilities)}`` and the slope ``density``.

    The inclination law is ``averaged["eta"]``, isotropic when absent. Below the grid the weight is
    ``c y^a`` with ``a = (p - 1)/2 + 1/3`` (``Fbar ~ y^(1/3)``), whose integral is added in closed form.
    """
    if not isinstance(averaged, Mapping):
        raise TypeError(
            'continuum_window takes the averaged laws as {"eta": (nodes, probabilities)}. Evaluate a '
            "quadrature rule first, for example rule({}), and pass its nodes and probabilities"
        )
    if not 0 < eps < 0.5:
        raise ValueError(f"eps must lie in (0, 0.5), not {eps!r}")
    cum, tail, slope = _continuum_cumulative(averaged, density)
    total = cum[-1]
    target_lo, target_hi = eps * total, (1 - eps) * total
    if target_lo < tail:
        y_lo = float(np.exp(_LOG_Y[0]) * (target_lo / tail) ** (1.0 / slope))
    else:
        y_lo = float(np.exp(np.interp(target_lo, cum, _LOG_Y)))
    return y_lo, float(np.exp(np.interp(target_hi, cum, _LOG_Y)))


def continuum_reference_outside(channels, cells, gamma_min, field, eta, density=2.6):
    """Share of reference I outside disjoint gamma intervals on channel quadrature."""
    cum, tail, slope = _continuum_cumulative({"eta": eta}, density)
    nu = np.asarray(channels.nodes, dtype=float)
    response = np.asarray(channels.response_weights())
    beta = (density - 1) / 2
    response = response * (nu / nu.max()) ** (-beta)
    band_total = np.zeros(channels.n_ch)
    band_inside = np.zeros_like(band_total)
    nodes, probabilities = (np.asarray(x) for x in field)
    if np.any(~np.isfinite(nodes) | (nodes < 0)) or not nodes.size:
        raise ValueError("reference field nodes must be finite and nonnegative")
    ref = nodes.max()
    if not ref > 0:
        return None

    def cumulative(scale, gamma):
        if not np.isfinite(gamma):
            return np.zeros_like(scale)
        log_y = scale - 2 * math.log(gamma)
        area = np.interp(log_y, _LOG_Y, cum)
        low = log_y < _LOG_Y[0]
        area[low] = tail * np.exp((log_y[low] - _LOG_Y[0]) * slope)
        return area

    edges = tuple((max(lo, gamma_min), hi) for lo, hi in cells if hi > gamma_min)
    for b, probability in zip(nodes, probabilities, strict=True):
        if b == 0 or probability == 0:
            continue
        scale = np.log(nu) - math.log(A1_HZ_PER_G) - math.log(b)
        total = cumulative(scale, gamma_min)
        inside = sum(cumulative(scale, lo) - cumulative(scale, hi) for lo, hi in edges)
        weight = probability * (b / ref) ** (1 + beta)
        band_inside += weight * np.sum(inside * response, axis=-1)
        band_total += weight * np.sum(total * response, axis=-1)
    if np.any(band_total <= 0):
        return None
    return np.clip(1 - band_inside / band_total, 0, 1)


def continuum_mixture_window(averaged, field, *, density=2.6, eps=1e-3):
    """Reference window on independent field quadrature nodes, in units of the largest node's field."""
    if not 0 < eps < 0.5:
        raise ValueError("eps must be in (0, 0.5)")
    b, probability = (np.asarray(a, dtype=float) for a in field)
    if (
        b.ndim != 1
        or probability.shape != b.shape
        or np.any(~np.isfinite(b) | (b < 0))
        or np.any(~np.isfinite(probability) | (probability < 0))
    ):
        raise ValueError("field nodes need finite nonnegative fields and probabilities")
    keep = (b > 0) & (probability > 0)
    if not np.any(keep):
        raise ValueError(
            "the continuum reference mixture needs emission at a positive field"
        )
    b, probability = b[keep], probability[keep]
    ref = float(b.max())
    shifts = np.log(b / ref)
    mass = probability * np.exp(shifts * ((density + 1) / 2))
    mass /= mass.sum()
    cum, tail, slope = _continuum_cumulative(averaged, density)

    def cdf(log_y):
        x = log_y - shifts
        part = np.interp(x, _LOG_Y, cum)
        low = x < _LOG_Y[0]
        part[low] = tail * np.exp((x[low] - _LOG_Y[0]) * slope)
        return np.dot(mass, part / cum[-1])

    def quantile(p):
        lo = _LOG_Y[0] + shifts.min() + np.log(p) / slope
        hi = _LOG_Y[-1]
        for _ in range(56):
            mid = (hi + lo) / 2
            if cdf(mid) < p:
                lo = mid
            else:
                hi = mid
        return float(np.exp((hi + lo) / 2))

    return (quantile(eps), quantile(1 - eps)), ref


def energy_range(
    window: tuple[float, float], nu: tuple[float, float], field: tuple[float, float]
) -> tuple[float, float]:
    """``(gamma_lo, gamma_hi)`` from the ``y`` window, the channel-support edges and the field support."""
    (y_lo, y_hi), (nu_lo, nu_hi), (b_lo, b_hi) = window, nu, field
    if not (0 < nu_lo <= nu_hi and 0 < b_lo <= b_hi and 0 < y_lo < y_hi):
        raise ValueError(
            "energy_range needs positive, ordered frequencies, fields and window"
        )
    return math.sqrt(nu_lo / (A1_HZ_PER_G * b_hi * y_hi)), math.sqrt(
        nu_hi / (A1_HZ_PER_G * b_lo * y_lo)
    )
