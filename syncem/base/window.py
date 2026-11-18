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

__all__ = ["A1_HZ_PER_G", "continuum_window", "energy_range"]

A1_HZ_PER_G = 3.0 * E_ESU / (4.0 * math.pi * M_E * C_CGS)
_LOG_Y = np.linspace(math.log(1e-8), math.log(200.0), 6000)


def _isotropic() -> tuple[np.ndarray, np.ndarray]:
    t, w = np.polynomial.legendre.leggauss(48)
    return 0.5 * t + 0.5, 0.5 * w


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
    nodes, probs = (
        np.asarray(a, dtype=float) for a in averaged.get("eta", _isotropic())
    )
    if (
        nodes.shape != probs.shape
        or not np.isclose(probs.sum(), 1.0, rtol=1e-10)
        or np.any(np.abs(nodes) >= 1)
    ):
        raise ValueError(
            "the eta law needs nodes inside (-1, 1) and probabilities summing to one"
        )
    if not 0 < eps < 0.5:
        raise ValueError(f"eps must lie in (0, 0.5), not {eps!r}")
    slope = (density - 1.0) / 2.0 + 1.0 / 3.0
    if not slope > 0:
        raise ValueError(
            f"density={density!r}: the weight diverges at small y for density <= 1/3"
        )
    s = np.sqrt(1.0 - nodes**2)
    y = np.exp(_LOG_Y)
    x = jnp.asarray(y[:, None] / s[None, :])
    fbar = np.asarray(
        jnp.sum(jnp.asarray(probs * s)[None, :] * F(jnp.minimum(x, 1e4)), axis=1)
    )
    weight = fbar * y ** ((density - 3.0) / 2.0) * y
    tail = weight[0] / slope
    cum = tail + np.concatenate(
        [[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(_LOG_Y))]
    )
    total = cum[-1]
    target_lo, target_hi = eps * total, (1 - eps) * total
    if target_lo < tail:
        y_lo = float(y[0] * (target_lo / tail) ** (1.0 / slope))
    else:
        y_lo = float(np.exp(np.interp(target_lo, cum, _LOG_Y)))
    return y_lo, float(np.exp(np.interp(target_hi, cum, _LOG_Y)))


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
