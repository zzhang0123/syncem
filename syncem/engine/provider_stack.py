"""The stack provider: analytic energy columns of the continuum kernel from S1 tables (SPEC Sec. 8.1, 8.3).

LABEL ``eq: illustrated derivative spectra``. For the continuum kernel ``K = A_1 s B W(nu / (a_1 s B gamma^2))``
with ``s = sqrt(1 - eta^2)``, ``d / d ln gamma = 2 (-x d/dx)``, so the ``r``-th energy derivative of the field
average is ``2^r A_1 s T_W^(r)(y)`` with ``T`` from S1 (``field_average.FieldAverage``). ``channel_kernels``
integrates it over the channel nodes, with the phase law inside the channel integral (``P`` has no azimuth
factor). ``point_series`` turns it into the Taylor coefficients in the lattice chart: ``(h^r / r!) d^r K /
d(ln gamma)^r`` for a log chart, and the composition with ``ln(1 + eps u)`` for an affine chart. These are the
columns of a product cell whose only celled variable is ``gamma`` and whose field law is averaged by one S1 table
(the cached-S1 mode). Other cases go through ``_stack_series`` (per-node mode, field cells by the scaling
relation).

Units: those of the kernel, erg s^-1 sr^-1 Hz^-1 per electron for unit-area channels. Shapes: ``channel_kernels``
returns the shape ``(order + 1, n_g, n_e, J)``. ``point_series`` returns ``(order + 1, J)``. Not certified: the S1 spline
and lookup errors and the channel quadrature, validated against the study stack and the jet.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from ..constants import C_CGS, E_ESU, M_E
from .field_average import FieldAverage

__all__ = ["A1_UG", "A_1_UG", "channel_kernels", "log_to_affine", "point_series"]

_UG = 1e-6
A1_UG = (
    math.sqrt(3.0) * E_ESU**3 * _UG / (4.0 * math.pi * M_E * C_CGS**2)
)  # A_theta per uG of B_perp
A_1_UG = 3.0 * E_ESU * _UG / (4.0 * math.pi * M_E * C_CGS)  # a_B per uG of B_perp, Hz


def _channel_arrays(
    channels: Any, phase: Any, depth: float = 0.0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.asarray(channels.nodes, float)
    weights = np.asarray(
        (
            channels.response_weights()
            if hasattr(channels, "response_weights")
            else channels.weights * channels.node_values()
        ),
        float,
    )
    screen = (
        np.ones_like(nodes, dtype=complex)
        if phase in (None, False)
        else np.asarray(phase.weights(nodes, depth))
    )
    return nodes, weights, screen


def channel_kernels(
    table: FieldAverage,
    channels: Any,
    phase: Any,
    gammas: Any,
    eta_s: Any,
    order: int,
    chunk_points: int = 3_000_000,
    depth: float = 0.0,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``KI`` real and ``KQ`` complex, ``(order + 1, n_g, n_e, J)``: ``d^r / d(ln gamma)^r`` of the channel kernels.

    ``arrays`` holds ``_channel_arrays(channels, phase, depth)`` computed by the caller, which then skips that
    computation (decision 145). The caller keeps it consistent with ``channels``, ``phase`` and ``depth``.
    """
    nodes, weights, screen = (
        _channel_arrays(channels, phase, depth) if arrays is None else arrays
    )
    gammas = np.atleast_1d(np.asarray(gammas, float))
    eta_s = np.atleast_1d(np.asarray(eta_s, float))
    if np.any(eta_s <= 0) or np.any(eta_s > 1):
        raise ValueError(
            "sqrt(1 - eta^2) must lie in (0, 1]: an inclination node at |eta| = 1 gives no emission"
        )
    lnnu = np.log(nodes / A_1_UG)
    wq = weights * screen
    n_g, n_e, n_j = gammas.size, eta_s.size, nodes.shape[0]
    ki = np.zeros((order + 1, n_g, n_e, n_j))
    kq = np.zeros((order + 1, n_g, n_e, n_j), complex)
    per = max(1, int(chunk_points // (n_e * lnnu.size)))
    pref = A1_UG * eta_s
    for s in range(0, n_g, per):
        g = gammas[s : s + per]
        lny = (
            lnnu[None, None]
            - 2.0 * np.log(g)[:, None, None, None]
            - np.log(eta_s)[None, :, None, None]
        )
        tf = table.evaluate(lny, "F", order)
        tg = table.evaluate(lny, "G", order)
        for r in range(order + 1):
            fac = 2.0**r * pref[None, :, None]
            ki[r, s : s + per] = fac * np.einsum("gejt,jt->gej", tf[r], weights)
            kq[r, s : s + per] = -fac * (
                np.einsum("gejt,jt->gej", tg[r], wq.real)
                + 1j * np.einsum("gejt,jt->gej", tg[r], wq.imag)
            )
    return ki, kq


def log_to_affine(series: np.ndarray, eps: float) -> np.ndarray:
    """Coefficients in ``u`` of ``f(ln gamma_b + ln(1 + eps u))`` from ``a_n = (1/n!) d^n f / d(ln gamma)^n``."""
    order = series.shape[0] - 1
    t = np.array([0.0] + [(-1) ** (k + 1) * eps**k / k for k in range(1, order + 1)])
    power = np.zeros(order + 1)
    power[0] = 1.0
    out = np.zeros_like(series)
    for n in range(order + 1):
        out = (
            out + np.multiply.outer(power, series[n])
            if series.ndim > 1
            else out + power * series[n]
        )
        power = np.convolve(power, t)[: order + 1]
    return out


def point_series(
    table: FieldAverage,
    channels: Any,
    phase: Any,
    gamma: float,
    chart: str,
    half_width: float,
    eta_s: Any,
    eta_probs: Any,
    order: int,
    depth: float = 0.0,
    arrays: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``(I, P)`` Taylor coefficients ``(order + 1, J)`` in the cell's energy chart, ``eta`` averaged by ``eta_probs``.

    ``eta_probs`` has shape ``(n_e,)`` or ``(n_e, *L)`` (signed Legendre weights), giving ``(order + 1, J, *L)``.
    ``arrays`` passes precomputed channel arrays to ``channel_kernels``.
    """
    ki, kq = channel_kernels(
        table, channels, phase, [gamma], eta_s, order, depth=depth, arrays=arrays
    )
    probs = np.asarray(eta_probs, float)
    i = np.einsum("rej,e...->rj...", ki[:, 0], probs)
    p = np.einsum("rej,e...->rj...", kq[:, 0], probs)
    inv = np.array([1.0 / math.factorial(r) for r in range(order + 1)])
    shape = (order + 1,) + (1,) * (i.ndim - 1)
    a_i, a_p = i * inv.reshape(shape), p * inv.reshape(shape)
    if chart == "log":
        h = np.array([half_width**r for r in range(order + 1)]).reshape(shape)
        return a_i * h, a_p * h
    if chart == "affine":
        eps = half_width / gamma
        return log_to_affine(a_i, eps), log_to_affine(a_p, eps)
    raise ValueError(f"chart must be 'log' or 'affine', not {chart!r}")
