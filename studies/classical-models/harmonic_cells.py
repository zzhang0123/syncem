"""Frequency-averaged harmonic emissivity of an isotropic-pitch population at one (gamma, eta, x).

Library for harmonic_correction.py.  Units and line formulae are those of kp_harmonic.py
(manuscript Eqs. `eq: harmonic frequencies`--`eq: Vn`; e = c = 1, nu_B = 1).  The exact
monochromatic emissivity is the sum over integer harmonics

    j_S(nu) = sum_m f_S(m),   f_S(m) = p(mu_m) S_m(mu_m) m / (nu^2 beta eta),
    mu_m = (1 - m/nu)/(beta eta).

A channel many line spacings wide sees its average over frequency.  By Poisson summation the
average over one line spacing keeps only the k = 0 term, int dm f_S(m); the k != 0 terms are
the line ripple, which averages out.  With m = nu D(mu), D = 1 - beta mu eta, dm = nu beta eta dmu:

    jbar_S(nu) = int dmu p(mu) S_{m(mu)}(mu) m(mu)/nu,

with J_m(m z) at real order m.  This form has no 1/eta and holds at eta = 0.  It is checked
against the exact integer sum averaged over line phases (kp_harmonic.harmonic_j) in
harmonic_correction.py.  The integral runs over the pitch angle alpha = theta + u/gamma,
|u| <= W, on Gauss-Legendre panels refined towards u = 0.

Bessel routes (kp_harmonic.BESSEL): "scipy" (jv, jvp at real order) and "olver" (DLMF
10.20.4/10.20.7 with B0, C0; relative error O(m^-2)).  "auto" uses scipy up to order 1e8 and
Olver above it.
"""
from __future__ import annotations

import math

import numpy as np
from numpy.polynomial.legendre import leggauss

import kernels as K
import kp_harmonic as KH

SCIPY_MAX_ORDER = 1e8
PANEL_EDGES = (0.0, 0.025, 0.05, 0.1, 0.2, 0.4, 0.8, 1.6, 3.2, 6.4, 12.8, 25.6)


def u_nodes(W: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric GL panels in u on [-W, W], refined towards 0."""
    edges = np.array([e for e in PANEL_EDGES if e < W] + [W])
    t, w = leggauss(n)
    us, ws = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        us.append(0.5 * (b - a) * t + 0.5 * (a + b))
        ws.append(0.5 * (b - a) * w)
    u, wu = np.concatenate(us), np.concatenate(ws)
    return np.concatenate([-u[::-1], u]), np.concatenate([wu[::-1], wu])


def averaged_j(gamma: float, eta: float, nu: float, route: str = "auto", W: float = 40.0, n: int = 32):
    """(jbar_I, jbar_Q, max order) for isotropic pitch, p(mu) = 1/2."""
    beta = math.sqrt(1.0 - 1.0 / gamma**2)
    omb = 1.0 / (gamma**2 * (1.0 + beta))  # 1 - beta
    st = math.sqrt((1.0 - eta) * (1.0 + eta))
    theta = math.acos(eta)
    u, wu = u_nodes(W, n)
    d = u / gamma
    alpha = theta + d
    ok = (alpha > 0.0) & (alpha < math.pi)
    u, wu, d, alpha = u[ok], wu[ok], d[ok], alpha[ok]
    mu, sa = np.cos(alpha), np.sin(alpha)
    mu_m_eta = -2.0 * np.sin(theta + 0.5 * d) * np.sin(0.5 * d)  # cos(theta + d) - cos(theta)
    bme = beta * mu_m_eta - omb * eta  # beta mu - eta
    D = 1.0 - beta * mu * eta
    m = nu * D
    bperp = beta * sa
    s = ((1.0 - eta * eta) / gamma**2 + bme * bme) / (D * D)  # 1 - z^2 as a sum of squares
    z = bperp * st / D
    use = route if route != "auto" else ("scipy" if m.max() <= SCIPY_MAX_ORDER else "olver")
    J, Jp = KH.BESSEL[use](m, s, z)
    C2 = 2.0 * np.pi * m * m / D**3
    A2, P2 = C2 * (bme / st * J) ** 2, C2 * (bperp * Jp) ** 2
    wgt = wu * 0.5 * (m / nu) * sa / gamma  # p(mu) m/nu |dmu/du|
    return float(np.sum(wgt * (A2 + P2))), float(np.sum(wgt * (A2 - P2))), float(m.max()), use


def continuum(gamma: float, eta: float, x: float) -> tuple[float, float]:
    """K^iso_I = A F(x), K^iso_Q = -A G(x), A = sqrt(3) gamma sin(theta)/2 (module units)."""
    st = math.sqrt((1.0 - eta) * (1.0 + eta))
    A = math.sqrt(3.0) * gamma * st / 2.0
    return A * float(K.F(np.array([x]))[0]), -A * float(K.G(np.array([x]))[0])


def nu_of(gamma: float, eta: float, x: float) -> float:
    return x * 1.5 * gamma**3 * math.sqrt((1.0 - eta) * (1.0 + eta))


def cell(args) -> dict:
    """gamma^2 (jbar/K - 1) for I and Q at one (gamma, eta, x); optional refinement checks."""
    gamma, eta, x, route, refine = args
    nu = nu_of(gamma, eta, x)
    jI, jQ, mmax, used = averaged_j(gamma, eta, nu, route)
    KI, KQ = continuum(gamma, eta, x)
    g2 = gamma**2
    out = {"gamma": gamma, "eta": eta, "x": x, "nu": nu, "max_order": mmax, "route": used,
           "c_I": g2 * (jI / KI - 1.0), "c_Q": g2 * (jQ / KQ - 1.0)}
    if refine:
        jI2, jQ2, _, _ = averaged_j(gamma, eta, nu, route, W=60.0, n=64)
        out["refined_minus_adopted_c_I"] = g2 * (jI2 / KI - 1.0) - out["c_I"]
        out["refined_minus_adopted_c_Q"] = g2 * (jQ2 / KQ - 1.0) - out["c_Q"]
    return out


def integer_sum_one_phase(args) -> dict:
    """Exact integer-harmonic sum (kp_harmonic.harmonic_j, scipy jv) at one frequency shift
    nu0 + spacing * k / n_phase, spacing = 1/(1 - beta eta^2) (line spacing at the cone);
    returned as j/K - 1 for I and Q."""
    gamma, eta, x, k, n_phase = args
    beta = math.sqrt(1.0 - 1.0 / gamma**2)
    spacing = 1.0 / (1.0 - beta * eta * eta)
    nu = nu_of(gamma, eta, x) + spacing * k / n_phase
    xk = nu / (1.5 * gamma**3 * math.sqrt(1.0 - eta * eta))
    jI, jQ, _, n_lines = KH.harmonic_j(gamma, eta, nu, ("iso",), bessel="scipy", W=40.0 if x < 0.05 else 25.0)
    KI, KQ = continuum(gamma, eta, xk)
    return {"gamma": gamma, "eta": eta, "x": x, "k": k, "dI": jI / KI - 1.0, "dQ": jQ / KQ - 1.0,
            "n_lines": n_lines}
