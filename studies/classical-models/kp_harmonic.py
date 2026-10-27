"""Exact harmonic sums for an anisotropic pitch density at one (gamma, eta, nu).

Library for kp_bridge.py.  Units (as in the beaming-limit survey of inputs/beaming-survey): e = c = 1 and the
relativistic gyrofrequency nu_B = e B / (2 pi gamma m c) = 1.  Manuscript Eqs.
`eq: harmonic frequencies`--`eq: Vn` give, for a pitch density p(mu) normalised on [-1, 1]
and eta = cos(theta) > 0,

    j_S(nu) = sum_m p(mu_m) S_m(mu_m) m / (nu^2 beta eta),   mu_m = (1 - m/nu)/(beta eta),
    S_m = C2 (a_par^2 J^2 +- b_perp^2 J'^2),  V_m = 2 C2 b_perp a_par J J',
    C2 = 2 pi m^2 / D^3,  D = m/nu,  a_par = (eta - beta mu)/sin(theta),
    J = J_m(m z),  z = b_perp sin(theta)/D.

This is a sum over integer harmonics at one monochromatic frequency; it is exact for the
helical-orbit model (no continuum approximation).  The continuum (beaming-limit) reference
is 2 p(eta) K^iso_S with K^iso_I = A F(x), K^iso_Q = -A G(x), A = sqrt(3) gamma sin(theta)/2,
x = nu/(1.5 gamma^3 sin(theta)) (manuscript `eq: beaming limit`, `eq: directional continuum`).

Bessel functions: two independent routes.
  * "scipy": scipy.special.jv / jvp at integer order (AMOS; refuses orders > ~1.07e9).
  * "olver": DLMF 10.20.4 and 10.20.7 (uniform Airy expansions) with the k = 0 terms
    A_0 = D_0 = 1 and B_0, C_0 (DLMF 10.20.10-11).  The neglected terms are O(m^-2)
    relative (measured against mpmath in kp_bridge.py checks).  1 - z^2 is formed as a sum
    of squares, (1 - eta^2)/gamma^2 + (beta mu - eta)^2 over D^2, so the turning-point
    coordinate keeps full precision at gamma ~ 1e3 (m ~ 1e9).
"""
from __future__ import annotations

import math
from functools import lru_cache

import numpy as np
from numpy.polynomial import chebyshev as cheb
from scipy import special

CHUNK = 500_000
S_SPLIT = 0.1  # below: Chebyshev fit of B0, C0 in s = 1 - z^2; above: closed form
AIRY_ZERO = 200.0  # Ai(200) ~ e^-1886: lines beyond this Airy argument are exactly 0 in float64


# ---------------------------------------------------------------- Olver coefficients
def _b0c0_closed(s, zeta):
    """DLMF 10.20.10-11 closed forms (float64; used for s >= S_SPLIT)."""
    r = np.sqrt(s)
    b0 = -5.0 / (48.0 * zeta**2) + zeta**-0.5 * (5.0 / (24.0 * r**3) - 1.0 / (8.0 * r))
    c0 = 7.0 / (48.0 * zeta) + zeta**0.5 * (-7.0 / (24.0 * r**3) + 3.0 / (8.0 * r))
    return b0, c0


def _zeta_from_w(w):
    """zeta with (2/3) zeta^{3/2} = atanh(w) - w, w = sqrt(1 - z^2) in [0, 1)."""
    w = np.asarray(w, dtype=float)
    w2 = w * w
    series = np.zeros_like(w)
    for k in range(16, 0, -1):  # sum_{k>=1} w^{2k-2}/(2k+1), Horner in w^2
        series = series * w2 + 1.0 / (2 * k + 1)
    t_small = w2 * w * series
    with np.errstate(divide="ignore", invalid="ignore"):
        t_big = np.arctanh(np.minimum(w, 1.0 - 1e-16)) - w
    t = np.where(w < 0.2, t_small, t_big)
    return (1.5 * t) ** (2.0 / 3.0)


@lru_cache(maxsize=1)
def _b0c0_chebyshev(deg: int = 24):
    """Chebyshev fits of B0(s), C0(s) on [0, S_SPLIT], from 60-digit mpmath values."""
    import mpmath as mp

    mp.mp.dps = 60
    nodes = 0.5 * S_SPLIT * (1.0 + np.cos(np.pi * (np.arange(2 * deg) + 0.5) / (2 * deg)))
    b, c = [], []
    for s in nodes:
        s_ = mp.mpf(s)
        w = mp.sqrt(s_)
        zeta = (mp.mpf(3) / 2 * (mp.atanh(w) - w)) ** (mp.mpf(2) / 3)
        b.append(float(-5 / (48 * zeta**2) + zeta ** mp.mpf(-0.5)
                       * (5 / (24 * w**3) - 1 / (8 * w))))
        c.append(float(7 / (48 * zeta) + zeta ** mp.mpf(0.5)
                       * (-7 / (24 * w**3) + 3 / (8 * w))))
    t = 2.0 * nodes / S_SPLIT - 1.0
    return cheb.chebfit(t, b, deg), cheb.chebfit(t, c, deg)


def b0c0(s, zeta):
    s = np.asarray(s, dtype=float)
    cb, cc = _b0c0_chebyshev()
    t = 2.0 * np.minimum(s, S_SPLIT) / S_SPLIT - 1.0
    b_fit, c_fit = cheb.chebval(t, cb), cheb.chebval(t, cc)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        b_cl, c_cl = _b0c0_closed(np.maximum(s, S_SPLIT), np.maximum(zeta, 1e-300))
    small = s < S_SPLIT
    return np.where(small, b_fit, b_cl), np.where(small, c_fit, c_cl)


def bessel_olver(m, s, z):
    """J_m(m z), J'_m(m z) from the uniform expansion; s = 1 - z^2 supplied accurately."""
    m = np.asarray(m, dtype=float)
    w = np.sqrt(s)
    zeta = _zeta_from_w(w)
    m13 = np.cbrt(m)
    a = m13 * m13 * zeta
    far = a > AIRY_ZERO
    ai, aip, _, _ = special.airy(np.where(far, 0.0, a))
    ai, aip = np.where(far, 0.0, ai), np.where(far, 0.0, aip)
    b0, c0 = b0c0(s, zeta)
    phi = (4.0 * zeta / s) ** 0.25
    J = phi * (ai / m13 + aip * b0 / (m * m13 * m13))
    with np.errstate(divide="ignore", invalid="ignore"):
        Jp = -(2.0 / z) / phi * (ai * c0 / (m * m13) + aip / (m13 * m13))
    return J, np.where(far, 0.0, Jp)


def bessel_scipy(m, s, z):
    x = m * z
    return special.jv(m, x), special.jvp(m, x)


def bessel_cubic_airy(m, s, z):
    """The manuscript pipeline's large-order branch (radio-fit/stable_lines.py):
    scale = (2/(m z))^{1/3}, arg = m (1-z) scale, J = scale Ai(arg), J' = -scale^2 Ai'(arg)."""
    one_minus_z = s / (1.0 + z)
    scale = np.cbrt(2.0 / (m * z))
    arg = m * one_minus_z * scale
    ai, aip, _, _ = special.airy(np.minimum(arg, AIRY_ZERO))
    ai, aip = np.where(arg > AIRY_ZERO, 0.0, ai), np.where(arg > AIRY_ZERO, 0.0, aip)
    return scale * ai, -scale**2 * aip


BESSEL = {"scipy": bessel_scipy, "olver": bessel_olver, "cubic_airy": bessel_cubic_airy}


# ---------------------------------------------------------------- pitch densities
def pdf_value(pdf, mu):
    """Unnormalised pitch densities.  pdf = ("iso",), ("kp_edge", mu_c, a), ("kp_smooth", k2, a).

    kp_edge is the exact Kardashev-Pacholczyk conditional density at fixed gamma above
    gamma_c(mu=0): 1 - b(mu) gamma t = 1.5 b_s gamma t (mu^2 - mu_c^2), so
    p(mu | gamma) is proportional to (mu^2 - mu_c^2)_+^a with a = p_inj - 2.
    kp_smooth is the same density below gamma_c(0): (mu^2 + kappa^2)^a.
    """
    mu = np.asarray(mu, dtype=float)
    if pdf[0] == "iso":
        return np.full_like(mu, 0.5)
    if pdf[0] == "kp_edge":
        mu_c, a = pdf[1], pdf[2]
        am = np.abs(mu)
        d = (am - mu_c) * (am + mu_c)
        return np.where(d > 0, np.abs(d) ** a, 0.0)
    if pdf[0] == "kp_smooth":
        return (mu * mu + pdf[1]) ** pdf[2]
    if pdf[0] == "P2":  # (1/2)[1 + P_2(mu)]
        return 0.5 * (1.0 + 0.5 * (3.0 * mu * mu - 1.0))
    if pdf[0] == "exp":  # e^mu / (2 sinh 1)
        return np.exp(mu) / (2.0 * np.sinh(1.0))
    if pdf[0] == "sin4":  # (15/16)(1 - mu^2)^2
        return 15.0 / 16.0 * (1.0 - mu * mu) ** 2
    raise ValueError(pdf)


def pdf_norm(pdf):
    """Z = int_{-1}^{1} p dmu (for the isotropic-equivalent error measure)."""
    from scipy import integrate

    if pdf[0] == "iso":
        return 1.0
    if pdf[0] == "kp_edge":
        mu_c, a = pdf[1], pdf[2]
        val, _ = integrate.quad(lambda m: (m + mu_c) ** a, mu_c, 1.0, weight="alg",
                                wvar=(a, 0.0), epsabs=0, epsrel=1e-13, limit=200)
        return 2.0 * val
    val, _ = integrate.quad(lambda m: pdf_value(pdf, m), -1, 1, epsabs=0, epsrel=1e-13)
    return val


# ---------------------------------------------------------------- harmonic sum
def _line_terms(m, gamma, eta, nu, pdf, bessel):
    beta = math.sqrt(1.0 - 1.0 / gamma**2)
    st = math.sqrt((1.0 - eta) * (1.0 + eta))
    D = m / nu
    bmu_minus_eta = (nu * (1.0 - eta * eta) - m) / (nu * eta)  # beta mu_m - eta, exact form
    mu = (bmu_minus_eta + eta) / beta
    ok = np.abs(mu) < 1.0
    m, D, mu, bme = m[ok], D[ok], mu[ok], bmu_minus_eta[ok]
    bperp = beta * np.sqrt((1.0 - mu) * (1.0 + mu))
    s = ((1.0 - eta * eta) / gamma**2 + bme * bme) / (D * D)  # 1 - z^2, sum of squares
    z = bperp * st / D
    J, Jp = BESSEL[bessel](m, s, z)
    C2 = 2.0 * np.pi * m * m / D**3
    apar = -bme / st
    A2, P2 = C2 * (apar * J) ** 2, C2 * (bperp * Jp) ** 2
    w = pdf_value(pdf, mu) * m / (nu * nu * beta * eta)
    return np.sum(w * (A2 + P2)), np.sum(w * (A2 - P2)), np.sum(w * 2 * C2 * bperp * apar * J * Jp)


def line_powers(m, gamma, mu, eta, bessel="olver"):
    """Per-line (I_m, Q_m, V_m) in the module units at fixed (mu, eta) and orders m."""
    m = np.asarray(m, dtype=float)
    beta = math.sqrt(1.0 - 1.0 / gamma**2)
    st = math.sqrt((1.0 - eta) * (1.0 + eta))
    D = (1.0 / gamma**2) / (1.0 + beta) + beta * (1.0 - mu * eta)
    bme = beta * mu - eta
    bperp = beta * math.sqrt((1.0 - mu) * (1.0 + mu))
    s = ((1.0 - eta * eta) / gamma**2 + bme * bme) / (D * D)
    z = bperp * st / D
    J, Jp = BESSEL[bessel](m, np.full_like(m, s), np.full_like(m, z))
    C2 = 2.0 * np.pi * m * m / D**3
    apar = -bme / st
    A2, P2 = C2 * (apar * J) ** 2, C2 * (bperp * Jp) ** 2
    return A2 + P2, A2 - P2, 2.0 * C2 * bperp * apar * J * Jp


def harmonic_j(gamma, eta, nu, pdf, bessel="scipy", W=25.0):
    """(j_I, j_Q, j_V, n_lines) at monochromatic nu; window |alpha - theta| <= W/gamma."""
    if not 0.0 < eta < 1.0:
        raise ValueError("needs 0 < eta < 1")
    beta = math.sqrt(1.0 - 1.0 / gamma**2)
    theta = math.acos(eta)
    mu_lo = math.cos(min(math.pi, theta + W / gamma))
    mu_hi = math.cos(max(0.0, theta - W / gamma))
    m_lo = max(1, math.floor(nu * (1.0 - beta * eta * mu_hi)))
    m_hi = math.ceil(nu * (1.0 - beta * eta * mu_lo))
    acc = [[], [], []]
    for start in range(m_lo, m_hi + 1, CHUNK):
        m = np.arange(start, min(start + CHUNK, m_hi + 1), dtype=float)
        for lst, v in zip(acc, _line_terms(m, gamma, eta, nu, pdf, bessel)):
            lst.append(float(v))
    return math.fsum(acc[0]), math.fsum(acc[1]), math.fsum(acc[2]), m_hi - m_lo + 1


def continuum(gamma, eta, nu):
    """K^iso_I, K^iso_Q in the same units (no pitch factor)."""
    import kernels as K  # studies/classical-models/lib (F, G tested to 2.4e-14 vs mpmath)

    st = math.sqrt((1.0 - eta) * (1.0 + eta))
    A = math.sqrt(3.0) * gamma * st / 2.0
    x = nu / (1.5 * gamma**3 * st)
    return A * float(K.F(np.array([x]))[0]), -A * float(K.G(np.array([x]))[0]), x


def nu_for_x(gamma, eta, x):
    return x * 1.5 * gamma**3 * math.sqrt((1.0 - eta) * (1.0 + eta))


# ---------------------------------------------------------------- leading-order profile
def beaming_profile(u, x, stokes="I"):
    """Classical angular profile of single-electron emission at offset u = gamma psi from
    the cone (Jackson 14.79 / Westfold): (1+u^2)^2 [K_{2/3}^2(xi) +- u^2/(1+u^2) K_{1/3}^2(xi)],
    xi = (x/2)(1+u^2)^{3/2}; sign + for I, - for |Q|.  Unnormalised."""
    u = np.asarray(u, dtype=float)
    q = 1.0 + u * u
    xi = 0.5 * x * q**1.5
    e = np.exp(-2.0 * xi)
    k23, k13 = special.kve(2.0 / 3.0, xi), special.kve(1.0 / 3.0, xi)
    sgn = 1.0 if stokes == "I" else -1.0
    return q * q * e * (k23**2 + sgn * (u * u / q) * k13**2)
