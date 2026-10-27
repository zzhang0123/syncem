"""Continuum (beaming-limit) synchrotron kernels and constants, Gaussian cgs.

Radiation model (manuscript Eqs. `eq: beaming limit`, `eq: directional continuum`):
    K^iso_I = A_theta F(x),  K^iso_Q = -A_theta G(x)   (natural basis, no V)
    A_theta = sqrt(3) e^3 B_perp / (4 pi m_e c^2),  x = nu / (a_B gamma^2),
    a_B = 3 e B_perp / (4 pi m_e c),  B_perp = B sqrt(1 - eta^2),
    F(x) = x int_x^inf K_{5/3},  G(x) = x K_{2/3}(x).
These are NOT the harmonic kernels; results built on them are continuum results.

Log-derivatives.  For W(x) = x int_0^inf exp(-x cosh u) w(u) du with
w_F(u) = cosh(5u/3)/cosh(u) (gives F) and w_G(u) = cosh(2u/3) (gives G),
    (-x d/dx)^r W(x) = x int_0^inf exp(-z) P_r(z) w(u) du,  z = x cosh u,
with P_0 = 1, P_{r+1} = (z-1) P_r - z P_r'  (the recursion of
studies/galprop-benchmark/moment_kernel.py, extended to G).  Since
x = nu/(a_B gamma^2), d/d(ln gamma) = 2 (-x d/dx).

All evaluations are returned in log-safe form: W^{(r)}(x) = x e^{-x} E_0(x) f_r(x)
with E_0 = int exp(-x(cosh u - 1)) w(u) du > 0 and ratios f_r = E_r/E_0.
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
from numpy.polynomial import polynomial as npoly
from numpy.polynomial.legendre import leggauss
from scipy import special

# --- constants (cgs).  E_ESU, M_E, C_CGS, C_SI are identical to syncem.constants
# (unchanged since SyncMoments d51f9be; the harmonic pipeline's values).
E_ESU = 4.803204712570263e-10
M_E = 9.1093837139e-28
C_CGS = 2.99792458e10
C_SI = 2.99792458e8
SIGMA_T = 6.6524587321e-25  # Thomson cross-section, CODATA 2018 [cm^2]
EV_ERG = 1.602176634e-12
MUG = 1e-6  # 1 microgauss in gauss
SEC_PER_MYR = 3.15576e13  # Julian Myr
A1 = np.sqrt(3.0) * E_ESU**3 * MUG / (4.0 * np.pi * M_E * C_CGS**2)  # A_theta per muG of B_perp
a1 = 3.0 * E_ESU * MUG / (4.0 * np.pi * M_E * C_CGS)  # a_B per muG of B_perp [Hz]
LOSS_COEFF = 4.0 * SIGMA_T / (3.0 * M_E * C_CGS)  # b = LOSS_COEFF * U [s^-1 per erg cm^-3]

MAX_ORDER = 8
DEFAULT_NODES = 200
DEFAULT_CUTOFF = 70.0


@lru_cache(maxsize=4)
def derivative_polynomials(order: int) -> tuple:
    """Coefficient arrays (ascending powers of z) of P_0..P_order."""
    polys = [np.array([1.0])]
    for _ in range(order):
        p = polys[-1]
        polys.append(npoly.polysub(npoly.polymul(p, [-1.0, 1.0]),
                                   npoly.polymul(npoly.polyder(p), [0.0, 1.0])))
    return tuple(polys)


@lru_cache(maxsize=4)
def _gauss(nodes: int):
    return leggauss(nodes)


def _weight(u, which):
    if which == "F":
        return np.cosh(5.0 * u / 3.0) / np.cosh(u)
    if which == "G":
        return np.cosh(2.0 * u / 3.0)
    raise ValueError("which must be 'F' or 'G'")


def scaled_log_derivatives(x, order=0, which="F", nodes=DEFAULT_NODES, cutoff=DEFAULT_CUTOFF):
    """Return (ln E_0(x), f(x)) with W^{(r)} = x e^{-x} E_0 f_r, f shape x.shape+(order+1,).

    W = F or G, W^{(r)} = (-x d/dx)^r W.  Gauss-Legendre on u in [0, u_max] with
    x (cosh u_max - 1) = cutoff; truncated tail < e^{-cutoff} (times a polynomial).
    """
    xx = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(xx)) or np.any(xx <= 0):
        raise ValueError("kernel arguments must be finite and positive")
    if not 0 <= order <= MAX_ORDER:
        raise ValueError(f"order must be in 0..{MAX_ORDER}")
    flat = xx.ravel()
    q, wq = _gauss(nodes)
    polys = derivative_polynomials(order)
    ln_e0 = np.empty(flat.size)
    ratios = np.empty((flat.size, order + 1))
    for s in range(0, flat.size, 2048):
        xv = flat[s:s + 2048, None]
        umax = np.arccosh(1.0 + cutoff / xv)
        u = 0.5 * (q + 1.0) * umax
        z = xv * np.cosh(u)
        base = np.exp(-2.0 * xv * np.sinh(0.5 * u) ** 2) * _weight(u, which) * wq * (0.5 * umax)
        e0 = base.sum(axis=1)
        ln_e0[s:s + 2048] = np.log(e0)
        ratios[s:s + 2048, 0] = 1.0
        for r in range(1, order + 1):
            ratios[s:s + 2048, r] = (base * npoly.polyval(z, polys[r])).sum(axis=1) / e0
    return ln_e0.reshape(xx.shape), ratios.reshape(xx.shape + (order + 1,))


def log_derivatives(x, order=0, which="F", **kw):
    """(-x d/dx)^r W(x), r = 0..order, shape x.shape + (order+1,).  Underflows to 0."""
    xx = np.asarray(x, dtype=float)
    ln_e0, f = scaled_log_derivatives(xx, order, which, **kw)
    return (xx * np.exp(ln_e0 - xx))[..., None] * f


def F(x, **kw):
    """F(x) = x int_x^inf K_{5/3}(t) dt."""
    return log_derivatives(x, 0, "F", **kw)[..., 0]


def G(x, **kw):
    """G(x) = x K_{2/3}(x)."""
    return log_derivatives(x, 0, "G", **kw)[..., 0]


def G_bessel(x):
    """Independent route for G via scipy kve."""
    x = np.asarray(x, dtype=float)
    return x * special.kve(2.0 / 3.0, x) * np.exp(-x)


def random_field_kernel(x):
    """Crusius & Schlickeiser (1986) / Ghisellini et al. (1988) kernel R(x).

    R(x) = (x^2/2) K_{4/3}(x/2) K_{1/3}(x/2) - (3x^3/20)[K_{4/3}^2(x/2) - K_{1/3}^2(x/2)],
    equal to the isotropic field-orientation average <sin(theta) F(x/sin(theta))>
    with x defined from the total field (checked in core_checks.py).
    """
    x = np.asarray(x, dtype=float)
    h = 0.5 * x
    k43, k13 = special.kve(4.0 / 3.0, h), special.kve(1.0 / 3.0, h)
    return np.exp(-x) * (0.5 * x**2 * k43 * k13 - 0.15 * x**3 * (k43 - k13) * (k43 + k13))


def mellin_F(m):
    """int_0^inf x^m F(x) dx, m > -4/3 (R&L)."""
    g = special.gamma
    return 2.0 ** (m + 1) / (m + 2) * g(m / 2 + 7.0 / 3) * g(m / 2 + 2.0 / 3)


def mellin_G(m):
    """int_0^inf x^m G(x) dx, m > -4/3."""
    g = special.gamma
    return 2.0**m * g(m / 2 + 4.0 / 3) * g(m / 2 + 2.0 / 3)


def powerlaw_pol_fraction(p):
    """Linear-polarisation fraction of an unbounded power law: (p+1)/(p+7/3)."""
    return (p + 1.0) / (p + 7.0 / 3.0)


F_SMALL_X_COEFF = 4.0 * np.pi / (np.sqrt(3.0) * special.gamma(1.0 / 3.0)) * 2.0 ** (-1.0 / 3.0)
"""F(x) ~ F_SMALL_X_COEFF x^{1/3} for x -> 0 (= 2.1495...)."""
