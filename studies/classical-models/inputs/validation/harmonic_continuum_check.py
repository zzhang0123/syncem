"""Pitch-averaged harmonic spectrum versus the continuum kernels of Appendix A.

Checks main.tex Eqs. (2.1)-(2.9) (harmonic amplitudes, source-time D^-3
normalisation, Stokes signs) against Eq. (A.5) (directional continuum kernels
K_I = A_theta F(x), K_Q = -A_theta G(x)) over a wide frequency range.

Electrons are isotropic in pitch angle, the observer sits at a fixed angle
theta to the field, and the field magnitude and gamma are fixed. Averaging the
harmonic line comb over mu = cos(alpha) is exact for this population:

    K_nu = sum_m (1/2) S_m(mu_m) * m nu_B / (nu^2 beta eta),
    mu_m = (1 - m nu_B / nu) / (beta eta),   |mu_m| <= 1,

because nu_m(mu) = m nu_B / (1 - beta mu eta) is monotonic in mu for eta > 0.
Units: e = m_e = c = B = 1, so omega_G = 1 and frequencies are in units of
nu_G = 1 / (2 pi). Ratios to the continuum kernel are unit free.

Run from the manuscript root:
  python -m validation.harmonic_continuum_check
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from scipy.integrate import quad
from scipy.special import jv, jvp, kv

OUT = Path(__file__).resolve().parent
THETA = np.pi / 3
ETA, SIN_T = np.cos(THETA), np.sin(THETA)
NU_G = 1 / (2 * np.pi)
A_THETA = np.sqrt(3) * SIN_T / (4 * np.pi)
A_B = 3 * SIN_T / (4 * np.pi)
GAMMAS = (3.0, 10.0, 30.0, 100.0)
X_GRID = np.geomspace(1e-3, 10.0, 49)


def harmonic_stokes(m, mu, gamma):
    """I_m, Q_m, V_m per unit source time and solid angle (natural basis)."""
    beta = np.sqrt(1 - 1 / gamma**2)
    b_par, b_perp = beta * mu, beta * np.sqrt(np.clip(1 - mu**2, 0, None))
    d = 1 - beta * mu * ETA
    x = m * b_perp * SIN_T / d
    c2 = m**2 / (gamma**2 * 2 * np.pi * d**3)
    j, jp = jv(m, x), jvp(m, x)
    a_par = (ETA - b_par) / SIN_T * j
    a_perp = b_perp * jp
    i = c2 * (a_par**2 + a_perp**2)
    q = c2 * (a_par**2 - a_perp**2)
    v = 2 * c2 * b_perp * (ETA - b_par) / SIN_T * j * jp
    return i, q, v


def pitch_averaged(nu, gamma, half_width):
    """Harmonic I, Q, V per unit frequency at fixed theta, isotropic pitch.

    Only harmonics whose root lies within half_width of the beaming cone
    mu = eta are summed; convergence in half_width is checked separately.
    """
    beta = np.sqrt(1 - 1 / gamma**2)
    nu_b = NU_G / gamma
    lo = max(-1.0, ETA - half_width)
    hi = min(1.0, ETA + half_width)
    m_lo = max(1, int(np.floor(nu * (1 - beta * ETA * hi) / nu_b)))
    m_hi = int(np.ceil(nu * (1 - beta * ETA * lo) / nu_b))
    out = np.zeros(3)
    for start in range(m_lo, m_hi + 1, 2_000_000):
        m = np.arange(start, min(start + 2_000_000, m_hi + 1), dtype=float)
        mu = (1 - m * nu_b / nu) / (beta * ETA)
        keep = (mu >= lo) & (mu <= hi)
        m, mu = m[keep], mu[keep]
        s = np.array(harmonic_stokes(m, mu, gamma))
        out += (s * (0.5 * m * nu_b / (nu**2 * beta * ETA))).sum(axis=1)
    return out


def synch_F(x):
    return x * quad(lambda t: kv(5 / 3, t), x, np.inf, epsabs=0, epsrel=1e-12, limit=200)[0]


def continuum(nu, gamma):
    x = nu / (A_B * gamma**2)
    return A_THETA * synch_F(x), -A_THETA * x * kv(2 / 3, x)


def cone_half_width(x, gamma, factor):
    """Beaming half-width in mu around eta, widened at low x."""
    return min(2.0, factor * max(1.0, x ** (-1 / 3)) / gamma)


def low_harmonic_detail():
    """Resolve the gamma=10 low-frequency structure using the full pitch domain."""
    rows = []
    gamma = 10.0
    for x in np.geomspace(1e-3, 1e-2, 1001):
        nu = x * A_B * gamma**2
        h = pitch_averaged(nu, gamma, 2.0)
        ci, cq = continuum(nu, gamma)
        rows.append(dict(gamma=gamma, x=float(x), nu_over_nuG=float(nu / NU_G),
                         harmonic_I=h[0], harmonic_Q=h[1], harmonic_V=h[2],
                         continuum_I=ci, continuum_Q=cq))
    return rows


def lienard_check(gamma):
    """Pitch- and direction-averaged harmonic power versus the Lienard power.

    With e=m=c=B=1, P_tot(alpha) = (2/3) gamma^2 beta_perp^2 (Eq. A.4 with
    omega_B = 1/gamma). Isotropic pitch gives <beta_perp^2> = 2 beta^2 / 3.
    The harmonic side sums line-integrated powers over m and integrates
    over mu and eta; it does not use the frequency root formula.
    """
    beta2 = 1 - 1 / gamma**2
    target = (2 / 3) * gamma**2 * (2 * beta2 / 3)
    nodes, weights = np.polynomial.legendre.leggauss(200)
    total = 0.0
    global ETA, SIN_T
    eta0, sin0 = ETA, SIN_T
    for eta, w_eta in zip(nodes, weights):
        ETA, SIN_T = eta, np.sqrt(1 - eta**2)
        mu = nodes[:, None]
        m = np.arange(1, int(80 * gamma**3) + 2, dtype=float)[None, :]
        i, _, _ = harmonic_stokes(m, mu, gamma)
        total += w_eta * 2 * np.pi * np.sum(weights[:, None] / 2 * i)
    ETA, SIN_T = eta0, sin0
    return total / target - 1


def main():
    t0 = time.time()
    rows = []
    for gamma in GAMMAS:
        for x in X_GRID:
            nu = x * A_B * gamma**2
            h = pitch_averaged(nu, gamma, cone_half_width(x, gamma, 25.0))
            h_wide = pitch_averaged(nu, gamma, cone_half_width(x, gamma, 50.0)) if gamma >= 30 else h
            ci, cq = continuum(nu, gamma)
            rows.append(dict(gamma=gamma, x=float(x), nu_over_nuG=float(nu / NU_G),
                             harmonic_I=h[0], harmonic_Q=h[1], harmonic_V=h[2],
                             continuum_I=ci, continuum_Q=cq,
                             cone_convergence=float(abs(h_wide[0] / h[0] - 1)) if h[0] else 0.0))
        print(f"gamma={gamma} done at {time.time() - t0:.0f}s", flush=True)
    lienard = {str(g): lienard_check(g) for g in (1.5, 2.0)}
    result = dict(theta=THETA, gammas=GAMMAS, x_grid=X_GRID.tolist(), rows=rows,
                  low_harmonic_detail=low_harmonic_detail(),
                  lienard_relative_error=lienard,
                  note="x = nu / (a_B gamma^2); continuum omits V; harmonic exact within the helical-orbit model")
    (OUT / "harmonic_continuum_results.json").write_text(json.dumps(result, indent=1))
    print(json.dumps(lienard), f"total {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
