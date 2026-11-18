"""Tail criterion of PAPER:1059 as a window in y = nu / (a_1 B gamma^2) for the eta-averaged continuum kernel."""
import numpy as np
from scipy import integrate, special, optimize

def F(x):
    x = np.asarray(x, float)
    out = np.empty_like(x)
    for i, xv in np.ndenumerate(x):
        out[i] = xv * integrate.quad(lambda t: special.kv(5.0 / 3.0, t), xv, np.inf, epsabs=0, epsrel=1e-10)[0]
    return out

# tabulate F on a log grid
lx = np.linspace(np.log(1e-9), np.log(60.0), 1400)
Ft = F(np.exp(lx))
def Fi(x):
    return np.where(x > 60.0, 0.0, np.exp(np.interp(np.log(np.maximum(x, 1e-9)), lx, np.log(np.maximum(Ft, 1e-300)))))

def window(eta_nodes, eta_w, p=2.6, eps=1e-3):
    s = np.sqrt(1 - eta_nodes**2)
    ly = np.linspace(np.log(1e-8), np.log(200.0), 6000)
    y = np.exp(ly)
    Fbar = np.sum(eta_w[None, :] * s[None, :] * Fi(y[:, None] / s[None, :]), axis=1)
    integrand = Fbar * y ** ((p - 3) / 2) * y        # d ln y
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(ly))])
    tot = cum[-1]
    y_lo = np.exp(np.interp(eps * tot, cum, ly))
    y_hi = np.exp(np.interp((1 - eps) * tot, cum, ly))
    return y_lo, y_hi

t, w = np.polynomial.legendre.leggauss(48)
a1 = 4.1988e6  # Hz per gauss, a_B = a1 * B_perp
for label, lo, hi in [("eta isotropic", 0.0, 1.0), ("eta U[-0.5,0.5]", 0.0, 0.5), ("eta = 0 (B_perp = B)", None, None)]:
    if lo is None:
        en, ew = np.array([0.0]), np.array([1.0])
    else:
        en = 0.5 * (hi - lo) * t + 0.5 * (hi + lo); ew = 0.5 * w
    for eps in (1e-3, 5e-4):
        ylo, yhi = window(en, ew, eps=eps)
        print(f"{label:22s} eps={eps:g}: y_lo={ylo:.3e} y_hi={yhi:.3f}")

def rng(nu_min, nu_max, Bmin, Bmax, ylo, yhi):
    return np.sqrt(nu_min / (a1 * Bmax * yhi)), np.sqrt(nu_max / (a1 * Bmin * ylo))

en = 0.5 * t + 0.5; ew = 0.5 * w
ylo, yhi = window(en, ew, eps=1e-3)
g0, g1 = rng(0.408e9, 23e9, 5e-6, 5e-6, ylo, yhi)
h = 0.2420
n = int(np.ceil(np.log(g1 / g0) / (2 * h)))
print("first screen (0.408-23 GHz, 5 uG, isotropic):", g0, g1, "cells", n, "moments", 3 * n)
en = 0.25 * t + 0.25; ew = 0.5 * w
ylo, yhi = window(en, ew, eps=1e-3)
print("paper stats 0.1-10 GHz, U[4,6]:", rng(1e8, 1e10, 4e-6, 6e-6, ylo, yhi))
print("paper stats 0.1-300 GHz, U[4,6]:", rng(1e8, 3e11, 4e-6, 6e-6, ylo, yhi))
# paper statement: gamma <= 2.96e6 within 1e-3 at 300 GHz; with <B>=5: y at edge
print("y at gamma=2.96e6, 300 GHz, 5 uG:", 3e11 / (a1 * 5e-6 * 2.96e6**2), " y at gamma=300, 0.1 GHz, 5 uG:", 1e8 / (a1 * 5e-6 * 300.0**2))
print("ln(2.96e6/300)/38 =", np.log(2.9626e6 / 300) / 38)
