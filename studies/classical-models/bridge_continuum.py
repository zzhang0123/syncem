"""Rybicki-Lightman continuum channel kernel for the Sec. 4.3 radio-fit configuration.

Every ensemble, channel, screen and basis setting copies radio_numerics.py of the
harmonic pipeline (inputs/harmonic-radio-fit/radio_numerics.py; the pipeline's file has sha
bd477eca..., and the copy differs from it only in its PACKAGE path). Only the radiation model differs: isotropic-pitch beaming limit
K_I = A_theta F(x), K_Q = -A_theta G(x), x = nu/(a_B gamma^2), no V.

Field-strength and field-angle nodes can be chosen equal to the harmonic ones
("matched"), so that harmonic minus continuum isolates the radiation model from
the B and eta quadrature, which both pipelines then share.
"""

from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.interpolate import CubicSpline
from scipy.special import kve

# Constants identical to syncem.constants (unchanged since SyncMoments d51f9be)
# (checked against the imported module in bridge_harmonic_probe.py).
E_ESU = 4.803204712570263e-10
M_E = 9.1093837139e-28
C_CGS = 2.99792458e10
C_SI_M = 2.99792458e8

# Configuration copied from radio_numerics.py (checked against results_final.json).
B0 = 5e-6
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
_bx, _bw = leggauss(256)
BUMP_AREA = float(np.dot(_bw, np.exp(1 - 1 / (1 - _bx**2))))

ADOPTED = dict(field=8, eta=7)  # harmonic adopted (refined) B, eta nodes
COARSE = dict(field=6, eta=5)  # harmonic coarse-check B, eta nodes

X_LO, X_HI = 1e-7, 800.0  # table range; F, G < 1e-340 above X_HI
_LX = np.linspace(np.log(X_LO), np.log(X_HI), 12001)


def uniform(lo, hi, nodes):
    """Gauss-Legendre nodes and probability weights of U[lo, hi] (as harmonic)."""
    x, w = leggauss(nodes)
    return (lo + hi) / 2 + (hi - lo) * x / 2, w / 2


def f_times_exp(x, step=0.01, umax=24.0):
    """F(x) e^x = x int_0^inf exp(-x(cosh u - 1)) cosh(5u/3)/cosh(u) du.

    Trapezoid rule on an even analytic integrand (error ~ exp(-pi^2/step)).
    """
    u = np.arange(0.0, umax + step / 2, step)
    w = np.full(u.size, step)
    w[0] = step / 2
    shape = np.cosh(5 * u / 3) / np.cosh(u)
    out = np.empty(np.size(x))
    xs = np.atleast_1d(np.asarray(x, dtype=float))
    for start in range(0, xs.size, 400):
        block = xs[start : start + 400, None]
        out[start : start + 400] = block[:, 0] * np.sum(
            w * shape * np.exp(-block * np.expm1(np.log(np.cosh(u)))), axis=1
        )
    return out


def _tables():
    x = np.exp(_LX)
    return {
        "F": CubicSpline(_LX, np.log(f_times_exp(x))),
        "K53": CubicSpline(_LX, np.log(kve(5 / 3, x))),
        "K23": CubicSpline(_LX, np.log(kve(2 / 3, x))),
        "K13": CubicSpline(_LX, np.log(kve(1 / 3, x))),
    }


TABLES = _tables()


def special(x, which):
    """Spline-tabulated F or K_nu at x; zero above X_HI."""
    x = np.asarray(x, dtype=float)
    if np.any(x < X_LO):
        raise ValueError("argument below table range")
    out = np.zeros_like(x)
    ok = x < X_HI
    out[ok] = np.exp(TABLES[which](np.log(x[ok])) - x[ok])
    return out


def shape_functions(x, order):
    """d^r/du^r of F(e^u) and G(e^u), u = ln x, for r = 0..order (lists)."""
    f = special(x, "F")
    k23 = special(x, "K23")
    if order == 0:
        return [f], [x * k23]
    k53, k13 = special(x, "K53"), special(x, "K13")
    x2, x3 = x * x, x * x * x
    fs = [f, f - x2 * k53, f - (4 / 3) * x2 * k53 + x3 * k23]
    gs = [
        x * k23,
        x * k23 / 3 - x2 * k13,
        x * k23 / 9 - 2 * x2 * k13 + x3 * k23,
    ]
    return fs[: order + 1], gs[: order + 1]


class ContinuumKernel:
    """Channel-averaged continuum kernel Kbar_X,j(gamma) for X = I, Re P, Im P."""

    def __init__(self, field=8, eta=7, channel=400, centres=CENTRES):
        fields, fw = uniform(4e-6, 6e-6, field)
        etas, ew = uniform(-0.5, 0.5, eta)
        bperp = (fields[:, None] * np.sqrt(1 - etas[None, :] ** 2)).ravel()
        self.weight = (fw[:, None] * ew[None, :]).ravel()
        self.amp = np.sqrt(3) * E_ESU**3 * bperp / (4 * np.pi * M_E * C_CGS**2)
        self.a_b = 3 * E_ESU * bperp / (4 * np.pi * M_E * C_CGS)
        t, tw = leggauss(channel)
        self.wt = tw * np.exp(1 - 1 / (1 - t**2)) / BUMP_AREA
        self.nu = centres[:, None] * (1 + HALF * t[None, :])
        lam2 = (C_SI_M / self.nu) ** 2
        self.screen = np.exp(2j * SCREEN_MEAN * lam2 - 2 * SCREEN_SD**2 * lam2**2)
        self.settings = dict(field=field, eta=eta, channel=channel)

    def log_derivatives(self, gamma, order=0):
        """d^r Kbar / d(ln gamma)^r, r = 0..order; array (order+1, J, 3)."""
        n_r = order + 1
        acc_i = np.zeros((n_r,) + self.nu.shape)
        acc_q = np.zeros((n_r,) + self.nu.shape)
        for q in range(self.amp.size):
            x = self.nu / (self.a_b[q] * gamma**2)
            fs, gs = shape_functions(x, order)
            for r in range(n_r):
                acc_i[r] += self.weight[q] * self.amp[q] * fs[r]
                acc_q[r] -= self.weight[q] * self.amp[q] * gs[r]
        out = np.zeros((n_r, self.nu.shape[0], 3))
        for r in range(n_r):
            scale = (-2.0) ** r  # d/d ln gamma = -2 d/d ln x
            p = SKY * np.sum(self.wt * acc_q[r] * self.screen, axis=1)
            out[r, :, 0] = scale * np.sum(self.wt * acc_i[r], axis=1)
            out[r, :, 1] = scale * p.real
            out[r, :, 2] = scale * p.imag
        return out

    def __call__(self, gamma):
        return self.log_derivatives(gamma, 0)[0]


def fd_pure(kernel, step):
    """(12, 3, J, 3) Taylor coefficients by the harmonic 5-point stencil in x_b."""
    pure = np.zeros((12, 3, len(CENTRES), 3))
    for b, (mid, half) in enumerate(zip(LOG_MIDS, LOG_HALF)):
        v = np.stack([kernel(float(np.exp(mid + half * step * o))) for o in range(-2, 3)])
        pure[b, 0] = v[2]
        pure[b, 1] = (v[0] - 8 * v[1] + 8 * v[3] - v[4]) / (12 * step)
        pure[b, 2] = (-v[0] + 16 * v[1] - 30 * v[2] + 16 * v[3] - v[4]) / (24 * step**2)
    return pure


def analytic_pure(kernel):
    """(12, 3, J, 3) Taylor coefficients D_b,r = (h_b^r / r!) d^r Kbar/d(ln gamma)^r."""
    pure = np.zeros((12, 3, len(CENTRES), 3))
    for b, (mid, half) in enumerate(zip(LOG_MIDS, LOG_HALF)):
        d = kernel.log_derivatives(float(np.exp(mid)), 2)
        pure[b, 0], pure[b, 1], pure[b, 2] = d[0], half * d[1], half**2 * d[2] / 2
    return pure


def basis_from_pure(pure):
    """(J, 3, 36) response matrix, column 3b+r = REF_BIN_MASS[b] * D_b,r."""
    out = np.zeros((len(CENTRES), 3, 36))
    for b in range(12):
        for r in range(3):
            out[:, :, 3 * b + r] = REF_BIN_MASS[b] * pure[b, r]
    return out


def energy_nodes(energy):
    """Truth energy nodes and weights exactly as radio_numerics.truth()."""
    x, w = leggauss(energy)
    norm = (ENERGY_EDGES[-1] ** (1 - SLOPE) - ENERGY_EDGES[0] ** (1 - SLOPE)) / (1 - SLOPE)
    gammas, weights, xs, bins = [], [], [], []
    for b, (mid, half) in enumerate(zip(LOG_MIDS, LOG_HALF)):
        g = np.exp(mid + half * x)
        gammas.append(g)
        weights.append(half * w * g ** (1 - SLOPE) / norm)
        xs.append(x)
        bins.append(np.full(energy, b))
    return (np.concatenate(gammas), np.concatenate(weights), np.concatenate(xs),
            np.concatenate(bins))


def truth_and_moments(kernel, energy=8):
    """Direct ensemble spectra (J, 3), node kernels, and true scaled moments (36)."""
    gammas, weights, xs, bins = energy_nodes(energy)
    node_kernels = np.stack([kernel(float(g)) for g in gammas])
    truth = np.einsum("n,njs->js", weights, node_kernels)
    moments = np.zeros(36)
    for b in range(12):
        sel = bins == b
        for r in range(3):
            moments[3 * b + r] = np.sum(weights[sel] * xs[sel] ** r) / REF_BIN_MASS[b]
    return truth, moments, node_kernels, gammas, weights
