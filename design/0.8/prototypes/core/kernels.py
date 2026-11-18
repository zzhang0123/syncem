"""Point kernels K(point) -> (I (n_ch,), P (n_ch,) complex).

A point is a dict of scalar values keyed by variable name. P carries the Faraday phase
inside the channel integral and no azimuth factor exp(2i phi).

* :class:`ContinuumChannelKernel` wraps ``syncem.model.ContinuumKernel.kernels`` (F and G of
  ``syncem.ultrarel``) and integrates over bump channels as the study's ``config.py``
  (h_j = 0.2 nu_j, unit-area response rho_1). B is in microgauss.
* :class:`PolynomialKernel` is a polynomial in chart variables with exact Taylor coefficients.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate

import syncem  # noqa: F401  (enables float64)
from syncem.constants import C_SI_M
from syncem.model import ContinuumKernel

MUG = 1e-6


def rho1(t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    ok = np.abs(t) < 1
    out[ok] = np.exp(-t[ok] ** 2 / (1 - t[ok] ** 2))
    return out


Z1 = integrate.quad(lambda t: float(rho1(np.array(t))), -1, 1, epsabs=0, epsrel=1e-13, limit=400)[0]


@dataclass(frozen=True, eq=False)
class Channels:
    """Channel quadrature: sum_t w[j, t] f(nu[j, t]) = int R_j(nu) f(nu) dnu."""

    nu: np.ndarray  # (J, T) Hz
    w: np.ndarray  # (J, T)

    @property
    def n_ch(self) -> int:
        return self.nu.shape[0]


def bump_channels(nu_lo=1e8, nu_hi=1e10, n=41, half=0.2, n_nodes=64) -> Channels:
    """The paper's channels: centres log-spaced, support nu_j (1 +- half), unit area."""
    t, w = leggauss(n_nodes)
    centres = np.geomspace(nu_lo, nu_hi, n)
    nu = centres[:, None] * (1.0 + half * t[None, :])
    return Channels(nu, np.broadcast_to(w * rho1(t) / Z1, nu.shape).copy())


class ContinuumChannelKernel:
    """Continuum kernel of syncem integrated over channels, variables gamma, B [uG], eta, depth.

    ``screen=(mean, sd)`` multiplies P by the Gaussian characteristic function
    exp(2i lam^2 mean - 2 lam^4 sd^2) inside the channel integral. A ``depth`` value in the
    point multiplies P by exp(2i lam^2 depth). ``x_min`` is passed to ContinuumKernel. The
    default 1e-6 raises in the top E1 cell at 0.1 GHz (x about 6e-7), so the default here is 1e-9.
    """

    variables = ("gamma", "B", "eta", "depth")

    def __init__(self, channels: Channels, *, screen=None, n_nodes_F=128, x_min=1e-9):
        self.channels = channels
        self.base = ContinuumKernel(n_nodes_F=n_nodes_F, x_min=x_min)
        self.nu = jnp.asarray(channels.nu)
        self.w = jnp.asarray(channels.w)
        self.lam2 = jnp.asarray((C_SI_M / channels.nu) ** 2)
        if screen is None:
            self.phase = jnp.ones_like(self.lam2, dtype=complex)
        else:
            m, s = screen
            self.phase = jnp.exp(2j * self.lam2 * m - 2.0 * self.lam2**2 * s**2)

    @property
    def n_ch(self) -> int:
        return self.channels.n_ch

    def __call__(self, p):
        k_I, k_Q = self.base.kernels(self.nu, p["gamma"], p["B"] * MUG, p["eta"])
        phase = self.phase
        if "depth" in p:
            phase = phase * jnp.exp(2j * self.lam2 * p["depth"])
        return jnp.sum(self.w * k_I, axis=-1), jnp.sum(self.w * k_Q * phase, axis=-1)


class PolynomialKernel:
    """K = sum_alpha c_alpha prod_v t_v^alpha_v with t_v = (ln x_v - c_v)/s_v (log) or (x_v - c_v)/s_v.

    ``transforms[v] = (chart, centre, scale)``. ``coef_I[alpha]`` is (n_ch,) real and
    ``coef_P[alpha]`` (n_ch,) complex. :meth:`taylor` gives the exact coefficients
    (1/r!) d^r/du^r in the cell chart, which is affine in t_v for the matching chart.
    """

    def __init__(self, transforms: dict, coef_I: dict, coef_P: dict):
        self.variables = tuple(transforms)
        self.transforms = transforms
        self.alphas = tuple(coef_I)
        self.cI = np.stack([np.asarray(coef_I[a], float) for a in self.alphas])
        self.cP = np.stack([np.asarray(coef_P[a], complex) for a in self.alphas])
        self.n_ch = self.cI.shape[1]

    @classmethod
    def random(cls, transforms: dict, degree: int, n_ch: int, seed=0) -> "PolynomialKernel":
        rng = np.random.default_rng(seed)
        d = len(transforms)
        alphas = [a for a in np.ndindex(*(degree + 1,) * d) if sum(a) <= degree]
        cI = {a: rng.normal(size=n_ch) for a in alphas}
        cP = {a: rng.normal(size=n_ch) + 1j * rng.normal(size=n_ch) for a in alphas}
        return cls(transforms, cI, cP)

    def _t(self, v, x, xp):
        chart, c, s = self.transforms[v]
        return ((xp.log(x) if chart == "log" else x) - c) / s

    def __call__(self, p):
        t = [self._t(v, p[v], jnp) for v in self.variables]
        mono = jnp.stack([math.prod([t[i] ** a[i] for i in range(len(t))]) for a in self.alphas])
        return mono @ jnp.asarray(self.cI), mono @ jnp.asarray(self.cP)

    def taylor(self, ref: dict, hw: dict, exponents) -> tuple[np.ndarray, np.ndarray]:
        """Exact (1/r!) d^r/du^r at u = 0, shapes (n_rows, n_ch). The cell chart must match."""
        tb = {v: float(self._t(v, ref[v], np)) for v in self.variables}
        sl = {v: hw[v] / self.transforms[v][2] for v in self.variables}
        coef = np.zeros((len(exponents), len(self.alphas)))
        for k, r in enumerate(exponents):
            for m, a in enumerate(self.alphas):
                coef[k, m] = math.prod(
                    math.comb(a[i], r[i]) * tb[v] ** (a[i] - r[i]) * sl[v] ** r[i] if r[i] <= a[i] else 0.0
                    for i, v in enumerate(self.variables))
        return coef @ self.cI, coef @ self.cP
