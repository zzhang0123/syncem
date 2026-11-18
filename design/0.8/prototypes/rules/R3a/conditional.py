"""Conditional laws of a resolved child on resolved parents, and their child-cell moments.

g[c, s](parents) = E[1_c(x) u_c(x)^s | parents] for child cell c (Spec Sec. 5.2 R3a).
Three ways to compute g are provided, because the specification leaves the quadrature open.

* ``exact``: closed-form partial moments (truncated normal in ln B, or lognormal partial
  moments with a binomial expansion in the affine chart). The reference.
* ``fixed``: Gauss-Legendre nodes fixed in the child cell's chart, weighted by the
  conditional density at the parent point (frozen-node reweighting). Derivatives in the
  parents are those of the density, so the jet is correct when the rule resolves the density.
* ``moving``: the conditional's own nodes (Gauss-Hermite in ln B, moving with the median)
  multiplied by the indicator of the child cell. The indicator has zero derivative, so the
  jet drops the boundary flux at the child-cell edges.

All functions are written in jnp so that ``jax.jacfwd`` gives the Taylor coefficients h.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np
from jax.scipy.special import ndtr
from numpy.polynomial.hermite import hermgauss
from numpy.polynomial.legendre import leggauss

SQ2PI = np.sqrt(2.0 * np.pi)


def child_bounds(cells):
    """Arrays lo, hi, ref, hw of a core Cells axis. B is positive, so an affine lo of -inf is 0."""
    lo = np.where(np.isfinite(cells.lo), cells.lo, 0.0)
    lo = np.maximum(lo, 0.0)
    return lo, np.asarray(cells.hi, float), np.asarray(cells.reference, float), np.asarray(cells.half_width, float)


def _trunc_normal_moments(m, sd, xl, xh, S):
    """M_k = int_{xl}^{xh} x^k N(x; m, sd^2) dx for k = 0..S. xl, xh may be infinite."""
    # xl, xh are static floats: infinite bounds are handled in Python, so no inf enters the trace
    lo_fin, hi_fin = bool(np.isfinite(xl)), bool(np.isfinite(xh))
    zl = (xl - m) / sd if lo_fin else None
    zh = (xh - m) / sd if hi_fin else None
    M = [(ndtr(zh) if hi_fin else 1.0) - (ndtr(zl) if lo_fin else 0.0) + 0.0 * m]
    fl = jnp.exp(-0.5 * zl**2) / (SQ2PI * sd) if lo_fin else 0.0
    fh = jnp.exp(-0.5 * zh**2) / (SQ2PI * sd) if hi_fin else 0.0
    xlf = xl if lo_fin else 0.0
    xhf = xh if hi_fin else 0.0
    for k in range(1, S + 1):
        bnd = xhf ** (k - 1) * fh - xlf ** (k - 1) * fl
        prev2 = M[k - 2] if k >= 2 else 0.0
        M.append(m * M[k - 1] + (k - 1) * sd**2 * prev2 - sd**2 * bnd)
    return jnp.stack(M)


@dataclass(frozen=True)
class LogNormalConditional:
    """B | gamma ~ LogNormal with median m0 (gamma / gamma_p)^beta and width sigma in ln B.

    ``mu(point)`` is ln of the median. With a log chart on gamma, mu is affine in u_gamma, so
    without child-cell edges g[c, s] is a polynomial of degree s in u_gamma (log chart on B).
    ``depth_slope`` adds a dependence on a second parent: mu += depth_slope * (depth - 8).
    """

    m0: float
    gamma_p: float
    beta: float
    sigma: float
    depth_slope: float = 0.0

    def mu(self, point):
        mu = jnp.log(self.m0) + self.beta * jnp.log(point["gamma"] / self.gamma_p)
        if self.depth_slope:
            mu = mu + self.depth_slope * (point["depth"] - 8.0)
        return mu

    def density(self, B, point):
        z = (jnp.log(B) - self.mu(point)) / self.sigma
        return jnp.exp(-0.5 * z**2) / (SQ2PI * self.sigma * B)

    def breaks_lnB(self, point, span=12.0, step=1.0):
        """Panel breaks in ln B that resolve the density (median +- span sigma)."""
        return np.asarray(self.mu(point)) + self.sigma * np.arange(-span, span + step / 2, step)

    # ---------------------------------------------------------------- g by three methods
    def g_exact(self, point, chart, cells, S):
        lo, hi, ref, hw = child_bounds(cells)
        mu = self.mu(point)
        if chart == "log":
            out = []
            for c in range(lo.size):
                xl = (np.log(lo[c]) - np.log(ref[c])) / hw[c] if lo[c] > 0 else -np.inf
                xh = (np.log(hi[c]) - np.log(ref[c])) / hw[c] if np.isfinite(hi[c]) else np.inf
                m = (mu - np.log(ref[c])) / hw[c]
                out.append(_trunc_normal_moments(m, self.sigma / hw[c], xl, xh, S))
            return jnp.stack(out)
        # affine: Gauss-Legendre in z = (ln B - mu) / sigma over the cell clipped to |z| <= 14,
        # 8 equal subpanels of 16 nodes. The closed form below (g_closed) loses digits for wide laws.
        t, w = leggauss(16)
        out = []
        for c in range(lo.size):
            ylo = np.log(lo[c]) if lo[c] > 0 else -np.inf
            yhi = np.log(hi[c]) if np.isfinite(hi[c]) else np.inf
            zl = jnp.clip((ylo - mu) / self.sigma, -14.0, 14.0)
            zh = jnp.clip((yhi - mu) / self.sigma, -14.0, 14.0)
            e = zl + (zh - zl) * jnp.linspace(0.0, 1.0, 9)
            z = (0.5 * (e[1:] - e[:-1])[:, None] * t + 0.5 * (e[1:] + e[:-1])[:, None]).ravel()
            wz = (0.5 * (e[1:] - e[:-1])[:, None] * w).ravel()
            x = (jnp.exp(mu + self.sigma * z) - ref[c]) / hw[c]
            q = wz * jnp.exp(-0.5 * z**2) / SQ2PI
            out.append(jnp.stack([jnp.sum(q * x**s) for s in range(S + 1)]))
        return jnp.stack(out)

    def g_closed(self, point, chart, cells, S):
        """Affine chart: E[B^j 1(lo < B < hi)] in closed form, then the binomial expansion of x^s."""
        lo, hi, ref, hw = child_bounds(cells)
        mu = self.mu(point)
        out = []
        for c in range(lo.size):
            ylo = np.log(lo[c]) if lo[c] > 0 else -np.inf
            yhi = np.log(hi[c]) if np.isfinite(hi[c]) else np.inf
            EB = [jnp.exp(j * mu + 0.5 * j**2 * self.sigma**2)
                  * (ndtr((yhi - mu - j * self.sigma**2) / self.sigma)
                     - ndtr((ylo - mu - j * self.sigma**2) / self.sigma)) for j in range(S + 1)]
            row = []
            for s in range(S + 1):
                acc = 0.0
                for j in range(s + 1):
                    acc = acc + _comb(s, j) * (-ref[c]) ** (s - j) * EB[j]
                row.append(acc / hw[c] ** s)
            out.append(jnp.stack(row))
        return jnp.stack(out)

    def g_fixed(self, point, chart, cells, S, n=32):
        """Gauss-Legendre nodes fixed on each child cell in its chart, density reweighting."""
        lo, hi, ref, hw = child_bounds(cells)
        if not np.all(np.isfinite(hi)) or np.any(lo <= 0) and chart == "log":
            raise ValueError("fixed nodes need bounded child cells")
        t, w = leggauss(n)
        out = []
        for c in range(lo.size):
            B = ref[c] * np.exp(hw[c] * t) if chart == "log" else ref[c] + hw[c] * t
            jac = hw[c] * B if chart == "log" else hw[c] * np.ones_like(t)
            p = self.density(jnp.asarray(B), point) * jac * w
            out.append(jnp.stack([jnp.sum(p * t**s) for s in range(S + 1)]))
        return jnp.stack(out)

    def g_moving(self, point, chart, cells, S, n=64):
        """The conditional's own Gauss-Hermite nodes in ln B times the child-cell indicator."""
        lo, hi, ref, hw = child_bounds(cells)
        t, w = hermgauss(n)
        B = jnp.exp(self.mu(point) + np.sqrt(2.0) * self.sigma * t)
        wq = w / np.sqrt(np.pi)
        out = []
        for c in range(lo.size):
            inside = (B >= lo[c]) & (B < hi[c])
            x = jnp.log(B / ref[c]) / hw[c] if chart == "log" else (B - ref[c]) / hw[c]
            out.append(jnp.stack([jnp.sum(jnp.where(inside, wq * x**s, 0.0)) for s in range(S + 1)]))
        return jnp.stack(out)

    def g(self, point, chart, cells, S, method="exact", **kw):
        return {"exact": self.g_exact, "fixed": self.g_fixed, "moving": self.g_moving}[method](
            point, chart, cells, S, **kw)

    def taylor_u(self, gamma_b, h_b, chart, cells, S, K, n=16, span=14.0):
        """h[c, s, k] = (1/k!) d^k g[c, s] / du^k at u = 0, u = ln(gamma / gamma_b) / h_b, NumPy.

        mu is affine in u with slope a = beta h_b, and d^k/dmu^k of the normal density in ln B is
        sigma^-k He_k(z) times the density (He the probabilists' Hermite polynomials). The cell
        integrals use Gauss-Legendre panels of unit width in z, so narrow laws are resolved.
        Only the gamma dependence is differentiated (depth_slope must be 0).
        """
        if self.depth_slope:
            raise ValueError("taylor_u differentiates in gamma only")
        lo, hi, ref, hw = child_bounds(cells)
        mu0 = np.log(self.m0) + self.beta * np.log(gamma_b / self.gamma_p)
        a = self.beta * h_b
        t, w = leggauss(n)
        out = np.zeros((lo.size, S + 1, K + 1))
        fact = np.array([math.factorial(k) for k in range(K + 1)], float)
        for c in range(lo.size):
            ylo = np.log(lo[c]) if lo[c] > 0 else -np.inf
            yhi = np.log(hi[c]) if np.isfinite(hi[c]) else np.inf
            zl, zh = max((ylo - mu0) / self.sigma, -span), min((yhi - mu0) / self.sigma, span)
            if zh <= zl:
                continue
            zb = np.unique(np.concatenate([[zl, zh], np.arange(np.ceil(zl), np.floor(zh) + 1)]))
            z = np.concatenate([0.5 * (q - p) * t + 0.5 * (q + p) for p, q in zip(zb[:-1], zb[1:])])
            wz = np.concatenate([0.5 * (q - p) * w for p, q in zip(zb[:-1], zb[1:])])
            phi = np.exp(-0.5 * z**2) / SQ2PI  # density in z, since p(B) dB = phi(z) dz
            y = mu0 + self.sigma * z
            x = (y - np.log(ref[c])) / hw[c] if chart == "log" else (np.exp(y) - ref[c]) / hw[c]
            He = [np.ones_like(z), z]
            for k in range(2, K + 1):
                He.append(z * He[k - 1] - (k - 1) * He[k - 2])
            for s in range(S + 1):
                for k in range(K + 1):
                    out[c, s, k] = (a / self.sigma) ** k / fact[k] * np.sum(wz * phi * x**s * He[k])
        return out

    # ---------------------------------------------------------------- rules for the averaged route
    def hermite_rule(self, n=40):
        """Moving Gauss-Hermite rule of the full law (no child cells): exact for polynomials in ln B."""
        t, w = hermgauss(n)
        t, wq = jnp.asarray(t), jnp.asarray(w / np.sqrt(np.pi))

        def rule(point):
            return jnp.exp(self.mu(point) + np.sqrt(2.0) * self.sigma * t), wq

        return rule

    def cell_rule(self, chart, cells, n=48):
        """Fixed Gauss-Legendre nodes on the child cells with density weights (mass inside cells)."""
        lo, hi, ref, hw = child_bounds(cells)
        t, w = leggauss(n)
        Bs, J = [], []
        for c in range(lo.size):
            B = ref[c] * np.exp(hw[c] * t) if chart == "log" else ref[c] + hw[c] * t
            Bs.append(B)
            J.append((hw[c] * B if chart == "log" else hw[c] * np.ones_like(t)) * w)
        Bn, Jn = jnp.asarray(np.concatenate(Bs)), jnp.asarray(np.concatenate(J))

        def rule(point):
            return Bn, self.density(Bn, point) * Jn

        return rule


@dataclass(frozen=True)
class UniformShiftConditional:
    """B | gamma ~ U[c(gamma) - delta, c(gamma) + delta], c = c0 + kappa ln(gamma / gamma_p).

    c is affine in u_gamma. While the support stays inside one affine child cell, g[c, s] is
    a polynomial of degree s in u_gamma, the exactness case of R3a.
    """

    c0: float
    gamma_p: float
    kappa: float
    delta: float

    def centre(self, point):
        return self.c0 + self.kappa * jnp.log(point["gamma"] / self.gamma_p)

    def density(self, B, point):
        c = self.centre(point)
        return jnp.where(jnp.abs(B - c) <= self.delta, 0.5 / self.delta, 0.0)

    def breaks_B(self, point):
        c = float(self.centre(point))
        return np.array([c - self.delta, c + self.delta])

    def g_exact(self, point, chart, cells, S):
        if chart != "affine":
            raise ValueError("UniformShiftConditional is written for an affine child chart")
        lo, hi, ref, hw = child_bounds(cells)
        c = self.centre(point)
        out = []
        for k in range(lo.size):
            a = jnp.clip(c - self.delta, lo[k], hi[k])
            b = jnp.clip(c + self.delta, lo[k], hi[k])
            xa, xb = (a - ref[k]) / hw[k], (b - ref[k]) / hw[k]
            out.append(jnp.stack([(xb ** (s + 1) - xa ** (s + 1)) / (s + 1) * hw[k] / (2 * self.delta)
                                  for s in range(S + 1)]))
        return jnp.stack(out)

    def g(self, point, chart, cells, S, method="exact", **kw):
        if method != "exact":
            raise ValueError("only exact g for the uniform conditional")
        return self.g_exact(point, chart, cells, S)

    def hermite_rule(self, n=16):
        t, w = leggauss(n)
        t, w = jnp.asarray(t), jnp.asarray(0.5 * w)

        def rule(point):
            return self.centre(point) + self.delta * t, w

        return rule


def _comb(n, k):
    from math import comb
    return float(comb(n, k))
