"""Beaming-limit column rules (Spec Sec. 4.2, 4.3, 8.1) built on the prototype core.

Under ``Beaming(variable="mu", partner="eta")`` the natural kernel tends to
``K_nat -> 2 K_iso(eta) delta(mu - eta)``. Every construction here takes the factor as an
argument (``factor``, default 2) so that the tests can show what the factor 1 gives.

Constructions
* Pitch Legendre (mu on Legendre(L), eta averaged with probabilities q_e at nodes eta_e):
  column l = sum_e q_e (factor/2)(2l+1) P_l(eta_e) Kbar(eta_e). With factor 2 these are the
  eta weights (2l+1) P_l(eta) of ``ST/lib/basis.py:186-191``.
* Diagonal Legendre (mu on Cells, eta on Legendre(L_eta)): the cell columns are the Taylor
  coefficients in mu of (factor/2)(2k+1) P_k(mu) Kbar(eta = mu), paired with
  <1_b(mu) u^r P_k(eta)>.
* Diagonal density (mu on Cells, eta averaged with density q): the cell columns are the Taylor
  coefficients in mu of factor q(mu) Kbar(eta = mu), paired with <1_b(mu) u^r>.

The direct reference integrates the population density on the diagonal mu = eta:
X = int deta int dgamma factor n(gamma, mu = eta, eta) Kbar(gamma, eta), with n the joint
population density in (mu, eta). It is formed as a core Measure over (gamma, eta) whose
weights already hold factor n(gamma, eta, eta), and evaluated by ``core.direct``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

import core as C

BEAMING_FACTOR = 2.0


# ------------------------------------------------------------------ quadrature of eta laws
def gl_panels(breaks, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Composite Gauss-Legendre nodes and integration weights between consecutive breaks."""
    t, w = leggauss(n)
    b = np.asarray(breaks, float)
    xs = [0.5 * (c - a) * t + 0.5 * (c + a) for a, c in zip(b[:-1], b[1:])]
    ws = [0.5 * (c - a) * w for a, c in zip(b[:-1], b[1:])]
    return np.concatenate(xs), np.concatenate(ws)


def uniform_law(lo: float, hi: float, n: int, panels: int = 1) -> tuple[np.ndarray, np.ndarray]:
    """Nodes and probabilities of U[lo, hi] by composite Gauss-Legendre."""
    x, w = gl_panels(np.linspace(lo, hi, panels + 1), n)
    return x, w / (hi - lo)


def union_law(intervals, n: int) -> tuple[np.ndarray, np.ndarray]:
    """Uniform law on a union of disjoint intervals, probabilities summing to one."""
    xs, ws = zip(*(gl_panels([a, b], n) for a, b in intervals))
    x, w = np.concatenate(xs), np.concatenate(ws)
    return x, w / w.sum()


def half_range(lo: float, hi: float, n: int) -> tuple[np.ndarray, np.ndarray]:
    """The half-range shortcut for an even law U[-hi, hi]: nodes on [0, hi], probabilities on [0, hi]."""
    if lo != -hi:
        raise ValueError("half_range is defined for U[-a, a]")
    return uniform_law(0.0, hi, n)


def mirror(nodes, probs) -> tuple[np.ndarray, np.ndarray]:
    """Full-range rule from a half-range rule of an even law: nodes at +-x, probabilities halved."""
    return np.concatenate([-nodes[::-1], nodes]), 0.5 * np.concatenate([probs[::-1], probs])


def legendre_matrix(x, L: int) -> np.ndarray:
    """(n, L+1) values P_l(x)."""
    x = np.asarray(x, float)
    return np.stack([eval_legendre(l, x) for l in range(L + 1)], axis=-1)


def pitch_weights(eta, probs, L: int, factor: float = BEAMING_FACTOR, mask_odd: bool = False) -> np.ndarray:
    """(n_eta, L+1) weights q_e (factor/2)(2l+1) P_l(eta_e). ``mask_odd`` zeroes odd l."""
    ell = np.arange(L + 1)
    W = np.asarray(probs)[:, None] * legendre_matrix(eta, L) * (factor / 2.0) * (2 * ell + 1)
    return W * (ell % 2 == 0) if mask_odd else W


def contract(columns: dict, W: np.ndarray) -> dict:
    """Columns (n_rows, n_ch, n_eta) contracted with eta weights (n_eta, L+1)."""
    return {k: np.einsum("rcn,nl->rcl", v, W) for k, v in columns.items()}


# ------------------------------------------------------------------ kernels
class EtaNodesKernel:
    """Kernel values at fixed eta nodes as a trailing axis: (I (n_ch, n_eta), P)."""

    def __init__(self, base, eta_nodes):
        self.base = base
        self.eta = jnp.asarray(eta_nodes, float)
        self.n_ch = base.n_ch

    def __call__(self, p):
        I, P = jax.vmap(lambda e: self.base({**p, "eta": e}))(self.eta)
        return I.T, P.T


class PitchLegendreKernel:
    """K(point) (factor/2)(2l+1) P_l(eta): the rule route of the pitch-Legendre columns."""

    def __init__(self, base, L: int, factor: float = BEAMING_FACTOR):
        self.base, self.L, self.factor = base, int(L), float(factor)
        self.n_ch = base.n_ch

    def _P(self, x):
        return _legendre_jnp(x, self.L) * (self.factor / 2.0) * (2 * jnp.arange(self.L + 1) + 1)

    def __call__(self, p):
        I, P = self.base(p)
        w = self._P(p["eta"])
        return I[:, None] * w[None, :], P[:, None] * w[None, :]


class DiagonalLegendreKernel:
    """mu on Cells, eta on Legendre(L): (factor/2)(2k+1) P_k(mu) K(eta = mu), shape (n_ch, L+1)."""

    def __init__(self, base, L: int, factor: float = BEAMING_FACTOR):
        self.base, self.L, self.factor = base, int(L), float(factor)
        self.n_ch = base.n_ch

    def __call__(self, p):
        q = {k: v for k, v in p.items() if k != "mu"}
        I, P = self.base({**q, "eta": p["mu"]})
        w = _legendre_jnp(p["mu"], self.L) * (self.factor / 2.0) * (2 * jnp.arange(self.L + 1) + 1)
        return I[:, None] * w[None, :], P[:, None] * w[None, :]


class DiagonalDensityKernel:
    """mu on Cells, eta averaged with density q: factor q(mu) K(eta = mu), shape (n_ch,).

    ``density`` is a jnp function of mu (and may read other point values through ``point``).
    """

    def __init__(self, base, density, factor: float = BEAMING_FACTOR):
        self.base, self.density, self.factor = base, density, float(factor)
        self.n_ch = base.n_ch

    def __call__(self, p):
        q = {k: v for k, v in p.items() if k != "mu"}
        I, P = self.base({**q, "eta": p["mu"]})
        d = self.factor * self.density(p["mu"])
        return d * I, d * P


def _legendre_jnp(x, L: int):
    """(L+1,) values P_l(x) by the three-term recurrence (traceable)."""
    vals = [jnp.ones_like(x), x]
    for l in range(1, L):
        vals.append(((2 * l + 1) * x * vals[l] - l * vals[l - 1]) / (l + 1))
    return jnp.stack(vals[: L + 1])


# ------------------------------------------------------------------ helpers
def eta_resolved_columns(base, table, eta_nodes, averaged=(), fixed=None, outer_batch=1) -> dict:
    """Jet columns with eta kept as a node axis: {"I": (n_rows, n_ch, n_eta), "P": ...}."""
    avg = C.AveragedKernel(EtaNodesKernel(base, eta_nodes), list(averaged), fixed=fixed,
                           outer_batch=outer_batch)
    return C.build_columns(avg, table)


def rel_to_max(a, b) -> float:
    """max |a - b| / max |b|."""
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(b)))


def verdict(rel_err: float) -> str:
    """Boundary-validation verdict of the user's rule."""
    return "OK" if rel_err < 1e-3 else ("WARN" if rel_err < 1e-1 else "BAD")
