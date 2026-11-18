"""Kernels, averaged laws and populations for the field-cell rules.

* The continuum channel kernel of the core with E1 angles and screen. ``eta`` is averaged on
  [0, 1/2] by 12 Gauss-Legendre nodes (the kernel depends on eta through sqrt(1 - eta^2)).
* ``eta_given_B``: an inclination law U[0, e(B)] whose upper edge moves with B. It breaks the
  precondition of the field-scaling relation (Spec Sec. 8.1).
* ``FactorKernel``: multiplies P by a resolved azimuth factor c(point).
* Electron laws for an averaged gamma: a power law on fixed panels (independent of B), and
  the E6 cut-off law p(gamma | B) whose support ends at gamma_c(B).
* Populations as core ``Measure`` objects over (gamma, B): the E6 joint law, and a power law
  with a field law conditional on gamma.
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi

import core as C
from core.paper import GAMMA_HI, GAMMA_LO, SCREEN

# E6 constants (study cost.gamma_c_local): Thomson cross-section loss with 1 eV cm^-3 radiation, 19 Myr
SIGMA_T, M_E, C_CGS = 6.6524587321e-25, 9.1093837015e-28, 2.99792458e10
LOSS = 4.0 * SIGMA_T / (3.0 * M_E * C_CGS)
EV_ERG, SEC_PER_MYR, MUG = 1.602176634e-12, 3.15576e13, 1e-6
P_E6, T_E6 = 2.6, 19.0

_E, _WE = leggauss(12)
ETA_NODES, ETA_PROBS = 0.25 * (_E + 1.0), _WE * 0.5


def gamma_c(B, t_myr=T_E6, u_rad_ev=1.0, xp=np):
    """E6 cut-off Lorentz factor for a local field B [uG] (study cost.gamma_c_local)."""
    b_s = LOSS * (B * MUG) ** 2 / (8 * np.pi)
    return 1.0 / ((b_s + LOSS * u_rad_ev * EV_ERG) * t_myr * SEC_PER_MYR)


def continuum(n_chan_nodes=64) -> C.ContinuumChannelKernel:
    return C.ContinuumChannelKernel(C.bump_channels(n_nodes=n_chan_nodes), screen=SCREEN)


def eta_rule():
    return C.fixed(ETA_NODES, ETA_PROBS)


def eta_given_B(scale=5.0):
    """U[0, e(B)] with e(B) = 0.9 B/(B + scale): a law of the inclination that depends on B."""
    return C.uniform(0.0, lambda p: 0.9 * p["B"] / (p["B"] + scale), 12)


class FactorKernel:
    """Wraps a point kernel and multiplies P by c(point) (a resolved azimuth factor)."""

    def __init__(self, kernel, factor):
        self.kernel, self.factor, self.n_ch = kernel, factor, kernel.n_ch

    def __call__(self, p):
        I, P = self.kernel(p)
        return I, P * self.factor(p)


# ------------------------------------------------------------------ electron laws for an averaged gamma
def _gl_panels(lo, hi, n_panels, n):
    t, w = leggauss(n)
    e = np.linspace(np.log(lo), np.log(hi), n_panels + 1)
    u = (0.5 * np.diff(e)[:, None] * t + 0.5 * (e[1:] + e[:-1])[:, None]).ravel()
    wu = (0.5 * np.diff(e)[:, None] * w).ravel()
    return u, wu


def power_law_rule(slope=2.6, lo=GAMMA_LO, hi=GAMMA_HI, n_panels=19, n=6):
    """gamma ~ gamma^-slope on [lo, hi], fixed nodes on panels equal in ln gamma, probabilities."""
    u, wu = _gl_panels(lo, hi, n_panels, n)
    g = np.exp(u)
    p = wu * g ** (1.0 - slope)
    return C.fixed(g, p / p.sum())


def cutoff_rule(slope=P_E6, lo=GAMMA_LO, n_panels=12, n=6):
    """p(gamma | B) proportional to gamma^-slope (1 - gamma/gamma_c(B))^(slope-2) on [lo, gamma_c(B)].

    Panels equal in ln gamma between ln lo and ln gamma_c(B). The last panel is Gauss-Jacobi
    with the weight of the cut-off, so the nodes move with B and the jet differentiates them.
    """
    t, w = leggauss(n)
    tj, wj = roots_jacobi(n, slope - 2.0, 0.0)
    t, w, tj, wj = (jnp.asarray(a) for a in (t, w, tj, wj))
    k = jnp.arange(n_panels)

    def rule(point):
        uc = jnp.log(gamma_c(point["B"], xp=jnp))
        ua = jnp.log(lo)
        hw = 0.5 * (uc - ua) / n_panels
        mid = ua + hw * (2 * k + 1)
        u_reg = (mid[:-1, None] + hw * t[None, :]).ravel()
        w_reg = jnp.tile(hw * w, n_panels - 1)
        g = jnp.exp(u_reg)
        dens = g ** (1.0 - slope) * (-jnp.expm1(jnp.log(g) - uc)) ** (slope - 2.0)
        u_j = mid[-1] + hw * tj
        gj = jnp.exp(u_j)
        d = uc - u_j
        ratio = -jnp.expm1(-d) / d
        dens_j = gj ** (1.0 - slope) * ratio ** (slope - 2.0)
        nodes = jnp.concatenate([g, gj])
        p = jnp.concatenate([w_reg * dens, hw ** (slope - 1.0) * wj * dens_j])
        return nodes, p / jnp.sum(p)

    return rule


# ------------------------------------------------------------------ populations over (gamma, B)
def _gamma_panels_cut(edges, gc, n, slope=P_E6):
    """Unnormalised E6 measure in gamma for one field value: cell-edge panels, Jacobi at gamma_c."""
    t, w = leggauss(n)
    tj, wj = roots_jacobi(n, slope - 2.0, 0.0)
    u_e = np.log(edges[edges < gc])
    u_e = np.append(u_e, np.log(gc))
    gs, ws = [], []
    uc = np.log(gc)
    for k, (ua, ub) in enumerate(zip(u_e[:-1], u_e[1:])):
        hw = 0.5 * (ub - ua)
        if k == len(u_e) - 2:
            u = ua + hw * (tj + 1.0)
            d = uc - u
            ratio = -np.expm1(-d) / d
            gs.append(np.exp(u))
            ws.append(hw ** (slope - 1.0) * wj * np.exp(u) ** (1.0 - slope) * ratio ** (slope - 2.0))
        else:
            u = ua + hw * (t + 1.0)
            g = np.exp(u)
            gs.append(g)
            ws.append(hw * w * g ** (1.0 - slope) * (-np.expm1(u - uc)) ** (slope - 2.0))
    return np.concatenate(gs), np.concatenate(ws)


def field_nodes(B_edges, n_B, chart="affine"):
    """Gauss-Legendre nodes of a field law split at the B cell edges, weights of dB (affine) or dln B."""
    m = C.panel_measure("B", lambda x: np.ones_like(x), B_edges, n=n_B, chart=chart)
    return m.nodes["B"], m.weights


def e6_measure(gamma_edges, B_edges, B_density, n_B=8, n_g=12, slope=P_E6, normalise_column=True):
    """N(gamma, B) = P(B) gamma^-slope (1 - gamma/gamma_c(B))^(slope-2) on [gamma_lo, gamma_c(B)].

    ``B_density(B)`` is P(B) per uG. ``normalise_column``: the total column is one.
    Field panels split at ``B_edges``, energy panels at ``gamma_edges`` and at gamma_c(B).
    """
    bn, bw = field_nodes(B_edges, n_B, chart="affine")
    G, Bs, W = [], [], []
    for b, wb in zip(bn, bw * B_density(bn)):
        g, w = _gamma_panels_cut(np.asarray(gamma_edges), gamma_c(b), n_g, slope)
        G.append(g)
        Bs.append(np.full(g.size, b))
        W.append(wb * w)
    W = np.concatenate(W)
    if normalise_column:
        W = W / W.sum()
    return C.Measure({"gamma": np.concatenate(G), "B": np.concatenate(Bs)}, W)


def conditional_field_measure(gamma_edges, B_edges, centre, half=0.5, slope=2.6, n_g=8, n_B=6):
    """gamma^-slope on the gamma edges times B ~ U[centre(gamma) - half, centre(gamma) + half].

    A field law conditional on the resolved gamma. Field panels split at the B cell edges.
    """
    gm = C.panel_measure("gamma", lambda x: x ** (-slope), gamma_edges, n=n_g, chart="log")
    t, w = leggauss(n_B)
    G, Bs, W = [], [], []
    be = np.asarray(B_edges)
    for g, wg in zip(gm.nodes["gamma"], gm.weights):
        a, b = centre(g) - half, centre(g) + half
        cuts = np.unique(np.r_[a, be[(be > a) & (be < b)], b])
        for lo, hi in zip(cuts[:-1], cuts[1:]):
            x = 0.5 * (hi - lo) * t + 0.5 * (hi + lo)
            G.append(np.full(n_B, g))
            Bs.append(x)
            W.append(wg * 0.5 * (hi - lo) * w / (2 * half))
    W = np.concatenate(W)
    return C.Measure({"gamma": np.concatenate(G), "B": np.concatenate(Bs)}, W / W.sum())
