"""Family moments on the E1 cells for the fit profiles (NNLS truth, family fits).

Family: n(gamma) = A gamma^-p exp(-gamma / gamma_c), theta = (ln A, p, ln gamma_c), with the
normalisation set so that A = 1, p = 2.6, gamma_c = inf gives unit order-0 moments.
Moments a_br = (1/w_b) int_cell n(gamma) x_b^r dgamma, x_b = ln(gamma / gamma_b) / h, by a
32-node Gauss-Legendre rule per cell in ln gamma. Columns are the study's w_b D_{j;br}.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

import _common  # noqa: F401  (paths)
from config import PAPER

N_RULE = 32
_t, _w = leggauss(N_RULE)


def rule(cfg=PAPER):
    """(ln gamma nodes (n_b, 32), x nodes (32,), ln-gamma weights (n_b, 32)) of every cell."""
    lg = np.log(cfg.gamma_b)[:, None] + cfg.h * _t[None, :]
    return lg, _t, np.broadcast_to(cfg.h * _w, lg.shape)


def ref_norm(cfg=PAPER) -> float:
    """Normalisation c with sum_b int_cell c gamma^-2.6 dgamma = 1."""
    k = 1.0 - cfg.ref_slope
    return k / (cfg.gamma_hi**k - cfg.gamma_lo**k)


def moments_np(theta, N: int, cfg=PAPER) -> np.ndarray:
    """(n_b * (N+1),) moments in study column order b*(N+1) + r."""
    lnA, p, lngc = theta
    lg, x, wl = rule(cfg)
    g = np.exp(lg)
    dens = ref_norm(cfg) * np.exp(lnA - p * lg - g * np.exp(-lngc)) * g  # dgamma = gamma dln gamma
    xr = x[None, :] ** np.arange(N + 1)[:, None]  # (N+1, 32)
    a = np.einsum("bq,rq->br", dens * wl, xr) / cfg.w_b[:, None]
    return a.ravel()


def moments_jax(theta, N: int, cfg=PAPER):
    import jax.numpy as jnp

    lnA, p, lngc = theta[0], theta[1], theta[2]
    lg, x, wl = rule(cfg)
    lg, wl = jnp.asarray(lg), jnp.asarray(wl)
    g = jnp.exp(lg)
    dens = ref_norm(cfg) * jnp.exp(lnA - p * lg - g * jnp.exp(-lngc)) * g
    xr = jnp.asarray(x[None, :] ** np.arange(N + 1)[:, None])
    a = jnp.einsum("bq,rq->br", dens * wl, xr) / jnp.asarray(cfg.w_b)[:, None]
    return a.ravel()
