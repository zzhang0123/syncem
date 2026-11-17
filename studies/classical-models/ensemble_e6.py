"""E6 on any channel configuration: JP ageing with synchrotron losses in the local field (cost.joint_case).

N(gamma, B) = P(B) (A/1e4)(gamma/1e4)^-p (1 - gamma/gamma_c(B))^(p-2), P(B) = U[4, 6] muG,
gamma_c(B) = 1/((b_s(B) + b_c) t) (cost.gamma_c_local), t = 19 Myr, p = 2.6, A for unit total column.
The definitions, the 16-node Gauss-Legendre rule in B and the fixed-B kernel tables are those of cost.py,
which compares them with the paper's E6 numbers; this module evaluates them on other channel configurations
(all share the energy cells of config.PAPER). joint_columns(cfg, N) generalises cost.joint_columns (six
monomials of total degree <= 2 in x_gamma and x_B = (B - 5 muG)/(1 muG)) to total degree <= N through the
continuum identity expanded by field_coefficients(N); N = 2 reproduces cost.joint_columns bit for bit.
"""
from __future__ import annotations

from functools import lru_cache
from math import factorial

import numpy as np

import cost as C  # also puts lib/ on the path
import ensembles as E
from config import PAPER, gl_interval

P = 2.6
T_MYR = 19.0
N_B = 16
MONOMIALS = C.MONOMIALS


@lru_cache(maxsize=None)
def _b_rule():
    bn, bw = gl_interval(N_B, 4.0, 6.0)
    return bn, bw / 2.0  # weights of the uniform density on [4, 6]


@lru_cache(maxsize=None)
def _tables():
    return tuple(C.fixed_b_table(bv) for bv in _b_rule()[0])


@lru_cache(maxsize=None)
def _tab0():
    return C.fixed_b_table(C.B0)


def _cell_moments(cfg=PAPER) -> np.ndarray:
    """(N_B, n_bins, 3) unit-amplitude JP moments at each B node."""
    bn, _ = _b_rule()
    th = np.c_[np.ones(N_B), np.full(N_B, P), np.log([C.gamma_c_local(bv, T_MYR) for bv in bn])]
    return E.jp_moments(cfg, th, 2)


@lru_cache(maxsize=None)
def a_unit() -> float:
    """Amplitude for unit total electron column (as cost.joint_case)."""
    _, bw = _b_rule()
    a_k = _cell_moments()
    return 1.0 / float(np.sum(bw[:, None] * a_k[:, :, 0] * PAPER.w_b[None, :]))


def direct(cfg, n: int = 24, n_b: int = N_B) -> np.ndarray:
    """(J, 3) direct channel spectrum: sum over B nodes of the JP measure through the fixed-B table.

    n: energy nodes per cell; n_b: B nodes (other values than N_B, for node-doubling checks, build
    their own tables).
    """
    if n_b == N_B:
        (bn, bw), tables = _b_rule(), _tables()
    else:
        bn, bw = gl_interval(n_b, 4.0, 6.0)
        bw = bw / 2.0
        tables = [C.fixed_b_table(bv) for bv in bn]
    tot = np.zeros((cfg.n_channels, 3))
    for bv, wv, tb in zip(bn, bw, tables):
        m = C.jp_measure(cfg, (a_unit(), P, np.log(C.gamma_c_local(bv, T_MYR))), n)
        tot += wv * E.direct_from_measure(cfg, m, table=tb)
    return tot


def monomials(N: int = 2) -> tuple:
    """(r, s) of x_gamma^r x_B^s with r + s <= N, ordered by s then r; N = 2 gives cost.MONOMIALS."""
    return tuple((r, s) for s in range(N + 1) for r in range(N + 1 - s))


def joint_moments(N: int = 2, cfg=PAPER) -> np.ndarray:
    """(n_bins, n_monomials) joint energy-field moments in the monomials of monomials(N) (as cost.joint_case),
    on the energy cells of cfg (default those of config.PAPER)."""
    bn, bw = _b_rule()
    th = np.c_[np.ones(N_B), np.full(N_B, P), np.log([C.gamma_c_local(bv, T_MYR) for bv in bn])]
    a = a_unit() * E.jp_moments(cfg, th, N)  # (N_B, n_bins, N + 1)
    xb = (bn - C.B0) / C.DB
    mons = monomials(N)
    m = np.zeros((cfg.n_bins, len(mons)))
    for j, (r, s) in enumerate(mons):
        m[:, j] = np.sum(bw[:, None] * xb[:, None] ** s * a[:, :, r], axis=0)
    return m


def field_coefficients(N: int) -> np.ndarray:
    """alpha (N + 1, N + 1): K(u, eps) = (1 + eps) G(u + ln(1 + eps)/2) = sum_s eps^s sum_k alpha[s, k] G^(k)(u).

    The continuum identity Kbar(u, B) = lam G(u + ln(lam)/2), lam = B/B0 = 1 + eps (cost.joint_columns), expanded
    to order N in eps: alpha[s, k] is the eps^s coefficient of (1 + eps) w(eps)^k / k!, w = ln(1 + eps)/2.
    """
    w = np.zeros(N + 1)
    w[1:] = [0.5 * (-1) ** (n + 1) / n for n in range(1, N + 1)]  # ln(1 + eps)/2
    alpha = np.zeros((N + 1, N + 1))
    wk = np.zeros(N + 1)
    wk[0] = 1.0  # w^0
    for k in range(N + 1):
        pk = np.convolve([1.0, 1.0], wk)[:N + 1] / factorial(k)  # (1 + eps) w^k / k!
        alpha[:, k] = pk
        wk = np.convolve(wk, w)[:N + 1]
    return alpha


def joint_columns(cfg, N: int = 2) -> np.ndarray:
    """(J, 3, n_monomials * n_bins) joint energy-field columns; column index = n_monomials * cell + monomial.

    Column (r, s) = (h^r/r!) (DB/B0)^s sum_k alpha[s, k] G^(k + r)(u_b), with G = Kbar(., B0) and its
    u-derivatives from basis.kbar; for N = 2 this is cost.joint_columns.
    """
    kI, kQ = C.B.kbar(cfg, cfg.gamma_b, order=N, table=_tab0())  # (N + 1, nb, J): d^r/du^r
    alpha, e, h = field_coefficients(N), C.DB / C.B0, cfg.h
    cols = []
    for k in (kI, kQ):
        c = [(h**r / factorial(r)) * e**s * sum(alpha[s, q] * k[q + r] for q in range(s + 1))
             for r, s in monomials(N)]
        cols.append(np.stack(c, axis=1))  # (nb, n_monomials, J)
    st = C.B.to_stokes(cols[0], cols[1], cfg.azimuth_factor) * cfg.w_b[:, None, None, None]
    return np.transpose(st, (2, 3, 0, 1)).reshape(cfg.n_channels, 3, -1)


def keep_joint(first_cell: int, N: int = 2, n_bins: int = PAPER.n_bins) -> np.ndarray:
    """Column indices of the joint basis for cells first_cell, ..., n_bins - 1."""
    n = len(monomials(N))
    return np.arange(n * first_cell, n * n_bins)
