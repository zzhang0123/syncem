"""Channel-averaged continuum kernels Kbar_{X,j}(gamma) and the moment basis.

Pipeline
  1. KernelTable: field-strength-averaged kernels T_W^{(r)}(y) = <B W^{(r)}(y/B)>_B for
     W = F, G and r = 0..max_order, with y = nu/(a1 gamma^2 s), s = sqrt(1-eta^2) and B in
     muG; W^{(r)} = (-x d/dx)^r W.  Stored as cubic splines in ln y of ln T_W (positive)
     and of the signed ratios tau_r = T^{(r)}/T (no clamped logs anywhere).
  2. channel_kernels: eta-resolved channel kernels, with the screen inside the channel
     integral, as d^r/d(ln gamma)^r:
       KI_r(gamma, eta) = 2^r A1 s sum_t w_t T_F tau^F_r,
       KQ_r(gamma, eta) = -2^r A1 s sum_t w_t S(nu_t) T_G tau^G_r   (complex, no azimuth)
     in erg s^-1 Hz^-1 sr^-1 per electron (unit-area channel responses).
  3. kbar: eta average with weights q(eta) (isotropic pitch) or q(eta)(2l+1)P_l(eta)
     (pitch-Legendre column l, beaming limit 2p(mu=eta) = sum_l b_l P_l(eta)).
  4. Columns w_b D_{j;br}, D_{j;br} = (h_b^r/r!) d^r Kbar/d(ln gamma)^r at gamma_b
     (= (1/r!) d^r/dx_b^r Kbar(gamma_b e^{h_b x_b}) at x_b = 0).

Stokes layout everywhere: arrays (..., J, 3) with [I, Q, U]; P = Q + iU.
Column order: index b*(N+1) + r (equals the harmonic pipeline's 3b + r for N = 2).
"""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from math import factorial
from pathlib import Path

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.special import eval_legendre, logsumexp

import kernels as K
from config import Config

HERE = Path(__file__).resolve().parent
CACHE = HERE.parent / "cache"
TABLE_LNY = (np.log(1e-5), np.log(1e6))
TABLE_STEP = 0.0025
TABLE_ORDER = 6


class KernelTable:
    """Spline tables of ln T_W(y) and tau_{W,r}(y), W in {F, G}, r = 1..max_order."""

    def __init__(self, b_nodes, b_weights, max_order=TABLE_ORDER, lny_range=TABLE_LNY,
                 step=TABLE_STEP, nodes=K.DEFAULT_NODES, cutoff=K.DEFAULT_CUTOFF):
        self.max_order = int(max_order)
        self.b_nodes = np.asarray(b_nodes, float)
        self.b_weights = np.asarray(b_weights, float)
        n = int(round((lny_range[1] - lny_range[0]) / step)) + 1
        self.lny = np.linspace(lny_range[0], lny_range[1], n)
        self.x0, self.dx = self.lny[0], self.lny[1] - self.lny[0]
        self.coef = {}
        for which in ("F", "G"):
            ln_t, tau = self.direct(np.exp(self.lny), which, nodes=nodes, cutoff=cutoff)
            rows = [CubicSpline(self.lny, ln_t).c]
            rows += [CubicSpline(self.lny, tau[:, r]).c for r in range(1, self.max_order + 1)]
            self.coef[which] = np.stack(rows)  # (1+max_order, 4, n-1)

    def direct(self, y, which, nodes=K.DEFAULT_NODES, cutoff=K.DEFAULT_CUTOFF):
        """Quadrature (no spline): ln T_W(y), tau (n_y, max_order+1)."""
        y = np.atleast_1d(np.asarray(y, float))
        x = y[:, None] / self.b_nodes[None, :]
        ln_e0, f = K.scaled_log_derivatives(x, self.max_order, which, nodes=nodes, cutoff=cutoff)
        terms = np.log(self.b_weights)[None, :] - x + ln_e0  # ln(w_q e^{-x} E_0); T = y sum_q
        lse = logsumexp(terms, axis=1)
        soft = np.exp(terms - lse[:, None])
        tau = np.einsum("yq,yqr->yr", soft, f)
        return np.log(y) + lse, tau

    def evaluate(self, lny, which, order):
        """T^{(r)}(y) for r = 0..order at ln y (any shape); list of arrays.  T = 0 above table."""
        if order > self.max_order:
            raise ValueError("order exceeds table max_order")
        lny = np.asarray(lny, float)
        if np.any(lny < self.lny[0]):
            raise ValueError("y below kernel-table range")
        hi = lny > self.lny[-1]
        pos = np.clip((lny - self.x0) / self.dx, 0.0, len(self.lny) - 1 - 1e-9)
        i = pos.astype(np.int64)
        t = (pos - i) * self.dx
        c = self.coef[which]

        def horner(k):
            ck = c[k]
            return ((ck[0][i] * t + ck[1][i]) * t + ck[2][i]) * t + ck[3][i]

        tval = np.exp(horner(0))
        tval[hi] = 0.0
        out = [tval]
        for r in range(1, order + 1):
            out.append(tval * horner(r))
        return out

    def spline_check(self, which, n_points=4001, order=None, seed=1):
        """Max relative spline error of T^{(r)} at random interior points (vs direct)."""
        order = self.max_order if order is None else order
        rng = np.random.default_rng(seed)
        lny = np.sort(rng.uniform(self.lny[0], np.log(5e4), n_points))
        ln_t, tau = self.direct(np.exp(lny), which)
        keep = ln_t > -650.0  # T representable in double precision
        lny, ln_t, tau = lny[keep], ln_t[keep], tau[keep]
        ev = self.evaluate(lny, which, order)
        out = {"n_points": int(keep.sum()), "y_max_checked": float(np.exp(lny.max())),
               "ln_T_max_abs": float(np.max(np.abs(np.log(ev[0]) - ln_t)))}
        for r in range(1, order + 1):
            ref = np.exp(ln_t) * tau[:, r]
            scale = np.exp(ln_t) * np.maximum(1.0, np.abs(tau[:, r]))
            out[f"r{r}_max_err_over_T_max(1,|tau|)"] = float(np.max(np.abs(ev[r] - ref) / scale))
        return out


def _table_key(cfg: Config, max_order: int) -> tuple:
    return (cfg.b_kind, cfg.b_lo, cfg.b_hi, cfg.b_rms, cfg.n_b, cfg.maxwell_cut, max_order)


@lru_cache(maxsize=8)
def _cached_table(key: tuple) -> KernelTable:
    b_kind, b_lo, b_hi, b_rms, n_b, cut, max_order = key
    from config import PAPER
    from dataclasses import replace

    cfg = replace(PAPER, b_kind=b_kind, b_lo=b_lo, b_hi=b_hi, b_rms=b_rms, n_b=n_b, maxwell_cut=cut)
    src = hashlib.sha256((HERE / "kernels.py").read_bytes() + (HERE / "basis.py").read_bytes()).hexdigest()[:16]
    name = "table_" + hashlib.sha256(repr(key).encode()).hexdigest()[:16] + f"_{src}.npz"
    path = CACHE / name
    b, w = cfg.b_nodes_weights
    if path.exists():
        d = np.load(path)
        tab = KernelTable.__new__(KernelTable)
        tab.max_order, tab.b_nodes, tab.b_weights = int(d["max_order"]), d["b_nodes"], d["b_weights"]
        tab.lny = d["lny"]
        tab.x0, tab.dx = tab.lny[0], tab.lny[1] - tab.lny[0]
        tab.coef = {"F": d["coefF"], "G": d["coefG"]}
        return tab
    tab = KernelTable(b, w, max_order=max_order)
    CACHE.mkdir(exist_ok=True)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, max_order=tab.max_order, b_nodes=tab.b_nodes, b_weights=tab.b_weights,
             lny=tab.lny, coefF=tab.coef["F"], coefG=tab.coef["G"])
    os.replace(tmp, path)
    return tab


def get_table(cfg: Config, max_order: int = TABLE_ORDER) -> KernelTable:
    """Kernel table for cfg's field-strength distribution (memory + disk cached)."""
    return _cached_table(_table_key(cfg, max_order))


# ---------------------------------------------------------------------------- kernels
def channel_kernels(cfg: Config, gammas, eta_s=None, order=0, table=None, chunk_points=3_000_000):
    """eta-resolved channel kernels and ln-gamma derivatives.

    Returns KI (order+1, n_g, n_e, J) real and KQ (order+1, n_g, n_e, J) complex, where
    KQ includes the screen but not the azimuth factor; r-th entry = d^r/d(ln gamma)^r.
    eta_s: values of sqrt(1-eta^2) (default cfg.eta_s).
    """
    table = get_table(cfg) if table is None else table
    gammas = np.atleast_1d(np.asarray(gammas, float))
    eta_s = cfg.eta_s if eta_s is None else np.atleast_1d(np.asarray(eta_s, float))
    lnnu = np.log(cfg.chan_nu / K.a1)  # (J, T)
    wt = cfg.chan_w
    wq = wt[None, :] * cfg.screen  # (J, T)
    ng, ne = gammas.size, eta_s.size
    J = lnnu.shape[0]
    KI = np.zeros((order + 1, ng, ne, J))
    KQ = np.zeros((order + 1, ng, ne, J), complex)
    per = max(1, int(chunk_points // (ne * lnnu.size)))
    pref = K.A1 * eta_s  # (n_e,)
    for s in range(0, ng, per):
        g = gammas[s:s + per]
        lny = lnnu[None, None] - 2.0 * np.log(g)[:, None, None, None] - np.log(eta_s)[None, :, None, None]
        tf = table.evaluate(lny, "F", order)
        tg = table.evaluate(lny, "G", order)
        for r in range(order + 1):
            fac = 2.0**r * pref[None, :, None]
            KI[r, s:s + per] = fac * np.einsum("gejt,t->gej", tf[r], wt)
            KQ[r, s:s + per] = -fac * (np.einsum("gejt,jt->gej", tg[r], wq.real)
                                       + 1j * np.einsum("gejt,jt->gej", tg[r], wq.imag))
    return KI, KQ


def eta_weights(cfg: Config, pitch_l: int | None = None) -> np.ndarray:
    """eta weights for the isotropic average (pitch_l None) or pitch-Legendre column l."""
    e, w = cfg.eta_nodes_weights
    if pitch_l is None:
        return w
    return w * (2 * pitch_l + 1) * eval_legendre(pitch_l, e)


def kbar(cfg: Config, gammas, order=0, weights=None, pitch_l=None, table=None):
    """eta-averaged channel kernels: (KIbar, KQbar) with shapes (order+1, n_g, J).

    weights: (n_e,) or (n_g, n_e) eta weights; default eta_weights(cfg, pitch_l).
    KQbar excludes the azimuth factor (multiply by cfg.azimuth_factor for separable P).
    """
    KI, KQ = channel_kernels(cfg, gammas, order=order, table=table)
    w = eta_weights(cfg, pitch_l) if weights is None else np.asarray(weights)
    if w.ndim == 1:
        return np.einsum("rgej,e->rgj", KI, w), np.einsum("rgej,e->rgj", KQ, w)
    return np.einsum("rgej,ge->rgj", KI, w), np.einsum("rgej,ge->rgj", KQ, w)


def to_stokes(kI, kQ, factor):
    """[I, Re(factor kQ), Im(factor kQ)] stacked on a new last axis."""
    p = factor * kQ
    return np.stack([kI, p.real, p.imag], axis=-1)


# ---------------------------------------------------------------------------- basis
def derivative_spectra(cfg: Config, N: int, pitch_l=None, table=None):
    """D^I (n_bins, N+1, J) and D^Q (n_bins, N+1, J) complex (no azimuth factor).

    D_{j;br} = (1/r!) d^r/dx_b^r Kbar(gamma_b e^{h x_b}) at x_b = 0, analytic derivatives.
    """
    kI, kQ = kbar(cfg, cfg.gamma_b, order=N, pitch_l=pitch_l, table=table)  # (N+1, nb, J)
    fac = np.array([cfg.h**r / factorial(r) for r in range(N + 1)])
    DI = np.transpose(kI, (1, 0, 2)) * fac[None, :, None]
    DQ = np.transpose(kQ, (1, 0, 2)) * fac[None, :, None]
    return DI, DQ


def stencil_derivative_spectra(cfg: Config, step=0.04, pitch_l=None, table=None):
    """D_0, D_1, D_2 by the paper's 5-point stencil in x_b (step 0.04), for cross-checks."""
    offs = np.arange(-2, 3)
    g = (cfg.gamma_b[:, None] * np.exp(cfg.h * step * offs[None, :])).ravel()
    kI, kQ = kbar(cfg, g, order=0, pitch_l=pitch_l, table=table)
    out = []
    for k in (kI[0], kQ[0]):
        v = k.reshape(cfg.n_bins, 5, -1)
        d0 = v[:, 2]
        d1 = (v[:, 0] - 8 * v[:, 1] + 8 * v[:, 3] - v[:, 4]) / (12 * step)
        d2 = (-v[:, 0] + 16 * v[:, 1] - 30 * v[:, 2] + 16 * v[:, 3] - v[:, 4]) / (12 * step**2) / 2
        out.append(np.stack([d0, d1, d2], axis=1))
    return out[0], out[1]


def separable_columns(cfg: Config, DI, DQ, factor=None):
    """(J, 3, n_bins*(N+1)) columns w_b D with P = factor * D^Q (default azimuth factor)."""
    factor = cfg.azimuth_factor if factor is None else factor
    nb, n1, J = DI.shape
    st = to_stokes(DI, DQ, factor)  # (nb, n1, J, 3)
    st = st * cfg.w_b[:, None, None, None]
    return np.transpose(st, (2, 3, 0, 1)).reshape(J, 3, nb * n1)


def joint_columns(cfg: Config, DI, DQ):
    """(J, 3, 3*n) columns for unknowns [a_br (n), Re M2_br (n), Im M2_br (n)].

    I rows: w_b D^I a; P rows: sum w_b D^Q M2 with M2 = <e^{2i phi} x_b^r>_bin / w_b.
    """
    nb, n1, J = DI.shape
    n = nb * n1
    wI = (DI * cfg.w_b[:, None, None]).transpose(2, 0, 1).reshape(J, n)
    wQ = (DQ * cfg.w_b[:, None, None]).transpose(2, 0, 1).reshape(J, n)
    C = np.zeros((J, 3, 3 * n))
    C[:, 0, :n] = wI
    C[:, 1, n:2 * n], C[:, 1, 2 * n:] = wQ.real, -wQ.imag
    C[:, 2, n:2 * n], C[:, 2, 2 * n:] = wQ.imag, wQ.real
    return C


def q_columns(cfg: Config, DQ):
    """(J, n) complex Q-kernel columns w_b D^Q (screen included, no azimuth factor)."""
    nb, n1, J = DQ.shape
    return (DQ * cfg.w_b[:, None, None]).transpose(2, 0, 1).reshape(J, nb * n1)


def build_basis(cfg: Config, N=2, variant="separable", pitch_l=None, table=None):
    """Convenience: columns for variant in {'separable', 'joint', 'q'}."""
    DI, DQ = derivative_spectra(cfg, N, pitch_l=pitch_l, table=table)
    if variant == "separable":
        return separable_columns(cfg, DI, DQ)
    if variant == "joint":
        return joint_columns(cfg, DI, DQ)
    if variant == "q":
        return q_columns(cfg, DQ)
    raise ValueError(variant)


def predict(C, a):
    """S = C a for C (J, 3, n) and a (n,) or (12, N+1); returns (J, 3)."""
    return np.einsum("jsk,k->js", C, np.asarray(a).ravel())


def bin_derivative_grid(cfg: Config, order: int, n_grid=65, pitch_l=None, table=None):
    """d^order/dx_b^order of Kbar on a grid x_b in [-1, 1] for every bin.

    Returns (x grid, dI (n_bins, n_grid, J), dQ (n_bins, n_grid, J) complex, no azimuth).
    """
    x = np.linspace(-1.0, 1.0, n_grid)
    g = (cfg.gamma_b[:, None] * np.exp(cfg.h * x[None, :])).ravel()
    kI, kQ = kbar(cfg, g, order=order, pitch_l=pitch_l, table=table)
    sc = cfg.h**order
    return x, sc * kI[order].reshape(cfg.n_bins, n_grid, -1), sc * kQ[order].reshape(cfg.n_bins, n_grid, -1)
