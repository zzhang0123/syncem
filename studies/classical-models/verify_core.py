"""Independent verifier for the core part (regression.py and forward.py).

Imports none of the author's analysis code (lib/config.py, lib/basis.py, lib/ensembles.py,
regression.py, forward.py, core_checks.py).  lib/kernels.py is imported only to spot-check
its F and G at five points against scipy kv + quad; it is not used for any number below.

Independent route:
  * F(x) = x int_x^inf K_{5/3} by a backward panel recursion on scaled kve(5/3), spline in ln x;
  * (x d/dx)^r F and G by a symbolic Bessel recursion
      D F = F - x^2 K_{5/3},  D[x^a K_n] = a x^a K_n - (x^{a+1}/2)(K_{n-1} + K_{n+1}),
    evaluated with scipy kve (the author uses an integral representation and a polynomial
    recursion in z = x cosh u);
  * B, eta, channel and energy quadratures with node counts different from the author's;
  * cut-off panels by the substitution 1 - gamma/gamma_c = w^5 (author: Gauss-Jacobi / QAWS).
Writes verify_core_results.json next to this script.  Run: python verify_core.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy import integrate, special
from scipy.interpolate import CubicSpline
from scipy.special import eval_legendre, kve, logsumexp

T0 = time.time()
HERE = Path(__file__).resolve().parent
OUT = HERE / "verify_core_results.json"
HARM = HERE / "inputs" / "harmonic-radio-fit" / "trusted_truth_refined.npz"

# ----------------------------------------------------------------------------- constants (cgs)
E_ESU = 1.602176634e-19 * 2.99792458e9
M_E = 9.1093837139e-28
C_CGS = 2.99792458e10
C_SI = 2.99792458e8
SIGMA_T = 6.6524587321e-25
EV = 1.602176634e-12
MYR = 3.15576e13
A1 = np.sqrt(3.0) * E_ESU**3 * 1e-6 / (4 * np.pi * M_E * C_CGS**2)  # A_theta per muG
a1 = 3.0 * E_ESU * 1e-6 / (4 * np.pi * M_E * C_CGS)  # a_B per muG [Hz]

# ----------------------------------------------------------------------------- paper configuration
NU = np.geomspace(1e8, 1e10, 41)
FIT = np.arange(0, 41, 2)
HELD = np.arange(1, 41, 2)
G_LO, G_HI, NBIN = 300.0, 1e5, 12
LEDGE = np.linspace(np.log(G_LO), np.log(G_HI), NBIN + 1)
EDGE = np.exp(LEDGE)
H = 0.5 * (LEDGE[1] - LEDGE[0])
UB = 0.5 * (LEDGE[1:] + LEDGE[:-1])
GB = np.exp(UB)
WB = np.diff(EDGE**-1.6) / -1.6
WB = WB / WB.sum()
ZETA = 0.4 * np.exp(0.5j)
SEED = 20260928
N_T, N_BU, N_ETA, N_E = 300, 20, 16, 40  # author: 400, 16, 12, 24


def rho1(t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    ok = np.abs(t) < 1
    out[ok] = np.exp(-t[ok] ** 2 / (1 - t[ok] ** 2))
    return out


Z1 = integrate.quad(lambda t: np.exp(-t * t / (1 - t * t)), -1, 1, epsabs=0, epsrel=1e-13, limit=500)[0]


def gl(n, lo, hi):
    t, w = leggauss(n)
    return 0.5 * (hi - lo) * t + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


def channels(n_t=N_T):
    t, w = leggauss(n_t)
    W = w * rho1(t) / Z1
    nu = NU[:, None] * (1 + 0.2 * t[None, :])
    lam2 = (C_SI / nu) ** 2
    scr = np.exp(2j * lam2 * 8.0 - 2 * lam2**2 * 0.25)
    return nu, W, scr


# ----------------------------------------------------------------------------- F table
def build_F_table():
    xs = [np.geomspace(1e-9, 62.5, int(np.log(62.5e9) / 0.004) + 1)]
    xs.append(np.arange(62.5, 2.0e4, 0.25)[1:])
    x = np.concatenate(xs)
    tq, wq = leggauss(10)
    a, b = x[:-1], x[1:]
    tt = 0.5 * (b - a)[:, None] * (tq[None] + 1) + a[:, None]
    I = (0.5 * (b - a)[:, None] * wq[None] * kve(5 / 3, tt) * np.exp(-(tt - a[:, None]))).sum(1)
    S = np.empty_like(x)
    S[-1] = kve(5 / 3, x[-1]) * (1 - 0.5 / x[-1])
    dec = np.exp(-(b - a))
    for k in range(len(x) - 2, -1, -1):
        S[k] = I[k] + dec[k] * S[k + 1]
    return CubicSpline(np.log(x), np.log(x * S)), x  # ln Ftilde(x), Ftilde = F e^x


LNFT, XGRID = build_F_table()


def F_scaled(x):
    """Ftilde(x) = F(x) e^x (valid for 1e-9 <= x <= 2e4)."""
    return np.exp(LNFT(np.log(x)))


def F_quad(x):
    return x * integrate.quad(lambda t: special.kv(5 / 3, t), x, np.inf, epsabs=0, epsrel=1e-13, limit=400)[0]


def D(expr):
    out = {}

    def add(k, c):
        out[k] = out.get(k, 0.0) + c

    for k, c in expr.items():
        if k == "F":
            add("F", c)
            add((2, 5), -c)
        else:
            a, n3 = k
            if a:
                add((a, n3), a * c)
            add((a + 1, abs(n3 - 3)), -0.5 * c)
            add((a + 1, n3 + 3), -0.5 * c)
    return out


def exprs(which, rmax):
    e = [{"F": 1.0} if which == "F" else {(1, 2): 1.0}]
    for _ in range(rmax):
        e.append(D(e[-1]))
    return e


EXPR = {w: exprs(w, 5) for w in "FG"}


def DrW_scaled(x, which, r):
    """e^x (x d/dx)^r W(x)."""
    tot = np.zeros_like(x)
    for k, c in EXPR[which][r].items():
        if k == "F":
            tot += c * F_scaled(x)
        else:
            a, n3 = k
            tot += c * x**a * kve(n3 / 3, x)
    return tot


class PhiTable:
    """Phi_r(y) = sum_i c_i (D^r W)(y/s_i) for a discrete s-distribution (c_i = weight * s_i)."""

    def __init__(self, s, c, rmax, ly=(np.log(1e-4), np.log(4e4)), dl=0.004):
        self.ly = np.arange(ly[0], ly[1] + dl, dl)
        self.rmax = rmax
        y = np.exp(self.ly)
        self.sp = {}
        for which in "FG":
            lnphi = np.empty(y.size)
            rat = np.empty((rmax + 1, y.size))
            for s0 in range(0, y.size, 400):
                yy = y[s0:s0 + 400, None]
                x = yy / s[None]
                big = x > 2.0e4
                xs = np.where(big, 1.0, x)
                w0 = DrW_scaled(xs, which, 0)
                lw = np.log(c)[None] + np.log(w0) - x
                lw[big] = -np.inf
                m = lw.max(1, keepdims=True)
                om = np.exp(np.log(c)[None] - x - m)
                om[big] = 0.0
                den = (om * w0).sum(1)
                lnphi[s0:s0 + 400] = m[:, 0] + np.log(den)
                for r in range(rmax + 1):
                    rat[r, s0:s0 + 400] = (om * DrW_scaled(xs, which, r)).sum(1) / den
            self.sp[which] = (CubicSpline(self.ly, lnphi), [CubicSpline(self.ly, rat[r]) for r in range(rmax + 1)])

    def __call__(self, y, which, r=0):
        lnp, rat = self.sp[which]
        ly = np.log(y)
        out = np.exp(lnp(ly)) * (rat[r](ly) if r else 1.0)
        return np.where(ly > self.ly[-1], 0.0, out)


def sdist_uniform(etas, weta, n_b=N_BU):
    b, wb = gl(n_b, 4.0, 6.0)
    wb = wb / 2.0
    s = (b[:, None] * np.sqrt(1 - etas[None] ** 2)).ravel()
    w = (wb[:, None] * weta[None]).ravel()
    return s, w * s


def sdist_maxwell(etas, weta, n_b=160, bmax=35.0):
    sig = 5.0 / np.sqrt(3.0)
    b, wb = gl(n_b, 0.0, bmax)
    pdf = np.sqrt(2 / np.pi) * b**2 / sig**3 * np.exp(-b * b / (2 * sig * sig))
    wb = wb * pdf
    s = (b[:, None] * np.sqrt(1 - etas[None] ** 2)).ravel()
    w = (wb[:, None] * weta[None]).ravel()
    return s, w * s, float(wb.sum())


# ----------------------------------------------------------------------------- channel kernels, basis
CH = channels()


def kbar(table, gam, r=0, ch=CH, screen=True):
    """(K,41) I and P-kernel (-G, with screen, no azimuth) channel kernels at gamma (order r in D)."""
    nu, W, scr = ch
    y = nu[None] / (a1 * np.asarray(gam)[:, None, None] ** 2)
    kI = A1 * (table(y, "F", r) * W).sum(-1)
    kP = -A1 * (table(y, "G", r) * W * (scr if screen else 1.0)).sum(-1)
    return kI, kP


def basis(table, N, ch=CH):
    """Columns (41, 2, nb*(N+1)): component 0 = I (real), 1 = P-kernel (complex), unit azimuth."""
    CI = np.zeros((41, NBIN, N + 1))
    CP = np.zeros((41, NBIN, N + 1), complex)
    for r in range(N + 1):
        kI, kP = kbar(table, GB, r, ch)
        f = (-2 * H) ** r / special.factorial(r)
        CI[:, :, r] = (WB[:, None] * f * kI).T
        CP[:, :, r] = (WB[:, None] * f * kP).T
    return CI.reshape(41, -1), CP.reshape(41, -1)


def stokes(sI, sP, zeta):
    P = zeta * sP
    return np.stack([sI, P.real, P.imag], -1)


def design(CI, CP, zeta):
    return np.stack([CI, (zeta * CP).real, (zeta * CP).imag], 1)  # (41, 3, ncol)


# ----------------------------------------------------------------------------- energy measures
def measure(dens, cut=None, n=N_E, nw=60):
    """Bin-aligned nodes (gamma, mass, bin) for density dens(g) on [300, min(1e5, cut))."""
    g_all, m_all, b_all = [], [], []
    for b in range(NBIN):
        lo, hi = EDGE[b], EDGE[b + 1]
        if cut is not None and lo >= cut:
            continue
        if cut is not None and hi > cut:  # 1 - g/cut = w^5
            wmax = (1 - lo / cut) ** 0.2
            w, ww = gl(nw, 0.0, wmax)
            g = cut * (1 - w**5)
            m = dens(g) * 5 * cut * w**4 * ww
        else:
            u, wu = gl(n, np.log(lo), np.log(hi))
            g = np.exp(u)
            m = dens(g) * g * wu
        g_all.append(g)
        m_all.append(m)
        b_all.append(np.full(g.size, b))
    return np.concatenate(g_all), np.concatenate(m_all), np.concatenate(b_all)


def moments(meas, N):
    g, m, b = meas
    x = (np.log(g) - UB[b]) / H
    a = np.zeros((NBIN, N + 1))
    for r in range(N + 1):
        np.add.at(a[:, r], b, m * x**r)
    return a / WB[:, None]


def direct(table, meas, ch=CH, screen=True):
    g, m, _ = meas
    kI, kP = kbar(table, g, 0, ch, screen)
    return m @ kI, m @ kP


def maxres(S, C, a):
    pred = np.einsum("jsk,k->js", C, a.ravel())
    return np.max(np.abs(S - pred) / S[:, :1], axis=0)


# ----------------------------------------------------------------------------- run
def main():
    res = {"description": __doc__, "checks": {}, "values": {}}
    chk, val = res["checks"], res["values"]

    # --- spot checks of F, G (mine and lib/kernels.py) against scipy kv + quad
    xs5 = np.array([1e-3, 0.1, 1.0, 5.0, 20.0])
    fq = np.array([F_quad(x) for x in xs5])
    gq = xs5 * special.kv(2 / 3, xs5)
    sys.path.insert(0, str(HERE / "lib"))
    import kernels as LK  # noqa: E402  (spot check only)
    chk["F_mine_vs_quad_rel"] = (F_scaled(xs5) * np.exp(-xs5) / fq - 1).tolist()
    chk["F_libkernels_vs_quad_rel"] = (LK.F(xs5) / fq - 1).tolist()
    chk["G_libkernels_vs_kv_rel"] = (LK.G(xs5) / gq - 1).tolist()
    # symbolic derivative recursion vs central differences of the order below
    xx = np.geomspace(1e-3, 50, 9)
    dl = 1e-4
    rec = {}
    for w in "FG":
        errs = []
        for r in range(1, 5):
            fd = (DrW_scaled(xx * np.exp(dl), w, r - 1) * np.exp(-xx * np.exp(dl))
                  - DrW_scaled(xx * np.exp(-dl), w, r - 1) * np.exp(-xx * np.exp(-dl))) / (2 * dl)
            an = DrW_scaled(xx, w, r) * np.exp(-xx)
            errs.append(float(np.max(np.abs(fd - an) / (np.abs(an) + 1e-3 * np.max(np.abs(an))))))
        rec[w] = errs
    chk["D^r_recursion_vs_central_difference_rel(r=1..4)"] = rec

    # --- tables
    e_iso, we_iso = gl(N_ETA, 0.0, 0.5)
    we_iso = we_iso / 0.5
    t0 = time.time()
    TAB = PhiTable(*sdist_uniform(e_iso, we_iso), rmax=4)
    val["time_iso_table_s"] = time.time() - t0
    sM, cM, massM = sdist_maxwell(e_iso, we_iso)
    TABM = PhiTable(sM, cM, rmax=0)
    chk["maxwell_mass"] = massM
    # Quadrature self-check: coarser channel/B/eta nodes
    e_c, we_c = gl(10, 0.0, 0.5)
    TABc = PhiTable(*sdist_uniform(e_c, we_c / 0.5, n_b=12), rmax=2)
    CHc = channels(200)

    # --- E1 and the regression
    p = 2.6
    K1 = (p - 1) / (G_LO ** (1 - p) - G_HI ** (1 - p))
    m1 = measure(lambda g: K1 * g**-p)
    chk["E1_column"] = float(m1[1].sum())
    a1m = moments(m1, 4)
    sI, sP = direct(TAB, m1)
    S1 = stokes(sI, sP, ZETA)
    sIc, sPc = direct(TABc, measure(lambda g: K1 * g**-p, n=24), CHc)
    chk["E1_direct_coarse_quadrature_change_over_I"] = np.max(np.abs(stokes(sIc, sPc, ZETA) - S1) / S1[:, :1], 0).tolist()
    if HARM.exists():
        Sh = np.load(HARM)["truth"]
        chk["E1_mine_minus_harmonic_truth_over_I_IQU"] = np.max(np.abs(S1 - Sh) / Sh[:, :1], 0).tolist()
    auth = json.load(open(HERE / "forward_results.json"))
    ADS = {k: np.array(v) for k, v in auth["direct_spectra_IQU_per_unit_column"].items()}
    chk["E1_mine_minus_author_direct_over_I_IQU"] = np.max(np.abs(S1 - ADS["E1"]) / S1[:, :1], 0).tolist()

    CI2, CP2 = basis(TAB, 2)
    C2 = design(CI2, CP2, ZETA)
    CI2c, CP2c = basis(TABc, 2, CHc)
    chk["N2_columns_coarse_change_over_I"] = float(np.max(np.abs(design(CI2c, CP2c, ZETA) - C2) / S1[:, :1, None]))
    a2 = a1m[:, :3]
    sig = 0.01 * S1[:, :1] * np.ones((1, 3))
    Dm = (C2[FIT] / sig[FIT][..., None]).reshape(-1, C2.shape[-1])
    u, s, vt = np.linalg.svd(Dm, full_matrices=False)
    keep = s >= 10.0
    R = int(keep.sum())
    noise = np.random.default_rng(SEED).normal(size=(21, 3))
    d = S1[FIT] + noise * sig[FIT]
    beta = (u[:, keep].T @ (d / sig[FIT]).ravel()) / s[keep]
    ahat = vt[keep].T @ beta
    pred = np.einsum("jsk,k->js", C2, ahat)
    chi2 = float(np.sum(((d - pred[FIT]) / sig[FIT]) ** 2))
    eps = S1 - np.einsum("jsk,k->js", C2, a2.ravel())
    bias = u[:, keep].T @ (eps[FIT] / sig[FIT]).ravel()
    dev = np.abs(pred - S1) / S1[:, :1]
    jmax = int(np.unravel_index(np.argmax(dev), dev.shape)[0])
    dof = d.size - R
    val["regression"] = {
        "retained": R, "s7": float(s[6]), "s8": float(s[7]),
        "max_bias_sigma": float(np.max(np.abs(bias))), "bias_modes_sigma": bias.tolist(),
        "chi2_seeded": chi2, "dof": dof,
        "chi2_pvalue_upper": float(scipy.stats.chi2.sf(chi2, dof)) if hasattr(scipy, "stats") else None,
        "max_fit_minus_direct_over_I_41ch": float(dev.max()), "argmax_channel": jmax,
        "argmax_is_withheld": bool(jmax % 2 == 1),
        "max_finite_response_discrepancy_over_I": float(np.max(np.abs(eps) / S1[:, :1])),
        "beta_hat_minus_beta_true_over_sigma": ((beta - vt[keep] @ a2.ravel()) * s[keep]).tolist(),
    }

    # --- E1 analytic checks
    nu_c, Wc, _ = CH
    sIn, sPn = direct(TAB, m1, screen=False)
    Pi = -sPn.real / sIn  # azimuth factor 1, no screen
    val["E1_Pi_0.1GHz"] = float(Pi[0])
    val["E1_Pi_10GHz"] = float(Pi[-1])
    val["Pi_unbounded"] = (p + 1) / (p + 7 / 3)
    m = (p - 3) / 2
    MF = 2 ** (m + 1) / (m + 2) * special.gamma(m / 2 + 7 / 3) * special.gamma(m / 2 + 2 / 3)
    sU, cU = sdist_uniform(e_iso, we_iso)
    sfac = np.sum(cU * (a1 * sU) ** ((p - 1) / 2))  # sum w s (a1 s)^((p-1)/2)
    I_unb = A1 * K1 * 0.5 * MF * sfac * (nu_c ** ((1 - p) / 2) * Wc).sum(-1)
    val["E1_unbounded_RL_over_direct_minus1_max"] = float(np.max(I_unb / sI - 1))
    val["E1_direct_over_unbounded_at_10GHz"] = float(sI[-1] / I_unb[-1])

    # --- forward tests, iso basis N=2 and N=4
    CI4, CP4 = basis(TAB, 4)
    C4 = design(CI4, CP4, ZETA)
    fwd = {}
    fwd["E1_N2"] = maxres(S1, C2, a2).tolist()
    fwd["E1_N4"] = maxres(S1, C4, a1m).tolist()
    # E2
    p1, p2, ss, gbr = 2.0, 3.2, 2.0, 7800.0
    f2 = lambda g: (g / gbr) ** -p1 * (1 + (g / gbr) ** ss) ** (-(p2 - p1) / ss)
    m2u = measure(f2)
    m2 = (m2u[0], m2u[1] / m2u[1].sum(), m2u[2])
    S2 = stokes(*direct(TAB, m2), ZETA)
    fwd["E2_N2"] = maxres(S2, C2, moments(m2, 2)).tolist()
    chk["E2_mine_minus_author_direct_over_I_IQU"] = np.max(np.abs(S2 - ADS["E2"]) / S2[:, :1], 0).tolist()
    # E3
    f3 = lambda g: (g / 1e4) ** -2.6 * np.exp(0.2**2 * np.log(g / 1e4) ** 2 / 2)
    m3u = measure(f3)
    m3 = (m3u[0], m3u[1] / m3u[1].sum(), m3u[2])
    S3 = stokes(*direct(TAB, m3), ZETA)
    fwd["E3_N2"] = maxres(S3, C2, moments(m3, 2)).tolist()
    chk["E3_mine_minus_author_direct_over_I_IQU"] = np.max(np.abs(S3 - ADS["E3"]) / S3[:, :1], 0).tolist()

    # --- E4 ageing
    mean_b2 = (6.0**3 - 4.0**3) / (3 * 2.0)  # <B^2> for U[4,6]
    loss = 4 * SIGMA_T / (3 * M_E * C_CGS)
    b_s = loss * mean_b2 * 1e-12 / (8 * np.pi)
    b_c = loss * 1.0 * EV
    t = 19.0 * MYR
    gc_jp = 1 / ((b_s + b_c) * t)
    bkp = lambda mu: 1.5 * b_s * (1 - mu**2) + b_c
    gc_kp = lambda mu: 1 / (bkp(mu) * t)
    mean_bperp = 5.0 * np.sum(we_iso * np.sqrt(1 - e_iso**2))
    val["ageing"] = {"mean_B2": mean_b2, "b_s": b_s, "b_c": b_c, "gamma_c_JP": gc_jp,
                     "nu_c_JP_GHz": a1 * mean_bperp * gc_jp**2 / 1e9,
                     "gamma_c_KP(0,0.5,1)": [gc_kp(0.0), gc_kp(0.5), gc_kp(1.0)],
                     "step_if_injection_cut_at_1e5": 1e5 / (1 + 1e5 * (b_s + b_c) * t)}
    alpha = p - 2
    fjp = lambda g: g**-p * np.clip(1 - g / gc_jp, 0, None) ** alpha
    mju = measure(fjp, cut=gc_jp)
    mj = (mju[0], mju[1] / mju[1].sum(), mju[2])
    SJ = stokes(*direct(TAB, mj), ZETA)
    fwd["E4JP_N2"] = maxres(SJ, C2, moments(mj, 2)).tolist()
    fwd["E4JP_N4"] = maxres(SJ, C4, moments(mj, 4)).tolist()
    chk["E4JP_mine_minus_author_direct_over_I_IQU"] = np.max(np.abs(SJ - ADS["E4-JP"]) / SJ[:, :1], 0).tolist()

    # --- E4 KP: N(g,mu) = (K/2) g^-p (1 - b(mu) g t)^alpha, int_{-1}^{1} dmu int dg N = 1
    def kp_meas(mu, scale=1.0):
        gc = gc_kp(mu)
        return measure(lambda g: scale * g**-p * np.clip(1 - g / gc, 0, None) ** alpha, cut=gc)

    mu_e = np.sqrt(1 - (1 / (EDGE[9] * t) - b_c) / (1.5 * b_s))  # gamma_c(mu_e) = edge 9
    mus, wmu = [], []
    for lo, hi in ((0.0, mu_e), (mu_e, 1.0)):
        mm, ww = gl(32, lo, hi)
        mus.append(mm)
        wmu.append(ww)
    mus, wmu = np.concatenate(mus), np.concatenate(wmu)
    tot = sum(w_ * kp_meas(mu)[1].sum() for mu, w_ in zip(mus, wmu))  # int_0^1 dmu int g^-p(...)
    Kkp = 1.0 / tot  # (K/2) * 2 int_0^1 = K int_0^1
    eta_k, wet_k = gl(24, 0.0, 0.5)
    wet_k = wet_k / 0.5
    sI_kp = np.zeros(41)
    sP_kp = np.zeros(41, complex)
    a_eff = np.zeros((NBIN, 5))
    col_eff = 0.0
    tabs_eta = []
    for e, w_ in zip(eta_k, wet_k):
        tb = PhiTable(*sdist_uniform(np.array([e]), np.array([1.0])), rmax=4)
        tabs_eta.append(tb)
        mk = kp_meas(e, scale=Kkp)  # 2 N(g, eta) = K g^-p (...)
        i_, p_ = direct(tb, mk)
        sI_kp += w_ * i_
        sP_kp += w_ * p_
        a_eff += w_ * moments(mk, 4)
        col_eff += w_ * mk[1].sum()
    SK = stokes(sI_kp, sP_kp, ZETA)
    val["KP_effective_column"] = float(col_eff)
    chk["E4KP_mine_minus_author_direct_over_I_IQU"] = np.max(np.abs(SK - ADS["E4-KP"]) / SK[:, :1], 0).tolist()
    for N_, C_ in ((2, C2), (4, C4)):
        fwd[f"E4KP_eff_iso_N{N_}"] = maxres(SK, C_, a_eff[:, :N_ + 1]).tolist()
    fwd["E4KP_eff_iso_N3"] = maxres(SK, design(*[c.reshape(41, NBIN, 5)[:, :, :4].reshape(41, -1)
                                                for c in (CI4, CP4)], ZETA), a_eff[:, :4]).tolist()
    # pitch-Legendre columns l = 0..L and moments of b_l(g) = int_{-1}^{1} P_l N dmu (N = 4)
    ls = [0, 2, 4, 6, 8]
    cols_eta = [basis(tb, 4) for tb in tabs_eta]
    al = {l: np.zeros((NBIN, 5)) for l in ls}
    for mu, w_ in zip(mus, wmu):
        mk = kp_meas(mu, scale=Kkp)  # = 2 N(g, mu); 2 int_0^1 dmu N P_l = int_0^1 dmu (2N) P_l
        a_ = moments(mk, 4)
        for l in ls:
            al[l] += w_ * eval_legendre(l, mu) * a_
    leg = {}
    for L in (0, 2, 4, 8):
        pred = np.zeros((41, 3))
        for l in [x for x in ls if x <= L]:
            CIl = sum(w_ * (2 * l + 1) * eval_legendre(l, e) * c[0] for e, w_, c in zip(eta_k, wet_k, cols_eta))
            CPl = sum(w_ * (2 * l + 1) * eval_legendre(l, e) * c[1] for e, w_, c in zip(eta_k, wet_k, cols_eta))
            pred += np.einsum("jsk,k->js", design(CIl, CPl, ZETA), al[l].ravel())
        leg[f"L{L}"] = np.max(np.abs(SK - pred) / SK[:, :1], 0).tolist()
    fwd["E4KP_pitch_legendre_N4"] = leg
    fwd["E4KP_marginal_L0_N2"] = maxres(SK, C2, al[0][:, :3]).tolist()

    # --- E5
    def pl(pp):
        Kp = (pp - 1) / (G_LO ** (1 - pp) - G_HI ** (1 - pp))
        return measure(lambda g: Kp * g**-pp)

    mA, mB = pl(3.1), pl(2.1)
    iA, pA = direct(TAB, mA)
    iB, pB = direct(TAB, mB)
    fA = iB[20] / (iA[20] + iB[20])
    fB = 1 - fA
    zA, zB = ZETA, 0.4 * np.exp(2j * (0.25 + np.pi / 4))
    S5 = np.stack([fA * iA + fB * iB, (fA * zA * pA + fB * zB * pB).real, (fA * zA * pA + fB * zB * pB).imag], -1)
    chk["E5_mine_minus_author_direct_over_I_IQU"] = np.max(np.abs(S5 - ADS["E5"]) / S5[:, :1], 0).tolist()
    aA, aB = moments(mA, 2), moments(mB, 2)
    a5 = fA * aA + fB * aB
    M2 = fA * zA * aA + fB * zB * aB
    pred_joint = np.stack([CI2 @ a5.ravel(), (CP2 @ M2.ravel()).real, (CP2 @ M2.ravel()).imag], -1)
    fwd["E5_joint_N2"] = np.max(np.abs(S5 - pred_joint) / S5[:, :1], 0).tolist()
    fwd["E5_separable_N2"] = maxres(S5, C2, a5).tolist()
    sepP = ZETA * (CP2 @ a5.ravel())
    fwd["E5_separable_N2_absP_over_I"] = float(np.max(np.abs(S5[:, 1] + 1j * S5[:, 2] - sepP) / S5[:, 0]))
    val["E5_fA_fB"] = [float(fA), float(fB)]
    lam2c = (C_SI / NU) ** 2
    P5 = S5[:, 1] + 1j * S5[:, 2]
    sel = np.abs(P5) / S5[:, 0] > 0.03
    chi_d = np.degrees(0.5 * np.unwrap(np.angle(P5[sel] * np.exp(-2j * lam2c[sel] * 8.0))))
    val["E5_chi_derot_swing_deg"] = float(chi_d.max() - chi_d.min())

    # --- E1M: Maxwellian direct, U[4,6] basis with E1 moments
    S1M = stokes(*direct(TABM, m1), ZETA)
    chk["E1M_mine_minus_author_direct_over_I_IQU"] = np.max(np.abs(S1M - ADS["E1M"]) / S1M[:, :1], 0).tolist()
    fwd["E1M_U46_N2"] = maxres(S1M, C2, a2).tolist()
    fwd["E1M_U46_N1"] = maxres(S1M, design(*[c.reshape(41, NBIN, 3)[:, :, :2].reshape(41, -1)
                                            for c in (CI2, CP2)], ZETA), a1m[:, :2]).tolist()
    # --- is the KP floor a property of the isotropic basis or of the declared truth?
    # Fit an isotropic-pitch JP spectrum (free A, p, gamma_c) to the KP direct spectrum,
    # then evaluate the iso basis at that JP's moments.
    from scipy.optimize import least_squares

    def jp_direct(th):
        lnA, pp, lgc = th
        gc = np.exp(lgc)
        mm = measure(lambda g: np.exp(lnA) * (g / 1e4) ** -pp * np.clip(1 - g / gc, 0, None) ** (pp - 2), cut=gc)
        return stokes(*direct(TAB, mm), ZETA), mm

    sigK = 0.01 * SK[:, :1]
    # amplitude start: match I at 1 GHz with p = 2.6 and gamma_c = gamma_c,JP
    Sj0, _ = jp_direct(np.array([0.0, 2.6, np.log(gc_jp)]))
    th0 = np.array([np.log(SK[20, 0] / Sj0[20, 0]), 2.6, np.log(gc_jp)])
    fit = least_squares(lambda th: ((jp_direct(th)[0] - SK) / sigK).ravel(), th0, x_scale=[0.01, 0.01, 0.01],
                        xtol=1e-12, ftol=1e-14, gtol=1e-14)
    SJf, mJf = jp_direct(fit.x)
    val["KP_isotropic_JP_fit"] = {
        "theta_lnA_p_lngc": fit.x.tolist(), "gamma_c_eff": float(np.exp(fit.x[2])),
        "gamma_c_eff_over_gamma_c_JP": float(np.exp(fit.x[2]) / gc_jp),
        "max_abs_JPfit_minus_KP_over_I_IQU": np.max(np.abs(SJf - SK) / SK[:, :1], 0).tolist(),
        "iso_basis_at_JPfit_moments_minus_KP_over_I_N2": maxres(SK, C2, moments(mJf, 2)).tolist(),
        "iso_basis_at_JPfit_moments_minus_KP_over_I_N4": maxres(SK, C4, moments(mJf, 4)).tolist(),
        "nfev": int(fit.nfev),
    }
    val["forward"] = fwd

    # --- internal-consistency checks of the author's JSONs
    reg = json.load(open(HERE / "regression_results.json"))["continuum"]
    summ = auth["summary_max_resid_over_I_IQU"]
    ic = {}
    ic["regression_chi2_minus_dof_in_sd"] = (reg["chi2_fit_seeded"] - reg["residual_dof"]) / np.sqrt(2 * reg["residual_dof"])
    ic["E1_N2_forward_equals_regression_discrepancy"] = abs(summ["E1"]["N2"][0] - reg["max_finite_response_discrepancy_over_I"])
    mono = {}
    for k, v in summ.items():
        I_ = [v[f"N{n}"][0] for n in range(5)]
        mono[k] = bool(all(I_[n + 1] <= I_[n] * 1.0000001 for n in range(4)))
    ic["I_residual_monotone_in_N"] = mono
    ratios = {k: [v[f"N{n}"][0] / max(v[f"N{n}"][1], v[f"N{n}"][2]) for n in range(5)] for k, v in summ.items()}
    ic["I_over_max(Q,U)_ratio_range"] = [min(min(r) for r in ratios.values()), max(max(r) for r in ratios.values())]
    env = {}
    for name, ens in auth["ensembles"].items():
        fw = ens.get("forward") or ens.get("forward_joint_azimuth") or ens.get("forward_effective_diagonal_iso_basis") \
            or ens.get("forward_maxwell_basis")
        if fw:
            env[name] = [fw[f"N{n}"].get("max_resid_over_envelope") for n in range(5)]
    ic["max_resid_over_envelope_by_N"] = env
    pleg = auth["ensembles"]["E4-KP"]["forward_pitch_legendre_vs_L"]
    ic["pitch_legendre_N2_I_by_L"] = {k: v[0] for k, v in pleg["N2"].items()}
    res["internal_consistency"] = ic

    # --- comparison table
    A = {
        "retained": (reg["retained"], val["regression"]["retained"]),
        "s7": (reg["s7_s8"][0], val["regression"]["s7"]),
        "s8": (reg["s7_s8"][1], val["regression"]["s8"]),
        "max_bias_sigma": (reg["max_bias_sigma"], val["regression"]["max_bias_sigma"]),
        "chi2_seeded": (reg["chi2_fit_seeded"], val["regression"]["chi2_seeded"]),
        "fit_max_over_I": (reg["fit_max_over_I_all41"], val["regression"]["max_fit_minus_direct_over_I_41ch"]),
        "E1_N2_I": (summ["E1"]["N2"][0], fwd["E1_N2"][0]),
        "E1_N4_I": (summ["E1"]["N4"][0], fwd["E1_N4"][0]),
        "E2_N2_I": (summ["E2"]["N2"][0], fwd["E2_N2"][0]),
        "E4JP_N2_I": (summ["E4-JP"]["N2"][0], fwd["E4JP_N2"][0]),
        "E4KP_eff_N2_I": (summ["E4-KP"]["N2"][0], fwd["E4KP_eff_iso_N2"][0]),
        "E4KP_eff_N4_I": (summ["E4-KP"]["N4"][0], fwd["E4KP_eff_iso_N4"][0]),
        "E4KP_marg_N2_I": (summ["E4-KP_marginal(L=0)"]["N2"][0], fwd["E4KP_marginal_L0_N2"][0]),
        "E4KP_leg_N4_L8_I": (pleg["N4"]["L8"][0], leg["L8"][0]),
        "E4KP_leg_N4_L4_I": (pleg["N4"]["L4"][0], leg["L4"][0]),
        "E5_sep_N2_Q": (summ["E5_separable_paper_factor"]["N2"][1], fwd["E5_separable_N2"][1]),
        "E5_joint_N2_I": (summ["E5"]["N2"][0], fwd["E5_joint_N2"][0]),
        "E1M_U46_N2_I": (summ["E1M_U46_basis"]["N2"][0], fwd["E1M_U46_N2"][0]),
        "E1_Pi_0.1GHz": (auth["e1_analytic"]["Pi_direct_no_screen_at_0.1_0.316_1_3.16_10GHz"][0], val["E1_Pi_0.1GHz"]),
        "E1_unbounded_excess": (auth["e1_analytic"]["max_abs(I_direct/I_unbounded - 1)"],
                                1 - val["E1_direct_over_unbounded_at_10GHz"]),
        "gamma_c_JP": (auth["ageing"]["gamma_c_JP"], gc_jp),
        "nu_c_GHz": (auth["ageing"]["nu_c_JP=a1<B_perp>gamma_c^2_Hz"] / 1e9, val["ageing"]["nu_c_JP_GHz"]),
        "KP_eff_column": (auth["ensembles"]["E4-KP"]["effective_column"], col_eff),
        "E5_fA": (auth["ensembles"]["E5"]["fractions_A_B_crossing_channel"][0], float(fA)),
        "E5_chi_swing_deg": (auth["ensembles"]["E5"]["chi_derotated_swing_deg_where_P/I>0.03"],
                             val["E5_chi_derot_swing_deg"]),
    }
    res["comparison"] = {k: {"author": float(a), "verifier": float(b),
                             "rel_diff": float(abs(a - b) / max(abs(a), abs(b), 1e-300))} for k, (a, b) in A.items()}

    src = Path(__file__).read_bytes()
    res["provenance"] = {
        "script": Path(__file__).name, "script_sha256": hashlib.sha256(src).hexdigest(),
        "inputs_sha256": {pth.relative_to(HERE).as_posix(): hashlib.sha256(pth.read_bytes()).hexdigest()
                          for pth in (HERE / "forward_results.json", HERE / "regression_results.json",
                                      HERE / "lib" / "kernels.py", HARM) if pth.exists()},
        "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
        "quadrature": {"channel": N_T, "B_uniform": N_BU, "eta": N_ETA, "energy_per_bin": N_E,
                       "cut_panel_w_nodes": 60, "KP_eta": 24, "KP_mu": "32+32 split at gamma_c(mu)=edge 9",
                       "maxwell_B": "160 GL on [0,35] muG"},
        "wall_clock_s": time.time() - T0,
    }
    OUT.write_text(json.dumps(res, indent=1, default=float))
    for k, v in res["comparison"].items():
        print(f"{k:22s} {v['author']:.6g} {v['verifier']:.6g} {v['rel_diff']:.2e}")
    print(json.dumps(chk, indent=0, default=float)[:3000])
    print(json.dumps(ic, default=float)[:2000])


if __name__ == "__main__":
    import scipy.stats  # noqa: F401

    main()
