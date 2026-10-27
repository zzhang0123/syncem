#!/usr/bin/env python
"""Independent verification of the fits (fits.py / fits_results.json).

Independent of the author's analysis code: only lib/kernels.py is imported, and only for
the log-safe F(x) (spot-checked against scipy kv + quad first).  Everything else is rebuilt
here:
  * ln-derivatives of F and G from Bessel identities (D = x d/dx):
      DF = F - x^2 K53,  D^2F = F - (4/3) x^2 K53 + x^3 K23,
      DG = (5/3) G - x^2 K53,  D^2G = (25/9) G - 2 x^2 K53 + x^3 K23,
    checked against mpmath numerical derivatives;
  * B-averaged kernel tables h_S^r(v) = <B W_S^(r)(v / (a1 B))>_B on a fine ln v grid
    (cubic spline; accuracy measured against exact node sums), eta handled by exact
    Gauss-Legendre sums with K(u; eta) = s_eta h(u / s_eta), s_eta = sqrt(1 - eta^2);
  * channel response, screen, azimuth factor, basis columns, ensembles (E1, E2, E3, E4-JP,
    E4-KP, E5, E1M), seeded noise, whitened TSVD/UPRE, coverage over an independent set of
    4000 noise realisations (different seed from the author), NNLS witness, Hausdorff,
    E5 joint and free-factor fits, de-rotated angle.
Writes verify_fits_results.json next to this script.  Run: python verify_fits.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import mpmath as mp
import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy import integrate, special
from scipy.interpolate import CubicSpline
from scipy.optimize import minimize, nnls
from scipy.stats import chi2 as chi2d
from scipy.stats import ncx2, norm

T0 = time.time()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import kernels as LK  # noqa: E402  (F only)

# ----------------------------------------------------------------------------- constants
E_ESU, M_E, C_CGS, C_SI = 4.80320471257e-10, 9.1093837015e-28, 2.99792458e10, 2.99792458e8
SIGMA_T, EV, MYR = 6.6524587321e-25, 1.602176634e-12, 3.15576e13
A1 = 3.0 * E_ESU * 1e-6 / (4.0 * np.pi * M_E * C_CGS)  # a_B per muG of B_perp [Hz]

# ----------------------------------------------------------------------------- configuration
NU = np.geomspace(1e8, 1e10, 41)
FIT, HELD = np.arange(0, 41, 2), np.arange(1, 41, 2)
NB, NORD = 12, 2
LE = np.linspace(np.log(300.0), np.log(1e5), NB + 1)
HB = 0.5 * (LE[1] - LE[0])
GB = np.exp(0.5 * (LE[1:] + LE[:-1]))
_m = np.diff(np.exp(LE) ** (-1.6)) / (-1.6)
WB = _m / _m.sum()
AZ = 0.4 * np.exp(0.5j)
N_T, N_ETA, N_BU, N_BM, N_G = 400, 24, 32, 160, 32


def gl(n, a, b):
    t, w = leggauss(n)
    return 0.5 * (b - a) * t + 0.5 * (b + a), 0.5 * (b - a) * w


def rho1(t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    ok = np.abs(t) < 1
    out[ok] = np.exp(-t[ok] ** 2 / (1 - t[ok] ** 2))
    return out


Z1 = integrate.quad(lambda t: np.exp(-t * t / (1 - t * t)), -1, 1, epsabs=0, epsrel=1e-13, limit=200)[0]


def channels(nt):
    t, w = leggauss(nt)
    cw = w * rho1(t) / Z1
    nu = NU[:, None] * (1 + 0.2 * t[None, :])
    lam2 = (C_SI / nu) ** 2
    scr = np.exp(2j * lam2 * 8.0 - 2.0 * lam2**2 * 0.25)
    return nu, cw, scr


ETA, QETA = gl(N_ETA, 0.0, 0.5)
QETA = QETA / 0.5
RES = {"provenance": {}, "checks": {}, "ensembles": {}}


# ----------------------------------------------------------------------------- kernels
def F_parts(x):
    lnE0, _ = LK.scaled_log_derivatives(x, 0, "F")
    k53, k23 = special.kve(5 / 3, x), special.kve(2 / 3, x)
    e0 = np.exp(lnE0)
    q53, q23 = x * k53 / e0, x**2 * k23 / e0  # x^2 K53 / F, x^3 K23 / F
    return np.log(x) - x + lnE0, 1 - q53, 1 - (4 / 3) * q53 + q23


def G_parts(x):
    k53, k23 = special.kve(5 / 3, x), special.kve(2 / 3, x)
    rr = x * k53 / k23
    return np.log(x) + np.log(k23) - x, 5 / 3 - rr, 25 / 9 - 2 * rr + x**2


def kernel_checks():
    out = {"F_G_spot_vs_scipy_quad": []}
    for x in [1e-3, 0.1, 1.0, 5.0, 20.0]:
        Fq = x * integrate.quad(lambda t: special.kv(5 / 3, t), x, np.inf, epsabs=0, epsrel=1e-12, limit=500)[0]
        Gq = x * special.kv(2 / 3, x)
        out["F_G_spot_vs_scipy_quad"].append({"x": x, "F_lib": float(LK.F(x)), "F_quad": Fq,
                                              "relF": float(abs(LK.F(x) / Fq - 1)),
                                              "relG": float(abs(LK.G(x) / Gq - 1))})
    mp.mp.dps = 30
    Fm = lambda x: x * mp.quad(lambda t: mp.besselk(mp.mpf(5) / 3, t), [x, mp.inf])
    Gm = lambda x: x * mp.besselk(mp.mpf(2) / 3, x)
    rows = []
    for x in [0.01, 0.7, 4.0, 15.0]:
        for name, f, parts in (("F", Fm, F_parts), ("G", Gm, G_parts)):
            g = lambda s: f(mp.e**s)
            s0 = mp.log(x)
            d0, d1, d2 = g(s0), mp.diff(g, s0, 1), mp.diff(g, s0, 2)
            lnW, r1, r2 = parts(np.array([x]))
            rows.append({"W": name, "x": x, "ratio1_err": float(abs(r1[0] - float(d1 / d0))),
                         "ratio2_err": float(abs(r2[0] - float(d2 / d0)) / max(1, abs(float(d2 / d0)))),
                         "lnW_err": float(abs(lnW[0] - float(mp.log(d0))))})
    out["ln_derivative_identities_vs_mpmath"] = rows
    return out


LV = np.linspace(np.log(4e-3), np.log(2.5e5), 12001)


def btable_exact(lv, bn, bw):
    x = np.exp(lv)[:, None] / (A1 * bn[None, :])
    lw = np.log(bw) + np.log(bn)
    out = {}
    for S, (lnW, r1, r2) in (("I", F_parts(x)), ("Q", G_parts(x))):
        a = lnW + lw[None, :]
        mx = a.max(1, keepdims=True)
        e = np.exp(a - mx)
        tot = e.sum(1)
        pi = e / tot[:, None]
        out[S] = (mx[:, 0] + np.log(tot), (pi * r1).sum(1), (pi * r2).sum(1))
    return out


def build_btable(bn, bw, label):
    ex = btable_exact(LV, bn, bw)
    tab = {S: [CubicSpline(LV, v) for v in ex[S]] for S in ex}
    rng = np.random.default_rng(1)
    lt = rng.uniform(LV[0], np.log(2e5), 400)
    chk = btable_exact(lt, bn, bw)
    err = {}
    for S in tab:
        ok = chk[S][0] > -700
        err[S] = [float(np.max(np.abs(tab[S][0](lt[ok]) - chk[S][0][ok])))] + [
            float(np.max(np.abs(tab[S][r](lt[ok]) - chk[S][r][ok]) / np.maximum(1, np.abs(chk[S][r][ok]))))
            for r in (1, 2)]
    RES["checks"][f"btable_spline_err_{label}"] = err
    return tab


def chan_kernels(tab, gam, order, nt=N_T, eta=None):
    """KI (order+1, J, G) real and KP (order+1, J, G) complex (Q-kernel -G, screen, no azimuth).

    eta=None: isotropic eta average (exact GL sum); eta=float: that eta only.
    """
    nu, cw, scr = channels(nt)
    gam = np.atleast_1d(gam)
    KI = np.zeros((order + 1, 41, gam.size))
    KP = np.zeros((order + 1, 41, gam.size), complex)
    etas = (ETA, QETA) if eta is None else (np.array([eta]), np.array([1.0]))
    for s0 in range(0, gam.size, 96):
        g = gam[s0:s0 + 96]
        for e, q in zip(*etas):
            s = np.sqrt(1 - e * e)
            lv = np.log(nu[:, :, None] / (g[None, None, :] ** 2 * s))
            for S in ("I", "Q"):
                h0 = s * np.exp(tab[S][0](lv))
                for r in range(order + 1):
                    val = h0 if r == 0 else h0 * tab[S][r](lv)
                    if S == "I":
                        KI[r, :, s0:s0 + 96] += q * np.einsum("t,jtg->jg", cw, val)
                    else:
                        KP[r, :, s0:s0 + 96] -= q * np.einsum("jt,jtg->jg", cw * scr, val)
    return KI, KP


def basis(tab, nt=N_T):
    KI, KP = chan_kernels(tab, GB, NORD, nt)
    fac = np.array([(-2 * HB) ** r / np.prod(np.arange(1, r + 1)) for r in range(NORD + 1)])
    wI = np.einsum("b,r,rjb->jbr", WB, fac, KI).reshape(41, -1)
    wP = np.einsum("b,r,rjb->jbr", WB, fac, KP).reshape(41, -1)  # complex, no azimuth factor
    return wI, wP


# ----------------------------------------------------------------------------- measures
def measure(shape, gc=None, alpha=0.0, n=N_G):
    """Nodes (gamma, weight dgamma * shape, bin, x_b); shape(g, onem) with onem = 1 - g/gc."""
    G, W, Bn = [], [], []
    for b in range(NB):
        la, lb = LE[b], LE[b + 1]
        if gc is not None and np.log(gc) <= la:
            break
        if gc is not None and np.log(gc) < lb:
            wmax = (1 - np.exp(la) / gc) ** 0.2
            wn, ww = gl(n, 0.0, wmax)
            g, onem, jac = gc * (1 - wn**5), wn**5, 5 * gc * wn**4 * ww
        else:
            ln, lw = gl(n, la, lb)
            g = np.exp(ln)
            onem = 1 - g / gc if gc is not None else np.ones_like(g)
            jac = g * lw
        G.append(g)
        W.append(jac * shape(g, onem))
        Bn.append(np.full(n, b))
    G, W, Bn = map(np.concatenate, (G, W, Bn))
    return G, W, Bn, (np.log(G) - np.log(GB[Bn])) / HB


def moments(m):
    G, W, Bn, X = m
    a = np.zeros((NB, NORD + 1))
    for r in range(NORD + 1):
        np.add.at(a[:, r], Bn, W * X**r)
    return (a / WB[:, None]).ravel()


def direct(tab, m, az=AZ, eta=None, nt=N_T):
    KI, KP = chan_kernels(tab, m[0], 0, nt, eta)
    sI, sP = KI[0] @ m[1], az * (KP[0] @ m[1])
    return np.stack([sI, sP.real, sP.imag], axis=-1)


def unit(m):
    t = m[1].sum()
    return (m[0], m[1] / t, m[2], m[3])


def pl(p):
    return lambda g, o: (g / 1e4) ** (-p)


# ----------------------------------------------------------------------------- fitting
def whiten(C, S):
    sig = 0.01 * S[:, 0]
    Af = (C[FIT] / sig[FIT, None, None]).reshape(63, -1)
    Ah = (C[HELD] / sig[HELD, None, None]).reshape(60, -1)
    return sig, Af, Ah


def svd_k(A):
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    k = int(np.sum(s > 1e-13 * s[0]))
    return u[:, :k], s[:k], vt[:k]


def tsvd(name, C, S, a_true, zf, zh, rng_real, pair_rtol=None, nreal=4000):
    sig, Af, Ah = whiten(C, S)
    u, s, vt = svd_k(Af)
    k = s.size
    allowed = np.ones(k, bool) if pair_rtol is None else np.append(s[:-1] > s[1:] * (1 + pair_rtol), True)
    Sfw = (S[FIT] / sig[FIT, None]).ravel()
    Shw = (S[HELD] / sig[HELD, None]).ravel()
    dfw, dhw = Sfw + zf.ravel(), Shw + zh.ravel()
    Rs = np.arange(1, k + 1)

    def curve(d):
        p = d @ u
        return p, np.sum(d**2, -1)[..., None] - np.cumsum(p**2, -1)

    proj, c2 = curve(dfw)
    upre = np.where(allowed, c2 + 2 * Rs, np.inf)
    R = int(Rs[np.argmin(upre)])
    eps = S - np.einsum("jsn,n->js", C, a_true)
    epsw = (eps[FIT] / sig[FIT, None]).ravel()
    bias = u.T @ epsw
    beta_t = vt @ a_true
    bdev = proj[:R] - s[:R] * beta_t[:R]  # (beta_hat - beta)/sigma
    a_hat = vt[:R].T @ (proj[:R] / s[:R])
    pred = np.einsum("jsn,n->js", C, a_hat)
    H = (Ah @ vt[:R].T) / s[:R]
    r_h = H @ proj[:R] - dhw
    M = np.linalg.inv(np.eye(60) + H @ H.T)
    # mean withheld residual for noise-free data (truncation + representation), and expected predictive chi2
    mu = H @ (u[:, :R].T @ (Sfw)) - Shw
    a_nf = vt[:R].T @ ((u[:, :R].T @ Sfw) / s[:R])
    pred_nf = np.einsum("jsn,n->js", C, a_nf)
    S0 = np.einsum("jsn,n->js", C, a_true)  # eps = 0 data
    S0f, S0h = (S0[FIT] / sig[FIT, None]).ravel(), (S0[HELD] / sig[HELD, None]).ravel()
    mu0 = H @ (u[:, :R].T @ S0f) - S0h
    out = {"E_chi2_held_pred_eps0_truncation_only": float(60 + mu0 @ M @ mu0),
           "eps_outside_first_R_modes": {int(r): float(epsw @ epsw - bias[:r] @ bias[:r]) for r in (3, 5, 7, 9, 11, 14)},
           "rank": k, "R_star": R, "s_R": float(s[R - 1]), "chi2_fit": float(c2[R - 1]), "dof": 63 - R,
           "chi2_held_raw": float(r_h @ r_h), "chi2_held_pred": float(r_h @ M @ r_h),
           "E_chi2_held_raw_if_eps0_and_no_trunc": float(60 + np.sum(H**2)),
           "E_chi2_held_pred_including_mean_residual": float(60 + mu @ M @ mu),
           "max_err": float(np.max(np.abs(pred - S) / S[:, 0:1])),
           "max_err_noise_free": float(np.max(np.abs(pred_nf - S) / S[:, 0:1])),
           "max_abs_beta_dev_sigma": float(np.max(np.abs(bdev))),
           "max_abs_bias_sigma": float(np.max(np.abs(bias[:R]))), "bias_mode": int(np.argmax(np.abs(bias[:R])) + 1),
           "eps_w_norm2": float(epsw @ epsw), "eps_w_outside_span": float(epsw @ epsw - bias @ bias),
           "eps_w_outside_retained_R_star": float(epsw @ epsw - bias[:R] @ bias[:R]),
           "picard_chi2": {int(r): float(c2[r - 1]) for r in range(3, 15) if r <= k},
           "s_first8": s[:8].tolist()}
    if pair_rtol is not None:
        out["n_degenerate_pairs"] = int(np.sum(~allowed[:-1]))
    # Hausdorff on the TSVD minimum-norm moments (first 36 entries)
    a = a_hat[:36].reshape(NB, 3)
    tol = 1e-12 * np.max(np.abs(a_true[:36]))
    viol = (a[:, 0] < -tol) | (np.abs(a[:, 1]) > a[:, 0] + tol) | (a[:, 2] < -tol) | (a[:, 2] > a[:, 0] + tol) | \
           (a[:, 1] ** 2 > a[:, 0] * a[:, 2] + tol * np.max(np.abs(a_true[:36])))
    out["hausdorff_bins_violating"] = int(viol.sum())
    # noise realisations (independent seed)
    Zf = rng_real.standard_normal((nreal, 63))
    Zh = rng_real.standard_normal((nreal, 60))
    P, C2 = curve(Sfw + Zf)
    Rn = Rs[np.argmin(np.where(allowed, C2 + 2 * Rs, np.inf), axis=1)]
    dev = P - s * beta_t  # (beta_hat - beta) / sigma per mode, all modes
    cov = np.mean(np.abs(dev[:, :R]) <= 1, axis=0)
    q = chi2d.ppf(norm.cdf(1) - norm.cdf(-1), R)
    joint_fixed = float(np.mean(np.sum(dev[:, :R] ** 2, 1) <= q))
    cs = np.cumsum(dev**2, 1)
    joint_post = float(np.mean(cs[np.arange(nreal), Rn - 1] <= chi2d.ppf(0.6826894921, Rn)))
    an = norm.cdf(1 - bias[:R]) - norm.cdf(-1 - bias[:R])
    out.update({"R_star_p16_p50_p84": np.percentile(Rn, [16, 50, 84]).tolist(),
                "coverage_min_mean": [float(cov.min()), float(cov.mean())],
                "coverage_analytic_min_mean": [float(an.min()), float(an.mean())],
                "coverage_joint_fixed": joint_fixed,
                "coverage_joint_fixed_analytic": float(ncx2.cdf(q, R, float(bias[:R] @ bias[:R]))) if R > 0 else None,
                "coverage_joint_post": joint_post,
                "pull_mean_sd_mode1": [float(dev[:, 0].mean()), float(dev[:, 0].std())]})
    # predictive chi2 over realisations at their own R* (median and >99% fraction)
    pc = np.empty(nreal)
    for Rv in np.unique(Rn):
        sel = Rn == Rv
        Hh = (Ah @ vt[:Rv].T) / s[:Rv]
        rr = P[sel, :Rv] @ Hh.T - (Shw + Zh[sel])
        Mv = np.linalg.inv(np.eye(60) + Hh @ Hh.T)
        pc[sel] = np.einsum("ni,ij,nj->n", rr, Mv, rr)
    out["chi2_held_pred_median"] = float(np.median(pc))
    out["frac_chi2_held_pred_above_99"] = float(np.mean(pc > chi2d.ppf(0.99, 60)))
    RES["ensembles"][name] = out
    return out, dict(u=u, s=s, vt=vt, R=R, a_hat=a_hat, proj=proj, Af=Af, dfw=dfw, sig=sig, pred=pred)


def nnls_witness(C36, S, zf, zh, rows=(0, 1, 2)):
    xq = np.cos(np.pi * np.arange(17) / 16)
    pw = np.stack([xq**r for r in range(3)])
    At = np.einsum("jsbr,rq->jsbq", C36.reshape(41, 3, NB, 3), pw).reshape(41, 3, -1)[:, list(rows)]
    sig = 0.01 * S[:, 0]
    Af = (At[FIT] / sig[FIT, None, None]).reshape(len(FIT) * len(rows), -1)
    dfw = (S[FIT][:, list(rows)] / sig[FIT, None] + zf[:, list(rows)]).ravel()
    w, rn = nnls(Af, dfw, maxiter=200 * Af.shape[1])
    wnf, rnf = nnls(Af, (S[FIT][:, list(rows)] / sig[FIT, None]).ravel(), maxiter=200 * Af.shape[1])
    wb = w.reshape(NB, 17).sum(1)
    return {"chi2_seeded": float(rn**2), "chi2_noise_free": float(rnf**2),
            "implied_column": float(WB @ wb), "bin1_a0": float(wb[1])}


# ----------------------------------------------------------------------------- main
def main():
    RES["checks"]["kernels"] = kernel_checks()
    bu, wu = gl(N_BU, 4.0, 6.0)
    wu = wu / 2.0
    tabU = build_btable(bu, wu, "U46")
    sgm = 5.0 / np.sqrt(3)
    bm, wm = gl(N_BM, 1e-3, 30.0)
    wm = wm * np.sqrt(2 / np.pi) * bm**2 / sgm**3 * np.exp(-bm**2 / (2 * sgm**2))
    RES["checks"]["maxwell_weight_sum"] = float(wm.sum())
    tabM = build_btable(bm, wm, "Maxwell")
    t1 = time.time()

    wI, wP = basis(tabU)
    wI2, wP2 = basis(tabU, nt=2 * N_T)
    RES["checks"]["basis_channel_node_doubling_max_rel"] = float(max(
        np.max(np.abs(wI2 - wI)) / np.max(np.abs(wI)), np.max(np.abs(wP2 - wP)) / np.max(np.abs(wP))))
    Csep = np.stack([wI, (AZ * wP).real, (AZ * wP).imag], axis=1)  # (41, 3, 36)

    # ensembles
    ens = {}
    m = unit(measure(pl(2.6)))
    ens["E1"] = (direct(tabU, m), moments(m))
    ens["E1M"] = (direct(tabM, m), moments(m))
    m = unit(measure(lambda g, o: (g / 7800) ** -2.0 * (1 + (g / 7800) ** 2) ** (-0.6)))
    ens["E2"] = (direct(tabU, m), moments(m))
    m = unit(measure(lambda g, o: (g / 1e4) ** -2.6 * np.exp(0.02 * np.log(g / 1e4) ** 2)))
    ens["E3"] = (direct(tabU, m), moments(m))
    lc = 4 * SIGMA_T / (3 * M_E * C_CGS)
    b_s, b_c, t = lc * (152 / 6) * 1e-12 / (8 * np.pi), lc * EV, 19 * MYR
    gc_jp = 1 / ((b_s + b_c) * t)
    m = unit(measure(lambda g, o: (g / 1e4) ** -2.6 * o**0.6, gc=gc_jp))
    ens["E4-JP"] = (direct(tabU, m), moments(m))
    # KP: N(g, mu) = (K/2) g^-p (1 - b(mu) g t)^(p-2); beaming: emission at eta uses 2 N(g, eta)
    gck = lambda mu: 1 / ((1.5 * b_s * (1 - mu**2) + b_c) * t)
    mus, wmu = gl(64, 0.0, 1.0)
    tot = sum(w_ * measure(lambda g, o: (g / 1e4) ** -2.6 * o**0.6, gc=gck(mu))[1].sum() for mu, w_ in zip(mus, wmu))
    # cross-check one inner integral by QAWS
    g0 = gck(0.3)
    qa = integrate.quad(lambda y: (y * g0 / 1e4) ** -2.6 * g0, 300 / g0, 1, weight="alg", wvar=(0, 0.6),
                        epsabs=0, epsrel=1e-12, limit=400)[0]
    RES["checks"]["kp_inner_qaws_rel"] = float(abs(measure(lambda g, o: (g / 1e4) ** -2.6 * o**0.6, gc=g0)[1].sum() / qa - 1))
    Kn = 1 / tot
    SK = np.zeros((41, 3))
    aK = np.zeros(36)
    col = 0.0
    for e, q in zip(ETA, QETA):
        mm = measure(lambda g, o: Kn * (g / 1e4) ** -2.6 * o**0.6, gc=gck(e))
        SK += q * direct(tabU, mm, eta=e)
        aK += q * moments(mm)
        col += q * mm[1].sum()
    ens["E4-KP"] = (SK, aK)
    RES["checks"]["kp"] = {"gamma_c_jp": gc_jp, "gamma_c_kp_0_half_1": [gck(0.0), gck(0.5), gck(1.0)],
                           "effective_column": col}
    # E5
    mA, mB = unit(measure(pl(3.1))), unit(measure(pl(2.1)))
    SA, SB = direct(tabU, mA, az=1.0), direct(tabU, mB, az=1.0)
    j1 = 20
    fA = SB[j1, 0] / (SA[j1, 0] + SB[j1, 0])
    fB = 1 - fA
    cA, cB = 0.4 * np.exp(0.5j), 0.4 * np.exp(2j * (0.25 + np.pi / 4))
    PA, PB = SA[:, 1] + 1j * SA[:, 2], SB[:, 1] + 1j * SB[:, 2]
    S5 = np.stack([fA * SA[:, 0] + fB * SB[:, 0], *(lambda P: (P.real, P.imag))(fA * cA * PA + fB * cB * PB)], -1)
    a5 = fA * moments(mA) + fB * moments(mB)
    M2 = fA * cA * moments(mA) + fB * cB * moments(mB)
    RES["checks"]["E5_fA"] = float(fA)
    t2 = time.time()

    # compare direct spectra with the author's products (data-level cross-check only)
    prod = np.load(HERE / "fits_products.npz")
    keymap = {"E1": "E1", "E2": "E2", "E3": "E3", "E4-JP": "E4_JP", "E4-KP": "E4_KP", "E1M": "E1M", "E5": "E5"}
    allS = dict(ens)
    allS["E5"] = (S5, None)
    def _nrm(S_):
        return S_ / np.max(S_[:, 0])
    RES["checks"]["direct_vs_author_max_over_I_scale_free"] = {
        k: float(np.max(np.abs(_nrm(allS[k][0]) - _nrm(prod[keymap[k] + "_S_direct"])) / _nrm(allS[k][0])[:, 0:1]))
        for k in allS}
    RES["checks"]["a_true_vs_author_max_abs"] = {
        k: float(np.max(np.abs(ens[k][1] - prod[keymap[k] + "_a_true"]))) for k in ens}

    zf = np.random.default_rng(20260928).normal(size=(21, 3))
    zh = np.random.default_rng(20260929).normal(size=(20, 3))
    Sa = prod["E1_S_direct"]
    RES["checks"]["seeded_noise_vs_author_max_abs"] = {
        "fit": float(np.max(np.abs((prod["E1_data_fit"] - Sa[FIT]) / (0.01 * Sa[FIT, 0:1]) - zf))),
        "held": float(np.max(np.abs((prod["E1_data_held"] - Sa[HELD]) / (0.01 * Sa[HELD, 0:1]) - zh)))}

    fits = {}
    for i, k in enumerate(["E1", "E2", "E3", "E4-JP", "E4-KP", "E1M"]):
        fits[k] = tsvd(k, Csep, ens[k][0], ens[k][1], zf, zh, np.random.default_rng(777 + i))
        RES["ensembles"][k]["s_vs_author_max_rel"] = float(np.max(np.abs(
            fits[k][1]["s"] - prod[keymap[k] + "_s"][:fits[k][1]["s"].size]) / prod[keymap[k] + "_s"][0]))
        RES["ensembles"][k]["nnls"] = nnls_witness(Csep, ens[k][0], zf, zh)
    # E1 regression rule s >= 10
    f1 = fits["E1"][1]
    R10 = int(np.sum(f1["s"] >= 10))
    RES["checks"]["E1_threshold_rule"] = {"R": R10, "chi2": float(f1["dfw"] @ f1["dfw"] - np.sum(f1["proj"][:R10] ** 2))}

    # spikiness test of the "eps inside column span" claim: an arbitrary non-smooth spectrum
    rng = np.random.default_rng(5)
    def spiky(g, o):
        lg = np.log(g)
        return sum(np.exp(-0.5 * ((lg - c) / 0.03) ** 2) for c in rng.uniform(LE[0], LE[-1], 6)) + 1e-3 * (g / 1e4) ** -2
    ms = unit(measure(spiky, n=128))
    Ss = direct(tabU, ms)
    sig, Af, _ = whiten(Csep, Ss)
    u, s, vt = svd_k(Af)
    ew = ((Ss - np.einsum("jsn,n->js", Csep, moments(ms)))[FIT] / sig[FIT, None]).ravel()
    RES["checks"]["spiky_spectrum_eps"] = {"eps_w_norm2": float(ew @ ew),
                                           "outside_span": float(ew @ ew - np.sum((u.T @ ew) ** 2))}

    # ------------------------------------------------------------------ E5 joint
    CI = np.stack([wI, 0 * wI, 0 * wI], 1)
    Z = np.zeros_like(wI)
    Cj = np.concatenate([
        np.stack([wI, Z, Z], 1),
        np.stack([Z, wP.real, wP.imag], 1),
        np.stack([Z, -wP.imag, wP.real], 1)], axis=2)  # (41, 3, 108): [a, Re M2, Im M2]
    a5full = np.concatenate([a5, M2.real, M2.imag])
    fj = tsvd("E5_joint", Cj, S5, a5full, zf, zh, np.random.default_rng(999), pair_rtol=1e-8)
    RES["ensembles"]["E5_joint"]["nnls_I_rows"] = nnls_witness(Csep, S5, zf, zh, rows=(0,))
    # angles
    lam2c = (C_SI / NU) ** 2
    def chi_deg(Sx):
        return np.degrees(0.5 * np.angle((Sx[:, 1] + 1j * Sx[:, 2]) * np.exp(-2j * lam2c * 8.0)))
    chi_t, chi_j = chi_deg(S5), chi_deg(fj[1]["pred"])
    sel = np.hypot(S5[:, 1], S5[:, 2]) / S5[:, 0] > 0.03
    # linearised sigma of chi from Cov(a_hat) = V_R D^-2 V_R^T
    R5, vt5, s5 = fj[1]["R"], fj[1]["vt"], fj[1]["s"]
    covA = (vt5[:R5].T / s5[:R5] ** 2) @ vt5[:R5]
    Ppred = fj[1]["pred"][:, 1] + 1j * fj[1]["pred"][:, 2]
    J = 0.5 * np.imag((Cj[:, 1] + 1j * Cj[:, 2]) / Ppred[:, None])
    sig_chi = np.degrees(np.sqrt(np.einsum("jn,nm,jm->j", J, covA, J)))

    # ------------------------------------------------------------------ E5 free complex factor
    sig5 = 0.01 * S5[:, 0]
    dfw5 = (S5[FIT] / sig5[FIT, None] + zf).ravel()
    dhw5 = (S5[HELD] / sig5[HELD, None] + zh).ravel()

    def design(c):
        P = c * wP
        return np.stack([wI, P.real, P.imag], 1)

    def curve_c(c):
        A = (design(c)[FIT] / sig5[FIT, None, None]).reshape(63, -1)
        u, s, vt = svd_k(A)
        p = u.T @ dfw5
        c2 = dfw5 @ dfw5 - np.cumsum(p**2)
        c2 = np.concatenate([c2, np.full(40 - c2.size, c2[-1])])
        return c2

    grid = [a * np.exp(1j * ph) for a in np.linspace(0.02, 1.0, 50) for ph in np.linspace(-np.pi, np.pi, 96, endpoint=False)]
    gv = np.array([curve_c(c) for c in grid])
    best = {}
    for R in range(1, 27):
        cand = []
        for jj in np.argsort(gv[:, R - 1])[:4]:
            x0 = np.array([grid[jj].real, grid[jj].imag])
            r = minimize(lambda v: curve_c(v[0] + 1j * v[1])[R - 1], x0, method="Nelder-Mead",
                         options={"xatol": 1e-9, "fatol": 1e-9, "maxiter": 3000,
                                  "initial_simplex": [x0, x0 + [0.01, 0], x0 + [0, 0.01]]})
            cand.append((r.fun, r.x[0] + 1j * r.x[1]))
        best[R] = min(cand, key=lambda z: z[0])
    upre_s = {R: best[R][0] + 2 * (R + 2) for R in best}
    Rs_ = min(upre_s, key=upre_s.get)
    cst = best[Rs_][1]
    Cst = design(cst)
    _, Afs, Ahs = whiten(Cst, S5)
    u, s, vt = svd_k(Afs)
    p = u.T @ dfw5
    ahat = vt[:Rs_].T @ (p[:Rs_] / s[:Rs_])
    pred_s = np.einsum("jsn,n->js", Cst, ahat)
    H = (Ahs @ vt[:Rs_].T) / s[:Rs_]
    rh = H @ p[:Rs_] - dhw5
    chi_s = chi_deg(pred_s)
    RES["ensembles"]["E5_free_c"] = {
        "R_star": int(Rs_), "c": [float(cst.real), float(cst.imag)], "chi2_fit": float(best[Rs_][0]),
        "full_rank_chi2_fit": float(best[26][0]), "chi2_held_raw": float(rh @ rh),
        "chi2_held_pred_fixed_c": float(rh @ np.linalg.inv(np.eye(60) + H @ H.T) @ rh),
        "max_err": float(np.max(np.abs(pred_s - S5) / S5[:, 0:1])),
        "upre_sep_minus_joint_seeded": float(upre_s[Rs_] - (fj[0]["chi2_fit"] + 2 * fj[0]["R_star"])),
        "upre_curve_R5_to_12": {R: float(upre_s[R]) for R in range(5, 13)}}
    idx = {"0.398": 12, "1.0": 20, "10": 40}
    RES["ensembles"]["E5_angles"] = {
        "chi_deg_truth_joint_sigma_free": {k: [float(chi_t[j]), float(chi_j[j]), float(sig_chi[j]), float(chi_s[j])]
                                           for k, j in idx.items()},
        "n_channels_P_over_I_gt_3pct": int(sel.sum()),
        "swing_truth_joint_free": [float(np.ptp(chi_t[sel])), float(np.ptp(chi_j[sel])), float(np.ptp(chi_s[sel]))],
        "joint_max_abs_dev_deg": float(np.max(np.abs(chi_j - chi_t)[sel])),
        "joint_max_abs_dev_sigma": float(np.max((np.abs(chi_j - chi_t) / sig_chi)[sel])),
        "free_max_abs_dev_deg": float(np.max(np.abs(chi_s - chi_t)[sel]))}
    RES["timings_s"] = {"tables": t1 - T0, "ensembles": t2 - t1, "total": time.time() - T0}


if __name__ == "__main__":
    main()
    src = Path(__file__).read_bytes()
    RES["provenance"] = {"script": Path(__file__).name, "script_sha256": hashlib.sha256(src).hexdigest(),
                         "inputs_sha256": {f: hashlib.sha256((HERE / f).read_bytes()).hexdigest()
                                           for f in ("lib/kernels.py", "fits_products.npz", "fits_results.json")},
                         "numpy": np.__version__, "scipy": scipy.__version__, "mpmath": mp.__version__,
                         "wall_clock_s": time.time() - T0}
    (HERE / "verify_fits_results.json").write_text(json.dumps(RES, indent=1, default=float))
    print(json.dumps(RES, indent=1, default=float))
