"""Independent verifier for the KP-bridge part (kp_bridge.py).

Recomputes headline numbers from scratch and checks the internal consistency of
kp_bridge_results.json.  It does NOT import kp_harmonic.py or kp_bridge.py.  F and G are
computed here from scipy.special.kv + quad; lib/kernels.py is imported only to spot-check
5 values of F and G against them.

Units differ from the author's on purpose: e = m_e = c = 1 and B = 1, so the source-time
gyrofrequency is omega_B = 1/gamma and nu_B = 1/(2 pi gamma).  The harmonic sum is built
directly from the paper: Eq. (2.1) (nu_m = m nu_B / D,
x_m = m beta_perp sin(theta)/D), Eqs. (2.2) and (2.3)
(C_m^2 = m^2 omega_B^2 / (2 pi D^3)) and Eq. (2.4a); the delta function in nu is integrated over
mu at fixed eta, giving the weight p(mu_m) D_m^2 / (m nu_B beta eta).
Continuum: 2 p(eta) K^iso with K_I = A F(x), K_Q = -A G(x), A = sqrt(3) sin(theta)/(4 pi),
x = nu / (a_B gamma^2), a_B = 3 sin(theta)/(4 pi)   (Eq. (2.8)).

Run: python verify_kpbridge.py [--workers N]
Writes verify_kpbridge_results.json next to this script.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import hashlib
import json
import math
import platform
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy import integrate, optimize, special

HERE = Path(__file__).resolve().parent
A_KP = 0.6          # p_inj - 2, p_inj = 2.6 (E4)
P_INJ = 2.6
MU_C = 0.25
THETA_C = math.acos(MU_C)
CHUNK = 400_000
T0 = time.time()


# ============================================================ continuum (own F, G)
def F_ref(x):
    v = integrate.quad(lambda t: special.kv(5.0 / 3.0, t), x, np.inf, epsabs=0,
                       epsrel=1e-12, limit=400)[0]
    return x * v


def G_ref(x):
    return x * special.kv(2.0 / 3.0, x)


def a_B(eta):
    return 3.0 * math.sqrt((1 - eta) * (1 + eta)) / (4 * math.pi)


def amp(eta):
    return math.sqrt(3.0) * math.sqrt((1 - eta) * (1 + eta)) / (4 * math.pi)


def x_of(gamma, eta, nu):
    return nu / (a_B(eta) * gamma**2)


def nu_of(gamma, eta, x):
    return x * a_B(eta) * gamma**2


def K_iso(gamma, eta, nu):
    x = x_of(gamma, eta, nu)
    return amp(eta) * F_ref(x), -amp(eta) * G_ref(x)


# ============================================================ pitch densities
def pdf_eval(pdf, mu):
    mu = np.asarray(mu, dtype=float)
    k = pdf[0]
    if k == "iso":
        return np.full_like(mu, 0.5)
    if k == "kp":  # (mu^2 - mu_c^2)_+^a
        d = (np.abs(mu) - pdf[1]) * (np.abs(mu) + pdf[1])
        return np.where(d > 0, np.abs(d) ** pdf[2], 0.0)
    if k == "kpN":  # scaled KP: N(g, mu) = g^-p (1 - g (2 - mu^2)/(2 Gam))_+^a
        g, Gam = pdf[1], pdf[2]
        v = 1.0 - g * (2.0 - mu * mu) / (2.0 * Gam)
        return g**-P_INJ * np.where(v > 0, np.abs(v) ** A_KP, 0.0)
    if k == "P2":
        return 0.25 * (1.0 + 3.0 * mu * mu)
    if k == "exp":
        return np.exp(mu)
    if k == "sin4":
        return (1.0 - mu * mu) ** 2
    if k == "kps":  # (mu^2 + kappa^2)^a
        return (mu * mu + pdf[1]) ** pdf[2]
    raise ValueError(pdf)


def dlogs(pdf, mu):
    """Analytic (p'/p, p''/p) at mu."""
    k = pdf[0]
    if k == "iso":
        return 0.0, 0.0
    if k == "P2":
        return 6 * mu / (1 + 3 * mu * mu), 6 / (1 + 3 * mu * mu)
    if k == "exp":
        return 1.0, 1.0
    if k == "sin4":
        return -4 * mu / (1 - mu * mu), (12 * mu * mu - 4) / (1 - mu * mu) ** 2
    if k in ("kps", "kp"):
        a = pdf[2]
        s = mu * mu + (pdf[1] if k == "kps" else -pdf[1] ** 2)
        return 2 * a * mu / s, 2 * a / s + 4 * a * (a - 1) * mu * mu / s**2
    raise ValueError(pdf)


# ============================================================ harmonic sum (own)
def window_W(x):
    """Half-window in units of 1/gamma: classical profile exponent 2 xi >= 80, plus 3."""
    umax = math.sqrt(max((80.0 / x) ** (2.0 / 3.0) - 1.0, 0.0))
    return max(6.0, umax + 3.0)


def harmonic_sum(gamma, eta, nu, pdf, method="scipy", W=None):
    """(j_I, j_Q, n_lines) at monochromatic nu, lines with |alpha - theta| <= W/gamma."""
    beta = math.sqrt((1 - 1 / gamma) * (1 + 1 / gamma))
    nuB = 1.0 / (2 * math.pi * gamma)
    st = math.sqrt((1 - eta) * (1 + eta))
    if W is None:
        W = window_W(x_of(gamma, eta, nu))
    th = math.acos(eta)
    mu_hi = math.cos(max(0.0, th - W / gamma))
    mu_lo = math.cos(min(math.pi, th + W / gamma))
    m_lo = max(1, math.floor(nu * (1 - beta * eta * mu_hi) / nuB))
    m_hi = math.ceil(nu * (1 - beta * eta * mu_lo) / nuB)
    sI, sQ = [], []
    for s0 in range(m_lo, m_hi + 1, CHUNK):
        m = np.arange(s0, min(s0 + CHUNK, m_hi + 1), dtype=float)
        D = m * nuB / nu
        mu = (1 - D) / (beta * eta)
        ok = np.abs(mu) < 1
        m, D, mu = m[ok], D[ok], mu[ok]
        dpar = (D - st * st) / eta  # = eta - beta mu, without cancellation
        bperp = beta * np.sqrt((1 - mu) * (1 + mu))
        z = bperp * st / D
        if method == "scipy":
            X = m * z
            J = special.jv(m, X)
            Jp = special.jv(m - 1, X) - (m / X) * J
        elif method == "cubic":  # radio-fit/stable_lines.py large-order branch
            omz = (((st / gamma) ** 2 + dpar**2) / (D * D)) / (1 + z)
            scale = np.cbrt(2 / (m * z))
            arg = m * omz * scale
            ai, aip, _, _ = special.airy(np.minimum(arg, 10.0))
            keep = (arg <= 10.0) & (z >= 0.9)
            J = np.where(keep, scale * ai, 0.0)
            Jp = np.where(keep, -scale**2 * aip, 0.0)
        else:
            raise ValueError(method)
        C2 = m * m / (2 * math.pi * gamma**2 * D**3)
        Ap = (dpar / st) * J
        Ak = bperp * Jp
        w = pdf_eval(pdf, mu) * D * D / (m * nuB * beta * eta) * C2
        sI.append(float(np.sum(w * (Ap * Ap + Ak * Ak))))
        sQ.append(float(np.sum(w * (Ap * Ap - Ak * Ak))))
    return math.fsum(sI), math.fsum(sQ), m_hi - m_lo + 1


def worker(task):
    key, gamma, eta, nu, pdf, method, W = task
    jI, jQ, n = harmonic_sum(gamma, eta, nu, pdf, method, W)
    return key, jI, jQ, n


def task_cost(t):
    _, gamma, eta, nu, _, _, W = t
    return nu * 2 * math.pi * gamma * eta * 2 * (W or 12) / gamma


# ============================================================ classical angular profile (own)
def prof(u, x, S="I"):
    u = np.asarray(u, dtype=float)
    q = 1 + u * u
    xi = 0.5 * x * q**1.5
    k23, k13 = special.kve(2 / 3, xi), special.kve(1 / 3, xi)
    sg = 1.0 if S == "I" else -1.0
    return q * q * np.exp(-2 * xi) * (k23**2 + sg * (u * u / q) * k13**2)


U_MAX = 40.0


def prof_int(fun, lo, hi, pts=()):
    edges = sorted({lo, hi, *[p for p in pts if lo < p < hi]})
    return sum(integrate.quad(fun, a, b, epsabs=0, epsrel=1e-12, limit=400)[0]
               for a, b in zip(edges[:-1], edges[1:]))


_norm_cache = {}


def prof_norm(x, S):
    if (x, S) not in _norm_cache:
        _norm_cache[(x, S)] = 2 * prof_int(lambda u: prof(u, x, S), 0, U_MAX, (1, 3, 8))
    return _norm_cache[(x, S)]


def u2_mean(x, S="I"):
    return 2 * prof_int(lambda u: u * u * prof(u, x, S), 0, U_MAX, (1, 3, 8)) / prof_norm(x, S)


def e_edge(k, x, S="I", a=A_KP):
    """int Phihat(u) (k+u)_+^a du, via t = k+u = s^b, b = 1/(1+a): t^a dt = b ds."""
    if k <= -U_MAX:
        return 0.0
    b = 1.0 / (1.0 + a)
    tmax = k + U_MAX
    tb = [t for t in (k - 3, k - 1, k, k + 1, k + 3, k + 8) if 0 < t < tmax]
    sb = [t ** (1 + a) for t in tb]
    val = prof_int(lambda s: prof(s**b - k, x, S), 0.0, tmax ** (1 + a), sb)
    return b * val / prof_norm(x, S)


def f_edge(k, x, S="I"):
    return e_edge(k, x, S) / k**A_KP - 1.0


# ============================================================ task builders
def eta_of_k(gamma, k):
    return math.cos(THETA_C - k / gamma)


def phase_nus(gamma, eta, x, nph):
    beta = math.sqrt((1 - 1 / gamma) * (1 + 1 / gamma))
    nuB = 1.0 / (2 * math.pi * gamma)
    period = nuB / (1 - beta * eta * eta)  # one line spacing at mu = eta
    nu0 = nu_of(gamma, eta, x)
    return [nu0 + period * j / nph for j in range(nph)]


KP = ("kp", MU_C, A_KP)
ISO = ("iso",)
SMOOTH = (ISO, ("P2",), ("exp",), ("sin4",), ("kps", 0.1, A_KP))
XS = (0.1, 1.0, 3.0)
CUBIC_ETAS = (0.1, 0.3, 0.5)
CUBIC_XS = (0.01, 0.1, 1.0, 3.0)


def build_point_tasks():
    tasks = []
    # fixed eta (B1)
    for g in (20, 40, 80, 160):
        for eta in (0.5, 0.35):
            for x in XS:
                for pdf, nph in ((KP, 4), (ISO, 2)):
                    for j, nu in enumerate(phase_nus(g, eta, x, nph)):
                        tasks.append((("fix", g, eta, x, pdf[0], j), g, eta, nu, pdf, "scipy", None))
    for pdf in (KP, ISO):
        nu = phase_nus(1000, 0.5, 0.1, 1)[0]
        tasks.append((("fix", 1000, 0.5, 0.1, pdf[0], 0), 1000, 0.5, nu, pdf, "scipy", None))
    # near edge (B2)
    for g, nph in ((20, 8), (80, 8), (160, 4), (320, 2)):
        for k in (0.0, 1.0, 4.0):
            eta = eta_of_k(g, k)
            for x in XS:
                for j, nu in enumerate(phase_nus(g, eta, x, nph)):
                    tasks.append((("edge", g, k, x, j), g, eta, nu, KP, "scipy", None))
    for k in (0.0, 1.0, 4.0):
        eta = eta_of_k(1000, k)
        tasks.append((("edge", 1000, k, 0.1, 0), 1000, eta, phase_nus(1000, eta, 0.1, 1)[0], KP,
                      "scipy", None))
    # smooth-density structure (C) at gamma = 80
    for eta in (0.5, 0.35):
        for x in XS:
            for pdf in SMOOTH[1:]:
                for j, nu in enumerate(phase_nus(80, eta, x, 2)):
                    tasks.append((("smooth", 80, eta, x, pdf[0], j), 80, eta, nu, pdf, "scipy", None))
    # cubic-Airy branch (G), same nu as phase 0 of the iso fixed-eta points
    for g in (40, 80, 160):
        for x in XS:
            nu = phase_nus(g, 0.5, x, 2)[0]
            tasks.append((("cubic", g, x), g, 0.5, nu, ISO, "cubic", None))
    tasks.append((("cubic", 1000, 0.1), 1000, 0.5, phase_nus(1000, 0.5, 0.1, 1)[0], ISO, "cubic",
                  None))
    # cubic-Airy shift versus eta and x (relevance to the eta-averaged c(x) of bridge.py)
    for g in (80, 160):
        for eta in CUBIC_ETAS:
            for x in CUBIC_XS:
                nu = nu_of(g, eta, x)
                for meth in ("scipy", "cubic"):
                    tasks.append((("cubeta", g, eta, x, meth), g, eta, nu, ISO, meth, None))
    # window checks
    for g in (20, 80):
        for x in XS:
            for lab, eta, pdf in (("k=1", eta_of_k(g, 1.0), KP), ("eta=0.5", 0.5, ISO)):
                nu = phase_nus(g, eta, x, 1)[0]
                W = window_W(x_of(g, eta, nu))
                tasks.append((("win", g, x, lab, "W"), g, eta, nu, pdf, "scipy", W))
                tasks.append((("win", g, x, lab, "1.6W"), g, eta, nu, pdf, "scipy", 1.6 * W))
    return tasks


ETA_LO, ETA_HI = 0.1, 0.5


def q_w(eta, kind):
    eta = np.asarray(eta, float)
    if kind == "uniform":
        return np.full_like(eta, 1 / (ETA_HI - ETA_LO))
    return np.sin(np.pi * (eta - ETA_LO) / (ETA_HI - ETA_LO)) ** 2 / (0.5 * (ETA_HI - ETA_LO))


def eta_grid(gamma, width, n):
    t, w = leggauss(n)
    nodes, wts = [], []
    for a, b in ((ETA_LO, MU_C), (MU_C, ETA_HI)):
        npan = max(1, math.ceil((b - a) * gamma / width))
        e = np.linspace(a, b, npan + 1)
        for lo, hi in zip(e[:-1], e[1:]):
            nodes += list(0.5 * (hi - lo) * t + 0.5 * (hi + lo))
            wts += list(0.5 * (hi - lo) * w)
    return np.array(nodes), np.array(wts)


E_CASES = [(20, 1.0, (0.4, 10)), (20, 1.0, (0.2, 10)), (40, 1.0, (0.4, 10)), (80, 1.0, (0.4, 10)),
           (20, 3.0, (0.4, 10)), (40, 3.0, (0.4, 10))]


def build_e_tasks():
    tasks, meta = [], []
    for g, x, res in E_CASES:
        nu = nu_of(g, 0.5, x)
        nodes, wts = eta_grid(g, *res)
        pdfs = (KP, ISO) if res == (0.4, 10) and g <= 40 else (KP,)
        for pdf in pdfs:
            meta.append((g, x, res, pdf, nodes, wts, nu))
            for i, e in enumerate(nodes):
                tasks.append((("E", g, x, res, pdf[0], i), g, float(e), nu, pdf, "scipy", None))
    return tasks, meta


ETA_F = 0.5


def gamma_grid(Gam, width, n):
    gc = 2 * Gam / (2 - ETA_F**2)
    band = 6.0
    pieces = [(Gam / 2, gc - band, 3, 16)]
    nf = int(round(2 * band / width))
    pieces += [(gc - band + i * width, gc - band + (i + 1) * width, 1, n) for i in range(nf)]
    pieces += [(gc + band, 2 * Gam, 2, 16)]
    t, w = {}, {}
    nodes, wts = [], []
    for lo, hi, npan, nn in pieces:
        tt, ww = leggauss(nn)
        e = np.linspace(lo, hi, npan + 1)
        for a, b in zip(e[:-1], e[1:]):
            nodes += list(0.5 * (b - a) * tt + 0.5 * (b + a))
            wts += list(0.5 * (b - a) * ww)
    return np.array(nodes), np.array(wts), gc


F_CASES = [(20, 1.0, (0.2, 8)), (20, 1.0, (0.1, 8)), (40, 1.0, (0.2, 8)), (20, 3.0, (0.2, 8))]


def build_f_tasks():
    tasks, meta = [], []
    for Gam, x, res in F_CASES:
        nodes, wts, gc = gamma_grid(Gam, *res)
        nu = nu_of(gc, ETA_F, x)  # x defined at gamma_c(eta); physical nu fixed
        meta.append((Gam, x, res, nodes, wts, gc, nu))
        for i, g in enumerate(nodes):
            tasks.append((("F", Gam, x, res, i), float(g), ETA_F, nu, ("kpN", float(g), float(Gam)),
                          "scipy", None))
    return tasks, meta


def run(pool, tasks):
    order = sorted(tasks, key=lambda t: -task_cost(t))
    return {r[0]: r[1:] for r in pool.imap_unordered(worker, order, chunksize=1)}


# ============================================================ analysis
def summarise_points(res):
    out = {"fixed_eta": {}, "edge": {}, "smooth_fit": {}, "cubic": {}, "window": {}}
    # fixed eta
    for g in (20, 40, 80, 160, 1000):
        for eta in (0.5, 0.35):
            for x in XS:
                for pk, pdf, nph in (("kp", KP, 4), ("iso", ISO, 2)):
                    nph = 1 if g == 1000 else nph
                    keys = [("fix", g, eta, x, pk, j) for j in range(nph)]
                    if keys[0] not in res:
                        continue
                    nus = phase_nus(g, eta, x, nph)
                    d = []
                    for kk, nu in zip(keys, nus):
                        KI, KQ = K_iso(g, eta, nu)
                        pe = float(pdf_eval(pdf, eta))
                        d.append(res[kk][0] / (2 * pe * KI) - 1)
                    out["fixed_eta"][f"g={g},eta={eta},x={x},{pk}"] = {
                        "g2d_I": g * g * float(np.mean(d)),
                        "phase_spread_g2d": g * g * float(np.ptp(d)), "n_lines": res[keys[0]][2]}
    # near edge
    Z = 2 * integrate.quad(lambda m: (m + MU_C) ** A_KP, MU_C, 1, weight="alg", wvar=(A_KP, 0),
                           epsabs=0, epsrel=1e-13)[0]
    out["Z_kp"] = Z
    for g, nph in ((20, 8), (80, 8), (160, 4), (320, 2), (1000, 1)):
        for k in (0.0, 1.0, 4.0):
            eta = eta_of_k(g, k)
            for x in XS:
                keys = [("edge", g, k, x, j) for j in range(nph)]
                if keys[0] not in res:
                    continue
                nus = phase_nus(g, eta, x, nph)
                rI, rQ = [], []
                for kk, nu in zip(keys, nus):
                    KI, KQ = K_iso(g, eta, nu)
                    rI.append(res[kk][0] / KI)
                    rQ.append(res[kk][1] / KQ)
                pe = float(pdf_eval(KP, eta))
                pg = float(pdf_eval(KP, math.cos(THETA_C - 1 / g)))
                row = {"n_phases": nph, "n_lines": res[keys[0]][2],
                       "e_I": float(np.mean(rI)) / (2 * pg), "e_Q": float(np.mean(rQ)) / (2 * pg),
                       "g06_d_iso_I": g**0.6 * float(np.mean(rI) - 2 * pe) / Z}
                if pe > 0:
                    dI = np.array(rI) / (2 * pe) - 1
                    dQ = np.array(rQ) / (2 * pe) - 1
                    row.update(d_I=float(dI.mean()), d_Q=float(dQ.mean()),
                               phase_spread_I=float(np.ptp(dI)))
                out["edge"][f"g={g},k={k},x={x}"] = row
    # smooth-density structure fit at gamma = 80
    for eta in (0.5, 0.35):
        for x in XS:
            A, y = [], []
            for pdf in SMOOTH:
                if pdf[0] == "iso":
                    v = out["fixed_eta"][f"g=80,eta={eta},x={x},iso"]["g2d_I"]
                else:
                    nus = phase_nus(80, eta, x, 2)
                    ds = []
                    for j, nu in enumerate(nus):
                        KI, _ = K_iso(80, eta, nu)
                        ds.append(res[("smooth", 80, eta, x, pdf[0], j)][0]
                                  / (2 * float(pdf_eval(pdf, eta)) * KI) - 1)
                    v = 6400 * float(np.mean(ds))
                A.append([1.0, *dlogs(pdf, eta)])
                y.append(v)
            A, y = np.array(A), np.array(y)
            c, *_ = np.linalg.lstsq(A, y, rcond=None)
            l1, l2 = dlogs(KP, eta)
            kp_h = out["fixed_eta"][f"g=80,eta={eta},x={x},kp"]["g2d_I"]
            out["smooth_fit"][f"eta={eta},x={x}"] = {
                "c": c.tolist(), "max_abs_resid": float(np.max(np.abs(y - A @ c))),
                "c2_profile": u2_mean(x) * (1 - eta * eta) / 2,
                "kp_pred": float(c @ [1, l1, l2]), "kp_harm": kp_h,
                "kp_rel_diff": float(c @ [1, l1, l2]) / kp_h - 1}
    # cubic
    for g, x in [(g, x) for g in (40, 80, 160) for x in XS] + [(1000, 0.1)]:
        ref = res[("fix", g, 0.5, x, "iso", 0)]
        cub = res[("cubic", g, x)]
        out["cubic"][f"g={g},x={x}"] = {"g2_I": g * g * (cub[0] / ref[0] - 1),
                                        "g2_Q": g * g * (cub[1] / ref[1] - 1)}
    out["cubic_eta"] = {}
    for g in (80, 160):
        for eta in CUBIC_ETAS:
            for x in CUBIC_XS:
                ex, cu = res[("cubeta", g, eta, x, "scipy")], res[("cubeta", g, eta, x, "cubic")]
                KI, _ = K_iso(g, eta, nu_of(g, eta, x))
                out["cubic_eta"][f"g={g},eta={eta},x={x}"] = {
                    "g2d_true_I": g * g * (ex[0] / KI - 1), "g2_cubic_shift_I": g * g * (cu[0] / ex[0] - 1)}
    # window
    for g in (20, 80):
        for x in XS:
            for lab in ("k=1", "eta=0.5"):
                a, b = res[("win", g, x, lab, "W")], res[("win", g, x, lab, "1.6W")]
                out["window"][f"g={g},x={x},{lab}"] = abs(b[0] / a[0] - 1)
    return out


def summarise_e(meta, res):
    out = []
    for g, x, resl, pdf, nodes, wts, nu in meta:
        jI = np.array([res[("E", g, x, resl, pdf[0], i)][0] for i in range(len(nodes))])
        KIn = np.array([K_iso(g, float(e), nu)[0] for e in nodes])
        cI = 2 * pdf_eval(pdf, nodes) * KIn
        for kind in ("uniform", "smooth"):
            q = q_w(nodes, kind)
            H = float(np.sum(wts * q * jI))

            def cf(e):
                return float(q_w(e, kind)) * 2 * K_iso(g, e, nu)[0]

            if pdf[0] == "kp":
                C = integrate.quad(lambda e: cf(e) * (e + MU_C) ** A_KP, MU_C, ETA_HI,
                                   weight="alg", wvar=(A_KP, 0), epsabs=0, epsrel=1e-12,
                                   limit=400)[0]
            else:
                C = integrate.quad(lambda e: 0.5 * cf(e), ETA_LO, ETA_HI, epsabs=0, epsrel=1e-12,
                                   limit=400)[0]
            out.append({"gamma": g, "x": x, "panel": list(resl), "pdf": pdf[0], "q": kind,
                        "g2_rel_I": g * g * (H / C - 1),
                        "L1_rel_I": float(np.sum(wts * q * np.abs(jI - cI))) / C})
    return out


def summarise_f(meta, res):
    out = []
    for Gam, x, resl, nodes, wts, gc, nu in meta:
        H = float(np.sum(wts * np.array([res[("F", Gam, x, resl, i)][0]
                                         for i in range(len(nodes))])))

        def cf(g):
            return 2 * g**-P_INJ * gc**-A_KP * K_iso(g, ETA_F, nu)[0]

        C = integrate.quad(cf, Gam / 2, gc, weight="alg", wvar=(0, A_KP), epsabs=0, epsrel=1e-12,
                           limit=400)[0]
        out.append({"Gamma": Gam, "x": x, "panel": list(resl), "n_nodes": len(nodes),
                    "G2_rel_I": Gam**2 * (H / C - 1)})
    return out


def profile_checks():
    out = {"norm_ratio": {}, "u2": {}, "f": {}, "e0": {}, "thr_1pct": {}, "thr_01pct": {}}
    for x in (0.01, 0.1, 1.0, 3.0, 10.0):
        out["norm_ratio"][f"x={x}"] = [prof_norm(x, "I") * x * x / F_ref(x),
                                       prof_norm(x, "Q") * x * x / G_ref(x),
                                       2 * math.pi / math.sqrt(3)]
    for x in XS:
        out["u2"][f"x={x}"] = [u2_mean(x, "I"), u2_mean(x, "Q")]
        for k in (0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            out["f"][f"x={x},k={k}"] = [f_edge(k, x, "I"), f_edge(k, x, "Q")]
        out["e0"][f"x={x}"] = e_edge(0.0, x, "I")
        # largest k with |f_I| = eps (f_I < 0 and increasing towards 0 beyond its minimum)
        ks = np.geomspace(0.3, 300, 300)
        fk = np.array([f_edge(k, x) for k in ks])
        for eps, lab in ((1e-2, "thr_1pct"), (1e-3, "thr_01pct")):
            i = np.nonzero(np.abs(fk) >= eps)[0][-1]
            root = optimize.brentq(lambda k: abs(f_edge(k, x)) - eps, ks[i], ks[i + 1], xtol=1e-6)
            out[lab][f"x={x}"] = root
    return out


def physical():
    sigma_T, me, c = 6.6524587321e-25, 9.1093837139e-28, 2.99792458e10
    ev, myr = 1.602176634e-12, 3.15576e13
    coeff = 4 * sigma_T / (3 * me * c)
    b2 = (6e-6**3 - 4e-6**3) / (3 * 2e-6)  # <B^2> for U[4,6] muG
    b_s = coeff * b2 / (8 * math.pi)
    b_c = coeff * 1.0 * ev
    t = 19 * myr

    def gc(mu):
        return 1 / ((1.5 * b_s * (1 - mu * mu) + b_c) * t)

    def mu_c(g):
        v = 1 - (1 / (g * t) - b_c) / (1.5 * b_s)
        return math.sqrt(v) if 0 <= v <= 1 else float("nan")

    g_half = optimize.brentq(lambda g: mu_c(g) - 0.5, gc(0) * 1.0001, gc(1) * 0.9999)
    # check of the KP edge form at gamma = g_half:  1 - b(mu) g t = 1.5 b_s g t (mu^2 - mu_c^2)
    mus = np.linspace(0.5, 1, 6)
    lhs = 1 - (1.5 * b_s * (1 - mus**2) + b_c) * g_half * t
    rhs = 1.5 * b_s * g_half * t * (mus**2 - 0.25)
    return {"b_s": b_s, "b_c": b_c, "gamma_c(0)": gc(0), "gamma_c(1/2)": gc(0.5),
            "gamma_c(1)": gc(1), "gamma_mu_c_half": g_half, "mu_c(16500)": mu_c(16500),
            "mu_c(30000)": mu_c(30000), "edge_form_max_abs_diff": float(np.max(np.abs(lhs - rhs)))}


def kernels_spot_check():
    sys.path.insert(0, str(HERE / "lib"))
    import kernels as K  # noqa: E402

    xs = np.array([1e-3, 0.1, 1.0, 3.0, 10.0])
    return {"x": xs.tolist(),
            "F_rel": [float(K.F(np.array([x]))[0] / F_ref(x) - 1) for x in xs],
            "G_rel": [float(K.G(np.array([x]))[0] / G_ref(x) - 1) for x in xs]}


# ============================================================ consistency of author JSON
def author_checks():
    R = json.load(open(HERE / "kp_bridge_results.json"))
    out = {}
    comp = R["D_profile"]["harm_vs_pred"]
    for S in ("I", "Q"):
        rows = [r for r in comp if r["S"] == S and r["bessel"] == "olver" and 0 < r["k"] <= 4 + 1e-9]
        out[f"max_abs_gamma_resid_{S}"] = max(abs(r["gamma_x_resid"]) for r in rows)
    sp = {}
    for p in R["B_points"]:
        if p["pdf"][0] == "kp_edge" and p["label"].startswith("k=") and "phase_spread_I" in p \
                and p["bessel"] == "olver" and p["n_phases"] > 1:
            sp.setdefault(p["gamma"], []).append(p["phase_spread_I"])
    gs = sorted(sp)
    mx = [max(sp[g]) for g in gs]
    out["phase_spread_max"] = dict(zip(map(str, gs), mx))
    out["phase_spread_slope"] = float(np.polyfit(np.log(gs), np.log(mx), 1)[0])
    E = [r for r in R["E_eta_integrated"] if r["pdf"] == "kp_edge" and r["q"] == "uniform"
         and r["x_at_eta_half"] == 1.0 and r["panel_width_gamma"] == 0.5]
    E = sorted(E, key=lambda r: r["gamma"])
    l1 = [r["L1_rel_I_node_quadrature"] for r in E]
    out["L1_kp_uniform_x1"] = l1
    out["L1_slope"] = float(np.polyfit(np.log([r["gamma"] for r in E]), np.log(l1), 1)[0])
    C = R["C_smooth_fit"]["fits"]
    out["C_max_fit_resid"] = max(max(abs(v) for v in f["fit_resid"]) for f in C.values())
    out["C_kp_pred_vs_harm_g160"] = {
        k: f["kp"]["fit_pred_g2d"] / f["kp"]["harm_g2d"] - 1 for k, f in C.items()
        if k.startswith("gamma=160")}
    out["C_c2_fit_vs_profile"] = {k: [f["c0_c1_c2"][2], f["c2_profile_pred"]] for k, f in C.items()}
    so = R["A_checks"]["scipy_vs_olver_max_g2_abs_diff_d_iso_I"]
    out["scipy_vs_olver_abs_d"] = {k: v / float(k.split("=")[1]) ** 2 for k, v in so.items()}
    Fg = R["F_gamma_integrated"]
    out["F_kp_over_iso"] = {f"G={r['Gamma']},x={r['x_at_gamma_c_eta']}":
                            r["G2_rel_I"] / s["G2_rel_I"] for r in Fg for s in Fg
                            if r["kind"] == "kp" and s["kind"] == "jp_iso" and r["Gamma"] == s["Gamma"]
                            and r["x_at_gamma_c_eta"] == s["x_at_gamma_c_eta"]
                            and r["panel_width"] == s["panel_width"] == 0.25}
    # table entries pulled from JSON
    pick = {}
    for p in R["B_points"]:
        if p["bessel"] != "olver":
            continue
        tag = (p["gamma"], p["x"], p["label"], p["pdf"][0])
        pick[tag] = p
    out["json_fixed_eta_g2d"] = {f"g={g},x={x},{lab},{pk}": pick[(g, x, lab, pk)]["g2_d_rel_I"]
                                 for g in (20, 80, 1000) for x in XS for lab in ("eta=0.5", "eta=0.35")
                                 for pk in ("kp_edge", "iso") if (g, x, lab, pk) in pick}
    out["json_edge_d"] = {f"g={g},x={x},k={k}": pick[(g, x, f"k={k}", "kp_edge")]["d_rel_I"]
                          for g in (20, 80, 1000) for x in XS for k in (1, 4)}
    out["json_edge_e0"] = {f"g={g},x={x}": pick[(g, x, "k=0", "kp_edge")]["e_harm_I"]
                           for g in (20, 80, 1000) for x in XS}
    out["json_g06_diso_k0"] = {f"g={g},x={x}": g**0.6 * pick[(g, x, "k=0", "kp_edge")]["d_iso_I"]
                               for g in (20, 80, 1000) for x in XS}
    P = R["D_profile"]["prediction"]
    out["json_f"] = {f"x={x:g},k={k}": P[f"x={x:g},I"]["f_pred"][P[f"x={x:g},I"]["k"].index(k)]
                     for x in XS for k in (1, 4)}
    out["json_u2"] = {f"x={x:g}": P[f"x={x:g},I"]["u2"] for x in XS}
    out["json_thr"] = R["D_profile"]["k_threshold_I"]
    out["json_E"] = [(r["gamma"], r["x_at_eta_half"], r["pdf"], r["q"], r["panel_width_gamma"],
                      r["g2_rel_I"]) for r in R["E_eta_integrated"]]
    out["json_F"] = [(r["Gamma"], r["x_at_gamma_c_eta"], r["kind"], r["panel_width"], r["G2_rel_I"])
                     for r in Fg]
    out["json_G"] = [(r["x"], r["label"], r["pdf"], r["g2_cubic_over_olver_minus1_I"])
                     for r in R["G_cubic_airy"]["sums"]]
    out["json_H"] = R["H_physical"]
    return out


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=24)
    args = ap.parse_args()
    results = {"kernels_spot_check": kernels_spot_check(), "physical": physical()}
    print("spot/physical", time.time() - T0, flush=True)
    ctx = get_context("spawn")
    with ctx.Pool(args.workers) as pool:
        pt = build_point_tasks()
        et, emeta = build_e_tasks()
        ft, fmeta = build_f_tasks()
        print("tasks", len(pt), len(et), len(ft), flush=True)
        res = run(pool, pt + et + ft)
    print("harmonic sums", time.time() - T0, flush=True)
    results["points"] = summarise_points(res)
    results["eta_integrated"] = summarise_e(emeta, res)
    results["gamma_integrated"] = summarise_f(fmeta, res)
    print("summaries", time.time() - T0, flush=True)
    results["profile"] = profile_checks()
    print("profile", time.time() - T0, flush=True)
    results["author_json_checks"] = author_checks()
    results["provenance"] = {
        "script": Path(__file__).name, "script_sha256": sha(__file__),
        "inputs_sha256": {"kp_bridge_results.json": sha(HERE / "kp_bridge_results.json"),
                          "lib/kernels.py": sha(HERE / "lib/kernels.py")},
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__},
        "workers": args.workers, "wall_clock_s": time.time() - T0,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    with open(HERE / "verify_kpbridge_results.json", "w") as fh:
        json.dump(results, fh, indent=1, default=float)
    print("done", time.time() - T0)


if __name__ == "__main__":
    main()
