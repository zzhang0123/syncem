"""KP harmonic bridge: exact harmonic sums versus the beaming limit 2 p(eta) K^iso for the
Kardashev-Pacholczyk pitch density, whose conditional form at fixed gamma has an edge.

Physics [derived]: for KP, 1 - b(mu) gamma t = 1.5 b_s gamma t (mu^2 - mu_c^2) above
gamma_c(mu=0), so p(mu | gamma) is proportional to (mu^2 - mu_c(gamma)^2)_+^{p-2} for ANY
(b_s, b_c, t); only mu_c and p = 2.6 matter.  We therefore hold mu_c = 0.25 fixed and vary
gamma in the radiation kernel (the KP shape at the energy where mu_c(gamma) = 0.25).

Parts (all harmonic sums are exact integer-harmonic sums at one frequency, kp_harmonic.py):
  A  checks: Olver vs mpmath, B0/C0 fit, scipy vs Olver, window, profile normalisation,
     regression against check_beaming.json of the earlier beaming-limit survey
     (inputs/beaming-survey).
  B  pointwise (gamma, eta, x) grid, eta = cos(theta_c - k/gamma) near the edge and two
     fixed eta; frequency-phase average over one line spacing.
  C  smooth-density structure fit gamma^2 d = c0 + c1 p'/p + c2 p''/p at eta = 1/2.
  D  leading-order near-edge prediction from the classical angular profile.
  E  eta-integrated (line-of-sight average) test; F gamma-integrated ordered-field KP test.
  G  accuracy of the manuscript pipeline's cubic-Airy line branch (stable_lines.py).
Writes kp_bridge_results.json.  Run: python kp_bridge.py [--workers N]
"""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import argparse
import json
import math
import sys
import time
from functools import lru_cache
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate

HERE = Path(__file__).resolve().parent
for p in (str(HERE), str(HERE / "lib")):
    if p not in sys.path:
        sys.path.insert(0, p)
import kp_harmonic as H  # noqa: E402
from provenance import provenance, write_json  # noqa: E402

T0 = time.time()
CHECK_BEAMING = HERE / "inputs" / "beaming-survey" / "check_beaming.json"
STABLE_LINES = HERE / "inputs" / "harmonic-radio-fit" / "stable_lines.py"

A_KP = 0.6  # p_inj - 2 with p_inj = 2.6 (E4)
MU_C = 0.25
KP = ("kp_edge", MU_C, A_KP)
ISO = ("iso",)
K_LIST = (-3.0, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)
ETA_FIXED = (0.35, 0.5)
X_LIST = (0.1, 1.0, 3.0)
W_DEFAULT = 25.0


# ------------------------------------------------------------------ worker
def _init_worker():
    for p in (str(HERE), str(HERE / "lib")):
        if p not in sys.path:
            sys.path.insert(0, p)


def harm_task(task):
    """task = (key, gamma, eta, nu, pdf, bessel, W) -> (key, I, Q, V, n, KI, KQ, x)."""
    key, gamma, eta, nu, pdf, bessel, W = task
    I, Q, V, n = H.harmonic_j(gamma, eta, nu, pdf, bessel, W)
    KI, KQ, x = H.continuum(gamma, eta, nu)
    return key, I, Q, V, n, KI, KQ, x


def cost(task):
    _, gamma, eta, nu, _, bessel, W = task
    st2 = 1.0 - eta * eta
    return nu * eta * 2.0 * math.sqrt(st2) * W / gamma * (2.0 if bessel == "scipy" else 1.0)


def run_tasks(pool, tasks):
    order = sorted(range(len(tasks)), key=lambda i: -cost(tasks[i]))
    res = pool.map(harm_task, [tasks[i] for i in order], chunksize=1)
    return {r[0]: r[1:] for r in res}


# ------------------------------------------------------------------ geometry helpers
THETA_C = math.acos(MU_C)


def eta_of_k(gamma, k):
    return math.cos(THETA_C - k / gamma)


def n_phases(gamma):
    return 8 if gamma <= 80 else (4 if gamma <= 160 else (2 if gamma <= 320 else 1))


def phase_nus(gamma, eta, x, nph):
    beta = math.sqrt(1.0 - 1.0 / gamma**2)
    period = 1.0 / (1.0 - beta * eta * eta)  # shifts the mu comb by one line spacing
    nu0 = H.nu_for_x(gamma, eta, x)
    return [nu0 + period * j / nph for j in range(nph)]


# ------------------------------------------------------------------ B: pointwise grid
def point_specs():
    """List of (label, gamma, eta, x, pdf, bessel, nph)."""
    specs = []
    for gamma in (20, 40, 80, 160, 320, 1000):
        methods = ("scipy", "olver") if gamma <= 80 else ("olver",)
        etas = [(f"k={k:g}", eta_of_k(gamma, k)) for k in K_LIST]
        etas += [(f"eta={e:g}", e) for e in ETA_FIXED]
        for x in X_LIST:
            for bes in methods:
                for lab, eta in etas:
                    for pdf in (KP, ISO):
                        specs.append((lab, gamma, eta, x, pdf, bes, n_phases(gamma)))
                specs.append(("eta=0.5", gamma, 0.5, x, ("kp_smooth", 0.1, A_KP), bes,
                              n_phases(gamma)))
    # scipy cross-check at gamma = 1000 (orders ~1.3e8 < AMOS limit) and window check
    for lab, k in (("k=0", 0.0), ("k=1", 1.0), ("k=4", 4.0)):
        specs.append((lab, 1000, eta_of_k(1000, k), 0.1, KP, "scipy", 1))
    specs.append(("eta=0.5", 1000, 0.5, 0.1, ISO, "scipy", 1))
    return specs


def point_tasks(specs, W=W_DEFAULT, tag="B"):
    tasks = []
    for i, (lab, gamma, eta, x, pdf, bes, nph) in enumerate(specs):
        for j, nu in enumerate(phase_nus(gamma, eta, x, nph)):
            tasks.append(((tag, i, j), gamma, eta, nu, pdf, bes, W))
    return tasks


def summarise_point(spec, rows):
    lab, gamma, eta, x, pdf, bes, nph = spec
    pe = float(H.pdf_value(pdf, eta))
    Z = H.pdf_norm(pdf)
    rI = [r[0] / r[4] for r in rows]  # j_I / K_I (no pitch factor)
    rQ = [r[1] / r[5] for r in rows]
    out = {"label": lab, "gamma": gamma, "eta": eta, "x": x, "pdf": list(pdf), "bessel": bes,
           "n_phases": nph, "n_lines": rows[0][3], "p_eta": pe, "Z": Z,
           "gammaV_over_I": float(np.mean([gamma * r[2] / r[0] for r in rows])),
           "jI_over_KI": float(np.mean(rI)), "jQ_over_KQ": float(np.mean(rQ))}
    if pdf[0] == "kp_edge":
        p_g = float(H.pdf_value(pdf, math.cos(THETA_C - 1.0 / gamma)))
        out["k"] = gamma * (THETA_C - math.acos(eta))
        out["p_gamma"] = p_g
        out["e_harm_I"] = float(np.mean(rI)) / (2 * p_g)
        out["e_harm_Q"] = float(np.mean(rQ)) / (2 * p_g)
        out["e_cont"] = pe / p_g
    out["d_iso_I"] = float(np.mean(rI) - 2 * pe) / Z
    out["d_iso_Q"] = float(np.mean(rQ) - 2 * pe) / Z
    if pe > 0:
        dI = np.array(rI) / (2 * pe) - 1
        dQ = np.array(rQ) / (2 * pe) - 1
        out.update(d_rel_I=float(dI.mean()), d_rel_Q=float(dQ.mean()),
                   g2_d_rel_I=float(gamma**2 * dI.mean()), g2_d_rel_Q=float(gamma**2 * dQ.mean()),
                   phase_spread_I=float(dI.max() - dI.min()),
                   phase_spread_Q=float(dQ.max() - dQ.min()))
    return out


# ------------------------------------------------------------------ D: profile prediction
@lru_cache(maxsize=64)
def profile_norm(x, stokes):
    return integrate.quad(lambda u: H.beaming_profile(u, x, stokes), -60, 60, points=[0],
                          epsabs=0, epsrel=1e-12, limit=400)[0]


def e_pred(k, x, stokes="I", a=A_KP):
    """int Phihat(u) (k + u)_+^a du: leading-order edge-smoothed density / C gamma^-a."""
    U = 60.0

    def f(u):
        return H.beaming_profile(u, x, stokes)

    if k <= -U:
        return 0.0
    if k < U:  # algebraic weight (u + k)^a on [-k, U]
        val = integrate.quad(f, -k, U, weight="alg", wvar=(a, 0.0), epsabs=0, epsrel=1e-11,
                             limit=400)[0]
    else:
        val = integrate.quad(lambda u: f(u) * (k + u) ** a, -U, U, points=[0], epsabs=0,
                             epsrel=1e-12, limit=400)[0]
    return val / profile_norm(x, stokes)


def profile_moment(x, n, stokes="I"):
    return integrate.quad(lambda u: u**n * H.beaming_profile(u, x, stokes), -60, 60,
                          points=[0], epsabs=0, epsrel=1e-12, limit=400)[0] / profile_norm(x, stokes)


def part_d(points):
    k_grid = [-3, -2, -1.5, -1, -0.5, 0, 0.25, 0.5, 1, 1.5, 2, 3, 4, 6, 8, 12, 16, 24, 32]
    pred = {}
    for x in X_LIST:
        for S in ("I", "Q"):
            e = [e_pred(k, x, S) for k in k_grid]
            f = [ei / k**A_KP - 1 if k > 0 else None for ei, k in zip(e, k_grid)]
            pred[f"x={x:g},{S}"] = {"k": k_grid, "e_pred": e, "f_pred": f,
                                    "u2": profile_moment(x, 2, S)}
    # thresholds: smallest k beyond which |f| < eps (scan a fine grid)
    thr = {}
    for x in X_LIST:
        ks = np.geomspace(0.05, 200, 400)
        fI = np.array([e_pred(k, x, "I") / k**A_KP - 1 for k in ks])
        for eps in (1e-1, 1e-2, 1e-3):
            bad = np.nonzero(np.abs(fI) >= eps)[0]
            thr[f"x={x:g},eps={eps:g}"] = float(ks[bad[-1] + 1]) if bad.size else float(ks[0])
    comp = []
    for r in points:
        if r["pdf"][0] != "kp_edge" or not r["label"].startswith("k="):
            continue
        for S in ("I", "Q"):
            k = r["k"]
            ep = e_pred(k, r["x"], S)
            eh = r[f"e_harm_{S}"]  # j_Q / K_Q > 0 (both carry the sign of -G)
            row = {"S": S, "gamma": r["gamma"], "x": r["x"], "k": k, "bessel": r["bessel"],
                   "e_harm": eh, "e_pred": ep, "e_cont": r["e_cont"]}
            if k > 0:
                row["d_harm"] = eh / r["e_cont"] - 1
                row["d_pred"] = ep / k**A_KP - 1
                row["gamma_x_resid"] = r["gamma"] * (row["d_harm"] - row["d_pred"])
            else:
                row["gamma_x_resid_e"] = r["gamma"] * (eh - ep)
            comp.append(row)
    return {"prediction": pred, "k_threshold_I": thr, "harm_vs_pred": comp}


# ------------------------------------------------------------------ C: smooth structure fit
SMOOTH_PDFS = (ISO, ("P2",), ("exp",), ("sin4",), ("kp_smooth", 0.1, A_KP))


def log_derivs(pdf, mu, h=1e-4):
    f = lambda m: float(H.pdf_value(pdf, m))  # noqa: E731
    p0, pp, pm = f(mu), f(mu + h), f(mu - h)
    return (pp - pm) / (2 * h * p0), (pp - 2 * p0 + pm) / (h * h * p0)


def part_c_specs():
    specs = []
    for gamma in (80, 160):
        for x in X_LIST:
            for e in ETA_FIXED:
                for pdf in SMOOTH_PDFS:
                    specs.append((f"eta={e:g}", gamma, e, x, pdf, "olver", n_phases(gamma)))
    return specs


def part_c(rows_c, points):
    """Fit gamma^2 d = c0 + c1 p'/p + c2 p''/p over 5 smooth densities at each fixed eta and
    compare with the KP edge density at the same eta (mu-derivatives)."""
    def find(pname, gamma, x, e):
        rr = [p for p in points if p["pdf"][0] == pname and p["gamma"] == gamma and p["x"] == x
              and abs(p["eta"] - e) < 1e-12 and p["bessel"] == "olver"]
        return rr[0]["g2_d_rel_I"] if rr else None

    out = {}
    for gamma, x, e in ((g, x, e) for g in (80, 160) for x in X_LIST for e in ETA_FIXED):
        sel = [r for r in rows_c if r["gamma"] == gamma and r["x"] == x
               and abs(r["eta"] - e) < 1e-12]
        Amat = np.array([[1.0, *log_derivs(tuple(r["pdf"]), e)] for r in sel])
        y = np.array([r["g2_d_rel_I"] for r in sel])
        coef, *_ = np.linalg.lstsq(Amat, y, rcond=None)
        l1, l2 = log_derivs(KP, e)
        u2 = profile_moment(x, 2, "I")
        out[f"gamma={gamma},x={x:g},eta={e:g}"] = {
            "pdfs": [r["pdf"] for r in sel], "g2d": y.tolist(),
            "c0_c1_c2": coef.tolist(), "fit_resid": (y - Amat @ coef).tolist(),
            "c2_profile_pred": u2 * (1 - e * e) / 2.0, "c1_profile_part": -u2 * e / 2.0,
            "kp": {"dlnp": l1, "d2p_over_p": l2, "harm_g2d": find("kp_edge", gamma, x, e),
                   "iso_g2d_same_eta": find("iso", gamma, x, e),
                   "fit_pred_g2d": float(coef @ [1.0, l1, l2])}}
    return out


# ------------------------------------------------------------------ E: eta-integrated
ETA_LO, ETA_HI = 0.1, 0.5


def q_weight(eta, kind):
    if kind == "uniform":
        return np.full_like(np.asarray(eta, float), 1.0 / (ETA_HI - ETA_LO))
    return np.sin(np.pi * (np.asarray(eta) - ETA_LO) / (ETA_HI - ETA_LO)) ** 2 / (
        0.5 * (ETA_HI - ETA_LO))


def eta_nodes(gamma, per_width=0.5, n=8):
    """GL panels of width <= per_width/gamma on [ETA_LO, MU_C] and [MU_C, ETA_HI]."""
    t, w = leggauss(n)
    nodes, wts = [], []
    for a, b in ((ETA_LO, MU_C), (MU_C, ETA_HI)):
        npan = max(1, math.ceil((b - a) * gamma / per_width))
        edges = np.linspace(a, b, npan + 1)
        for lo, hi in zip(edges[:-1], edges[1:]):
            nodes.extend(0.5 * (hi - lo) * t + 0.5 * (hi + lo))
            wts.extend(0.5 * (hi - lo) * w)
    return np.array(nodes), np.array(wts)


def part_e_tasks():
    tasks, meta = [], []
    for gamma in (20, 40, 80):
        for x in (1.0, 3.0):
            nu = H.nu_for_x(gamma, 0.5, x)
            for res in ((0.5, 8), (0.25, 8)) if gamma == 40 else ((0.5, 8),):
                nodes, wts = eta_nodes(gamma, *res)
                for pdf in (KP, ISO):
                    meta.append((gamma, x, nu, res, pdf, nodes, wts))
                    for i, e in enumerate(nodes):
                        tasks.append((("E", gamma, x, res, pdf[0], i), gamma, float(e), nu, pdf,
                                      "scipy", W_DEFAULT))
    return tasks, meta


def cont_eta_integral(gamma, nu, pdf, kind):
    def g(e):
        KI, KQ, _ = H.continuum(gamma, e, nu)
        return q_weight(e, kind) * 2.0 * np.array([KI, KQ])

    out = []
    for comp in (0, 1):
        if pdf[0] == "kp_edge":
            f = lambda e: g(e)[comp] * (e + MU_C) ** A_KP  # noqa: E731
            val = integrate.quad(f, MU_C, ETA_HI, weight="alg", wvar=(A_KP, 0.0), epsabs=0,
                                 epsrel=1e-12, limit=400)[0]
        else:
            val = integrate.quad(lambda e: g(e)[comp] * 0.5, ETA_LO, ETA_HI, epsabs=0,
                                 epsrel=1e-12, limit=400)[0]
        out.append(val)
    return out


def part_e(meta, res_map):
    out = []
    for gamma, x, nu, resl, pdf, nodes, wts in meta:
        rows = [res_map[("E", gamma, x, resl, pdf[0], i)] for i in range(len(nodes))]
        jI = np.array([r[0] for r in rows])
        jQ = np.array([r[1] for r in rows])
        cI = np.array([2 * H.pdf_value(pdf, e) * r[4] for e, r in zip(nodes, rows)])
        for kind in ("uniform", "smooth"):
            q = q_weight(nodes, kind)
            CI, CQ = cont_eta_integral(gamma, nu, pdf, kind)
            HI, HQ = float(np.sum(wts * q * jI)), float(np.sum(wts * q * jQ))
            l1 = float(np.sum(wts * q * np.abs(jI - cI))) / CI
            out.append({"gamma": gamma, "x_at_eta_half": x, "pdf": pdf[0], "q": kind,
                        "panel_width_gamma": resl[0], "n_nodes": len(nodes),
                        "rel_I": HI / CI - 1, "rel_Q": HQ / CQ - 1,
                        "g2_rel_I": gamma**2 * (HI / CI - 1), "g2_rel_Q": gamma**2 * (HQ / CQ - 1),
                        "L1_rel_I_node_quadrature": l1})
    return out


# ------------------------------------------------------------------ F: gamma-integrated
ETA_F = 0.5


def kp_scaled(gamma, Gam, mu_obs=None):
    """Scaled KP: gamma_c(mu=0) = Gam, gamma_c(1) = 2 Gam; returns (pdf, amplitude)."""
    amp = gamma**-2.6 * (gamma / (2.0 * Gam)) ** A_KP
    if gamma >= Gam:
        return ("kp_edge", math.sqrt(max(0.0, 2.0 - 2.0 * Gam / gamma)), A_KP), amp
    return ("kp_smooth", 2.0 * Gam / gamma - 2.0, A_KP), amp


def gamma_nodes(Gam, band=8.0, width=0.25, n=6, jp=False):
    """GL nodes on [Gam/2, 2 Gam] (KP), fine panels of `width` within +-band of gamma_c(eta).
    jp=True: nodes on [Gam/2, gc] only, last panel Gauss-Jacobi with weight (gc - gamma)^a;
    returns flag gj (True where the weight already contains (gc - gamma)^a)."""
    from scipy.special import roots_jacobi

    gc = 2.0 * Gam / (2.0 - ETA_F**2)
    nfine = int(round(band / width))
    pieces = [(Gam / 2.0, gc - band, 40)]
    edges = gc - band + width * np.arange(2 * nfine + 1)
    pieces += [(lo, hi, n) for lo, hi in zip(edges[:-1], edges[1:])]
    pieces += [(gc + band, 2.0 * Gam, 16)]
    if jp:
        pieces = [pc for pc in pieces if pc[1] <= gc - 0.5 * width]
    nodes, wts, gj = [], [], []
    for lo, hi, nn in pieces:
        tt, ww = leggauss(nn)
        nodes.extend(0.5 * (hi - lo) * tt + 0.5 * (hi + lo))
        wts.extend(0.5 * (hi - lo) * ww)
        gj.extend([False] * nn)
    if jp:
        tt, ww = roots_jacobi(n, A_KP, 0.0)
        h = 0.5 * width
        nodes.extend(gc - h + h * tt)
        wts.extend(h ** (1.0 + A_KP) * ww)
        gj.extend([True] * n)
    return np.array(nodes), np.array(wts), gc, np.array(gj)


def part_f_tasks():
    tasks, meta = [], []
    for Gam in (20, 40, 80):
        for x in (0.3, 1.0, 3.0):
            for res in ((0.25, 6), (0.125, 6)) if (Gam == 40 and x == 1.0) else ((0.25, 6),):
                for kind in ("kp", "jp_iso"):
                    nodes, wts, gc, gj = gamma_nodes(Gam, width=res[0], n=res[1],
                                                     jp=(kind == "jp_iso"))
                    nut = x * 1.5 * gc**2 * math.sqrt(1 - ETA_F**2)  # nu / nu_0
                    meta.append((Gam, x, res, kind, nodes, wts, gc, gj, nut))
                    for i, g in enumerate(nodes):
                        pdf = kp_scaled(g, Gam)[0] if kind == "kp" else ISO
                        tasks.append((("F", Gam, x, res, kind, i), float(g), ETA_F, nut * g, pdf,
                                      "olver", W_DEFAULT))
    return tasks, meta


def part_f(meta, res_map):
    """Harmonic and continuum int dgamma N(gamma, mu) K / gamma at fixed eta (ordered field).
    kp: N = gamma^-p (1 - b(mu) gamma t)_+^a (scaled KP);  jp_iso: the same energy profile
    along mu = eta, N = gamma^-p (1 - gamma/gc)_+^a, but isotropic pitch (no pitch edge)."""
    out = []
    for Gam, x, res, kind, nodes, wts, gc, gj, nut in meta:
        HI = HQ = 0.0
        for i, g in enumerate(nodes):
            r = res_map[("F", Gam, x, res, kind, i)]
            if kind == "kp":
                amp = kp_scaled(g, Gam)[1]
            else:  # iso pdf returns 1/2: amplitude 2 N(gamma) gives density N(gamma) for all mu
                shape = gc**-A_KP if gj[i] else max(0.0, 1.0 - g / gc) ** A_KP
                amp = 2.0 * g**-2.6 * shape
            HI += wts[i] * amp * r[0] / g
            HQ += wts[i] * amp * r[1] / g
        CI, CQ = cont_gamma_integral(Gam, nut, gc)
        out.append({"Gamma": Gam, "x_at_gamma_c_eta": x, "kind": kind, "gamma_c_eta": gc,
                    "panel_width": res[0], "n_nodes": len(nodes),
                    "rel_I": HI / CI - 1, "rel_Q": HQ / CQ - 1,
                    "G2_rel_I": Gam**2 * (HI / CI - 1), "G2_rel_Q": Gam**2 * (HQ / CQ - 1)})
    return out


def cont_gamma_integral(Gam, nut, gc):
    """int dgamma N(gamma, eta) 2 K^iso / gamma, N = gamma^-p (1 - gamma/gc)_+^a (both kinds)."""
    def f(g, comp):
        KI, KQ, _ = H.continuum(g, ETA_F, nut * g)
        return g**-2.6 * gc**-A_KP * 2.0 * (KI, KQ)[comp] / g

    vals = []
    for comp in (0, 1):
        v = integrate.quad(lambda g: f(g, comp), Gam / 2.0, gc, weight="alg", wvar=(0.0, A_KP),
                           epsabs=0, epsrel=1e-12, limit=400)[0]
        vals.append(v)
    return vals


# ------------------------------------------------------------------ A: checks
def check_olver_mpmath():
    import mpmath as mp

    mp.mp.dps = 40
    rows = []
    for nu in (100, 300, 1000):
        for z in (0.95, 0.99, 0.999):
            try:
                J = mp.besselj(nu, nu * mp.mpf(z))
                Jp = mp.besselj(nu, nu * mp.mpf(z), derivative=1)
            except ValueError:
                continue
            s = 1.0 - z * z
            Jo, Jpo = H.bessel_olver(np.array([float(nu)]), np.array([s]), np.array([z]))
            rows.append({"m": nu, "z": z, "relerr_J": float(Jo[0] / J - 1),
                         "relerr_Jp": float(Jpo[0] / Jp - 1)})
    return rows


def check_b0c0_fit():
    import mpmath as mp

    mp.mp.dps = 60
    worst = [0.0, 0.0, 0.0]
    for s in np.geomspace(1e-9, 0.5, 60):
        w = mp.sqrt(mp.mpf(s))
        zeta = (mp.mpf(3) / 2 * (mp.atanh(w) - w)) ** (mp.mpf(2) / 3)
        b = -5 / (48 * zeta**2) + zeta ** mp.mpf(-0.5) * (5 / (24 * w**3) - 1 / (8 * w))
        c = 7 / (48 * zeta) + zeta ** mp.mpf(0.5) * (-7 / (24 * w**3) + 3 / (8 * w))
        zf = H._zeta_from_w(np.array([math.sqrt(s)]))
        bf, cf = H.b0c0(np.array([s]), zf)
        worst[0] = max(worst[0], abs(float(bf[0] / b - 1)))
        worst[1] = max(worst[1], abs(float(cf[0] / c - 1)))
        worst[2] = max(worst[2], abs(float(zf[0] / zeta - 1)))
    return {"s_range": [1e-9, 0.5], "max_relerr_B0": worst[0], "max_relerr_C0": worst[1],
            "max_relerr_zeta": worst[2]}


def check_profile_norm():
    import kernels as K

    rows = []
    for x in (0.01, 0.1, 1.0, 3.0, 10.0):
        rows.append({"x": x, "intPhiI_x2_over_F": profile_norm(x, "I") * x**2 / float(K.F(np.array([x]))[0]),
                     "intPhiQ_x2_over_G": profile_norm(x, "Q") * x**2 / float(K.G(np.array([x]))[0])})
    return {"rows": rows, "expected": 2 * math.pi / math.sqrt(3)}


def check_regression(pool):
    old = json.loads(CHECK_BEAMING.read_text())
    eta = math.cos(old["theta"])
    pdfs = {"iso": ISO, "P2_a1": ("P2",), "exp_k1": ("exp",), "sin4": ("sin4",)}
    tasks = []
    for c in old["cases"]:
        nu = H.nu_for_x(c["gamma"], eta, c["x"])
        tasks.append(((c["pdf"], c["x"], c["gamma"]), c["gamma"], eta, nu, pdfs[c["pdf"]],
                      "scipy", 25.0))
    res = run_tasks(pool, tasks)
    worst = 0.0
    for c in old["cases"]:
        r = res[(c["pdf"], c["x"], c["gamma"])]
        pe = float(H.pdf_value(pdfs[c["pdf"]], eta))
        worst = max(worst, abs(r[0] / (2 * pe * r[4]) - 1 - c["dI"]))
    return {"n_cases": len(old["cases"]), "max_abs_diff_dI": worst}


def check_window(pool):
    specs = [(f"k={k:g}", 40, eta_of_k(40, k), 0.1, KP, "scipy", 1) for k in (-1.5, 0.0, 1.0, 4.0)]
    a = run_tasks(pool, point_tasks(specs, 25.0, "W25"))
    b = run_tasks(pool, point_tasks(specs, 40.0, "W40"))
    return [{"label": s[0], "rel_change_I_W25_vs_W40":
             a[("W25", i, 0)][0] / b[("W40", i, 0)][0] - 1} for i, s in enumerate(specs)]


def check_methods(points):
    """scipy vs olver differences in d_rel / e_harm at gamma <= 1000."""
    key = lambda r: (r["label"], r["gamma"], r["x"], r["pdf"][0])  # noqa: E731
    sc = {key(r): r for r in points if r["bessel"] == "scipy"}
    worst = {}
    for r in points:
        if r["bessel"] != "olver" or key(r) not in sc:
            continue
        s = sc[key(r)]
        if r["n_phases"] != s["n_phases"]:
            continue
        v = abs(r["d_iso_I"] - s["d_iso_I"]) * r["gamma"] ** 2
        g = r["gamma"]
        worst[g] = max(worst.get(g, 0.0), v)
    return {f"gamma={g}": v for g, v in sorted(worst.items())}


# ------------------------------------------------------------------ G: cubic-Airy branch
def check_cubic_branch(pool):
    from syncem_compat import use_syncem  # noqa: PLC0415

    use_syncem()  # registers syncem under the name stable_lines imports; adds STABLE_LINES.parent to sys.path
    import stable_lines as SL  # noqa: E402

    gamma, eta, B = 1000.0, 0.5, 5e-6
    rows = []
    for dk in (0.0, 2.0, 5.0):
        mu = math.cos(math.acos(eta) - dk / gamma)
        nu = H.nu_for_x(gamma, eta, 1.0)
        m0 = round(nu * (1 - math.sqrt(1 - gamma**-2) * eta * mu))
        orders = np.array([2.0e5, 5.0e5, m0 * 0.5, m0, m0 * 1.5]).round()
        Is, Qs, _, _ = SL.stable_high_order_line_powers(orders, gamma, B, mu, eta)
        Io, Qo, _ = H.line_powers(orders, gamma, mu, eta, "olver")
        Ic, Qc, _ = H.line_powers(orders, gamma, mu, eta, "cubic_airy")
        unit = Is[0] / Io[0]  # orders < 1e6 use scipy jv in stable_lines
        rows.append({"dk": dk, "orders": orders.tolist(),
                     "stable_over_olver_minus1": (Is / Io / unit - 1).tolist(),
                     "stable_over_mycubic_minus1": (Is / Ic / unit - 1).tolist(),
                     "unit_consistency_m=5e5": float(Is[1] / Io[1] / unit - 1)})
    sums = []
    tasks = []
    for x in X_LIST:
        for lab, e in (("eta=0.5", 0.5), ("k=1", eta_of_k(1000, 1.0)), ("k=4", eta_of_k(1000, 4.0))):
            for pdf in (ISO, KP) if lab == "eta=0.5" else (KP,):
                nu = H.nu_for_x(1000, e, x)
                tasks.append((("G", x, lab, pdf[0]), 1000.0, e, nu, pdf, "cubic_airy", W_DEFAULT))
    res = run_tasks(pool, tasks)
    for (tag, x, lab, pname), r in res.items():
        sums.append({"x": x, "label": lab, "pdf": pname, "I": r[0], "Q": r[1], "KI": r[4],
                     "KQ": r[5]})
    return {"lines": rows, "sums": sums}


# ------------------------------------------------------------------ H: physical E4-KP map
def physical_map(thr):
    import ensembles as E

    ag = E.Ageing()
    out = {"ageing": ag.describe(), "mu_c_of_gamma": {}}
    for g in (16500, 17000, 18767, 22000, 26000, 30000):
        mc = float(ag.mu_at_cut(g))
        out["mu_c_of_gamma"][str(g)] = None if math.isnan(mc) else mc
    g = 18767.0
    out["band_halfwidth_rad_at_gamma_18767"] = {k: v / g for k, v in thr.items()}
    return out


# ------------------------------------------------------------------ main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=26)
    args = ap.parse_args()
    results = {"config": {"mu_c": MU_C, "a": A_KP, "theta_c": THETA_C, "k_list": K_LIST,
                          "eta_fixed": ETA_FIXED, "x_list": X_LIST, "W": W_DEFAULT,
                          "units": "e=c=1, nu_B = eB/(2 pi gamma m c) = 1",
                          "k_definition": "eta = cos(theta_c - k/gamma), theta_c = acos(mu_c)"}}
    ctx = get_context("spawn")
    with ctx.Pool(args.workers, initializer=_init_worker) as pool:
        t = time.time()
        results["A_checks"] = {"olver_vs_mpmath": check_olver_mpmath(),
                               "b0c0_fit": check_b0c0_fit(),
                               "profile_norm": check_profile_norm(),
                               "regression_check_beaming": check_regression(pool),
                               "window": check_window(pool)}
        print("checks", time.time() - t, flush=True)
        specs = point_specs() + part_c_specs()
        res = run_tasks(pool, point_tasks(specs))
        print("points", time.time() - t, flush=True)
        allrows = [summarise_point(s, [res[("B", i, j)] for j in range(s[6])])
                   for i, s in enumerate(specs)]
        nB = len(point_specs())
        points, rows_c = allrows[:nB], allrows[nB:]
        results["B_points"] = points
        results["A_checks"]["scipy_vs_olver_max_g2_abs_diff_d_iso_I"] = check_methods(points)
        results["C_smooth_fit"] = {"rows": rows_c, "fits": part_c(rows_c, points)}
        results["D_profile"] = part_d(points)
        tasks_e, meta_e = part_e_tasks()
        results["E_eta_integrated"] = part_e(meta_e, run_tasks(pool, tasks_e))
        print("E", time.time() - t, flush=True)
        tasks_f, meta_f = part_f_tasks()
        results["F_gamma_integrated"] = part_f(meta_f, run_tasks(pool, tasks_f))
        print("F", time.time() - t, flush=True)
        results["G_cubic_airy"] = check_cubic_branch(pool)
        for srow in results["G_cubic_airy"]["sums"]:
            ref = [p for p in points if p["gamma"] == 1000 and p["bessel"] == "olver"
                   and p["x"] == srow["x"] and p["label"] == srow["label"]
                   and p["pdf"][0] == srow["pdf"]][0]
            srow["g2_cubic_over_olver_minus1_I"] = 1e6 * (srow["I"] / srow["KI"] / ref["jI_over_KI"] - 1)
            srow["g2_cubic_over_olver_minus1_Q"] = 1e6 * (srow["Q"] / srow["KQ"] / ref["jQ_over_KQ"] - 1)
            srow["g2_olver_d_rel_I"] = ref.get("g2_d_rel_I")
        print("G", time.time() - t, flush=True)
    results["H_physical"] = physical_map(results["D_profile"]["k_threshold_I"])
    results["provenance"] = provenance(__file__, T0, inputs=[HERE / "kp_harmonic.py",
                                                             CHECK_BEAMING, STABLE_LINES,
                                                             HERE / "lib" / "syncem_compat.py"])
    results["provenance"]["mpmath"] = __import__("mpmath").__version__
    results["provenance"]["workers"] = args.workers
    write_json(HERE / "kp_bridge_results.json", results)
    print("done", time.time() - T0, flush=True)


if __name__ == "__main__":
    main()
