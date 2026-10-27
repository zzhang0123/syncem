"""Independent verifier for the params part (params.py), continuum kernels only.

Everything is rebuilt here: channels, field/eta/azimuth/screen averages, bins,
reference metric, ensembles, moment basis, UPRE rank, compressed and full fits.  The
author's analysis code (params.py, paramlib.py, params_degeneracy.py, lib/basis.py,
lib/config.py, lib/ensembles.py) is NOT imported.  lib/kernels.py is imported only for
F(x) and G(x), after a spot check against scipy.special.kv + quad.

Independent choices (deliberately different from the author's):
  * kernel ln-derivatives (x d/dx)^r W from Bessel identities generated symbolically
    (D(x^m K_nu) = (m - nu) x^m K_nu - x^{m+1} K_{|nu-1|}, DF = F - x^2 K_{5/3}),
    each step checked against a finite difference;
  * field and eta folded into one product rule (24 x 16 nodes for U[4,6]; 128 x 16 for
    the Maxwellian), channel response by 320-node Gauss-Legendre;
  * Taylor columns from the analytic derivative tables; moments by 48-node GL per bin and
    Gauss-Jacobi on the JP/KP cut-off panel.
Also checks the author's params_results.json / params_realisations.npz for internal
consistency (pulls, coverage, chi2 vs dof, R* claims, sd ratios).

Writes verify_params_results.json.  Run: python verify_params.py
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import sys
import time
from fractions import Fraction
from math import factorial
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np  # noqa: E402
import scipy  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402
from scipy import integrate, special  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402
from scipy.special import roots_jacobi  # noqa: E402

T0 = time.time()
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import kernels as _K  # noqa: E402  (F and G only)

N_REAL_E1 = int(os.environ.get("VP_NREAL_E1", "300"))
N_REAL_JP = int(os.environ.get("VP_NREAL_JP", "200"))
N_REAL_E2 = int(os.environ.get("VP_NREAL_E2", "200"))
OUT = {}

# ----------------------------------------------------------------------------- constants (cgs)
E_ESU, M_E, C_CGS, C_SI = 4.803204712570263e-10, 9.1093837139e-28, 2.99792458e10, 2.99792458e8
SIGMA_T, EV, MYR, MUG = 6.6524587321e-25, 1.602176634e-12, 3.15576e13, 1e-6
A1 = np.sqrt(3.0) * E_ESU**3 * MUG / (4 * np.pi * M_E * C_CGS**2)
a1 = 3.0 * E_ESU * MUG / (4 * np.pi * M_E * C_CGS)
LOSS = 4 * SIGMA_T / (3 * M_E * C_CGS)
GAMMA0 = 1e4
SEED = 20260928
XMAX = 690.0


def tic(label, t):
    OUT.setdefault("timings_s", {})[label] = round(time.time() - t, 2)


# ----------------------------------------------------------------------------- 1. F, G spot check
def F_ref(x):
    f = lambda v: special.kv(5.0 / 3.0, x * np.exp(v)) * x * np.exp(v)  # noqa: E731
    vmax = np.log(max(900.0 / x, 2.0)) + 1.0
    val, _ = integrate.quad(f, 0.0, vmax, epsabs=0, epsrel=1e-13, limit=800)
    return x * val


def spot_check():
    xs = [1e-3, 0.1, 1.0, 3.0, 20.0]
    rows = []
    for x in xs:
        fr, gr = F_ref(x), x * special.kv(2.0 / 3.0, x)
        fk, gk = float(_K.F(np.array([x]))[0]), float(_K.G(np.array([x]))[0])
        rows.append({"x": x, "F_kernels": fk, "F_scipy_quad": fr, "relF": abs(fk / fr - 1),
                     "G_kernels": gk, "G_scipy": gr, "relG": abs(gk / gr - 1)})
    return rows


# ----------------------------------------------------------------------------- 2. derivatives
def dpoly(order, which):
    """[(cF, {(m, nu): c})] for r = 0..order: D^r W = cF F + sum c x^m K_nu, D = x d/dx."""
    cF = Fraction(1) if which == "F" else Fraction(0)
    terms = {} if which == "F" else {(1, Fraction(2, 3)): Fraction(1)}
    out = [(cF, dict(terms))]
    for _ in range(order):
        new = {}

        def add(key, v):
            new[key] = new.get(key, Fraction(0)) + v
        if cF:
            add((2, Fraction(5, 3)), -cF)
        for (m, nu), c in terms.items():
            add((m, nu), c * (m - nu))
            add((m + 1, abs(nu - 1)), -c)
        terms = {k: v for k, v in new.items() if v != 0}
        out.append((cF, dict(terms)))
    return out


class OneD:
    """ln F on a fine ln x grid (from kernels.F); Bessel terms evaluated exactly."""

    def __init__(self, step=5e-4):
        lx = np.arange(np.log(1e-6), np.log(XMAX) + step, step)
        self.lx_lo, self.lx_hi = lx[0], lx[-1]
        self.lnF = CubicSpline(lx, np.log(_K.F(np.exp(lx))))
        rng = np.random.default_rng(3)
        xt = np.exp(rng.uniform(np.log(1e-6), np.log(XMAX) - 0.01, 3000))
        self.spline_check = float(np.max(np.abs(np.exp(self.lnF(np.log(xt))) / _K.F(xt) - 1)))

    def dW(self, x, order, which):
        """List r = 0..order of D^r W(x); zero where x > XMAX."""
        x = np.asarray(x, float)
        ok = x <= XMAX
        xx = np.where(ok, x, 1.0)
        lx = np.log(xx)
        Fv = np.exp(self.lnF(np.clip(lx, self.lx_lo, self.lx_hi))) if which == "F" else None
        bes = {nu: special.kve(float(nu), xx) for nu in (Fraction(5, 3), Fraction(2, 3), Fraction(1, 3))}
        out = []
        for cF, terms in dpoly(order, which):
            v = float(cF) * Fv if cF else np.zeros_like(xx)
            for (m, nu), c in terms.items():
                v = v + float(c) * np.exp(m * lx - xx) * bes[nu]
            out.append(np.where(ok, v, 0.0))
        return out


def derivative_fd_check(od):
    """D^r W vs a 5-point difference (in ln x) of the analytic D^{r-1} W, r = 1..4."""
    res = {}
    d = 1e-3
    for which in ("F", "G"):
        for x in (1e-3, 0.3, 2.0, 15.0):
            lx = np.log(x)
            pts = np.exp(lx + d * np.array([-2, -1, 1, 2]))
            vals = od.dW(pts, 4, which)
            cen = od.dW(np.array([x]), 4, which)
            for r in range(1, 5):
                v = vals[r - 1]
                fd = (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * d)
                scale = max(abs(cen[r][0]), abs(cen[0][0]))
                res[f"{which}_x{x}_r{r}"] = float(abs(fd - cen[r][0]) / scale)
    return max(res.values()), res


# ----------------------------------------------------------------------------- 3. geometry
NT = int(os.environ.get("VP_NT", "320"))
CENTRES = np.geomspace(1e8, 1e10, 41)
FIT, HELD = np.arange(0, 41, 2), np.arange(1, 41, 2)
FACTOR = 0.4 * np.exp(0.5j)


def rho1(t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    ok = np.abs(t) < 1
    out[ok] = np.exp(-t[ok] ** 2 / (1 - t[ok] ** 2))
    return out


Z1 = integrate.quad(lambda t: float(rho1(np.array(t))), -1, 1, epsabs=0, epsrel=1e-13, limit=500)[0]


class Channels:
    def __init__(self, nt=NT):
        t, w = leggauss(nt)
        self.w = w * rho1(t) / Z1
        self.nu = CENTRES[:, None] * (1.0 + 0.2 * t[None, :])
        self.lnnu = np.log(self.nu)
        lam2 = (C_SI / self.nu) ** 2
        self.wscr = self.w[None, :] * np.exp(2j * lam2 * 8.0 - 2.0 * lam2**2 * 0.5**2)


def gl(n, lo, hi):
    t, w = leggauss(n)
    return 0.5 * (hi - lo) * t + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


ETA, WETA = gl(16, 0.0, 0.5)
WETA = WETA / 0.5
SETA = np.sqrt(1 - ETA**2)


def field_nodes(kind):
    if kind == "U46":
        b, w = gl(24, 4.0, 6.0)
        return b, w / 2.0
    if kind == "maxwell":
        sig = 5.0 / np.sqrt(3.0)
        b, w = gl(128, 0.0, 30.0)
        return b, w * np.sqrt(2 / np.pi) * b**2 / sig**3 * np.exp(-b**2 / (2 * sig**2))
    raise ValueError(kind)


class Table:
    """H_W^{(r)}(s) = sum_k pref_k D^r W(e^s / c_k), s = ln(nu / gamma^2), splined in s."""

    def __init__(self, od, b, wb, eta_s, weta, order, step, s_lo=-5.6, s_hi=12.0):
        c = a1 * (b[:, None] * eta_s[None, :]).ravel()
        pref = (wb[:, None] * weta[None, :] * A1 * b[:, None] * eta_s[None, :]).ravel()
        s = np.arange(s_lo, s_hi + step, step)
        self.s_lo, self.order = s[0], order
        self.lnH, self.rho, self.smax = {}, {}, {}
        for which in ("F", "G"):
            H = np.zeros((order + 1, s.size))
            for i in range(0, s.size, 400):
                x = np.exp(s[i:i + 400])[:, None] / c[None, :]
                vals = od.dW(x, order, which)
                for r in range(order + 1):
                    H[r, i:i + 400] = vals[r] @ pref
            good = H[0] > 1e-300
            last = int(np.max(np.nonzero(good)[0]))
            self.smax[which] = s[last]
            ss = s[:last + 1]
            self.lnH[which] = CubicSpline(ss, np.log(H[0, :last + 1]))
            self.rho[which] = [None] + [CubicSpline(ss, H[r, :last + 1] / H[0, :last + 1]) for r in range(1, order + 1)]

    def H(self, which, s, order):
        if np.any(s < self.s_lo):
            raise ValueError("s below table")
        sm = self.smax[which]
        ok = s <= sm
        sc = np.minimum(s, sm)
        h0 = np.where(ok, np.exp(self.lnH[which](sc)), 0.0)
        return [h0] + [h0 * self.rho[which][r](sc) for r in range(1, order + 1)]


def kbar(tab, ch, gammas, order=0, ln_seta=0.0, seta=1.0):
    """d^r/d(ln gamma)^r of channel-averaged K_I (real) and K_P (complex, factor and screen included)."""
    g = np.atleast_1d(np.asarray(gammas, float))
    KI = np.zeros((order + 1, g.size, 41))
    KP = np.zeros((order + 1, g.size, 41), complex)
    per = max(1, int(3e6 // ch.lnnu.size))
    for i in range(0, g.size, per):
        s = ch.lnnu[None] - 2 * np.log(g[i:i + per])[:, None, None] - ln_seta
        hf, hg = tab.H("F", s, order), tab.H("G", s, order)
        for r in range(order + 1):
            fac = (-2.0) ** r * seta
            KI[r, i:i + per] = fac * np.einsum("gjt,t->gj", hf[r], ch.w)
            KP[r, i:i + per] = -FACTOR * fac * np.einsum("gjt,jt->gj", hg[r], ch.wscr)
    return KI, KP


def stokes(kI, kP):
    return np.stack([kI, kP.real, kP.imag], axis=-1)


# ----------------------------------------------------------------------------- 4. bins, metric
LE = np.linspace(np.log(300.0), np.log(1e5), 13)
EDGES = np.exp(LE)
H_B = 0.5 * (LE[1] - LE[0])
GB = np.exp(0.5 * (LE[1:] + LE[:-1]))
WB = np.diff(EDGES ** (-1.6)) / (-1.6)
WB = WB / WB.sum()


def basis(tab, ch, N):
    """C (41, 3, 12 (N+1)) and the raw D_r^I, D_r^P (N+1, 12, 41)."""
    KI, KP = kbar(tab, ch, GB, N)
    fac = np.array([H_B**r / factorial(r) for r in range(N + 1)])[:, None, None]
    DI, DP = KI * fac, KP * fac
    C = np.zeros((41, 3, 12 * (N + 1)))
    for b in range(12):
        for r in range(N + 1):
            C[:, :, b * (N + 1) + r] = WB[b] * stokes(DI[r, b], DP[r, b])
    return C, DI, DP


# ----------------------------------------------------------------------------- 5. measures
XG48, WG48 = leggauss(48)


def smooth_nodes(n=48, lo=300.0, hi=1e5):
    """GL in ln gamma per bin (restricted to [lo, hi] within bins); (gamma, w, bin, x_b)."""
    t, w = leggauss(n)
    G, W, Bn, X = [], [], [], []
    for b in range(12):
        ua, ub = max(LE[b], np.log(lo)), min(LE[b + 1], np.log(hi))
        if ub <= ua:
            continue
        u = 0.5 * (ub - ua) * t + 0.5 * (ub + ua)
        g = np.exp(u)
        G.append(g)
        W.append(0.5 * (ub - ua) * w * g)
        Bn.append(np.full(n, b))
        X.append((u - np.log(GB[b])) / H_B)
    return np.concatenate(G), np.concatenate(W), np.concatenate(Bn), np.concatenate(X)


def _gj(a, gc, alpha, n):
    t, w = roots_jacobi(n, alpha, 0.0)
    g = a + (gc - a) * (1 + t) / 2
    return g, w * ((gc - a) / (2 * gc)) ** alpha * (gc - a) / 2


def cut_nodes(gc, alpha, n=48, lo=300.0):
    """Nodes/weights for int f(gamma) (1 - gamma/gc)^alpha dgamma on [lo, gc] per bin."""
    G, W, Bn = [], [], []
    for b in range(12):
        glo, ghi = max(EDGES[b], lo), EDGES[b + 1]
        if gc <= glo or ghi <= lo:
            continue
        if gc <= ghi:
            g, w = _gj(glo, gc, alpha, n)
        elif gc - ghi > 0.3 * (ghi - glo):
            ua, ub = np.log(glo), np.log(ghi)
            u = 0.5 * (ub - ua) * XG48 + 0.5 * (ub + ua)
            g = np.exp(u)
            w = 0.5 * (ub - ua) * WG48 * g * (1 - g / gc) ** alpha
        else:
            g1, w1 = _gj(glo, gc, alpha, n)
            g2, w2 = _gj(ghi, gc, alpha, n)
            g, w = np.concatenate([g1, g2]), np.concatenate([w1, -w2])
        G.append(g)
        W.append(w)
        Bn.append(np.full(g.size, b))
    g = np.concatenate(G)
    bb = np.concatenate(Bn)
    return g, np.concatenate(W), bb, (np.log(g) - np.log(GB[bb])) / H_B


def moments_from(g, w, bb, x, dens, N=2):
    a = np.zeros((12, N + 1))
    for r in range(N + 1):
        np.add.at(a[:, r], bb, w * dens * x**r)
    return (a / WB[:, None]).ravel()


NODES = smooth_nodes()


def mom_PL(phi, N=2):
    g, w, bb, x = NODES
    return moments_from(g, w, bb, x, np.exp(phi[0]) / GAMMA0 * (g / GAMMA0) ** (-phi[1]), N)


def dens_SBPL(phi, g, s=2.0):
    y = g / np.exp(phi[3])
    return np.exp(phi[0]) / GAMMA0 * y ** (-phi[1]) * (1 + y**s) ** (-(phi[2] - phi[1]) / s)


def mom_SBPL(phi, N=2):
    g, w, bb, x = NODES
    return moments_from(g, w, bb, x, dens_SBPL(phi, g), N)


def mom_SBPLs(phi, N=2):
    g, w, bb, x = NODES
    return moments_from(g, w, bb, x, dens_SBPL(phi[:4], g, np.exp(phi[4])), N)


def mom_MIX(phi, N=2):
    g, w, bb, x = NODES
    y = np.log(g / GAMMA0)
    return moments_from(g, w, bb, x, np.exp(phi[0]) / GAMMA0 * np.exp(-phi[1] * y + 0.5 * phi[2] * y**2), N)


def bperp_mean():
    return 5.0 * 2 * integrate.quad(lambda e: np.sqrt(1 - e**2), 0, 0.5, epsabs=0, epsrel=1e-13)[0]


LN_NUCONV = np.log(a1 * bperp_mean())


def mom_JP(phi, N=2):
    gc = np.exp(0.5 * (phi[2] - LN_NUCONV))
    g, w, bb, x = cut_nodes(gc, phi[1] - 2.0)
    return moments_from(g, w, bb, x, np.exp(phi[0]) / GAMMA0 * (g / GAMMA0) ** (-phi[1]), N)


def direct(tab, ch, g, w):
    KI, KP = kbar(tab, ch, g, 0)
    return stokes(w @ KI[0], w @ KP[0])


# ----------------------------------------------------------------------------- 6. fitting
class Problem:
    def __init__(self, C, S):
        self.n = C.shape[-1]
        self.sig = 0.01 * S[:, 0]
        self.sf, self.sh = np.repeat(self.sig[FIT], 3), np.repeat(self.sig[HELD], 3)
        self.W = C[FIT].reshape(-1, self.n) / self.sf[:, None]
        self.Wh = C[HELD].reshape(-1, self.n) / self.sh[:, None]
        self.U, self.s, self.Vt = np.linalg.svd(self.W, full_matrices=False)
        self.nrank = int(np.sum(self.s > 1e-13 * self.s[0]))

    def upre(self, dw):
        proj = self.U.T @ dw
        chi2 = (dw @ dw - np.cumsum(proj**2))[:self.nrank]
        R = np.arange(1, chi2.size + 1)
        return int(R[np.argmin(chi2 + 2 * R)]), chi2

    def maps(self, dw, mode, R):
        if mode == "comp":
            return self.U[:, :R].T @ dw, self.s[:R, None] * self.Vt[:R]
        return dw, self.W


def jac_a(mom, phi):
    h = 1e-6 * np.maximum(1.0, np.abs(phi))
    cols = []
    for i in range(len(phi)):
        e = np.zeros(len(phi))
        e[i] = h[i]
        cols.append((mom(phi + e) - mom(phi - e)) / (2 * h[i]))
    return np.stack(cols, axis=1)


def fit(prob, mom, phi0, dw, mode, R=None):
    y, A = prob.maps(dw, mode, R)
    sol = least_squares(lambda p: y - A @ mom(p), np.asarray(phi0, float), jac=lambda p: -(A @ jac_a(mom, p)),
                        method="lm", xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000)
    J = -(A @ jac_a(mom, sol.x))
    cov = np.linalg.inv(J.T @ J)
    return {"phi": sol.x, "sd": np.sqrt(np.diag(cov)), "chi2": float(np.sum(sol.fun**2)), "ok": bool(sol.success)}


def fisher_sd(prob, mom, phi, mode, R=None):
    _, A = prob.maps(np.zeros(prob.W.shape[0]), mode, R)
    J = A @ jac_a(mom, phi)
    return np.sqrt(np.diag(np.linalg.inv(J.T @ J)))


def seeded_data(S, prob):
    z_fit = np.random.default_rng(SEED).normal(size=(21, 3))
    z_held = np.random.default_rng(SEED + 1).normal(size=(20, 3))
    dw = (S[FIT] + z_fit * prob.sig[FIT, None]).ravel() / prob.sf
    dhw = (S[HELD] + z_held * prob.sig[HELD, None]).ravel() / prob.sh
    return dw, dhw


def analyse(name, S, C, mom, phi_true, a_true=None):
    prob = Problem(C, S)
    dw, dhw = seeded_data(S, prob)
    dw0 = S[FIT].ravel() / prob.sf
    R, chi2R = prob.upre(dw)
    beta_R = prob.U[:, :R].T @ dw
    a_tsvd = prob.Vt[:R].T @ (beta_R / prob.s[:R])
    out = {"R_star": R, "numerical_rank": prob.nrank, "chi2_fit_TSVD_at_Rstar": float(chi2R[R - 1]),
           "chi2_held_TSVD_at_Rstar": float(np.sum((dhw - prob.Wh @ a_tsvd) ** 2))}
    for mode, RR in (("comp", R), ("full", None)):
        f = fit(prob, mom, phi_true, dw, mode, RR)
        out[f"seeded_{mode}"] = {"phi": f["phi"].tolist(), "sd": f["sd"].tolist(), "chi2": f["chi2"]}
    sd_c, sd_f = fisher_sd(prob, mom, phi_true, "comp", R), fisher_sd(prob, mom, phi_true, "full")
    out["fisher_sd_full_at_truth"] = sd_f.tolist()
    out["sd_comp_over_full_at_truth_Rstar"] = (sd_c / sd_f).tolist()
    out["sd_comp_over_full_R_scan"] = {f"R{r}": (fisher_sd(prob, mom, phi_true, "comp", r) / sd_f).tolist()
                                       for r in range(max(3, len(phi_true)), 15)}
    for mode, RR, sd in (("comp", R, sd_c), ("full", None, sd_f)):
        f = fit(prob, mom, phi_true, dw0, mode, RR)
        out[f"noisefree_{mode}"] = {"phi": f["phi"].tolist(), "shift_over_sd": ((f["phi"] - phi_true) / sd).tolist(),
                                    "chi2": f["chi2"]}
    if a_true is not None:
        eps = S - np.einsum("jsk,k->js", C, a_true)
        epsw = eps[FIT].ravel() / prob.sf
        out["max_eps_over_I_IQU"] = (np.max(np.abs(eps) / S[:, :1], axis=0)).tolist()
        out["norm_eps_w"] = float(np.linalg.norm(epsw))
    return out, prob


def run_realisations(prob, S, mom, phi_start, R_seed, n_real, k):
    rows = []
    for kk in range(n_real):
        z = np.random.default_rng([SEED, 1, kk]).normal(size=(41, 3))
        dw = (S[FIT] + z[FIT] * prob.sig[FIT, None]).ravel() / prob.sf
        R = prob.upre(dw)[0]
        fc = fit(prob, mom, phi_start, dw, "comp", max(R, k))
        fs = fit(prob, mom, phi_start, dw, "comp", R_seed)
        ff = fit(prob, mom, phi_start, dw, "full")
        rows.append(np.concatenate([[R], fc["phi"], fc["sd"], [fc["chi2"], fc["ok"]], fs["phi"], fs["sd"],
                                    ff["phi"], ff["sd"], [ff["chi2"], ff["ok"]]]))
    return np.array(rows)


def pull_stats(phi, sd, ref):
    p = (phi - ref[None]) / sd
    n = p.shape[0]
    return {"n": n, "mean": p.mean(0).tolist(), "sd": p.std(0, ddof=1).tolist(),
            "frac_|pull|<1": np.mean(np.abs(p) < 1, axis=0).tolist(), "binomial_sd": float(np.sqrt(0.6827 * 0.3173 / n))}


def compare_with_author(mine, author, k, m):
    """Per-realisation agreement with the author's npz rows (same seeds)."""
    n = mine.shape[0]
    width = 2 * k + 2 + 2 * m + 1
    A = author[:n]
    a_comp = A[:, 1:1 + 2 * k]
    a_full = A[:, 1 + 2 * width:1 + 2 * width + 2 * k]
    m_comp = mine[:, 1:1 + 2 * k]
    off = 1 + 2 * k + 2 + 2 * k
    m_full = mine[:, off:off + 2 * k]
    return {"n_compared": n, "R_star_identical_fraction": float(np.mean(mine[:, 0] == A[:, 0])),
            "max|dphi|/sd_comp_Rstar": float(np.max(np.abs(m_comp[:, :k] - a_comp[:, :k]) / a_comp[:, k:])),
            "max|dsd|/sd_comp_Rstar": float(np.max(np.abs(m_comp[:, k:] / a_comp[:, k:] - 1))),
            "max|dphi|/sd_full": float(np.max(np.abs(m_full[:, :k] - a_full[:, :k]) / a_full[:, k:]))}


# ----------------------------------------------------------------------------- 7. author's files
def author_consistency(res, npz):
    out = {}
    for name, t in res["tasks"].items():
        key = name.replace("-", "_")
        arr = npz[key]
        k = len(t["coords"])
        m = len(t["invariant_names"])
        width = 2 * k + 2 + 2 * m + 1
        ref = np.asarray(t["reference_point"]["phi"])
        o = {"k": k, "min_R_star": int(np.min(arr[:, 0])), "R_star_mode": int(np.bincount(arr[:, 0].astype(int)).argmax())}
        for i, lab in enumerate(("comp_Rstar", "comp_Rseed", "full")):
            blk = arr[:, 1 + i * width:1 + (i + 1) * width]
            ok = blk[:, -1] == 1.0
            phi, sd = blk[ok, :k], blk[ok, k:2 * k]
            ps = pull_stats(phi, sd, ref)
            js = t["realisations"][lab]["params_vs_reference"]
            o[lab] = {"n_failed": int((~ok).sum()), "pull_mean": ps["mean"], "pull_sd": ps["sd"],
                      "coverage": ps["frac_|pull|<1"],
                      "max|recomputed - json| (mean, sd, cov)": float(np.max(np.abs(np.concatenate([
                          np.asarray(ps["mean"]) - js["pull_mean"], np.asarray(ps["sd"]) - js["pull_sd"],
                          np.asarray(ps["frac_|pull|<1"]) - js["coverage_|pull|<1"]])))),
                      "chi2_mean": float(np.mean(blk[ok, 2 * k])), "chi2_held_mean": float(np.mean(blk[ok, 2 * k + 1])),
                      "estimator_mean_minus_ref_over_fisher_sd": ((phi.mean(0) - ref) / np.asarray(t["fisher_sd_full_at_ref"])).tolist(),
                      "estimator_sd_over_median_sd_hat": (phi.std(0, ddof=1) / np.median(sd, 0)).tolist()}
            if lab == "comp_Rstar":
                Rk = np.maximum(arr[ok, 0], k)
                o[lab]["expected_chi2_beta_mean_(R*-k)"] = float(np.mean(Rk - k))
            if lab == "full":
                o[lab]["expected_chi2_mean_(63-k+noisefree)"] = 63 - k + t["noisefree"]["full"]["chi2"]
        o["sd_ratio_scan_max_R>=7"] = float(max(max(v["fisher_sd_ratio_comp_full"]) for kk, v in t["R_scan_seeded_comp"].items()
                                                if int(kk[1:]) >= 7))
        o["sd_ratio_scan_R3-5"] = {kk: max(v["fisher_sd_ratio_comp_full"]) for kk, v in t["R_scan_seeded_comp"].items()
                                   if int(kk[1:]) <= 5}
        out[name] = o
    return out


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def jsonable(x):
    if isinstance(x, dict):
        return {str(k): jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return jsonable(x.tolist())
    if isinstance(x, (np.floating, np.integer, np.bool_)):
        return x.item()
    if isinstance(x, complex):
        return [x.real, x.imag]
    return x


# ----------------------------------------------------------------------------- main
def main():
    res = json.loads((HERE / "params_results.json").read_text())
    npz = np.load(HERE / "params_realisations.npz")
    OUT["provenance_check_author"] = {f: {"recorded": rec, "on_disk": sha(HERE / f), "match": rec == sha(HERE / f)}
                                      for f, rec in (("params.py", res["provenance"]["script_sha256"]),
                                                     ("paramlib.py", res["provenance"]["inputs_sha256"]["paramlib.py"]),
                                                     ("params_degeneracy.py", res["provenance"]["inputs_sha256"]["params_degeneracy.py"]))}
    t = time.time()
    OUT["spot_check_F_G"] = spot_check()
    od = OneD()
    OUT["lnF_1d_spline_max_rel_err"] = od.spline_check
    mx, _ = derivative_fd_check(od)
    OUT["derivative_recursion_fd_check_max_rel"] = mx
    tic("kernels", t)

    t = time.time()
    bU, wU = field_nodes("U46")
    tabU = Table(od, bU, wU, SETA, WETA, 4, 0.001)
    bM, wM = field_nodes("maxwell")
    tabM = Table(od, bM, wM, SETA, WETA, 0, 0.0025)
    tabB = Table(od, bU, wU, np.array([1.0]), np.array([1.0]), 0, 0.001)  # B-only (eta-resolved via shift)
    ch, ch2 = Channels(NT), Channels(int(1.5 * NT))
    tic("tables", t)

    t = time.time()
    C2, DI, DP = basis(tabU, ch, 4)  # orders to 4 (for rho_3)
    Cn2 = np.zeros((41, 3, 36))
    for b in range(12):
        Cn2[:, :, 3 * b:3 * b + 3] = C2[:, :, 5 * b:5 * b + 3]
    C3 = np.zeros((41, 3, 48))
    for b in range(12):
        C3[:, :, 4 * b:4 * b + 4] = C2[:, :, 5 * b:5 * b + 4]
    tic("basis", t)

    # ---------------- ensembles and truths
    g, w, bb, x = NODES
    g2, w2, bb2, x2 = smooth_nodes(96)
    shapePL = lambda gg: (gg / GAMMA0) ** -2.6 / GAMMA0  # noqa: E731
    normPL = 1.0 / np.sum(w * shapePL(g))
    phiE1 = np.array([np.log(normPL), 2.6])
    S_E1 = direct(tabU, ch, g, w * normPL * shapePL(g))
    S_E1_dbl = direct(tabU, ch, g2, w2 * normPL * shapePL(g2))
    S_E1_ch = direct(tabU, ch2, g, w * normPL * shapePL(g))
    a_E1 = mom_PL(phiE1)
    author_S = np.asarray(res["tasks"]["E1"]["S_direct_IQU"])
    OUT["E1_direct_checks"] = {
        "gamma_node_doubling_max/I": float(np.max(np.abs(S_E1 - S_E1_dbl) / S_E1[:, :1])),
        "channel_nodes_x1.5_max/I": float(np.max(np.abs(S_E1 - S_E1_ch) / S_E1[:, :1])),
        "vs_author_S_direct_max/I_IQU": (np.max(np.abs(S_E1 - author_S) / author_S[:, :1], axis=0)).tolist()}
    # analytic R&L check of the slope away from edges
    p = 2.6
    OUT["E1_local_slope_ch20"] = float(-(np.log(S_E1[21, 0]) - np.log(S_E1[19, 0])) / (np.log(CENTRES[21]) - np.log(CENTRES[19])))

    t = time.time()
    OUT["E1"], probE1 = analyse("E1", S_E1, Cn2, mom_PL, phiE1, a_E1)
    q = 1.8
    sq = 2 * integrate.quad(lambda e: (1 - e**2) ** (q / 2), 0, 0.5, epsabs=0, epsrel=1e-13)[0]
    bU_q = (6 ** (q + 1) - 4 ** (q + 1)) / (2 * (q + 1))

    def inv_PL(phi, bq_fun):
        pp = phi[1]
        qq = 0.5 * (pp + 1)
        s_q = 2 * integrate.quad(lambda e: (1 - e**2) ** (qq / 2), 0, 0.5, epsabs=0, epsrel=1e-13)[0]
        return phi[0] + (pp - 1) * np.log(GAMMA0) + np.log(bq_fun(qq) * s_q)

    bqU = lambda qq: (6 ** (qq + 1) - 4 ** (qq + 1)) / (2 * (qq + 1))  # noqa: E731
    sigM = 5.0 / np.sqrt(3.0)
    bqM = lambda qq: (2 * sigM**2) ** (qq / 2) * special.gamma((3 + qq) / 2) / special.gamma(1.5)  # noqa: E731
    OUT["E1"]["invariant_truth"] = float(inv_PL(phiE1, bqU))
    tic("E1_seeded", t)

    # ---------------- E1M
    t = time.time()
    S_M = direct(tabM, ch, g, w * normPL * shapePL(g))
    OUT["E1M"], _ = analyse("E1M", S_M, Cn2, mom_PL, phiE1)
    ratio = bqM(q) / bU_q
    OUT["E1M"]["field_moment_ratio_<B_perp^1.8>_M/U"] = float(ratio)
    inv_true_M = inv_PL(phiE1, bqM)
    for lab in ("noisefree_comp", "seeded_comp"):
        ph = np.array(OUT["E1M"][lab]["phi"])
        dinv = inv_PL(ph, bqU) - inv_true_M
        OUT["E1M"][lab]["dlnA"] = float(ph[0] - phiE1[0])
        OUT["E1M"][lab]["invariant_minus_truth"] = float(dinv)
        OUT["E1M"][lab]["decomp_(dlnA - ln ratio)"] = float(ph[0] - phiE1[0] - np.log(ratio))
        OUT["E1M"][lab]["decomp_dp*ln(gamma0)"] = float((ph[1] - 2.6) * np.log(GAMMA0))
        OUT["E1M"][lab]["decomp_d ln<B_perp^((p+1)/2)>_U"] = float(inv_PL(np.array([0, ph[1]]), bqU) - (ph[1] - 1) * np.log(GAMMA0)
                                                                    - (inv_PL(np.array([0, 2.6]), bqU) - 1.6 * np.log(GAMMA0)))
    OUT["E1M"]["S_maxwell_gamma_node_doubling"] = float(np.max(np.abs(direct(tabM, ch, g2, w2 * normPL * shapePL(g2)) - S_M) / S_M[:, :1]))
    tic("E1M", t)

    # ---------------- E2 (s = 2)
    t = time.time()
    phiE2 = np.array([0.0, 2.0, 3.2, np.log(7800.0)])
    normE2 = 1.0 / np.sum(w * dens_SBPL(phiE2, g))
    phiE2[0] = np.log(normE2)
    S_E2 = direct(tabU, ch, g, w * dens_SBPL(phiE2, g))
    OUT["E2_s2"], probE2 = analyse("E2", S_E2, Cn2, mom_SBPL, phiE2, mom_SBPL(phiE2))
    OUT["E2_s2"]["truth"] = phiE2.tolist()
    phiE2s = np.concatenate([phiE2, [np.log(2.0)]])
    sdfs = fisher_sd(probE2, mom_SBPLs, phiE2s, "full")
    OUT["E2_s2"]["fisher_sd_full_s_free_at_truth"] = sdfs.tolist()
    tic("E2", t)

    # ---------------- E3
    t = time.time()
    phiE3 = np.array([0.0, 2.6, 0.04])
    y3 = np.log(g / GAMMA0)
    sh3 = np.exp(-2.6 * y3 + 0.02 * y3**2) / GAMMA0
    phiE3[0] = np.log(1.0 / np.sum(w * sh3))
    S_E3 = direct(tabU, ch, g, w * np.exp(phiE3[0]) * sh3)
    OUT["E3"], _ = analyse("E3", S_E3, Cn2, mom_MIX, phiE3, mom_MIX(phiE3))
    OUT["E3"]["truth"] = phiE3.tolist()
    tic("E3", t)

    # ---------------- E4-JP truth and fits
    t = time.time()
    b_s = LOSS * (76.0 / 3.0) * MUG**2 / (8 * np.pi)
    b_c = LOSS * 1.0 * EV
    t_age = 19.0 * MYR
    gc = 1.0 / ((b_s + b_c) * t_age)
    gj, wj, bj, xj = cut_nodes(gc, 0.6)
    gj2, wj2, _, _ = cut_nodes(gc, 0.6, n=96)
    shJP = lambda gg: (gg / GAMMA0) ** -2.6 / GAMMA0  # noqa: E731
    normJP = 1.0 / np.sum(wj * shJP(gj))
    normJP2 = 1.0 / np.sum(wj2 * shJP(gj2))
    phiJP = np.array([np.log(normJP), 2.6, LN_NUCONV + 2 * np.log(gc)])
    S_JP = direct(tabU, ch, gj, wj * normJP * shJP(gj))
    S_JP2 = direct(tabU, ch, gj2, wj2 * normJP2 * shJP(gj2))
    OUT["E4JP_truth"] = {"b_s": b_s, "b_c": b_c, "U_B_eV": (76 / 3) * MUG**2 / (8 * np.pi) / EV, "gamma_c": gc,
                         "ln(a1<B_perp>)": LN_NUCONV, "<B_perp>": bperp_mean(), "ln_nu_b": phiJP[2],
                         "nu_b_GHz": np.exp(phiJP[2]) / 1e9, "lnA": phiJP[0],
                         "node_doubling_max/I": float(np.max(np.abs(S_JP - S_JP2) / S_JP[:, :1])),
                         "vs_author_S_direct_max/I_IQU": (np.max(np.abs(S_JP - np.asarray(res["tasks"]["E4-JP"]["S_direct_IQU"]))
                                                                 / S_JP[:, :1], axis=0)).tolist()}
    OUT["E4-JP"], probJP = analyse("E4-JP", S_JP, Cn2, mom_JP, phiJP, mom_JP(phiJP))
    tic("E4JP", t)

    # ---------------- KP predictions, KP direct spectrum, JP-family fits
    t = time.time()
    bkp = lambda mu: 1.5 * b_s * (1 - mu**2) + b_c  # noqa: E731
    bjp = b_s + b_c
    et, ew = gl(64, 0.0, 0.5)
    ew = ew / 0.5
    rat = bkp(et) / bjp
    wem = ew * (1 - et**2) ** 0.9
    wem = wem / wem.sum()
    OUT["KP_predictions"] = {"range": [float(bkp(0.5) / bjp), float(bkp(0.0) / bjp)], "uniform_mean": float(ew @ rat),
                             "uniform_mean_closed_form": float((1.5 * b_s * 11 / 12 + b_c) / bjp),
                             "emission_weighted_mean": float(wem @ rat)}
    # KP direct spectrum: S = int_{-1/2}^{1/2} deta int dgamma 2 N(gamma, eta) K(gamma, eta)
    mu_n, mu_w = gl(48, 0.0, 1.0)
    I_mu = []
    for mu in mu_n:
        gg, ww, _, _ = cut_nodes(1.0 / (bkp(mu) * t_age), 0.6)
        I_mu.append(np.sum(ww * shJP(gg)))
    K_kp = 1.0 / float(np.sum(mu_w * np.array(I_mu)))  # int_{-1}^{1} dmu (K/2) ... = K int_0^1 dmu ...
    eta_k, weta_k = gl(24, 0.0, 0.5)
    S_KP = np.zeros((41, 3))
    for e_, we_ in zip(eta_k, weta_k):
        gce = 1.0 / (bkp(e_) * t_age)
        gg, ww, _, _ = cut_nodes(gce, 0.6)
        se = np.sqrt(1 - e_**2)
        S_KP += 2 * we_ * direct_eta(tabB, ch, gg, ww * K_kp * shJP(gg), se)
    OUT["KP_direct"] = {"vs_author_max/I_IQU": (np.max(np.abs(S_KP - np.asarray(res["tasks"]["E4-KP_JPfamily"]["S_direct_IQU"]))
                                                       / S_KP[:, :1], axis=0)).tolist()}
    probKP = Problem(Cn2, S_KP)
    dwK, _ = seeded_data(S_KP, probKP)
    RK = probKP.upre(dwK)[0]
    dwK0 = S_KP[FIT].ravel() / probKP.sf

    def teff(lnnu):
        return gc / np.exp(0.5 * (lnnu - LN_NUCONV))

    kpfit = {"R_star_seeded": RK}
    for mode, RR in (("comp", RK), ("full", None)):
        f = fit(probKP, mom_JP, phiJP, dwK0, mode, RR)
        resid = np.einsum("jsk,k->js", Cn2, mom_JP(f["phi"])) - S_KP
        kpfit[f"noisefree_{mode}"] = {"phi": f["phi"].tolist(), "t_eff/t": float(teff(f["phi"][2])), "chi2": f["chi2"],
                                      "max|C a(phi) - S_KP|/I_IQU_all41": (np.max(np.abs(resid) / S_KP[:, :1], axis=0)).tolist()}
        f = fit(probKP, mom_JP, phiJP, dwK, mode, RR)
        kpfit[f"seeded_{mode}"] = {"phi": f["phi"].tolist(), "t_eff/t": float(teff(f["phi"][2])),
                                   "sd_t_eff/t": float(0.5 * f["sd"][2] * teff(f["phi"][2]))}

    # exact JP forward fit (no moments), amplitude profiled
    def jp_spec(xx):
        gg, ww, _, _ = cut_nodes(float(np.exp(xx[1])), float(xx[0]) - 2.0)
        return direct(tabU, ch, gg, ww * (gg / GAMMA0) ** (-xx[0]) / GAMMA0)

    def jp_res(xx):
        mw = jp_spec(xx)[FIT].ravel() / probKP.sf
        A = mw @ dwK0 / (mw @ mw)
        return dwK0 - A * mw

    sol = least_squares(jp_res, np.array([2.6, np.log(gc)]), method="lm", diff_step=1e-4, xtol=1e-10, ftol=1e-10)
    sbest = jp_spec(sol.x)
    mw = sbest[FIT].ravel() / probKP.sf
    Abest = mw @ dwK0 / (mw @ mw)
    kpfit["exact_JP_forward"] = {"p": float(sol.x[0]), "gamma_c": float(np.exp(sol.x[1])),
                                 "t_eff/t": float(gc / np.exp(sol.x[1])), "chi2": float(np.sum(sol.fun**2)),
                                 "max|S_fit - S_KP|/I_IQU": (np.max(np.abs(Abest * sbest - S_KP) / S_KP[:, :1], axis=0)).tolist()}
    OUT["KP_JPfamily"] = kpfit
    tic("KP", t)

    # ---------------- null vector (E1)
    t = time.time()
    nv = {}
    for N in (2, 3):
        a = mom_PL(phiE1, N).reshape(12, N + 1)
        dSI = np.zeros(41)
        dSP = np.zeros(41, complex)
        for r in range(N + 1):
            dSI += (WB * a[:, r]) @ (DI[r] + (r + 1) / (2 * H_B) * DI[r + 1])
            dSP += (WB * a[:, r]) @ (DP[r] + (r + 1) / (2 * H_B) * DP[r + 1])
        rI = (N + 1) / (2 * H_B) * (WB * a[:, N]) @ DI[N + 1]
        rP = (N + 1) / (2 * H_B) * (WB * a[:, N]) @ DP[N + 1]
        Jw = stokes(dSI, dSP)[FIT].ravel() / probE1.sf
        rw = stokes(rI, rP)[FIT].ravel() / probE1.sf
        Wn = (Cn2 if N == 2 else C3)[FIT].reshape(63, -1) / probE1.sf[:, None]
        U, s, _ = np.linalg.svd(Wn, full_matrices=False)
        Uk = U[:, s > 1e-13 * s[0]]
        perp = Jw - Uk @ (Uk.T @ Jw)
        perpR = {f"R{R}": float(np.linalg.norm(Jw - U[:, :R] @ (U[:, :R].T @ Jw))) for R in (7, 9, 12, 20)}
        nv[f"N{N}"] = {"|(I-U_R U_R^T) J_lnB|_w": perpR,
                       "|J_lnB|_w": float(np.linalg.norm(Jw)), "|rho_N|_w": float(np.linalg.norm(rw)),
                       "ratio": float(np.linalg.norm(rw) / np.linalg.norm(Jw)),
                       "1/|rho_N|": float(1 / np.linalg.norm(rw)),
                       "|(I-P_W) J_lnB|_w": float(np.linalg.norm(perp)),
                       "marginal_sd_lnB_free_moments=1/|(I-P_W)J|": float(1 / np.linalg.norm(perp))}
    OUT["null_vector_E1"] = nv
    tic("null_vector", t)

    # ---------------- realisations (same seeds as the author) and comparison with the npz
    t = time.time()
    for name, prob, S, mom, phi, nre, key, k, m in (
            ("E1", probE1, S_E1, mom_PL, phiE1, N_REAL_E1, "E1", 2, 1),
            ("E4-JP", probJP, S_JP, mom_JP, phiJP, N_REAL_JP, "E4_JP", 3, 2),
            ("E2_s2", probE2, S_E2, mom_SBPL, phiE2, N_REAL_E2, "E2_s2", 4, 3)):
        Rs = OUT[name]["R_star"]
        arr = run_realisations(prob, S, mom, phi, Rs, nre, k)
        OUT.setdefault("realisations", {})[name] = {
            "n": nre, "n_failed_comp_or_full": int(np.sum((arr[:, 1 + 2 * k + 1] == 0) | (arr[:, -1] == 0))),
            "pulls_comp_Rstar": pull_stats(arr[:, 1:1 + k], arr[:, 1 + k:1 + 2 * k], phi),
            "pulls_full": pull_stats(arr[:, 1 + 4 * k + 2:1 + 5 * k + 2], arr[:, 1 + 5 * k + 2:1 + 6 * k + 2], phi),
            "chi2_beta_mean": float(np.mean(arr[:, 1 + 2 * k])), "E[R*-k]": float(np.mean(np.maximum(arr[:, 0], k) - k)),
            "chi2_full_mean": float(np.mean(arr[:, -2])),
            "vs_author_npz": compare_with_author(arr, npz[key], k, m)}
        tic(f"realisations_{name}", t)
        t = time.time()

    OUT["author_internal_consistency"] = author_consistency(res, npz)
    OUT["config_used"] = {"channel_nodes": NT, "B_nodes_U46": 24, "B_nodes_maxwell": "128 on [0,30] muG", "eta_nodes": 16,
                          "gamma_nodes_per_bin": 48, "table_step_s": [0.001, 0.0025], "Z1": Z1, "h_b": H_B}
    OUT["provenance"] = {"script": Path(__file__).name, "script_sha256": sha(__file__),
                         "inputs_sha256": {f: sha(HERE / f) for f in ("params_results.json", "params_realisations.npz",
                                                                      "lib/kernels.py")},
                         "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__},
                         "wall_clock_s": time.time() - T0, "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z")}
    (HERE / "verify_params_results.json").write_text(json.dumps(jsonable(OUT), indent=1))
    print("done", time.time() - T0)


def direct_eta(tabB, ch, g, w, se):
    KI, KP = kbar(tabB, ch, g, 0, ln_seta=np.log(se), seta=se)
    return stokes(w @ KI[0], w @ KP[0])


if __name__ == "__main__":
    main()
