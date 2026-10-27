"""Independent verification of the baselines (baselines.py) and two cost claims (cost.py).

Nothing from the author's analysis code is imported.  lib/kernels.py is imported only for
F and G, and only to spot-check them against scipy (kv + quad); the kernel used below is an
independent table built here from scipy.special.kv:

  F(x)   = x int_x^inf K_5/3          (cumulative quad of kv, 9000 segments)
  theta F = F - x^2 K_5/3             theta = x d/dx
  theta^2 F = F - (4/3) x^2 K_5/3 + x^3 K_2/3
  G = x K_2/3, theta G = x K_2/3 / 3 - x^2 K_1/3, theta^2 G = x K_2/3 / 9 - 2 x^2 K_1/3 + x^3 K_2/3

Basis columns: d/dx_b = h d/dln gamma = -2 h theta, so
  D_0 = Kbar, D_1 = -2h theta Kbar, D_2 = 2 h^2 theta^2 Kbar, column = w_b D.
Quadrature differs from the author's on purpose: B 24-node GL on [4,6], eta 20-node GL on
[0,1/2], channel 600-node GL on the bump, gamma 30 panels x 20 nodes in ln gamma, JP cut-off
panel mapped by u = u_c - Delta tau^5 (integrand analytic in tau).

Writes verify_baselines_results.json.  Run: python verify_baselines.py
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy import integrate, optimize, special
from scipy.interpolate import CubicSpline

T0 = time.time()
HERE = Path(__file__).resolve().parent
GALPROP_CSV = HERE / "inputs" / "galprop-benchmark" / "regional_spectra.csv"
AUTHOR_JSON = HERE / "baselines_results.json"
OUT_JSON = HERE / "verify_baselines_results.json"

# ----------------------------------------------------------------------------- constants (cgs)
E_ESU = 4.803204712570263e-10
M_E = 9.1093837015e-28
C_CGS = 2.99792458e10
C_SI = 2.99792458e8
SIGMA_T = 6.6524587321e-25
EV = 1.602176634e-12
MYR = 3.15576e13
A_NU = 3.0 * E_ESU * 1e-6 / (4.0 * np.pi * M_E * C_CGS)  # a_B per muG of B_perp [Hz]

# ----------------------------------------------------------------------------- configuration
NU = np.geomspace(1e8, 1e10, 41)
FIT = np.arange(0, 41, 2)
HELD = np.arange(1, 41, 2)
G_LO, G_HI, NB = 300.0, 1e5, 12
LOG_EDGES = np.linspace(np.log(G_LO), np.log(G_HI), NB + 1)
H = 0.5 * (LOG_EDGES[1] - LOG_EDGES[0])
GB = np.exp(0.5 * (LOG_EDGES[1:] + LOG_EDGES[:-1]))
_m = np.diff(np.exp(LOG_EDGES) ** (1 - 2.6)) / (1 - 2.6)
WB = _m / _m.sum()
AZ = 0.4 * np.exp(0.5j)
SEED_FIT = 20260928
MY_SEED = 777001
NREAL = 500


def sha(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()


# ----------------------------------------------------------------------------- 1. spot-check lib F, G
def F_quad(x):
    v, _ = integrate.quad(lambda t: special.kv(5.0 / 3.0, t), x, np.inf, epsabs=0, epsrel=1e-12, limit=500)
    return x * v


def spot_check_lib():
    sys.path.insert(0, str(HERE / "lib"))
    import kernels as LK  # noqa: E402  (F and G only)

    xs = np.array([1e-4, 1e-2, 0.3, 1.0, 3.0, 10.0, 30.0])
    out = []
    for x in xs:
        fr, gr = F_quad(x), x * special.kv(2.0 / 3.0, x)
        fl, gl = float(LK.F(np.array([x]))[0]), float(LK.G(np.array([x]))[0])
        out.append({"x": x, "F_rel": abs(fl / fr - 1), "G_rel": abs(gl / gr - 1)})
    return out, LK


# ----------------------------------------------------------------------------- 2. own kernel tables
def kernel_tables():
    lx = np.linspace(np.log(1e-9), np.log(200.0), 9001)
    x = np.exp(lx)
    k53 = lambda t: special.kv(5.0 / 3.0, t)
    seg = np.array([integrate.quad(k53, x[i], x[i + 1], epsabs=0, epsrel=1e-13, limit=200)[0]
                    for i in range(len(x) - 1)])
    tail = integrate.quad(k53, x[-1], np.inf, epsabs=0, epsrel=1e-12)[0]
    fint = np.append(np.cumsum(seg[::-1])[::-1] + tail, tail)
    F = x * fint
    K53, K23, K13 = special.kv(5 / 3, x), special.kv(2 / 3, x), special.kv(1 / 3, x)
    tF = F - x**2 * K53
    ttF = F - (4 / 3) * x**2 * K53 + x**3 * K23
    G = x * K23
    tG = x * K23 / 3 - x**2 * K13
    ttG = x * K23 / 9 - 2 * x**2 * K13 + x**3 * K23
    tab = {"F": [CubicSpline(lx, np.log(F)), CubicSpline(lx, tF / F), CubicSpline(lx, ttF / F)],
           "G": [CubicSpline(lx, np.log(G)), CubicSpline(lx, tG / G), CubicSpline(lx, ttG / G)],
           "lo": lx[0], "hi": lx[-1]}
    return tab


def eval_W(tab, which, r, x):
    """theta^r W(x) from the table; 0 above the table (x > 200, W < e^-195)."""
    lx = np.log(x)
    out = np.zeros_like(x)
    ok = lx <= tab["hi"]
    if np.any(lx[ok] < tab["lo"]):
        raise ValueError("x below table")
    v = np.exp(tab[which][0](lx[ok]))
    if r > 0:
        v = v * tab[which][r](lx[ok])
    out[ok] = v
    return out


def field_nodes(nb=24, ne=20):
    b, wb = leggauss(nb)
    b, wb = 5.0 + b, 0.5 * wb  # U[4,6]: nodes 5 + t, prob weights w/2
    e, we = leggauss(ne)
    e, we = 0.25 * (e + 1), 0.5 * we  # eta on [0, 1/2], prob weights sum to 1 (sum 0.25 w = 0.5 -> /0.5)
    we = we / we.sum()
    bp = (b[:, None] * np.sqrt(1 - e[None, :] ** 2)).ravel()
    w = (wb[:, None] * we[None, :]).ravel()
    return bp, w


def averaged_tables(tab, bp, w):
    """H_{W,r}(y) = sum_k w_k Bperp_k theta^r W(y/(a Bperp_k)), y = nu/gamma^2; spline in ln y."""
    ly = np.linspace(np.log(1e-6), np.log(190.0 * A_NU * bp.min()), 7001)  # top: every x <= 190 in table
    y = np.exp(ly)
    x = y[:, None] / (A_NU * bp[None, :])
    out = {}
    for which in ("F", "G"):
        h = [np.sum(w * bp * eval_W(tab, which, r, x), axis=1) for r in range(3)]
        out[which] = [CubicSpline(ly, np.log(h[0])), CubicSpline(ly, h[1] / h[0]), CubicSpline(ly, h[2] / h[0])]
    out["hi"] = ly[-1]
    out["lo"] = ly[0]
    return out


def eval_H(ht, which, r, y):
    ly = np.log(y)
    out = np.zeros_like(y)
    ok = ly <= ht["hi"]
    if np.any(ly[ok] < ht["lo"]):
        raise ValueError("y below table")
    v = np.exp(ht[which][0](ly[ok]))
    if r > 0:
        v = v * ht[which][r](ly[ok])
    out[ok] = v
    return out


# ----------------------------------------------------------------------------- channels
def rho1(t):
    t = np.asarray(t, float)
    out = np.zeros_like(t)
    ok = np.abs(t) < 1
    out[ok] = np.exp(-t[ok] ** 2 / (1 - t[ok] ** 2))
    return out


Z1 = integrate.quad(lambda t: float(rho1(np.array([t]))[0]), -1, 1, epsabs=0, epsrel=1e-13, limit=400)[0]


def channels(n=600):
    t, w = leggauss(n)
    wt = w * rho1(t) / Z1
    nu = NU[:, None] * (1 + 0.2 * t[None, :])
    lam2 = (C_SI / nu) ** 2
    scr = np.exp(2j * lam2 * 8.0 - 2 * lam2**2 * 0.25)
    return nu, wt, scr


# ----------------------------------------------------------------------------- ensembles
def gl_panels(lo, hi, npan=30, n=20):
    t, w = leggauss(n)
    edges = np.linspace(lo, hi, npan + 1)
    u = (0.5 * np.diff(edges)[:, None] * t + 0.5 * (edges[1:] + edges[:-1])[:, None]).ravel()
    wu = (0.5 * np.diff(edges)[:, None] * w).ravel()
    return u, wu


def measure(kind, norm=True, n=40, **kw):
    """Nodes gamma and weights m = N dgamma on the support; panels split at every bin edge.

    JP: the panel ending at the cut-off is mapped by u = u_c - d tau^5 (integrand analytic in tau).
    norm=True scales to a unit column.
    """
    t, w = leggauss(n)
    if kind == "JP":
        p, gc = kw["p"], kw["gc"]
        uc = np.log(gc)
        inner = LOG_EDGES[(LOG_EDGES > np.log(G_LO)) & (LOG_EDGES < uc)]
        pts = np.r_[np.log(G_LO), inner]
        us, ws = [], []
        for ua, ub in zip(pts[:-1], pts[1:]):
            u = 0.5 * (ub - ua) * t + 0.5 * (ub + ua)
            us.append(u)
            ws.append(0.5 * (ub - ua) * w * np.exp(u) ** (1 - p) * (1 - np.exp(u - uc)) ** (p - 2))
        d = uc - pts[-1]
        tau, wt = 0.5 * (t + 1), 0.5 * w
        dd = d * tau**5
        rat = -np.expm1(-dd) / dd
        us.append(uc - dd)
        ws.append(wt * np.exp(uc - dd) ** (1 - p) * d ** (p - 2) * tau ** (5 * (p - 2)) * rat ** (p - 2) * 5 * d * tau**4)
        g, m = np.exp(np.concatenate(us)), np.concatenate(ws)
    else:
        u = (0.5 * np.diff(LOG_EDGES)[:, None] * t + 0.5 * (LOG_EDGES[1:] + LOG_EDGES[:-1])[:, None]).ravel()
        wu = (0.5 * np.diff(LOG_EDGES)[:, None] * w).ravel()
        g = np.exp(u)
        if kind == "PL":
            nn = g ** (-kw["p"])
        elif kind == "SBPL":
            y = g / kw["gb"]
            nn = y ** (-kw["p1"]) * (1 + y ** kw["s"]) ** (-(kw["p2"] - kw["p1"]) / kw["s"])
        elif kind == "MIX":
            y = np.log(g / 1e4)
            nn = np.exp(-kw["pbar"] * y + 0.5 * kw["sig"] ** 2 * y**2)
        else:
            raise ValueError(kind)
        m = wu * g * nn
    return (g, m / m.sum()) if norm else (g, m)


def bin_moments(g, m, gscale=1.0):
    """a_br = (1/w_b) sum m x_b^r for nodes at gamma' = gscale * gamma (edge bins absorb |x| > 1)."""
    u = np.log(g * gscale)
    bi = np.clip(np.searchsorted(LOG_EDGES, u, side="right") - 1, 0, NB - 1)
    xb = (u - np.log(GB[bi])) / H
    a = np.zeros((NB, 3))
    for r in range(3):
        np.add.at(a[:, r], bi, m * xb**r)
    return (a / WB[:, None]).ravel()


def direct(ht, ch, g, m, factor=AZ):
    nu, wt, scr = ch
    I = np.zeros(41)
    P = np.zeros(41, complex)
    for s in range(0, len(g), 50):
        y = nu[None] / g[s:s + 50, None, None] ** 2
        I += m[s:s + 50] @ (eval_H(ht, "F", 0, y) @ wt)
        P += m[s:s + 50] @ ((-eval_H(ht, "G", 0, y) * scr[None]) @ wt)
    return I, factor * P


def columns(ht, ch):
    nu, wt, scr = ch
    CI = np.zeros((41, NB, 3))
    CQ = np.zeros((41, NB, 3), complex)
    fac = [1.0, -2 * H, 2 * H**2]
    for b in range(NB):
        y = nu / GB[b] ** 2
        for r in range(3):
            CI[:, b, r] = WB[b] * fac[r] * (eval_H(ht, "F", r, y) @ wt)
            CQ[:, b, r] = WB[b] * fac[r] * ((-eval_H(ht, "G", r, y) * scr) @ wt)
    return CI.reshape(41, -1), AZ * CQ.reshape(41, -1)


def mono_columns(ht, nu):
    C = np.zeros((len(nu), NB, 3))
    fac = [1.0, -2 * H, 2 * H**2]
    for b in range(NB):
        for r in range(3):
            C[:, b, r] = WB[b] * fac[r] * eval_H(ht, "F", r, nu / GB[b] ** 2)
    return C.reshape(len(nu), -1)


# ----------------------------------------------------------------------------- TSVD
def tsvd(C_fit, sig_fit):
    u, s, vt = np.linalg.svd(C_fit / sig_fit[:, None], full_matrices=False)
    return u, s, vt


def tsvd_coef(svd, d_fit, sig_fit, R):
    u, s, vt = svd
    return vt[:R].T @ ((u[:, :R].T @ (d_fit / sig_fit)) / s[:R])


def upre(svd, d_fit, sig_fit):
    u, s, vt = svd
    dw = d_fit / sig_fit
    pr = u.T @ dw
    chi = np.sum(dw**2) - np.cumsum(pr**2)
    return int(np.argmin(chi + 2 * np.arange(1, len(s) + 1))) + 1, chi


def tr_hht(svd, C_held, sig_held, R):
    u, s, vt = svd
    Hm = (C_held / sig_held[:, None]) @ vt[:R].T / s[:R]
    return float(np.sum(Hm**2))


# ----------------------------------------------------------------------------- spectral laws (own VarPro)
def law_basis(name, phi, L):
    if name == "s1":
        return [np.exp(phi[0] * L)]
    if name == "s3":
        return [np.exp(phi[0] * L + phi[1] * L**2)]
    if name == "logcubic":
        return [np.exp(phi[0] * L + phi[1] * L**2 + phi[2] * L**3)]
    e = np.exp(phi[0] * L)
    if name == "bmom3":
        return [e, 0.5 * e * L**2]
    if name == "bmom4":
        return [e, e * L, 0.5 * e * L**2]
    raise ValueError(name)


def law_design(name, phi, L, wt):
    return np.stack([g @ wt for g in law_basis(name, phi, L)], axis=1)


def law_solve(name, phi, L, wt, d, sig):
    Gm = law_design(name, phi, L, wt)
    c = np.linalg.lstsq(Gm / sig[:, None], d / sig, rcond=None)[0]
    return c, (Gm @ c - d) / sig


def fit_law(name, Lfit, wt, d, sig, starts=None):
    nphi = {"s1": 1, "s3": 2, "logcubic": 3, "bmom3": 1, "bmom4": 1}[name]
    if nphi == 1:
        grid = np.arange(-4.0, 2.0001, 0.01)
        cost = [np.sum(law_solve(name, [b], Lfit, wt, d, sig)[1] ** 2) for b in grid]
        b0 = grid[int(np.argmin(cost))]
        res = optimize.minimize_scalar(lambda b: np.sum(law_solve(name, [b], Lfit, wt, d, sig)[1] ** 2),
                                       bounds=(b0 - 0.02, b0 + 0.02), method="bounded",
                                       options={"xatol": 1e-12})
        phi = np.array([res.x])
    else:
        Lc = np.log(np.exp(Lfit) @ wt)  # ln of each row's weighted mean frequency (start only)
        pc = np.polyfit(Lc, np.log(d), nphi)[::-1][1:]  # c1, c2, (c3)
        cand = [pc] + ([] if starts is None else starts)
        rng = np.random.default_rng(1)
        cand += [pc + rng.normal(scale=[0.3, 0.05, 0.02][:nphi]) for _ in range(12)]
        best = None
        for s0 in cand:
            r = optimize.least_squares(lambda ph: law_solve(name, ph, Lfit, wt, d, sig)[1], s0,
                                       xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000)
            if best is None or r.cost < best.cost:
                best = r
        phi = best.x
    c, res = law_solve(name, phi, Lfit, wt, d, sig)
    return phi, c, float(np.sum(res**2))


def law_predict(name, phi, c, L, wt):
    return law_design(name, phi, L, wt) @ c


# ----------------------------------------------------------------------------- electron families
_XQ, _WQ = leggauss(48)


def fam_moments(name, phi):
    g = GB[:, None] * np.exp(H * _XQ[None, :])
    if name == "ePL":
        n = (g / 1e4) ** (-phi[0])
    elif name == "eLP":
        y = np.log(g / 1e4)
        n = np.exp(-phi[0] * y - phi[1] * y**2)
    elif name == "eSBPL":
        y = g / np.exp(phi[2])
        n = y ** (-phi[0]) * (1 + y**2) ** (-(phi[1] - phi[0]) / 2)
    else:
        raise ValueError(name)
    a = np.stack([np.sum(H * _WQ * g * n * _XQ**r, axis=1) for r in range(3)], axis=1)
    return (a / WB[:, None]).ravel()


def fit_family(name, C_fit, d, sig, starts):
    def res(phi):
        v = C_fit @ fam_moments(name, phi)
        A = np.sum(v * d / sig**2) / np.sum(v**2 / sig**2)
        return (A * v - d) / sig

    best = None
    for s0 in starts:
        r = optimize.least_squares(res, s0, xtol=1e-14, ftol=1e-14, gtol=1e-14, max_nfev=4000)
        if best is None or r.cost < best.cost:
            best = r
    phi = best.x
    v = C_fit @ fam_moments(name, phi)
    A = np.sum(v * d / sig**2) / np.sum(v**2 / sig**2)
    return phi, A


# ----------------------------------------------------------------------------- main
def main():
    out = {"script": Path(__file__).name, "script_sha256": sha(__file__),
           "inputs_sha256": {"inputs/galprop-benchmark/regional_spectra.csv": sha(GALPROP_CSV), "lib/kernels.py": sha(HERE / "lib" / "kernels.py"),
                             "baselines_results.json (author, read as data)": sha(AUTHOR_JSON)},
           "versions": {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}}
    AJ = json.loads(AUTHOR_JSON.read_text())["results"]

    spot, LK = spot_check_lib()
    out["lib_spot_check"] = spot
    tab = kernel_tables()
    xs = np.geomspace(1e-5, 60, 400)
    out["own_F_table_vs_lib_F_max_rel"] = float(np.max(np.abs(eval_W(tab, "F", 0, xs) / LK.F(xs) - 1)))
    out["own_G_table_vs_scipy_max_rel"] = float(np.max(np.abs(eval_W(tab, "G", 0, xs) / (xs * special.kv(2 / 3, xs)) - 1)))
    # theta-derivative formulas vs finite differences of lib F (step 1e-4 in ln x)
    e = 1e-4
    fd1 = (LK.F(xs * np.exp(e)) - LK.F(xs * np.exp(-e))) / (2 * e)
    fd2 = (LK.F(xs * np.exp(e)) - 2 * LK.F(xs) + LK.F(xs * np.exp(-e))) / e**2
    sc = np.max(np.abs(LK.F(xs)))
    out["thetaF_vs_fd_max_over_maxF"] = float(np.max(np.abs(eval_W(tab, "F", 1, xs) - fd1)) / sc)
    out["theta2F_vs_fd_max_over_maxF"] = float(np.max(np.abs(eval_W(tab, "F", 2, xs) - fd2)) / sc)

    bp, w = field_nodes()
    ht = averaged_tables(tab, bp, w)
    ch = channels(600)
    ch300 = channels(300)

    # ---- ensembles
    b_s = 4 * SIGMA_T / (3 * M_E * C_CGS) * (152.0 / 6.0) * 1e-12 / (8 * np.pi)
    b_c = 4 * SIGMA_T / (3 * M_E * C_CGS) * EV
    gc = 1.0 / ((b_s + b_c) * 19.0 * MYR)
    out["E4_gamma_c_JP"] = gc
    ens = {
        "E1": measure("PL", p=2.6),
        "E2": measure("SBPL", p1=2.0, p2=3.2, s=2.0, gb=7800.0),
        "E3": measure("MIX", pbar=2.6, sig=0.2),
        "E4-JP": measure("JP", p=2.6, gc=gc),
    }
    spectra = {k: direct(ht, ch, *v) for k, v in ens.items()}
    # E5 two-zone (intensity crosses at channel 20)
    zA, zB = measure("PL", p=3.1), measure("PL", p=2.1)
    IA, PA = direct(ht, ch, *zA, factor=0.4 * np.exp(0.5j))
    IB, PB = direct(ht, ch, *zB, factor=0.4 * np.exp(2j * (0.25 + np.pi / 4)))
    fa = IB[20] / (IA[20] + IB[20])
    spectra["E5"] = (fa * IA + (1 - fa) * IB, fa * PA + (1 - fa) * PB)
    out["E5_fA"] = fa
    # node checks: channel 300 vs 600, gamma doubling for E1, E4-JP
    I300 = direct(ht, ch300, *ens["E1"])[0]
    out["E1_channel_300_vs_600_max_over_I"] = float(np.max(np.abs(I300 / spectra["E1"][0] - 1)))
    # author direct spectra (data) vs mine
    zf = np.load(HERE / "forward_spectra.npz")
    out["direct_vs_author_forward_spectra_max_over_I"] = {
        k: float(np.max(np.abs(zf[k.replace("-", "_")][:, 0] / zf[k.replace("-", "_")][:, 0].max()
                               - spectra[k][0] / spectra[k][0].max()) / (spectra[k][0] / spectra[k][0].max())))
        for k in spectra}

    CI, CP = columns(ht, ch)
    # forward representation check at N = 2 for E1 (true moments)
    a1 = bin_moments(*ens["E1"])
    out["E1_forward_N2_max_over_I"] = float(np.max(np.abs(CI @ a1 - spectra["E1"][0]) / spectra["E1"][0]))

    # ---- synthetic: TSVD and laws, noise-free
    nu, wt, scr = ch
    L = np.log(nu / 1e9)
    synth = {}
    noise_fit = np.random.default_rng(SEED_FIT).normal(size=(21, 3))[:, 0]
    noise_held = np.random.default_rng(SEED_FIT + 1).normal(size=(20, 3))[:, 0]
    for k, (I, P) in spectra.items():
        sig = 0.01 * I
        svd = tsvd(CI[FIT], sig[FIT])
        r = {}
        for R in (2, 3, 4, 7, 8, 9, 12):
            a = tsvd_coef(svd, I[FIT], sig[FIT], R)
            r[f"moment_R{R}_max_res_all"] = float(np.max(np.abs(CI @ a - I) / I))
            r[f"moment_R{R}_tr_HHt_held"] = tr_hht(svd, CI[HELD], sig[HELD], R)
            r[f"moment_R{R}_chi2_held_nf"] = float(np.sum(((CI @ a - I)[HELD] / sig[HELD]) ** 2))
        for name in ("s1", "s3", "bmom3", "logcubic", "bmom4"):
            phi, c, chi2 = fit_law(name, L[FIT], wt, I[FIT], sig[FIT])
            pred = law_predict(name, phi, c, L, wt)
            r[f"{name}_max_res_all"] = float(np.max(np.abs(pred - I) / I))
            r[f"{name}_chi2_fit"] = chi2
            if name == "bmom3":
                r["bmom3_omega2"] = float(c[1] / c[0])
        # seeded UPRE
        d = I[FIT] + sig[FIT] * noise_fit
        rs, chi = upre(svd, d, sig[FIT])
        r["seeded_R_star"] = rs
        a = tsvd_coef(svd, d, sig[FIT], rs)
        r["seeded_chi2_fit"] = float(chi[rs - 1])
        dh = I[HELD] + sig[HELD] * noise_held
        r["seeded_chi2_held"] = float(np.sum(((CI[HELD] @ a - dh) / sig[HELD]) ** 2))
        synth[k] = r
    # matching electron families
    fam = {}
    for k, name, starts in (("E1", "ePL", [[2.0], [2.6], [3.2]]),
                            ("E2", "eSBPL", [[2.0, 3.2, np.log(7800)], [2.2, 3.0, np.log(5000)], [1.8, 3.4, np.log(12000)]]),
                            ("E3", "eLP", [[2.6, 0.0], [2.6, -0.02], [2.4, 0.05]])):
        I = spectra[k][0]
        sig = 0.01 * I
        phi, A = fit_family(name, CI[FIT], I[FIT], sig[FIT], starts)
        a = A * fam_moments(name, phi)
        fam[f"{k}_{name}"] = {"phi": phi.tolist(), "max_res_all": float(np.max(np.abs(CI @ a - I) / I)),
                              "maxP_res_over_I": float(np.max(np.abs(CP @ a - spectra[k][1]) / I))}
    out["families"] = fam

    # ---- seeded E1: P prediction at 8.9 GHz (channel 39) from the I-only TSVD at R*
    I, P = spectra["E1"]
    sig = 0.01 * I
    svd = tsvd(CI[FIT], sig[FIT])
    d = I[FIT] + sig[FIT] * noise_fit
    rs = upre(svd, d, sig[FIT])[0]
    a = tsvd_coef(svd, d, sig[FIT], rs)
    u_, s_, vt_ = svd
    cov = vt_[:rs].T @ np.diag(s_[:rs] ** -2) @ vt_[:rs]
    j = 39
    Pp = CP[j] @ a
    grad = np.real(np.conj(Pp) * CP[j]) / abs(Pp)
    sd = float(np.sqrt(grad @ cov @ grad))
    out["E1_seeded_P_at_8p9GHz"] = {"nu_GHz": NU[j] / 1e9, "R_star": rs, "pred_absP_over_Itrue": float(abs(Pp) / I[j]),
                                    "pred_absP_over_Ipred": float(abs(Pp) / (CI[j] @ a)),
                                    "true_absP_over_I": float(abs(P[j]) / I[j]),
                                    "miss_in_sigma_of_pred": float((abs(Pp) - abs(P[j])) / sd)}

    # ---- Monte Carlo (own seed): UPRE ranks, chi2_held, P miss (moment basis), s3/logcubic chi2_held on E1
    rng = np.random.default_rng(MY_SEED)
    nf = rng.normal(size=(NREAL, 21))
    nh = rng.normal(size=(NREAL, 20))
    mc = {}
    for k, (I, P) in spectra.items():
        sig = 0.01 * I
        svd = tsvd(CI[FIT], sig[FIT])
        Rs, ch2, pm = [], [], []
        for i in range(NREAL):
            d = I[FIT] + sig[FIT] * nf[i]
            rs = upre(svd, d, sig[FIT])[0]
            a = tsvd_coef(svd, d, sig[FIT], rs)
            Rs.append(rs)
            ch2.append(np.sum(((CI[HELD] @ a - (I[HELD] + sig[HELD] * nh[i])) / sig[HELD]) ** 2))
            pm.append(np.max(np.abs(CP @ a - P) / I))
        mc[k] = {"R_star_median": float(np.median(Rs)), "R_star_p16_p84": [float(np.percentile(Rs, 16)), float(np.percentile(Rs, 84))],
                 "chi2_held_median": float(np.median(ch2)), "chi2_held_mean": float(np.mean(ch2)),
                 "frac_chi2_held_above_31.41": float(np.mean(np.array(ch2) > 31.41)),
                 "maxP_res_over_I_median_pct": 100 * float(np.median(pm)),
                 "chi2_held_p84_p95_p99_max": [float(np.percentile(ch2, q)) for q in (84, 95, 99, 100)],
                 "frac_chi2_held_above_100": float(np.mean(np.array(ch2) > 100)),
                 "R_star_hist": {int(v): int(c) for v, c in zip(*np.unique(Rs, return_counts=True))},
                 "chi2_held_median_by_R_star": {int(v): float(np.median(np.array(ch2)[np.array(Rs) == v])) for v in np.unique(Rs)}}
    I = spectra["E1"][0]
    sig = 0.01 * I
    for name in ("s3", "logcubic"):
        phi0, c0, _ = fit_law(name, L[FIT], wt, I[FIT], sig[FIT])
        ch2 = []
        for i in range(NREAL):
            d = I[FIT] + sig[FIT] * nf[i]
            rr = optimize.least_squares(lambda ph: law_solve(name, ph, L[FIT], wt, d, sig[FIT])[1], phi0,
                                        xtol=1e-12, ftol=1e-12)
            c, _ = law_solve(name, rr.x, L[FIT], wt, d, sig[FIT])
            pr = law_predict(name, rr.x, c, L[HELD], wt)
            ch2.append(np.sum(((pr - (I[HELD] + sig[HELD] * nh[i])) / sig[HELD]) ** 2))
        mc["E1"][f"{name}_chi2_held_median"] = float(np.median(ch2))
    out["monte_carlo_own_seed"] = {"seed": MY_SEED, "n_real": NREAL, **mc}
    out["synthetic_noise_free"] = synth

    # ---- GALPROP
    raw = np.loadtxt(GALPROP_CSV, delimiter=",", skiprows=1)
    idx = np.arange(26, 51)
    gnu, gI = raw[idx, 0] * 1e9, raw[idx, 3]
    tr, he = idx % 2 == 0, idx % 2 == 1
    sig = 0.01 * gI
    CM = mono_columns(ht, gnu)
    svd = tsvd(CM[tr], sig[tr])
    gal = {"nu_GHz_range": [gnu[0] / 1e9, gnu[-1] / 1e9], "n_fit": int(tr.sum()), "n_held": int(he.sum())}
    for R in (4, 7, 9, 12):
        a = tsvd_coef(svd, gI[tr], sig[tr], R)
        fr = (CM @ a - gI) / gI
        gal[f"moment_R{R}_held_rms_max_pct"] = [100 * float(np.sqrt(np.mean(fr[he] ** 2))), 100 * float(np.max(np.abs(fr[he])))]
    Lg = np.log(gnu / 1e9)[:, None]
    one = np.array([1.0])
    for name in ("s1", "s3", "logcubic", "bmom3", "bmom4"):
        phi, c, _ = fit_law(name, Lg[tr], one, gI[tr], sig[tr])
        fr = (law_predict(name, phi, c, Lg, one) - gI) / gI
        gal[f"{name}_held_rms_max_pct"] = [100 * float(np.sqrt(np.mean(fr[he] ** 2))), 100 * float(np.max(np.abs(fr[he])))]
        if name == "bmom3":
            gal["bmom3_omega2"] = float(c[1] / c[0])
    cpf = np.polyfit(Lg[tr, 0], np.log(gI[tr]), 3)
    fr = np.exp(np.polyval(cpf, Lg[:, 0])) / gI - 1
    gal["logspace_logcubic_held_rms_pct"] = 100 * float(np.sqrt(np.mean(fr[he] ** 2)))
    phi, A = fit_family("eSBPL", CM[tr], gI[tr], sig[tr], [[2.0, 3.2, np.log(7800)], [1.6, 3.0, np.log(3000)], [2.2, 3.6, np.log(2e4)]])
    fr = (A * CM @ fam_moments("eSBPL", phi) - gI) / gI
    gal["eSBPL_held_rms_max_pct"] = [100 * float(np.sqrt(np.mean(fr[he] ** 2))), 100 * float(np.max(np.abs(fr[he])))]
    gal["eSBPL_phi"] = phi.tolist()
    rng = np.random.default_rng(MY_SEED + 1)
    Rs, rms = [], []
    for i in range(NREAL):
        d = gI[tr] + sig[tr] * rng.normal(size=tr.sum())
        rs = upre(svd, d, sig[tr])[0]
        a = tsvd_coef(svd, d, sig[tr], rs)
        Rs.append(rs)
        rms.append(np.sqrt(np.mean(((CM @ a - gI) / gI)[he] ** 2)))
    gal["UPRE_R_star_median_p16_p84"] = [float(np.median(Rs)), float(np.percentile(Rs, 16)), float(np.percentile(Rs, 84))]
    gal["UPRE_held_rms_median_pct"] = 100 * float(np.median(rms))
    # absorption: formula and a turnover-based estimate from the published spectrum itself
    nl = raw[26, 0]
    gal["tau_ff_formula_EM_1_5"] = [3.28e-7 * 0.8**-1.35 * nl**-2.1 * em for em in (1.0, 5.0)]
    lo = np.arange(0, 26)  # 0.001-0.095 GHz
    lnu = np.log(raw[lo, 0])
    est = {}
    for geom in ("screen", "slab"):
        def model(p, lnu=lnu):
            tau = np.exp(p[4]) * np.exp(-2.1 * lnu)
            latt = -tau if geom == "screen" else np.log(-np.expm1(-tau)) - np.log(tau)
            return np.polyval(p[:4], lnu) + latt
        best = None
        for lt in np.linspace(-18, -8, 21):
            r_ = optimize.least_squares(lambda p: model(p) - np.log(raw[lo, 3]),
                                        np.r_[np.polyfit(lnu[12:], np.log(raw[lo[12:], 3]), 3), lt])
            if best is None or r_.cost < best.cost:
                best = r_
        tau_l = float(np.exp(best.x[4]) * nl**-2.1)
        est[geom] = {"tau_at_0.1145GHz": tau_l, "attenuation_at_0.1145GHz": float(1 - (np.exp(-tau_l) if geom == "screen" else -np.expm1(-tau_l) / tau_l)),
                     "rms_log_resid": float(np.sqrt(np.mean(best.fun**2)))}
    gal["absorption_from_turnover"] = est
    out["galprop"] = gal

    # ---- cost: scaling identity and the B-derivative column identity (continuum, own kernel)
    gam = np.geomspace(500, 5e4, 7)
    nn = np.geomspace(1e8, 1e10, 9)
    lam = 1.3
    k1 = lam * 5.0 * eval_W(tab, "F", 0, nn[None] / (A_NU * 5.0 * lam * gam[:, None] ** 2))
    k2 = lam * 5.0 * eval_W(tab, "F", 0, nn[None] / (A_NU * 5.0 * (gam[:, None] * np.sqrt(lam)) ** 2))
    okk = k2 > 1e-280
    out["scaling_identity_max_rel"] = float(np.max(np.abs(k1[okk] / k2[okk] - 1)))

    def Dcol(Bf, r, g0, nu_):
        x = nu_ / (A_NU * Bf * g0**2)
        f = [1.0, -2 * H, 2 * H**2, None][r]
        if r == 3:  # (1/3!) d^3/dx_b^3 = -(8/6) h^3 theta^3; theta^3 F by FD of theta^2 F
            e3 = 1e-4
            t3 = (eval_W(tab, "F", 2, x * np.exp(e3)) - eval_W(tab, "F", 2, x * np.exp(-e3))) / (2 * e3)
            return Bf * (-(8 / 6) * H**3) * t3
        return Bf * f * eval_W(tab, "F", r, x)

    g0, nu_ = 7000.0, np.geomspace(1e8, 1e10, 41)
    eb = 1e-4
    errs = []
    for r in (0, 1, 2):
        lhs = (Dcol(5.0 * np.exp(eb), r, g0, nu_) - Dcol(5.0 * np.exp(-eb), r, g0, nu_)) / (2 * eb)
        rhs = Dcol(5.0, r, g0, nu_) + (r + 1) * Dcol(5.0, r + 1, g0, nu_) / (2 * H)
        errs.append(float(np.max(np.abs(lhs - rhs)) / np.max(np.abs(Dcol(5.0, 0, g0, nu_)))))
    out["B_dB_column_identity_max_over_max_D0_r012"] = errs
    # ---- joint (gamma, B): marginal-moment and push-forward routes (own fixed-B tables, no identity used)
    bnj, bwj = leggauss(24)
    bnj, bwj = 5.0 + bnj, 0.5 * bwj
    en, we = leggauss(20)
    en, we = 0.25 * (en + 1), 0.5 * we
    we = we / we.sum()
    seta = np.sqrt(1 - en**2)
    LC = 4 * SIGMA_T / (3 * M_E * C_CGS)
    S_I, S_P = np.zeros(41), np.zeros(41, complex)
    a_marg = np.zeros(NB * 3)
    a_pf = np.zeros(NB * 3)
    gcs = []
    for bv, wv in zip(bnj, bwj):
        gcb = 1.0 / ((LC * (bv * 1e-6) ** 2 / (8 * np.pi) + LC * EV) * 19.0 * MYR)
        gcs.append(gcb)
        g, m = measure("JP", norm=False, p=2.6, gc=gcb)
        htb = averaged_tables(tab, bv * seta, we)
        Ib, Pb = direct(htb, ch, g, m)
        S_I += wv * Ib
        S_P += wv * Pb
        a_marg += wv * bin_moments(g, m)
        a_pf += wv * (bv / 5.0) * bin_moments(g, m, np.sqrt(bv / 5.0))
    ht0 = averaged_tables(tab, 5.0 * seta, we)
    CI0, CP0 = columns(ht0, ch)
    def jerr(ci, cp, a):
        return [float(np.max(np.abs(ci @ a - S_I) / S_I)), float(np.max(np.abs(cp @ a - S_P) / S_I))]
    out["joint_gamma_B"] = {"gamma_c_range": [min(gcs), max(gcs)],
                            "preaveraged_basis_marginal_moments_max_rel_I_P": jerr(CI, CP, a_marg),
                            "pushforward_fixed_B0_max_rel_I_P": jerr(CI0, CP0, a_pf)}
    # ---- GALPROP: refit after removing the turnover-based slab attenuation estimate
    raw_ = np.loadtxt(GALPROP_CSV, delimiter=",", skiprows=1)
    idx_ = np.arange(26, 51)
    nu_g = raw_[idx_, 0]
    tau1 = out["galprop"]["absorption_from_turnover"]["slab"]["tau_at_0.1145GHz"] / raw_[26, 0] ** -2.1
    tau_g = tau1 * nu_g**-2.1
    gI0 = raw_[idx_, 3]
    gIc = gI0 / (-np.expm1(-tau_g) / tau_g)
    tr_, he_ = idx_ % 2 == 0, idx_ % 2 == 1
    CMg = mono_columns(ht, nu_g * 1e9)
    dab = {}
    for lab, dat in (("published", gI0), ("de_absorbed_slab", gIc)):
        sg = 0.01 * dat
        sv = tsvd(CMg[tr_], sg[tr_])
        for R in (9, 12):
            a = tsvd_coef(sv, dat[tr_], sg[tr_], R)
            fr = (CMg @ a - dat) / dat
            dab[f"{lab}_moment_R{R}_held_rms_max_pct"] = [100 * float(np.sqrt(np.mean(fr[he_] ** 2))), 100 * float(np.max(np.abs(fr[he_])))]
        phi, c, _ = fit_law("logcubic", np.log(nu_g)[tr_, None], np.array([1.0]), dat[tr_], sg[tr_])
        fr = (law_predict("logcubic", phi, c, np.log(nu_g)[:, None], np.array([1.0])) - dat) / dat
        dab[f"{lab}_logcubic_held_rms_max_pct"] = [100 * float(np.sqrt(np.mean(fr[he_] ** 2))), 100 * float(np.max(np.abs(fr[he_])))]
    dab["slab_attenuation_at_nodes_26_27_28"] = (1 - (-np.expm1(-tau_g[:3]) / tau_g[:3])).tolist()
    out["galprop_deabsorption_sensitivity"] = dab
    # turnover-fit robustness: degree and range
    rob = []
    for deg in (2, 3, 4):
        for top in (18, 22, 26):
            lo_ = np.arange(0, top)
            ln_ = np.log(raw_[lo_, 0])
            def mdl(pp, ln_=ln_, deg=deg):
                ta = np.exp(pp[-1]) * np.exp(-2.1 * ln_)
                return np.polyval(pp[:-1], ln_) + np.log(-np.expm1(-ta)) - np.log(ta)
            best = None
            for lt in np.linspace(-18, -8, 21):
                rr = optimize.least_squares(lambda pp: mdl(pp) - np.log(raw_[lo_, 3]),
                                            np.r_[np.polyfit(ln_[top // 2:], np.log(raw_[lo_[top // 2:], 3]), deg), lt])
                if best is None or rr.cost < best.cost:
                    best = rr
            rob.append({"deg": deg, "top_GHz": float(raw_[top - 1, 0]), "tau_0.1145": float(np.exp(best.x[-1]) * raw_[26, 0] ** -2.1),
                        "rms_log": float(np.sqrt(np.mean(best.fun**2)))})
    out["galprop_turnover_slab_robustness"] = rob
    # ---- bmom4 (beta_bar and omega_1 both free): parameter correlation at the noise-free optimum
    bdeg = {}
    for k in ("E1", "E4-JP"):
        I = spectra[k][0]
        sg = 0.01 * I
        phi, c, _ = fit_law("bmom4", L[FIT], wt, I[FIT], sg[FIT])
        th0 = np.array([phi[0], np.log(c[0]), c[1] / c[0], c[2] / c[0]])
        def mod(th, I=I):
            e_ = np.exp(th[0] * L[FIT])
            return np.exp(th[1]) * ((e_ * (1 + th[2] * L[FIT] + 0.5 * th[3] * L[FIT] ** 2)) @ wt) / sg[FIT]
        J = np.stack([(mod(th0 + dv) - mod(th0 - dv)) / (2e-6) for dv in 1e-6 * np.eye(4)], axis=1)
        cv = np.linalg.inv(J.T @ J)
        sd_ = np.sqrt(np.diag(cv))
        bdeg[k] = {"theta_beta_lnA_w1_w2": th0.tolist(), "sd": sd_.tolist(),
                   "corr_beta_w1": float(cv[0, 2] / (sd_[0] * sd_[2])), "cond_JtJ": float(np.linalg.cond(J.T @ J))}
    out["bmom4_degeneracy"] = bdeg
    # ---- UPRE tail with the search capped at R <= 14 (Picard range)
    cap = {}
    rng = np.random.default_rng(MY_SEED)
    nf = rng.normal(size=(NREAL, 21))
    nh = rng.normal(size=(NREAL, 20))
    for k in ("E1", "E4-JP"):
        I = spectra[k][0]
        sg = 0.01 * I
        sv = tsvd(CI[FIT], sg[FIT])
        c2 = []
        for i in range(NREAL):
            d = I[FIT] + sg[FIT] * nf[i]
            _, chi = upre(sv, d, sg[FIT])
            rs = int(np.argmin((chi + 2 * np.arange(1, len(chi) + 1))[:14])) + 1
            a = tsvd_coef(sv, d, sg[FIT], rs)
            c2.append(np.sum(((CI[HELD] @ a - (I[HELD] + sg[HELD] * nh[i])) / sg[HELD]) ** 2))
        cap[k] = {"median": float(np.median(c2)), "mean": float(np.mean(c2)),
                  "p95_p99_max": [float(np.percentile(c2, q)) for q in (95, 99, 100)]}
    out["upre_capped_R14_chi2_held"] = cap
    out["wall_clock_s"] = time.time() - T0

    # ---- comparison table with author values (author JSON read as data)
    def A_(e, m, key="max_res_all"):
        return AJ[e]["noise_free"][m][key]

    cmp_rows = []
    def row(q, av, mv, tol_rel=0.05, tol_abs=None):
        ok = (abs(mv - av) <= tol_abs) if tol_abs is not None else (abs(mv - av) <= tol_rel * max(abs(av), 1e-30))
        cmp_rows.append({"quantity": q, "author": av, "verifier": mv, "agree": bool(ok)})

    for e in ("E1", "E2", "E3", "E4-JP", "E5"):
        for m in ("s1", "s3", "logcubic", "bmom4"):
            row(f"{e} {m} noise-free max|res|/I (%)", 100 * A_(e, m), 100 * synth[e][f"{m}_max_res_all"])
        row(f"{e} bmom3 omega2", A_(e, "bmom3", "params")[2] / A_(e, "bmom3", "params")[1], synth[e]["bmom3_omega2"], tol_rel=0.05)
        for R in (2, 7, 9):
            row(f"{e} TSVD R={R} noise-free max|res|/I (%)", 100 * AJ[e]["noise_free_tsvd_curve"][R - 1]["max_res_all"],
                100 * synth[e][f"moment_R{R}_max_res_all"])
            row(f"{e} TSVD R={R} tr(HH^T) held", AJ[e]["noise_free_tsvd_curve"][R - 1]["tr_HHt_held"], synth[e][f"moment_R{R}_tr_HHt_held"])
        row(f"{e} seeded UPRE R*", AJ[e]["seeded"]["moment_UPRE"]["R_star"], synth[e]["seeded_R_star"], tol_abs=0)
        row(f"{e} seeded chi2_fit at R*", AJ[e]["seeded"]["moment_UPRE"]["chi2_fit"], synth[e]["seeded_chi2_fit"], tol_rel=0.02)
        row(f"{e} seeded chi2_held at R*", AJ[e]["seeded"]["moment_UPRE"]["chi2_held"], synth[e]["seeded_chi2_held"], tol_rel=0.02)
        s = AJ[e]["monte_carlo"]["summary"]["moment_UPRE"]
        row(f"{e} MC median chi2_held, basis UPRE", s["chi2_held"]["median"], mc[e]["chi2_held_median"], tol_rel=0.05)
        if "maxP_over_I_all" in s:
            row(f"{e} MC median max|dP|/I basis (%)", 100 * s["maxP_over_I_all"]["median"], mc[e]["maxP_res_over_I_median_pct"], tol_rel=0.08)
    for k, nm in (("E1", "ePL"), ("E2", "eSBPL"), ("E3", "eLP")):
        row(f"{k} {nm} noise-free max|res|/I (%)", 100 * A_(k, nm), 100 * fam[f"{k}_{nm}"]["max_res_all"], tol_rel=0.1)
    for nm in ("s3", "logcubic"):
        row(f"E1 MC median chi2_held {nm}", AJ["E1"]["monte_carlo"]["summary"][nm]["chi2_held"]["median"],
            mc["E1"][f"{nm}_chi2_held_median"], tol_rel=0.05)
    G_ = AJ["GALPROP"]
    for R in (7, 9, 12):
        c = G_["noise_free_tsvd_curve"][R - 1]
        row(f"GALPROP TSVD R={R} held RMS (%)", 100 * c["rms_res_held"], gal[f"moment_R{R}_held_rms_max_pct"][0])
        row(f"GALPROP TSVD R={R} held max (%)", 100 * c["max_res_held"], gal[f"moment_R{R}_held_rms_max_pct"][1])
    for nm in ("s3", "logcubic", "bmom4"):
        row(f"GALPROP {nm} held RMS (%)", 100 * G_["noise_free"][nm]["rms_res_held"], gal[f"{nm}_held_rms_max_pct"][0])
        row(f"GALPROP {nm} held max (%)", 100 * G_["noise_free"][nm]["max_res_held"], gal[f"{nm}_held_rms_max_pct"][1])
    row("GALPROP eSBPL held RMS (%)", 100 * G_["noise_free"]["eSBPL"]["rms_res_held"], gal["eSBPL_held_rms_max_pct"][0], tol_rel=0.1)
    row("GALPROP bmom3 omega2", G_["noise_free"]["bmom3"]["params"][2] / G_["noise_free"]["bmom3"]["params"][1], gal["bmom3_omega2"])
    out["comparison"] = cmp_rows
    OUT_JSON.write_text(json.dumps(out, indent=1, default=float))
    for r in cmp_rows:
        print(f"{r['quantity']:<48s} {r['author']:>12.5g} {r['verifier']:>12.5g} {'OK' if r['agree'] else 'DIFF'}")
    print(json.dumps({k: out[k] for k in ("lib_spot_check", "own_F_table_vs_lib_F_max_rel", "own_G_table_vs_scipy_max_rel",
                                          "thetaF_vs_fd_max_over_maxF", "theta2F_vs_fd_max_over_maxF", "E4_gamma_c_JP", "E5_fA",
                                          "E1_channel_300_vs_600_max_over_I", "direct_vs_author_forward_spectra_max_over_I",
                                          "E1_forward_N2_max_over_I", "families", "E1_seeded_P_at_8p9GHz",
                                          "scaling_identity_max_rel", "B_dB_column_identity_max_over_max_D0_r012", "joint_gamma_B", "galprop_deabsorption_sensitivity", "galprop_turnover_slab_robustness", "bmom4_degeneracy", "upre_capped_R14_chi2_held", "wall_clock_s")},
                     indent=1, default=float))
    print(json.dumps(out["galprop"], indent=1, default=float))
    print(json.dumps(out["monte_carlo_own_seed"], indent=1, default=float))


if __name__ == "__main__":
    main()
