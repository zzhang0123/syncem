"""Independent check of the harmonic-vs-continuum bridge (bridge.py / bridge_results.json).

Written from scratch; imports nothing from bridge*.py. It
  1. loads the paper's harmonic matrix from results_final.json and checks the array layout
     against the npz caches, the configuration block, the noise draw and the refit;
  2. builds the Rybicki-Lightman continuum kernel with its own F/G (F by scipy quad over
     kve(5/3), tabulated on a fine ln x grid; G = x K_2/3 directly), its own channel,
     field-strength, field-direction, azimuth and screen averaging, and the paper's
     5-point stencil, plus analytic ln-gamma derivatives;
  3. recomputes every headline comparison of bridge_results.json;
  4. uses the harmonic probe arrays (bridge_harmonic_probe.npz, produced by the original
     radio_numerics.kernel) only as harmonic data, after spot-checking a few of them by
     re-running the harmonic kernel (inputs/harmonic-radio-fit/radio_numerics.py, run with
     syncem through lib/syncem_compat.py).

Writes verify_bridge_results.json next to this file.  Run: python verify_bridge.py
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import sys
import time
from multiprocessing import get_context
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import numpy as np
import scipy
from numpy.polynomial.legendre import leggauss
from scipy.integrate import quad
from scipy.interpolate import CubicSpline
from scipy.special import gamma as Gamma
from scipy.special import kve

HERE = Path(__file__).resolve().parent
RADIO = HERE / "inputs" / "harmonic-radio-fit"  # radio_numerics.py: only its PACKAGE path edited
LIB = HERE / "lib"
PROBE = HERE / "bridge_harmonic_probe.npz"
BRIDGE_JSON = HERE / "bridge_results.json"
OUT = HERE / "verify_bridge_results.json"

# Physical constants: values of syncem.constants (unchanged since SyncMoments d51f9be); read, not imported.
E_ESU = 4.803204712570263e-10
M_E = 9.1093837139e-28
C_CGS = 2.99792458e10
C_SI_M = 2.99792458e8

# Configuration block of results_final.json (re-derived and checked in check_inputs).
CENTRES = np.geomspace(1e8, 1e10, 41)
HALF = 0.2
EDGES = np.geomspace(300.0, 1e5, 13)
LOG_EDGES = np.log(EDGES)
MIDS = 0.5 * (LOG_EDGES[1:] + LOG_EDGES[:-1])
HB = 0.5 * np.diff(LOG_EDGES)
SLOPE = 2.6
SKY = 0.4 * np.exp(0.5j)
RM_MEAN, RM_SD = 8.0, 0.5
NOISE, SEED, THRESHOLD = 0.01, 20260928, 10.0
STEP = 0.04
X_MIN, X_MAX = 1e-5, 800.0

# ---------------------------------------------------------------- F and G functions


def lnF_plus_x_quad(x: float) -> float:
    """ln F(x) + x with F(x) = x int_x^inf K_5/3(t) dt, by scipy quad on kve."""

    def integrand(s):
        return kve(5.0 / 3.0, x + s) * math.exp(-s)

    cuts = [0.0]
    s = x
    while s < 1.0:
        cuts.append(s)
        s *= 4.0
    cuts += [1.0, 4.0, 16.0, 64.0]
    total = 0.0
    for a, b in zip(cuts[:-1], cuts[1:]):
        total += quad(integrand, a, b, epsabs=0.0, epsrel=1e-13, limit=200)[0]
    return math.log(x) + math.log(total)


class Shapes:
    """F from a cubic spline of ln F + x in u = ln x (nodes by quad); G and K_nu direct."""

    def __init__(self, u_nodes: np.ndarray, g_nodes: np.ndarray):
        self.spline = CubicSpline(u_nodes, g_nodes)
        self.umin, self.umax = u_nodes[0], u_nodes[-1]

    def F(self, x: np.ndarray) -> np.ndarray:
        if np.min(x) < X_MIN:
            raise ValueError(f"x below table: {np.min(x)}")
        u = np.log(np.clip(x, X_MIN, X_MAX))
        out = np.exp(self.spline(u) - np.minimum(x, X_MAX))
        return np.where(x > X_MAX, 0.0, out)

    @staticmethod
    def K(nu: float, x: np.ndarray) -> np.ndarray:
        with np.errstate(under="ignore"):
            return kve(nu, x) * np.exp(-x)

    def G(self, x):
        return x * self.K(2.0 / 3.0, x)

    def F1(self, x):  # dF/dln x
        return self.F(x) - x**2 * self.K(5.0 / 3.0, x)

    def F2(self, x):  # d2F/dln x2
        return self.F(x) - 4.0 / 3.0 * x**2 * self.K(5.0 / 3.0, x) + x**3 * self.K(2.0 / 3.0, x)

    def G1(self, x):
        return x * self.K(2.0 / 3.0, x) / 3.0 - x**2 * self.K(1.0 / 3.0, x)

    def G2(self, x):
        return x * self.K(2.0 / 3.0, x) / 9.0 - 2.0 * x**2 * self.K(1.0 / 3.0, x) + x**3 * self.K(2.0 / 3.0, x)


_SHAPES: Shapes | None = None


def _init_worker(u_nodes, g_nodes):
    global _SHAPES
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _SHAPES = Shapes(u_nodes, g_nodes)


# ---------------------------------------------------------------- continuum kernel


def gl_uniform(lo: float, hi: float, n: int):
    x, w = leggauss(n)
    return 0.5 * (lo + hi) + 0.5 * (hi - lo) * x, 0.5 * w


def bump(t):
    out = np.zeros_like(t)
    inside = np.abs(t) < 1
    out[inside] = np.exp(1.0 - 1.0 / (1.0 - t[inside] ** 2))
    return out


_bx, _bw = leggauss(256)
BUMP_AREA_GL256 = float(np.dot(_bw, bump(_bx)))  # the harmonic code's normalisation constant


def channel_grid(n_nu: int):
    """Frequencies (41, n_nu) and weights for int R_j(nu) f(nu) dnu, R_j as in the paper code."""
    t, w = leggauss(n_nu)
    nu = CENTRES[:, None] * (1.0 + HALF * t[None, :])
    weight = (w * bump(t) / BUMP_AREA_GL256)[None, :] * np.ones((41, 1))
    lam2 = (C_SI_M / nu) ** 2
    screen = np.exp(2j * RM_MEAN * lam2 - 2.0 * RM_SD**2 * lam2**2)
    return nu, weight, screen


def continuum_kernel(task):
    """Continuum (41,3) [I, Re P, Im P] at one gamma; derivative order 0, 1 or 2 in ln x."""
    gamma, n_b, n_eta, n_nu, order = task
    sh = _SHAPES
    fields, fw = gl_uniform(4e-6, 6e-6, n_b)
    etas, ew = gl_uniform(-0.5, 0.5, n_eta)
    bperp = (fields[:, None] * np.sqrt(1.0 - etas[None, :] ** 2)).ravel()
    wfe = (fw[:, None] * ew[None, :]).ravel()
    amp = math.sqrt(3.0) * E_ESU**3 * bperp / (4.0 * math.pi * M_E * C_CGS**2)
    a_b = 3.0 * E_ESU * bperp / (4.0 * math.pi * M_E * C_CGS)
    nu, weight, screen = channel_grid(n_nu)
    x = nu[None, :, :] / (a_b[:, None, None] * gamma**2)
    fI = {0: sh.F, 1: sh.F1, 2: sh.F2}[order](x)
    fQ = -{0: sh.G, 1: sh.G1, 2: sh.G2}[order](x)
    pref = (wfe * amp)[:, None, None] * weight[None, :, :]
    i_val = np.sum(pref * fI, axis=(0, 2))
    p_val = SKY * np.sum(pref * fQ * screen[None, :, :], axis=(0, 2))
    return np.stack([i_val, p_val.real, p_val.imag], axis=-1)


def harmonic_kernel(task):
    """radio_numerics.kernel of the harmonic pipeline (run with syncem) at one gamma."""
    gamma, settings = task
    sys.path.insert(0, str(LIB))
    from syncem_compat import use_syncem  # noqa: E402

    use_syncem()
    import radio_numerics as rn  # noqa: E402

    return rn.kernel(float(gamma), settings)[0]


# ---------------------------------------------------------------- helpers


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def stencil_gammas(step: float):
    return [float(np.exp(MIDS[b] + HB[b] * step * o)) for b in range(12) for o in range(-2, 3)]


def assemble_basis(values: np.ndarray, step: float, masses: np.ndarray):
    """values (12*5, 41, 3) -> basis (41,3,36), pure (12,3,41,3) with the paper's stencil."""
    v = values.reshape(12, 5, 41, 3)
    pure = np.empty((12, 3, 41, 3))
    pure[:, 0] = v[:, 2]
    pure[:, 1] = (v[:, 0] - 8 * v[:, 1] + 8 * v[:, 3] - v[:, 4]) / (12 * step)
    pure[:, 2] = (-v[:, 0] + 16 * v[:, 1] - 30 * v[:, 2] + 16 * v[:, 3] - v[:, 4]) / (24 * step**2)
    basis = np.empty((41, 3, 36))
    for b in range(12):
        for r in range(3):
            basis[:, :, 3 * b + r] = masses[b] * pure[b, r]
    return basis, pure


def analytic_basis(d0, d1, d2, masses):
    """d_r (12,41,3) = <shape^(r) in ln x> at bin centres -> pure D_r in x_b."""
    pure = np.stack([d0, -2.0 * HB[:, None, None] * d1, 2.0 * HB[:, None, None] ** 2 * d2], axis=1)
    basis = np.empty((41, 3, 36))
    for b in range(12):
        for r in range(3):
            basis[:, :, 3 * b + r] = masses[b] * pure[b, r]
    return basis, pure


def truth_nodes(n_e: int = 8):
    x, w = leggauss(n_e)
    norm = (EDGES[-1] ** (1 - SLOPE) - EDGES[0] ** (1 - SLOPE)) / (1 - SLOPE)
    gam, wts, xb = [], [], []
    for b in range(12):
        g = np.exp(MIDS[b] + HB[b] * x)
        gam.append(g)
        wts.append(HB[b] * w * g ** (1 - SLOPE) / norm)
        xb.append(x)
    return np.concatenate(gam), np.concatenate(wts), np.concatenate(xb)


def fit(c, truth, sigma, data, use):
    design = c[use].reshape(-1, 36) / sigma.reshape(-1, 1)
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    keep = s >= THRESHOLD
    est = (vt[keep].T / s[keep]) @ u[:, keep].T @ (data / sigma).ravel()
    return dict(u=u[:, keep], s=s, sr=s[keep], v=vt[keep], est=est, design=design)


def column_ratio(delta, ref_basis):
    """max_j |delta_{j,s,k}| / max_j |C_{j,I,k}| -> (12 bins, 3 r, 3 Stokes)."""
    den = np.max(np.abs(ref_basis[:, 0, :]), axis=0)
    ratio = np.max(np.abs(delta), axis=0) / den[None, :]
    return ratio.T.reshape(12, 3, 3)


def principal_sines(a, b):
    """Sines of principal angles between row spaces of orthonormal a, b (k x n)."""
    resid = b - (b @ a.T) @ a
    return np.sort(np.linalg.svd(resid, compute_uv=False))


def mellin_F(s):
    return 2 ** (s + 1) / (s + 2) * Gamma(s / 2 + 7 / 3) * Gamma(s / 2 + 2 / 3)


def mellin_G(s):
    return 2**s * Gamma(s / 2 + 2 / 3) * Gamma(s / 2 + 4 / 3)


# ---------------------------------------------------------------- step 1: inputs


def check_inputs():
    res = json.loads((RADIO / "results_final.json").read_text())
    arr = {k: np.asarray(v) for k, v in res["arrays"].items()}
    cfg = res["configuration"]
    basis_npz = np.load(RADIO / "trusted_basis_refined.npz")
    truth_npz = np.load(RADIO / "trusted_truth_refined.npz")
    C = arr["response_matrix"]
    S = arr["truth_IQU"]
    a = arr["true_moments"]
    pure = basis_npz["pure_coefficients"]
    masses = np.diff(EDGES ** (1 - SLOPE))
    masses = masses / masses.sum()
    chk = {}
    chk["shapes"] = {"response_matrix": list(C.shape), "pure_coefficients": list(pure.shape)}
    chk["json_vs_npz_basis_max_abs"] = float(np.max(np.abs(C - basis_npz["basis"])))
    chk["json_vs_npz_truth_max_abs"] = float(np.max(np.abs(S - truth_npz["truth"])))
    chk["json_vs_npz_moments_max_abs"] = float(np.max(np.abs(a - truth_npz["moments"])))
    lay = np.stack([masses[b] * pure[b, r] for b in range(12) for r in range(3)], axis=-1)
    chk["column_3b_plus_r_equals_mass_b_times_pure_b_r_max_rel"] = float(
        np.max(np.abs(lay - C)) / np.max(np.abs(C)))
    alt = np.stack([masses[b] * pure[b, r] for r in range(3) for b in range(12)], axis=-1)
    chk["alternative_column_12r_plus_b_max_rel_(should_be_large)"] = float(
        np.max(np.abs(alt - C)) / np.max(np.abs(C)))
    chk["config_bin_masses_max_abs"] = float(np.max(np.abs(masses - np.asarray(cfg["reference_bin_masses"]))))
    chk["config_log_edges_max_abs"] = float(np.max(np.abs(LOG_EDGES - np.asarray(cfg["log_bin_edges"]))))
    chk["config_centres_max_rel"] = float(np.max(np.abs(CENTRES / np.asarray(cfg["channel_centres_Hz"]) - 1)))
    chk["config_scalars"] = {k: cfg[k] for k in ("B_gauss_uniform", "screen_gaussian_mean_sd_rad_m2",
                                                  "channel_halfwidth_fraction", "noise_fraction_of_I",
                                                  "seed", "SVD_threshold", "azimuth_density",
                                                  "eta_density", "mu_density", "gamma_bounds")}
    use = np.arange(0, 41, 2)
    chk["fit_channels_even"] = bool(np.array_equal(arr["fit_channel_indices"], use))
    sigma = NOISE * S[use, 0, None] * np.ones((1, 3))
    chk["noise_sigma_max_rel"] = float(np.max(np.abs(sigma / arr["noise_sigma_IQU"] - 1)))
    z = np.random.default_rng(SEED).normal(size=(21, 3))
    data = S[use] + z * sigma
    chk["noisy_data_reproduced_max_over_sigma"] = float(np.max(np.abs(data - arr["noisy_data_IQU"]) / sigma))
    f = fit(C, S, sigma, data, use)
    chk["refit_singular_values_max_rel"] = float(np.max(np.abs(f["s"] / np.asarray(res["fit"]["singular_values"]) - 1)))
    chk["refit_retained"] = int(len(f["sr"]))
    chk["refit_estimate_max_abs"] = float(np.max(np.abs(f["est"] - np.asarray(res["fit"]["estimated_moments"]))))
    rep = f["u"].T @ ((S - np.einsum("jsk,k->js", C, a))[use] / sigma).ravel()
    sign = np.sign(np.sum(f["v"] * np.asarray(res["fit"]["retained_combinations"]), axis=1))
    chk["refit_representation_bias_max_abs"] = float(np.max(np.abs(
        sign * rep - np.asarray(res["fit"]["representation_mode_bias_over_sigma"]))))
    xg, wg = leggauss(8)
    mom = np.array([np.sum(HB[b] * wg * np.exp(MIDS[b] + HB[b] * xg) ** (1 - SLOPE) * xg**r)
                    for b in range(12) for r in range(3)])
    mom /= ((EDGES[-1] ** (1 - SLOPE) - EDGES[0] ** (1 - SLOPE)) / (1 - SLOPE)) * np.repeat(masses, 3)
    chk["true_moments_recomputed_max_abs"] = float(np.max(np.abs(mom - a)))
    chk["stokes_I_positive_decreasing"] = bool(np.all(S[:, 0] > 0) and np.all(np.diff(S[:, 0]) < 0))
    harm = dict(C=C, S=S, a=a, pure=pure, masses=masses, use=use, sigma=sigma, z=z, data=data,
                fit=f, res=res,
                C_coarse=np.load(RADIO / "trusted_basis.npz")["basis"],
                C_half=np.load(RADIO / "trusted_basis_halfstep.npz")["basis"],
                S_coarse=np.load(RADIO / "trusted_truth.npz")["truth"])
    return chk, harm


# ---------------------------------------------------------------- step 2: F table


def build_table(pool):
    du = 0.004
    u = np.arange(math.log(X_MIN), math.log(X_MAX) + du, du)
    g = np.array(pool.map(lnF_plus_x_quad, np.exp(u).tolist(), chunksize=64))
    sh = Shapes(u, g)
    rng = np.random.default_rng(1)
    utest = rng.uniform(u[0], u[-2], 400)
    gq = np.array(pool.map(lnF_plus_x_quad, np.exp(utest).tolist(), chunksize=16))
    spline_err = float(np.max(np.abs(sh.spline(utest) - gq)))
    import mpmath as mp

    mp.mp.dps = 30
    xs = [1e-4, 1e-2, 0.3, 1.0, 4.0, 30.0, 200.0]
    mp_err = []
    for xv in xs:
        X = mp.mpf(xv)
        Fm = X * mp.quad(lambda t: mp.besselk(mp.mpf(5) / 3, t), mp.linspace(X, X + 60, 61) + [mp.inf])
        Gm = X * mp.besselk(mp.mpf(2) / 3, X)
        mp_err.append([xv, float(abs(sh.F(np.array([xv]))[0] / Fm - 1)),
                       float(abs(sh.G(np.array([xv]))[0] / Gm - 1))])
    mell = {}
    for s in (1.0, 2.0):
        fi = quad(lambda uu: math.exp((s + 1) * uu) * sh.F(np.array([math.exp(uu)]))[0],
                  u[0], u[-1], limit=1000, epsrel=1e-12)[0]
        gi = quad(lambda uu: math.exp((s + 1) * uu) * sh.G(np.array([math.exp(uu)]))[0],
                  u[0], u[-1], limit=1000, epsrel=1e-12)[0]
        mell[f"s={s}"] = {"F_rel_err": abs(fi / mellin_F(s) - 1), "G_rel_err": abs(gi / mellin_G(s) - 1)}
    tb, wb = leggauss(4096)
    exact_area = quad(lambda t: math.exp(1 - 1 / (1 - t * t)), -1, 1, epsabs=0, epsrel=1e-13, limit=400)[0]
    info = {
        "table": f"u=ln x on [{u[0]:.3f},{u[-1]:.3f}], step {du}, {len(u)} quad nodes, cubic spline of ln F + x",
        "spline_vs_direct_quad_400_random_max_abs_lnF": spline_err,
        "mpmath_dps30_rel_err_[x,F,G]": mp_err,
        "mellin_integral_checks": mell,
        "bump_area_gl256_minus_quad_rel": BUMP_AREA_GL256 / exact_area - 1,
        "bump_area_gl4096_minus_quad_rel": float(np.dot(wb, bump(tb))) / exact_area - 1,
    }
    return (u, g), info


# ---------------------------------------------------------------- step 3: compute


def run_continuum(pool, table):
    probe = np.load(PROBE)
    tg, tw, _ = truth_nodes(8)
    tg32, tw32, _ = truth_nodes(32)
    jobs = {
        "fd04": [(g, 8, 7, 400, 0) for g in stencil_gammas(0.04)],
        "fd04_nu800": [(g, 8, 7, 800, 0) for g in stencil_gammas(0.04)],
        "fd02": [(g, 8, 7, 400, 0) for g in stencil_gammas(0.02)],
        "fd01": [(g, 8, 7, 400, 0) for g in stencil_gammas(0.01)],
        "truth_E32": [(g, 8, 7, 400, 0) for g in tg32],
        "fd04_coarse": [(g, 6, 5, 400, 0) for g in stencil_gammas(0.04)],
        "fd02_coarse": [(g, 6, 5, 400, 0) for g in stencil_gammas(0.02)],
        "an1": [(float(np.exp(m)), 8, 7, 400, 1) for m in MIDS],
        "an2": [(float(np.exp(m)), 8, 7, 400, 2) for m in MIDS],
        "truth": [(g, 8, 7, 400, 0) for g in tg],
        "truth_nu800": [(g, 8, 7, 800, 0) for g in tg],
        "truth_BE16": [(g, 16, 16, 400, 0) for g in tg],
        "probe": [(float(g), 8, 7, 400, 0) for g in probe["gamma"]],
    }
    flat = [(k, i, t) for k, v in jobs.items() for i, t in enumerate(v)]
    t0 = time.perf_counter()
    out = pool.map(continuum_kernel, [t for _, _, t in flat], chunksize=2)
    seconds = time.perf_counter() - t0
    vals = {k: np.empty((len(v), 41, 3)) for k, v in jobs.items()}
    for (k, i, _), o in zip(flat, out):
        vals[k][i] = o
    vals["_tw32"] = tw32
    return vals, seconds, (tg, tw)


def run_harmonic_spotchecks(pool):
    probe = np.load(PROBE)
    adopted = dict(field=8, eta=7, pitch=18, channel=64)
    refined = dict(field=8, eta=7, pitch=26, channel=96)
    tasks = [(float(probe["gamma"][5]), adopted), (float(probe["gamma"][20]), adopted),
             (float(np.exp(MIDS[6])), adopted), (float(probe["gamma"][10]), refined)]
    t0 = time.perf_counter()
    out = pool.map(harmonic_kernel, tasks, chunksize=1)
    seconds = time.perf_counter() - t0
    pure = np.load(RADIO / "trusted_basis_refined.npz")["pure_coefficients"]
    refs = [probe["adopted"][5], probe["adopted"][20], pure[6, 0], probe["refined"][10]]
    labels = ["probe adopted k=5 (gamma=1000)", "probe adopted k=20 (gamma=31623)",
              "cache pure_coefficients[6,0] (bin-centre, adopted)", "probe refined k=10 (gamma=3162)"]
    return {lab: float(np.max(np.abs(o - r)) / np.max(np.abs(r))) for lab, o, r in zip(labels, out, refs)}, seconds


# ---------------------------------------------------------------- step 4: compare

A_B5 = 3.0 * E_ESU * 5e-6 / (4.0 * math.pi * M_E * C_CGS)  # a_B at B_perp = 5 muG


def loc(table):
    i = np.unravel_index(np.argmax(table), table.shape)
    return [int(v) for v in i]


def compare_columns(h, vals):
    m = h["masses"]
    Cc, pure_c = assemble_basis(vals["fd04"], 0.04, m)
    Cc800, _ = assemble_basis(vals["fd04_nu800"], 0.04, m)
    Ca, pure_a = analytic_basis(vals["fd04"][2::5], vals["an1"], vals["an2"], m)
    Ch, S, a = h["C"], h["S"], h["a"]
    I = S[:, 0]
    d = column_ratio(Ch - Cc, Ch)  # (12 bins, 3 r, 3 Stokes)
    gb2 = np.exp(2 * MIDS)[:, None] * d[:, :, 0]
    high = np.exp(MIDS) >= 1e3
    st = column_ratio(Cc - Ca, Ch)
    Cc01, _ = assemble_basis(vals["fd01"], 0.01, m)
    st01 = column_ratio(Cc01 - Ca, Ch)
    out = {
        "channel_nodes_400_vs_800_basis_max_over_colmax": float(np.max(column_ratio(Cc800 - Cc, Ch))),
        "harm_minus_cont_fd04_max_over_colmax_IQU": [float(np.max(d[:, :, s])) for s in range(3)],
        "argmax_bin_r_IQU": [loc(d[:, :, s]) for s in range(3)],
        "table_bin_r_I": d[:, :, 0].tolist(),
        "table_bin_r_Q": d[:, :, 1].tolist(),
        "table_bin_r_U": d[:, :, 2].tolist(),
        "gamma_b2_times_r0_I_per_bin": gb2[:, 0].tolist(),
        "gamma_b2_times_r0_I_range": [float(gb2[:, 0].min()), float(gb2[:, 0].max())],
        "bins_gamma_b_ge_1e3_r0_max_IQU": [float(np.max(d[high, 0, s])) for s in range(3)],
        "bins_8_11_r12_max_over_Stokes": float(np.max(d[8:, 1:, :])),
        "bins_8_11_table_maxStokes": np.max(d[8:], axis=2).tolist(),
        "x_at_bin0_channel0_B5": float(CENTRES[0] / (A_B5 * np.exp(2 * MIDS[0]))),
        "stencil_only_cont_fd04_minus_analytic_max_over_colmax": float(np.max(st)),
        "stencil_fd01_minus_analytic_max_over_colmax": float(np.max(st01)),
        "stencil_fd04_over_fd01_ratio": float(np.max(st) / np.max(st01)),
        "argmax_gamma_b_IQU": [float(np.exp(MIDS[loc(d[:, :, s])[0]])) for s in range(3)],
        "stencil_only_argmax_bin_r_stokes": [int(v) for v in np.unravel_index(np.argmax(st), st.shape)],
        "stencil_only_at_true_moments_max_over_I_IQU": [
            float(v) for v in np.max(np.abs(np.einsum("jsk,k->js", Cc - Ca, a)) / I[:, None], axis=0)],
        "harm_minus_cont_fd04_at_true_moments_max_over_I_IQU": [
            float(v) for v in np.max(np.abs(np.einsum("jsk,k->js", Ch - Cc, a)) / I[:, None], axis=0)],
        "harm_minus_cont_analytic_max_over_colmax_IQU": [
            float(np.max(column_ratio(Ch - Ca, Ch)[:, :, s])) for s in range(3)],
    }
    return out, dict(Cc=Cc, Ca=Ca, pure_c=pure_c, d=d)


def compare_truth(h, vals, nodes):
    tg, tw = nodes
    Sc = np.einsum("n,njs->js", tw, vals["truth"])
    S800 = np.einsum("n,njs->js", tw, vals["truth_nu800"])
    S16 = np.einsum("n,njs->js", tw, vals["truth_BE16"])
    I = h["S"][:, 0]
    dS = h["S"] - Sc
    dP = np.abs((h["S"][:, 1] - Sc[:, 1]) + 1j * (h["S"][:, 2] - Sc[:, 2]))
    out = {
        "harm_minus_cont_max_over_I_IQU": [float(v) for v in np.max(np.abs(dS) / I[:, None], axis=0)],
        "argmax_channel_IQU": [int(v) for v in np.argmax(np.abs(dS) / I[:, None], axis=0)],
        "argmax_channel_GHz_IQU": [float(CENTRES[v] / 1e9) for v in np.argmax(np.abs(dS) / I[:, None], axis=0)],
        "cont_energy_nodes_8_vs_32_max_over_I": float(np.max(np.abs(np.einsum("n,njs->js", vals["_tw32"], vals["truth_E32"]) - Sc) / I[:, None])),
        "absP_max_over_I": float(np.max(dP / I)),
        "cont_channel_nodes_400_vs_800_max_over_I": float(np.max(np.abs(S800 - Sc) / I[:, None])),
        "cont_BE_8x7_vs_16x16_max_over_I": float(np.max(np.abs(S16 - Sc) / I[:, None])),
        "harm_minus_cont_BE16_max_over_I_IQU": [
            float(v) for v in np.max(np.abs(h["S"] - S16) / I[:, None], axis=0)],
        "harm_coarse_minus_adopted_max_over_I": float(np.max(np.abs(h["S_coarse"] - h["S"]) / I[:, None])),
    }
    return out, Sc, dS


def aligned(v_ref, v):
    return np.sign(np.sum(v_ref * v, axis=1))[:, None]


def compare_fits(h, Cc, Sc, Ca):
    use, sig_h, z = h["use"], h["sigma"], h["z"]
    fh = h["fit"]
    a = h["a"]
    fc_hs = fit(Cc, h["S"], sig_h, h["data"], use)  # continuum basis, harmonic sigma and data
    fa_hs = fit(Ca, h["S"], sig_h, h["data"], use)  # analytic-derivative continuum columns
    sig_c = NOISE * Sc[use, 0, None] * np.ones((1, 3))
    data_c = Sc[use] + z * sig_c
    fc = fit(Cc, Sc, sig_c, data_c, use)  # continuum pipeline, own truth/sigma, same draws
    sh, sc, so = fh["s"], fc_hs["s"], fc["s"]

    def rep(f, C, S, sig):
        return f["u"].T @ ((S - np.einsum("jsk,k->js", C, a))[use] / sig).ravel()

    rep_h = rep(fh, h["C"], h["S"], sig_h)
    sgn = aligned(fh["v"], fc["v"])[:, 0]
    rep_c = sgn * rep(fc, Cc, Sc, sig_c)
    pull_h = fh["sr"] * (fh["v"] @ (fh["est"] - a))
    pull_c = sgn * fc["sr"] * (fc["v"] @ (fc["est"] - a))
    shift = fh["sr"] * (fh["v"] @ (fc_hs["est"] - fh["est"]))
    rel = np.abs(sc / sh - 1)
    out = {
        "s7_s8_harmonic": sh[6:8].tolist(),
        "s7_s8_continuum_harmonic_sigma": sc[6:8].tolist(),
        "s7_s8_continuum_own_sigma": so[6:8].tolist(),
        "s7_s8_rel_diff_harmonic_sigma": (sc[6:8] / sh[6:8] - 1).tolist(),
        "s7_s8_rel_diff_own_sigma": (so[6:8] / sh[6:8] - 1).tolist(),
        "s1_s15_max_rel_diff": float(np.max(rel[:15])),
        "s16_s23_rel_diff_range": [float(rel[15:23].min()), float(rel[15:23].max())],
        "singular_values_harmonic": sh.tolist(),
        "singular_values_continuum": sc.tolist(),
        "retained_harmonic_cont_contown": [len(fh["sr"]), len(fc_hs["sr"]), len(fc["sr"])],
        "principal_sines_coeff_space_harmonic_sigma": principal_sines(fh["v"], fc_hs["v"]).tolist(),
        "principal_sines_coeff_space_analytic_columns": principal_sines(fh["v"], fa_hs["v"]).tolist(),
        "principal_sines_coeff_space_own_sigma": principal_sines(fh["v"], fc["v"]).tolist(),
        "representation_bias_harmonic": rep_h.tolist(),
        "representation_bias_diff_max": float(np.max(np.abs(rep_h - rep_c))),
        "pull_harmonic": pull_h.tolist(),
        "pull_diff_max": float(np.max(np.abs(pull_h - pull_c))),
        "harmonic_data_continuum_basis_shift_over_sigma": shift.tolist(),
        "harmonic_data_continuum_basis_shift_max": float(np.max(np.abs(shift))),
        "finite_representation_max_over_I_harm_cont": [
            float(np.max(np.abs(h["S"] - np.einsum("jsk,k->js", h["C"], a)) / h["S"][:, :1])),
            float(np.max(np.abs(Sc - np.einsum("jsk,k->js", Cc, a)) / Sc[:, :1]))],
    }
    return out


def c_functions(Kh, Kc, gam):
    """c_I, c_P on (n_gamma, 41) and selection by per-gamma peak and x <= 30."""
    g2 = gam[:, None] ** 2
    cI = g2 * (Kh[..., 0] / Kc[..., 0] - 1)
    cP = g2 * np.abs((Kh[..., 1] - Kc[..., 1]) + 1j * (Kh[..., 2] - Kc[..., 2])) / Kc[..., 0]
    x = CENTRES[None, :] / (A_B5 * g2)
    sel = (Kc[..., 0] > 1e-12 * np.max(Kc[..., 0], axis=1, keepdims=True)) & (x <= 30)
    return cI, cP, x, sel


def probe_analysis(vals):
    pr = np.load(PROBE)
    gam = pr["gamma"]
    Kc = vals["probe"]
    out = {}
    cI, cP, x, sel = c_functions(pr["refined"], Kc, gam)
    cIa, cPa, _, _ = c_functions(pr["adopted"], Kc, gam)
    out["points_selected"] = int(sel.sum())
    sel_glob = (Kc[..., 0] > 1e-12 * Kc[..., 0].max()) & (x <= 30)
    out["points_selected_global_peak_rule"] = int(sel_glob.sum())
    dI = cI / gam[:, None] ** 2
    ns, xs = [], []
    for k in range(len(gam) - 1):
        for j in range(41 - 4):
            if sel[k, j] and sel[k + 1, j + 4]:
                ns.append(math.log(dI[k, j] / dI[k + 1, j + 4]) / math.log(gam[k + 1] / gam[k]))
                xs.append(x[k, j])
    ns = np.array(ns)
    out["exponent_refined"] = {"n_pairs": len(ns), "median": float(np.median(ns)),
                               "p05_p95": np.percentile(ns, [5, 95]).tolist(),
                               "min_max": [float(ns.min()), float(ns.max())],
                               "x_range": [float(min(xs)), float(max(xs))]}
    xv, cv = x[sel], cI[sel]
    out["c_I_all_negative"] = bool(np.all(cv < 0))
    out["abs_c_I_range"] = [float(np.abs(cv).min()), float(np.abs(cv).max())]
    out["c_P_max_refined"] = float(cP[sel].max())
    A = np.stack([xv ** (-2 / 3), np.ones_like(xv), xv], axis=1) / np.abs(cv)[:, None]
    coef = np.linalg.lstsq(A, np.ones_like(xv), rcond=None)[0]
    fitv = -(coef[0] * xv ** (-2 / 3) + coef[1] + coef[2] * xv)
    rr = np.abs(fitv / cv - 1)
    out["c_I_fit_relative_lsq"] = {"alpha_beta_delta": coef.tolist(), "median_rel_res": float(np.median(rr)),
                                   "max_rel_res": float(rr.max()), "x_range": [float(xv.min()), float(xv.max())]}
    bc = np.array([0.364654, 0.179217, 0.547487])
    rb = np.abs(-(bc[0] * xv ** (-2 / 3) + bc[1] + bc[2] * xv) / cv - 1)
    out["bridge_fit_on_my_c_I"] = {"median_rel_res": float(np.median(rb)), "max_rel_res": float(rb.max())}
    near = lambda x0: float(np.abs(cv[np.argmin(np.abs(np.log10(xv / x0)))]))  # noqa: E731
    out["abs_c_I_data_nearest_point_x"] = {"1e-3": near(1e-3), "0.6": near(0.6), "27": near(27.0)}
    out["abs_c_I_fit_at_x"] = {k: float(coef[0] * x0 ** (-2 / 3) + coef[1] + coef[2] * x0)
                               for k, x0 in (("1e-3", 1e-3), ("0.6", 0.6), ("27", 27.0))}
    out["adopted_minus_refined_max_gamma2_rel_I"] = float(np.max(np.abs(cIa - cI)[sel]))
    r2 = {}
    for i, g in enumerate(pr["refined2_gamma"]):
        k = int(np.argmin(np.abs(gam - g)))
        c2I, c2P, _, s2 = c_functions(pr["refined2"][i:i + 1], Kc[k:k + 1], gam[k:k + 1])
        if g >= 500:
            r2[f"{g:.0f}"] = [float(np.max(np.abs(c2I - cI[k:k + 1])[s2])),
                              float(np.max(np.abs(c2P - cP[k:k + 1])[s2]))]
    out["refined2_minus_refined_c_units_I_P_by_gamma"] = r2
    return out, coef


def holdout_and_envelope(h, pure_c, nodes, vals, bridge):
    ef = bridge["error_function"]
    nI = np.asarray(ef["envelope_I"]["log10x_nodes"]); eI = np.asarray(ef["envelope_I"]["c_env"])
    nP = np.asarray(ef["envelope_P"]["log10x_nodes"]); eP = np.asarray(ef["envelope_P"]["c_env"])
    gb = np.exp(MIDS)
    cI, cP, x, sel = c_functions(h["pure"][:, 0], pure_c[:, 0], gb)
    lx = np.log10(x)
    out = {"bin_centre_points": int(sel.sum()),
           "max_abs_cI_over_env": float(np.max((np.abs(cI) / np.interp(lx, nI, eI))[sel])),
           "max_cP_over_env": float(np.max((cP / np.interp(lx, nP, eP))[sel])),
           "max_abs_cI_bin_centres": float(np.max(np.abs(cI)[sel])),
           "max_cP_bin_centres": float(np.max(cP[sel]))}
    tg, tw = nodes
    KI = vals["truth"][..., 0]  # (96, 41) continuum
    xn = CENTRES[None, :] / (A_B5 * tg[:, None] ** 2)
    fI = np.where(xn > 30, 1.0, np.minimum(np.interp(np.log10(xn), nI, eI) / tg[:, None] ** 2, 1.0))
    fP = np.where(xn > 30, 2.0, np.minimum(np.interp(np.log10(xn), nP, eP) / tg[:, None] ** 2, 2.0))
    E_I = np.einsum("n,nj->j", tw, fI * KI)
    E_P = np.einsum("n,nj->j", tw, fP * KI)
    I = h["S"][:, 0]
    ec = bridge["envelope_checks"]
    E_I_b = np.asarray(ec["E_I_over_I_per_channel"]) * I
    E_P_b = np.asarray(ec["E_P_over_I_per_channel"]) * I
    out["E_reconstructed_vs_bridge_max_rel_I_P"] = [float(np.max(np.abs(E_I / E_I_b - 1))),
                                                     float(np.max(np.abs(E_P / E_P_b - 1)))]
    out["E_over_I_max_bridge_I_P"] = [float(np.max(E_I_b / I)), float(np.max(E_P_b / I))]
    mg2 = np.einsum("n,nj->j", tw, KI / tg[:, None] ** 2) / np.einsum("n,nj->j", tw, KI)
    out["E_I_over_meang2_I_range"] = [float(v) for v in (E_I_b / (mg2 * I)).min(keepdims=True).tolist()
                                      + (E_I_b / (mg2 * I)).max(keepdims=True).tolist()]
    out["E_P_over_meang2_I_range"] = [float((E_P_b / (mg2 * I)).min()), float((E_P_b / (mg2 * I)).max())]
    out["meang2_inv_sqrt_ch0_ch40"] = [float(mg2[0] ** -0.5), float(mg2[-1] ** -0.5)]
    return out, E_I_b, E_P_b, mg2


def propagation(h, Cc, dS, E_I, E_P, mg2, coef):
    use, sig = h["use"], h["sigma"]
    fh = h["fit"]
    I = h["S"][:, 0]
    dCa = np.einsum("jsk,k->js", h["C"] - Cc, h["a"])
    E3 = np.stack([E_I, E_P, E_P], axis=1)
    ew = (E3[use] / sig).ravel()
    U = fh["u"]
    Kmat = (fh["v"].T / fh["sr"]) @ U.T  # moments per whitened datum
    bound = np.abs(U.T) @ ew
    act_t = U.T @ (dS[use] / sig).ravel()
    act_b = U.T @ (dCa[use] / sig).ravel()
    mb = np.abs(Kmat) @ ew
    ma = Kmat @ (dS[use] / sig).ravel()
    dP = np.abs(dS[:, 1] + 1j * dS[:, 2])
    dPa = np.abs(dCa[:, 1] + 1j * dCa[:, 2])
    ceff = dS[:, 0] / (I * mg2)
    s = (SLOPE - 3) / 2
    pred = -(coef[0] * mellin_F(s + 1 - 2 / 3) + coef[1] * mellin_F(s + 1) + coef[2] * mellin_F(s + 2)) / mellin_F(s + 1)
    bc = [0.364654, 0.179217, 0.547487]
    pred_b = -(bc[0] * mellin_F(s + 1 - 2 / 3) + bc[1] * mellin_F(s + 1) + bc[2] * mellin_F(s + 2)) / mellin_F(s + 1)
    return {
        "mode_bound_max_over_sigma": float(bound.max()),
        "mode_actual_truth_max_over_sigma": float(np.max(np.abs(act_t))),
        "mode_actual_basis_at_a_max_over_sigma": float(np.max(np.abs(act_b))),
        "bound_holds": bool(np.all(np.abs(act_t) <= bound) and np.all(np.abs(act_b) <= bound)),
        "moment_bound_max": float(mb.max()),
        "moment_actual_max": float(np.max(np.abs(ma))),
        "coverage_truth_I_P": [float(np.max(np.abs(dS[:, 0]) / E_I)),
                               float(np.max(np.maximum.reduce([np.abs(dS[:, 1]), np.abs(dS[:, 2]), dP]) / E_P))],
        "coverage_basis_I_P": [float(np.max(np.abs(dCa[:, 0]) / E_I)),
                               float(np.max(np.maximum.reduce([np.abs(dCa[:, 1]), np.abs(dCa[:, 2]), dPa]) / E_P))],
        "c_eff_I_range": [float(ceff.min()), float(ceff.max())],
        "c_eff_I_spread_rel": float((ceff.max() - ceff.min()) / abs(ceff.mean())),
        "c_eff_P_max": float(np.max(dP / (I * mg2))),
        "c_eff_predicted_unbounded_power_law_my_fit": float(pred),
        "c_eff_predicted_unbounded_power_law_bridge_coefficients": float(pred_b),
    }


def stencil_checks(h, vals, d_meas, pure_c, coef):
    m = h["masses"]
    Cc04, _ = assemble_basis(vals["fd04_coarse"], 0.04, m)
    Cc02, _ = assemble_basis(vals["fd02_coarse"], 0.02, m)
    dh = column_ratio(h["C_half"] - h["C_coarse"], h["C_coarse"])
    dc = column_ratio(Cc02 - Cc04, h["C_coarse"])
    dd = column_ratio((h["C_half"] - h["C_coarse"]) - (Cc02 - Cc04), h["C_coarse"])
    pr = np.load(PROBE)
    bins = [int(b) for b in pr["stencil_bins"]]
    colmax = np.max(np.abs(h["C"][:, 0, :]), axis=0).reshape(12, 3)
    ref = np.max(np.abs((pr["stencil_refined_pure"] - pure_c[bins]) * m[bins, None, None, None]),
                 axis=(2, 3)) / colmax[bins]
    adp = np.max(np.abs((h["pure"][bins] - pure_c[bins]) * m[bins, None, None, None]), axis=(2, 3)) / colmax[bins]
    # model Delta K_I = c_I(x) gamma^-2 K_I(cont) on the stencil, compared with the measured columns
    v = vals["fd04"].reshape(12, 5, 41, 3)[..., 0]
    gs = np.array(stencil_gammas(0.04)).reshape(12, 5)
    xs = CENTRES[None, None, :] / (A_B5 * gs[..., None] ** 2)
    model = -(coef[0] * xs ** (-2 / 3) + coef[1] + coef[2] * xs) / gs[..., None] ** 2 * v
    bc = np.array([0.364654, 0.179217, 0.547487])
    model_b = -(bc[0] * xs ** (-2 / 3) + bc[1] + bc[2] * xs) / gs[..., None] ** 2 * v

    def stc(w):
        return np.stack([w[:, 2], (w[:, 0] - 8 * w[:, 1] + 8 * w[:, 3] - w[:, 4]) / (12 * STEP),
                         (-w[:, 0] + 16 * w[:, 1] - 30 * w[:, 2] + 16 * w[:, 3] - w[:, 4]) / (24 * STEP**2)], axis=1)

    out = {}
    meas = (h["pure"][..., 0] - pure_c[..., 0]) * m[:, None, None]  # (12,3,41)
    for lab, mod in (("my_fit", model), ("bridge_coefficients", model_b)):
        pd = stc(mod) * m[:, None, None]
        maxratio = np.max(np.abs(pd), axis=2) / np.max(np.abs(meas), axis=2)
        resid = np.max(np.abs(pd - meas), axis=2) / np.max(np.abs(meas), axis=2)
        out[f"model_{lab}"] = {
            "pred_over_meas_max_ratio_table": maxratio.tolist(),
            "max_dev_of_ratio_r1_bins0_6": float(np.max(np.abs(maxratio[:7, 1] - 1))),
            "max_dev_of_ratio_r2_bins0_3": float(np.max(np.abs(maxratio[:4, 2] - 1))),
            "meas_over_pred_r2_bins4_7": (1 / maxratio[4:8, 2]).tolist(),
            "channelwise_resid_over_meas_r1_bins0_6_max": float(np.max(resid[:7, 1])),
            "channelwise_resid_over_meas_r2_bins0_3_max": float(np.max(resid[:4, 2])),
        }
    out["measured_r2_bins4_7_over_colmax"] = d_meas[4:8, 2, 0].tolist()
    out["halfstep_harmonic_max_over_colmax"] = float(np.max(dh))
    out["halfstep_continuum_max_over_colmax"] = float(np.max(dc))
    out["halfstep_difference_max_over_colmax"] = float(np.max(dd))
    out["refined_harmonic_minus_continuum_bins8_11"] = ref.tolist()
    out["refined_harmonic_minus_continuum_bins8_11_max"] = float(ref.max())
    out["adopted_harmonic_minus_continuum_bins8_11_max"] = float(adp.max())
    return out


def main():
    t_start = time.perf_counter()
    report = {"script": Path(__file__).name, "script_sha256": sha256(Path(__file__))}
    inputs = [RADIO / f for f in ("results_final.json", "trusted_basis_refined.npz", "trusted_truth_refined.npz",
                                  "trusted_basis.npz", "trusted_basis_halfstep.npz", "trusted_truth.npz",
                                  "radio_numerics.py", "stable_lines.py")]
    inputs += [PROBE, BRIDGE_JSON, LIB / "syncem_compat.py"]
    report["input_sha256"] = {p.resolve().relative_to(HERE).as_posix(): sha256(p) for p in inputs}
    import mpmath

    sys.path.insert(0, str(LIB))
    from syncem_compat import check_record, use_syncem  # noqa: E402

    syncem = use_syncem()
    report["versions"] = {"python": platform.python_version(), "numpy": np.__version__,
                          "scipy": scipy.__version__, "mpmath": mpmath.__version__,
                          "jax": getattr(sys.modules.get("jax"), "__version__", "not imported"),
                          "syncem": syncem.__version__}
    report["syncem_check"] = check_record()
    report["inputs_and_layout"], h = check_inputs()
    bridge = json.loads(BRIDGE_JSON.read_text())
    ctx = get_context("spawn")
    with ctx.Pool(20, initializer=_init_worker, initargs=(np.zeros(4) + np.arange(4), np.zeros(4))) as pool:
        table, report["shape_functions"] = build_table(pool)
    with ctx.Pool(20, initializer=_init_worker, initargs=table) as pool:
        vals, sec, nodes = run_continuum(pool, table)
        report["continuum_seconds_wall"] = sec
        report["harmonic_spotcheck_max_abs_over_max"], report["harmonic_spotcheck_seconds"] = (
            run_harmonic_spotchecks(pool))
    report["columns"], cc = compare_columns(h, vals)
    report["truth"], Sc, dS = compare_truth(h, vals, nodes)
    report["fits"] = compare_fits(h, cc["Cc"], Sc, cc["Ca"])
    report["probe"], coef = probe_analysis(vals)
    env, E_I, E_P, mg2 = holdout_and_envelope(h, cc["pure_c"], nodes, vals, bridge)
    report["holdout_and_envelope"] = env
    report["propagation_and_compact"] = propagation(h, cc["Cc"], dS, E_I, E_P, mg2, coef)
    report["stencil"] = stencil_checks(h, vals, cc["d"], cc["pure_c"], coef)
    report["wall_seconds"] = time.perf_counter() - t_start
    OUT.write_text(json.dumps(report, indent=1, default=float))
    print(json.dumps({k: report[k] for k in report if k not in ("input_sha256",)}, indent=1, default=float)[:20000])


if __name__ == "__main__":
    main()
