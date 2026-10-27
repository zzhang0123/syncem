"""Physical parameters from the fitted moment combinations, continuum kernels.

For each task (ensemble, parametric family): paper geometry, separable N = 2 basis on
U[4,6] muG, noise 0.01 I_j, 21 fitted / 20 withheld channels.
  * seeded realisation (paper seed for the fitted rows): UPRE R*, compressed fit
    chi2_beta(phi) (primary) and full linear likelihood (reference), multi-start check,
    withheld chi2, invariants;
  * noise-free fits: representation bias in sd units, its linear prediction G eps_w and the
    envelope |G| E_w with E the measured N = 2 Taylor envelope (forward.py definition);
  * exact-forward reference (no moment truncation) for the smooth families;
  * compressed fits versus R = 3..14 (seeded data);
  * N_REAL noise realisations: R* per realisation (primary), fixed R_seed, full; pulls,
    coverage, sd_comp/sd_full, invariant pulls.
Degeneracy (params_degeneracy.py): kernel scaling identity, derivative identity, Jacobian
null vector, E2 (ln B, ln gamma_b) ridge; E1M (Maxwellian data, U[4,6] basis); KP data
with the JP family; E2a exploratory run on an UNVERIFIED transcription of Orlando (2018)
Table 5 (not for the paper until checked against the PDF).  E2a runs only when
inputs/orlando2018_PDDE_all_electron_LIS_UNVERIFIED.csv is present.  That file is not
distributed with the study; without it params.py skips E2a, prints a line saying so and
records the skip under E2a_input in params_results.json.  No other task depends on E2a.

Writes params_results.json and params_realisations.npz.  Run: python params.py
"""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from dataclasses import dataclass  # noqa: E402
from functools import cached_property  # noqa: E402
from math import factorial  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402
from scipy.interpolate import CubicSpline  # noqa: E402
from scipy.optimize import least_squares  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))
import basis as B  # noqa: E402
import ensembles as E  # noqa: E402
import kernels as K  # noqa: E402
import paramlib as P  # noqa: E402
import params_degeneracy as D  # noqa: E402
from config import PAPER, PAPER_MAXWELL  # noqa: E402
from provenance import provenance, relative, sha256, write_json  # noqa: E402

T0 = time.time()
N_REAL = int(os.environ.get("PARAMS_NREAL", "1000"))
SEED = PAPER.seed
WORKERS = 28
NU_CONV = float(np.log(K.a1 * PAPER.mean_bperp))  # ln(a1 <B_perp>) of the assumed U[4,6] field
LIS_CSV = HERE / "inputs" / "orlando2018_PDDE_all_electron_LIS_UNVERIFIED.csv"
ME_MEV = 0.51099895
TIMINGS = {}
MULTISTART_SCALE = {"PL": [0.3], "SBPL_s2": [0.3, 0.3, 0.5], "SBPL_sfree": [0.3, 0.3, 0.5, 0.5],
                    "MIX": [0.3, 0.05], "JP": [0.3, 0.7]}


def tic(label, t):
    TIMINGS[label] = TIMINGS.get(label, 0.0) + time.time() - t


# ----------------------------------------------------------------------------- E2a (unverified)
@dataclass(frozen=True)
class TabulatedLIS(E.Ensemble1D):
    """N(gamma) proportional to J(E) = E2J / E^2 (E kinetic, MeV); cubic spline of ln E2J in ln E."""

    name: str = "E2a-unverified"
    lnE: tuple = ()
    lnE2J: tuple = ()

    @cached_property
    def spline(self):
        return CubicSpline(np.array(self.lnE), np.array(self.lnE2J))

    def shape(self, g):
        Ek = (np.asarray(g, float) - 1.0) * ME_MEV
        return np.exp(self.spline(np.log(Ek))) / Ek**2


def load_lis():
    rows = [r.split(",") for r in LIS_CSV.read_text().splitlines() if r and not r.startswith(("#", "E_"))]
    E_mev = np.array([float(r[0]) for r in rows])
    e2j = np.array([float(r[1]) for r in rows])
    knots = 1.0 + E_mev / ME_MEV
    kinks = tuple(float(k) for k in knots if 300.0 < k < 1e5)
    return TabulatedLIS(lnE=tuple(np.log(E_mev)), lnE2J=tuple(np.log(e2j)), kinks=kinks)


# ----------------------------------------------------------------------------- helpers
_GRID3 = {}


def envelope_N2(cfg, abs3, factor):
    """Measured N = 2 Taylor envelope (J, 3) (forward.py definition, 65-point grid)."""
    if id(cfg) not in _GRID3:
        _GRID3[id(cfg)] = B.bin_derivative_grid(cfg, 3)
    _, dI, dQ = _GRID3[id(cfg)]
    p = factor * dQ
    env = np.stack([abs3 @ np.max(np.abs(dI), axis=1), abs3 @ np.max(np.abs(p.real), axis=1),
                    abs3 @ np.max(np.abs(p.imag), axis=1)], axis=-1)
    return env / factorial(3)


class FineTable:
    """Exact-forward reference for smooth families: Kbar on 64 GL nodes per ln-gamma bin."""

    def __init__(self, cfg, n=64):
        t, w = leggauss(n)
        u = (np.log(cfg.gamma_b)[:, None] + cfg.h * t[None, :]).ravel()
        self.g = np.exp(u)
        self.wq = np.tile(cfg.h * w, cfg.n_bins) * self.g
        kI, kQ = B.kbar(cfg, self.g, order=0)
        self.K = B.to_stokes(kI[0], kQ[0], cfg.azimuth_factor)  # (n_g, J, 3)

    def spectrum(self, key, phi):
        return np.einsum("g,gjs->js", self.wq * family_density(key, phi, self.g), self.K)


def family_density(key, phi, g):
    """N(gamma) of a smooth family in fit coordinates (per unit gamma)."""
    th = P.Family(key).to_lib(phi)[0]
    if key == "PL":
        n = (g / E.GAMMA0) ** (-th[1])
    elif key.startswith("SBPL"):
        y = g / np.exp(th[3])
        n = y ** (-th[1]) * (1.0 + y ** th[4]) ** (-(th[2] - th[1]) / th[4])
    elif key == "MIX":
        y = np.log(g / E.GAMMA0)
        n = np.exp(-th[1] * y + 0.5 * th[2] * y**2)
    else:
        raise ValueError(key)
    return th[0] / E.GAMMA0 * n


def exact_forward_fit(spec_fn, prob, d_fit, phi0):
    """Full whitened chi2 with an exact forward model spec_fn(phi) -> (J, 3)."""
    fit = prob.cfg.fit_idx

    def res(phi):
        return ((spec_fn(phi)[fit] - d_fit) / prob.sig[fit]).ravel()

    sol = least_squares(res, phi0, method="lm", x_scale="jac", xtol=1e-12, ftol=1e-12, gtol=1e-12)
    J = sol.jac
    cov = np.linalg.inv(J.T @ J)
    return {"phi": sol.x, "sd": np.sqrt(np.diag(cov)), "cov": cov, "chi2": float(2 * sol.cost),
            "nfev": int(sol.nfev), "success": bool(sol.success)}


def starts(key, center, n=12, seed=7):
    rng = np.random.default_rng(seed)
    sc = np.array([0.0] + MULTISTART_SCALE[key])
    return [center] + [center + sc * rng.uniform(-1, 1, size=sc.size) for _ in range(n - 1)]


def fit_summary(fam, cfg_field, prob, f, dhw, true_inv=None, phi_true=None):
    a = fam.moments(prob.cfg, f["phi"])[0]
    # the analyst's invariants use the ASSUMED field statistics (the basis config);
    # true_inv (if given) uses the field that generated the data
    inv, inv_sd = fam.invariants_with_errors(prob.cfg, f["phi"], f["cov"])
    out = {"phi": f["phi"], "sd": f["sd"], "chi2": f["chi2"],
           "chi2_held": float(np.sum((dhw - prob.Wh @ a) ** 2)), "invariants": inv, "invariants_sd": inv_sd}
    if phi_true is not None:
        out["phi_minus_truth"] = f["phi"] - phi_true
        out["pull_vs_truth"] = (f["phi"] - phi_true) / f["sd"]
    if true_inv is not None:
        out["invariants_minus_truth"] = inv - true_inv
        out["invariant_pull_vs_truth"] = (inv - true_inv) / inv_sd
    return out


# ----------------------------------------------------------------------------- tasks
def prepare_tasks():
    """Direct spectra, truths and envelopes for every task."""
    ens = E.reference_ensembles(PAPER)
    t = time.time()
    lis = load_lis() if LIS_CSV.exists() else None
    tasks = {}
    spec = [("E1", "E1", "PL", "PAPER"), ("E2_s2", "E2", "SBPL_s2", "PAPER"),
            ("E2_sfree", "E2", "SBPL_sfree", "PAPER"), ("E3", "E3", "MIX", "PAPER"),
            ("E4-JP", "E4-JP", "JP", "PAPER"), ("E1M_U46basis", "E1M", "PL", "MAXWELL"),
            ("E4-KP_JPfamily", "E4-KP", "JP", "PAPER"), ("E2a_unverified_s2", "E2a", "SBPL_s2", "PAPER"),
            ("E2a_unverified_sfree", "E2a", "SBPL_sfree", "PAPER")]
    if lis is None:
        spec = [s for s in spec if s[1] != "E2a"]
        print(f"E2a skipped: {relative(LIS_CSV)} is absent", flush=True)
    cache = {}
    for name, ek, key, field in spec:
        fam = P.Family(key, nu_conv=NU_CONV)
        cfg_field = PAPER_MAXWELL if field == "MAXWELL" else PAPER
        if ek not in cache:
            if ek == "E4-KP":
                kp = ens[ek]
                S = kp.direct(PAPER, n=24, n_eta=16)
                a = kp.effective_moments(PAPER, 2)
                abs3 = kp.effective_moments(PAPER, 2, absolute_m=3)
                chk = kp.direct_check(PAPER)
            elif ek == "E2a":
                S, a, abs3 = lis.direct(PAPER), lis.moments(PAPER, 2), lis.abs_moments(PAPER, 3)
                chk = lis.direct_check(PAPER)
            else:
                e = ens[ek]
                S = e.direct(cfg_field)
                a, abs3 = e.moments(PAPER, 2), e.abs_moments(PAPER, 3)
                chk = e.direct_check(cfg_field)
            cache[ek] = (S, a, abs3, chk)
        S, a, abs3, chk = cache[ek]
        phi_true = fam.from_lib(ens[ek].theta) if ek in ("E1", "E2", "E3", "E4-JP", "E1M") else None
        tasks[name] = {"ens": ek, "key": key, "family": fam, "cfg_field": cfg_field, "S": S, "a_true": a,
                       "abs3": abs3, "phi_true": phi_true, "direct_check": chk}
    tic("prepare_tasks", t)
    return tasks, ens, lis


def analyse_seeded(name, task, C, fine):
    """Seeded fit, multi-start, noise-free bias and envelope, exact-forward reference, R scan."""
    fam, S, cfg_field = task["family"], task["S"], task["cfg_field"]
    prob = P.Whitened(PAPER, C, S)
    fit_i, held_i = PAPER.fit_idx, PAPER.held_idx
    z_fit = PAPER.draw_noise(len(fit_i))
    z_held = PAPER.draw_noise(len(held_i), seed=SEED + 1)  # same withheld draw as fits.py
    d_fit = S[fit_i] + z_fit * prob.sig[fit_i]
    dw, dhw = prob.whiten(d_fit), prob.whiten_held(S[held_i] + z_held * prob.sig[held_i])
    dw0, dhw0 = prob.whiten(S[fit_i]), prob.whiten_held(S[held_i])
    R, chi2R, crit = prob.upre(dw)
    phi_true = task["phi_true"]
    if phi_true is not None:
        center = phi_true
    elif task["key"] == "JP":  # KP data: JP truth of the same injection and age
        center = fam.from_lib(E.reference_ensembles()["E4-JP"].theta)
    else:  # E2a: start from the E2 truth
        center = fam.from_lib(E.SmoothBroken().theta)
    true_inv = fam.invariants(cfg_field, phi_true) if phi_true is not None else None
    out = {"family": fam.key, "coords": fam.names, "invariant_names": fam.invariant_names(),
           "phi_true": phi_true, "invariants_true": true_inv, "R_star_seeded": R,
           "upre_scan_R_max_(numerical_rank)": prob.numerical_rank,
           "singular_values": prob.s, "upre_criterion_R1_20": crit[:20], "chi2_fit_vs_R_1_20": chi2R[:20]}
    # multi-start on the seeded data
    ms = {}
    for mode, RR in (("comp", R), ("full", None)):
        res = P.multistart(prob, fam, PAPER, dw, mode, RR, starts(fam.key, center))
        best = res[0]
        spread = np.max([np.max(np.abs(r["phi"] - best["phi"]) / best["sd"]) for r in res
                         if r["chi2"] < best["chi2"] + 1e-6])
        ms[mode] = {"n_starts": len(res), "n_at_best_chi2(+1e-6)": int(sum(r["chi2"] < best["chi2"] + 1e-6 for r in res)),
                    "max_phi_spread_at_best_over_sd": float(spread),
                    "chi2_other_minima": sorted({round(r["chi2"], 4) for r in res if r["chi2"] >= best["chi2"] + 1e-6})}
        out[f"seeded_{mode}"] = fit_summary(fam, cfg_field, prob, best, dhw, true_inv, phi_true)
    out["multistart"] = ms
    # TSVD reference on held channels
    a_tsvd = prob.tsvd_moments(dw, R)
    out["seeded_tsvd_chi2_held"] = float(np.sum((dhw - prob.Wh @ a_tsvd) ** 2))
    if phi_true is not None:
        ref = phi_true
    elif task["ens"] == "E4-KP":
        ex = kp_exact_jp_fit(task, prob)
        out["exact_JP_forward_fit_noisefree"] = ex
        ref = np.array([ex["lnA_family"], ex["p"], NU_CONV + 2.0 * ex["ln_gamma_c"]])
    else:
        f = exact_forward_fit(lambda phi: fine.spectrum(fam.key, phi), prob, S[fit_i], out["seeded_full"]["phi"])
        ref = f["phi"]
    out["reference_point"] = {"phi": ref, "kind": "truth" if phi_true is not None else
                              "pseudo-truth: exact-forward family fit to the noise-free direct spectrum"}
    # Fisher sd ratio at the reference point
    sd_c, _ = P.fisher_sd(prob, fam, PAPER, ref, "comp", R)
    sd_f, _ = P.fisher_sd(prob, fam, PAPER, ref, "full")
    out["fisher_sd_full_at_ref"] = sd_f
    out["fisher_sd_comp_over_full_at_R_seed"] = sd_c / sd_f
    # noise-free fits: representation (or misspecification) bias
    nf = {}
    for mode, RR in (("comp", R), ("full", None)):
        f = P.fit(prob, fam, PAPER, dw0, mode, RR, ref)
        nf[mode] = fit_summary(fam, cfg_field, prob, f, dhw0, true_inv, phi_true)
        nf[mode]["shift_over_fisher_sd"] = (f["phi"] - ref) / (sd_c if mode == "comp" else sd_f)
    out["noisefree"] = nf
    if phi_true is not None:
        eps = S - B.predict(C, task["a_true"])
        eps_w = eps[fit_i].ravel() / prob.sf
        lin, envb = {}, {}
        env = envelope_N2(PAPER, task["abs3"], PAPER.azimuth_factor)
        Ew = env[fit_i].ravel() / prob.sf
        for mode, RR, sd in (("comp", R, sd_c), ("full", None, sd_f)):
            G = P.bias_response(prob, fam, PAPER, phi_true, mode, RR)
            lin[mode] = G @ eps_w / sd
            envb[mode] = np.abs(G) @ Ew / sd
        out["representation"] = {
            "max_eps_over_I_IQU": np.max(np.abs(eps) / S[:, :1], axis=0),
            "envelope_max_over_I_IQU": np.max(env / S[:, :1], axis=0),
            "envelope_bounds_eps_all_cells": bool(np.all(np.abs(eps) <= env)),
            "norm_bound_|eps_w|_2_(any_param_shift<=this_many_sd)": float(np.linalg.norm(eps_w)),
            "norm_bound_|E_w|_2_(envelope_version)": float(np.linalg.norm(Ew)),
            "linear_bias_over_sd": lin, "envelope_bias_over_sd": envb}
    # exact-forward reference (no moment truncation) for smooth families
    if fine is not None and fam.key in ("PL", "SBPL_s2", "SBPL_sfree", "MIX"):
        spec_fn = lambda phi: fine.spectrum(fam.key, phi)  # noqa: E731
        ex = {"nodes": int(fine.g.size)}
        if phi_true is not None and task["ens"] != "E1M":
            ex["max|S_fine(phi_true) - S_direct|/I_IQU"] = np.max(np.abs(spec_fn(phi_true) - S) / S[:, :1], axis=0)
        for dname, dd in (("noisefree", S[fit_i]), ("seeded", d_fit)):
            f = exact_forward_fit(spec_fn, prob, dd, ref)
            ex[dname] = {"phi": f["phi"], "sd": f["sd"], "chi2": f["chi2"]}
            if phi_true is not None:
                ex[dname]["shift_over_sd"] = (f["phi"] - phi_true) / f["sd"]
        out["exact_forward"] = ex
    # R scan (Picard-type) on the seeded data
    scan = {}
    for RR in range(max(3, fam.k), 15):
        f = P.fit(prob, fam, PAPER, dw, "comp", RR, out["seeded_full"]["phi"])
        sdc, _ = P.fisher_sd(prob, fam, PAPER, ref, "comp", RR)
        scan[f"R{RR}"] = {"phi": f["phi"], "sd": f["sd"], "chi2_beta": f["chi2"], "fisher_sd_ratio_comp_full": sdc / sd_f}
    out["R_scan_seeded_comp"] = scan
    return out, prob, R


def run_realisations(pool, name, task, C, R_seed, phi_start, phi_retry):
    fam = task["family"]
    payload = {"cfg_basis": PAPER, "cfg_field": task["cfg_field"], "family": fam, "C": C, "S": task["S"],
               "seed": SEED, "R_seed": R_seed, "phi_start": phi_start, "phi_retry": phi_retry}
    chunks = np.array_split(np.arange(N_REAL), WORKERS)
    futs = [pool.submit(P.realisation_chunk, payload, list(c)) for c in chunks]
    return np.vstack([f.result() for f in futs])


def summarise_realisations(task, arr, phi_ref, inv_ref, noisefree):
    fam = task["family"]
    k, m = fam.k, len(fam.invariant_names())
    width = 2 * k + 2 + 2 * m + 1
    out = {"N_real": int(arr.shape[0]), "noise": "default_rng([20260928, 1, k]).normal(size=(41, 3)), k = 0..N-1",
           "R_star_counts": {int(v): int(c) for v, c in zip(*np.unique(arr[:, 0], return_counts=True))}}
    labels = ("comp_Rstar", "comp_Rseed", "full")
    blocks = {}
    for i, lab in enumerate(labels):
        blk = arr[:, 1 + i * width: 1 + (i + 1) * width]
        phi, sd = blk[:, :k], blk[:, k:2 * k]
        chi2, held = blk[:, 2 * k], blk[:, 2 * k + 1]
        inv, inv_sd = blk[:, 2 * k + 2:2 * k + 2 + m], blk[:, 2 * k + 2 + m:2 * k + 2 + 2 * m]
        ok = blk[:, -1] == 1.0
        blocks[lab] = (phi, sd)
        s = {"n_failed": int((~ok).sum()), "chi2_mean": float(np.nanmean(chi2)), "chi2_held_mean": float(np.nanmean(held)),
             "params_vs_reference": P.summarise(phi, phi_ref, sd)}
        nfm = noisefree["comp" if lab.startswith("comp") else "full"]["phi"]
        s["params_vs_noisefree_fit"] = P.summarise(phi, nfm, sd)
        if inv_ref is not None:
            s["invariants_vs_truth"] = P.summarise(inv, inv_ref, inv_sd)
        out[lab] = s
    out["sd_comp_Rstar_over_full_median"] = np.nanmedian(blocks["comp_Rstar"][1] / blocks["full"][1], axis=0)
    out["sd_comp_Rseed_over_full_median"] = np.nanmedian(blocks["comp_Rseed"][1] / blocks["full"][1], axis=0)
    out["corr(phi_comp_Rstar, phi_full)"] = [float(np.corrcoef(blocks["comp_Rstar"][0][:, j], blocks["full"][0][:, j])[0, 1])
                                            if np.all(np.isfinite(blocks["comp_Rstar"][0][:, j])) else None for j in range(k)]
    return out


# ----------------------------------------------------------------------------- KP vs JP
def kp_predictions(ens):
    """t_eff/t = b_KP(eta)/b_JP and the implied cut-offs, over eta ~ U[-1/2, 1/2]."""
    ag = E.AGEING
    e, w = np.polynomial.legendre.leggauss(64)
    eta = 0.25 * (e + 1.0)
    w = 0.25 * w / 0.5  # probability weights on [0, 1/2]
    ratio = ag.b_kp(eta) / (ag.b_s + ag.b_c)
    gc = ag.gamma_c_kp(eta)
    sperp = np.sqrt(1.0 - eta**2)
    p = 2.6
    wem = w * sperp ** ((p + 1) / 2)
    wem /= wem.sum()
    bw, bwt = PAPER.b_nodes_weights
    nu_c = K.a1 * np.sum(bwt * bw) * sperp * gc**2  # a1 <B> B_perp/B gamma_c(eta)^2
    return {
        "b_JP_s^-1": ag.b_s + ag.b_c, "t_Myr": ag.t_myr, "gamma_c_JP": ag.gamma_c_jp,
        "nu_b_JP=a1<B_perp>gamma_c_JP^2_Hz": float(np.exp(NU_CONV) * ag.gamma_c_jp**2),
        "t_eff_over_t_range_eta_in_[0,1/2]": [float(ratio.min()), float(ratio.max())],
        "t_eff_over_t_mean_uniform_eta": float(np.sum(w * ratio)),
        "t_eff_over_t_mean_emission_weighted_(1-eta^2)^((p+1)/4)": float(np.sum(wem * ratio)),
        "t_eff_over_t_from_emission_weighted_<ln gamma_c>": float(ag.gamma_c_jp / np.exp(np.sum(wem * np.log(gc)))),
        "nu_c(eta)=a1<B>B_perp/B gamma_c(eta)^2_range_Hz": [float(nu_c.min()), float(nu_c.max())],
        "nu_b_pred_emission_weighted_<ln gamma_c>_Hz": float(np.exp(NU_CONV + 2 * np.sum(wem * np.log(gc)))),
        "definition": "t_eff = 1/(b_JP gamma_c,fit); nu_b = a1 <B_perp> gamma_c^2 with <B_perp> of U[4,6], eta ~ U[-1/2,1/2]",
    }


def kp_exact_jp_fit(task, prob):
    """Fit (p, ln gamma_c) of the exact JP direct spectrum (amplitude profiled) to noise-free KP data."""
    S = task["S"]
    fit = PAPER.fit_idx
    dw = prob.whiten(S[fit])
    calls = {"n": 0}

    def model(x):
        calls["n"] += 1
        s = E.JaffePerola(p=float(x[0]), gamma_c=float(np.exp(x[1]))).direct(PAPER)
        mw = s[fit].ravel() / prob.sf
        A = float(mw @ dw / (mw @ mw))
        return A, s, mw

    def res(x):
        A, _, mw = model(x)
        return dw - A * mw

    t = time.time()
    x0 = np.array([2.6, np.log(E.AGEING.gamma_c_jp)])
    sol = least_squares(res, x0, method="lm", diff_step=1e-4, xtol=1e-10, ftol=1e-10)
    A, s, _ = model(sol.x)
    A_family = A * E.JaffePerola(p=float(sol.x[0]), gamma_c=float(np.exp(sol.x[1]))).norm
    J = sol.jac
    cov = np.linalg.inv(J.T @ J)
    return {"p": float(sol.x[0]), "ln_gamma_c": float(sol.x[1]), "gamma_c": float(np.exp(sol.x[1])),
            "lnA_family": float(np.log(A_family)),
            "sd_(p, ln gamma_c)_amplitude_profiled": np.sqrt(np.diag(cov)), "chi2_noisefree": float(2 * sol.cost),
            "t_eff_over_t": float(E.AGEING.gamma_c_jp / np.exp(sol.x[1])),
            "nu_b_Hz": float(np.exp(NU_CONV + 2 * sol.x[1])),
            "max|S_JPfit - S_KP|/I_IQU_all41": np.max(np.abs(A * s - S) / S[:, :1], axis=0),
            "direct_calls": calls["n"], "seconds": time.time() - t}


# ----------------------------------------------------------------------------- degeneracy driver
def degeneracy(pool, tasks, ens):
    out = {}
    S1 = tasks["E1"]["S"]
    t = time.time()
    lams = (0.5, 0.8, 1.25, 2.0)
    futs = [pool.submit(D.kernel_identity_one, lam, S1[:, 0]) for lam in lams]
    out["kernel_scaling_identity"] = [f.result() for f in futs]
    tic("kernel_identity", t)
    t = time.time()
    deltas = (0.01, 0.02)
    lnl = sorted({round(s * d, 12) for d in deltas for s in (-2, -1, 1, 2)} | {0.0})
    futs = [pool.submit(D.derivative_spectra_at, x, 4) for x in lnl]
    spectra = dict(f.result() for f in futs)
    out["derivative_identity"] = D.derivative_identity(spectra, deltas, S1[:, 0])
    nv = {}
    for nm in ("E1", "E2_s2"):
        e = ens[tasks[nm]["ens"]]
        nv[nm] = D.null_vector_residual(tasks[nm]["S"], {2: e.moments(PAPER, 2), 3: e.moments(PAPER, 3)},
                                        tasks[nm]["S"][:, 0], spectra[0.0])
    out["jacobian_null_vector"] = nv
    tic("derivative_identity", t)
    # E2 ridge
    t = time.time()
    task = tasks["E2_s2"]
    S = task["S"]
    z_fit = PAPER.draw_noise(len(PAPER.fit_idx))
    R_seed = task["_R_seed"]
    grid = (-0.4, -0.3, -0.2, -0.1, -0.05, 0.0, 0.05, 0.1, 0.2, 0.3, 0.4)
    dl = tuple(np.round(np.linspace(-0.5, 0.5, 21), 10))
    futs = [pool.submit(D.ridge_one, x, S, z_fit, task["phi_true"], R_seed, dl) for x in grid]
    rows = [f.result() for f in futs]
    out["E2_ridge"] = {"ln_lambda_grid": grid, "dln_gamma_b_grid_(absolute_offset_from_truth)": dl,
                       "R_seed": R_seed, "family": "SBPL_s2 (lnA, p1, p2, ln gamma_b), s = 2",
                       "rows": rows, "summary": ridge_summary(rows, grid)}
    tic("ridge", t)
    return out


def ridge_summary(rows, grid):
    i0 = grid.index(0.0)
    out = {}
    for sup in ("transformed", "fixed"):
        for dname in ("noisefree", "seeded"):
            for k in ("full", "comp_Rseed"):
                c0 = rows[i0][sup][dname][k]["chi2"]
                out[f"{sup}/{dname}/{k}/delta_chi2_vs_lnlam"] = [r[sup][dname][k]["chi2"] - c0 for r in rows]
                out[f"{sup}/{dname}/{k}/ln_gamma_b_fit"] = [r[sup][dname][k]["phi"][3] for r in rows]
                phi0 = np.asarray(rows[i0][sup][dname][k]["phi"])
                dev = [np.asarray(r[sup][dname][k]["phi"]) - phi0
                       - np.array([-0.5, 0.0, 0.0, -0.5]) * g for r, g in zip(rows, grid)]
                out[f"{sup}/{dname}/{k}/phi(lam)_minus_transform_of_phi(1)_per_lnlam"] = dev
                out[f"{sup}/{dname}/{k}/max|phi(lam) - transform of phi(1)|_over_sd(1)"] = float(np.max(
                    [np.max(np.abs(v) / np.asarray(rows[i0][sup][dname][k]["sd"])) for v in dev]))
                out[f"{sup}/{dname}/{k}/invariants_minus_lam1"] = [
                    (np.asarray(r[sup][dname][k]["invariants"]) - np.asarray(rows[i0][sup][dname][k]["invariants"])).tolist()
                    for r in rows]
    for sup in ("transformed", "fixed"):
        lg = np.array(out[f"{sup}/noisefree/full/ln_gamma_b_fit"])
        out[f"{sup}/noisefree/full/slope_dln_gamma_b_dln_B"] = float(np.polyfit(grid, lg, 1)[0])
    return out


# ----------------------------------------------------------------------------- main
def main():
    t = time.time()
    tasks, ens, lis = prepare_tasks()
    t = time.time()
    C = B.build_basis(PAPER, 2)
    fine = FineTable(PAPER)
    tic("basis_and_fine_table", t)
    results = {"description": __doc__, "config": PAPER.describe(), "N_real": N_REAL, "seed": SEED,
               "nu_conv_ln(a1<B_perp>)": NU_CONV, "tasks": {}}
    real_store = {}
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=WORKERS, mp_context=ctx) as pool:
        for name, task in tasks.items():
            t = time.time()
            seeded, prob, R = analyse_seeded(name, task, C, fine)
            task["_R_seed"] = R
            tic(f"seeded_{name}", t)
            if name == "E4-KP_JPfamily":
                t = time.time()
                seeded["predictions_b_KP/b_JP"] = kp_predictions(ens)
                teff = lambda lnnu: E.AGEING.gamma_c_jp / np.exp(0.5 * (np.asarray(lnnu) - NU_CONV))  # noqa: E731
                seeded["t_eff_over_t_seeded"] = {m: float(teff(seeded[f"seeded_{m}"]["phi"][2])) for m in ("comp", "full")}
                seeded["t_eff_over_t_seeded_sd"] = {m: float(0.5 * seeded[f"seeded_{m}"]["sd"][2] * teff(seeded[f"seeded_{m}"]["phi"][2]))
                                                    for m in ("comp", "full")}
                for mode in ("comp", "full"):
                    phi = seeded["noisefree"][mode]["phi"]
                    seeded["noisefree"][mode]["t_eff_over_t"] = float(E.AGEING.gamma_c_jp / np.exp(0.5 * (phi[2] - NU_CONV)))
                    seeded["noisefree"][mode]["nu_b_Hz"] = float(np.exp(phi[2]))
                tic("kp_extras", t)
            t = time.time()
            ref = np.asarray(seeded["reference_point"]["phi"])
            arr = run_realisations(pool, name, task, C, R, ref, seeded["noisefree"]["full"]["phi"])
            real_store[name] = arr
            inv_ref = seeded["invariants_true"] if task["phi_true"] is not None else None
            seeded["realisations"] = summarise_realisations(task, arr, ref, inv_ref, seeded["noisefree"])
            seeded["realisations"]["reference_point"] = seeded["reference_point"]["kind"]
            if name == "E4-KP_JPfamily":
                k = task["family"].k
                lnnu = arr[:, 1 + 2]  # comp_Rstar block, coordinate ln nu_b
                te = E.AGEING.gamma_c_jp / np.exp(0.5 * (lnnu - NU_CONV))
                seeded["realisations"]["t_eff_over_t_comp_Rstar_mean_sd"] = [float(np.nanmean(te)), float(np.nanstd(te, ddof=1))]
                del k
            if name == "E1M_U46basis":
                q = 0.5 * (2.6 + 1.0)
                seeded["field_moment_ratio_<B_perp^1.8>_Maxwell_over_U46"] = float(
                    P._bperp_moment(PAPER_MAXWELL, q) / P._bperp_moment(PAPER, q))
            tic(f"realisations_{name}", t)
            if task["ens"] == "E2a":
                g = np.geomspace(300.0, 1e5, 400)
                lr = np.log(family_density(task["key"], seeded["reference_point"]["phi"], g) / lis.density(g))
                band = (g >= 1e3) & (g <= 5e4)
                seeded["pseudo_truth_vs_LIS_electron_space"] = {
                    "max|ln N_family/N_LIS|_300-1e5": float(np.max(np.abs(lr))),
                    "max|ln N_family/N_LIS|_1e3-5e4": float(np.max(np.abs(lr[band]))),
                    "LIS_local_index_at_gamma_[1e3,3e3,1e4,3e4]": (-np.gradient(np.log(lis.density(g)), np.log(g)))[
                        [int(np.argmin(np.abs(g - v))) for v in (1e3, 3e3, 1e4, 3e4)]]}
            seeded["direct_check"] = task["direct_check"]
            seeded["S_direct_IQU"] = task["S"]
            results["tasks"][name] = seeded
            print(name, "R*", R, "comp", np.round(seeded["seeded_comp"]["phi"], 4), flush=True)
        results["degeneracy"] = degeneracy(pool, tasks, ens)
    if lis is None:
        results["E2a_input"] = {"file": relative(LIS_CSV), "present": False,
                                "status": "skipped: input file absent (not distributed); E2a tasks not run"}
    else:
        results["E2a_input"] = {"file": relative(LIS_CSV), "present": True, "sha256": sha256(LIS_CSV),
                                "verified_against_pdf": False,
                                "status": "exploratory; transcribed from the ar5iv HTML version of Orlando (2018), "
                                          "Table 5, not checked against the journal PDF; not used in the paper"}
    fs = np.load(HERE / "forward_spectra.npz")
    results["consistency_with_forward_spectra_npz"] = {
        k: float(np.max(np.abs(tasks[n]["S"] - fs[k]) / fs[k][:, :1])) for n, k in
        (("E1", "E1"), ("E2_s2", "E2"), ("E3", "E3"), ("E4-JP", "E4_JP"), ("E4-KP_JPfamily", "E4_KP"), ("E1M_U46basis", "E1M"))}
    np.savez(HERE / "params_realisations.npz", **{k.replace("-", "_"): v for k, v in real_store.items()})
    results["timings_s"] = TIMINGS
    results["provenance"] = provenance(__file__, T0, inputs=(HERE / "paramlib.py", HERE / "params_degeneracy.py",
                                                             *([LIS_CSV] if lis is not None else []),
                                                             HERE / "forward_spectra.npz"))
    write_json(HERE / "params_results.json", results)
    print("done", time.time() - T0)


if __name__ == "__main__":
    main()
