"""Baselines at equal parameter count, intensity only, continuum kernel.

Part A (synthetic): E1, E2, E3, E4-JP, E4-KP, E5 (and E1M as a field-misspecified extra) direct
intensities in the 41 paper channels, 1% noise.  Models fitted to the 21 fitted channels and
validated on the 20 withheld channels:
  * spectral laws (spectral_laws.py): PySM s1 (k = 2), s3 (3), beta-moment with omega_1 = 0
    (bmom3, 3), log-cubic (4), beta-moment with omega_1 free (bmom4, 4), each integrated
    through the same channel responses;
  * the moment basis (N = 2, 36 columns), noise-whitened TSVD on intensity rows, at the UPRE
    rank R* searched over R <= 14 (the Picard range; the uncapped R* over R = 1..rank is
    kept as moment_UPRE_uncapped) and at R = 2, 3, 4 (equal count);
  * electron-spectrum families through the same basis (ePL 2, eLP 3, eSBPL 4 parameters).
  Polarisation: moment-basis models predict P through the separable P columns (paper azimuth
  factor 0.4 e^{0.5i}, screen inside the channel integral).  The laws predict P only with the
  extra assumption of spectral_laws.polarisation_with_assumption.
  Noise: seeded realisation (fitted rows: I column of cfg.draw_noise(21); withheld rows: I column
  of default_rng(seed + 1).normal(size=(20, 3))) and NREAL common-random-number realisations.

Part B (GALPROP): published |b| >= 60 deg intensity (inputs/galprop-benchmark/regional_spectra.csv),
native nodes 0.114-9.1 GHz (original indices 26..50), even indices fitted (13), odd withheld (12),
monochromatic evaluation with the paper's B ~ U[4,6] muG and eta ~ U[-1/2,1/2], moment basis on
[300, 1e5] with 12 bins, N = 2, versus the same laws and families.  Weights and UPRE use an
assumed 1% noise sd; reported both noise-free (representation) and over 1% noise realisations.

Writes baselines_results.json and baselines_plotdata.npz.  Run: python baselines.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import multiprocessing as mp  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import chi2 as chi2_dist  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))
import basis as B  # noqa: E402
import baseline_models as BM  # noqa: E402
import ensembles as E  # noqa: E402
import galprop_absorption as GA  # noqa: E402
import kernels as K  # noqa: E402
import spectral_laws as SL  # noqa: E402
from config import PAPER, PAPER_MAXWELL  # noqa: E402
from provenance import provenance, relative, sha256, write_json  # noqa: E402

T0 = time.time()
NREAL = 500
MC_SEED = 20261027
CHUNK = 50
N_WORKERS = 16
GALPROP_DIR = HERE / "inputs" / "galprop-benchmark"
GALPROP_CSV = GALPROP_DIR / "regional_spectra.csv"
GALPROP_CURV = GALPROP_DIR / "curvature_results.json"
GALPROP_MOMFIT = GALPROP_DIR / "moment_fit_results.json"
FORWARD_NPZ = HERE / "forward_spectra.npz"
SYNTH = ("E1", "E2", "E3", "E4-JP", "E4-KP", "E5", "E1M")
METRICS = ("chi2_fit", "chi2_held", "max_res_all", "max_res_held", "rms_res_held", "maxP_over_I_all",
           "maxP_over_I_all_with_assumption")


class MonoConfig:
    """Paper geometry with each channel replaced by one frequency (unit weight, no screen)."""

    def __init__(self, cfg, nu):
        self._cfg = cfg
        self.centres = np.asarray(nu, float)
        self.chan_nu = self.centres[:, None]
        self.chan_w = np.array([1.0])
        self.screen = np.ones_like(self.chan_nu, dtype=complex)
        self.n_channels = len(self.centres)

    def __getattr__(self, name):
        return getattr(self._cfg, name)


# ----------------------------------------------------------------------------- worker
_CTX = {}


def _init_worker(ctxs):
    _CTX.update(ctxs)


def _run_chunk(task):
    key, z = task
    ctx = _CTX[key]
    out = []
    for zz in z:
        d = ctx["I_true"] + zz * ctx["sig"]
        out.append(BM.fit_all(ctx, d[ctx["fit_idx"]], d[ctx["held_idx"]]))
    return key, out


# ----------------------------------------------------------------------------- helpers
def summarise(runs: list, crit: float) -> dict:
    """Median, mean and 16/84/95/99 percentiles of each metric for every model over realisations."""
    out = {}
    for model in runs[0]:
        rec = {"n_params": runs[0][model]["n_params"]}
        for m in METRICS:
            if m in runs[0][model]:
                v = np.array([r[model][m] for r in runs])
                rec[m] = {"median": float(np.median(v)), "mean": float(np.mean(v)),
                          **{f"p{q}": float(np.percentile(v, q)) for q in (16, 84, 95, 99)},
                          "max": float(np.max(v))}
        v = np.array([r[model]["chi2_held"] for r in runs])
        rec["frac_chi2_held_above_95pct"] = float(np.mean(v > crit))
        if model in ("moment_UPRE", "moment_UPRE_uncapped"):
            rs = np.array([r[model]["R_star"] for r in runs])
            rec["R_star"] = {"median": float(np.median(rs)), "p16": float(np.percentile(rs, 16)),
                             "p84": float(np.percentile(rs, 84)),
                             "histogram": {int(k): int(np.sum(rs == k)) for k in np.unique(rs)},
                             "frac_ge_15": float(np.mean(rs >= 15))}
        if model == "moment_UPRE":
            ru = np.array([r[model]["R_star_uncapped"] for r in runs])
            rec["R_star_uncapped_frac_ge_15"] = float(np.mean(ru >= 15))
            rec["frac_cap_binding"] = float(np.mean(ru != np.array([r[model]["R_star"] for r in runs])))
            rec["r_cap"] = runs[0][model]["r_cap"]
        if "params" in runs[0][model]:
            P = np.array([r[model]["params"] for r in runs])
            rec["params_median"] = np.median(P, axis=0).tolist()
            if model in ("bmom3", "bmom4"):
                w2 = P[:, -1] / P[:, 1]
                rec["omega_2"] = {"median": float(np.median(w2)), "frac_negative": float(np.mean(w2 < 0))}
            rec["frac_flagged"] = float(np.mean([r[model]["flag"] for r in runs]))
        out[model] = rec
    return out


def tsvd_curve(ctx: dict, d_fit, d_held) -> list:
    """Noise-free TSVD metrics for every rank."""
    tf = BM.tsvd_fit(ctx["svd"], d_fit, ctx["sig"][ctx["fit_idx"]])
    rows = []
    for R in range(1, len(ctx["svd"]["s"]) + 1):
        a = BM.tsvd_moments(ctx["svd"], tf, R)
        pP = ctx["CP"] @ a if ctx["CP"] is not None else None
        m = BM.metrics(ctx["CI"] @ a, ctx["I_true"], d_fit, d_held, ctx["sig"], ctx["fit_idx"],
                       ctx["held_idx"], pP, ctx["P_true"])
        Hw = (ctx["CI"][ctx["held_idx"]] / ctx["sig"][ctx["held_idx"], None]) @ (
            ctx["svd"]["vt"][:R].T / ctx["svd"]["s"][:R])
        rows.append({"R": R, **m, "upre_noisefree": float(tf["upre"][R - 1]),
                     "s_R_over_s_1": float(ctx["svd"]["s"][R - 1] / ctx["svd"]["s"][0]),
                     "tr_HHt_held": float(np.sum(Hw**2))})
    return rows


def predictive_trace(ctx: dict, model: str, params) -> float:
    """tr(H H^T) of the linearised fit: withheld-row noise propagated from the fitted rows,
    H = J_held (J_fit^T J_fit)^-1 J_fit^T (whitened Jacobians at the noise-free fit).
    Expected raw chi2_held = n_held + tr(H H^T) + (noise-free chi2_held)."""
    params = np.asarray(params, float)
    fi, hi, sig = ctx["fit_idx"], ctx["held_idx"], ctx["sig"]
    if model in SL.LAWS:
        k = SL.N_NONLIN[model]

        def pred(p):
            return SL.predict(model, p, ctx["grid"])
        lin = list(range(k, len(params)))
    else:
        k = 1

        def pred(p):
            return BM.family_prediction(model, p, ctx["CI"])
        lin = [0]
    J = np.zeros((len(sig), len(params)))
    for i in range(len(params)):
        if i in lin:  # prediction is linear in these coefficients
            e = np.zeros_like(params)
            e[i] = 1.0
            J[:, i] = pred(params * (np.arange(len(params)) != i) + e) - pred(params * (np.arange(len(params)) != i))
        else:
            h = 1e-6 * max(abs(params[i]), 1e-3)
            pp, pm = params.copy(), params.copy()
            pp[i] += h
            pm[i] -= h
            J[:, i] = (pred(pp) - pred(pm)) / (2 * h)
    J = J / sig[:, None]
    Jf, Jh = J[fi], J[hi]
    col = np.linalg.norm(Jf, axis=0)
    Jf, Jh = Jf / col, Jh / col
    G = np.linalg.pinv(Jf.T @ Jf)
    return float(np.trace(Jh @ G @ Jh.T))


def moment_bands(ctx: dict, R: int) -> dict:
    """1-sigma noise bands of the moment-basis TSVD prediction at rank R (whitened Cov)."""
    cov = BM.tsvd_cov(ctx["svd"], R)
    out = {"I_sd": np.sqrt(np.einsum("jk,kl,jl->j", ctx["CI"], cov, ctx["CI"]))}
    if ctx["CP"] is not None:
        cq, cu = ctx["CP"].real, ctx["CP"].imag
        out["Q_sd"] = np.sqrt(np.einsum("jk,kl,jl->j", cq, cov, cq))
        out["U_sd"] = np.sqrt(np.einsum("jk,kl,jl->j", cu, cov, cu))
        out["QU_cov"] = np.einsum("jk,kl,jl->j", cq, cov, cu)
    return out


def abs_p_sd(P, bands):
    """Linearised sd of |P| from the Q, U covariance."""
    n = np.stack([P.real, P.imag]) / np.maximum(np.abs(P), 1e-300)
    var = n[0] ** 2 * bands["Q_sd"] ** 2 + n[1] ** 2 * bands["U_sd"] ** 2 + 2 * n[0] * n[1] * bands["QU_cov"]
    return np.sqrt(np.clip(var, 0, None))


def clean(res: dict) -> dict:
    return {k: {kk: vv for kk, vv in v.items()} for k, v in res.items()}


# ----------------------------------------------------------------------------- part A setup
def synthetic_direct(cfg) -> tuple[dict, dict]:
    ens = E.reference_ensembles(cfg)
    spectra = {}
    for name in SYNTH:
        c = PAPER_MAXWELL if name == "E1M" else cfg
        spectra[name] = ens[name].direct(c)
    ref = np.load(FORWARD_NPZ)
    check = {name: float(np.max(np.abs(spectra[name] - ref[name.replace("-", "_")]) / spectra[name][:, :1]))
             for name in SYNTH}
    return spectra, check


def synthetic_contexts(cfg, spectra):
    C = B.build_basis(cfg, 2)
    CI, CP = C[:, 0, :], C[:, 1, :] + 1j * C[:, 2, :]
    grid = SL.channel_grid(cfg.chan_nu, cfg.chan_w)
    ctxs = {}
    for name, S in spectra.items():
        I = S[:, 0]
        sig = cfg.noise_frac * I
        ctxs[name] = dict(CI=CI, CP=CP, grid=grid, fit_idx=cfg.fit_idx, held_idx=cfg.held_idx, sig=sig,
                          I_true=I, P_true=S[:, 1] + 1j * S[:, 2], factor=cfg.azimuth_factor,
                          screen=cfg.screen, svd=BM.tsvd_setup(CI[cfg.fit_idx], sig[cfg.fit_idx]))
    return ctxs


def seeded_noise_synthetic(cfg):
    zf = cfg.draw_noise(len(cfg.fit_idx))[:, 0]
    zh = np.random.default_rng(cfg.seed + 1).normal(size=(len(cfg.held_idx), 3))[:, 0]
    z = np.zeros(cfg.n_channels)
    z[cfg.fit_idx], z[cfg.held_idx] = zf, zh
    return z


# ----------------------------------------------------------------------------- part B setup
def galprop_data():
    raw = np.loadtxt(GALPROP_CSV, delimiter=",", skiprows=1)
    k = np.arange(raw.shape[0])
    grid_err = float(np.max(np.abs(raw[:, 0] * 1e9 / (1e6 * 1.2**k) - 1)))
    sel = (raw[:, 0] >= 0.1) & (raw[:, 0] <= 10.0)
    idx = k[sel]
    return {"nu": raw[sel, 0] * 1e9, "I": raw[sel, 3], "orig_index": idx, "train": idx % 2 == 0,
            "grid_rel_err_vs_1e6x1.2^k": grid_err}


def galprop_checks(cfg, g, CI_mono, table) -> dict:
    """Reproduce the archived log-space fits; check the monochromatic basis three ways."""
    import json

    out = {}
    curv = json.loads(GALPROP_CURV.read_text())["bands"]["current_band"]["regions"]["high_latitude"]
    tr, he = g["train"], ~g["train"]
    lnf = np.log(g["nu"] / 1e9)
    for deg, key in ((1, "power_law"), (2, "log_parabola"), (3, "log_cubic")):
        c = np.polynomial.polynomial.polyfit(lnf[tr], np.log(g["I"][tr]), deg)
        p = np.exp(np.polynomial.polynomial.polyval(lnf, c))
        rms = float(np.sqrt(np.mean(((p[he] - g["I"][he]) / g["I"][he]) ** 2)))
        out[f"logspace_{key}_withheld_rms_minus_archived"] = rms - curv[key]["withheld_rms_fraction"]
    mf = json.loads(GALPROP_MOMFIT.read_text())["bands"]["current_band"]["regions"]["high_latitude"]
    out["archived_fractional_logcubic_withheld"] = mf["baselines"]["log_cubic"]["withheld"]
    # narrow channels through the library channel path
    cn = replace(cfg, nu_lo=g["nu"][0], nu_hi=g["nu"][-1], n_channels=len(g["nu"]), chan_half=1e-4)
    out["narrow_channel_centres_rel_err"] = float(np.max(np.abs(cn.centres / g["nu"] - 1)))
    CIn = B.build_basis(cn, 2, table=table)[:, 0, :]
    out["narrow_channel_minus_mono_max_over_col_max"] = float(np.max(np.abs(CIn - CI_mono)) / np.max(np.abs(CI_mono)))
    out["narrow_channel_minus_mono_max_rel_per_row"] = float(np.max(
        np.abs(CIn - CI_mono) / np.max(np.abs(CI_mono), axis=1, keepdims=True)))
    # table-free kernel average (direct F quadrature at every B, eta node)
    mono = MonoConfig(cfg, g["nu"])
    bn, bw = cfg.b_nodes_weights
    en, ew = cfg.eta_nodes_weights
    s = np.sqrt(1 - en**2)
    errs = []
    for gam in (1e3, 1e4, 5e4):
        kI, _ = B.kbar(mono, [gam], order=0, table=table)
        x = g["nu"][:, None, None] / (K.a1 * bn[None, :, None] * s[None, None, :] * gam**2)
        ref = np.sum(bw[None, :, None] * ew[None, None, :] * K.A1 * bn[None, :, None] * s[None, None, :] * K.F(x), axis=(1, 2))
        errs.append(float(np.max(np.abs(kI[0, 0] - ref)) / ref.max()))
    out["table_free_kbar_max_rel_err_gamma_1e3_1e4_5e4"] = errs
    # analytic vs 5-point stencil columns (monochromatic)
    DI, _ = B.derivative_spectra(mono, 2, table=table)
    SI, _ = B.stencil_derivative_spectra(mono, table=table)
    scale = np.max(np.abs(DI[:, 0, :]))
    out["analytic_minus_stencil_max_over_global_max"] = float(np.max(np.abs(DI - SI)) / scale)
    # representation of E1 (monochromatic) by C a_true
    e1 = E.PowerLaw()
    m = e1.measure(cfg)
    kI, _ = B.kbar(mono, m.gamma, order=0, table=table)
    S1 = m.w @ kI[0]
    out["E1_mono_representation_max_rel"] = float(np.max(np.abs(CI_mono @ e1.moments(cfg, 2).ravel() - S1) / S1))
    return out


def galprop_context(cfg, g, table):
    mono = MonoConfig(cfg, g["nu"])
    DI, DQ = B.derivative_spectra(mono, 2, table=table)
    CI = B.separable_columns(mono, DI, DQ)[:, 0, :]
    fi, hi = np.where(g["train"])[0], np.where(~g["train"])[0]
    sig = PAPER.noise_frac * g["I"]
    ctx = dict(CI=CI, CP=None, grid=SL.mono_grid(g["nu"]), fit_idx=fi, held_idx=hi, sig=sig, I_true=g["I"],
               P_true=None, factor=None, screen=None, svd=BM.tsvd_setup(CI[fi], sig[fi]))
    return ctx


def dense_predictions(cfg, g, table, res_curves: dict, nu_dense) -> dict:
    """Noise-free GALPROP fits evaluated on a dense frequency grid (model curves, not data)."""
    mono = MonoConfig(cfg, nu_dense)
    DI, DQ = B.derivative_spectra(mono, 2, table=table)
    CId = B.separable_columns(mono, DI, DQ)[:, 0, :]
    gd = SL.mono_grid(nu_dense)
    out = {}
    for model, c in res_curves.items():
        if model.startswith("moment"):
            out[model] = CId @ c["a"]
        elif model in SL.LAWS:
            out[model] = SL.predict(model, np.asarray(c["params"]), gd)
        else:
            out[model] = BM.family_prediction(model, np.asarray(c["params"]), CId)
    return out


def fit_verification(ctxs: dict) -> dict:
    """Noise-free fits: VarPro laws vs full-parameter multi-start LSQ; families vs random starts;
    TSVD chi2 bookkeeping vs direct residuals; channel weights sum."""
    from scipy.optimize import least_squares

    out = {"laws": {}, "families": {}, "tsvd_chi2_formula_max_abs_diff": 0.0}
    rng = np.random.default_rng(3)
    for key, ctx in ctxs.items():
        fi = ctx["fit_idx"]
        I, sig = ctx["I_true"], ctx["sig"]
        s0 = float(np.median(I[fi]))
        gf = ctx["grid"].rows(fi)
        for law in SL.LAWS:
            f = SL.fit(law, gf, I[fi] / s0, sig[fi] / s0)
            out["laws"][f"{key}/{law}"] = SL.multistart_check(law, gf, I[fi] / s0, sig[fi] / s0, f["chi2"])
        for fam in BM.FAMILIES:
            f = BM.fit_family(fam, ctx["CI"][fi], I[fi], sig[fi])
            lo, hi_b = (np.array(b) for b in BM.FAMILY_BOUNDS[fam])
            dw = I[fi] / sig[fi]

            def resid(phi):
                g = (ctx["CI"][fi] @ BM.family_moments(fam, phi).ravel()) / sig[fi]
                return (g @ dw) / (g @ g) * g - dw

            best = np.inf
            for _ in range(10):
                x0 = lo + (hi_b - lo) * rng.uniform(0.1, 0.9, size=len(lo))
                try:
                    r = least_squares(resid, x0, bounds=(lo, hi_b), method="trf", xtol=1e-12, ftol=1e-12,
                                      gtol=1e-12, max_nfev=3000)
                    best = min(best, 2 * r.cost)
                except ValueError:
                    continue
            out["families"][f"{key}/{fam}"] = {"production_chi2": f["chi2"], "best_random_start_chi2": best,
                                                "production_not_worse": bool(f["chi2"] <= best * (1 + 1e-6) + 1e-9)}
        tf = BM.tsvd_fit(ctx["svd"], I[fi], sig[fi])
        for R in (3, 7, 10):
            a = BM.tsvd_moments(ctx["svd"], tf, R)
            direct = float(np.sum(((ctx["CI"][fi] @ a - I[fi]) / sig[fi]) ** 2))
            out["tsvd_chi2_formula_max_abs_diff"] = max(out["tsvd_chi2_formula_max_abs_diff"],
                                                        abs(direct - tf["chi2_fit"][R - 1]))
    out["all_laws_varpro_not_worse"] = all(v["varpro_not_worse"] for v in out["laws"].values())
    out["all_families_not_worse"] = all(v["production_not_worse"] for v in out["families"].values())
    out["channel_weights_sum_minus_1"] = float(PAPER.chan_w.sum() - 1.0)
    return out


# ----------------------------------------------------------------------------- main
def main():
    cfg = PAPER
    table = B.get_table(cfg)
    timings = {}
    t = time.time()
    spectra, direct_check = synthetic_direct(cfg)
    timings["direct_spectra_s"] = time.time() - t
    ctxs = synthetic_contexts(cfg, spectra)
    g = galprop_data()
    ctxs["GALPROP"] = galprop_context(cfg, g, table)
    checks = {"direct_vs_forward_spectra_npz_max_over_I": direct_check, "galprop": galprop_checks(
        cfg, g, ctxs["GALPROP"]["CI"], table), "fit_verification": fit_verification(ctxs)}

    results, plot = {}, {"centres_Hz": cfg.centres, "fit_idx": cfg.fit_idx, "held_idx": cfg.held_idx}
    # noise-free and seeded fits
    for key, ctx in ctxs.items():
        I = ctx["I_true"]
        nf, nf_curves = BM.fit_all(ctx, I[ctx["fit_idx"]], I[ctx["held_idx"]], keep_curves=True)
        for model in nf:
            if "params" in nf[model]:
                tr = predictive_trace(ctx, model, nf[model]["params"])
                nf[model]["tr_HHt_held"] = tr
                nf[model]["expected_raw_chi2_held"] = len(ctx["held_idx"]) + tr + nf[model]["chi2_held"]
        if key == "GALPROP":
            z = np.zeros(len(I))
            z[ctx["fit_idx"]] = np.random.default_rng(cfg.seed).normal(size=len(ctx["fit_idx"]))
            z[ctx["held_idx"]] = np.random.default_rng(cfg.seed + 1).normal(size=len(ctx["held_idx"]))
        else:
            z = seeded_noise_synthetic(cfg)
        d = I + z * ctx["sig"]
        sd, sd_curves = BM.fit_all(ctx, d[ctx["fit_idx"]], d[ctx["held_idx"]], keep_curves=True)
        r_star = sd["moment_UPRE"]["R_star"]
        results[key] = {"noise_free_note": "fits to the noise-free spectrum with the assumed sd as weights; "
                                             "its moment_UPRE entry applies UPRE to noise-free data and is not a rank "
                                             "selection (use noise_free_tsvd_curve and the Monte Carlo R*)",
                        "noise_free": clean(nf), "noise_free_tsvd_curve": tsvd_curve(ctx, I[ctx["fit_idx"]], I[ctx["held_idx"]]),
                        "seeded": clean(sd), "seeded_noise_z": z}
        pre = f"{key}__"
        plot[pre + "I_true"] = I
        plot[pre + "sig"] = ctx["sig"]
        plot[pre + "seeded_data"] = d
        if ctx["P_true"] is not None:
            plot[pre + "P_true"] = ctx["P_true"]
        bands = moment_bands(ctx, r_star)
        plot[pre + "seeded__moment_UPRE__I_sd"] = bands["I_sd"]
        for model, c in sd_curves.items():
            for q, v in c.items():
                if v is not None and q != "a":
                    plot[f"{pre}seeded__{model}__{q}"] = v
        if ctx["CP"] is not None:
            Pm = sd_curves["moment_UPRE"]["P"]
            plot[pre + "seeded__moment_UPRE__absP_sd"] = abs_p_sd(Pm, bands)
        for model, c in nf_curves.items():
            plot[f"{pre}noisefree__{model}__I"] = c["I"]
            if c.get("P") is not None:
                plot[f"{pre}noisefree__{model}__P"] = c["P"]
            if c.get("P_with_assumption") is not None:
                plot[f"{pre}noisefree__{model}__P_with_assumption"] = c["P_with_assumption"]
        if key == "GALPROP":
            nu_dense = np.geomspace(g["nu"][0], g["nu"][-1], 241)
            # noise-free moment curve at the MC median UPRE rank is added after the MC below
            nf_par = {m: {"params": nf[m]["params"]} for m in nf if "params" in nf[m]}
            dense = dense_predictions(cfg, g, table, nf_par, nu_dense)
            plot["GALPROP__nu"] = g["nu"]
            plot["GALPROP__train"] = g["train"]
            plot["GALPROP__orig_index"] = g["orig_index"]
            plot["GALPROP__nu_dense"] = nu_dense
            for m, v in dense.items():
                plot[f"GALPROP__noisefree__{m}__I_dense"] = v

    # Monte Carlo
    t = time.time()
    z_syn = np.random.default_rng(MC_SEED).normal(size=(NREAL, cfg.n_channels))
    z_gal = np.random.default_rng(MC_SEED + 1).normal(size=(NREAL, len(g["nu"])))
    tasks = []
    for key in ctxs:
        z = z_gal if key == "GALPROP" else z_syn
        tasks += [(key, z[i:i + CHUNK]) for i in range(0, NREAL, CHUNK)]
    with mp.get_context("spawn").Pool(N_WORKERS, initializer=_init_worker, initargs=(ctxs,)) as pool:
        mc = {}
        for key, out in pool.imap(_run_chunk, tasks):
            mc.setdefault(key, []).extend(out)
    timings["monte_carlo_s"] = time.time() - t
    for key, runs in mc.items():
        n_held = len(ctxs[key]["held_idx"])
        crit = float(chi2_dist.ppf(0.95, n_held))
        results[key]["monte_carlo"] = {"n_real": len(runs), "chi2_held_95pct_point": crit,
                                       "n_held": n_held, "summary": summarise(runs, crit)}
        if key == "GALPROP":
            ctx = ctxs[key]
            rs = int(round(np.median([r["moment_UPRE"]["R_star"] for r in runs])))
            I = ctx["I_true"]
            tf = BM.tsvd_fit(ctx["svd"], I[ctx["fit_idx"]], ctx["sig"][ctx["fit_idx"]])
            a = BM.tsvd_moments(ctx["svd"], tf, rs)
            plot["GALPROP__noisefree__moment_at_MC_median_Rstar__I"] = ctx["CI"] @ a
            plot["GALPROP__MC_median_Rstar"] = rs
            nu_dense = plot["GALPROP__nu_dense"]
            dense = dense_predictions(cfg, g, table, {"moment": {"a": a}}, nu_dense)
            plot["GALPROP__noisefree__moment_at_MC_median_Rstar__I_dense"] = dense["moment"]
            results[key]["noise_free_moment_at_MC_median_Rstar"] = next(
                r for r in results[key]["noise_free_tsvd_curve"] if r["R"] == rs)

    t = time.time()
    raw = np.loadtxt(GALPROP_CSV, delimiter=",", skiprows=1)
    absorption = GA.analyse(raw, g["orig_index"], ctxs["GALPROP"])
    timings["absorption_s"] = time.time() - t
    out = {
        "description": __doc__,
        "noise": {"frac": cfg.noise_frac, "seed_fit_rows": cfg.seed, "seed_withheld_rows": cfg.seed + 1,
                  "mc_seed_synthetic": MC_SEED, "mc_seed_galprop": MC_SEED + 1, "n_real": NREAL,
                  "galprop_sigma": "assumed 0.01 * I_GALPROP (no noise in the published spectrum)"},
        "models": {
            "laws": {law: SL.n_params(law) for law in SL.LAWS},
            "families": {f: BM.family_n_params(f) for f in BM.FAMILIES},
            "moment": ("TSVD of whitened intensity rows, N = 2, 12 bins on [300, 1e5]; UPRE R* = argmin chi2_fit + 2R "
                       f"over R <= {BM.R_CAP} (moment_UPRE); moment_UPRE_uncapped searches R = 1..rank"),
            "law_pivot_GHz": 1.0,
            "law_P": "none; with_assumption = local power-law fraction (see spectral_laws.py)",
        },
        "results": results,
        "checks": checks,
        "absorption_caveat": absorption,
        "config": cfg.describe(),
        "timings_s": timings,
        "plotdata_npz": relative(HERE / "baselines_plotdata.npz"),
        "provenance": provenance(__file__, T0, inputs=(GALPROP_CSV, GALPROP_CURV, GALPROP_MOMFIT, FORWARD_NPZ,
                                                         HERE / "spectral_laws.py", HERE / "baseline_models.py",
                                                         HERE / "galprop_absorption.py")),
    }
    np.savez(HERE / "baselines_plotdata.npz", **plot)
    out["plotdata_sha256"] = sha256(HERE / "baselines_plotdata.npz")
    write_json(HERE / "baselines_results.json", out)
    print("done", time.time() - T0)


if __name__ == "__main__":
    main()
