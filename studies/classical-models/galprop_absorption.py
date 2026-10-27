"""Model-dependent absorption estimate for the published GALPROP |b| >= 60 deg spectrum.

Used by baselines.py (results under ``absorption_caveat`` in baselines_results.json).

The published spectrum (regional_spectra.csv, column high_latitude) extends to 1 MHz.  Below
a few MHz it turns over: it rises with frequency, peaks, and only then falls as a synchrotron
spectrum.  A turnover of this kind is the signature of free-free absorption mixed with the
emission.  The estimate below fits, to the nodes below 0.1 GHz,

    ln I(nu) = poly_deg(ln nu) + ln[(1 - e^-tau)/tau]   (mixed slab)   or   ... - tau   (screen),
    tau = tau_0 (nu / GHz)^-2.1,

and reads off tau at the lowest node used in the baselines (0.1145 GHz).  The polynomial
stands for the unabsorbed spectrum, so the result depends on its degree, on the frequency
range of the fit and on the geometry: it is a model-dependent estimate, not a measurement of
GALPROP's absorption settings (which cannot be audited from the FITS file).  The baselines are
then refitted to the spectrum divided by the slab attenuation to show what the estimate changes.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import baseline_models as BM
import spectral_laws as SL

FF_INDEX = 2.1
DEGREES = (2, 3, 4)
TOPS = (18, 22, 26)  # fit nodes 0..top-1 (top = 26 means nu <= 0.0954 GHz)
CENTRAL = (3, 26)
LN_TAU0_STARTS = np.linspace(-18.0, -8.0, 21)


def attenuation(tau: np.ndarray, geometry: str) -> np.ndarray:
    """Observed / unabsorbed intensity."""
    tau = np.asarray(tau, float)
    return np.exp(-tau) if geometry == "screen" else -np.expm1(-tau) / tau


def turnover(nu_ghz: np.ndarray, I: np.ndarray, below_ghz: float = 2.5e-3) -> dict:
    """Low-frequency log slope (nodes <= below_ghz) and the frequency of the intensity peak."""
    lo = nu_ghz <= below_ghz * (1 + 1e-9)
    slope = np.polyfit(np.log(nu_ghz[lo]), np.log(I[lo]), 1)[0]
    k = int(np.argmax(I))
    # parabola in (ln nu, ln I) through the three nodes around the maximum
    c = np.polyfit(np.log(nu_ghz[k - 1:k + 2]), np.log(I[k - 1:k + 2]), 2)
    peak = float(np.exp(-c[1] / (2 * c[0])))
    return {"low_freq_log_slope": float(slope), "slope_nodes_GHz": nu_ghz[lo].tolist(),
            "peak_node_GHz": float(nu_ghz[k]), "peak_parabolic_GHz": peak}


def fit_tau(nu_ghz: np.ndarray, I: np.ndarray, deg: int, top: int, geometry: str) -> dict:
    """Least squares in ln I over nodes 0..top-1; multi-start in ln tau_0."""
    lnu, y = np.log(nu_ghz[:top]), np.log(I[:top])

    def model(p):
        tau = np.exp(p[-1] - FF_INDEX * lnu)
        return np.polyval(p[:-1], lnu) + np.log(np.maximum(attenuation(tau, geometry), 1e-300))

    poly0 = np.polyfit(lnu[top // 2:], y[top // 2:], deg)
    best = None
    for lt in LN_TAU0_STARTS:
        r = least_squares(lambda p: model(p) - y, np.r_[poly0, lt], xtol=1e-14, ftol=1e-14, gtol=1e-14,
                          max_nfev=20000)
        if best is None or r.cost < best.cost:
            best = r
    tau0 = float(np.exp(best.x[-1]))
    return {"deg": deg, "top_index": top, "top_GHz": float(nu_ghz[top - 1]), "geometry": geometry,
            "tau0_at_1GHz": tau0, "rms_ln_resid": float(np.sqrt(np.mean(best.fun**2))),
            "converged": bool(best.success)}


def _refit(ctx_ci: np.ndarray, nu_hz: np.ndarray, fit_idx, held_idx, I: np.ndarray, R_list) -> dict:
    """Withheld fractional rms / max of the moment TSVD at fixed R and of log-cubic (1% weights)."""
    sig = 0.01 * I
    svd = BM.tsvd_setup(ctx_ci[fit_idx], sig[fit_idx])
    tf = BM.tsvd_fit(svd, I[fit_idx], sig[fit_idx])
    out = {}

    def stats(pred):
        fr = (pred[held_idx] - I[held_idx]) / I[held_idx]
        return [100 * float(np.sqrt(np.mean(fr**2))), 100 * float(np.max(np.abs(fr)))]

    for R in R_list:
        out[f"moment_R{R}_held_rms_max_pct"] = stats(ctx_ci @ BM.tsvd_moments(svd, tf, R))
    grid = SL.mono_grid(nu_hz)
    f = SL.fit("logcubic", grid.rows(fit_idx), I[fit_idx], sig[fit_idx])
    out["logcubic_held_rms_max_pct"] = stats(SL.predict("logcubic", f["params"], grid))
    return out


def analyse(raw: np.ndarray, sel_idx: np.ndarray, ctx: dict, R_list=(7, 9, 12)) -> dict:
    """Turnover, tau estimates over (degree, range, geometry), and de-absorbed refits.

    raw: full CSV array (frequency_ghz in column 0, high_latitude in column 3);
    sel_idx: original indices of the baseline nodes (0.1145-9.1 GHz);
    ctx: baselines GALPROP context (monochromatic CI on the selected nodes, fit/held indices).
    """
    nu_all, I_all = raw[:, 0], raw[:, 3]
    nu_l = float(nu_all[sel_idx[0]])
    fits = [fit_tau(nu_all, I_all, d, t, g) for g in ("slab", "screen") for d in DEGREES for t in TOPS]
    for f in fits:
        f["tau_at_lowest_node"] = f["tau0_at_1GHz"] * nu_l ** -FF_INDEX
        f["attenuation_at_lowest_node"] = float(1 - attenuation(f["tau_at_lowest_node"], f["geometry"]))
    slab = [f for f in fits if f["geometry"] == "slab"]
    taus = np.array([f["tau_at_lowest_node"] for f in slab])
    central = next(f for f in slab if (f["deg"], f["top_index"]) == CENTRAL)
    nu_sel = nu_all[sel_idx]
    att = attenuation(central["tau0_at_1GHz"] * nu_sel ** -FF_INDEX, "slab")
    I_pub = I_all[sel_idx]
    fi, hi = ctx["fit_idx"], ctx["held_idx"]
    refit = {"published": _refit(ctx["CI"], nu_sel * 1e9, fi, hi, I_pub, R_list),
             "de_absorbed_central_slab": _refit(ctx["CI"], nu_sel * 1e9, fi, hi, I_pub / att, R_list)}
    return {
        "statement": ("GALPROP absorption settings cannot be audited from the published FITS "
                      "(inputs/README.md). The estimate below uses the low-frequency turnover "
                      "of the published spectrum itself."),
        "model": ("ln I = poly_deg(ln nu) + ln A(tau), tau = tau_0 (nu/GHz)^-2.1; A = (1 - e^-tau)/tau "
                  "(mixed slab) or e^-tau (screen); least squares in ln I on nodes 0..top-1"),
        "turnover": turnover(nu_all, I_all),
        "fits": fits,
        "slab_tau_at_lowest_node_min_max": [float(taus.min()), float(taus.max())],
        "slab_attenuation_at_lowest_node_min_max": [float(1 - attenuation(taus.min(), "slab")),
                                                    float(1 - attenuation(taus.max(), "slab"))],
        "central_fit": {"deg": CENTRAL[0], "top_index": CENTRAL[1]},
        "central_slab_tau_at_lowest_node": central["tau_at_lowest_node"],
        "central_slab_attenuation_first_three_nodes": (1 - att[:3]).tolist(),
        "refit_withheld": refit,
        "lowest_node_GHz": nu_l,
        "label": "model-dependent estimate from the turnover, not a bound; depends on degree, range, geometry",
    }
