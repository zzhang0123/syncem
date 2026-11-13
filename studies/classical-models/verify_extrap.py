"""Checks of the extrapolation test (fits_extrap.py and the scripts that read its outputs).

The values of the extrapolation table are recomputed from the per-realisation arrays of
fits_extrap_products.npz and compared with fits_extrap_results.json:
  * per prediction channel, the median and 84th percentile of |dI|/I, |d|P||/I and |d chi| for
    every model and noise level;
  * per electron family, the median in-band chi2 and the fraction rejected at p < 0.001.
The values that have no key in fits_extrap_results.json are computed and stored here: the
16th percentile of |dI|/I at 300 GHz, the table row of each family (chi2/dof, rejection rate,
median and 16-84% range of |dI|/I at 300 GHz), and the fraction of realisations in which the
in-band chi2 of the right family is lower than that of the wrong family by more than the
p = 0.01 point of chi2 with one degree of freedom.

The basis diagnostics are recomputed with numpy from the whitened fit design (not through
fits.LinearFit): the norm fraction of the 300 GHz intensity row outside the retained right
singular vectors at the median R*, compared with fits_extrap_noisefree_results.json.

The electron families are refitted to the noise-free fit-band data, and the 300 GHz intensity error
of each fit is recorded. For the right family this is the extrapolation error that remains without
noise (N = 2 representation and the shape of the family), not a pass/fail check.

The dense curves of extrap_curves.npz share the first channel (1 GHz) with the fit grid and the
last channel (300 GHz) with the prediction grid. There the seed-0 data, truth and model curves
must equal the arrays of fits_extrap_products.npz.

The column setup (fits_extrap.columns, fits_extrap.truth) and the family fit
(baseline_models.fit_family) are imported from the study code, so this is a consistency check of
the saved results, not an independent rebuild of the kernels.

Writes verify_extrap_results.json.  Run: python verify_extrap.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import baseline_models as BM  # noqa: E402
import fits_extrap as X  # noqa: E402

F = X.F
HERE = Path(__file__).resolve().parent
PRODUCTS = HERE / "fits_extrap_products.npz"
RESULTS = HERE / "fits_extrap_results.json"
NOISEFREE = HERE / "fits_extrap_noisefree_results.json"
CURVES = HERE / "extrap_curves.npz"
OUT = HERE / "verify_extrap_results.json"
TAG = f"{X.NOISE[0]:g}"  # the table is at 1% noise
# (ensemble, right family, wrong family) as drawn in fig_cmb_extrapolation.py
PAIRS = (("E2", "eSBPL", "eLP"), ("E3", "eLP", "ePL"))
P_SELECT = 0.01  # threshold of the in-band chi2 difference (one degree of freedom)
QUANTITIES = ("dI_over_I", "dP_over_I", "dchi_deg")
LIMITS = {"summary_rel": 1e-9, "chi2_rel": 1e-12, "rejected_abs": 0.0, "noisefree_row_abs": 1e-8,
          "curves_rel": 1e-9}
T0 = time.time()


def rel(a, b) -> float:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    if not np.array_equal(np.isfinite(a), np.isfinite(b)):
        return float("inf")
    return float(np.max(np.abs(a[ok] - b[ok]) / np.maximum(np.abs(b[ok]), 1e-300))) if ok.any() else 0.0


def summary_checks(z, summ) -> dict:
    """Largest relative difference of the recomputed per-channel medians and 84th percentiles."""
    worst = {"median": 0.0, "p84": 0.0, "max_over_channels_of_median": 0.0}
    for tag, rows in summ.items():
        for name, rec in rows.items():
            for model, mrec in rec.items():
                key = f"{name.replace('-', '_')}_{tag}_{model}_err"
                if key not in z.files:
                    continue
                e = z[key]
                for j, q in enumerate(QUANTITIES):
                    med = np.nanmedian(e[..., j], axis=0)
                    worst["median"] = max(worst["median"], rel(med, mrec[q]["median"]))
                    worst["p84"] = max(worst["p84"], rel(np.nanpercentile(e[..., j], 84, axis=0), mrec[q]["p84"]))
                    worst["max_over_channels_of_median"] = max(
                        worst["max_over_channels_of_median"], rel(np.nanmax(med), mrec[q]["max_over_channels_of_median"]))
    return worst


def family_rows(z, summ) -> tuple[dict, dict]:
    """Table rows of every family at TAG, and their checks against the JSON."""
    rows, checks = {}, {"chi2_median_rel": 0.0, "frac_rejected_abs": 0.0, "n_failed": 0}
    for name, rec in summ[TAG].items():
        k = name.replace("-", "_")
        for fam in X.FAMILIES:
            if fam not in rec:
                continue
            chi2 = z[f"{k}_{TAG}_{fam}_chi2"]
            dof = rec[fam]["dof"]
            rej = float(np.mean(chi2 > chi2_dist.isf(1e-3, dof)))
            checks["chi2_median_rel"] = max(checks["chi2_median_rel"], rel(np.median(chi2), rec[fam]["chi2_median"]))
            checks["frac_rejected_abs"] = max(checks["frac_rejected_abs"], abs(rej - rec[fam]["frac_rejected_p001"]))
            checks["n_failed"] += rec[fam]["n_failed"]
            dI = z[f"{k}_{TAG}_{fam}_err"][:, :, 0]
            med = np.nanmedian(dI, axis=0)
            rows[f"{name}_{fam}"] = {
                "chi2_over_dof_median": float(np.median(chi2) / dof), "rejected_pct": 100 * rej,
                "dI_over_I_300GHz_pct_median_p16_p84": [100 * float(np.nanpercentile(dI[:, -1], q)) for q in (50, 16, 84)],
                "largest_median_at_300GHz": bool(int(np.nanargmax(med)) == med.size - 1)}
    return rows, checks


def selection_fractions(z, summ) -> dict:
    """Fraction of realisations where chi2(wrong) - chi2(right) exceeds the p = 0.01 point of chi2_1."""
    crit = float(chi2_dist.isf(P_SELECT, 1))
    out = {"threshold": crit}
    for name, right, wrong in PAIRS:
        rec = summ[TAG][name]
        if rec[right]["n_failed"] or rec[wrong]["n_failed"]:
            raise SystemExit(f"{name}: failed family fits, the chi2 arrays are not aligned by realisation")
        k = name.replace("-", "_")
        d = z[f"{k}_{TAG}_{wrong}_chi2"] - z[f"{k}_{TAG}_{right}_chi2"]
        out[name] = {"right": right, "wrong": wrong, "fraction_right_lower_by_more_than_threshold": float(np.mean(d > crit)),
                     "median_chi2_wrong_minus_right": float(np.median(d))}
    return out


def setup():
    ens = F.E.reference_ensembles(X.PAPER)
    cf, full_f = X.columns(X.FIT_CFG)
    cp, full_p = X.columns(X.PRED_CFG)
    Sf, af, _ = X.truth(X.FIT_CFG, ens)
    Sp, _, _ = X.truth(X.PRED_CFG, ens)
    return cf, cp, full_f, full_p, Sf, Sp


def row300_outside(summ, noisefree, cf, cp, Sf) -> dict:
    """Norm fraction of the 300 GHz I row outside span(V_R), from a numpy SVD of the whitened fit design."""
    out, worst = {}, 0.0
    for tag, rows in summ.items():
        frac = float(tag)
        for name in rows:
            key = "joint" if name == "E5" else "sep"
            sig = frac * Sf[name][:, 0]
            A = (cf[key] / sig[:, None, None]).reshape(-1, cf[key].shape[-1])
            _, s, vt = np.linalg.svd(A, full_matrices=False)
            vt = vt[s > 1e-13 * s[0]]
            R = int(round(rows[name]["basis"]["R_star_median"]))
            c = cp[key][-1, 0, :]
            V = vt[:R].T
            val = float(np.linalg.norm(c - V @ (V.T @ c)) / np.linalg.norm(c))
            ref = noisefree["results"][tag][name]["row300_norm_fraction_outside_VR"]
            out[f"{name}_{tag}"] = {"R": R, "verifier": val, "noisefree_json": ref}
            worst = max(worst, abs(val - ref))
    out["max_abs_diff"] = worst
    return out


def noisefree_families(full_f, full_p, Sf, Sp) -> dict:
    """Families fitted to noise-free fit-band I, Q, U (1% weights); |dI|/I at 300 GHz."""
    out = {}
    for name, right, wrong in PAIRS:
        sig = X.NOISE[0] * Sf[name][:, 0]
        for fam in (right, wrong):
            res = BM.fit_family(fam, full_f.reshape(-1, full_f.shape[-1]), Sf[name].ravel(), np.repeat(sig, 3))
            pred = F.B.predict(full_p, res["params"][0] * BM.family_moments(fam, res["params"][1:]).ravel())
            out[f"{name}_{fam}"] = {"role": "right" if fam == right else "wrong", "chi2_noisefree": float(res["chi2"]),
                                    "dI_over_I_300GHz": float(abs(pred[-1, 0] - Sp[name][-1, 0]) / Sp[name][-1, 0]),
                                    "max_dI_over_I_40_300GHz": float(np.max(np.abs(pred[:, 0] - Sp[name][:, 0]) / Sp[name][:, 0]))}
    return out


def curve_checks(z, cv) -> dict:
    """extrap_curves.npz at its shared end channels against fits_extrap_products.npz (seed 0, 1%)."""
    nd = cv["nu_dense"]
    if not (np.isclose(nd[0], z["nu_fit"][0], rtol=1e-12) and np.isclose(nd[-1], z["nu_pred"][-1], rtol=1e-12)):
        raise SystemExit("extrap_curves.npz end channels do not coincide with the fit and prediction grids")
    worst = 0.0
    for name in X.ROWS:
        k = name.replace("-", "_")
        worst = max(worst, rel(cv[f"data_fit_{name}"], z[f"{k}_{TAG}_data_seed0"]),
                    rel(cv[f"truth_{name}"][-1], z[f"{k}_S_pred"][-1]),
                    rel(cv[f"truth_{name}"][0], z[f"{k}_S_fit"][0]),
                    rel(cv[f"{name}_basis"][-1], z[f"{k}_{TAG}_basis_pred_seed0"][-1]))
        models = [*X.LAWS] + ([*X.FAMILIES] if name != "E5" else [])
        for m in models:
            worst = max(worst, rel(cv[f"{name}_{m}"][-1], z[f"{k}_{TAG}_{m}_pred_seed0"][-1]))
    return {"max_rel_diff_at_shared_channels": worst}


def main() -> None:
    z = np.load(PRODUCTS)
    summ = json.loads(RESULTS.read_text())["summary"]
    noisefree = json.loads(NOISEFREE.read_text())
    cv = np.load(CURVES)
    cf, cp, full_f, full_p, Sf, Sp = setup()
    rows, fam_checks = family_rows(z, summ)
    res = {
        "description": __doc__, "noise": TAG, "pairs": [list(p) for p in PAIRS], "limits": LIMITS,
        "summary_vs_arrays_max_rel": summary_checks(z, summ), "family_checks": fam_checks,
        "table_rows": rows, "selection": selection_fractions(z, summ),
        "row300_outside_VR": row300_outside(summ, noisefree, cf, cp, Sf),
        "noisefree_family_fits": noisefree_families(full_f, full_p, Sf, Sp),
        "curves": curve_checks(z, cv)}
    failed = [k for k, v, lim in (
        ("summary", max(res["summary_vs_arrays_max_rel"].values()), LIMITS["summary_rel"]),
        ("chi2_median", fam_checks["chi2_median_rel"], LIMITS["chi2_rel"]),
        ("frac_rejected", fam_checks["frac_rejected_abs"], LIMITS["rejected_abs"]),
        ("row300", res["row300_outside_VR"]["max_abs_diff"], LIMITS["noisefree_row_abs"]),
        ("curves", res["curves"]["max_rel_diff_at_shared_channels"], LIMITS["curves_rel"])) if not v <= lim]
    res["failed"] = failed
    res["provenance"] = F.provenance(__file__, T0, inputs=(PRODUCTS, RESULTS, NOISEFREE, CURVES))
    F.write_json(OUT, res)
    print(json.dumps({k: res[k] for k in ("summary_vs_arrays_max_rel", "family_checks", "table_rows", "selection",
                                          "curves", "failed")}, indent=1))
    if failed:
        raise SystemExit(f"verify_extrap: checks failed: {', '.join(failed)}")
    print("done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
