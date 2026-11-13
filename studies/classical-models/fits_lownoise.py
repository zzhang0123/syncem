"""Seeded UPRE fits of the moment_fit_gallery_compact rows repeated at a lower noise level.

The rows are E2, E4-KP and E5. The fits use the fits.py basis, direct spectra, true moments and
the same standard-normal draw (paper seed) for the seeded realisation, with the noise sd set to
frac * I_j for frac = 1% and 0.01%. The 1% run must reproduce the predictions and ranks saved in
fits_products.npz, which checks the driver. For E5 the separable fit with one free complex
azimuth factor (fits.FreeFactorFit) is repeated at 1% and compared with fits_results.json.

Reads fits_results.json and fits_products.npz (python fits.py).
Writes fits_lownoise_products.npz and fits_lownoise_results.json.  Run: python fits_lownoise.py
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import fits as F

HERE = Path(__file__).resolve().parent
ROWS = ("E2", "E4-KP", "E5")
LOW = 1e-4
ARCHIVE = HERE / "fits_products.npz"
RESULTS = HERE / "fits_results.json"
OUT_NPZ = HERE / "fits_lownoise_products.npz"
OUT_JSON = HERE / "fits_lownoise_results.json"
T0 = time.time()


def seeded_fit(name: str, cols: dict, S: dict, a: dict, zf: np.ndarray, zh: np.ndarray) -> dict:
    """UPRE fit of one seeded realisation under the current F.CFG noise level."""
    cplx = (np.arange(F.NE, 2 * F.NE), np.arange(2 * F.NE, 3 * F.NE)) if name == "E5" else None
    lf = F.LinearFit(name, F.basis_for(name, cols), S[name], a[name], cplx=cplx)
    sc = lf.scan(lf.Sf + zf, lf.Sh + zh)
    R = int(sc["R_star"][0])
    pred = F.B.predict(lf.C, lf.a_hat(sc["proj"][0], R))
    return {"R_star": R, "pred": pred, "S": lf.S,
            "chi2_fit": float(sc["chi2_fit"][0, R - 1]), "dof_fit": int(len(lf.Sf) - R),
            "max_fit_minus_direct_over_I": float(np.max(np.abs(pred - lf.S) / lf.S[:, :1]))}


def main() -> None:
    cols, S, a, _, _ = F.build_inputs()
    zf, zh = F.seeded_noise()
    base = F.CFG
    archive = dict(np.load(ARCHIVE))
    out_npz, summary, checks = {"centres": base.centres}, {}, {}
    for frac in (base.noise_frac, LOW):
        F.CFG = replace(base, noise_frac=frac)
        for name in ROWS:
            k = name.replace("-", "_")
            res = seeded_fit(name, cols, S, a, zf, zh)
            tag = f"{frac:g}"
            out_npz[f"{k}_pred_{tag}"] = res["pred"]
            out_npz[f"{k}_S_direct"] = res["S"]
            out_npz[f"{k}_R_star_{tag}"] = res["R_star"]
            summary.setdefault(tag, {})[name] = {key: v for key, v in res.items() if key not in ("pred", "S")}
            if frac == base.noise_frac:
                I = res["S"][:, :1]
                checks[f"{name}_pred_vs_archive_max_over_I"] = float(
                    np.max(np.abs(res["pred"] - archive[f"{k}_pred"]) / I))
                checks[f"{name}_R_star_minus_archive"] = res["R_star"] - int(archive[f"{k}_R_star"])
    F.CFG = base
    # E5 comparator: one free azimuth factor shared by all cells (fits.run_free_factor, seeded part)
    ff = F.FreeFactorFit(cols["wI"], cols["wQ"], S["E5"])
    sep = ff.fit(ff.Sf + zf, ff.Sh + zh)
    ref = json.loads(RESULTS.read_text())["E5"]["separable_free_factor"]["seeded"]
    out_npz["E5_free_factor_pred"] = sep["pred"]
    checks["E5_free_factor_R_star_minus_archive"] = sep["R_star"] - ref["R_star"]
    checks["E5_free_factor_chi2_fit_minus_archive"] = sep["chi2_fit"] - ref["chi2_fit"]
    checks["E5_free_factor_c_minus_archive"] = float(abs(sep["c"] - complex(*ref["c"])))
    np.savez(OUT_NPZ, **out_npz)
    F.write_json(OUT_JSON, {"description": __doc__, "rows": list(ROWS), "noise_fracs": [base.noise_frac, LOW],
                            "summary": summary, "checks": checks,
                            "provenance": F.provenance(__file__, T0, inputs=(ARCHIVE, RESULTS, HERE / "fits.py"))})
    print("checks:", checks)
    for tag, rec in summary.items():
        print(tag, {n: (r["R_star"], round(100 * r["max_fit_minus_direct_over_I"], 4)) for n, r in rec.items()})
    print("done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
