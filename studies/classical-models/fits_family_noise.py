"""E2 electron-family fits over 1-40 and 1-100 GHz at two noise levels, evaluated over 1-300 GHz.

For figures/cmb_extrapolation (a, b). E2 is fitted with the eSBPL (correct) and eLP (incorrect) families through the full
separable columns (fits_extrap.family_fit, I, Q and U, the paper's azimuth factor) of the first 21
(1-40 GHz) and 26 (1-100.6 GHz) channels of the nested grid of fits_cmb_bands.py (fits_cmb_bands.FULL_CFG,
whose first 21 channels are the fit channels of fits_extrap.py):
  * "1-40" and "1-100" name the bands by their rounded upper edges, as in fits_cmb_bands.py.
  * The "1-40" fits use the columns and truth of fits_extrap.FIT_CFG itself, so that they can be checked
    against fits_extrap.py to CHECK_RTOL (see main).
The fitted moments are evaluated on the 121 channels of extrap_curves.DENSE_CFG (1-300 GHz), whose last
channel is the 300 GHz prediction channel of fits_extrap.py.

Noise: N_REAL realisations with standard deviation frac * I_j on I, Q and U, at 1% and 0.01%, from the
standard normals of fits_cmb_bands.shared_noise, so that both bands fit the same data as the free-moment
fits of fits_cmb_bands.py. The first 21 channels use the normals of fits_extrap.py, so the "1-40" fits
reproduce its 300 GHz errors and in-band chi2; the script stops otherwise. For each band, family and
noise level the JSON gives the median and 16-84% range of |I_fit - I|/I in each dense channel and at
300 GHz, and the in-band chi2 (median, fraction rejected at p < 0.001).

Reads fits_extrap_products.npz (for the check). Writes fits_family_noise_results.json and
fits_family_noise_products.npz: the signed ratio I_fit/I of every realisation on the dense channels, the
in-band chi2, the channel centres, the true spectra and the 1% data of the first realisation (the data
drawn in panels (a, b)) on the fitted channels of each band.
Run: python fits_family_noise.py
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

import fits as F  # first: puts lib/ on the path
import baseline_models as BM
import extrap_curves as EC
import fits_cmb_bands as CB
import fits_extrap as X
from config import PAPER

HERE = Path(__file__).resolve().parent
NAME = "E2"
FAMILIES = ("eSBPL", "eLP")  # (correct, incorrect)
NOISE = (1e-2, 1e-4)
BANDS = {"1-40": 21, "1-100": 26}  # number of leading channels of fits_cmb_bands.FULL_CFG fitted
DRAWN = 0  # the realisation whose 1% data and fits panels (a, b) draw (the first, as extrap_curves.py)
CHECK_RTOL = 1e-6
EXTRAP_PRODUCTS = HERE / "fits_extrap_products.npz"
OUT_NPZ = HERE / "fits_family_noise_products.npz"
OUT_JSON = HERE / "fits_family_noise_results.json"
T0 = time.time()


def fit_all(fam, rows, S, frac, Z, full_d):
    """Fits of every realisation: signed I_fit/I on the dense channels (before division) and in-band chi2."""
    sig = frac * S[:, 0]
    jobs = [(fam, rows, (S + (frac * S[:, :1]) * Z[i]).ravel(), np.repeat(sig, 3), full_d)
            for i in range(Z.shape[0])]
    with ProcessPoolExecutor(max_workers=X.N_WORKERS) as pool:
        res = list(pool.map(X.family_fit, jobs, chunksize=10))
    if any(r is None for r in res):
        raise SystemExit(f"{fam} {frac:g}: a family fit failed")
    return np.stack([r[0][:, 0] for r in res]), np.array([r[1] for r in res])


def check_against_extrap(z, fam, tag, ratio, chi2) -> dict:
    """The 1-40 GHz fits must reproduce the 300 GHz |dI|/I and in-band chi2 of fits_extrap.py."""
    ref = z[f"{NAME}_{tag}_{fam}_err"][:, -1, 0]
    d_err = float(np.max(np.abs(np.abs(ratio[:, -1] - 1) - ref) / ref))
    ref_chi = z[f"{NAME}_{tag}_{fam}_chi2"]
    d_chi = float(np.max(np.abs(chi2 - ref_chi) / ref_chi))
    if not (d_err <= CHECK_RTOL and d_chi <= CHECK_RTOL):
        raise SystemExit(f"1-40 {fam} {tag}: differs from fits_extrap.py (err {d_err:.3g}, chi2 {d_chi:.3g})")
    return {"300GHz_err_max_rel_diff": d_err, "chi2_max_rel_diff": d_chi}


def summarise(ratio, chi2, dof) -> dict:
    e = np.abs(ratio - 1)
    return {"dI_over_I": {"median": np.median(e, axis=0).tolist(), "p16": np.percentile(e, 16, axis=0).tolist(),
                          "p84": np.percentile(e, 84, axis=0).tolist()},
            "dI_over_I_300GHz_median_p16_p84": [float(np.median(e[:, -1])), float(np.percentile(e[:, -1], 16)),
                                                float(np.percentile(e[:, -1], 84))],
            "chi2_median": float(np.median(chi2)),
            "frac_rejected_p001": float(np.mean(chi2 > X.chi2_crit(dof, 1e-3)))}


def main() -> None:
    if not np.allclose(CB.FULL_CFG.centres[:X.FIT_CFG.n_channels], X.FIT_CFG.centres, rtol=1e-12):
        raise SystemExit("the first 21 channels differ from the fit channels of fits_extrap.py")
    ens = F.E.reference_ensembles(PAPER)
    _, full_c = X.columns(CB.FULL_CFG)
    _, full_f = X.columns(X.FIT_CFG)
    _, full_d = X.columns(EC.DENSE_CFG)
    Sc, _, _ = X.truth(CB.FULL_CFG, ens)
    Sf, _, _ = X.truth(X.FIT_CFG, ens)
    Sd, _, _ = X.truth(EC.DENSE_CFG, ens)
    Z_full = CB.shared_noise()
    z = np.load(EXTRAP_PRODUCTS)
    out_npz = {"centres_dense": EC.DENSE_CFG.centres, "centres_fit": CB.FULL_CFG.centres,
               f"truth_{NAME}": Sd[NAME][:, 0], f"truth_fit_{NAME}": Sc[NAME]}
    summary, checks = {}, {}
    for band, n_fit in BANDS.items():
        # 1-40 GHz: the inputs of fits_extrap.py itself. Its channels agree with the first 21 of the nested
        # grid to 1e-14, but the nonlinear family fits turn that rounding into 300 GHz differences of up to
        # 5e-4 of small errors, which would hide a real discrepancy in the check.
        cols, St = (full_f, Sf) if band == "1-40" else (full_c, Sc)
        rows = cols[:n_fit].reshape(-1, cols.shape[-1])
        S = St[NAME][:n_fit]
        out_npz[f"data_{NAME}_{band}_0.01"] = S + (1e-2 * S[:, :1]) * Z_full[DRAWN, :n_fit]
        out_npz[f"sigma_{NAME}_{band}_0.01"] = 1e-2 * S[:, 0]
        for fam in FAMILIES:
            dof = 3 * n_fit - BM.family_n_params(fam)
            for frac in NOISE:
                tag = f"{frac:g}"
                pred, chi2 = fit_all(fam, rows, S, frac, Z_full[:, :n_fit], full_d)
                ratio = pred / Sd[NAME][None, :, 0]
                if band == "1-40":
                    checks[f"{band}_{fam}_{tag}"] = check_against_extrap(z, fam, tag, ratio, chi2)
                out_npz[f"{NAME}_{band}_{fam}_{tag}_ratio"] = ratio
                out_npz[f"{NAME}_{band}_{fam}_{tag}_chi2"] = chi2
                s = summarise(ratio, chi2, dof)
                s["drawn_realisation_chi2"] = float(chi2[DRAWN])
                summary.setdefault(band, {"n_channels": n_fit,
                                          "upper_centre_GHz": float(CB.FULL_CFG.centres[n_fit - 1] / 1e9)}) \
                    .setdefault(fam, {"role": "correct" if fam == FAMILIES[0] else "incorrect",
                                      "dof": int(dof)})[tag] = s
                print(band, fam, tag, "300 GHz median %", round(100 * s["dI_over_I_300GHz_median_p16_p84"][0], 4),
                      "chi2 median", round(s["chi2_median"], 1), "/", dof, "rejected", s["frac_rejected_p001"],
                      flush=True)
    np.savez(OUT_NPZ, **out_npz)
    F.write_json(OUT_JSON, {"description": __doc__, "ensemble": NAME, "families": list(FAMILIES),
                            "noise_fracs": list(NOISE), "bands": BANDS, "drawn_realisation": DRAWN,
                            "n_real": X.N_REAL, "seed": X.SEED, "fit_config": CB.FULL_CFG.describe(),
                            "dense_config": EC.DENSE_CFG.describe(), "checks": checks, "summary": summary,
                            "provenance": F.provenance(__file__, T0, inputs=(HERE / "fits_extrap.py",
                                                                             HERE / "fits_cmb_bands.py",
                                                                             HERE / "baseline_models.py",
                                                                             EXTRAP_PRODUCTS))})
    print("checks", checks, "done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
