"""Electron families fitted over 1-40 GHz at three noise levels, evaluated over 1-300 GHz.

For Fig. 6(a, b). E2 is fitted with the eSBPL (correct) and eLP (incorrect) families, and E3 with the
eLP (correct) and ePL (incorrect) families, the pairs drawn in Fig. 6(a, b), through the full
separable columns of the 21 fit channels of
fits_extrap.py (fits_extrap.family_fit, I, Q and U, the paper's azimuth factor). The fitted
moments are evaluated on the 121 channels of extrap_curves.DENSE_CFG (1-300 GHz), whose last
channel is the 300 GHz prediction channel of fits_extrap.py.

Noise: N_REAL realisations with standard deviation frac * I_j on I, Q and U of the fitted
channels, from the standard normals of fits_extrap.py (same seed), at 1%, 0.01% and 0.001%. At 1%
and 0.01% the fits reproduce the 300 GHz errors and in-band chi2 of fits_extrap.py; the script
stops otherwise. For each ensemble, family and noise level the JSON gives the median and 16-84% range of
|I_fit - I|/I in each dense channel and at 300 GHz, and the in-band chi2 (median, fraction rejected
at p < 0.001).

Reads fits_extrap_products.npz (for the check). Writes fits_family_noise_results.json and
fits_family_noise_products.npz (the signed ratio I_fit/I of every realisation on the dense channels).
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
import fits_extrap as X
from config import PAPER

HERE = Path(__file__).resolve().parent
FAMILIES = {"E2": ("eSBPL", "eLP"), "E3": ("eLP", "ePL")}  # (correct, incorrect)
NOISE = (1e-2, 1e-4, 1e-5)
EXTRAP_PRODUCTS = HERE / "fits_extrap_products.npz"
OUT_NPZ = HERE / "fits_family_noise_products.npz"
OUT_JSON = HERE / "fits_family_noise_results.json"
T0 = time.time()


def main() -> None:
    ens = F.E.reference_ensembles(PAPER)
    _, full_f = X.columns(X.FIT_CFG)
    _, full_d = X.columns(EC.DENSE_CFG)
    Sf, _, _ = X.truth(X.FIT_CFG, ens)
    Sd, _, _ = X.truth(EC.DENSE_CFG, ens)
    rows = full_f.reshape(-1, full_f.shape[-1])
    nf = X.FIT_CFG.n_channels
    Z = np.random.default_rng(X.SEED).standard_normal((X.N_REAL, nf, 3))
    z = np.load(EXTRAP_PRODUCTS)
    out_npz = {"centres_dense": EC.DENSE_CFG.centres, **{f"truth_{n}": Sd[n][:, 0] for n in FAMILIES}}
    summary, checks = {}, {}
    for name, fam in ((n, f) for n, fs_ in FAMILIES.items() for f in fs_):
        dof = 3 * nf - BM.family_n_params(fam)
        for frac in NOISE:
            tag = f"{frac:g}"
            sig = frac * Sf[name][:, 0]
            jobs = [(fam, rows, (Sf[name] + (frac * Sf[name][:, :1]) * Z[i]).ravel(), np.repeat(sig, 3), full_d)
                    for i in range(X.N_REAL)]
            with ProcessPoolExecutor(max_workers=X.N_WORKERS) as pool:
                res = list(pool.map(X.family_fit, jobs, chunksize=10))
            if any(r is None for r in res):
                raise SystemExit(f"{name} {fam} {tag}: a family fit failed")
            ratio = np.stack([r[0][:, 0] for r in res]) / Sd[name][None, :, 0]
            chi2 = np.array([r[1] for r in res])
            if tag in ("0.01", "0.0001"):
                ref = z[f"{name}_{tag}_{fam}_err"][:, -1, 0]
                d_err = float(np.max(np.abs(np.abs(ratio[:, -1] - 1) - ref) / ref))
                d_chi = float(np.max(np.abs(chi2 - z[f"{name}_{tag}_{fam}_chi2"]) / z[f"{name}_{tag}_{fam}_chi2"]))
                if not (d_err <= 1e-6 and d_chi <= 1e-6):
                    raise SystemExit(f"{name} {fam} {tag}: differs from fits_extrap.py (err {d_err:.3g}, chi2 {d_chi:.3g})")
                checks[f"{name}_{fam}_{tag}"] = {"300GHz_err_max_rel_diff": d_err, "chi2_max_rel_diff": d_chi}
            e = np.abs(ratio - 1)
            out_npz[f"{name}_{fam}_{tag}_ratio"] = ratio
            out_npz[f"{name}_{fam}_{tag}_chi2"] = chi2
            summary.setdefault(name, {}).setdefault(fam, {"role": "correct" if fam == FAMILIES[name][0] else "incorrect",
                                                       "dof": int(dof)})[tag] = s = {
                "dI_over_I": {"median": np.median(e, axis=0).tolist(), "p16": np.percentile(e, 16, axis=0).tolist(),
                              "p84": np.percentile(e, 84, axis=0).tolist()},
                "dI_over_I_300GHz_median_p16_p84": [float(np.median(e[:, -1])), float(np.percentile(e[:, -1], 16)),
                                                    float(np.percentile(e[:, -1], 84))],
                "chi2_median": float(np.median(chi2)),
                "frac_rejected_p001": float(np.mean(chi2 > X.chi2_crit(dof, 1e-3)))}
            print(name, fam, tag, "300 GHz median %", round(100 * s["dI_over_I_300GHz_median_p16_p84"][0], 4),
                  "chi2 median", round(s["chi2_median"], 1), "/", dof, "rejected", s["frac_rejected_p001"], flush=True)
    np.savez(OUT_NPZ, **out_npz)
    F.write_json(OUT_JSON, {"description": __doc__, "families": FAMILIES, "noise_fracs": list(NOISE),
                            "n_real": X.N_REAL, "seed": X.SEED, "dense_config": EC.DENSE_CFG.describe(),
                            "checks": checks, "summary": summary,
                            "provenance": F.provenance(__file__, T0, inputs=(HERE / "fits_extrap.py",
                                                                             HERE / "baseline_models.py",
                                                                             EXTRAP_PRODUCTS))})
    print("checks", checks, "done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
