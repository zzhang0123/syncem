"""Free-moment fit of E2 over the 10-300 GHz window, for Fig. 6(c).

The fit uses the fits.py machinery (UPRE rank, whitened TSVD, N = 2 separable basis) with the
channels of config.CMB: 41 logarithmically spaced centres over 10-300 GHz, of which the 21 even
channels are fitted and the 20 odd channels are withheld. No electron family is assumed.
The basis keeps the energy cells this window is sensitive to, cells 6-19 of the 19-cell grid
(gamma >= edges[5]), 42 energy columns. Truth, moments and columns are computed on the full
grid and the retained cells are selected, so no electron is reassigned between cells. The script
records the largest contribution of the omitted cells to I_j (omitted_cells_max_over_I).

Noise: one seeded realisation (paper seed, the standard normals of fits.seeded_noise) scaled
to a standard deviation of 1% and 0.01% of I_j, so both noise levels share one realisation.
For each level the JSON gives the rank R*, chi2_fit and its degrees of freedom, and the
largest |fit - true| over the 41 channels relative to I_j (Stokes I alone and I, Q, U).

Reads nothing. Writes fits_cmb_results.json and fits_cmb_products.npz (channel centres, fitted
channel indices, true I/Q/U, fitted data and noise sd at 1%, predictions at both levels).
Run: python fits_cmb.py
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import fits as F
from config import CMB

HERE = Path(__file__).resolve().parent
NAME = "E2"
FIRST_CELL = 5  # zero-based index of the lowest retained cell
KEEP = np.arange(3 * FIRST_CELL, 3 * CMB.n_bins)  # cell-major columns b*3 + r
NOISE = (0.01, 1e-4)
OUT_NPZ = HERE / "fits_cmb_products.npz"
OUT_JSON = HERE / "fits_cmb_results.json"
T0 = time.time()


def inputs() -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Retained separable columns, direct spectrum and true moments of E2 in the CMB window."""
    DI, DQ = F.B.derivative_spectra(CMB, F.N_ORDER)
    full = F.B.separable_columns(CMB, DI, DQ)
    ens = F.E.reference_ensembles(CMB)[NAME]
    S = ens.direct(CMB, n=24)
    a_full = ens.moments(CMB, F.N_ORDER, n=24).ravel()
    drop = np.setdiff1d(np.arange(full.shape[-1]), KEEP)
    omitted = float(np.max(np.abs(F.B.predict(full[:, :, drop], a_full[drop])) / S[:, :1]))
    return full[:, :, KEEP], S, a_full[KEEP], omitted


def seeded_fit(C: np.ndarray, S: np.ndarray, a: np.ndarray, zf: np.ndarray, zh: np.ndarray) -> dict:
    """UPRE fit of the seeded realisation at the noise level of F.CFG."""
    lf = F.LinearFit(NAME, C, S, a)
    sc = lf.scan(lf.Sf + zf, lf.Sh + zh)
    R = int(sc["R_star"][0])
    pred = F.B.predict(lf.C, lf.a_hat(sc["proj"][0], R))
    err = np.abs(pred - S) / S[:, :1]
    return {"R_star": R, "pred": pred, "sigma": lf.sigma,
            "data_fit": S[F.FIT] + lf.sigma[F.FIT] * zf.reshape(-1, 3),
            "chi2_fit": float(sc["chi2_fit"][0, R - 1]), "dof_fit": int(len(lf.Sf) - R),
            "max_dI_over_I": float(np.max(err[:, 0])), "max_IQU_err_over_I": float(np.max(err)),
            "rms_held_dI_over_I": float(np.sqrt(np.mean(err[F.HELD, 0] ** 2)))}


def main() -> None:
    F.CFG = CMB
    C, S, a, omitted = inputs()
    zf, zh = F.seeded_noise()
    out_npz = {"centres": CMB.centres, "fit_idx": F.FIT, "S_direct": S}
    summary = {}
    for frac in NOISE:
        F.CFG = replace(CMB, noise_frac=frac)
        tag = f"{frac:g}"
        res = seeded_fit(C, S, a, zf, zh)
        out_npz[f"pred_{tag}"] = res["pred"]
        out_npz[f"data_fit_{tag}"], out_npz[f"sigma_{tag}"] = res["data_fit"], res["sigma"]
        summary[tag] = {k: v for k, v in res.items() if k not in ("pred", "sigma", "data_fit")}
        print(tag, summary[tag])
    F.CFG = CMB
    np.savez(OUT_NPZ, **out_npz)
    F.write_json(OUT_JSON, {
        "description": __doc__, "ensemble": NAME, "noise_fracs": list(NOISE), "config": CMB.describe(),
        "retained_cells": {"first_zero_based": FIRST_CELL, "gamma_lo": float(CMB.edges[FIRST_CELL]),
                           "n_cells": int(CMB.n_bins - FIRST_CELL), "n_energy_columns": int(KEEP.size)},
        "omitted_cells_max_over_I": omitted, "summary": summary,
        "provenance": F.provenance(__file__, T0, inputs=(HERE / "fits.py",))})
    print("omitted cells", omitted, "done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
