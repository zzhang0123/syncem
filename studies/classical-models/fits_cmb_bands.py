"""Free-moment fits of one data set over three nested bands from 1 GHz, evaluated over 1-300 GHz, for figures/cmb_extrapolation (c).

The fits use the fits.py machinery (whitened TSVD, UPRE rank, N = 2 separable basis, no electron
family) for E2 and E3, and keep cells 4-19 of the 19-cell grid (gamma >= 1.28e3), as fits_extrap.py
does. The channels form one logarithmic grid that extends the 21 fit channels of fits_extrap.py
(1-40 GHz, ratio 40^(1/20)) to 32 channels, up to 40^(31/20) GHz = 304.2 GHz. The three bands fit the
first 21 (1-40 GHz), 26 (1-100.6 GHz) and 32 (1-304.2 GHz) channels of the same data:
  * "1-40", "1-100", "1-300" name the bands by their rounded upper edges.
Truth, moments and columns are computed on the full cell grid and the retained cells are selected.
The fitted moments are evaluated on the 121 channels of extrap_curves.DENSE_CFG (1-300 GHz), whose
last channel is the 300 GHz prediction channel of fits_extrap.py.

Noise: N_REAL realisations of independent Gaussian noise with standard deviation frac * I_j on I, Q
and U of the 32 channels at 1% and 0.01%, shared by the three bands. The first 21 channels use the standard normals
of fits_extrap.py (its seed and array shape), so the "1-40" fits reproduce the 300 GHz errors of its
free-moment ("basis") fits at 1% and 0.01% noise to EXTRAP_RTOL; the script stops otherwise. Channels 22-32 use a
second stream (seed + 1).
For each ensemble, band and noise level the JSON gives the median and 16-84% range of R* and of
|I_fit - I|/I in each dense channel and at 300 GHz.

Reads fits_extrap_products.npz (for the check). Writes fits_cmb_bands_results.json and
fits_cmb_bands_products.npz: the signed ratio I_fit/I of every realisation on the dense channels, the
channel centres, the true spectra on them, and the data and noise of a typical realisation per ensemble
and noise level (typical_realisation: closest to the median 300 GHz error in all three bands).
Run: python fits_cmb_bands.py
"""
from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import extrap_curves as EC
import fits as F
import fits_extrap as X
from config import PAPER

HERE = Path(__file__).resolve().parent
ROWS = ("E2", "E3")
N_REAL = X.N_REAL
SEED = X.SEED
NOISE = (1e-2, 1e-4)  # 1% and 0.01%, the noise levels of figures/cmb_extrapolation
RATIO = 40.0 ** (1 / 20)  # channel spacing of the 21 fit channels of fits_extrap.py
N_FULL = 32
FULL_CFG = replace(PAPER, nu_lo=1e9, nu_hi=1e9 * RATIO ** (N_FULL - 1), n_channels=N_FULL)
BANDS = {"1-40": 21, "1-100": 26, "1-300": 32}  # number of leading channels of FULL_CFG fitted
EXTRAP_PRODUCTS = HERE / "fits_extrap_products.npz"
OUT_NPZ = HERE / "fits_cmb_bands_products.npz"
OUT_JSON = HERE / "fits_cmb_bands_results.json"
T0 = time.time()


def full_inputs():
    """Separable columns, direct spectra and true moments on the 32 channels, retained cells only."""
    DI, DQ = F.B.derivative_spectra(FULL_CFG, F.N_ORDER)
    cols = F.B.separable_columns(FULL_CFG, DI, DQ)[:, :, X.KEEP]
    ens = F.E.reference_ensembles(PAPER)
    S = {n: ens[n].direct(FULL_CFG, n=24) for n in ROWS}
    a = {n: ens[n].moments(FULL_CFG, F.N_ORDER, n=24).ravel()[X.KEEP] for n in ROWS}
    return cols, S, a


def shared_noise() -> np.ndarray:
    """Standard normals of the 32 channels: those of fits_extrap.py for the first 21, a second stream after."""
    n0 = X.FIT_CFG.n_channels
    first = np.random.default_rng(SEED).standard_normal((N_REAL, n0, 3))
    rest = np.random.default_rng(SEED + 1).standard_normal((N_REAL, N_FULL - n0, 3))
    return np.concatenate([first, rest], axis=1)


def fit_band(n_fit, full, Z_full, dense_cols, S_dense) -> dict:
    """UPRE fits of all realisations of the first n_fit channels; I_fit/I on the dense channels."""
    cols_full, S_full, a = full
    cols, S = cols_full[:n_fit], {n: S_full[n][:n_fit] for n in ROWS}
    F.FIT, F.HELD = np.arange(n_fit), np.arange(n_fit, n_fit + EC.DENSE_CFG.n_channels)
    Z = Z_full[:, :n_fit]
    out = {}
    for frac in NOISE:
        F.CFG = replace(PAPER, noise_frac=frac)
        for name in ROWS:
            lf = F.LinearFit(name, np.concatenate([cols, dense_cols], axis=0),
                             np.concatenate([S[name], S_dense[name]], axis=0), a[name])
            sc = lf.scan(lf.Sf[None, :] + Z.reshape(N_REAL, -1), np.repeat(lf.Sh[None, :], N_REAL, axis=0))
            R = sc["R_star"]
            I_d = np.stack([F.B.predict(dense_cols, lf.a_hat(sc["proj"][i], int(R[i])))[:, 0]
                            for i in range(N_REAL)])
            out[(name, f"{frac:g}")] = {"R": R, "ratio": I_d / S_dense[name][None, :, 0]}
    return out


EXTRAP_RTOL = 1e-4  # the inputs agree with those of fits_extrap.py to 1e-14 (channel centres, columns,
# truth); extrapolating to 300 GHz amplifies that rounding along weakly constrained directions to < 1e-4 (largest 9.4e-5)


def check_against_extrap(res) -> dict:
    """The 1-40 GHz fits must reproduce the 300 GHz |dI|/I of the basis fits of fits_extrap.py."""
    z = np.load(EXTRAP_PRODUCTS)
    out = {}
    for name in ROWS:
        for tag in ("0.01", "0.0001"):
            ref = z[f"{name}_{tag}_basis_err"][:, -1, 0]
            diff = float(np.max(np.abs(np.abs(res[(name, tag)]["ratio"][:, -1] - 1) - ref) / ref))
            if not diff <= EXTRAP_RTOL:
                raise SystemExit(f"1-40 {name} {tag}: 300 GHz errors differ from fits_extrap.py by {diff:.3g}")
            out[f"{name}_{tag}_300GHz_max_rel_diff"] = diff
    return out


def typical_realisation(band_ratios) -> tuple[int, float]:
    """The realisation whose 300 GHz error is closest to the median in every band: it minimises, over the
    realisations, the largest distance of the fractional rank of |I_fit/I - 1| at 300 GHz from 0.5."""
    ranks = []
    for r in band_ratios:
        e = np.abs(r[:, -1] - 1)
        ranks.append((np.argsort(np.argsort(e)) + 0.5) / e.size)
    dev = np.max(np.abs(np.array(ranks) - 0.5), axis=0)
    k = int(np.argmin(dev))
    return k, float(dev[k])


def summarise(r) -> dict:
    e, R = np.abs(r["ratio"] - 1), r["R"]
    return {"R_star_median": float(np.median(R)), "R_star_p16_p84": np.percentile(R, [16, 84]).tolist(),
            "dI_over_I": {"median": np.median(e, axis=0).tolist(), "p16": np.percentile(e, 16, axis=0).tolist(),
                          "p84": np.percentile(e, 84, axis=0).tolist()},
            "dI_over_I_300GHz_median_p16_p84": [float(np.median(e[:, -1])), float(np.percentile(e[:, -1], 16)),
                                                float(np.percentile(e[:, -1], 84))]}


def main() -> None:
    if not np.allclose(FULL_CFG.centres[:X.FIT_CFG.n_channels], X.FIT_CFG.centres, rtol=1e-12):
        raise SystemExit("the first 21 channels differ from the fit channels of fits_extrap.py")
    DId, DQd = F.B.derivative_spectra(EC.DENSE_CFG, F.N_ORDER)
    dense_cols = F.B.separable_columns(EC.DENSE_CFG, DId, DQd)[:, :, X.KEEP]
    ens = F.E.reference_ensembles(PAPER)
    S_dense = {n: ens[n].direct(EC.DENSE_CFG, n=24) for n in ROWS}
    full = full_inputs()
    Z_full = shared_noise()
    out_npz = {"centres_dense": EC.DENSE_CFG.centres, "centres_fit": FULL_CFG.centres,
               **{f"truth_{n}": S_dense[n][:, 0] for n in ROWS}, **{f"truth_fit_{n}": full[1][n] for n in ROWS}}
    summary, checks, ratios = {}, {}, {}
    for band, n_fit in BANDS.items():
        res = fit_band(n_fit, full, Z_full, dense_cols, S_dense)
        ratios[band] = {key: r["ratio"] for key, r in res.items()}
        if band == "1-40":
            checks = check_against_extrap(res)
        for (name, tag), r in res.items():
            out_npz[f"{name}_{band}_{tag}_ratio"] = r["ratio"]
            out_npz[f"{name}_{band}_{tag}_R"] = r["R"]
            summary.setdefault(band, {}).setdefault(tag, {})[name] = s = summarise(r)
            print(band, tag, name, "R*", s["R_star_median"], "300 GHz median/p16/p84 %",
                  [round(100 * v, 3) for v in s["dI_over_I_300GHz_median_p16_p84"]], flush=True)
        summary[band]["n_channels"] = n_fit
        summary[band]["upper_centre_GHz"] = float(FULL_CFG.centres[n_fit - 1] / 1e9)
    typical = {}
    for name in ROWS:
        for frac in NOISE:
            tag = f"{frac:g}"
            k, dev = typical_realisation([ratios[b][(name, tag)] for b in BANDS])
            typical[f"{name}_{tag}"] = {"index": k, "max_rank_deviation_from_median": dev}
            sigma = frac * full[1][name][:, :1]
            out_npz[f"data_{name}_{tag}"] = full[1][name] + sigma * Z_full[k]
            out_npz[f"sigma_{name}_{tag}"] = sigma[:, 0]
            out_npz[f"typical_{name}_{tag}"] = k
    print("typical realisations", typical)
    np.savez(OUT_NPZ, **out_npz)
    F.write_json(OUT_JSON, {"description": __doc__, "rows": list(ROWS), "noise_fracs": list(NOISE),
                            "fit_config": FULL_CFG.describe(),
                            "n_real": N_REAL, "seed": SEED, "dense_config": EC.DENSE_CFG.describe(),
                            "retained_cells_first_zero_based": X.FIRST_CELL, "checks": checks,
                            "typical_realisation": typical, "summary": summary,
                            "provenance": F.provenance(__file__, T0, inputs=(HERE / "fits.py", HERE / "fits_extrap.py",
                                                                             HERE / "extrap_curves.py", EXTRAP_PRODUCTS))})
    print("checks", checks, "done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
