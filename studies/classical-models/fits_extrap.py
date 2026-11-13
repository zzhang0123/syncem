"""Extrapolation test for the CMB window: fit 1-40 GHz, predict 40-300 GHz.

Same populations (truth on [300, 2.96e6]), geometry, N = 2 basis and noise model as fits.py.
Fit channels: 21 log-spaced centres over 1-40 GHz; prediction channels: 25 over 40-300 GHz
(both with the paper's channel response, h_j = 0.2 nu_j).  The basis keeps cells 4-19 of the
common 19-cell grid (gamma >= 1.28e3); the omitted electrons contribute < 1e-8 of I_j over
0.8-360 GHz (recorded below).  Truth, moments and columns are computed on the full grid and the
retained cells are selected.  E5 zone fractions keep the radio definition (equal I at 1 GHz).

Fits reuse fits.LinearFit (whitened TSVD, UPRE rank, E5 complex pairs) with FIT = fit channels
and HELD = prediction channels.  Spectral laws (spectral_laws.py: s1, s3, logcubic, bmom3, bmom4)
are fitted to I on the same channels; their polarisation follows the PySM convention of a fixed
polarisation fraction and intrinsic angle, P = c <I_law x screen>_channel with one complex c fitted to
Q, U (the same known Faraday screen as the basis).  Electron-spectrum families through the basis
(baseline_models.py: ePL, eLP, eSBPL; all 19 cells, since a family sets every cell's moments) are
fitted to I, Q, U of E2 and E3 with the paper's azimuth factor (separable columns); E5, whose
azimuth depends on energy, has no family fit.

Per ensemble and noise level: N_REAL realisations; for each prediction channel the errors
|I_pred - I|/I, |(|P_pred| - |P|)|/I and |chi_pred - chi| (deg, where P/I > 3%) are summarised by
their median and 84th percentile (the full error arrays are saved). Electron families also record
the in-band chi2 of every realisation. verify_extrap.py checks the summary against the arrays.

Writes fits_extrap_results.json and fits_extrap_products.npz.  Run: python fits_extrap.py
"""
from __future__ import annotations

import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist

import fits as F
import baseline_models as BM
import spectral_laws as SL
from config import PAPER

HERE = Path(__file__).resolve().parent
ROWS = ("E2", "E3", "E5")
NOISE = (0.01, 1e-4)
N_REAL = 500
SEED = 20261112
FIRST_CELL = 3  # zero-based: keep cells 4-19 (gamma >= 1.28e3)
KEEP = np.arange(3 * FIRST_CELL, 3 * PAPER.n_bins)
LAWS = ("s1", "s3", "logcubic", "bmom3", "bmom4")
FAMILIES = ("ePL", "eLP", "eSBPL")
POL_MIN = 0.03
FIT_CFG = replace(PAPER, nu_lo=1e9, nu_hi=4e10, n_channels=21)
PRED_CFG = replace(PAPER, nu_lo=4e10, nu_hi=3e11, n_channels=25)
N_WORKERS = min(20, os.cpu_count() or 1)
T0 = time.time()


def columns(cfg):
    DI, DQ = F.B.derivative_spectra(cfg, F.N_ORDER)
    ne = 3 * cfg.n_bins
    jk = np.concatenate([KEEP, ne + KEEP, 2 * ne + KEEP])
    full_sep = F.B.separable_columns(cfg, DI, DQ)
    return {"sep": full_sep[:, :, KEEP], "joint": F.B.joint_columns(cfg, DI, DQ)[:, :, jk]}, full_sep


def truth(cfg, ens):
    S, a, a_full = {}, {}, {}
    for name in ("E2", "E3"):
        S[name] = ens[name].direct(cfg, n=24)
        a_full[name] = ens[name].moments(cfg, F.N_ORDER, n=24).ravel()
        a[name] = a_full[name][KEEP]
    e5 = ens["E5"]
    fr = e5.fractions(PAPER, 48)
    S["E5"] = e5.direct(cfg, 24, fractions=fr)
    a5, m2 = e5.moments(cfg, F.N_ORDER, 24, fr)
    a["E5"] = np.concatenate([a5.ravel()[KEEP], m2.real.ravel()[KEEP], m2.imag.ravel()[KEEP]])
    return S, a, a_full


def chi2_crit(dof, p):
    return float(chi2_dist.isf(p, dof))


def stokes_errors(pred, true):
    """Per channel: |dI|/I, |d|P||/I, |d chi| [deg] (nan where P/I < POL_MIN)."""
    I = true[..., 0]
    Pt = true[..., 1] + 1j * true[..., 2]
    Pp = pred[..., 1] + 1j * pred[..., 2]
    dI = np.abs(pred[..., 0] - I) / I
    dP = np.abs(np.abs(Pp) - np.abs(Pt)) / I
    dchi = np.degrees(np.abs(0.5 * np.angle(Pp / Pt)))
    dchi = np.where(np.abs(Pt) / I > POL_MIN, dchi, np.nan)
    return np.stack([dI, dP, dchi], axis=-1)


def law_fit(args):
    """One realisation, one law: fit I on the fit grid, P = c I_law; predict the prediction grid."""
    law, d_fit, sig_fit, g_fit, g_pred = args
    try:
        res = SL.fit(law, g_fit, d_fit[:, 0], sig_fit)
    except Exception:  # pragma: no cover - a failed nonlinear fit is recorded, not hidden
        return None
    I_pred = SL.predict(law, res["params"], g_pred)
    w = 1.0 / sig_fit**2
    Pd = d_fit[:, 1] + 1j * d_fit[:, 2]
    shape_fit = screened(law, res["params"], FIT_CFG)
    c = np.sum(w * np.conj(shape_fit) * Pd) / np.sum(w * np.abs(shape_fit) ** 2)
    P = c * screened(law, res["params"], PRED_CFG)
    return np.stack([I_pred, P.real, P.imag], axis=-1)


def screened(law, params, cfg):
    """Channel average of I_law(nu) times the known screen factor <exp(2 i lambda^2 Psi)>."""
    J, T = cfg.chan_nu.shape
    nodes = SL.predict(law, params, SL.mono_grid(cfg.chan_nu.ravel())).reshape(J, T)
    return np.sum(cfg.chan_w[None, :] * nodes * cfg.screen, axis=1)


def family_fit(args):
    """One realisation, one electron family: fit I, Q, U through the full separable columns."""
    fam, rows_fit, d_fit, sig_fit, C_pred = args
    try:
        res = BM.fit_family(fam, rows_fit, d_fit, sig_fit)
    except Exception:  # pragma: no cover
        return None
    pred = F.B.predict(C_pred, res["params"][0] * BM.family_moments(fam, res["params"][1:]).ravel())
    return pred, res["chi2"], res["params"]


def main() -> None:
    ens = F.E.reference_ensembles(PAPER)
    cf, full_f = columns(FIT_CFG)
    cp, full_p = columns(PRED_CFG)
    Sf, af, aff = truth(FIT_CFG, ens)
    Sp, _, _ = truth(PRED_CFG, ens)
    drop = np.setdiff1d(np.arange(3 * PAPER.n_bins), KEEP)
    omitted = {n: max(float(np.max(np.abs(F.B.predict(fc[:, :, drop], aff[n][drop])) / S[n][:, :1])) for fc, S in
                      ((full_f, Sf), (full_p, Sp)))
               for n in ("E2", "E3")}
    nf, npred = FIT_CFG.n_channels, PRED_CFG.n_channels
    F.FIT, F.HELD = np.arange(nf), np.arange(nf, nf + npred)
    g_fit = SL.channel_grid(FIT_CFG.chan_nu, FIT_CFG.chan_w)
    g_pred = SL.channel_grid(PRED_CFG.chan_nu, PRED_CFG.chan_w)
    rng = np.random.default_rng(SEED)
    Z = rng.standard_normal((N_REAL, nf, 3))
    out_npz = {"nu_fit": FIT_CFG.centres, "nu_pred": PRED_CFG.centres}
    summary = {}
    for frac in NOISE:
        F.CFG = replace(PAPER, noise_frac=frac)
        tag = f"{frac:g}"
        for name in ROWS:
            k = name.replace("-", "_")
            C = np.concatenate([(cf["joint"] if name == "E5" else cf["sep"]),
                                (cp["joint"] if name == "E5" else cp["sep"])], axis=0)
            S = np.concatenate([Sf[name], Sp[name]], axis=0)
            nk = KEEP.size
            cplx = (np.arange(nk, 2 * nk), np.arange(2 * nk, 3 * nk)) if name == "E5" else None
            lf = F.LinearFit(name, C, S, af[name], cplx=cplx)
            sig_fit = frac * Sf[name][:, 0]
            dw = lf.Sf[None, :] + Z.reshape(N_REAL, -1)
            sc = lf.scan(dw, np.repeat(lf.Sh[None, :], N_REAL, axis=0))
            R = sc["R_star"]
            preds = np.stack([F.B.predict(cp["joint"] if name == "E5" else cp["sep"],
                                          lf.a_hat(sc["proj"][i], int(R[i]))) for i in range(N_REAL)])
            err = stokes_errors(preds, Sp[name][None])
            rec = {"basis": {"R_star_median": float(np.median(R)), "R_star_p16_p84": np.percentile(R, [16, 84]).tolist()}}
            out_npz[f"{k}_{tag}_basis_err"] = err
            out_npz[f"{k}_S_pred"], out_npz[f"{k}_S_fit"] = Sp[name], Sf[name]
            out_npz[f"{k}_{tag}_basis_pred_seed0"] = preds[0]
            out_npz[f"{k}_{tag}_data_seed0"] = Sf[name] + (frac * Sf[name][:, :1]) * Z[0]
            for law in LAWS:
                jobs = [(law, Sf[name] + (frac * Sf[name][:, :1]) * Z[i], sig_fit, g_fit, g_pred) for i in range(N_REAL)]
                with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                    lp = list(pool.map(law_fit, jobs, chunksize=10))
                ok = [p for p in lp if p is not None]
                le = stokes_errors(np.stack(ok), Sp[name][None])
                out_npz[f"{k}_{tag}_{law}_err"] = le
                out_npz[f"{k}_{tag}_{law}_pred_seed0"] = lp[0] if lp[0] is not None else np.full_like(Sp[name], np.nan)
                rec[law] = {"n_failed": len(lp) - len(ok)}
            if name != "E5":
                rows_fit = full_f.reshape(-1, full_f.shape[-1])
                for fam in FAMILIES:
                    jobs = [(fam, rows_fit, (Sf[name] + (frac * Sf[name][:, :1]) * Z[i]).ravel(),
                             np.repeat(sig_fit, 3), full_p) for i in range(N_REAL)]
                    with ProcessPoolExecutor(max_workers=N_WORKERS) as pool:
                        fp = list(pool.map(family_fit, jobs, chunksize=10))
                    ok = [q for q in fp if q is not None]
                    out_npz[f"{k}_{tag}_{fam}_err"] = stokes_errors(np.stack([q[0] for q in ok]), Sp[name][None])
                    out_npz[f"{k}_{tag}_{fam}_pred_seed0"] = fp[0][0]
                    chi2 = np.array([q[1] for q in ok])
                    dof = 3 * nf - BM.family_n_params(fam)
                    out_npz[f"{k}_{tag}_{fam}_chi2"] = chi2
                    rec[fam] = {"n_failed": len(fp) - len(ok), "dof": int(dof),
                                "chi2_median": float(np.median(chi2)),
                                "frac_rejected_p001": float(np.mean(chi2 > chi2_crit(dof, 1e-3)))}
            keys = ["basis", *LAWS] + ([*FAMILIES] if name != "E5" else [])
            for key in keys:
                e = out_npz[f"{k}_{tag}_{key}_err"]
                rec[key].update({q: {"median": np.nanmedian(e[..., j], axis=0).tolist(),
                                     "p84": np.nanpercentile(e[..., j], 84, axis=0).tolist(),
                                     "max_over_channels_of_median": float(np.nanmax(np.nanmedian(e[..., j], axis=0)))}
                                 for j, q in enumerate(("dI_over_I", "dP_over_I", "dchi_deg"))})
            summary.setdefault(tag, {})[name] = rec
            print(tag, name, "R*", rec["basis"]["R_star_median"],
                  {key: round(100 * rec[key]["dI_over_I"]["max_over_channels_of_median"], 4) for key in keys})
    np.savez(HERE / "fits_extrap_products.npz", **out_npz)
    F.write_json(HERE / "fits_extrap_results.json", {
        "description": __doc__, "rows": list(ROWS), "noise_fracs": list(NOISE), "n_real": N_REAL, "seed": SEED,
        "fit_config": FIT_CFG.describe(), "pred_config": PRED_CFG.describe(),
        "retained_cells": {"first_zero_based": FIRST_CELL, "gamma_lo": float(PAPER.edges[FIRST_CELL]),
                           "n_cells": int(PAPER.n_bins - FIRST_CELL), "n_energy_columns": int(KEEP.size)},
        "omitted_cells_max_over_I": omitted, "summary": summary,
        "provenance": F.provenance(__file__, T0, inputs=(HERE / "fits.py", HERE / "spectral_laws.py",
                                                         HERE / "baseline_models.py"))})
    print("omitted", omitted, "done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
