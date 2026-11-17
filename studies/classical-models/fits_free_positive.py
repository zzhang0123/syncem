"""Free moments with positivity: the extrapolation test of fits_mixture_forms.py with non-negative cell measures.

The moments of the kept energy cells (4-19, as the free-moment fits) are restricted to those of a
non-negative electron distribution. Within each cell the distribution is represented by non-negative point
masses at fixed positions, so the fitted moments lie in (an inner approximation of) the cone of realisable
moments and satisfy the Hausdorff conditions by construction; no electron-spectrum family is assumed and no
relation links the cells:
  * E2, E3, E4-JP: 17 Chebyshev-Lobatto atoms x_q in [-1, 1] per cell (fits.Z_ATOMS, as fits.atom_columns);
    an atom contributes x_q^r to the moment of order r, r = 0..N.
  * E6: atoms on the grid of 17 x_gamma by 9 x_B Chebyshev-Lobatto points per cell, x_B = (B - 5 muG)/(1 muG)
    in [-1, 1] (the field range of E6), contributing x_gamma^r x_B^s to the joint moment (r, s) of
    ensemble_e6.monomials(N).
The atom columns are the order-N basis columns times these moment maps. The fit is NNLS (scipy.optimize.nnls)
on the whitened I, Q, U rows of the fitted channels, with the columns scaled to unit norm; no rank is chosen.
Channels, bands (1-40, 1-100, 1-145 GHz of fits_cmb_bands.FULL_CFG), noise normals (fits_cmb_bands.shared_noise,
1% and 0.01% of I_j), true spectra and the dense evaluation are those of fits_mixture_forms.py, with the
N = 2 basis ("pos"; see Caveats for why not N = 4).
For the figure (fig_extrapolation_positive.py --source positive) the products also carry, unchanged, the
free-moment, mom2 and cum2 results of fits_mixture_forms (N = 2) for the bands 1-40 and 1-100 GHz, with the
true spectra, I_PL and the drawn data at 1% and 0.01% noise; the JSON summary repeats their entries.
Beyond the error statistics of fits_mixture_forms.error_stats, the JSON gives for each ensemble, band and noise
level: the polarisation errors |P_fit - P|/I and /I_PL at 100 and 300 GHz (P = Q + iU, fits_mixture_forms.pol_stats);
the 16, 50 and 84% points of the predicted I/I_PL at 300 GHz with the true value; the fraction of fits whose
in-band chi2 has p < 0.001 (and p < 0.05) with n_data minus the number of active atoms as degrees of freedom; and,
for E2, E3 and E4-JP over 1-40 and 1-100 GHz at 1% noise (first N_BOUND realisations), the largest 300 GHz
intensity, in units of I_PL, of non-negative fits whose chi2 exceeds the NNLS minimum by at most 1
(max_I300_over_IPL_dchi2_1; Lagrangian maximisation with L-BFGS-B and bisection in the multiplier).
Checks: the implied moments of the first 20 realisations of each N = 2 separable group satisfy the degree-2
Hausdorff conditions; this holds by construction for non-negative point masses in [-1, 1] and only guards
against rounding.
Caveats (from an independent check): positivity applies to the cell measure. At N = 2 every atom column has
positive intensity on 1-300 GHz, so the model intensity is non-negative. At N = 4 the atoms are order-4
Taylor polynomials of the cell kernel, and some (12 of 272 for E4-JP, 244 of 2448 for E6) emit negatively
near the cut-off, so non-negative weights would not mean non-negative emission; the study uses N = 2 only. The near-zero NNLS prediction
above the cut-off of E4-JP and E6 is a point estimate on the w = 0 boundary; with chi2 allowed to rise by 1,
positive fits of E4-JP at 1% noise reach a median of about 2e-5 I_PL at 300 GHz over 1-100 GHz and about 0.2 I_PL
over 1-40 GHz (max_I300_over_IPL_dchi2_1). It also relies
on noise proportional to I_j and on the N = 2 misfit of the cut-off.

Reads fits_mixture_forms_{products.npz,results.json}. Writes fits_free_positive_{results.json,products.npz}.
Run: python fits_free_positive.py
"""
from __future__ import annotations

import json
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.stats import chi2 as chi2_dist

import fits as F  # first: puts lib/ on the path
import ensemble_e6 as E6
import extrap_curves as EC
import fits_cmb_bands as CB
import fits_extrap as X
import fits_mixture_forms as FM
from config import PAPER

HERE = Path(__file__).resolve().parent
ROWS = FM.ROWS
ORDERS = {2: "pos"}  # N = 4 Taylor atoms can emit negatively near a cut-off (see Caveats)
BANDS = {"1-40": 21, "1-100": 26, "1-145": 28}
NOISE = (1e-2, 1e-4)
N_REAL = FM.N_REAL
XB_ATOMS = np.cos(np.pi * np.arange(9) / 8)  # Chebyshev-Lobatto on [-1, 1] for x_B (E6)
COPY_MODELS = ("free", "mom2", "cum2")
COPY_BANDS = ("1-40", "1-100")
BOUND_ROWS, BOUND_BANDS, BOUND_FRAC, N_BOUND = ("E2", "E3", "E4-JP"), ("1-40", "1-100"), 1e-2, 100
FORMS_NPZ, FORMS_JSON = HERE / "fits_mixture_forms_products.npz", HERE / "fits_mixture_forms_results.json"
OUT_NPZ, OUT_JSON = HERE / "fits_free_positive_products.npz", HERE / "fits_free_positive_results.json"
T0 = time.time()


def columns(name: str, cfg, N: int) -> np.ndarray:
    """Order-N columns of cells 4-19 on the channels of cfg: the joint energy-field columns for E6, the separable
    energy columns otherwise (fits_mixture_forms.free_columns, for any N)."""
    if name == "E6":
        return E6.joint_columns(cfg, N)[:, :, E6.keep_joint(X.FIRST_CELL, N)]
    DI, DQ = F.B.derivative_spectra(cfg, N)
    return F.B.separable_columns(cfg, DI, DQ)[:, :, np.arange((N + 1) * X.FIRST_CELL, (N + 1) * cfg.n_bins)]


def moment_map(name: str, N: int) -> np.ndarray:
    """(n_moments_per_cell, n_atoms_per_cell): moments of each unit point mass of one cell."""
    if name == "E6":
        xg, xb = np.meshgrid(F.Z_ATOMS, XB_ATOMS, indexing="ij")
        return np.stack([(xg ** r * xb ** s).ravel() for r, s in E6.monomials(N)])
    return np.stack([F.Z_ATOMS ** r for r in range(N + 1)])


def atom_columns(cols: np.ndarray, M: np.ndarray) -> np.ndarray:
    """(J, 3, n_cells * n_mom) moment columns -> (J, 3, n_cells * n_atoms) atom columns."""
    J, st, n = cols.shape
    c = cols.reshape(J, st, n // M.shape[0], M.shape[0])
    return np.einsum("jscm,ma->jsca", c, M).reshape(J, st, -1)


_A = _D = _SCALE = _C = None  # worker state: unit-norm whitened rows, dense I, Q, U rows, scales, 300 GHz I row


def _init(A, D, scale, C):
    global _A, _D, _SCALE, _C
    _A, _D, _SCALE, _C = A, D, scale, C


def _solve(d):
    """NNLS of one realisation: ((J, 3) prediction on the dense channels, sum of squared residuals in units of
    I_j, active atoms, weights)."""
    w, rn = nnls(_A, d, maxiter=50 * _A.shape[1])
    w = w / _SCALE
    return (_D @ w).reshape(-1, 3), float(rn ** 2), int(np.count_nonzero(w)), w


def _qp(lam, x0, d):
    """min |A w - d|^2 - lam C.w over w >= 0 (L-BFGS-B); returns (w, |A w - d|^2)."""
    def f(w):
        r = _A @ w - d
        return float(r @ r - lam * (_C @ w)), 2.0 * (_A.T @ r) - lam * _C
    res = minimize(f, x0, jac=True, method="L-BFGS-B", bounds=[(0.0, None)] * len(x0),
                   options=dict(maxiter=20000, ftol=1e-16, gtol=1e-14))
    r = _A @ res.x - d
    return res.x, float(r @ r)


def _bound(args):
    """Largest 300 GHz intensity (in units of I_PL there) over non-negative fits with chi2 <= chi2_min + 1.

    Lagrangian: w(lam) maximises C.w - |A w - d|^2 / lam; chi2(lam) rises with lam, which is bracketed and bisected
    in log space until chi2 - chi2_min = 1 (in units of the noise, i.e. frac^2 in the I_j units of A and d)."""
    d, frac = args
    w0, rn = nnls(_A, d, maxiter=50 * _A.shape[1])
    s_min, target = float(rn ** 2), frac ** 2
    best, x, lam_lo, lam_hi = float(_C @ w0), w0, 0.0, 1e-8
    for _ in range(80):  # bracket
        w, s = _qp(lam_hi, x, d)
        if s - s_min > target:
            break
        best, x, lam_lo, lam_hi = max(best, float(_C @ w)), w, lam_hi, lam_hi * 4.0
    for _ in range(40):  # bisect
        lam = np.sqrt(lam_hi * max(lam_lo, lam_hi * 1e-12))
        w, s = _qp(lam, x, d)
        if s - s_min <= target:
            best, x, lam_lo = max(best, float(_C @ w)), w, lam
        else:
            lam_hi = lam
    return best


def fit_group(name, N, band, n_fit, S_full, S_dense, Z_full, ipl300):
    """All realisations at both noise levels for one ensemble, basis order and band; S_dense (J, 3) on the dense
    channels, ipl300 = I_PL at 300 GHz."""
    M = moment_map(name, N)
    cf = atom_columns(columns(name, CB.FULL_CFG, N)[:n_fit], M)
    cd = atom_columns(columns(name, EC.DENSE_CFG, N), M)
    S = S_full[:n_fit]
    rows = cf.reshape(-1, cf.shape[-1]) / np.repeat(S[:, 0], 3)[:, None]  # whitened up to the factor frac
    scale = np.linalg.norm(rows, axis=0)
    keep = scale > 1e-12 * scale.max()
    A = np.where(keep[None, :], rows / np.where(keep, scale, 1.0)[None, :], 0.0)
    out = {}
    scale_k = np.where(keep, scale, np.inf)
    c300 = cd[-1, 0, :] / scale_k / ipl300  # 300 GHz intensity per unit normalised weight, in units of I_PL
    with ProcessPoolExecutor(max_workers=X.N_WORKERS, initializer=_init,
                             initargs=(A, cd.reshape(-1, cd.shape[-1]), scale_k, c300)) as pool:
        for frac in NOISE:
            jobs = [((S + (frac * S[:, :1]) * Z_full[i, :n_fit]).ravel() / np.repeat(S[:, 0], 3)) for i in range(N_REAL)]
            res = list(pool.map(_solve, jobs, chunksize=10))
            pred = np.stack([r[0] for r in res])
            rec = {"ratio": pred[..., 0] / S_dense[None, :, 0], "dP": FM.pol_error(pred, S_dense),
                   "chi2": np.array([r[1] for r in res]) / frac ** 2,
                   "n_active": np.array([r[2] for r in res]), "w_sample": np.stack([r[3] for r in res[:20]])}
            if name in BOUND_ROWS and band in BOUND_BANDS and frac == BOUND_FRAC:
                rec["bound300"] = np.array(list(pool.map(_bound, [(jobs[i], frac) for i in range(N_BOUND)], chunksize=2)))
            out[f"{frac:g}"] = rec
    return out, M, int(cf.shape[-1])


def hausdorff_check(w_sample, M, n_cells) -> dict:
    """Degree-2 Hausdorff conditions of the implied cell moments (N = 2 separable fits)."""
    a = np.einsum("ma,ica->icm", M, w_sample.reshape(len(w_sample), n_cells, -1))
    scale = float(np.max(np.abs(a[..., 0])))
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    tol = 1e-12 * scale
    viol = (a0 < -tol) | (np.abs(a1) > a0 + tol) | (a2 < -tol) | (a2 > a0 + tol) | (a1 ** 2 > a0 * a2 + tol * scale)
    return {"n_fits": int(len(w_sample)), "cells_violating": int(viol.sum())}


def main() -> None:
    ens = F.E.reference_ensembles(PAPER)
    Z_full = CB.shared_noise()
    S_full = {n: FM.truth(ens, n, CB.FULL_CFG) for n in ROWS}
    zf, jf = np.load(FORMS_NPZ), json.loads(FORMS_JSON.read_text())
    out_npz = {k: zf[k] for k in ("centres_dense", "centres_fit")}
    summary, checks, sizes, pol_frac = {}, {}, {}, {}
    for name in ROWS:
        S_dense = FM.truth(ens, name, EC.DENSE_CFG)
        pol_frac[name] = float(np.abs(S_dense[-1, 1] + 1j * S_dense[-1, 2]) / S_dense[-1, 0])
        if not np.array_equal(S_dense[:, 0], zf[f"truth_{name}"]):
            raise SystemExit(f"{name}: dense truth differs from fits_mixture_forms_products.npz")
        t_ipl = zf[f"truth_{name}"] / zf[f"ipl_{name}"]
        for key in (f"truth_{name}", f"ipl_{name}"):
            out_npz[key] = zf[key]
        for band in COPY_BANDS:
            n_fit = BANDS[band]
            for frac in NOISE:  # the drawn data at both noise levels (the 1% data equal those of the forms study)
                S = S_full[name][:n_fit]
                out_npz[f"data_{name}_{band}_{frac:g}"] = S + (frac * S[:, :1]) * Z_full[FM.DRAWN, :n_fit]
                out_npz[f"sigma_{name}_{band}_{frac:g}"] = frac * S[:, 0]
            # the forms study draws its 1-40 GHz data on fits_extrap.FIT_CFG, whose channels equal the first 21
            # of the nested grid to 1e-14
            if not np.allclose(out_npz[f"data_{name}_{band}_0.01"], zf[f"data_{name}_{band}_0.01"], rtol=1e-12, atol=0):
                raise SystemExit(f"{name} {band}: the 1% data differ from fits_mixture_forms_products.npz")
            for m in COPY_MODELS:
                for tag in ("0.01", "0.0001"):
                    out_npz[f"{name}_{m}_{band}_{tag}_ratio"] = zf[f"{name}_{m}_{band}_{tag}_ratio"]
                    rec = dict(jf["summary"][name][m][band][tag])
                    p300 = zf[f"{name}_{m}_{band}_{tag}_ratio"][:, -1] * t_ipl[-1]
                    rec.update({"pred_over_IPL_300GHz_p16_p50_p84": np.percentile(p300, [16, 50, 84]).tolist(),
                                "truth_over_IPL_300GHz": float(t_ipl[-1])})
                    summary.setdefault(name, {}).setdefault(m, {}).setdefault(band, {})[tag] = rec
        for N, model in ORDERS.items():
            for band, n_fit in BANDS.items():
                res, M, n_atoms = fit_group(name, N, band, n_fit, S_full[name], S_dense, Z_full,
                                            float(zf[f"ipl_{name}"][-1]))
                sizes[f"{name}_{model}"] = n_atoms
                for tag, r in res.items():
                    out_npz[f"{name}_{model}_{band}_{tag}_ratio"] = r["ratio"]
                    out_npz[f"{name}_{model}_{band}_{tag}_chi2"] = r["chi2"]
                    out_npz[f"{name}_{model}_{band}_{tag}_n_active"] = r["n_active"]
                    s = FM.error_stats(r["ratio"], t_ipl)
                    s.update(FM.pol_stats(r["dP"], t_ipl))
                    pred = r["ratio"] * t_ipl[None, :]
                    dof = 3 * n_fit - r["n_active"]
                    s.update({"pred_over_IPL_300GHz_p16_p50_p84": np.percentile(pred[:, -1], [16, 50, 84]).tolist(),
                              "truth_over_IPL_300GHz": float(t_ipl[-1]),
                              "frac_rejected_p001_dof_n_minus_active": float(np.mean(
                                  r["chi2"] > chi2_dist.isf(1e-3, np.maximum(dof, 1)))),
                              "frac_p_below_005_dof_n_minus_active": float(np.mean(
                                  r["chi2"] > chi2_dist.isf(0.05, np.maximum(dof, 1))))})
                    if "bound300" in r:
                        b = r["bound300"]
                        s["max_I300_over_IPL_dchi2_1"] = {"n": int(b.size), "median": float(np.median(b)),
                                                          "p16": float(np.percentile(b, 16)),
                                                          "p84": float(np.percentile(b, 84))}
                    s.update({"chi2_median": float(np.median(r["chi2"])), "n_data": 3 * n_fit,
                              "n_active_median": float(np.median(r["n_active"])),
                              "min_I_pred_over_IPL_median_above_band": float(np.median(np.min(
                                  (r["ratio"] * t_ipl[None, :])[:, EC.DENSE_CFG.centres > CB.FULL_CFG.centres[n_fit - 1]],
                                  axis=1)))})
                    summary.setdefault(name, {}).setdefault(model, {}).setdefault(band, {})[tag] = s
                    if N == 2 and name != "E6":
                        checks[f"hausdorff_{name}_{band}_{tag}"] = hausdorff_check(r["w_sample"], M,
                                                                                   r["w_sample"].shape[1] // M.shape[1])
                    e = 100 * s["dI_over_I_300GHz_median_p16_p84"][0]
                    print(name, model, band, tag, f"300 GHz median {e:.4g}%", "chi2 median",
                          round(s["chi2_median"], 1), "/", 3 * n_fit, "active", s["n_active_median"], flush=True)
    bad = {k: v for k, v in checks.items() if v["cells_violating"]}
    if bad:
        raise SystemExit(f"Hausdorff violations in positive fits: {bad}")
    np.savez(OUT_NPZ, **out_npz)
    F.write_json(OUT_JSON, {"description": __doc__, "rows": list(ROWS), "orders": {str(k): v for k, v in ORDERS.items()},
                            "bands": BANDS, "noise_fracs": list(NOISE), "n_real": N_REAL, "seed": X.SEED,
                            "drawn_realisation": FM.DRAWN, "n_atoms": sizes,
                            "truth_P_over_I_300GHz": pol_frac, "checks": checks, "summary": summary,
                            "provenance": F.provenance(__file__, T0, inputs=(HERE / "fits_mixture_forms.py",
                                                                             HERE / "ensemble_e6.py", HERE / "fits.py",
                                                                             FORMS_NPZ, FORMS_JSON))})
    print("checks ok", len(checks), "done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
