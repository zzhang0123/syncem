"""Positivity with a soft truncation: Tikhonov fits of the free-moment basis, with and without non-negative cells.

The free-moment fits of fits_mixture_forms.py (whitened TSVD, UPRE rank R*) set the poorly constrained
combinations to zero; the non-negative fits of fits_free_positive.py (NNLS on 17 point masses per cell) only
require each cell distribution to be non-negative and truncate nothing. This study separates the two effects
on the 300 GHz prediction, for E2, E3 and E4-JP, bands 1-40 and 1-100 GHz, 1% and 0.01% noise, with the 48
moment columns of cells 4-19 (N = 2), the shared noise normals and the dense evaluation channels of those
studies. In whitened units (sigma = frac I_j on I, Q and U), with the SVD of fits.svd_signed (singular values below
1e-13 s_max dropped, as in fits.LinearFit), per realisation:
  * "free":       TSVD at the UPRE rank R* (chi2 + 2R), as fits_mixture_forms.free_fits;
  * "free_R+k":   TSVD at rank R* + k (k = 2, 4, 8, 16, capped at the numerical rank);
  * "ridge":      Tikhonov, min |A a - d|^2 + alpha |a|^2 over the moments a (the norm of the TSVD minimum-norm
                  solution, so TSVD and Tikhonov share one SVD basis), alpha minimising
                  UPRE(alpha) = |r|^2 + 2 sum_i s_i^2/(s_i^2 + alpha) on a log grid;
  * "posridge":   the same objective with non-negative cells, a = blockdiag(M) w, w >= 0 (M: moments of the 17
                  Chebyshev-Lobatto point masses of a cell), at alpha = alpha_UPRE ("posridge_0.1" and
                  "posridge_0.01": alpha = 0.1 and 0.01 alpha_UPRE, for sensitivity only); solved as NNLS of
                  [A blockdiag(M); sqrt(alpha) blockdiag(M)] w = [d; 0] with unit-norm columns;
  * "pos":        the same with alpha = 0 (NNLS), which reproduces fits_free_positive.py.
The script stops unless "free" reproduces the free-moment ratios of fits_mixture_forms_products.npz and "pos"
those of fits_free_positive_products.npz to 1e-4 (max |difference| of I_fit relative to the larger of I_PL and
|I_fit|; ill-conditioned directions near the rank cut amplify rounding to ~1e-5).
The JSON summary is indexed [ensemble][method][band][noise], as in fits_free_positive_results.json, with the
moment and cumulant forms (mom2, cum2) copied from there. Every method has the 300 GHz errors, the 16/50/84%
points of the predicted I/I_PL at 300 GHz and the median whitened chi2 over the fitted channels; free, ridge,
posridge and pos also have the per-channel error medians (fits_mixture_forms.error_stats), the polarisation
errors (fits_mixture_forms.pol_stats) and the fraction rejected by the in-band chi2 test at p < 0.001, with dof
= n_data - R* (free), n_data - sum_i s_i^2/(s_i^2 + alpha) (ridge), n_data - sum_i t_i^2/(t_i^2 + alpha) over the
singular values t_i of A restricted to the moments spanned by the active point masses (posridge) and n_data -
active point masses (pos); ridge also has the median alpha / s_max^2 and effective number of parameters.
Writes fits_positive_ridge_results.json and fits_positive_ridge_products.npz (signed I_fit/I on the dense
channels per realisation for free, ridge, posridge, pos, mom2, cum2, with the truth, I_PL and the drawn data
copied from fits_free_positive_products.npz) for fig_extrapolation_positive.py (the default --source ridge).
Run: python fits_positive_ridge.py [n_real]
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
from scipy.optimize import nnls
from scipy.stats import chi2 as chi2_dist

import fits as F  # first: puts lib/ on the path
import extrap_curves as EC
import fits_cmb_bands as CB
import fits_free_positive as P
import fits_mixture_forms as FM
from config import PAPER

HERE = Path(__file__).resolve().parent
ROWS = ("E2", "E3", "E4-JP")
BANDS = {"1-40": 21, "1-100": 26}
NOISE = (1e-2, 1e-4)
R_EXTRA = (2, 4, 8, 16)
ALPHA_MULT = (1.0, 0.1, 0.01)
LOG_ALPHA = np.linspace(-30.0, 2.0, 321)  # alpha / s_max^2
OUT_JSON, OUT_NPZ = HERE / "fits_positive_ridge_results.json", HERE / "fits_positive_ridge_products.npz"
COPY_MODELS = ("mom2", "cum2")
T0 = time.time()


def tsvd(A: np.ndarray, D: np.ndarray, R: int) -> np.ndarray:
    """Moments at rank R for the rows of D (n_real, n_rows); A = U S V^T."""
    U, s, Vt = F.svd_signed(A)
    return ((D @ U[:, :R]) / s[:R]) @ Vt[:R]


def upre_rank(A: np.ndarray, D: np.ndarray) -> np.ndarray:
    """UPRE rank per realisation (chi2 + 2R), with LinearFit's rule of not splitting a degenerate pair."""
    U, s, _ = F.svd_signed(A)
    proj = D @ U
    chi2 = np.sum(D ** 2, axis=1)[:, None] - np.cumsum(proj ** 2, axis=1)
    Rs = np.arange(1, s.size + 1)
    allowed = np.append(s[:-1] > s[1:] * (1 + F.PAIR_RTOL), True)
    return Rs[np.argmin(np.where(allowed[None, :], chi2 + 2 * Rs, np.inf), axis=1)]


def ridge(A: np.ndarray, D: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Tikhonov moments at the UPRE alpha of each realisation; returns (moments, alpha, effective parameters)."""
    U, s, Vt = F.svd_signed(A)
    proj = D @ U
    alphas = s[0] ** 2 * 10.0 ** LOG_ALPHA
    f = s[None, :] ** 2 / (s[None, :] ** 2 + alphas[:, None])  # (n_alpha, k)
    resid = (np.sum(D ** 2, axis=1) - np.sum(proj ** 2, axis=1))[:, None] + (proj ** 2) @ ((1 - f) ** 2).T
    upre = resid + 2 * np.sum(f, axis=1)[None, :]
    best = np.argmin(upre, axis=1)
    fb = f[best]
    return ((fb / s) * proj) @ Vt, alphas[best], np.sum(fb, axis=1)


def pos_ridge(A: np.ndarray, Mblk: np.ndarray, d: np.ndarray, alpha: float) -> np.ndarray:
    """Moments of min |A a - d|^2 + alpha |a|^2 over a = Mblk w, w >= 0."""
    Aw = A @ Mblk
    if alpha > 0:
        Aw = np.vstack([Aw, np.sqrt(alpha) * Mblk])
        d = np.concatenate([d, np.zeros(Mblk.shape[0])])
    norm = np.linalg.norm(Aw, axis=0)
    keep = norm > 1e-300
    w = np.zeros(Aw.shape[1])
    wn, _ = nnls(Aw[:, keep] / norm[keep], d, maxiter=200 * Aw.shape[1])
    w[keep] = wn / norm[keep]
    return Mblk @ w


def active_dof(A: np.ndarray, Mblk: np.ndarray, w: np.ndarray, alpha: float) -> float:
    """Effective number of parameters of a penalised non-negative fit: with Q an orthonormal basis of the moments
    spanned by the active point masses, sum_i t_i^2/(t_i^2 + alpha) over the singular values t_i of A Q."""
    act = w > 0
    if not np.any(act):
        return 0.0
    u, s, _ = np.linalg.svd(Mblk[:, act], full_matrices=False)
    Q = u[:, s > s[0] * 1e-12]
    t = np.linalg.svd(A @ Q, compute_uv=False)
    return float(np.sum(t ** 2 / (t ** 2 + alpha)))


def pos_ridge_w(A: np.ndarray, Mblk: np.ndarray, d: np.ndarray, alpha: float) -> tuple[np.ndarray, int, np.ndarray]:
    """pos_ridge, also returning the number of non-zero point masses and the weights."""
    Aw = A @ Mblk
    if alpha > 0:
        Aw = np.vstack([Aw, np.sqrt(alpha) * Mblk])
        d = np.concatenate([d, np.zeros(Mblk.shape[0])])
    norm = np.linalg.norm(Aw, axis=0)
    keep = norm > 1e-300
    w = np.zeros(Aw.shape[1])
    wn, _ = nnls(Aw[:, keep] / norm[keep], d, maxiter=200 * Aw.shape[1])
    w[keep] = wn / norm[keep]
    return Mblk @ w, int(np.count_nonzero(w)), w


def summarise(a, A, D, cd, S_dense, t_ipl, dof=None, full=False) -> tuple[dict, np.ndarray]:
    """Error statistics and the signed I_fit/I on the dense channels for moments a (n_real, 48).

    full: also the per-channel error medians (fits_mixture_forms.error_stats), the polarisation errors
    (fits_mixture_forms.pol_stats) and, given dof per realisation, the in-band chi2 test at p >= 0.001."""
    pred = np.einsum("jsn,rn->rjs", cd, a)
    ratio = pred[..., 0] / S_dense[None, :, 0]
    stats = FM.error_stats(ratio, t_ipl)
    out = stats if full else {k: v for k, v in stats.items() if k.endswith("300GHz_median_p16_p84")}
    out["pred_over_IPL_300GHz_p16_p50_p84"] = np.percentile(ratio[:, -1] * t_ipl[-1], [16, 50, 84]).tolist()
    chi2 = np.sum((a @ A.T - D) ** 2, axis=1)
    out["chi2_median"] = float(np.median(chi2))
    if full:
        out.update(FM.pol_stats(FM.pol_error(pred, S_dense), t_ipl))
        if dof is not None:
            p = chi2_dist.sf(chi2, np.maximum(dof, 1.0))
            out.update({"dof_median": float(np.median(dof)), "frac_rejected_p001": float(np.mean(p < 1e-3))})
    return out, ratio


def run_group(name, band, n_fit, frac, S_full, S_dense, Z, t_ipl, n_real, stored) -> tuple[dict, dict, dict]:
    """All fits of one ensemble, band and noise level: (summaries by method, signed ratios, checks)."""
    cf = P.columns(name, CB.FULL_CFG, 2)[:n_fit]
    cd = P.columns(name, EC.DENSE_CFG, 2)
    S = S_full[:n_fit]
    sig = frac * np.repeat(S[:, 0], 3)
    A = cf.reshape(-1, cf.shape[-1]) / sig[:, None]
    D = np.stack([(S + (frac * S[:, :1]) * Z[i, :n_fit]).ravel() for i in range(n_real)]) / sig[None, :]
    n_data = A.shape[0]
    M = P.moment_map(name, 2)
    Mblk = np.kron(np.eye(cf.shape[-1] // M.shape[0]), M)
    tag = FM.tag_of(frac)
    res, ratios = {}, {}
    R = upre_rank(A, D)
    rank = F.svd_signed(A)[1].size  # numerical rank, as in fits.LinearFit
    a_free = np.stack([tsvd(A, D[i:i + 1], int(R[i]))[0] for i in range(n_real)])
    res["free"], ratios["free"] = summarise(a_free, A, D, cd, S_dense, t_ipl, n_data - R, full=True)
    res["free"]["R_star_median"] = float(np.median(R))
    for k in R_EXTRA:
        a_k = np.stack([tsvd(A, D[i:i + 1], int(min(R[i] + k, rank)))[0] for i in range(n_real)])
        res[f"free_R+{k}"], _ = summarise(a_k, A, D, cd, S_dense, t_ipl)
    a_r, alpha, neff = ridge(A, D)
    res["ridge"], ratios["ridge"] = summarise(a_r, A, D, cd, S_dense, t_ipl, n_data - neff, full=True)
    smax2 = F.svd_signed(A)[1][0] ** 2
    res["ridge"].update({"alpha_over_smax2_median": float(np.median(alpha / smax2)),
                         "n_eff_median": float(np.median(neff))})
    for m in ALPHA_MULT:
        key = "posridge" if m == 1.0 else f"posridge_{m:g}"
        if m == 1.0:
            fits = [pos_ridge_w(A, Mblk, D[i], alpha[i]) for i in range(n_real)]
            a_p = np.stack([f[0] for f in fits])
            df = np.array([active_dof(A, Mblk, f[2], alpha[i]) for i, f in enumerate(fits)])
            res[key], ratios[key] = summarise(a_p, A, D, cd, S_dense, t_ipl, n_data - df, full=True)
            res[key].update({"df_active_median": float(np.median(df)),
                             "n_active_median": float(np.median([f[1] for f in fits]))})
        else:
            a_p = np.stack([pos_ridge(A, Mblk, D[i], m * alpha[i]) for i in range(n_real)])
            res[key], _ = summarise(a_p, A, D, cd, S_dense, t_ipl)
    fits0 = [pos_ridge_w(A, Mblk, D[i], 0.0) for i in range(n_real)]  # dof n - active masses, as fits_free_positive
    a_0, n_act = np.stack([f[0] for f in fits0]), np.array([f[1] for f in fits0])
    res["pos"], ratios["pos"] = summarise(a_0, A, D, cd, S_dense, t_ipl, n_data - n_act, full=True)
    checks = {}
    for key, z in (("free", stored["forms"]), ("pos", stored["pos"])):
        ref = z[f"{name}_{key}_{band}_{tag}_ratio"][:n_real]
        scale = np.maximum(1.0, np.abs(ref) * t_ipl[None, :])  # I_PL, or the prediction where it is larger
        checks[key] = float(np.max(np.abs(ratios[key] - ref) * t_ipl[None, :] / scale))
    return res, ratios, checks


def main() -> None:
    n_real = int(sys.argv[1]) if len(sys.argv) > 1 else FM.N_REAL
    ens = F.E.reference_ensembles(PAPER)
    Z = CB.shared_noise()
    stored = {"forms": np.load(HERE / "fits_mixture_forms_products.npz"),
              "pos": np.load(HERE / "fits_free_positive_products.npz")}
    js_pos = json.loads((HERE / "fits_free_positive_results.json").read_text())
    zp = stored["pos"]
    out_npz = {k: zp[k] for k in ("centres_dense", "centres_fit")}
    nu = EC.DENSE_CFG.centres / 1e9
    summary, checks = {}, {}
    for name in ROWS:
        S_full, S_dense = FM.truth(ens, name, CB.FULL_CFG), FM.truth(ens, name, EC.DENSE_CFG)
        t_ipl = S_dense[:, 0] / FM.reference_law(nu, S_dense[:, 0])(nu)
        for key in (f"truth_{name}", f"ipl_{name}"):
            out_npz[key] = zp[key]
        for band, n_fit in BANDS.items():
            for frac in NOISE:
                tag = FM.tag_of(frac)
                res, ratios, chk = run_group(name, band, n_fit, frac, S_full, S_dense, Z, t_ipl, n_real, stored)
                checks.setdefault(name, {}).setdefault(band, {})[tag] = chk
                for method, rec in res.items():
                    summary.setdefault(name, {}).setdefault(method, {}).setdefault(band, {})[tag] = rec
                for method, r in ratios.items():
                    out_npz[f"{name}_{method}_{band}_{tag}_ratio"] = r
                for m in COPY_MODELS:  # the electron families, as in fits_free_positive (the figure's third column)
                    out_npz[f"{name}_{m}_{band}_{tag}_ratio"] = zp[f"{name}_{m}_{band}_{tag}_ratio"]
                    summary[name].setdefault(m, {}).setdefault(band, {})[tag] = js_pos["summary"][name][m][band][tag]
                for k in (f"data_{name}_{band}_{tag}", f"sigma_{name}_{band}_{tag}"):
                    out_npz[k] = zp[k]
                key = "dI_over_IPL_300GHz_median_p16_p84" if name == "E4-JP" else "dI_over_I_300GHz_median_p16_p84"
                print(name, band, tag, {m: round(100 * v[key][0], 4) for m, v in res.items()}, "checks", chk,
                      flush=True)
    worst = max(v for n in checks.values() for b in n.values() for t in b.values() for v in t.values())
    if worst > 1e-4:
        raise SystemExit(f"free/pos do not reproduce the stored studies: max relative |dI| = {worst:.3g}")
    np.savez(OUT_NPZ, **out_npz)
    F.write_json(OUT_JSON, {"description": __doc__, "rows": list(ROWS), "bands": BANDS, "noise_fracs": list(NOISE),
                            "n_real": n_real, "drawn_realisation": FM.DRAWN, "r_extra": list(R_EXTRA),
                            "alpha_mult": list(ALPHA_MULT),
                            "log10_alpha_over_smax2_grid": [float(LOG_ALPHA[0]), float(LOG_ALPHA[-1]),
                                                            int(LOG_ALPHA.size)],
                            "checks_max_rel_dI": checks, "summary": summary,
                            "provenance": F.provenance(__file__, T0, inputs=(HERE / "fits_mixture_forms.py",
                                                                             HERE / "fits_free_positive.py",
                                                                             HERE / "fits_cmb_bands.py"))})
    print("done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
