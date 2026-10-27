"""Fits for the reference ensembles E1-E5, plus E1M (E2a was not built).

Model d = C a + eps + n on the 21 fitted channels (63 data, I/Q/U), noise sd 0.01 I_j^direct.
  * Noise-whitened SVD Sigma^-1/2 C = U D V^T; rank R* = argmin_R [chi2_fit(R) + 2R] (UPRE,
    Mallows C_p); withheld channels (60 data) only validate.
  * beta_hat = D_R^-1 U_R^T Sigma^-1/2 d, Cov(beta_hat) = D_R^-2, beta_true = V_R^T a_true.
  * Representation bias of mode i in its own sigma: U_i^T Sigma^-1/2 eps, eps = S_direct - C a_true.
  * Coverage of the noise-only 1-sigma intervals of beta over noise realisations (fixed R and
    per-realisation UPRE R), Picard outputs versus R, NNLS witness on 17 Chebyshev-Lobatto
    atoms per bin, Hausdorff conditions of the TSVD minimum-norm moments.
  * E5: joint per-bin complex azimuth basis (108 columns) versus the separable basis with one
    free complex azimuth factor c (bilinear; variable projection: for each R the rank-R TSVD
    chi2 is minimised over c; UPRE counts R + 2 parameters), and the de-rotated angle chi(nu).

Continuum (beaming-limit) kernels only; N = 2 basis (36 columns).  Seeds: fitted rows use the
paper draw (Config.draw_noise(21), seed 20260928); withheld rows of that realisation use seed
20260929; the noise realisations use one standard-normal array from seed 20261027, shared by
all ensembles.  Writes fits_results.json and fits_products.npz.  Run: python fits.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json
import sys
import time
from multiprocessing import get_context
from pathlib import Path

import numpy as np
from scipy.optimize import minimize, nnls
from scipy.stats import chi2 as chi2dist
from scipy.stats import ncx2, norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import basis as B  # noqa: E402
import ensembles as E  # noqa: E402
from config import PAPER, PAPER_MAXWELL  # noqa: E402
from provenance import provenance, write_json  # noqa: E402

T0 = time.time()
CFG = PAPER
FIT, HELD = CFG.fit_idx, CFG.held_idx
N_ORDER = 2
N_REAL = 4000  # linear fits (coverage, R* distribution)
N_REAL_NNLS = 200
N_REAL_SEP = 400  # E5 free-factor comparator (nonlinear in c)
SEED_HELD = CFG.seed + 1
SEED_REAL = 20261027
R_PICARD = tuple(range(3, 15))
R_PICARD_E5 = tuple(range(3, 31))
N_ATOMS = 17
Z_ATOMS = np.cos(np.pi * np.arange(N_ATOMS) / (N_ATOMS - 1))  # Chebyshev-Lobatto on [-1, 1]
N_AZ_ATOMS = 16  # azimuth atoms for the E5 extension: e^{2 i phi_k}, phi_k = pi k / 16
COVER = float(norm.cdf(1.0) - norm.cdf(-1.0))
POL_MIN = 0.03  # chi(nu) is summarised where P/I > 3%
CHUNK = 200
N_WORKERS = 24
TIMINGS = {}
FORWARD_NPZ = HERE / "forward_spectra.npz"
REGRESSION_JSON = HERE / "regression_results.json"


def tic(label, t):
    TIMINGS[label] = TIMINGS.get(label, 0.0) + time.time() - t


# ----------------------------------------------------------------------------- helpers
def stack_rows(X, rows):
    """(J, n_st, ...) -> (len(rows) * n_st, ...) in (channel, Stokes) order."""
    X = np.asarray(X)[rows]
    return X.reshape(X.shape[0] * X.shape[1], *X.shape[2:])


def svd_signed(A, rtol=1e-13):
    """Thin SVD with numerically zero directions dropped and signs fixed (largest |V| entry > 0)."""
    u, s, vt = np.linalg.svd(A, full_matrices=False)
    keep = s > s[0] * rtol
    u, s, vt = u[:, keep], s[keep], vt[keep]
    sgn = np.sign(vt[np.arange(s.size), np.argmax(np.abs(vt), axis=1)])
    return u * sgn, s, vt * sgn[:, None]


def seeded_noise():
    """Standard normals for the seeded realisation: fitted rows (paper draw) and withheld rows."""
    return CFG.draw_noise(len(FIT)).ravel(), CFG.draw_noise(len(HELD), seed=SEED_HELD).ravel()


def realisation_noise(n):
    rng = np.random.default_rng(SEED_REAL)
    Z = rng.standard_normal((n, CFG.n_channels, 3))
    return Z[:, FIT].reshape(n, -1), Z[:, HELD].reshape(n, -1), Z


def pct(x, q=(16, 50, 84)):
    return [float(v) for v in np.percentile(np.asarray(x, float), q)]


def max_rel(pred, S):
    """max |pred - S| / I_j over channels and components, with its location."""
    rel = np.abs(pred - S) / S[:, 0:1]
    j, k = np.unravel_index(np.argmax(rel), rel.shape)
    return {"value": float(rel[j, k]), "channel": int(j), "stokes": "IQU"[k],
            "withheld": bool(j in set(HELD.tolist())),
            "fitted_max": float(rel[FIT].max()), "withheld_max": float(rel[HELD].max())}


def wrap_half(d):
    """Wrap a polarisation-angle difference to (-pi/2, pi/2]."""
    return 0.5 * np.angle(np.exp(2j * d))


PAIR_RTOL = 1e-9


def canonical_pairs(A, u, s, vt, ix, iy):
    """Canonical basis inside exactly degenerate singular pairs of a complex-structured block.

    For the E5 joint basis the P rows act on M2 = x + i y only, through the real form of a
    complex matrix, so its singular values come in exact pairs whose 2-planes are invariant
    under J: (x, y) -> (-y, x).  Inside each pair choose r1 with complex form z = x + i y whose
    largest |z| entry is real positive, and r2 = J r1.  Returns (u, vt, n_pairs, max_offplane).
    """
    u, vt = u.copy(), vt.copy()
    i, n_pairs, worst = 0, 0, 0.0
    while i < s.size - 1:
        if abs(s[i] - s[i + 1]) > PAIR_RTOL * s[i]:
            i += 1
            continue
        z = vt[i, ix] + 1j * vt[i, iy]
        z = z * np.exp(-1j * np.angle(z[np.argmax(np.abs(z))]))
        r1 = np.zeros(vt.shape[1])
        r1[ix], r1[iy] = z.real, z.imag
        r2 = np.zeros_like(r1)
        r2[ix], r2[iy] = -z.imag, z.real
        plane = vt[i:i + 2]
        worst = max(worst, float(np.linalg.norm(r1 - plane.T @ (plane @ r1))),
                    float(np.linalg.norm(r2 - plane.T @ (plane @ r2))))
        vt[i], vt[i + 1] = r1, r2
        u[:, i], u[:, i + 1] = A @ r1 / s[i], A @ r2 / s[i + 1]
        n_pairs += 1
        i += 2
    return u, vt, n_pairs, worst


# ----------------------------------------------------------------------------- linear TSVD
class LinearFit:
    """Whitened TSVD / UPRE machinery for one ensemble and one linear basis C (J, 3, n)."""

    def __init__(self, name, C, S, a_true, cplx=None):
        self.name, self.C, self.S = name, C, S
        self.a_true = np.asarray(a_true, float).ravel()
        self.n = C.shape[-1]
        self.sigma = CFG.noise_sigma(S)
        self.sf, self.sh = stack_rows(self.sigma, FIT), stack_rows(self.sigma, HELD)
        self.Af = stack_rows(C, FIT) / self.sf[:, None]
        self.Ah = stack_rows(C, HELD) / self.sh[:, None]
        self.u, self.s, self.vt = svd_signed(self.Af)
        self.k = self.s.size
        self.pairs = {"n_pairs": 0, "max_offplane": 0.0}
        if cplx is not None:
            self.u, self.vt, npair, off = canonical_pairs(self.Af, self.u, self.s, self.vt, *cplx)
            self.pairs = {"n_pairs": npair, "max_offplane": off}
        # truncations that do not split an exactly degenerate pair
        gap = np.append(self.s[:-1] > self.s[1:] * (1 + PAIR_RTOL), True)
        self.allowed = gap
        self.Sf, self.Sh = stack_rows(S, FIT) / self.sf, stack_rows(S, HELD) / self.sh
        self.eps = S - B.predict(C, self.a_true)
        self.bias = self.u.T @ (stack_rows(self.eps, FIT) / self.sf)
        self.beta_true = self.vt @ self.a_true
        self.Hh = (self.Ah @ self.vt.T) / self.s
        self.Hall = (C.reshape(-1, self.n) @ self.vt.T) / self.s
        self.Sall, self.Iall = S.reshape(-1), np.repeat(S[:, 0], 3)

    def scan(self, dw, dhw):
        """All-R outputs for whitened data batches dw (n, 63), dhw (n, 60); index R-1."""
        dw, dhw = np.atleast_2d(dw), np.atleast_2d(dhw)
        proj = dw @ self.u
        Rs = np.arange(1, self.k + 1)
        chi2_fit = np.sum(dw**2, axis=1)[:, None] - np.cumsum(proj**2, axis=1)
        held, err = [], []
        for s0 in range(0, len(dw), CHUNK):
            p = proj[s0:s0 + CHUNK]
            ph = np.cumsum(self.Hh[None] * p[:, None, :], axis=2)
            held.append(np.sum((ph - dhw[s0:s0 + CHUNK, :, None]) ** 2, axis=1))
            pa = np.cumsum(self.Hall[None] * p[:, None, :], axis=2)
            err.append(np.max(np.abs(pa - self.Sall[None, :, None]) / self.Iall[None, :, None], axis=1))
        upre = chi2_fit + 2 * Rs
        R_star = Rs[np.argmin(np.where(self.allowed[None, :], upre, np.inf), axis=1)]
        return {"proj": proj, "chi2_fit": chi2_fit, "upre": upre, "chi2_held": np.concatenate(held),
                "err": np.concatenate(err), "R_star": R_star}

    def predictive_held(self, proj, dhw, R):
        """r^T (I + H_R H_R^T)^-1 r on withheld rows, r = prediction - data; chi2_60 if eps = 0 (fixed R)."""
        H = self.Hh[:, :R]
        r = np.atleast_2d(proj)[:, :R] @ H.T - np.atleast_2d(dhw)
        M = np.linalg.inv(np.eye(H.shape[0]) + H @ H.T)
        return np.einsum("ni,ij,nj->n", r, M, r)

    def predictive_held_at(self, proj, dhw, Rn):
        out = np.zeros(len(Rn))
        for R in np.unique(Rn):
            m = Rn == R
            out[m] = self.predictive_held(proj[m], dhw[m], int(R))
        return out

    def eps_diagnostics(self):
        """|eps_w|^2 on fitted rows and its part outside the column span (visible to chi2)."""
        ew = stack_rows(self.eps, FIT) / self.sf
        inside = self.u.T @ ew
        return {"eps_w_norm2_fit": float(ew @ ew), "eps_w_outside_span_chi2": float(ew @ ew - inside @ inside),
                "eps_w_inside_span_norm2": float(inside @ inside)}

    def a_hat(self, proj, R):
        return (proj[:R] / self.s[:R]) @ self.vt[:R]

    def expected_chi2_held(self, R):
        """E[chi2_held] for eps = 0: 60 + tr(H_R H_R^T) (prediction variance on withheld rows)."""
        return float(len(self.Sh) + np.sum(self.Hh[:, :R] ** 2))


def hausdorff(a36, scale):
    """Degree-2 Hausdorff conditions on [-1, 1] per bin for scaled moments (..., 36).

    Returns (n_bins_violating (...), per-condition boolean arrays (..., 12), worst violation / scale).
    """
    a = np.asarray(a36, float).reshape(*np.shape(a36)[:-1], CFG.n_bins, 3)
    a0, a1, a2 = a[..., 0], a[..., 1], a[..., 2]
    tol = 1e-12 * scale
    cond = {"a0>=0": -a0, "|a1|<=a0": np.abs(a1) - a0, "a2>=0": -a2, "a2<=a0": a2 - a0,
            "a1^2<=a0*a2": (a1**2 - a0 * a2) / max(scale, 1e-300)}
    viol = {k: v > tol for k, v in cond.items()}
    anyv = np.any(np.stack(list(viol.values())), axis=0)
    worst = np.max(np.stack([np.where(v > tol, v, 0.0) for v in cond.values()]), axis=0) / scale
    return anyv.sum(axis=-1), viol, worst.max(axis=-1)


def hausdorff_detail(a36, scale):
    n, viol, worst = hausdorff(a36, scale)
    a = np.asarray(a36).reshape(CFG.n_bins, 3)
    bins = [{"bin": b, "a": a[b].tolist(), "violated": [k for k, v in viol.items() if v[b]]}
            for b in range(CFG.n_bins) if any(v[b] for v in viol.values())]
    return {"n_bins_violating": int(n), "worst_violation_over_max_a0_true": float(worst),
            "per_condition_n_bins": {k: int(v.sum()) for k, v in viol.items()}, "bins": bins}


def azimuth_bound(a36, m2_36, scale):
    """Necessary conditions |M2_br| <= a_b0 (r = 0, 1, 2) and |M2_b2| <= a_b2; bins violating."""
    a = np.asarray(a36).reshape(-1, CFG.n_bins, 3)
    m = np.asarray(m2_36).reshape(-1, CFG.n_bins, 3)
    tol = 1e-12 * scale
    v = np.any(np.abs(m) > a[..., 0:1] + tol, axis=-1) | (np.abs(m[..., 2]) > a[..., 2] + tol)
    return v.sum(axis=-1)


# ----------------------------------------------------------------------------- reports
def seeded_report(lf, zf, zh, picard_R):
    """Seeded realisation: UPRE curve, beta_hat, bias, chi2, max error, Hausdorff; Picard rows."""
    sc = lf.scan(lf.Sf + zf, lf.Sh + zh)
    nf = lf.scan(lf.Sf, lf.Sh)
    R = int(sc["R_star"][0])
    proj = sc["proj"][0]
    a_hat = lf.a_hat(proj, R)
    pred = B.predict(lf.C, a_hat)
    dev = proj[:R] - lf.s[:R] * lf.beta_true[:R]
    scale = float(np.max(np.abs(lf.a_true[: 3 * CFG.n_bins].reshape(-1, 3)[:, 0])))
    pred_h = stack_rows(pred, HELD) / lf.sh
    out = {
        "R_star": R, "rank": lf.k, "n_columns": lf.n,
        "upre_curve": sc["upre"][0, :min(lf.k, 40)].tolist(),
        "chi2_fit": float(sc["chi2_fit"][0, R - 1]), "dof_fit": int(len(lf.Sf) - R),
        "chi2_held": float(sc["chi2_held"][0, R - 1]),
        "chi2_held_vs_direct": float(np.sum((pred_h - lf.Sh) ** 2)),
        "expected_chi2_held_if_eps_0": lf.expected_chi2_held(R), "n_held": int(len(lf.Sh)),
        "chi2_held_predictive": float(lf.predictive_held(proj, lf.Sh + zh, R)[0]),
        "chi2_held_predictive_note": "r^T (I + H_R H_R^T)^-1 r; chi2 with 60 dof if eps = 0 at fixed R",
        "eps_diagnostics": lf.eps_diagnostics(), "degenerate_pairs": lf.pairs,
        "max_fit_minus_direct_over_I": max_rel(pred, lf.S),
        "noise_free_at_R_star": {"chi2_fit": float(nf["chi2_fit"][0, R - 1]),
                                 "max_fit_minus_direct_over_I": float(nf["err"][0, R - 1])},
        "singular_values": lf.s.tolist(),
        "sigma_beta": (1.0 / lf.s[:R]).tolist(),
        "beta_hat": (proj[:R] / lf.s[:R]).tolist(), "beta_true": lf.beta_true[:R].tolist(),
        "beta_hat_minus_true_over_sigma": dev.tolist(),
        "rep_bias_over_sigma": lf.bias[:R].tolist(),
        "max_abs_rep_bias_over_sigma": float(np.max(np.abs(lf.bias[:R]))),
        "max_finite_response_discrepancy_over_I": float(np.max(np.abs(lf.eps) / lf.S[:, 0:1])),
        "hausdorff_tsvd": hausdorff_detail(a_hat[:36], scale),
        "hausdorff_tsvd_noise_free": hausdorff_detail(lf.a_hat(nf["proj"][0], R)[:36], scale),
        "hausdorff_true_moments_n_bins_violating": int(hausdorff(lf.a_true[:36], scale)[0]),
    }
    rows = []
    for Rp in picard_R:
        if Rp > lf.k:
            break
        if not lf.allowed[Rp - 1]:
            continue  # would split an exactly degenerate pair (E5 joint): convention-dependent
        ah = lf.a_hat(proj, Rp)
        rows.append({"R": Rp, "s_R": float(lf.s[Rp - 1]), "chi2_fit": float(sc["chi2_fit"][0, Rp - 1]),
                     "chi2_held": float(sc["chi2_held"][0, Rp - 1]), "upre": float(sc["upre"][0, Rp - 1]),
                     "max_err_over_I": float(sc["err"][0, Rp - 1]),
                     "noise_free_max_err_over_I": float(nf["err"][0, Rp - 1]),
                     "noise_free_chi2_fit": float(nf["chi2_fit"][0, Rp - 1]),
                     "max_abs_rep_bias_over_sigma": float(np.max(np.abs(lf.bias[:Rp]))),
                     "chi2_held_predictive": float(lf.predictive_held(proj, lf.Sh + zh, Rp)[0]),
                     "hausdorff_bins_violating": int(hausdorff(ah[:36], scale)[0])})
    nmode = min(lf.k, max(picard_R) + 6)
    modes = {"s": lf.s[:nmode].tolist(),
             "abs_coeff_noise_free_|u^T S_w|": np.abs(nf["proj"][0, :nmode]).tolist(),
             "abs_coeff_seeded_|u^T d_w|": np.abs(proj[:nmode]).tolist(),
             "rep_bias_over_sigma": lf.bias[:nmode].tolist(),
             "beta_true": lf.beta_true[:nmode].tolist()}
    return out, {"per_R": rows, "per_mode": modes}, {"a_hat": a_hat, "proj": proj, "pred": pred}


def realisation_report(lf, zf, zh, R0, picard_rows):
    """Noise realisations: R* distribution, chi2/error at R*, coverage, Hausdorff frequency."""
    sc = lf.scan(lf.Sf[None] + zf, lf.Sh[None] + zh)
    n = len(zf)
    idx = np.arange(n)
    Rn = sc["R_star"]
    dev = sc["proj"] - lf.s * lf.beta_true
    inside = np.abs(dev) <= 1.0
    Rmax = int(Rn.max())
    kept = np.arange(lf.k)[None, :] < Rn[:, None]
    post = [float(inside[kept[:, i], i].mean()) for i in range(Rmax)]
    q = np.cumsum(dev**2, axis=1)
    scale = float(np.max(np.abs(lf.a_true[:36].reshape(-1, 3)[:, 0])))
    a_hat_n = ((sc["proj"] * kept) / lf.s) @ lf.vt
    hz = hausdorff(a_hat_n[:, :36], scale)[0]
    vals, cnt = np.unique(Rn, return_counts=True)
    sc["pred_held_star"] = lf.predictive_held_at(sc["proj"], zh + lf.Sh, Rn)
    out = {
        "n_realisations": n, "seed": SEED_REAL,
        "R_star_p16_p50_p84": pct(Rn), "R_star_histogram": {int(v): int(c) for v, c in zip(vals, cnt)},
        "chi2_fit_at_R_star_p16_p50_p84": pct(sc["chi2_fit"][idx, Rn - 1]),
        "chi2_held_at_R_star_p16_p50_p84": pct(sc["chi2_held"][idx, Rn - 1]),
        "chi2_held_predictive_at_R_star_p16_p50_p84": pct(sc["pred_held_star"]),
        "fraction_chi2_held_predictive_above_chi2_60_99pct": float(np.mean(
            sc["pred_held_star"] > chi2dist.ppf(0.99, len(lf.Sh)))),
        "max_err_over_I_at_R_star_p16_p50_p84": pct(sc["err"][idx, Rn - 1]),
        "coverage_fixed_R": {
            "R": R0, "nominal": COVER, "binomial_sd_at_nominal": float(np.sqrt(COVER * (1 - COVER) / n)),
            "per_mode_empirical": inside[:, :R0].mean(axis=0).tolist(),
            "per_mode_analytic": (norm.cdf(1 - lf.bias[:R0]) - norm.cdf(-1 - lf.bias[:R0])).tolist(),
            "joint_ellipsoid_empirical": float(np.mean(q[:, R0 - 1] <= chi2dist.ppf(COVER, R0))),
            "joint_ellipsoid_analytic": float(ncx2.cdf(chi2dist.ppf(COVER, R0), R0,
                                                       float(np.sum(lf.bias[:R0] ** 2)))),
        },
        "coverage_post_selection": {
            "per_mode_empirical_when_retained": post,
            "fraction_retaining_mode": kept[:, :Rmax].mean(axis=0).tolist(),
            "joint_ellipsoid_empirical": float(np.mean(q[idx, Rn - 1] <= chi2dist.ppf(COVER, Rn))),
        },
        "hausdorff_tsvd_fraction_with_violation": float(np.mean(hz > 0)),
        "hausdorff_tsvd_bins_violating_p16_p50_p84": pct(hz),
    }
    for row in picard_rows:
        Rp = row["R"]
        row["realisations_median_chi2_fit"] = float(np.median(sc["chi2_fit"][:, Rp - 1]))
        row["realisations_median_chi2_held"] = float(np.median(sc["chi2_held"][:, Rp - 1]))
        row["realisations_median_chi2_held_predictive"] = float(np.median(
            lf.predictive_held(sc["proj"], zh + lf.Sh, Rp)))
        row["realisations_median_max_err_over_I"] = float(np.median(sc["err"][:, Rp - 1]))
        row["joint_coverage_empirical"] = float(np.mean(q[:, Rp - 1] <= chi2dist.ppf(COVER, Rp)))
        row["min_per_mode_coverage_analytic"] = float(np.min(norm.cdf(1 - lf.bias[:Rp])
                                                             - norm.cdf(-1 - lf.bias[:Rp])))
        row["fraction_realisations_choosing_R"] = float(np.mean(Rn == Rp))
    return out, sc, a_hat_n


# ----------------------------------------------------------------------------- NNLS witness
def atom_columns(C36):
    """(J, n_st, 36) moment columns -> (J, n_st, 12 * 17) point-mass atom columns."""
    c = C36.reshape(*C36.shape[:-1], CFG.n_bins, N_ORDER + 1)
    pw = np.stack([Z_ATOMS**r for r in range(N_ORDER + 1)])
    return np.einsum("jsbr,rq->jsbq", c, pw).reshape(*C36.shape[:-1], -1)


def azimuth_atom_columns(wI, wQ):
    """E5 extension: atoms (b, x_q, phi_k) with I += w x^r, M2 += w e^{2 i phi_k} x^r."""
    th = 2.0 * np.pi * np.arange(N_AZ_ATOMS) / N_AZ_ATOMS
    pw = np.stack([Z_ATOMS**r for r in range(N_ORDER + 1)])
    aI = np.einsum("jbr,rq->jbq", wI.reshape(-1, CFG.n_bins, 3), pw)
    aQ = np.einsum("jbr,rq->jbq", wQ.reshape(-1, CFG.n_bins, 3), pw)
    Ia = np.broadcast_to(aI[..., None], aI.shape + (N_AZ_ATOMS,))
    P = aQ[..., None] * np.exp(1j * th)[None, None, None, :]
    cols = np.stack([Ia, P.real, P.imag], axis=1)  # (J, 3, nb, nq, nk)
    return cols.reshape(cols.shape[0], 3, -1)


class AtomWitness:
    """NNLS on nonnegative atom weights; cols (J, n_st, n_atoms), S and sigma (J, n_st)."""

    def __init__(self, cols, S, sigma):
        self.cols, self.S = cols, S
        self.sf, self.sh = stack_rows(sigma, FIT), stack_rows(sigma, HELD)
        self.Af = stack_rows(cols, FIT) / self.sf[:, None]
        self.Sh = stack_rows(S, HELD) / self.sh

    def run(self, dw, dhw):
        w, rn = nnls(self.Af, dw, maxiter=200 * self.Af.shape[1])
        pred = np.einsum("jsk,k->js", self.cols, w)
        rel = np.abs(pred - self.S) / self.S[:, 0:1]
        return {"chi2_fit": float(rn**2),
                "chi2_held": float(np.sum((stack_rows(pred, HELD) / self.sh - dhw) ** 2)),
                "max_err_over_I": float(rel.max()), "n_active_atoms": int(np.sum(w > 0)), "w": w}


def implied_column(w, n_az=1):
    """Total electron column sum_b w_b a_b0 implied by atom weights."""
    wb = w.reshape(CFG.n_bins, N_ATOMS * n_az).sum(axis=1)
    return float(np.sum(CFG.w_b * wb))


def nnls_report(wit, Sf, Sh, zf, zh, zfR, zhR, n_az=1, tsvd_chi2=None, full_ls_chi2=None):
    t = time.time()
    seed = wit.run(Sf + zf, Sh + zh)
    nf = wit.run(Sf, Sh)
    real = [wit.run(Sf + a, Sh + b) for a, b in zip(zfR, zhR)]
    out = {"n_atoms": int(wit.Af.shape[1]), "n_data_fit": int(wit.Af.shape[0]),
           "seeded": {k: v for k, v in seed.items() if k != "w"},
           "seeded_implied_total_column": implied_column(seed["w"], n_az),
           "seeded_implied_a_b0": seed["w"].reshape(CFG.n_bins, -1).sum(axis=1).tolist(),
           "noise_free_implied_total_column": implied_column(nf["w"], n_az),
           "noise_free_implied_a_b0": nf["w"].reshape(CFG.n_bins, -1).sum(axis=1).tolist(),
           "noise_free": {k: v for k, v in nf.items() if k != "w"},
           "realisations": {"n": len(real), "chi2_fit_p16_p50_p84": pct([r["chi2_fit"] for r in real]),
                            "chi2_held_p16_p50_p84": pct([r["chi2_held"] for r in real]),
                            "max_err_over_I_p16_p50_p84": pct([r["max_err_over_I"] for r in real])},
           "seconds": time.time() - t}
    if tsvd_chi2 is not None:
        out["seeded_tsvd_chi2_fit_at_R_star"] = tsvd_chi2
    if full_ls_chi2 is not None:
        out["seeded_full_rank_ls_chi2_fit"] = full_ls_chi2
    return out


# ----------------------------------------------------------------------------- E5 comparator
class FreeFactorFit:
    """Separable P = c * (w_b D^Q) a with one free complex factor c shared by all bins."""

    def __init__(self, wI, wQ, S):
        self.wI, self.wQ, self.S = wI, wQ, S
        self.n = wI.shape[1]
        self.sigma = CFG.noise_sigma(S)
        self.sf, self.sh = stack_rows(self.sigma, FIT), stack_rows(self.sigma, HELD)
        self.Sf, self.Sh = stack_rows(S, FIT) / self.sf, stack_rows(S, HELD) / self.sh

    def columns(self, c):
        P = c * self.wQ
        return np.stack([self.wI, P.real, P.imag], axis=1)  # (J, 3, n)

    def chi2_curve(self, c, dw):
        u, s, vt = svd_signed(stack_rows(self.columns(c), FIT) / self.sf[:, None])
        proj = u.T @ dw
        return dw @ dw - np.cumsum(proj**2)

    def solve(self, c, R, dw):
        """Rank-R TSVD at fixed c: moments, prediction (J, 3), withheld response H_R (60, R)."""
        C = self.columns(c)
        u, s, vt = svd_signed(stack_rows(C, FIT) / self.sf[:, None])
        a = ((u.T @ dw)[:R] / s[:R]) @ vt[:R]
        H = (stack_rows(C, HELD) / self.sh[:, None]) @ vt[:R].T / s[:R]
        return a, B.predict(C, a), H

    def profile(self, dw, n_phase=24, amps=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0)):
        """For every R: c_R = argmin_c chi2(R; c) (grid start + Nelder-Mead); returns c (k,), chi2 (k,)."""
        grid = [a * np.exp(1j * ph) for a in amps for ph in np.linspace(-np.pi, np.pi, n_phase, endpoint=False)]
        curves = np.array([self.chi2_curve(c, dw) for c in grid])
        k = curves.shape[1]
        c_best, f_best = np.zeros(k, complex), np.zeros(k)
        for R in range(1, k + 1):
            starts = [grid[int(np.argmin(curves[:, R - 1]))]]
            if R > 1:
                starts.append(c_best[R - 2])
            best = None
            for c0 in starts:
                x0 = np.array([c0.real, c0.imag])
                res = minimize(lambda v: self.chi2_curve(v[0] + 1j * v[1], dw)[R - 1], x0, method="Nelder-Mead",
                               options={"initial_simplex": [x0, x0 + [0.03, 0], x0 + [0, 0.03]],
                                        "xatol": 1e-7, "fatol": 1e-7, "maxiter": 600})
                if best is None or res.fun < best.fun:
                    best = res
            c_best[R - 1], f_best[R - 1] = best.x[0] + 1j * best.x[1], best.fun
        return c_best, f_best

    def fit(self, dw, dhw):
        c_R, chi2_R = self.profile(dw)
        Rs = np.arange(1, len(chi2_R) + 1)
        upre = chi2_R + 2 * (Rs + 2)
        R = int(Rs[np.argmin(upre)])
        a, pred, H = self.solve(c_R[R - 1], R, dw)
        r = stack_rows(pred, HELD) / self.sh - dhw
        pred_held = float(r @ np.linalg.solve(np.eye(len(r)) + H @ H.T, r))
        return {"R_star": R, "c": c_R[R - 1], "chi2_fit": float(chi2_R[R - 1]),
                "chi2_held": float(r @ r), "chi2_held_predictive_fixed_c": pred_held,
                "max_err_over_I": float(np.max(np.abs(pred - self.S) / self.S[:, 0:1])),
                "a": a, "pred": pred, "c_R": c_R, "chi2_R": chi2_R, "upre": upre}


_WORKER = {}


def _init_worker(wI, wQ, S):
    os.environ["OPENBLAS_NUM_THREADS"] = "1"
    _WORKER["ff"] = FreeFactorFit(wI, wQ, S)


def _worker_fit(args):
    zf, zh = args
    ff = _WORKER["ff"]
    r = ff.fit(ff.Sf + zf, ff.Sh + zh)
    keys = ("R_star", "c", "chi2_fit", "chi2_held", "chi2_held_predictive_fixed_c", "max_err_over_I")
    return {k: r[k] for k in keys} | {"chi2_R": r["chi2_R"]}


# ----------------------------------------------------------------------------- E5 angles
def chi_sigma(C, a_hat, vt, s, R):
    """Linearised 1-sigma of chi_j = arg(P_j)/2 from Cov(a_hat) = V_R D_R^-2 V_R^T."""
    pred = B.predict(C, a_hat)
    P = pred[:, 1] + 1j * pred[:, 2]
    g = 0.5 * (P.real[:, None] * C[:, 2, :] - P.imag[:, None] * C[:, 1, :]) / np.abs(P[:, None]) ** 2
    G = (g @ vt[:R].T) / s[:R]
    return np.sqrt(np.sum(G**2, axis=1))


def e5_angles(lf, seeded_a, seeded_R, sc_real, sep_pred, sep_pred_nf, pol):
    """chi(nu) for truth, joint fit (band from covariance and from realisations), separable fit."""
    chi_true = E.derotated_angle(CFG, lf.S)
    pred = B.predict(lf.C, seeded_a)
    chi_joint = E.derotated_angle(CFG, pred)
    sig = chi_sigma(lf.C, seeded_a, lf.vt, lf.s, seeded_R)
    a_nf = lf.a_hat(lf.Sf @ lf.u, seeded_R)
    pred_nf = B.predict(lf.C, a_nf)
    sig_nf = chi_sigma(lf.C, a_nf, lf.vt, lf.s, seeded_R)
    # realisation spread at the same R (circular, relative to the noise-free prediction)
    kept = sc_real["proj"][:, :seeded_R] / lf.s[:seeded_R]
    a_n = kept @ lf.vt[:seeded_R]
    Pc = lf.C[:, 1, :] + 1j * lf.C[:, 2, :]
    P_n = a_n @ Pc.T
    P0 = pred_nf[:, 1] + 1j * pred_nf[:, 2]
    spread = np.std(0.5 * np.angle(P_n / P0[None, :]), axis=0)
    sel = pol > POL_MIN
    chi_sep = E.derotated_angle(CFG, sep_pred)
    chi_sep_nf = E.derotated_angle(CFG, sep_pred_nf)
    d_joint = wrap_half(chi_joint - chi_true)
    d_nf = wrap_half(E.derotated_angle(CFG, pred_nf) - chi_true)
    d_sep = wrap_half(chi_sep - chi_true)
    swing = lambda c: float(np.degrees(np.ptp(np.unwrap(2 * c[sel]) / 2)))  # noqa: E731
    return {
        "nu_GHz": (CFG.centres / 1e9).tolist(), "P_over_I_true": pol.tolist(),
        "selection": f"P/I > {POL_MIN}", "n_channels_selected": int(sel.sum()),
        "chi_true_deg": np.degrees(chi_true).tolist(),
        "chi_joint_seeded_deg": np.degrees(chi_joint).tolist(),
        "chi_joint_sigma_deg_linearised_seeded": np.degrees(sig).tolist(),
        "chi_joint_sigma_deg_linearised_noise_free": np.degrees(sig_nf).tolist(),
        "chi_joint_sigma_deg_realisations_fixed_R": np.degrees(spread).tolist(),
        "chi_separable_free_factor_seeded_deg": np.degrees(chi_sep).tolist(),
        "chi_separable_free_factor_noise_free_deg": np.degrees(chi_sep_nf).tolist(),
        "swing_deg_selected": {"true": swing(chi_true), "joint_seeded": swing(chi_joint),
                               "separable_seeded": swing(chi_sep), "separable_noise_free": swing(chi_sep_nf)},
        "max_abs_joint_minus_true_deg_selected": float(np.degrees(np.max(np.abs(d_joint[sel])))),
        "max_abs_joint_minus_true_over_sigma_selected": float(np.max(np.abs(d_joint[sel]) / sig[sel])),
        "fraction_selected_true_within_joint_1sigma": float(np.mean(np.abs(d_joint[sel]) <= sig[sel])),
        "max_abs_noise_free_joint_bias_over_sigma_selected": float(np.max(np.abs(d_nf[sel]) / sig_nf[sel])),
        "max_abs_separable_minus_true_deg_selected": float(np.degrees(np.max(np.abs(d_sep[sel])))),
        "max_ratio_linearised_over_realisation_sigma_selected": float(np.max(sig_nf[sel] / spread[sel])),
        "min_ratio_linearised_over_realisation_sigma_selected": float(np.min(sig_nf[sel] / spread[sel])),
    }


# ----------------------------------------------------------------------------- driver
LINEAR = ("E1", "E2", "E3", "E4-JP", "E4-KP", "E5", "E1M")
NPZ_KEY = {"E4-JP": "E4_JP", "E4-KP": "E4_KP"}


def build_inputs():
    """Basis columns, direct spectra and true moments (E4-KP: effective diagonal; E5: (a, M2))."""
    t = time.time()
    DI, DQ = B.derivative_spectra(CFG, N_ORDER)
    cols = {"sep": B.separable_columns(CFG, DI, DQ), "joint": B.joint_columns(CFG, DI, DQ),
            "wI": np.real(B.q_columns(CFG, DI)), "wQ": B.q_columns(CFG, DQ)}
    tic("basis_N2", t)
    t = time.time()
    ens = E.reference_ensembles(CFG)
    S, a = {}, {}
    for name in ("E1", "E2", "E3", "E4-JP"):
        S[name], a[name] = ens[name].direct(CFG, n=24), ens[name].moments(CFG, N_ORDER, n=24).ravel()
    kp = ens["E4-KP"]
    S["E4-KP"], a["E4-KP"] = kp.direct(CFG, n=24, n_eta=16), kp.effective_moments(CFG, N_ORDER, 24, 16).ravel()
    e5 = ens["E5"]
    fr = e5.fractions(CFG, 48)
    S["E5"] = e5.direct(CFG, 24, fractions=fr)
    a5, m2 = e5.moments(CFG, N_ORDER, 24, fr)
    a["E5"] = np.concatenate([a5.ravel(), m2.real.ravel(), m2.imag.ravel()])
    a["E5-sep"] = a5.ravel()
    S["E1M"], a["E1M"] = ens["E1M"].direct(PAPER_MAXWELL, n=24), ens["E1M"].moments(CFG, N_ORDER, 24).ravel()
    tic("direct_spectra_and_moments", t)
    ref = np.load(FORWARD_NPZ)
    check = {k: float(np.max(np.abs(S[k] - ref[NPZ_KEY.get(k, k)]) / S[k][:, 0:1])) for k in LINEAR}
    return cols, S, a, check, ens


def basis_for(name, cols):
    return cols["joint"] if name == "E5" else cols["sep"]


def nnls_inputs(name, cols, lf, zf, zh, zfR, zhR):
    """Witness inputs: separable atoms (I, Q, U) or, for E5, I-only atoms."""
    if name != "E5":
        return AtomWitness(atom_columns(cols["sep"]), lf.S, lf.sigma), lf.Sf, lf.Sh, zf, zh, zfR, zhR
    sel = lambda z, m: z.reshape(-1, m, 3)[..., 0].reshape(z.shape[:-1] + (m,))  # noqa: E731
    wit = AtomWitness(atom_columns(cols["sep"][:, 0:1, :]), lf.S[:, 0:1], lf.sigma[:, 0:1])
    Sf = lf.S[FIT, 0] / lf.sigma[FIT, 0]
    Sh = lf.S[HELD, 0] / lf.sigma[HELD, 0]
    return wit, Sf, Sh, sel(zf, len(FIT)), sel(zh, len(HELD)), sel(zfR, len(FIT)), sel(zhR, len(HELD))


def run_linear(name, cols, S, a, noise):
    zf, zh, zfR, zhR = noise
    t = time.time()
    cplx = (np.arange(36, 72), np.arange(72, 108)) if name == "E5" else None
    lf = LinearFit(name, basis_for(name, cols), S[name], a[name], cplx=cplx)
    picard_R = R_PICARD_E5 if name == "E5" else R_PICARD
    seeded, picard, prod = seeded_report(lf, zf, zh, picard_R)
    real, sc, _ = realisation_report(lf, zfR, zhR, seeded["R_star"], picard["per_R"])
    tic(f"tsvd_{name}", t)
    t = time.time()
    wit, Sf, Sh, nzf, nzh, nzfR, nzhR = nnls_inputs(name, cols, lf, zf, zh, zfR, zhR)
    full_ls = None if name == "E5" else float(lf.scan(lf.Sf + zf, lf.Sh + zh)["chi2_fit"][0, -1])
    nn = nnls_report(wit, Sf, Sh, nzf, nzh, nzfR[:N_REAL_NNLS], nzhR[:N_REAL_NNLS],
                     tsvd_chi2=None if name == "E5" else seeded["chi2_fit"], full_ls_chi2=full_ls)
    nn["moments_shared_with"] = "I only (E5: P has its own M2 moments)" if name == "E5" else "I, Q, U"
    tic(f"nnls_{name}", t)
    res = {"seeded": seeded, "realisations": real, "picard": picard, "nnls_witness": nn}
    if name == "E5":
        res.update(e5_joint_extras(lf, cols, prod, seeded["chi2_fit"], zf, zh, zfR, zhR))
    return lf, res, sc, prod


def e5_joint_extras(lf, cols, prod, seeded_chi2, zf, zh, zfR, zhR):
    """E5: azimuth-atom NNLS (I, Q, U) and the necessary bound |M2| <= a on the TSVD moments."""
    t = time.time()
    wit = AtomWitness(azimuth_atom_columns(cols["wI"], cols["wQ"]), lf.S, lf.sigma)
    nn = nnls_report(wit, lf.Sf, lf.Sh, zf, zh, zfR[:N_REAL_NNLS], zhR[:N_REAL_NNLS], n_az=N_AZ_ATOMS,
                     tsvd_chi2=seeded_chi2)
    tic("nnls_E5_azimuth_atoms", t)
    scale = float(np.max(lf.a_true[:36].reshape(-1, 3)[:, 0]))
    ah = prod["a_hat"]
    m2 = ah[36:72] + 1j * ah[72:]
    m2_true = lf.a_true[36:72] + 1j * lf.a_true[72:]
    return {"nnls_witness_azimuth_atoms": nn,
            "azimuth_bound_tsvd_bins_violating": int(azimuth_bound(ah[:36], m2, scale)[0]),
            "azimuth_bound_true_bins_violating": int(azimuth_bound(lf.a_true[:36], m2_true, scale)[0])}


def run_free_factor(cols, S, zf, zh, zfR, zhR):
    """E5 comparator: seeded and noise-free fits, grid verification, realisations (pool)."""
    t = time.time()
    ff = FreeFactorFit(cols["wI"], cols["wQ"], S["E5"])
    seed = ff.fit(ff.Sf + zf, ff.Sh + zh)
    nf = ff.fit(ff.Sf, ff.Sh)
    R = seed["R_star"]
    dense = [a * np.exp(1j * ph) for a in np.linspace(0.02, 1.2, 60)
             for ph in np.linspace(-np.pi, np.pi, 144, endpoint=False)]
    dvals = np.array([ff.chi2_curve(c, ff.Sf + zf)[R - 1] for c in dense])
    dense_min = float(dvals.min())
    restarts = []
    for j in np.argsort(dvals)[:5]:
        x0 = np.array([dense[j].real, dense[j].imag])
        restarts.append(minimize(lambda v: ff.chi2_curve(v[0] + 1j * v[1], ff.Sf + zf)[R - 1], x0,
                                 method="Nelder-Mead", options={"initial_simplex": [x0, x0 + [0.01, 0], x0 + [0, 0.01]],
                                                                "xatol": 1e-8, "fatol": 1e-8, "maxiter": 2000}).fun)
    tic("free_factor_seeded", t)
    t = time.time()
    ctx = get_context("spawn")
    with ctx.Pool(N_WORKERS, initializer=_init_worker, initargs=(cols["wI"], cols["wQ"], S["E5"])) as pool:
        real = pool.map(_worker_fit, list(zip(zfR[:N_REAL_SEP], zhR[:N_REAL_SEP])), chunksize=4)
    tic("free_factor_realisations", t)
    picard = [{"R": r, "chi2_fit": float(seed["chi2_R"][r - 1]), "upre_R_plus_2": float(seed["upre"][r - 1]),
               "c_abs": float(abs(seed["c_R"][r - 1])), "c_arg_rad": float(np.angle(seed["c_R"][r - 1]))}
              for r in R_PICARD_E5 if r <= len(seed["chi2_R"])]
    out = {
        "model": "I = wI a, P = c (w_b D^Q) a; a (36) shared with I, c complex free; UPRE counts R + 2",
        "seeded": {"R_star": R, "c": [seed["c"].real, seed["c"].imag], "c_abs": float(abs(seed["c"])),
                   "c_arg_rad": float(np.angle(seed["c"])), "chi2_fit": seed["chi2_fit"],
                   "dof_fit": int(len(ff.Sf) - R - 2), "chi2_held": seed["chi2_held"],
                   "max_fit_minus_direct_over_I": max_rel(seed["pred"], ff.S),
                   "chi2_held_predictive_fixed_c": seed["chi2_held_predictive_fixed_c"],
                   "full_rank_chi2_fit": float(seed["chi2_R"][-1]), "numerical_rank": int(len(seed["chi2_R"])),
                   "check_dense_grid_min_minus_adopted_at_R_star": dense_min - seed["chi2_fit"],
                   "check_restarts_from_5_best_grid_points_min_minus_adopted": float(min(restarts) - seed["chi2_fit"]),
                   "upre_curve": seed["upre"].tolist()},
        "noise_free": {"R_star": nf["R_star"], "c": [nf["c"].real, nf["c"].imag], "chi2_fit": nf["chi2_fit"],
                       "max_fit_minus_direct_over_I": float(nf["max_err_over_I"]),
                       "full_rank_chi2_fit": float(nf["chi2_R"][-1])},
        "picard": picard,
        "realisations": {"n": len(real), "seed": SEED_REAL,
                         "R_star_p16_p50_p84": pct([r["R_star"] for r in real]),
                         "chi2_fit_p16_p50_p84": pct([r["chi2_fit"] for r in real]),
                         "chi2_held_p16_p50_p84": pct([r["chi2_held"] for r in real]),
                         "chi2_held_predictive_fixed_c_p16_p50_p84": pct([r["chi2_held_predictive_fixed_c"]
                                                                          for r in real]),
                         "max_err_over_I_p16_p50_p84": pct([r["max_err_over_I"] for r in real]),
                         "full_rank_chi2_fit_p16_p50_p84": pct([r["chi2_R"][-1] for r in real]),
                         "c_abs_p16_p50_p84": pct([abs(r["c"]) for r in real])},
    }
    return out, seed, nf, real


def compare_e5(sc_joint, real_sep, lf_joint):
    """Realisation-by-realisation joint versus free-factor comparison (same noise)."""
    n = len(real_sep)
    idx = np.arange(n)
    Rj = sc_joint["R_star"][:n]
    up_j = sc_joint["upre"][idx, Rj - 1]
    up_s = np.array([r["chi2_fit"] + 2 * (r["R_star"] + 2) for r in real_sep])
    ch_j = sc_joint["pred_held_star"][:n]
    ch_s = np.array([r["chi2_held_predictive_fixed_c"] for r in real_sep])
    full_s = np.array([r["chi2_R"][-1] for r in real_sep])
    dof_full = np.array([len(lf_joint.Sf) - len(r["chi2_R"]) - 2 for r in real_sep])
    crit = float(chi2dist.ppf(0.99, len(lf_joint.Sh)))
    return {"n": n, "chi2_held_statistic": "predictive (I + H H^T)^-1 form; separable at fixed fitted c",
            "upre_separable_minus_joint_p16_p50_p84": pct(up_s - up_j),
            "fraction_upre_prefers_joint": float(np.mean(up_j < up_s)),
            "chi2_held_separable_minus_joint_p16_p50_p84": pct(ch_s - ch_j),
            "fraction_chi2_held_lower_for_joint": float(np.mean(ch_j < ch_s)),
            "chi2_60_99pct": crit,
            "fraction_separable_chi2_held_above_99pct": float(np.mean(ch_s > crit)),
            "fraction_joint_chi2_held_above_99pct": float(np.mean(ch_j > crit)),
            "raw_chi2_held_separable_minus_joint_p16_p50_p84": pct(
                np.array([r["chi2_held"] for r in real_sep]) - sc_joint["chi2_held"][idx, Rj - 1]),
            "separable_full_rank_dof_p50": float(np.median(dof_full)),
            "fraction_separable_full_rank_chi2_fit_above_99pct_of_its_dof":
                float(np.mean(full_s > chi2dist.ppf(0.99, dof_full)))}


def fixed_factor_e5(cols, S, a, zf, zh):
    """Reference: separable basis with the paper's fixed factor 0.4 e^{0.5 i}."""
    lf = LinearFit("E5-fixed", cols["sep"], S["E5"], a["E5-sep"])
    sc = lf.scan(lf.Sf + zf, lf.Sh + zh)
    R = int(sc["R_star"][0])
    return {"R_star": R, "chi2_fit": float(sc["chi2_fit"][0, R - 1]), "chi2_held": float(sc["chi2_held"][0, R - 1]),
            "max_fit_minus_direct_over_I": float(sc["err"][0, R - 1])}


def regression_check(lf_e1, zf, zh):
    """Paper threshold rule s >= 10 on E1 must reproduce regression_results.json."""
    ref = json.loads(REGRESSION_JSON.read_text())["continuum"]
    R = int(np.sum(lf_e1.s >= 10.0))
    sc = lf_e1.scan(lf_e1.Sf + zf, lf_e1.Sh + zh)
    est = np.abs(lf_e1.vt[:R] @ lf_e1.a_hat(sc["proj"][0], R))
    return {"R_threshold_s>=10": R, "chi2_fit": float(sc["chi2_fit"][0, R - 1]),
            "max_abs_diff_|beta_hat|_vs_regression_estimated_modes": float(np.max(np.abs(
                est - np.abs(ref["estimated_modes"])))),
            "max_abs_diff_|rep_bias|_vs_regression": float(np.max(np.abs(
                np.abs(lf_e1.bias[:R]) - np.abs(ref["representation_mode_bias_over_sigma"])))),
            "chi2_fit_regression": ref["chi2_fit_seeded"],
            "max_bias_sigma": float(np.max(np.abs(lf_e1.bias[:R]))), "max_bias_regression": ref["max_bias_sigma"],
            "s7_s8": [float(lf_e1.s[6]), float(lf_e1.s[7])], "s7_s8_regression": ref["s7_s8"],
            "fit_max_over_I_all41": float(sc["err"][0, R - 1]),
            "fit_max_regression": ref["fit_max_over_I_all41"]}


def span_check(fits):
    """Independent lstsq: part of eps_w (fitted rows) outside the full column span, per ensemble."""
    out = {}
    for name, lf in fits.items():
        ew = stack_rows(lf.eps, FIT) / lf.sf
        x, *_ = np.linalg.lstsq(lf.Af, ew, rcond=None)
        r = ew - lf.Af @ x
        out[name] = {"lstsq_outside_span_chi2": float(r @ r), "eps_w_norm2": float(ew @ ew),
                     "rank_used": int(lf.k), "s_min_over_s_max": float(lf.s[-1] / lf.s[0])}
    return out


def summary_tables(res):
    rows = {}
    for name in LINEAR:
        r = res[name]
        sd, rl, nn = r["seeded"], r["realisations"], r["nnls_witness"]
        cf, cp = rl["coverage_fixed_R"], rl["coverage_post_selection"]
        rows[name] = {
            "R_star": sd["R_star"], "R_star_p16_p50_p84": rl["R_star_p16_p50_p84"],
            "s_R_star": sd["singular_values"][sd["R_star"] - 1],
            "chi2_fit": sd["chi2_fit"], "dof_fit": sd["dof_fit"], "chi2_held": sd["chi2_held"],
            "expected_chi2_held_if_eps_0": sd["expected_chi2_held_if_eps_0"],
            "chi2_held_predictive": sd["chi2_held_predictive"],
            "chi2_held_predictive_p50_realisations": rl["chi2_held_predictive_at_R_star_p16_p50_p84"][1],
            "eps_w_norm2_fit": sd["eps_diagnostics"]["eps_w_norm2_fit"],
            "eps_w_outside_span_chi2": sd["eps_diagnostics"]["eps_w_outside_span_chi2"],
            "max_err_over_I": sd["max_fit_minus_direct_over_I"]["value"],
            "max_err_noise_free": sd["noise_free_at_R_star"]["max_fit_minus_direct_over_I"],
            "max_abs_beta_dev_sigma": float(np.max(np.abs(sd["beta_hat_minus_true_over_sigma"]))),
            "max_abs_rep_bias_sigma": sd["max_abs_rep_bias_over_sigma"],
            "coverage_fixed_min_mean": [min(cf["per_mode_empirical"]), float(np.mean(cf["per_mode_empirical"]))],
            "coverage_joint_fixed": cf["joint_ellipsoid_empirical"],
            "coverage_joint_post": cp["joint_ellipsoid_empirical"],
            "nnls_chi2_seeded": nn["seeded"]["chi2_fit"], "nnls_chi2_noise_free": nn["noise_free"]["chi2_fit"],
            "nnls_chi2_median": nn["realisations"]["chi2_fit_p16_p50_p84"][1],
            "nnls_implied_column_seeded": nn["seeded_implied_total_column"],
            "hausdorff_bins_violating_seeded": sd["hausdorff_tsvd"]["n_bins_violating"],
            "hausdorff_bins_violating_noise_free": sd["hausdorff_tsvd_noise_free"]["n_bins_violating"],
            "hausdorff_fraction_realisations": rl["hausdorff_tsvd_fraction_with_violation"],
        }
    return rows


def save_products(fits, prods, res, path):
    zf, zh = seeded_noise()
    arrays = {}
    for name, lf in fits.items():
        p, key = prods[name], name.replace("-", "_")
        R = int(res[name]["seeded"]["R_star"])
        arrays.update({f"{key}_s": lf.s, f"{key}_vt": lf.vt, f"{key}_u": lf.u, f"{key}_R_star": R,
                       f"{key}_beta_hat": p["proj"][:R] / lf.s[:R], f"{key}_a_hat": p["a_hat"],
                       f"{key}_a_true": lf.a_true, f"{key}_S_direct": lf.S, f"{key}_sigma": lf.sigma,
                       f"{key}_eps": lf.eps, f"{key}_rep_bias_sigma": lf.bias, f"{key}_pred": p["pred"],
                       f"{key}_data_fit": lf.S[FIT] + zf.reshape(-1, 3) * lf.sigma[FIT],
                       f"{key}_data_held": lf.S[HELD] + zh.reshape(-1, 3) * lf.sigma[HELD]})
    arrays.update(fit_idx=FIT, held_idx=HELD, centres=CFG.centres)
    np.savez(path, **arrays)


def main():
    cols, S, a, direct_check, ens = build_inputs()
    zf, zh = seeded_noise()
    zfR, zhR, _ = realisation_noise(N_REAL)
    noise = (zf, zh, zfR, zhR)
    res, fits, prods, scans = {}, {}, {}, {}
    for name in LINEAR:
        fits[name], res[name], scans[name], prods[name] = run_linear(name, cols, S, a, noise)
        sd = res[name]["seeded"]
        print(name, "R*", sd["R_star"], "chi2", round(sd["chi2_fit"], 2), "held", round(sd["chi2_held"], 2),
              "err", round(sd["max_fit_minus_direct_over_I"]["value"], 5),
              "bias", round(sd["max_abs_rep_bias_over_sigma"], 3), flush=True)
    sep, sep_seed, sep_nf, sep_real = run_free_factor(cols, S, zf, zh, zfR, zhR)
    print("E5 free factor", sep["seeded"]["R_star"], sep["seeded"]["chi2_fit"], sep["seeded"]["chi2_held"], flush=True)
    lf5 = fits["E5"]
    pol = np.hypot(S["E5"][:, 1], S["E5"][:, 2]) / S["E5"][:, 0]
    e5 = {"joint": "see ensembles.E5", "separable_free_factor": sep,
          "separable_fixed_paper_factor": fixed_factor_e5(cols, S, a, zf, zh),
          "joint_vs_separable_realisations": compare_e5(scans["E5"], sep_real, lf5),
          "chi_nu": e5_angles(lf5, prods["E5"]["a_hat"], res["E5"]["seeded"]["R_star"], scans["E5"],
                              sep_seed["pred"], sep_nf["pred"], pol)}
    out = {
        "description": __doc__,
        "settings": {"N": N_ORDER, "n_fit_data": int(3 * len(FIT)), "n_held_data": int(3 * len(HELD)),
                     "seed_fit_rows": CFG.seed, "seed_held_rows": SEED_HELD, "seed_realisations": SEED_REAL,
                     "n_realisations": N_REAL, "n_realisations_nnls": N_REAL_NNLS,
                     "n_realisations_free_factor": N_REAL_SEP, "atoms": Z_ATOMS.tolist(),
                     "n_azimuth_atoms": N_AZ_ATOMS, "coverage_nominal": COVER,
                     "truth": {"E4-KP": "effective diagonal moments", "E5": "(a, Re M2, Im M2)",
                               "E1M": "E1 moments; data from the Maxwellian field, basis U[4,6]"},
                     "E2a": "not built"},
        "summary": summary_tables(res),
        "ensembles": res, "E5": e5,
        "checks": {"direct_vs_forward_spectra_npz_max_over_I": direct_check,
                   "E1_threshold_rule_regression": regression_check(fits["E1"], zf, zh),
                   "eps_outside_column_span_lstsq": span_check(fits)},
        "config": CFG.describe(), "timings_s": TIMINGS,
    }
    save_products(fits, prods, res, HERE / "fits_products.npz")
    out["provenance"] = provenance(__file__, T0, inputs=(FORWARD_NPZ, REGRESSION_JSON))
    write_json(HERE / "fits_results.json", out)
    print(json.dumps(out["summary"], indent=1, default=float))
    print(json.dumps(out["checks"], indent=1, default=float))


if __name__ == "__main__":
    main()
