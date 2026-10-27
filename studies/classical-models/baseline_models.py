"""Models compared in baselines.py: moment-basis TSVD with UPRE, electron-spectrum families through
the moment basis, and the generic spectral laws of spectral_laws.py.

All fits are weighted least squares on intensity rows only, with known noise sd.
Moment-basis models predict P through the separable P columns (paper azimuth factor and screen
inside the channel integral); the spectral laws predict P only under the explicit extra
assumption documented in spectral_laws.py.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import least_squares

import ensembles as E
import spectral_laws as SL
from config import PAPER

GAMMA0 = E.GAMMA0
R_EQUAL = (2, 3, 4)
R_CAP = 14  # UPRE search limited to R <= 14 (the Picard range R = 3..14)


# ----------------------------------------------------------------------------- moment TSVD
def tsvd_setup(CI_fit: np.ndarray, sig_fit: np.ndarray) -> dict:
    """Whitened SVD of the fitted intensity rows."""
    u, s, vt = np.linalg.svd(CI_fit / sig_fit[:, None], full_matrices=False)
    return {"u": u, "s": s, "vt": vt}


def tsvd_fit(svd: dict, d_fit: np.ndarray, sig_fit: np.ndarray, r_cap: int = R_CAP) -> dict:
    """UPRE rank R* = argmin_R [chi2_fit(R) + 2R] over R = 1..min(rank, r_cap); moments for every R.

    ``R_star_uncapped`` is the minimiser over the full range R = 1..rank (kept for reporting).
    """
    dw = d_fit / sig_fit
    proj = svd["u"].T @ dw
    rmax = len(svd["s"])
    tail = np.sum(dw**2) - np.cumsum(proj**2)  # chi2_fit(R) for R = 1..rmax
    upre = tail + 2 * np.arange(1, rmax + 1)
    r_star = int(np.argmin(upre[:min(rmax, r_cap)])) + 1
    coeff = proj / svd["s"]
    return {"proj": proj, "coeff": coeff, "chi2_fit": tail, "upre": upre, "R_star": r_star,
            "R_star_uncapped": int(np.argmin(upre)) + 1, "r_cap": r_cap}


def tsvd_moments(svd: dict, fit: dict, R: int) -> np.ndarray:
    return svd["vt"][:R].T @ fit["coeff"][:R]


def tsvd_cov(svd: dict, R: int) -> np.ndarray:
    """Cov(a_hat) = V_R D_R^-2 V_R^T (whitened noise)."""
    v = svd["vt"][:R].T / svd["s"][:R]
    return v @ v.T


# ----------------------------------------------------------------------------- electron families
def _lp_shape(g, th):
    y = np.log(g / GAMMA0)
    return np.exp(-th[:, 0] * y - th[:, 1] * y**2)


def family_moments(name: str, phi, cfg=PAPER) -> np.ndarray:
    """Unit-amplitude moments (12, 3) of an electron-spectrum family (N = 2).

    ePL   (A, p):            N = (A/g0)(g/g0)^-p
    eLP   (A, p, kappa):     N = (A/g0)(g/g0)^-p exp(-kappa ln^2(g/g0))   (kappa signed)
    eSBPL (A, p1, p2, ln g_b), s = 2 fixed: smoothly broken power law of E2
    """
    phi = np.asarray(phi, float)
    if name == "ePL":
        return E.powerlaw_moments(cfg, np.r_[1.0, phi], 2)[0]
    if name == "eLP":
        return E._smooth_family_moments(cfg, _lp_shape, np.r_[1.0, phi], 2)[0]
    if name == "eSBPL":
        return E.smooth_broken_moments(cfg, np.r_[1.0, phi, 2.0], 2)[0]
    raise ValueError(name)


FAMILY_BOUNDS = {
    "ePL": ([0.5], [6.0]),
    "eLP": ([0.5, -0.5], [6.0, 0.5]),
    "eSBPL": ([0.0, 0.0, np.log(100.0)], [7.0, 7.0, np.log(3e5)]),
}
FAMILIES = ("ePL", "eLP", "eSBPL")


def family_n_params(name: str) -> int:
    return len(FAMILY_BOUNDS[name][0]) + 1


def family_starts(name: str, p0: float):
    if name == "ePL":
        return [np.array([p]) for p in (2.0, 2.6, 3.2)]
    if name == "eLP":
        return [np.array([p0, k]) for k in (-0.05, 0.0, 0.05)]
    return [np.array([max(p0 - 0.5, 0.1), p0 + 0.5, np.log(g)]) for g in (2e3, 7e3, 2e4, 6e4)]


def fit_family(name: str, C_rows: np.ndarray, d: np.ndarray, sig: np.ndarray, p0: float = 2.6) -> dict:
    """Weighted LSQ of C_rows a(theta) to d; amplitude solved linearly (variable projection)."""
    dw = d / sig

    def gvec(phi):
        return (C_rows @ family_moments(name, phi).ravel()) / sig

    def resid(phi):
        g = gvec(phi)
        return (g @ dw) / (g @ g) * g - dw

    lo, hi = FAMILY_BOUNDS[name]
    best = None
    for x0 in family_starts(name, p0):
        x0 = np.clip(x0, np.array(lo) + 1e-6, np.array(hi) - 1e-6)
        r = least_squares(resid, x0, bounds=(lo, hi), method="trf", xtol=1e-12, ftol=1e-12,
                          gtol=1e-12, max_nfev=3000)
        if best is None or r.cost < best.cost:
            best = r
    g = gvec(best.x)
    amp = (g @ dw) / (g @ g)
    at_bound = bool(np.any(np.isclose(best.x, lo, atol=1e-5) | np.isclose(best.x, hi, atol=1e-5)))
    return {"params": np.r_[amp, best.x], "chi2": float(2 * best.cost), "at_bound": at_bound}


def family_prediction(name: str, params, C: np.ndarray) -> np.ndarray:
    """C (rows, 36) times a(params)."""
    return C @ (params[0] * family_moments(name, params[1:]).ravel())


# ----------------------------------------------------------------------------- one fit of every model
def metrics(pred_I, I_true, d_fit, d_held, sig, fit_idx, held_idx, pred_P=None, P_true=None) -> dict:
    out = {
        "chi2_fit": float(np.sum(((pred_I[fit_idx] - d_fit) / sig[fit_idx]) ** 2)),
        "chi2_held": float(np.sum(((pred_I[held_idx] - d_held) / sig[held_idx]) ** 2)),
        "max_res_all": float(np.max(np.abs(pred_I - I_true) / I_true)),
        "max_res_held": float(np.max(np.abs(pred_I[held_idx] - I_true[held_idx]) / I_true[held_idx])),
        "rms_res_held": float(np.sqrt(np.mean(((pred_I[held_idx] - I_true[held_idx]) / I_true[held_idx]) ** 2))),
    }
    if pred_P is not None and P_true is not None:
        out["maxP_over_I_all"] = float(np.max(np.abs(pred_P - P_true) / I_true))
    return out


def fit_all(ctx: dict, d_fit: np.ndarray, d_held: np.ndarray, keep_curves: bool = False) -> dict:
    """Fit every model to one intensity realisation.

    ctx keys: CI (J, 36), CP (J, 36) complex or None, grid (spectral_laws.Grid, all rows),
    fit_idx, held_idx, sig (J,), I_true (J,), P_true (J,) complex or None,
    factor (complex), screen (J, T) complex or None, svd (tsvd_setup on fitted rows).
    """
    fi, hi = ctx["fit_idx"], ctx["held_idx"]
    sig, I_true, P_true = ctx["sig"], ctx["I_true"], ctx["P_true"]
    CI, CP = ctx["CI"], ctx["CP"]
    res, curves = {}, {}

    # moment basis, TSVD: UPRE rank capped at R_CAP (moment_UPRE) and uncapped (moment_UPRE_uncapped)
    tf = tsvd_fit(ctx["svd"], d_fit, sig[fi])
    for R in sorted({*R_EQUAL, tf["R_star"], tf["R_star_uncapped"]}):
        a = tsvd_moments(ctx["svd"], tf, R)
        pI = CI @ a
        pP = CP @ a if CP is not None else None
        m = metrics(pI, I_true, d_fit, d_held, sig, fi, hi, pP, P_true)
        m["n_params"] = R
        if R == tf["R_star"]:
            res["moment_UPRE"] = {**m, "R_star": tf["R_star"], "R_star_uncapped": tf["R_star_uncapped"],
                                  "r_cap": tf["r_cap"]}
            if keep_curves:
                curves["moment_UPRE"] = {"I": pI, "P": pP, "a": a}
        if R == tf["R_star_uncapped"]:
            res["moment_UPRE_uncapped"] = {**m, "R_star": tf["R_star_uncapped"]}
        if R in R_EQUAL:
            res[f"moment_R{R}"] = m

    # spectral laws
    grid = ctx["grid"]
    for law in SL.LAWS:
        f = SL.fit(law, grid.rows(fi), d_fit, sig[fi])
        pI = SL.predict(law, f["params"], grid)
        pP = None
        if ctx.get("screen") is not None:
            pP = SL.polarisation_with_assumption(law, f["params"], grid, ctx["factor"], ctx["screen"])
        m = metrics(pI, I_true, d_fit, d_held, sig, fi, hi, pP, P_true)
        if "maxP_over_I_all" in m:  # laws carry P only under the extra assumption
            m["maxP_over_I_all_with_assumption"] = m.pop("maxP_over_I_all")
        m.update(n_params=SL.n_params(law), params=f["params"].tolist(), flag=f["edge_or_fail"])
        res[law] = m
        if keep_curves:
            curves[law] = {"I": pI, "P_with_assumption": pP}
    p0 = 1.0 - 2.0 * res["s1"]["params"][0]  # electron index start from the s1 slope

    # electron-spectrum families through the moment basis
    for fam in FAMILIES:
        f = fit_family(fam, CI[fi], d_fit, sig[fi], p0=p0)
        pI = family_prediction(fam, f["params"], CI)
        pP = family_prediction(fam, f["params"], CP) if CP is not None else None
        m = metrics(pI, I_true, d_fit, d_held, sig, fi, hi, pP, P_true)
        m.update(n_params=family_n_params(fam), params=f["params"].tolist(), flag=f["at_bound"])
        res[fam] = m
        if keep_curves:
            curves[fam] = {"I": pI, "P": pP}
    return (res, curves) if keep_curves else res
