"""Tails of the UPRE rank and of the withheld predictive chi2 for the main I, Q, U fits.

Re-runs the 4000 linear-fit realisations of fits.py (same seeds, same LinearFit machinery,
imported from fits.py) and reports, per ensemble:
  * the R* distribution (histogram, fractions with R* >= 15 and >= 20, maximum);
  * the predictive withheld chi2 r^T (I + H_R H_R^T)^-1 r at the per-realisation R*:
    mean, median, 95th and 99th percentiles, maximum, fraction above the chi2_60 99% point;
  * its expectation at the seeded R* (fixed R), 60 + mu^T (I + H H^T)^-1 mu, with mu the
    mean withheld residual left by truncation and representation error (whitened);
  * a decision rule: the search range R = 1..rank is changed only if the
    99th percentile exceeds twice that expectation.  A capped search (R <= 14, pair-respecting
    for E5) is evaluated alongside as a sensitivity check whatever the outcome.

Writes fits_tails_results.json.  Run: python fits_tails.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.stats import chi2 as chi2dist  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))
import fits as F  # noqa: E402
from provenance import provenance, write_json  # noqa: E402

T0 = time.time()
R_CAP = 14
CAP_E5 = 28  # E5 joint: the same cap per complex pair structure is not defined; 2 x 14 real modes


def expectation_fixed_R(lf, R: int) -> float:
    """60 + mu^T (I + H_R H_R^T)^-1 mu at fixed R (whitened withheld rows)."""
    H = lf.Hh[:, :R]
    mu = H @ (lf.u[:, :R].T @ lf.Sf) - lf.Sh
    M = np.linalg.inv(np.eye(H.shape[0]) + H @ H.T)
    return float(len(lf.Sh) + mu @ M @ mu)


def r_star_capped(lf, upre: np.ndarray, cap: int) -> np.ndarray:
    allowed = lf.allowed.copy()
    allowed[cap:] = False
    return np.arange(1, lf.k + 1)[np.argmin(np.where(allowed[None, :], upre, np.inf), axis=1)]


def tail_stats(v: np.ndarray, crit: float) -> dict:
    return {"mean": float(np.mean(v)), "median": float(np.median(v)), "p95": float(np.percentile(v, 95)),
            "p99": float(np.percentile(v, 99)), "max": float(np.max(v)),
            "frac_above_chi2_60_99pct": float(np.mean(v > crit))}


def one_ensemble(name, cols, S, a, zfR, zhR, crit) -> dict:
    cplx = (np.arange(36, 72), np.arange(72, 108)) if name == "E5" else None
    lf = F.LinearFit(name, F.basis_for(name, cols), S[name], a[name], cplx=cplx)
    zf, zh = F.seeded_noise()
    R_seed = int(lf.scan(lf.Sf + zf, lf.Sh + zh)["R_star"][0])
    sc = lf.scan(lf.Sf[None] + zfR, lf.Sh[None] + zhR)
    Rn, idx = sc["R_star"], np.arange(len(zfR))
    pred = lf.predictive_held_at(sc["proj"], zhR + lf.Sh, Rn)
    vals, cnt = np.unique(Rn, return_counts=True)
    out = {
        "rank": lf.k, "R_seeded": R_seed,
        "R_star_histogram": {int(v): int(c) for v, c in zip(vals, cnt)},
        "R_star_frac_ge_15": float(np.mean(Rn >= 15)), "R_star_frac_ge_20": float(np.mean(Rn >= 20)),
        "R_star_max": int(Rn.max()), "R_star_p50_p95_p99": [float(np.percentile(Rn, q)) for q in (50, 95, 99)],
        "chi2_held_predictive_at_R_star": tail_stats(pred, crit),
        "chi2_held_raw_at_R_star": tail_stats(sc["chi2_held"][idx, Rn - 1], crit),
        "max_err_over_I_at_R_star": {"median": float(np.median(sc["err"][idx, Rn - 1])),
                                     "p95": float(np.percentile(sc["err"][idx, Rn - 1], 95)),
                                     "p99": float(np.percentile(sc["err"][idx, Rn - 1], 99))},
        "expectation_predictive_fixed_R_seeded": expectation_fixed_R(lf, R_seed),
        "expectation_predictive_by_R": {int(R): expectation_fixed_R(lf, int(R)) for R in vals},
    }
    hi_mask = Rn >= 15
    if hi_mask.any():
        out["chi2_held_predictive_given_R_star_ge_15"] = tail_stats(pred[hi_mask], crit)
    out["rule_p99_over_2x_expectation"] = out["chi2_held_predictive_at_R_star"]["p99"] / (
        2.0 * out["expectation_predictive_fixed_R_seeded"])
    out["rule_triggered"] = bool(out["rule_p99_over_2x_expectation"] > 1.0)
    cap = CAP_E5 if name == "E5" else R_CAP
    Rc = r_star_capped(lf, sc["upre"], cap)
    pc = lf.predictive_held_at(sc["proj"], zhR + lf.Sh, Rc)
    out["sensitivity_capped"] = {"cap": cap, "frac_changed": float(np.mean(Rc != Rn)),
                                 "chi2_held_predictive_at_R_star": tail_stats(pc, crit),
                                 "max_err_over_I_at_R_star_p99": float(np.percentile(sc["err"][idx, Rc - 1], 99))}
    return out


def main():
    cols, S, a, _check, _ens = F.build_inputs()
    zfR, zhR, _ = F.realisation_noise(F.N_REAL)
    crit = float(chi2dist.ppf(0.99, 3 * len(F.HELD)))
    res = {}
    for name in F.LINEAR:
        t = time.time()
        res[name] = one_ensemble(name, cols, S, a, zfR, zhR, crit)
        res[name]["seconds"] = time.time() - t
        r = res[name]
        print(name, "R*>=15", r["R_star_frac_ge_15"], "pred mean/p95/p99",
              [round(r["chi2_held_predictive_at_R_star"][k], 1) for k in ("mean", "p95", "p99")],
              "E", round(r["expectation_predictive_fixed_R_seeded"], 2), "trig", r["rule_triggered"], flush=True)
    out = {
        "description": __doc__,
        "settings": {"n_realisations": F.N_REAL, "seed_realisations": F.SEED_REAL, "search_range": "R = 1..rank "
                     "(fits.py; E5 joint: pair-respecting truncations)", "chi2_60_99pct": crit,
                     "decision_rule": "change the search range only if p99 of the predictive chi2_held at R* "
                                      "exceeds twice its fixed-R expectation at the seeded R*"},
        "ensembles": res,
        "any_rule_triggered": any(r["rule_triggered"] for r in res.values()),
        "provenance": provenance(__file__, T0, inputs=(HERE / "fits.py", F.FORWARD_NPZ)),
    }
    write_json(HERE / "fits_tails_results.json", out)
    print("done", time.time() - T0)


if __name__ == "__main__":
    main()
