"""Spectral (SED) error along the E2 field-energy ridge at expansion orders N = 2, 3, 4.

The set-up is that of params_degeneracy.ridge_one: SBPL_s2 family, full likelihood, noise sd
0.01 I_j, fits started at the transformed truth, energy support scaled as B^-1/2 or fixed.
The ln(lambda) grid is finer, and three orders are fitted. For each (N, support, ln lambda),
for noise-free data and for the seeded draw of params.py, the script records
  * chi2 in units of the 1% noise, the fit status, phi minus the transformed truth and the
    Gauss-Newton sd
  * the SED error max |C a(phi) - S| / I_j over all 41 channels and I, Q, U, the same over the
    fitted and the withheld channels, and the rms
  * the largest 1-sigma SED band sqrt(diag(J cov J^T)) / I_j of the fit
The summary gives, per (support, N), the range of each error along the ridge, the parabolic
vertex of the noise-free chi2 in ln B and its curvature at the true field.

Check: at N = 2, chi2 and phi on the 11 points of params_results.json -> degeneracy.E2_ridge
are reproduced to 1e-8, or the script stops without writing.

Reads params_results.json. Writes ridge_sed_results.json.  Run: python ridge_sed.py
(about 30 s with 17 worker processes, 0.5 GB peak memory in the parent process)
"""
from __future__ import annotations

import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))
import basis as B  # noqa: E402
import ensembles as E  # noqa: E402
import paramlib as P  # noqa: E402
import params_degeneracy as D  # noqa: E402
from config import PAPER  # noqa: E402
from provenance import provenance, write_json  # noqa: E402

T0 = time.time()
ORDERS = (2, 3, 4)
GRID = tuple(float(x) for x in np.round(np.arange(-0.4, 0.4001, 0.05), 10))
SHIFT = np.array([-0.5, 0.0, 0.0, -0.5])  # d phi / d ln lambda on the ridge (ln A, p1, p2, ln gamma_b)
PARAMS_JSON = HERE / "params_results.json"


def sed_record(C, fam, cl, prob, f, S, N):
    """SED error of the fitted spectrum against the direct spectrum, in units of I_j."""
    a, Ja = fam.moments_and_jacobian(cl, f["phi"], N)
    pred = C @ a
    err = np.abs(pred - S) / S[:, :1]
    Js = C.reshape(-1, C.shape[-1]) @ Ja
    band = np.sqrt(np.einsum("ik,kl,il->i", Js, f["cov"], Js)).reshape(S.shape) / S[:, :1]
    return {"chi2": f["chi2"], "success": f["success"], "status": f["status"], "sd": f["sd"],
            "sed_max": float(err.max()), "sed_max_fitted": float(err[cl.fit_idx].max()),
            "sed_max_withheld": float(err[cl.held_idx].max()), "sed_rms": float(np.sqrt(np.mean(err**2))),
            "sed_band_1sigma_max": float(band.max()), "_phi": f["phi"]}


def one(args):
    lnlam, S, phi_true, z_fit = args
    lam = float(np.exp(lnlam))
    tab = D.table_for(D.scaled_cfg(PAPER, lam))
    fam = P.Family("SBPL_s2")
    phi_exp = phi_true + SHIFT * lnlam
    out = {"ln_lambda": lnlam}
    for support in ("transformed", "fixed"):
        cl = D.scaled_cfg(PAPER, lam, support == "transformed")
        rec = {}
        for N in ORDERS:
            C = B.build_basis(cl, N, table=tab)
            prob = P.Whitened(cl, C, S)
            rec[str(N)] = {}
            for dname, zz in (("noisefree", 0.0 * z_fit), ("seeded", z_fit)):
                dw = prob.whiten(S[cl.fit_idx] + zz * prob.sig[cl.fit_idx])
                f = P.fit(prob, fam, cl, dw, "full", None, phi_exp, N=N)
                r = sed_record(C, fam, cl, prob, f, S, N)
                r["phi"] = r.pop("_phi")
                r["phi_minus_transformed_truth"] = r["phi"] - phi_exp
                rec[str(N)][dname] = r
        out[support] = rec
    return out


def vertex(x, y):
    """Abscissa of the parabola through the smallest grid value and its two neighbours."""
    i = min(max(int(np.argmin(y)), 1), len(y) - 2)
    h = x[1] - x[0]
    fm, f0, fp = y[i - 1], y[i], y[i + 1]
    curv = fm - 2.0 * f0 + fp
    if np.ptp(y) <= 1e-9 * abs(f0) or curv <= 0.0:  # flat ridge (exact symmetry): no minimum
        return float("nan")
    return float(x[i] + h * (fm - fp) / (2.0 * curv))


def summarise(rows):
    x = np.array(GRID)
    i0 = GRID.index(0.0)
    out = {}
    for sup in ("transformed", "fixed"):
        for N in map(str, ORDERS):
            for dn in ("noisefree", "seeded"):
                col = [r[sup][N][dn] for r in rows]
                chi = np.array([c["chi2"] for c in col])
                key = f"{sup}/N{N}/{dn}"
                out[f"{key}/chi2_minus_chi2_at_truth"] = chi - chi[i0]
                out[f"{key}/chi2_at_truth"] = float(chi[i0])
                out[f"{key}/chi2_min_dlnB_parabola"] = vertex(x, chi)
                for q in ("sed_max", "sed_max_withheld", "sed_band_1sigma_max"):
                    v = np.array([c[q] for c in col])
                    out[f"{key}/{q}"] = v
                    out[f"{key}/{q}_range"] = [float(v.min()), float(v.max())]
                out[f"{key}/all_fits_converged"] = all(c["success"] for c in col)
            nf = np.array([r[sup][N]["noisefree"]["chi2"] for r in rows])
            c = 0.5 * (nf[i0 - 1] + nf[i0 + 1]) - nf[i0]
            out[f"{sup}/N{N}/noisefree/chi2_curvature_per_lnB2_at_truth"] = float(c / (x[1] - x[0]) ** 2)
    return out


def checks_against_params(rows):
    """N = 2 chi2 and phi against params_results.json's E2_ridge (same data, start and code path)."""
    rg = json.loads(PARAMS_JSON.read_text())["degeneracy"]["E2_ridge"]
    by_x = {round(r["ln_lambda"], 10): r for r in rows}
    worst = {"chi2_rel": 0.0, "phi_abs": 0.0}
    for x, ref in zip(rg["ln_lambda_grid"], rg["rows"]):
        mine = by_x[round(x, 10)]
        for sup in ("transformed", "fixed"):
            for dn in ("noisefree", "seeded"):
                a, b = mine[sup]["2"][dn], ref[sup][dn]["full"]
                worst["chi2_rel"] = max(worst["chi2_rel"], abs(a["chi2"] - b["chi2"]) / max(b["chi2"], 1e-300))
                worst["phi_abs"] = max(worst["phi_abs"], float(np.max(np.abs(np.asarray(a["phi"]) - b["phi"]))))
    return {"N2_vs_params_E2_ridge_max_rel_chi2": worst["chi2_rel"],
            "N2_vs_params_E2_ridge_max_abs_phi": worst["phi_abs"], "points_compared": len(rg["rows"])}


def main():
    ens = E.reference_ensembles(PAPER)
    S = ens["E2"].direct(PAPER)
    phi_true = P.Family("SBPL_s2").from_lib(ens["E2"].theta)
    z_fit = PAPER.draw_noise(len(PAPER.fit_idx))  # the seeded draw of params.py's ridge
    with ProcessPoolExecutor(max_workers=min(len(GRID), os.cpu_count() or 4)) as pool:
        rows = list(pool.map(one, [(x, S, phi_true, z_fit) for x in GRID]))
    checks = checks_against_params(rows)
    if checks["N2_vs_params_E2_ridge_max_rel_chi2"] > 1e-8 or checks["N2_vs_params_E2_ridge_max_abs_phi"] > 1e-8:
        raise AssertionError(f"N = 2 ridge does not reproduce params_results.json: {checks}")
    results = {
        "description": "SED error along the E2 field-energy ridge (Sec. 5.4) at N = 2, 3, 4. "
                       "Errors in units of I_j, chi2 in units of the 1% noise.",
        "ln_lambda_grid": GRID, "orders": ORDERS, "family": "SBPL_s2 (lnA, p1, p2, ln gamma_b), s = 2",
        "rows": rows, "summary": summarise(rows), "checks": checks,
        "provenance": provenance(__file__, T0, inputs=(HERE / "paramlib.py", HERE / "params_degeneracy.py",
                                                       PARAMS_JSON))}
    write_json(HERE / "ridge_sed_results.json", results)
    sm = results["summary"]
    for sup in ("transformed", "fixed"):
        for N in map(str, ORDERS):
            k = f"{sup}/N{N}"
            print(f"{k:>16}: chi2 min at dlnB={sm[k + '/noisefree/chi2_min_dlnB_parabola']:+.3f}  "
                  f"SED noise-free {100 * np.array(sm[k + '/noisefree/sed_max_range'])} %  "
                  f"seeded {100 * np.array(sm[k + '/seeded/sed_max_range'])} %  "
                  f"band {100 * np.array(sm[k + '/seeded/sed_band_1sigma_max_range'])} %")
    print("checks", checks, f"done in {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
