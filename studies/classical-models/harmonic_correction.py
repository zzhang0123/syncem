"""Harmonic-minus-continuum coefficients c_I(x, eta), c_Q(x, eta) for isotropic pitch, and their
propagation to the reference geometry (E1, E2, E4-JP).

c_S(x, eta) = gamma^2 (jbar_S / K^cont_S - 1), with jbar the frequency-averaged exact harmonic
emissivity (Poisson k = 0 term; harmonic_cells.py) and K^cont = A F(x), -A G(x).  Tables on
x = 1e-3 ... 31.6 (12 points per decade; the x <= 10 points coincide with
inputs/validation/harmonic_continuum_results.json) and eta = 0, 0.05, ..., 0.5, at gamma = 100 and 300.

Checks: refinement (W 40 -> 60, 32 -> 64 nodes per panel) in every cell; scipy versus Olver
Bessel routes; the exact integer-harmonic sum averaged over 16 line phases; the exact-sum
points of kp_bridge_results.json (isotropic, Olver); inputs/validation/harmonic_continuum_results.json
(theta = pi/3, gamma = 100; read only).

Propagation [run + gamma^-2 extrapolation]: Delta S_j = int dN(gamma) gamma^-2 <c K^cont>_j, the
average over the channel response, B ~ U[4,6] muG, eta ~ U[-1/2,1/2], the screen and the azimuth
factor, with c from the gamma = 300 table (gamma = 100 as a check).  Outside the tabulated x
range c is continued as x^-2/3 below 1e-3 and held constant above 31.6 (both variants reported).

Writes harmonic_correction_results.json and harmonic_correction_tables.npz.
Run: python harmonic_correction.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from dataclasses import replace  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from scipy.interpolate import RectBivariateSpline  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))
import ensembles as E  # noqa: E402
import harmonic_cells as HC  # noqa: E402
import kernels as K  # noqa: E402
from config import PAPER  # noqa: E402
from provenance import provenance, sha256, write_json  # noqa: E402

T0 = time.time()
GAMMAS = (100.0, 300.0)
X_GRID = np.concatenate([np.geomspace(1e-3, 10.0, 49), 10.0 * 10 ** (np.arange(1, 7) / 12.0)])
ETA_GRID = np.round(np.arange(0.0, 0.501, 0.05), 10)
N_WORKERS = 26
N_PHASE = 16
KPB_JSON = HERE / "kp_bridge_results.json"
VAL_JSON = HERE / "inputs" / "validation" / "harmonic_continuum_results.json"
FORWARD_NPZ = HERE / "forward_spectra.npz"
PROPAGATE = ("E1", "E2", "E4-JP")
N_CHAN = 100  # channel quadrature nodes for the correction (checked against 200)
LNX_FINE = np.linspace(np.log(1e-7), np.log(300.0), 8001)


# ----------------------------------------------------------------------------- tables
def tables(pool) -> dict:
    tasks = [(g, e, x, "auto", True) for g in GAMMAS for e in ETA_GRID for x in X_GRID]
    rows = pool.map(HC.cell, tasks, chunksize=4)
    out = {}
    for g in GAMMAS:
        sel = [r for r in rows if r["gamma"] == g]
        cI = np.array([r["c_I"] for r in sel]).reshape(len(ETA_GRID), len(X_GRID)).T
        cQ = np.array([r["c_Q"] for r in sel]).reshape(len(ETA_GRID), len(X_GRID)).T
        ref = max(max(abs(r["refined_minus_adopted_c_I"]), abs(r["refined_minus_adopted_c_Q"])) for r in sel)
        out[g] = {"c_I": cI, "c_Q": cQ, "max_refined_minus_adopted": ref,
                  "routes": {k: sum(r["route"] == k for r in sel) for k in ("scipy", "olver")},
                  "max_order": max(r["max_order"] for r in sel)}
    return out


def scaling_summary(tab: dict) -> dict:
    d = {}
    for S in ("c_I", "c_Q"):
        a, b = tab[100.0][S], tab[300.0][S]
        diff = b - a
        i = np.unravel_index(np.argmax(np.abs(diff)), diff.shape)
        rel = np.abs(diff) / np.abs(b)
        j = np.unravel_index(np.argmax(rel), rel.shape)
        d[S] = {"max_abs_change_100_to_300": float(np.abs(diff).max()), "at_x_eta": [float(X_GRID[i[0]]), float(ETA_GRID[i[1]])],
                "max_rel_change": float(rel.max()), "rel_at_x_eta": [float(X_GRID[j[0]]), float(ETA_GRID[j[1]])],
                "median_rel_change": float(np.median(rel)),
                "range_gamma300": [float(b.min()), float(b.max())]}
    return d


# ----------------------------------------------------------------------------- cross-checks
def check_routes(pool) -> dict:
    cells = [(g, e, x) for g in GAMMAS for e in (0.0, 0.25, 0.5) for x in (1e-3, 0.1, 1.0, 3.0)
             if HC.nu_of(g, e, x) * 1.0 <= HC.SCIPY_MAX_ORDER]
    sc = pool.map(HC.cell, [(g, e, x, "scipy", False) for g, e, x in cells])
    ol = pool.map(HC.cell, [(g, e, x, "olver", False) for g, e, x in cells])
    dI = [abs(a["c_I"] - b["c_I"]) for a, b in zip(sc, ol)]
    dQ = [abs(a["c_Q"] - b["c_Q"]) for a, b in zip(sc, ol)]
    return {"n_cells": len(cells), "max_abs_diff_c_I": float(max(dI)), "max_abs_diff_c_Q": float(max(dQ)),
            "cells": [list(c) for c in cells]}


def check_integer_sum(pool, tab) -> dict:
    cells = [(100.0, e, x) for e in (0.5, 0.25, 0.1) for x in (0.01, 0.1, 1.0)] + [(100.0, 0.1, 3.0)]
    tasks = [(g, e, x, k, N_PHASE) for g, e, x in cells for k in range(N_PHASE)]
    res = pool.map(HC.integer_sum_one_phase, tasks, chunksize=1)
    rows, worst = [], 0.0
    for g, e, x in cells:
        r = [t for t in res if (t["gamma"], t["eta"], t["x"]) == (g, e, x)]
        cI, cQ = g**2 * np.mean([t["dI"] for t in r]), g**2 * np.mean([t["dQ"] for t in r])
        mine = HC.cell((g, e, x, "auto", False))
        rows.append({"gamma": g, "eta": e, "x": x, "integer_sum_c_I": float(cI), "integer_sum_c_Q": float(cQ),
                     "averaged_c_I": mine["c_I"], "averaged_c_Q": mine["c_Q"],
                     "phase_spread_c_I": float(g**2 * np.ptp([t["dI"] for t in r])),
                     "n_lines_per_phase": int(r[0]["n_lines"])})
        worst = max(worst, abs(cI - mine["c_I"]), abs(cQ - mine["c_Q"]))
    return {"n_phase": N_PHASE, "rows": rows, "max_abs_diff_c": float(worst)}


def check_kp_bridge(pool) -> dict:
    pts = [p for p in json.loads(KPB_JSON.read_text())["B_points"]
           if p["pdf"][0] == "iso" and p["bessel"] == "olver" and p["gamma"] >= 160 and 0 < p["eta"] <= 0.5 + 1e-12]
    mine = pool.map(HC.cell, [(float(p["gamma"]), float(p["eta"]), float(p["x"]), "auto", False) for p in pts])
    dI = [abs(m["c_I"] - p["g2_d_rel_I"]) for m, p in zip(mine, pts)]
    dQ = [abs(m["c_Q"] - p["g2_d_rel_Q"]) for m, p in zip(mine, pts)]
    sel = [(p["gamma"], p["x"], round(p["eta"], 4), p["g2_d_rel_I"], m["c_I"]) for m, p in zip(mine, pts)
           if p["label"] == "eta=0.5"]
    return {"n_points": len(pts), "max_abs_diff_c_I": float(max(dI)), "max_abs_diff_c_Q": float(max(dQ)),
            "eta_half_rows_gamma_x_eta_kpbridge_mine": sel}


def check_validation(tab) -> dict:
    v = json.loads(VAL_JSON.read_text())
    rows = [r for r in v["rows"] if r["gamma"] == 100.0]
    ie = int(np.argmin(np.abs(ETA_GRID - 0.5)))
    out = []
    for r in rows:
        ix = int(np.argmin(np.abs(X_GRID / r["x"] - 1)))
        if abs(X_GRID[ix] / r["x"] - 1) > 1e-9:
            continue
        vI = 1e4 * (r["harmonic_I"] / r["continuum_I"] - 1)
        vQ = 1e4 * (r["harmonic_Q"] / r["continuum_Q"] - 1)
        out.append((r["x"], vI, float(tab[100.0]["c_I"][ix, ie]), vQ, float(tab[100.0]["c_Q"][ix, ie])))
    dI = max(abs(o[1] - o[2]) for o in out)
    dQ = max(abs(o[3] - o[4]) for o in out)
    pick = {f"x={o[0]:.4g}": {"validation_c_I": o[1], "table_c_I": o[2], "validation_c_Q": o[3], "table_c_Q": o[4]}
            for o in out if any(abs(o[0] / t - 1) < 1e-6 for t in (1e-3, 0.1, 1.0, 10.0))}
    return {"theta": v["theta"], "gamma": 100.0, "n_x": len(out), "max_abs_diff_c_I": float(dI),
            "max_abs_diff_c_Q": float(dQ), "selected": pick}


# ----------------------------------------------------------------------------- propagation
def fine_tables(cfg, c_I, c_Q, extrap: str):
    """H_S(ln x, eta_k) = c_S(x, eta_k) W_S(x) on LNX_FINE for the cfg eta nodes."""
    eta_nodes = cfg.eta_nodes_weights[0]
    lnx = np.log(X_GRID)
    xf = np.exp(LNX_FINE)
    inside = np.clip(LNX_FINE, lnx[0], lnx[-1])
    F, G = K.F(xf), K.G(xf)
    out = {}
    for S, c, W in (("I", c_I, F), ("Q", c_Q, G)):
        spl = RectBivariateSpline(lnx, ETA_GRID, c, kx=3, ky=3)
        cf = spl(inside, eta_nodes)  # (n_fine, n_eta), held constant outside the grid
        if extrap == "x^-2/3":
            low = LNX_FINE < lnx[0]
            cf[low] *= np.exp(-(2.0 / 3.0) * (LNX_FINE[low] - lnx[0]))[:, None]
        out[S] = (cf * W[:, None]).T  # (n_eta, n_fine)
    out["F"], out["G"] = F, G
    return out


def propagate(cfg, ens, H, chunk: int = 48) -> dict:
    """Delta S_j and the plain continuum S_j (check) with gamma^-2 weights; per-channel arrays."""
    m = ens.measure(cfg, n=24)
    lnnu = np.log(cfg.chan_nu)  # (J, T)
    bn, bw = cfg.b_nodes_weights
    en, ew = cfg.eta_nodes_weights
    s = np.sqrt(1.0 - en**2)
    wq = cfg.chan_w[None, :] * cfg.screen
    acc = {k: np.zeros(cfg.n_channels, complex if "Q" in k else float)
           for k in ("dI", "Q", "I", "I_g2", "dQ", "I_lowx", "I_highx")}
    for c0 in range(0, len(m.gamma), chunk):
        g, w = m.gamma[c0:c0 + chunk], m.w[c0:c0 + chunk]
        g2 = (1.0 / g**2)[:, None]
        for k in range(len(en)):
            lnx = (lnnu[None, None] - np.log(K.a1 * bn * s[k])[None, :, None, None]
                   - 2.0 * np.log(g)[:, None, None, None])  # (n_g, n_B, J, T)
            amp = w[:, None] * bw[None, :] * ew[k] * K.A1 * bn[None, :] * s[k]  # (n_g, n_B)
            F0 = np.interp(lnx, LNX_FINE, H["F"])
            acc["dI"] += np.einsum("gb,gbjt,t->j", amp * g2, np.interp(lnx, LNX_FINE, H["I"][k]), cfg.chan_w)
            acc["dQ"] += -np.einsum("gb,gbjt,jt->j", amp * g2, np.interp(lnx, LNX_FINE, H["Q"][k]), wq)
            acc["I"] += np.einsum("gb,gbjt,t->j", amp, F0, cfg.chan_w)
            acc["Q"] += -np.einsum("gb,gbjt,jt->j", amp, np.interp(lnx, LNX_FINE, H["G"]), wq)
            acc["I_g2"] += np.einsum("gb,gbjt,t->j", amp * g2, F0, cfg.chan_w)
            acc["I_lowx"] += np.einsum("gb,gbjt,t->j", amp, F0 * (lnx < np.log(X_GRID[0])), cfg.chan_w)
            acc["I_highx"] += np.einsum("gb,gbjt,t->j", amp, F0 * (lnx > np.log(X_GRID[-1])), cfg.chan_w)
    f = cfg.azimuth_factor
    I = acc["I"]
    return {"I": I, "P": f * acc["Q"], "dI": acc["dI"], "dP": f * acc["dQ"], "mean_inv_g2": acc["I_g2"] / I,
            "frac_lowx": acc["I_lowx"] / I, "frac_highx": acc["I_highx"] / I}


def summarise_prop(r: dict, cfg) -> dict:
    I = r["I"]
    rI, rP = r["dI"] / I, np.abs(r["dP"]) / I
    jI, jP = int(np.argmax(np.abs(rI))), int(np.argmax(rP))
    ceff = rI / r["mean_inv_g2"]
    return {"max_abs_dI_over_I": float(np.abs(rI).max()), "at_GHz_I": float(cfg.centres[jI] / 1e9),
            "max_abs_dP_over_I": float(rP.max()), "at_GHz_P": float(cfg.centres[jP] / 1e9),
            "max_abs_dQ_dU_over_I": [float(np.abs(r["dP"].real / I).max()), float(np.abs(r["dP"].imag / I).max())],
            "c_eff_I_range": [float(ceff.min()), float(ceff.max())],
            "dP_over_I_over_mean_inv_g2_max": float((rP / r["mean_inv_g2"]).max()),
            "mean_inv_g2_range": [float(r["mean_inv_g2"].min()), float(r["mean_inv_g2"].max())],
            "max_emission_fraction_x_below_1e-3": float(r["frac_lowx"].max()),
            "max_emission_fraction_x_above_grid": float(r["frac_highx"].max())}


def run_propagation(tab) -> tuple[dict, dict]:
    cfg = replace(PAPER, n_chan_nodes=N_CHAN)
    ens = E.reference_ensembles(cfg)
    ref = np.load(FORWARD_NPZ)
    out, arrays = {}, {}
    variants = {"gamma300_xm23": (300.0, "x^-2/3"), "gamma300_clamp": (300.0, "clamp"),
                "gamma100_xm23": (100.0, "x^-2/3")}
    H = {v: fine_tables(cfg, tab[g]["c_I"], tab[g]["c_Q"], ex) for v, (g, ex) in variants.items()}
    for name in PROPAGATE:
        rec = {}
        for v in variants:
            r = propagate(cfg, ens[name], H[v])
            rec[v] = summarise_prop(r, cfg)
            if v == "gamma300_xm23":
                S = ref[name.replace("-", "_")]
                rec["check_continuum_I_vs_forward_max_rel"] = float(np.max(np.abs(r["I"] / S[:, 0] - 1)))
                rec["check_continuum_P_vs_forward_max_over_I"] = float(np.max(np.abs(
                    r["P"] - (S[:, 1] + 1j * S[:, 2])) / S[:, 0]))
                idx = [int(np.argmin(np.abs(cfg.centres - f))) for f in (1e8, 1e9, 1e10)]
                rec["mean_inv_g2_at_0.1_1_10_GHz"] = [float(r["mean_inv_g2"][i]) for i in idx]
                rec["mean_inv_g2_pow_minus_half_at_0.1_1_10_GHz"] = [float(r["mean_inv_g2"][i] ** -0.5) for i in idx]
                arrays[f"{name}__dI_over_I"] = r["dI"] / r["I"]
                arrays[f"{name}__dP_over_I"] = r["dP"] / r["I"]
                arrays[f"{name}__mean_inv_g2"] = r["mean_inv_g2"]
        out[name] = rec
    # channel-node check for E1
    cfg2 = replace(PAPER, n_chan_nodes=2 * N_CHAN)
    r2 = propagate(cfg2, E.reference_ensembles(cfg2)["E1"], fine_tables(cfg2, tab[300.0]["c_I"], tab[300.0]["c_Q"], "x^-2/3"))
    out["E1"]["check_channel_nodes_200_vs_100_dI_over_I_max_abs_diff"] = float(np.max(np.abs(
        r2["dI"] / r2["I"] - arrays["E1__dI_over_I"])))
    return out, arrays


def main():
    timings = {}
    with get_context("spawn").Pool(N_WORKERS) as pool:
        t = time.time()
        tab = tables(pool)
        timings["tables_s"] = time.time() - t
        t = time.time()
        checks = {"routes_scipy_vs_olver": check_routes(pool), "kp_bridge_exact_sums": check_kp_bridge(pool),
                  "validation_theta_pi_3": check_validation(tab)}
        timings["checks_s"] = time.time() - t
        t = time.time()
        checks["integer_sum_phase_averaged"] = check_integer_sum(pool, tab)
        timings["integer_sum_s"] = time.time() - t
    t = time.time()
    prop, arrays = run_propagation(tab)
    timings["propagation_s"] = time.time() - t
    np.savez(HERE / "harmonic_correction_tables.npz", x=X_GRID, eta=ETA_GRID,
             **{f"c_{S}_gamma{int(g)}": tab[g][f"c_{S}"] for g in GAMMAS for S in ("I", "Q")}, **arrays)
    out = {
        "description": __doc__,
        "grid": {"x": X_GRID.tolist(), "eta": ETA_GRID.tolist(), "gammas": list(GAMMAS)},
        "tables_meta": {str(int(g)): {k: v for k, v in tab[g].items() if k not in ("c_I", "c_Q")} for g in GAMMAS},
        "tables_c_I": {str(int(g)): tab[g]["c_I"] for g in GAMMAS},
        "tables_c_Q": {str(int(g)): tab[g]["c_Q"] for g in GAMMAS},
        "gamma_scaling": scaling_summary(tab),
        "checks": checks,
        "propagation": prop,
        "propagation_label": "run + gamma^-2 extrapolation (c from gamma = 300 applied for gamma in [300, 1e5])",
        "timings_s": timings,
    }
    out["tables_npz_sha256"] = sha256(HERE / "harmonic_correction_tables.npz")
    out["provenance"] = provenance(__file__, T0, inputs=(HERE / "harmonic_cells.py", HERE / "kp_harmonic.py",
                                                         KPB_JSON, VAL_JSON, FORWARD_NPZ))
    write_json(HERE / "harmonic_correction_results.json", out)
    print(json.dumps({"scaling": out["gamma_scaling"], "checks": {k: {kk: vv for kk, vv in v.items() if kk.startswith("max")}
                                                                  for k, v in checks.items()}}, indent=1))
    print(json.dumps(prop, indent=1))
    print("done", time.time() - T0)


if __name__ == "__main__":
    main()
