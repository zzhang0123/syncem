"""Regression test: the paper's Sec 5.2 / App C.3-C.4 fit rebuilt with the continuum library.

Paper power-law truth (E1), 1% noise with seed 20260928, 21 fitted channels, absolute
SVD threshold s >= 10 (noise sd <= 0.1 in w_b-scaled coordinates).  Compared with the
paper's numbers and with the frozen harmonic caches (inputs/harmonic-radio-fit, read only).
Writes regression_results.json.  Run: python regression.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import basis as B  # noqa: E402
import ensembles as E  # noqa: E402
from config import PAPER  # noqa: E402
from provenance import provenance, write_json  # noqa: E402

T0 = time.time()
HARM = HERE / "inputs" / "harmonic-radio-fit"
H_BASIS, H_TRUTH, H_RESULTS = HARM / "trusted_basis_refined.npz", HARM / "trusted_truth_refined.npz", \
    HARM / "results_final.json"
PAPER_NUMBERS = {"retained": 7, "s7": 10.213, "s8": 5.453, "max_bias_sigma": 0.263,
                 "max_finite_response_discrepancy_over_I": 0.000863, "chi2": 51.12,
                 "fit_max_over_I": 0.0101}
THRESHOLD = 10.0


def solve(cfg, C, S_direct, data_fit):
    """Whitened SVD of the fitted rows; absolute threshold s >= THRESHOLD."""
    use = cfg.fit_idx
    sigma = cfg.noise_sigma(S_direct)[use]
    design = C[use].reshape(-1, C.shape[-1]) / sigma.reshape(-1, 1)
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    keep = s >= THRESHOLD
    est = (vt[keep].T / s[keep]) @ u[:, keep].T
    a_hat = est @ (data_fit / sigma).ravel()
    pred = np.einsum("jsk,k->js", C, a_hat)
    return dict(s=s, keep=keep, u=u[:, keep], vt=vt[keep], a_hat=a_hat, pred=pred, sigma=sigma)


def analyse(cfg, C, S, a_true, noise):
    use = cfg.fit_idx
    sig_all = cfg.noise_sigma(S)
    data = S[use] + noise * sig_all[use]
    f = solve(cfg, C, S, data)
    delta = S - B.predict(C, a_true)
    bias = f["u"].T @ (delta[use] / f["sigma"]).ravel()
    pred0 = B.predict(C, (f["vt"].T @ (np.diag(1 / f["s"][f["keep"]]) @ f["u"].T)) @ (S[use] / f["sigma"]).ravel())
    return {
        "retained": int(f["keep"].sum()),
        "singular_values": f["s"].tolist(),
        "s7_s8": [float(f["s"][6]), float(f["s"][7])],
        "max_finite_response_discrepancy_over_I": float(np.max(np.abs(delta) / S[:, 0:1])),
        "representation_mode_bias_over_sigma": bias.tolist(),
        "max_bias_sigma": float(np.max(np.abs(bias))),
        "chi2_fit_seeded": float(np.sum(((data - f["pred"][use]) / f["sigma"]) ** 2)),
        "residual_dof": int(data.size - f["keep"].sum()),
        "fit_max_over_I_all41": float(np.max(np.abs(f["pred"] - S) / S[:, 0:1])),
        "fit_max_over_I_withheld": float(np.max(np.abs(f["pred"] - S)[cfg.held_idx] / S[cfg.held_idx, 0:1])),
        "noisefree_fit_max_over_I": float(np.max(np.abs(pred0 - S) / S[:, 0:1])),
        "true_modes": (f["vt"] @ a_true.ravel()).tolist(),
        "estimated_modes": (f["vt"] @ f["a_hat"]).tolist(),
        "mode_sigma": (1 / f["s"][f["keep"]]).tolist(),
        "_vt": f["vt"],
    }


def main():
    cfg = PAPER
    e1 = E.PowerLaw()
    timings = {}
    t = time.time()
    S = e1.direct(cfg, n=24)
    a = e1.moments(cfg, 2, n=24)
    timings["direct_E1_s"] = time.time() - t
    t = time.time()
    C = B.build_basis(cfg, 2)
    timings["basis_N2_s"] = time.time() - t
    check = e1.direct_check(cfg, 24)
    noise = cfg.draw_noise(len(cfg.fit_idx))
    cont = analyse(cfg, C, S, a, noise)

    # refined quadrature (channels, B, eta doubled; energy nodes doubled)
    ref_cfg = replace(cfg, n_chan_nodes=800, n_b=32, n_eta=24)
    S_r, a_r = e1.direct(ref_cfg, n=48), e1.moments(ref_cfg, 2, n=48)
    C_r = B.build_basis(ref_cfg, 2)
    cont_ref = analyse(ref_cfg, C_r, S_r, a_r, noise)

    # harmonic caches
    hb, ht = np.load(H_BASIS), np.load(H_TRUTH)
    C_h, S_h, a_h = hb["basis"], ht["truth"], ht["moments"]
    harm = analyse(cfg, C_h, S_h, a_h.reshape(12, 3), noise)
    Ih = S_h[:, 0:1]
    bridge = {
        "truth_max_abs_diff_over_I_IQU": np.max(np.abs(S - S_h) / Ih, axis=0).tolist(),
        "basis_at_true_moments_max_abs_diff_over_I_IQU": np.max(
            np.abs(B.predict(C, a) - B.predict(C_h, a_h)) / Ih, axis=0).tolist(),
        "columns_max_abs_diff_over_I": float(np.max(np.abs(C - C_h) / Ih[:, :, None])),
        "true_moments_max_abs_diff": float(np.max(np.abs(a.ravel() - a_h))),
        "s7_s8_diff": [cont["s7_s8"][0] - harm["s7_s8"][0], cont["s7_s8"][1] - harm["s7_s8"][1]],
        "max_bias_sigma_diff": cont["max_bias_sigma"] - harm["max_bias_sigma"],
        "chi2_diff_same_seed": cont["chi2_fit_seeded"] - harm["chi2_fit_seeded"],
        "retained_projector_diff_spectral_norm": float(np.linalg.norm(
            cont["_vt"].T @ cont["_vt"] - harm["_vt"].T @ harm["_vt"], 2)),
    }
    for d in (cont, cont_ref, harm):
        d.pop("_vt")
    match = {
        "retained_7": cont["retained"] == PAPER_NUMBERS["retained"],
        "s7_abs_diff_from_10.213": abs(cont["s7_s8"][0] - 10.213),
        "s8_abs_diff_from_5.453": abs(cont["s7_s8"][1] - 5.453),
        "max_bias_abs_diff_from_0.263": abs(cont["max_bias_sigma"] - 0.263),
        "continuum_vs_harmonic_truth_below_1e-6_I": bool(max(bridge["truth_max_abs_diff_over_I_IQU"]) < 1e-6),
        "continuum_vs_harmonic_Ca_below_1e-6_I": bool(
            max(bridge["basis_at_true_moments_max_abs_diff_over_I_IQU"]) < 1e-6),
    }
    out = {
        "description": __doc__,
        "seed": cfg.seed, "threshold_s": THRESHOLD, "noise_frac": cfg.noise_frac,
        "paper_numbers": PAPER_NUMBERS,
        "continuum": cont,
        "continuum_refined_quadrature": {k: cont_ref[k] for k in (
            "retained", "s7_s8", "max_bias_sigma", "max_finite_response_discrepancy_over_I", "chi2_fit_seeded")},
        "refinement_shift": {
            "s7": cont_ref["s7_s8"][0] - cont["s7_s8"][0], "s8": cont_ref["s7_s8"][1] - cont["s7_s8"][1],
            "max_bias_sigma": cont_ref["max_bias_sigma"] - cont["max_bias_sigma"],
            "truth_max_change_over_I": float(np.max(np.abs(S_r - S) / S[:, 0:1]))},
        "harmonic_cache_refit": harm,
        "bridge_continuum_minus_harmonic": bridge,
        "match": match,
        "energy_node_doubling": check,
        "config": cfg.describe(),
        "timings_s": timings,
        "provenance": provenance(__file__, T0, inputs=(H_BASIS, H_TRUTH, H_RESULTS)),
    }
    write_json(HERE / "regression_results.json", out)
    print({k: out["continuum"][k] for k in ("retained", "s7_s8", "max_bias_sigma", "chi2_fit_seeded",
                                            "fit_max_over_I_all41", "max_finite_response_discrepancy_over_I")})
    print(match)
    print(bridge)


if __name__ == "__main__":
    main()
