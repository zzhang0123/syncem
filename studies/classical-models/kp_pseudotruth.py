"""E4-KP against a pseudo-truth moment vector: the moments of the JP spectrum with t_eff.

The reference truth of E4-KP in forward.py and fits.py is the effective diagonal moments of
N_eff(gamma) = int deta q(eta) 2 N(gamma, eta).  That vector is not the one the data measure:
an isotropic-pitch JP spectrum with t_eff/t = 1.1475 (params_results.json ->
tasks.E4-KP_JPfamily.reference_point, the exact-forward JP fit to the noise-free KP direct
spectrum) reproduces the KP spectrum to < 1e-4 I_j.  Its N = 2 moments are used here as the
pseudo-truth of E4-KP, and the fit-summary quantities of fits.py are recomputed against them
(LinearFit, seeded_report and realisation_report imported from fits.py; same seeds and 4000
realisations).  The values against that reference truth are recomputed alongside.

Writes kp_pseudotruth_results.json and kp_pseudotruth.npz.  Run: python kp_pseudotruth.py
"""
from __future__ import annotations

import json
import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))
import basis as B  # noqa: E402
import ensembles as E  # noqa: E402
import fits as F  # noqa: E402
import paramlib as P  # noqa: E402
from config import PAPER  # noqa: E402
from provenance import provenance, relative, sha256, write_json  # noqa: E402

T0 = time.time()
CFG = PAPER
PARAMS_JSON = HERE / "params_results.json"
FORWARD_NPZ = HERE / "forward_spectra.npz"


def pseudo_truth(params: dict):
    """(JP ensemble at t_eff, amplitude factor, phi, lib theta) from the params reference point."""
    task = params["tasks"]["E4-KP_JPfamily"]
    phi = np.asarray(task["reference_point"]["phi"], float)
    fam = P.Family("JP", nu_conv=params["nu_conv_ln(a1<B_perp>)"])
    theta = fam.to_lib(phi)[0]  # (A, p, ln gamma_c)
    jp = E.JaffePerola(name="E4-KP pseudo-truth", p=float(theta[1]), gamma_c=float(np.exp(theta[2])))
    scale = float(theta[0] / jp.norm)  # family N = A * shape = scale * unit-column density
    return jp, scale, phi, theta, task


def forward_residuals(S_kp, jp, scale, theta, Ns=(2, 4)) -> dict:
    """max_j |S - C a|/I_j (I, Q, U) for the pseudo-truth moments at each N; JP-vs-KP spectrum."""
    S_jp = scale * jp.direct(CFG)
    I = S_kp[:, :1]
    out = {"max_abs_S_JP_minus_S_KP_over_I_IQU": np.max(np.abs(S_jp - S_kp) / I, axis=0).tolist()}
    for N in Ns:
        C = B.build_basis(CFG, N)
        a = E.jp_moments(CFG, theta, N)[0].ravel()
        a_chk = scale * jp.moments(CFG, N).ravel()
        out[f"N{N}"] = {"max_abs_S_KP_minus_C_a_over_I_IQU": np.max(np.abs(S_kp - B.predict(C, a)) / I, axis=0).tolist(),
                        "max_abs_S_JP_minus_C_a_over_I_IQU": np.max(np.abs(S_jp - B.predict(C, a)) / I, axis=0).tolist(),
                        "family_moments_vs_quadrature_moments_max_abs": float(np.max(np.abs(a - a_chk)))}
    return out, S_jp


def summary(lf, zf, zh, zfR, zhR) -> dict:
    seeded, picard, prod = F.seeded_report(lf, zf, zh, F.R_PICARD)
    real, sc, _ = F.realisation_report(lf, zfR, zhR, seeded["R_star"], picard["per_R"])
    cf, cp = real["coverage_fixed_R"], real["coverage_post_selection"]
    b = np.asarray(seeded["rep_bias_over_sigma"])
    return {
        "R_star_seeded": seeded["R_star"],
        "max_abs_rep_bias_over_sigma": seeded["max_abs_rep_bias_over_sigma"],
        "mode_of_max_bias": int(np.argmax(np.abs(b))) + 1,
        "rep_bias_over_sigma": seeded["rep_bias_over_sigma"],
        "beta_hat_minus_true_over_sigma": seeded["beta_hat_minus_true_over_sigma"],
        "max_abs_beta_dev_sigma": float(np.max(np.abs(seeded["beta_hat_minus_true_over_sigma"]))),
        "eps_diagnostics": seeded["eps_diagnostics"],
        "max_finite_response_discrepancy_over_I": seeded["max_finite_response_discrepancy_over_I"],
        "coverage_fixed_R": {"R": cf["R"], "per_mode_analytic": cf["per_mode_analytic"],
                             "per_mode_empirical": cf["per_mode_empirical"],
                             "per_mode_analytic_min_mean": [float(np.min(cf["per_mode_analytic"])),
                                                            float(np.mean(cf["per_mode_analytic"]))],
                             "per_mode_empirical_min_mean": [float(np.min(cf["per_mode_empirical"])),
                                                             float(np.mean(cf["per_mode_empirical"]))],
                             "joint_ellipsoid_analytic": cf["joint_ellipsoid_analytic"],
                             "joint_ellipsoid_empirical": cf["joint_ellipsoid_empirical"]},
        "coverage_post_selection_joint": cp["joint_ellipsoid_empirical"],
        "n_realisations": real["n_realisations"],
        "hausdorff_true_moments_n_bins_violating": seeded["hausdorff_true_moments_n_bins_violating"],
    }, prod, lf


def main():
    params = json.loads(PARAMS_JSON.read_text())
    jp, scale, phi, theta, task = pseudo_truth(params)
    ens = E.reference_ensembles(CFG)
    kp = ens["E4-KP"]
    S_kp = kp.direct(CFG, n=24, n_eta=16)
    ref = np.load(FORWARD_NPZ)["E4_KP"]
    fwd, S_jp = forward_residuals(S_kp, jp, scale, theta)
    a_pseudo = E.jp_moments(CFG, theta, 2)[0].ravel()
    a_spec = kp.effective_moments(CFG, 2, 24, 16).ravel()
    C = B.build_basis(CFG, 2)
    zf, zh = F.seeded_noise()
    zfR, zhR, _ = F.realisation_noise(F.N_REAL)
    res = {}
    lfs = {}
    for label, a in (("pseudo_truth_JP_t_eff", a_pseudo), ("spec_truth_effective_diagonal", a_spec)):
        lf = F.LinearFit("E4-KP", C, S_kp, a)
        res[label], prod, lfs[label] = summary(lf, zf, zh, zfR, zhR)
    lfp = lfs["pseudo_truth_JP_t_eff"]
    fits_json = json.loads((HERE / "fits_results.json").read_text())["ensembles"]["E4-KP"]["seeded"]
    checks = {
        "S_KP_vs_forward_npz_max_over_I": float(np.max(np.abs(S_kp - ref) / S_kp[:, :1])),
        "spec_truth_bias_vs_fits_json_max_abs": float(np.max(np.abs(
            np.asarray(res["spec_truth_effective_diagonal"]["rep_bias_over_sigma"])
            - np.asarray(fits_json["rep_bias_over_sigma"])))),
        "t_eff_over_t_params_json": task["exact_JP_forward_fit_noisefree"]["t_eff_over_t"],
        "gamma_c_eff_vs_params_json_rel": float(np.exp(theta[2]) / task["exact_JP_forward_fit_noisefree"]["gamma_c"] - 1),
    }
    np.savez(HERE / "kp_pseudotruth.npz", a_pseudo=a_pseudo, a_spec=a_spec, S_kp=S_kp, S_jp=S_jp,
             pred_pseudo=B.predict(C, a_pseudo), eps_pseudo=lfp.eps, rep_bias_sigma_pseudo=lfp.bias,
             beta_pseudo=lfp.beta_true, s=lfp.s, vt=lfp.vt)
    out = {
        "description": __doc__,
        "pseudo_truth": {"source": "params_results.json -> tasks.E4-KP_JPfamily.reference_point",
                         "phi_lnA_p_ln_nu_b": phi.tolist(), "theta_A_p_ln_gamma_c": theta.tolist(),
                         "gamma_c_eff": float(np.exp(theta[2])),
                         "t_eff_over_t": task["exact_JP_forward_fit_noisefree"]["t_eff_over_t"],
                         "column": float(np.sum(CFG.w_b * a_pseudo.reshape(12, 3)[:, 0])),
                         "spec_truth_column": float(np.sum(CFG.w_b * a_spec.reshape(12, 3)[:, 0]))},
        "forward": fwd,
        "fit_summary": res,
        "checks": checks,
        "npz": relative(HERE / "kp_pseudotruth.npz"),
    }
    out["npz_sha256"] = sha256(HERE / "kp_pseudotruth.npz")
    out["provenance"] = provenance(__file__, T0, inputs=(PARAMS_JSON, FORWARD_NPZ, HERE / "fits.py",
                                                         HERE / "paramlib.py", HERE / "fits_results.json"))
    write_json(HERE / "kp_pseudotruth_results.json", out)
    for k, v in res.items():
        print(k, "max|Ke|", round(v["max_abs_rep_bias_over_sigma"], 3), "mode", v["mode_of_max_bias"],
              "cov fixed analytic", round(v["coverage_fixed_R"]["joint_ellipsoid_analytic"], 3),
              "post", v["coverage_post_selection_joint"], "per-mode min/mean", v["coverage_fixed_R"]["per_mode_analytic_min_mean"])
    print(json.dumps(fwd, indent=1))
    print(json.dumps(checks, indent=1))
    print("done", time.time() - T0)


if __name__ == "__main__":
    main()
