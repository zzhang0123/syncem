"""Harmonic-versus-continuum bridge for the paper's linear radio basis (Sec. 4.3 fit).

Loads the frozen harmonic response matrix and truth of the manuscript fit
(inputs/harmonic-radio-fit, read only), rebuilds the same matrix with the
Rybicki-Lightman continuum kernels (bridge_continuum.py), and reports:
column, at-true-moment and direct-truth differences; singular values; retained
counts; principal angles; retained combinations and biases; the per-gamma error
function c(x) gamma^-2 from bridge_harmonic_probe.npz; and an empirical envelope
E_j propagated through |K| E.

Run: python bridge.py
Needs bridge_harmonic_probe.npz (python bridge_harmonic_probe.py) and, if present,
bridge_mpmath_check.json (python bridge_mpmath_check.py).
Writes bridge_results.json and figures/bridge_error_function.{pdf,png}.
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402
import numpy as np  # noqa: E402
import scipy  # noqa: E402

import bridge_continuum as bc  # noqa: E402

HERE = Path(__file__).resolve().parent
LIB = HERE / "lib"
RADIO = HERE / "inputs" / "harmonic-radio-fit"
INPUTS = ("results_final.json", "trusted_basis_refined.npz", "trusted_truth_refined.npz",
          "trusted_basis.npz", "trusted_basis_halfstep.npz", "trusted_truth.npz",
          "finalize.py", "radio_numerics.py", "stable_lines.py")
NOISE, THRESHOLD, SEED = 0.01, 10.0, 20260928
USE = np.arange(0, 41, 2)
A_B5 = 3 * bc.E_ESU * bc.B0 / (4 * np.pi * bc.M_E * bc.C_CGS)  # x_ref = nu/(A_B5 gamma^2)
X_VALID = 30.0  # probe: harmonic omits Airy-truncated lines visibly above x ~ 40
TAIL_FLOOR = 1e-12  # ignore channels with K_I below this fraction of the gamma's peak
MARGIN = 1.25  # declared safety factor on the empirical envelope
WINDOW_DEX = 0.1


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    """Path relative to this directory, for the provenance block."""
    return Path(path).resolve().relative_to(HERE).as_posix()


def rel_max(a, b) -> float:
    return float(np.max(np.abs(a - b)) / np.max(np.abs(b)))


# ----------------------------------------------------------------------------- harmonic


def solve(c, data, sigma):
    """Whitened truncated-SVD fit, identical to finalize.solve()."""
    design = c[USE].reshape(-1, 36) / sigma.reshape(-1, 1)
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    keep = s >= THRESHOLD
    estimator = (vt[keep].T / s[keep]) @ u[:, keep].T
    return {"u": u[:, keep], "s": s, "vt": vt[keep], "keep": int(keep.sum()),
            "estimator": estimator, "estimate": estimator @ (data / sigma).ravel(),
            "u_all": u, "vt_all": vt}


def load_harmonic() -> tuple[dict, dict]:
    report = json.loads((RADIO / "results_final.json").read_text())
    arr = {k: np.asarray(v) for k, v in report["arrays"].items()}
    npz = {name: np.load(RADIO / name) for name in INPUTS if name.endswith(".npz")}
    basis, truth = npz["trusted_basis_refined.npz"], npz["trusted_truth_refined.npz"]
    pure = basis["pure_coefficients"]
    layout = max(rel_max(basis["basis"][:, :, 3 * b + r], bc.REF_BIN_MASS[b] * pure[b, r])
                 for b in range(12) for r in range(3))
    cfg = report["configuration"]
    sigma = NOISE * arr["truth_IQU"][USE, 0, None] * np.ones((1, 3))
    draws = np.random.default_rng(SEED).normal(size=sigma.shape)
    checks = {
        "json_vs_npz_response_max_abs": float(np.max(np.abs(arr["response_matrix"] - basis["basis"]))),
        "json_vs_npz_truth_max_abs": float(np.max(np.abs(arr["truth_IQU"] - truth["truth"]))),
        "json_vs_npz_moments_max_abs": float(np.max(np.abs(arr["true_moments"] - truth["moments"]))),
        "layout_column_3b_plus_r_equals_binmass_times_pure_max_rel": layout,
        "config_reference_bin_masses_max_abs": float(np.max(np.abs(np.asarray(cfg["reference_bin_masses"]) - bc.REF_BIN_MASS))),
        "config_log_edges_max_abs": float(np.max(np.abs(np.asarray(cfg["log_bin_edges"]) - bc.LOG_EDGES))),
        "config_centres_max_rel": float(np.max(np.abs(np.asarray(cfg["channel_centres_Hz"]) / bc.CENTRES - 1))),
        "fit_channels_are_even_indices": bool(np.array_equal(arr["fit_channel_indices"], USE)),
        "noise_sigma_max_abs_rel": float(np.max(np.abs(arr["noise_sigma_IQU"] / sigma - 1))),
        "noisy_data_reproduced_max_abs_over_sigma": float(np.max(np.abs(arr["noisy_data_IQU"] - (truth["truth"][USE] + draws * sigma)) / sigma)),
    }
    h = {"C": basis["basis"], "S": truth["truth"], "a": truth["moments"], "pure": pure,
         "data": arr["noisy_data_IQU"], "sigma": sigma, "draws": draws,
         "pure_coarse": npz["trusted_basis.npz"]["pure_coefficients"],
         "pure_halfstep": npz["trusted_basis_halfstep.npz"]["pure_coefficients"],
         "S_coarse": npz["trusted_truth.npz"]["truth"], "fit_json": report["fit"]}
    fit = solve(h["C"], h["data"], sigma)
    delta = h["S"] - np.einsum("jsk,k->js", h["C"], h["a"])
    bias = fit["u"].T @ (delta[USE] / sigma).ravel()
    checks["refit_singular_values_max_rel"] = float(np.max(np.abs(fit["s"] / np.asarray(report["fit"]["singular_values"]) - 1)))
    checks["refit_retained_count"] = fit["keep"]
    checks["refit_representation_bias_max_abs_diff"] = float(np.max(np.abs(bias - np.asarray(report["fit"]["representation_mode_bias_over_sigma"]))))
    checks["refit_estimate_max_abs_diff"] = float(np.max(np.abs(fit["estimate"] - np.asarray(report["fit"]["estimated_moments"]))))
    return h, checks


# ---------------------------------------------------------------------------- continuum


def build_continuum() -> tuple[dict, dict, dict]:
    t0 = time.perf_counter()
    kern = {"adopted": bc.ContinuumKernel(8, 7, 400), "coarse": bc.ContinuumKernel(6, 5, 400),
            "converged": bc.ContinuumKernel(20, 20, 400)}
    pure = {"fd04": bc.fd_pure(kern["adopted"], 0.04), "fd02": bc.fd_pure(kern["adopted"], 0.02),
            "fd01": bc.fd_pure(kern["adopted"], 0.01), "an": bc.analytic_pure(kern["adopted"]),
            "coarse_fd04": bc.fd_pure(kern["coarse"], 0.04), "coarse_fd02": bc.fd_pure(kern["coarse"], 0.02),
            "converged_an": bc.analytic_pure(kern["converged"])}
    truth = {"adopted_e8": bc.truth_and_moments(kern["adopted"], 8),
             "adopted_e32": bc.truth_and_moments(kern["adopted"], 32),
             "coarse_e6": bc.truth_and_moments(kern["coarse"], 6),
             "converged_e8": bc.truth_and_moments(kern["converged"], 8)}
    gs = (float(bc.GAMMA_MIDS[0]), float(bc.GAMMA_MIDS[6]), float(bc.GAMMA_MIDS[11]))
    ch800 = bc.ContinuumKernel(8, 7, 800)
    sa, sc = truth["adopted_e8"][0], truth["converged_e8"][0]
    quad = {
        "channel_nodes_400_vs_800_max_over_max": max(rel_max(kern["adopted"](g), ch800(g)) for g in gs),
        "B_eta_nodes_8x7_vs_20x20_truth_max_over_local_I": float(np.max(np.abs(sa - sc) / sc[:, :1])),
        "energy_nodes_8_vs_32_truth_max_over_local_I": float(np.max(np.abs(sa - truth["adopted_e32"][0]) / sa[:, :1])),
        "energy_nodes_8_vs_32_moments_max_abs": float(np.max(np.abs(truth["adopted_e8"][1] - truth["adopted_e32"][1]))),
        "analytic_vs_fd04_max_over_column_max": col_stats(pure["fd04"] - pure["an"], pure["an"])["max"],
        "analytic_vs_fd01_max_over_column_max": col_stats(pure["fd01"] - pure["an"], pure["an"])["max"],
        "seconds": time.perf_counter() - t0,
    }
    return kern, pure, truth | {"quadrature": quad}


# ------------------------------------------------------------------------------ metrics


def col_stats(dpure, ref_pure) -> dict:
    """max_j |dD| / max_j |D_I| per (bin, r, component); D are (12,3,J,3) Taylor coefficients."""
    scale = np.max(np.abs(ref_pure[:, :, :, 0]), axis=2)  # (12,3)
    per = np.max(np.abs(dpure), axis=2) / scale[:, :, None]  # (12,3,3)
    return {"max": float(per.max()), "per_component_IQU": per.max(axis=(0, 1)).tolist(),
            "table_bin_r": per.max(axis=2).tolist(),
            "gamma_b2_times_table_bin_r": (per.max(axis=2) * bc.GAMMA_MIDS[:, None] ** 2).tolist(),
            "x_ref_at_r0_column_max_I_per_bin": (
                bc.CENTRES[np.argmax(np.abs(ref_pure[:, 0, :, 0]), axis=1)] / (A_B5 * bc.GAMMA_MIDS**2)).tolist()}


def at_moments(dC, a, S) -> dict:
    d = np.abs(np.einsum("jsk,k->js", dC, a)) / S[:, :1]
    return {"max_over_I_IQU": d.max(axis=0).tolist(), "max_over_I": float(d.max()),
            "max_over_I_fit_channels": float(d[USE].max())}


def truth_diff(S1, S2, Sref) -> dict:
    d = np.abs(S1 - S2) / Sref[:, :1]
    p = np.hypot(S1[:, 1] - S2[:, 1], S1[:, 2] - S2[:, 2]) / Sref[:, 0]
    return {"max_over_I_IQU": d.max(axis=0).tolist(), "max_absP_over_I": float(p.max()),
            "argmax_channel_I": int(np.argmax(d[:, 0])), "per_channel_over_I": d.tolist()}


def principal_sines(a, b) -> list:
    """Sines of the principal angles between row spaces of orthonormal a, b (k x n)."""
    resid = b.T - a.T @ (a @ b.T)
    return np.sort(np.linalg.svd(resid, compute_uv=False)).tolist()


def align(fit, ref) -> dict:
    sign = np.sign(np.sum(fit["vt"] * ref["vt"], axis=1))
    return fit | {"vt": fit["vt"] * sign[:, None], "u": fit["u"] * sign[None, :],
                  "estimator": fit["estimator"]}


def combos(fit, a_true, S, C, sigma) -> dict:
    delta = S - np.einsum("jsk,k->js", C, a_true)
    mode_sigma = 1 / fit["s"][: fit["keep"]]
    true, est = fit["vt"] @ a_true, fit["vt"] @ fit["estimate"]
    return {"retained": fit["keep"], "true_modes": true.tolist(), "estimated_modes": est.tolist(),
            "mode_sigma": mode_sigma.tolist(),
            "estimate_minus_true_over_sigma": ((est - true) / mode_sigma).tolist(),
            "representation_bias_over_sigma": (fit["u"].T @ (delta[USE] / sigma).ravel()).tolist(),
            "finite_representation_max_over_I": float(np.max(np.abs(delta) / S[:, :1]))}


def compare_fits(h, cont_pure, S_c) -> dict:
    """Matrix comparison at common sigma, continuum pipeline, and cross analyses."""
    C_c = bc.basis_from_pure(cont_pure)
    fh = solve(h["C"], h["data"], h["sigma"])
    fc = align(solve(C_c, h["data"], h["sigma"]), fh)  # harmonic data, continuum basis
    sig_c = NOISE * S_c[USE, 0, None] * np.ones((1, 3))
    data_c = S_c[USE] + h["draws"] * sig_c
    fcc = align(solve(C_c, data_c, sig_c), fh)  # all-continuum pipeline
    fhc = solve(h["C"], data_c, sig_c)  # continuum data, harmonic basis
    sh, sc = fh["s"], fc["s"]
    shift = fh["vt"] @ (fc["estimate"] - fh["estimate"]) * sh[: fh["keep"]]
    shift_c = fh["vt"] @ (fhc["estimate"] - fh["estimate"]) * sh[: fh["keep"]]
    return {
        "singular_values_harmonic": sh.tolist(), "singular_values_continuum": sc.tolist(),
        "singular_values_rel_diff_first10": (np.abs(sc[:10] / sh[:10] - 1)).tolist(),
        "singular_values_continuum_own_sigma": fcc["s"].tolist(),
        "s7_s8_harmonic": [float(sh[6]), float(sh[7])], "s7_s8_continuum": [float(sc[6]), float(sc[7])],
        "retained_harmonic": fh["keep"], "retained_continuum": fc["keep"],
        "retained_continuum_own_sigma": fcc["keep"],
        "principal_sines_coefficient_space": principal_sines(fh["vt"], fc["vt"]),
        "principal_sines_data_space": principal_sines(fh["u"].T, fc["u"].T),
        "retained_projector_difference_spectral_norm": float(np.linalg.norm(fh["vt"].T @ fh["vt"] - fc["vt"].T @ fc["vt"], 2)),
        "harmonic_pipeline": combos(fh, h["a"], h["S"], h["C"], h["sigma"]),
        "continuum_pipeline_own_truth_same_draws": combos(fcc, h["a"], S_c, C_c, sig_c),
        "harmonic_data_continuum_basis": combos(fc, h["a"], h["S"], C_c, h["sigma"]),
        "shift_harmonic_modes_continuum_basis_vs_harmonic_basis_over_sigma": shift.tolist(),
        "shift_harmonic_modes_continuum_data_vs_harmonic_data_over_sigma": shift_c.tolist(),
        "_fits": (fh, fc),
    }


# ------------------------------------------------------------------------ error function


def probe_c(probe, kern) -> dict:
    gam = probe["gamma"]
    xg = bc.CENTRES[None, :] / (A_B5 * gam[:, None] ** 2)
    cont = np.stack([kern(float(g)) for g in gam])
    ok = (cont[:, :, 0] > TAIL_FLOOR * cont[:, :, :1].max(axis=1)) & (xg <= X_VALID)
    out = {"gamma": gam, "x": xg, "cont": cont, "ok": ok}
    for label in ("adopted", "refined", "coarse"):
        h = probe[label]
        g2 = gam[:, None] ** 2
        with np.errstate(divide="ignore", invalid="ignore"):
            out[f"cI_{label}"] = g2 * (h[:, :, 0] - cont[:, :, 0]) / cont[:, :, 0]
            out[f"cP_{label}"] = g2 * np.hypot(h[:, :, 1] - cont[:, :, 1], h[:, :, 2] - cont[:, :, 2]) / cont[:, :, 0]
    return out


def refined2_convergence(probe, pc) -> dict:
    """gamma^2 |K_refined - K_refined2| / K_cont (I and |P|) on the selected points, per gamma."""
    out = {}
    for i, g in enumerate(probe["refined2_gamma"]):
        k = int(np.argmin(np.abs(pc["gamma"] - g)))
        ok, c = pc["ok"][k], pc["cont"][k]
        if not np.any(ok):
            continue
        d = probe["refined"][k] - probe["refined2"][i]
        out[f"{g:.0f}"] = [float(g * g * np.max(np.abs(d[ok, 0]) / c[ok, 0])),
                           float(g * g * np.max(np.hypot(d[ok, 1], d[ok, 2]) / c[ok, 0]))]
    return out


def equal_x_exponents(pc) -> dict:
    """gamma_{k+1}/gamma_k = 10^0.1 shifts x by four channels exactly: test the gamma^-2 law."""
    rel = pc["cI_refined"] / pc["gamma"][:, None] ** 2
    ratio = np.log(10**0.1)
    exps, xs = [], []
    for k in range(len(pc["gamma"]) - 1):
        for j in range(41 - 4):
            if pc["ok"][k, j] and pc["ok"][k + 1, j + 4] and abs(pc["cI_refined"][k, j]) > 0:
                exps.append(np.log(rel[k, j] / rel[k + 1, j + 4]) / ratio)
                xs.append(pc["x"][k, j])
    exps = np.asarray(exps)
    return {"n_pairs": int(exps.size), "median": float(np.median(exps)),
            "p05_p95": np.percentile(exps, [5, 95]).tolist(), "min_max": [float(exps.min()), float(exps.max())],
            "x_range": [float(min(xs)), float(max(xs))]}


def fit_cI(pc) -> dict:
    """Least squares c_I(x) = -(alpha x^-2/3 + beta + delta x) on the refined probe."""
    ok = pc["ok"] & np.isfinite(pc["cI_refined"])
    x, c = pc["x"][ok], pc["cI_refined"][ok]
    design = -np.stack([x ** (-2 / 3), np.ones_like(x), x], axis=1) / np.abs(c)[:, None]
    coef, *_ = np.linalg.lstsq(design, c / np.abs(c), rcond=None)
    model = -(coef[0] * x ** (-2 / 3) + coef[1] + coef[2] * x)
    res = np.abs(model / c - 1)
    return {"alpha_beta_delta": coef.tolist(), "max_rel_residual": float(res.max()),
            "median_rel_residual": float(np.median(res)), "x_range": [float(x.min()), float(x.max())],
            "all_negative": bool(np.all(c < 0))}


def cI_model(coef, x):
    return -(coef[0] * x ** (-2 / 3) + coef[1] + coef[2] * x)


def envelope(pc, key) -> dict:
    """Windowed upper envelope of |c| over adopted and refined probe points, times MARGIN."""
    vals = np.concatenate([np.abs(pc[f"{key}_adopted"][pc["ok"]]), np.abs(pc[f"{key}_refined"][pc["ok"]])])
    lx = np.log10(np.concatenate([pc["x"][pc["ok"]]] * 2))
    edges = np.arange(np.floor(lx.min() / WINDOW_DEX) * WINDOW_DEX, lx.max() + WINDOW_DEX, WINDOW_DEX)
    idx = np.clip(np.digitize(lx, edges) - 1, 0, edges.size - 2)
    peak = np.array([vals[idx == i].max() if np.any(idx == i) else np.nan for i in range(edges.size - 1)])
    good = np.isfinite(peak)
    centres = (edges[:-1] + edges[1:]) / 2
    peak = np.where(good, peak, np.interp(centres, centres[good], peak[good]))
    hull = np.array([peak[max(0, i - 1): i + 2].max() for i in range(peak.size)])
    note = "windowed max of |c| (adopted and refined probe), +-1 window, times margin"
    if key == "cP":
        # Small-x P data come only from Faraday-depolarised channels; do not let the
        # envelope fall towards small x: continue the x < 1 peak as x^-2/3 (declared choice).
        small = centres < 0
        i_pk = int(np.argmax(np.where(small, hull, -np.inf)))
        ext = hull[i_pk] * 10 ** ((2 / 3) * (centres[i_pk] - centres))
        hull = np.where(small & (centres < centres[i_pk]), np.maximum(hull, ext), hull)
        note += "; below the x<1 peak continued as x^-2/3"
    return {"log10x_nodes": centres.tolist(), "c_env": (MARGIN * hull).tolist(),
            "margin": MARGIN, "window_dex": WINDOW_DEX, "x_valid_max": X_VALID, "construction": note}


def c_env_eval(env, x):
    """Log-log interpolation; x^-2/3 extrapolation below, cap handled by caller above X_VALID."""
    lxn, cn = np.asarray(env["log10x_nodes"]), np.asarray(env["c_env"])
    lx = np.log10(x)
    inside = np.interp(lx, lxn, np.log10(np.maximum(cn, 1e-300)))
    below = np.log10(cn[0]) - (2 / 3) * (lx - lxn[0])
    return 10 ** np.where(lx < lxn[0], below, inside)


def envelope_E(env_i, env_p, node_kernels, gammas, weights) -> tuple[np.ndarray, np.ndarray]:
    """E_X,j = sum_n w_n min(c_env(x_jn)/gamma_n^2, cap) K_I,j(gamma_n); cap 1 (I), 2 (P)."""
    x = bc.CENTRES[None, :] / (A_B5 * gammas[:, None] ** 2)
    g2 = gammas[:, None] ** 2
    rel_i = np.where(x <= X_VALID, np.minimum(c_env_eval(env_i, x) / g2, 1.0), 1.0)
    rel_p = np.where(x <= X_VALID, np.minimum(c_env_eval(env_p, x) / g2, 2.0), 2.0)
    k_i = node_kernels[:, :, 0]
    return np.einsum("n,nj->j", weights, rel_i * k_i), np.einsum("n,nj->j", weights, rel_p * k_i)


def holdout(h, cont_pure, env_i, env_p) -> dict:
    """Bin-centre kernels (gamma_b not on the probe grid) against the envelope."""
    xs = bc.CENTRES[None, :] / (A_B5 * bc.GAMMA_MIDS[:, None] ** 2)
    kh, kc = h["pure"][:, 0], cont_pure[:, 0]
    g2 = bc.GAMMA_MIDS[:, None] ** 2
    ok = (kc[:, :, 0] > TAIL_FLOOR * kc[:, :, :1].max(axis=1)) & (xs <= X_VALID)
    ci = np.abs(g2 * (kh[:, :, 0] - kc[:, :, 0]) / kc[:, :, 0])
    cp = g2 * np.hypot(kh[:, :, 1] - kc[:, :, 1], kh[:, :, 2] - kc[:, :, 2]) / kc[:, :, 0]
    return {"n_points": int(ok.sum()),
            "max_cI_over_env": float(np.max(ci[ok] / c_env_eval(env_i, xs[ok]))),
            "max_cP_over_env": float(np.max(cp[ok] / c_env_eval(env_p, xs[ok]))),
            "max_abs_cI_bin_centres": float(ci[ok].max()), "max_cP_bin_centres": float(cp[ok].max())}


def propagate(h, fits, E_i, E_p, dS, dCa) -> dict:
    """|K| E bounds for retained modes (sigma units) and for the 36 moments."""
    fh, fc = fits
    rows = np.stack([E_i, E_p, E_p], axis=1)[USE] / h["sigma"]
    out = {}
    for name, fit in (("harmonic_basis", fh), ("continuum_basis", fc)):
        bound = np.abs(fit["u"].T) @ rows.ravel()
        actual = fit["u"].T @ (dS[USE] / h["sigma"]).ravel()
        actual_c = fit["u"].T @ (dCa[USE] / h["sigma"]).ravel()
        mom_b = np.abs(fit["estimator"]) @ rows.ravel()
        mom_a = fit["estimator"] @ (dS[USE] / h["sigma"]).ravel()
        out[name] = {"mode_bound_over_sigma": bound.tolist(),
                     "mode_actual_truth_diff_over_sigma": actual.tolist(),
                     "mode_actual_basis_diff_at_true_moments_over_sigma": actual_c.tolist(),
                     "max_mode_bound_over_sigma": float(bound.max()),
                     "bound_holds_truth": bool(np.all(np.abs(actual) <= bound)),
                     "bound_holds_basis": bool(np.all(np.abs(actual_c) <= bound)),
                     "moment_bound_max": float(mom_b.max()), "moment_actual_max": float(np.abs(mom_a).max()),
                     "moment_bound_holds": bool(np.all(np.abs(mom_a) <= mom_b))}
    return out


def c_eff_power_law(coef, p=bc.SLOPE) -> float:
    """[derived] For N ~ gamma^-p on (0, inf) and one B_perp: dS_I/I = c_eff <gamma^-2>,
    c_eff = int x^(mu+1) c_I(x) F dx / int x^(mu+1) F dx, mu = (p-3)/2 (c_I from the fit)."""
    lx = np.linspace(np.log(1e-5), np.log(200.0), 20001)
    x = np.exp(lx)
    w = x ** ((p - 3) / 2 + 2) * bc.special(x, "F")  # extra x from dx = x dlnx
    return float(np.trapezoid(w * cI_model(coef, x), lx) / np.trapezoid(w, lx))


def compact_form(dS, S, node_kernels, gammas, weights, E_i, E_p, coef) -> dict:
    """dS_j / I_j = c_eff,j <gamma^-2>_j with <gamma^-2>_j the I-weighted mean."""
    mean_g2 = np.einsum("n,nj->j", weights * gammas**-2.0, node_kernels[:, :, 0]) / S[:, 0]
    ci = dS[:, 0] / S[:, 0] / mean_g2  # signed: harmonic minus continuum
    cp = np.hypot(dS[:, 1], dS[:, 2]) / S[:, 0] / mean_g2
    ei, ep = E_i / S[:, 0] / mean_g2, E_p / S[:, 0] / mean_g2
    return {"mean_inv_gamma2_per_channel": mean_g2.tolist(),
            "gamma_eff_per_channel": (mean_g2**-0.5).tolist(),
            "c_eff_I_per_channel": ci.tolist(), "c_eff_P_per_channel": cp.tolist(),
            "c_eff_I_range": [float(ci.min()), float(ci.max())],
            "c_eff_P_range": [float(cp.min()), float(cp.max())],
            "c_eff_I_predicted_from_fit_unbounded_power_law": c_eff_power_law(coef),
            "E_I_over_meang2_I_range": [float(ei.min()), float(ei.max())],
            "E_P_over_meang2_I_range": [float(ep.min()), float(ep.max())]}


def stencil_checks(h, cpure, probe) -> dict:
    """Harmonic half-step change vs the continuum's (coarse B/eta nodes), and the
    high-gamma columns rebuilt with refined harmonic pitch/order quadrature."""
    dh = h["pure_halfstep"] - h["pure_coarse"]
    dc = cpure["coarse_fd02"] - cpure["coarse_fd04"]
    bins = probe["stencil_bins"]
    ref = cpure["fd04"][bins]
    scale = np.max(np.abs(ref[:, :, :, 0]), axis=2)[:, :, None]
    table = lambda d: (np.max(np.abs(d), axis=2) / scale).max(axis=2).tolist()  # noqa: E731
    return {"harmonic_halfstep_minus_step04": col_stats(dh, h["pure_coarse"]),
            "continuum_halfstep_minus_step04": col_stats(dc, cpure["coarse_fd04"]),
            "difference_of_the_two": col_stats(dh - dc, h["pure_coarse"]),
            "high_gamma_bins": bins.tolist(),
            "adopted_harmonic_minus_continuum_table_bin_r": table(h["pure"][bins] - ref),
            "refined_harmonic_minus_continuum_table_bin_r": table(probe["stencil_refined_pure"] - ref),
            "adopted_minus_refined_harmonic_table_bin_r": table(h["pure"][bins] - probe["stencil_refined_pure"])}


def predicted_columns(kern, coef, h, cpure) -> dict:
    """Stencil derivative of the model dK_I = gamma^-2 c_I(x) K_I against measured columns."""
    step = 0.04
    pred = np.zeros((12, 3, len(bc.CENTRES)))
    for b, (mid, half) in enumerate(zip(bc.LOG_MIDS, bc.LOG_HALF)):
        v = []
        for o in range(-2, 3):
            g = float(np.exp(mid + half * step * o))
            x = bc.CENTRES / (A_B5 * g * g)
            v.append(cI_model(coef, x) / g**2 * kern(g)[:, 0])
        v = np.asarray(v)
        pred[b, 0] = v[2]
        pred[b, 1] = (v[0] - 8 * v[1] + 8 * v[3] - v[4]) / (12 * step)
        pred[b, 2] = (-v[0] + 16 * v[1] - 30 * v[2] + 16 * v[3] - v[4]) / (24 * step**2)
    meas = h["pure"][:, :, :, 0] - cpure["fd04"][:, :, :, 0]
    scale = np.max(np.abs(cpure["fd04"][:, :, :, 0]), axis=2)
    pm = np.max(np.abs(pred), axis=2) / scale
    mm = np.max(np.abs(meas), axis=2) / scale
    rm = np.max(np.abs(meas - pred), axis=2) / scale
    return {"predicted_table_bin_r": pm.tolist(), "measured_table_bin_r": mm.tolist(),
            "residual_table_bin_r": rm.tolist(),
            "residual_over_measured_max_per_r": (rm / mm).max(axis=0).tolist()}


# ------------------------------------------------------------------------------- figure


def figure(pc, coef, env_i, env_p, diffs, E_i, E_p, S_h, fits) -> list:
    sys.path.insert(0, str(LIB))
    import figure_style  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    figure_style.configure()
    fig, axes = plt.subplots(2, 2, figsize=(figure_style.WIDTH, 4.6), layout="constrained")
    cmap = plt.get_cmap("cividis")
    lg = np.log10(pc["gamma"])
    norm = matplotlib.colors.Normalize(lg.min(), lg.max())
    xx = np.geomspace(4e-4, X_VALID, 300)
    for ax, key, env, lab in ((axes[0, 0], "cI", env_i, r"$\gamma^2\,|K^{\rm harm}_I/K^{\rm cont}_I-1|$"),
                              (axes[0, 1], "cP", env_p, r"$\gamma^2\,|\Delta P|/K^{\rm cont}_I$")):
        ok = pc["ok"]
        ax.scatter(pc["x"][ok], np.abs(pc[f"{key}_adopted"][ok]), s=3, color="0.7", lw=0, label="adopted quadrature")
        sc = ax.scatter(pc["x"][ok], np.abs(pc[f"{key}_refined"][ok]), s=5, c=lg[np.nonzero(ok)[0]],
                        cmap=cmap, norm=norm, lw=0, label="refined quadrature")
        ax.plot(xx, c_env_eval(env, xx), color="#D55E00", lw=1.0, ls="--", label=r"envelope $c_{\rm env}$")
        if key == "cI":
            ax.plot(xx, np.abs(cI_model(coef, xx)), color="k", lw=0.7, label=r"fit $|c_I(x)|$")
        ax.set(xscale="log", yscale="log", xlabel=r"$x=\nu_j/(a_B\gamma^2)$, $B_\perp=5\,\mu$G", ylabel=lab)
    axes[0, 0].set_ylim(0.3, 3000)
    axes[0, 1].set_ylim(1e-4, 3000)
    axes[0, 0].legend(frameon=False, loc="upper center", fontsize=8, markerscale=2.5, ncol=2,
                      columnspacing=0.8, handletextpad=0.3)
    cb = fig.colorbar(sc, ax=axes[0, :], pad=0.01, aspect=30)
    cb.set_label(r"$\log_{10}\gamma$")
    ax = axes[1, 0]
    nu = bc.CENTRES / 1e9
    ax.plot(nu, diffs["truth_I"], color="#0072B2", lw=1.2, label=r"$|S^{\rm harm}_I-S^{\rm cont}_I|$")
    ax.plot(nu, diffs["truth_P"], color="#009E73", lw=1.2, label=r"$|\Delta S_P|$")
    ax.plot(nu, diffs["basis_I"], color="#0072B2", lw=0.9, ls=":", label=r"$|[(C^{\rm harm}-C^{\rm cont})a]_I|$")
    ax.plot(nu, E_i / S_h[:, 0], color="#D55E00", lw=1.0, ls="--", label=r"$E_{I,j}$")
    ax.plot(nu, E_p / S_h[:, 0], color="#CC79A7", lw=1.0, ls="--", label=r"$E_{P,j}$")
    ax.axhline(NOISE, color="0.55", lw=0.8, ls=":")
    ax.text(0.105, NOISE * 1.6, r"noise $\sigma_j/I_j$", color="0.4", fontsize=8)
    ax.plot(nu, diffs["representation"], color="0.55", lw=0.8)
    ax.text(0.105, 2.2 * diffs["representation"][0], r"$|S-Ca|/I_j$ (paper)", color="0.4", fontsize=8)
    ax.set(xscale="log", yscale="log", xlabel=r"Channel centre $\nu_j$ [GHz]", ylabel=r"Difference $/\,I_j$",
           ylim=(1e-14, 0.3))
    ax.set_xticks([0.1, 1, 10], labels=["0.1", "1", "10"])
    ax.legend(frameon=False, fontsize=8, ncol=2, loc="lower right", columnspacing=0.8, handlelength=1.6)
    ax = axes[1, 1]
    fh, fc = fits
    k = np.arange(1, 37)
    ax.plot(k, fh["s"], "o", ms=3.2, color="#0072B2", mfc="none", label="harmonic")
    ax.plot(k, fc["s"], "x", ms=3.2, color="#D55E00", label="continuum")
    ax.axhline(THRESHOLD, color="0.6", lw=0.7, ls="--")
    ax.set(yscale="log", xlabel="Singular-value index", ylabel=r"Whitened singular value $s_A$", ylim=(1e-17, 1e3))
    ax.legend(frameon=False, fontsize=8)
    for i, a in enumerate(axes.flat):
        a.set_title(f"({chr(97 + i)})", loc="left")
    out = HERE / "figures"
    out.mkdir(exist_ok=True)
    fig.set_figwidth(figure_style.WIDTH)
    from matplotlib.text import Text  # noqa: PLC0415

    for label in fig.findobj(Text):
        label.set_fontsize(min(9, max(8, label.get_fontsize())))
    paths = [out / "bridge_error_function.pdf", out / "bridge_error_function.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=200)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------------- main


def strip(obj):
    if isinstance(obj, dict):
        return {k: strip(v) for k, v in obj.items() if not k.startswith("_")}
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (list, tuple)):
        return [strip(v) for v in obj]
    return obj


def main() -> None:
    t0 = time.perf_counter()
    h, load_checks = load_harmonic()
    kern, cpure, ctruth = build_continuum()
    S_c, a_c, nodes_k, nodes_g, nodes_w = ctruth["adopted_e8"]
    S_cv = ctruth["converged_e8"][0]
    C_fd, C_an = bc.basis_from_pure(cpure["fd04"]), bc.basis_from_pure(cpure["an"])
    probe = np.load(HERE / "bridge_harmonic_probe.npz")
    probe_meta = json.loads((HERE / "bridge_harmonic_probe.json").read_text())
    for name in ("E_ESU", "M_E", "C_CGS", "C_SI_M"):
        if probe_meta["constants"][name] != getattr(bc, name):
            raise ValueError(f"constant {name} differs from the syncem.constants value used by the harmonic probe")
    columns = {
        "harmonic_minus_continuum_fd04_matched": col_stats(h["pure"] - cpure["fd04"], h["pure"]),
        "harmonic_minus_continuum_analytic_matched": col_stats(h["pure"] - cpure["an"], h["pure"]),
        "continuum_fd04_minus_analytic_stencil_only": col_stats(cpure["fd04"] - cpure["an"], cpure["an"]),
        "continuum_analytic_matched_minus_converged_B_eta": col_stats(cpure["an"] - cpure["converged_an"], cpure["an"]),
        "harmonic_minus_continuum_analytic_converged": col_stats(h["pure"] - cpure["converged_an"], h["pure"]),
        "harmonic_coarse_minus_continuum_fd04_coarse_matched": col_stats(h["pure_coarse"] - cpure["coarse_fd04"], h["pure_coarse"]),
    }
    moments = {
        "harmonic_minus_continuum_fd04_matched": at_moments(h["C"] - C_fd, h["a"], h["S"]),
        "harmonic_minus_continuum_analytic_matched": at_moments(h["C"] - C_an, h["a"], h["S"]),
        "continuum_fd04_minus_analytic": at_moments(C_fd - C_an, h["a"], h["S"]),
        "harmonic_minus_continuum_analytic_converged": at_moments(h["C"] - bc.basis_from_pure(cpure["converged_an"]), h["a"], h["S"]),
        "continuum_vs_harmonic_true_moments_max_abs": float(np.max(np.abs(a_c - h["a"]))),
    }
    truths = {
        "harmonic_minus_continuum_matched": truth_diff(h["S"], S_c, h["S"]),
        "harmonic_minus_continuum_converged": truth_diff(h["S"], S_cv, h["S"]),
        "harmonic_coarse_minus_continuum_coarse_matched": truth_diff(h["S_coarse"], ctruth["coarse_e6"][0], h["S"]),
        "harmonic_coarse_minus_adopted": truth_diff(h["S_coarse"], h["S"], h["S"]),
    }
    fits_fd = compare_fits(h, cpure["fd04"], S_c)
    fits_an = compare_fits(h, cpure["an"], S_c)
    pc = probe_c(probe, kern["adopted"])
    coef_fit = fit_cI(pc)
    env_i, env_p = envelope(pc, "cI"), envelope(pc, "cP")
    E_i, E_p = envelope_E(env_i, env_p, nodes_k, nodes_g, nodes_w)
    dS = h["S"] - S_c
    dCa = np.einsum("jsk,k->js", h["C"] - C_fd, h["a"])
    absP = lambda d: np.hypot(d[:, 1], d[:, 2])  # noqa: E731
    env_checks = {
        "E_I_bounds_truth_I": bool(np.all(np.abs(dS[:, 0]) <= E_i)),
        "E_P_bounds_truth_Q_U_P": bool(np.all(np.maximum(np.abs(dS[:, 1:]).max(axis=1), absP(dS)) <= E_p)),
        "E_I_bounds_basis_at_true_moments_I": bool(np.all(np.abs(dCa[:, 0]) <= E_i)),
        "E_P_bounds_basis_at_true_moments_P": bool(np.all(absP(dCa) <= E_p)),
        "max_truth_I_over_E_I": float(np.max(np.abs(dS[:, 0]) / E_i)),
        "max_truth_P_over_E_P": float(np.max(absP(dS) / E_p)),
        "max_basis_I_over_E_I": float(np.max(np.abs(dCa[:, 0]) / E_i)),
        "max_basis_P_over_E_P": float(np.max(absP(dCa) / E_p)),
        "E_I_over_I_max": float(np.max(E_i / h["S"][:, 0])), "E_P_over_I_max": float(np.max(E_p / h["S"][:, 0])),
        "E_I_over_I_per_channel": (E_i / h["S"][:, 0]).tolist(), "E_P_over_I_per_channel": (E_p / h["S"][:, 0]).tolist(),
    }
    error_function = {
        "definition": "c_I(x)=gamma^2 (K_harm,I/K_cont,I - 1), c_P(x)=gamma^2 |P_harm-P_cont|/K_cont,I per channel j and gamma; x=nu_j/(a_B gamma^2) with a_B at B_perp=5 muG; continuum at the harmonic's B and eta nodes",
        "probe_gamma_grid": pc["gamma"].tolist(),
        "points_used": int(pc["ok"].sum()),
        "selection": f"K_cont,I > {TAIL_FLOOR} x peak and x <= {X_VALID}",
        "equal_x_scaling_exponent_refined": equal_x_exponents(pc),
        "c_I_fit_refined": coef_fit,
        "adopted_minus_refined_max_gamma2_rel_I": float(np.nanmax(np.abs(pc["cI_adopted"] - pc["cI_refined"])[pc["ok"]])),
        "refined_minus_refined2_max_gamma2_rel_I_by_gamma": refined2_convergence(probe, pc),
        "c_I_abs_range_refined": [float(np.nanmin(np.abs(pc["cI_refined"][pc["ok"]]))), float(np.nanmax(np.abs(pc["cI_refined"][pc["ok"]])))],
        "c_P_max_refined": float(np.nanmax(pc["cP_refined"][pc["ok"]])),
        "envelope_I": env_i, "envelope_P": env_p,
        "holdout_bin_centres": holdout(h, cpure["an"], env_i, env_p),
        "predicted_vs_measured_columns_I": predicted_columns(kern["adopted"], coef_fit["alpha_beta_delta"], h, cpure),
    }
    compact = compact_form(dS, h["S"], nodes_k, nodes_g, nodes_w, E_i, E_p, coef_fit["alpha_beta_delta"])
    prop_fd = propagate(h, fits_fd["_fits"], E_i, E_p, dS, dCa)
    diffs = {"truth_I": np.abs(dS[:, 0]) / h["S"][:, 0], "truth_P": absP(dS) / h["S"][:, 0],
             "basis_I": np.abs(dCa[:, 0]) / h["S"][:, 0],
             "representation": np.max(np.abs(h["S"] - np.einsum("jsk,k->js", h["C"], h["a"])), axis=1) / h["S"][:, 0]}
    paths = figure(pc, coef_fit["alpha_beta_delta"], env_i, env_p, diffs, E_i, E_p, h["S"], fits_fd["_fits"])
    results = {
        "description": "Harmonic (paper, adopted quadrature) vs Rybicki-Lightman continuum response for the Sec. 4.3 radio fit",
        "load_checks": load_checks,
        "continuum_quadrature": ctruth["quadrature"],
        "column_differences_over_column_max_I": columns,
        "at_true_moments": moments,
        "direct_truth": truths,
        "fit_comparison_fd04_matched": fits_fd,
        "fit_comparison_analytic_matched": fits_an,
        "stencil": stencil_checks(h, cpure, probe),
        "error_function": error_function,
        "compact_form": compact,
        "envelope_checks": env_checks,
        "propagation_fd04_matched": prop_fd,
        "figures": {p.name: sha(p) for p in paths},
        "shape_function_mpmath_check": mpmath_check(),
        "notes": NOTES,
        "provenance": provenance(t0),
    }
    (HERE / "bridge_results.json").write_text(json.dumps(strip(results), indent=1) + "\n")
    print(json.dumps(strip(headline(results)), indent=1))


def headline(r) -> dict:
    f = r["fit_comparison_fd04_matched"]
    return {
        "load_checks": r["load_checks"],
        "column_max": {k: v["max"] for k, v in r["column_differences_over_column_max_I"].items()},
        "at_true_moments": {k: (v["max_over_I_IQU"] if isinstance(v, dict) else v) for k, v in r["at_true_moments"].items()},
        "truth": {k: v["max_over_I_IQU"] for k, v in r["direct_truth"].items()},
        "s7_s8": [f["s7_s8_harmonic"], f["s7_s8_continuum"]], "retained": [f["retained_harmonic"], f["retained_continuum"]],
        "principal_sines": f["principal_sines_coefficient_space"],
        "bias_h": f["harmonic_pipeline"]["representation_bias_over_sigma"],
        "bias_c": f["continuum_pipeline_own_truth_same_draws"]["representation_bias_over_sigma"],
        "shift": f["shift_harmonic_modes_continuum_basis_vs_harmonic_basis_over_sigma"],
        "error_function": {k: r["error_function"][k] for k in ("equal_x_scaling_exponent_refined", "c_I_fit_refined", "holdout_bin_centres")},
        "envelope": {k: v for k, v in r["envelope_checks"].items() if not k.endswith("channel")},
        "propagation": {k: {kk: vv for kk, vv in v.items() if not isinstance(vv, list)} for k, v in r["propagation_fd04_matched"].items()},
        "quadrature": r["continuum_quadrature"],
    }


def mpmath_check() -> dict:
    path = HERE / "bridge_mpmath_check.json"
    if not path.exists():
        return {"status": "not run"}
    rec = json.loads(path.read_text())
    stale = rec["bridge_continuum_sha256"] != sha(HERE / "bridge_continuum.py")
    return {"max_rel_err": rec["max_rel_err"], "x_points": [r["x"] for r in rec["rows"]],
            "stale_against_current_bridge_continuum": stale, "file_sha256": sha(path)}


NOTES = {
    "run": [
        "Harmonic matrix, truth, moments and noisy data loaded from results_final.json and the npz caches; refit reproduces singular values, retained count and biases exactly.",
        "Continuum matrix built at the harmonic's own B (8 GL) and eta (7 GL) nodes, same channels, screen, sky factor, bin masses and 5-point stencil (step 0.04), plus analytic ln-gamma derivatives.",
        "Harmonic kernel evaluated (radio_numerics.kernel of the harmonic pipeline, run with syncem, whose constants and line modules equal SyncMoments d51f9be up to the rename) on gamma = 10^(2.5..5.0 step 0.1) at adopted, coarse, refined (pitch 26, order 96) and, for even k, refined2 (pitch 36, order 128) quadrature; and the 5-point stencil of bins 8-11 at refined quadrature.",
    ],
    "inferred": [
        "c(x) gamma^-2 is an empirical description: exponent 2 is measured from equal-x pairs over 7.5e-4 <= x <= 27; the three-term c_I fit is not an asymptotic expansion.",
        "The envelope E_j is an empirical upper envelope over the probed gammas, channels and this ensemble (isotropic pitch, B ~ U[4,6] muG, eta ~ U[-1/2,1/2], this screen); it is validated on held-out bin-centre gammas and on the integrated truth, not proved.",
        "The adopted harmonic's own pitch/order quadrature error (up to ~3e-9 of K at gamma ~ 8e4) is part of the paper's matrix; the envelope covers it at the probed gammas.",
    ],
}


def provenance(t0) -> dict:
    files = {rel(RADIO / n): sha(RADIO / n) for n in INPUTS}
    for n in ("bridge_harmonic_probe.npz", "bridge_harmonic_probe.json", "bridge_continuum.py",
              "bridge_harmonic_probe.py"):
        files[rel(HERE / n)] = sha(HERE / n)
    files[rel(LIB / "figure_style.py")] = sha(LIB / "figure_style.py")
    return {"script": rel(Path(__file__)), "script_sha256": sha(Path(__file__)),
            "input_sha256": files,
            "versions": {"python": platform.python_version(), "numpy": np.__version__,
                         "scipy": scipy.__version__, "matplotlib": matplotlib.__version__},
            "runtime_seconds": time.perf_counter() - t0,
            "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


if __name__ == "__main__":
    main()
