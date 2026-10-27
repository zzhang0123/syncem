"""Display data for the classical-model figures and tables (Figs. 8-10, tables/*.tex).

Everything quoted in a figure or table is read from the saved results of the analysis
runs (fits, forward, params, baselines, cost, verify_fits).  This module computes only the
display quantities that those runs did not store, and checks each one against the saved
results wherever they overlap:

* intrinsic direct spectra (no screen, unit azimuth factor) for the polarisation fraction;
* N(gamma) curves of the reference ensembles, the E4-KP effective diagonal density
  N_eff(gamma) = int deta q(eta) 2 N(gamma, eta) and <P_2(mu) | gamma>;
* the noise-only 1-sigma band of each seeded UPRE fit, Cov(a_hat) = V_R D_R^-2 V_R^T;
* for E4-KP, the prediction and retained-mode deviations at the pseudo-truth moments of
  kp_pseudotruth.py (JP spectrum with t_eff), checked against its saved arrays;
* a(theta_hat) for E2 and the compressed and full chi2 surfaces of the seeded E2 data over
  (p2, ln gamma_b), profiled over (ln A, p1);
* pair-invariant summaries for the E5 joint fit, whose retained modes contain exactly
  degenerate singular pairs;
* the GALPROP |b| >= 60 deg spectrum of baselines.py divided by a ln-space power law, with the
  noise-free model predictions.  The moment basis at R = 12 is not in baselines_plotdata.npz;
  it is recomputed from the saved node intensities with baselines.py's own functions, and the
  recomputation is checked by reproducing the saved R = 7 (MC median R*) arrays.

Continuum (beaming-limit) kernels only; nothing here is harmonic.
"""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import roots_jacobi
from scipy.stats import chi2 as chi2_dist
from scipy.stats import norm

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))

import basis as B  # noqa: E402
import ensembles as E  # noqa: E402
import paramlib as P  # noqa: E402
from config import PAPER  # noqa: E402

INPUTS = {
    "fits": HERE / "fits_results.json",
    "fits_npz": HERE / "fits_products.npz",
    "forward": HERE / "forward_results.json",
    "forward_npz": HERE / "forward_spectra.npz",
    "params": HERE / "params_results.json",
    "baselines": HERE / "baselines_results.json",
    "cost": HERE / "cost_results.json",
    "verify_fits": HERE / "verify_fits_results.json",
    "kp_pseudo": HERE / "kp_pseudotruth_results.json",
    "kp_pseudo_npz": HERE / "kp_pseudotruth.npz",
    "baselines_npz": HERE / "baselines_plotdata.npz",
}
# Producing scripts whose recorded SHA-256 is compared with the file on disk.
PRODUCERS = {"fits": "fits.py", "forward": "forward.py", "params": "params.py",
             "baselines": "baselines.py", "cost": "cost.py", "verify_fits": "verify_fits.py",
             "kp_pseudo": "kp_pseudotruth.py"}

ENSEMBLES = ("E1", "E2", "E3", "E4-JP", "E4-KP", "E5", "E1M")
GALLERY = ("E1", "E2", "E4-KP", "E5")  # Fig. 9 rows (E3 stays in the tables)
PSEUDO = "E4-KP"  # plotted against the JP(t_eff) pseudo-truth moments (kp_pseudotruth.py)
GALPROP_R_HIGH = 12  # second moment-basis rank drawn in galprop_fit (the first is the MC median R*)
GALPROP_LAWS = ("logcubic", "eSBPL")  # four-parameter law and electron family drawn in galprop_fit
PAIR_RTOL = 1e-9  # relative gap below which two singular values form a degenerate pair
CHECK_TOL = 1e-9  # agreement required between a recomputed and a saved quantity [I_j units]


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def npz_key(name: str) -> str:
    return name.replace("-", "_")


# ----------------------------------------------------------------------------- inputs
def load_inputs() -> tuple[dict, dict, dict]:
    """(data, input SHA-256s, producer-script hash checks)."""
    data, shas = {}, {}
    for key, path in INPUTS.items():
        shas[str(path)] = sha256(path)
        if path.suffix == ".json":
            data[key] = json.loads(path.read_text())
        else:
            with np.load(path) as z:
                data[key] = {k: z[k] for k in z.files}
    hashes = {}
    for key, script in PRODUCERS.items():
        recorded = data[key]["provenance"]["script_sha256"]
        hashes[script] = {"recorded": recorded, "on_disk": sha256(HERE / script),
                          "match": recorded == sha256(HERE / script)}
    return data, shas, hashes


# ----------------------------------------------------------------------------- Fig. 8
def kp_effective_density(kp, gammas, n_eta=64) -> np.ndarray:
    """N_eff(gamma) = (1/eta_max) int_0^eta_max 2 N(gamma, eta) deta (E4-KP, beaming limit).

    N(gamma, mu) = K * 0.5 (gamma/G0)^-p (1 - b(mu) gamma t)^(p-2) / G0 on b(mu) gamma t < 1.
    For gamma above gamma_c(0) the integrand vanishes for |eta| < mu_c(gamma) and behaves
    as (eta - mu_c)^(p-2) above it, so that panel uses Gauss-Jacobi nodes.
    """
    ag, eta_max, alpha = kp.ageing, PAPER.eta_max, kp.alpha
    pref = kp.norm_unscaled_inv * 0.5 / E.GAMMA0
    tg, wg = leggauss(n_eta)
    tj, wj = roots_jacobi(n_eta, 0.0, alpha)
    out = np.zeros(len(gammas))
    for i, g in enumerate(gammas):
        amp = pref * (g / E.GAMMA0) ** (-kp.p)
        if g >= ag.gamma_c_kp(eta_max) or g < kp.lo:
            continue
        mc = ag.mu_at_cut(g)
        if not np.isfinite(mc) or mc <= 0.0:  # no cut inside [0, eta_max]
            eta = 0.5 * eta_max * (tg + 1.0)
            f = np.clip(1.0 - ag.b_kp(eta) * g * ag.t, 0.0, None) ** alpha
            integral = 0.5 * eta_max * np.sum(wg * f)
        else:
            half = 0.5 * (eta_max - mc)
            eta = mc + half * (tj + 1.0)
            # 1 - b(eta) g t = 1.5 b_s g t (eta - mc)(eta + mc)
            f = (1.5 * ag.b_s * g * ag.t * (eta + mc)) ** alpha
            integral = half ** (alpha + 1.0) * np.sum(wj * f)
        out[i] = 2.0 * amp * integral / eta_max
    return out


def bin_masses(cfg, density_fn, kinks=(), n=64) -> np.ndarray:
    """int_bin N dgamma by Gauss-Legendre in ln gamma on panels split at the kinks (check only)."""
    t, w = leggauss(n)
    out = np.zeros(cfg.n_bins)
    for b in range(cfg.n_bins):
        lo, hi = cfg.log_edges[b], cfg.log_edges[b + 1]
        pts = np.unique([lo, hi, *[np.log(k) for k in kinks if lo < np.log(k) < hi]])
        for ua, ub in zip(pts[:-1], pts[1:]):
            u = 0.5 * (ub - ua) * t + 0.5 * (ua + ub)
            out[b] += 0.5 * (ub - ua) * np.sum(w * density_fn(np.exp(u)) * np.exp(u))
    return out


def electron_curves(fwd: dict) -> dict:
    """gamma^2.6 N(gamma) / (gamma^2.6 N_E1) for the reference ensembles; KP <P_2 | gamma>."""
    ens = E.reference_ensembles(PAPER)
    g = np.clip(np.exp(np.linspace(np.log(PAPER.gamma_lo), np.log(PAPER.gamma_hi), 4001)),
                PAPER.gamma_lo, PAPER.gamma_hi)
    curves = {k: ens[k].density(g) for k in ("E1", "E2", "E3", "E4-JP")}
    fa, fb, _ = fwd["ensembles"]["E5"]["fractions_A_B_crossing_channel"]
    za, zb = ens["E5"].zones
    curves["E5"] = fa * za.density(g) + fb * zb.density(g)
    curves["E4-KP"] = kp_effective_density(ens["E4-KP"], g)
    ref = curves["E1"]
    rel = {k: v / ref for k, v in curves.items()}
    # checks: KP bin masses against the forward-run effective moments; unit columns
    kp_fwd = fwd["ensembles"]["E4-KP"]
    kp = ens["E4-KP"]
    kinks = (kp.ageing.gamma_c_kp(0.0), kp.ageing.gamma_c_kp(PAPER.eta_max))
    masses = bin_masses(PAPER, lambda x: kp_effective_density(kp, x), kinks)
    a_eff = np.asarray(kp_fwd["moments_effective_N2"])[:, 0] * PAPER.w_b
    checks = {"kp_effective_column_GL_minus_forward": float(masses.sum() - kp_fwd["effective_column"]),
              "kp_bin_mass_max_abs_diff_vs_effective_moments": float(np.max(np.abs(masses - a_eff)))}
    for k in ("E1", "E2", "E3", "E4-JP"):
        col = bin_masses(PAPER, ens[k].density, (fwd["ageing"]["gamma_c_JP"],)).sum()
        checks[f"{k}_unit_column_GL_minus_1"] = float(col - 1.0)
    g_p2 = np.exp(np.linspace(np.log(PAPER.gamma_lo), np.log(ens["E4-KP"].ageing.gamma_c_kp(1.0)) - 1e-6, 400))
    p2 = ens["E4-KP"].p2_conditional(g_p2)
    p2_json = kp_fwd["P2_conditional"]
    p2_chk = ens["E4-KP"].p2_conditional(np.asarray(p2_json["gamma"]))
    checks["kp_P2_conditional_vs_forward_max_abs"] = float(np.nanmax(np.abs(p2_chk - np.asarray(p2_json["value"]))))
    return {"gamma": g, "rel_density": rel, "gamma_p2": g_p2, "p2": p2, "checks": checks,
            "gamma_c": {"E4-JP": float(fwd["ageing"]["gamma_c_JP"]),
                        "KP_mu0": float(kp_fwd["params"]["gamma_c_KP(mu=0)"]),
                        "KP_mu_half": float(kp_fwd["params"]["gamma_c_KP(mu=1/2)"]),
                        "KP_mu1": float(kp_fwd["params"]["gamma_c_KP(mu=1)"])}}


def intrinsic_spectra(fwd: dict, fwd_npz: dict) -> dict:
    """Direct spectra without the screen and with unit azimuth factor per zone."""
    cfg = replace(PAPER, screen_mean=0.0, screen_sd=0.0, azimuth_amp=2.0, azimuth_phase=0.0)
    ens = E.reference_ensembles(PAPER)
    fr = fwd["ensembles"]["E5"]["fractions_A_B_crossing_channel"]
    out = {k: ens[k].direct(cfg) for k in ("E1", "E2", "E3", "E4-JP", "E4-KP")}
    out["E5"] = E.TwoZone(amp=2.0).direct(cfg, fractions=(fr[0], fr[1], int(fr[2])))
    checks = {}
    for k, s in out.items():  # the screen and azimuth factor do not touch I
        ref = fwd_npz[npz_key(k)]
        checks[f"{k}_I_vs_forward_max_rel"] = float(np.max(np.abs(s[:, 0] / ref[:, 0] - 1.0)))
    frac = {k: np.hypot(s[:, 1], s[:, 2]) / s[:, 0] for k, s in out.items()}
    e1a = fwd["e1_analytic"]["Pi_direct_no_screen_at_0.1_0.316_1_3.16_10GHz"]
    idx = [int(np.argmin(np.abs(PAPER.centres - f))) for f in (1e8, 10**8.5, 1e9, 10**9.5, 1e10)]
    checks["E1_Pi_vs_forward_e1_analytic_max_abs"] = float(np.max(np.abs(frac["E1"][idx] - np.asarray(e1a))))
    chi_int = np.degrees(0.5 * np.angle(out["E5"][:, 1] + 1j * out["E5"][:, 2]))
    chi_json = np.asarray(fwd["ensembles"]["E5"]["chi_intrinsic_no_screen_deg"])
    checks["E5_chi_intrinsic_vs_forward_max_abs_deg"] = float(np.max(np.abs(
        (chi_int - chi_json + 90.0) % 180.0 - 90.0)))
    return {"fraction": frac, "checks": checks}


# ----------------------------------------------------------------------------- Fig. 9
def fit_bands(fits_npz: dict, fits: dict, kp_npz: dict) -> dict:
    """Seeded UPRE fit, true-moment prediction and their 1-sigma bands (I and |P|).

    For E4-KP the dashed reference is the prediction at the pseudo-truth moments (JP with t_eff).
    """
    C_sep = B.build_basis(PAPER, 2)
    C_joint = B.build_basis(PAPER, 2, "joint")
    out, checks = {}, {}
    for name in GALLERY:
        k = npz_key(name)
        C = C_joint if name == "E5" else C_sep
        S, pred, eps = fits_npz[f"{k}_S_direct"], fits_npz[f"{k}_pred"], fits_npz[f"{k}_eps"]
        R = int(fits_npz[f"{k}_R_star"])
        vt, s = fits_npz[f"{k}_vt"][:R], fits_npz[f"{k}_s"][:R]
        a_hat, a_true = fits_npz[f"{k}_a_hat"], fits_npz[f"{k}_a_true"]
        I = S[:, :1]
        checks[f"{name}_C_a_hat_vs_saved_pred"] = float(np.max(np.abs(B.predict(C, a_hat) - pred) / I))
        checks[f"{name}_C_a_true_vs_S_minus_eps"] = float(np.max(np.abs(B.predict(C, a_true) - (S - eps)) / I))
        cov_a = (vt.T / s**2) @ vt
        G = np.einsum("jsn,nm,jtm->jst", C, cov_a, C)  # (J, 3, 3) prediction covariance
        P = np.hypot(pred[:, 1], pred[:, 2])
        gq, gu = pred[:, 1] / P, pred[:, 2] / P
        var_p = gq**2 * G[:, 1, 1] + gu**2 * G[:, 2, 2] + 2 * gq * gu * G[:, 1, 2]
        fwd = S - eps
        if name == PSEUDO:
            fwd = B.predict(C, kp_npz["a_pseudo"])
            checks[f"{name}_C_a_pseudo_vs_saved_pred_pseudo"] = float(np.max(np.abs(fwd - kp_npz["pred_pseudo"]) / I))
            checks[f"{name}_S_direct_vs_pseudo_npz_S_kp"] = float(np.max(np.abs(S - kp_npz["S_kp"]) / I))
        out[name] = {"S": S, "pred": pred, "forward_true": fwd, "sd_I": np.sqrt(G[:, 0, 0]),
                     "sd_absP": np.sqrt(var_p), "sigma": fits_npz[f"{k}_sigma"],
                     "data_fit": fits_npz[f"{k}_data_fit"], "data_held": fits_npz[f"{k}_data_held"],
                     "R_star": R}
        max_err = float(np.max(np.abs(pred - S) / I))
        checks[f"{name}_max_err_vs_fits_json"] = max_err - fits["summary"]["E5" if name == "E5" else name]["max_err_over_I"]
    chi = fits["E5"]["chi_nu"]
    chi_pred = np.degrees(E.derotated_angle(PAPER, out["E5"]["pred"]))
    sel = np.asarray(chi["P_over_I_true"]) > 0.03
    dchi = (chi_pred - np.asarray(chi["chi_joint_seeded_deg"]) + 90.0) % 180.0 - 90.0
    checks["E5_chi_joint_from_pred_vs_json_max_abs_deg_selected"] = float(np.max(np.abs(dchi[sel])))
    return {"fits": out, "centres": fits_npz["centres"], "fit_idx": fits_npz["fit_idx"],
            "held_idx": fits_npz["held_idx"], "checks": checks}


# ----------------------------------------------------------------------------- Fig. 10
def degenerate_pairs(s: np.ndarray, R: int) -> list[tuple[int, int]]:
    """Index pairs (i, i+1) inside the first R modes with s_i = s_{i+1} to PAIR_RTOL."""
    pairs, i = [], 0
    while i < R - 1:
        if abs(s[i] - s[i + 1]) <= PAIR_RTOL * s[i]:
            pairs.append((i, i + 1))
            i += 2
        else:
            i += 1
    return pairs


def mode_deviations(fits: dict, fits_npz: dict, kp: dict, kp_npz: dict) -> dict:
    """Per-mode (beta_hat - beta)/sigma and K eps/sigma; E5 pairs as invariant 2-D norms.

    For a degenerate pair only the pair norm is basis independent. The noise deviation is
    reported as the one-degree-of-freedom equivalent z with the same tail probability,
    z = Phi^-1(1 - exp(-r^2/2)/2), and the bias as its 2-D norm |b|.
    E4-KP is taken relative to the pseudo-truth moments of kp_pseudotruth.py.
    """
    out = {}
    for name in ENSEMBLES:
        sd = fits["ensembles"][name]["seeded"]
        dev = np.asarray(sd["beta_hat_minus_true_over_sigma"])
        R = len(dev)
        bias = np.asarray(fits_npz[f"{npz_key(name)}_rep_bias_sigma"])[:R]
        if name == PSEUDO:
            ps = kp["fit_summary"]["pseudo_truth_JP_t_eff"]
            s_ = fits_npz[f"{npz_key(name)}_s"][:R]
            vt = fits_npz[f"{npz_key(name)}_vt"][:R]
            shift = s_ * (vt @ (kp_npz["a_pseudo"] - fits_npz[f"{npz_key(name)}_a_true"]))
            pseudo_dev = np.asarray(ps["beta_hat_minus_true_over_sigma"])
            out_chk = float(np.max(np.abs(dev - shift - pseudo_dev)))
            dev, bias = pseudo_dev, np.asarray(ps["rep_bias_over_sigma"])[:R]
        pairs = degenerate_pairs(fits_npz[f"{npz_key(name)}_s"], R)
        singles = sorted(set(range(R)) - {i for p in pairs for i in p})
        rec = {"singles": singles, "dev": dev[singles], "bias": bias[singles], "pairs": pairs}
        if pairs:
            r2 = np.array([dev[i] ** 2 + dev[j] ** 2 for i, j in pairs])
            rec["pair_z_equiv"] = norm.isf(0.5 * np.exp(-0.5 * r2))
            rec["pair_bias_norm"] = np.array([np.hypot(bias[i], bias[j]) for i, j in pairs])
        inv_bias = np.concatenate([np.abs(rec["bias"]), rec.get("pair_bias_norm", [])])
        rec["max_abs_bias_invariant"] = float(np.max(inv_bias))
        rec["max_abs_bias_json"] = float(fits["summary"][name]["max_abs_rep_bias_sigma"])
        if name == PSEUDO:
            rec["max_abs_bias_json"] = float(kp["fit_summary"]["pseudo_truth_JP_t_eff"]["max_abs_rep_bias_over_sigma"])
            rec["check_pseudo_dev_consistency"] = out_chk
            rec["truth"] = "JP(t_eff) pseudo-truth"
        rec["n_pairs_in_R"] = len(pairs)
        out[name] = rec
    return out


class Fixed2:
    """A paramlib family with coordinates idx (sorted) held at values (profile grids)."""

    def __init__(self, fam, idx, values):
        self.fam, self.idx, self.values = fam, tuple(idx), tuple(values)
        self.k = fam.k - len(self.idx)

    def _full(self, phi):
        phi = np.atleast_2d(np.asarray(phi, float))
        for i, v in zip(self.idx, self.values):
            phi = np.insert(phi, i, v, axis=1)
        return phi

    def moments(self, cfg, phi, N=2):
        return self.fam.moments(cfg, self._full(phi), N)

    def steps(self, phi):
        return 1e-5 * np.maximum(1.0, np.abs(phi))

    def moments_and_jacobian(self, cfg, phi, N=2):
        phi = np.asarray(phi, float)
        h = self.steps(phi)
        pts = np.vstack([phi[None], phi[None] + np.diag(h), phi[None] - np.diag(h)])
        a = self.moments(cfg, pts, N)
        return a[0], (a[1:self.k + 1] - a[self.k + 1:]).T / (2.0 * h[None, :])


def e2_parameters(params: dict, fits_npz: dict, n_grid=41, span=4.0) -> dict:
    """a(theta_hat) for E2 and the seeded compressed / full chi2 over (p2, ln gamma_b)."""
    task = params["tasks"]["E2_s2"]
    fam = P.Family("SBPL_s2")
    S = fits_npz["E2_S_direct"]
    C = B.build_basis(PAPER, 2)
    prob = P.Whitened(PAPER, C, S)
    dw = prob.whiten(fits_npz["E2_data_fit"])
    R = prob.upre(dw)[0]
    checks = {"R_star_recomputed_minus_json": R - task["R_star_seeded"]}
    best = {}
    for mode in ("comp", "full"):
        ref = task[f"seeded_{mode}"]
        f = P.fit(prob, fam, PAPER, dw, mode, R, np.asarray(ref["phi"]))
        checks[f"{mode}_refit_max_shift_over_sd"] = float(np.max(np.abs(f["phi"] - ref["phi"]) / ref["sd"]))
        checks[f"{mode}_refit_chi2_minus_json"] = f["chi2"] - ref["chi2"]
        best[mode] = f
    a_true = fits_npz["E2_a_true"]
    checks["a(theta_true)_vs_a_true_max_abs"] = float(np.max(np.abs(
        fam.moments(PAPER, np.asarray(task["phi_true"]))[0] - a_true)))
    grids = {}
    t = np.linspace(-span, span, n_grid)
    for mode in ("comp", "full"):
        f = best[mode]
        # grid along the principal axes of the Gauss-Newton (p2, ln gamma_b) covariance, so the
        # narrow direction of the strongly correlated pair is resolved as well as the long one
        lam, Q = np.linalg.eigh(f["cov"][2:4, 2:4])
        L = Q * np.sqrt(lam)
        p2 = f["phi"][2] + L[0, 0] * t[:, None] + L[0, 1] * t[None, :]
        lg = f["phi"][3] + L[1, 0] * t[:, None] + L[1, 1] * t[None, :]
        surf = np.full((n_grid, n_grid), np.nan)
        start_row = f["phi"][:2].copy()
        for i in range(n_grid):
            start = start_row.copy()
            for j in range(n_grid):
                fx = Fixed2(fam, (2, 3), (p2[i, j], lg[i, j]))
                r = P.fit(prob, fx, PAPER, dw, mode, R, start)
                surf[i, j] = r["chi2"]
                start = r["phi"]
                if j == 0:
                    start_row = r["phi"].copy()
        grids[mode] = {"p2": p2, "ln_gamma_b": lg, "chi2": surf, "chi2_min": f["chi2"],
                       "phi": f["phi"], "sd": f["sd"], "cov": f["cov"]}
        checks[f"{mode}_grid_min_minus_best"] = float(np.nanmin(surf) - f["chi2"])
        # multistart spot check of the profile at the grid corners along the long axis
        worst = 0.0
        for i, j in ((0, n_grid // 2), (n_grid - 1, n_grid // 2), (n_grid // 2, 0), (n_grid // 2, n_grid - 1)):
            fx = Fixed2(fam, (2, 3), (p2[i, j], lg[i, j]))
            alt = min(P.fit(prob, fx, PAPER, dw, mode, R, s0)["chi2"]
                      for s0 in (f["phi"][:2], np.asarray(task["phi_true"])[:2], f["phi"][:2] + [0.3, 0.05]))
            worst = max(worst, surf[i, j] - alt)
        checks[f"{mode}_grid_profile_vs_multistart_excess"] = float(worst)
    a_hat_theta = {m: fam.moments(PAPER, best[m]["phi"])[0] for m in ("comp", "full")}
    return {"grids": grids, "a_true": a_true, "a_theta": a_hat_theta, "a_tsvd": fits_npz["E2_a_hat"],
            "phi_true": np.asarray(task["phi_true"]), "R": R, "checks": checks,
            "levels_2d": [float(chi2_dist.ppf(q, 2)) for q in (norm.cdf(1) - norm.cdf(-1),
                                                               norm.cdf(2) - norm.cdf(-2))]}


# ----------------------------------------------------------------------------- GALPROP figure
def _node_positions(nu_dense: np.ndarray, nu: np.ndarray) -> np.ndarray:
    """Index of the dense frequency nearest (in ln nu) to each node."""
    return np.argmin(np.abs(np.log(nu_dense)[None, :] - np.log(nu)[:, None]), axis=1)


def _moment_curves(z: dict, ranks: tuple[int, int]) -> tuple[dict, dict]:
    """Noise-free moment-basis predictions (nodes, dense grid) at the given ranks.

    Recomputed with baselines.py's galprop_context / dense_predictions from the saved node
    intensities; the first rank (the MC median R*) is checked against the saved arrays.
    """
    import baseline_models as BM  # imported here: only this figure needs the baselines code
    import baselines as BL

    nu, nu_d, I = z["GALPROP__nu"], z["GALPROP__nu_dense"], z["GALPROP__I_true"]
    g = {"nu": nu, "train": z["GALPROP__train"].astype(bool), "I": I}
    table = B.get_table(PAPER)
    ctx = BL.galprop_context(PAPER, g, table)
    fi = ctx["fit_idx"]
    tf = BM.tsvd_fit(ctx["svd"], I[fi], ctx["sig"][fi])
    a = {R: BM.tsvd_moments(ctx["svd"], tf, R) for R in ranks}
    dense = BL.dense_predictions(PAPER, g, table, {f"moment_R{R}": {"a": a[R]} for R in ranks}, nu_d)
    curves = {f"moment_R{R}": (ctx["CI"] @ a[R], dense[f"moment_R{R}"]) for R in ranks}
    saved = (z["GALPROP__noisefree__moment_at_MC_median_Rstar__I"],
             z["GALPROP__noisefree__moment_at_MC_median_Rstar__I_dense"])
    new = curves[f"moment_R{ranks[0]}"]
    checks = {"moment_median_rank_nodes_recomputed_minus_saved_max_rel": float(np.max(np.abs(new[0] / saved[0] - 1))),
              "moment_median_rank_dense_recomputed_minus_saved_max_rel": float(np.max(np.abs(new[1] / saved[1] - 1)))}
    return curves, checks


def galprop_fit(bl: dict, z: dict) -> dict:
    """GALPROP |b| >= 60 deg spectrum and noise-free fits, relative to a ln-space power law.

    The power law is fitted by unweighted least squares in ln I to all 25 nodes. Node values and
    dense-grid curves of the log-cubic law and the broken power-law family, and of the moment
    basis at the MC median R*, are read from baselines_plotdata.npz; the moment basis at
    GALPROP_R_HIGH is recomputed (see _moment_curves).
    """
    nu, nu_d, I = z["GALPROP__nu"], z["GALPROP__nu_dense"], z["GALPROP__I_true"]
    train = z["GALPROP__train"].astype(bool)
    r_med = int(z["GALPROP__MC_median_Rstar"])
    ranks = (r_med, GALPROP_R_HIGH)
    curves, checks = _moment_curves(z, ranks)
    for m in GALPROP_LAWS:
        curves[m] = (z[f"GALPROP__noisefree__{m}__I"], z[f"GALPROP__noisefree__{m}__I_dense"])
    pos = _node_positions(nu_d, nu)
    checks["nu_dense_at_nodes_max_rel"] = float(np.max(np.abs(nu_d[pos] / nu - 1)))
    models = {}
    for key, (node, dense) in curves.items():
        checks[f"{key}_dense_at_nodes_max_rel"] = float(np.max(np.abs(dense[pos] / node - 1)))
        frac = (node - I) / I
        models[key] = {"I": node, "I_dense": dense, "res_frac": frac, "res_pct": 100 * frac,
                       "rms_held": float(np.sqrt(np.mean(frac[~train] ** 2))),
                       "max_held": float(np.max(np.abs(frac[~train])))}
    json_median = bl["results"]["GALPROP"]["monte_carlo"]["summary"]["moment_UPRE"]["R_star"]["median"]
    checks["moment_median_rank_npz_minus_json"] = float(r_med - json_median)
    checks["plotdata_sha256_mismatch"] = float(sha256(INPUTS["baselines_npz"]) != bl["plotdata_sha256"])
    c = np.polyfit(np.log(nu), np.log(I), 1)
    n_params = {m: int(bl["results"]["GALPROP"]["noise_free"][m]["n_params"]) for m in GALPROP_LAWS}
    return {"nu": nu, "nu_dense": nu_d, "train": train, "I": I, "ranks": ranks, "n_params": n_params,
            "pl": np.exp(np.polyval(c, np.log(nu))), "pl_dense": np.exp(np.polyval(c, np.log(nu_d))),
            "pl_coeff": c, "models": models, "checks": checks}


def build_all(data: dict) -> dict:
    """All derived display data; each block carries its own checks."""
    return {"electrons": electron_curves(data["forward"]),
            "intrinsic": intrinsic_spectra(data["forward"], data["forward_npz"]),
            "bands": fit_bands(data["fits_npz"], data["fits"], data["kp_pseudo_npz"]),
            "modes": mode_deviations(data["fits"], data["fits_npz"], data["kp_pseudo"], data["kp_pseudo_npz"]),
            "e2": e2_parameters(data["params"], data["fits_npz"]),
            "galprop": galprop_fit(data["baselines"], data["baselines_npz"])}
