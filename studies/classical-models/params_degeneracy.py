"""Field-strength degeneracy checks for params.py, continuum kernels only.

* Kernel scaling identity  Kbar_{lam B}(gamma) = lam Kbar_B(gamma sqrt(lam))  for the
  channel-, field-, eta- and screen-averaged kernels (B distribution U[4,6] scaled by lam),
  at table level without splines, and for the derivative columns on bins scaled by 1/sqrt(lam).
* Derivative identity  B d_B D_br = D_br + (r+1) D_{b,r+1} / (2 h_b), with B d_B the
  generator of the scale family U[4 lam, 6 lam] (d/d ln lam at lam = 1), by a 5-point
  finite difference in ln lam (two steps) against analytic columns.
* Null vector of the Jacobian of S = C(B) a in (a, ln B): residual
  rho_N = (N+1)/(2h) sum_b w_b a_bN D_{b,N+1}  (whitened norm).
* (ln B, ln gamma_b) ridge for E2: SBPL fits with the basis built for U[4 lam, 6 lam], with
  the energy support scaled by 1/sqrt(lam) ("transformed") or fixed at [300, 1e5] ("fixed").

Each ``*_one`` function is self-contained so it can run in a worker process.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

import basis as B
import paramlib as P
from config import PAPER


def scaled_cfg(cfg, lam, transform_support=False):
    kw = {"b_lo": cfg.b_lo * lam, "b_hi": cfg.b_hi * lam}
    if transform_support:
        kw.update(gamma_lo=cfg.gamma_lo / np.sqrt(lam), gamma_hi=cfg.gamma_hi / np.sqrt(lam))
    return replace(cfg, **kw)


def table_for(cfg):
    """In-memory kernel table (not written to the disk cache)."""
    return B.KernelTable(*cfg.b_nodes_weights)


# ----------------------------------------------------------------------------- identities
def kernel_identity_one(lam, I_ref):
    """Scaling identity at one lam; I_ref (J,) = data scale (E1 direct I) for column checks."""
    cfg1, tab1 = PAPER, B.get_table(PAPER)
    cl = scaled_cfg(PAPER, lam)
    tabl = table_for(cl)
    g = np.geomspace(300.0, 1e5, 49)
    kIl, kQl = B.kbar(cl, g, order=0, table=tabl)
    kI1, kQ1 = B.kbar(cfg1, g * np.sqrt(lam), order=0, table=tab1)
    sc = np.max(np.abs(kIl[0]), axis=1, keepdims=True)
    big = kIl[0] > 1e-6 * sc
    out = {"lam": lam,
           "kbar_I_max_abs_diff_over_max_j": float(np.max(np.abs(kIl[0] - lam * kI1[0]) / sc)),
           "kbar_Q_max_abs_diff_over_max_j_I": float(np.max(np.abs(kQl[0] - lam * kQ1[0]) / sc)),
           "kbar_I_max_rel_diff_where_I>1e-6max": float(np.max(np.abs(kIl[0] - lam * kI1[0])[big]
                                                               / kIl[0][big]))}
    y = np.geomspace(1e-3, 3e2, 300)
    for which in ("F", "G"):
        lnl, taul = tabl.direct(y, which)
        ln1, tau1 = tab1.direct(y / lam, which)
        out[f"table_{which}_no_spline_max|lnT_lam - ln(lam T_1(y/lam))|"] = float(np.max(np.abs(lnl - np.log(lam) - ln1)))
        out[f"table_{which}_no_spline_max|tau_lam - tau_1|/max(1,|tau_1|)_r<=6"] = float(
            np.max(np.abs(taul - tau1) / np.maximum(1.0, np.abs(tau1))))
    ct = scaled_cfg(PAPER, lam, transform_support=True)
    DIl, DQl = B.derivative_spectra(ct, 4, table=tabl)
    DI1, DQ1 = B.derivative_spectra(cfg1, 4, table=tab1)
    col = np.max(np.abs(DI1), axis=2, keepdims=True)
    out["columns_transformed_bins_N4_max|D_lam - lam D_1|/max_j|D_1|_I"] = float(np.max(np.abs(DIl - lam * DI1) / col))
    out["columns_transformed_bins_N4_max|D_lam - lam D_1|/max_j|D_1|_Q"] = float(
        np.max(np.abs(DQl - lam * DQ1) / np.max(np.abs(DQ1), axis=2, keepdims=True)))
    wb = cfg1.w_b[:, None, None]
    out["columns_transformed_bins_N4_max w_b|D_lam - lam D_1|/I_ref_j (I, Q)"] = [
        float(np.max(wb * np.abs(DIl - lam * DI1) / I_ref[None, None])),
        float(np.max(wb * np.abs(DQl - lam * DQ1) / I_ref[None, None]))]
    return out


def derivative_spectra_at(lnlam, N):
    cl = scaled_cfg(PAPER, float(np.exp(lnlam)))
    return lnlam, B.derivative_spectra(cl, N, table=table_for(cl))


def derivative_identity(spectra, deltas, I_ref):
    """spectra: dict lnlam -> (DI, DQ) at fixed bins (N >= 3); lnlam = 0 must have N = 4."""
    DI0, DQ0 = spectra[0.0]
    h = PAPER.h
    wb = PAPER.w_b[:, None, None]
    res = {"h_b": h, "steps_ln_lambda": list(deltas), "per_step": {},
           "richardson": "lhs_R = (16 lhs(d1) - lhs(d2))/15 with d2 = 2 d1 (removes the O(d^4) stencil error)"}

    def fd(d, k):
        v = [spectra[round(s * d, 12)][k][:, :4] for s in (-2, -1, 1, 2)]
        return (v[0] - 8 * v[1] + 8 * v[2] - v[3]) / (12 * d)

    variants = {f"{d}": (fd(d, 0), fd(d, 1)) for d in deltas}
    d1, d2 = deltas
    variants["richardson"] = ((16 * fd(d1, 0) - fd(d2, 0)) / 15, (16 * fd(d1, 1) - fd(d2, 1)) / 15)
    for label, (lhs_I, lhs_Q) in variants.items():
        rows = {}
        for r in range(4):
            rhs_I = DI0[:, r] + (r + 1) / (2 * h) * DI0[:, r + 1]
            rhs_Q = DQ0[:, r] + (r + 1) / (2 * h) * DQ0[:, r + 1]
            dI, dQ = np.abs(lhs_I[:, r] - rhs_I), np.abs(lhs_Q[:, r] - rhs_Q)
            colI = np.max(np.abs(lhs_I[:, r]), axis=1, keepdims=True)
            colQ = np.max(np.abs(lhs_Q[:, r]), axis=1, keepdims=True)
            term_I = (r + 1) / (2 * h) * np.abs(DI0[:, r + 1])
            rows[f"r{r}"] = {
                "max|lhs-rhs|/max_j|lhs|_I_bins3-11": float(np.max((dI / colI)[3:])),
                "max|lhs-rhs|/max_j|lhs|_Q_bins3-11": float(np.max((dQ / colQ)[3:])),
                "max|lhs-rhs|/max_j|lhs|_I_all_bins": float(np.max(dI / colI)),
                "max w_b|lhs-rhs|/I_ref_j (I, Q)": [float(np.max(wb[:, 0] * dI / I_ref[None])),
                                                    float(np.max(wb[:, 0] * dQ / I_ref[None]))],
                "size_of_(r+1)D_{r+1}/(2h)_term_over_max_j|lhs|_I_bins3-11": float(np.max((term_I / colI)[3:])),
            }
        res["per_step"][label] = rows
    return res


def null_vector_residual(S, a_by_N, I_scale, spectra0):
    """Whitened norms of J_lnB = dS/dlnB|_a and rho_N for moments a (12, N+1), N = 2, 3."""
    DI0, DQ0 = spectra0
    cfg, h = PAPER, PAPER.h
    sig = cfg.noise_sigma(S)[cfg.fit_idx].ravel()
    out = {}
    for N, a in a_by_N.items():
        dS_I = np.zeros(cfg.n_channels)
        dS_Q = np.zeros(cfg.n_channels, complex)
        for r in range(N + 1):
            dS_I += (cfg.w_b * a[:, r]) @ (DI0[:, r] + (r + 1) / (2 * h) * DI0[:, r + 1])
            dS_Q += (cfg.w_b * a[:, r]) @ (DQ0[:, r] + (r + 1) / (2 * h) * DQ0[:, r + 1])
        rho_I = (N + 1) / (2 * h) * (cfg.w_b * a[:, N]) @ DI0[:, N + 1]
        rho_Q = (N + 1) / (2 * h) * (cfg.w_b * a[:, N]) @ DQ0[:, N + 1]
        J = B.to_stokes(dS_I, dS_Q, cfg.azimuth_factor)
        rho = B.to_stokes(rho_I, rho_Q, cfg.azimuth_factor)
        Jw = J[cfg.fit_idx].ravel() / sig
        rw = rho[cfg.fit_idx].ravel() / sig
        out[f"N{N}"] = {"|J_lnB|_whitened": float(np.linalg.norm(Jw)), "|rho_N|_whitened": float(np.linalg.norm(rw)),
                        "ratio": float(np.linalg.norm(rw) / np.linalg.norm(Jw)),
                        "max_j|rho_N|/I_j": float(np.max(np.abs(rho) / I_scale[:, None])),
                        "dlnB_for_dchi2=1_if_moments_compensate": float(1.0 / np.linalg.norm(rw))}
    return out


# ----------------------------------------------------------------------------- E2 ridge
def ridge_one(lnlam, S, z_fit, phi_true, R_seed, dlngb_grid):
    lam = float(np.exp(lnlam))
    tab = table_for(scaled_cfg(PAPER, lam))
    fam = P.Family("SBPL_s2")
    shift = np.array([-0.5 * lnlam, 0.0, 0.0, -0.5 * lnlam])
    phi_exp = phi_true + shift
    out = {"ln_lambda": lnlam}
    for support in ("transformed", "fixed"):
        cl = scaled_cfg(PAPER, lam, support == "transformed")
        C = B.build_basis(cl, 2, table=tab)
        prob = P.Whitened(cl, C, S)
        rec = {}
        for dname, zz in (("noisefree", 0.0 * z_fit), ("seeded", z_fit)):
            d = S[cl.fit_idx] + zz * prob.sig[cl.fit_idx]
            dw = prob.whiten(d)
            Rst = prob.upre(dw)[0]
            fits = {"full": P.fit(prob, fam, cl, dw, "full", None, phi_exp),
                    "comp_Rseed": P.fit(prob, fam, cl, dw, "comp", R_seed, phi_exp)}
            if dname == "seeded":
                fits["comp_Rstar"] = P.fit(prob, fam, cl, dw, "comp", Rst, phi_exp)
            r2 = {}
            for k, f in fits.items():
                inv, inv_sd = fam.invariants_with_errors(cl, f["phi"], f["cov"])
                r2[k] = {"phi": f["phi"], "sd": f["sd"], "chi2": f["chi2"],
                         "phi_minus_transform_of_truth": f["phi"] - phi_exp,
                         "invariants": inv, "invariants_sd": inv_sd}
            r2["R_star_upre"] = Rst
            rec[dname] = r2
        # 2-D profile chi2 (full, noise-free) over absolute ln gamma_b
        dw0 = prob.whiten(S[cl.fit_idx])
        grid = []
        for dl in dlngb_grid:
            fx = P.Fixed(fam, 3, float(phi_true[3] + dl))
            x0 = np.delete(phi_exp, 3)
            x0 = P.profile_amplitude(prob, fx, cl, dw0, x0)
            f = P.fit(prob, fx, cl, dw0, "full", None, x0)
            grid.append(f["chi2"])
        rec["profile_grid_full_noisefree_chi2"] = grid
        out[support] = rec
    return out
