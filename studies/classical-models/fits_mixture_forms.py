"""Free moments and the truncated power-law-mixture forms of Sec. 4.3, fitted to E2, E3, E4-JP and E6 over
1-40 and 1-100 GHz and evaluated over 1-300 GHz (the extrapolation figure, fig_extrapolation_positive.py).

Channels, noise and evaluation follow fits_cmb_bands.py and fits_family_noise.py:
  * The fitted channels are the first 21 (1-40 GHz) and 26 (1-100.6 GHz) channels of the nested
    logarithmic grid fits_cmb_bands.FULL_CFG. The fits are evaluated on the 121 channels of
    extrap_curves.DENSE_CFG (1-300 GHz).
  * Noise: N_REAL realisations with standard deviation frac * I_j on I, Q and U, at 1% and 0.01%, from
    the standard normals of fits_cmb_bands.shared_noise, so that every model fits the same data.
Models:
  * "free": free moments, the whitened TSVD with UPRE rank of fits.LinearFit on the columns of cells
    4-19 (fits_extrap.FIRST_CELL), as fits_cmb_bands.py: the separable energy columns for E2, E3 and
    E4-JP, and for E6 the joint energy-field columns of ensemble_e6.py ((N + 1)(N + 2)/2 monomials per
    cell for basis order N; six, the paper's E6 basis, at N = 2). For E2 and E3 the script stops unless the signed ratios I_fit/I reproduce
    fits_cmb_bands_products.npz to FREE_RTOL.
  * "mom<n>", "cum<n>" for n = 2, 3, 4: the moment and cumulant forms of mixture_forms.py, truncated at
    second, third and fourth order, fitted to I, Q and U through the full separable energy columns (all cells,
    the marginal field statistics), as the electron families of fits_family_noise.py; for E6 they
    therefore ignore its correlation of energy with field strength. The "1-40" form fits use the
    columns and truth of fits_extrap.FIT_CFG (as fits_family_noise.py). cum2 is the eLP family: the
    script stops unless its E2 fits reproduce the eLP fits of fits_family_noise_products.npz (both bands,
    both noise levels) and its E3 1-40 GHz fits reproduce the eLP fits of fits_extrap_products.npz: the
    in-band chi2 to CHI2_RTOL and the 300 GHz error |I_fit - I|/I to FORM_RTOL relative or FORM_ATOL
    absolute (the eLP errors reach 2e-7 at 0.01% noise, where the optimiser's different parametrisation,
    k_2 = -2 kappa, sets the agreement).
For each ensemble, model, band and noise level the JSON gives the median and 16-84% range of
|I_fit - I|/I and of |I_fit - I|/I_PL on the dense channels, at 100 and at 300 GHz, the same for the
polarisation error |P_fit - P|/I and /I_PL at 100 and 300 GHz (P = Q + iU, pol_stats), the median R* (free
moments) and, for the forms, the median in-band chi2, the fraction rejected at p < 0.001 and the fraction
of fits at a bound. I_PL is the power law fitted (least squares in ln I) to the true intensity on the dense
channels up to 40 GHz (reference_law), the normalisation of the figure's spectra.
E4-JP has no electrons above gamma_c = 19673 (E6 above 16950-22900), so its intensity is 6.5e-4 (E6 3.8e-4)
of I_PL at 100 GHz and about 6e-12 (6e-13) at 300 GHz: there |I_fit - I|/I measures the suppression of the
truth more than the model, and |I_fit - I|/I_PL compares the models.
The rejection fractions assume noise proportional to I_j in every channel. At 0.01% noise they and the
in-band chi2 include the representation error of the order-N cell basis; at N = 2 it rejects even the
exact form of E3 (cum2), at N = 4 it adds about 10 to the median chi2 of that form. "noisefree" records,
per form and band, the 300 GHz error, the largest in-band error and chi2 (at 1% noise) of a fit to
noise-free data, and "basis_floor" the error of the order-N basis at the true moments (joint moments for
E6) over 1-100 GHz and at 300 GHz.
The direct spectra of E4-JP and E6 are checked by doubling the quadrature nodes on the dense channels
(energy nodes 24 -> 48, and for E6 field nodes 16 -> 32); the largest change in units of I is recorded.

Reads fits_cmb_bands_products.npz, fits_family_noise_products.npz and fits_extrap_products.npz (checks).
Writes fits_mixture_forms_results.json and fits_mixture_forms_products.npz: the signed ratio I_fit/I of
every realisation on the dense channels, the in-band chi2 and bound flags of the forms, the channel
centres, the true spectra, and the 1% data of realisation DRAWN on the fitted channels of each band.
--basis-order N (default 2) sets the Taylor order of the energy-cell basis for every model (free moments,
the forms, and E6's joint energy-field columns of total degree <= N); for N != 2 the outputs are
fits_mixture_forms_N<N>_{results.json,products.npz} and the checks against the N = 2 studies are skipped.
Run: python fits_mixture_forms.py [--basis-order N]
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from pathlib import Path

import numpy as np

import fits as F  # first: puts lib/ on the path
import ensemble_e6 as E6
import extrap_curves as EC
import fits_cmb_bands as CB
import fits_extrap as X
import mixture_forms as MF
from config import PAPER

HERE = Path(__file__).resolve().parent
ROWS = ("E2", "E3", "E4-JP", "E6")
NOISE = (1e-2, 1e-4)
BANDS = {"1-40": 21, "1-100": 26}  # number of leading channels of fits_cmb_bands.FULL_CFG fitted
FORMS = MF.FORMS
N_REAL = X.N_REAL
DRAWN = 0  # the realisation whose 1% data and fits the figure draws
FREE_RTOL = 1e-9
FORM_RTOL, FORM_ATOL, CHI2_RTOL = 1e-4, 1e-6, 1e-6
CB_PRODUCTS = HERE / "fits_cmb_bands_products.npz"
FAMILY_PRODUCTS = HERE / "fits_family_noise_products.npz"
EXTRAP_PRODUCTS = HERE / "fits_extrap_products.npz"
N_BASIS = 2  # Taylor order of the energy-cell basis (--basis-order); 2 is the paper's basis
T0 = time.time()


def out_paths(stem: str) -> tuple[Path, Path]:
    """(npz, json) of a study: <stem>_products.npz for N_BASIS = 2, <stem>_N<N>_products.npz otherwise."""
    suffix = "" if N_BASIS == 2 else f"_N{N_BASIS}"
    return HERE / f"{stem}{suffix}_products.npz", HERE / f"{stem}{suffix}_results.json"


def keep_free() -> np.ndarray:
    """Separable columns of cells 4-19 at order N_BASIS (fits_extrap.KEEP for N_BASIS = 2)."""
    n1 = N_BASIS + 1
    return np.arange(n1 * X.FIRST_CELL, n1 * PAPER.n_bins)


def full_columns(cfg) -> np.ndarray:
    """Separable columns of all cells at order N_BASIS (fits_extrap.columns(cfg)[1] for N_BASIS = 2)."""
    DI, DQ = F.B.derivative_spectra(cfg, N_BASIS)
    return F.B.separable_columns(cfg, DI, DQ)


def tag_of(frac: float) -> str:
    return f"{frac:g}"


def truth(ens, name, cfg) -> np.ndarray:
    """(J, 3) direct channel spectrum of one ensemble on cfg."""
    return E6.direct(cfg) if name == "E6" else ens[name].direct(cfg, n=24)


# ----------------------------------------------------------------------------- free moments
def free_columns(name, cfg) -> np.ndarray:
    """Columns of cells 4-19: joint energy-field columns for E6, separable energy columns otherwise."""
    if name == "E6":
        return E6.joint_columns(cfg, N_BASIS)[:, :, E6.keep_joint(X.FIRST_CELL, N_BASIS)]
    return full_columns(cfg)[:, :, keep_free()]


def free_fits(ens, Z_full, S_dense) -> dict:
    """UPRE fits of every realisation in each band; signed I_fit/I on the dense channels and R*."""
    cols_full = {n: free_columns(n, CB.FULL_CFG) for n in ROWS}
    dense_cols = {n: free_columns(n, EC.DENSE_CFG) for n in ROWS}
    S_full = {n: truth(ens, n, CB.FULL_CFG) for n in ROWS}
    a_full = {n: (E6.joint_moments(N_BASIS).ravel()[E6.keep_joint(X.FIRST_CELL, N_BASIS)] if n == "E6"
                  else ens[n].moments(CB.FULL_CFG, N_BASIS, n=24).ravel()[keep_free()]) for n in ROWS}
    out = {}
    for band, n_fit in BANDS.items():
        F.FIT, F.HELD = np.arange(n_fit), np.arange(n_fit, n_fit + EC.DENSE_CFG.n_channels)
        Z = Z_full[:, :n_fit]
        for frac in NOISE:
            F.CFG = replace(PAPER, noise_frac=frac)
            for name in ROWS:
                lf = F.LinearFit(name, np.concatenate([cols_full[name][:n_fit], dense_cols[name]], axis=0),
                                 np.concatenate([S_full[name][:n_fit], S_dense[name]], axis=0), a_full[name])
                sc = lf.scan(lf.Sf[None, :] + Z.reshape(N_REAL, -1), np.repeat(lf.Sh[None, :], N_REAL, axis=0))
                R = sc["R_star"]
                pred = np.stack([F.B.predict(dense_cols[name], lf.a_hat(sc["proj"][i], int(R[i])))
                                 for i in range(N_REAL)])
                out[(name, band, tag_of(frac))] = {"ratio": pred[..., 0] / S_dense[name][None, :, 0], "R": R,
                                                   "dP": pol_error(pred, S_dense[name])}
                print("free", band, tag_of(frac), name, "R* median", float(np.median(R)), flush=True)
    F.CFG = PAPER
    return out


def check_free(res) -> dict:
    """E2 and E3 must reproduce the signed ratios of fits_cmb_bands.py."""
    z = np.load(CB_PRODUCTS)
    out = {}
    for name in ("E2", "E3"):
        for band in BANDS:
            for frac in NOISE:
                tag = tag_of(frac)
                ref = z[f"{name}_{band}_{tag}_ratio"]
                diff = float(np.max(np.abs(res[(name, band, tag)]["ratio"] - ref) / np.abs(ref)))
                if not diff <= FREE_RTOL:
                    raise SystemExit(f"free {name} {band} {tag}: differs from fits_cmb_bands.py by {diff:.3g}")
                out[f"free_{name}_{band}_{tag}_max_rel_diff"] = diff
    return out


# ----------------------------------------------------------------------------- mixture forms
def form_inputs(ens):
    """Full separable columns and truth on the fitted grid of each band and on the dense grid."""
    full_c, full_f, full_d = (full_columns(cfg) for cfg in (CB.FULL_CFG, X.FIT_CFG, EC.DENSE_CFG))
    S_c = {n: truth(ens, n, CB.FULL_CFG) for n in ROWS}
    S_f = {n: truth(ens, n, X.FIT_CFG) for n in ROWS}
    # 1-40 GHz: the inputs of fits_extrap.py itself (for the eLP check); other bands: the nested grid
    grids = {band: (full_f, S_f) if band == "1-40" else (full_c, S_c) for band in BANDS}
    return grids, full_d


def form_fits(grids, full_d, Z_full, S_dense) -> dict:
    """All realisations of every form, ensemble, band and noise level: signed I_fit/I, chi2, bound flags."""
    out = {}
    with ProcessPoolExecutor(max_workers=X.N_WORKERS) as pool:
        for band, n_fit in BANDS.items():
            cols, S_all = grids[band]
            rows = cols[:n_fit].reshape(-1, cols.shape[-1])
            for name in ROWS:
                S = S_all[name][:n_fit]
                for frac in NOISE:
                    sig = np.repeat(frac * S[:, 0], 3)
                    for form in FORMS:
                        jobs = [(form, rows, (S + (frac * S[:, :1]) * Z_full[i, :n_fit]).ravel(), sig, full_d,
                                 N_BASIS) for i in range(N_REAL)]
                        res = list(pool.map(MF.fit_form, jobs, chunksize=10))
                        if any(r is None for r in res):
                            raise SystemExit(f"{form} {name} {band} {tag_of(frac)}: a fit failed")
                        full = np.stack([r[0] for r in res])
                        pred = full[..., 0]
                        out[(name, form, band, tag_of(frac))] = {
                            "ratio": pred / S_dense[name][None, :, 0], "dP": pol_error(full, S_dense[name]),
                            "chi2": np.array([r[1] for r in res]),
                            "at_bound": np.array([r[3] for r in res]),
                            "params": np.stack([r[2] for r in res])}
                        e300 = np.abs(out[(name, form, band, tag_of(frac))]["ratio"][:, -1] - 1)
                        print(form, band, tag_of(frac), name, "300 GHz median %", f"{100 * np.median(e300):.4g}",
                              "chi2 median", round(float(np.median(out[(name, form, band, tag_of(frac))]["chi2"])), 1),
                              flush=True)
    return out


def check_cum2(res) -> dict:
    """cum2 is eLP: E2 against fits_family_noise.py (both bands), E3 1-40 GHz against fits_extrap.py."""
    zf, zx = np.load(FAMILY_PRODUCTS), np.load(EXTRAP_PRODUCTS)
    out = {}
    for frac in NOISE:
        tag = tag_of(frac)
        refs = [("E2", band, np.abs(zf[f"E2_{band}_eLP_{tag}_ratio"][:, -1] - 1), zf[f"E2_{band}_eLP_{tag}_chi2"])
                for band in BANDS]
        refs.append(("E3", "1-40", zx[f"E3_{tag}_eLP_err"][:, -1, 0], zx[f"E3_{tag}_eLP_chi2"]))
        for name, band, ref_err, ref_chi in refs:
            r = res[(name, "cum2", band, tag)]
            diff = np.abs(np.abs(r["ratio"][:, -1] - 1) - ref_err)
            ok_err = bool(np.all(diff <= np.maximum(FORM_RTOL * ref_err, FORM_ATOL)))
            d_chi = float(np.max(np.abs(r["chi2"] - ref_chi) / ref_chi))
            rec = {"300GHz_err_max_rel_diff": float(np.max(diff / ref_err)),
                   "300GHz_err_max_abs_diff": float(np.max(diff)), "chi2_max_rel_diff": d_chi}
            if not (ok_err and d_chi <= CHI2_RTOL):
                raise SystemExit(f"cum2 {name} {band} {tag}: differs from eLP {rec}")
            out[f"cum2_{name}_{band}_{tag}"] = rec
    return out


def check_nested(res) -> dict:
    """Each order's chi2 must not exceed the previous order's, realisation by realisation: exactly for the
    linear moment forms, and to the optimiser's tolerance for the nested cumulant fits."""
    out = {}
    for (name, form, band, tag), r in res.items():
        kind, order = MF.kind_order(form)
        if order == 2:
            continue
        lower = res[(name, f"{kind}{order - 1}", band, tag)]["chi2"]
        worst = float(np.max((r["chi2"] - lower) / lower))
        if worst > 1e-8:
            raise SystemExit(f"{form} {name} {band} {tag}: chi2 exceeds that of order {order - 1} by {worst:.3g}")
        out[f"{form}_{name}_{band}_{tag}_max_rel_rise"] = worst
    return out


# ----------------------------------------------------------------------------- summary
def reference_law(nu_ghz, truth_i, hi=40.0):
    """Power law fitted (least squares in ln I) to the truth at nu <= hi GHz; returns a callable of nu [GHz]."""
    m = nu_ghz <= hi * (1 + 1e-9)
    slope, icpt = np.polyfit(np.log(nu_ghz[m]), np.log(truth_i[m]), 1)
    return lambda x: np.exp(icpt + slope * np.log(x))


K100 = int(np.argmin(np.abs(EC.DENSE_CFG.centres - 1e11)))


def pol_error(pred: np.ndarray, S_dense: np.ndarray) -> np.ndarray:
    """|P_fit - P|/I per realisation and dense channel, P = Q + iU (amplitude and angle errors together)."""
    P_fit = pred[..., 1] + 1j * pred[..., 2]
    P = S_dense[:, 1] + 1j * S_dense[:, 2]
    return np.abs(P_fit - P[None, :]) / S_dense[None, :, 0]


def pol_stats(dP: np.ndarray, truth_over_ipl) -> dict:
    """Median and 16-84% range of |P_fit - P|/I and /I_PL at 100 and 300 GHz."""
    out = {}
    for key, e in (("dP_over_I", dP), ("dP_over_IPL", dP * truth_over_ipl[None, :])):
        for label, k in (("100GHz", K100), ("300GHz", -1)):
            x = e[:, k]
            out[f"{key}_{label}_median_p16_p84"] = [float(np.median(x)), float(np.percentile(x, 16)),
                                                    float(np.percentile(x, 84))]
    return out


def error_stats(ratio, truth_over_ipl) -> dict:
    """Median and 16-84% range of |I_fit - I|/I and |I_fit - I|/I_PL per dense channel, at 100 and 300 GHz."""
    def mpp(x):
        return [float(np.median(x)), float(np.percentile(x, 16)), float(np.percentile(x, 84))]

    out = {"dense_index_100GHz": K100}
    for key, e in (("dI_over_I", np.abs(ratio - 1)), ("dI_over_IPL", np.abs(ratio - 1) * truth_over_ipl[None, :])):
        out[key] = {"median": np.median(e, axis=0).tolist(), "p16": np.percentile(e, 16, axis=0).tolist(),
                    "p84": np.percentile(e, 84, axis=0).tolist()}
        out[f"{key}_100GHz_median_p16_p84"] = mpp(e[:, K100])
        out[f"{key}_300GHz_median_p16_p84"] = mpp(e[:, -1])
    return out


def noisefree_and_floor(ens, grids, full_d, S_dense) -> tuple[dict, dict]:
    """Noise-free form fits per ensemble, band and form, and the basis error (order N_BASIS) at the true moments."""
    nu = EC.DENSE_CFG.centres
    nf, floor = {}, {}
    for name in ROWS:
        I = S_dense[name][:, 0]
        for band, n_fit in BANDS.items():
            cols, S_all = grids[band]
            S = S_all[name][:n_fit]
            inband = nu <= CB.FULL_CFG.centres[n_fit - 1] * (1 + 1e-9)
            for form in FORMS:
                pred, chi2, params, at_bound = MF.fit_form((form, cols[:n_fit].reshape(-1, cols.shape[-1]), S.ravel(),
                                                            np.repeat(1e-2 * S[:, 0], 3), full_d, N_BASIS))
                e = np.abs(pred[:, 0] / I - 1)
                nf.setdefault(name, {}).setdefault(form, {})[band] = {
                    "dI_over_I_300GHz": float(e[-1]), "max_in_band_dI_over_I": float(np.max(e[inband])),
                    "chi2_at_1pct_noise": float(chi2), "at_bound": bool(at_bound), "params": params.tolist()}
        if name == "E6":
            pred = F.B.predict(E6.joint_columns(EC.DENSE_CFG, N_BASIS), E6.joint_moments(N_BASIS).ravel())
            basis = "joint energy-field columns at the true joint moments"
        else:
            pred = F.B.predict(full_d, ens[name].moments(EC.DENSE_CFG, N_BASIS, n=24).ravel())
            basis = "separable energy columns at the true moments"
        e = np.abs(pred[:, 0] / I - 1)
        floor[name] = {"basis": basis, "max_dI_over_I_1_100GHz": float(np.max(e[nu <= 1.01e11])),
                       "dI_over_I_300GHz": float(e[-1])}
    return nf, floor


def main() -> None:
    global N_BASIS
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--basis-order", type=int, default=2, help="Taylor order N of the energy-cell basis")
    N_BASIS = ap.parse_args().basis_order
    out_npz_path, out_json_path = out_paths("fits_mixture_forms")
    if not np.allclose(CB.FULL_CFG.centres[:X.FIT_CFG.n_channels], X.FIT_CFG.centres, rtol=1e-12):
        raise SystemExit("the first 21 channels differ from the fit channels of fits_extrap.py")
    ens = F.E.reference_ensembles(PAPER)
    S_dense = {n: truth(ens, n, EC.DENSE_CFG) for n in ROWS}
    doubled = {"E4-JP": ens["E4-JP"].direct(EC.DENSE_CFG, n=48), "E6": E6.direct(EC.DENSE_CFG, n=48, n_b=32)}
    nodes = {f"{n}_dense_max_change_over_I": float(np.max(np.abs(S2 - S_dense[n]) / S_dense[n][:, :1]))
             for n, S2 in doubled.items()}
    Z_full = CB.shared_noise()
    fit_state = (F.FIT, F.HELD, F.CFG)
    free = free_fits(ens, Z_full, S_dense)
    F.FIT, F.HELD, F.CFG = fit_state
    skipped = "skipped: the reference products use the N = 2 basis"
    checks = {"free": check_free(free) if N_BASIS == 2 else skipped}
    grids, full_d = form_inputs(ens)
    forms = form_fits(grids, full_d, Z_full, S_dense)
    checks["cum2_is_eLP"] = check_cum2(forms) if N_BASIS == 2 else skipped
    checks["nested_chi2"] = check_nested(forms)

    noisefree, floor = noisefree_and_floor(ens, grids, full_d, S_dense)
    out_npz = {"centres_dense": EC.DENSE_CFG.centres, "centres_fit": CB.FULL_CFG.centres}
    summary = {}
    nu_ghz = EC.DENSE_CFG.centres / 1e9
    for name in ROWS:
        out_npz[f"truth_{name}"] = S_dense[name][:, 0]
        ipl = reference_law(nu_ghz, S_dense[name][:, 0])(nu_ghz)
        out_npz[f"ipl_{name}"] = ipl
        t_ipl = S_dense[name][:, 0] / ipl
        for band, n_fit in BANDS.items():
            S = grids[band][1][name][:n_fit]
            out_npz[f"data_{name}_{band}_0.01"] = S + (1e-2 * S[:, :1]) * Z_full[DRAWN, :n_fit]
            out_npz[f"sigma_{name}_{band}_0.01"] = 1e-2 * S[:, 0]
            for frac in NOISE:
                tag = tag_of(frac)
                r = free[(name, band, tag)]
                out_npz[f"{name}_free_{band}_{tag}_ratio"] = r["ratio"]
                out_npz[f"{name}_free_{band}_{tag}_R"] = r["R"]
                s = error_stats(r["ratio"], t_ipl)
                s.update(pol_stats(r["dP"], t_ipl))
                s["R_star_median"] = float(np.median(r["R"]))
                summary.setdefault(name, {}).setdefault("free", {}).setdefault(band, {})[tag] = s
                for form in FORMS:
                    r = forms[(name, form, band, tag)]
                    for key in ("ratio", "chi2", "at_bound", "params"):
                        out_npz[f"{name}_{form}_{band}_{tag}_{key}"] = r[key]
                    dof = 3 * n_fit - MF.n_params(form)
                    s = error_stats(r["ratio"], t_ipl)
                    s.update(pol_stats(r["dP"], t_ipl))
                    s.update({"dof": int(dof), "chi2_median": float(np.median(r["chi2"])),
                              "frac_rejected_p001": float(np.mean(r["chi2"] > X.chi2_crit(dof, 1e-3))),
                              "frac_at_bound": float(np.mean(r["at_bound"])),
                              "params_median": np.median(r["params"], axis=0).tolist()})
                    summary[name].setdefault(form, {}).setdefault(band, {})[tag] = s
    np.savez(out_npz_path, **out_npz)
    F.write_json(out_json_path, {"description": __doc__, "basis_order": N_BASIS, "rows": list(ROWS), "forms": list(FORMS), "P0": MF.P0,
                            "k_bounds": [list(b) for b in MF.K_BOUNDS], "noise_fracs": list(NOISE), "bands": BANDS,
                            "drawn_realisation": DRAWN, "n_real": N_REAL, "seed": X.SEED,
                            "fit_config": CB.FULL_CFG.describe(), "dense_config": EC.DENSE_CFG.describe(),
                            "retained_cells_first_zero_based_free": X.FIRST_CELL, "direct_node_doubling": nodes,
                            "checks": checks, "noisefree": noisefree, "basis_floor": floor, "summary": summary,
                            "provenance": F.provenance(__file__, T0, inputs=(HERE / "mixture_forms.py",
                                                                             HERE / "ensemble_e6.py", HERE / "cost.py",
                                                                             HERE / "fits.py", HERE / "fits_extrap.py",
                                                                             HERE / "fits_cmb_bands.py",
                                                                             HERE / "extrap_curves.py",
                                                                             CB_PRODUCTS, FAMILY_PRODUCTS,
                                                                             EXTRAP_PRODUCTS))})
    print("checks", checks, "done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
