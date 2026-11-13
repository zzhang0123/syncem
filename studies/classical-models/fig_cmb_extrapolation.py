"""CMB-window figure: electron families fitted over 1-40 GHz and free moments fitted over 10-300 GHz.

Three columns, 1% noise.
(a) E2, smoothly broken power law, and (b) E3, index mixture, from extrap_curves.npz
(extrap_curves.py, first noise realisation): intensity divided by a power law fitted to the truth
over 1-40 GHz, with the fitted data, the truth, the correct electron family and one incorrect
family that the in-band chi2 accepts, over 1-300 GHz. The fitted part (below 40 GHz) is shaded grey.
Above 40 GHz a light band of each family's colour gives the 16-84% range of its predicted intensity
over the 500 noise realisations of fits_extrap.py. The bands are computed here: the 500 family fits
of fits_extrap.py are repeated (fits_extrap.family_fit, same seed and noise) and predicted on the
25 prediction channels and on the 121 dense channels of extrap_curves.py. The script stops unless
the prediction-channel errors reproduce the saved |dI|/I arrays of fits_extrap_products.npz, the
first realisation reproduces the dense curves of extrap_curves.npz, and the two grids agree at
300 GHz.
(c) E2 fitted with free moments over 10-300 GHz, from fits_cmb_products.npz (fits_cmb.py): the same
ratio with a power law fitted to the truth over 10-300 GHz, the fitted data and the fit at 1%
noise. Below it, at half its height, the intensity error |fit - true|/I_j in per cent in each of
the 41 channels (21 fitted, 20 withheld) at 1% and 0.01% noise (one realisation, scaled), with the
two noise levels as reference. Panel (c) has no shading, since the whole window is fitted.

The 500-realisation statistics of the drawn families (in-band chi2/dof, rejection rate, median
and 16-84% error at 40 and 300 GHz, signed 16-84% range of I_pred/I - 1 at 300 GHz) are recorded
from fits_extrap_products.npz after checking that they reproduce fits_extrap_results.json. The
errors drawn in (c) are checked against fits_cmb_results.json. A figure legend in the style of the fit gallery names the curves, and the
family names are written on the curves. Text positions of the labels are placed by hand.
Style: lib/figure_style_paper.py.

Reads fits_extrap_products.npz, fits_extrap_results.json (python fits_extrap.py),
extrap_curves.npz (python extrap_curves.py), fits_cmb_products.npz and fits_cmb_results.json
(python fits_cmb.py).
Writes figures/cmb_extrapolation.{pdf,png} and fig_cmb_extrapolation_results.json.
Run: python fig_cmb_extrapolation.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import figure_style_paper as fs  # noqa: E402
from provenance import provenance, relative, sha256, write_json  # noqa: E402
import extrap_curves as EC  # noqa: E402
import fits as F  # noqa: E402
import fits_extrap as X  # noqa: E402
from config import PAPER  # noqa: E402

T0 = time.time()
PRODUCTS = HERE / "fits_extrap_products.npz"
RESULTS = HERE / "fits_extrap_results.json"
CURVES = HERE / "extrap_curves.npz"
CMB_PRODUCTS = HERE / "fits_cmb_products.npz"
CMB_RESULTS = HERE / "fits_cmb_results.json"
OUT_JSON = HERE / "fig_cmb_extrapolation_results.json"

TAG = "0.01"
CMB_TAGS = ("0.01", "0.0001")
NAME = "cmb_extrapolation"
NU_LO, NU_FIT_HI, NU_HI = 0.9, 40.0, 330.0  # GHz: x range of (a, b) and fit-band edge
NU_LO_C = 9.0  # GHz: lower x limit of (c)
ERR_LIM_C = (1e-5, 40.0)  # intensity error in (c), per cent of I_j; the lowest 0.01%-noise channel is 1.3e-5 %
ERR_TICKS_C = {1e-4: r"$10^{-4}$", 1e-2: "0.01", 1.0: "1"}  # the Fig. 5 error axis
ALPHA_WRONG, ALPHA_RIGHT = 0.25, 0.33  # 16-84% bands of the drawn families
SED_LIM = {"E2": (0.56, 1.1), "E3": (0.97, 1.11), "c": (0.955, 1.04)}  # no error bar or band clipped
FS = 8
TEXT_GREY = "#555555"
WRONG_TEXT = "#A84200"  # darkened vermilion for text: darker than the grey labels in greyscale
FREE_TEXT = "#8A5A00"  # darkened orange for text
LS_WRONG = (0, (4, 1.6))
LS_FREE = (0, (5, 1.2, 1.2, 1.2))  # dash-dot: the free-moment fit
FIT_FILL = "#F2F2F2"
NOISE_GREY = "#999999"
NOISE_LS = (0, (1, 1.5))
FREE = fs.ENSEMBLE["E2"]  # the E2 row colour of the fit gallery
FREE_LOW = "#9A6200"  # orange mixed with ink: the 0.01% error curve
FREE_HALO = [pe.Stroke(linewidth=1.6, foreground=fs.INK, alpha=0.55), pe.Normal()]
TRUTH_KW = dict(color="#ADADAD", lw=3.2, solid_capstyle="butt", zorder=1)
DATA_KW = dict(fmt="o", ms=2.0, color=fs.INK, ecolor=fs.INK, elinewidth=0.5, capsize=0, zorder=5)
# (ensemble, right family, wrong family accepted by the in-band chi2, title)
PANELS = (
    ("E2", "eSBPL", "eLP", "(a) Smoothly broken power law (E2)"),
    ("E3", "eLP", "ePL", "(b) Index mixture (E3)"),
)
TITLE_C = "(c) Free moments (E2)"


def stats(z, ens, model, tag=TAG):
    e = z[f"{ens}_{tag}_{model}_err"][:, :, 0]
    return (np.nanmedian(e, axis=0), np.nanpercentile(e, 16, axis=0),
            np.nanpercentile(e, 84, axis=0))


def check_against_json(z, summ) -> dict:
    """The recorded medians and 84th percentiles must reproduce the JSON summary (rtol 1e-9)."""
    out = {}
    for ens, right, wrong, _ in PANELS:
        for m in (right, wrong):
            med, _, hi = stats(z, ens, m)
            ref = summ[TAG][ens][m]["dI_over_I"]
            for q, val in (("median", med), ("p84", hi)):
                diff = float(np.max(np.abs(val - ref[q]) / np.abs(np.asarray(ref[q]))))
                if not diff <= 1e-9:
                    raise SystemExit(f"{ens} {m} {q}: recorded values differ from the JSON by {diff:.3g}")
                out[f"{ens}_{m}_{q}_max_rel_diff"] = diff
    return out


def cmb_errors(zc) -> dict:
    """|fit - true|/I_j of the free-moment fit in each channel, per noise level."""
    S = zc["S_direct"]
    return {t: np.abs(zc[f"pred_{t}"][:, 0] - S[:, 0]) / S[:, 0] for t in CMB_TAGS}


def check_cmb(err, cmb) -> dict:
    """The largest plotted error at each noise level must equal max_dI_over_I of fits_cmb_results.json."""
    out = {}
    for t in CMB_TAGS:
        ref = cmb["summary"][t]["max_dI_over_I"]
        diff = abs(float(np.max(err[t])) - ref) / ref
        if not diff <= 1e-12:
            raise SystemExit(f"(c) {t}: plotted maximum error differs from the JSON by {diff:.3g}")
        out[f"c_{t}_max_dI_over_I_rel_diff"] = diff
    return out


def _refit(ens, fams):
    """Repeat the 500 family fits of fits_extrap.py at 1% noise, predicting on the prediction and dense grids."""
    pens = F.E.reference_ensembles(PAPER)
    _, full_f = X.columns(X.FIT_CFG)
    _, full_p = X.columns(X.PRED_CFG)
    _, full_d = X.columns(EC.DENSE_CFG)
    Sf, _, _ = X.truth(X.FIT_CFG, pens)
    Z = np.random.default_rng(X.SEED).standard_normal((X.N_REAL, X.FIT_CFG.n_channels, 3))
    rows, C, n_p = full_f.reshape(-1, full_f.shape[-1]), np.concatenate([full_p, full_d]), full_p.shape[0]
    out = {}
    for e in ens:
        sig = float(TAG) * Sf[e][:, 0]
        for fam in fams[e]:
            jobs = [(fam, rows, (Sf[e] + (float(TAG) * Sf[e][:, :1]) * Z[i]).ravel(), np.repeat(sig, 3), C)
                    for i in range(X.N_REAL)]
            with ProcessPoolExecutor(max_workers=X.N_WORKERS) as pool:
                res = list(pool.map(X.family_fit, jobs, chunksize=10))
            if any(r is None for r in res):
                raise SystemExit(f"{e} {fam}: a family fit failed, unlike fits_extrap.py")
            pred = np.stack([r[0][:, 0] for r in res])
            out[(e, fam)] = (pred[:, :n_p], pred[:, n_p:])
    return out


def prediction_bands(z, cv) -> tuple[dict, dict]:
    """16-84% range of the predicted intensity of each drawn family over the 500 realisations.

    Returns the bands on the dense grid and the checks: the refitted prediction-channel errors must
    reproduce the saved |dI|/I (rtol 1e-9), the first realisation the dense curve of extrap_curves.npz,
    and the 300 GHz channel of the two grids must agree.
    """
    refit = _refit([p[0] for p in PANELS], {p[0]: p[1:3] for p in PANELS})
    bands, checks = {}, {}
    for (ens, fam), (pp, pd) in refit.items():
        S = z[f"{ens}_S_pred"][:, 0]
        saved = z[f"{ens}_{TAG}_{fam}_err"][:, :, 0]
        diffs = {"saved_err": float(np.max(np.abs(np.abs(pp - S) / S - saved) / saved)),
                 "dense_seed0": float(np.max(np.abs(pd[0] - cv[f"{ens}_{fam}"][:, 0]) / cv[f"{ens}_{fam}"][:, 0])),
                 "300GHz_grids": float(np.max(np.abs(pd[:, -1] - pp[:, -1]) / pp[:, -1]))}
        for k, d in diffs.items():
            if not d <= 1e-9:
                raise SystemExit(f"{ens} {fam}: refitted predictions differ from {k} by {d:.3g}")
            checks[f"{ens}_{fam}_refit_{k}_max_rel_diff"] = d
        rel = pd / cv[f"truth_{ens}"][None, :, 0] - 1
        bands[(ens, fam)] = {"lo": np.percentile(pd, 16, axis=0), "hi": np.percentile(pd, 84, axis=0),
                             "signed_300GHz_median_p16_p84": [float(np.median(rel[:, -1])),
                                                              float(np.percentile(rel[:, -1], 16)),
                                                              float(np.percentile(rel[:, -1], 84))]}
    return bands, checks


def freq_axis(ax, labels, lo=NU_LO, ticks=(1, 10, 40, 100, 300), shade=True):
    ax.set_xscale("log")
    ax.set_xlim(lo, NU_HI)
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if shade:
        ax.axvspan(lo, NU_FIT_HI, color=FIT_FILL, lw=0, zorder=0)
        ax.axvline(NU_FIT_HI, color=fs.GREY, lw=fs.LW_REF, zorder=0.5)
    if labels:
        ax.set_xticklabels([f"{t:g}" for t in ticks])
        ax.set_xlabel("Frequency [GHz]")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def reference_law(nu, truth_i, hi=NU_FIT_HI):
    """Power law fitted (least squares in log I) to the truth at the frequencies nu <= hi."""
    m = nu <= hi * (1 + 1e-9)
    slope, icpt = np.polyfit(np.log(nu[m]), np.log(truth_i[m]), 1)
    return lambda x: np.exp(icpt + slope * np.log(x))


def draw_band(ax, nu, band, ref, colour, alpha):
    """Fill the band above the fitted range, interpolated in log frequency from 40 GHz."""
    x = np.geomspace(NU_FIT_HI, nu[-1], 200)
    lo, hi = (np.interp(np.log(x), np.log(nu), band[k]) for k in ("lo", "hi"))
    ax.fill_between(x, lo / ref(x), hi / ref(x), color=colour, alpha=alpha, lw=0, zorder=2)
    return float(np.min(lo / ref(x))), float(np.max(hi / ref(x)))


def sed_panel(ax, cv, ens, right, wrong, title, bands):
    nu, nf = cv["nu_dense"] / 1e9, cv["nu_fit"] / 1e9
    truth = cv[f"truth_{ens}"][:, 0]
    ref = reference_law(nu, truth)
    ax.plot(nu, truth / ref(nu), **TRUTH_KW)
    extent = [draw_band(ax, nu, bands[(ens, f)], ref, c, a)
              for f, c, a in ((wrong, fs.VERMILION, ALPHA_WRONG), (right, fs.BLUE, ALPHA_RIGHT))]
    lo, hi = min(e[0] for e in extent), max(e[1] for e in extent)
    if not SED_LIM[ens][0] < lo and hi < SED_LIM[ens][1]:
        raise SystemExit(f"{ens}: band range {lo:.4f}-{hi:.4f} is clipped by SED_LIM {SED_LIM[ens]}")
    ax.plot(nu, cv[f"{ens}_{wrong}"][:, 0] / ref(nu), color=fs.VERMILION, lw=1.2, ls=LS_WRONG, zorder=3)
    ax.plot(nu, cv[f"{ens}_{right}"][:, 0] / ref(nu), color=fs.BLUE, lw=1.2, zorder=4)
    d = cv[f"data_fit_{ens}"][:, 0]
    ax.errorbar(nf, d / ref(nf), yerr=cv[f"sigma_fit_{ens}"] / ref(nf), **DATA_KW)
    freq_axis(ax, labels=True)
    fs.panel_title(ax, title)
    ax.set_ylim(*SED_LIM[ens])
    return {m: float(cv[f"{ens}_{m}"][-1, 0] / truth[-1] - 1) for m in (right, wrong)}


def sed_panel_c(ax, zc) -> dict:
    """(c): truth, 1% data on the 21 fitted channels and the free-moment fit at 1% noise."""
    nu, S, fi = zc["centres"] / 1e9, zc["S_direct"][:, 0], zc["fit_idx"]
    ref = reference_law(nu, S, hi=NU_HI)
    ax.plot(nu, S / ref(nu), **TRUTH_KW)
    ax.plot(nu, zc["pred_0.01"][:, 0] / ref(nu), color=FREE, lw=1.2, ls=LS_FREE, zorder=4,
            path_effects=FREE_HALO)
    ax.errorbar(nu[fi], zc["data_fit_0.01"][:, 0] / ref(nu[fi]), yerr=zc["sigma_0.01"][fi, 0] / ref(nu[fi]),
                **DATA_KW)
    freq_axis(ax, labels=False, lo=NU_LO_C, ticks=(10, 30, 100, 300), shade=False)
    fs.panel_title(ax, TITLE_C)
    ax.set_ylim(*SED_LIM["c"])
    ax.text(0.5, 0.04, "fit: 10--300 GHz", transform=ax.transAxes, color=TEXT_GREY, ha="center", va="bottom",
            fontsize=FS)
    return {"truth_over_law_min_max": [float(np.min(S / ref(nu))), float(np.max(S / ref(nu)))]}


def error_panel_c(ax, zc, err) -> None:
    """(c) lower panel: per-channel intensity error in per cent at 1% (heavy) and 0.01% (thin) noise."""
    nu = zc["centres"] / 1e9
    for pct in (1.0, 1e-2):
        ax.axhline(pct, color=NOISE_GREY, lw=fs.LW_REF, ls=NOISE_LS, zorder=1)
    ax.plot(nu, 100 * err["0.01"], color=FREE, lw=1.3, zorder=3)
    ax.plot(nu, 100 * err["0.0001"], color=FREE_LOW, lw=0.75, zorder=3)
    freq_axis(ax, labels=True, lo=NU_LO_C, ticks=(10, 30, 100, 300), shade=False)
    ax.set_yscale("log")
    ax.set_ylim(*ERR_LIM_C)
    ax.yaxis.set_major_locator(FixedLocator(list(ERR_TICKS_C)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ERR_TICKS_C.get(v, f"{v:g}")))
    ax.yaxis.set_minor_locator(FixedLocator([10.0**p for p in (-5, -3, -1)]))
    ax.yaxis.set_minor_formatter(NullFormatter())


def label_panels(a, b, c, ce):
    """Family names on the curves and the fitted ranges. The legend gives the roles and the other curves."""
    kw = dict(fontsize=FS)
    a.text(55, 0.95, "eSBPL", color=fs.BLUE, ha="left", **kw)
    a.text(200, 0.59, "eLP", color=WRONG_TEXT, ha="right", **kw)
    b.text(160, 1.077, "eLP", color=fs.BLUE, ha="right", **kw)
    b.text(300, 0.980, "ePL", color=WRONG_TEXT, ha="right", **kw)
    b.set_yticks([0.98, 1.0, 1.02, 1.04, 1.06, 1.08, 1.1])
    for ax, y, va in ((a, 0.04, "bottom"), (b, 0.96, "top")):  # clear of the data in each panel
        ax.text(np.sqrt(NU_LO * NU_FIT_HI), y, "fit: 1--40 GHz", transform=ax.get_xaxis_transform(),
                color=TEXT_GREY, ha="center", va=va, **kw)
    ce.text(12, 2.6, r"$1\%$ noise", color=FREE_TEXT, ha="left", va="bottom", **kw)
    ce.text(12, 2.0e-5, r"$0.01\%$ noise", color=FREE_TEXT, ha="left", va="bottom", **kw)


def legend(fig):
    """Figure legend in the style of the fit gallery: two rows of frameless entries above the panels."""
    data = (Line2D([], [], marker="|", ls="", ms=6, mew=0.5, color=fs.INK),
            Line2D([], [], marker="o", ls="", ms=2.6, color=fs.INK))
    handles = [Line2D([], [], **{k: v for k, v in TRUTH_KW.items() if k != "zorder"}), data,
               Line2D([], [], color=fs.BLUE, lw=1.2), Line2D([], [], color=fs.VERMILION, lw=1.2, ls=LS_WRONG),
               (Patch(color=fs.BLUE, alpha=ALPHA_RIGHT, lw=0), Patch(color=fs.VERMILION, alpha=ALPHA_WRONG, lw=0)),
               Line2D([], [], color=FREE, lw=1.2, ls=LS_FREE, path_effects=FREE_HALO),
               Line2D([], [], color=NOISE_GREY, lw=1.0, ls=NOISE_LS)]
    labels = ["true spectrum", r"data, $1\%$ noise", "fit, correct electron family",
              "fit, incorrect electron family", r"$16$--$84\%$ of 500 realisations", "fit, free moments",
              "noise level"]
    fig.legend(handles=handles, labels=labels, loc="upper center", ncol=4, bbox_to_anchor=(0.54, 1.0),
               columnspacing=1.2, handlelength=2.4, handletextpad=0.5,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0), data: HandlerTuple(ndivide=1)})


def layout(fig):
    """(a) and (b) span the full height; (c) is an intensity panel over an error panel of half its height."""
    grid = fig.add_gridspec(2, 3, height_ratios=(2, 1), hspace=0.08, wspace=0.42, left=0.085, right=0.99,
                            bottom=0.155, top=0.80)
    a, b = fig.add_subplot(grid[:, 0]), fig.add_subplot(grid[:, 1])
    c = fig.add_subplot(grid[0, 2])
    return a, b, c, fig.add_subplot(grid[1, 2], sharex=c)


def report(z, summ, sed, cmb, err, sed_c, bands) -> dict:
    """Values of the drawn families (500 realisations) and of the free-moment fit drawn in (c)."""
    out = {}
    for ens, right, wrong, _ in PANELS:
        for role, fam in (("right", right), ("wrong", wrong)):
            i = summ[TAG][ens][fam]
            med, lo, hi = stats(z, ens, fam)
            out[f"{ens}_{fam}"] = {
                "role": role, "chi2_median": i["chi2_median"], "dof": i["dof"],
                "chi2_median_over_dof": i["chi2_median"] / i["dof"], "frac_rejected_p001": i["frac_rejected_p001"],
                "dI_over_I_300GHz_median_p16_p84": [float(med[-1]), float(lo[-1]), float(hi[-1])],
                "dI_over_I_40GHz_median": float(med[0]),
                "seed0_rel_intensity_offset_300GHz": sed[ens][fam],
                "signed_rel_intensity_offset_300GHz_median_p16_p84":
                    bands[(ens, fam)]["signed_300GHz_median_p16_p84"]}
            print(ens, fam, out[f"{ens}_{fam}"])
    out["nu_fit_last_nu_pred_first_last_GHz"] = [float(z["nu_fit"][-1] / 1e9), float(z["nu_pred"][0] / 1e9),
                                                 float(z["nu_pred"][-1] / 1e9)]
    out["c_E2_free_moments"] = {t: {"R_star": cmb["summary"][t]["R_star"], "chi2_fit": cmb["summary"][t]["chi2_fit"],
                                    "dof_fit": cmb["summary"][t]["dof_fit"],
                                    "max_dI_over_I": float(np.max(err[t])),
                                    "median_dI_over_I": float(np.median(err[t]))} for t in CMB_TAGS}
    out["c_E2_free_moments"].update(sed_c)
    print("(c)", out["c_E2_free_moments"])
    return out


def main():
    fs.configure()
    z = np.load(PRODUCTS)
    summ = json.loads(RESULTS.read_text())["summary"]
    cv = np.load(CURVES)
    zc = np.load(CMB_PRODUCTS)
    cmb = json.loads(CMB_RESULTS.read_text())
    err = cmb_errors(zc)
    bands, band_checks = prediction_bands(z, cv)
    checks = {**check_against_json(z, summ), **check_cmb(err, cmb), **band_checks}
    fig = plt.figure(figsize=(fs.WIDTH, 2.75))
    a, b, c, ce = layout(fig)
    sed = {}
    for ax, (ens, right, wrong, title) in zip((a, b), PANELS):
        sed[ens] = sed_panel(ax, cv, ens, right, wrong, title, bands)
    sed_c = sed_panel_c(c, zc)
    error_panel_c(ce, zc, err)
    label_panels(a, b, c, ce)
    legend(fig)
    a.set_ylabel(r"Intensity / power law")
    ce.set_ylabel(r"Error [\%]")
    fs.tighten_y_labels((a, b, c, ce))
    values = report(z, summ, sed, cmb, err, sed_c, bands)
    (HERE / "figures").mkdir(exist_ok=True)
    fs.save(fig, NAME)
    outputs = {relative(HERE / "figures" / f"{NAME}.{ext}"): sha256(HERE / "figures" / f"{NAME}.{ext}")
               for ext in ("pdf", "png")}
    write_json(OUT_JSON, {"description": __doc__, "noise": TAG, "panels": [list(p) for p in PANELS],
                          "panel_c": {"ensemble": "E2", "title": TITLE_C, "noise": list(CMB_TAGS)},
                          "checks": checks, "values": values, "figures": outputs,
                          "provenance": provenance(__file__, T0, inputs=(PRODUCTS, RESULTS, CURVES, CMB_PRODUCTS,
                                                                         CMB_RESULTS,
                                                                         HERE / "fits_extrap.py",
                                                                         HERE / "lib" / "figure_style_paper.py"))})


if __name__ == "__main__":
    main()
