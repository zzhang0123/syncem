"""CMB-window figure: extrapolation of E2 with electron families and with free moments, one row of three panels.
It was the extrapolation figure of the paper up to 0.7.9; fig_extrapolation_positive.py now draws that figure.

Every panel is a spectrum over two error subpanels on a shared 1-300 GHz axis, laid out in inches (LAY).
The spectra show I/I_PL, the intensity divided by a power law fitted to the truth over 1-40 GHz, for fits
to data with 1% noise in (a, b) and 0.01% noise in (c), on one shared range. The error subpanels give the
median |I_fit - I|/I over 500 noise realisations on the 121 dense channels of extrap_curves.py, in per
cent, for fits to data with 1% and with 0.01% noise (named in each subpanel), on one error axis with a
dotted 1% reference line. All panels fit the same E2 data set (fits_cmb_bands.shared_noise).
(a, b) The correct electron family (eSBPL) and an incorrect one (eLP), fitted over 1-40 GHz in (a) and
1-100 GHz in (b) (shaded grey). The spectra show the 1% data and fits of the first realisation and, above
the fitted band, a light band of each family's colour for the 16-84% range of its predicted intensity.
(a) uses extrap_curves.npz and repeats the 500 family fits of fits_extrap.py here (fits_extrap.family_fit,
same seed and noise) for the bands; the script stops unless the prediction-channel errors reproduce the
saved |dI|/I arrays of fits_extrap_products.npz, the first realisation reproduces the dense curves of
extrap_curves.npz, the two grids agree at 300 GHz, and the 1% medians equal those of fits_family_noise.py,
which gives the error subpanels of (a, b) and the spectrum of (b). The script also stops unless the data of
(b) repeat those of (a) in the first 21 channels.
(c) Free moments, as (a, b): the E2 data of a typical realisation (closest to the median 300 GHz error in
all three bands, fits_cmb_bands.typical_realisation) on the 32 channels of fits_cmb_bands.py
(1-304 GHz) and its fits over the first 21, 26 and 32 channels (1-40, 1-100 and 1-300 GHz), with the
16-84% range over the realisations above the 40 and 100 GHz band edges. The fitted bands are nested grey
shadings (1-100 GHz light, 1-40 GHz darker), the curve colour follows them (one orange hue, darker for a
narrower band), and an in-panel legend pairs each shading with its curve.
Every drawn median is checked against the JSON of its study. A figure legend names the curve roles.
Style: lib/figure_style_paper.py.

Reads fits_extrap_products.npz, fits_extrap_results.json (python fits_extrap.py),
extrap_curves.npz (python extrap_curves.py), fits_cmb_bands_products.npz and
fits_cmb_bands_results.json (python fits_cmb_bands.py), fits_family_noise_products.npz and
fits_family_noise_results.json (python fits_family_noise.py).
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
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402

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
BANDS_PRODUCTS = HERE / "fits_cmb_bands_products.npz"
BANDS_RESULTS = HERE / "fits_cmb_bands_results.json"
FAMILY_PRODUCTS = HERE / "fits_family_noise_products.npz"
FAMILY_RESULTS = HERE / "fits_family_noise_results.json"
OUT_JSON = HERE / "fig_cmb_extrapolation_results.json"

TAG = "0.01"
NAME = "cmb_extrapolation"
NU_LO, NU_FIT_HI, NU_HI = 0.9, 40.0, 330.0  # GHz: x range of all panels and fit-band edge of (a) and of the 1-40 band of (c)
ERR_FREQ_TICKS = (1, 10, 40, 100, 300)  # GHz: every error subpanel labels the band edges 40 and 100 GHz
ERR_TICKS = {1e-4: "0.0001", 1e-3: "0.001", 1e-2: "0.01", 1e-1: "0.1", 1.0: "1", 10.0: "10", 100.0: "100"}
ERR_LIM = (2e-3, 60.0)  # median |dI|/I in per cent, the same in every error subpanel
TICK_MIN_IN = 0.12  # inches: least distance between a labelled decade and the "noise" tick
# Layout in inches (figure width fs.WIDTH): margins, column gap, legend block, panel title, spectrum and
# error-subpanel heights, the gap between a spectrum and its subpanels, tick-label and axis-label rows.
LAY = dict(left=0.42, right=0.10, gap=(0.12, 0.12), legend=0.40, title=0.17, sed=1.00, sub_gap=0.035, err=0.46,
           ticks=0.17, xlabel=0.20, bottom=0.04)
FIG_H = (LAY["legend"] + LAY["title"] + LAY["sed"] + 2 * (LAY["sub_gap"] + LAY["err"]) + LAY["ticks"]
         + LAY["xlabel"] + LAY["bottom"])
YLABEL_X_IN = 0.31  # inches left of the axes for every y label of the left column, so they align
C_NOISE = ("0.01", "0.0001")  # (c): error subpanels at 1% and 0.01% noise, as (a, b)
C_SED_NOISE = "0.0001"  # (c): the spectrum shows the free-moment fits at 0.01% noise
BANDS = ("1-300", "1-100", "1-40")  # free-moment fitted bands of (c), in drawing order
BAND_EDGES = {"1-40": 40.0, "1-100": 100.6, "1-300": 304.2}  # GHz: upper fitted channel centre (nested grid)
# Each error subpanel has one noise level, drawn as one dotted line labelled "noise level"; curve labels name
# methods or fitted bands, never a percentage. In (c) the fitted bands are nested grey shadings (1-100 GHz
# light, 1-40 GHz darker on top of it; 1-300 GHz is the whole axis), and the curve colour follows the
# shading: one orange hue stepped in OKLCH lightness, darker for a narrower band. Checked with the dataviz
# palette validator (adjacent pairs, light surface): normal-vision and CVD Delta E >= 15.9; the light step's
# 2.1:1 contrast is relieved by the direct labels. The middle step is drawn wide and the others thinner on
# top, so curves that coincide inside the fitted bands stay visible.
BAND_COLOUR = {"1-40": "#813C0B", "1-100": "#BB6802", "1-300": "#EFA000"}
BAND_LW = {"1-40": 1.0, "1-100": 2.2, "1-300": 1.2}
BAND_Z = {"1-40": 5, "1-100": 3, "1-300": 4}
BAND_ALPHA = {"1-40": 0.22, "1-100": 0.30}  # 16-84% ranges above the band edge
SHADE = {"1-100": "#F2F2F2", "1-40": "#E2E2E2"}  # nested fitted-band shading of (c)
ALPHA_WRONG, ALPHA_RIGHT = 0.25, 0.33  # 16-84% bands of the drawn families
SED_LIM = {"E2": (0.48, 1.1)}  # no error bar or band clipped; shared by the three spectra
FS = 8
TEXT_GREY = "#555555"
WRONG_TEXT = "#A84200"  # darkened vermilion for text: darker than the grey labels in greyscale
FREE_TEXT = "#8A5A00"  # darkened orange for text
INK_TEXT = "#333333"  # direct labels of the free-moment fits in (c): ink, the line carries the colour
LS_WRONG = (0, (4, 1.6))
LS_FREE = "-"  # solid like the full-basis fits of figures/moment_fit_gallery_compact, where dash-dot marks a misspecified model
FIT_FILL = "#F2F2F2"
NOISE_GREY = "#999999"
NOISE_LS = (0, (1, 1.5))
FREE = fs.ENSEMBLE["E2"]  # the E2 row colour of the fit gallery
FREE_LOW = "#9A6200"  # orange mixed with ink: the 0.01% error curve
TRUTH_KW = dict(color="#ADADAD", lw=3.2, solid_capstyle="butt", zorder=1)
DATA_KW = dict(fmt="o", ms=2.0, color=fs.INK, ecolor=fs.INK, elinewidth=0.5, capsize=0, zorder=5)
# (ensemble, right family, wrong family accepted by the in-band chi2, title) of (a); (b) fits the same
# families over BAND_B
PANELS = (
    ("E2", "eSBPL", "eLP", "(a) Electron families, 1--40 GHz"),
)
BAND_A, BAND_B = "1-40", "1-100"  # fitted bands of (a) and (b), keys of fits_family_noise.py
TITLE_B = "(b) Electron families, 1--100 GHz"
TITLE_C = "(c) Free moments"
AB_NOISE = ("0.01", "0.0001")  # (a, b) error subpanels: fits to data with 1% and 0.01% noise
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
                             "err_med": np.median(np.abs(rel), axis=0),
                             "signed_300GHz_median_p16_p84": [float(np.median(rel[:, -1])),
                                                              float(np.percentile(rel[:, -1], 16)),
                                                              float(np.percentile(rel[:, -1], 84))]}
    return bands, checks


def freq_axis(ax, labels, lo=NU_LO, ticks=(1, 10, 40, 300), shade=True, edge=NU_FIT_HI):
    """Log frequency axis, shaded up to the fitted-band edge."""
    ax.set_xscale("log")
    ax.set_xlim(lo, NU_HI)
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if shade:
        ax.axvspan(lo, edge, color=FIT_FILL, lw=0, zorder=0)
        ax.axvline(edge, color=fs.GREY, lw=fs.LW_REF, zorder=0.5)
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


def draw_band(ax, nu, band, ref, colour, alpha, edge=NU_FIT_HI):
    """Fill the band above the fitted range, interpolated in log frequency from the band edge."""
    x = np.geomspace(edge, nu[-1], 200)
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
    if not (SED_LIM[ens][0] < lo and hi < SED_LIM[ens][1]):
        raise SystemExit(f"{ens}: band range {lo:.4f}-{hi:.4f} is clipped by SED_LIM {SED_LIM[ens]}")
    ax.plot(nu, cv[f"{ens}_{wrong}"][:, 0] / ref(nu), color=fs.VERMILION, lw=1.2, ls=LS_WRONG, zorder=3)
    ax.plot(nu, cv[f"{ens}_{right}"][:, 0] / ref(nu), color=fs.BLUE, lw=1.2, zorder=4)
    d = cv[f"data_fit_{ens}"][:, 0]
    ax.errorbar(nf, d / ref(nf), yerr=cv[f"sigma_fit_{ens}"] / ref(nf), **DATA_KW)
    freq_axis(ax, labels=False)
    fs.panel_title(ax, title)
    ax.set_ylim(*SED_LIM[ens])
    return {m: float(cv[f"{ens}_{m}"][-1, 0] / truth[-1] - 1) for m in (right, wrong)}


def family_sed_b(ax, zf, jf, cv, title) -> dict:
    """(b) upper, as (a): the truth, the 1% data of the first realisation on the channels of BAND_B, its fits
    with the two families of (a), and the 16-84% range of each family's prediction above the band edge, from
    fits_family_noise.py. The data must repeat those of (a) in the first 21 channels."""
    ens, right, wrong, _ = PANELS[0]
    nu, truth = zf["centres_dense"] / 1e9, zf[f"truth_{ens}"]
    ref = reference_law(nu, truth)
    edge = jf["summary"][BAND_B]["upper_centre_GHz"]
    n = jf["summary"][BAND_B]["n_channels"]
    k = int(jf["drawn_realisation"])
    d, sig = zf[f"data_{ens}_{BAND_B}_{TAG}"][:, 0], zf[f"sigma_{ens}_{BAND_B}_{TAG}"]
    n_a = cv["nu_fit"].size
    diff = float(np.max(np.abs(d[:n_a] - cv[f"data_fit_{ens}"][:, 0]) / np.abs(cv[f"data_fit_{ens}"][:, 0])))
    if not diff <= 1e-10:
        raise SystemExit(f"(b): the data differ from those of (a) in the first {n_a} channels by {diff:.3g}")
    ax.plot(nu, truth / ref(nu), **TRUTH_KW)
    extent, out = [], {"data_vs_a_max_rel_diff": diff, "band_edge_GHz": edge, "realisation": k}
    for fam, colour, alpha, kw in ((wrong, fs.VERMILION, ALPHA_WRONG, dict(lw=1.2, ls=LS_WRONG, zorder=3)),
                                   (right, fs.BLUE, ALPHA_RIGHT, dict(lw=1.2, zorder=4))):
        r = zf[f"{ens}_{BAND_B}_{fam}_{TAG}_ratio"]
        band = {"lo": np.percentile(r, 16, axis=0) * truth, "hi": np.percentile(r, 84, axis=0) * truth}
        extent.append(draw_band(ax, nu, band, ref, colour, alpha, edge=edge))
        ax.plot(nu, r[k] * truth / ref(nu), color=colour, **kw)
        out[f"{fam}_drawn_rel_offset_300GHz"] = float(r[k, -1] - 1)
    lo, hi = min(e[0] for e in extent), max(e[1] for e in extent)
    if not (SED_LIM[ens][0] < lo and hi < SED_LIM[ens][1]):
        raise SystemExit(f"(b): band range {lo:.4f}-{hi:.4f} is clipped by SED_LIM {SED_LIM[ens]}")
    nf = zf["centres_fit"][:n] / 1e9
    ax.errorbar(nf, d / ref(nf), yerr=sig / ref(nf), **DATA_KW)
    freq_axis(ax, labels=False, edge=edge)
    fs.panel_title(ax, title)
    ax.set_ylim(*SED_LIM[ens])
    return out


def medians(ratio) -> np.ndarray:
    """Median |I_fit/I - 1| over the realisations in each dense channel."""
    return np.median(np.abs(ratio - 1), axis=0)


def check_medians(drawn: dict, ref: dict, rtol=1e-12) -> dict:
    """Every drawn median curve must reproduce its reference (JSON summary or refit)."""
    out = {}
    for key, m in drawn.items():
        r = np.asarray(ref[key])
        diff = float(np.max(np.abs(m - r) / r))
        if not diff <= rtol:
            raise SystemExit(f"{key}: drawn medians differ from the reference by {diff:.3g}")
        out[f"{key}_median_rel_diff"] = diff
    return out


def error_axis(ax, row, shade=()):
    """Error subpanel: frequency axis over 1-300 GHz with tick labels and a logarithmic median-error axis
    labelled at every decade, or at every second decade where the decades are closer than TICK_MIN_IN.
    The noise of the fitted data is the "data: x% noise" text; the dotted lines are reference error levels.
    shade: (upper edge, colour) of fitted-band shadings, widest first."""
    freq_axis(ax, labels=True, ticks=ERR_FREQ_TICKS, shade=False)
    shade_bands(ax, shade)
    ax.set_yscale("log")
    lo, hi = ERR_LIM
    ax.set_ylim(lo, hi)
    in_per_decade = ax.get_position().height * ax.figure.get_figheight() / np.log10(hi / lo)
    step = 1 if in_per_decade >= TICK_MIN_IN else 2
    major = [10.0**p for p in range(-4, 3) if p % step == 0 and lo < 10.0**p < hi]
    ax.yaxis.set_major_locator(FixedLocator(major))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: ERR_TICKS.get(v, f"{v:g}")))
    ax.yaxis.set_minor_locator(NullLocator())


def shade_bands(ax, shade):
    for hi, colour in shade:
        ax.axvspan(NU_LO, hi, color=colour, lw=0, zorder=0)
        ax.axvline(hi, color=fs.GREY, lw=fs.LW_REF, zorder=0.5)


REF_ERRORS = (1.0,)  # per cent: dotted reference level of every error subpanel


def ref_lines(ax):
    """Dotted reference level at 1% error, the same in every error subpanel, for comparison.
    The noise of the fitted data is named in the subpanel text, not drawn."""
    for y in REF_ERRORS:
        ax.axhline(y, color=NOISE_GREY, lw=fs.LW_REF, ls=NOISE_LS, zorder=1)


def family_error(ax, ens, right, wrong, tag, zf, jf, band, bands=None) -> tuple[dict, dict]:
    """(a, b) subpanel: median |dI|/I of the correct and incorrect families fitted over band to data with the
    noise of tag, on the dense channels. At 1% in (a) these are the fits drawn above; their medians must equal
    those of the refitted fits of prediction_bands (bands)."""
    nu = zf["centres_dense"] / 1e9
    ref_lines(ax)
    drawn, ref, out = {}, {}, {}
    for fam, kw in ((wrong, dict(color=fs.VERMILION, lw=1.2, ls=LS_WRONG, zorder=3)), (right, dict(color=fs.BLUE, lw=1.2, zorder=4))):
        m = medians(zf[f"{ens}_{band}_{fam}_{tag}_ratio"])
        ax.plot(nu, 100 * m, **kw)
        drawn[f"{ens}_{band}_{fam}_{tag}"] = m
        ref[f"{ens}_{band}_{fam}_{tag}"] = jf["summary"][band][fam][tag]["dI_over_I"]["median"]
        out[f"{fam}_300GHz"] = float(100 * m[-1])
    edge = jf["summary"][band]["upper_centre_GHz"]
    error_axis(ax, tag, shade=((edge, FIT_FILL),))
    noise_text(ax, tag, FIT_FILL)
    checks = check_medians(drawn, ref)
    if bands is not None:
        checks.update(check_medians({f"{ens}_{f}_refit": drawn[f"{ens}_{band}_{f}_{tag}"] for f in (right, wrong)},
                                    {f"{ens}_{f}_refit": bands[(ens, f)]["err_med"] for f in (right, wrong)}, rtol=1e-9))
    return {f"{ens}_{band}_{tag}_error_percent": out}, checks


NESTED = ((BAND_EDGES["1-100"], SHADE["1-100"]), (BAND_EDGES["1-40"], SHADE["1-40"]))


def free_sed(ax, zb, noise, title) -> dict:
    """(c) upper, as (a, b): the truth, the data of a typical realisation on the 32 channels and the
    free-moment fits of those data over the first 21, 26 and 32 channels (1-40, 1-100 and 1-300 GHz), as
    I/I_PL with the law fitted to the truth over 1-40 GHz, and the 16-84% range of each fit over the
    realisations above its band edge when that edge is below 300 GHz."""
    nu, truth = zb["centres_dense"] / 1e9, zb["truth_E2"]
    ref = reference_law(nu, truth)
    t = truth / ref(nu)
    ax.plot(nu, t, **TRUTH_KW)
    nf = zb["centres_fit"] / 1e9
    ax.errorbar(nf, zb[f"data_E2_{noise}"][:, 0] / ref(nf), yerr=zb[f"sigma_E2_{noise}"] / ref(nf), **DATA_KW)
    k = int(zb[f"typical_E2_{noise}"])  # typical realisation (fits_cmb_bands.typical_realisation)
    lo_all, hi_all = [float(np.min(t))], [float(np.max(t))]
    for band in BANDS:
        r = zb[f"E2_{band}_{noise}_ratio"]
        med, lo, hi = r[k], np.percentile(r, 16, axis=0), np.percentile(r, 84, axis=0)  # med: the drawn realisation
        if band in BAND_ALPHA:
            up = nu >= BAND_EDGES[band] * (1 - 1e-9)
            ax.fill_between(nu[up], (lo * t)[up], (hi * t)[up], color=BAND_COLOUR[band], alpha=BAND_ALPHA[band],
                            lw=0, zorder=2)
            lo_all.append(float(np.min((lo * t)[up])))
            hi_all.append(float(np.max((hi * t)[up])))
        ax.plot(nu, med * t, color=BAND_COLOUR[band], lw=BAND_LW[band], zorder=BAND_Z[band], solid_capstyle="butt")
    freq_axis(ax, labels=False, ticks=ERR_FREQ_TICKS, shade=False)
    shade_bands(ax, NESTED)
    ylim = SED_LIM["E2"]  # shared with (a, b)
    if not (ylim[0] < min(lo_all) and max(hi_all) < ylim[1]):
        raise SystemExit(f"(c): range {min(lo_all):.3f}-{max(hi_all):.3f} is clipped by SED_LIM {ylim}")
    ax.set_ylim(*ylim)
    ax.set_yticks([t for t in (0.2, 0.4, 0.6, 0.8, 1.0) if ylim[0] < t < ylim[1]])
    fs.panel_title(ax, title)
    return {"ylim": list(ylim), "truth_over_law_min_max": [float(np.min(t)), float(np.max(t))], "realisation": k}


def noise_text(ax, tag, background):
    """Names the noise of the fitted data in the top left corner of an error subpanel, on the colour of the
    shading behind it so that the reference lines stop at the text."""
    ax.text(0.015, 0.93, f"data: ${100 * float(tag):g}\\%$ noise", transform=ax.transAxes, color=TEXT_GREY,
            ha="left", va="top", fontsize=FS, zorder=6,
            bbox=dict(boxstyle="square,pad=0.12", facecolor=background, edgecolor="none"))


def free_error(ax, zb, jb, noise) -> tuple[dict, dict]:
    """(c) subpanel: median |dI|/I of the free-moment fits over the three bands at one noise level."""
    nu = zb["centres_dense"] / 1e9
    ref_lines(ax)
    drawn, ref, out = {}, {}, {}
    for band in BANDS:
        m = medians(zb[f"E2_{band}_{noise}_ratio"])
        ax.plot(nu, 100 * m, color=BAND_COLOUR[band], lw=BAND_LW[band], zorder=BAND_Z[band], solid_capstyle="butt")
        drawn[f"{band}_{noise}"] = m
        ref[f"{band}_{noise}"] = jb["summary"][band][noise]["E2"]["dI_over_I"]["median"]
        out[f"{band}_300GHz"] = float(100 * m[-1])
    error_axis(ax, noise, shade=NESTED)
    noise_text(ax, noise, SHADE["1-40"])
    return out, check_medians(drawn, ref)


def band_legend(ax):
    """In-panel legend of (c): each entry pairs the shading of a fitted band with the colour of its fit."""
    def entry(band):
        line = Line2D([], [], color=BAND_COLOUR[band], lw=min(BAND_LW[band], 1.6), solid_capstyle="butt")
        if band in SHADE:
            return (Patch(facecolor=SHADE[band], edgecolor=fs.GREY, lw=0.4), line)
        return (Patch(facecolor="white", edgecolor=fs.GREY, lw=0.4), line)
    handles = [entry(b) for b in ("1-40", "1-100", "1-300")]
    labels = ["fit 1--40 GHz", "fit 1--100 GHz", "fit 1--300 GHz"]
    leg = ax.legend(handles, labels, loc="lower left", bbox_to_anchor=(0.0, 0.14), handlelength=2.0,
                    handletextpad=0.5, borderaxespad=0.4,
                    labelspacing=0.3, frameon=True, framealpha=1.0, edgecolor="none", fontsize=FS,
                    handler_map={tuple: HandlerTuple(ndivide=1, pad=0.0)})
    leg.set_zorder(10)


LABEL_POS = {"a": {"eSBPL": (55, 0.95, "left"), "eLP": (200, 0.59, "right")},
             "b": {"eSBPL": (330, 0.95, "right"), "eLP": (250, 0.585, "right")}}  # (x GHz, y, ha) in I/I_PL


def label_panels(axes, edge_b):
    """Family names on the curves of (a, b) and the noise of the drawn data in the fitted band of each spectrum."""
    a, ae1, ae2, b, be1, be2, c, ce1, ce2 = axes
    kw = dict(fontsize=FS)
    for ax, key in ((a, "a"), (b, "b")):
        for fam, colour in (("eSBPL", fs.BLUE), ("eLP", WRONG_TEXT)):
            x, y, ha = LABEL_POS[key][fam]
            ax.text(x, y, fam, color=colour, ha=ha, **kw)
    for ax, edge, noise in ((a, NU_FIT_HI, TAG), (b, edge_b, TAG), (c, NU_FIT_HI, C_SED_NOISE)):
        ax.text(np.sqrt(NU_LO * edge), 0.04, f"${100 * float(noise):g}\\%$ noise", transform=ax.get_xaxis_transform(),
                color=TEXT_GREY, ha="center", va="bottom", **kw)


def legend(fig):
    """Figure legend in the style of the fit gallery: two rows of frameless entries above the panels."""
    data = Line2D([], [], marker="o", ls="", ms=2.6, color=fs.INK)  # the 1% error bars are hidden by the markers
    handles = [Line2D([], [], **{k: v for k, v in TRUTH_KW.items() if k != "zorder"}), data,
               Line2D([], [], color=fs.BLUE, lw=1.2), Line2D([], [], color=fs.VERMILION, lw=1.2, ls=LS_WRONG),
               (Patch(color=fs.BLUE, alpha=ALPHA_RIGHT, lw=0), Patch(color=fs.VERMILION, alpha=ALPHA_WRONG, lw=0),
                Patch(color=BAND_COLOUR["1-100"], alpha=BAND_ALPHA["1-100"], lw=0)),
               tuple(Line2D([], [], color=BAND_COLOUR[b], lw=1.4, solid_capstyle="butt") for b in ("1-40", "1-100", "1-300"))]
    labels = ["true spectrum", "data", "fit, correct electron family", "fit, incorrect electron family",
              r"$16$--$84\%$ of 500 realisations", "fit, free moments"]
    fig.legend(handles=handles, labels=labels, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 1.0),
               columnspacing=1.2, handlelength=2.4, handletextpad=0.5,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0)})


def layout(fig):
    """One row of three panels, placed in inches (LAY), each a spectrum over two error subpanels on a shared
    frequency axis."""
    W, H, L = fig.get_figwidth(), fig.get_figheight(), LAY
    width = (W - L["left"] - L["right"] - sum(L["gap"])) / 3
    heights = (L["sed"], L["err"], L["err"])
    axes, x = [], L["left"]
    for j in range(3):
        y, cell = H - L["legend"] - L["title"], []
        for k, h in enumerate(heights):
            y -= h + (L["sub_gap"] if k else 0)
            cell.append(fig.add_axes([x / W, y / H, width / W, h / H], sharex=cell[0] if cell else None))
        axes += cell
        x += width + (L["gap"][j] if j < 2 else 0)
    a, ae1, ae2, b, be1, be2, c, ce1, ce2 = axes
    for ax in (b, c):  # the same E2 truth and normalisation in every spectrum
        ax.sharey(a)
    for ax in (be1, ce1):
        ax.sharey(ae1)
    for ax in (be2, ce2):
        ax.sharey(ae2)
    return axes


def left_ylabels(fig, axes):
    """y labels of the left column at one distance from the axes; the two error subpanels of (a) share one."""
    a, ae1, ae2 = axes[:3]
    W = fig.get_figwidth()
    for ax, text in ((a, r"$I/I_{\rm PL}$"), (ae1, r"Error [\%]")):
        ax.set_ylabel(text)
        x = -YLABEL_X_IN / (ax.get_position().width * W)
        y = 0.5
        if ax is ae1:  # centre on the pair (a) error subpanels
            p1, p2 = ae1.get_position(), ae2.get_position()
            y = ((p1.y1 + p2.y0) / 2 - p1.y0) / p1.height
        ax.yaxis.set_label_coords(x, y)


def report(z, summ, sed, bands) -> dict:
    """Values of the drawn families of (a, b) over 500 realisations."""
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
    return out


def main():
    fs.configure()
    z = np.load(PRODUCTS)
    summ = json.loads(RESULTS.read_text())["summary"]
    cv = np.load(CURVES)
    zb, jb = np.load(BANDS_PRODUCTS), json.loads(BANDS_RESULTS.read_text())
    zf, jf = np.load(FAMILY_PRODUCTS), json.loads(FAMILY_RESULTS.read_text())
    for zz in (zb, zf):
        if not np.allclose(zz["centres_dense"], cv["nu_dense"], rtol=1e-12):
            raise SystemExit("the dense channels of the study products and extrap_curves.py differ")
    bands, band_checks = prediction_bands(z, cv)
    fig = plt.figure(figsize=(fs.WIDTH, FIG_H))
    axes = layout(fig)
    a, ae1, ae2, b, be1, be2, c, ce1, ce2 = axes
    sed, values, checks = {}, {}, {**check_against_json(z, summ), **band_checks}
    ens, right, wrong, title = PANELS[0]
    sed[ens] = sed_panel(a, cv, ens, right, wrong, title, bands)
    values["b_sed"] = family_sed_b(b, zf, jf, cv, TITLE_B)
    checks["b_data_vs_a_max_rel_diff"] = values["b_sed"]["data_vs_a_max_rel_diff"]
    for errs, band in (((ae1, ae2), BAND_A), ((be1, be2), BAND_B)):
        for axe, tag in zip(errs, AB_NOISE):
            v, k = family_error(axe, ens, right, wrong, tag, zf, jf, band,
                                bands if (band == BAND_A and tag == TAG) else None)
            values.update(v)
            checks.update(k)
    values["c_sed"] = free_sed(c, zb, C_SED_NOISE, TITLE_C)
    for axe, tag in zip((ce1, ce2), C_NOISE):
        v, k = free_error(axe, zb, jb, tag)
        values[f"c_{tag}_error_percent"] = v
        checks.update(k)
    band_legend(c)
    label_panels(axes, values["b_sed"]["band_edge_GHz"])
    legend(fig)
    left_ylabels(fig, axes)
    # Spectra and error subpanels each share one range across the row: tick labels only in (a).
    for ax in (b, c, be1, be2, ce1, ce2):
        ax.tick_params(axis="y", labelleft=False)
    for ax in (a, b, c, ae1, be1, ce1):  # tick labels only under the lowest subpanel of each panel
        ax.tick_params(axis="x", labelbottom=False)
        ax.set_xlabel("")
    for ax in (ae2, ce2):  # one frequency label, under (b)
        ax.set_xlabel("")
    fs.tighten_y_labels(axes)
    values.update(report(z, summ, sed, bands))
    print("values", {k: v for k, v in values.items() if not k.startswith(("E2_", "E3_", "nu_"))})
    (HERE / "figures").mkdir(exist_ok=True)
    fs.save(fig, NAME)
    outputs = {relative(HERE / "figures" / f"{NAME}.{ext}"): sha256(HERE / "figures" / f"{NAME}.{ext}")
               for ext in ("pdf", "png")}
    write_json(OUT_JSON, {"description": __doc__, "noise": TAG, "panels": [list(p) for p in PANELS],
                          "panel_b": [TITLE_B, BAND_B], "panel_c": [TITLE_C, list(BANDS), list(C_NOISE)],
                          "checks": checks, "values": values, "figures": outputs,
                          "provenance": provenance(__file__, T0, inputs=(PRODUCTS, RESULTS, CURVES, BANDS_PRODUCTS,
                                                                         BANDS_RESULTS, FAMILY_PRODUCTS, FAMILY_RESULTS,
                                                                         HERE / "fits_extrap.py",
                                                                         HERE / "lib" / "figure_style_paper.py"))})


if __name__ == "__main__":
    main()
