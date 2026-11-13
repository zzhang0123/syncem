"""CMB-window extrapolation figure: right and wrong electron models fitted through the basis.

Two columns, (E2) smoothly broken power law and (E3) index mixture, 1% noise.
Top row, from extrap_curves.npz (extrap_curves.py, first noise realisation): intensity divided by a
power law fitted to the truth over 1-40 GHz, with the fitted data, the truth, the right electron
model and one wrong electron model that the in-band chi2 accepts, over 1-300 GHz.
Bottom row, from fits_extrap_products.npz and fits_extrap_results.json (fits_extrap.py): intensity
error |dI|/I of the same two models (median and 16-84% band over realisations) with the 1% noise
level as reference.
Both rows share a 1-300 GHz logarithmic frequency axis whose fitted part (below 40 GHz, shaded) is
drawn at one third of the logarithmic scale of the predicted part.
The plotted medians and 84th percentiles are checked against the JSON summary, and the in-band
chi2/dof, rejection rates and 300 GHz errors of the drawn families are recorded with the check.
Text positions of the labels are placed by hand. Style: lib/figure_style_paper.py.

Reads fits_extrap_products.npz, fits_extrap_results.json (python fits_extrap.py) and
extrap_curves.npz (python extrap_curves.py).
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
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import figure_style_paper as fs  # noqa: E402
from provenance import provenance, relative, sha256, write_json  # noqa: E402

T0 = time.time()
PRODUCTS = HERE / "fits_extrap_products.npz"
RESULTS = HERE / "fits_extrap_results.json"
CURVES = HERE / "extrap_curves.npz"
OUT_JSON = HERE / "fig_cmb_extrapolation_results.json"

TAG = "0.01"
NAME = "cmb_extrapolation"
NU_LO, NU_FIT_HI, NU_HI = 0.9, 40.0, 330.0  # GHz: x range and fit-band edge
K_FIT = 1.0 / 3.0  # logarithmic scale of the fitted part relative to the predicted part
ERR_LIM = (6e-4, 0.6)  # intensity error, fraction of I
FS = 8
TEXT_GREY = "#555555"
WRONG_TEXT = "#A84200"  # darkened vermilion for text: darker than the grey labels in greyscale
LS_WRONG = (0, (4, 1.6))
FIT_FILL = "#F2F2F2"
NOISE_GREY = "#999999"
ALPHA_WRONG, ALPHA_RIGHT = 0.25, 0.33
RIGHT_LABEL_Y = {"E2": 3.6e-3, "E3": 2.4e-3}  # top of the right-model label, under its band
TRUTH_KW = dict(color="#ADADAD", lw=3.2, solid_capstyle="butt", zorder=1)
DATA_KW = dict(fmt="o", ms=2.0, color=fs.INK, ecolor=fs.INK, elinewidth=0.5, capsize=0, zorder=5)
# (ensemble, right family, wrong family accepted by the in-band chi2, title)
PANELS = (
    ("E2", "eSBPL", "eLP", "(a) Smoothly broken power law (E2)"),
    ("E3", "eLP", "ePL", "(b) Index mixture (E3)"),
)


def stats(z, ens, model, tag=TAG):
    e = z[f"{ens}_{tag}_{model}_err"][:, :, 0]
    return (np.nanmedian(e, axis=0), np.nanpercentile(e, 16, axis=0),
            np.nanpercentile(e, 84, axis=0))


def check_against_json(z, summ) -> dict:
    """The plotted medians and 84th percentiles must reproduce the JSON summary (rtol 1e-9)."""
    out = {}
    for ens, right, wrong, _ in PANELS:
        for m in (right, wrong):
            med, _, hi = stats(z, ens, m)
            ref = summ[TAG][ens][m]["dI_over_I"]
            for q, val in (("median", med), ("p84", hi)):
                diff = float(np.max(np.abs(val - ref[q]) / np.abs(np.asarray(ref[q]))))
                if not diff <= 1e-9:
                    raise SystemExit(f"{ens} {m} {q}: plotted values differ from the JSON by {diff:.3g}")
                out[f"{ens}_{m}_{q}_max_rel_diff"] = diff
    return out


def _fwd(nu):
    x = np.log(np.clip(nu, 1e-3, None))
    return np.where(x < np.log(NU_FIT_HI), np.log(NU_FIT_HI) + K_FIT * (x - np.log(NU_FIT_HI)), x)


def _inv(x):
    x = np.asarray(x, dtype=float)
    return np.exp(np.where(x < np.log(NU_FIT_HI), np.log(NU_FIT_HI) + (x - np.log(NU_FIT_HI)) / K_FIT, x))


NU_FIT_MID = float(_inv(0.5 * (_fwd(NU_LO) + _fwd(NU_FIT_HI))))


def freq_axis(ax, labels):
    ax.set_xscale("function", functions=(_fwd, _inv))
    ax.set_xlim(NU_LO, NU_HI)
    ax.xaxis.set_major_locator(FixedLocator([1, 10, 40, 100, 300]))
    ax.xaxis.set_minor_locator(FixedLocator([2, 3, 4, 5, 6, 7, 8, 9, 20, 30, 50, 60, 70, 80, 90, 200]))
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.axvspan(NU_LO, NU_FIT_HI, color=FIT_FILL, lw=0, zorder=0)
    ax.axvline(NU_FIT_HI, color=fs.GREY, lw=fs.LW_REF, zorder=0.5)
    if labels:
        ax.set_xticklabels(["1", "10", "40", "100", "300"])
        ax.set_xlabel("Frequency [GHz]")
    else:
        ax.tick_params(axis="x", labelbottom=False)


def reference_law(nu, truth_i):
    """Power law fitted (least squares in log I) to the truth over the fitted 1-40 GHz part."""
    m = nu <= NU_FIT_HI * (1 + 1e-9)
    slope, icpt = np.polyfit(np.log(nu[m]), np.log(truth_i[m]), 1)
    return lambda x: np.exp(icpt + slope * np.log(x))


def sed_panel(ax, cv, ens, right, wrong, title):
    nu, nf = cv["nu_dense"] / 1e9, cv["nu_fit"] / 1e9
    truth = cv[f"truth_{ens}"][:, 0]
    ref = reference_law(nu, truth)
    ax.plot(nu, truth / ref(nu), **TRUTH_KW)
    ax.plot(nu, cv[f"{ens}_{wrong}"][:, 0] / ref(nu), color=fs.VERMILION, lw=1.2, ls=LS_WRONG, zorder=3)
    ax.plot(nu, cv[f"{ens}_{right}"][:, 0] / ref(nu), color=fs.BLUE, lw=1.2, zorder=4)
    d = cv[f"data_fit_{ens}"][:, 0]
    ax.errorbar(nf, d / ref(nf), yerr=cv[f"sigma_fit_{ens}"] / ref(nf), **DATA_KW)
    freq_axis(ax, labels=False)
    fs.panel_title(ax, title)
    return {m: float(cv[f"{ens}_{m}"][-1, 0] / truth[-1] - 1) for m in (right, wrong)}


def error_panel(ax, z, ens, right, wrong):
    nu = z["nu_pred"] / 1e9
    for fam, colour, ls, alpha in ((wrong, fs.VERMILION, LS_WRONG, ALPHA_WRONG),
                                   (right, fs.BLUE, "-", ALPHA_RIGHT)):
        med, lo, hi = stats(z, ens, fam)
        ax.fill_between(nu, lo, hi, color=colour, alpha=alpha, lw=0, zorder=1)
        ax.plot(nu, med, color=colour, lw=1.6, ls=ls, zorder=4)
    ax.axhline(0.01, color=NOISE_GREY, lw=fs.LW_REF, ls=(0, (1, 1.5)), zorder=1)
    freq_axis(ax, labels=True)
    ax.set_yscale("log")
    ax.set_ylim(*ERR_LIM)
    ax.set_yticks([0.001, 0.01, 0.1])
    ax.set_yticklabels([r"0.1\%", r"1\%", r"10\%"])
    ax.yaxis.set_minor_locator(FixedLocator(
        [k * 10.0**p for p in (-3, -2, -1) for k in range(2, 10)]))
    ax.yaxis.set_minor_formatter(NullFormatter())


def label_sed(a, b):
    kw = dict(fontsize=FS, linespacing=1.0)
    a.text(1.1, 0.585, "data\n(1\\% noise)", color=fs.INK, ha="left", va="bottom", **kw)
    a.text(300, 0.92, "truth", color=TEXT_GREY, ha="right", va="bottom", **kw)
    a.text(300, 0.875, "right family (eSBPL)", color=fs.BLUE, ha="right", va="bottom", **kw)
    a.text(48, 0.66, "wrong family (eLP),\naccepted in band", color=WRONG_TEXT, ha="left", va="center", **kw)
    a.set_ylim(0.56, 1.1)
    b.text(160, 1.059, "right family (eLP)", color=fs.BLUE, ha="right", va="bottom", **kw)
    b.text(300, 1.050, "truth", color=TEXT_GREY, ha="right", va="top", **kw)
    b.text(300, 1.0, "wrong family (ePL),\naccepted in band", color=WRONG_TEXT, ha="right", va="top", **kw)


def label_err(axes, z):
    kw = dict(fontsize=FS, linespacing=1.0)
    for ax, (ens, right, wrong, _) in zip(axes, PANELS):
        ax.text(NU_FIT_MID, ERR_LIM[1] / 1.15, "fitted\n1--40 GHz", color=TEXT_GREY,
                ha="center", va="top", **kw)
        ax.text(NU_FIT_MID, 0.0108, "1\\% noise", color=TEXT_GREY, ha="center", va="bottom", **kw)
        wmed = stats(z, ens, wrong)[2]
        ax.text(300, wmed[-1] * 1.12, f"wrong ({wrong})", color=WRONG_TEXT, ha="right", va="bottom", **kw)
        ax.text(300, RIGHT_LABEL_Y[ens], f"right ({right})", color=fs.BLUE, ha="right", va="top", **kw)


def layout(fig):
    tot_w, tot_h = fig.get_figwidth() * 72, fig.get_figheight() * 72
    left, gap, right_pad = 40.0, 30.0, 4.0
    w = (tot_w - left - gap - right_pad) / 2
    bottom, gap_rows, top = 27.0, 16.0, 14.0
    h = (tot_h - bottom - gap_rows - top) / 2
    rows = []
    for y in (bottom + h + gap_rows, bottom):
        rows.append([fig.add_axes([x / tot_w, y / tot_h, w / tot_w, h / tot_h])
                     for x in (left, left + w + gap)])
    return rows


def report(z, summ, sed) -> dict:
    """Values of the drawn families: in-band chi2, rejection rate and |dI|/I at 40 and 300 GHz."""
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
                "seed0_rel_intensity_offset_300GHz": sed[ens][fam]}
            print(ens, fam, out[f"{ens}_{fam}"])
    out["nu_fit_last_nu_pred_first_last_GHz"] = [float(z["nu_fit"][-1] / 1e9), float(z["nu_pred"][0] / 1e9),
                                                 float(z["nu_pred"][-1] / 1e9)]
    return out


def main():
    fs.configure()
    z = np.load(PRODUCTS)
    summ = json.loads(RESULTS.read_text())["summary"]
    cv = np.load(CURVES)
    checks = check_against_json(z, summ)
    fig = plt.figure(figsize=(fs.WIDTH, 3.9))
    top, low = layout(fig)
    sed = {}
    for ax, (ens, right, wrong, title) in zip(top, PANELS):
        sed[ens] = sed_panel(ax, cv, ens, right, wrong, title)
    for ax, (ens, right, wrong, _), letter in zip(low, PANELS, ("(c)", "(d)")):
        error_panel(ax, z, ens, right, wrong)
        fs.panel_title(ax, letter)
    label_sed(*top)
    label_err(low, z)
    top[0].set_ylabel(r"Intensity / power law")
    low[0].set_ylabel(r"Intensity error $|\Delta I|/I$")
    fs.tighten_y_labels(top + low)
    values = report(z, summ, sed)
    (HERE / "figures").mkdir(exist_ok=True)
    fs.save(fig, NAME)
    outputs = {relative(HERE / "figures" / f"{NAME}.{ext}"): sha256(HERE / "figures" / f"{NAME}.{ext}")
               for ext in ("pdf", "png")}
    write_json(OUT_JSON, {"description": __doc__, "noise": TAG, "panels": [list(p) for p in PANELS],
                          "checks": checks, "values": values, "figures": outputs,
                          "provenance": provenance(__file__, T0, inputs=(PRODUCTS, RESULTS, CURVES,
                                                                         HERE / "lib" / "figure_style_paper.py"))})


if __name__ == "__main__":
    main()
