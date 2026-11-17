"""Extrapolation figure of the paper (`fig: cmb extrapolation`, figures/extrapolation_positive): free moments,
non-negative cells and the second-order moment and cumulant forms of Sec. 4.3.

With its default arguments the script draws the paper's figure. Rows: E2, E3 and E4-JP. Columns: free moments,
non-negative cells (the Tikhonov fits with non-negative cells, "posridge" of fits_positive_ridge.py) and the
electron family (the moment and cumulant forms truncated at second order). Every panel is a spectrum over two
error subpanels on a shared 1-300 GHz axis.
  * Spectrum: I/I_PL, the intensity divided by I_PL, the power law fitted to the truth over 1-40 GHz
    (fits_mixture_forms.reference_law). Points: the data (noise --sed-noise, 0.01% by default) of realisation
    fits_mixture_forms.DRAWN over the --sed-band band (shaded). Lines: the per-channel median of the predictions
    of the fits to the 500 realisations, drawn at the width of that noise level; above the band, the shading
    (a fan-shaped legend swatch) gives their 16-84% range.
    The free-moment and non-negative columns draw one model each, the family column the moment form (solid)
    and the cumulant form (dashed).
  * Error subpanels: the median over the realisations of |I_fit - I|/I (E2, E3) or of |I_fit - I|/I_PL
    (E4-JP, E6, whose true intensity falls by 1e-12 above the cut-off, so |I_fit - I|/I would measure the
    truth rather than the fit), in per cent, for fits over 1-40 GHz (upper) and 1-100 GHz (lower), each
    with its fitted band shaded, at 1% noise (thick, in an opaque tint of the curve colour) and 0.01% noise
    (thin, full colour; the spectrum medians use the same thin width), with a dotted 1% reference line.
Colours: free moments and non-negative cells ink (both solid; the column titles name them), moment form blue,
cumulant form vermilion; the cumulant form is also dashed, so the forms stay apart in greyscale, and the
noise levels differ in line width. The legend has three untitled groups, each above one column of panels:
the spectrum elements, the medians (three-colour swatches: width = noise level in every column) with the
1% reference line, and the two forms. Every drawn median is checked against the results JSON of the source;
the script stops if an error median leaves the top of its axis, and records the frequencies where one passes
below it (small in-band errors in units of I_PL above the cut-off). Style: lib/figure_style_paper.py.

--source picks the study drawn: ridge (default, fits_positive_ridge_*), positive (fits_free_positive_*, non-negative
cells without the penalty, column 'pos') or forms (fits_mixture_forms_*: free moments and the moment and cumulant
forms at the two orders of --orders, rows up to E6). --columns overrides the three column groups, e.g.
'free;pos;mom2,cum2'. --bound-note (where every cumulant fit ends on a bound of k, Appendix B) and --axis-note
(where the free-moment range leaves a fixed spectrum axis) add in-panel notes; both are off for the paper's figure.
Reads <source>_products.npz and <source>_results.json. Writes figures/<name>.{pdf,png} and fig_<name>_results.json,
with the name extrapolation_positive (ridge), extrapolation_positive_nnls (positive) or extrapolation_forms (forms)
unless --name is given.
Run: python fig_extrapolation_positive.py [ENSEMBLE ...] [--source ridge|positive|forms] [--columns GROUPS]
     [--orders 2,4] [--bands 1-40,1-100] [--noise 0.01,0.0001] [--sed-band 1-100] [--sed-noise 0.0001] [--name NAME]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import to_hex, to_rgb  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.legend_handler import HandlerPatch, HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Polygon  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter, NullLocator  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import figure_style_paper as fs  # noqa: E402
from provenance import provenance, relative, sha256, write_json  # noqa: E402

T0 = time.time()

ALL_ROWS = ("E2", "E3", "E4-JP", "E6")
DEFAULT_ROWS = ("E2", "E3", "E4-JP")  # the rows of the paper's figure
ORDER_NAME = {2: "second order", 3: "third order", 4: "fourth order"}
GROUP_TITLE = {"free": "free moments", "pos": "non-negative cells", "posridge": "non-negative cells",
               "mom2,cum2": "electron family"}  # the second-order moment and cumulant forms of Sec. 4.3
SED_BAND, SED_TAG = "1-100", "0.01"  # set in main: the band and noise of the fits drawn in the spectra
NOISE = ("0.01", "0.0001")  # set in main: noise levels drawn in the error subpanels
ERR_BANDS = ("1-40", "1-100")  # set in main: one error subpanel per band
SOURCES = {"ridge": "fits_positive_ridge",  # Tikhonov fits with non-negative cells ("posridge"; and free, mom2, cum2)
           "positive": "fits_free_positive",  # non-negative cells without the penalty ("pos"; and free, mom2, cum2)
           "forms": "fits_mixture_forms"}  # free moments and the moment and cumulant forms up to fourth order
DEFAULT_COLUMNS = {"ridge": "free;posridge;mom2,cum2", "positive": "free;pos;mom2,cum2", "forms": None}
DEFAULT_NAME = {"ridge": "extrapolation_positive", "positive": "extrapolation_positive_nnls",
                "forms": "extrapolation_forms"}
BAND_EDGE = {"1-40": 40.0, "1-100": 100.6, "1-145": 145.5}  # GHz: upper fitted channel centre
NU_LO, NU_HI = 0.9, 330.0
FREQ_TICKS = (1, 10, 40, 100, 300)

COLOUR = {"free": fs.INK, "pos": fs.INK, "mom": fs.BLUE, "cum": fs.VERMILION}
LS = {"free": "-", "pos": "-", "mom": "-", "cum": (0, (4, 1.6))}  # free and pos share a style: the column titles name them
NOISE_STYLE = {"0.01": dict(lw=2.2), "0.0001": dict(lw=1.0)}  # 1%: thick, drawn in TINT; 0.01%: thin, full colour
TINT_ALPHA = 0.45  # the 1% curves are opaque blends of the curve colour with white (no brown where forms overlap)
TINT_FRAC = {fs.BLUE: 0.6, fs.VERMILION: 0.4}  # the two forms' tints differ in lightness, so they part in greyscale
PAIR_GAP_IN = 0.16  # gap between the free and non-negative columns, which share their spectrum y axis
BAND_ALPHA = {"free": 0.15, "pos": 0.15, "mom": 0.22, "cum": 0.22}
TRUTH_KW = dict(color="#C4C4C4", lw=3.2, solid_capstyle="butt", zorder=1)
DATA_KW = dict(fmt="o", ms=1.8, color=fs.INK, ecolor=fs.INK, elinewidth=0.5, capsize=0, zorder=6)
FIT_FILL = "#F2F2F2"
NOISE_GREY = "#999999"
FS = 7.5  # in-panel annotations
LEGEND_FS, LEGEND_INSET = 8.0, 0.06  # legend font (pt, as in the other figures of the paper) and top inset (in)
TEXT_GREY = "#555555"

# Error metric per row (key of the JSON summary) and its axis label.
METRIC = {"E2": "dI_over_I", "E3": "dI_over_I", "E4-JP": "dI_over_IPL", "E6": "dI_over_IPL"}
METRIC_LABEL = {"dI_over_I": r"Error [\%]", "dI_over_IPL": r"Error [\% of $I_{\rm PL}$]"}
# Spectrum limits: None takes the drawn content plus a margin; the free column and the two form columns
# of a row are scaled separately. Fixed limits are used where the free-moment range leaves any useful scale.
SED_LIM_FREE = {"E4-JP": (-0.4, 2.4), "E6": (-0.4, 2.4)}
ERR_LIM = {"dI_over_I": (1e-3, 1e3), "dI_over_IPL": (0.9e-4, 1e6)}  # per cent, for 1% and 0.01% noise
ERR_TICK_STEP = {"dI_over_I": 2, "dI_over_IPL": 4}  # labelled decades: every second or fourth
LAY = dict(left=0.47, right=0.08, gap=(0.36, 0.12), legend=0.50, title=0.16, sed=0.80, sub_gap=0.03, err=0.34,
           row_gap=0.10, ticks=0.16, xlabel=0.18, bottom=0.03)
YLABEL_X_IN = 0.34
NOTE_Y_EDGE = 0.72  # axes fraction: where the band-edge line stops in a spectrum panel that carries a note


def tint(colour: str) -> str:
    """Opaque blend of colour with white at TINT_FRAC (TINT_ALPHA by default)."""
    rgb = np.array(to_rgb(colour))
    return to_hex(1.0 - TINT_FRAC.get(colour, TINT_ALPHA) * (1.0 - rgb))


def kind(model: str) -> str:
    return model if model in ("free", "pos") else model[:3]


def freq_axis(ax, labels, edge=None):
    ax.set_xscale("log")
    ax.set_xlim(NU_LO, NU_HI)
    ax.xaxis.set_major_locator(FixedLocator(list(FREQ_TICKS)))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=np.arange(1, 10)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    line = None
    if edge is not None:
        ax.axvspan(NU_LO, edge, color=FIT_FILL, lw=0, zorder=0)
        line = ax.axvline(edge, color=fs.GREY, lw=fs.LW_REF, zorder=0.5)
    if labels:
        ax.set_xticklabels([f"{t:g}" for t in FREQ_TICKS])
    else:
        ax.tick_params(axis="x", labelbottom=False)
    return line


def sed_panel(ax, z, js, name, models, nu, nu_fit, drawn) -> tuple[float, float, dict, tuple, object]:
    """Truth, the drawn 1% data over 1-100 GHz, the median predictions and the 16-84% ranges above the band."""
    truth, ipl = z[f"truth_{name}"], z[f"ipl_{name}"]
    ax.plot(nu, truth / ipl, **TRUTH_KW)
    data, sig = z[f"data_{name}_{SED_BAND}_{SED_TAG}"], z[f"sigma_{name}_{SED_BAND}_{SED_TAG}"]
    nf = nu_fit[:data.shape[0]]
    ref_f = np.exp(np.interp(np.log(nf), np.log(nu), np.log(ipl)))
    ax.errorbar(nf, data[:, 0] / ref_f, yerr=sig / ref_f, **DATA_KW)
    vals = [truth / ipl, (data[:, 0] - sig) / ref_f, (data[:, 0] + sig) / ref_f]
    above = nu >= BAND_EDGE[SED_BAND] * (1 - 1e-9)
    notes, med_vals = {}, []
    for model in models:
        k = kind(model)
        pred = z[f"{name}_{model}_{SED_BAND}_{SED_TAG}_ratio"] * truth[None, :] / ipl[None, :]
        med = np.median(pred, axis=0)
        ax.plot(nu, med, color=COLOUR[k] if SED_TAG == "0.0001" else tint(COLOUR[k]), ls=LS[k],
                lw=NOISE_STYLE[SED_TAG]["lw"], zorder=4)  # the width of the noise level, as in the error subpanels
        lo, hi = np.percentile(pred[:, above], 16, axis=0), np.percentile(pred[:, above], 84, axis=0)
        if k == "cum" and SED_TAG != "0.0001":  # dashed edge: the two forms' ranges stay apart in greyscale
            ax.fill_between(nu[above], lo, hi, facecolor=COLOUR[k], edgecolor=COLOUR[k], alpha=BAND_ALPHA[k],
                            lw=0.6, linestyle=LS[k], zorder=0.8)
        else:
            ax.fill_between(nu[above], lo, hi, color=COLOUR[k], alpha=BAND_ALPHA[k], lw=0, zorder=0.8)  # under the truth
        vals += [med, lo, hi]
        med_vals.append(med)
        if k == "cum" and js["summary"][name][model][SED_BAND][SED_TAG]["frac_at_bound"] == 1.0:
            notes[model] = "cumulant fits\nat bounds of $k$"
    line = freq_axis(ax, labels=False, edge=BAND_EDGE[SED_BAND])
    flat, mflat = (np.concatenate([np.ravel(v) for v in x]) for x in (vals, med_vals))
    return float(np.min(flat)), float(np.max(flat)), notes, (float(np.min(mflat)), float(np.max(mflat))), line


def error_panel(ax, z, js, name, models, band, nu, labels, band_label=True) -> dict:
    """Median error of fits over band at both noise levels; checked against the JSON and the axis limits."""
    metric = METRIC[name]
    lo, hi = ERR_LIM[metric]
    t_ipl = z[f"truth_{name}"] / z[f"ipl_{name}"]
    ax.axhline(1.0, color=NOISE_GREY, lw=fs.LW_REF, ls=(0, (1, 1.5)), zorder=1)
    out = {}
    for model in models:
        k = kind(model)
        for tag in NOISE:
            e = np.abs(z[f"{name}_{model}_{band}_{tag}_ratio"] - 1)
            if metric == "dI_over_IPL":
                e = e * t_ipl[None, :]
            m = np.median(e, axis=0)
            rec = js["summary"][name][model][band][tag]
            if not np.allclose(m, np.asarray(rec[metric]["median"]), rtol=1e-12, atol=0.0):
                raise SystemExit(f"{name} {model} {band} {tag}: drawn medians differ from the JSON")
            if not np.all(100 * m < hi):
                raise SystemExit(f"{name} {model} {band} {tag}: median up to {100 * m.max():.3g}% "
                                 f"leaves the axis {lo:g}-{hi:g}%")
            if np.any(100 * m <= lo):  # small in-band errors may pass below the axis; recorded
                out[f"{model}_{band}_{tag}_below_axis_GHz"] = [float(nu[100 * m <= lo].min()),
                                                               float(nu[100 * m <= lo].max())]
            ax.plot(nu, 100 * m, color=COLOUR[k] if tag == "0.0001" else tint(COLOUR[k]), ls=LS[k],
                    solid_capstyle="butt", zorder=3 + (tag == "0.0001"), **NOISE_STYLE[tag])
            out[f"{model}_{band}_{tag}_{metric}_300GHz_pct"] = float(100 * m[-1])
            out[f"{model}_{band}_{tag}_{metric}_100GHz_pct"] = float(100 * m[rec["dense_index_100GHz"]])
    freq_axis(ax, labels=labels, edge=BAND_EDGE[band])
    ax.set_yscale("log")
    ax.set_ylim(lo, hi)
    decades = range(int(np.ceil(np.log10(lo))), int(np.floor(np.log10(hi))) + 1)
    step = ERR_TICK_STEP[metric]
    major = [10.0**p for p in decades if p % step == 0 and lo < 10.0**p < hi]
    ax.yaxis.set_major_locator(FixedLocator(major))
    ax.yaxis.set_major_formatter(FuncFormatter(
        lambda v, _: f"{v:g}" if 1e-2 <= v <= 1e2 else rf"$10^{{{int(round(np.log10(v)))}}}$"))
    ax.yaxis.set_minor_locator(NullLocator())
    if band_label:  # first column only: the rows of error subpanels share their band across the columns
        ax.text(0.02, 0.9, f"fit {band.replace('-', '--')} GHz", transform=ax.transAxes, color=TEXT_GREY,
                ha="left", va="top", fontsize=FS, zorder=6,
                bbox=dict(boxstyle="square,pad=0.1", facecolor=FIT_FILL, edgecolor="none"))
    return out


def layout(fig, rows):
    """Panels in inches (LAY): one row per ensemble, each panel a spectrum over two error subpanels."""
    W, H, L = fig.get_figwidth(), fig.get_figheight(), LAY
    width = (W - L["left"] - L["right"] - sum(L["gap"])) / 3  # the first gap leaves room for y tick labels
    axes = {}
    y_top = H - L["legend"]
    for i, name in enumerate(rows):
        y = y_top - L["title"]
        for j in range(3):
            yy, cell = y, []
            x = L["left"] + j * width + sum(L["gap"][:j])
            for k, h in enumerate((L["sed"],) + (L["err"],) * len(ERR_BANDS)):
                yy -= h + (L["sub_gap"] if k else 0)
                cell.append(fig.add_axes([x / W, yy / H, width / W, h / H], sharex=cell[0] if cell else None))
            axes[(name, j)] = cell
        y_top = yy - L["row_gap"]
    return axes


def fig_height(n_rows: int) -> float:
    L = LAY
    row = L["title"] + L["sed"] + len(ERR_BANDS) * (L["sub_gap"] + L["err"])
    return L["legend"] + n_rows * row + (n_rows - 1) * L["row_gap"] + L["ticks"] + L["xlabel"] + L["bottom"]


class HandlerFan(HandlerPatch):
    """Legend swatch of a 16-84% range: a wedge opening to the right, like the fan above the band."""

    def create_artists(self, legend, orig_handle, xdescent, ydescent, width, height, fontsize, trans):
        x0, y0 = -xdescent, -ydescent
        yc = y0 + height / 2
        pts = [(x0, yc - 0.06 * height), (x0 + width, y0 - 0.15 * height), (x0 + width, y0 + 1.15 * height),
               (x0, yc + 0.06 * height)]
        return [Polygon(pts, closed=True, facecolor=orig_handle.get_facecolor(), edgecolor="none", transform=trans)]


def legend(fig, col_x):
    """Three untitled groups, each left-aligned above one column of panels (col_x: figure-fraction x0):
    the spectrum elements; the medians, whose width gives the noise level in every column and in both the
    spectra and the errors; and the two electron-family forms of the third column."""
    lw_thin, lw_thick = NOISE_STYLE["0.0001"]["lw"], NOISE_STYLE["0.01"]["lw"]
    cols = (COLOUR["free"], COLOUR["mom"], COLOUR["cum"])
    thin = tuple(Line2D([], [], color=c, lw=lw_thin, solid_capstyle="butt") for c in cols)
    thick = tuple(Line2D([], [], color=tint(c), lw=lw_thick, solid_capstyle="butt") for c in cols)
    groups = [([Line2D([], [], **{k: v for k, v in TRUTH_KW.items() if k != "zorder"}),
                Line2D([], [], marker="o", ls="", ms=2.4, color=fs.INK),
                Patch(color=COLOUR["free"], alpha=BAND_ALPHA["free"], lw=0)],
               ["true spectrum", rf"data, ${100 * float(SED_TAG):g}\%$ noise", r"$16$--$84\%$ of 500 fits"]),
              ([thin, thick, Line2D([], [], color=NOISE_GREY, lw=fs.LW_REF, ls=(0, (1, 1.5)))],
               [r"median, $0.01\%$ noise", r"median, $1\%$ noise", r"$1\%$ error"]),
              ([Line2D([], [], color=COLOUR["mom"], lw=lw_thin), Line2D([], [], color=COLOUR["cum"], lw=lw_thin,
                                                                       ls=LS["cum"])],
               ["moment form", "cumulant form"])]
    top = 1.0 - LEGEND_INSET / fig.get_figheight()
    for x, (handles, labels) in zip(col_x, groups):
        fig.legend(handles=handles, labels=labels, loc="upper left", bbox_to_anchor=(x, top), ncol=1,
                   fontsize=LEGEND_FS, frameon=False, borderpad=0.0, borderaxespad=0.0, handlelength=2.4,
                   handletextpad=0.5, labelspacing=0.3,
                   handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0), Patch: HandlerFan()})


def set_sed_limits(ax, lo, hi, fixed=None) -> list:
    lim = fixed if fixed is not None else (lo - 0.06 * (hi - lo), hi + 0.06 * (hi - lo))
    ax.set_ylim(*lim)
    return list(lim)


def main():
    ap = argparse.ArgumentParser(description="extrapolation figure: free moments, non-negative cells, electron family")
    ap.add_argument("rows", nargs="*", default=list(DEFAULT_ROWS))
    ap.add_argument("--orders", default="2,4", help="two truncation orders, e.g. 2,4")
    ap.add_argument("--columns", default=None,
                    help="three column groups, e.g. 'free;pos;mom2,cum2'; default from --source (forms: --orders)")
    ap.add_argument("--source", choices=tuple(SOURCES), default="ridge")
    ap.add_argument("--bands", default="1-40,1-100", help="fitted bands of the error subpanels")
    ap.add_argument("--noise", default="0.01,0.0001", help="noise levels drawn in the error subpanels")
    ap.add_argument("--sed-band", default="1-100")
    ap.add_argument("--sed-noise", default="0.0001")
    ap.add_argument("--axis-note", action=argparse.BooleanOptionalAction, default=False,
                    help="note free-moment panels whose content leaves the fixed spectrum axis")
    ap.add_argument("--bound-note", action=argparse.BooleanOptionalAction, default=False,
                    help="note panels whose cumulant fits all end on a bound of k")
    ap.add_argument("--name", default=None, help="output name; default from --source (DEFAULT_NAME)")
    args = ap.parse_args()
    if args.name is None:
        args.name = DEFAULT_NAME[args.source]
    if args.columns is None:
        args.columns = DEFAULT_COLUMNS[args.source]
    global SED_BAND, SED_TAG, NOISE, ERR_BANDS
    sed_lim_free = SED_LIM_FREE
    SED_BAND, SED_TAG = args.sed_band, args.sed_noise
    NOISE, ERR_BANDS = tuple(args.noise.split(",")), tuple(args.bands.split(","))
    products = HERE / f"{SOURCES[args.source]}_products.npz"
    results = HERE / f"{SOURCES[args.source]}_results.json"
    out_json = HERE / f"fig_{args.name}_results.json"
    rows = tuple(args.rows)
    orders = tuple(int(o) for o in args.orders.split(","))
    if not set(rows) <= set(ALL_ROWS) or len(orders) != 2 or not set(orders) <= set(ORDER_NAME):
        raise SystemExit(f"rows must be among {ALL_ROWS} and two orders among {tuple(ORDER_NAME)}")
    columns = (("free",),) + tuple((f"mom{o}", f"cum{o}") for o in orders)
    titles = ("free moments",) + tuple(ORDER_NAME[o] for o in orders)
    if args.columns:
        columns = tuple(tuple(g.split(",")) for g in args.columns.split(";"))
        if len(columns) != 3:
            raise SystemExit("--columns needs three groups")
        titles = tuple(GROUP_TITLE.get(",".join(g), ORDER_NAME.get(int(g[0][-1]), g[0]) if g[0][:3] in ("mom", "cum")
                                       else g[0]) for g in columns)
    pair_free = (args.source in ("positive", "ridge") and columns[0] == ("free",)
                 and columns[1] in (("pos",), ("posridge",)))
    if pair_free:  # the third column carries its own y tick labels, the second shares the first's
        LAY["gap"] = (PAIR_GAP_IN, 0.41)  # room for the third column's error tick labels
    fs.configure()
    z = np.load(products)
    js = json.loads(results.read_text())
    drawn = int(js["drawn_realisation"])
    nu, nu_fit = z["centres_dense"] / 1e9, z["centres_fit"] / 1e9
    fig = plt.figure(figsize=(fs.WIDTH, fig_height(len(rows))))
    axes = layout(fig, rows)
    report, letters = {"rows": list(rows), "orders": list(orders), "columns": [list(g) for g in columns],
                       "argv": sys.argv[1:], "source": args.source, "bands": list(ERR_BANDS),
                       "noise": list(NOISE), "sed_band": SED_BAND, "sed_noise": SED_TAG}, iter("abcdefghijkl")
    for name in rows:
        ext, edge_lines, med_ranges, rec = {}, {}, {}, report.setdefault(name, {})
        for j, models in enumerate(columns):
            sed, *errs = axes[(name, j)]
            lo, hi, notes, med_range, edge_line = sed_panel(sed, z, js, name, models, nu, nu_fit, drawn)
            ext[j], edge_lines[j], med_ranges[j] = (lo, hi), edge_line, med_range
            sed.set_title(f"({next(letters)}) {name}, {titles[j]}", fontsize=8.5)
            if notes and args.bound_note:
                sed.text(0.97, 0.94, next(iter(notes.values())), transform=sed.transAxes, color="#A84200",
                         ha="right", va="top", fontsize=FS, linespacing=1.1, zorder=7)
                edge_line.set_ydata([0.0, NOTE_Y_EDGE])  # the band-edge line stops below the note
                rec[f"col{j}_note"] = notes
            for ax, band in zip(errs, ERR_BANDS):
                rec.update(error_panel(ax, z, js, name, models, band, nu, labels=name == rows[-1] and ax is errs[-1],
                                       band_label=j == 0))
            if j:
                for a, b in zip(errs, axes[(name, 0)][1:]):
                    a.sharey(b)
                    a.tick_params(axis="y", labelleft=pair_free and j == 2)  # the third column has its own labels
        free_ax = axes[(name, 0)][0]
        if pair_free:  # free and free >= 0 share one spectrum range; the form column keeps its own
            ext[0] = (min(ext[0][0], ext[1][0]), max(ext[0][1], ext[1][1]))
        rec["sed_ylim_free"] = set_sed_limits(free_ax, *ext[0], sed_lim_free.get(name))
        if args.axis_note and name in sed_lim_free and (ext[0][0] < sed_lim_free[name][0]
                                                         or ext[0][1] > sed_lim_free[name][1]):
            lim = sed_lim_free[name]
            median_out = med_ranges[0][0] < lim[0] or med_ranges[0][1] > lim[1]
            text = "median and 16--84\\%\nrange leave the axis" if median_out else "16--84\\% range\nleaves the axis"
            free_ax.text(0.97, 0.94, text, transform=free_ax.transAxes,
                         color=TEXT_GREY, ha="right", va="top", fontsize=FS, linespacing=1.1, zorder=7)
            edge_lines[0].set_ydata([0.0, NOTE_Y_EDGE])
            rec["sed_free_content_range"], rec["sed_free_median_range"] = list(ext[0]), list(med_ranges[0])
        if pair_free:
            axes[(name, 1)][0].sharey(free_ax)
            axes[(name, 1)][0].tick_params(axis="y", labelleft=False)
            rec["sed_ylim_forms"] = set_sed_limits(axes[(name, 2)][0], *ext[2])
        else:
            flo, fhi = min(ext[1][0], ext[2][0]), max(ext[1][1], ext[2][1])
            rec["sed_ylim_forms"] = set_sed_limits(axes[(name, 1)][0], flo, fhi)
            axes[(name, 2)][0].sharey(axes[(name, 1)][0])
            axes[(name, 2)][0].tick_params(axis="y", labelleft=False)
        W = fig.get_figwidth()
        sed0, e10, *rest = axes[(name, 0)]
        sed0.set_ylabel(r"$I/I_{\rm PL}$")
        e10.set_ylabel(METRIC_LABEL[METRIC[name]])
        p1, p2 = e10.get_position(), (rest[-1] if rest else e10).get_position()
        x = -YLABEL_X_IN / (p1.width * W)
        sed0.yaxis.set_label_coords(x, 0.5)
        e10.yaxis.set_label_coords(x, ((p1.y1 + p2.y0) / 2 - p1.y0) / p1.height)
    fig.text(0.53, LAY["bottom"] / fig.get_figheight(), "Frequency [GHz]", ha="center", va="bottom")
    legend(fig, [axes[(rows[0], j)][0].get_position().x0 for j in range(3)])
    out_dir = HERE / "figures"
    out_dir.mkdir(exist_ok=True)
    paths = [out_dir / f"{args.name}.pdf", out_dir / f"{args.name}.png"]
    fig.savefig(paths[0])
    fig.savefig(paths[1], dpi=220)
    plt.close(fig)
    write_json(out_json, {"description": __doc__, "drawn_realisation": drawn, "values": report,
                          "outputs": {relative(p): sha256(p) for p in paths},
                          "provenance": provenance(__file__, T0, inputs=(products, results))})
    print("rows", rows, "written", [relative(p) for p in paths])


if __name__ == "__main__":
    main()
