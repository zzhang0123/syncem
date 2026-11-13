"""Energy basis spectra D_{j;br} of three cells (figures/harmonic_derivative_spectra).

Columns D^X_{j;br}, r = 0, 1, 2, of three energy cells, from basis.derivative_spectra: the continuum
(beaming-limit) kernels averaged over the field-strength, field-direction, azimuth and screen
distributions of the paper configuration and integrated over the channel response (k = 1,
h_j = 0.2 nu_j). A column is a smooth function of the channel centre, so the curves evaluate it
for DENSE log-spaced centres over 0.1-10 GHz. Every tenth of them is one of the 41 paper channels
(checked to equality). The polarised columns are P = <e^{2i phi}> D^Q, drawn after removing the
mean screen rotation at each centre, P exp(-2i <Psi> lambda_j^2). The "field" frame used here also
removes the phase of <e^{2i phi}>, so Q' and U' are in the frame of the mean projected field.
All panels are divided by A = max_{j, b shown} D^I_{j;b0} over the 41 channels.

The results JSON records the largest |U'| of the highest cell at r = 2 (panel i), whose curve is
left unlabelled because it stays within CAPTION_BOUND of zero, with that check. Label positions
are placed automatically, except those in LABEL_AT, which are placed by hand.
Style: lib/figure_style_paper.py.

Writes figures/harmonic_derivative_spectra.{pdf,png}, derivative_spectra_figure.npz (the plotted
arrays) and derivative_spectra_figure_results.json.  Run: python derivative_spectra_figure.py
"""
from __future__ import annotations

import json
import sys
import time
import warnings
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, MaxNLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import basis as B  # noqa: E402
import figure_style_paper as fs  # noqa: E402
from config import PAPER  # noqa: E402
from kernels import C_SI  # noqa: E402
from provenance import provenance, relative, sha256, write_json  # noqa: E402

T0 = time.time()
NPZ = HERE / "derivative_spectra_figure.npz"
OUT_JSON = HERE / "derivative_spectra_figure_results.json"
CAPTION_BOUND = 2e-4  # stated bound on |U'| of the unlabelled curve in panel (i)

NAME = "harmonic_derivative_spectra"
BINS = (3, 6, 8)  # indices into PAPER.gamma_b, low to high energy
MARKERS = ("o", "s", "^")  # one shape per bin, in every panel
DENSE = 10 * (PAPER.n_channels - 1) + 1  # every tenth dense centre is a paper channel
MARK_EVERY = 5  # identity markers on every fifth paper channel
MARK_FLOOR = 0.04  # ... where |curve| exceeds this fraction of the panel's largest |value|
FRAME = "field"  # P-row representation drawn in the figure
GHZ = 1e9


def columns(cfg=PAPER) -> dict:
    """Columns of the shown bins on the dense centres and the paper channels, normalised by A."""
    bins = list(BINS)
    dense = replace(cfg, n_channels=DENSE)
    DI, DQ = B.derivative_spectra(dense, 2)
    DI41, DQ41 = B.derivative_spectra(cfg, 2)
    if not (np.array_equal(dense.centres[::10], cfg.centres)
            and np.array_equal(DI[:, :, ::10], DI41) and np.array_equal(DQ[:, :, ::10], DQ41)):
        raise RuntimeError("dense centres do not reproduce the paper channels")
    A = float(DI41[bins, 0].max())
    sky = np.exp(-2j * cfg.screen_mean * (C_SI / dense.centres) ** 2)
    P = cfg.azimuth_factor * DQ[bins] * sky[None, None, :] / A
    return dict(nu=dense.centres / GHZ, nu41=cfg.centres / GHZ, gamma_b=cfg.gamma_b[bins], A=A,
                I=DI[bins] / A, P_sky=P, P_field=P * np.exp(-1j * np.angle(cfg.azimuth_factor)))


def _frequency_axis(ax, nu41, label: bool) -> None:
    ax.set_xscale("log")
    ax.set_xlim(0.09, 11.0)  # as figure_plots.ghz_axis
    ax.xaxis.set_major_locator(FixedLocator([0.1, 1.0, 10.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(FixedLocator(nu41))  # minor ticks: the 41 paper channel centres
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.tick_params(axis="x", which="minor", length=2.2, width=0.4, color=fs.GREY)
    if label:
        ax.set_xlabel("Frequency [GHz]")
    else:
        ax.tick_params(labelbottom=False)


def _curve(ax, nu, y, k, filled: bool, scale: float, ls="-") -> None:
    """Dense curve with identity markers on every MARK_EVERY-th paper channel.

    Markers are omitted where the curve lies within MARK_FLOOR * scale of zero, where all
    curves overlap the zero line and a marker would identify nothing.
    """
    color = fs.SERIES[k]
    ax.plot(nu, y, color=color, ls=ls, lw=fs.LW, zorder=2)
    step = 10 * MARK_EVERY
    idx = np.arange(10 * k, nu.size, step)  # paper channels, offset by one channel per bin
    keep = idx[np.abs(y[idx]) > MARK_FLOOR * scale]
    # every curve keeps at least one marker, at the marked channel of its largest |value|
    idx = keep if keep.size else idx[[int(np.argmax(np.abs(y[idx])))]]
    ax.plot(nu[idx], y[idx], ls="", marker=MARKERS[k], ms=4.4 if MARKERS[k] == "^" else 3.8, zorder=3,
            mfc=color if filled else "white", mec=color if not filled else "white",
            mew=0.9 if not filled else 0.5)


# One common signed display scale; match the linear/log slopes at +/- LINTHRESH.
LINTHRESH = 1e-4
SYMLOG_BASE = 10.0
LINSCALE = (1 - 1 / SYMLOG_BASE) / np.log(SYMLOG_BASE)
RESPONSE_LIMITS = (-1.2, 10.0)  # one decade of headroom above 1 holds the (a) label of the 18372 column


def _response_axis(ax, d, label_bottom: bool, label_left: bool) -> None:
    ax.set_yscale("symlog", base=SYMLOG_BASE, linthresh=LINTHRESH, linscale=LINSCALE)
    ax.set_ylim(*RESPONSE_LIMITS)
    ax.yaxis.set_major_locator(FixedLocator([-1, -0.01, 0, 0.01, 1]))
    labels = {-1: r"$-1$", -0.01: r"$-10^{-2}$", 0: "0", 0.01: r"$10^{-2}$", 1: "$1$"}
    ax.yaxis.set_major_formatter(FuncFormatter(lambda value, _: labels.get(value, f"{value:g}")))
    ax.yaxis.set_minor_locator(FixedLocator([]))
    ax.tick_params(axis="y", labelleft=label_left)
    ax.axhspan(-LINTHRESH, LINTHRESH, color=fs.LIGHT, alpha=0.7, lw=0, zorder=-2)
    ax.axhline(0, color="#cfcfcf", lw=0.35, zorder=-1)
    _frequency_axis(ax, d["nu41"], label=label_bottom)
    # Ordinary log-frequency ticks replace the channel-centre comb.
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 5)))


# Cells are told apart by weight and opacity as well as hue, monotone in energy, so they stay distinct in
# greyscale; the heavy low-energy curve is drawn first and the thinner ones on top of it.
CELL_LW = (1.7, 1.1, 0.6)
CELL_ALPHA = (1.0, 1.0, 0.9)  # the hairline stays above 0.5 pt of ink in print
# Direct labels: placed automatically next to their own curve where no other curve is close (points).
LABEL_FONTSIZE = 8
LABEL_GAP_PT = 1.5  # clear space between a label and every curve and label
# (reach to own curve, extra clearance from others, largest ratio own/other distance); the last try is a
# fallback that only requires the own curve to be the nearest, and it is reported when used
LABEL_TRIES = ((4.0, 5.0, 0.5), (6.0, 3.0, 0.5), (9.0, 1.5, 0.5), (12.0, 0.0, 0.5), (12.0, 0.0, 1.0))
# Preferred label frequency [GHz] by (panel letter, cell), where the automatic choice would sit next to a
# frequency at which two cells cross or run together: the label then goes within LABEL_NEAR_DEX of it.
LABEL_NEAR = {("a", 1): 1.8, ("b", 1): 1.2, ("e", 1): 2.5, ("g", 0): 0.7, ("h", 0): 0.45, ("i", 0): 0.3}
LABEL_NEAR_DEX = 0.25
# Hand-placed label centres (frequency [GHz], value) where the automatic boxes, with their padding, find no
# clear slot: in (a) the 18372 label sits above the plateau of its curve, in the headroom above 1 where no
# other curve passes; on the right of (d) the 18372 curve lies 1.5-2 decades below the 6977 curve, so the
# label goes in that gap, just inside the curve it names.
LABEL_AT = {("a", 2): (4.5, 3.6), ("d", 2): (8.3, -0.045)}
# Unlabelled curves: in (i) the 18372 column stays within |value| < 1.7e-4, inside or at the edge of the
# shaded linear band, next to the other columns, so no label position identifies it; it is the thinnest line
# and is labelled in (g) and (h).
LABEL_SKIP = {("i", 2)}


def _display_curve(ax, nu, y) -> np.ndarray:
    """Curve in display coordinates, resampled so that consecutive points are at most 1 px apart."""
    p = ax.transData.transform(np.column_stack([nu, y]))
    n = np.maximum(1, np.ceil(np.hypot(*np.diff(p, axis=0).T)).astype(int))
    parts = [np.linspace(p[i], p[i + 1], n[i], endpoint=False) for i in range(len(p) - 1)]
    return np.vstack(parts + [p[-1:]])


def _distances(boxes, pts) -> np.ndarray:
    """Smallest distance from each box (rows x0, y0, x1, y1) to the points, in display units."""
    out = np.empty(len(boxes))
    for s in range(0, len(boxes), 512):
        b = boxes[s:s + 512, :, None]
        dx = np.maximum(np.maximum(b[:, 0] - pts[:, 0], pts[:, 0] - b[:, 2]), 0.0)
        dy = np.maximum(np.maximum(b[:, 1] - pts[:, 1], pts[:, 1] - b[:, 3]), 0.0)
        out[s:s + 512] = np.hypot(dx, dy).min(axis=1)
    return out


def _place_labels(ax, lines, texts, near=None, fixed=None) -> None:
    """Write each cell's label beside its own curve, clear of all curves and of the other labels.

    Candidate label boxes lie on a grid over the panel (1 pt across, 0.25 pt up), outside the shaded band. A box is accepted when its own curve is within
    `reach` and every other curve is at least `extra` farther away, so the nearest curve identifies the cell
    without relying on colour. The best box maximises the clearance from the other curves beyond its distance to its own (capped at 20 pt).
    Labels with the fewest acceptable boxes are placed first; the tries relax reach and clearance in turn.
    near maps a cell to a preferred frequency [GHz]; boxes centred farther than LABEL_NEAR_DEX from it are used
    only if none near it is acceptable. fixed maps a cell to a hand-placed label centre (GHz, value).
    """
    near, fixed = near or {}, fixed or {}
    fig = ax.figure
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72.0
    gap = LABEL_GAP_PT * pt
    frame = ax.get_window_extent(renderer)
    curves = [_display_curve(ax, line.get_xdata(), line.get_ydata()) for line in lines]
    sizes = []
    for text in texts:
        probe = ax.text(0, 0, text or "", fontsize=LABEL_FONTSIZE)
        extent = probe.get_window_extent(renderer)
        probe.remove()
        sizes.append((extent.width + pt, extent.height))

    def score(k, placed, reach, extra, fraction, prefer=True):
        w, h = sizes[k]
        cx, cy = np.meshgrid(np.arange(frame.x0 + pt + w / 2, frame.x1 - pt - w / 2, pt),
                             np.arange(frame.y0 + pt + h / 2, frame.y1 - pt - h / 2, pt / 4))
        if prefer and k in near:
            freq = ax.transData.inverted().transform(np.column_stack([cx.ravel(), cy.ravel()]))[:, 0]
            inside = np.abs(np.log10(freq / near[k])) <= LABEL_NEAR_DEX
            cx, cy = cx.ravel()[inside], cy.ravel()[inside]
        boxes = np.column_stack([cx.ravel() - w / 2, cy.ravel() - h / 2, cx.ravel() + w / 2, cy.ravel() + h / 2])
        band = ax.transData.transform([(1.0, -LINTHRESH), (1.0, LINTHRESH)])[:, 1]  # shaded linear band
        boxes = boxes[(boxes[:, 1] > band[1] + 0.5 * pt) | (boxes[:, 3] < band[0] - 0.5 * pt)]
        for b in placed:  # keep clear of the labels already written
            boxes = boxes[~((boxes[:, 0] < b[2] + gap) & (b[0] < boxes[:, 2] + gap)
                            & (boxes[:, 1] < b[3] + gap) & (b[1] < boxes[:, 3] + gap))]
        own = _distances(boxes, curves[k])
        keep = (own >= gap) & (own <= reach * pt)
        boxes, own = boxes[keep], own[keep]
        others = np.full(len(boxes), np.inf)
        for j, c in enumerate(curves):
            if j != k:
                others = np.minimum(others, _distances(boxes, c))
        keep = others >= np.maximum(np.maximum(gap, own + extra * pt), own / fraction)
        return own[keep] - np.minimum(others[keep] - own[keep], 20 * pt), boxes[keep]

    placed = []
    for k, xy in fixed.items():
        cx, cy = ax.transData.transform(xy)
        w, h = sizes[k]
        placed.append(np.array([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]))
        ax.text(*xy, texts[k], ha="center", va="center", fontsize=LABEL_FONTSIZE, color=fs.INK, zorder=5)
    free = [k for k in range(len(texts)) if k not in fixed and texts[k] is not None]
    order = sorted(free, key=lambda k: len(score(k, placed, *LABEL_TRIES[0])[0]))
    where = ax.get_title(loc="left")
    for k in order:
        for prefer, (reach, extra, fraction) in [(p, t) for p in (True, False) for t in LABEL_TRIES]:
            values, boxes = score(k, placed, reach, extra, fraction, prefer)
            if len(values):
                if k in near and not prefer:
                    warnings.warn(f"label {texts[k]!r} in panel {where!r}: no acceptable position near "
                                  f"{near[k]} GHz, placed elsewhere", stacklevel=2)
                if fraction == 1.0:
                    warnings.warn(f"label {texts[k]!r} in panel {where!r}: placed by the fallback rule "
                                  "(own curve only the nearest)", stacklevel=2)
                break
        else:
            raise RuntimeError(f"no clear position for the label {texts[k]!r} in panel {ax.get_title(loc='left')!r}")
        box = boxes[int(np.argmin(values))]
        placed.append(box)
        centre = ax.transData.inverted().transform(((box[0] + box[2]) / 2, (box[1] + box[3]) / 2))
        ax.text(*centre, texts[k], ha="center", va="center", fontsize=LABEL_FONTSIZE, color=fs.INK, zorder=5)


FIG_HEIGHT = 5.0  # inches
TOP_GAP = 0.50  # gap below the intensity row (tick labels and titles), in panel heights
POL_HSPACE = 0.32  # gap between the Q'/U' rows (titles only), in panel heights


def draw(d: dict, frame: str = FRAME):
    """Spectra on one symmetric logarithmic scale, with wider Q'/U' frequency axes and direct cell labels."""
    P = d[f"P_{frame}"]
    fig = plt.figure(figsize=(fs.WIDTH, FIG_HEIGHT))
    block = 3 + 2 * POL_HSPACE  # Q'/U' block height in panel heights
    outer = fig.add_gridspec(2, 1, left=0.085, right=0.99, top=0.955, bottom=0.09,
                             hspace=TOP_GAP / ((1 + block) / 2), height_ratios=[1, block])
    pol = outer[1].subgridspec(3, 1, hspace=POL_HSPACE)
    letters = iter("abcdefghi")
    master = None
    labelled = []

    def panel(cell, values, order, component, bottom, left, title):
        nonlocal master
        ax = fig.add_subplot(cell, sharex=master, sharey=master)
        if master is None:
            master = ax
        _response_axis(ax, d, label_bottom=bottom, label_left=left)
        lines = [ax.plot(d["nu"], y, color=fs.SERIES[k], ls="-", lw=CELL_LW[k], alpha=CELL_ALPHA[k],
                    solid_joinstyle="round", zorder=3 + 0.1 * k)[0] for k, y in enumerate(values)]
        fs.panel_title(ax, rf"({next(letters)}) {title}")
        labelled.append((ax, lines))
        return ax

    intensity = outer[0].subgridspec(1, 3, wspace=0.12)
    for order in range(3):
        ax = panel(intensity[0, order], d["I"][:, order], order, "I", False, order == 0,
                   rf"Intensity $I$, order $r={order}$")
        ax.tick_params(labelbottom=True)  # tick labels only: these axes do not line up with the rows below
    for order in range(3):
        pair = pol[order].subgridspec(1, 2, wspace=0.12)
        # Q' and U' are Re P' and Im P': Stokes Q and U in the frame of the mean projected field, after
        # removing the mean Faraday rotation at each channel centre
        for col, (symbol, values) in enumerate(((r"Q'", P.real), (r"U'", P.imag))):
            title = rf"Stokes ${symbol}$, order $r={order}$" if order == 0 else rf"${symbol}$, $r={order}$"
            panel(pair[0, col], values[:, order], order, symbol, order == 2, col == 0, title)
    fig.text(0.01, 0.5, r"Basis spectrum / $I_{\rm max}$", rotation=90, va="center")
    fig.canvas.draw()  # final layout: the labeller works in display coordinates
    names = [f"{g:.0f}" for g in d["gamma_b"]]
    for i, (ax, lines) in enumerate(labelled):
        letter = "abcdefghi"[i]
        near = {k: x for (p, k), x in LABEL_NEAR.items() if p == letter}
        fixed = {k: xy for (p, k), xy in LABEL_AT.items() if p == letter}
        texts = [rf"$\gamma_b={names[0]}$", *names[1:]] if i == 0 else list(names)
        texts = [None if (letter, k) in LABEL_SKIP else t for k, t in enumerate(texts)]
        _place_labels(ax, lines, texts, near, fixed)
    return fig


def draw_weak_polarisation(d: dict, frame: str = FRAME):
    """Separate detail view of the lowest-energy cell, with an explicit 10^3 scale."""
    values = 1e3 * d[f"P_{frame}"][0]
    fig, axes = plt.subplots(1, 3, figsize=(fs.WIDTH, 2.05), sharex=True,
                             gridspec_kw=dict(left=0.095, right=0.99, top=0.80, bottom=0.25,
                                              wspace=0.32))
    for r, ax in enumerate(axes):
        ax.axhline(0, color=fs.GREY, lw=fs.LW_REF, zorder=0)
        scale = float(np.abs(values[r]).max())
        _curve(ax, d["nu"], values[r].real, 0, filled=True, scale=scale)
        _curve(ax, d["nu"], values[r].imag, 0, filled=False, scale=scale, ls=(0, (4, 2)))
        _frequency_axis(ax, d["nu41"], label=True)
        ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
        fs.panel_title(ax, rf"({chr(97 + r)}) $r={r}$")
    axes[0].set_ylabel(r"$10^3\widetilde P'_{jbr}/A$")
    fig.suptitle(rf"Low-energy cell $\gamma_b={d['gamma_b'][0]:.0f}$: polarisation magnified $\times10^3$",
                 y=0.99, fontsize=8.5)
    key = [Line2D([], [], color=fs.INK, lw=fs.LW, marker="o", ms=3.6, mec="white", mew=0.5, label="Re"),
           Line2D([], [], color=fs.INK, lw=fs.LW, ls=(0, (4, 2)), marker="o", ms=3.6, mfc="white",
                  mec=fs.INK, mew=0.9, label="Im")]
    axes[2].legend(handles=key, loc="upper right", handlelength=1.5, borderaxespad=0.1)
    return fig


def unlabelled_bound(d: dict) -> dict:
    """Largest |value| of each skipped curve (LABEL_SKIP) against CAPTION_BOUND."""
    out = {}
    for letter, k in sorted(LABEL_SKIP):
        row = "abcdefghi".index(letter) - 3  # panels d-i: Q', U' at r = 0, 1, 2
        order, part = row // 2, ("real", "imag")[row % 2]
        peak = float(np.max(np.abs(getattr(d[f"P_{FRAME}"][k, order], part))))
        out[f"panel_{letter}_cell_{k}"] = {"gamma_b": float(d["gamma_b"][k]), "order": order,
                                           "component": "Q'" if part == "real" else "U'",
                                           "max_abs_over_A": peak, "bound": CAPTION_BOUND,
                                           "within_bound": bool(peak < CAPTION_BOUND)}
    return out


def main() -> None:
    fs.configure()
    d = columns()
    np.savez(NPZ, bins=np.array(BINS), frame=FRAME, **d)
    (HERE / "figures").mkdir(exist_ok=True)
    fs.save(draw(d), NAME)
    checks = unlabelled_bound(d)
    figures = {relative(HERE / "figures" / f"{NAME}.{e}"): sha256(HERE / "figures" / f"{NAME}.{e}")
               for e in ("pdf", "png")}
    write_json(OUT_JSON, {"description": __doc__, "bins": list(BINS), "gamma_b": d["gamma_b"].tolist(),
                          "A": d["A"], "frame": FRAME, "dense_centres": DENSE,
                          "dense_vs_paper_channels": "equal (checked in columns())",
                          "hand_placed_labels": {f"{p}_{k}": list(xy) for (p, k), xy in LABEL_AT.items()},
                          "unlabelled_curves": checks, "npz": relative(NPZ), "npz_sha256": sha256(NPZ),
                          "figures": figures,
                          "provenance": provenance(__file__, T0, inputs=(HERE / "lib" / "figure_style_paper.py",))})
    print(json.dumps(checks, indent=1))
    if not all(v["within_bound"] for v in checks.values()):
        raise SystemExit(f"an unlabelled curve exceeds {CAPTION_BOUND:g}")


if __name__ == "__main__":
    main()
