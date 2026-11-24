"""Shared manuscript style: final-width figures with the manuscript's TeX fonts and one palette.

Every manuscript figure is drawn at its printed size (WIDTH = text width, 440 pt, or
HALF_WIDTH), in newtx at 8-9 pt, with left-aligned "(a) Title" panel headings, left and
bottom spines only, no grid and frameless legends.

Palette (Okabe & Ito, colour-blind safe), with fixed roles across the paper:
  INK      direct, exact or reference curves
  SERIES   ordered sweeps (first, second, third, fourth curve of a family)
  ENSEMBLE the reference ensembles of Sec. 5; E1 is the ratio reference and is drawn in INK,
           and the two ageing variants share one colour with different line styles
  GREY     secondary reference lines and annotations; LIGHT for bands and shading
"""

from collections.abc import Sequence
from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import matplotlib.pyplot as plt
from cycler import cycler
from matplotlib.axes import Axes
from matplotlib.text import Text
from matplotlib.figure import Figure
from matplotlib.transforms import Bbox

WIDTH = 6.1
HALF_WIDTH = WIDTH / 2
ROOT = Path(__file__).resolve().parents[1]

BLUE, VERMILION, GREEN, PURPLE, ORANGE, SKY = (
    "#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
)
INK, GREY, LIGHT = "#222222", "#767676", "#E6E6E6"
SERIES = (BLUE, VERMILION, GREEN, PURPLE)
SERIES_MARKER = ("o", "s", "^", "D")  # same order: identity survives greyscale printing
ENSEMBLE = {
    "E1": INK, "E1M": INK, "E2": ORANGE, "E3": GREEN,
    "E4-JP": PURPLE, "E4-KP": PURPLE, "E5": BLUE,
}
ENSEMBLE_LS = {
    "E1": "-", "E1M": ":", "E2": "-", "E3": "-",
    "E4-JP": "-", "E4-KP": "--", "E5": "-.",
}
# Sparse markers that keep the ensembles apart in greyscale where several share one panel (Fig. 8);
# the two ageing variants share a marker as they share a colour.
ENSEMBLE_MARKER = {
    "E1": None, "E1M": None, "E2": "s", "E3": "o",
    "E4-JP": "^", "E4-KP": "^", "E5": None,
}
LW, LW_REF = 1.1, 0.6


def configure() -> None:
    """Use newtx text and mathematics at readable final publication sizes."""
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8.5,
            "text.usetex": True,
            "text.latex.preamble": r"\usepackage{newtxtext,newtxmath}",
            "axes.labelsize": 8.5,
            "axes.titlesize": 8.5,
            "axes.titlelocation": "left",
            "axes.titlepad": 4.0,
            "legend.fontsize": 8,
            "legend.frameon": False,
            "legend.handlelength": 1.8,
            "legend.borderaxespad": 0.3,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "xtick.major.size": 3.0,
            "ytick.major.size": 3.0,
            "xtick.minor.size": 1.8,
            "ytick.minor.size": 1.8,
            "xtick.major.width": 0.6,
            "ytick.major.width": 0.6,
            "xtick.minor.width": 0.5,
            "ytick.minor.width": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.6,
            "axes.grid": False,
            "axes.prop_cycle": cycler(color=SERIES),
            "lines.linewidth": LW,
            "lines.markersize": 3.5,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_title(ax: Axes, text: str) -> None:
    """Left-aligned panel heading, e.g. "(a) Electron spectra"."""
    ax.set_title(text, loc="left", pad=4.0)


# Horizontal packing of a row of panels (points).
ROW_GAP_PT = 6.0  # clear space between any item of one panel and any item of the next at its height
ROW_V_CLEAR_PT = 2.0  # items closer than this vertically count as being at the same height
ROW_EDGE_PT = 1.5  # clear space at the left and right figure edges
Y_LABELPAD_PT = 2.0  # y-label to tick-label distance in packed rows (matplotlib default 4)
Y_TICKPAD_PT = 2.0  # tick-label to tick distance in packed rows (matplotlib default 3.5)


def tighten_y_labels(axes: Sequence[Axes]) -> None:
    """Pull y labels and y tick labels towards their axis to free horizontal space."""
    for ax in axes:
        ax.yaxis.labelpad = Y_LABELPAD_PT
        ax.tick_params(axis="y", which="both", pad=Y_TICKPAD_PT)


def panel_items(ax: Axes, renderer) -> list[Bbox]:
    """Display-space extents of everything a panel draws: plot area, titles, axis labels,
    tick labels inside the view, legend, and free text such as annotations."""
    items = [ax.get_window_extent(renderer)]
    # Direct Text children are the three title slots, ax.text and annotations.
    texts = [c for c in ax.get_children() if isinstance(c, Text)]
    texts += [ax.xaxis.label, ax.yaxis.label]
    for axis, (lo, hi) in ((ax.xaxis, ax.get_xlim()), (ax.yaxis, ax.get_ylim())):
        lo, hi = min(lo, hi), max(lo, hi)
        for tick in axis.get_major_ticks() + axis.get_minor_ticks():
            if tick.label1.get_visible() and lo <= tick.get_loc() <= hi:
                texts.append(tick.label1)
    items += [t.get_window_extent(renderer) for t in texts if t.get_visible() and t.get_text()]
    if ax.get_legend() is not None:
        items.append(ax.get_legend().get_window_extent(renderer))
    return items


def _same_height(a: Bbox, b: Bbox, margin: float) -> bool:
    return a.y0 < b.y1 + margin and b.y0 < a.y1 + margin


def pack_row(fig: Figure, axes: Sequence[Axes], gap_pt: float = ROW_GAP_PT,
             v_clear_pt: float = ROW_V_CLEAR_PT, edge_pt: float = ROW_EDGE_PT) -> dict:
    """Give a row of panels equal widths and the narrowest gaps their decorations allow.

    The spacing between panel i's right edge and panel i+1's left edge is the smallest
    that keeps every item of panel i (panel_items) gap_pt clear of every item of panel
    i+1 at the same height (vertical tolerance v_clear_pt). Items at different heights may
    share horizontal space, so the last x tick label of one panel does not push the next
    panel's y label away. Offsets are measured from the panel edge each item is anchored
    to; widening a panel only moves its items away from the gaps, so one measurement is
    conservative. Vertical placement is unchanged. Call after all panel content except
    text whose placement needs the final transforms; such text must stay inside its axes.

    Raises if the packed row overlaps, comes closer than gap_pt or leaves the figure.
    """
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    pt = fig.dpi / 72.0  # display units per point
    margin = v_clear_pt * pt
    boxes = [ax.get_window_extent(renderer) for ax in axes]
    items = [panel_items(ax, renderer) for ax in axes]
    spacing = []  # right edge of panel i to left edge of panel i+1, in points
    for i in range(len(axes) - 1):
        need = [(a.x1 - boxes[i].x1) - (b.x0 - boxes[i + 1].x0)
                for a in items[i] for b in items[i + 1] if _same_height(a, b, margin)]
        spacing.append(max(need) / pt + gap_pt)
    left = (boxes[0].x0 - min(b.x0 for b in items[0])) / pt
    right = max(max(b.x1 for b in items[-1]) - boxes[-1].x1, 0.0) / pt
    total_pt = fig.get_figwidth() * 72.0
    width_pt = (total_pt - 2 * edge_pt - left - right - sum(spacing)) / len(axes)
    if width_pt <= 0:
        raise ValueError(f"row does not fit: panel width {width_pt:.1f} pt")
    x_pt = edge_pt + left
    for i, ax in enumerate(axes):
        pos = ax.get_position()
        ax.set_position([x_pt / total_pt, pos.y0, width_pt / total_pt, pos.height])
        x_pt += width_pt + (spacing[i] if i < len(spacing) else 0.0)
    fig.canvas.draw()
    items = [panel_items(ax, renderer) for ax in axes]
    clear = [min(b.x0 - a.x1 for a in items[i] for b in items[i + 1] if _same_height(a, b, margin)) / pt
             for i in range(len(axes) - 1)]
    lo_x, hi_x = min(b.x0 for b in items[0]), max(b.x1 for b in items[-1])
    if (clear and min(clear) < gap_pt - 0.25) or lo_x < 0 or hi_x > total_pt * pt:
        raise AssertionError(f"packed row overlaps or leaves the figure: clear gaps {clear} pt")
    return dict(panel_width_pt=round(float(width_pt), 2),
                spacing_pt=[round(float(s), 2) for s in spacing],
                min_clear_gap_pt=[round(float(c), 2) for c in clear])


def save(fig: Figure, name: str, width: float = WIDTH) -> None:
    """Save fixed-width PDF/PNG pairs without shrinking the typeset labels."""
    fig.set_figwidth(width)
    for label in fig.findobj(Text):
        label.set_fontsize(min(9, max(8, label.get_fontsize())))
    # Finalize locally before replacing files in the synced manuscript directory.
    with TemporaryDirectory(prefix="synchro-figure-") as temporary:
        base = Path(temporary)
        fig.savefig(base / f"{name}.pdf")
        fig.savefig(base / f"{name}.png", dpi=200)
        for extension in ("pdf", "png"):
            destination = ROOT / "figures" / f"{name}.{extension}"
            staged = destination.with_suffix(f".{extension}.tmp")
            shutil.copyfile(base / destination.name, staged)
            staged.replace(destination)
    plt.close(fig)
