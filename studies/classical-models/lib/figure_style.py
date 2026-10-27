"""Shared manuscript style: final-width figures with the manuscript's TeX fonts."""

from pathlib import Path
from tempfile import TemporaryDirectory
import shutil
import matplotlib.pyplot as plt
from matplotlib.text import Text
from matplotlib.figure import Figure

WIDTH = 6.1
ROOT = Path(__file__).resolve().parents[1]


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
            "legend.fontsize": 8,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def save(fig: Figure, name: str) -> None:
    """Save fixed-width PDF/PNG pairs without shrinking the typeset labels."""
    fig.set_figwidth(WIDTH)
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
