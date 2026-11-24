"""Paired Figure 1: spatial geometry and harmonic lines in a fixed channel.

Run ``python -m validation.single_particle_figure`` from the repository root.
Both panels use the original checked drawing functions at the printed font size.
"""

from pathlib import Path
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .angle_geometry import geometry
from .narrative_figures import moving_lines
from .figure_style import WIDTH, configure, save

ROOT = Path(__file__).resolve().parents[1]
NAME = "single_particle_geometry_channels"


def paired_figure():
    """Draw both panels without shrinking text or changing harmonic inputs."""
    configure()
    fig = plt.figure(figsize=(WIDTH, 2.65))
    geometry_ax = fig.add_axes([0.015, 0.07, 0.485, 0.83])
    channel_ax = fig.add_axes([0.59, 0.20, 0.395, 0.66])
    record = {
        "geometry": geometry(geometry_ax),
        "moving_lines": moving_lines(channel_ax),
    }
    channel_ax.set(xlabel=r"Frequency $\nu/\nu_*$", ylabel="Normalised line power")
    channel_ax.set_xticks([2, 4, 6, 8])
    channel_ax.set_yticks([0, 0.5, 1.0, 1.5])
    fig.text(0.025, 0.955, "(a) Geometry", ha="left", va="top")
    fig.text(0.59, 0.955, "(b) Harmonics and channel", ha="left", va="top")
    save(fig, NAME)
    record["scope"] = (
        "Panel (a) is a generic direction schematic. Panel (b) retains the "
        "original gamma=20, mu=eta=0.5 example; the panel angles are not shared."
    )
    return record


def main():
    result = paired_figure()
    inputs = (
        "validation/single_particle_figure.py",
        "validation/angle_geometry.py",
        "validation/narrative_figures.py",
        "validation/figure_style.py",
        "validation/full_response.py",
    )
    result["source_sha256"] = {
        name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest() for name in inputs
    }
    result["output_sha256"] = {
        f"figures/{NAME}.{ext}": hashlib.sha256(
            (ROOT / "figures" / f"{NAME}.{ext}").read_bytes()
        ).hexdigest()
        for ext in ("pdf", "png")
    }
    (ROOT / "validation/single_particle_figure_results.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
