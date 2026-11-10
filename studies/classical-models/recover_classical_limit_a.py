"""Recover the Fig. 3(a) curves from the vector PDF drawn by the manuscript's render_old_panels.py.

The generator input of that panel (smooth-forward/results_fine.json in a removed Codex
worktree) no longer exists. Matplotlib wrote each 105-point curve as one unsimplified
PDF path (fewer than 128 vertices), so the data coordinates follow exactly from the
axis transform. The panel had log axes with xlim = (3, 1000) Hz and ylim = (1e-4, 100).

The PDF is a byte copy of figures/classical_limit.pdf at manuscript commit 11b2f95
(inputs/manuscript-fig3/). Checks against inputs/manuscript-fig3/render_old_panels_manifest.json:
  * every vertex inside the canvas lies on np.geomspace(3, 1000, 105) Hz,
  * the peak values and positions of the three curves.
Vertices that matplotlib clipped at the canvas edge lie below the axes and are kept as
recovered so that a redraw with ymin >= 1e-4 reproduces the visible curves.

Run from this directory:  python recover_classical_limit_a.py
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
from pdfminer.high_level import extract_pages
from pdfminer.layout import LTCurve

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "inputs/manuscript-fig3/classical_limit_11b2f95.pdf"
MANIFEST = HERE / "inputs/manuscript-fig3/render_old_panels_manifest.json"
OUT = HERE / "recover_classical_limit_a_results.json"

XLIM, YLIM = (3.0, 1000.0), (1e-4, 100.0)
GRID = np.geomspace(3.0, 1000.0, 105)
# Stroke colour of each curve in the old panel (Okabe-Ito blue, vermilion, green).
CURVES = {
    "narrowB_1.1-1.2": dict(rgb=(0.0, 0.447, 0.698), gamma=(1.1, 1.2), B_muG=(4.5, 5.5)),
    "broadB_1.1-1.2": dict(rgb=(0.835, 0.369, 0.0), gamma=(1.1, 1.2), B_muG=(2.0, 8.0)),
    "broadB_2-4": dict(rgb=(0.0, 0.620, 0.451), gamma=(2.0, 4.0), B_muG=(2.0, 8.0)),
}


def walk(obj):
    if isinstance(obj, LTCurve):
        yield obj
    if hasattr(obj, "__iter__"):
        for child in obj:
            yield from walk(child)


def axes_box(items: list[LTCurve], x_max: float) -> tuple[float, float, float, float]:
    """Left and bottom spines of the first panel give its axes rectangle."""
    spines = [c for c in items if c.x0 < x_max and len(c.pts) == 2 and c.linewidth == 0.6]
    left = next(c for c in spines if abs(c.pts[0][0] - c.pts[1][0]) < 1e-6 and c.height > 50)
    bottom = next(c for c in spines if abs(c.pts[0][1] - c.pts[1][1]) < 1e-6 and c.width > 50)
    return left.pts[0][0], bottom.pts[1][0], bottom.pts[0][1], left.pts[1][1]


def to_data(points: np.ndarray, box: tuple[float, float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    x0, x1, y0, y1 = box
    lx = np.log10(XLIM[0]) + (points[:, 0] - x0) / (x1 - x0) * np.log10(XLIM[1] / XLIM[0])
    ly = np.log10(YLIM[0]) + (points[:, 1] - y0) / (y1 - y0) * np.log10(YLIM[1] / YLIM[0])
    return 10**lx, 10**ly


def recover() -> dict:
    digest = hashlib.sha256(SOURCE.read_bytes()).hexdigest()
    page = next(iter(extract_pages(str(SOURCE))))
    items = list(walk(page))
    box = axes_box(items, page.width / 3)
    paths = [c for c in items if c.x0 < page.width / 3 and len(c.pts) > 10]
    manifest = json.loads(MANIFEST.read_text())["summary_numbers_from_plotted_arrays"]["a"]
    curves, checks = {}, {}
    for name, meta in CURVES.items():
        path = next(p for p in paths if np.allclose(p.stroking_color, meta["rgb"], atol=2e-3))
        x, y = to_data(np.asarray(path.pts, float), box)
        index = np.abs(np.log(x[:, None] / GRID[None, :])).argmin(axis=1)
        offset = np.abs(np.log(x / GRID[index]))
        on_grid = offset < 1e-4
        x = np.where(on_grid, GRID[index], x)  # snap to the exact grid nodes
        peak = int(np.argmax(y))
        want = manifest[name]
        checks[name] = dict(
            n_vertices=len(x), n_on_grid=int(on_grid.sum()),
            max_log_offset_on_grid=float(offset[on_grid].max()),
            peak_relative=float(y[peak]), manifest_peak_relative=want["peak_relative"],
            peak_hz=float(x[peak]), manifest_peak_hz=want["peak_hz"],
        )
        if abs(y[peak] / want["peak_relative"] - 1) > 1e-6 or abs(x[peak] / want["peak_hz"] - 1) > 1e-4:
            raise AssertionError(f"{name}: recovered peak does not match the manifest")
        if not on_grid[(y >= YLIM[0])].all():
            raise AssertionError(f"{name}: a visible vertex lies off the 105-point grid")
        curves[name] = dict(gamma=meta["gamma"], B_muG=meta["B_muG"],
                            nu_hz=x.tolist(), I_rel=y.tolist(), on_grid=on_grid.tolist())
    return dict(
        source=str(SOURCE.relative_to(HERE)), source_sha256=digest,
        source_git_commit="11b2f95 (last change of figures/classical_limit.pdf)",
        method="pdfminer vertex extraction, inverse log-axis transform, snap to geomspace(3,1000,105)",
        axes_box_pt=[float(v) for v in box], xlim_hz=XLIM, ylim=YLIM,
        normalisation="peak of the broad-field gamma 1.1-1.2 curve (as in the old panel)",
        ensemble="uniform in gamma and B, isotropic pitch, eta uniform on [-1/2,1/2], f = 8% channels",
        curves=curves, checks=checks,
    )


if __name__ == "__main__":
    result = recover()
    OUT.write_text(json.dumps(result, indent=1))
    print(json.dumps(result["checks"], indent=1))
