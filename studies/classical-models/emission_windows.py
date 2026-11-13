"""Where the reference ensembles emit, and where their spectra are cut off (schematic figure).

Monochromatic continuum emission with the paper's default geometry (B ~ U[4, 6] muG,
eta ~ U[-1/2, 1/2], isotropic pitch angles, <e^{2 i phi}> = 0.4 e^{0.5 i}, Gaussian screen with
mean 8 and sd 0.5 rad m^-2).  The directional continuum kernels are A_theta F(x) for I and
-A_theta G(x) for Q in the field frame, x = nu / (nu_0 gamma^2 sin theta), nu_0 = 3 e B/(4 pi m c).
The script checks this monochromatic implementation against the study kernels (lib/basis.py)
on narrow channels.

Stokes V vanishes for the symmetric eta distribution.  Panel (c) shows |V|/I for one field
inclination, theta = 75 deg, from the power-law fit of Pandya et al. (2016, Eq. 33), which is
valid where the spectrum follows the power law (fit error up to 25%).

The results JSON records the intensity relative to the power law at fixed frequencies (read
from the saved curves), |V|/I, the screen rotation of the polarisation angle at 1 GHz
(lambda^2 times the mean rotation measure) and the screen depolarisation factor at 0.2 GHz.
The 0.2 GHz of the panel (b) annotation is placed by hand and is recorded as such. Text positions
of all labels are placed by hand. Style: lib/figure_style_paper.py.

Writes emission_windows_results.json, emission_windows_curves.npz and
figures/emission_windows.{pdf,png}.
Run: python emission_windows.py   (python emission_windows.py --replot redraws from the npz only)
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patheffects as pe  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.integrate import quad  # noqa: E402
from scipy.special import kv  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import ensembles as E  # noqa: E402
import figure_style_paper as fs  # noqa: E402
from config import PAPER  # noqa: E402
from provenance import provenance, relative, sha256, write_json  # noqa: E402

T0 = time.time()
E_ESU, M_E, C_CGS, C_SI = 4.80320471257e-10, 9.1093837015e-28, 2.99792458e10, 2.99792458e8
NU0_PER_MUG = 3 * E_ESU * 1e-6 / (4 * np.pi * M_E * C_CGS)  # Hz per muG
P_INDEX = 2.6
THETA_V = np.radians(75.0)
NU = np.geomspace(1e6, 1e15, 700)  # Hz
LNG = np.linspace(np.log(1.0), np.log(1e9), 5000)
RADIO, CMB_BAND = (0.1, 10.0), (10.0, 300.0)  # GHz
PROBE_GHZ = (0.1, 1, 10, 30, 100, 300)
ROTATION_GHZ = 1.0  # frequency of the stated screen rotation
DEPOL_NOTE_GHZ = 0.2  # frequency in the panel (b) annotation, placed by hand
FIGURE = "emission_windows"

# synchrotron functions F(x) = x int_x^inf K_5/3, G(x) = x K_2/3(x)
_XS = np.geomspace(1e-9, 40.0, 1500)
_F = np.array([x * quad(lambda u: kv(5 / 3, np.exp(u)) * np.exp(u), np.log(x), np.log(120.0), limit=400)[0]
               for x in _XS])


def F(x):
    out = np.zeros_like(x)
    mid = (x >= _XS[0]) & (x <= _XS[-1])
    out[mid] = np.exp(np.interp(np.log(x[mid]), np.log(_XS), np.log(_F)))
    lo, hi = x < _XS[0], x > _XS[-1]
    out[lo] = 2.1495 * x[lo] ** (1 / 3)
    out[hi] = np.sqrt(np.pi * x[hi] / 2) * np.exp(-np.minimum(x[hi], 700.0))
    return out


def G(x):
    return x * kv(2 / 3, np.minimum(x, 700.0))


def stokes(density, nu=NU, cfg=PAPER):
    """(I, Q + iU) per unit column, monochromatic, paper geometry and screen."""
    b, wb = cfg.b_nodes_weights
    eta, we = cfg.eta_nodes_weights
    g = np.exp(LNG)
    wg = density(g) * g  # integrand per unit ln gamma
    I = np.zeros(nu.size)
    P = np.zeros(nu.size)
    for bk, wbk in zip(b, wb):
        for ek, wek in zip(eta, we):
            s = np.sqrt(1 - ek**2)
            x = nu[:, None] / (NU0_PER_MUG * bk * s * g[None, :] ** 2)
            a = wbk * wek * bk * s
            I += a * np.trapezoid(wg * F(x), LNG, axis=1)
            P -= a * np.trapezoid(wg * G(x), LNG, axis=1)
    lam2 = (C_SI / nu) ** 2
    screen = np.exp(2j * lam2 * cfg.screen_mean - 2 * lam2**2 * cfg.screen_sd**2)
    return I, P * cfg.azimuth_factor * screen


def check_against_study_kernels() -> float:
    """Max |I_mono / I_study - 1| for E1 on 41 narrow channels over 0.1-10 GHz."""
    cfg = replace(PAPER, chan_half=0.002, n_chan_nodes=64)
    e1 = E.reference_ensembles(cfg)["E1"]
    s_study = e1.direct(cfg, n=24)
    I_mono, P_mono = stokes(lambda g: np.where((g >= cfg.gamma_lo) & (g <= cfg.gamma_hi), g**-P_INDEX, 0.0),
                            nu=cfg.centres, cfg=cfg)
    k = s_study[:, 0] / I_mono
    p_ratio = np.abs(s_study[:, 1] + 1j * s_study[:, 2]) / s_study[:, 0] / (np.abs(P_mono) / I_mono)
    return float(np.max(np.abs(k / k.mean() - 1))), float(np.max(np.abs(p_ratio - 1)))


def v_over_i(nu, b_mug=5.0, theta=THETA_V, p=P_INDEX):
    """|j_V/j_I| of an isotropic power law (Pandya et al. 2016, Eq. 33 fit)."""
    nu_c = 2.799e6 * b_mug * 1e-6
    return 171 / 250 * np.sqrt(p) / np.tan(theta) * (nu / (3 * nu_c * np.sin(theta))) ** -0.5


CURVES = HERE / "emission_windows_curves.npz"


def replot() -> None:
    """Redraw from the saved curves (python emission_windows.py --replot)."""
    d = np.load(CURVES)
    curves = {k: {q: d[f"{k}__{q}"] for q in ("I_over_PL", "Q_over_I", "U_over_I")}
              for k in ("old_range", "new_range", "aged_E4JP")}
    plot(d["ghz"], curves, d["vI"], d["valid"], float(d["gc"]))


def main() -> None:
    gc = E.AGEING.gamma_c_jp
    box = lambda lo, hi: (lambda g: np.where((g >= lo) & (g <= hi), g**-P_INDEX, 0.0))  # noqa: E731
    aged = lambda g: np.where((g >= 300) & (g < gc), g**-P_INDEX * np.clip(1 - g / gc, 0, None) ** (P_INDEX - 2), 0.0)  # noqa: E731
    I_ref, _ = stokes(box(1.0, 1e9))
    curves = {}
    for key, dens in {"old_range": box(300, 1e5), "new_range": box(300, PAPER.gamma_hi), "aged_E4JP": aged}.items():
        I, P = stokes(dens)
        curves[key] = {"I_over_PL": I / I_ref, "Q_over_I": P.real / I, "U_over_I": P.imag / I}
    vI = v_over_i(NU)
    valid = np.abs(curves["new_range"]["I_over_PL"] - 1) < 0.05
    chk_I, chk_P = check_against_study_kernels()
    ghz = NU / 1e9
    np.savez(CURVES, ghz=ghz, vI=vI, valid=valid, gc=gc,
             **{f"{k}__{q}": v for k, c in curves.items() for q, v in c.items()})
    saved = np.load(CURVES)
    probe = {f"{f:g}": {k: float(np.interp(np.log(f), np.log(saved["ghz"]), saved[f"{k}__I_over_PL"]))
                        for k in curves} for f in PROBE_GHZ}
    probe_direct = {f"{f:g}": {k: float(np.interp(np.log(f), np.log(ghz), c["I_over_PL"])) for k, c in curves.items()}
                    for f in PROBE_GHZ}
    probe_diff = max(abs(probe[f][k] - probe_direct[f][k]) for f in probe for k in probe[f])
    plot(ghz, curves, vI, valid, gc)
    write_json(HERE / "emission_windows_results.json", {
        "description": __doc__, "I_over_PL_at_GHz": probe,
        "V_over_I_theta75_at_GHz": {f"{f:g}": float(v_over_i(f * 1e9)) for f in (0.1, 1, 10, 100, 300)},
        "screen": screen_values(ghz, curves["new_range"]),
        "checks": {"monochromatic_vs_study_kernel_E1_narrow_channels": {"I_shape_max_rel": chk_I,
                                                                         "P_over_I_max_rel": chk_P},
                   "I_over_PL_from_npz_minus_computed_max_abs": probe_diff},
        "curves_npz": relative(CURVES), "curves_npz_sha256": sha256(CURVES),
        "figures": {relative(p): sha256(p) for p in (HERE / "figures" / f"{FIGURE}.{e}" for e in ("pdf", "png"))},
        "provenance": provenance(__file__, T0, inputs=(HERE / "lib" / "figure_style_paper.py",))})
    print("check vs study kernels:", chk_I, chk_P)
    print(probe)


def screen_values(ghz, c) -> dict:
    """Screen rotation at ROTATION_GHZ and depolarisation at the annotated DEPOL_NOTE_GHZ."""
    lam2_rot = (C_SI / (ROTATION_GHZ * 1e9)) ** 2
    lam2_dep = (C_SI / (DEPOL_NOTE_GHZ * 1e9)) ** 2
    p_over_i = np.hypot(c["Q_over_I"], c["U_over_I"])
    return {"rotation_deg_at_GHz": {f"{ROTATION_GHZ:g}": float(np.degrees(lam2_rot * PAPER.screen_mean))},
            "depolarisation_factor_exp(-2 lambda^4 sd^2)_at_GHz": {
                f"{DEPOL_NOTE_GHZ:g}": float(np.exp(-2 * lam2_dep**2 * PAPER.screen_sd**2))},
            "abs_P_over_I_at_GHz": {f"{f:g}": float(np.interp(np.log(f), np.log(ghz), p_over_i))
                                    for f in (DEPOL_NOTE_GHZ, ROTATION_GHZ, 10.0)},
            "depolarisation_annotation_GHz": DEPOL_NOTE_GHZ,
            "depolarisation_annotation_note": "frequency placed by hand in the panel (b) text, not computed"}


WINDOW_STYLE = [(RADIO, "radio band\n0.1–10 GHz", "#EFEFEF"),
                (CMB_BAND, "CMB band\n10–300 GHz", "#DCDCDC")]
NOTE = "#6E6E6E"  # commentary text and leaders: darker than fs.GREY so it survives projection


def _windows(ax, label: bool) -> None:
    for (lo, hi), name, shade in WINDOW_STYLE:
        ax.axvspan(lo, hi, color=shade, lw=0, zorder=0)
        if label:
            ax.text(np.sqrt(lo * hi), 0.985, name, transform=ax.get_xaxis_transform(),
                    ha="center", va="top", fontsize=7.5, linespacing=1.05)
    ax.axvline(CMB_BAND[0], color="white", lw=0.8, zorder=0.5)


def _leader(ax, text, xy, xytext, **kw) -> None:
    ax.annotate(text, xy=xy, xytext=xytext, fontsize=8, color=NOTE, linespacing=1.1,
                arrowprops=dict(arrowstyle="-", color=NOTE, lw=0.6, shrinkA=1, shrinkB=1), **kw)


def _leader_from_line(ax, text, i, xy, side="right") -> None:
    """Straight leader from the left or right end of line i of a multi-line text to xy (data)."""
    fig = ax.figure
    r = fig.canvas.get_renderer()
    lines = text.get_text().split("\n")
    full = text.get_window_extent(r)
    probe = ax.text(0, 0, lines[i], fontsize=text.get_fontsize(), transform=None)
    w = probe.get_window_extent(r).width
    probe.remove()
    x0 = full.x0 if side == "left" else full.x0 + w
    y0 = full.y1 - (i + 0.5) * full.height / len(lines)
    pad = 2.0 if side == "right" else -2.0
    start = ax.transData.inverted().transform((x0 + pad, y0))
    ax.plot([start[0], xy[0]], [start[1], xy[1]], color=NOTE, lw=0.6, zorder=1.5)


def _panel_intensity(ax, ghz, c) -> None:
    aged = "#A34D84"  # E4-JP hue darkened so the line stays a data curve on the CMB band and in greyscale
    aged_text = "#6E2F57"  # same hue, dark like the other curve labels (commentary stays mid-grey)
    halo = [pe.withStroke(linewidth=1.5, foreground=WINDOW_STYLE[0][2])]
    ax.axhline(1.0, color=fs.GREY, lw=fs.LW_REF, ls=":")
    ax.plot(ghz, c["old_range"]["I_over_PL"], color=fs.INK, lw=fs.LW, ls="--", zorder=2)
    ax.plot(ghz, c["aged_E4JP"]["I_over_PL"], color=aged, lw=1.6, ls=(0, (6, 1.5, 1.2, 1.5)), zorder=2)
    ax.plot(ghz, c["new_range"]["I_over_PL"], color=fs.INK, lw=1.6, zorder=3)
    ax.set_yscale("log")
    ax.set_ylim(1e-3, 6)
    ax.set_yticks([1e-3, 1e-2, 1e-1, 1.0])
    ax.set_ylabel(r"Intensity / power law")
    ax.text(8e5, 1.35, r"$300\le\gamma\le2.96\times10^6$ (wide energy range)", fontsize=8,
            ha="right", va="bottom")
    ax.text(8e2, 0.2, r"$300\le\gamma\le10^5$" + "\n(narrow energy range)", fontsize=8,
            ha="left", va="center", linespacing=1.1)
    ax.text(0.12, 0.012, "aged electrons (E4-JP)\n" + r"only up to $\gamma_{\rm c}\approx2\times10^4$",
            fontsize=8, color=aged_text, ha="left", va="center", linespacing=1.1, path_effects=halo)
    _leader(ax, "turnover at the critical\nfrequency of the least\nenergetic electrons\n"
            r"($\gamma=300$, $\approx2$ MHz)",
            xy=(2.6e-3, 0.6), xytext=(1.3e-3, 0.0065), va="center")
    cut = ("cut-off near the critical\nfrequency of the most\nenergetic electrons\n"
           r"($\gamma=10^5$ dashed, $2.96\times10^6$ solid)")
    note = ax.text(2.0e3, 0.0045, cut, fontsize=8, color=NOTE, va="center", ha="left", linespacing=1.1)
    _leader_from_line(ax, note, 0, (7e2, 0.045), side="left")
    _leader_from_line(ax, note, 0, (6.0e5, 0.045))
    fs.panel_title(ax, r"(a) Intensity relative to a $\gamma^{-2.6}$ power law")


def _panel_linear(axp, ghz, c) -> None:
    q, u = c["new_range"]["Q_over_I"], c["new_range"]["U_over_I"]
    u_ink = "#5A5A5A"
    axp.axhline(0.0, color=fs.GREY, lw=fs.LW_REF, ls=":")
    axp.plot(ghz, u, color=u_ink, lw=fs.LW, ls="--")
    axp.plot(ghz, q, color=fs.INK, lw=fs.LW)
    axp.set_ylim(-0.37, 0.33)
    axp.set_yticks([-0.2, 0.0, 0.2])
    axp.set_ylabel("Polarisation\nfraction")
    i3 = int(np.argmin(np.abs(ghz - 3e3)))
    axp.text(3e3, q[i3] + 0.02, r"$Q/I$", fontsize=8, ha="center", va="bottom")
    axp.text(3e3, u[i3] + 0.02, r"$U/I$", fontsize=8, ha="center", va="bottom", color=u_ink)
    far = axp.text(1.5e-3, 0.315, "Faraday rotation\n(oscillations) and\ndepolarisation\n"
                   "(below about 0.2 GHz)", fontsize=8, color=NOTE, ha="left", va="top", linespacing=1.1)
    _leader_from_line(axp, far, 1, (0.33, 0.2))
    axp.text(1.5e-3, -0.33, "same electrons as the\nsolid curve in (a)", fontsize=8,
             color=NOTE, ha="left", va="bottom", linespacing=1.1)
    fs.panel_title(axp, r"(b) Linear polarisation, single frequencies")


def _panel_circular(axv, ghz, vI, valid) -> None:
    axv.plot(ghz[valid], vI[valid], color=fs.INK, lw=fs.LW)
    axv.set_yscale("log")
    axv.set_ylim(3e-7, 3e-3)
    axv.set_yticks([1e-6, 1e-5, 1e-4, 1e-3])
    axv.set_ylabel("Circular\nfraction ($|V|/I$)")
    axv.text(4e-2, 5.5e-4, r"$\propto\nu^{-1/2}$", fontsize=8, ha="left", va="bottom")
    fs.panel_title(axv, r"(c) Circular polarisation, one field direction ($\theta=75^\circ$, $B=5\,\mu$G)")


def plot(ghz, c, vI, valid, gc) -> None:
    fs.configure()
    fig, axes = plt.subplots(3, 1, figsize=(fs.WIDTH, 5.0), sharex=True,
                             gridspec_kw=dict(height_ratios=[1.35, 0.85, 0.58]))
    fig.subplots_adjust(top=0.955, bottom=0.075, left=0.105, right=0.985, hspace=0.32)
    ax, axp, axv = axes
    for a in axes:
        _windows(a, a is ax)
        a.set_xscale("log")
    axv.set_xlim(1e-3, 1e6)  # set before the panels: leaders are placed from rendered text extents
    _panel_intensity(ax, ghz, c)
    _panel_linear(axp, ghz, c)
    _panel_circular(axv, ghz, vI, valid)
    axv.set_xlabel(r"Frequency [GHz]")
    fig.align_ylabels(axes)
    (HERE / "figures").mkdir(exist_ok=True)
    fs.save(fig, FIGURE)


if __name__ == "__main__":
    replot() if "--replot" in sys.argv else main()
