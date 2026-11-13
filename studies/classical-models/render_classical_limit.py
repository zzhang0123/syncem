"""Draw Fig. 3 (figures/classical_limit.{pdf,png}) in the notation of Sec. 3.4.

No physics is recomputed. Inputs:
  * recover_classical_limit_a_results.json   panel (a), from recover_classical_limit_a.py. The
    script SHA-256 recorded in its provenance block must match recover_classical_limit_a.py.
  * inputs/validation/harmonic_continuum_results.json   panels (b, c), theta = pi/3, SHA-256 pinned.
Style: lib/figure_style_paper.py, a byte copy of the manuscript's validation/figure_style.py.
Writes figures/classical_limit.{pdf,png} and render_classical_limit_results.json, with the
SHA-256 of both inputs, of the style module and of the figures and a provenance block.

Panel (b) shows |lhs/rhs - 1| for the two sides of main.tex Eq. (3.10), the
isotropic pitch average of the harmonic kernel against the continuum kernel K^iso_X,
and panel (c) V/I of the pitch-averaged harmonic kernels. x = nu / (a_B gamma^2). Panel (a) gives channel intensities I_j^obs against
nu_j / nu_0, nu_0 = e B_0 / (2 pi m_e c) with B_0 = 5 muG, so that the plot does not
suggest Hz frequencies that lie below the plasma frequency.

Run from this directory:  python render_classical_limit.py
"""

from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

sys.dont_write_bytecode = True

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, NullFormatter  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
from figure_style_paper import (  # noqa: E402
    GREY, INK, LW, SERIES, WIDTH, configure, pack_row, panel_title, save,
    tighten_y_labels)
from provenance import provenance, write_json  # noqa: E402

T0 = time.time()
PANEL_A = HERE / "recover_classical_limit_a_results.json"
PANEL_A_PRODUCER = HERE / "recover_classical_limit_a.py"
INPUTS = {
    "continuum": (HERE / "inputs/validation/harmonic_continuum_results.json",
                  "aad5d7140bfa7e66818a0659c3982b6496fb18e87e439e7375e2feceba55cd12"),
}
STYLE = HERE / "lib" / "figure_style_paper.py"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()
E_ESU, M_E_G, C_CM_S = 4.80320471e-10, 9.1093837015e-28, 2.99792458e10
B0_GAUSS = 5e-6
NU0_HZ = E_ESU * B0_GAUSS / (2 * np.pi * M_E_G * C_CM_S)  # 13.996 Hz
GAMMAS = (10.0, 30.0, 100.0)
LW_Q = 0.8
# (key, legend label, colour, linestyle): ink for the two gamma = 1.1-1.2 ensembles,
# grey for gamma = 2-4. The blue/vermilion/green series mark gamma in panels (b, c).
LOW = (("narrowB_1.1-1.2", r"$1.1$--$1.2$, \ $4.5$--$5.5$", INK, "--"),
       ("broadB_1.1-1.2", r"$1.1$--$1.2$, \ $2$--$8$", INK, "-"),
       ("broadB_2-4", r"$2$--$4$, \ $2$--$8$", GREY, "-"))


def load() -> dict:
    data = {}
    for key, (path, digest) in INPUTS.items():
        found = sha256(path)
        if found != digest:
            raise ValueError(f"{key}: SHA-256 {found} != pinned {digest} ({path})")
        data[key] = json.loads(path.read_text())
    panel_a = json.loads(PANEL_A.read_text())
    recorded, found = panel_a["provenance"]["script_sha256"], sha256(PANEL_A_PRODUCER)
    if recorded != found:
        raise ValueError(f"{PANEL_A.name} was written by a different {PANEL_A_PRODUCER.name} "
                         f"(recorded SHA-256 {recorded}, file {found}); rerun it")
    data["panel_a"] = panel_a
    return data


def continuum_curves(cont: dict) -> dict:
    """Same selection and merge as validation/spectral_figures.harmonic_continuum."""
    out = {}
    for g in GAMMAS:
        rows = [r for r in cont["rows"] if r["gamma"] == g and r["harmonic_I"] > 0]
        detail = [r for r in cont["low_harmonic_detail"] if r["gamma"] == g]
        if detail:
            lo, hi = detail[0]["x"], detail[-1]["x"]
            rows = sorted(detail + [r for r in rows if r["x"] < lo or r["x"] > hi],
                          key=lambda r: r["x"])
        out[g] = dict(
            x=np.array([r["x"] for r in rows]),
            err_i=np.abs([r["harmonic_I"] / r["continuum_I"] - 1 for r in rows]),
            err_q=np.abs([r["harmonic_Q"] / r["continuum_Q"] - 1 for r in rows]),
            v_over_i=np.array([r["harmonic_V"] / r["harmonic_I"] for r in rows]),
        )
    return out


def panel_low_energy(ax, panel: dict) -> None:
    for key, label, colour, ls in LOW:
        c = panel["curves"][key]
        ax.plot(np.asarray(c["nu_hz"]) / NU0_HZ, c["I_rel"], label=label, color=colour, ls=ls, lw=LW)
    ax.set(xscale="log", yscale="log", xlabel=r"$\nu_j/\nu_0$",
           ylabel=r"$I_j^{\rm obs}$ [arb.\ units]",
           xlim=(3 / NU0_HZ, 1000 / NU0_HZ), ylim=(1e-4, 1e3))
    ax.xaxis.set_major_locator(FixedLocator([1, 10]))
    ax.set_xticklabels(["1", "10"])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.yaxis.set_major_locator(FixedLocator([1e-4, 1e-2, 1, 1e2]))
    panel_title(ax, "(a) Low-energy harmonics")
    legend = ax.legend(loc="upper right", title=r"$\gamma$, \ $B$ [$\mu$G]", handlelength=1.6,
                       labelspacing=0.15, borderaxespad=0.1, title_fontsize=8)
    legend._legend_box.align = "left"


def panel_continuum(ax_err, ax_v, curves: dict) -> None:
    for colour, g in zip(SERIES, GAMMAS):
        c = curves[g]
        ax_err.plot(c["x"], c["err_i"], color=colour, lw=LW)
        ax_err.plot(c["x"], c["err_q"], color=colour, lw=LW_Q, ls="--")
        ax_v.plot(c["x"], c["v_over_i"], color=colour, lw=LW)
    for ax in (ax_err, ax_v):
        ax.set(xscale="log", xlabel=r"$x=\nu/(a_B\gamma^2)$", xlim=(1e-3, 10))
    ax_err.set(yscale="log", ylim=(1e-5, 1), ylabel="Relative difference")
    ax_err.yaxis.set_major_locator(FixedLocator([1e-5, 1e-3, 1e-1, 1]))
    panel_title(ax_err, "(b) Continuum-limit error")
    style = [Line2D([], [], color=INK, lw=LW), Line2D([], [], color=INK, lw=LW_Q, ls="--")]
    legend = ax_err.legend(style, ["$X=I$", "$X=Q$"], loc="lower left", ncol=2, columnspacing=1.0,
                           title=r"$\theta=\pi/3$", title_fontsize=8, borderaxespad=0.1)
    legend._legend_box.align = "left"
    # gamma = 10 changes sign near x = 1e-3, where its curve leaves the log axis downwards.
    ax_v.set(yscale="log", ylim=(3e-3, 2), ylabel=r"$V/I$")
    ax_v.yaxis.set_major_locator(FixedLocator([1e-2, 1e-1, 1]))
    panel_title(ax_v, "(c) Circular polarisation")


def label_gammas(ax, curves: dict, key: str, x_lab: float, below: float) -> None:
    """Labels set just below each gamma family's curve, along its local slope.

    Same placement as render_old_panels.label_gammas. Call after the layout is final.
    """
    for g in GAMMAS:
        c = curves[g]
        keep = c[key] > 0  # gamma = 10 V/I changes sign near x = 1e-3
        xs, ys = np.log(c["x"][keep]), np.log(c[key][keep])
        f = lambda x: float(np.exp(np.interp(np.log(x), xs, ys)))  # noqa: E731
        p0, p1 = ax.transData.transform([(x_lab / 1.4, f(x_lab / 1.4)), (x_lab * 1.4, f(x_lab * 1.4))])
        angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
        ax.text(x_lab, f(x_lab) / below, fr"$\gamma={g:g}$", rotation=angle, rotation_mode="anchor",
                ha="center", va="center")


def summary(curves: dict) -> dict:
    out = {}
    for g in GAMMAS:
        c = curves[g]
        mid = (c["x"] >= 0.1) & (c["x"] <= 10)
        out[f"gamma={g:g}"] = {
            "gamma2_err_I_range_x0.1-10": [float(g**2 * c["err_i"][mid].min()), float(g**2 * c["err_i"][mid].max())],
            "gamma2_err_Q_range_x0.1-10": [float(g**2 * c["err_q"][mid].min()), float(g**2 * c["err_q"][mid].max())],
            "gamma2_err_IQ_max_at_x1e-3": float(g**2 * max(c["err_i"][0], c["err_q"][0])),
        }
    return out


def main() -> None:
    configure()
    data = load()
    curves = continuum_curves(data["continuum"])
    height = 2.4
    fig, (ax_a, ax_b, ax_c) = plt.subplots(1, 3, figsize=(WIDTH, height),
                                           gridspec_kw=dict(bottom=0.44 / height, top=0.86))
    panel_low_energy(ax_a, data["panel_a"])
    panel_continuum(ax_b, ax_c, curves)
    tighten_y_labels((ax_a, ax_b, ax_c))
    layout = pack_row(fig, (ax_a, ax_b, ax_c))
    label_gammas(ax_b, curves, "err_i", 2e-2, 1.85)  # slope-following labels need the final transforms
    label_gammas(ax_c, curves, "v_over_i", 0.3, 1.55)
    save(fig, "classical_limit", WIDTH)
    record = dict(nu0_hz=NU0_HZ, layout=layout, summary=summary(curves),
                  outputs={n: sha256(HERE / "figures" / n)
                           for n in ("classical_limit.pdf", "classical_limit.png")})
    record["provenance"] = provenance(__file__, T0, inputs=[PANEL_A, *(p for p, _ in INPUTS.values()), STYLE])
    write_json(HERE / "render_classical_limit_results.json", record)
    print(json.dumps({k: v for k, v in record.items() if k != "provenance"}, indent=1))


if __name__ == "__main__":
    main()
