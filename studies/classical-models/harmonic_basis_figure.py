"""Basis spectra at gamma0 = 100 without the continuum approximation (figures/harmonic_basis_spectra).

Finite-gamma harmonic kernels from syncem.model.build_high_order_basis (the exploratory high-order
route: its numerical error is reported unbounded), at the reference (gamma0, B0) = (100, 5 muG), with
Truncation(L_mu = 1, L_eta = 1, N = 1, depth_degree = 0): Legendre pairs l, k <= 1, order 0 and the
first derivatives in gamma and B, scaled by s_gamma = 0.2 gamma0 and s_B = 1 muG, and no Faraday depth.
Channels: unit-area bump responses (normalisation "unit_integral", the response family of the
examples) with half-width 0.2 nu_j at centres log-spaced in y = nu/nu_0, nu_0 = e B0/(2 pi m_e c).
Channels are built in chunks under the 65536-node angular guard of the route. Each channel's columns
are independent of the others.

The figure has columns I, Q' (the polarisation in the frame of the projected field, U' = 0 without
Faraday rotation) and V, rows (l, k) = (0,0), (1,1) for I and Q' and (1,0), (0,1) for V, and in each
panel order 0, d_gamma and d_B, each divided by its largest absolute value.

The results JSON records: the change of every column between CHECK_NODES and NODES angular nodes,
the agreement of the I and Q' columns with the continuum (l, k) projections of the study kernels
(one common constant), the departure of the field derivatives from the continuum identity
s_B d_B = (s_B/B0)(K + 0.5 dK/dln gamma), the peaks of V relative to the order-0 intensity, and U'.
Style: lib/figure_style_paper.py.

Writes figures/harmonic_basis_spectra.{pdf,png}, harmonic_basis_figure.npz and
harmonic_basis_figure_results.json.  Run: python harmonic_basis_figure.py  (about 30 minutes)
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FuncFormatter, LogLocator, NullFormatter  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402
from scipy.special import eval_legendre  # noqa: E402

HERE = Path(__file__).resolve().parent
PACKAGE_ROOT = HERE.parents[1]
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(PACKAGE_ROOT))
import basis as B  # noqa: E402
import cost  # noqa: E402
import figure_style_paper as fs  # noqa: E402
import syncem  # noqa: E402
from config import PAPER  # noqa: E402
from provenance import provenance, relative, sha256, write_json  # noqa: E402
from syncem.constants import C_CGS, E_ESU, M_E  # noqa: E402
from syncem.model import Channels, Reference, Support, Truncation, build_high_order_basis  # noqa: E402

if Path(syncem.__file__).resolve().parents[1] != PACKAGE_ROOT:
    raise RuntimeError("syncem did not resolve to the package root of this repository")

T0 = time.time()
NAME = "harmonic_basis_spectra"
NPZ = HERE / "harmonic_basis_figure.npz"
OUT_JSON = HERE / "harmonic_basis_figure_results.json"
GAMMA0, B0 = 100.0, 5e-6
NU0 = E_ESU * B0 / (2 * np.pi * M_E * C_CGS)  # Hz
Y = np.geomspace(300.0, 1e5, 121)  # nu/nu_0
NODES, CHECK_NODES = 96, 48
GUARD = 65536  # angular nodes per build_high_order_basis call (route resource guard)
ROUTE = dict(rtol=1e-6, atol=1e-35, atol_gamma=1e-35, atol_B=1e-30)
SCALES = (0.2 * GAMMA0, 1e-6, 1.0)
CURVES = ((0, 0), (1, 0), (0, 1))  # (r, s): order 0, d_gamma, d_B
STYLE = {(0, 0): dict(color="#104281", lw=1.6, ls="-"),
         (1, 0): dict(color="#2a78d6", lw=1.3, ls=(0, (5, 1.6))),
         (0, 1): dict(color=fs.VERMILION, lw=1.2, ls=(0, (5, 1.4, 1.2, 1.4)))}
NAME_CURVE = {(0, 0): "order 0", (1, 0): r"$\partial_{\gamma}$", (0, 1): r"$\partial_B$"}
SYMBOL = {"I": r"\widetilde I", "Q'": r"\widetilde Q'", "V": r"\widetilde V"}


def build(nodes: int) -> dict:
    """Columns {(component, l, k, r, s, b): (n_ch,)} of the route at nodes x nodes angular nodes."""
    ref = Reference(GAMMA0, B0, depth_ref=0.0, scales=SCALES)
    sup = Support((0.8 * GAMMA0, 1.2 * GAMMA0), (4e-6, 6e-6), (-1.0, 1.0), truncated=False)
    chunk = max(1, GUARD // (3 * nodes * nodes))  # the ridge rule uses up to three panels per mu node
    parts = []
    for s in range(0, Y.size, chunk):
        yy = Y[s:s + chunk]
        b = build_high_order_basis(Channels.bump(yy * NU0, 0.2 * yy * NU0, normalisation="unit_integral"),
                                   Truncation(1, 1, 1, depth_degree=0), ref, support=sup, n_mu=nodes,
                                   n_eta=nodes, **ROUTE)
        parts.append((np.asarray(b.I_basis), np.asarray(b.V_basis), np.asarray(b.P_basis)))
        h0, h2 = b.index.h0, b.index.h2
    I, V, P = (np.concatenate([p[i] for p in parts]) for i in range(3))
    col = {}
    for comp, arr, rows in (("I", I, h0), ("V", V, h0), ("Q'", P.real, h2), ("U'", P.imag, h2)):
        for i, row in enumerate(rows):
            col[(comp,) + tuple(int(v) for v in row)] = arr[:, i]
    return col


def continuum() -> dict:
    """Continuum (l, k) projections of the study kernels at (gamma0, B0) on the same channels, z-scaled."""
    cfg = replace(PAPER, nu_lo=Y[0] * NU0, nu_hi=Y[-1] * NU0, n_channels=Y.size, screen_mean=0.0, screen_sd=0.0)
    t, w = leggauss(256)
    e, w = 0.5 * (t + 1.0), 0.5 * w
    KI, KQ = B.channel_kernels(cfg, np.array([GAMMA0]), eta_s=np.sqrt(1.0 - e**2), order=1,
                               table=cost.fixed_b_table(B0 / 1e-6))
    s_g, s_B = 0.2, SCALES[1] / B0  # s_gamma d/dgamma = 0.2 d/dln(gamma) at gamma0; s_B d/dB = (s_B/B0) B d/dB
    out = {}
    for l, k in ((0, 0), (1, 1)):
        wt = (2 * l + 1) * (2 * k + 1) / 2 * 2 * w * eval_legendre(l, e) * eval_legendre(k, e)
        for comp, K in (("I", KI), ("Q'", KQ.real)):
            G0, G1 = np.einsum("rgej,e->rgj", K, wt)[:, 0]
            out[(comp, l, k, 0, 0, 0)] = G0
            out[(comp, l, k, 1, 0, 0)] = s_g * G1
            out[(comp, l, k, 0, 1, 0)] = s_B * (G0 + 0.5 * G1)  # field-scaling identity, exact in the continuum
    return out


def checks(col: dict, coarse: dict) -> dict:
    nonzero = [key for key, v in col.items() if np.abs(v).max() > 0]
    conv = {}
    for comp in ("I", "Q'", "V"):
        conv[comp] = float(max(np.abs(col[k] - coarse[k]).max() / np.abs(col[k]).max() for k in nonzero
                               if k[0] == comp))
    cont = continuum()
    k0 = np.abs(col[("I", 0, 0, 0, 0, 0)]).max() / np.abs(cont[("I", 0, 0, 0, 0, 0)]).max()
    vs_cont = float(max(np.abs(col[k] - k0 * c).max() / np.abs(k0 * c).max() for k, c in cont.items()))
    ident = {}
    for comp in ("I", "Q'"):
        for l, k in ((0, 0), (1, 1)):
            G0, Gg = col[(comp, l, k, 0, 0, 0)], col[(comp, l, k, 1, 0, 0)]
            target = SCALES[1] / B0 * (G0 + 0.5 * Gg / 0.2)  # (s_B/B0)(K + 0.5 dK/dln gamma)
            dB = col[(comp, l, k, 0, 1, 0)]
            ident[f"{comp} ({l},{k})"] = float(np.abs(dB - target).max() / np.abs(dB).max())
    I00 = np.abs(col[("I", 0, 0, 0, 0, 0)]).max()
    v01, v10 = col[("V", 0, 1, 0, 0, 0)], col[("V", 1, 0, 0, 0, 0)]
    return {"nodes": NODES, "check_nodes": CHECK_NODES,
            "max_change_check_to_nodes_by_component": conv,
            "I_Q_vs_continuum_common_constant_max_rel": vs_cont,
            "dB_minus_continuum_identity_max_rel": ident,
            "V_order0_peak_over_I00_peak": {"(0,1)": float(np.abs(v01).max() / I00),
                                            "(1,0)": float(np.abs(v10).max() / I00)},
            "V_10_vs_01_shape_max_diff": float(np.abs(v10 / np.abs(v10).max() - v01 / np.abs(v01).max()).max()),
            "U_over_Q_max": float(max(np.abs(col[k]).max() for k in col if k[0] == "U'")
                                  / max(np.abs(col[k]).max() for k in col if k[0] == "Q'")),
            "odd_I_even_V_max": float(max(np.abs(col[k]).max() for k in col
                                          if (k[0] == "I" and (k[1] + k[2]) % 2) or (k[0] == "V" and (k[1] + k[2]) % 2 == 0)))}


def draw(col: dict):
    fig = plt.figure(figsize=(fs.WIDTH, 2.85))
    gs = fig.add_gridspec(2, 3, left=0.075, right=0.99, top=0.80, bottom=0.15, hspace=0.55, wspace=0.10)
    letters = iter("abcdef")
    for ri, (even, odd) in enumerate((((0, 0), (1, 0)), ((1, 1), (0, 1)))):
        for j, (comp, pair) in enumerate((("I", even), ("Q'", even), ("V", odd))):
            ax = fig.add_subplot(gs[ri, j])
            ax.set_xscale("log")
            ax.set_xlim(Y[0], Y[-1])
            ax.xaxis.set_major_locator(LogLocator(base=10))
            ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: rf"$10^{{{int(round(np.log10(v)))}}}$"))
            ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5)))
            ax.xaxis.set_minor_formatter(NullFormatter())
            if ri == 1:
                ax.set_xlabel(r"$\nu/\nu_0$")
            else:
                ax.tick_params(labelbottom=False)
            ax.axhline(0, color=fs.GREY, lw=0.4, zorder=0)
            for rs in CURVES:
                v = col[(comp, pair[0], pair[1], rs[0], rs[1], 0)]
                ax.plot(Y, v / np.abs(v).max(), solid_capstyle="round", zorder=3, **STYLE[rs])
            ax.set_ylim(-1.15, 1.15)
            ax.set_yticks([-1, 0, 1])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}$"))
            ax.tick_params(labelleft=(j == 0))
            fs.panel_title(ax, rf"({next(letters)}) ${SYMBOL[comp]}$, $(\ell,k)=({pair[0]},{pair[1]})$")
    handles = [Line2D([], [], **STYLE[rs], label=NAME_CURVE[rs]) for rs in CURVES]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 0.995), handlelength=2.8,
               columnspacing=2.0, frameon=False)
    fig.text(0.006, 0.47, r"Basis spectrum / peak $|{\rm value}|$", rotation=90, va="center")
    return fig


def main() -> None:
    fs.configure()
    col = build(NODES)
    coarse = build(CHECK_NODES)
    result = checks(col, coarse)
    keys = sorted(col)
    np.savez(NPZ, y=Y, nu0=NU0, gamma0=GAMMA0, B0=B0, keys=np.array([str(k) for k in keys]),
             columns=np.array([col[k] for k in keys]))
    (HERE / "figures").mkdir(exist_ok=True)
    fs.save(draw(col), NAME)
    figures = {relative(HERE / "figures" / f"{NAME}.{e}"): sha256(HERE / "figures" / f"{NAME}.{e}")
               for e in ("pdf", "png")}
    write_json(OUT_JSON, {"description": __doc__, "gamma0": GAMMA0, "B0_G": B0, "nu0_Hz": NU0,
                          "y_range": [float(Y[0]), float(Y[-1]), int(Y.size)], "scales": list(SCALES),
                          "route_settings": ROUTE, "syncem_version": syncem.__version__, "checks": result,
                          "npz": relative(NPZ), "npz_sha256": sha256(NPZ), "figures": figures,
                          "provenance": provenance(__file__, T0, inputs=(HERE / "lib" / "figure_style_paper.py",
                                                                         HERE / "cost.py"))})
    print(json.dumps(result, indent=1))


if __name__ == "__main__":
    main()
