"""Basis spectra of two energy cells by Legendre pair, in the continuum limit (figures/basis_spectra).

The continuum kernels emit only at mu = eta, so K^nat(mu, eta) = 2 K^iso(eta) delta(mu - eta) and the
Legendre projection of Eq. legendre spectral response is
    K_{lk} = (2l+1)(2k+1)/2 int_{-1}^{1} P_l(eta) P_k(eta) K^iso(eta) d eta.
K^iso is even in eta, so the projection uses Gauss-Legendre nodes on [0, 1] doubled (N_ETA nodes).
The kernels are those of the paper configuration at fixed B0 = 5 muG (cost.fixed_b_table), with
the channel response (k = 1, h_j = 0.2 nu_j) and the independent Gaussian screen. The figure shows
cells 7 and 10 (gamma_b = 11322 and 48377), pairs (l, k) = (0, 0) and (2, 0), and for each the
order-0, first and second derivatives in u_b, (h_b^r/r!) d^r/d(ln gamma)^r. Columns: I, and
Q' + i U' = <e^{2i phi}> K_Q exp(-2i <Psi> lambda_j^2) exp(-i arg <e^{2i phi}>), Stokes Q and U in the
frame of the mean projected field after removal of the mean screen rotation at each channel centre.
Each curve is divided by its largest absolute value. Curves evaluate log-spaced centres over
0.1-300 GHz (the main band and the extrapolation band of Sec. 5.3) with the spacing of DENSE centres
over 0.1-10 GHz, every tenth of which is one of the 41 paper channels.

The results JSON records the checks behind the caption and the citing text: eta quadrature
convergence, the (0,0) projection against basis.kbar over all field directions, l <-> k symmetry
and the zero of (1,0) on a full [-1, 1] rule, the field derivatives against a finite-difference
stencil in B (they equal combinations of the order-0 and energy spectra), the depolarisation below
DEPOL_NU_GHZ on centres that end there, U' without a mean Faraday depth, and the order-0 peak
frequencies, and for panels (i, l) the low-frequency-tail relation of the u_b spectra to order 0.
Style: lib/figure_style_paper.py.

Writes figures/basis_spectra.{pdf,png}, basis_spectra_figure.npz and basis_spectra_figure_results.json.
Run: python basis_spectra_figure.py
"""
from __future__ import annotations

import json
import sys
import time
from dataclasses import replace
from math import factorial
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, LogLocator, NullFormatter  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402
from scipy.special import eval_legendre  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import basis as B  # noqa: E402
import cost  # noqa: E402
import figure_style_paper as fs  # noqa: E402
from config import PAPER  # noqa: E402
from kernels import C_SI  # noqa: E402
from provenance import provenance, relative, sha256, write_json  # noqa: E402

T0 = time.time()
NAME = "basis_spectra"
NPZ = HERE / "basis_spectra_figure.npz"
OUT_JSON = HERE / "basis_spectra_figure_results.json"
CELLS = (7, 10)  # indices into PAPER.gamma_b; order-0 intensity peaks: checks.order0_intensity_peak_GHz
PAIRS = ((0, 0), (2, 0))
COMPONENTS = ("I", "Q'", "U'")
ORDERS = (0, 1, 2)
N_ETA, N_ETA_CHECK = 256, 128  # Gauss-Legendre nodes on [0, 1]
DENSE = 10 * (PAPER.n_channels - 1) + 1  # over 0.1-10 GHz every tenth dense centre is a paper channel
NU_TOP_GHZ = 300.0  # upper end of the figure
DEPOL_NU_GHZ, DEPOL_FRACTION = 0.3, 0.1  # caption
TAIL_NU_GHZ = 2.0  # caption (i, l): the U' panels of the higher-energy cell lie in its low-frequency tail
STENCIL_STEP = 0.04  # in x_B = (B - B0)/(1 muG), as cost.joint_columns_stencil_check
GHZ = 1e9

# Taylor order is ordered: one hue stepped in lightness (validated ordinal ramp, dataviz), dark = order 0,
# with a distinct dash per order so the curves stay distinct in greyscale.
STYLE = {0: dict(color="#104281", lw=1.6, ls="-"),
         1: dict(color="#2a78d6", lw=1.3, ls=(0, (5, 1.6))),
         2: dict(color="#6da7ec", lw=1.6, ls=(0, (1.3, 1.3)))}
NAME_ORDER = {0: "order 0", 1: r"$\partial_{u_b}$", 2: r"$\tfrac12\partial_{u_b}^2$"}
SYMBOL = {"I": r"\widetilde I", "Q'": r"\widetilde Q'", "U'": r"\widetilde U'"}


# ----------------------------------------------------------------------------- projections
def eta_rule(n: int) -> tuple[np.ndarray, np.ndarray]:
    """Gauss-Legendre nodes and weights on [0, 1]."""
    t, w = leggauss(n)
    return 0.5 * (t + 1.0), 0.5 * w


def projections(cfg, pairs, table, n_eta=N_ETA, order=2, gammas=None) -> tuple[dict, dict]:
    """K_{lk} for even l + k: {pair: (order+1, cells, J)} for I and complex Q (screen, no azimuth factor)."""
    e, w = eta_rule(n_eta)
    gammas = cfg.gamma_b[list(CELLS)] if gammas is None else gammas
    KI, KQ = B.channel_kernels(cfg, gammas, eta_s=np.sqrt(1.0 - e**2), order=order, table=table)
    I, Q = {}, {}
    for l, k in pairs:
        if (l + k) % 2:
            raise ValueError("odd pairs vanish for an even kernel; use full_range_projection")
        wt = (2 * l + 1) * (2 * k + 1) / 2 * 2 * w * eval_legendre(l, e) * eval_legendre(k, e)  # x2: [-1, 0]
        I[(l, k)] = np.einsum("rgej,e->rgj", KI, wt)
        Q[(l, k)] = np.einsum("rgej,e->rgj", KQ, wt)
    return I, Q


def full_range_projection(cfg, pair, table, n=2 * N_ETA) -> np.ndarray:
    """Order-0 intensity projection on a Gauss-Legendre rule over the whole of [-1, 1] (no parity used)."""
    t, w = leggauss(n)
    KI, _ = B.channel_kernels(cfg, cfg.gamma_b[list(CELLS)], eta_s=np.sqrt(1.0 - t**2), order=0, table=table)
    l, k = pair
    return np.einsum("gej,e->gj", KI[0], (2 * l + 1) * (2 * k + 1) / 2 * w * eval_legendre(l, t)
                     * eval_legendre(k, t))  # the /2 (not /4) carries the 2 of K^nat = 2 K^iso delta(mu - eta)


def u_orders(K: np.ndarray, h: float) -> np.ndarray:
    """(h^r/r!) d^r/d(ln gamma)^r for r = 0..2: the u_b spectra of Eq. illustrated derivative spectra."""
    return np.array([h**r / factorial(r) for r in ORDERS])[:, None, None] * K


def field_frame(cfg, Q: np.ndarray) -> np.ndarray:
    rot = np.exp(-2j * cfg.screen_mean * (C_SI / cfg.centres) ** 2) * np.exp(-1j * np.angle(cfg.azimuth_factor))
    return cfg.azimuth_factor * Q * rot


def spectra(cfg, table) -> dict:
    I, Q = projections(cfg, PAIRS, table)
    out = {}
    for p in PAIRS:
        P = field_frame(cfg, u_orders(Q[p], cfg.h))
        out[("I", p)] = u_orders(I[p], cfg.h)
        out[("Q'", p)], out[("U'", p)] = P.real, P.imag
    return out


# ----------------------------------------------------------------------------- checks
def check_quadrature_and_projection(dense, table) -> dict:
    I256, _ = projections(dense, PAIRS, table, N_ETA)
    I128, _ = projections(dense, PAIRS, table, N_ETA_CHECK)
    conv = max(float(np.abs(I256[p] - I128[p]).max() / np.abs(I256[p]).max()) for p in PAIRS)
    kI, _ = B.kbar(replace(dense, eta_max=1.0, n_eta=N_ETA), dense.gamma_b[list(CELLS)], order=2, table=table)
    vs_kbar = float(np.abs(I256[(0, 0)] - kI).max() / np.abs(kI).max())
    Isym, _ = projections(dense, ((0, 2), (2, 0)), table)
    sym = float(np.abs(Isym[(0, 2)] - Isym[(2, 0)]).max() / np.abs(Isym[(2, 0)]).max())
    odd = full_range_projection(dense, (1, 0), table)
    even = full_range_projection(dense, (0, 0), table)
    return {"eta_quadrature_128_vs_256_max_rel": conv, "projection_00_vs_kbar_eta_max_1_max_rel": vs_kbar,
            "symmetry_02_vs_20_max_rel": sym,
            "odd_pair_10_over_00_full_range_rule": float(np.abs(odd).max() / np.abs(even).max())}


def check_field_derivatives(dense) -> dict:
    """Field derivatives by a 5-point stencil in x_B against e (K + 0.5 dK/dln gamma), per pair and order."""
    e = cost.DB / cost.B0
    offs = (-2, -1, 1, 2)
    tables = {o: cost.fixed_b_table(cost.B0 + o * STENCIL_STEP * cost.DB) for o in offs}
    tab0 = cost.fixed_b_table(cost.B0)
    base = projections(dense, PAIRS, tab0, order=1)[0]
    vals = {o: projections(dense, PAIRS, tables[o], order=0)[0] for o in offs}
    out = {}
    for p in PAIRS:
        v = {o: vals[o][p][0] for o in offs}
        d1 = (v[-2] - 8 * v[-1] + 8 * v[1] - v[2]) / (12 * STENCIL_STEP)
        ident = e * (base[p][0] + 0.5 * base[p][1])
        out[f"{p}"] = float(np.abs(d1 - ident).max() / np.abs(ident).max())
    return {"stencil_dB_vs_identity_max_rel": out, "step_xB": STENCIL_STEP,
            "note": "includes the stencil's own truncation error; the identity is exact for the continuum kernels"}


def check_depolarisation(table) -> dict:
    low = replace(PAPER, nu_hi=DEPOL_NU_GHZ * GHZ, n_channels=201)  # centres end at DEPOL_NU_GHZ
    _, Q = projections(low, PAIRS, table, order=0)
    _, Q0 = projections(replace(low, screen_mean=0.0, screen_sd=0.0), PAIRS, table, order=0)
    ratio = {f"{p}": (np.abs(Q[p][0]) / np.abs(Q0[p][0])).max(axis=1).tolist() for p in PAIRS}
    return {"up_to_GHz": DEPOL_NU_GHZ, "bound": DEPOL_FRACTION, "max_ratio_order0_by_pair_and_cell": ratio,
            "within_bound": bool(max(max(r) for r in ratio.values()) < DEPOL_FRACTION)}


def check_u_origin(dense, table) -> dict:
    cfg = replace(dense, screen_mean=0.0)
    _, Q = projections(cfg, PAIRS, table)
    P = {p: field_frame(cfg, u_orders(Q[p], cfg.h)) for p in PAIRS}
    return {"U_over_Q_max_without_mean_depth": float(max(np.abs(P[p].imag).max() / np.abs(P[p].real).max()
                                                         for p in PAIRS))}


def check_tail(S, nu, h) -> dict:
    """Panels (i, l): in the low-frequency tail K ~ gamma^(-2/3), so the u_b spectra are (-2h/3)^r/r! times order 0.

    For the higher-energy cell, U' at nu < TAIL_NU_GHZ where |order 0| exceeds 10% of its largest value: the
    ratios order 1 / order 0 and order 2 / order 0 against -2h/3 and 2h^2/9, and the largest difference of the
    curves as drawn (each divided by its largest absolute value).
    """
    out = {"prediction_order1_over_order0": -2 * h / 3, "prediction_order2_over_order0": 2 * h * h / 9}
    for p in PAIRS:
        Y = S[("U'", p)][:, 1]
        sel = (np.abs(Y[0]) > 0.1 * np.abs(Y[0]).max()) & (nu < TAIL_NU_GHZ)
        r1, r2 = Y[1][sel] / Y[0][sel], Y[2][sel] / Y[0][sel]
        n = [y / np.abs(y).max() for y in Y]
        out[f"U' {p}"] = {"order1_over_order0": [float(r1.min()), float(r1.max())],
                          "order2_over_order0": [float(r2.min()), float(r2.max())],
                          "drawn_order2_minus_order0_max": float(np.abs(n[2] - n[0]).max()),
                          "drawn_order1_plus_order0_max": float(np.abs(n[1] + n[0]).max())}
    return out


# ----------------------------------------------------------------------------- figure
def frequency_axis(ax, bottom: bool) -> None:
    ax.set_xscale("log")
    ax.set_xlim(0.1, NU_TOP_GHZ)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 1.0, 10.0, 100.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_locator(LogLocator(base=10, subs=(2, 3, 5)))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if bottom:
        ax.set_xlabel("Frequency [GHz]")
    else:
        ax.tick_params(labelbottom=False)


def draw(nu, gamma_b, S):
    fig = plt.figure(figsize=(fs.WIDTH, 4.7))
    gs = fig.add_gridspec(4, 3, left=0.075, right=0.99, top=0.885, bottom=0.085, hspace=0.62, wspace=0.10)
    rows = [(p, c) for c in range(len(CELLS)) for p in PAIRS]
    letters = iter("abcdefghijkl")
    for i, (p, c) in enumerate(rows):
        for j, comp in enumerate(COMPONENTS):
            ax = fig.add_subplot(gs[i, j])
            frequency_axis(ax, bottom=(i == len(rows) - 1))
            ax.axhline(0, color=fs.GREY, lw=0.4, zorder=0)
            for r in ORDERS:
                y = S[(comp, p)][r, c]
                ax.plot(nu, y / np.abs(y).max(), solid_capstyle="round", zorder=3 + r, **STYLE[r])
            ax.set_ylim(-1.15, 1.15)
            ax.set_yticks([-1, 0, 1])
            ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"${v:g}$"))
            ax.tick_params(labelleft=(j == 0))
            fs.panel_title(ax, rf"({next(letters)}) ${SYMBOL[comp]}$, $({p[0]},{p[1]})$, "
                               rf"$\gamma_b={gamma_b[c]:.0f}$")
    handles = [Line2D([], [], **STYLE[r], label=NAME_ORDER[r]) for r in ORDERS]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.53, 0.995), handlelength=2.8,
               columnspacing=2.0, frameon=False)
    fig.text(0.006, 0.48, r"Basis spectrum / peak $|{\rm value}|$", rotation=90, va="center")
    return fig


def main() -> None:
    fs.configure()
    table = cost.fixed_b_table(cost.B0)
    step = np.log10(PAPER.nu_hi / PAPER.nu_lo) / (DENSE - 1)  # dex per dense step
    n_wide = int(np.ceil(np.log10(NU_TOP_GHZ * GHZ / PAPER.nu_lo) / step)) + 1
    dense = replace(PAPER, nu_hi=PAPER.nu_lo * 10 ** (step * (n_wide - 1)), n_channels=n_wide)
    if not np.allclose(dense.centres[:DENSE:10], PAPER.centres, rtol=1e-12, atol=0):
        raise RuntimeError("dense centres do not reproduce the paper channels")
    S = spectra(dense, table)
    S41 = spectra(PAPER, table)
    if not all(np.allclose(S[k][..., :DENSE:10], S41[k], rtol=1e-9, atol=1e-12 * np.abs(S41[k]).max()) for k in S):
        raise RuntimeError("dense spectra do not reproduce the paper-channel spectra")
    nu = dense.centres / GHZ
    gamma_b = PAPER.gamma_b[list(CELLS)]
    checks = {"dense_vs_paper_channels": "paper channels reproduced to 1e-9 (checked in main())",
              "centres_GHz": [float(nu[0]), float(nu[-1]), int(nu.size)]}
    checks.update(check_quadrature_and_projection(dense, table))
    checks["field_derivatives"] = check_field_derivatives(dense)
    checks["depolarisation"] = check_depolarisation(table)
    checks.update(check_u_origin(dense, table))
    I00 = S[("I", (0, 0))][0]
    checks["order0_intensity_peak_GHz"] = nu[I00.argmax(axis=1)].tolist()
    checks["tail_panels_i_l"] = check_tail(S, nu, dense.h)
    np.savez(NPZ, nu=nu, gamma_b=gamma_b, cells=np.array(CELLS), pairs=np.array(PAIRS),
             **{f"{c}_{p[0]}{p[1]}".replace("'", "p"): v for (c, p), v in S.items()})
    (HERE / "figures").mkdir(exist_ok=True)
    fs.save(draw(nu, gamma_b, S), NAME)
    figures = {relative(HERE / "figures" / f"{NAME}.{e}"): sha256(HERE / "figures" / f"{NAME}.{e}")
               for e in ("pdf", "png")}
    write_json(OUT_JSON, {"description": __doc__, "cells": list(CELLS), "gamma_b": gamma_b.tolist(),
                          "pairs": [list(p) for p in PAIRS], "B0_muG": cost.B0, "n_eta": N_ETA,
                          "centres": n_wide, "checks": checks, "npz": relative(NPZ),
                          "npz_sha256": sha256(NPZ), "figures": figures,
                          "provenance": provenance(__file__, T0, inputs=(HERE / "lib" / "figure_style_paper.py",
                                                                         HERE / "cost.py"))})
    print(json.dumps(checks, indent=1))
    bad = [checks["depolarisation"]["within_bound"], checks["symmetry_02_vs_20_max_rel"] < 1e-12,
           checks["odd_pair_10_over_00_full_range_rule"] < 1e-12, checks["U_over_Q_max_without_mean_depth"] < 1e-12]
    if not all(bad):
        raise SystemExit("a caption check failed")


if __name__ == "__main__":
    main()
