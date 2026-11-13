"""Drawing code for the classical-model figures (house style, final text width).

reference_ensembles     electron spectra, channel intensities, intrinsic polarisation
moment_fit_gallery      seeded UPRE fits of E1, E2, E4-KP, E5 (E4-KP reference: JP(t_eff) moments)
moments_and_parameters  E2 moments, retained combinations, parameters, B-gamma ridge
galprop_fit             noise-free fits to the GALPROP |b| >= 60 deg spectrum (external test)

Compact variants (same panel functions, fewer panels):
reference_ensembles_compact     old panels (a) and (b) in one row
moment_fit_gallery_compact      Stokes gallery of rows GALLERY_COMPACT (E2, E4-KP, E5): I, Q, U
                                fits with the 1% and the low-noise fits (fig_stokes_fit_gallery)
moments_and_parameters_compact  old panels (a) and (d) in one row, relabelled (a) and (b)

field_energy_ridge      contours of old panel (d) above, SED error along the ridge at N = 2 and 4 below
                        (ridge_sed_results.json), unlettered

Colours, line styles and line widths come from lib/figure_style_paper.py.
All arrays come from figure_data.build_all() and the loaded results; no number is typed here.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patheffects as pe
from matplotlib.legend_handler import HandlerTuple
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnnotationBbox, DrawingArea, HPacker, TextArea
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, Formatter, FuncFormatter, MaxNLocator, NullFormatter, NullLocator
from scipy.interpolate import RectBivariateSpline

import ensembles as E
from config import PAPER
import figure_style_paper as fs

# One colour and line style per ensemble, shared by every paper figure (lib/figure_style_paper.py).
COLOR = dict(fs.ENSEMBLE)
STYLE = dict(fs.ENSEMBLE_LS)
LABEL = {"E1": "E1 power law", "E2": "E2 broken power law", "E3": "E3 index mixture",
         "E4-JP": "E4-JP ageing", "E4-KP": "E4-KP ageing", "E5": "E5 two zones",
         "E1M": r"E1M Maxwellian $B$"}
FACET = {"E1": "E1 power law", "E2": "E2 broken PL", "E3": "E3 mixture", "E4-JP": "E4-JP ageing",
         "E4-KP": "E4-KP ageing", "E5": "E5 two zones"}
SHORT = {"E1": "E1", "E2": "E2", "E3": "E3", "E4-JP": "E4-JP", "E4-KP": "E4-KP", "E5": "E5", "E1M": "E1M"}
DIRECT, GREY = fs.INK, fs.GREY
Q_LS = (0, (1.6, 1.0))  # Q residual: short dense dashes, unlike the grey reference curves
WITNESS = fs.VERMILION  # nonnegative-witness markers (moments_and_parameters); not an ensemble colour
PSEUDO_LS = (0, (5.0, 1.4, 1.2, 1.4, 1.2, 1.4))  # E4-KP reference curve: moments of the JP spectrum with t_eff
GHZ = 1e9


RADIO_AXIS = dict(xlim=(0.09, 11.0), ticks=(0.1, 1.0, 10.0))


def ghz_axis(ax, label=True, xlim=RADIO_AXIS["xlim"], ticks=RADIO_AXIS["ticks"]):
    ax.set_xscale("log")
    ax.set_xlim(*xlim)
    ax.xaxis.set_major_locator(FixedLocator(list(ticks)))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if label:
        ax.set_xlabel("Frequency [GHz]")


def log_ticks(ax, ticks=(0.2, 0.5, 1.0, 2.0)):
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def panel(ax, text):
    fs.panel_title(ax, text)


# ----------------------------------------------------------------------------- reference_ensembles
REF_ORDER = ("E1", "E2", "E3", "E4-JP", "E4-KP", "E5")  # ensembles drawn in (a), (c); (b) adds E1M


MARK_START = {"E2": 0.03, "E3": 0.10, "E4-JP": 0.40, "E4-KP": 0.46}  # stagger the sparse markers


def _ens_line(ax, x, y, k, markers=True):
    """Ensemble curve in its colour, line style and sparse markers (readable in greyscale)."""
    style = dict(color=COLOR[k], ls=STYLE[k], lw=fs.LW)
    marker = fs.ENSEMBLE_MARKER[k]
    if markers and marker:
        hollow = k == "E4-KP"
        style.update(marker=marker, ms=3.4, markevery=(MARK_START[k], 0.15),
                     mfc="white" if hollow else COLOR[k], mec=COLOR[k] if hollow else "white",
                     mew=0.8 if hollow else 0.5)
    ax.plot(x, y, **style)


def _end_labels(ax, x_end, ends: dict, sep=0.1) -> None:
    """Labels right of the axes at each curve's end value, spread to >= sep dex, with leader lines."""
    keys = sorted(ends, key=lambda k: ends[k])
    want = np.log10([ends[k] for k in keys])
    pos = want.copy()
    for _ in range(200):  # relax overlapping labels apart, keeping their mean
        gap = np.diff(pos)
        push = np.clip(sep - gap, 0.0, None) / 2
        if not push.any():
            break
        pos[:-1] -= push
        pos[1:] += push
    x_text = x_end * 1.55
    for k, y0, y1 in zip(keys, want, pos):
        ax.annotate(k, xy=(x_end * 1.04, 10**y0), xytext=(x_text, 10**y1), ha="left", va="center",
                    annotation_clip=False,
                    arrowprops=None if abs(y1 - y0) < 0.02 else
                    dict(arrowstyle="-", color=GREY, lw=0.5, shrinkA=1.5, shrinkB=0.5))


def _panel_electron_spectra(ax_a, ax_p2, el, labels=False):
    """(a) electron spectra relative to E1, with the E4-KP pitch-anisotropy subpanel below."""
    gc = el["gamma_c"]
    for ax in (ax_a, ax_p2):  # E4-KP cut-offs for 0 <= |eta| <= 1/2
        ax.axvspan(gc["KP_mu0"], gc["KP_mu_half"], color=fs.LIGHT, lw=0, zorder=0)
    g = el["gamma"]
    rel = {k: np.where(el["rel_density"][k] > 0, el["rel_density"][k], np.nan) for k in REF_ORDER}
    for k in REF_ORDER:
        _ens_line(ax_a, g, rel[k], k, markers=not labels)
    ax_a.set(xscale="log", yscale="log", xlim=(PAPER.gamma_lo, PAPER.gamma_hi), ylim=(0.12, 4.0))
    ax_a.set_ylabel(r"$\mathcal{N}(\gamma)/\mathcal{N}_{\rm E1}(\gamma)$")
    ax_a.tick_params(labelbottom=False)
    log_ticks(ax_a)
    panel(ax_a, r"(a) Electron spectra")
    if labels:  # where the curves separate; E1M has the E1 electron spectrum
        i2, i5 = np.nanargmax(rel["E2"]), np.nanargmin(rel["E5"])
        ax_a.text(g[i2], rel["E2"][i2] * 1.12, "E2", ha="center", va="bottom")
        ax_a.text(g[i5], rel["E5"][i5] / 1.12, "E5", ha="center", va="top")
        x13 = 4.0e4
        j = np.argmin(np.abs(g - x13))
        ax_a.text(x13, rel["E1"][j] * 1.07, "E1", ha="center", va="bottom")
        ax_a.text(x13, rel["E3"][j] / 1.07, "E3", ha="center", va="top")
        ax_a.text(gc["KP_mu0"] / 1.12, 0.16, "E4-KP", ha="right", va="center")
        ax_a.text(gc["KP_mu_half"] * 1.25, 0.16, "E4-JP", ha="left", va="center")
    _ens_line(ax_p2, el["gamma_p2"], el["p2"], "E4-KP", markers=not labels)
    ax_p2.set(xscale="log", xlim=(PAPER.gamma_lo, PAPER.gamma_hi), ylim=(-0.1, 1.1))
    ax_p2.yaxis.set_major_locator(FixedLocator([0.0, 1.0]))
    ax_p2.set_xlabel(r"Lorentz factor $\gamma$")
    ax_p2.set_ylabel(r"$\langle \mathsf{L}_2\rangle_{\rm ang}$")
    ax_p2.text(0.03, 0.68, "E4-KP", transform=ax_p2.transAxes, va="center")


def _panel_channel_intensity(ax_b, nu, fnpz, labels=False, label_sep=0.1):
    """(b) channel intensities relative to E1."""
    I1 = fnpz["E1"][:, 0]
    ratio = {k: fnpz[k.replace("-", "_")][:, 0] / I1 for k in REF_ORDER + ("E1M",)}
    for k in REF_ORDER + ("E1M",):
        _ens_line(ax_b, nu, ratio[k], k, markers=not labels)
    ghz_axis(ax_b)
    ax_b.set(yscale="log", ylim=(0.07 if labels else 0.12, 4.0))
    log_ticks(ax_b, (0.1, 0.2, 0.5, 1.0, 2.0) if labels else (0.2, 0.5, 1.0, 2.0))
    ax_b.set_ylabel(r"$I_j/I_j^{\rm E1}$")
    panel(ax_b, r"(b) Channel intensities")
    if labels:
        _end_labels(ax_b, nu[-1], {k: v[-1] for k, v in ratio.items()}, sep=label_sep)


def _panel_polarisation_fraction(ax_c, nu, intr, fwd):
    """(c) ordered-field polarisation fraction against (p+1)/(p+7/3)."""
    for k in REF_ORDER:
        ax_c.plot(nu, intr["fraction"][k], color=COLOR[k], ls=STYLE[k], lw=1.1)
    pi_pl = fwd["e1_analytic"]["Pi_unbounded_(p+1)/(p+7/3)"]
    ax_c.axhline(pi_pl, color=GREY, lw=0.6, ls=":")
    ax_c.annotate(r"$(p+1)/(p+7/3)$ for $p=2.6$", xy=(0.15, pi_pl), xytext=(0.12, 0.655),
                  color=GREY, va="center", arrowprops=dict(arrowstyle="-", color=GREY, lw=0.5))
    ghz_axis(ax_c)
    ax_c.set_ylabel(r"$|P_j|/I_j$")
    panel(ax_c, r"(c) Polarisation fraction, ordered field, no screen")


def _panel_e5_angle(ax_d, nu, fwd):
    """(d) E5 polarisation angle without a screen and after de-rotation."""
    e5 = fwd["ensembles"]["E5"]
    sel = np.asarray(e5["P_over_I"]) > 0.03
    ax_d.plot(nu, e5["chi_intrinsic_no_screen_deg"], color=COLOR["E5"], ls="-", lw=1.1,
              label="no screen")
    ax_d.plot(nu[sel], np.asarray(e5["chi_derotated_deg"])[sel], "o", ms=3, mfc="white",
              mec=COLOR["E5"], mew=0.9, label=r"screen, de-rotated ($|P|/I>3\%$)")
    for z, key in (("A", "phi_a"), ("B", "phi_b")):
        chi_z = np.degrees(0.5 * np.angle(-np.exp(2j * e5["params"][key])))
        ax_d.axhline(chi_z, color=GREY, lw=0.6, ls=":")
        ax_d.text(9.5, chi_z + 1.0, f"zone {z} alone", ha="right", va="bottom", color=GREY)
    ghz_axis(ax_d)
    ax_d.set_ylim(-80, -26)
    ax_d.set_ylabel(r"E5 angle $\chi_j$ [deg]")
    ax_d.legend(loc="upper left", bbox_to_anchor=(0.0, 0.88), frameon=False, handlelength=1.6,
                borderaxespad=0.1)
    panel(ax_d, r"(d) E5 polarisation angle")


def _ensemble_legend(fig):
    """Figure legend of the seven ensembles, two rows of four above the panels."""
    keys = ("E1", "E1M", "E2", "E3", "E4-JP", "E4-KP", "E5")  # column-major: related ensembles share a column
    handles = [Line2D([], [], color=COLOR[k], ls=STYLE[k], lw=fs.LW, label=LABEL[k]) for k in keys]
    fig.legend(handles=handles, loc="upper center", ncol=4, bbox_to_anchor=(0.53, 1.0),
               columnspacing=1.4, handlelength=2.2)


def _electron_axes(fig, cell):
    """(a) and its pitch-anisotropy subpanel in one grid cell."""
    sub = cell.subgridspec(2, 1, height_ratios=[2.6, 1], hspace=0.08)
    return fig.add_subplot(sub[0]), fig.add_subplot(sub[1])


def fig_reference_ensembles(D: dict, data: dict):
    fnpz = data["forward_npz"]
    nu = fnpz["centres"] / GHZ
    fig = plt.figure(figsize=(6.1, 4.9))
    outer = fig.add_gridspec(2, 2, left=0.095, right=0.985, top=0.865, bottom=0.085,
                             hspace=0.42, wspace=0.3)
    ax_a, ax_p2 = _electron_axes(fig, outer[0, 0])
    ax_b, ax_c, ax_d = (fig.add_subplot(outer[i, j]) for i, j in ((0, 1), (1, 0), (1, 1)))
    _panel_electron_spectra(ax_a, ax_p2, D["electrons"])
    _panel_channel_intensity(ax_b, nu, fnpz)
    _panel_polarisation_fraction(ax_c, nu, D["intrinsic"], data["forward"])
    _panel_e5_angle(ax_d, nu, data["forward"])
    _ensemble_legend(fig)
    return fig


def fig_reference_ensembles_compact(D: dict, data: dict):
    """Panels (a) and (b) of fig_reference_ensembles in one row, labelled directly (no legend)."""
    fnpz = data["forward_npz"]
    nu = fnpz["centres"] / GHZ
    height = 2.4
    fig = plt.figure(figsize=(6.1, height))
    outer = fig.add_gridspec(1, 2, left=0.095, right=0.905, top=1 - 0.24 / height, bottom=0.42 / height,
                             wspace=0.3)
    sub = outer[0, 0].subgridspec(2, 1, height_ratios=[2.5, 1], hspace=0.12)
    ax_a, ax_p2 = fig.add_subplot(sub[0]), fig.add_subplot(sub[1])
    ax_b = fig.add_subplot(outer[0, 1])
    _panel_electron_spectra(ax_a, ax_p2, D["electrons"], labels=True)
    # Keep directly labelled, nearly coincident E1/E1M/E3 curves legible at the
    # shorter printed height; the leader lines still end at the original data.
    _panel_channel_intensity(ax_b, nu, fnpz, labels=True, label_sep=0.14)
    fig.align_ylabels([ax_a, ax_p2])
    return fig


# ----------------------------------------------------------------------------- moment_fit_gallery
def _power_law_reference(nu, I):
    c = np.polyfit(np.log(nu), np.log(I), 1)
    return np.exp(np.polyval(c, np.log(nu)))


def _row_intensity(ax, nu, f, fit_i, held_i, color, ref_ls="--"):
    ref = _power_law_reference(nu, f["S"][:, 0])
    pred, sd = f["pred"][:, 0], f["sd_I"]
    ax.fill_between(nu, (pred - sd) / ref, (pred + sd) / ref, color=color, alpha=0.35, lw=0)
    ax.plot(nu, f["S"][:, 0] / ref, color=DIRECT, lw=1.0)
    ax.plot(nu, f["forward_true"][:, 0] / ref, color=GREY, ls=ref_ls, lw=0.9)
    sig = f["sigma"][:, 0] / ref
    ax.errorbar(nu[fit_i], f["data_fit"][:, 0] / ref[fit_i], sig[fit_i], fmt="o", ms=2.4,
                color=color, mec="black", mew=0.4, elinewidth=0.6, capsize=0)
    ax.errorbar(nu[held_i], f["data_held"][:, 0] / ref[held_i], sig[held_i], fmt="s", ms=2.4,
                mfc="white", mec=color, color=color, mew=0.7, elinewidth=0.6, capsize=0)


P_ERR_LS = (0, (3.0, 1.4))  # |Delta P| / I in the log residual panel
RES_YLIM = (3e-4, 6.0)  # per cent of I_j


def _row_residual(ax, nu, f, color, ref_ls="--", label=True):
    """|fit - direct| / I_j for I (solid) and P = Q + iU (dashed) at each noise level (dotted line).

    label=False omits the noise-level labels (drawn in the top row only; the lines are the same in every row).
    """
    I = f["S"][:, 0]
    levels = [(PAPER.noise_frac, f["pred"])]
    if "pred_low" in f:
        levels.append((f["noise_low"], f["pred_low"]))
    for frac, pred in levels:
        d = pred - f["S"]
        ax.axhline(100 * frac, color=GREY, ls=":", lw=fs.LW_REF, zorder=1)
        ax.plot(nu, 100 * np.abs(d[:, 0]) / I, color=color, lw=1.0, zorder=3)
        ax.plot(nu, 100 * np.abs(d[:, 1] + 1j * d[:, 2]) / I, color=color, ls=P_ERR_LS, lw=0.8, zorder=3)
        if label:
            ax.text(nu[0] * 1.04, 100 * frac * 1.35, f"{100 * frac:g}\\% noise", ha="left", va="bottom",
                    color=fs.INK, fontsize=8)
    ax.set_yscale("log")
    ax.set_ylim(*RES_YLIM)


def _row_polarisation(ax, nu, f, fit_i, held_i, color, ref_ls="--"):
    frac = lambda s: np.hypot(s[..., 1], s[..., 2]) / s[..., 0]  # noqa: E731
    pred = f["pred"]
    pfit, sd = frac(pred), f["sd_absP"] / pred[:, 0]
    ax.fill_between(nu, pfit - sd, pfit + sd, color=color, alpha=0.35, lw=0)
    ax.plot(nu, frac(f["S"]), color=DIRECT, lw=1.0)
    ax.plot(nu, frac(f["forward_true"]), color=GREY, ls=ref_ls, lw=0.9)
    sig = f["sigma"][:, 1]
    for idx, d, kw in ((fit_i, f["data_fit"], dict(fmt="o", mec="black", mew=0.4)),
                       (held_i, f["data_held"], dict(fmt="s", mfc="white", mec=color, mew=0.7))):
        ax.errorbar(nu[idx], frac(d), sig[idx] / d[:, 0], ms=2.4, color=color, elinewidth=0.6, capsize=0, **kw)
    ax.set_ylim(-0.02, None)


def _row_angle(ax, nu, f, fit_i, held_i, chi, color):
    sel = np.asarray(chi["P_over_I_true"]) > 0.03
    true, joint = np.asarray(chi["chi_true_deg"]), np.asarray(chi["chi_joint_seeded_deg"])
    sd = np.asarray(chi["chi_joint_sigma_deg_linearised_seeded"])
    free = np.asarray(chi["chi_separable_free_factor_seeded_deg"])
    ax.fill_between(nu[sel], (joint - sd)[sel], (joint + sd)[sel], color=color, alpha=0.35, lw=0)
    ax.plot(nu[sel], true[sel], color=DIRECT, lw=1.0)
    ax.fill_between([], [], [], color=color, alpha=0.35, lw=0, label=r"joint $(a,M^{(2)})$")
    ax.plot(nu[sel], free[sel], color=fs.VERMILION, ls="-.", lw=fs.LW, label="one azimuth\nfactor")
    d_all = np.empty_like(f["S"])
    d_all[fit_i], d_all[held_i] = f["data_fit"], f["data_held"]
    ang = np.degrees(E.derotated_angle(PAPER, d_all))  # same de-rotation as the fits run
    ang = joint + (ang - joint + 90.0) % 180.0 - 90.0  # branch nearest the fit
    err = np.degrees(0.5 * f["sigma"][:, 1] / np.hypot(d_all[:, 1], d_all[:, 2]))
    for idx, kw in ((fit_i, dict(fmt="o", mec="black", mew=0.4)),
                    (held_i, dict(fmt="s", mfc="white", mec=color, mew=0.7))):
        keep = idx[sel[idx]]
        ax.errorbar(nu[keep], ang[keep], err[keep], ms=2.4, color=color, elinewidth=0.6, capsize=0, **kw)
    # every plotted point and error bar inside the axes; head room above for the legend
    lo = min(np.min((ang - err)[sel]), np.min(true[sel]), np.min(free[sel]))
    ax.set_ylim(5.0 * np.floor(lo / 5.0) - 1.0, -33)
    # right of the lowest data point's error bar, clear of the row label at x = 1.03
    ax.legend(loc="lower right", bbox_to_anchor=(1.01, 0.0), frameon=False, handlelength=1.2, handletextpad=0.4,
              borderpad=0.2, borderaxespad=0.0, labelspacing=0.3)


GALLERY_COMPACT = ("E2", "E4-KP", "E5")  # rows of moment_fit_gallery_compact (E1 row dropped)


def fig_moment_fit_gallery(D: dict, data: dict, rows: tuple[str, ...] | None = None):
    """One row per ensemble; rows=None draws every fit in figure_data.GALLERY order."""
    bd = D["bands"]
    nu = bd["centres"] / GHZ
    fit_i, held_i = bd["fit_idx"], bd["held_idx"]
    rows = tuple(bd["fits"]) if rows is None else tuple(rows)
    missing = [r for r in rows if r not in bd["fits"]]
    if missing or not rows:
        raise KeyError(f"gallery rows {missing or rows} not in figure_data.GALLERY {tuple(bd['fits'])}")
    n = len(rows)
    compact = rows == GALLERY_COMPACT
    height = 1.24 * n + 0.94 if compact else 1.48 * n + 0.88
    legend_height = 0.80 if compact else 0.74
    fig, axes = plt.subplots(n, 3, figsize=(6.1, height), sharex=True,
                             gridspec_kw=dict(left=0.085, right=0.948, top=1 - legend_height / height,
                                              bottom=0.44 / height,
                                              hspace=0.18 if compact else 0.12, wspace=0.34))
    chi = data["fits"]["E5"]["chi_nu"]
    for row, name in enumerate(rows):
        f, c = bd["fits"][name], COLOR[name]
        ls = PSEUDO_LS if name == "E4-KP" else "--"
        ax_i, ax_r, ax_p = axes[row]
        _row_intensity(ax_i, nu, f, fit_i, held_i, c, ls)
        _row_residual(ax_r, nu, f, c, ls, label=(row == 0))
        if name == "E5":
            _row_angle(ax_p, nu, f, fit_i, held_i, chi, c)
            ax_p.set_ylabel(r"$\chi_j$ [deg]")
        else:
            _row_polarisation(ax_p, nu, f, fit_i, held_i, c, ls)
            ax_p.set_ylabel(r"$|P_j|/I_j$")
        ax_i.set_ylabel(r"$I_j/I_{\rm PL}$")
        ax_r.set_ylabel(r"Error [\%]")
        ax_p.text(1.03, 0.5, f"{FACET[name]}\n$R_\\star={f['R_star']}$", transform=ax_p.transAxes,
                  rotation=-90, ha="left", va="center", ma="center", linespacing=1.1)
        for ax in axes[row]:
            ghz_axis(ax, label=(row == n - 1))
    for ax, t in zip(axes[0], (r"(a) Intensity", r"(b) $|{\rm fit}-{\rm direct}|/I_j$", r"(c) Polarisation")):
        panel(ax, t)
    fit_handle = Patch(facecolor=GREY, alpha=0.45, lw=0)
    handles = [Line2D([], [], color=DIRECT, lw=1.0, label="direct"),
               Line2D([], [], color=GREY, ls="--", lw=0.9, label="true moments, $N=2$"),
               Line2D([], [], color=GREY, ls=PSEUDO_LS, lw=0.9, label=r"E4-KP: JP($t_{\rm eff}$) moments, $N=2$"),
               fit_handle,
               Line2D([], [], marker="o", ls="", ms=3, color=GREY, mec="black", mew=0.4, label="data, fitted"),
               Line2D([], [], marker="s", ls="", ms=3, mfc="white", mec=GREY, mew=0.7, label="data, withheld"),
               Line2D([], [], color=GREY, ls=":", lw=fs.LW_REF, label="noise level"),
               Line2D([], [], color=DIRECT, lw=1.0, label="$I$ error"),
               Line2D([], [], color=DIRECT, ls=P_ERR_LS, lw=0.8, label="$P$ error")]
    labels = [r"UPRE fit $\pm1\sigma$" if h is fit_handle else h.get_label() for h in handles]
    order = [0, 3, 6, 1, 4, 7, 2, 5, 8]  # column-major fill: three rows of three
    fig.legend(handles=[handles[i] for i in order], labels=[labels[i] for i in order], loc="upper center",
               ncol=3, frameon=False, bbox_to_anchor=(0.53, 1.0), columnspacing=1.6, handlelength=2.4)
    return fig


# ----------------------------------------------------------------------------- Stokes gallery (moment_fit_gallery_compact)
# Column headings carry the plotted quantity, so the panels need no y labels.
STOKES_COLUMNS = ((0, r"(a) Intensity / power law, $I_j/I_{\rm PL}$"), (1, r"(b) Stokes $Q_j/I_j$"), (2, r"(c) Stokes $U_j/I_j$"))
# Row titles: the model in plain words, its code in parentheses.
ROW_TITLE = {"E1": "Power law (E1)", "E2": "Smoothly broken power law (E2)",
             "E3": "Index mixture (E3)", "E4-JP": "Ageing, isotropised pitch angles (E4-JP)",
             "E4-KP": "Ageing, fixed pitch angles (E4-KP)", "E5": "Two power-law zones (E5)",
             "E1M": r"Power law, Maxwellian field strength (E1M)"}
# inches; "gap" is the band above each block that holds its row title
STOKES_HEIGHTS = dict(main=0.80, inner=0.05, sub=0.38, gap=0.24, top=0.56, bottom=0.36)
STOKES_ERR_LIM = (4e-5, 12.0)  # per cent of I_j; the axis line sits below the floor, the top clears the E5 dash-dot error
# Q/U/I error panels are not clipped: errors below the axis leave the panel through the bottom
STOKES_ERR_TICKS = {1e-4: r"$10^{-4}$", 1e-2: "0.01", 1.0: "1"}
# error curves: the 1% fit heavy in the row colour, the low-noise fit thin in a darker tone of it
# (weight and tone, not hue or markers), so the thin curve keeps its contrast in greyscale
ERR_STYLE = {"high": dict(lw=1.3), "low": dict(lw=0.75)}
ERR_LOW_DARKEN = 0.35  # fraction of INK mixed into the row colour for the low-noise curve
# noise-level reference, drawn above the error curves so it shows where they sit on it
NOISE_STYLE = dict(color="#8C8C8C", lw=0.5, dashes=(1, 2))
NOISE_ZORDER = {"high": 3, "low": 1.5}  # above the heavy 1% curve, below the thin low-noise curve
NOISE_LEGEND_STYLE = dict(color="#8C8C8C", lw=1.0, dashes=(1, 1.5))  # heavier so the handle shows at legend size
FREE_FACTOR_LS = "-."
TRUE_STYLE = dict(color="#C8C8C8", lw=4.5, solid_capstyle="round", zorder=1)  # wide light band under the fit
FIT_LW = 0.9
# thin dark underlay: the fit stays darker than the true-spectrum band in greyscale, whatever the row colour
FIT_HALO = [pe.Stroke(linewidth=FIT_LW + 0.4, foreground=fs.INK, alpha=0.55), pe.Normal()]
DATA_STYLE = dict(fmt="o", ms=2.4, color=fs.INK, mec=fs.INK, elinewidth=0.6, capsize=0, zorder=4)


def _err_color(color, style):
    """Row colour for the 1% error curve, a darker tone of it for the low-noise curve."""
    if style == "high":
        return color
    c, ink = np.array(mcolors.to_rgb(color)), np.array(mcolors.to_rgb(fs.INK))
    return tuple((1 - ERR_LOW_DARKEN) * c + ERR_LOW_DARKEN * ink)


def _stokes_main(ax, nu, f, k, color, fit_i):
    """True spectrum (wide light line), fitted data with 1-sigma noise bars, fit (thin line in the row colour).

    I is shown relative to a power law fitted to the true spectrum, Q and U relative to I_j.
    """
    I = f["S"][:, 0]
    norm = _power_law_reference(nu, I) if k == 0 else I
    ax.plot(nu, f["S"][:, k] / norm, **TRUE_STYLE)
    ax.plot(nu, f["pred"][:, k] / norm, color=color, lw=FIT_LW, zorder=3, path_effects=FIT_HALO)
    if k > 0 and "pred_free_factor" in f:
        ff = f["pred_free_factor"][:, k] / I
        ax.plot(nu, ff, color=fs.VERMILION, ls=FREE_FACTOR_LS, lw=FIT_LW, zorder=3)
        if k == 1:  # direct label inside the empty lower right, under the dash-dot tail
            ax.text(0.99, 0.01, "energy-\nindependent\nazimuth", transform=ax.transAxes, ha="right", va="bottom",
                    color=_err_color(fs.VERMILION, "low"), fontsize="x-small", linespacing=0.95)
    ax.errorbar(nu[fit_i], f["data_fit"][:, k] / norm[fit_i], f["sigma"][fit_i, k] / norm[fit_i], **DATA_STYLE)
    ax.yaxis.set_major_locator(MaxNLocator(3, steps=[1, 2, 2.5, 5], min_n_ticks=3))
    ax.yaxis.set_major_formatter(CommonDecimals())


class CommonDecimals(Formatter):
    """Every tick on an axis printed with the same number of decimals; zero printed as 0."""

    def __call__(self, x, pos=None):
        if abs(x) < 1e-12:
            return "$0$"
        places = max((len(f"{v:g}".partition(".")[2]) for v in self.locs if "e" not in f"{v:g}"), default=0)
        return f"${x:.{places}f}$"


def _stokes_error(ax, nu, f, k, color):
    """|fit - direct| / I_j at 1% noise and at the low noise level; dotted lines mark the noise levels.

    For Q and U of E5 the 1%-noise fit with energy-independent azimuth is added in its dash-dot style.
    """
    I = f["S"][:, 0]
    levels = [(PAPER.noise_frac, f["pred"], "high")]
    if "pred_low" in f:
        levels.append((f["noise_low"], f["pred_low"], "low"))
    for frac, pred, style in levels:
        ax.axhline(100 * frac, zorder=NOISE_ZORDER[style], **NOISE_STYLE)
        err = 100 * np.abs(pred[:, k] - f["S"][:, k]) / I
        ax.plot(nu, err, color=_err_color(color, style), zorder=2, **ERR_STYLE[style])
    if k > 0 and "pred_free_factor" in f:
        err = 100 * np.abs(f["pred_free_factor"][:, k] - f["S"][:, k]) / I
        ax.plot(nu, err, color=fs.VERMILION, ls=FREE_FACTOR_LS, lw=FIT_LW, zorder=2)
    ax.set_yscale("log", nonpositive="mask")
    ax.set_ylim(*STOKES_ERR_LIM)
    ax.yaxis.set_major_locator(FixedLocator(list(STOKES_ERR_TICKS)))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: STOKES_ERR_TICKS.get(v, f"{v:g}")))
    ax.yaxis.set_minor_locator(NullLocator())


def _row_title(ax, name, R_star, color):
    """Row title above the block's left panel, led by a short segment in the row's fit colour."""
    swatch = DrawingArea(14, 6, 0, 0)
    swatch.add_artist(Line2D([0, 14], [3, 3], color=color, lw=FIT_LW, path_effects=FIT_HALO))
    text = TextArea(rf"{ROW_TITLE.get(name, name)}, $R_\star$ = {R_star}")
    box = HPacker(children=[swatch, text], align="center", pad=0, sep=4)
    ax.add_artist(AnnotationBbox(box, (0, 1), xycoords="axes fraction", xybox=(0, 4), boxcoords="offset points",
                                 box_alignment=(0, 0), frameon=False, pad=0, annotation_clip=False))


def fig_stokes_fit_gallery(D: dict, data: dict, rows: tuple[str, ...] = GALLERY_COMPACT):
    """Columns I, Q, U. One block per ensemble: spectrum above a flat error panel sharing the frequency axis."""
    COLS = STOKES_COLUMNS
    bd = D["bands"]
    nu = bd["centres"] / GHZ
    fit_i = bd["fit_idx"]
    h = STOKES_HEIGHTS
    ratios = [h["gap"], h["main"], h["inner"], h["sub"]] * len(rows)
    height = h["top"] + sum(ratios) + h["bottom"]
    fig = plt.figure(figsize=(fs.WIDTH, height))
    top = 1 - h["top"] / height
    gs = fig.add_gridspec(len(ratios), 3, height_ratios=ratios, hspace=0, wspace=0.32, left=0.105, right=0.99,
                          top=top, bottom=h["bottom"] / height)
    share, columns = None, [[] for _ in COLS]
    for r, name in enumerate(rows):
        f, c = bd["fits"][name], COLOR[name]
        last = r == len(rows) - 1
        for col, (k, heading) in enumerate(COLS):
            ax = fig.add_subplot(gs[4 * r + 1, col], sharex=share)
            share = share or ax
            sub = fig.add_subplot(gs[4 * r + 3, col], sharex=share)
            _stokes_main(ax, nu, f, k, c, fit_i)
            _stokes_error(sub, nu, f, k, c)
            if col == 0:
                sub.set_ylabel("Error\n" r"[\%]", linespacing=1.0, labelpad=2)  # two short lines fit the flat panel
                _row_title(ax, name, f["R_star"], c)

            for a in (ax, sub):
                ghz_axis(a, label=False)
                a.tick_params(labelbottom=(a is sub and last))
            if r == 0:  # column heading centred over its column, regular weight as the row titles
                pos = ax.get_position()
                fig.text((pos.x0 + pos.x1) / 2, top + 3 / 72 / height, heading, ha="center",
                         va="baseline", fontsize=plt.rcParams["axes.titlesize"])
            columns[col] += [ax, sub]
    for axes in columns:
        fig.align_ylabels(axes)
    mid = columns[1][-1].get_position()  # one frequency label, centred under the middle column
    fig.text((mid.x0 + mid.x1) / 2, 0.03 / height, "Frequency [GHz]", ha="center", va="bottom",
             fontsize=plt.rcParams["axes.labelsize"])

    def row_lines(**kw):  # one short segment per row colour, as drawn in the panels
        return tuple(Line2D([], [], color=COLOR[name], **kw) for name in rows)

    low = 100 * bd["fits"][rows[0]].get("noise_low", 0.0)
    data_handle = (Line2D([], [], marker="|", ls="", ms=7, mew=0.6, color=fs.INK),  # dot over a 1-sigma bar
                   Line2D([], [], marker="o", ls="", ms=3, color=fs.INK))
    # column-major fill with ncol=4: (true, data) | (the two fits) | (the two errors) | (noise level)
    handles = [Line2D([], [], **{**{k: v for k, v in TRUE_STYLE.items() if k != "zorder"}, "solid_capstyle": "butt"}),
               data_handle,
               row_lines(lw=FIT_LW, path_effects=FIT_HALO),
               Line2D([], [], color=fs.VERMILION, ls=FREE_FACTOR_LS, lw=FIT_LW),
               row_lines(**ERR_STYLE["high"]),
               tuple(Line2D([], [], color=_err_color(COLOR[name], "low"), **ERR_STYLE["low"]) for name in rows),
               Line2D([], [], **NOISE_LEGEND_STYLE)]
    labels = ["true spectrum", r"data, $1\%$ noise",
              "fit, full basis (row colour)", "fit, energy-independent azimuth",
              r"fit error in \% of $I_j$, $1\%$ noise", rf"fit error in \% of $I_j$, ${low:g}\%$ noise", "noise level"]
    fig.legend(handles=handles, labels=labels, loc="upper center", ncol=4, frameon=False,
               bbox_to_anchor=(0.53, 1.0), columnspacing=1.2, handlelength=2.4, handletextpad=0.5,
               handler_map={tuple: HandlerTuple(ndivide=None, pad=0.0), data_handle: HandlerTuple(ndivide=1)})
    return fig


# ----------------------------------------------------------------------------- moments_and_parameters
def _range(*arrays, pad=0.08):
    v = np.concatenate([np.ravel(a) for a in arrays])
    lo, hi = float(np.min(v)), float(np.max(v))
    d = pad * (hi - lo)
    return lo - d, hi + d


def _panel_moments(axs, D, data, letter="a"):
    e2 = D["e2"]
    edges, gb = PAPER.edges, PAPER.gamma_b
    nnls = np.asarray(data["fits"]["ensembles"]["E2"]["nnls_witness"]["seeded_implied_a_b0"])
    a_true, a_th = e2["a_true"].reshape(-1, 3), e2["a_theta"]["comp"].reshape(-1, 3)
    a_ts = e2["a_tsvd"].reshape(-1, 3)
    for r, ax in enumerate(axs):
        lo, hi = _range(a_true[:, r], a_th[:, r], a_ts[:, r], [0.0])
        if r == 0:
            hi = lo + (hi - lo) / 0.6  # two legend rows above the off-scale markers
        ax.stairs(a_true[:, r], edges, color=DIRECT, lw=1.1, baseline=None)
        ax.stairs(a_th[:, r], edges, color=COLOR["E2"], lw=1.1, ls="--", baseline=None)
        ax.axhline(0, color=GREY, lw=fs.LW_REF, zorder=0)
        ax.plot(gb, a_ts[:, r], "o", ms=3, mfc="white", mec="black", mew=0.8)
        ax.set(xscale="log", xlim=(edges[0], edges[-1]), ylim=(lo, hi))
        ax.set_ylabel([r"$a_{b0}$", r"$a_{b1}$", r"$a_{b2}$"][r])
    top = _range(a_true[:, 0], a_th[:, 0], a_ts[:, 0], [0.0])[1]
    ins = nnls <= top
    # The a_b0 scale ends at `top`; above it are the legend and the off-scale markers.
    axs[0].spines["left"].set_bounds(axs[0].get_ylim()[0], top)
    axs[0].yaxis.set_major_locator(FixedLocator([t for t in (0, 1, 2, 3, 4) if t <= top]))
    axs[0].plot(gb[ins], nnls[ins], "D", ms=3, color=WITNESS, mec="black", mew=0.4)
    for g, v in zip(gb[~ins], nnls[~ins]):  # off scale: value written beside an upward triangle
        axs[0].plot(g, top, "^", ms=4, color=WITNESS, mec="black", mew=0.4)
        right = g * 1.12 * 1.35 < edges[-1]
        axs[0].text(g * 1.12 if right else g / 1.12, top, f"{v:.0f}", va="center", ha="left" if right else "right")
    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)
    axs[-1].set_xlabel(r"Lorentz factor $\gamma$")
    handles = [Line2D([], [], color=DIRECT, lw=1.1, label="true"),
               Line2D([], [], color=COLOR["E2"], lw=1.1, ls="--", label=r"$a(\hat\theta)$"),
               Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec="black", label=r"min.-norm, $R_\star$"),
               Line2D([], [], marker="D", ls="", ms=3, color=WITNESS, mec="black", mew=0.4,
                      label="nonneg. witness")]
    axs[0].legend(handles=handles, loc="upper left", frameon=False, ncol=2, columnspacing=1.2,
                  handlelength=1.4, handletextpad=0.4, borderaxespad=0.05)
    panel(axs[0], rf"({letter}) E2 bin moments")


def _offscale_label(ax, x, dev, bias, lim):
    """Arrow marker at the axis edge for an off-scale deviation; values written beside it.

    An off-scale bias is given in the text only, so no marker sits on top of the arrow.
    """
    off_d, off_b = abs(dev) > lim, abs(bias) > lim
    if not (off_d or off_b):
        return
    side = np.sign(dev) if off_d else np.sign(bias)
    y = side * (lim - 0.3)
    lines = []
    if off_d:
        ax.plot(x, y, "v" if dev < 0 else "^", color="0.4", ms=4, mec="black", mew=0.4)
        lines.append(f"${dev:.1f}$")
    else:
        ax.plot(x, dev, "o", color="0.4", ms=3, mec="black", mew=0.3)
    if off_b:
        lines.append(rf"bias ${bias:.1f}$")
    ax.text(x - 1.3, y, "\n".join(lines), ha="right", va="bottom" if side < 0 else "top",
            linespacing=1.0)


def _panel_modes(ax, D):
    lim, x0, ticks, labels = 4.5, 0.0, [], []
    ax.axhspan(-1, 1, color="0.88", lw=0, zorder=0)
    ax.axhline(0, color="0.6", lw=0.5)
    for name in ("E1", "E2", "E3", "E4-JP", "E4-KP", "E5", "E1M"):
        m = D["modes"][name]
        order = sorted([(i, "s") for i in m["singles"]] + [(p[0], "p") for p in m["pairs"]])
        pos = {key: x0 + j for j, key in enumerate(order)}
        xs = np.array([pos[(i, "s")] for i in m["singles"]])
        dev, bias = np.asarray(m["dev"]), np.asarray(m["bias"])
        ok_d, ok_b = np.abs(dev) <= lim, np.abs(bias) <= lim
        ax.plot(xs[ok_d], dev[ok_d], "o", color=COLOR[name], ms=3, mec="black", mew=0.3)
        ax.plot(xs[ok_b], bias[ok_b], "_", color="black", ms=6, mew=1.2)
        for x, d, b in zip(xs, dev, bias):
            _offscale_label(ax, x, d, b, lim)
        if m["pairs"]:
            xp = [pos[(p[0], "p")] for p in m["pairs"]]
            ax.plot(xp, m["pair_z_equiv"], "D", ms=3.2, mfc="white", mec=COLOR[name], mew=0.9)
            ax.plot(xp, m["pair_bias_norm"], "_", color="black", ms=6, mew=1.2)
        ticks.append(x0 + 0.5 * (len(order) - 1))
        labels.append(SHORT[name] + (r"$^\dagger$" if m.get("truth") else ""))
        x0 += len(order) + 1.5
        ax.axvline(x0 - 1.25, color="0.8", lw=0.5)
    ax.set_xlim(-1.0, x0 - 1.25)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(ticks, labels)
    ax.tick_params(axis="x", length=0)
    ax.set_ylabel(r"$(\hat\beta_i-\beta_i)/\sigma_i$")
    handles = [Line2D([], [], marker="o", ls="", ms=3, color="0.5", mec="black", mew=0.3, label="noisy fit"),
               Line2D([], [], marker="_", ls="", ms=6, mew=1.2, color="black", label=r"bias $V_R^{\mathsf{T}}K\epsilon/\sigma$"),
               Line2D([], [], marker="D", ls="", ms=3.2, mfc="white", mec="0.4", label="E5 pair")]
    ax.legend(handles=handles, loc="upper left", frameon=False, ncol=3, handlelength=1.0,
              columnspacing=0.8, borderaxespad=0.1)
    if any(D["modes"][k].get("truth") for k in D["modes"]):
        ax.text(0.015, 0.855, r"$^\dagger$E4-KP relative to JP($t_{\rm eff}$) moments", transform=ax.transAxes,
                ha="left", va="top")
    panel(ax, r"(b) Retained combinations, noisy data")


def _panel_contours(ax, D):
    e2 = D["e2"]
    styles = {"comp": dict(colors=COLOR["E2"], linestyles="-", linewidths=1.1),
              "full": dict(colors="black", linestyles="--", linewidths=0.9)}
    for mode in ("comp", "full"):
        g = e2["grids"][mode]
        ax.contour(g["p2"], g["ln_gamma_b"], g["chi2"] - np.nanmin(g["chi2"]), levels=e2["levels_2d"], **styles[mode])
        ax.plot(g["phi"][2], g["phi"][3], "x" if mode == "comp" else "+", color=styles[mode]["colors"], ms=5, mew=1.0)
    t = e2["phi_true"]
    ax.plot(t[2], t[3], "*", color="black", ms=7, mfc="white", mew=0.8)
    ax.set_xlabel(r"$p_2$")
    ax.set_ylabel(r"$\ln\gamma_{\rm br}$")
    handles = [Line2D([], [], color=COLOR["E2"], lw=1.1, marker="x", ms=5, mew=1.0,
                      label=rf"compressed, $R_\star={e2['R']}$"),
               Line2D([], [], color="black", ls="--", lw=0.9, marker="+", ms=5, mew=1.0, label="full likelihood"),
               Line2D([], [], marker="*", ls="", color="black", mfc="white", ms=7, label="truth")]
    ax.legend(handles=handles, loc="upper left", frameon=False, handlelength=1.6, borderaxespad=0.1)
    panel(ax, r"(c) E2 parameters: 68\% and 95\% regions")


RIDGE_LEGEND = dict(loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.1)
RIDGE_LINE_HANDLE = dict(color=GREY, ls=":", lw=0.6, label=r"$\gamma_{\rm br}\propto B^{-1/2}$")


def _panel_ridge(ax2, ax1, D, data, letter="d"):
    """Ridge contours above and Delta chi2 along the ridge below (fig_moments_and_parameters)."""
    _ridge_contours(ax2, D, data, letter)
    _ridge_delta_chi2(ax1, data)
    handles = [Line2D([], [], color=COLOR["E2"], lw=1.1, label=r"support scales with $B$"),
               Line2D([], [], color=DIRECT, ls="--", lw=0.9, label="support fixed"),
               Line2D([], [], **RIDGE_LINE_HANDLE)]
    ax2.legend(handles=handles, **RIDGE_LEGEND)
    ax1.legend(handles=[Line2D([], [], color="0.3", lw=0.9, label="noise-free"),
                        Line2D([], [], marker="o", ls="", ms=2.6, mfc="white", mec="0.3", label="noisy data")],
               loc="center left", bbox_to_anchor=(0.0, 0.62), ncol=2, frameon=False, handlelength=1.6,
               borderaxespad=0.1, columnspacing=1.0)


def _ridge_contours(ax2, D, data, letter):
    """68% noise-free chi2 contours of the E2 ridge for the scaled and the fixed energy support."""
    rg = data["params"]["degeneracy"]["E2_ridge"]
    lnl = np.asarray(rg["ln_lambda_grid"])
    dl = np.asarray(rg["dln_gamma_b_grid_(absolute_offset_from_truth)"])
    lv = D["e2"]["levels_2d"][:1]
    xf, yf = np.linspace(lnl[0], lnl[-1], 161), np.linspace(dl[0], dl[-1], 161)
    styles = {"transformed": dict(colors=COLOR["E2"], linestyles="-", linewidths=1.1),
              "fixed": dict(colors=DIRECT, linestyles="--", linewidths=0.9)}
    for sup in ("transformed", "fixed"):
        grid = np.array([r[sup]["profile_grid_full_noisefree_chi2"] for r in rg["rows"]])
        best = min(np.min(grid), min(r[sup]["noisefree"]["full"]["chi2"] for r in rg["rows"]))
        spl = RectBivariateSpline(lnl, dl, grid - best, kx=3, ky=3)
        X, Y = np.meshgrid(xf, yf, indexing="ij")
        ax2.contour(X, Y, spl(xf, yf), levels=lv, **styles[sup])
    ax2.plot(lnl, -0.5 * lnl, color=GREY, lw=0.6, ls=":")
    ax2.set_ylim(-0.45, 0.85)
    ax2.yaxis.set_major_locator(FixedLocator([-0.4, 0.0, 0.4, 0.8]))
    ax2.set_ylabel(r"$\Delta\ln\gamma_{\rm br}$")
    ax2.tick_params(labelbottom=False)
    panel(ax2, (rf"({letter}) " if letter else "") + r"E2 field--energy ridge, 68\%")
    ax2.set_xlim(lnl[0] - 0.02, lnl[-1] + 0.02)


def _ridge_delta_chi2(ax1, data):
    """Delta chi2 along the ridge, noise-free (lines) and seeded (markers), N = 2."""
    rg = data["params"]["degeneracy"]["E2_ridge"]
    lnl = np.asarray(rg["ln_lambda_grid"])
    sm = rg["summary"]
    ax1.axhline(1.0, color=GREY, lw=0.6, ls=":")
    ax1.axhline(0.0, color=GREY, lw=fs.LW_REF)
    ax1.plot(lnl, sm["transformed/noisefree/full/delta_chi2_vs_lnlam"], color=COLOR["E2"], lw=1.1)
    ax1.plot(lnl, sm["transformed/seeded/full/delta_chi2_vs_lnlam"], "o", color=COLOR["E2"], ms=2.6, mfc="white", mew=0.8)
    ax1.plot(lnl, sm["fixed/noisefree/full/delta_chi2_vs_lnlam"], color=DIRECT, ls="--", lw=0.9)
    ax1.plot(lnl, sm["fixed/seeded/full/delta_chi2_vs_lnlam"], "s", color=DIRECT, ms=2.6, mfc="white", mew=0.7)
    ax1.set_ylim(-0.45, 1.3)
    ax1.set_xlabel(r"$\Delta\ln B$")
    ax1.set_ylabel(r"$\Delta\chi^2$ on ridge")
    ax1.set_xlim(lnl[0] - 0.02, lnl[-1] + 0.02)
    ax1.text(lnl[0], 1.05, r"$\Delta\chi^2=1$", va="bottom", ha="left", color=fs.INK)


RIDGE_SED_ORDERS = {"2": dict(lw_scale=1.0, alpha=1.0), "4": dict(lw_scale=0.75, alpha=0.5)}


def _ridge_sed(ax1, data):
    """Largest SED error of the ridge fits over all channels and I, Q, U, in per cent of I_j.

    Lines: noise-free fits at N = 2 (full weight) and N = 4 (lighter), labelled directly.
    Markers: fits to the seeded 1% noise draw at N = 2.  Line style gives the energy support,
    as in the contour panel above.
    """
    rs = data["ridge_sed"]
    x = np.asarray(rs["ln_lambda_grid"])
    sm = rs["summary"]
    lines = {"transformed": dict(color=COLOR["E2"], ls="-", lw=1.1),
             "fixed": dict(color=DIRECT, ls="--", lw=0.9)}
    marks = {"transformed": dict(marker="o", color=COLOR["E2"], mew=0.8), "fixed": dict(marker="s", color=DIRECT, mew=0.7)}
    for N, look in RIDGE_SED_ORDERS.items():
        for sup, st in lines.items():
            ax1.plot(x, 100 * np.asarray(sm[f"{sup}/N{N}/noisefree/sed_max"]), color=st["color"], ls=st["ls"],
                     lw=st["lw"] * look["lw_scale"], alpha=look["alpha"])
    for start, (sup, mk) in enumerate(marks.items()):  # alternate points: the two sets nearly coincide
        y = 100 * np.asarray(sm[f"{sup}/N2/seeded/sed_max"])
        ax1.plot(x[start::2], y[start::2], ls="", ms=2.6, mfc="white", **mk)
    ax1.axhline(100 * PAPER.noise_frac, color=GREY, lw=0.6, ls=":")
    ax1.set_yscale("log")
    ax1.set_ylim(0.006, 2.5)
    log_ticks(ax1, ticks=(0.01, 0.1, 1.0))
    i0 = list(x).index(0.0)
    for N, xl in (("2", -0.12), ("4", -0.3)):
        y0 = 100 * sm[f"transformed/N{N}/noisefree/sed_max"][i0]
        ax1.text(xl, y0 * 1.25, rf"$N={N}$", ha="center", va="bottom", color=fs.INK)
    ax1.text(x[0], 100 * PAPER.noise_frac * 1.12, "noise", va="bottom", ha="left", color=fs.INK)
    ax1.set_xlabel(r"$\Delta\ln B$")
    ax1.set_ylabel(r"SED error [\% of $I_j$]")
    ax1.set_xlim(x[0] - 0.02, x[-1] + 0.02)


def _moment_axes(fig, cell) -> list:
    """Three stacked subpanels (a_b0, a_b1, a_b2) sharing the energy axis."""
    sa = cell.subgridspec(3, 1, height_ratios=[2.4, 1, 1], hspace=0.1)
    axs = [fig.add_subplot(sa[i]) for i in range(3)]
    for ax in axs[1:]:
        ax.sharex(axs[0])
    return axs


def _ridge_axes(fig, cell) -> tuple:
    """Ridge contours above, Delta chi^2 along the ridge below, shared Delta ln B axis."""
    sd = cell.subgridspec(2, 1, height_ratios=[1.7, 1], hspace=0.1)
    ax2 = fig.add_subplot(sd[0])
    ax1 = fig.add_subplot(sd[1], sharex=ax2)
    return ax2, ax1


def fig_moments_and_parameters(D: dict, data: dict):
    fig = plt.figure(figsize=(6.1, 5.6))
    outer = fig.add_gridspec(2, 2, width_ratios=[1.12, 1], height_ratios=[1.25, 1], left=0.09,
                             right=0.985, top=0.955, bottom=0.075, hspace=0.36, wspace=0.3)
    _panel_moments(_moment_axes(fig, outer[0, 0]), D, data)
    _panel_contours(fig.add_subplot(outer[0, 1]), D)
    _panel_modes(fig.add_subplot(outer[1, 0]), D)
    _panel_ridge(*_ridge_axes(fig, outer[1, 1]), D, data)
    return fig


def fig_moments_and_parameters_compact(D: dict, data: dict):
    """Old panels (a) and (d) in one row, relabelled (a) and (b); same column widths as the full figure."""
    height = 3.0
    fig = plt.figure(figsize=(6.1, height))
    outer = fig.add_gridspec(1, 2, width_ratios=[1.12, 1], left=0.09, right=0.985, top=1 - 0.2 / height,
                             bottom=0.4 / height, wspace=0.3)
    _panel_moments(_moment_axes(fig, outer[0, 0]), D, data, letter="a")
    _panel_ridge(*_ridge_axes(fig, outer[0, 1]), D, data, letter="b")
    return fig


def fig_field_energy_ridge(D: dict, data: dict):
    """Ridge contours (N = 2) above and the SED error along the ridge (N = 2 and 4) below, at text width.

    The legend marks each energy support with the marker of its seeded fits in the panel below.
    """
    width, height = 6.1, 3.2
    fig = plt.figure(figsize=(width, height))
    outer = fig.add_gridspec(1, 1, left=0.6 / width, right=1 - 0.09 / width, top=1 - 0.2 / height,
                             bottom=0.4 / height)
    sd = outer[0, 0].subgridspec(2, 1, height_ratios=[1.3, 1], hspace=0.1)
    ax2 = fig.add_subplot(sd[0])
    ax1 = fig.add_subplot(sd[1], sharex=ax2)
    _ridge_contours(ax2, D, data, letter=None)
    handles = [Line2D([], [], color=COLOR["E2"], lw=1.1, marker="o", ms=2.6, mfc="white", mew=0.8,
                      label=r"support $\propto B^{-1/2}$"),
               Line2D([], [], color=DIRECT, ls="--", lw=0.9, marker="s", ms=2.6, mfc="white", mew=0.7,
                      label="support fixed"),
               Line2D([], [], **RIDGE_LINE_HANDLE)]
    ax2.legend(handles=handles, **RIDGE_LEGEND)
    _ridge_sed(ax1, data)
    return fig


# ----------------------------------------------------------------------------- GALPROP
# (colour, line style) per model; the broken power-law family takes the E2 colour (same form).
GALPROP_STYLE = {"moment_lo": ("#0072B2", "-"), "moment_hi": ("#009E73", "-"),
                 "logcubic": ("#CC79A7", "--"), "eSBPL": ("#E69F00", "-.")}
GALPROP_NAME = {"logcubic": "log-cubic law", "eSBPL": "broken PL electrons"}


def _galprop_entries(G: dict) -> list[tuple[str, str, str, str]]:
    """(model key, colour, line style, legend label with the withheld rms) in legend order."""
    out = []
    for slot, R in zip(("moment_lo", "moment_hi"), G["ranks"]):
        out.append((f"moment_R{R}", *GALPROP_STYLE[slot], rf"moment basis, $R={R}$"))
    for m in G["n_params"]:
        out.append((m, *GALPROP_STYLE[m], rf"{GALPROP_NAME[m]}, $k={G['n_params'][m]}$"))
    return [(k, c, ls, rf"{lab} ({100 * G['models'][k]['rms_held']:.3f}\%)") for k, c, ls, lab in out]


def _galprop_nodes(ax, nu, y, train, color, mfc_fit, ms=3.0, zorder=3):
    ax.plot(nu[train], y[train], "o", ms=ms, color=color, mfc=mfc_fit, mec="black" if mfc_fit != "white" else color,
            mew=0.5, ls="", zorder=zorder)
    ax.plot(nu[~train], y[~train], "s", ms=ms, mfc="white", mec=color, mew=0.8, ls="", zorder=zorder)


def fig_galprop_fit(D: dict, data: dict):
    G = D["galprop"]
    nu, nu_d, tr = G["nu"] / GHZ, G["nu_dense"] / GHZ, G["train"]
    fig = plt.figure(figsize=(6.1, 2.6))
    gs = fig.add_gridspec(1, 2, left=0.085, right=0.985, top=0.755, bottom=0.165, wspace=0.27)
    ax_a, ax_b = fig.add_subplot(gs[0]), fig.add_subplot(gs[1], sharex=None)
    entries = _galprop_entries(G)
    for key, c, ls, _ in entries:
        ax_a.plot(nu_d, G["models"][key]["I_dense"] / G["pl_dense"], color=c, ls=ls, lw=1.0, zorder=2)
    _galprop_nodes(ax_a, nu, G["I"] / G["pl"], tr, DIRECT, "black", ms=3.2)
    ax_a.set_ylabel(r"$I(\nu)/I_{\rm PL}(\nu)$")
    panel(ax_a, r"(a) GALPROP $|b|\geq60^\circ$, relative to a power law")

    ax_b.axhline(0, color="0.6", lw=0.5, zorder=0)
    for x in nu[~tr]:
        ax_b.axvline(x, color="0.9", lw=0.6, zorder=0)
    for key, c, ls, _ in entries:
        r = G["models"][key]["res_pct"]
        ax_b.plot(nu, r, color=c, ls=ls, lw=0.9, zorder=2)
        _galprop_nodes(ax_b, nu, r, tr, c, c, ms=2.6)
    lim = 1.12 * max(np.max(np.abs(G["models"][k]["res_pct"])) for k, *_ in entries)
    ax_b.set_ylim(-lim, lim)
    ax_b.set_ylabel(r"Residual [\%]")
    panel(ax_b, r"(b) $({\rm model}-{\rm GALPROP})/{\rm GALPROP}$")
    for ax in (ax_a, ax_b):
        ax.set_xscale("log")
        ax.set_xlim(nu[0] / 1.12, nu[-1] * 1.12)
        ax.xaxis.set_major_locator(FixedLocator([0.2, 0.5, 1.0, 2.0, 5.0]))
        ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xlabel("Frequency [GHz]")
    handles = [Line2D([], [], marker="o", ls="", ms=3.2, color=DIRECT, mec="black", label="GALPROP, fitted node"),
               Line2D([], [], marker="s", ls="", ms=3.2, mfc="white", mec=DIRECT, mew=0.8,
                      label="GALPROP, withheld node")]
    handles += [Line2D([], [], color=c, ls=ls, lw=1.0, label=lab) for _, c, ls, lab in entries]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,  # columns: data, moments, laws
               bbox_to_anchor=(0.53, 1.0), columnspacing=1.4, handlelength=2.2)
    return fig
