"""Drawing code for the classical-model figures (house style, final text width).

Fig. 8  reference_ensembles     electron spectra, channel intensities, intrinsic polarisation
Fig. 9  moment_fit_gallery      seeded UPRE fits of E1, E2, E4-KP, E5 (E4-KP reference: JP(t_eff) moments)
Fig. 10 moments_and_parameters  E2 moments, retained combinations, parameters, B-gamma ridge
        galprop_fit             noise-free fits to the GALPROP |b| >= 60 deg spectrum (external test)

Compact variants with the panels shown in Figs. 8-10 of the paper (same panel functions, fewer panels):
        reference_ensembles_compact     old panels (a) and (b) in one row
        moment_fit_gallery_compact      rows GALLERY_COMPACT (E2, E4-KP, E5)
        moments_and_parameters_compact  old panels (a) and (d) in one row, relabelled (a) and (b)

All arrays come from figure_data.build_all() and the loaded results; no number is typed here.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter
from scipy.interpolate import RectBivariateSpline

import ensembles as E
from config import PAPER

# Okabe-Ito colours (colour-blind safe), one per ensemble, used in every figure.
COLOR = {"E1": "#0072B2", "E2": "#E69F00", "E3": "#009E73", "E4-JP": "#CC79A7",
         "E4-KP": "#D55E00", "E5": "#56B4E9", "E1M": "#666666"}
STYLE = {"E1": "-", "E2": "-", "E3": "-", "E4-JP": "-", "E4-KP": "--", "E5": "-.", "E1M": ":"}
LABEL = {"E1": "E1 power law", "E2": "E2 broken power law", "E3": "E3 index mixture",
         "E4-JP": "E4-JP ageing", "E4-KP": "E4-KP ageing", "E5": "E5 two zones",
         "E1M": r"E1M Maxwellian $B$"}
FACET = {"E1": "E1 power law", "E2": "E2 broken PL", "E3": "E3 mixture", "E4-JP": "E4-JP ageing",
         "E4-KP": "E4-KP ageing", "E5": "E5 two zones"}
SHORT = {"E1": "E1", "E2": "E2", "E3": "E3", "E4-JP": "E4-JP", "E4-KP": "E4-KP", "E5": "E5", "E1M": "E1M"}
DIRECT, GREY = "black", "0.45"
PSEUDO_LS = (0, (5.0, 1.4, 1.2, 1.4, 1.2, 1.4))  # E4-KP reference curve: moments of the JP spectrum with t_eff
GHZ = 1e9


def ghz_axis(ax, label=True):
    ax.set_xscale("log")
    ax.set_xlim(0.09, 11.0)
    ax.xaxis.set_major_locator(FixedLocator([0.1, 1.0, 10.0]))
    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.xaxis.set_minor_formatter(NullFormatter())
    if label:
        ax.set_xlabel("Frequency [GHz]")


def log_ticks(ax, ticks=(0.2, 0.5, 1.0, 2.0)):
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
    ax.yaxis.set_minor_formatter(NullFormatter())


def panel(ax, text, x=0.0, y=1.03, ha="left"):
    ax.text(x, y, text, transform=ax.transAxes, ha=ha, va="bottom")


# ----------------------------------------------------------------------------- Fig. 8
REF_ORDER = ("E1", "E2", "E3", "E4-JP", "E4-KP", "E5")  # ensembles drawn in (a), (c); (b) adds E1M


def _panel_electron_spectra(ax_a, ax_p2, el):
    """(a) electron spectra relative to E1, with the E4-KP pitch-anisotropy subpanel below."""
    for k in REF_ORDER:
        y = np.where(el["rel_density"][k] > 0, el["rel_density"][k], np.nan)
        ax_a.plot(el["gamma"], y, color=COLOR[k], ls=STYLE[k], lw=1.1)
    ax_a.set(xscale="log", yscale="log", xlim=(300, 1e5), ylim=(0.12, 4.0))
    ax_a.set_ylabel(r"$\mathcal{N}(\gamma)/\mathcal{N}_{\rm E1}(\gamma)$")
    ax_a.tick_params(labelbottom=False)
    log_ticks(ax_a)
    panel(ax_a, r"(a) Electron spectra, unit column")
    ax_p2.plot(el["gamma_p2"], el["p2"], color=COLOR["E4-KP"], ls=STYLE["E4-KP"], lw=1.1)
    gc = el["gamma_c"]
    for key, ls in (("KP_mu0", ":"), ("KP_mu_half", ":")):
        for ax in (ax_a, ax_p2):
            ax.axvline(gc[key], color=GREY, lw=0.6, ls=ls, zorder=0)
    ax_p2.text(gc["KP_mu0"] / 1.07, 0.7, r"$\gamma_c^{\rm KP}(0),\,\gamma_c^{\rm KP}(\frac12)$",
               ha="right", va="center", color=GREY)
    ax_p2.set(xscale="log", xlim=(300, 1e5), ylim=(-0.05, 1.05))
    ax_p2.set_xlabel(r"Lorentz factor $\gamma$")
    ax_p2.set_ylabel(r"$\langle P_2(\mu)\rangle_{\rm ang}$")
    ax_p2.text(0.03, 0.62, "E4-KP pitch\nanisotropy", transform=ax_p2.transAxes, va="center")


def _panel_channel_intensity(ax_b, nu, fnpz):
    """(b) channel intensities relative to E1."""
    I1 = fnpz["E1"][:, 0]
    for k in REF_ORDER + ("E1M",):
        ax_b.plot(nu, fnpz[k.replace("-", "_")][:, 0] / I1, color=COLOR[k], ls=STYLE[k], lw=1.1)
    ghz_axis(ax_b)
    ax_b.set(yscale="log", ylim=(0.12, 4.0))
    log_ticks(ax_b)
    ax_b.set_ylabel(r"$I_j/I_j^{\rm E1}$")
    panel(ax_b, r"(b) Channel intensity, unit column")


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
    handles = [Line2D([], [], color=COLOR[k], ls=STYLE[k], lw=1.1, label=LABEL[k]) for k in REF_ORDER + ("E1M",)]
    fig.legend(handles=handles, loc="upper center", ncol=4, frameon=False, bbox_to_anchor=(0.53, 1.0),
               columnspacing=1.2, handlelength=2.2)


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
    """Panels (a) and (b) of fig_reference_ensembles in one row, same margins and column widths."""
    fnpz = data["forward_npz"]
    nu = fnpz["centres"] / GHZ
    height = 2.75
    fig = plt.figure(figsize=(6.1, height))
    outer = fig.add_gridspec(1, 2, left=0.095, right=0.985, top=1 - 0.66 / height, bottom=0.42 / height,
                             wspace=0.3)
    ax_a, ax_p2 = _electron_axes(fig, outer[0, 0])
    ax_b = fig.add_subplot(outer[0, 1])
    _panel_electron_spectra(ax_a, ax_p2, D["electrons"])
    _panel_channel_intensity(ax_b, nu, fnpz)
    _ensemble_legend(fig)
    return fig


# ----------------------------------------------------------------------------- Fig. 9
def _power_law_reference(nu, I):
    c = np.polyfit(np.log(nu), np.log(I), 1)
    return np.exp(np.polyval(c, np.log(nu)))


def _row_intensity(ax, nu, f, fit_i, held_i, color, ref_ls="--"):
    ref = _power_law_reference(nu, f["S"][:, 0])
    pred, sd = f["pred"][:, 0], f["sd_I"]
    ax.fill_between(nu, (pred - sd) / ref, (pred + sd) / ref, color=color, alpha=0.35, lw=0)
    ax.plot(nu, f["S"][:, 0] / ref, color=DIRECT, lw=1.0)
    ax.plot(nu, f["forward_true"][:, 0] / ref, color=GREY, ls=ref_ls, lw=0.9)
    ax.plot(nu, pred / ref, color=color, lw=0.8)
    sig = f["sigma"][:, 0] / ref
    ax.errorbar(nu[fit_i], f["data_fit"][:, 0] / ref[fit_i], sig[fit_i], fmt="o", ms=2.4,
                color=color, mec="black", mew=0.4, elinewidth=0.6, capsize=0)
    ax.errorbar(nu[held_i], f["data_held"][:, 0] / ref[held_i], sig[held_i], fmt="s", ms=2.4,
                mfc="white", mec=color, color=color, mew=0.7, elinewidth=0.6, capsize=0)


def _row_residual(ax, nu, f, color, ref_ls="--"):
    I = f["S"][:, 0]
    ax.axhspan(-1, 1, color="0.88", lw=0, zorder=0)
    ax.axhline(0, color="0.6", lw=0.5)
    r = 100 * (f["pred"] - f["S"]) / I[:, None]
    band = 100 * f["sd_I"] / I
    ax.fill_between(nu, r[:, 0] - band, r[:, 0] + band, color=color, alpha=0.35, lw=0)
    ax.plot(nu, 100 * (f["forward_true"][:, 0] - I) / I, color=GREY, ls=ref_ls, lw=0.9)
    ax.plot(nu, r[:, 0], color=color, lw=1.0)
    ax.plot(nu, r[:, 1], color="black", ls=(0, (3, 1.5)), lw=0.7)
    ax.plot(nu, r[:, 2], color="black", ls=(0, (1, 1.2)), lw=0.8)
    lim = max(1.6, 1.15 * np.max(np.abs(np.concatenate([r.ravel(), 100 * (f["forward_true"][:, 0] - I) / I]))))
    ax.set_ylim(-lim, lim)


def _row_polarisation(ax, nu, f, fit_i, held_i, color, ref_ls="--"):
    frac = lambda s: np.hypot(s[..., 1], s[..., 2]) / s[..., 0]  # noqa: E731
    pred = f["pred"]
    pfit, sd = frac(pred), f["sd_absP"] / pred[:, 0]
    ax.fill_between(nu, pfit - sd, pfit + sd, color=color, alpha=0.35, lw=0)
    ax.plot(nu, frac(f["S"]), color=DIRECT, lw=1.0)
    ax.plot(nu, frac(f["forward_true"]), color=GREY, ls=ref_ls, lw=0.9)
    ax.plot(nu, pfit, color=color, lw=0.8)
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
    ax.plot(nu[sel], joint[sel], color=color, lw=0.8, label=r"joint $(a,M^{(2)})$")
    ax.plot(nu[sel], free[sel], color="black", ls="-.", lw=0.9, label="one azimuth\nfactor")
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
    height = 1.48 * n + 0.88  # 6.8 in for four rows: fits the paper's text block with a 10-line caption
    fig, axes = plt.subplots(n, 3, figsize=(6.1, height), sharex=True,
                             gridspec_kw=dict(left=0.085, right=0.948, top=1 - 0.74 / height,
                                              bottom=0.44 / height, hspace=0.12, wspace=0.34))
    chi = data["fits"]["E5"]["chi_nu"]
    for row, name in enumerate(rows):
        f, c = bd["fits"][name], COLOR[name]
        ls = PSEUDO_LS if name == "E4-KP" else "--"
        ax_i, ax_r, ax_p = axes[row]
        _row_intensity(ax_i, nu, f, fit_i, held_i, c, ls)
        _row_residual(ax_r, nu, f, c, ls)
        if name == "E5":
            _row_angle(ax_p, nu, f, fit_i, held_i, chi, c)
            ax_p.set_ylabel(r"$\chi_j$ [deg]")
        else:
            _row_polarisation(ax_p, nu, f, fit_i, held_i, c, ls)
            ax_p.set_ylabel(r"$|P_j|/I_j$")
        ax_i.set_ylabel(r"$I_j/I_{\rm PL}$")
        ax_r.set_ylabel(r"Residual [\%]")
        ax_p.text(1.03, 0.5, f"{FACET[name]}\n$R_\\star={f['R_star']}$", transform=ax_p.transAxes,
                  rotation=-90, ha="left", va="center", ma="center", linespacing=1.1)
        for ax in axes[row]:
            ghz_axis(ax, label=(row == n - 1))
    for ax, t in zip(axes[0], (r"(a) Intensity", r"(b) $({\rm fit}-{\rm direct})/I_j$", r"(c) Polarisation")):
        panel(ax, t)
    fit_handle = (Patch(facecolor="0.55", alpha=0.5, lw=0), Line2D([], [], color="0.3", lw=0.8))
    handles = [Line2D([], [], color=DIRECT, lw=1.0, label="direct"),
               Line2D([], [], color=GREY, ls="--", lw=0.9, label="true moments, $N=2$"),
               Line2D([], [], color=GREY, ls=PSEUDO_LS, lw=0.9, label=r"E4-KP: JP($t_{\rm eff}$) moments, $N=2$"),
               fit_handle,
               Line2D([], [], marker="o", ls="", ms=3, color="0.4", mec="black", mew=0.4, label="data, fitted"),
               Line2D([], [], marker="s", ls="", ms=3, mfc="white", mec="0.3", mew=0.7, label="data, withheld"),
               Patch(facecolor="0.88", label=r"$\pm1\%$ noise"),
               Line2D([], [], color="black", ls=(0, (3, 1.5)), lw=0.7, label="$Q$ residual"),
               Line2D([], [], color="black", ls=(0, (1, 1.2)), lw=0.8, label="$U$ residual")]
    labels = [h.get_label() if not isinstance(h, tuple) else r"UPRE fit $\pm1\sigma$" for h in handles]
    order = [0, 3, 6, 1, 4, 7, 2, 5, 8]  # column-major fill: three rows of three
    fig.legend(handles=[handles[i] for i in order], labels=[labels[i] for i in order], loc="upper center",
               ncol=3, frameon=False, bbox_to_anchor=(0.53, 1.0), columnspacing=1.6, handlelength=2.4)
    return fig


# ----------------------------------------------------------------------------- Fig. 10
def _range(*arrays, pad=0.08):
    v = np.concatenate([np.ravel(a) for a in arrays])
    lo, hi = float(np.min(v)), float(np.max(v))
    d = pad * (hi - lo)
    return lo - d, hi + d


def _panel_moments(axs, D, data, letter="a"):
    e2 = D["e2"]
    edges, gb = PAPER.edges, PAPER.gamma_b
    nnls = np.asarray(data["fits"]["ensembles"]["E2"]["nnls_witness"]["seeded_implied_a_b0"])
    a_true, a_th = e2["a_true"].reshape(12, 3), e2["a_theta"]["comp"].reshape(12, 3)
    a_ts = e2["a_tsvd"].reshape(12, 3)
    for r, ax in enumerate(axs):
        lo, hi = _range(a_true[:, r], a_th[:, r], a_ts[:, r], [0.0])
        if r == 0:
            hi = lo + (hi - lo) / 0.6  # two legend rows above the off-scale markers
        ax.stairs(a_true[:, r], edges, color=DIRECT, lw=1.1, baseline=None)
        ax.stairs(a_th[:, r], edges, color=COLOR["E2"], lw=1.1, ls="--", baseline=None)
        ax.axhline(0, color="0.7", lw=0.5, zorder=0)
        ax.plot(gb, a_ts[:, r], "o", ms=3, mfc="white", mec="black", mew=0.8)
        ax.set(xscale="log", xlim=(edges[0], edges[-1]), ylim=(lo, hi))
        ax.set_ylabel([r"$a_{b0}$", r"$a_{b1}$", r"$a_{b2}$"][r])
    top = _range(a_true[:, 0], a_th[:, 0], a_ts[:, 0], [0.0])[1]
    ins = nnls <= top
    axs[0].plot(gb[ins], nnls[ins], "D", ms=3, color=COLOR["E3"], mec="black", mew=0.4)
    for g, v in zip(gb[~ins], nnls[~ins]):  # off scale: value written beside an upward triangle
        axs[0].plot(g, top, "^", ms=4, color=COLOR["E3"], mec="black", mew=0.4)
        right = g * 1.12 * 1.35 < edges[-1]
        axs[0].text(g * 1.12 if right else g / 1.12, top, f"{v:.0f}", va="center", ha="left" if right else "right")
    for ax in axs[:-1]:
        ax.tick_params(labelbottom=False)
    axs[-1].set_xlabel(r"Lorentz factor $\gamma$")
    handles = [Line2D([], [], color=DIRECT, lw=1.1, label="true"),
               Line2D([], [], color=COLOR["E2"], lw=1.1, ls="--", label=r"$a(\hat\theta)$"),
               Line2D([], [], marker="o", ls="", ms=3, mfc="white", mec="black", label=r"min.-norm, $R_\star$"),
               Line2D([], [], marker="D", ls="", ms=3, color=COLOR["E3"], mec="black", mew=0.4,
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


def _panel_ridge(ax2, ax1, D, data, letter="d"):
    rg = data["params"]["degeneracy"]["E2_ridge"]
    lnl = np.asarray(rg["ln_lambda_grid"])
    dl = np.asarray(rg["dln_gamma_b_grid_(absolute_offset_from_truth)"])
    lv = D["e2"]["levels_2d"][:1]
    xf, yf = np.linspace(lnl[0], lnl[-1], 161), np.linspace(dl[0], dl[-1], 161)
    styles = {"transformed": dict(colors=COLOR["E2"], linestyles="-", linewidths=1.1),
              "fixed": dict(colors="black", linestyles="--", linewidths=0.9)}
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
    panel(ax2, rf"({letter}) E2 field--energy ridge, 68\%")
    sm = rg["summary"]
    ax1.axhline(1.0, color=GREY, lw=0.6, ls=":")
    ax1.axhline(0.0, color="0.7", lw=0.5)
    ax1.plot(lnl, sm["transformed/noisefree/full/delta_chi2_vs_lnlam"], color=COLOR["E2"], lw=1.1)
    ax1.plot(lnl, sm["transformed/seeded/full/delta_chi2_vs_lnlam"], "o", color=COLOR["E2"], ms=2.6, mfc="white", mew=0.8)
    ax1.plot(lnl, sm["fixed/noisefree/full/delta_chi2_vs_lnlam"], color="black", ls="--", lw=0.9)
    ax1.plot(lnl, sm["fixed/seeded/full/delta_chi2_vs_lnlam"], "o", color="black", ms=2.6, mfc="white", mew=0.7)
    ax1.set_ylim(-0.45, 1.3)
    ax1.set_xlabel(r"$\Delta\ln B$")
    ax1.set_ylabel(r"$\Delta\chi^2$ on ridge")
    for ax in (ax1, ax2):
        ax.set_xlim(lnl[0] - 0.02, lnl[-1] + 0.02)
    handles = [Line2D([], [], color=COLOR["E2"], lw=1.1, label=r"support scales with $B$"),
               Line2D([], [], color="black", ls="--", lw=0.9, label="support fixed"),
               Line2D([], [], color=GREY, ls=":", lw=0.6, label=r"$\gamma_{\rm br}\propto B^{-1/2}$")]
    ax2.legend(handles=handles, loc="upper right", frameon=False, handlelength=1.6, borderaxespad=0.1)
    ax1.legend(handles=[Line2D([], [], color="0.3", lw=0.9, label="noise-free"),
                        Line2D([], [], marker="o", ls="", ms=2.6, mfc="white", mec="0.3", label="noisy data")],
               loc="center left", bbox_to_anchor=(0.0, 0.62), ncol=2, frameon=False, handlelength=1.6,
               borderaxespad=0.1, columnspacing=1.0)


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
