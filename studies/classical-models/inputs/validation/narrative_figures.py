"""Message-led manuscript figures; saved comparisons plus a harmonic illustration.

Run: python -m validation.narrative_figures
The moving-line illustration has a complete finite harmonic sum on its support.
The other numerical panels replot saved model-specific results, not new fits.
"""

from pathlib import Path
import hashlib
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
from scipy.special import jv, jvp

from . import full_response as harmonic

from .figure_style import (
    BLUE,
    GREY,
    LIGHT,
    LW,
    LW_REF,
    PURPLE,
    SERIES,
    VERMILION,
    WIDTH,
    configure,
    save,
)

ROOT = Path(__file__).resolve().parents[1]

configure()


def overview():
    fig, ax = plt.subplots(figsize=(8.3, 2.8))
    ax.set(xlim=(0, 10), ylim=(0, 3))
    ax.axis("off")

    def box(x, y, width, title, body, color):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                1.03,
                boxstyle="round,pad=.04,rounding_size=.07",
                facecolor=color,
                edgecolor="#A1ADB2",
                lw=0.8,
            )
        )
        ax.text(
            x + width / 2,
            y + 0.77,
            title,
            ha="center",
            va="center",
            fontsize=11,
            fontweight="bold",
        )
        ax.text(
            x + width / 2,
            y + 0.34,
            body,
            ha="center",
            va="center",
            fontsize=10,
            linespacing=1.4,
        )

    box(
        0.06,
        1.65,
        2.95,
        "One joint population",
        "Electron + field properties\nand Faraday depth",
        "#EDF3F6",
    )
    box(
        0.06,
        0.2,
        2.95,
        "Radiation + channels",
        "Integrate the harmonic lines\nover each frequency channel",
        "#F8F0E8",
    )
    box(
        3.52,
        1.65,
        2.7,
        "Joint statistics",
        "Angular weights\nand physical moments",
        "#EDF3F6",
    )
    box(
        3.52,
        0.2,
        2.7,
        "Response basis",
        "Project the angles; expand\nthe remaining parameters",
        "#F8F0E8",
    )
    box(
        7,
        0.92,
        2.9,
        "Stokes predictions",
        "Same statistics, all channels\n" + r"$\sum$ moment $\times$ response",
        "#EDF4EE",
    )
    for xy, end in [
        ((3.07, 2.17), (3.46, 2.17)),
        ((3.07, 0.72), (3.46, 0.72)),
        ((6.28, 2.17), (6.94, 1.62)),
        ((6.28, 0.72), (6.94, 1.22)),
    ]:
        ax.annotate(
            "",
            xy=end,
            xytext=xy,
            arrowprops={"arrowstyle": "->", "lw": 1.5, "color": "#536775"},
        )
    fig.tight_layout(pad=0.3)
    save(fig, "population_response_overview")


def moving_lines(ax=None):
    gamma, mu, eta = 20.0, 0.5, 0.5
    beta = np.sqrt(1 - gamma**-2)
    d = 1 - beta * mu * eta
    # The only illustrated channel is the centre-4 channel of full_response.py.
    # y_m=m*b/D, y_max=6.6, b>=0.8, so m>=7 never enters its support.
    m = np.arange(1, 8)
    x = m * beta * np.sqrt(1 - mu * mu) * np.sqrt(1 - eta * eta) / d
    parallel = (eta - beta * mu) / np.sqrt(1 - eta * eta) * jv(m, x)
    perpendicular = beta * np.sqrt(1 - mu * mu) * jvp(m, x)
    power0 = m * m / (gamma * gamma * d**3) * (parallel**2 + perpendicular**2)

    def channel(b, frozen=False):
        frequencies = m / d if frozen else m * b / d
        return float(np.sum(b * b * power0 * harmonic.response(frequencies)[1]))

    bvals = np.linspace(0.8, 1.2, 161)
    dynamic = np.array([channel(b) for b in bvals])
    frozen = np.array([channel(b, True) for b in bvals])
    # Independent vectorised harmonic implementation, including more harmonics.
    oracle = np.array(
        [
            harmonic.direct_kernel(
                np.array([0.0, b - 1, 0.0]), np.array(mu), np.array(eta), 40
            )[0, 1].real
            for b in bvals
        ]
    )
    np.testing.assert_allclose(dynamic, oracle, rtol=3e-13, atol=1e-16)
    assert 7 * 0.8 / d > 6.6
    norm = channel(1.0)
    np.testing.assert_allclose(frozen / norm, bvals**2, rtol=1e-14)
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(WIDTH, 2.15), layout="constrained")
    y = np.linspace(0.2, 8, 801)
    response = harmonic.response(y)[1]
    ax.fill_between(y, response, color=LIGHT, lw=0)
    ax.plot(y, response, color=GREY, lw=LW_REF)
    # B/B0 = 1 and 1.2 are the first and second members of an ordered family.
    for b, color, marker in [(1.0, SERIES[0], "o"), (1.2, SERIES[1], "s")]:
        loc, heights = m * b / d, b * b * power0 / power0[m / d < 8].max()
        visible = loc < 8
        ax.vlines(loc[visible], 0, heights[visible], colors=color, lw=LW)
        ax.plot(
            loc[visible],
            heights[visible],
            marker,
            color=color,
            ls="none",
            label=rf"$B/B_0={b:g}$",
        )
    ax.set(
        xlim=(0.2, 8.3),
        ylim=(0, 1.7),
        xlabel=r"Frequency $\nu/\nu_*$",
        ylabel="Line power / reference peak",
    )
    ax.legend(loc="upper left")
    ax.text(4.0, 1.05, "Receiving band", ha="center", color=GREY)
    if standalone:
        save(fig, "moving_harmonic_channels")
    return {
        "gamma": gamma,
        "mu": mu,
        "eta": eta,
        "harmonics_summed": 7,
        "channel_centre": 4.0,
        "half_width": 2.6,
        "field_range": [0.8, 1.2],
        "oracle_max_absolute_difference": float(np.max(np.abs(dynamic - oracle))),
        "power_normalisation": "e^2 Omega_0^2/(2 pi c), divided out in dynamic and oracle channel checks",
        "plotted_line_normalisation": "Largest displayed line at B/B0=1",
    }


def results(hdata, cdata):
    checked = 0
    harmonic_errors = []
    for degree in [2, 4, 8]:
        rows = sorted(
            [r for r in hdata["rows"] if r["width_scale"] == 1 and r["L_mu"] == degree],
            key=lambda r: r["N"],
        )
        assert [r["N"] for r in rows] == [0, 1, 2]
        errors = []
        for row in rows:
            direct, finite = np.array(row["direct_IQUV"]), np.array(row["finite_IQUV"])
            err = np.max(np.abs(finite - direct) / direct[0])
            np.testing.assert_allclose(
                err, row["max_abs_stokes_error_over_channel_I"], rtol=1e-12
            )
            errors.append(100 * err)
            checked += 1
        harmonic_errors.append(errors)
    continuum_errors = {}
    for width in [0.5, 1.0]:
        errs = []
        for row in [r for r in cdata["rows"] if r["width"] == width]:
            direct = np.array(row["direct"])
            arr = [
                float(np.max(np.abs(np.array(p) - direct) / direct[:, :1]))
                for p in row["predictions"]
            ]
            np.testing.assert_allclose(arr, row["errors_over_channel_I"], rtol=1e-12)
            errs.append(arr)
            checked += 1
        continuum_errors[str(width)] = (100 * np.max(errs, axis=0)).tolist()
    # Figure 6: diagnose two approximation steps before showing their total.
    # Each diagnostic is a maximum norm, not an additive error budget.
    degrees = np.array([2, 4, 8])
    diagnostic_rows = [
        next(
            r
            for r in hdata["rows"]
            if r["width_scale"] == 1 and r["L_mu"] == degree and r["N"] == 2
        )
        for degree in degrees
    ]
    angular_error = 100 * np.array(
        [r["angular_only_error_over_channel_I"] for r in diagnostic_rows]
    )
    parameter_error = 100 * np.array(
        [r["taylor_vs_angular_only_over_channel_I"] for r in diagnostic_rows]
    )
    total = np.array(harmonic_errors)
    # The norm triangle inequalities must hold; the norms need not add.
    assert np.all(total[:, 2] <= angular_error + parameter_error + 1e-10)
    assert np.all(total[:, 2] >= np.abs(angular_error - parameter_error) - 1e-10)
    fig = plt.figure(figsize=(6.1, 3.45))
    axes = [fig.add_axes((x, 0.27, 0.355, 0.57)) for x in (0.12, 0.62)]
    axes[0].plot(
        degrees,
        angular_error,
        "o-",
        color=VERMILION,
        label="Angular truncation",
        ms=4,
    )
    axes[0].plot(
        degrees,
        parameter_error,
        "s-",
        color=PURPLE,
        label=r"Quadratic expansion ($N=2$)",
        ms=4,
    )
    axes[1].plot(
        degrees,
        total[:, 0],
        "o-",
        color="#777777",
        mfc="white",
        label=r"Zeroth-order response ($N=0$)",
        ms=4,
    )
    axes[1].plot(
        degrees,
        total[:, 2],
        "o-",
        color=BLUE,
        label=r"Quadratic response ($N=2$)",
        ms=4,
    )
    for ax, title in zip(
        axes, ["(a) Error introduced at each step", "(b) Final prediction error"]
    ):
        ax.set(
            xscale="linear",
            yscale="log",
            xlim=(1.5, 8.5),
            xticks=degrees,
            xlabel=r"Angular expansion order $L$",
            title=title,
        )
        ax.minorticks_off()
        ax.grid(axis="y", alpha=0.15)
        ax.legend(
            loc="upper left",
            bbox_to_anchor=(-0.03, -0.24),
            frameon=False,
            handlelength=1.5,
            borderaxespad=0,
            labelspacing=0.4,
        )
    axes[0].set(
        ylim=(1e-4, 20),
        yticks=[1e-4, 0.001, 0.01, 0.1, 1, 10],
        yticklabels=["0.0001", "0.001", "0.01", "0.1", "1", "10"],
        ylabel=r"Step error [\% of intensity]",
    )
    axes[1].set(
        ylim=(0.08, 12),
        yticks=[0.1, 1, 10],
        yticklabels=["0.1", "1", "10"],
        ylabel=r"Total error [\% of intensity]",
    )
    axes[1].annotate(
        r"$0.175\%$",
        (8, total[-1, 2]),
        xytext=(-5, 9),
        textcoords="offset points",
        ha="right",
        color=BLUE,
    )
    save(fig, "population_response_results")

    # Compact main-text comparison; the harmonic diagnostic above is in the appendix.
    fig = plt.figure(figsize=(6.1, 2.85))
    fig.text(
        0.53,
        0.97,
        "Same marginals; reversed energy--field and depth--field correlations",
        ha="center",
        va="top",
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#292929",
                marker="o",
                mfc="none",
                ls="none",
                label="Direct population integrals",
            ),
            Line2D(
                [],
                [],
                color=BLUE,
                marker="x",
                ls="none",
                label=r"Quadratic moments ($N=2$)",
            ),
        ],
        loc="lower center",
        bbox_to_anchor=(0.53, 0.01),
        ncol=2,
        frameon=False,
    )
    minus = next(r for r in cdata["rows"] if r["width"] == 1 and r["correlation"] < 0)
    plus = next(r for r in cdata["rows"] if r["width"] == 1 and r["correlation"] > 0)
    norm = np.array(plus["direct"])[:, :1]
    direct_difference = (
        100 * (np.array(plus["direct"]) - np.array(minus["direct"])) / norm
    )
    finite_difference = (
        100
        * (np.array(plus["predictions"][2]) - np.array(minus["predictions"][2]))
        / norm
    )
    freq = np.array(cdata["configuration"]["frequency_centres_Hz"]) / 1e9
    halfwidth = cdata["configuration"]["fractional_channel_halfwidth"] * freq
    residual = finite_difference - direct_difference
    limits = [(-0.48, 0.035), (-1.55, 0.8), (-0.045, 0.34)]
    for k, (xpos, title) in enumerate(
        zip(
            [0.11, 0.425, 0.74],
            [r"(a) Intensity $I$", r"(b) Stokes $Q$", r"(c) Stokes $U$"],
        )
    ):
        ax = fig.add_axes((xpos, 0.28, 0.235, 0.49))
        ax.axhline(0, color="#999999", lw=0.6)
        ax.hlines(
            direct_difference[:, k],
            freq - halfwidth,
            freq + halfwidth,
            color="#999999",
            lw=0.8,
            zorder=1,
        )
        ax.vlines(
            freq,
            direct_difference[:, k],
            finite_difference[:, k],
            color="#999999",
            lw=0.9,
            zorder=1,
        )
        ax.plot(
            freq,
            direct_difference[:, k],
            "o",
            mfc="none",
            color="#292929",
            ms=5,
            mew=1.2,
        )
        ax.plot(freq, finite_difference[:, k], "x", color=BLUE, ms=5, mew=1.2)
        ax.set(
            ylim=limits[k],
            title=title,
            xscale="log",
            xlim=(0.22, 1.55),
            xticks=freq,
            xticklabels=["0.3", "0.6", "1.2"],
            xlabel="Band centre [GHz]",
        )
        ax.minorticks_off()
        if k == 0:
            ax.set_ylabel(r"$100\,\Delta S/I_b$")
        if k == 2:
            relative_error = abs(residual[0, k] / direct_difference[0, k])
            ax.annotate(
                "Prediction error:\n" + rf"{100 * relative_error:.0f}\% of signal",
                xy=(freq[0], (direct_difference[0, k] + finite_difference[0, k]) / 2),
                xytext=(0.46, 0.13),
                ha="left",
                va="center",
                color=BLUE,
                arrowprops={"arrowstyle": "->", "color": BLUE, "lw": 0.8},
                bbox={
                    "facecolor": "white",
                    "edgecolor": "none",
                    "alpha": 0.85,
                    "pad": 0.6,
                },
            )
    contrast = float(np.max(np.abs(direct_difference)))
    recorded = next(r for r in cdata["correlation_contrasts"] if r["width"] == 1)
    np.testing.assert_allclose(
        contrast, 100 * recorded["direct_difference_over_plus_I"], rtol=1e-12
    )
    save(fig, "correlation_response_results")
    return {
        "saved_rows_checked": checked,
        "continuum_max_discrepancy_percent": continuum_errors,
        "continuum_correlation_contrast_percent": contrast,
        "scope": "Finite forward comparisons within the specified emission models",
    }


def main():
    paths = [
        ROOT / "validation/full_response_results.json",
        ROOT / "validation/continuum_population_results.json",
    ]
    hdata, cdata = [json.loads(p.read_text()) for p in paths]
    if hdata["configuration"]["quick"]:
        raise ValueError("Require full saved results")
    overview()
    record = {
        "moving_lines": moving_lines(),
        "results": results(hdata, cdata),
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "style_sha256": hashlib.sha256(
            (ROOT / "validation/figure_style.py").read_bytes()
        ).hexdigest(),
        "source_sha256": {
            p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in paths
        },
    }
    (ROOT / "validation/narrative_figures_results.json").write_text(
        json.dumps(record, indent=2) + "\n"
    )
    print(json.dumps(record, indent=2))


if __name__ == "__main__":
    main()
