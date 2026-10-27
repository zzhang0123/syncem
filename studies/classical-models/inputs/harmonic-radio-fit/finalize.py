"""Fit authenticated radio harmonic data, propagate discrepancies, and plot."""

from __future__ import annotations

import json
import os
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", str(Path(__file__).resolve().parent / ".mplcache"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from cache_io import file_sha, load
import radio_numerics as n

ROOT = Path(__file__).resolve().parent
SEED = 20260928
NOISE = 0.01
THRESHOLD = 10.0


def solve(c, data, sigma, use):
    design = c[use].reshape(-1, 36) / sigma.reshape(-1, 1)
    u, s, vt = np.linalg.svd(design, full_matrices=False)
    keep = s >= THRESHOLD
    estimator = (vt[keep].T / s[keep]) @ u[:, keep].T
    estimate = estimator @ (data / sigma).ravel()
    return {
        "estimate": estimate,
        "prediction": np.einsum("jsk,k->js", c, estimate),
        "singular_values": s,
        "modes": vt[keep],
        "left": u[:, keep],
        "mode_sigma": 1 / s[keep],
        "estimator": estimator,
        "design": design,
    }


def main():
    specifications = {
        "basis": (
            "trusted_basis_refined.npz",
            "basis",
            dict(field=8, eta=7, pitch=18, channel=64, step=0.04),
        ),
        "truth": (
            "trusted_truth_refined.npz",
            "truth",
            dict(field=8, eta=7, pitch=18, channel=64, energy=8),
        ),
        "basis_coarse": (
            "trusted_basis.npz",
            "basis",
            dict(field=6, eta=5, pitch=14, channel=48, step=0.04),
        ),
        "truth_coarse": (
            "trusted_truth.npz",
            "truth",
            dict(field=6, eta=5, pitch=14, channel=48, energy=6),
        ),
        "halfstep": (
            "trusted_basis_halfstep.npz",
            "basis",
            dict(field=6, eta=5, pitch=14, channel=48, step=0.02),
        ),
    }
    arrays, records = {}, {}
    for key, (filename, kind, settings) in specifications.items():
        arrays[key], records[key] = load(ROOT / filename, kind, settings)
    response = arrays["basis"]["basis"]
    truth, moments = arrays["truth"]["truth"], arrays["truth"]["moments"]
    use = np.arange(0, len(n.CENTRES), 2)
    sigma = NOISE * truth[use, 0, None] * np.ones((1, 3))
    data = truth[use] + np.random.default_rng(SEED).normal(size=sigma.shape) * sigma
    fit = solve(response, data, sigma, use)
    prediction, modes = fit["prediction"], fit["modes"]
    true_modes = modes @ moments
    estimated_modes = modes @ fit["estimate"]
    finite = np.einsum("jsk,k->js", response, moments)
    discrepancy = truth - finite
    noise_mode_bias = fit["left"].T @ (discrepancy[use] / sigma).ravel()
    covariance = fit["estimator"] @ fit["estimator"].T
    fit_sigma = np.sqrt(np.einsum("jsk,kl,jsl->js", response, covariance, response))
    variations = {
        "truth_quadrature": arrays["truth_coarse"]["truth"] - truth,
        "response_quadrature_at_true_moments": np.einsum(
            "jsk,k->js", arrays["basis_coarse"]["basis"] - response, moments
        ),
        "response_halfstep_at_true_moments": np.einsum(
            "jsk,k->js",
            arrays["halfstep"]["basis"] - arrays["basis_coarse"]["basis"],
            moments,
        ),
    }
    checks = {}
    for key, delta in variations.items():
        bias = fit["left"].T @ (delta[use] / sigma).ravel()
        checks[key] = {
            "max_over_local_I": float(np.max(abs(delta) / truth[:, 0, None])),
            "retained_mode_bias_over_sigma": bias.tolist(),
            "max_retained_mode_bias_over_sigma": float(np.max(abs(bias))),
        }
    alternative_fits = {
        key: solve(arrays[key]["basis"], data, sigma, use)
        for key in ("basis_coarse", "halfstep")
    }
    for key, other in alternative_fits.items():
        checks[f"refit_{key}"] = {
            "retained_count": len(other["modes"]),
            "prediction_change_max_over_I": float(
                np.max(abs(other["prediction"] - prediction) / truth[:, 0, None])
            ),
            "adopted_mode_shift_over_sigma": (
                (modes @ (other["estimate"] - fit["estimate"])) / fit["mode_sigma"]
            ).tolist(),
            "retained_projector_difference_spectral_norm": float(
                np.linalg.norm(other["modes"].T @ other["modes"] - modes.T @ modes, 2)
            ),
        }
    moment_array = moments.reshape(12, 3)
    true_physical_margin = {
        "min_mass": float(np.min(moment_array[:, 0])),
        "min_mass_minus_abs_first": float(
            np.min(moment_array[:, 0] - abs(moment_array[:, 1]))
        ),
        "min_mass_times_second_minus_first_squared": float(
            np.min(moment_array[:, 0] * moment_array[:, 2] - moment_array[:, 1] ** 2)
        ),
        "min_second": float(np.min(moment_array[:, 2])),
        "min_mass_minus_second": float(np.min(moment_array[:, 0] - moment_array[:, 2])),
        "total_column": float(np.dot(n.REF_BIN_MASS, moment_array[:, 0])),
    }
    checks["SVD_identities"] = {
        "projector_spectral_norm_error": float(
            np.linalg.norm(fit["estimator"] @ fit["design"] - modes.T @ modes, 2)
        ),
        "mode_bias_identity_max_absolute": float(
            np.max(
                abs(
                    modes @ fit["estimator"] @ (truth[use] / sigma).ravel()
                    - true_modes
                    - noise_mode_bias * fit["mode_sigma"]
                )
            )
        ),
    }
    if max(checks["SVD_identities"].values()) > 1e-10:
        raise ValueError(
            "retained projection or discrepancy propagation identity fails"
        )
    if any(value < -1e-12 for value in true_physical_margin.values()):
        raise ValueError("true energy moments violate compact-support constraints")
    if not np.all(np.diff(truth[:, 0]) < 0):
        raise ValueError("the radio reference intensity is not decreasing")
    if (
        max(checks[key]["max_retained_mode_bias_over_sigma"] for key in variations)
        > 0.1
    ):
        raise ValueError("numerical sensitivity exceeds 0.1 mode standard deviation")
    for key in alternative_fits:
        refit_check = checks[f"refit_{key}"]
        if np.max(abs(np.asarray(refit_check["adopted_mode_shift_over_sigma"]))) > 0.1:
            raise ValueError(
                "refitted retained-mode shift exceeds 0.1 standard deviation"
            )
        if refit_check["retained_projector_difference_spectral_norm"] > 1e-3:
            raise ValueError("refitted retained projector changes by more than 1e-3")
    if max(len(other["modes"]) for other in alternative_fits.values()) != len(
        modes
    ) or min(len(other["modes"]) for other in alternative_fits.values()) != len(modes):
        raise ValueError("retained count changes under numerical refinement")
    if np.max(abs(noise_mode_bias)) > 0.5:
        raise ValueError("representation bias exceeds 0.5 mode standard deviation")
    scale = float(np.max(truth[:, 0]))
    report = {
        "configuration": n.model()
        | {
            "noise_fraction_of_I": NOISE,
            "seed": SEED,
            "SVD_threshold": THRESHOLD,
            "truth_column": 1.0,
        },
        "fit": {
            "number_of_channels": len(use),
            "number_of_data": data.size,
            "number_of_coefficients": 36,
            "retained_count": len(modes),
            "chi2": float(np.sum(((data - prediction[use]) / sigma) ** 2)),
            "residual_dof": int(data.size - len(modes)),
            "singular_values": fit["singular_values"].tolist(),
            "finite_representation_max_over_I": float(
                np.max(abs(discrepancy) / truth[:, 0, None])
            ),
            "fit_max_over_I": float(
                np.max(abs(prediction - truth) / truth[:, 0, None])
            ),
            "heldout_max_over_I": float(
                np.max(abs(prediction[1::2] - truth[1::2]) / truth[1::2, 0, None])
            ),
            "representation_mode_bias_over_sigma": noise_mode_bias.tolist(),
            "representation_max_mode_bias_over_sigma": float(
                np.max(abs(noise_mode_bias))
            ),
            "true_modes": true_modes.tolist(),
            "estimated_modes": estimated_modes.tolist(),
            "mode_sigma": fit["mode_sigma"].tolist(),
            "retained_combinations": modes.tolist(),
            "estimated_moments": fit["estimate"].tolist(),
            "representative_is_not_constrained_to_be_physical": True,
        },
        "checks": checks,
        "truth_energy_moment_constraints": true_physical_margin,
        "radiation": {
            "intensity_strictly_decreasing": True,
            "I_Hz_endpoints": truth[[0, -1], 0].tolist(),
            "effective_spectral_slope": float(
                np.log(truth[-1, 0] / truth[0, 0])
                / np.log(n.CENTRES[-1] / n.CENTRES[0])
            ),
            "omitted_line_envelope_quadrature_max_over_I": float(
                np.max(arrays["truth"]["omitted_tail"] / truth[:, 0])
            ),
            "V_zero_by_simultaneous_mu_eta_reversal": True,
        },
        "acceptance_gate": {
            "passed": True,
            "max_numerical_mode_shift_sigma": 0.1,
            "max_refitted_adopted_mode_shift_sigma": 0.1,
            "max_retained_projector_difference_spectral_norm": 1e-3,
            "max_representation_mode_shift_sigma": 0.5,
            "meaning": "Finite numerical and representation sensitivity thresholds for this declared example, not global error bounds.",
        },
        "scope": "Recovery is conditional on known independent B, pitch, field-direction, azimuth and Faraday distributions. Only source-scaled piecewise energy moments are fitted. Retained Airy-line and accelerated harmonic-summation errors lack an ensemble-wide rigorous bound. Numerical refinements do not replace those missing bounds.",
        "arrays": {
            "centres_Hz": n.CENTRES.tolist(),
            "response_matrix": response.tolist(),
            "truth_IQU": truth.tolist(),
            "true_moments": moments.tolist(),
            "finite_at_true_moments_IQU": finite.tolist(),
            "fitted_IQU": prediction.tolist(),
            "fitted_mean_sigma_IQU": fit_sigma.tolist(),
            "fit_channel_indices": use.tolist(),
            "noisy_data_IQU": data.tolist(),
            "noise_sigma_IQU": sigma.tolist(),
        },
        "provenance": {
            "input_cache_records": records,
            "input_cache_sha256": {
                filename: file_sha(ROOT / filename)
                for filename, _, _ in specifications.values()
            },
            "finalizer_sha256": file_sha(__file__),
        },
    }
    plot_sed(truth, prediction, fit_sigma, data, sigma, use, scale)
    plot_modes(true_modes, estimated_modes, fit["mode_sigma"])
    report["derivative_figure"] = plot_derivatives(arrays, records)
    report["output_sha256"] = {
        path.name: file_sha(path)
        for path in sorted(ROOT.glob("radio_*.pdf")) + sorted(ROOT.glob("radio_*.png"))
    }
    (ROOT / "results_final.json").write_text(json.dumps(report, indent=2) + "\n")
    print(
        json.dumps(
            {
                "fit": report["fit"]
                | {
                    "retained_combinations": "saved in report",
                    "estimated_moments": "saved in report",
                },
                "checks": checks,
                "radiation": report["radiation"],
            },
            indent=2,
        )
    )


def style():
    plt.rcParams.update(
        {
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelsize": 10,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "pdf.fonttype": 42,
        }
    )


def save(fig, name):
    for suffix in ("pdf", "png"):
        fig.savefig(ROOT / f"{name}.{suffix}", dpi=240)
    plt.close(fig)


def plot_sed(truth, prediction, errors, data, sigma, use, scale):
    style()
    fig, axes = plt.subplots(
        1, 3, figsize=(8.8, 3.1), sharex=True, layout="constrained"
    )
    frequency = n.CENTRES / 1e9
    for k, (ax, label) in enumerate(zip(axes, ("I", "Q", "U"))):
        ax.plot(
            frequency, truth[:, k] / scale, color="0.15", lw=1.7, label="Full ensemble"
        )
        ax.plot(
            frequency,
            prediction[:, k] / scale,
            color="#D55E00",
            lw=1.6,
            ls="--",
            label="Moment fit",
        )
        ax.fill_between(
            frequency,
            (prediction[:, k] - errors[:, k]) / scale,
            (prediction[:, k] + errors[:, k]) / scale,
            color="#D55E00",
            alpha=0.2,
            lw=0,
        )
        ax.errorbar(
            frequency[use],
            data[:, k] / scale,
            yerr=sigma[:, k] / scale,
            color="#0072B2",
            marker="o",
            ms=2.8,
            lw=0.7,
            ls="none",
            label="Synthetic data",
        )
        ax.set_xscale("log")
        ax.set_title(f"({chr(97+k)}) Stokes ${label}$", loc="left")
        ax.set_xlabel(r"Channel centre $\nu_j$ [GHz]")
        ax.set_ylabel(f"${label}_j / I_{{\\rm max}}$")
        ax.set_xticks([0.1, 1, 10], labels=["0.1", "1", "10"])
        if k == 0:
            ax.set_yscale("log")
            ax.legend(frameon=False, fontsize=8.5, loc="lower left")
        else:
            ax.axhline(0, color="0.75", lw=0.7)
        ax.set_xlim(frequency[0], frequency[-1])
    save(fig, "radio_sed_fit")


def plot_modes(truth, estimate, errors):
    style()
    fig, ax = plt.subplots(figsize=(7.2, 2.8), layout="constrained")
    x = np.arange(1, len(truth) + 1)
    ax.errorbar(
        x,
        estimate,
        yerr=errors,
        color="#D55E00",
        marker="o",
        ms=4,
        ls="none",
        capsize=3,
        label=r"Fit $\pm1\sigma$",
    )
    ax.plot(
        x, truth, color="0.15", marker="s", ms=4, ls="none", label="True combination"
    )
    ax.set_xlabel("Retained energy-moment combination")
    ax.set_ylabel(r"$\beta_A$")
    ax.set_xticks(x)
    ax.legend(frameon=False, ncol=2)
    ax.axhline(0, color="0.8", lw=0.7)
    save(fig, "radio_moment_combinations")


def plot_derivatives(arrays, records):
    style()
    selected = [3, 6, 8]
    pure = arrays["basis"]["pure_coefficients"]
    scale = float(np.max(pure[selected, 0, :, 0]))
    fig, axes = plt.subplots(
        2, 3, figsize=(9.0, 5.6), sharex=True, layout="constrained"
    )
    colors = ["#0072B2", "#D55E00", "#009E73"]
    titles = [
        r"Reference $\overline{\mathcal{K}}$",
        r"First derivative $\partial_x\overline{\mathcal{K}}$",
        r"Second term $\frac{1}{2}\partial_x^2\overline{\mathcal{K}}$",
    ]
    for r in range(3):
        for b, color in zip(selected, colors):
            axes[0, r].plot(
                n.CENTRES / 1e9,
                pure[b, r, :, 0] / scale,
                color=color,
                lw=1.5,
                label=rf"$\gamma_b={n.GAMMA_MIDS[b]:.0f}$",
            )
            axes[1, r].plot(
                n.CENTRES / 1e9, pure[b, r, :, 1] / scale, color=color, lw=1.5
            )
            axes[1, r].plot(
                n.CENTRES / 1e9, pure[b, r, :, 2] / scale, color=color, lw=1.5, ls="--"
            )
        axes[0, r].set_title(titles[r], fontsize=10)
        axes[1, r].set_xlabel(r"Channel centre $\nu_j$ [GHz]")
    axes[0, 0].set_ylabel(r"Intensity coefficient $/A$")
    axes[1, 0].set_ylabel(r"Polarisation coefficient $/A$")
    axes[0, 0].legend(frameon=False, fontsize=8.5)
    axes[1, 0].legend(
        handles=[
            Line2D([], [], color="0.3", label="Re"),
            Line2D([], [], color="0.3", ls="--", label="Im"),
        ],
        frameon=False,
        fontsize=9,
    )
    for ax in axes.flat:
        ax.set_xscale("log")
        ax.set_xlim(0.1, 10)
        ax.set_xticks([0.1, 1, 10], labels=["0.1", "1", "10"])
        ax.axhline(0, color="0.75", lw=0.7)
    save(fig, "radio_derivative_spectra")
    coarse = arrays["basis_coarse"]["pure_coefficients"]
    halfstep = arrays["halfstep"]["pure_coefficients"]
    return {
        "bins_zero_based": selected,
        "gamma_centres": n.GAMMA_MIDS[selected].tolist(),
        "coordinate_half_widths_in_log_gamma": n.LOG_HALF[selected].tolist(),
        "definition": "D_b,r = (1/r!) partial_x^r Kbar(gamma_b exp(h_b x)) at x=0; nuisance distributions averaged exactly as in SED fit before differentiation",
        "common_normalisation": scale,
        "quadrature_max_component_difference_over_A": float(
            np.max(abs(pure[selected] - coarse[selected])) / scale
        ),
        "halfstep_max_component_difference_over_A": float(
            np.max(abs(halfstep[selected] - coarse[selected])) / scale
        ),
        "cache_identity": records["basis"]["sources"],
    }


if __name__ == "__main__":
    main()
