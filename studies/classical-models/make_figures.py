"""Regenerate the classical-model figures (Figs. 8-10, galprop_fit) and LaTeX tables from saved results.

Run: python make_figures.py

Inputs (read only, SHA-256 recorded): fits_results.json, fits_products.npz,
forward_results.json, forward_spectra.npz, params_results.json, baselines_results.json,
cost_results.json, verify_fits_results.json, kp_pseudotruth_results.json, kp_pseudotruth.npz,
baselines_plotdata.npz.  The producing scripts' recorded hashes are compared with the files on disk.
galprop_fit recomputes one curve (moment basis at R = 12, not saved) with baselines.py's functions;
see figure_data.galprop_fit and the galprop/* checks.

Needs verify_fits_results.json (python verify_fits.py) besides the producing scripts' outputs.

Outputs (this directory only):
  figures/{reference_ensembles,moment_fit_gallery,moments_and_parameters,galprop_fit}.{pdf,png}
  figures/{reference_ensembles,moment_fit_gallery,moments_and_parameters}_compact.{pdf,png}
      (fewer panels, same panel functions; the panels shown in Figs. 8-10 of the paper,
       while the full figures have all panels)
  tables/{ensembles,fit_summary,parameter_recovery,baselines,cost}.tex
  tables/{fit_summary,baselines}_compact.tex (the layout of Tables 4 and 5 of the paper, to cross-check
      their entries)
  make_figures_results.json (provenance, every consistency check, output hashes)
  make_figures_data.npz     (the derived display arrays drawn in the figures)

Continuum (beaming-limit) kernels only.
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import shutil  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from tempfile import TemporaryDirectory  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.text import Text  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT_FIG, OUT_TAB = HERE / "figures", HERE / "tables"
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))
from figure_style import WIDTH, configure  # noqa: E402

import figure_data as FD  # noqa: E402
import figure_plots as FP  # noqa: E402
import figure_tables as FT  # noqa: E402
from provenance import provenance, relative, sha256, write_json  # noqa: E402

CHECK_LIMITS = {  # (substring of the check name, largest allowed |value|)
    "C_a_hat_vs_saved_pred": 1e-12, "C_a_true_vs_S_minus_eps": 1e-12, "max_err_vs_fits_json": 1e-12,
    "chi_joint_from_pred": 1e-9, "_I_vs_forward": 1e-12, "Pi_vs_forward": 1e-12, "chi_intrinsic_vs": 1e-9,
    "kp_": 1e-9, "unit_column": 1e-9, "refit": 1e-6, "R_star_recomputed": 0, "a(theta_true)": 1e-12,
    "grid_min_minus_best": 1e-6, "grid_profile_vs_multistart": 1e-6,
    "C_a_pseudo_vs_saved_pred_pseudo": 1e-12, "S_direct_vs_pseudo_npz": 1e-12, "pseudo_dev_consistency": 1e-9,
    # galprop_fit: plotted withheld rms / max against baselines_results.json (fractions of I), the
    # recomputed moment curve against the saved one, dense curves at the nodes, input identity
    "rms_held_minus_json": 1e-12, "max_held_minus_json": 1e-12, "legend_rms_minus_json": 1e-12,
    "recomputed_minus_saved": 1e-12,
    "dense_at_nodes": 1e-12, "median_rank": 0, "plotdata_sha256_mismatch": 0, "rank_json_minus_npz": 0,
}


def save(fig, name: str) -> dict:
    """House save (figure_style.save of the paper's figures) with a local destination."""
    fig.set_figwidth(WIDTH)
    for label in fig.findobj(Text):
        label.set_fontsize(min(9, max(8, label.get_fontsize())))
    written = {}
    with TemporaryDirectory(prefix="classical-figure-") as tmp:
        for ext, kw in (("pdf", {}), ("png", {"dpi": 200})):
            staged = Path(tmp) / f"{name}.{ext}"
            fig.savefig(staged, **kw)
            dest = OUT_FIG / staged.name
            shutil.copyfile(staged, dest.with_suffix(f".{ext}.tmp"))
            dest.with_suffix(f".{ext}.tmp").replace(dest)
            written[relative(dest)] = sha256(dest)
    size = [round(float(v), 3) for v in fig.get_size_inches()]
    plt.close(fig)
    return {"files": written, "size_inches": size}


def flatten_checks(D: dict) -> dict:
    out = {}
    for block in ("electrons", "intrinsic", "bands", "e2"):
        for k, v in D[block]["checks"].items():
            out[f"{block}/{k}"] = float(v)
    for name, rec in D["modes"].items():
        if "check_pseudo_dev_consistency" in rec:
            out[f"modes/{name}_pseudo_dev_consistency"] = float(rec["check_pseudo_dev_consistency"])
    for k, v in D["galprop"]["checks"].items():
        out[f"galprop/{k}"] = float(v)
    return out


def galprop_json_checks(G: dict, bl: dict) -> dict:
    """Withheld rms and max of the plotted node residuals minus baselines_results.json.

    Moment basis: noise_free_tsvd_curve at each drawn rank (and, for the MC median rank, the
    noise_free_moment_at_MC_median_Rstar record); laws and families: noise_free.<model>.
    """
    res = bl["results"]["GALPROP"]
    curve = {r["R"]: r for r in res["noise_free_tsvd_curve"]}
    ref = {f"moment_R{R}": curve[R] for R in G["ranks"]}
    ref.update({m: res["noise_free"][m] for m in G["n_params"]})
    held = ~G["train"]
    out = {}
    for key, rec in ref.items():
        r = G["models"][key]["res_pct"][held] / 100.0  # the array drawn in panel (b)
        out[f"galprop/{key}_rms_held_minus_json"] = float(np.sqrt(np.mean(r**2)) - rec["rms_res_held"])
        out[f"galprop/{key}_max_held_minus_json"] = float(np.max(np.abs(r)) - rec["max_res_held"])
        out[f"galprop/{key}_legend_rms_minus_json"] = float(G["models"][key]["rms_held"] - rec["rms_res_held"])
    med = res["noise_free_moment_at_MC_median_Rstar"]
    out["galprop/MC_median_rank_json_minus_npz"] = float(med["R"] - G["ranks"][0])
    out[f"galprop/moment_R{G['ranks'][0]}_rms_held_minus_json_median_record"] = float(
        G["models"][f"moment_R{G['ranks'][0]}"]["rms_held"] - med["rms_res_held"])
    return out


def assert_checks(checks: dict, hashes: dict) -> dict:
    status = {}
    for name, value in checks.items():
        limit = next((lim for key, lim in CHECK_LIMITS.items() if key in name), None)
        if limit is None:
            raise KeyError(f"no tolerance for check {name}")
        ok = abs(value) <= limit
        status[name] = {"value": value, "limit": limit, "ok": ok}
        if not ok:
            raise AssertionError(f"check failed: {name} = {value} (limit {limit})")
    bad = [s for s, h in hashes.items() if not h["match"]]
    if bad:
        raise AssertionError(f"producer scripts changed since their results were written: {bad}")
    return status


def display_arrays(D: dict) -> dict:
    arr = {"gamma": D["electrons"]["gamma"], "gamma_p2": D["electrons"]["gamma_p2"],
           "kp_P2_conditional": D["electrons"]["p2"]}
    for k, v in D["electrons"]["rel_density"].items():
        arr[f"rel_density__{k}"] = v
    for k, v in D["intrinsic"]["fraction"].items():
        arr[f"intrinsic_P_over_I__{k}"] = v
    for k, f in D["bands"]["fits"].items():
        arr[f"fit_sd_I__{k}"], arr[f"fit_sd_absP__{k}"] = f["sd_I"], f["sd_absP"]
    for mode, g in D["e2"]["grids"].items():
        arr[f"e2_{mode}_p2"], arr[f"e2_{mode}_ln_gamma_b"], arr[f"e2_{mode}_chi2"] = g["p2"], g["ln_gamma_b"], g["chi2"]
        arr[f"e2_{mode}_cov"] = g["cov"]
        arr[f"e2_a_theta_{mode}"] = D["e2"]["a_theta"][mode]
    G = D["galprop"]
    for k in ("nu", "nu_dense", "train", "I", "pl", "pl_dense"):
        arr[f"galprop_{k}"] = G[k]
    for key, rec in G["models"].items():
        arr[f"galprop_{key}_I"], arr[f"galprop_{key}_I_dense"] = rec["I"], rec["I_dense"]
        arr[f"galprop_{key}_residual_pct"] = rec["res_pct"]
    return arr


def main() -> None:
    t0 = time.time()
    OUT_FIG.mkdir(exist_ok=True)
    OUT_TAB.mkdir(exist_ok=True)
    configure()
    data, input_shas, hashes = FD.load_inputs()
    D = FD.build_all(data)
    status = assert_checks({**flatten_checks(D), **galprop_json_checks(D["galprop"], data["baselines"])}, hashes)
    outputs = {"reference_ensembles": save(FP.fig_reference_ensembles(D, data), "reference_ensembles"),
               "moment_fit_gallery": save(FP.fig_moment_fit_gallery(D, data), "moment_fit_gallery"),
               "moments_and_parameters": save(FP.fig_moments_and_parameters(D, data), "moments_and_parameters"),
               "galprop_fit": save(FP.fig_galprop_fit(D, data), "galprop_fit")}
    compact = {"reference_ensembles": lambda: FP.fig_reference_ensembles_compact(D, data),
               "moment_fit_gallery": lambda: FP.fig_moment_fit_gallery(D, data, rows=FP.GALLERY_COMPACT),
               "moments_and_parameters": lambda: FP.fig_moments_and_parameters_compact(D, data)}
    outputs.update({f"{k}_compact": save(draw(), f"{k}_compact") for k, draw in compact.items()})
    tables = {}
    for name, tex in FT.all_tables(data, D["modes"]).items():
        path = OUT_TAB / f"{name}.tex"
        path.write_text(f"% Generated by make_figures.py from saved results; do not edit by hand.\n{tex}")
        tables[relative(path)] = sha256(path)
    np.savez(HERE / "make_figures_data.npz", **display_arrays(D))
    modes = {k: {"pairs": v["pairs"], "max_abs_bias_invariant": v["max_abs_bias_invariant"],
                 "max_abs_bias_json": v["max_abs_bias_json"],
                 "pair_z_equiv": v.get("pair_z_equiv", []), "pair_bias_norm": v.get("pair_bias_norm", [])}
             for k, v in D["modes"].items()}
    lib_inputs = [HERE / n for n in ("figure_data.py", "figure_plots.py", "figure_tables.py", "paramlib.py",
                                     "baselines.py", "baseline_models.py")]
    lib_inputs.append(HERE / "lib" / "figure_style.py")
    prov = provenance(__file__, t0, inputs=[*map(Path, input_shas), *lib_inputs])
    write_json(HERE / "make_figures_results.json", {
        "description": "Figures 8-10, galprop_fit and LaTeX tables of the classical-model section, drawn from saved "
                       "results (galprop_fit: moment basis at R = 12 recomputed, see galprop/* checks). The *_compact "
                       "figures have the panels shown in Figs. 8-10 of the paper; the *_compact tables have the "
                       "layout of Tables 4 and 5 of the paper.",
        "figures": outputs, "tables": tables, "data_npz": relative(HERE / "make_figures_data.npz"),
        "producer_script_hashes": hashes, "checks": status, "e5_mode_pairs": modes,
        "e2_contour_levels_delta_chi2": D["e2"]["levels_2d"], "provenance": prov})
    print(f"done in {time.time() - t0:.1f} s; {len(status)} checks passed")


if __name__ == "__main__":
    main()
