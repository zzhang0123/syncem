"""Intensity-only baselines of baselines.py repeated at lower noise levels.

Same channels, models, seeds, rank cap (R <= 14) and 500 common-random-number realisations as
baselines.py, with the noise sd set to frac * I for frac in NOISE_FRACS. The 1% run reproduces the
Monte Carlo medians of baselines_results.json and is kept as a consistency check
(archive_1pct_max_abs_diff_rms_held).

Reads baselines_results.json (python baselines.py).
Writes baselines_noise_sweep_results.json.  Run: python baselines_noise_sweep.py
"""
from __future__ import annotations

import json
import multiprocessing as mp
import time
from dataclasses import replace
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist

import baselines as BL

HERE = Path(__file__).resolve().parent
NOISE_FRACS = (1e-2, 1e-3, 1e-4)
ARCHIVE = HERE / "baselines_results.json"
OUT = HERE / "baselines_noise_sweep_results.json"
T0 = time.time()


def contexts(frac: float, spectra: dict, g: dict, table) -> dict:
    """Fit contexts with noise sd = frac * I for the synthetic ensembles and GALPROP."""
    cfg = replace(BL.PAPER, noise_frac=frac)
    ctxs = BL.synthetic_contexts(cfg, spectra)
    gal = BL.galprop_context(cfg, g, table)
    sig = frac * g["I"]
    fi = gal["fit_idx"]
    ctxs["GALPROP"] = {**gal, "sig": sig, "svd": BL.BM.tsvd_setup(gal["CI"][fi], sig[fi])}
    return ctxs


def monte_carlo(ctxs: dict, n_channels: int, n_gal: int) -> dict:
    """Run the baselines.py Monte Carlo (same seeds and chunking) on the given contexts."""
    z_syn = np.random.default_rng(BL.MC_SEED).normal(size=(BL.NREAL, n_channels))
    z_gal = np.random.default_rng(BL.MC_SEED + 1).normal(size=(BL.NREAL, n_gal))
    tasks = []
    for key in ctxs:
        z = z_gal if key == "GALPROP" else z_syn
        tasks += [(key, z[i:i + BL.CHUNK]) for i in range(0, BL.NREAL, BL.CHUNK)]
    runs: dict = {}
    with mp.get_context("spawn").Pool(BL.N_WORKERS, initializer=BL._init_worker, initargs=(ctxs,)) as pool:
        for key, out in pool.imap(BL._run_chunk, tasks):
            runs.setdefault(key, []).extend(out)
    summary = {}
    for key, rr in runs.items():
        crit = float(chi2_dist.ppf(0.95, len(ctxs[key]["held_idx"])))
        summary[key] = BL.summarise(rr, crit)
    return summary


def compact(summary: dict) -> dict:
    """Median withheld rms (fraction of I) per model, plus the UPRE rank statistics."""
    out = {}
    for key, models in summary.items():
        rec = {m: models[m]["rms_res_held"]["median"] for m in models if "rms_res_held" in models[m]}
        up = models["moment_UPRE"]
        rec["R_star_median"] = up["R_star"]["median"]
        rec["frac_cap_binding"] = up.get("frac_cap_binding")
        out[key] = rec
    return out


def archive_check(compact_1pct: dict) -> float:
    """Largest |sweep - archive| of the 1% medians over models and ensembles."""
    arch = json.loads(ARCHIVE.read_text())
    worst = 0.0
    for key, rec in compact_1pct.items():
        ref = arch["results"][key]["monte_carlo"]["summary"]
        for model, val in rec.items():
            if model in ref and "rms_res_held" in ref[model]:
                worst = max(worst, abs(val - ref[model]["rms_res_held"]["median"]))
    return worst


def main() -> None:
    cfg = BL.PAPER
    table = BL.B.get_table(cfg)
    spectra, _ = BL.synthetic_direct(cfg)
    g = BL.galprop_data()
    results, timings = {}, {}
    for frac in NOISE_FRACS:
        t = time.time()
        summary = monte_carlo(contexts(frac, spectra, g, table), cfg.n_channels, len(g["nu"]))
        results[f"{frac:g}"] = {"compact_median_rms_held": compact(summary), "summary": summary}
        timings[f"{frac:g}"] = time.time() - t
        print(f"noise {frac:g}: {timings[f'{frac:g}']:.1f} s")
    check = archive_check(results["0.01"]["compact_median_rms_held"])
    out = {
        "description": __doc__,
        "noise_fracs": list(NOISE_FRACS),
        "rank_cap": BL.BM.R_CAP,
        "n_real": BL.NREAL,
        "mc_seeds": {"synthetic": BL.MC_SEED, "galprop": BL.MC_SEED + 1},
        "archive_1pct_max_abs_diff_rms_held": check,
        "results": results,
        "timings_s": timings,
        "provenance": BL.provenance(__file__, T0, inputs=(ARCHIVE, HERE / "baselines.py",
                                                          HERE / "baseline_models.py",
                                                          HERE / "spectral_laws.py")),
    }
    BL.write_json(OUT, out)
    print("archive check (max |diff| at 1%):", check)
    print("done", time.time() - T0)


if __name__ == "__main__":
    main()
