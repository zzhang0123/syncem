"""Replay the synthetic intensity-only electron-family rows of paper Table 5.

Run from the repository root with
python -m tests.v090.replay_paper_baseline_families --workers 4

The command writes one JSON record per campaign and exits with status 1
if any median differs at the precision printed in the paper.
"""

import argparse
import importlib
import json
import multiprocessing as mp
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from time import perf_counter
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from tests.v080 import _study

_CASES = ("E1", "E2", "E3", "E4-JP", "E4-KP", "E5")
_NOISE_LEVELS = ("0.0001", "0.01")


def _study_baseline():
    """Load the original fit procedure in this runner's dedicated process."""
    sys.path[:0] = [str(_study.STUDY / "lib"), str(_study.STUDY)]
    try:
        return importlib.import_module("baseline_models")
    finally:
        del sys.path[:2]


def _physical_inputs():
    # Import SyncEM only in the parent process to avoid starting JAX in workers.
    import syncem as se

    channels = se.paper.channels()
    common = se.paper.regime_basis("E1", channels, check=False)
    kp = se.paper.regime_basis("E4-KP", channels, check=False)
    basis = se.paper.energy_basis(common, "E1", order=2)
    lattice = basis.lattice
    cells = lattice.cells
    geometry = SimpleNamespace(
        gamma_b=np.asarray(cells.reference[:, 0]),
        h=np.asarray(cells.half_width[:, :1]),
        w_b=np.asarray(lattice.weights),
    )
    intensity_design = np.asarray(basis.design())[:, 0, :]
    response_weights = np.asarray(channels.response_weights())
    np.testing.assert_allclose(
        response_weights,
        np.broadcast_to(response_weights[0], response_weights.shape),
        atol=1e-15,
    )
    spectra = {}
    for name in _CASES:
        regime = kp if name == "E4-KP" else common
        spectra[name] = np.asarray(
            regime.predict(se.paper.population(name), check=True).I
        )
    return (
        intensity_design,
        geometry,
        np.asarray(channels.nodes),
        response_weights[0],
        spectra,
    )


def _campaign(task):
    name, level, intensity, design, geometry, nodes, weights, draws, reference = task
    baseline = _study_baseline()
    laws = baseline.SL
    fitted, held = slice(None, None, 2), slice(1, None, 2)
    grid = laws.channel_grid(nodes, weights).rows(fitted)
    sigma = float(level) * intensity
    samples = intensity + sigma * draws
    errors = {family: [] for family in baseline.FAMILIES}

    # The original quadrature broadcasts over the physical cell references and
    # widths here. The original default geometry was only the 19 upper cells.
    family_moments = partial(baseline.family_moments, cfg=geometry)
    with patch.object(baseline, "family_moments", family_moments):
        for sample in samples[:, fitted]:
            # The signed intensity slope sets the original electron index start.
            s1 = laws.fit("s1", grid, sample, sigma[fitted])
            p0 = 1.0 - 2.0 * s1["params"][0]
            for family in baseline.FAMILIES:
                result = baseline.fit_family(
                    family, design[fitted], sample, sigma[fitted], p0=p0
                )
                prediction = baseline.family_prediction(
                    family, result["params"], design[held]
                )
                fractional = prediction / intensity[held] - 1.0
                errors[family].append(np.sqrt(np.mean(fractional**2)))

    results = {}
    for family, error in errors.items():
        actual = float(np.median(error))
        expected = reference[family]
        results[family] = {
            "rms": actual,
            "archived_rms": expected,
            "difference": actual - expected,
            "same_two_significant_figures": format(actual, ".2g")
            == format(expected, ".2g"),
        }
    return name, level, results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=1)
    options = parser.parse_args()
    start = perf_counter()
    print(json.dumps({"start_load": os.getloadavg()}), flush=True)
    filename = "baselines_noise_sweep_results.json"
    archive = _study.STUDY / filename
    expected = _study.read_sums(_study.STUDY / "SHA256SUMS")[filename]
    assert _study.sha256_file(archive) == expected
    sweep = json.loads(archive.read_text())
    design, geometry, nodes, weights, spectra = _physical_inputs()
    print(json.dumps({"physical_response_s": perf_counter() - start}), flush=True)
    rng = np.random.default_rng(sweep["mc_seeds"]["synthetic"])
    draws = rng.standard_normal((sweep["n_real"], len(nodes)))
    tasks = [
        (
            name,
            level,
            spectra[name],
            design,
            geometry,
            nodes,
            weights,
            draws,
            sweep["results"][level]["compact_median_rms_held"][name],
        )
        for name in _CASES
        for level in _NOISE_LEVELS
    ]
    count = 0
    discrepancies = 0
    max_difference = 0.0
    with ProcessPoolExecutor(
        max_workers=options.workers, mp_context=mp.get_context("spawn")
    ) as executor:
        for name, level, results in executor.map(_campaign, tasks):
            for model in results.values():
                count += 1
                discrepancies += int(not model["same_two_significant_figures"])
                max_difference = max(max_difference, abs(model["difference"]))
            print(
                json.dumps({"ensemble": name, "noise": level, "models": results}),
                flush=True,
            )
    print(
        json.dumps(
            {
                "medians": count,
                "different_at_printed_precision": discrepancies,
                "max_absolute_difference": max_difference,
                "finish_load": os.getloadavg(),
                "seconds": perf_counter() - start,
            }
        ),
        flush=True,
    )
    return int(discrepancies != 0)


if __name__ == "__main__":
    raise SystemExit(main())
