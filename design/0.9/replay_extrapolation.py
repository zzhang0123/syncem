"""Replay the paper's 300 GHz intensity, polarisation and E4-JP bound.

Run from the repository root with the v0.7.10 study archive extracted.
Set one BLAS thread before starting Python. The report keeps full
precision and records which medians round to the published values.
Pass --curves to compare the full 121-channel intensity predictions.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
from time import perf_counter

import numpy as np

import syncem as se
from syncem.fitting.family_fit import fit_family
from syncem.fitting.moment_form import fit_moment_form

_ENS = ("E2", "E3", "E4-JP")
_BANDS = {"1-40": 21, "1-100": 26}
_NOISE = ("0.0001", "0.01")
_METHODS = ("free", "ridge", "posridge", "pos", "mom2", "cum2")
_SEED = 20261112
_ARCHIVE_SHA = {
    "fits_positive_ridge_products.npz": "d22f12806b429b380bbc5975ada7fb38476fd9388a5dbfbec9d32a9985d1ea8d",
    "fits_positive_ridge_results.json": "baf64d41e56a23922085a92d3a73d77cdd0ae3a5c153d592a90293936f4c3b3e",
    "fits_free_positive_results.json": "519d09298ae269c52b59aa049157e3b8837f275bbda4dde8b7e149a8ea288ad5",
}
_PUBLISHED_I = {
    ("E2", "1-40"): ("19/45", "11/47", "8.9/46", "69/95"),
    ("E2", "1-100"): ("1.6/16", "1.7/16", "1.3/15", "4.4/26"),
    ("E3", "1-40"): ("17/65", "18/61", "15/59", "41/85"),
    ("E3", "1-100"): ("2.3/26", "2.2/25", "1.8/24", "4.3/17"),
}
_PUBLISHED_P = {
    "E2": ("0.68/4.2", "0.47/3.7", "7.3/7.2", "2.9/2.9"),
    "E3": ("1.4/5.7", "1.2/5.4", "0.084/0.20", "0.0045/0.21"),
    "E4-JP": ("1.4e2/78", "2e-10/2e-10", "0.16/0.16", "0.0091/0.0091"),
}


def _archive(directory, curves=False):
    for file, expected in _ARCHIVE_SHA.items():
        with (directory / file).open("rb") as source:
            actual = hashlib.file_digest(source, "sha256").hexdigest()
        if actual != expected:
            raise ValueError(f"v0.7.10 archive digest differs for {file}")
    with np.load(
        directory / "fits_positive_ridge_products.npz", allow_pickle=False
    ) as raw:
        old_stokes = {
            name: {
                "first_draw": raw[f"data_{name}_1-100_0.01"],
                "i_300": float(raw[f"truth_{name}"][-1]),
                "ipl_300": float(raw[f"ipl_{name}"][-1]),
            }
            for name in _ENS
        }
        if curves:
            centres = raw["centres_dense"]
            for name, case in old_stokes.items():
                case["centres"] = centres
                case["truth"] = raw[f"truth_{name}"]
                case["ipl"] = raw[f"ipl_{name}"]
    fits, bounds = (
        json.loads((directory / file).read_text())["summary"]
        for file in (
            "fits_positive_ridge_results.json",
            "fits_free_positive_results.json",
        )
    )
    return old_stokes, fits, bounds


def _noise_draws():
    # Generate the original full shapes. Changing the first dimension before
    # drawing would still agree for the first band but alter seed provenance.
    first = np.random.default_rng(_SEED).standard_normal((500, 21, 3))
    rest = np.random.default_rng(_SEED + 1).standard_normal((500, 11, 3))
    return np.concatenate((first, rest), axis=1)


def _basis(args):
    training, dense = se.paper.extrapolation_channels()
    channels = se.Channels.at(
        np.r_[
            training.centres_hz[:26],
            dense.centres_hz if args.curves else dense.centres_hz[-1:],
        ],
        unit="Hz",
        half_width=0.2,
        n_nu=400,
    )
    regime = se.paper.regime_basis(
        "E1", channels, check=args.refine, moment_nodes=args.moment_nodes
    )
    basis = se.paper.energy_basis(regime, "E1")
    energy_floor = se.paper.cells().edges[3]
    physical = basis.lattice
    # Keep the paper's 16 original high-energy cells and their absolute weights.
    selected = np.flatnonzero(physical.cells.lo[:, 0] >= energy_floor)
    restricted = physical.select(gamma=selected)
    np.testing.assert_array_equal(
        restricted.cells.lo[:, 0], se.paper.cells().edges[3:-1]
    )
    short = se.paper.energy_basis(regime, "E1", restricted)
    return regime, basis, short


def _rounded(actual, published):
    from decimal import Decimal

    target = Decimal(published)
    places = -target.as_tuple().exponent
    return round(float(actual), places) == float(target)


def _published(name, band, noise, method):
    tag = _NOISE.index(noise)
    out = {}
    if (name, band) in _PUBLISHED_I and method in _METHODS[:4]:
        out["I"] = _PUBLISHED_I[name, band][_METHODS.index(method)].split("/")[tag]
    pol_methods = ("free", "posridge", "mom2", "cum2")
    if band == "1-100" and method in pol_methods:
        out["P"] = _PUBLISHED_P[name][pol_methods.index(method)].split("/")[tag]
    return out


def _observations(channels, stokes, sigma):
    return se.StokesData(channels, stokes=stokes, sigma=sigma[..., None], unit="cgs")


def _family_predictions(method, source, fitted, target, stokes, sigma):
    lo = float(fitted.lattice.cells.lo[0, 0])
    hi = float(fitted.lattice.cells.hi[-1, 0])
    if method == "mom2":
        gamma = se.PowerLawMixture.moment_form(2, 2.6, lo=lo, hi=hi)
    else:
        gamma = se.PowerLawMixture.cumulant_form(
            se.Free(2.6, bounds=(0.5, 6.0)),
            se.Free(0.0, bounds=(-1.0, 1.0)),
            lo=lo,
            hi=hi,
        )
    family = source.replace(gamma=gamma)
    method = fit_moment_form if method == "mom2" else fit_family
    result = []
    for one in stokes:
        observation = _observations(fitted.channels, one, sigma)
        fit = method(fitted, observation, family)
        result.append(np.asarray(target.predict(fit.moments, check=False).stokes))
    return np.asarray(result)


def _linear_fit(method, source, basis, channels, stokes, sigma):
    marker = (
        se.NonNegativeCells(point_masses=17)
        if method.startswith("pos")
        else se.FreeMoments()
    )
    return se.fit(
        source.replace(gamma=marker),
        _observations(channels, stokes, sigma),
        basis=basis,
        penalty=(
            None if method == "free" else se.Penalty(0 if method == "pos" else None)
        ),
        check=False,
        chunk_bytes=2**25,
    )


def _population_truth(regime, source, archived, noise, archived_observations):
    # Undo the first stored Gaussian draw exactly in channel-major IQU order.
    truth_fit_archived = (
        archived["first_draw"]
        - 0.01
        * (archived["first_draw"][:, :1] / (1 + 0.01 * noise[:26, :1]))
        * noise[:26]
    )
    prediction = regime.predict(source, check=True)
    truth_fit, truth_prediction = np.split(np.asarray(prediction.stokes), [26])
    endpoint = truth_prediction[-1]
    summary = {
        "fitted_relative_max": float(
            np.max(np.abs(truth_fit - truth_fit_archived) / truth_fit_archived[:, :1])
        ),
        "endpoint_relative_I": float(endpoint[0] / archived["i_300"] - 1),
        "endpoint_P_over_I": float(np.hypot(*endpoint[1:]) / endpoint[0]),
        "source_columns": prediction.plan.source_columns,
        "pairing_refinement_max_over_I": float(
            np.max(
                np.asarray(prediction.ledger["pairing_refinement"].value)
                / np.asarray(prediction.I)[:, None]
            )
        ),
        "harmonic_ledger": {
            t.name: {
                "kind": t.kind,
                "max_abs": None if t.value is None else float(np.max(np.abs(t.value))),
                "relative_to": t.relative_to,
            }
            for t in prediction.ledger.terms
            if t.name.startswith("harmonic_")
        },
    }
    if archived_observations:
        truth_fit = truth_fit_archived
    return truth_fit, truth_prediction, summary


def _metrics(pred, truth, scale, norm, archived, published):
    diff = pred - truth
    di = np.abs(diff[:, 0]) / scale
    dp = np.hypot(diff[:, 1], diff[:, 2]) / scale
    result = {}
    for name, samples in (("I", di), ("P", dp)):
        actual = float(np.median(samples))
        ref = archived[f"d{name}_over_{norm}_300GHz_median_p16_p84"][0]
        result[f"d{name}"] = {
            "median": actual,
            "archived_median": ref,
            "archived_delta": actual - ref,
        }
        if name in published:
            result[f"d{name}"]["paper_percent"] = published[name]
            result[f"d{name}"]["rounds_to_paper"] = _rounded(
                100 * actual, published[name]
            )
    return result


def _power_law(centres, intensity):
    """The Figure 7 log least-squares reference on channels through 40 GHz."""
    log_frequency = np.log(centres / 1e9)
    use = centres <= 40e9 * (1 + 1e-9)
    slope, offset = np.polyfit(log_frequency[use], np.log(intensity[use]), 1)
    return np.exp(offset + slope * log_frequency)


def _curve_reference(channels, truth, archived, summary):
    centres = np.asarray(channels.centres_hz)
    np.testing.assert_allclose(centres, archived["centres"], rtol=2e-14, atol=0)
    ipl = _power_law(archived["centres"], archived["truth"])
    np.testing.assert_allclose(ipl, archived["ipl"], rtol=2e-13, atol=0)
    fitted_ipl = _power_law(centres, truth)
    summary["dense_truth_max_relative_I"] = float(
        np.max(np.abs(truth / archived["truth"] - 1))
    )
    summary["dense_reference_power_law_max_relative"] = float(
        np.max(np.abs(fitted_ipl / ipl - 1))
    )
    return fitted_ipl


def _curve_metrics(pred, truth, ipl, reference, archived_ratio, norm):
    """Compare the intensity draws, envelopes and error panels in figure units."""
    actual = pred / ipl
    archival = archived_ratio * (reference["truth"] / reference["ipl"])
    # The frozen replay scales the comparison by the larger of I_PL and the
    # archived prediction's magnitude in each draw and channel.
    sample_change = np.abs(actual - archival) / np.maximum(1, np.abs(archival))
    worst_draw, worst_channel = np.unravel_index(
        np.argmax(sample_change), sample_change.shape
    )
    max_sample_change = sample_change[worst_draw, worst_channel]
    actual_error = np.abs(pred - truth) / (truth if norm == "I" else ipl)
    archived_error = np.abs(archived_ratio - 1)
    if norm == "IPL":
        archived_error *= reference["truth"] / reference["ipl"]
    envelope = np.percentile(actual, [16, 50, 84], axis=0)
    archived_envelope = np.percentile(archival, [16, 50, 84], axis=0)
    return {
        "channels": truth.size,
        "max_draw_change_divided_by_larger_of_ipl_and_archived_prediction": float(
            max_sample_change
        ),
        "within_frozen_curve_comparison_tolerance": bool(max_sample_change < 1e-4),
        "draws_exceeding_frozen_tolerance": int(
            np.count_nonzero(np.max(sample_change, axis=1) >= 1e-4)
        ),
        "worst_draw_index": int(worst_draw),
        "worst_channel_index": int(worst_channel),
        "max_envelope_change_over_ipl": float(
            np.max(np.abs(envelope - archived_envelope))
        ),
        "max_median_error_change": float(
            np.max(
                np.abs(
                    np.median(actual_error, axis=0) - np.median(archived_error, axis=0)
                )
            )
        ),
    }


def _write(output, report):
    output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")


def replay(args):
    start = perf_counter()
    start_load = os.getloadavg()
    old_truth, old_fits, old_bounds = _archive(args.archive, args.curves)
    draws = _noise_draws()
    regime, full, short = _basis(args)
    n_fit = _BANDS
    indices = {band: regime.channels[: n_fit[band]] for band in args.bands}
    prediction = regime.channels[26:]
    bands = (
        {band: full.at(channels) for band, channels in indices.items()}
        if any(method in _METHODS[4:] for method in args.methods)
        else {}
    )
    target = full.at(prediction) if bands else None
    report = {
        "archive_sha256": _ARCHIVE_SHA,
        "draws": args.draws,
        "syncem_version": se.__version__,
        "profile_draws": args.profile_draws,
        "seed": _SEED,
        "archived_observations": args.archived_observations,
        "moment_nodes": args.moment_nodes,
        "harmonic_refinement": args.refine,
        "start_load": start_load,
        "post_basis_load": os.getloadavg(),
        "continuum_cells": regime.continuum.lattice.n_cells,
        "harmonic_cells": regime.harmonic.lattice.n_cells,
        "cases": {},
    }
    for name in args.ensembles:
        source = se.paper.population(name)
        truth_fit, truth_prediction, checks = _population_truth(
            regime, source, old_truth[name], draws[0], args.archived_observations
        )
        endpoint = truth_prediction[-1]
        if args.curves:
            ipl = _curve_reference(
                prediction, truth_prediction[:, 0], old_truth[name], checks
            )
        case = report["cases"].setdefault(name, {"truth_check": checks, "bands": {}})
        norm = "I" if name != "E4-JP" else "IPL"
        scale = endpoint[0] if norm == "I" else old_truth[name]["ipl_300"]
        for band, channels in indices.items():
            for noise in args.noise:
                n = n_fit[band]
                sigma = float(noise) * truth_fit[:n, 0]
                stokes = truth_fit[:n] + sigma[:, None] * draws[: args.draws, :n]
                summary = case["bands"].setdefault(band, {}).setdefault(noise, {})
                for method in args.methods:
                    if method in _METHODS[:4]:
                        fit = _linear_fit(
                            method, source, short, channels, stokes, sigma
                        )
                        pred = np.asarray(fit.predict(prediction).stokes)
                    else:
                        pred = _family_predictions(
                            method, source, bands[band], target, stokes, sigma
                        )
                    reference = old_fits[name][method][band][noise]
                    summary[method] = _metrics(
                        pred[:, -1],
                        endpoint,
                        scale,
                        norm,
                        reference,
                        (
                            _published(name, band, noise, method)
                            if args.draws == 500
                            else {}
                        ),
                    )
                    if args.curves:
                        with np.load(
                            args.archive / "fits_positive_ridge_products.npz",
                            allow_pickle=False,
                        ) as raw:
                            archived_ratio = raw[
                                f"{name}_{method}_{band}_{noise}_ratio"
                            ][: args.draws]
                        summary[method]["curves"] = _curve_metrics(
                            pred[:, :, 0],
                            truth_prediction[:, 0],
                            ipl,
                            old_truth[name],
                            archived_ratio,
                            norm,
                        )
                    print(name, band, noise, method, summary[method], flush=True)
                    _write(args.output, report)
                if name == "E4-JP" and noise == "0.01" and args.profile_draws:
                    count = args.profile_draws
                    profile_stokes = truth_fit[:n] + sigma[:, None] * draws[:count, :n]
                    positive = _linear_fit(
                        "pos", source, short, channels, profile_stokes, sigma
                    )
                    bounds = (
                        np.asarray(
                            positive.profile_max(prediction[-1:], delta_chi2=1.0).value
                        )[:, 0]
                        / scale
                    )
                    reference = old_bounds[name]["pos"][band][noise][
                        "max_I300_over_IPL_dchi2_1"
                    ]
                    median = float(np.median(bounds))
                    summary["unpenalized_delta_chi2_1"] = {
                        "median": median,
                        "archived_median": reference["median"],
                        "draws": count,
                    }
                    if count == 100:
                        text = "0.21" if band == "1-40" else "1.7e-5"
                        summary["unpenalized_delta_chi2_1"]["rounds_to_paper"] = (
                            _rounded(median, text)
                        )
                    print(
                        name,
                        band,
                        "bound",
                        summary["unpenalized_delta_chi2_1"],
                        flush=True,
                    )
                    _write(args.output, report)
    report["seconds"] = perf_counter() - start
    report["end_load"] = os.getloadavg()
    _write(args.output, report)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    parser.add_argument(
        "--archive", type=Path, default=Path("studies/classical-models")
    )
    parser.add_argument(
        "--draws", type=int, default=500, choices=range(1, 501), metavar="1..500"
    )
    parser.add_argument(
        "--profile-draws", type=int, default=100, choices=range(101), metavar="0..100"
    )
    parser.add_argument("--ensembles", nargs="+", choices=_ENS, default=_ENS)
    parser.add_argument("--methods", nargs="*", choices=_METHODS, default=_METHODS)
    parser.add_argument("--noise", nargs="+", choices=_NOISE, default=_NOISE)
    parser.add_argument("--bands", nargs="+", choices=_BANDS, default=list(_BANDS))
    parser.add_argument("--moment-nodes", type=int, default=12)
    parser.add_argument("--no-refine", dest="refine", action="store_false")
    parser.add_argument("--archived-observations", action="store_true")
    parser.add_argument(
        "--curves", action="store_true", help="Replay all 121 prediction channels."
    )
    replay(parser.parse_args())


if __name__ == "__main__":
    main()
