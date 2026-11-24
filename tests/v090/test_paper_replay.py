"""Radio and 300 GHz regression runs using both branches at the paper's resolution."""

import json
import runpy
from pathlib import Path

import numpy as np
import pytest

import syncem as se
from tests.v080 import _study

pytestmark = pytest.mark.slow

_SEPARABLE = ("E1", "E2", "E3", "E4-JP", "E4-KP", "E1M")
_JOINT = {"E5": ("gamma", "phi"), "E6": ("gamma", "B")}


@pytest.fixture(scope="module")
def radio():
    channels = se.paper.channels()
    sources = {case: se.paper.population(case) for case in (*_SEPARABLE, *_JOINT)}
    common = se.paper.regime_basis("E1", channels, check=False)
    bases = {name: common for name in sources}
    for name in ("E1M", "E4-KP", "E6"):
        bases[name] = se.paper.regime_basis(name, channels, check=False)
    spectra = {
        name: bases[name].predict(source, check=True)
        for name, source in sources.items()
    }
    return channels, sources, bases, spectra


def _study_results(filename):
    path = _study.STUDY / filename
    expected = _study.read_sums(_study.STUDY / "SHA256SUMS")
    assert _study.sha256_file(path) == expected[filename]
    return json.loads(path.read_text())


def _max_relative_error(prediction, truth):
    return float(np.max(np.abs(np.asarray(prediction) - truth) / truth[:, :1]))


def _check_seeded_fit(fit, truth, target, case):
    assert int(fit.rank) == target["R_star"], case
    assert int(fit.dof) == target["dof_fit"], case
    assert f"{float(fit.chi2):.1f}" == f"{target['chi2_fit']:.1f}", case
    error = _max_relative_error(fit.predict().stokes, truth)
    assert (
        f"{error:.2g}" == f"{target['max_fit_minus_direct_over_I']['value']:.2g}"
    ), case
    # The unrounded held-out diagnostics accompany the published table.
    assert float(fit.withheld.chi2) == pytest.approx(
        target["chi2_held"], abs=0.01
    ), case
    predictive = target["chi2_held_predictive"]
    assert float(fit.withheld.chi2_predictive) == pytest.approx(
        predictive, abs=0.01
    ), case


@pytest.fixture(scope="module")
def radio_observations(radio):
    channels, _, _, spectra = radio
    data = _study.ACCEPTANCE_DATA
    _study.verify_sums(data / "SHA256SUMS")
    cfg = _study.paper()
    held = np.arange(1, 41, 2)
    observations = {}
    with (
        np.load(data / "forward_spectra.npz") as archive,
        np.load(data / "fits_products.npz") as fitted,
    ):
        for name, prediction in spectra.items():
            key = name.replace("-", "_")
            if name == "E6":
                reference = _study.study().e6.direct(cfg)
                fitted_noise = cfg.draw_noise(21)
                held_noise = cfg.draw_noise(20, seed=cfg.seed + 1)
            else:
                reference = archive[key]
                np.testing.assert_array_equal(reference, fitted[f"{key}_S_direct"])
                sigma = fitted[f"{key}_sigma"]
                fitted_noise = (fitted[f"{key}_data_fit"] - reference[::2]) / sigma[::2]
                held_noise = (fitted[f"{key}_data_held"] - reference[1::2]) / sigma[
                    1::2
                ]
            truth = np.asarray(prediction.stokes)
            assert _max_relative_error(truth, reference) < 1.2e-6, name
            lower, upper = prediction.plan.source_columns
            assert lower > 0 and upper == pytest.approx(1.0, abs=2e-7), name

            # Keep the paper noise realisation and fractional noise model.
            sigma = cfg.noise_frac * truth[:, :1]
            observed = truth.copy()
            observed[::2] += sigma[::2] * fitted_noise
            observed[1::2] += sigma[1::2] * held_noise
            observations[name] = se.StokesData(
                channels, stokes=observed, sigma=sigma, unit="cgs"
            ).withhold(held)
    return observations


@pytest.fixture(scope="module")
def physical_radio(radio):
    _, sources, bases, _ = radio
    return {
        name: se.paper.energy_basis(bases[name], sources[name])
        for name in ("E1", *_JOINT)
    }


def test_forward_and_seeded_radio(radio, radio_observations, physical_radio):
    _, sources, _, spectra = radio
    data = _study.ACCEPTANCE_DATA
    seed_results = json.loads((data / "fits_results.json").read_text())["ensembles"]
    scalar = sources["E1"].replace(gamma=se.FreeMoments())
    for name, new in spectra.items():
        if name in _JOINT:
            names = _JOINT[name]
            specified = {
                key: law
                for key, law in sources[name].parts[0].factors.items()
                if key not in names
            }
            unknown = se.Population(se.FreeMoments(*names), **specified)
        else:
            unknown = scalar
        fit = se.fit(
            unknown,
            radio_observations[name],
            basis=physical_radio.get(name, physical_radio["E1"]),
            check=False,
        )
        if name == "E6":
            assert int(fit.rank) == 9, name
            assert int(fit.dof) == 54, name
            assert float(fit.chi2) == pytest.approx(48.8641959385, abs=1e-3), name
        else:
            _check_seeded_fit(
                fit, np.asarray(new.stokes), seed_results[name]["seeded"], name
            )


def test_fit_summary_prediction_errors(radio, physical_radio):
    _, sources, _, spectra = radio
    expected = _study_results("forward_results.json")["summary_max_resid_over_I_IQU"]
    pseudo = _study_results("kp_pseudotruth_results.json")
    amplitude, index, log_cut = pseudo["pseudo_truth"]["theta_A_p_ln_gamma_c"]
    jp_at_kp_age = sources["E1"].replace(
        gamma=se.AgedPowerLaw(
            index, gamma_cut=np.exp(log_cut), lo=100, amplitude=amplitude / 1e4
        )
    )
    factor = se.paper.specified()["phi"]
    first, second = (part.replace(phi=factor) for part in sources["E5"].parts)
    one_azimuth = first + second
    cases = (
        ("E1", sources["E1"], "E1"),
        ("E2", sources["E2"], "E2"),
        ("E3", sources["E3"], "E3"),
        ("E4-JP", sources["E4-JP"], "E4-JP"),
        ("E4-KP", jp_at_kp_age, "E4-KP"),
        ("E5", sources["E5"], "E5"),
        ("E1M", sources["E1"], "E1M_U46_basis"),
        ("E5", one_azimuth, "E5_separable_paper_factor"),
    )
    for order in (2, 4):
        common = physical_radio["E1"]
        joint = physical_radio["E5"]
        if order != 2:
            common = common.with_order(order)
            joint = joint.with_order(order)
        for name, population, key in cases:
            basis = joint if key == "E5" else common
            prediction = basis.predict(population, check=False)
            error = _max_relative_error(
                prediction.stokes, np.asarray(spectra[name].stokes)
            )
            order_key = f"N{order}"
            target = (
                pseudo["forward"][order_key]["max_abs_S_KP_minus_C_a_over_I_IQU"]
                if name == "E4-KP"
                else expected[key][order_key]
            )
            assert f"{error:.2g}" == f"{max(target):.2g}", (key, order)


def test_e5_single_fitted_azimuth(radio, radio_observations):
    _, sources, bases, spectra = radio
    archived = json.loads((_study.ACCEPTANCE_DATA / "fits_results.json").read_text())
    target = archived["E5"]["separable_free_factor"]["seeded"]
    source = sources["E1"]
    columns = se.paper.energy_basis(
        bases["E1"], source.replace(phi=se.AzimuthFactor(1))
    )
    unknown = source.replace(gamma=se.FreeMoments(), phi=se.FreeMoments())
    fit = se.fit(unknown, radio_observations["E5"], basis=columns, check=False)
    assert fit.factor == pytest.approx(complex(*target["c"]), abs=1e-3)
    _check_seeded_fit(
        fit,
        np.asarray(spectra["E5"].stokes),
        {**target, "chi2_held_predictive": target["chi2_held_predictive_fixed_c"]},
        "E5 shared azimuth",
    )


def test_e6_joint_and_marginalized_field(radio, physical_radio):
    _, sources, _, spectra = radio
    archived = _study_results("cost_results.json")["joint_gamma_B"]
    truth = np.asarray(spectra["E6"].stokes)
    source = sources["E6"]
    joint, independent = physical_radio["E6"], physical_radio["E1"]
    moments = joint.moments(source)
    field_axis = joint.lattice.cells.variables.index("B")
    field_power = joint.lattice.index.arrays.powers[:, field_axis]
    # The B-zero rows integrate out the field coordinate of the electron law.
    np.testing.assert_array_equal(joint.lattice.weights, independent.lattice.weights)
    marginal = se.Moments(independent.lattice, moments.values[field_power == 0])
    for basis, values, key in (
        (joint, moments, "moment_route_72_max_rel_err"),
        (independent, marginal, "preaveraged_36_with_marginal_moments_max_rel_err"),
    ):
        error = _max_relative_error(basis.predict(values, check=False).stokes, truth)
        assert f"{error:.2g}" == f"{archived[key]:.2g}", key


def _withheld_rms(prediction, truth):
    return np.sqrt(np.mean((np.asarray(prediction) / truth - 1) ** 2, axis=-1))


def test_empirical_radio_baselines(radio, physical_radio):
    channels, sources, _, spectra = radio
    study = _study.STUDY
    laws = runpy.run_path(str(study / "spectral_laws.py"))
    noise_free = _study_results("baselines_results.json")["results"]
    sweep = _study_results("baselines_noise_sweep_results.json")
    weights = np.asarray(channels.response_weights())
    np.testing.assert_allclose(
        weights, np.broadcast_to(weights[0], weights.shape), atol=1e-15
    )
    grid = laws["channel_grid"](channels.nodes, weights[0])
    fitted, held = slice(None, None, 2), slice(1, None, 2)
    fitted_grid, held_grid = grid.rows(fitted), grid.rows(held)
    rng = np.random.default_rng(sweep["mc_seeds"]["synthetic"])
    draw = rng.standard_normal((sweep["n_real"], len(weights)))

    basis = physical_radio["E1"]
    source = sources["E1"].replace(gamma=se.FreeMoments())
    for name in ("E1", "E2", "E3", "E4-JP", "E4-KP", "E5"):
        intensity = np.asarray(spectra[name].I)
        sigma = 0.01 * intensity
        data = se.StokesData(
            channels[fitted], I=intensity[fitted], sigma=sigma[fitted], unit="cgs"
        )
        fit = se.fit(source, data, basis=basis, rank=se.Rank(4), check=False)
        rms = _withheld_rms(fit.predict(channels[held]).I, intensity[held])
        assert (
            f"{rms:.2g}"
            == f"{noise_free[name]['noise_free']['moment_R4']['rms_res_held']:.2g}"
        ), name

        for level in ("0.0001", "0.01"):
            target = sweep["results"][level]["compact_median_rms_held"][name]
            sigma = float(level) * intensity
            samples = intensity + draw * sigma
            data = se.StokesData(
                channels[fitted], I=samples[:, fitted], sigma=sigma[fitted], unit="cgs"
            )
            fit = se.fit(
                source,
                data,
                basis=basis,
                rank=se.UPRE(cap=sweep["rank_cap"]),
                check=False,
            )
            errors = {
                "moment_UPRE": _withheld_rms(
                    fit.predict(channels[held]).I, intensity[held]
                )
            }
            assert np.median(np.asarray(fit.rank)) == target["R_star_median"], (
                name,
                level,
            )

            for law in laws["LAWS"]:
                predictions = []
                for sample in samples[:, fitted]:
                    params = laws["fit"](law, fitted_grid, sample, sigma[fitted])[
                        "params"
                    ]
                    predictions.append(laws["predict"](law, params, held_grid))
                errors[law] = _withheld_rms(predictions, intensity[held])

            for model, rms in errors.items():
                median = np.median(rms)
                assert f"{median:.2g}" == f"{target[model]:.2g}", (name, level, model)


def test_e2_extrapolation_500_draws():
    training, dense = se.paper.extrapolation_channels()
    training, prediction = training[:26], dense[-1:]
    combined = se.Channels.at(
        np.r_[training.centres_hz, prediction.centres_hz],
        unit="Hz",
        half_width=0.2,
        n_nu=400,
    )
    source = se.paper.population("E2")
    regime = se.paper.regime_basis("E2", combined, check=False)
    fit_cells = se.Lattice(gamma=se.paper.cells()).select(gamma=slice(3, None))
    basis = se.paper.energy_basis(regime, source, fit_cells)
    truth = regime.predict(source, check=True)
    seed = 20261112
    draws = np.concatenate(
        (
            np.random.default_rng(seed).standard_normal((500, 21, 3)),
            np.random.default_rng(seed + 1).standard_normal((500, 11, 3)),
        ),
        axis=1,
    )[:, :26]
    with np.load(_study.ACCEPTANCE_DATA / "e2_extrapolation.npz") as archived:
        for noise in ("0.0001", "0.01"):
            sigma = float(noise) * np.asarray(truth.I[:26])
            observed = np.asarray(truth.stokes[:26]) + sigma[:, None] * draws
            data = se.StokesData(
                training, stokes=observed, sigma=sigma[:, None], unit="cgs"
            )
            for label, marker, penalty in (
                ("free", se.FreeMoments(), None),
                ("ridge", se.FreeMoments(), se.Penalty()),
                ("posridge", se.NonNegativeCells(), se.Penalty()),
                ("pos", se.NonNegativeCells(), se.Penalty(0)),
            ):
                fit = se.fit(
                    source.replace(gamma=marker),
                    data,
                    basis=basis,
                    penalty=penalty,
                    check=False,
                    chunk_bytes=2**25,
                )
                ratio = np.asarray(fit.predict(prediction).I).ravel() / float(
                    truth.I[-1]
                )
                actual = np.median(abs(ratio - 1))
                expected = np.median(abs(archived[f"{label}_{noise}"] - 1))
                assert actual == pytest.approx(expected, abs=2e-8), (noise, label)

    pin = json.loads(
        (Path(__file__).parent / "data" / "e3_narrowband.json").read_text()
    )
    legacy_truth = np.asarray(pin["truth_fit"])
    sigma = pin["noise"] * legacy_truth[:, :1]
    data = se.StokesData(
        training[:21],
        stokes=legacy_truth + sigma * draws[0, :21],
        sigma=sigma,
        unit="cgs",
    )
    fit = se.fit(source.replace(gamma=se.FreeMoments()), data, basis=basis, check=False)
    assert int(fit.rank) == pin["rank"]
    assert float(fit.predict(prediction).I[0]) / pin["truth_300"] == pytest.approx(
        pin["ratio_300"], abs=1e-7
    )
