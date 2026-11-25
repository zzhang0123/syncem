"""Fits of physical families in the full and retained two-regime likelihoods."""

from dataclasses import replace

import numpy as np
import pytest

import syncem as se
from syncem.front._regime_family import fit_regime_family
from syncem.values.errors import DataError


@pytest.fixture(scope="module")
def experiment():
    field = "30 uG"
    population = se.Population(
        gamma=se.PowerLaw(2.6, lo=120, hi=5000, pivot=800, amplitude=0.08),
        B=field,
        mu=se.Isotropic(),
        eta=0.2,
        phi=se.AzimuthFactor(0.55 + 0.1j),
        depth=se.NoRotation(),
    )
    channels = se.Channels.log(2, 600, 10, unit="MHz", half_width=0.1, n_nu=24)
    basis = se.RegimeBasis.build(
        population,
        channels,
        gamma_switch=300,
        harmonic=se.Lattice(
            gamma=se.Cells.log(120, 300, 3, order=2), phi=se.AzimuthHarmonic()
        ),
        continuum=se.Lattice(
            nu_c=se.Cells.log((5, "MHz"), (4, "GHz"), 16, order=2),
            phi=se.AzimuthHarmonic(),
        ),
        reference_field=field,
        angular_nodes=8,
        moment_nodes=12,
        check=False,
    )
    return basis, population


def _data(basis, truth):
    spectrum = basis.predict(truth, check=False).to("Jy/sr")
    return se.StokesData(
        spectrum.channels,
        stokes=spectrum.stokes,
        sigma=0.03 * spectrum.I[:, None],
        unit=spectrum.unit,
    )


def _initial(name):
    bounds = (120, 5000)
    if name == "E1":
        return se.PowerLaw(
            se.Free(2.3, bounds=(1, 5)), lo=bounds[0], hi=bounds[1], pivot=800
        ), {"index": 2.6}
    if name == "E2":
        return se.BrokenPowerLaw(
            se.Free(1.75, bounds=(0.5, 4)),
            se.Free(3.5, bounds=(1, 5)),
            se.Free(900, bounds=(300, 2400), log=True),
            smoothness=2,
            lo=bounds[0],
            hi=bounds[1],
            pivot=800,
        ), {"index1": 1.8, "index2": 3.3, "gamma_break": 820.0}
    if name == "E3":
        return se.PowerLawMixture.cumulant_form(
            se.Free(2.5, bounds=(1, 5)),
            se.Free(0.025, bounds=(-0.2, 0.2)),
            lo=bounds[0],
            hi=bounds[1],
            pivot=800,
        ), {"k1": 2.65, "k2": 0.04}
    return se.AgedPowerLaw(
        se.Free(2.1, bounds=(1.2, 5)),
        gamma_cut=se.Free(2400, bounds=(1200, 3900), log=True),
        lo=bounds[0],
        pivot=800,
    ), {"index": 2.3, "explicit_cut": 2170.0}


@pytest.mark.parametrize("name", ["E1", "E2", "E3", "E4-JP"])
def test_recover_family_parameters_and_physical_covariance(experiment, name):
    basis, source = experiment
    initial, shape = _initial(name)
    parameters = {"amplitude": 0.08, **shape}
    truth = source.replace(gamma=initial._at(parameters))
    data = _data(basis, truth).withhold([0])
    fitted = fit_regime_family(basis, source.replace(gamma=initial), data)
    for key, expected in parameters.items():
        parameter = fitted.params[f"gamma.{key}"]
        assert parameter.value == pytest.approx(expected, rel=2e-6)
        assert parameter.identified and parameter.sigma > 0
    assert fitted.rank == len(parameters)
    assert fitted.dof == 27 - fitted.rank
    assert fitted.withheld.n == 3
    assert fitted.withheld.chi2 < 1e-10
    assert fitted.chi2 < 1e-10
    assert fitted.predict().unit == "Jy/sr"
    assert np.all(np.linalg.eigvalsh(fitted.parameter_covariance) > 0)
    np.testing.assert_allclose(fitted.predict().stokes, data.stokes, rtol=1e-6, atol=0)
    np.testing.assert_allclose(
        fitted.electron_spectrum(np.array([350, 900, 1100])),
        truth.factors["gamma"].density(np.array([350, 900, 1100])),
        rtol=2e-6,
        atol=0,
    )
    assert fitted.column.value > 0


def test_compressed_fit_recovers_the_full_likelihood_solution(experiment):
    basis, source = experiment
    raw = _data(basis, source)
    data = se.StokesData(
        raw.channels,
        stokes=raw.stokes
        + np.asarray(raw.sigma)
        * np.random.default_rng(2026).normal(size=raw.stokes.shape),
        sigma=raw.sigma,
        unit=raw.unit,
    )
    initial, _ = _initial("E1")
    source = source.replace(gamma=initial)
    direct = fit_regime_family(basis, source, data)
    moments = basis.fit(data, rank=se.FullRank(), keep_directions=True)
    compressed = fit_regime_family(basis, source, moments)
    for key in direct.params:
        assert compressed.params[key].value == pytest.approx(
            direct.params[key].value, rel=2e-7
        )
        assert compressed.params[key].sigma == pytest.approx(
            direct.params[key].sigma, rel=2e-7
        )
    assert compressed.dof == int(moments.rank) - len(compressed.params)
    np.testing.assert_allclose(
        compressed.predict().stokes, direct.predict().stokes, rtol=2e-7, atol=0
    )

    unretained = basis.fit(data, rank=se.FullRank())
    with pytest.raises(DataError, match="keep_directions=True"):
        fit_regime_family(basis, source, unretained)
    wrong = replace(basis, continuum=basis.harmonic)
    with pytest.raises(DataError, match="original branch moment coordinates"):
        fit_regime_family(wrong, source, moments)


def test_fixed_and_bounded_amplitude_only_fits(experiment):
    basis, truth = experiment
    data = _data(basis, truth)
    fixed = fit_regime_family(basis, truth, data)
    assert fixed.kind == "fixed"
    assert fixed.rank == 0 and fixed.dof == 30
    assert fixed.chi2 < 1e-20
    initial = truth.factors["gamma"]._at(
        {"amplitude": se.Free(0.04, bounds=(0.02, 0.06))}
    )
    limited = fit_regime_family(basis, truth.replace(gamma=initial), data)
    assert limited.params["gamma.amplitude"].value == 0.06
    assert limited.params["gamma.amplitude"].at_bound
    assert limited.chi2 > 10


def test_retained_subset_and_family_artifact(experiment, tmp_path):
    basis, truth = experiment
    spectrum = basis.predict(truth, check=False)
    subset = slice(None, None, 2)
    data = se.StokesData(
        basis.channels[subset],
        stokes=spectrum.stokes[subset],
        sigma=0.01 * spectrum.I[subset, None],
        unit="cgs",
    )
    linear = basis.fit(data, rank=se.FullRank(), keep_directions=True)
    fit = basis.fit_family(truth, linear)
    assert fit.chi2 < 1e-20
    assert "Observation units  cgs" in str(fit)
    path = tmp_path / "two-regime-family.npz"
    fit.save(path)
    assert path.stat().st_size > 0


@pytest.mark.parametrize("reverse", [False, True])
def test_withheld_rank_and_source_column_per_pixel(experiment, reverse):
    basis, truth = experiment
    spectrum = basis.predict(truth, check=False)
    y = np.stack((spectrum.stokes, 2 * spectrum.stokes))
    mask = np.ones_like(y, dtype=bool)
    mask[0, 0] = False
    selection = slice(None, None, -1 if reverse else 1)
    data = se.StokesData(
        basis.channels,
        stokes=y[selection],
        mask=mask[selection],
        sigma=0.02 * spectrum.I[:, None],
        unit="cgs",
    ).withhold([0])
    law = truth.factors["gamma"]._at({"amplitude": se.Free(0.04, bounds=(0.001, 1))})
    fit = se.fit(truth.replace(gamma=law), data, basis=basis, check=False)
    np.testing.assert_array_equal(fit.withheld.n, np.array([0, 3])[selection])
    assert np.max(fit.withheld.chi2) < 1e-16
    np.testing.assert_allclose(
        fit.source_column.sigma,
        fit.params["gamma.amplitude"].sigma
        * truth.factors["gamma"].integrated_column()
        / 0.08,
    )
