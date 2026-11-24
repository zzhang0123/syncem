"""Classical parameter coverage from the v0.7.10 study, SPEC Sec. 9.6.

The fixture extracts rank and the parameter and standard-error columns of
the comp_Rstar and full blocks from params_realisations.npz (SHA-256
201a3175155d792e0ba80ba61868498e09001e0bc03c6e9a458f80f38d789d51).
The original block layout and noise seeds are in classical-models/params.py.
To run one 1,000-draw case at a time, select the case by name, for example
``python -m pytest tests/acceptance/test_parameter_coverage.py -m slow -k E4-JP``.
"""

import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se
from tests.v080 import _study

_CASES = ("E1", "E2_s2", "E3", "E4-JP")
_ROUTES = ("comp_Rstar", "full")
_COUNT = 1000
_SEED = 20260928
_SUMMARY_SHA = "bb6ef5367b20899d3eeb6e1f7a19eb31b85b43cd1d09281598d6725087a15945"


@pytest.fixture(scope="module")
def archive():
    directory = _study.ACCEPTANCE_DATA
    _study.verify_sums(directory / "SHA256SUMS")
    summary = _study.STUDY / "params_results.json"
    assert _study.sha256_file(summary) == _SUMMARY_SHA
    tasks = json.loads(summary.read_text())["tasks"]
    verifier = json.loads((directory / "verify_params_results.json").read_text())
    offset = verifier["E4JP_truth"]["ln(a1<B_perp>)"]
    with np.load(directory / "parameter_coverage.npz", allow_pickle=False) as pins:
        cases = {
            name: {
                key: pins[f"{name.replace('-', '_')}_{key}"]
                for key in ("rank", *_ROUTES)
            }
            for name in _CASES
        }
    return tasks, cases, offset


def _covered(phi, sigma, truth):
    return np.abs(phi - truth) < sigma


def test_archived_parameter_coverage(archive):
    tasks, cases, _ = archive
    coverage = {route: [] for route in _ROUTES}
    for name in _CASES:
        task = tasks[name]
        truth = np.asarray(task["reference_point"]["phi"])
        assert cases[name]["rank"].shape == (_COUNT,)
        for route in _ROUTES:
            rows = cases[name][route]
            assert rows.shape == (_COUNT, 2 * truth.size)
            phi, sigma = np.split(rows, 2, axis=-1)
            assert np.all(np.isfinite(phi) & np.isfinite(sigma) & (sigma > 0))
            fraction = np.mean(_covered(phi, sigma, truth), axis=0)
            reference = task["realisations"][route]["params_vs_reference"]
            assert reference["n"] == _COUNT
            np.testing.assert_array_equal(
                fraction, reference["coverage_|pull|<1"], err_msg=f"{name} {route}"
            )
            coverage[route].extend(fraction)
    assert (min(coverage["comp_Rstar"]), max(coverage["comp_Rstar"])) == (0.657, 0.691)
    assert (min(coverage["full"]), max(coverage["full"])) == (0.656, 0.687)


@pytest.fixture(scope="module")
def spectral_models():
    population = se.paper.e1()
    cells = se.paper.cells()
    lattice = se.Lattice(gamma=cells)
    channels = se.paper.channels()
    basis = se.spectral_basis(population, channels, lattice=lattice, check=False)
    p = se.Free(2.6, bounds=(1.1, 5.0))
    limits = {"lo": float(cells.edges[0]), "hi": float(cells.edges[-1])}
    laws = {
        "E1": se.PowerLaw(p, **limits),
        "E2_s2": se.BrokenPowerLaw(
            se.Free(2.0, bounds=(0, 7)),
            se.Free(3.2, bounds=(0, 7)),
            se.Free(7800, bounds=(100, 3e5), log=True),
            **limits,
        ),
        "E3": se.PowerLawMixture.cumulant_form(
            p, se.Free(0.04, bounds=(-0.5, 0.5)), **limits
        ),
        "E4-JP": se.AgedPowerLaw(
            p, gamma_cut=se.Free(19000, bounds=(400, 1e6), log=True), lo=limits["lo"]
        ),
    }
    models = {name: population.replace(gamma=law) for name, law in laws.items()}
    free = population.replace(gamma=se.FreeMoments())
    return basis, free, models


def _paper_coordinates(fit, jp_offset):
    law = fit.electron_spectrum()
    names = tuple(key.split(".", 1)[1] for key in fit.params)

    def coordinates(x):
        model = law._at(dict(zip(names, x, strict=True)))
        values = [jnp.log(model.paper_coefficient().value)]
        for name, value in zip(names, x, strict=True):
            if name == "amplitude":
                continue
            if name == "gamma_break":
                value = jnp.log(value)
            elif name == "explicit_cut":
                value = jp_offset + 2 * jnp.log(value)
            values.append(value)
        return jnp.asarray(values)

    return jax.jit(coordinates), jax.jit(jax.jacfwd(coordinates))


@pytest.mark.parametrize("name", _CASES)
@pytest.mark.parametrize("count", [4, pytest.param(_COUNT, marks=pytest.mark.slow)])
def test_parameter_coverage(name, count, archive, spectral_models):
    tasks, cases, jp_offset = archive
    basis, free, models = spectral_models
    fitted = np.arange(0, 41, 2)
    channels = basis.channels[fitted]
    spectrum = np.asarray(tasks[name]["S_direct_IQU"])[fitted]
    noise = 0.01 * spectrum[:, :1]
    truth = np.asarray(tasks[name]["reference_point"]["phi"])
    pins = cases[name]
    estimates = {route: [] for route in _ROUTES}
    transform = None
    for draw in range(count):
        rng = np.random.default_rng([_SEED, 1, draw])
        observation = spectrum + noise * rng.normal(size=(41, 3))[fitted]
        data = se.StokesData(channels, stokes=observation, sigma=noise, unit="cgs")
        linear = se.fit(free, data, basis=basis, keep_directions=True, check=False)
        assert int(linear.rank) == pins["rank"][draw], f"{name} draw {draw}"
        targets = {"comp_Rstar": linear.retained, "full": data}
        for route, target in targets.items():
            fit = se.fit(models[name], target, basis=basis, check=False)
            if transform is None:
                transform = _paper_coordinates(fit, jp_offset)
            coordinates, jacobian = transform
            physical = jnp.asarray(
                [parameter.value for parameter in fit.params.values()]
            )
            phi = np.asarray(coordinates(physical))
            jac = np.asarray(jacobian(physical))
            sigma = np.sqrt(np.diag(jac @ np.asarray(fit.parameter_covariance) @ jac.T))
            estimates[route].append(np.concatenate((phi, sigma)))

    for route in _ROUTES:
        phi, sigma = np.split(np.asarray(estimates[route]), 2, axis=-1)
        phi_ref, sigma_ref = np.split(pins[route][:count], 2, axis=-1)
        label = f"{name} {route}"
        np.testing.assert_array_less(
            abs(phi - phi_ref) / sigma_ref, 0.01, err_msg=label
        )
        np.testing.assert_allclose(sigma, sigma_ref, rtol=0.003, atol=0, err_msg=label)
        np.testing.assert_array_equal(
            _covered(phi, sigma, truth),
            _covered(phi_ref, sigma_ref, truth),
            err_msg=label,
        )

    if count == _COUNT:
        comp, full = (
            np.asarray(estimates[route])[:, truth.size :] for route in _ROUTES
        )
        ratio = np.median(comp / full, axis=0)
        np.testing.assert_allclose(ratio, 1.0, rtol=0.003, atol=0)
