"""E2 extrapolation with the v0.7.10 noise draws and positive-ridge outputs.

The fixture extracts the 300 GHz endpoint columns, first noisy observation
and per-channel noise from fits_positive_ridge_products.npz. The source
archive SHA-256 is d22f12806b429b380bbc5975ada7fb38476fd9388a5dbfbec9d32a9985d1ea8d.
"""

import json

import numpy as np
import pytest

import syncem as se
from tests.v080 import _study


@pytest.fixture(scope="module")
def setup():
    directory = _study.ACCEPTANCE_DATA
    _study.verify_sums(directory / "SHA256SUMS")
    with np.load(directory / "e2_extrapolation.npz") as raw:
        numbers = {key: raw[key] for key in raw.files}
    training, dense = se.paper.extrapolation_channels()
    training = training[:26]
    lattice = se.Lattice(gamma=se.paper.cells()).select(gamma=slice(3, None))
    source = se.paper.e2()
    basis = se.spectral_basis(source, training, lattice=lattice, check=False)
    return numbers, training, dense[-1:], source, basis


@pytest.mark.parametrize("count", [4, pytest.param(500, marks=pytest.mark.slow)])
def test_300_ghz_realisations(setup, count):
    numbers, training, prediction, source, basis = setup
    seed = 20261112
    a = np.random.default_rng(seed).standard_normal((500, 21, 3))
    b = np.random.default_rng(seed + 1).standard_normal((500, 11, 3))
    draws = np.concatenate((a, b), axis=1)[:count, :26]
    free = source.replace(gamma=se.FreeMoments())
    positive = source.replace(gamma=se.NonNegativeCells())
    for tag in ("0.0001", "0.01"):
        sigma = numbers[f"sigma_{tag}"]
        observations = numbers[f"data_seed0_{tag}"] + sigma[:, None] * (
            draws - draws[0]
        )
        data = se.StokesData(
            training, stokes=observations, sigma=sigma[:, None], unit="cgs"
        )
        for method, model, penalty in (
            ("free", free, None),
            ("ridge", free, se.Penalty()),
            ("posridge", positive, se.Penalty()),
            ("pos", positive, se.Penalty(0)),
        ):
            fit = se.fit(
                model,
                data,
                basis=basis,
                penalty=penalty,
                check=False,
                chunk_bytes=2**25,
            )
            ratio = (
                np.asarray(fit.predict(prediction).I).ravel() / numbers["truth_I_300"]
            )
            archived = numbers[f"{method}_{tag}"][:count]
            if count == 500:
                assert np.median(abs(ratio - 1)) == pytest.approx(
                    np.median(abs(archived - 1)), abs=1e-8
                )
            else:
                np.testing.assert_allclose(ratio, archived, rtol=0, atol=1e-7)


def test_unconstrained_at_300_ghz(setup):
    numbers, training, prediction, source, basis = setup
    saved = _study.ACCEPTANCE_DATA / "fits_extrap_noisefree_results.json"
    expected = json.loads(saved.read_text())["results"]
    unknown = source.replace(gamma=se.FreeMoments())
    for tag in ("0.0001", "0.01"):
        data = se.StokesData(
            training[:21],
            stokes=numbers[f"data_seed0_{tag}"][:21],
            sigma=numbers[f"sigma_{tag}"][:21, None],
            unit="cgs",
        )
        reference = expected[tag]["E2"]
        fit = se.fit(
            unknown, data, basis=basis, rank=se.Rank(reference["R"]), check=False
        )
        fraction = fit.predict(prediction).unconstrained[0, 0]
        assert float(fraction) == pytest.approx(
            reference["row300_norm_fraction_outside_VR"], abs=1e-8
        )
