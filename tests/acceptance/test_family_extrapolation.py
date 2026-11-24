"""Archived 300 GHz eSBPL and cumulant fits in SPEC Sec. 9.6.

The compact fixture copies endpoint ratios, chi-square values and the first
high-noise observation. The v0.7.10 source archives and SHA-256 digests are
fits_family_noise_products.npz: b63e11111c0bf549b59715c8572bacffbe771e545c39bb73f72586a385cfc565
fits_mixture_forms_products.npz: 9171747a499526d83b77aed0a70993ae57586255188074fb2e271a2bdd9cffb3
"""

import numpy as np
import pytest

import syncem as se
from tests.v080 import _study


@pytest.fixture(scope="module")
def campaign():
    folder = _study.ACCEPTANCE_DATA
    _study.verify_sums(folder / "SHA256SUMS")
    with np.load(folder / "family_extrapolation.npz") as archive:
        data = {key: archive[key] for key in archive}
    seed = 20261112
    noise = np.concatenate(
        (
            np.random.default_rng(seed).standard_normal((500, 21, 3)),
            np.random.default_rng(seed + 1).standard_normal((500, 11, 3)),
        ),
        axis=1,
    )
    train, dense = se.paper.extrapolation_channels()
    cells = se.paper.cells()
    lattice = se.Lattice(gamma=cells)
    columns = se.spectral_basis(se.paper.e2(), train[:26], lattice=lattice, check=False)
    limits = {"lo": float(cells.edges[0]), "hi": float(cells.edges[-1])}
    laws = {
        "E2": se.BrokenPowerLaw(
            se.Free(2.1, bounds=(0, 7)),
            se.Free(3.1, bounds=(0, 7)),
            se.Free(7000, bounds=(100, 3e5), log=True),
            **limits,
        ),
        "E3": se.PowerLawMixture.cumulant_form(
            se.Free(2.6, bounds=(0.5, 6)),
            se.Free(0.0, bounds=(-1, 1)),
            **limits,
        ),
    }
    output = columns.at(dense[-1:])
    return data, noise, laws, {21: columns.at(train[:21]), 26: columns}, output


@pytest.mark.parametrize("count", [4, pytest.param(500, marks=pytest.mark.slow)])
@pytest.mark.parametrize(
    "name,band,level",
    [
        ("E2", "1-100", "0.0001"),
        ("E2", "1-100", "0.01"),
        ("E2", "1-40", "0.01"),
        ("E3", "1-100", "0.0001"),
        ("E3", "1-100", "0.01"),
    ],
)
def test_family_extrapolation(name, band, level, count, campaign):
    data, noise, laws, bases, output = campaign
    nr = 21 if band == "1-40" else 26
    columns = bases[nr]
    source = se.paper.e1().replace(gamma=laws[name])
    key = f"{name}_{band}"
    sig_high = data[f"{key}_sigma"]
    sigma = sig_high * (float(level) / 0.01)
    truth = data[f"{key}_data"] - sig_high[:, None] * noise[0, :nr]
    ref_ratio = data[f"{key}_{level}_ratio"][:count]
    ref_chi2 = data[f"{key}_{level}_chi2"][:count]
    ratios, chi2 = [], []
    for sample in truth + sigma[:, None] * noise[:count, :nr]:
        observed = se.StokesData(
            columns.channels, stokes=sample, sigma=sigma[:, None], unit="cgs"
        )
        result = se.fit(source, observed, basis=columns, check=False)
        predicted = output.predict(result.moments, check=False)
        ratios.append(float(predicted.I[0]) / data[f"{name}_truth"])
        chi2.append(float(result.chi2))
    ratios = np.asarray(ratios)
    np.testing.assert_allclose(ratios, ref_ratio, rtol=0, atol=1e-5)
    np.testing.assert_allclose(chi2, ref_chi2, rtol=0, atol=5e-7)
    assert np.median(abs(ratios - 1)) == pytest.approx(
        np.median(abs(ref_ratio - 1)), abs=1e-6
    )
