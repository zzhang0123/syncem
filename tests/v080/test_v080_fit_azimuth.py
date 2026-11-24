"""A polarisation fit needs a declared azimuth (SPEC decision 7, review P2-6)."""

import numpy as np
import pytest

import syncem as se


@pytest.fixture(scope="module")
def setup():
    channels = se.Channels.log(0.5, 2.0, 2, unit="GHz", half_width=0.2, n_nu=8)
    known = se.Population(
        gamma=se.paper.e1().factors["gamma"], B="5 uG", phi=se.UniformAzimuth()
    )
    lattice = se.Lattice(gamma=se.paper.cells(order=0))
    basis = se.spectral_basis(known, channels, lattice=lattice, check=False)
    spectrum = basis.predict(known, check=False)
    scale = float(np.max(np.abs(spectrum.I)))
    values = dict(I=spectrum.I, Q=spectrum.I * 0.1, U=spectrum.I * 0.05)
    return channels, lattice, basis, spectrum.unit, scale, values


def test_polarised_fit_without_azimuth_is_refused(setup):
    channels, lattice, basis, unit, scale, values = setup
    data = se.StokesData(channels, **values, unit=unit, sigma=0.01 * scale)
    undeclared = se.Population(gamma=se.FreeMoments(), B="5 uG")
    with pytest.raises(se.RoleError, match="azimuth 'phi' declared"):
        se.fit(undeclared, data, lattice=lattice, check=False)
    with pytest.raises(se.RoleError, match="azimuth 'phi' declared"):
        se.problem(undeclared, data, basis=basis, check=False)
    declared = undeclared.replace(phi=se.UniformAzimuth())
    se.fit(declared, data, basis=basis, check=False)


def test_intensity_fit_keeps_the_default_azimuth(setup):
    channels, lattice, basis, unit, scale, values = setup
    data = se.StokesData(channels, I=values["I"], unit=unit, sigma=0.01 * scale)
    se.fit(se.Population(gamma=se.FreeMoments(), B="5 uG"), data, basis=basis, check=False)


def test_masked_polarisation_rows_do_not_count_as_observed(setup):
    channels, lattice, basis, unit, scale, values = setup
    mask = np.array([[True, False, False], [True, False, False]])
    data = se.StokesData(channels, **values, unit=unit, sigma=0.01 * scale, mask=mask)
    se.fit(se.Population(gamma=se.FreeMoments(), B="5 uG"), data, basis=basis, check=False)


@pytest.mark.parametrize("method", [se.fit, se.problem])
@pytest.mark.parametrize(
    "response, mask, polarised",
    [
        ([[1, 0, 0, 0], [0, 0, 1, 0]], [True, True], False),
        ([[0, 1, 0, 0], [0, 0, 1, 0]], [True, False], True),
        ([[1, 0, 0, 0], [0, 0, 1, -0.3]], [False, True], True),
        ([[1, 0, 0, 0], [0, 0, 1, -0.3]], [True, False], False),
    ],
)
def test_response_in_the_observed_rows_sets_the_azimuth_requirement(
    setup, method, response, mask, polarised
):
    channels, _, basis, unit, scale, values = setup
    data = se.StokesData(
        channels,
        stokes=values["I"],
        components=("I", "Q"),
        response=response,
        mask=mask,
        sigma=scale,
        unit=unit,
    )
    unknown = se.Population(gamma=se.FreeMoments(), B="5 uG")
    if polarised:
        with pytest.raises(se.RoleError, match="azimuth 'phi' declared"):
            method(unknown, data, basis=basis, check=False)
    else:
        method(unknown, data, basis=basis, check=False)
