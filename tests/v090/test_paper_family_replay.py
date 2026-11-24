"""Replay the archived family extrapolation pins with the two-part design."""

import numpy as np
import pytest

import syncem as se
from tests.acceptance import test_family_extrapolation as reference

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def two_part():
    data, noise, laws, _, _ = reference.campaign.__wrapped__()
    laws = {name: law._at({"lo": 100.0}) for name, law in laws.items()}
    training, dense = se.paper.extrapolation_channels()
    channels = se.Channels.at(
        np.r_[training.centres_hz[:26], dense.centres_hz[-1]],
        unit="Hz",
        half_width=0.2,
        n_nu=400,
    )
    regime = se.paper.regime_basis("E1", channels, check=False)
    columns = se.paper.energy_basis(regime, "E1")
    return (
        data,
        noise,
        laws,
        {21: columns.at(training[:21]), 26: columns.at(training[:26])},
        columns.at(dense[-1:]),
    )


@pytest.mark.parametrize("count", [4, 500])
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
def test_original_family_pins(name, band, level, count, two_part):
    reference.test_family_extrapolation(name, band, level, count, two_part)
