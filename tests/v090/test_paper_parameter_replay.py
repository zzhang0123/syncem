"""Replay the original full and retained parameter campaigns through both regimes."""

import pytest

import syncem as se
from tests.acceptance import test_parameter_coverage as reference

pytestmark = pytest.mark.slow


@pytest.fixture(scope="module")
def two_part():
    metadata = reference.archive.__wrapped__()
    _, free, models = reference.spectral_models.__wrapped__()
    models = {
        name: model.replace(gamma=model.factors["gamma"]._at({"lo": 100.0}))
        for name, model in models.items()
    }
    channels = se.paper.channels()
    regime = se.paper.regime_basis("E1", channels, check=False)
    basis = se.paper.energy_basis(regime, "E1")
    return metadata, (basis, free, models)


@pytest.mark.parametrize("count", [4, 1000])
@pytest.mark.parametrize("name", ["E1", "E2_s2", "E3", "E4-JP"])
def test_original_parameter_pins(name, count, two_part):
    metadata, (basis, free, models) = two_part
    reference.test_parameter_coverage(name, count, metadata, (basis, free, models))
