"""Fixed-field reference for the signed G8 independence measurement."""

from dataclasses import replace

import numpy as np
import pytest

import syncem as se
from syncem.front._independence import independence_check
from syncem.front._independence_scaling import scaling_difference
from syncem.population._marginal import marginalize


@pytest.fixture(scope="module")
def energy_field():
    channels = se.Channels.log(0.8, 2, 2, unit="GHz", n_nu=3)
    g = se.Cells.log(6500, 12000, 2, order=1)
    b = se.Uniform(4, 6, nodes=4, unit="uG")

    def electron(B):
        return se.PowerLaw(2.6, lo=6500, hi=12000, column=B.value / 5)

    pop = se.Population(
        B=b,
        gamma=se.Conditional(electron, units={"B": "uG"}),
        eta=0.21,
        phi=se.AzimuthFactor(0.21 + 0.31j),
        depth=se.GaussianScreen(3.0, 0.1),
    )
    marginal, names = marginalize(pop, se.ContinuumKernel(), ("B",))
    baseline = se.Lattice(gamma=g)
    joint = se.Lattice(gamma=g, B=se.Cells.linear(4, 6, 1, order=0, unit="uG"))
    basis = se.spectral_basis(marginal, channels, lattice=baseline, check=False)
    return pop, marginal, names, basis, joint


def test_scaling_uses_joint_and_marginal_measures_against_direct_calls(energy_field):
    pop, marginal, names, basis, joint = energy_field
    difference = scaling_difference(basis, pop, names[0], 1.0)
    assert difference is not None
    correlated = replace(basis, lattice=joint).direct(pop).stokes
    independent = basis.direct(marginal).stokes
    want = np.array(independent - correlated)
    scale = np.max(correlated[:, 0])
    np.testing.assert_allclose(difference / scale, want / scale, rtol=0, atol=2e-7)
    assert np.min(np.abs(want[:, 1:])) > 0

    spectrum = basis.predict(marginal, check=False)
    measured = independence_check(spectrum, basis, pop, names, True)
    term = measured.ledger[names[0]]
    assert term.kind == "measured"
    assert term.signed and "scaling table" in term.note
    assert [x.name for x in measured.ledger.terms if x.name != names[0]] == [
        x.name for x in spectrum.ledger.terms
    ]
    np.testing.assert_array_equal(term.value, difference)


def test_scaling_budget_and_other_edges_decline(energy_field):
    pop, _, names, basis, _ = energy_field
    # The table alone fits this budget. The Cartesian field draw does not.
    assert scaling_difference(basis, pop, names[0], 1e-4) is None
    assert scaling_difference(basis, pop, "independence:gamma|eta", 1) is None
    dependent = pop.replace(
        phi=se.AzimuthFactor(se.Conditional(lambda B: B.value * 0j))
    )
    assert scaling_difference(basis, dependent, names[0], 1) is None


def test_public_predict_keeps_independence_when_raw_partition_exceeds_budget(
    energy_field, monkeypatch
):
    from syncem.front import _predictions

    pop, _, names, basis, _ = energy_field
    basis = replace(basis, forward_average=("B",))
    monkeypatch.setattr(_predictions, "estimate_direct", lambda _: float("inf"))
    measured = basis.predict(pop)
    term = measured.ledger[names[0]]
    assert measured.ledger["representation"].kind == "unbounded"
    assert term.kind == "measured"
    assert term.signed and "scaling table" in term.note
