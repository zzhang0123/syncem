"""Independent quadrature mixtures for explicitly dropped population edges."""

import numpy as np
import pytest
import syncem as se

from syncem.base.continuum import ContinuumKernel
from syncem.distributions import AzimuthFactor, Conditional, Free, PowerLaw, Uniform
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.population import FreeMoments, Population, Samples, pair
from syncem.population._marginal import marginalize
from syncem.population.declaration import law_at, parents
from syncem.values.errors import RoleError

KERNEL = ContinuumKernel()


def test_public_marginal_prediction_measures_the_removed_edge():
    @se.kernel(variables=(se.Variable("x"), se.Variable("y")), stokes=("I",))
    def model(nu, x, y):
        return {"I": x * y + 0 * nu}

    pop = se.Population(x=se.Uniform(1, 3, nodes=4), y=lambda x: x)
    ch = ([1.0, 2.0], "GHz")
    lattice = se.Lattice(y=se.Cells.linear(1, 3, 1, order=1))
    result = se.predict(pop, ch, lattice=lattice, kernel=model, average=("x",))
    np.testing.assert_allclose(result.I, 4.0)
    term = result.ledger["independence:y|x"]
    assert term.kind == "measured"
    np.testing.assert_allclose(term.value, -1 / 3)


def test_marginal_is_independent_and_retains_conditional_columns():
    field = Uniform(4.0, 6.0, unit="uG", nodes=2)

    def electrons(B):
        return PowerLaw(2.6, lo=500, hi=5000, column=(B.to("uG").value, "cm-2"))

    original = 2 * Population(B=field, eta=0.0, gamma=electrons)
    population, assumptions = marginalize(original, KERNEL, ("B",))

    assert assumptions == ("independence:gamma|B",)
    assert len(population.parts) == 2
    for part in population.parts:
        np.testing.assert_allclose(part.factors["B"].rule(), field.rule())
    assert all(not parents(part.factors["gamma"]) for part in population.parts)
    lattice = Lattice(gamma=Cells.log(500, 5000, 4, order=0))
    energy = [
        part.column_factor * Population(gamma=part.factors["gamma"])
        for part in population.parts
    ]
    assert float(lattice.column(pair(energy[0] + energy[1], lattice))) == pytest.approx(
        10.0, rel=1e-10
    )


def test_keeps_remaining_parent_declared_unit_parity_and_params():
    population = Population(
        B="5 uG",
        eta=Uniform(-0.9, 0.9),
        gamma=Conditional(
            lambda B, eta, scale: 400.0 + B.value * scale * eta**2,
            params={"scale": Free(2.0)},
            units={"B": "uG"},
            parity={"eta": "even"},
        ),
    )

    changed, assumptions = marginalize(population, KERNEL, ("B",))
    law = changed.at({"gamma.scale": 4.0}).factors["gamma"]

    assert assumptions == ("independence:gamma|B",)
    assert law.parents == ("eta",)
    assert law.parity == (("eta", "even"),)
    assert law_at(law, {"eta": -0.5}) == pytest.approx(405.0)


def test_marginalizes_a_conditional_spin_factor_without_splitting_phi():
    population = Population(
        B=Uniform(4.0, 6.0, unit="uG", nodes=4),
        eta=0.0,
        gamma=1500,
        phi=AzimuthFactor(Conditional(lambda B: B.value * 0.05j, units={"B": "uG"})),
    )
    changed, assumptions = marginalize(population, KERNEL, ("B",))

    assert assumptions == ("independence:phi|B",)
    assert len(changed.parts) == 1
    assert complex(changed.factors["phi"].circular(2)) == pytest.approx(0.25j)


def test_edges_to_siblings_get_separate_parent_draws():
    population = Population(
        eta=Uniform(-1.0, 1.0, nodes=2),
        B=lambda eta: (5.0 + eta, "uG"),
        gamma=lambda eta: 600 + 80 * eta,
    )
    changed, assumptions = marginalize(population, KERNEL, ("eta",))
    assert assumptions == ("independence:B|eta", "independence:gamma|eta")
    assert len(changed.parts) == 4
    np.testing.assert_allclose([p.column_factor for p in changed.parts], 0.25)
    assert len({(p.factors["B"][0], p.factors["gamma"]) for p in changed.parts}) == 4


@pytest.mark.parametrize(
    "population",
    (
        Population(eta=FreeMoments(support=(-1, 1)), gamma=lambda eta: 500 + eta),
        Population(eta=lambda B: 0.1, B="5 uG", gamma=lambda eta: 500 + eta),
        Population(Samples(eta=[0.1, 0.2]), gamma=lambda eta: 500 + eta),
        Population(phi=AzimuthFactor(0.4), gamma=lambda phi: 400),
    ),
)
def test_an_unavailable_parent_marginal_is_refused(population):
    name = "phi" if "phi" in population.factors else "eta"
    with pytest.raises(RoleError, match=f"specified independent marginal.*{name}"):
        marginalize(population, KERNEL, (name,))


def test_work_budget_and_no_effect_declaration():
    population = Population(eta=Uniform(-1, 1), gamma=lambda eta: 500 + eta)
    with pytest.raises(RoleError, match=r"limit is 8"):
        marginalize(population, KERNEL, ("eta",), max_components=8)
    changed, assumptions = marginalize(population, KERNEL, ("depth",), max_components=1)
    assert changed is population
    assert assumptions == ()
