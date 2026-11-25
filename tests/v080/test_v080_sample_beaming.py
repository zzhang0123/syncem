"""Atomic angular measures retain their declared T19 reference."""

import numpy as np
import pytest

import syncem as se
from tests.v080.test_v080_beaming_diagonal import _projected, _simple


def projector(sample, look, order):
    pitch = np.polynomial.legendre.legvander(sample, order)
    viewing = np.polynomial.legendre.legvander(look, order)
    return np.sum((2 * np.arange(order + 1) + 1) * pitch * viewing, axis=-1)


def modes(look):
    value = 2 + look + look**2 / 2
    return np.array([value, 0.3 * value, -0.15 * value, look]).T


@pytest.mark.parametrize("degree", [None, 0, 4, 8])
def test_fixed_sightline_pairs_the_sample_mass_at_the_requested_degree(degree):
    pitch = np.array([-0.4, 0.8])
    look = 0.25
    order = 8 if degree is None else degree
    lat = se.Lattice(**({} if degree is None else {"pitch": se.Legendre(degree)}))
    weights = np.array([1.0, 3.0])
    source = se.Population(
        se.Samples(pitch=pitch, weights=weights, column=7), look=look
    )
    basis = se.spectral_basis(
        source, ([1.0], "GHz"), kernel=_simple, lattice=lat, check=False
    )
    expected = (
        7 * np.average(projector(pitch, look, order), weights=weights) * modes(look)
    )
    predicted = basis.predict(source)
    for value in (predicted, basis.direct(source)):
        np.testing.assert_allclose(value.stokes[0], expected, rtol=0, atol=1e-12)
        term = value.ledger["representation:angular"]
        assert term.kind == "unbounded" and term.value is None
        assert any(f"Legendre({order})" in a.label for a in value.ledger.assumptions)
    np.testing.assert_allclose(predicted.ledger["representation"].value, 0, atol=1e-12)
    if degree is None:
        with pytest.raises(se.SpecifiedMismatch):
            basis.predict(se.Population(se.Samples(pitch=-pitch), look=look))
    else:
        continuous = se.Population(pitch=se.Isotropic(), look=look)
        reference = se.spectral_basis(
            continuous, ([1.0], "GHz"), kernel=_simple, lattice=lat, check=False
        )
        np.testing.assert_allclose(
            reference.predict(source).stokes[0], expected, atol=1e-12
        )
        assert (
            reference.direct(source).ledger["representation:angular"].kind
            == "unbounded"
        )
        for fn in (basis.predict, basis.direct):
            changed = fn(continuous)
            np.testing.assert_allclose(changed.stokes[0], modes(look), atol=2e-13)
            assert "representation:angular" not in {
                term.name for term in changed.ledger.terms
            }


@pytest.mark.parametrize("cell_look", [False, True])
def test_empirical_sum_through_a_view_density_checks_cell_ownership(cell_look):
    nodes = np.array([-0.8, -0.3, 0.1, 0.8])
    weights = np.array([1.0, 2.0, 3.0, 4.0])
    source = se.Population(
        se.Samples(pitch=nodes, weights=weights), look=se.Uniform(-0.5, 0.5)
    )
    lat = (
        se.Lattice(look=se.Cells.linear(0, 1, 1, order=0), pitch=se.Legendre(4))
        if cell_look
        else se.Lattice()
    )
    basis = se.spectral_basis(
        source, ([1.0], "GHz"), kernel=_simple, lattice=lat, check=False
    )
    values = (2 * weights / weights.sum())[:, None] * modes(nodes)
    direct = basis.direct(source)
    np.testing.assert_allclose(direct.stokes[0], values[1:3].sum(axis=0), atol=1e-13)
    if cell_look:
        np.testing.assert_allclose(
            basis.direct(source, within="cells").stokes[0], values[2], atol=1e-13
        )
    else:
        np.testing.assert_allclose(
            basis.predict(source).stokes, direct.stokes, atol=1e-13
        )
    assert "representation:angular" not in {term.name for term in direct.ledger.terms}


def test_joint_sample_keeps_each_angle_pair_and_reports_the_finite_projection():
    pitch = np.array([-0.3, 0.8])
    look = np.array([0.25, 0.7])
    weights = np.array([1.0, 3.0])
    pop = se.Population(se.Samples(pitch=pitch, look=look, weights=weights))
    lat = se.Lattice(pitch=se.Legendre(2), look=se.Legendre(4))
    basis = se.spectral_basis(
        pop, ([1.0], "GHz"), kernel=_simple, lattice=lat, check=False
    )
    expected = np.average(
        projector(pitch, look, 2)[:, None] * modes(look), weights=weights, axis=0
    )
    for spec in (basis.direct(pop), basis.predict(pop)):
        np.testing.assert_allclose(spec.stokes[0], expected, rtol=0, atol=1e-13)
        assert spec.ledger["representation:angular"].kind == "unbounded"


@pytest.mark.parametrize("point", [True, False])
def test_conditional_viewing_law_can_return_a_point_or_a_density(point):
    nodes = np.array([-0.4, 0.8])
    weights = np.array([1.0, 3.0])

    def view(pitch):
        angle = 0.25 * pitch.value
        return se.Delta(angle) if point else se.Uniform(-0.9, 0.9 + 0.1 * angle)

    source = se.Population(
        se.Samples(pitch=nodes, weights=weights), look=se.Conditional(view)
    )
    basis = se.spectral_basis(
        source, ([1.0], "GHz"), kernel=_simple, lattice=se.Lattice(), check=False
    )
    look = 0.25 * nodes if point else nodes
    factor = projector(nodes, look, 8) if point else 2 / (1.8 + 0.025 * nodes)
    expected = np.average(factor[:, None] * modes(look), weights=weights, axis=0)
    for value in (basis.direct(source), basis.predict(source)):
        np.testing.assert_allclose(value.stokes[0], expected, rtol=0, atol=1e-12)
        assert (
            "representation:angular" in {term.name for term in value.ledger.terms}
        ) == point


def test_default_isotropic_view_density_keeps_the_prediction_ledger_exact():
    source = se.Population(
        se.Samples(mu=[-0.5, 0.4], weights=[1.0, 3.0]), B="5 uG", gamma=1e4
    )
    basis = se.spectral_basis(
        source,
        ([1.0], "GHz"),
        kernel=se.ContinuumKernel(),
        lattice=se.Lattice(),
        check=False,
    )
    explicit = basis.direct(source.replace(eta=se.Isotropic()))
    for value in (basis.predict(source), basis.direct(source)):
        np.testing.assert_allclose(value.stokes, explicit.stokes, rtol=1e-12, atol=0)
        assert "representation:angular" not in {
            term.name for term in value.ledger.terms
        }


def test_constant_pitch_density_keeps_its_conditional_on_the_fixed_view_diagonal():
    def nuisance(pitch):
        return se.Delta(0.6 * pitch.value)

    look = 0.25
    pop = se.Population(
        pitch=se.Isotropic(), look=look, nuisance=se.Conditional(nuisance)
    )
    basis = se.spectral_basis(
        pop,
        ([1.0], "GHz"),
        kernel=_projected,
        lattice=se.Lattice(pitch=se.Legendre(0)),
        check=False,
    )
    value = 3 + look + 0.5 * look**2 + (0.6 * look) ** 2
    expected = [value, 0.4 * value, 0.1 * value, look * 0.6 * look]
    for spec in (basis.predict(pop), basis.direct(pop)):
        np.testing.assert_allclose(spec.stokes[0], expected, atol=1e-13)


@se.kernel(
    variables=(
        se.Variable("pitch", kind="cosine"),
        se.Variable("z"),
        se.Variable("look", kind="cosine"),
        se.Variable("x", kind="cosine"),
    ),
    capabilities=(se.Beaming("pitch", "look"),),
    stokes=("I", "P", "V"),
)
def _child_emission(nu, pitch, z, look, x):
    i = nu / 1e9 * (2 + z**2 + (pitch - look) ** 2)
    return {"I": i, "P": (0.3 - 0.1j) * i, "V": nu / 1e9 * z}


def _physical_child(pitch):
    return se.Delta(0.6 * pitch.value)


def _child_stokes(look):
    z = 0.6 * look
    i = 2 + z**2
    return np.array([i, 0.3 * i, -0.1 * i, z]).T


@pytest.mark.parametrize("x", [0.0, 0.5])
@pytest.mark.parametrize("axes", [(), ("pitch",), ("x", "pitch"), ("pitch", "x")])
def test_conditional_sightline_projects_the_physical_child_at_each_parent(x, axes):
    def view(x):
        return se.Delta(0.25 + 0.2 * x.value)

    source = se.Population(
        x=x,
        z=se.Conditional(_physical_child),
        pitch=se.Isotropic(),
        look=se.Conditional(view),
    )
    lattice = se.Lattice(
        **{name: se.Legendre(0 if name == "pitch" else 2) for name in axes}
    )
    basis = se.spectral_basis(
        source, ([1.0], "GHz"), kernel=_child_emission, lattice=lattice
    )
    for result in (basis.predict(source, check=True), basis.direct(source)):
        np.testing.assert_allclose(
            result.stokes[0], _child_stokes(0.25 + 0.2 * x), atol=1e-12
        )


@pytest.mark.parametrize("degree", [None, 0, 4])
def test_weighted_discrete_viewing_rule_preserves_the_pitch_sample_and_diagonal_child(
    degree,
):
    import jax.numpy as jnp
    from syncem.distributions._common import Law

    class TwoViews(Law):
        centre: object

        def rule(self, n=None):
            return self.centre + jnp.array([-0.3, 0.2]), jnp.array([0.25, 0.75])

        @property
        def support(self):
            return self.centre - 0.3, self.centre + 0.2

    def view(x):
        return TwoViews(0.1 + 0.25 * x.value)

    x = 0.4
    pitch, weights = np.array([-0.6, 0.5]), np.array([0.3, 0.7])
    source = se.Population(
        se.Samples(pitch=pitch, weights=weights),
        x=x,
        z=se.Conditional(_physical_child),
        look=se.Conditional(view),
    )
    lattice = se.Lattice(**({} if degree is None else {"pitch": se.Legendre(degree)}))
    basis = se.spectral_basis(
        source, ([1.0], "GHz"), kernel=_child_emission, lattice=lattice, check=False
    )
    angles = 0.1 + 0.25 * x + np.array([-0.3, 0.2])
    smoothing = np.array(
        [
            np.average(
                projector(pitch, q, 8 if degree is None else degree), weights=weights
            )
            for q in angles
        ]
    )
    want = np.average(
        smoothing[:, None] * _child_stokes(angles), weights=[0.25, 0.75], axis=0
    )
    for result in (basis.predict(source, check=True), basis.direct(source)):
        np.testing.assert_allclose(result.stokes[0], want, atol=1e-12)
        assert result.ledger["representation:angular"].kind == "unbounded"


@pytest.mark.parametrize("wrapper", [se.Conditional, lambda factory: factory])
def test_constant_viewing_factory_uses_point_placement(wrapper):
    pop = se.Population(
        pitch=se.Isotropic(),
        x=0.0,
        look=wrapper(lambda: 0.25),
        z=se.Conditional(_physical_child),
    )
    basis = se.spectral_basis(
        pop,
        ([1.0], "GHz"),
        kernel=_child_emission,
        lattice=se.Lattice(pitch=se.Legendre(0)),
        check=False,
    )
    for result in (basis.predict(pop, check=True), basis.direct(pop)):
        np.testing.assert_allclose(result.stokes[0], _child_stokes(0.25), atol=1e-12)
