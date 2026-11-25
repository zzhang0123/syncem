"""An in-process declaration match survives placing the same uncacheable law."""

import equinox as eqx
import numpy as np
import pytest

import syncem as se
from syncem.distributions._statistics import factor_key, factor_record


@se.kernel(
    variables=(se.Variable("look", kind="cosine"), se.Variable("pitch", kind="cosine")),
    stokes=("I",),
    capabilities=(se.Beaming("pitch", "look"),),
)
def kernel(nu, look, pitch):
    return {"I": nu / 1e9 * (2 + look + (pitch - look) ** 2)}


class Coefficient:
    def __init__(self, slope):
        self.slope = slope


def conditional(slope):
    coefficient = Coefficient(slope)

    @se.family(variable="pitch", support=(-1, 1))
    def density(pitch, a):
        return 1 + a * pitch

    def factory(look, contrast):
        return density(a=contrast * coefficient.slope * look.value, nodes=8)

    return se.Conditional(factory, params={"contrast": 1.0})


def test_same_conditional_matches_through_placement_and_real_changes_are_distinct():
    law = conditional(0.8)
    assert law.record()["fingerprint"] is None
    pop = se.Population(pitch=law, look=se.Isotropic())
    lat = se.Lattice()
    basis = se.spectral_basis(
        pop, ([1.0], "GHz"), kernel=kernel, lattice=lat, check=False
    )
    assert basis.key is None
    target = pop.replace(look=se.Isotropic())
    np.testing.assert_allclose(
        basis.predict(target).stokes, basis.direct(target).stokes, atol=1e-13
    )
    np.testing.assert_allclose(
        basis.predict(3 * target).stokes, 3 * basis.direct(target).stokes, atol=1e-13
    )
    for other in (
        conditional(0.4),
        eqx.tree_at(lambda node: node.params, law, {"contrast": 0.5}),
    ):
        with pytest.raises(se.SpecifiedMismatch):
            basis.predict(pop.replace(pitch=other))
        assert factor_key(other) != factor_key(law)


def test_local_family_key_preserves_parameters_and_cutoff_identity():
    coefficient = Coefficient(0.5)

    @se.family(variable="pitch", support=(-1, 1))
    def density(pitch, a):
        return 1 + coefficient.slope * a * pitch

    law = density(a=0.8)
    assert law.record()["fingerprint"] is None
    rebuilt = eqx.tree_at(lambda x: x.params, law, {"a": 0.8})
    assert factor_record(law) == factor_record(rebuilt)
    assert factor_key(law) != factor_key(density(a=0.4))


def test_equal_public_metadata_does_not_merge_distinct_local_cutoffs():
    from syncem.distributions.user import UserFamily

    coefficient = Coefficient(0.5)

    def shape(pitch, a):
        return 1 + a * pitch

    def cut_one(a):
        return a * coefficient.slope

    def cut_two(a):
        return 1 - a * coefficient.slope

    def make(cut):
        return UserFamily(shape, "pitch", (-1, 1), None, cut, (), 16, {"a": 0.5})

    first, second = make(cut_one), make(cut_two)
    assert first.record()["fingerprint"] is None
    assert first.record() == second.record()
    assert factor_key(first) == factor_key(make(cut_one))
    assert factor_key(first) != factor_key(second)


def test_callable_equality_cannot_merge_distinct_local_physics():
    from dataclasses import dataclass, field

    @se.family(variable="pitch", support=(-1, 1))
    def tilted(pitch, slope):
        return 1 + slope * pitch

    @dataclass(frozen=True)
    class AngularLaw:
        slope: float = field(compare=False)

        def __call__(self, look):
            return tilted(slope=self.slope * look.value, nodes=8)

    left, right = AngularLaw(0.75), AngularLaw(-0.75)
    assert left == right
    source = se.Population(pitch=se.Conditional(left), look=se.Isotropic())
    other = source.replace(pitch=se.Conditional(right))
    basis = se.spectral_basis(
        source, ([1.0], "GHz"), lattice=se.Lattice(), kernel=kernel, check=False
    )
    np.testing.assert_allclose(basis.predict(source).I, 2.5, atol=1e-13)
    np.testing.assert_allclose(basis.direct(other).I, 1.5, atol=1e-13)
    with pytest.raises(se.SpecifiedMismatch):
        basis.predict(other)
