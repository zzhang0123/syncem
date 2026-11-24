"""Structure of the population and role graph before numerical compilation."""

from dataclasses import dataclass

import numpy as np
import pytest

from syncem.base.continuum import ContinuumKernel
from syncem.distributions import (
    AzimuthDensity,
    AzimuthFactor,
    Conditional,
    CumulantScreen,
    Free,
    GaussianScreen,
    LogNormal,
    Uniform,
)
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.moments import Moments
from syncem.lattice.representations import AzimuthHarmonic, Legendre
from syncem.planning.roles import compile_roles
from syncem.population.blocks import FreeMoments, NonNegativeCells, Samples
from syncem.population.declaration import Population
from syncem.population.ray import Ray
from syncem.values.errors import RoleError

KERNEL = ContinuumKernel()
ENERGY = Cells.log(300, 1e5, 4)


def _energy(B):
    return Uniform(300, 1e5)


def _field(gamma):
    return Uniform(4, 6, unit="uG")


def _population(**overrides):
    return Population(**{"gamma": Uniform(300, 1e5), "B": "5 uG", **overrides})


def test_default_promotion_is_transitive_and_records_defaults():
    pop = _population(gamma=_energy, B=lambda eta: Uniform(4, 6, unit="uG"))
    plan = compile_roles(pop, KERNEL)
    assert plan.resolved == ("gamma", "B", "eta")
    assert plan.representations == {"gamma": "Cells", "B": "Cells", "eta": "Legendre"}
    component = plan.components[0]
    source_names = tuple(s.variables[0] for s in component.sources)
    assert (
        source_names.index("eta")
        < source_names.index("B")
        < source_names.index("gamma")
    )
    assert component.roles["B"].rule == component.roles["eta"].rule == "R3"
    assert component.roles["mu"].rule == "R0"
    assert component.defaults == (
        ("mu", "Isotropic"),
        ("eta", "Isotropic"),
        ("phi", "UniformAzimuth"),
        ("depth", "NoRotation"),
    )


def test_explicit_axes_follow_the_lattice_and_dropped_dependence_is_recorded():
    pop = _population(gamma=_energy)
    lattice = Lattice(gamma=ENERGY)
    with pytest.raises(RoleError, match=r"gamma depends on .*B.*average="):
        compile_roles(pop, KERNEL, lattice)
    with pytest.raises(RoleError, match="resolve="):
        compile_roles(pop, KERNEL, lattice, resolve=("B",))
    plan = compile_roles(pop, KERNEL, lattice, average=("B",))
    assert plan.resolved == ("gamma",)
    assert plan.assumptions == ("independence:gamma|B",)
    assert plan.components[0].dropped == (("gamma", "B"),)
    assert plan.components[0].source_for("gamma").parents == ("B",)


def test_free_resolved_parent_selects_conditional_linear_role():
    plan = compile_roles(
        Population(gamma=FreeMoments(), B=_field),
        KERNEL,
        resolve=("B",),
    )
    component = plan.components[0]
    assert component.source_for("gamma").declaration.variables == ("gamma",)
    assert component.roles["B"].rule == "R3a"
    assert component.roles["B"].how == "conditional linear moment map"


def test_resolved_block_and_averaged_free_law_are_distinct_roles():
    block = FreeMoments("gamma", "B", support={"B": ("1 uG", "9 uG")})
    plan = compile_roles(Population(block), KERNEL)
    comp = plan.components[0]
    assert comp.source_for("gamma") is comp.source_for("B")
    assert comp.source_for("B").support_for(KERNEL.variables[1]) == pytest.approx(
        (1e-6, 9e-6)
    )
    plan = compile_roles(_population(B=LogNormal("5 uG", Free(0.3))), KERNEL)
    assert plan.components[0].roles["B"].rule == "R6"
    assert plan.components[0].roles["B"].role == "basis parameter"
    plan = compile_roles(_population(phi=FreeMoments()), KERNEL)
    assert plan.components[0].roles["phi"].role == "azimuth parameter"
    assert "phi" not in plan.resolved


def test_duplicate_unknown_and_cyclic_declarations_raise():
    with pytest.raises(RoleError, match="more than once"):
        compile_roles(Population(FreeMoments("gamma", "B"), B="5 uG"), KERNEL)
    with pytest.raises(RoleError, match="cyclic"):
        compile_roles(_population(gamma=_energy, B=_field), KERNEL)
    with pytest.raises(RoleError, match=r"undeclared.*other"):
        compile_roles(_population(B=lambda other: "5 uG"), KERNEL)
    with pytest.raises(RoleError, match="the value or law of B"):
        compile_roles(Population(gamma=Uniform(300, 1e5)), KERNEL)


def test_depth_characteristic_precondition_and_azimuth_routes():
    depth = GaussianScreen(8, 0.5)
    pop = _population(depth=depth)
    comp = compile_roles(pop, KERNEL).components[0]
    assert comp.roles["depth"].how == "screen characteristic"
    assert comp.roles["phi"].rule == "R5(i)"
    moved = pop.replace(phi=lambda depth: AzimuthFactor(0.4))
    comp = compile_roles(moved, KERNEL).components[0]
    assert comp.roles["depth"].how == "quadrature"
    assert comp.roles["phi"].rule == "R5(iii)"
    with pytest.raises(RoleError, match="characteristic function only"):
        compile_roles(
            moved.replace(depth=CumulantScreen((8.0, 0.25, 0.0, 0.0))), KERNEL
        )
    tied = pop.replace(phi=AzimuthFactor(Conditional(lambda gamma: 0.2)))
    assert compile_roles(tied, KERNEL).components[0].roles["phi"].rule == "R5(ii)"
    tied = pop.replace(phi=lambda eta: AzimuthFactor(0.4))
    assert (
        compile_roles(tied, KERNEL, resolve=("eta",)).components[0].roles["phi"].rule
        == "R5(iv)"
    )


def test_fourier_structural_requires_more_than_harmonic_and_has_eager_guard():
    pop = _population(
        eta=lambda phi: Uniform(-0.5, 0.5), phi=AzimuthDensity(0.2, "0 deg")
    )
    plan = compile_roles(pop, KERNEL, resolve=("phi",))
    assert plan.representations["phi"] == "AzimuthFourier"
    assert plan.components[0].roles["phi"].rule == "R4a"
    with pytest.raises(RoleError, match="AzimuthFourier"):
        compile_roles(pop, KERNEL, Lattice(gamma=ENERGY, phi=AzimuthHarmonic()))
    comp = compile_roles(pop, KERNEL).components[0]
    assert comp.roles["phi"].how == "azimuth in nested quadrature"
    assert comp.roles["eta"].rule == "R4a"
    with pytest.raises(RoleError, match="angular nodes"):
        compile_roles(pop.replace(phi=AzimuthFactor(0.3)), KERNEL)


def test_joint_blocks_are_kept_whole_and_declarations_carry_role():
    @dataclass(frozen=True)
    class JointField:
        variables = ("B", "eta")

        def rule(self):
            pytest.fail("structural compilation should not evaluate a quadrature")

    pop = Population(JointField(), gamma=Uniform(300, 1e5))
    plan = compile_roles(pop, KERNEL, resolve=("B",))
    assert set(plan.resolved) == {"gamma", "B", "eta"}
    comp = compile_roles(pop, KERNEL).components[0]
    assert comp.source_for("B") is comp.source_for("eta")
    assert comp.roles["B"].how == "joint quadrature"
    with pytest.raises(RoleError, match="spans resolved and averaged"):
        compile_roles(
            pop, KERNEL, Lattice(gamma=ENERGY, B=Cells.linear(4, 6, 1, unit="uG"))
        )


def test_sum_groups_and_azimuth_promotion():
    common = _population()
    same = _population(gamma=Uniform(1e3, 1e5))
    changed = _population(B="6 uG")
    plan = compile_roles(common + changed + same, KERNEL)
    assert plan.basis_groups == ((0, 2), (1,))
    p1 = _population(phi="30 deg")
    p2 = _population(phi="50 deg")
    plan = compile_roles(p1 + p2, KERNEL)
    assert plan.representations["phi"] == "AzimuthHarmonic"
    assert plan.basis_groups == ((0, 1),)
    with pytest.raises(RoleError, match="differ in averaged phi"):
        compile_roles(p1 + p2, KERNEL, Lattice(gamma=ENERGY))
    mixed = p1.replace(eta=lambda phi: Uniform(-0.5, 0.5)) + p2
    plan = compile_roles(mixed, KERNEL)
    assert "phi" not in plan.resolved
    assert plan.basis_groups == ((0,), (1,))


def test_distinct_callable_captures_do_not_share_a_basis():
    def law(lo):
        return lambda gamma: Uniform(lo, 7, unit="uG")

    left = _population(B=law(5))
    right = _population(B=law(6))
    plan = compile_roles(left + right, KERNEL)
    assert plan.basis_groups == ((0,), (1,))


def test_samples_and_moments_supply_resolved_variables():
    samples = Samples(gamma=[1000, 2000], B=([4, 6], "uG"))
    plan = compile_roles(Population(samples), KERNEL)
    assert plan.resolved == ("gamma", "B")
    assert plan.components[0].roles["B"].how == "Samples"
    lat = Lattice(gamma=ENERGY, mu=Legendre(2), phi=AzimuthHarmonic())
    moments = Moments(lat, np.ones(lat.index.n))
    plan = compile_roles(Population(moments, B="5 uG"), KERNEL, lat)
    assert plan.resolved == ("gamma", "mu", "phi")
    assert plan.components[0].source_for("phi").kind == "Moments"


def test_cells_under_beaming_requires_inclination_density_or_legendre():
    pop = _population(mu=NonNegativeCells(support=(-1, 1)), eta=0.5)
    with pytest.raises(RoleError, match="has Cells under Beaming"):
        compile_roles(pop, KERNEL)
    lattice = Lattice(gamma=ENERGY, mu=Cells.linear(-1, 1, 2), eta=Legendre(8))
    plan = compile_roles(pop, KERNEL, lattice)
    assert plan.representations["mu"] == "Cells"


def test_ray_selects_direct_without_a_lattice_and_lists_missing_axes():
    ray = Ray(
        step=([1.0], "pc"),
        B_vector=([[1.0, 2.0, 3.0]], "uG"),
        thermal_density=([1.0], "cm^-3"),
        electrons=Uniform(300, 1e5),
        electron_density=([2.0], "cm^-3"),
    )
    plan = compile_roles(ray, KERNEL)
    assert plan.direct and plan.resolved == ()
    assert plan.components[0].roles["depth"].rule == "R8"
    with pytest.raises(RoleError, match=r"Ray needs axes.*depth"):
        compile_roles(ray, KERNEL, Lattice(gamma=ENERGY))
    lattice = Lattice(
        gamma=ENERGY,
        B=Cells.linear(1, 6, 1, unit="uG"),
        eta=Legendre(2),
        phi=AzimuthHarmonic(),
        depth=Cells.linear(-2, 2, 2),
    )
    plan = compile_roles(ray, KERNEL, lattice)
    assert not plan.direct
    assert plan.components[0].roles["depth"].how == "ray moments"
