"""Joint samples, unknown-moment declarations and discrete ray profiles."""

import json

import numpy as np
import pytest

from syncem.base.variables import SYNCHROTRON_VARIABLES, Variable
from syncem.lattice.cells import Cells
from syncem.population.blocks import FreeMoments, NonNegativeCells, Samples
from syncem.population.ray import Ray
from syncem.units import UnitError, convert
from syncem.values.errors import RoleError


def test_samples_keep_correlations_units_and_column():
    samples = Samples(
        gamma=[1000.0, 2000.0, 9000.0],
        B=([2.0, 5.0, 99.0], "uG"),
        eta=([0.0, 180.0, 45.0], "deg"),
        weights=[1e308, 5e307, 0.0],
        column="12 cm^-2",
    )
    placed = samples.placed(SYNCHROTRON_VARIABLES)
    nodes, columns = placed.mass_rule()
    np.testing.assert_array_equal(columns, [8.0, 4.0, 0.0])
    np.testing.assert_allclose(nodes["B"], [2e-6, 5e-6, 99e-6], rtol=2e-15)
    np.testing.assert_array_equal(nodes["eta"][:2], [1, -1])
    assert placed.given_units == (("gamma", None), ("B", "uG"), ("eta", "deg"))
    assert placed.units == (("gamma", ""), ("B", "G"), ("eta", ""))
    np.testing.assert_allclose(placed.support["B"], [2e-6, 5e-6])
    np.testing.assert_array_equal(
        placed.placed(SYNCHROTRON_VARIABLES).nodes["eta"], nodes["eta"]
    )
    np.testing.assert_allclose(
        columns @ (nodes["gamma"] * nodes["B"]), 0.056, rtol=2e-15
    )
    assert json.loads(json.dumps(placed.record(), allow_nan=False))["column"] == 12


def test_default_sample_weights_are_probabilities_and_allow_user_axes():
    block = Samples(a=([1, 2], "m"), b=[-1, 2]).placed(
        (Variable("a", unit="cm", kind="positive"), Variable("b"))
    )
    nodes, w = block.rule()
    _, mass = block.mass_rule()
    assert block.column is None
    np.testing.assert_array_equal(nodes["a"], [100.0, 200.0])
    np.testing.assert_array_equal(nodes["b"], [-1.0, 2.0])
    np.testing.assert_array_equal(w, [0.5, 0.5])
    np.testing.assert_array_equal(w, mass)


@pytest.mark.parametrize(
    "args, match",
    [
        ({}, "named variable"),
        ({"gamma": []}, "nonempty"),
        ({"gamma": [1, 2], "B": ([5], "uG")}, "equal length"),
        ({"gamma": [1, 2], "weights": [1]}, "same shape"),
        ({"gamma": [1, 2], "weights": [0, 0]}, "positive mass"),
        ({"gamma": [1, 2], "weights": [1, -1]}, "nonnegative"),
        ({"gamma": [1, 2], "column": "-1 cm^-2"}, "column"),
        ({"gamma": [1, 2], "column": ([1, 2], "cm^-2")}, "scalar"),
    ],
)
def test_invalid_sample_shapes_and_mass(args, match):
    with pytest.raises(ValueError, match=match):
        Samples(**args)


def test_samples_validate_at_placement():
    with pytest.raises(UnitError, match="B requires a unit"):
        Samples(B=[3, 4]).placed(SYNCHROTRON_VARIABLES)
    with pytest.raises(UnitError, match="B takes G"):
        Samples(B=([3, 4], "GHz")).placed(SYNCHROTRON_VARIABLES)
    with pytest.raises(ValueError, match="cosine"):
        Samples(eta=[0.9, 1.01]).placed(SYNCHROTRON_VARIABLES)
    with pytest.raises(RoleError, match="absent"):
        Samples(grain_radius=[1e-5, 2e-5]).placed(SYNCHROTRON_VARIABLES)


def test_free_moments_bind_keyword_and_resolve_support():
    unknown = FreeMoments(support={"B": ("0 uG", "8 uG")})
    assert not unknown.carries_values and unknown.variables == ()
    bound = unknown.bind("B")
    assert unknown.variables == () and bound.variables == ("B",)
    b = SYNCHROTRON_VARIABLES["B"]
    assert b.place(bound) is bound
    assert bound.support_for(b) == (0, 8e-6)
    assert bound.bind("B") is bound
    with pytest.raises(RoleError, match="cannot occupy"):
        bound.bind("gamma")
    with pytest.raises(RoleError, match="outside"):
        unknown.bind("eta")


def test_joint_markers_keep_cell_hints_and_point_masses():
    cells = Cells.linear(300, 700, 2)
    marker = NonNegativeCells(
        "gamma", "eta", cells={"gamma": cells}, support={"eta": (-0.2, 0.6)}
    )
    assert marker.point_masses == 17
    assert marker.cells_for(SYNCHROTRON_VARIABLES["gamma"]) is cells
    assert marker.support_for(SYNCHROTRON_VARIABLES["eta"]) == (-0.2, 0.6)
    assert (
        json.loads(json.dumps(marker.record(), allow_nan=False))["point_masses"] == 17
    )
    with pytest.raises(ValueError, match="distinct"):
        FreeMoments("gamma", "gamma")
    with pytest.raises(ValueError, match="positive integer"):
        NonNegativeCells("eta", point_masses=True)
    with pytest.raises(RoleError, match="mapping"):
        FreeMoments("gamma", "B", support=(1, 4))


class _Law:
    def __init__(self, label):
        self.label = label

    def rule(self):
        return np.array([1000.0]), np.ones(1)

    def record(self):
        return {"label": self.label}


def _ray(field, *, electrons, pitch=None):
    return Ray(
        step=([1.0, 2.0], "pc"),
        B_vector=(field, "uG"),
        thermal_density=([1.0, 2.0], "cm^-3"),
        electron_density=([3.0, 2.0], "cm^-3"),
        electrons=electrons,
        pitch=pitch,
    )


def test_ray_depth_orientation_and_inclination_at_emitting_midpoint():
    a, b = _Law("near"), _Law("far")
    ray = _ray([[3.0, 4.0, 2.0], [-1.0, 0.0, -1.0]], electrons=(a, b))
    points = ray.coordinates()
    assert ray.shape == () and ray.n_steps == 2
    np.testing.assert_allclose(points["depth"], [-2.436, -1.624], rtol=2e-15)
    np.testing.assert_allclose(points["B"], np.sqrt([29, 2]) * 1e-6, rtol=2e-15)
    np.testing.assert_allclose(points["eta"], np.array([2, -1]) / np.sqrt([29, 2]))
    np.testing.assert_allclose(points["phi"], [np.arctan2(4, 3), np.pi])
    np.testing.assert_array_equal(
        ray.columns, convert(np.array([3.0, 4.0]), "pc", "cm")
    )
    steps = tuple(ray.iter_steps())
    assert tuple(s.electrons for s in steps) == (a, b)
    assert steps[0].factors["B"].unit == "G"
    assert steps[0].factors["phi"].unit == "rad"
    assert (
        json.loads(json.dumps(ray.record(), allow_nan=False))["units"]["step"] == "cm"
    )


def test_ray_selects_sightlines_and_preserves_step_laws():
    a, b, c, d = (_Law(i) for i in range(4))
    ray = _ray(
        np.array(
            [
                [[3.0, 4.0, 2.0], [-1.0, 0.0, -1.0]],
                [[0.0, 0.0, 0.0], [1.0, 1.0, -2.0]],
            ]
        ),
        electrons=((a, b), (c, d)),
        pitch=(a, b),
    )
    assert ray.shape == (2,)
    first = ray[..., 0]
    assert tuple(s.electrons for s in first.iter_steps()) == (a, b)
    second = ray[1]
    (step,) = second.iter_steps()
    assert step.electrons is d and step.factors["mu"] is b
    reverse = ray[::-1]
    assert reverse[0].electrons == (c, d)
    with pytest.raises(IndexError, match="sightline"):
        second[0]
