"""Step 1 of design/0.8/SPEC.md Sec. 10.2: variables, placement checks and error classes (Sec. 2.4, 2.5, 3.2).

The acceptance cases ``B=se.Uniform(4, 6)``, ``B=se.Cells.linear(2, 8, 3)`` and
``eta=se.Uniform(10, 50, unit="deg")`` need classes of later steps. Here they are stand-ins that
carry the placement protocol (``given_unit``, ``is_point``, ``unit_remedy``, ``angle_remedy``).
Steps 2 and the distributions step repeat these tests with the real classes.
"""

import math
from dataclasses import dataclass

import pytest

from syncem.base.variables import SYNCHROTRON_VARIABLES, Variable
from syncem.values import errors as E
from syncem.values.errors import UnitError


@dataclass(frozen=True)
class StandIn:
    text: str
    given_unit: str | None
    is_point: bool = False
    remedy: str = ""
    angle: str = ""

    def __repr__(self):
        return self.text

    def unit_remedy(self, unit):
        return self.remedy.format(unit=unit)

    def angle_remedy(self):
        return self.angle


B = SYNCHROTRON_VARIABLES["B"]
ETA = SYNCHROTRON_VARIABLES["eta"]


def test_the_six_synchrotron_variables():
    table = {
        name: (v.kind, v.unit, v.bare, v.default)
        for name, v in SYNCHROTRON_VARIABLES.items()
    }
    assert table == {
        "gamma": ("positive", "", True, None),
        "B": ("positive", "G", False, None),
        "mu": ("cosine", "", True, "Isotropic"),
        "eta": ("cosine", "", True, "Isotropic"),
        "phi": ("angle", "rad", False, "UniformAzimuth"),
        "depth": ("real", "rad m^-2", True, "NoRotation"),
    }
    assert tuple(SYNCHROTRON_VARIABLES) == ("gamma", "B", "mu", "eta", "phi", "depth")


def test_bare_field_strength_raises():
    with pytest.raises(UnitError) as info:
        B.place(5.0)
    assert (
        str(info.value)
        == 'B=5.0 has no unit. Write B="5 uG", or se.Uniform(4, 6, unit="uG") for a range.'
    )


def test_range_without_unit_raises():
    uniform = StandIn(
        "se.Uniform(4, 6)", None, remedy='Write se.Uniform(4, 6, unit="{unit}").'
    )
    with pytest.raises(UnitError) as info:
        B.place(uniform)
    assert (
        str(info.value)
        == 'B=se.Uniform(4, 6) has no unit. Write se.Uniform(4, 6, unit="uG").'
    )


def test_cells_without_unit_raise():
    cells = StandIn(
        "se.Cells.linear(2, 8, 3)",
        None,
        remedy='Write se.Cells.linear("2 {unit}", "8 {unit}", 3), or pass unit="{unit}".',
    )
    with pytest.raises(UnitError) as info:
        B.place(cells)
    assert str(info.value) == (
        'B=se.Cells.linear(2, 8, 3) has no unit. Write se.Cells.linear("2 uG", "8 uG", 3), or pass unit="uG".'
    )


def test_range_of_angles_on_a_cosine_raises():
    angle_range = StandIn(
        'se.Uniform(10, 50, unit="deg")',
        "deg",
        angle='Write se.UniformAngle(10, 50, unit="deg") (uniform in the angle) or se.Uniform(0.64, 0.98) '
        "(uniform in the cosine).",
    )
    with pytest.raises(UnitError) as info:
        ETA.place(angle_range)
    assert str(info.value).startswith(
        'eta=se.Uniform(10, 50, unit="deg") is a range of angles on a cosine. Write'
    )


def test_angle_point_on_a_cosine_is_read_as_its_cosine():
    assert ETA.place("60 deg") == pytest.approx(0.5, rel=1e-15)


def test_distribution_of_another_variable_raises():
    screen = StandIn("GaussianScreen(...)", "rad m^-2")
    with pytest.raises(UnitError) as info:
        SYNCHROTRON_VARIABLES["phi"].place(screen)
    assert str(info.value) == (
        "phi=GaussianScreen(...) is a distribution of Faraday depth. phi is the projected-field azimuth. Write depth=."
    )


def test_wrong_dimension_without_a_matching_variable_raises():
    with pytest.raises(UnitError, match=r'B="5 GHz" has unit GHz \(frequency\)'):
        B.place("5 GHz")


def test_point_values_convert_to_canonical():
    assert B.place("5 uG") == pytest.approx(5e-6, rel=1e-15)
    assert SYNCHROTRON_VARIABLES["gamma"].place(1e4) == 1e4
    assert SYNCHROTRON_VARIABLES["depth"].place(8.0) == 8.0
    assert SYNCHROTRON_VARIABLES["depth"].place("8 rad m^-2") == 8.0
    assert SYNCHROTRON_VARIABLES["phi"].place("90 deg") == pytest.approx(
        math.pi / 2, rel=1e-15
    )


def test_objects_with_a_matching_unit_pass_unchanged():
    cells = StandIn('se.Cells.linear("2 uG", "8 uG", 3)', "uG")
    assert B.place(cells) is cells
    bare = StandIn("se.Uniform(0.1, 0.5)", None)
    assert ETA.place(bare) is bare


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("mu", 1.5),
        ("eta", -1.01),
        ("gamma", 0.0),
        ("B", "-5 uG"),
        ("gamma", float("nan")),
    ],
)
def test_kind_violations_raise(name, value):
    with pytest.raises(ValueError):
        SYNCHROTRON_VARIABLES[name].place(value)


def test_usual_range_gives_a_warning_line():
    assert B.outside_usual(5e-6) is None
    assert B.outside_usual(5.0) == "B = 5 G is outside the usual range 1e-09 to 0.001 G"
    assert SYNCHROTRON_VARIABLES["eta"].outside_usual(0.3) is None


def test_variable_is_static_and_hashable():
    v = Variable("a", "grain radius", "positive", "cm", False, (1e-7, 1e-3), None, "um")
    assert hash(v) == hash(
        Variable("a", "grain radius", "positive", "cm", False, (1e-7, 1e-3), None, "um")
    )
    with pytest.raises(ValueError):
        Variable("a", "grain radius", "sideways", "cm", False, None, None, "um")


def test_error_classes():
    names = (
        "UnitError",
        "RoleError",
        "RegimeError",
        "SpecifiedMismatch",
        "NothingToPredict",
        "DataError",
        "DepthExpansionError",
        "SizeError",
        "ImproperPosterior",
    )
    for name in names:
        cls = getattr(E, name)
        assert issubclass(cls, E.SyncEMError)
        assert issubclass(cls, ValueError)
    assert set(E.__all__) == {"SyncEMError", *names}


# ------------------------------------------------------------------ review fixes
import numpy as np  # noqa: E402


def test_pairs_are_placed():
    np.testing.assert_allclose(
        B.place((np.array([4.0, 6.0]), "uG")), [4e-6, 6e-6], rtol=1e-15
    )
    np.testing.assert_allclose(B.place(([4, 6], "uG")), [4e-6, 6e-6], rtol=1e-15)
    np.testing.assert_allclose(
        SYNCHROTRON_VARIABLES["depth"].place(([1.0, 2.0], "rad m^-2")), [1.0, 2.0]
    )
    assert ETA.place((60.0, "deg")) == pytest.approx(0.5, rel=1e-15)
    np.testing.assert_allclose(
        ETA.place((np.array([60.0, 120.0]), "deg")), [0.5, -0.5], atol=1e-15
    )


def test_cosine_of_an_angle_array_quantity():
    from syncem.units import Quantity

    np.testing.assert_allclose(
        ETA.place(Quantity(np.array([60.0, 120.0]), "deg")), [0.5, -0.5], atol=1e-15
    )


@dataclass(frozen=True)
class NoValues:
    text: str
    given_unit: str | None = None
    is_point: bool = False
    carries_values: bool = False

    def __repr__(self):
        return self.text


def test_declarations_without_values_need_no_unit():
    factor = NoValues("se.AzimuthFactor(0.4)")
    assert SYNCHROTRON_VARIABLES["phi"].place(factor) is factor
    free = NoValues("se.FreeMoments()")
    assert B.place(free) is free


@dataclass(frozen=True)
class AnglePoint:
    degrees: float
    given_unit: str = "deg"
    is_point: bool = True

    def as_cosine(self):
        return StandIn(
            f"se.Delta({math.cos(math.radians(self.degrees)):.6g})", None, is_point=True
        )


def test_angle_point_object_is_converted_to_its_cosine():
    placed = ETA.place(AnglePoint(60.0))
    assert placed.text == "se.Delta(0.5)"


def test_angle_point_object_without_conversion_raises():
    with pytest.raises(UnitError, match="cosine"):
        ETA.place(StandIn("se.Delta(60, unit='deg')", "deg", is_point=True))
