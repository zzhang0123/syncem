"""Step 1 of design/0.8/SPEC.md Sec. 10.2: boundary units and the parent value ``Quantity`` (Sec. 2.4)."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syncem import units as U
from syncem.units import Quantity
from syncem.values.errors import UnitError

YEAR_S = 365.25 * 86400.0


@pytest.mark.parametrize(
    ("text", "dimension", "expected"),
    [
        ("5 uG", "field", 5e-6),
        ("5 µG", "field", 5e-6),
        ("3 mG", "field", 3e-3),
        ("2 nT", "field", 2e-5),
        ("1.4 GHz", "frequency", 1.4e9),
        ("408 MHz", "frequency", 4.08e8),
        ("1e8 Hz", "frequency", 1e8),
        ("60 deg", "angle", math.pi / 3),
        ("0.5 rad", "angle", 0.5),
        ("19 Myr", "time", 19e6 * YEAR_S),
        ("8 rad m^-2", "depth", 8.0),
        ("8 rad/m^2", "depth", 8.0),
        ("8 rad m-2", "depth", 8.0),
        ("1e20 cm^-2", "column", 1e20),
        ("1 m^-2", "column", 1e-4),
        ("0.03 cm^-3", "density", 0.03),
        ("2 pc", "length", 2 * 3.0856775814913673e18),
    ],
)
def test_string_scalars_convert_once_to_canonical(text, dimension, expected):
    assert U.to_canonical(text, dimension, name="x") == pytest.approx(
        expected, rel=1e-15
    )


def test_canonical_units():
    assert U.CANONICAL == {
        "frequency": "Hz",
        "field": "G",
        "angle": "rad",
        "depth": "rad m^-2",
        "column": "cm^-2",
        "density": "cm^-3",
        "length": "cm",
        "time": "s",
        "dimensionless": "",
    }


def test_hz_against_ghz_pins():
    assert U.to_canonical("0.1 GHz", "frequency", name="nu") == 1e8
    assert U.to_canonical("10 GHz", "frequency", name="nu") == 1e10
    assert U.to_canonical(
        (np.array([0.1, 10.0]), "GHz"), "frequency", name="nu"
    ).tolist() == [1e8, 1e10]
    assert U.convert(1e9, "Hz", "GHz") == 1.0


def test_array_pairs_and_quantities():
    out = U.to_canonical((np.array([4.0, 6.0]), "uG"), "field", name="B")
    np.testing.assert_allclose(out, [4e-6, 6e-6], rtol=1e-15)
    assert U.to_canonical(Quantity(5.0, "uG"), "field", name="B") == pytest.approx(
        5e-6, rel=1e-15
    )


def test_astropy_quantity_is_accepted():
    astropy_units = pytest.importorskip("astropy.units")
    assert U.to_canonical(5 * astropy_units.uG, "field", name="B") == pytest.approx(
        5e-6, rel=1e-14
    )
    assert U.to_canonical(
        1.4 * astropy_units.GHz, "frequency", name="nu"
    ) == pytest.approx(1.4e9, rel=1e-15)


@pytest.mark.parametrize("dimension", ["dimensionless", "depth", "column"])
def test_bare_numbers_for_conventional_units(dimension):
    assert U.to_canonical(8.0, dimension, name="x") == 8.0
    np.testing.assert_array_equal(
        U.to_canonical(np.array([1.0, 2.0]), dimension, name="x"), [1.0, 2.0]
    )


@pytest.mark.parametrize("dimension", ["field", "frequency", "angle", "time"])
def test_bare_numbers_raise_for_dimensional_quantities(dimension):
    with pytest.raises(UnitError, match=r"x=5\.0 has no unit\. Write x=\"5 "):
        U.to_canonical(5.0, dimension, name="x")


def test_bare_array_raises_with_the_pair_spelling():
    with pytest.raises(UnitError, match=r"\(array, \"uG\"\)"):
        U.to_canonical(np.array([4.0, 6.0]), "field", name="B")


def test_wrong_dimension_raises():
    with pytest.raises(UnitError, match="frequency"):
        U.to_canonical("5 GHz", "field", name="B")


def test_unknown_unit_raises():
    with pytest.raises(UnitError, match="furlong"):
        U.to_canonical("5 furlong", "length", name="L")


@pytest.mark.parametrize("text", ["", "uG", "5 uG extra", "five uG"])
def test_malformed_strings_raise(text):
    with pytest.raises(UnitError):
        U.parse(text)


def test_parse_returns_value_and_unit():
    assert U.parse("5 uG") == (5.0, "uG")
    assert U.parse("  -2.5e-3   rad m^-2 ") == (-2.5e-3, "rad m^-2")
    assert U.parse("7") == (7.0, "")


def test_dimension_of():
    assert U.dimension_of("uG") == "field"
    assert U.dimension_of("rad/m^2") == "depth"
    assert U.dimension_of("") == "dimensionless"
    assert U.dimension_of("G^2") is None  # a product has no named dimension


# ------------------------------------------------------------------ Quantity


def test_quantity_is_a_pytree_with_a_static_unit():
    q = Quantity(5.0, "uG")
    leaves, treedef = jax.tree_util.tree_flatten(q)
    assert len(leaves) == 1
    assert jax.tree_util.tree_unflatten(treedef, leaves).unit == "uG"


def test_quantity_arithmetic_propagates_the_unit():
    b = Quantity(5.0, "uG")
    assert (b + "1 uG").to("uG").value == pytest.approx(6.0)
    assert (b + Quantity(1e-6, "G")).to("uG").value == pytest.approx(6.0)
    assert (2 * b).to("G").value == pytest.approx(1e-5)
    assert (b / 2).to("uG").value == pytest.approx(2.5)
    assert (-b).to("uG").value == pytest.approx(-5.0)
    assert (b**2).to("G^2").value == pytest.approx(25e-12, rel=1e-14)
    assert float(b / Quantity(1.0, "uG")) == pytest.approx(5.0)
    assert (b * Quantity(2.0, "Myr")).to("G s").value == pytest.approx(
        5e-6 * 2e6 * YEAR_S
    )


def test_quantity_comparison_converts():
    assert Quantity(5.0, "uG") < "1 mG"
    assert Quantity(5.0, "uG") >= Quantity(5e-6, "G")


def test_plain_float_in_a_dimensional_slot_is_refused():
    b = Quantity(5.0, "uG")
    with pytest.raises(UnitError, match="plain"):
        b + 1.0
    with pytest.raises(UnitError):
        float(b)
    with pytest.raises(UnitError):
        b + "1 GHz"


def test_dimensionless_quantity_takes_plain_numbers():
    q = Quantity(2.0, "")
    assert float(q + 1.0) == 3.0


def test_quantity_under_jit_and_grad():
    def loss(value):
        b = Quantity(value, "uG")
        return (b**2).to("uG^2").value

    assert jax.jit(loss)(3.0) == pytest.approx(9.0)
    assert jax.grad(loss)(3.0) == pytest.approx(6.0)
    out = jax.vmap(lambda v: (Quantity(v, "uG") * 2).to("G").value)(
        jnp.array([1.0, 2.0])
    )
    np.testing.assert_allclose(out, [2e-6, 4e-6])


def test_quantity_of_an_unknown_unit_raises():
    with pytest.raises(UnitError):
        Quantity(1.0, "furlong")


# ------------------------------------------------------------------ review fixes


def test_plain_number_on_a_scaled_dimensionless_unit():
    q = Quantity(2.0, "GHz s")
    assert float(q + 1.0) == pytest.approx(2e9 + 1.0)
    assert not (Quantity(0.5, "GHz s") < 1.0)


def test_numpy_arrays_defer_to_quantity():
    out = np.array([1.0, 2.0]) * Quantity(1.0, "GHz")
    assert isinstance(out, Quantity)
    np.testing.assert_allclose(out.to("Hz").value, [1e9, 2e9])
    assert isinstance(np.array([1.0, 2.0]) / Quantity(1.0, "GHz"), Quantity)


def test_traced_power_of_a_dimensionless_quantity():
    f = jax.jit(lambda a: (Quantity(2.0, "") ** a).value)
    assert f(1.5) == pytest.approx(2.0**1.5)
    assert jax.grad(lambda a: (Quantity(2.0, "") ** a).value)(1.5) == pytest.approx(
        2.0**1.5 * math.log(2.0)
    )
    with pytest.raises(UnitError):
        jax.jit(lambda a: (Quantity(2.0, "uG") ** a).value)(1.5)


def test_fractional_powers_round_trip():
    b = (Quantity(8.0, "G") ** (1 / 3)) ** 3
    assert U.to_canonical(b, "field", name="B") == pytest.approx(8.0)
    assert U.dimension_of((Quantity(4.0, "G^2") ** 0.5).unit) == "field"


def test_conversion_round_trips():
    xs = np.linspace(0.1, 100, 1000)
    assert all(Quantity(x * 1e9, "Hz").to("GHz").value == x for x in xs)
    assert U.convert(1.4e9, "Hz", "GHz") == 1.4


def test_bare_error_for_non_numbers():
    for value in (None, object(), 1j):
        with pytest.raises(UnitError):
            U.to_canonical(value, "frequency", name="nu")


def test_pairs_and_bare_values_under_jit():
    f = jax.jit(lambda v: U.to_canonical((v, "GHz"), "frequency", name="nu"))
    assert f(1.4) == pytest.approx(1.4e9)
    g = jax.jit(lambda v: U.to_canonical(v, "depth", name="depth"))
    assert g(1.4) == pytest.approx(1.4)
    assert jax.grad(lambda v: U.to_canonical((v, "GHz"), "frequency", name="nu"))(
        1.0
    ) == pytest.approx(1e9)


@pytest.mark.parametrize("text", ["60 μT", "60 µT", "3 μm", "5 μG"])
def test_both_micro_signs(text):
    assert U.parse(text)[0] > 0


def test_inverse_units_with_a_leading_one():
    assert U.to_canonical("1e20 1/cm^2", "column", name="N") == 1e20
    assert U.dimension_of("1/cm^3") == "density"


def test_reflected_power_of_a_dimensionless_quantity():
    assert float(2.0 ** Quantity(3.0, "")) == 8.0
    with pytest.raises(UnitError):
        2.0 ** Quantity(3.0, "uG")
