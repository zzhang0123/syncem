"""Signed ledger and Spectrum checks for SPEC sections 3.8 and 7.2."""

import json

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.model.errors import ErrorTerm as ModelErrorTerm
from syncem.records.ledger import ErrorTerm, Ledger
from syncem.units import UnitError
from syncem.values.errors import DataError
from syncem.values.spectrum import Spectrum, channel_conversion


def test_signed_partition_and_envelope_with_declared_replacement():
    a = ErrorTerm([[2.0, -1.0]], "measured", name="representation")
    b = ErrorTerm([[-1.0, 3.0]], "measured", name="outside_lattice")
    ledger = Ledger((a, b, ErrorTerm.unbounded(name="physical_kernel")))
    np.testing.assert_array_equal(
        ledger["representation"].value + ledger["outside_lattice"].value,
        [[1.0, 2.0]],
    )
    np.testing.assert_array_equal(ledger.sum("measured"), [[3.0, 4.0]])
    assert ledger.envelope() is None
    assert ledger.unbounded() == ("physical_kernel",)
    assert ledger.sum("unbounded") is None

    update = ledger.allow(
        "physical_kernel", ModelErrorTerm(0.5, "estimate"), note="reference model"
    )
    assert not update.by_kind("estimate")
    assert update.declared()[0].note == "reference model"
    assert update.assumptions[0].name == "physical_kernel"
    assert update.envelope().kind == "estimate"
    assert update.envelope().declared == ("physical_kernel",)
    np.testing.assert_array_equal(update.envelope().value, [[3.5, 4.5]])
    d = update.to_dict()
    assert json.loads(json.dumps(d, allow_nan=False)) == d
    assert ledger["physical_kernel"].kind == "unbounded"

    with pytest.raises(ValueError, match="nonempty note"):
        ledger.allow("physical_kernel", ErrorTerm(0.0, "bound"), note=" ")
    with pytest.raises(ValueError, match="only an unbounded"):
        update.allow("physical_kernel", ErrorTerm(0.0, "bound"), note="flow")
    with pytest.raises(eqx.EquinoxRuntimeError, match="nonnegative"):
        ErrorTerm(-1.0, "bound", name="physical_kernel")


def test_jy_conversion_preserves_signed_ledger_and_scales_noise_covariance():
    ch = Channels.at([20.0, 100.0], unit="GHz")
    inside = ErrorTerm([[1.0], [-1.0]], "measured", name="representation")
    spec = Spectrum(
        np.array([[[2.0], [-4.0]], [[3.0], [-5.0]]]) * 1e-23,
        ch,
        ledger=Ledger((inside.scaled(1e-23),)),
        sigma=[[1e-23], [2e-23]],
        cov=np.eye(2) * 1e-46,
        unconstrained=[[0.5], [0.1]],
    )
    jy = eqx.filter_jit(lambda s: s.to("Jy/sr"))(spec)
    np.testing.assert_allclose(jy.I, [[2.0, -4.0], [3.0, -5.0]], rtol=2e-15)
    np.testing.assert_allclose(jy.sigma, [[1.0], [2.0]], rtol=2e-15)
    np.testing.assert_allclose(jy.cov, np.eye(2), rtol=2e-15)
    np.testing.assert_allclose(jy.ledger["representation"].value, [[1.0], [-1.0]])
    np.testing.assert_array_equal(jy.unconstrained, spec.unconstrained)
    np.testing.assert_allclose(jy.to("cgs").stokes, spec.stokes, rtol=2e-15, atol=0)
    assert jy.Q is None and jy.V is None and jy.P is None
    with pytest.raises(UnitError):
        jy.to("K")


@pytest.mark.parametrize("unit", ["K_RJ", "K_CMB"])
def test_temperature_units_use_the_channel_passband(unit):
    ch = Channels.at([135.0, 280.0], unit="GHz", half_width=0.4, n_nu=32)
    nu = np.asarray(ch.nodes)
    kb, h, c = 1.380649e-16, 6.62607015e-27, 2.99792458e10
    if unit == "K_RJ":
        reference = 2 * kb * nu**2 / c**2
    else:
        temp = 2.7255
        x = h * nu / (kb * temp)
        reference = 2 * h * nu**3 / c**2 * np.exp(x) * (x / temp) / np.expm1(x) ** 2
    band = np.sum(ch.response_weights() * reference, axis=-1)
    spec = Spectrum(band[:, None], ch, ledger=Ledger())
    np.testing.assert_allclose(spec.to(unit).I, np.ones(2), rtol=3e-15)
    mono = Channels.at([135.0, 280.0], unit="GHz")
    assert np.max(np.abs(band * channel_conversion(mono, "cgs", unit) - 1)) > 1e-3


def test_response_uses_signed_differences_and_absolute_magnitudes():
    ch = Channels.at([20.0, 100.0], unit="GHz")
    target = Channels.at([30.0], unit="GHz")
    ledger = Ledger(
        (
            ErrorTerm([[2.0], [1.0]], "measured", name="representation"),
            ErrorTerm([[1.0], [3.0]], "measured", name="provider_agreement"),
        )
    )
    spec = Spectrum([[7.0], [9.0]], ch, ledger=ledger, cov=[[4.0, 1.0], [1.0, 9.0]])
    actual = spec.through([[1.0, -2.0]], channels=target)
    np.testing.assert_array_equal(actual.I, [-11.0])
    np.testing.assert_array_equal(actual.cov, [[36.0]])
    np.testing.assert_array_equal(actual.sigma, [[6.0]])
    np.testing.assert_array_equal(actual.ledger["representation"].value, [[0.0]])
    np.testing.assert_array_equal(actual.ledger["provider_agreement"].value, [[7.0]])
    with pytest.raises(DataError, match="response shape"):
        spec.through([[1.0, -2.0]])


def test_stokes_accessors_and_display_use_declared_layout():
    ch = Channels.at([100.0], unit="GHz")
    spec = Spectrum(
        [[0.7, -0.2]],
        ch,
        ledger=Ledger(),
        components=("Q", "U"),
    )
    np.testing.assert_allclose(spec.P, jnp.array([0.7 - 0.2j]))
    assert spec.I is None
    assert spec.table().splitlines() == [
        "nu (GHz) | Q (cgs) | U (cgs)",
        "100 | 0.7 | -0.2",
    ]
