"""Step 1 of design/0.8/SPEC.md Sec. 10.2: ``Channels`` constructors (Sec. 3.2) against the study channels."""

import sys
from pathlib import Path

import numpy as np
import pytest

from syncem.base.channels import SIGMA_RHO, Channels
from syncem.model import Channels as ModelChannels
from syncem.values.errors import UnitError

STUDY_LIB = Path(__file__).resolve().parents[2] / "studies" / "classical-models" / "lib"


@pytest.fixture(scope="module")
def paper():
    if str(STUDY_LIB) not in sys.path:
        sys.path.insert(0, str(STUDY_LIB))
    from config import PAPER  # noqa: PLC0415

    return PAPER


def test_log_equals_the_study_channels(paper):
    ch = Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2, n_nu=400)
    nodes = np.asarray(ch.nodes)
    weights = np.asarray(ch.response_weights())
    assert nodes.shape == weights.shape == (41, 400)
    assert np.max(np.abs(nodes / paper.chan_nu - 1)) < 1e-14
    assert np.max(np.abs(weights - paper.chan_w[None, :])) < 1e-14 * np.max(
        paper.chan_w
    )


def test_hz_against_ghz_pins():
    ghz = Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2)
    hz = Channels.log(1e8, 1e10, 41, unit="Hz", half_width=0.2)
    np.testing.assert_array_equal(np.asarray(ghz.centres_hz), np.asarray(hz.centres_hz))
    np.testing.assert_allclose(np.asarray(ghz.nodes), np.asarray(hz.nodes), rtol=1e-15)
    assert ghz.unit == "GHz" and hz.unit == "Hz"
    assert float(ghz.centres_hz[0]) == 1e8 and float(ghz.centres_hz[-1]) == 1e10


def test_fractional_width_is_the_spread_of_the_response():
    # PAPER eq: forward response families: h_j = f nu_j / sigma_rho, and f nu_j is the spread of R_j.
    assert SIGMA_RHO == pytest.approx(0.3976350541184696, rel=1e-13)
    f = 0.08
    ch = Channels.log(1, 300, 7, unit="GHz", fractional_width=f, n_nu=256)
    c = np.asarray(ch.centres_hz)[:, None]
    w = np.asarray(ch.response_weights())
    nu = np.asarray(ch.nodes)
    np.testing.assert_allclose(w.sum(axis=1), 1.0, rtol=1e-13)
    np.testing.assert_allclose(
        np.sqrt((w * (nu - c) ** 2).sum(axis=1)) / c[:, 0], f, rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(ch.widths_hz) / c[:, 0], f / SIGMA_RHO, rtol=1e-15
    )


def test_at_takes_given_centres():
    ch = Channels.at([1.4, 23.0, 408e-3], unit="GHz", half_width=0.1)
    np.testing.assert_allclose(
        np.asarray(ch.centres_hz), [1.4e9, 2.3e10, 4.08e8], rtol=1e-15
    )
    assert isinstance(ch, ModelChannels)


def test_monochromatic_channels():
    ch = Channels.at([1.4, 23.0], unit="GHz")
    assert ch.family == "monochromatic"
    np.testing.assert_array_equal(np.asarray(ch.nodes), [[1.4e9], [2.3e10]])
    np.testing.assert_array_equal(np.asarray(ch.response_weights()), [[1.0], [1.0]])
    with pytest.raises(ValueError, match="monochromatic"):
        ch(np.array([1.4e9]))


def test_indexing_selects_rows():
    ch = Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2)
    sub = ch[[0, 2, 40]]
    assert sub.n_ch == 3
    np.testing.assert_array_equal(
        np.asarray(sub.nodes), np.asarray(ch.nodes)[[0, 2, 40]]
    )
    np.testing.assert_array_equal(
        np.asarray(sub.response_weights()),
        np.asarray(ch.response_weights())[[0, 2, 40]],
    )
    assert ch[5:8].n_ch == 3
    assert ch[3].n_ch == 1
    assert sub.unit == "GHz"
    mono = Channels.at([1.0, 2.0, 3.0], unit="GHz")[1:]
    np.testing.assert_array_equal(np.asarray(mono.nodes), [[2e9], [3e9]])


def test_from_bandpass_tables_on_their_own_grids():
    t1 = (np.array([0.9, 1.0, 1.1, 1.2]), np.array([0.0, 1.0, 1.0, 0.0]))
    t2 = (np.array([2.0, 2.2, 2.4]), np.array([0.0, 2.0, 0.0]))
    ch = Channels.from_bandpass([t1, t2], unit="GHz", n_nu=256)
    assert ch.n_ch == 2
    np.testing.assert_allclose(
        np.asarray(ch.support), [[0.9e9, 1.2e9], [2.0e9, 2.4e9]], rtol=1e-15
    )
    # Unit area: the trapezoid of each table.
    area1, area2 = 0.2e9, 0.4e9 * 2.0 / 2
    np.testing.assert_allclose(
        np.asarray(ch(np.array([1.05e9])))[:, 0], [1.0 / area1, 0.0], rtol=1e-12
    )
    np.testing.assert_allclose(
        np.asarray(ch(np.array([2.2e9])))[:, 0], [0.0, 2.0 / area2], rtol=1e-12
    )


def test_from_bandpass_needs_zero_ends():
    with pytest.raises(ValueError, match="zero"):
        Channels.from_bandpass(
            [(np.array([1.0, 2.0]), np.array([1.0, 0.0]))], unit="GHz"
        )


@pytest.mark.parametrize(
    ("kwargs", "error", "match"),
    [
        (
            dict(unit=None, half_width=0.2),
            UnitError,
            r"Channels\.log\(0\.1, 10, 41\) has no unit",
        ),
        (dict(unit="uG", half_width=0.2), UnitError, "field"),
        (dict(unit="GHz", half_width=0.2, fractional_width=0.08), ValueError, "one of"),
        (dict(unit="GHz", half_width=1.0), ValueError, "half_width"),
        (dict(unit="GHz", fractional_width=-0.1), ValueError, "fractional_width"),
        (dict(unit="GHz", half_width=0.2, n_nu=0), ValueError, "n_nu"),
    ],
)
def test_constructor_refusals(kwargs, error, match):
    with pytest.raises(error, match=match):
        Channels.log(0.1, 10, 41, **kwargs)


@pytest.mark.parametrize(
    ("lo", "hi", "n"), [(10, 0.1, 41), (0.1, 10, 0), (0.1, 10, 2.5), (-1, 10, 3)]
)
def test_bad_grids_raise(lo, hi, n):
    with pytest.raises(ValueError):
        Channels.log(lo, hi, n, unit="GHz", half_width=0.2)


def test_single_channel_log():
    ch = Channels.log(1.0, 1.0, 1, unit="GHz", half_width=0.2)
    assert ch.n_ch == 1 and float(ch.centres_hz[0]) == 1e9


# ------------------------------------------------------------------ review fixes
from syncem.units import Quantity  # noqa: E402


@pytest.mark.parametrize(
    "frequencies",
    [
        (np.array([30.0, 100.0]), "GHz"),
        ([30.0, 100.0], "GHz"),
        Quantity(np.array([30.0, 100.0]), "GHz"),
        ["30 GHz", "100 GHz"],
    ],
)
def test_at_with_frequencies_that_carry_their_unit(frequencies):
    ch = Channels.at(frequencies, half_width=0.1)
    np.testing.assert_allclose(np.asarray(ch.centres_hz), [3e10, 1e11], rtol=1e-15)
    assert ch.unit == "GHz"


def test_at_with_an_astropy_quantity():
    u = pytest.importorskip("astropy.units")
    ch = Channels.at(np.array([30.0, 100.0]) * u.GHz, half_width=0.1)
    np.testing.assert_allclose(np.asarray(ch.centres_hz), [3e10, 1e11], rtol=1e-15)


def test_log_with_quantities_and_strings():
    a = Channels.log(Quantity(1.0, "GHz"), Quantity(10.0, "GHz"), 3, half_width=0.2)
    b = Channels.log("1 GHz", "10 GHz", 3, half_width=0.2)
    np.testing.assert_array_equal(np.asarray(a.centres_hz), np.asarray(b.centres_hz))
    assert a.unit == b.unit == "GHz"


def test_mixed_string_units_record_hz():
    assert Channels.at(["1 GHz", "500 MHz"], half_width=0.1).unit == "Hz"


def test_strings_with_unit_keyword_raise():
    with pytest.raises(UnitError, match="unit="):
        Channels.at(["30 GHz"], unit="GHz")
    with pytest.raises(UnitError, match="unit="):
        Channels.log("0.1 GHz", "10 GHz", 3, unit="GHz", half_width=0.2)


def test_from_bandpass_with_pairs_keeps_the_unit():
    ch = Channels.from_bandpass([((np.array([1.0, 2.0, 3.0]), "GHz"), [0.0, 1.0, 0.0])])
    assert ch.unit == "GHz"
    assert (
        Channels.from_bandpass([(["1 GHz", "2 GHz", "3 GHz"], [0.0, 1.0, 0.0])]).unit
        == "GHz"
    )


def test_single_channel_needs_equal_ends():
    with pytest.raises(ValueError, match="lo == hi"):
        Channels.log(1, 10, 1, unit="GHz", half_width=0.2)


def test_monochromatic_peak_scale_and_from_basis():
    mono = Channels.at([1.0, 2.0, 3.0], unit="GHz")
    np.testing.assert_array_equal(np.asarray(mono.peak_scale()), [1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="monochromatic"):
        Channels.from_basis(mono, np.eye(3)[:2])


def test_from_basis_keeps_the_reference_unit():
    ref = Channels.log(1, 10, 5, unit="GHz", half_width=0.2)
    assert Channels.from_basis(ref, np.eye(5)[:2]).unit == "GHz"
