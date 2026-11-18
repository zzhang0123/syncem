"""Step 2 of design/0.8/SPEC.md Sec. 10.2: ``Cells`` (Sec. 3.3, 4.1) and their placement (Sec. 2.4)."""

import math
import sys
from pathlib import Path

import numpy as np
import pytest

from syncem.base.variables import SYNCHROTRON_VARIABLES
from syncem.lattice.cells import Cells
from syncem.values.errors import UnitError

STUDY_LIB = Path(__file__).resolve().parents[2] / "studies" / "classical-models" / "lib"


@pytest.fixture(scope="module")
def paper():
    if str(STUDY_LIB) not in sys.path:
        sys.path.insert(0, str(STUDY_LIB))
    from config import PAPER  # noqa: PLC0415

    return PAPER


def test_paper_energy_cells(paper):
    # The study's upper end is 2962642.63 (PAPER: "gamma_max ~ 2.96e6"). SPEC examples round it to 2.9626e6.
    cells = Cells.log(300, paper.gamma_hi, 19)
    assert cells.n == 19 and cells.chart == "log" and cells.order == (2,) * 19
    np.testing.assert_allclose(np.asarray(cells.edges), paper.edges, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(cells.reference), paper.gamma_b, rtol=1e-14)
    np.testing.assert_allclose(np.asarray(cells.half_width), paper.h, rtol=1e-13)
    assert float(cells.half_width[0]) == pytest.approx(0.2420, abs=5e-5)
    assert float(cells.edges[0]) == 300.0 and float(cells.edges[-1]) == paper.gamma_hi


def test_linear_cells_with_units():
    cells = Cells.linear("2 uG", "8 uG", 3, order=1)
    np.testing.assert_allclose(
        np.asarray(cells.edges), [2e-6, 4e-6, 6e-6, 8e-6], rtol=1e-15
    )
    np.testing.assert_allclose(
        np.asarray(cells.reference), [3e-6, 5e-6, 7e-6], rtol=1e-15
    )
    np.testing.assert_allclose(np.asarray(cells.half_width), [1e-6] * 3, rtol=1e-14)
    assert cells.given_unit == "uG" and cells.order == (1, 1, 1)
    same = Cells.linear(2, 8, 3, order=1, unit="uG")
    np.testing.assert_array_equal(np.asarray(same.edges), np.asarray(cells.edges))


def test_log_field_cells():
    cells = Cells.log("0.5 uG", "50 uG", 4)
    np.testing.assert_allclose(
        np.asarray(cells.edges)[[0, -1]], [5e-7, 5e-5], rtol=1e-15
    )
    np.testing.assert_allclose(
        np.asarray(cells.half_width), math.log(100) / 8, rtol=1e-14
    )


def test_from_edges_with_one_order_per_cell():
    cells = Cells.from_edges([300, 1e3, 1e4, 3e6], order=(2, 2, 4))
    assert cells.order == (2, 2, 4)
    np.testing.assert_allclose(
        np.asarray(cells.reference), np.sqrt([3e5, 1e7, 3e10]), rtol=1e-15
    )


def test_single_cell_without_edges():
    cells = Cells.single("5 uG", scale="1 uG")
    assert cells.edges is None and cells.n == 1 and cells.chart == "affine"
    assert float(cells.reference[0]) == pytest.approx(5e-6) and float(
        cells.half_width[0]
    ) == pytest.approx(1e-6)
    paper_point = Cells.single(20.0, scale=20.0)
    assert float(paper_point.reference[0]) == 20.0 and paper_point.given_unit is None


def test_at_points():
    cells = Cells.at_points([1e3, 1e5], [0.1, 0.2], chart="log", order=3)
    assert cells.edges is None and cells.n == 2 and cells.order == (3, 3)


def test_shift_moves_by_a_fraction_of_the_half_width():
    cells = Cells.log(300, 2.9626e6, 19)
    shifted = cells.shift(0.5)
    ratio = np.asarray(shifted.reference) / np.asarray(cells.reference)
    np.testing.assert_allclose(
        np.log(ratio), 0.5 * np.asarray(cells.half_width), rtol=1e-12
    )
    lin = Cells.linear(0, 4, 2).shift(-0.5)
    np.testing.assert_allclose(np.asarray(lin.reference), [0.5, 2.5])


def test_refine_and_with_order():
    cells = Cells.log(300, 2.9626e6, 19)
    fine = cells.refine(2)
    assert fine.n == 38
    np.testing.assert_allclose(
        np.asarray(fine.edges)[::2], np.asarray(cells.edges), rtol=1e-13
    )
    assert cells.with_order(4).order == (4,) * 19
    assert Cells.from_edges([1, 2, 3], order=(1, 3)).refine(2).order == (1, 1, 3, 3)


@pytest.mark.parametrize(
    "call",
    [
        lambda: Cells.log(10, 1, 3),
        lambda: Cells.log(-1, 10, 3),
        lambda: Cells.log(1, 10, 0),
        lambda: Cells.linear(1, 10, 2.5),
        lambda: Cells.from_edges([1, 3, 2]),
        lambda: Cells.from_edges([1, 2, 3], order=(1,)),
        lambda: Cells.log(1, 10, 2, order=-1),
        lambda: Cells.from_edges([1, 2], chart="weird"),
        lambda: Cells.single(1.0, scale=0.0),
    ],
)
def test_bad_cells_raise(call):
    with pytest.raises(ValueError):
        call()


def test_mixed_units_raise():
    with pytest.raises(UnitError):
        Cells.linear("2 uG", "8 GHz", 3)
    with pytest.raises(UnitError):
        Cells.linear("2 uG", "8 uG", 3, unit="uG")


# ------------------------------------------------------------------ placement (step-1 acceptance, real class)


def test_bare_cells_on_field_strength_raise():
    with pytest.raises(UnitError) as info:
        SYNCHROTRON_VARIABLES["B"].place(Cells.linear(2, 8, 3))
    assert str(info.value) == (
        'B=se.Cells.linear(2, 8, 3) has no unit. Write se.Cells.linear("2 uG", "8 uG", 3), or pass unit="uG".'
    )


def test_cells_with_a_field_unit_place():
    cells = Cells.linear("2 uG", "8 uG", 3)
    assert SYNCHROTRON_VARIABLES["B"].place(cells) is cells
    assert SYNCHROTRON_VARIABLES["gamma"].place(Cells.log(300, 3e6, 2)) is not None


def test_cells_of_angles_on_a_cosine_raise():
    with pytest.raises(UnitError, match="range of angles on a cosine"):
        SYNCHROTRON_VARIABLES["eta"].place(Cells.linear("10 deg", "50 deg", 2))


def test_repr():
    assert repr(Cells.linear(2, 8, 3)) == "se.Cells.linear(2, 8, 3)"
    assert repr(Cells.log(300, 2.9626e6, 19)) == "se.Cells.log(300, 2962600.0, 19)"
    assert (
        repr(Cells.linear("2 uG", "8 uG", 3, order=1))
        == "se.Cells.linear('2 uG', '8 uG', 3, order=1)"
    )
