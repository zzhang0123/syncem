"""Regression tests for the step-2 review findings (lattice layer, design/0.8/SPEC.md Sec. 3.3, 4)."""

import math

import numpy as np
import pytest

from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.moments import Moments
from syncem.lattice.representations import AzimuthHarmonic, Legendre
from syncem.units import Quantity, UnitError

GAMMA_HI = 2962642.632683605


def energy():
    return Cells.log(300, GAMMA_HI, 19)


# ------------------------------------------------------------------ HIGH: totals on derived tensor lattices


def test_with_order_changes_the_total_on_a_tensor_lattice():
    lat = Lattice(gamma=energy())
    assert lat.with_order(4).n == 19 * 5 and lat.with_order(4).order == 4
    assert lat.with_order(1).n == 19 * 2
    two = Lattice(gamma=energy(), B=Cells.linear("2 uG", "8 uG", 3))
    assert two.with_order(1).n == 57 * 3


def test_replace_recomputes_an_implicit_total_and_keeps_an_explicit_one():
    lat = Lattice(gamma=energy())
    assert lat.replace(gamma=energy().with_order(4)).n == 19 * 5
    fixed = Lattice(gamma=energy(), B=Cells.linear("2 uG", "8 uG", 3), order=2)
    assert fixed.replace(gamma=energy().with_order(4)).order == 2


# ------------------------------------------------------------------ MEDIUM


def test_interval_with_an_infinite_end():
    lat = Lattice(gamma=Cells.from_edges([300, 1e3, 1e4, 3e6]))
    assert lat.index.cells_where(gamma=(1e4, math.inf)).tolist() == [2]
    assert lat.index.cells_where(gamma=(-math.inf, 1e4)).tolist() == [0, 1]


@pytest.mark.parametrize(
    "axes", [dict(mu=Legendre(2)), dict(phi=AzimuthHarmonic()), dict()]
)
def test_lattice_without_cells(axes):
    lat = Lattice(**axes)
    assert lat.n_cells == 1 and lat.n >= 1


def test_recentre_refuses_cells_that_differ_slightly():
    a = Lattice(B=Cells.linear("2 uG", "8 uG", 3))
    b = Lattice(B=Cells.linear("2.002 uG", "8.002 uG", 3))
    with pytest.raises(ValueError, match="same cells"):
        a.recentre(Moments(a, np.ones(a.n)), to=b)


def test_coarsen_refuses_a_target_that_the_sources_do_not_fill():
    fine = Lattice(B=Cells.linear("2 uG", "8 uG", 600, order=1))
    sub = Lattice.from_cells(fine.cells.rows(np.arange(1, 600)))
    with pytest.raises(ValueError, match="union"):
        sub.coarsen(
            Moments(sub, np.ones(sub.n)),
            to=Lattice(B=Cells.linear("2 uG", "8 uG", 1, order=1)),
        )


def test_maps_carry_masked_retained_and_component():
    a = Lattice(gamma=Cells.log(300, 3e5, 4, order=2))
    b = Lattice(gamma=Cells.log(300, 3e5, 4, order=2).shift(0.3))
    m = Moments(
        a, np.ones(a.n), origin="minimum_norm", masked=(4,), retained="r", component="c"
    )
    out = a.recentre(m, to=b)
    assert out.retained == "r" and out.component == "c" and out.origin == "minimum_norm"
    assert set(out.masked) == {4, 5}  # u^1 and u^2 of cell 1 depend on u^1 of cell 1
    coarse = Lattice(gamma=Cells.log(300, 3e5, 2, order=2))
    assert set(a.coarsen(m, to=coarse).masked) == {1, 2}


@pytest.mark.parametrize(
    "bad",
    [
        dict(variables=("mu",), lo=[[-3.0]], hi=[[3.0]], ref=[[0.0]]),
        dict(variables=("B",), lo=[[-1e-6]], hi=[[1e-6]], ref=[[0.0]]),
        dict(variables=("gamma",), lo=[[300.0]], hi=[[100.0]], ref=[[200.0]]),
        dict(variables=("gamma",), lo=[[300.0]], hi=[[np.nan]], ref=[[400.0]]),
        dict(variables=("gamma",), lo=[[300.0]], hi=[[1000.0]], ref=[[2000.0]]),
        dict(variables=("phi",), lo=[[0.0]], hi=[[1.0]], ref=[[0.5]]),
    ],
)
def test_from_cells_checks_boxes(bad):
    from syncem.lattice.table import CellTable

    chart = (("affine",),)
    table = CellTable(
        bad["variables"],
        np.array(bad["lo"]),
        np.array(bad["hi"]),
        np.array(bad["ref"]),
        np.array([[0.5]]),
        chart,
        ((2,),),
        None,
    )
    with pytest.raises(ValueError):
        Lattice.from_cells(table)


def test_from_cells_refuses_cells_in_representations():
    table = Lattice(gamma=energy()).cells
    with pytest.raises(ValueError):
        Lattice.from_cells(
            table, representations={"B": Cells.linear("2 uG", "8 uG", 3)}
        )
    with pytest.raises(ValueError):
        Lattice.from_cells(table, representations={"gamma": Legendre(3)})


@pytest.mark.parametrize(
    "call",
    [
        lambda: Cells.single("5 uG", scale=1),
        lambda: Cells.single("5 uG", scale="1 GHz"),
        lambda: Cells.single("5 uG", scale="1 uG", support=(2, 8)),
        lambda: Cells.single("5 uG", scale="1 uG", support=("8 uG", "2 uG")),
    ],
)
def test_single_cell_units_and_support(call):
    with pytest.raises((UnitError, ValueError)):
        call()


def test_single_cell_support_converts():
    cells = Cells.single("5 uG", scale="1 uG", support=("2 uG", "8 uG"))
    assert cells.support == pytest.approx((2e-6, 8e-6))


@pytest.mark.parametrize(
    "axes",
    [
        dict(B=Cells.at_points(["-5 uG"], ["1 uG"])),
        dict(mu=Cells.single(2.0, scale=0.1)),
    ],
)
def test_reference_points_outside_the_kind_raise(axes):
    with pytest.raises(ValueError):
        Lattice(**axes)


@pytest.mark.parametrize(
    "edges",
    [(np.array([2.0, 4.0, 8.0]), "uG"), Quantity(np.array([2.0, 4.0, 8.0]), "uG")],
)
def test_cells_from_arrays_with_units(edges):
    cells = Cells.from_edges(edges, chart="affine")
    np.testing.assert_allclose(cells.edges, [2e-6, 4e-6, 8e-6], rtol=1e-15)
    assert cells.given_unit == "uG"


def test_cells_from_astropy():
    u = pytest.importorskip("astropy.units")
    cells = Cells.from_edges(np.array([2.0, 4.0, 8.0]) * u.uG, chart="affine")
    np.testing.assert_allclose(cells.edges, [2e-6, 4e-6, 8e-6], rtol=1e-14)
    assert cells.given_unit is not None
    lin = Cells.linear(2 * u.uG, 8 * u.uG, 3)
    np.testing.assert_allclose(lin.edges, [2e-6, 4e-6, 6e-6, 8e-6], rtol=1e-14)


def test_position_with_interval_selectors():
    lat = Lattice(
        gamma=Cells.log(300, 3e5, 4), B=Cells.linear("2 uG", "8 uG", 3), order=2
    )
    e1 = float(lat.cells.hi[0, 0])
    assert lat.index.position(gamma=(300.0, e1), B=(2e-6, 4e-6)) == 0
    fine = lat.refine(where={"gamma": 0, "B": 0}, gamma=2, B=2)
    k = fine.index.position(gamma=(300.0, e1), B=(4e-6, 6e-6), powers={"B": 1})
    assert fine.index.rows[k].powers["B"] == 1
    with pytest.raises(ValueError, match="one cell"):
        fine.index.position(gamma=(300.0, e1), B=(2e-6, 4e-6))


# ------------------------------------------------------------------ LOW


def test_unknown_powers_and_conflicting_selectors_raise():
    lat = Lattice(gamma=Cells.from_edges([300, 1e3, 1e4, 3e6]), mu=Legendre(2))
    with pytest.raises(ValueError):
        lat.index.position(gamma=1, powers={"gama": 1})
    with pytest.raises(ValueError):
        lat.index.position(gamma=1, powers={"mu": 1})
    with pytest.raises(ValueError):
        lat.index.position(cell=1, gamma=2)
    with pytest.raises(ValueError):
        lat.index.select(powers={"gama": 0})
    with pytest.raises(ValueError):
        lat.index.position(gamma=1, nonsense=3)


def test_cell_rows_range_checks():
    lat = Lattice(gamma=Cells.from_edges([300, 1e3, 1e4, 3e6]))
    with pytest.raises(IndexError):
        lat.index.cell_rows(-1)
    with pytest.raises(IndexError):
        lat.index.cell_rows(0, block=5)


def test_negative_zero_does_not_change_the_key():
    a = Lattice(depth=Cells.from_edges([-1.0, 0.0, 1.0], chart="affine"))
    b = Lattice(depth=Cells.from_edges([-1.0, -0.0, 1.0], chart="affine"))
    assert a.key == b.key


def test_coarsen_refuses_cells_without_edges():
    two = Lattice(gamma=Cells.at_points([1e3, 1e4], [0.5, 0.5], chart="log", order=2))
    with pytest.raises(ValueError):
        two.coarsen(
            Moments(two, np.ones(two.n)),
            to=Lattice(gamma=Cells.single(3e3, 0.5, chart="log", order=2)),
        )


def test_type_errors_name_the_argument():
    lat = Lattice(gamma=Cells.log(300, 3e5, 3))
    with pytest.raises(TypeError, match="to="):
        lat.recentre(Moments(lat, np.ones(lat.n)), to=Cells.log(300, 3e5, 3))
    with pytest.raises(TypeError, match="Lattice"):
        Moments(object(), np.zeros(3))
    with pytest.raises(ValueError):
        Moments(lat, np.zeros(lat.n), masked=(1.7,))


def test_refine_and_shift_name_checks():
    lat = Lattice(gamma=Cells.log(300, 3e5, 3), mu=Legendre(2))
    for call in (
        lambda: lat.refine(where={"gamma": 0}, mu=2),
        lambda: lat.refine(where={"gamma": 0}, gamma=1.5),
        lambda: lat.refine(mu=2),
        lambda: lat.shift(mu=0.5),
        lambda: lat.refine(where={"gamma": 0}, gamma=2).shift(mu=0.5),
    ):
        with pytest.raises(ValueError):
            call()


def test_order_without_free_variables_raises():
    with pytest.raises(ValueError):
        Lattice(depth=Cells.linear(-1, 1, 3), order=5)


def test_cells_reference_inside_and_spelling_follows():
    with pytest.raises(ValueError):
        Cells.log(300, 3e5, 3).shift(3.0)
    assert (
        repr(Cells.log(300, 3e6, 19).refine(2))
        == "se.Cells.log(300, 3000000.0, 19).refine(2)"
    )
    assert (
        repr(Cells.log(300, 3e6, 19).with_order(4))
        == "se.Cells.log(300, 3000000.0, 19).with_order(4)"
    )
    assert Cells.log(2, 8, 3, order=1).unit_remedy("uG") == (
        'Write se.Cells.log("2 uG", "8 uG", 3, order=1), or pass unit="uG".'
    )


# ------------------------------------------------------------------ verification round


def test_recentre_accepts_edges_equal_to_rounding():
    a = Cells.linear(-0.3, 0.7, 10, order=2)
    b = Cells.from_edges(np.round(a.edges, 12), chart="affine", order=2)
    la, lb = Lattice(eta=a), Lattice(eta=b)
    out = la.recentre(Moments(la, np.ones(la.n)), to=lb)
    assert np.all(np.isfinite(np.asarray(out.values)))


def test_recentre_of_edgeless_cells_emits_no_warning():
    import warnings

    e1 = Lattice(gamma=Cells.single(1e4, 0.5, chart="log", order=3))
    e2 = Lattice(gamma=Cells.single(2e4, 0.7, chart="log", order=3))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        e1.recentre(Moments(e1, np.ones(e1.n)), to=e2)


def test_masked_rows_are_deduplicated_and_bools_refused():
    lat = Lattice(gamma=Cells.log(300, 3e5, 3))
    assert Moments(lat, np.zeros(lat.n), masked=(3, 3)).masked == (3,)
    with pytest.raises(ValueError):
        Moments(lat, np.zeros(lat.n), masked=[True])


def test_position_range_errors_name_the_variable():
    lat = Lattice(gamma=Cells.from_edges([300, 1e3, 1e4, 3e6]), mu=Legendre(2))
    assert lat.index.position(gamma=-1) == lat.index.position(
        gamma=2
    )  # negatives count from the end
    for kwargs in (
        dict(gamma=-4),
        dict(gamma=3),
        dict(cell=0, degrees={"mu": 3}),
        dict(cell=True),
    ):
        with pytest.raises(ValueError, match="gamma|mu|cell"):
            lat.index.position(**kwargs)


def test_tensor_refine_factor_error_names_refine():
    with pytest.raises(ValueError, match="refine"):
        Lattice(gamma=Cells.log(300, 3e5, 3)).refine(gamma=1.5)


def test_shift_to_a_cell_face_is_accepted():
    for cells in (
        Cells.log(300, 3e5, 3).shift(1.0),
        Cells.linear("2 uG", "8 uG", 3).shift(-1.0),
    ):
        lat = Lattice(gamma=cells) if cells.chart == "log" else Lattice(B=cells)
        Lattice.from_cells(lat.cells)


def test_bare_reference_with_unit_edges_names_the_unit():
    with pytest.raises(UnitError, match="uG"):
        Cells.log("2 uG", "8 uG", 2, reference=[3, 6])
    cells = Cells.log("2 uG", "8 uG", 2, reference=["3 uG", "6 uG"])
    np.testing.assert_allclose(cells.reference, [3e-6, 6e-6])


def test_positive_cells_may_start_at_zero():
    Lattice(B=Cells.linear("0 uG", "8 uG", 4))
    with pytest.raises(ValueError, match="at or above zero"):
        Lattice(B=Cells.linear("-1 uG", "8 uG", 4))
