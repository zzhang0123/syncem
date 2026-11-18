"""Step 2 of design/0.8/SPEC.md Sec. 10.2: ``Lattice``, ``CellTable`` and ``MomentIndex`` (Sec. 3.3, 4.1-4.4, 4.7)."""

import sys
import time
from pathlib import Path

import numpy as np
import pytest

from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import AzimuthHarmonic, Legendre
from syncem.values.errors import SizeError

STUDY_LIB = Path(__file__).resolve().parents[2] / "studies" / "classical-models" / "lib"


def energy():
    return Cells.log(300, 2.9626e6, 19)


# ------------------------------------------------------------------ the row counts of Sec. 4.7


def test_e1_rows():
    lat = Lattice(gamma=energy())
    assert lat.n == 57 and lat.n_cells == 19


def test_e5_joint_rows_and_labels():
    lat = Lattice(gamma=energy(), phi=AzimuthHarmonic())
    assert lat.n == 171
    assert lat.labels()[:4] == (
        "M0[gamma=0 | u^0]",
        "M0[gamma=0 | u^1]",
        "M0[gamma=0 | u^2]",
        "M0[gamma=1 | u^0]",
    )
    assert lat.labels()[57] == "Re M2[gamma=0 | u^0]"
    assert lat.labels()[114] == "Im M2[gamma=0 | u^0]"
    assert (
        lat.index.position(block="Im M2", gamma=7, powers={"gamma": 1})
        == 114 + 7 * 3 + 1
    )


def test_e6_rows():
    lat = Lattice(gamma=energy(), B=Cells.single("5 uG", scale="1 uG"), order=2)
    assert lat.n == 114
    monomials = {tuple(r.powers.values()) for r in lat.index.rows[:6]}
    assert monomials == {(0, 0), (1, 0), (0, 1), (2, 0), (1, 1), (0, 2)}


def test_e4_kp_rows():
    lat = Lattice(gamma=energy(), mu=Legendre(8))
    assert lat.n == 9 * 57 == 513


def test_energy_and_depth_rows_depth_outside_the_total():
    lat = Lattice(gamma=energy(), depth=Cells.linear(6, 10, 3), order=2)
    assert lat.n == 19 * 3 * 9 == 513
    powers = {
        tuple(r.powers[v] for v in ("gamma", "depth")) for r in lat.index.rows[:9]
    }
    assert powers == {(r, s) for r in range(3) for s in range(3)}


def test_benchmark_rows_with_depth_inside_the_total():
    single = dict(
        gamma=Cells.single(1e3, scale=1e2),
        B=Cells.single("5 uG", scale="1 uG"),
        depth=Cells.single(8.0, scale=1.0),
    )
    lat = Lattice(
        **single,
        mu=Legendre(8),
        eta=Legendre(8),
        phi=AzimuthHarmonic(),
        order=2,
        depth_in_total=True,
    )
    assert lat.n == 3 * 10 * 81 == 2430
    lat_out = Lattice(
        **single, mu=Legendre(8), eta=Legendre(8), phi=AzimuthHarmonic(), order=2
    )
    assert lat_out.n == 3 * (6 * 3) * 81


def test_position_is_b_times_n_plus_one_plus_r():
    for order in (0, 2, 4):
        lat = Lattice(gamma=energy().with_order(order))
        for b in (0, 7, 18):
            for r in range(order + 1):
                assert (
                    lat.index.position(gamma=b, powers={"gamma": r})
                    == b * (order + 1) + r
                )


def test_rows_view_matches_positions():
    lat = Lattice(gamma=energy(), mu=Legendre(2), phi=AzimuthHarmonic())
    for k in (0, 5, 56, 171, 400, lat.n - 1):
        row = lat.index.rows[k]
        assert (
            lat.index.position(
                block=row.block, cell=row.cell, powers=row.powers, degrees=row.degrees
            )
            == k
        )
    assert lat.index.rows[57 * 3].pair == 57 * 3 * 2


# ------------------------------------------------------------------ order rules


def test_order_none_takes_the_largest_axis_order():
    lat = Lattice(gamma=energy(), B=Cells.single("5 uG", scale="1 uG", order=1))
    assert lat.order == 2
    assert lat.n == 19 * 5  # (r, s) with r <= 2, s <= 1, r + s <= 2


def test_order_above_the_sum_of_axis_orders_raises():
    with pytest.raises(ValueError, match="with_order"):
        Lattice(gamma=energy(), order=4)


def test_unequal_orders_are_rows_of_the_table():
    lat = Lattice(gamma=Cells.from_edges([300, 1e3, 1e4, 3e6], order=(2, 2, 4)))
    assert lat.n == 3 + 3 + 5
    assert lat.index.position(gamma=2, powers={"gamma": 4}) == 10


# ------------------------------------------------------------------ selectors, refine, select, from_cells


def test_interval_selector():
    lat = Lattice(gamma=energy(), phi=AzimuthHarmonic())
    cells = lat.cells
    inside = [c for c in range(19) if cells.lo[c, 0] >= 1e4 and cells.hi[c, 0] <= 1e5]
    rows = lat.index.select(gamma=(1e4, 1e5))
    assert len(rows) == len(inside) * 3 * 3
    assert {lat.index.rows[k].cell for k in rows} == set(inside)


def test_refine_where_splits_only_the_selected_cell():
    gamma = Cells.log(300, 3e5, 4)
    field = Cells.linear("2 uG", "8 uG", 3)
    lat = Lattice(gamma=gamma, B=field, order=2)
    fine = lat.refine(where={"gamma": 0, "B": 0}, gamma=2, B=2)
    assert fine.n_cells == 12 - 1 + 4
    assert fine.n == fine.n_cells * 6
    assert fine.cells.tensor is None
    with pytest.raises(AttributeError, match="cells"):
        fine.axes  # noqa: B018
    corner = fine.index.select(gamma=(300.0, float(gamma.edges[1])), B=(2e-6, 4e-6))
    assert len({fine.index.rows[k].cell for k in corner}) == 4
    lo = np.asarray(fine.cells.lo)
    assert (
        np.isclose(lo[:, 0], 300.0).sum() == 2 + 2
    )  # two sub-cells of the corner and the B=1, B=2 cells


def test_select_gives_a_sub_lattice():
    lat = Lattice(gamma=energy())
    sub = lat.select(gamma=slice(3, None))
    assert sub.n_cells == 16 and sub.n == 48
    np.testing.assert_array_equal(
        np.asarray(sub.cells.lo), np.asarray(lat.cells.lo)[3:]
    )


def test_from_cells_rejects_overlaps_and_accepts_holes():
    lat = Lattice(gamma=energy())
    table = lat.cells
    keep = [0, 1, 5, 6]
    sub = Lattice.from_cells(table.rows(keep))
    assert sub.n_cells == 4 and sub.cells.tensor is None
    overlapping = table.rows([0, 1]).with_bounds(
        hi=[[float(table.hi[1, 0])], [float(table.hi[1, 0])]]
    )
    with pytest.raises(ValueError, match="overlap"):
        Lattice.from_cells(overlapping)


def test_axes_view_on_a_tensor_table():
    lat = Lattice(gamma=energy(), mu=Legendre(8))
    assert set(lat.axes) == {"gamma", "mu"}
    assert lat.axes["mu"].degree == 8


def test_placement_runs_at_construction():
    from syncem.values.errors import UnitError

    with pytest.raises(UnitError, match="has no unit"):
        Lattice(gamma=energy(), B=Cells.linear(2, 8, 3))
    with pytest.raises(ValueError, match="cosine"):
        Lattice(gamma=energy(), mu=Cells.linear(-2, 2, 2))
    with pytest.raises(ValueError, match="Legendre"):
        Lattice(gamma=Legendre(3))


# ------------------------------------------------------------------ reference weights and key


def test_reference_weights_follow_the_study():
    if str(STUDY_LIB) not in sys.path:
        sys.path.insert(0, str(STUDY_LIB))
    from config import PAPER  # noqa: PLC0415

    lat = Lattice(gamma=Cells.log(300, PAPER.gamma_hi, 19))
    np.testing.assert_allclose(np.asarray(lat.reference_weights), PAPER.w_b, rtol=1e-13)
    assert (
        float(Lattice(B=Cells.single("5 uG", scale="1 uG")).reference_weights[0]) == 1.0
    )
    weights = Lattice(
        gamma=Cells.log(300, PAPER.gamma_hi, 19), B=Cells.linear("2 uG", "8 uG", 3)
    ).reference_weights
    np.testing.assert_allclose(
        np.asarray(weights).reshape(19, 3).sum(axis=1) / 3, PAPER.w_b, rtol=1e-13
    )


def test_explicit_weights():
    w = np.linspace(1, 2, 19)
    lat = Lattice(gamma=energy(), weights=w)
    np.testing.assert_array_equal(np.asarray(lat.reference_weights), w)
    with pytest.raises(ValueError):
        Lattice(gamma=energy(), weights=np.ones(3))


def test_key_depends_on_cells_orders_and_weights():
    a, b = Lattice(gamma=energy()), Lattice(gamma=energy())
    assert a.key == b.key and len(a.key) == 64
    assert Lattice(gamma=energy().with_order(3)).key != a.key
    assert Lattice(gamma=energy(), weights=np.ones(19)).key != a.key
    assert Lattice(gamma=energy().shift(0.25)).key != a.key
    assert a.with_key("abc").key == "abc"


# ------------------------------------------------------------------ the G15 lattice and the size cap


def g15():
    return Lattice(
        gamma=Cells.log(300, 1e7, 22),
        B=Cells.log("1 uG", "30 uG", 4),
        eta=Legendre(8),
        phi=AzimuthHarmonic(),
        depth=Cells.linear(-60, 60, 48, order=3),
        order=2,
    )


def test_g15_lattice_builds_without_per_row_objects():
    start = time.perf_counter()
    lat = g15()
    elapsed = time.perf_counter() - start
    assert lat.n == 2_737_152 and lat.n_cells == 22 * 4 * 48
    assert lat.key is not None
    assert elapsed < 5.0
    assert (
        lat.index.position(
            block="Im M2",
            gamma=21,
            B=3,
            depth=47,
            powers={"gamma": 0, "B": 2, "depth": 3},
            degrees={"eta": 8},
        )
        == lat.n - 1
    )


def test_size_cap_refuses_with_the_numbers():
    lat = g15()
    with pytest.raises(SizeError) as info:
        lat.check_size(n_columns=1_140_480)
    assert str(info.value) == (
        "the lattice has 2 737 152 rows and 1 140 480 basis columns (21.9 MB of moments per sightline), "
        "above max_columns=20 000. Pass max_columns=1_200_000 to accept, or use fewer cells or a lower order."
    )
    lat.check_size(n_columns=1_140_480, max_columns=1_200_000)
