"""Step 2 of design/0.8/SPEC.md Sec. 10.2: ``Moments``, ``recentre``, ``coarsen``, ``column`` and ``raw`` (Sec. 3.3, 4.5)."""

import itertools

import numpy as np
import pytest

from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.moments import Moments
from syncem.lattice.representations import AzimuthHarmonic

GAMMA_CUT = 2.3e4  # a cut-off inside a cell


def gamma_density(g):
    return np.where(g < GAMMA_CUT, g**-2.2, 0.0)


def field_density(b):
    return 1.0 + 0.3 * np.sin(b / 1e-6)


def _panel(lo, hi, chart, n=48, breaks=()):
    """Gauss-Legendre nodes and weights in x on [lo, hi], split at breaks, uniform in the chart."""
    t, w = np.polynomial.legendre.leggauss(n)
    cuts = sorted({lo, hi, *[b for b in breaks if lo < b < hi]})
    xs, ws = [], []
    for a, b in zip(cuts[:-1], cuts[1:]):
        if chart == "log":
            la, lb = np.log(a), np.log(b)
            s = 0.5 * (lb - la) * t + 0.5 * (la + lb)
            xs.append(np.exp(s))
            ws.append(0.5 * (lb - la) * w * np.exp(s))
        else:
            xs.append(0.5 * (b - a) * t + 0.5 * (a + b))
            ws.append(0.5 * (b - a) * w)
    return np.concatenate(xs), np.concatenate(ws)


def direct_moments(lat, densities, breaks):
    """``a_alpha = (1/w_c) int psi_alpha prod_v f_v dx`` by product quadrature, one cell at a time."""
    table, index = lat.cells, lat.index
    values = np.zeros(lat.n)
    for c in range(lat.n_cells):
        grids = []
        for v, name in enumerate(table.variables):
            x, w = _panel(
                float(table.lo[c, v]),
                float(table.hi[c, v]),
                table.chart[c][v],
                breaks=breaks.get(name, ()),
            )
            ref, h = float(table.reference[c, v]), float(table.half_width[c, v])
            u = np.log(x / ref) / h if table.chart[c][v] == "log" else (x - ref) / h
            grids.append((u, w * densities[name](x)))
        for k in index.cell_rows(c):
            powers = index.rows[k].powers
            total = 1.0
            for (u, wf), name in zip(grids, table.variables):
                total *= np.sum(wf * u ** powers[name])
            values[k] = total / float(lat.reference_weights[c])
    return values


def rel(a, b):
    return np.max(np.abs(np.asarray(a) - np.asarray(b))) / np.max(np.abs(np.asarray(b)))


def test_recentre_is_exact():
    cells = Cells.log(300, 3e5, 6, order=3)
    a, b = Lattice(gamma=cells), Lattice(gamma=cells.shift(0.3))
    ma = Moments(a, direct_moments(a, {"gamma": gamma_density}, {"gamma": [GAMMA_CUT]}))
    mb = direct_moments(b, {"gamma": gamma_density}, {"gamma": [GAMMA_CUT]})
    out = a.recentre(ma, to=b)
    assert out.key == b.key
    assert rel(out.values, mb) < 1e-12


def test_coarsen_is_exact():
    fine, coarse = (Lattice(gamma=Cells.log(300, 3e5, n, order=3)) for n in (6, 3))
    mf = Moments(
        fine, direct_moments(fine, {"gamma": gamma_density}, {"gamma": [GAMMA_CUT]})
    )
    mc = direct_moments(coarse, {"gamma": gamma_density}, {"gamma": [GAMMA_CUT]})
    assert rel(fine.coarsen(mf, to=coarse).values, mc) < 1e-12


def test_corner_split_coarsens_back_to_the_parent_table():
    parent = Lattice(
        gamma=Cells.log(300, 3e5, 4), B=Cells.linear("2 uG", "8 uG", 3), order=2
    )
    split = parent.refine(where={"gamma": 0, "B": 0}, gamma=2, B=2)
    densities = {"gamma": gamma_density, "B": field_density}
    breaks = {"gamma": [GAMMA_CUT]}
    ms = Moments(split, direct_moments(split, densities, breaks))
    assert (
        rel(
            split.coarsen(ms, to=parent).values,
            direct_moments(parent, densities, breaks),
        )
        < 1e-12
    )


def test_corner_split_equals_the_uniform_refinement_on_shared_cells():
    parent = Lattice(
        gamma=Cells.log(300, 3e5, 4), B=Cells.linear("2 uG", "8 uG", 3), order=2
    )
    split = parent.refine(where={"gamma": 0, "B": 0}, gamma=2, B=2)
    uniform = Lattice(
        gamma=Cells.log(300, 3e5, 8), B=Cells.linear("2 uG", "8 uG", 6), order=2
    )
    densities = {"gamma": gamma_density, "B": field_density}
    ms, mu = direct_moments(split, densities, {}), direct_moments(
        uniform, densities, {}
    )
    corner = (300.0, float(np.asarray(parent.cells.hi)[0, 0]))
    rows_s = split.index.select(gamma=corner, B=(2e-6, 4e-6))
    rows_u = uniform.index.select(gamma=corner, B=(2e-6, 4e-6))
    assert len(rows_s) == len(rows_u) == 4 * 6
    np.testing.assert_allclose(ms[list(rows_s)], mu[list(rows_u)], rtol=1e-13)


def test_coarsen_refusals():
    fine = Lattice(gamma=Cells.log(300, 3e5, 6, order=3))
    m = Moments(fine, np.zeros(fine.n))
    with pytest.raises(ValueError, match="union"):
        fine.coarsen(m, to=Lattice(gamma=Cells.log(400, 3e5, 3, order=3)))
    with pytest.raises(ValueError, match="chart"):
        fine.coarsen(m, to=Lattice(gamma=Cells.linear(300, 3e5, 3, order=3)))
    with pytest.raises(ValueError, match="order"):
        fine.coarsen(m, to=Lattice(gamma=Cells.log(300, 3e5, 3, order=4)))


def test_moments_of_another_lattice_are_refused():
    a = Lattice(gamma=Cells.log(300, 3e5, 6))
    b = Lattice(gamma=Cells.log(300, 3e5, 5))
    with pytest.raises(ValueError, match="key"):
        a.column(Moments(b, np.zeros(b.n)))
    with pytest.raises(ValueError):
        Moments(a, np.zeros(a.n + 1))


def test_column_and_raw():
    lat = Lattice(gamma=Cells.log(300, 3e5, 6), phi=AzimuthHarmonic())
    values = np.zeros(lat.n)
    values[: lat.n // 3] = direct_moments(
        Lattice(gamma=Cells.log(300, 3e5, 6)),
        {"gamma": gamma_density},
        {"gamma": [GAMMA_CUT]},
    )
    m = Moments(lat, values)
    x, w = _panel(300.0, 3e5, "log", n=400, breaks=[GAMMA_CUT])
    assert float(lat.column(m)) == pytest.approx(
        float(np.sum(w * gamma_density(x))), rel=1e-12
    )
    raw = np.asarray(lat.raw(m))
    w_rows = np.repeat(np.asarray(lat.reference_weights), 3)
    np.testing.assert_allclose(
        raw[: lat.n // 3], w_rows * values[: lat.n // 3], rtol=1e-15
    )


def test_moments_fields_and_batches():
    lat = Lattice(gamma=Cells.log(300, 3e5, 6))
    m = Moments(lat, np.ones((4, lat.n)), origin="minimum_norm")
    assert (
        m.values.shape == (4, 18)
        and m.key == lat.key
        and m.masked == ()
        and m.origin == "minimum_norm"
    )
    assert "M0[gamma=0 | u^0]" in m.table()
    with pytest.raises(ValueError):
        Moments(lat, np.ones(lat.n), origin="guessed")


def test_cell_rows_and_monomials():
    lat = Lattice(
        gamma=Cells.log(300, 3e5, 2), B=Cells.single("5 uG", scale="1 uG"), order=2
    )
    rows = list(lat.index.cell_rows(1))
    assert rows == list(range(6, 12))
    exps = [tuple(lat.index.rows[k].powers.values()) for k in rows]
    assert exps == [
        e for e in itertools.chain([(0, 0)], [(1, 0), (0, 1)], [(2, 0), (1, 1), (0, 2)])
    ]
