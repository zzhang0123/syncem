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


@pytest.mark.parametrize("base", [0.0, 2.0**20, -(2.0**20)])
def test_narrow_cell_maps_depend_on_extent(base):
    width = 2.0**-22

    def lattice(start, stop, n=1):
        return Lattice(x=Cells.linear(start, stop, n, order=1), weights=np.ones(n))

    source = lattice(base, base + width)
    other = lattice(base + 2 * width, base + 3 * width)
    atom = Moments(source, [1.0, 0.0])
    with pytest.raises(ValueError, match="same cells"):
        source.recentre(atom, to=other)
    with pytest.raises(ValueError, match="unions of cells"):
        source.coarsen(atom, to=other)
    np.testing.assert_allclose(source.recentre(atom, to=source).values, atom.values)
    pair = lattice(base, base + 2 * width, 2)
    merged = pair.coarsen(
        Moments(pair, [1, 0, 3, 0]), to=lattice(base, base + 2 * width)
    )
    np.testing.assert_allclose(merged.values, [4, 1])


def test_coarsen_detects_an_interior_hole_even_if_its_volume_rounds_away():
    # A small square removed from the interior changes the float64 volume by < 1 ulp.
    edges = [0.0, 0.5, 0.5 + 2.0**-30, 1.0]
    axis = Cells.from_edges(edges, order=0, chart="affine")
    full = Lattice(x=axis, y=axis)
    sparse = Lattice.from_cells(full.cells.rows([0, 1, 2, 3, 5, 6, 7, 8]))
    unit = Cells.linear(0, 1, 1, order=0)
    target = Lattice(x=unit, y=unit)
    with pytest.raises(ValueError, match="not a union"):
        sparse.coarsen(Moments(sparse, np.ones(sparse.n)), to=target)
    joined = full.coarsen(Moments(full, np.ones(full.n)), to=target)
    np.testing.assert_allclose(
        target.column(joined), full.column(Moments(full, np.ones(full.n)))
    )


def test_one_representable_boundary_shift_excludes_a_unit_atom():
    lower = 0.5
    source = Lattice(x=Cells.linear(lower, 1, 1, order=1), weights=[1])
    at_edge = Moments(source, [1, -1])
    for different in (np.nextafter(lower, 1), lower + 5e-13):
        target = Lattice(x=Cells.linear(different, 1, 1, order=1), weights=[1])
        with pytest.raises(ValueError, match="same cells"):
            source.recentre(at_edge, to=target)
        with pytest.raises(ValueError, match="unions of cells"):
            source.coarsen(at_edge, to=target)


def test_coarsen_preserves_first_owner_of_shared_faces_and_corners():
    from syncem import Delta, Population
    from syncem.population.pairing import pair

    fine = Cells.linear(0, 4, 4, order=0)
    coarse = Cells.linear(0, 4, 2, order=0)
    for source, target, point in (
        (Lattice(x=fine), Lattice(x=coarse), Population(x=Delta(2))),
        (
            Lattice(x=fine, y=fine),
            Lattice(x=coarse, y=coarse),
            Population(x=Delta(2), y=Delta(2)),
        ),
    ):
        mass = pair(point, source)
        direct = pair(point, target)
        np.testing.assert_allclose(
            source.coarsen(mass, to=target).values, direct.values
        )
        reverse = Lattice.from_cells(
            target.cells.rows(list(range(target.n_cells))[::-1])
        )
        with pytest.raises(ValueError, match="ownership"):
            source.coarsen(mass, to=reverse)


def test_log_map_resolves_one_ulp_between_large_reference_points():
    from syncem import Delta, Population
    from syncem.population.pairing import pair

    a = 2.0**20
    ulp = np.spacing(a)
    edges = [a, a + 1024 * ulp]

    def lattice(ref):
        return Lattice(
            gamma=Cells.from_edges(edges, chart="log", reference=[ref], order=1),
            weights=[1],
        )

    x = a + 512 * ulp
    other = x + ulp
    src, dst = lattice(x), lattice(other)
    source = Population(gamma=Delta(x))
    mass = pair(source, src)
    expected = [1.0, np.log1p((x - other) / other) / dst.cells.half_width[0, 0]]
    for mapped in (
        pair(source, dst),
        src.recentre(mass, to=dst),
        src.coarsen(mass, to=dst),
    ):
        np.testing.assert_allclose(mapped.values, expected, rtol=2e-15, atol=0)


def test_narrow_log_refine_preserves_distinct_canonical_edges():
    lo = 2.0**20
    ulp = np.spacing(lo)
    axis = Cells.log(lo, lo + 4 * ulp, 2, order=1)
    np.testing.assert_array_equal(axis.edges, lo + ulp * np.array([0, 2, 4]))
    fine = axis.refine(2)
    np.testing.assert_array_equal(fine.edges, lo + ulp * np.arange(5))
    np.testing.assert_allclose(
        fine.half_width,
        0.5 * np.log1p(np.diff(fine.edges) / fine.edges[:-1]),
        rtol=2e-15,
    )


@pytest.mark.parametrize("chart", ["log", "affine"])
@pytest.mark.parametrize(
    "origin,width_ulps", [(2.0**-20, 4), (1.5 * 2.0**20, 32), (2.0**400, 1024)]
)
def test_rounded_continuous_nodes_keep_their_cell_and_local_moments(
    chart, origin, width_ulps
):
    from syncem import Delta, Population, Uniform
    from syncem.population.pairing import pair

    upper = origin + width_ulps * np.spacing(origin)
    make = Cells.log if chart == "log" else Cells.linear
    fine = Lattice(x=make(origin, upper, 2, order=3), weights=[1.0, 1.0])
    coarse = Lattice(x=make(origin, upper, 1, order=3), weights=[1.0])
    uniform = Population(x=Uniform(origin, upper))
    boundary = fine.cells.hi[0, 0]
    mixture = 0.7 * uniform + 0.3 * Population(x=Delta(boundary))
    roots, weights = np.polynomial.legendre.leggauss(12)

    for lattice in (fine, coarse):
        # Compute relative to physical faces before adding the large origin.
        expected = []
        for cell in range(lattice.n_cells):
            lo, hi = lattice.cells.lo[cell, 0], lattice.cells.hi[cell, 0]
            ref = lattice.cells.reference[cell, 0]
            half_width = lattice.cells.half_width[cell, 0]
            delta = (lo - ref) + 0.5 * (roots + 1) * (hi - lo)
            atom = boundary - ref
            if chart == "log":
                delta, atom = np.log1p(delta / ref), np.log1p(atom / ref)
            mass = 0.35 * (hi - lo) / (upper - origin)
            for power in range(4):
                val = mass * (weights @ ((delta / half_width) ** power))
                if cell == 0:
                    val += 0.3 * (atom / half_width) ** power
                expected.append(val)
        np.testing.assert_allclose(
            pair(mixture, lattice).values, expected, rtol=3e-14, atol=4e-15
        )
    result = fine.coarsen(pair(mixture, fine), to=coarse)
    np.testing.assert_allclose(
        result.values, pair(mixture, coarse).values, rtol=3e-14, atol=4e-15
    )


def test_normalised_smooth_families_conserve_column_on_narrow_log_cells():
    import jax.numpy as jnp
    from syncem import family, Population, PowerLaw
    from syncem.population.pairing import pair

    lo = 1.5 * 2.0**20
    hi = lo + 4 * np.spacing(lo)
    lat = Lattice(gamma=Cells.log(lo, hi, 2, order=0), weights=[1.0, 1.0])
    for law, total in (
        (PowerLaw(2, lo=lo, hi=hi, column=3), 3.0),
        (
            family(variable="gamma", support=(lo, hi))(
                lambda gamma: jnp.ones_like(gamma)
            )(),
            1.0,
        ),
    ):
        result = pair(Population(gamma=law), lat).values
        np.testing.assert_allclose(result, total * np.array([0.5, 0.5]), rtol=5e-14)
