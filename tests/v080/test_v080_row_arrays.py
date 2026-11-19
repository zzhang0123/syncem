"""Step 5b-1 of design/0.8/SPEC.md: ``MomentIndex.arrays`` and ``CellTable.geometry_digests`` (decisions 146, 147).

The row arrays are checked against a brute-force loop over ``index.row(k)``. Every comparison is exact,
because both sides are integer arithmetic on the same index. The digests are checked for the invariances
and changes that row 146 states.
"""

import time

import numpy as np
import pytest

from syncem.lattice.cells import Cells
from syncem.lattice.index import MomentIndex
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import AzimuthHarmonic, Legendre
from syncem.lattice.table import CellTable


def energy():
    return Cells.log(300, 2.9626e6, 19)


def _benchmark(depth_in_total):
    return Lattice(
        gamma=Cells.single(1e3, scale=1e2),
        B=Cells.single("5 uG", scale="1 uG"),
        depth=Cells.single(8.0, scale=1.0),
        mu=Legendre(8),
        eta=Legendre(8),
        phi=AzimuthHarmonic(),
        order=2,
        depth_in_total=depth_in_total,
    )


def _corner_split(order):
    parent = Lattice(
        gamma=Cells.log(300, 3e5, 4), B=Cells.linear("2 uG", "8 uG", 3), order=order
    )
    return parent.refine(where={"gamma": 0, "B": 0}, gamma=2, B=2)


# The lattices of tests/v080/test_v080_lattice.py (without G15), corner-split tables and Legendre lattices.
LATTICES = {
    "e1": lambda: Lattice(gamma=energy()),
    "e5": lambda: Lattice(gamma=energy(), phi=AzimuthHarmonic()),
    "e6": lambda: Lattice(
        gamma=energy(), B=Cells.single("5 uG", scale="1 uG"), order=2
    ),
    "e4_kp": lambda: Lattice(gamma=energy(), mu=Legendre(8)),
    "energy_depth": lambda: Lattice(
        gamma=energy(), depth=Cells.linear(6, 10, 3), order=2
    ),
    "benchmark_in_total": lambda: _benchmark(True),
    "benchmark_out_of_total": lambda: _benchmark(False),
    "order_0": lambda: Lattice(gamma=energy().with_order(0)),
    "order_4": lambda: Lattice(gamma=energy().with_order(4)),
    "rows_view": lambda: Lattice(gamma=energy(), mu=Legendre(2), phi=AzimuthHarmonic()),
    "order_none": lambda: Lattice(
        gamma=energy(), B=Cells.single("5 uG", scale="1 uG", order=1)
    ),
    "unequal_orders": lambda: Lattice(
        gamma=Cells.from_edges([300, 1e3, 1e4, 3e6], order=(2, 2, 4))
    ),
    "refine_where": lambda: _corner_split(2),
    "corner_split_order_1": lambda: _corner_split(1),
    "corner_split_legendre": lambda: Lattice(
        gamma=Cells.log(300, 3e5, 4),
        B=Cells.linear("2 uG", "8 uG", 3),
        mu=Legendre(3),
        eta=Legendre(2),
        phi=AzimuthHarmonic(),
        order=2,
    ).refine(where={"gamma": 3, "B": 2}, gamma=3),
    "select": lambda: Lattice(gamma=energy()).select(gamma=slice(3, None)),
    "from_cells": lambda: Lattice.from_cells(
        Lattice(gamma=energy()).cells.rows([0, 1, 5, 6])
    ),
    "axes_view": lambda: Lattice(gamma=energy(), mu=Legendre(8)),
    "single_field": lambda: Lattice(B=Cells.single("5 uG", scale="1 uG")),
    "gamma_field": lambda: Lattice(
        gamma=Cells.log(300, 2.9626e6, 19), B=Cells.linear("2 uG", "8 uG", 3)
    ),
    "explicit_weights": lambda: Lattice(gamma=energy(), weights=np.linspace(1, 2, 19)),
}


def brute_force(index):
    """The row arrays from one ``index.row(k)`` call per row."""
    names, legendre = index.table.variables, index.legendre
    dims = tuple(d + 1 for _, d in legendre)
    out = {k: [] for k in ("block", "cell", "pair", "monomial", "powers", "degrees")}
    for k in range(index.n):
        row = index.row(k)
        powers = tuple(row.powers[v] for v in names)
        degrees = tuple(row.degrees[n] for n, _ in legendre)
        out["block"].append(index.blocks.index(row.block))
        out["cell"].append(row.cell)
        out["pair"].append(int(np.ravel_multi_index(degrees, dims)) if dims else 0)
        out["monomial"].append(
            index.patterns[int(index.cell_pattern[row.cell])].index(powers)
        )
        out["powers"].append(powers)
        out["degrees"].append(degrees)
    shapes = {"powers": (index.n, len(names)), "degrees": (index.n, len(legendre))}
    return {
        k: np.array(v, dtype=np.int64).reshape(shapes.get(k, (index.n,)))
        for k, v in out.items()
    }


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_row_arrays_equal_the_row_loop(name):
    index = LATTICES[name]().index
    arrays, expected = index.arrays, brute_force(index)
    for field, value in expected.items():
        got = getattr(arrays, field)
        assert got.dtype == np.int64, field
        np.testing.assert_array_equal(got, value, err_msg=f"{name}: {field}")


def test_row_arrays_are_cached_and_read_only():
    index = LATTICES["e5"]().index
    arrays = index.arrays
    assert index.arrays is arrays
    with pytest.raises(ValueError):
        arrays.cell[0] = 1
    with pytest.raises(ValueError):
        arrays.powers[0, 0] = 1


def test_row_arrays_leave_the_index_key_and_equality_alone():
    a, b = LATTICES["e6"]().index, LATTICES["e6"]().index
    key = a.key
    _ = a.arrays
    assert a.key == key == b.key and a == b and hash(a) == hash(b)


def g15():
    return Lattice(
        gamma=Cells.log(300, 1e7, 22),
        B=Cells.log("1 uG", "30 uG", 4),
        eta=Legendre(8),
        phi=AzimuthHarmonic(),
        depth=Cells.linear(-60, 60, 48, order=3),
        order=2,
    )


def test_g15_row_arrays_without_row_objects(monkeypatch):
    """G15 ``lat8`` (2 737 152 rows) builds its row arrays without ``index.row``.

    Bound: a probe of the same vectorised build took 0.023 to 0.028 s over five runs at a 1-minute load of
    8.9 (scratchpad probe of step 5b-1). The bound of 1.0 s is a margin of 35 over the slowest run.
    """

    def refuse(self, k):
        raise AssertionError("index.row was called")

    monkeypatch.setattr(MomentIndex, "row", refuse)
    index = g15().index
    start = time.perf_counter()
    arrays = index.arrays
    elapsed = time.perf_counter() - start
    assert elapsed < 1.0
    assert arrays.cell.shape == (2_737_152,) and arrays.powers.shape == (2_737_152, 3)
    assert arrays.degrees.shape == (2_737_152, 1)
    names = index.table.variables
    rng = np.random.default_rng(0)
    for k in [0, index.n - 1, *rng.integers(0, index.n, 200)]:
        k = int(k)
        position = index.position(
            block=index.blocks[arrays.block[k]],
            cell=int(arrays.cell[k]),
            powers=dict(zip(names, (int(x) for x in arrays.powers[k]))),
            degrees={"eta": int(arrays.degrees[k, 0])},
        )
        assert position == k


# ------------------------------------------------------------------ geometry digests (decision 146)


def _digests(lattice):
    return lattice.cells.geometry_digests()


def _with(table, **changes):
    fields = dict(
        variables=table.variables,
        lo=table.lo,
        hi=table.hi,
        reference=table.reference,
        half_width=table.half_width,
        chart=table.chart,
        order=table.order,
        tensor=table.tensor,
        excluded=table.excluded,
    )
    fields.update(changes)
    return CellTable(**fields)


def _bumped(array, cell, factor):
    out = np.array(array, dtype=float)
    out[cell, 0] *= factor
    return out


def test_digests_are_one_hex_string_per_cell_and_distinct():
    d = _digests(LATTICES["gamma_field"]())
    assert len(d) == 57 and all(len(x) == 64 for x in d) and len(set(d)) == 57
    assert isinstance(d, tuple)


def test_digests_ignore_orders_and_weights():
    base = _digests(Lattice(gamma=energy()))
    assert _digests(Lattice(gamma=energy().with_order(4))) == base
    assert _digests(Lattice(gamma=energy(), weights=np.linspace(1, 2, 19))) == base
    assert (
        Lattice(gamma=energy().with_order(4)).index.key
        != Lattice(gamma=energy()).index.key
    )


def test_digests_ignore_the_cell_order_of_the_table():
    table = Lattice(gamma=energy()).cells
    permutation = np.random.default_rng(1).permutation(19)
    shuffled = table.rows(permutation).geometry_digests()
    assert shuffled == tuple(table.geometry_digests()[i] for i in permutation)


@pytest.mark.parametrize("field", ["lo", "hi", "reference", "half_width"])
def test_digests_change_with_each_array_in_that_cell_only(field):
    table = Lattice(gamma=energy()).cells
    factor = 1.0 + 1e-15 if field != "half_width" else 2.0
    changed = _with(table, **{field: _bumped(getattr(table, field), 4, factor)})
    assert float(getattr(changed, field)[4, 0]) != float(getattr(table, field)[4, 0])
    before, after = table.geometry_digests(), changed.geometry_digests()
    assert [i for i in range(19) if before[i] != after[i]] == [4]


def test_digests_change_with_the_chart():
    table = Lattice(gamma=energy()).cells
    charts = tuple(("linear",) if c == 2 else ch for c, ch in enumerate(table.chart))
    before, after = (
        table.geometry_digests(),
        _with(table, chart=charts).geometry_digests(),
    )
    assert [i for i in range(19) if before[i] != after[i]] == [2]


def test_digests_change_with_the_excluded_set_and_the_names():
    inside = _digests(
        Lattice(
            gamma=energy(), depth=Cells.linear(6, 10, 3), order=2, depth_in_total=True
        )
    )
    outside = _digests(Lattice(gamma=energy(), depth=Cells.linear(6, 10, 3), order=2))
    assert not set(inside) & set(outside)
    table = Lattice(gamma=energy()).cells
    renamed = _with(table, variables=("lorentz",))
    assert not set(renamed.geometry_digests()) & set(table.geometry_digests())


def test_digests_are_stable_under_select():
    lattice = Lattice(gamma=energy(), B=Cells.linear("2 uG", "8 uG", 3))
    parent = _digests(lattice)
    sub = lattice.select(gamma=slice(3, None))
    assert _digests(sub) == parent[9:]


def test_digests_are_stable_under_refine_for_unchanged_cells():
    parent = Lattice(
        gamma=Cells.log(300, 3e5, 4), B=Cells.linear("2 uG", "8 uG", 3), order=2
    )
    fine = parent.refine(where={"gamma": 0, "B": 0}, gamma=2, B=2)
    before, after = _digests(parent), _digests(fine)
    assert after[4:] == before[1:]
    assert not set(after[:4]) & set(before)


def test_digests_refuse_traced_cell_arrays():
    import jax

    table = Lattice(gamma=energy()).cells

    def digest_inside(lo):
        _with(table, lo=lo, tensor=None).geometry_digests()
        return lo

    with pytest.raises(ValueError, match="concrete"):
        jax.jit(digest_inside)(np.asarray(table.lo))
