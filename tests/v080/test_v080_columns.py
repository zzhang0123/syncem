"""Step 5b-1 of design/0.8/SPEC.md: the column map (Sec. 3.4, 4.3, 4.7, decisions 146 to 148).

Every map is compared field by field with the brute-force map of ``tests/v080/_column_cases.py``, built from
``index.row(k)`` one ``Row`` object at a time, so the vectorised passes of ``syncem.engine.columns`` are
checked against an independent formulation. Column counts are the static counts of Sec. 4.7 with hand-passed
``ColumnRules``. The step-7 acceptance of the counts 285 and 1306 from plan-computed rules stays in step 7
(decision 148). The key, refusal and module-contract tests are in ``test_v080_columns_keys.py``.
"""

from __future__ import annotations

import os
import time

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.engine.average import AveragedKernel, uniform_rule
from syncem.engine.columns import REASONS, ColumnMap, ColumnRules, column_map
from syncem.engine.provider_jet import jet_lattice
from syncem.lattice.cells import Cells
from syncem.lattice.index import MomentIndex
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import AzimuthHarmonic, Legendre
from syncem.values.errors import SizeError
from tests.v080._column_cases import (
    CANON,
    EPS,
    LATTICES,
    N_ETA,
    N_MU,
    assert_matches_brute,
    benchmark_kernel,
    cosine_kernel,
    merge_case,
)

# ------------------------------------------------------------------ 1. static counts of Sec. 4.7

COUNTS = [
    # name, kernel, rules, n_rows, n, n0, n2, |uses I|, |uses P|, |uses V|
    ("E1", "continuum", ColumnRules(), 57, 57, 57, 57, 57, 57, 0),
    ("E5", "continuum", ColumnRules(), 171, 171, 57, 57, 57, 114, 0),
    ("E6", "continuum", ColumnRules(), 114, 114, 114, 114, 114, 114, 0),
    (
        "energy-depth",
        "continuum",
        ColumnRules(depth_free_spin0=True),
        513,
        513,
        57,
        513,
        171,
        513,
        0,
    ),
    (
        "E4-KP",
        "continuum",
        ColumnRules(masks=(("mu", "single"),)),
        513,
        285,
        285,
        285,
        285,
        285,
        0,
    ),
    (
        "benchmark",
        "benchmark",
        ColumnRules(
            stokes=("I", "P", "V"),
            masks=(("mu", "joint"), ("eta", "joint")),
            depth_free_spin0=True,
        ),
        2430,
        1306,
        486,
        410,
        246,
        820,
        240,
    ),
]


def _kernel(name):
    return ContinuumKernel() if name == "continuum" else benchmark_kernel()


@pytest.mark.parametrize("case", COUNTS, ids=[c[0] for c in COUNTS])
def test_static_column_counts_of_sec_4_7(case):
    """Counts of SPEC Sec. 4.4 and 4.7 (exact), and the brute-force map on the same lattice (exact)."""
    name, kname, rules, n_rows, n, n0, n2, n_i, n_p, n_v = case
    lat, kern = LATTICES[name](), _kernel(kname)
    cm = column_map(lat, kern, rules)
    assert isinstance(cm, ColumnMap)
    assert (cm.n_rows, cm.n, cm.n0, cm.n2) == (n_rows, n, n0, n2)
    assert tuple(len(cm.uses[s]) for s in CANON) == (n_i, n_p, n_v)
    assert_matches_brute(cm, lat, kern, rules)


def test_energy_and_depth_cells_share_one_spin0_spectrum():
    """Sec. 4.3: with I and V depth-free, the s = 0 columns of the three depth cells repeat one spectrum."""
    lat = LATTICES["energy-depth"]()
    cm = column_map(lat, ContinuumKernel(), ColumnRules(depth_free_spin0=True))
    seen = set()
    for b in (0, 9, 18):
        for r in range(3):
            cols = [
                cm.col_of_row[lat.index.position(gamma=b, depth=d, powers={"gamma": r})]
                for d in range(3)
            ]
            shared = {int(cm.spin0_from[c]) for c in cols}
            assert len(shared) == 1 and shared.isdisjoint(seen)
            seen |= shared
            s1 = lat.index.position(gamma=b, depth=1, powers={"gamma": r, "depth": 1})
            assert cm.spin0_from[cm.col_of_row[s1]] == -1
            assert cm.spin2_from[cm.col_of_row[s1]] >= 0


def test_benchmark_blocks():
    """Sec. 4.7: 6 depth-free monomials x 81 pairs in M0, 10 monomials x 41 even pairs per spin-2 block."""
    lat = LATTICES["benchmark"]()
    rules = COUNTS[-1][2]
    cm = column_map(lat, benchmark_kernel(), rules)
    blocks = np.bincount(cm.column_block, minlength=3)
    assert tuple(blocks) == (486, 410, 410)
    assert dict(cm.reasons).keys() == {"depth", "parity"}
    assert len(dict(cm.reasons)["depth"]) == 4 * 81


# ------------------------------------------------------------------ 2. the default rules are the identity


@pytest.mark.parametrize("name", sorted(LATTICES))
def test_default_rules_are_the_identity(name):
    """``ColumnRules()`` drops and merges nothing when every block of the lattice is requested (decision 148)."""
    lat, kern = LATTICES[name](), ContinuumKernel()
    cm = column_map(lat, kern, ColumnRules(), max_columns=10**6)
    np.testing.assert_array_equal(cm.col_of_row, np.arange(lat.n))
    np.testing.assert_array_equal(cm.col_rows, np.arange(lat.n))
    assert cm.n == lat.n and cm.masked.size == 0 and cm.reasons == ()
    assert_matches_brute(cm, lat, kern, ColumnRules())


def test_unrequested_blocks_are_unused():
    lat = LATTICES["E5"]()
    rules = ColumnRules(stokes=("I",))
    cm = column_map(lat, ContinuumKernel(), rules)
    assert cm.n == 57 and cm.n2 == 0 and cm.uses["P"].size == 0
    np.testing.assert_array_equal(dict(cm.reasons)["unused"], np.arange(57, 171))
    assert_matches_brute(cm, lat, ContinuumKernel(), rules)


# ------------------------------------------------------------------ 3. algebra


ALGEBRA = {
    "merge": merge_case,
    "benchmark": lambda: (LATTICES["benchmark"](), benchmark_kernel(), COUNTS[-1][2]),
    "E4-KP": lambda: (
        LATTICES["E4-KP"](),
        ContinuumKernel(),
        ColumnRules(masks=(("mu", "single"),), symmetries=("pitch_reversal",)),
    ),
}


def test_merge_case_counts_and_labels():
    """Beaming merges (l, k) with (k, l) (Sec. 4.3): 4 of the 5 even pairs per cell and monomial remain."""
    lat, kern, rules = merge_case()
    cm = column_map(lat, kern, rules)
    assert (cm.n_rows, cm.n) == (54, 24)
    assert len(dict(cm.reasons)["beaming"]) == 6
    assert len(dict(cm.reasons)["parity"]) == 24
    labels = cm.labels(lat.index)
    assert len(labels) == cm.n
    merged = [j for j in range(cm.n) if len(cm.rows[j]) == 2]
    assert len(merged) == 6
    for j in range(cm.n):
        assert labels[j] == " + ".join(lat.index.label(int(k)) for k in cm.rows[j])
    assert labels[merged[0]] == (
        "M0[gamma=0 | u^0 P_0(mu) P_2(eta)] + M0[gamma=0 | u^0 P_2(mu) P_0(eta)]"
    )
    assert_matches_brute(cm, lat, kern, rules)


@pytest.mark.parametrize("name", sorted(ALGEBRA))
def test_apply_equals_the_matrix_bitwise(name):
    """``apply(v) == matrix() @ v``: both sum at most two rows per column in row order (exact)."""
    lat, kern, rules = ALGEBRA[name]()
    cm = column_map(lat, kern, rules)
    v = np.random.default_rng(1).normal(size=(3, cm.n_rows))
    got = cm.apply(v)
    assert got.shape == (3, cm.n)
    np.testing.assert_array_equal(got, (cm.matrix() @ v.T).T)
    np.testing.assert_array_equal(np.asarray(cm.apply(jnp.asarray(v))), got)
    np.testing.assert_array_equal(cm.apply(v[0]), got[0])


@pytest.mark.parametrize("name", sorted(ALGEBRA))
def test_lift_is_d_times_s(name):
    """``lift(D) == D @ S`` bitwise: each row has at most one column, so each entry is one product with 1.

    ``lift(D) @ v`` against ``D @ apply(v)`` differ by re-association only. The tolerance is the standard
    summation bound ``2 (n_rows + 2) eps`` times the sum of the absolute terms ``|D| S |v|`` (exact
    derivation, no probe).
    """
    lat, kern, rules = ALGEBRA[name]()
    cm = column_map(lat, kern, rules)
    rng = np.random.default_rng(2)
    D = rng.normal(size=(5, cm.n))
    v = rng.normal(size=cm.n_rows)
    S = cm.matrix().toarray()
    lifted = cm.lift(D)
    assert lifted.shape == (5, cm.n_rows)
    np.testing.assert_array_equal(lifted, D @ S)
    np.testing.assert_array_equal(lifted[:, cm.masked], 0.0)
    np.testing.assert_array_equal(np.asarray(cm.lift(jnp.asarray(D))), lifted)
    bound = 2 * (cm.n_rows + 2) * EPS * (np.abs(D) @ S @ np.abs(v))
    assert np.all(np.abs(lifted @ v - D @ cm.apply(v)) <= bound)


@pytest.mark.parametrize("name", sorted(ALGEBRA))
def test_masked_rows_and_reason_groups(name):
    lat, kern, rules = ALGEBRA[name]()
    cm = column_map(lat, kern, rules)
    np.testing.assert_array_equal(cm.masked, np.flatnonzero(cm.col_of_row == -1))
    groups = dict(cm.reasons)
    dropped = [g for r, g in cm.reasons if r != "beaming"]
    every = np.concatenate(dropped) if dropped else np.zeros(0, int)
    np.testing.assert_array_equal(np.sort(every), cm.masked)
    assert np.unique(every).size == every.size
    for k in groups.get("beaming", ()):
        assert cm.col_of_row[k] >= 0 and k not in cm.col_rows[cm.col_start[:-1]]
    assert set(groups) <= set(REASONS)
    assert len(cm.rows) == cm.n
    assert all(cm.col_of_row[k] == j for j in range(cm.n) for k in cm.rows[j])
    assert_matches_brute(cm, lat, kern, rules)


def test_pitch_reversal_drops_odd_pitch_degrees():
    lat, kern, rules = ALGEBRA["E4-KP"]()
    plain = column_map(lat, kern, ColumnRules(symmetries=("pitch_reversal",)))
    assert plain.n == 285 and set(dict(plain.reasons)) == {"symmetry"}
    masked = column_map(lat, kern, rules)
    assert masked.n == 285 and set(dict(masked.reasons)) == {"parity"}


def test_map_arrays_are_read_only():
    cm = column_map(LATTICES["E5"](), ContinuumKernel())
    for name in ("col_of_row", "col_rows", "spin0_from", "spin2_base"):
        with pytest.raises(ValueError):
            getattr(cm, name)[0] = 7
    with pytest.raises(ValueError):
        cm.uses["I"][0] = 7


# ------------------------------------------------------------------ 4. G15 without per-row objects


def g15():
    return Lattice(
        gamma=Cells.log(300, 1e7, 22),
        B=Cells.log("1 uG", "30 uG", 4),
        eta=Legendre(8),
        phi=AzimuthHarmonic(),
        depth=Cells.linear(-60, 60, 48, order=3),
        order=2,
    )


G15_RULES = ColumnRules(masks=(("eta", "single"),), depth_free_spin0=True)


def test_g15_column_map_without_row_objects(monkeypatch):
    """G15 ``lat8`` (2 737 152 rows) maps through ``MomentIndex.arrays`` with ``index.row`` disabled.

    Time bound 3.0 s. RED probe (``probe_g15_ops.py`` in the session scratchpad): the dominant vectorised
    operations of the passes (row arrays, use masks, one stable argsort, two ``np.unique``) took 0.127 to
    0.135 s over 3 runs at a 1-minute load of 11.4. The bound is 22 times the slowest probe run.
    Counts: 9 eta degrees keep 5 even ones, the 24 monomials (gamma, B) x depth^0..3 keep 6 with s = 0 in
    M0, and the 48 depth cells share one spin-0 spectrum per (gamma, B) cell.
    """
    lat = g15()
    _ = lat.index.arrays

    def refuse(self, k):
        raise AssertionError("index.row called")

    monkeypatch.setattr(MomentIndex, "row", refuse)
    start = time.perf_counter()
    cm = column_map(lat, ContinuumKernel(), G15_RULES, max_columns=2_000_000)
    elapsed = time.perf_counter() - start
    assert elapsed < 3.0, (elapsed, os.getloadavg())
    m0, m2 = 4224 * 5 * 6, 4224 * 5 * 24
    assert (cm.n_rows, cm.n, cm.n0, cm.n2) == (2_737_152, m0 + 2 * m2, 88 * 5 * 6, m2)
    with pytest.raises(SizeError, match="max_columns"):
        column_map(lat, ContinuumKernel(), G15_RULES)


# ------------------------------------------------------------------ 5. size boundary, no provider


def test_size_boundary_at_n_and_n_minus_one(monkeypatch):
    calls = []
    import syncem.engine.provider_jet as pj

    for name in ("jet_lattice", "jet_cell", "jet_columns"):
        monkeypatch.setattr(pj, name, lambda *a, **k: calls.append(name))
    lat, kern, rules = LATTICES["E4-KP"](), ContinuumKernel(), COUNTS[4][2]
    assert column_map(lat, kern, rules, max_columns=285).n == 285
    with pytest.raises(SizeError, match="285"):
        column_map(lat, kern, rules, max_columns=284)
    assert calls == []

    # ------------------------------------------------------------------ 6. the mask dispatcher through the jet


def _mask_columns(L, rule):
    lat = Lattice(a=Cells.log(1.0, 10.0, 2), mu=Legendre(L))
    avg = AveragedKernel(cosine_kernel(), [("eta", rule)], legendre={"mu": (L, N_MU)})
    ch = Channels.log(0.1, 10, 4, unit="GHz", half_width=0.2, n_nu=16)
    i, p = (np.asarray(x) for x in jet_lattice(avg, ch, lat))
    return lat, i, p


@pytest.mark.parametrize("L", [0, 8])
def test_mask_dispatcher_rules_on_against_off(L):
    """The odd-l mask of one resolved cosine is exact for an even partner law and loses columns otherwise.

    Even law (Gauss-Legendre on [-1, 1]): the odd-l columns with rules off are zero in exact arithmetic,
    since the nodes are symmetric. The bound is derived from the rounding of the two quadratures:
    ``(N_MU + N_ETA) (2 L + 1) / 2 eps = 4.5e-14`` of the channel's column maximum. Measured in the RED probe
    (``probe_mask.py``): 2.1e-16 for P and 1.4e-16 for I at a 1-minute load of 13.2.
    Failing side, ``eta ~ U[0, 1/2]``: the mean 1/4 gives exact Legendre coefficients. The l = 1 column of I
    is ``0.125 / 1.1`` and that of P is ``0.05`` of the l = 0 column, an exact identity of the polynomial
    kernel, which both Gauss-Legendre rules integrate without quadrature error. Tolerance 1e-12 (measured 8e-16 in the probe).
    """
    lat, kern = Lattice(a=Cells.log(1.0, 10.0, 2), mu=Legendre(L)), cosine_kernel()
    off = column_map(lat, kern, ColumnRules())
    on = column_map(lat, kern, ColumnRules(masks=(("mu", "single"),)))
    l = lat.index.arrays.degrees[:, 0]
    np.testing.assert_array_equal(off.col_of_row, np.arange(lat.n))
    np.testing.assert_array_equal(on.masked, np.flatnonzero(l % 2 == 1))
    assert on.n == lat.n - on.masked.size
    _, i, p = _mask_columns(L, uniform_rule(-1.0, 1.0, N_ETA))
    bound = (N_MU + N_ETA) * (2 * L + 1) / 2 * EPS
    for x in (i, p):
        scale = np.abs(x).max(axis=0)
        assert np.all(np.abs(x[on.masked]) <= bound * scale)
    if L == 0:
        assert on.masked.size == 0
        return
    _, i, p = _mask_columns(L, uniform_rule(0.0, 0.5, N_ETA))
    for c in (0, 1):
        k0 = lat.index.position(cell=c, degrees={"mu": 0})
        k1 = lat.index.position(cell=c, degrees={"mu": 1})
        assert k1 in on.masked
        np.testing.assert_allclose(i[k1] / i[k0], 0.125 / 1.1, rtol=1e-12)
        np.testing.assert_allclose(np.real(p[k1] / p[k0]), 0.05, rtol=1e-12)
