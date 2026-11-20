"""Boundary tests of the stack provider (design/0.8/SPEC.md Sec. 8.3, 8.8, step 5a of Sec. 10.2): reweighting and S0.

Each dispatch condition of ``_stack_series`` is tested on both sides: where the stack applies it is compared with
the jet (``provider_jet``) at the same cell, at extreme parameter values, and where it refuses the test asserts
``StackRefused`` and checks that the jet computes the cell. Tolerances come from probes on 2026-11-18 (machine load
average 16 to 24, which changes timings, not values). The measured value is in a comment next to each tolerance.
A sign error or a dropped term changes the compared columns by 1e-4 or more in every case measured (frozen weights
without reweighting, a flipped affine ``eps``, a dropped log coefficient).

This file holds the frozen-node reweighting at order 4, the S0 coverage and sharing, and the helpers that the
other stack files import. The refusals are in ``test_v080_stack_refusals.py``. The comparisons of the affine
chart, orders 0, 5 and 6, eta given B, field cells and strong fields are in ``test_v080_stack_against_jet.py``,
and the findings of the step-5a re-review in ``test_v080_stack_review.py``.

Memory (decision 170): until step 5b-3 one file held all four parts. Its fast run peaked at 9.10 GB of resident
memory with the jet of step 5b-2 and at 8.18 GB with the jet of step 5b-3, since the resident memory grew by 0.1
to 0.3 GB with each jet comparison and was not returned. After the split the fast run of this file alone peaked
at 6.47 and 6.71 GB in two runs (1-minute load 7.3 to 11.7 and 5.4 to 5.6) and the slow run at 1.88 GB (load
6.1 to 6.5). All loaded.
"""

import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.engine._stack_series import StackRefused, stack_cell, stack_lattice
from syncem.engine.average import AveragedKernel, uniform_rule
from syncem.engine.field_average import FieldAverage
from syncem.engine.primitive import PrimitiveTable
from syncem.engine.provider_jet import jet_cell, jet_lattice
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from tests.v080.test_v080_stack import (
    channel_max,
    lognormal_given_gamma,
    rel,
    screen,
    small_channels,
)

K = ContinuumKernel(x_min=1e-12)


def eta4():
    return ("eta", uniform_rule(0.0, 0.5, 4))


def gamma_lattice(n_cells, order, lo=300, hi=3e6, chart="log"):
    cells = (Cells.log if chart == "log" else Cells.linear)(
        lo, hi, n_cells, order=order
    )
    return Lattice(gamma=cells, order=order)


def scaled_error(stack, jet, scale):
    return float(np.max(np.abs(np.asarray(stack) - np.asarray(jet)) / scale))


def finite(a):
    return bool(np.all(np.isfinite(np.asarray(a))))


# ------------------------------------------------------------------ 1. frozen-node reweighting at N = 4


def _lognormal_avg(width, slope, n=16, k=K):
    return AveragedKernel(
        k, [("B", lognormal_given_gamma(width, slope, n)), eta4()], phase=screen()
    )


def _lattice_scale(avg, ch, n_cells):
    """Each channel's largest value over the lattice.

    Up to 19 cells: over every row of the stack lattice, as in ``test_b_given_celled_gamma_by_frozen_reweighting``.
    At 200 cells: over the order-0 rows of the jet at the 200 references (2 s instead of 35 s). That is a lower bound
    of the lattice maximum, so the test is stricter there.
    """
    if n_cells <= 19:
        li, lp = stack_lattice(avg, ch, gamma_lattice(n_cells, 4))
    else:
        li, lp = jet_lattice(avg, ch, gamma_lattice(n_cells, 0))
    return channel_max(li), channel_max(lp)


_REWEIGHT_CASES = [
    (w, s, n)
    for w in (0.01, 0.4)
    for s in (0.02, 0.1, 0.5)
    for n in (1, 19, 200)
    if not (
        w == 0.01 and s >= 0.1 and n == 1
    )  # the two cases of the amplification test below
]


@pytest.mark.parametrize(("width", "slope", "n_cells"), _REWEIGHT_CASES)
def test_reweighting_equals_the_jet_at_order_4(width, slope, n_cells):
    """``ln B ~ N(ln 5 uG + slope ln(gamma / 1e4), width^2)`` on 16 Gauss-Hermite nodes, order 4, cells first/mid/last."""
    ch, avg = small_channels(), _lognormal_avg(width, slope)
    lattice = gamma_lattice(n_cells, 4)
    scale_i, scale_p = _lattice_scale(avg, ch, n_cells)
    memo: dict = {}
    for c in sorted({0, n_cells // 2, n_cells - 1}):
        si, sp = stack_cell(avg, ch, lattice, c, memo=memo)
        ji, jp = jet_cell(avg, ch, lattice, c)
        err_i, err_p = scaled_error(si, ji, scale_i), scaled_error(sp, jp, scale_p)
        # probe: largest 1.5e-10 (I and P, width 0.01, slope 0.5, 19 cells, cell 9), 1.7e-11 on one cell at
        # width 0.01 and slope 0.02, below 1e-10 elsewhere
        assert err_i < 1e-9 and err_p < 1e-9, (width, slope, n_cells, c, err_i, err_p)
    mid = n_cells // 2
    naive = stack_cell(avg, ch, lattice, mid, reweight=False, memo=memo)[0]
    jet = jet_cell(avg, ch, lattice, mid)[0]
    # frozen weights without reweighting: 5.1e-4 (200 cells, width 0.4, slope 0.02) to 0.93 (one cell, slope 0.5)
    assert scaled_error(naive, jet, scale_i) > 1e-4


@pytest.mark.parametrize("slope", [0.1, 0.5])
def test_reweighting_of_a_narrow_law_across_one_wide_cell_refuses(slope):
    """One cell over ``[300, 3e6]`` (h = 4.6), width 0.01: the weights are amplified beyond the bound.

    Probe (2026-11-18, load 16 to 18): amplification ``max_k sum_q |w_q^(k)|`` 928, 5.8e5 and 3.6e8 at slopes 0.02,
    0.1 and 0.5, with stack-jet errors 1.7e-11, 8.5e-9 and 9.8e-6 of the lattice scale. The error is 1.5e-14 to
    5.4e-14 of the amplification wherever cancellation dominates, and the bound 1e4 keeps it below about 5e-10. The
    largest amplification of the accepted cases above is 2.8e3 (width 0.01, slope 0.5, 19 cells).
    """
    ch, avg = small_channels(), _lognormal_avg(0.01, slope)
    lattice = gamma_lattice(1, 4)
    with pytest.raises(StackRefused, match="amplifies the weights"):
        stack_cell(avg, ch, lattice, 0)
    assert finite(jet_cell(avg, ch, lattice, 0)[0])


def test_reweighting_of_a_narrow_law_across_one_wide_cell_at_a_small_slope_equals_the_jet():
    ch, avg = small_channels(), _lognormal_avg(0.01, 0.02)
    lattice = gamma_lattice(1, 4)
    scale_i, scale_p = _lattice_scale(avg, ch, 1)
    si, sp = stack_cell(avg, ch, lattice, 0)
    ji, jp = jet_cell(avg, ch, lattice, 0)
    # probe: 1.7e-11 (amplification 928)
    assert scaled_error(si, ji, scale_i) < 1e-9 and scaled_error(sp, jp, scale_p) < 1e-9


@pytest.mark.parametrize("slope", [0.02, 0.5])
def test_reweighting_refuses_an_unresolved_wide_law(slope):
    """Width 1.5 on 16 nodes: the reweighted moments of ``t^2`` or ``t^3`` (``t`` linear in ``B``) differ from the
    moving-node moments (probe: order 1, power 3, 0.491339 against 0.491528 at slope 0.02), since ``E[B^3]`` of a
    log-normal of width 1.5 grows as ``exp(10.1)`` and 16 nodes do not resolve it. Every cell refuses.
    """
    ch, avg = small_channels(), _lognormal_avg(1.5, slope)
    for n_cells in (1, 19, 200):
        lattice = gamma_lattice(n_cells, 4)
        for c in sorted({0, n_cells // 2, n_cells - 1}):
            with pytest.raises(
                StackRefused, match="does not reproduce the moving nodes"
            ):
                stack_cell(avg, ch, lattice, c)
    assert finite(jet_cell(avg, ch, gamma_lattice(19, 4), 9)[0])


@pytest.mark.slow
def test_reweighting_of_a_wide_law_on_32_nodes_matches_the_jet_to_its_quadrature_error():
    """Width 1.5, slope 0.1, 32 nodes: the check accepts, and the stack differs from the jet by the law quadrature.

    Probe (mid cell of 3, order 4, against the cell's channel maxima): stack - jet 2.2e-4. Against the 64-node jet
    the 32-node jet is off by 8.8e-4 and the 32-node stack by 6.6e-4, so the difference is the law quadrature,
    which both methods share, not the reweighting. ``x_min = 1e-30`` admits the 64-node field nodes up to 1e5 G.
    """
    k = ContinuumKernel(x_min=1e-30)
    ch, lattice = small_channels(), gamma_lattice(3, 4)
    si = stack_cell(_lognormal_avg(1.5, 0.1, 32, k), ch, lattice, 1)[0]
    j32 = jet_cell(_lognormal_avg(1.5, 0.1, 32, k), ch, lattice, 1)[0]
    j64 = jet_cell(_lognormal_avg(1.5, 0.1, 64, k), ch, lattice, 1)[0]
    assert rel(si, j32) < 5e-4  # measured 2.2e-4
    assert rel(si, j64) < rel(j32, j64)  # measured 6.6e-4 < 8.8e-4


# ------------------------------------------------------------------ 9. S0 coverage and sharing


def test_lookups_refuse_a_short_primitive_table():
    """For one node at 5 uG, ``needed_lnx`` is about ``(-22.35, 12.21)``."""
    lo, hi = FieldAverage.needed_lnx([5.0])
    for table in (
        PrimitiveTable.build(lo + 1.0, hi + 1.0),
        PrimitiveTable.build(lo - 1.0, hi - 1.0),
        PrimitiveTable.build(lo - 1.0, hi + 1.0, order=2),
    ):
        with pytest.raises(ValueError, match="does not cover the lookups"):
            FieldAverage.lookups([5.0], [1.0], primitive=table)
    table = PrimitiveTable.build(-1.0, 1.0, order=2)
    for lnx in (-1.5, 1.5):
        with pytest.raises(ValueError, match="leaves the primitive table"):
            table.lookup(np.array([lnx]), "F", 2)


def _count_builds(monkeypatch):
    calls = []
    build = PrimitiveTable.build.__func__

    def counted(cls, *args, **kwargs):
        calls.append(args)
        return build(cls, *args, **kwargs)

    monkeypatch.setattr(PrimitiveTable, "build", classmethod(counted))
    return calls


def test_one_s0_serves_every_s1_table_of_a_lattice(monkeypatch):
    """Per-node mode builds one S1 table per field node and cell (16 nodes on 3 cells), all from one S0."""
    calls = _count_builds(monkeypatch)
    avg = _lognormal_avg(0.4, 0.1)
    stack_lattice(avg, small_channels(), gamma_lattice(3, 2))
    assert len(calls) == 1


def test_each_law_reads_the_s0_of_its_quantised_span(monkeypatch):
    """Decision 158: a 1 G field needs ``ln x`` down to about -34.6, so it reads the S0 of span (-40, 20).

    The 4 to 6 uG laws read the S0 of the default span (-30, 20), built once, before and after the 1 G law.
    """
    calls = _count_builds(monkeypatch)
    ch, lattice, memo = small_channels(), gamma_lattice(3, 2), {}
    weak = AveragedKernel(
        K, [("B", uniform_rule(4e-6, 6e-6, 4)), eta4()], phase=screen()
    )
    strong = AveragedKernel(K, [eta4()], fixed={"B": 1.0}, phase=screen())
    other = AveragedKernel(
        K, [("B", uniform_rule(3e-6, 7e-6, 4)), eta4()], phase=screen()
    )
    stack_cell(weak, ch, lattice, 1, memo=memo)
    stack_cell(strong, ch, lattice, 1, memo=memo)
    stack_cell(other, ch, lattice, 2, memo=memo)
    assert calls == [(-30.0, 20.0), (-40.0, 20.0)]
    assert FieldAverage.needed_lnx([1e6])[0] > -40.0


def test_s0_builds_over_a_strong_field_lattice(monkeypatch):
    """Six ``B`` cells from 2 mG to 2 G: at most two S0 builds (decision 158), not one per cell.

    Their lower ``ln x`` ends run from -28.9 to -34.7, which round outward to -30 and -40. Probe of the rule before
    decision 144: 6 builds (lnx0 from -28.91 to -34.67), 3.4 s at load 17.
    """
    calls = _count_builds(monkeypatch)
    avg = AveragedKernel(K, [("eta", uniform_rule(0.0, 0.5, 6))], phase=screen())
    lattice = Lattice(
        gamma=Cells.log(30, 300, 1), B=Cells.log("2 mG", "2 G", 6), order=2
    )
    ch = Channels.log(0.1, 10, 5, unit="GHz", half_width=0.2, n_nu=32)
    stack_lattice(avg, ch, lattice)
    assert len(calls) <= 2 and len(set(calls)) == len(calls)
    assert set(calls) <= {(-30.0, 20.0), (-40.0, 20.0)}


@pytest.mark.parametrize(
    "nodes, spans",
    [
        (np.geomspace(2e3, 1e7, 8), [(-30.0, 20.0), (-40.0, 20.0)]),
        (np.geomspace(0.1, 1e-3, 8), [(-30.0, 20.0), (-30.0, 30.0)]),
    ],
)
def test_s0_builds_over_monotone_field_nodes(monkeypatch, nodes, spans):
    """One-node tables in ascending strong or descending weak fields: one S0 per span (8 builds each before 144)."""
    from syncem.engine._stack_series import _s1_table

    calls, memo = _count_builds(monkeypatch), {}
    for b in nodes:
        _s1_table(memo, "lookup", np.array([b]), np.ones(1))
    assert calls == spans
