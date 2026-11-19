"""Route dispatcher of step 5b-1 (design/0.8/SPEC.md decision 150, Sec. 8.3, 8.8, T2, T5, T16).

Each route is tested on both sides of its condition: where ``routes="auto"`` takes the stack the values equal the
stack, where it takes the jet the reason is pinned, and at the grid-reach switch cell the stack and the jet are
evaluated directly and compared. ``check_contract`` asserts the one-way contract of ``stack_precondition`` on every
auto case. Probes ran on 2026-11-19 at a 1-minute load of 7.6 to 14.3 (load changes timings, not values). Each
tolerance names its probe value and margin, a SPEC number or an exact identity.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.continuum import ContinuumKernel
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine import provider_jet
from syncem.engine._routes import check_stokes, route_cells, stack_precondition
from syncem.engine._stack_series import StackRefused, stack_lattice
from syncem.engine.average import AveragedKernel, NodeRule, uniform_rule
from syncem.engine.provider_jet import jet_cell, jet_lattice
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.records.records import RouteRecord
from tests.v080.test_v080_jet import _one
from tests.v080.test_v080_stack import lognormal_given_gamma, screen, small_channels
from tests.v080.test_v080_stack_boundaries import _lognormal_avg, gamma_lattice

K = ContinuumKernel(x_min=1e-12)
GAMMA_AVERAGED = "gamma averaged: Similarity variable not celled -> jet"
REACH = "below the field-average table"
ENDS = [0, -1]  # the lowest and highest channels


def eta4():
    return ("eta", uniform_rule(0.0, 0.5, 4))


def cached_avg():
    levels = [("B", uniform_rule(4e-6, 6e-6, 4)), eta4()]
    return AveragedKernel(K, levels, phase=screen())


def fixed_b_avg():
    return AveragedKernel(K, [eta4()], fixed={"B": 5e-6}, phase=screen())


def field_lattice(n_b=2, order=2):
    b = Cells.log("2 uG", "8 uG", n_b, order=order)
    return Lattice(gamma=Cells.log(300, 3e5, 2, order=order), B=b, order=order)


def check_contract(avg, lattice, records):
    """Decision 150: ``None`` wherever auto took the stack, and a reason only where auto took the jet with it."""
    for rec in records:
        reason = stack_precondition(avg, lattice, rec.cell)
        assert rec.provider != "stack" or reason is None, rec
        assert reason is None or (rec.provider, rec.reason) == ("jet", reason), rec


def assert_jet_rows(avg, ch, lat, i, p):
    ji, jp = jet_lattice(avg, ch, lat)
    # exact: the same jet routine on the same cells
    np.testing.assert_array_equal(i, np.asarray(ji))
    np.testing.assert_array_equal(p, np.asarray(jp))


def test_gamma_only_takes_the_stack_in_cached_s1_mode():
    ch, avg, lat = small_channels(), cached_avg(), gamma_lattice(4, 2)
    i, p, v, records = route_cells(avg, ch, lat, "auto")
    assert v is None
    assert records == tuple(
        RouteRecord(c, "stack", "cached-S1", "lookup") for c in (0, 1, 2, 3)
    )
    si, sp = stack_lattice(avg, ch, lat)
    # exact: the same stack calls with one memo
    np.testing.assert_array_equal(i, si)
    np.testing.assert_array_equal(p, sp)
    check_contract(avg, lat, records)


@pytest.mark.parametrize("with_b", [False, True])
def test_celled_eta_takes_the_jet_with_the_reason(with_b):
    g, eta = Cells.log(300, 3e5, 2, order=1), Cells.linear(0.0, 0.5, 2, order=1)
    b = {"B": Cells.log("2 uG", "8 uG", 1, order=1)} if with_b else {}
    levels = [] if with_b else [("B", uniform_rule(4e-6, 6e-6, 4))]
    lat = Lattice(gamma=g, **b, eta=eta, order=1)
    avg, ch = AveragedKernel(K, levels, phase=screen()), small_channels()
    i, p, _, records = route_cells(avg, ch, lat, "auto")
    names = lat.cells.variables
    text = f"celled variables {names}: the stack takes gamma and optionally B. The jet takes the cell"
    assert all(r.provider == "jet" and r.reason == text for r in records)
    assert_jet_rows(avg, ch, lat, i, p)
    check_contract(avg, lat, records)


def power_law_gamma(slope=2.6, lo=300.0, hi=3e6, n=16, cutoff=False):
    """``gamma ~ gamma^-slope`` on ``[lo, hi]``, Gauss-Legendre in ``ln gamma``. ``cutoff`` moves ``hi`` as ``B^-1/2``."""
    t, w = np.polynomial.legendre.leggauss(n)

    def fn(point):
        top = jnp.log(hi) - (0.5 * jnp.log(point["B"] / 5e-6) if cutoff else 0.0)
        g = jnp.exp(jnp.log(lo) + 0.5 * (top - jnp.log(lo)) * (t + 1))
        q = w * g ** (1 - slope)
        return g, q / jnp.sum(q)

    return NodeRule(fn, parents=("B",) if cutoff else ())


def five_point(avg, ch, ref, h, s):
    """Columns 1 and 2 in the chart ``z = ln(B / ref) / h`` from five-point differences of step ``s`` in ln B."""
    vals = [avg(ch, {"B": ref * np.exp(k * s)}) for k in (-2, -1, 0, 1, 2)]
    out = []
    for get in (lambda m: np.asarray(m.I), lambda m: np.asarray(m.P[0])):
        f = [get(m) for m in vals]
        d1 = (f[0] - 8 * f[1] + 8 * f[3] - f[4]) / (12 * s)
        d2 = (-f[0] + 16 * f[1] - 30 * f[2] + 16 * f[3] - f[4]) / (12 * s * s)
        out.append((h * d1, h * h * d2 / 2))
    return out


@pytest.mark.parametrize("cutoff", [False, True])
def test_field_cells_alone_take_the_jet_and_match_the_averaged_kernel(cutoff):
    """T5 (SPEC Sec. 9.5): 4 log cells on [0.5, 50] uG, ``gamma`` averaged, orders 0 to 2.

    Probe (load 14.3): the order-0 column equals the averaged kernel at the reference to 1.9e-15 per channel (bound
    1e-13, margin 50). The five-point differences at step 2e-3 in ln B agree with orders 1 and 2 to at most 2.6e-10
    pointwise at the lowest and highest channels (bound 2e-9, margin 8). Their error falls as ``s^4`` from
    ``s = 2e-2`` to ``2e-3``, so it is the truncation and rounding of the differences. A dropped factor ``h`` or a
    sign error gives errors of order 1.
    """
    ch, lat = small_channels(), Lattice(B=Cells.log("0.5 uG", "50 uG", 4), order=2)
    levels = [("gamma", power_law_gamma(cutoff=cutoff)), eta4()]
    avg = AveragedKernel(K, levels, phase=screen())
    i, p, _, records = route_cells(avg, ch, lat, "auto")
    assert all(r.provider == "jet" and r.reason == GAMMA_AVERAGED for r in records)
    assert all(stack_precondition(avg, lat, c) == GAMMA_AVERAGED for c in range(4))
    with pytest.raises(StackRefused):
        route_cells(avg, ch, lat, "stack", cells=(0,))
    for c in (0, 3):
        rows = lat.index.cell_rows(c)
        ref, h = float(lat.cells.reference[c, 0]), float(lat.cells.half_width[c, 0])
        f0 = avg(ch, {"B": ref})
        for cols, value in (
            (i[rows], np.asarray(f0.I)),
            (p[rows], np.asarray(f0.P[0])),
        ):
            assert np.max(np.abs(cols[0] - value) / np.abs(value)) < 1e-13
        for cols, diffs in zip((i[rows], p[rows]), five_point(avg, ch, ref, h, 2e-3)):
            for r, fd in zip((1, 2), diffs):
                err = np.abs(cols[r] - fd)[ENDS] / np.abs(cols[r])[ENDS]
                assert np.all(err < 2e-9), (c, r, err)
    check_contract(avg, lat, records)


def switch_lattice(order, n=8, lo=300.0, hi=1e9):
    return Lattice(gamma=Cells.log(lo, hi, n, order=order), order=order)


def test_auto_switches_from_the_stack_to_the_jet_at_the_grid_reach():
    """Channels from 0.08 GHz at 5 uG: references up to 6.0e7 are inside the S1 grid and 3.9e8 is not (probe)."""
    ch, avg, lat = small_channels(), fixed_b_avg(), switch_lattice(1)
    events: dict = {}
    _, _, _, records = route_cells(avg, ch, lat, "auto", events=events)
    assert [r.provider for r in records] == ["stack"] * 7 + ["jet"]
    assert all(r.mode == "per-node" and r.s1 == "lookup" for r in records[:7])
    assert REACH in records[7].reason and records[7].s1 is None
    assert all(stack_precondition(avg, lat, c) is None for c in range(8))
    check_contract(avg, lat, records)
    counts = {"stack cells": 7, "jet cells": 1, "core cells": 0, "jet groups": 1}
    assert events == {**counts, "jet padded cells": 1}


@pytest.mark.parametrize("order", [0, 2])
@pytest.mark.parametrize("s1", ["quadrature", "lookup"])
def test_last_stack_cell_equals_the_jet_within_check_a(order, s1):
    """Check (a) of SPEC Sec. 8.3 at ``r <= 2``: 1e-10 of the channel column maximum, 1e-8 of the cell's own peak.

    The lookup path is held to 1e-7 of the channel maximum (SPEC Sec. 8.2, 9.3, decision 140). Probe (load 13.4):
    4.7e-15 of the channel maximum and 3.8e-13 of the own peak on both paths at orders 0 and 2.
    """
    ch, avg, lat = small_channels(), fixed_b_avg(), switch_lattice(order)
    ji, jp = (np.asarray(a) for a in jet_lattice(avg, ch, lat))
    si, sp, _, rec = route_cells(avg, ch, lat, {6: "stack"}, s1=s1, cells=(6,))
    assert rec == (RouteRecord(6, "stack", "per-node", s1),)
    rows = lat.index.cell_rows(6)
    for s, j in ((si, ji), (sp, jp)):
        d = np.abs(s[rows] - j[rows])[:, ENDS]
        channel_max = np.abs(j).max(axis=0)[ENDS]
        own = np.abs(j[rows]).max(axis=1, keepdims=True)
        assert np.max(d / channel_max) < (1e-10 if s1 == "quadrature" else 1e-7)
        assert s1 == "lookup" or np.max(d / own) < 1e-8
    with pytest.raises(StackRefused, match=REACH):
        route_cells(avg, ch, lat, {7: "stack"}, cells=(7,))
    assert stack_precondition(avg, lat, 7) is None


@pytest.mark.parametrize("order", [0, 6])
def test_grid_reach_switch_at_orders_0_and_6_with_1_and_200_cells(order):
    """Stack only. The switch does not depend on the order: reach is set by the reference and the lowest channel."""
    ch, avg = small_channels(), fixed_b_avg()
    inside = switch_lattice(order, 1, 3e7, 1.2e8)
    assert route_cells(avg, ch, inside, "stack")[3][0].provider == "stack"
    with pytest.raises(StackRefused, match=REACH):
        route_cells(avg, ch, switch_lattice(order, 1, 1e8, 1e9), "stack")
    many = switch_lattice(order, 200)
    k = first_refused(avg, ch, switch_lattice(0, 200))
    assert 0 < k < 200 and stack_precondition(avg, many, k) is None
    route_cells(avg, ch, many, "stack", cells=(k - 1,))
    with pytest.raises(StackRefused, match=REACH):
        route_cells(avg, ch, many, "stack", cells=(k,))


def first_refused(avg, ch, lattice):
    """First refused cell. The accepted cells form a prefix."""
    accepted, memo = [], {}
    for c in range(lattice.n_cells):
        try:
            route_cells(avg, ch, lattice, "stack", cells=(c,), memo=memo)
            accepted.append(c)
        except StackRefused:
            pass
    assert accepted == list(range(len(accepted)))
    return len(accepted)


def missing_density():
    rule = NodeRule(lognormal_given_gamma(0.4, 0.1).fn, parents=("gamma",))
    return AveragedKernel(K, [("B", rule), eta4()], phase=screen())


@pytest.mark.parametrize(
    ("make", "n_cells", "order", "text"),
    [
        (lambda: _lognormal_avg(0.01, 0.5), 1, 4, "amplifies the weights"),
        (missing_density, 2, 2, "without a density"),
    ],
)
def test_evaluation_time_refusals_route_to_the_jet(make, n_cells, order, text):
    ch, avg, lat = small_channels(), make(), gamma_lattice(n_cells, order)
    i, p, _, records = route_cells(avg, ch, lat, "auto")
    assert all(r.provider == "jet" and text in r.reason for r in records)
    assert all(stack_precondition(avg, lat, c) is None for c in range(n_cells))
    assert_jet_rows(avg, ch, lat, i, p)


def test_a_celled_variable_that_is_also_averaged_is_not_routed():
    levels = [("gamma", power_law_gamma()), eta4()]
    avg = AveragedKernel(K, levels, fixed={"B": 5e-6}, phase=screen())
    lat = gamma_lattice(2, 1)
    for routes in ("auto", "jet", ("stack", "stack")):
        with pytest.raises(ValueError, match="celled and also averaged") as err:
            route_cells(avg, small_channels(), lat, routes)
        assert not isinstance(err.value, StackRefused)
    with pytest.raises(ValueError, match="celled and also averaged"):
        stack_precondition(avg, lat, 0)


BAD_ROUTES = ("fast", ("stack",), {0: "stack"}, ("stack", "table"), {5: "jet"})


def test_route_arguments_are_checked():
    ch, avg, lat = small_channels(), cached_avg(), gamma_lattice(2, 1)
    for bad in BAD_ROUTES:
        with pytest.raises(ValueError, match="routes"):
            route_cells(avg, ch, lat, bad)
    with pytest.raises(ValueError, match="cells"):
        route_cells(avg, ch, lat, "auto", cells=(2,))
    with pytest.raises(ValueError, match="V"):
        route_cells(avg, ch, lat, ("jet", "core"), with_v=True)


@kernel(variables=[Variable("a", kind="positive")], stokes=("I",))
def only_i(nu, a):
    return {"I": a * nu}


def test_stokes_requests_outside_the_kernel_raise():
    with pytest.raises(ValueError, match=r"\('I', 'P'\)"):
        check_stokes(K, "IV")
    with pytest.raises(ValueError, match=r"\('I',\)"):
        check_stokes(only_i, ("I", "P"))
    assert check_stokes(K, "IQU") == check_stokes(K, ("P", "I")) == ("I", "P")
    for bad in ("Q", "IU", "IX", ""):
        with pytest.raises(ValueError):
            check_stokes(K, bad)


@pytest.mark.parametrize("d", [1, 2])
def test_explicit_core_equals_the_jet(d):
    """Probe (load 13.4): 7.2e-16 of the channel maximum at d = 1 (bound 1e-14, the planning probe 6.5e-16) and
    5.4e-15 at d = 2 on a 2 x 2 (gamma, B) lattice (bound 5e-14, margin 9)."""
    ch = small_channels()
    if d == 1:
        avg, lat, tol = cached_avg(), gamma_lattice(3, 2), 1e-14
    else:
        avg, lat, tol = (
            AveragedKernel(K, [eta4()], phase=screen()),
            field_lattice(),
            5e-14,
        )
    n = lat.n_cells
    ci, cp, _, rec = route_cells(avg, ch, lat, ("core",) * n)
    ji, jp, _, _ = route_cells(avg, ch, lat, ("jet",) * n)
    assert all(r.provider == "core" and r.reason is None for r in rec)
    for c, j in ((ci, ji), (cp, jp)):
        assert np.max(np.abs(c - j) / np.abs(j).max(axis=0)) < tol


POLY_I = [(1.5, 2.0, 1.0), (-0.7, 0.0, 0.3), (2.2, -1.3, -0.6)]
POLY_V = [(1.5, 2.0, 0.4), (0.5, 1.0, -0.2), (-1.2, 0.5, 0.7)]


def poly(terms, x, y):
    return sum(c * x**p * y**q for p, q, c in terms)


XY = [Variable("x", kind="positive"), Variable("y", kind="positive")]


@kernel(variables=XY, stokes=("I", "P", "V"))
def poly_kernel(nu, x, y):
    ones, i = jnp.ones_like(nu), poly(POLY_I, x, y)
    return {"I": i * ones, "P": 0.3 * i * ones, "V": poly(POLY_V, x, y) * ones}


def closed_form_v(lat):
    """V per row of one block, before the channel response: ``sum_k d_k`` times the per-axis Taylor factors."""
    rows, t = lat.index.arrays, lat.cells
    out = np.empty(lat.index.block_size)
    for k, (c, (r, s)) in enumerate(zip(rows.cell, rows.powers)):
        x, y = ((t.chart[c][a], t.reference[c, a], t.half_width[c, a]) for a in (0, 1))
        terms = [
            d * _one(x[0], px, *x[1:], r) * _one(y[0], py, *y[1:], s)
            for px, py, d in POLY_V
        ]
        out[k] = sum(terms)
    return out


def test_v_rows_of_a_polynomial_user_kernel_equal_the_closed_form():
    """T16 reference and tolerance 1e-12 (``tests/v080/test_v080_jet.py``)."""
    x, y = Cells.log(1.0, 30.0, 3, order=3), Cells.linear(0.5, 1.5, 2, order=2)
    ch, lat, avg = (
        small_channels(3, 8),
        Lattice(x=x, y=y, order=3),
        AveragedKernel(poly_kernel),
    )
    _, _, v, records = route_cells(avg, ch, lat, "auto", with_v=True)
    assert all(r.provider == "jet" and "UserKernel" in r.reason for r in records)
    check_contract(avg, lat, records)
    unit = np.asarray(avg(ch, {"x": 1.0, "y": 1.0}).V) / poly(POLY_V, 1.0, 1.0)
    expect = closed_form_v(lat)[:, None] * unit[None, :]
    np.testing.assert_allclose(v, expect, rtol=1e-12, atol=1e-12 * np.abs(expect).max())


def test_jet_groups_are_padded_to_powers_of_two(monkeypatch):
    """Groups of 3, 4 and 5 cells of one pattern compile two routines, of 4 and 8 cells (Sec. 8.5)."""
    traces = []
    original = provider_jet.taylor_coefficients

    def counting(*args, **kwargs):
        traces.append(1)  # runs once per trace of the routine
        return original(*args, **kwargs)

    monkeypatch.setattr(provider_jet, "taylor_coefficients", counting)
    provider_jet._routine.cache_clear()
    ch, avg, lat = small_channels(3, 8), fixed_b_avg(), gamma_lattice(12, 1)
    padded = []
    for cells in ((0, 1, 2), (3, 4, 5, 6), (7, 8, 9, 10, 11)):
        events: dict = {}
        route_cells(avg, ch, lat, "jet", cells=cells, events=events)
        padded.append(events["jet padded cells"])
    assert padded == [4, 4, 8] and len(traces) == 2
    provider_jet._routine.cache_clear()


def test_jet_lattice_on_a_subset_equals_the_rows_of_the_whole():
    """Padded outputs are dropped. Measured difference 0.0 (load 7.6) against the whole lattice and ``jet_cell``."""
    ch, avg, lat = small_channels(3, 8), fixed_b_avg(), gamma_lattice(5, 2)
    whole = [np.asarray(a) for a in jet_lattice(avg, ch, lat)]
    part = jet_lattice(avg, ch, lat, cells=(4, 1, 2))
    one = jet_cell(avg, ch, lat, 1)
    sub = jet_lattice(avg, ch, lat, cells=(1,), exponents=((0,), (2,)))
    take = np.concatenate([np.arange(c * 3, c * 3 + 3) for c in (4, 1, 2)])
    for w, a, b, s in zip(whole, part, one, sub):
        np.testing.assert_array_equal(np.asarray(a), w[take])
        np.testing.assert_array_equal(np.asarray(b), w[3:6])
        np.testing.assert_array_equal(np.asarray(s), w[[3, 5]])
    with pytest.raises(ValueError, match="cells"):
        jet_lattice(avg, ch, lat, cells=(5,))
    with pytest.raises(ValueError, match="exponents"):
        jet_lattice(avg, ch, lat, exponents=((0, 1),))
