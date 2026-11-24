"""The memory step of the jet (design/0.8/SPEC.md decisions 165, 166 and 169, step 5b-3).

One compiled call per (cell, channel, node chunk). A channel of ``n_nu`` nodes runs in ``K = ceil(n_nu / 64)``
chunks of ``C = ceil(n_nu / K)`` nodes, the last padded with copies of the last node at weight 0, and the host
adds the chunk columns in chunk order. The tests cover the chunk plan, both sides of the chunk boundary against
one call over the whole channel, the memory budget at its boundary, the unavailable analysis, the order limit,
the trace slots of a routine and the estimate of one E1 call at order 4.

At 63 and 64 nodes a channel is one chunk and equals one call over the channel bit for bit. At 65, 128 and 400
nodes on the lowest and highest of 4 channels of 0.1 to 10 GHz, at orders 0 and 4, the chunked columns differ
from one call over the whole channel as one ``ResponseView`` by at most 6.6e-16 of the channel maximum (3.0 eps,
2026-11-19, jax 0.10.0 on CPU, 1-minute load 4.3 to 4.4). ``CHUNK_PINS`` holds each gap, and a test allows 4 times its pin or 4 eps.
"""

import jax
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.engine import provider_jet
from syncem.engine._jet_routine import (
    DEFAULT_JET_MAX_ORDER,
    DEFAULT_JET_MEM_BYTES,
    JET_MAX_ORDER_ENV,
    JET_MEM_ENV,
    JET_NODE_CHUNK,
    TRACE_SLOTS,
    JetMemoryError,
    ResponseView,
    chunk_plan,
    chunk_views,
)
from syncem.engine._routes import route_cells
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import build_from_averaged
from syncem.engine.provider_jet import jet_columns, jet_lattice, make_routine
from tests.v080.test_v080_jet_reference import e1_kernel
from tests.v080.test_v080_routes import eta4, fixed_b_avg, switch_lattice
from tests.v080.test_v080_stack import screen
from tests.v080.test_v080_stack_boundaries import gamma_lattice

K = ContinuumKernel(x_min=1e-12)
NAMES, CHARTS = ("gamma",), ("log",)
AXES = {"gamma": ("log", 3e3, 0.24)}
REF, HW = np.array([3e3]), np.array([0.24])
JET_EVENTS = ("jet groups", "jet calls", "jet compiles", "jet memory unchecked")


def chans(n, n_nu, lo=0.1, hi=10.0):
    return Channels.log(lo, hi, n, unit="GHz", half_width=0.2, n_nu=n_nu)


def exps(order):
    return tuple((n,) for n in range(order + 1))


def channel_gap(new, old):
    """Largest difference over the largest value of each channel (axis 1) of ``old``."""
    new, old = np.asarray(new), np.asarray(old)
    scale = np.abs(old).max(axis=0)
    diff = np.abs(new - old).max(axis=0)
    assert np.all(diff[scale == 0] == 0)
    return float(np.max(diff[scale > 0] / scale[scale > 0], initial=0.0))


def one_view(row):
    """The whole channel ``row`` as one ``ResponseView``, without chunks."""
    return ResponseView(row.nodes, row.weights, row.response_weights(), row.n_nu)


# ------------------------------------------------------------------ chunk plan and views


@pytest.mark.parametrize(
    "n_nu, plan",
    [(1, (1, 1)), (63, (1, 63)), (64, (1, 64)), (65, (2, 33)), (128, (2, 64))]
    + [(129, (3, 43)), (256, (4, 64)), (400, (7, 58))],
)
def test_the_chunk_plan(n_nu, plan):
    k, c = chunk_plan(n_nu)
    assert (k, c) == plan and c <= JET_NODE_CHUNK and (k - 1) * c < n_nu <= k * c


@pytest.mark.parametrize("bad", [0, -3, 2.5, True])
def test_the_chunk_plan_refuses_a_bad_count(bad):
    with pytest.raises(ValueError, match="n_nu"):
        chunk_plan(bad)


def test_chunk_views_pad_with_the_last_node_at_weight_zero():
    ch = chans(2, 65)
    views = chunk_views(ch, 1)
    assert len(views) == 2 and all(v.nodes.shape == (1, 33) for v in views)
    nodes = np.concatenate([np.asarray(v.nodes)[0] for v in views])
    weights = np.concatenate([np.asarray(v.weights)[0] for v in views])
    assert nodes[:65].tobytes() == np.asarray(ch.nodes)[1].tobytes()
    assert weights[:65].tobytes() == np.asarray(ch.weights)[1].tobytes()
    assert nodes[65] == nodes[64] and weights[65] == 0.0
    assert np.asarray(views[1].response_weights())[0, -1] == 0.0
    one = chunk_views(chans(3, 64), 2)
    assert len(one) == 1
    assert (
        np.asarray(one[0].nodes).tobytes()
        == np.asarray(chans(3, 64).nodes)[2:].tobytes()
    )


# ------------------------------------------------------------------ both sides of the chunk boundary

CHUNK_PINS = (
    {  # (n_nu, order, channel): measured gap over I and P, 1-minute load 4.3 to 4.4
        (65, 0, 0): 2.36e-16,
        (65, 0, 3): 1.39e-16,
        (65, 4, 0): 2.36e-16,
        (65, 4, 3): 4.16e-16,
        (128, 0, 0): 1.48e-16,
        (128, 0, 3): 1.41e-16,
        (128, 4, 0): 1.48e-16,
        (128, 4, 3): 2.77e-16,
        (400, 0, 0): 6.56e-16,
        (400, 0, 3): 1.41e-16,
        (400, 4, 0): 6.56e-16,
        (400, 4, 3): 1.41e-16,
    }
)


@pytest.mark.parametrize("n_nu", [63, 64, 65, 128, 400])
@pytest.mark.parametrize("order", [0, 4])
def test_chunked_columns_equal_one_call_over_the_channel(n_nu, order):
    ch, avg = chans(4, n_nu), fixed_b_avg()
    whole = make_routine(NAMES, CHARTS, exps(order))
    for j in (0, 3):
        row = ch[[j]]
        chunked = jet_columns(avg, row, AXES, list(exps(order)))
        single = whole(avg, one_view(row), REF, HW)[:2]
        for got, ref in zip(chunked, single):
            if n_nu <= JET_NODE_CHUNK:
                assert np.asarray(got).tobytes() == np.asarray(ref).tobytes()
            else:
                pin = CHUNK_PINS[(n_nu, order, j)]
                assert channel_gap(got, ref) <= max(4 * pin, 4 * np.finfo(float).eps)


# ------------------------------------------------------------------ the memory budget


def _case(order=2, n_nu=16):
    view = chunk_views(chans(2, n_nu), 0)[0]
    return (fixed_b_avg(), view, REF, HW), exps(order)


def test_the_default_budget_and_order_limit():
    assert DEFAULT_JET_MEM_BYTES == 4_000_000_000 and DEFAULT_JET_MAX_ORDER == 12
    assert TRACE_SLOTS == 4 and JET_NODE_CHUNK == 64


def test_the_budget_boundary(monkeypatch):
    args, e = _case()
    parts = make_routine(NAMES, CHARTS, e).estimate(*args)
    assert len(parts) == 3 and parts[0] > 0 and min(parts) >= 0
    total = sum(parts)
    monkeypatch.setenv(JET_MEM_ENV, str(total))
    make_routine(NAMES, CHARTS, e)(*args)
    monkeypatch.setenv(JET_MEM_ENV, str(total - 1))
    refused = make_routine(NAMES, CHARTS, e)
    with pytest.raises(JetMemoryError, match=JET_MEM_ENV):
        refused(*args)
    assert refused.programs == 0
    monkeypatch.setenv(JET_MEM_ENV, "0")
    make_routine(NAMES, CHARTS, e)(*args)
    monkeypatch.setenv(JET_MEM_ENV, "lots")
    with pytest.raises(ValueError, match=JET_MEM_ENV):
        make_routine(NAMES, CHARTS, e)(*args)


def test_a_tiny_budget_refuses_with_the_parts_and_the_remedies(monkeypatch):
    monkeypatch.setenv(JET_MEM_ENV, "1")
    with pytest.raises(JetMemoryError) as info:
        jet_lattice(fixed_b_avg(), chans(3, 16), gamma_lattice(2, 2))
    err, msg = info.value, str(info.value)
    assert isinstance(err, MemoryError) and err.kind == "budget"
    for part in (
        "('gamma',)",
        "order 2",
        "3 exponents",
        "one channel of 16 nodes per call",
    ):
        assert part in msg, part
    for part in ("3 channels", "2 cells", "temporary", "arguments", "outputs"):
        assert part in msg, part
    for part in ("SYNCEM_JET_MEM = 1 bytes", "order", "stack", "fewer nodes"):
        assert part in msg, part


@pytest.mark.parametrize("broken", ["none", "raise"])
def test_an_unavailable_analysis_runs_unchecked_and_warns_once(monkeypatch, broken):
    ch, avg, lat = chans(3, 65), fixed_b_avg(), gamma_lattice(2, 2)
    reference = jet_lattice(avg, ch, lat, routine=make_routine)

    def analysis(self):
        if broken == "raise":
            raise RuntimeError("no analysis on this backend")
        return None

    monkeypatch.setattr(jax.stages.Compiled, "memory_analysis", analysis)
    monkeypatch.setenv(JET_MEM_ENV, "1")
    events: dict = {}
    with pytest.warns(RuntimeWarning, match=jax.default_backend()) as caught:
        out = jet_lattice(avg, ch, lat, routine=make_routine, events=events)
    assert sum(issubclass(w.category, RuntimeWarning) for w in caught) == 1
    assert events["jet memory unchecked"] == events["jet compiles"] == 1
    assert events["jet calls"] == 2 * 3 * 2 and events["jet groups"] == 1
    for a, b in zip(out, reference):
        assert np.asarray(a).tobytes() == np.asarray(b).tobytes()


# ------------------------------------------------------------------ the order limit


def test_above_the_ceiling_raises_before_tracing(monkeypatch):
    def fail(*a, **k):
        pytest.fail("traced")

    monkeypatch.setattr(provider_jet, "taylor_mode_coefficients", fail)
    too_high = DEFAULT_JET_MAX_ORDER + 1
    with pytest.raises(JetMemoryError, match=f"order {too_high}") as info:
        jet_columns(fixed_b_avg(), chans(2, 8), AXES, list(exps(too_high)))
    msg = str(info.value)
    assert info.value.kind == "order" and f"SYNCEM_JET_MAX_ORDER={too_high}" in msg
    assert "stack" in msg and f"orders up to {DEFAULT_JET_MAX_ORDER}" in msg
    with pytest.raises(JetMemoryError, match="cell 0"):
        jet_lattice(fixed_b_avg(), chans(2, 8), gamma_lattice(2, too_high))
    with pytest.raises(JetMemoryError, match=f"order {too_high}"):
        make_routine(NAMES, CHARTS, exps(too_high))


def test_the_order_limit_follows_the_environment(monkeypatch):
    args, e = _case(order=2)
    monkeypatch.setenv(JET_MAX_ORDER_ENV, "1")
    with pytest.raises(JetMemoryError, match="orders up to 1"):
        jet_columns(args[0], chans(2, 8), AXES, list(e))
    monkeypatch.setenv(JET_MAX_ORDER_ENV, "2")
    assert np.all(np.isfinite(jet_columns(args[0], chans(2, 8), AXES, list(e))[0]))
    for bad in ("0", "-1", "two", "2.5"):
        monkeypatch.setenv(JET_MAX_ORDER_ENV, bad)
        with pytest.raises(ValueError, match=JET_MAX_ORDER_ENV):
            jet_columns(args[0], chans(2, 8), AXES, list(e))


def test_an_order_7_cell_names_the_reason_of_the_stack(monkeypatch):
    monkeypatch.setenv(JET_MAX_ORDER_ENV, "6")
    reason = "order 7 exceeds the S1 table order 6"
    lattice, ch = switch_lattice(7, 1, 3e7, 1.2e8), chans(2, 8)
    with pytest.raises(JetMemoryError, match=reason) as info:
        route_cells(fixed_b_avg(), ch, lattice, "auto")
    assert "cell 0" in str(info.value)
    with pytest.raises(JetMemoryError, match=reason):
        build_from_averaged(fixed_b_avg(), ch, lattice, azimuth_factor=0.4)


@pytest.mark.slow
def test_order_5_runs():
    """Order 5 is the largest order of the nested jet."""
    i, p = jet_columns(fixed_b_avg(), chans(2, 8), AXES, list(exps(5)))
    assert np.all(np.isfinite(np.asarray(i))) and np.all(np.isfinite(np.asarray(p)))


# ------------------------------------------------------------------ trace slots, events, defaults


def test_a_routine_keeps_at_most_four_programs():
    e, events = exps(1), {}
    for _ in range(6):
        avg = AveragedKernel(K, [eta4()], fixed={"B": 5e-6}, phase=screen())
        jet_lattice(avg, chans(2, 8), gamma_lattice(1, 1), events=events)
    assert events["jet compiles"] == 6
    assert provider_jet._direct(NAMES, CHARTS, e).programs == TRACE_SLOTS


def test_route_events_count_calls_and_compiles():
    events: dict = {}
    route_cells(fixed_b_avg(), chans(3, 65), gamma_lattice(2, 1), "jet", events=events)
    assert "jet padded cells" not in events
    assert {k: events[k] for k in JET_EVENTS} == {
        "jet groups": 1,
        "jet calls": 2 * 3 * 2,
        "jet compiles": 1,
        "jet memory unchecked": 0,
    }


def test_the_build_records_the_jet_defaults(monkeypatch):
    ch, lat = chans(2, 8), gamma_lattice(1, 1)
    basis = build_from_averaged(
        fixed_b_avg(), ch, lat, routes="jet", azimuth_factor=0.4, cache=False
    )
    defaults = dict(basis.record.defaults)
    assert defaults["jet_mem_bytes"] == DEFAULT_JET_MEM_BYTES
    assert defaults["jet_max_order"] == DEFAULT_JET_MAX_ORDER
    monkeypatch.setenv(JET_MEM_ENV, "8GB")
    monkeypatch.setenv(JET_MAX_ORDER_ENV, "3")
    basis = build_from_averaged(
        fixed_b_avg(), ch, lat, routes="jet", azimuth_factor=0.4, cache=False
    )
    assert dict(basis.record.defaults)["jet_mem_bytes"] == 8_000_000_000
    assert dict(basis.record.defaults)["jet_max_order"] == 3


def test_one_e1_call_at_order_4_is_estimated_below_0_2_gb():
    """400 nodes run in 7 chunks of 58. The audit computed about 98 MB of temporary buffer per call."""
    view = chunk_views(chans(1, 400, 1.0, 1.0), 0)[0]
    parts = make_routine(NAMES, CHARTS, exps(4)).estimate(e1_kernel(), view, REF, HW)
    assert sum(parts) < 0.2e9
