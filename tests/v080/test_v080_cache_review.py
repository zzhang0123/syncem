"""Findings of the step-5b-3 review on the stores (design/0.8/SPEC.md decisions 167, 168 and 171).

``route_cells`` without ``memo=`` reads a ``local_memo()`` closed at the end of the call, as ``stack_cell`` and
``stack_lattice`` do. ``at`` on a parent without the cache adds the counts of its memos to the events of the
derived record and the one-node note of decision 168 to its notes. A ``StageMemo`` holds the S0 table it built
as the copy the cache stored, so the arrays exist once while the store keeps them.
"""

from __future__ import annotations

import numpy as np

from syncem.base.channels import Channels
from syncem.engine import _stage_memo
from syncem.engine._keys import s0_key
from syncem.engine._routes import route_cells
from syncem.engine._stage_memo import PRIMITIVE_STEP, StageMemo
from syncem.engine.basis import build_from_averaged
from syncem.engine.cache import Cache
from syncem.engine.field_average import TABLE_ORDER
from syncem.engine.primitive import quantised_span
from tests.v080.test_v080_cache_nodes import _bound, distinct_case
from tests.v080.test_v080_stack import small_channels

NODE_EVENTS = ("one-node tables built", "one-node rebuilds", "one-node tables not kept")


def test_route_cells_without_a_memo_reads_a_closed_local_memo(monkeypatch):
    made = []

    def spy():
        memo = StageMemo(Cache.local())
        made.append(memo)
        return memo

    monkeypatch.setattr(_stage_memo, "local_memo", spy)
    avg, lattice, _ = distinct_case()
    out = route_cells(avg, small_channels(3, 8), lattice, "stack")
    assert [r.provider for r in out[3]] == ["stack"] * lattice.n_cells
    assert len(made) == 1
    memo = made[0]
    assert memo.counts["one-node tables built"] > 0
    # closed: no S0 table held, and the store keeps the S0 table only (8,960,032 bytes over the default span)
    assert memo._primitives == {} and memo.cache.nbytes < 9 * 10**6


def test_at_on_an_uncached_parent_records_its_memo_counts_and_note(monkeypatch):
    monkeypatch.setenv("SYNCEM_CACHE_MEM", str(_bound(0)))
    avg, lattice, factor = distinct_case()
    parent = build_from_averaged(
        avg, small_channels(3, 8), lattice, azimuth_factor=factor, cache=False
    )
    new = Channels.log(0.2, 20, 3, unit="GHz", half_width=0.2, n_nu=8)
    derived = parent.at(new)
    events = dict(derived.record.events)
    assert events["channel rows computed"] == 3
    built, rebuilt, not_kept = (events[name] for name in NODE_EVENTS)
    assert built > 0 and rebuilt > 0 and not_kept > 0
    assert events["S0 tables built"] == 1
    own = [n for n in derived.record.notes if n.startswith("one-node tables")]
    inherited = [n for n in parent.record.notes if n.startswith("one-node tables")]
    assert len(own) == len(inherited) + 1
    assert own[-1].startswith(
        f"one-node tables: {not_kept} not kept, {rebuilt} rebuilt"
    )


def test_a_built_s0_table_is_held_as_the_stored_copy():
    cache = Cache.local(mem_bytes=10**9)
    memo = StageMemo(cache)
    span = quantised_span(-30.0, 20.0)
    held = memo.primitive(span, TABLE_ORDER)
    stored = cache.get("s0", s0_key(span, TABLE_ORDER, PRIMITIVE_STEP))
    assert np.shares_memory(stored["coef_F"], held.coef["F"])
    assert np.shares_memory(stored["coef_G"], held.coef["G"])
    assert memo.primitive(span, TABLE_ORDER) is held
    assert memo.counts["S0 tables built"] == 1 and memo.counts["S0 tables read"] == 1
    memo.close()
