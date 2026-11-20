"""The one-node store and the local store of uncached builds (design/0.8/SPEC.md decisions 167 to 169).

``Store.admit`` keeps a new table only if it fits after evicting unprotected entries in least-recently-used
order, and otherwise stores and evicts nothing. ``NodeStore`` protects the one-node tables of its own build,
so a cyclic walk whose working set exceeds the bound keeps the first ``R`` tables and builds the others at
every read. With ``M`` distinct tables, ``A`` walks over them and ``R`` resident tables, a build makes
``M + (A - 1)(M - R)`` table builds when ``M > R`` (decision 168). The dry walk of ``stack_stage`` keys the
tables and builds none. ``Cache.local`` is the byte-bounded store of a ``cache=False`` or ``NotCacheable``
build, with no disk and no directory lookup (decision 167). ``Cache.clear(jax=True)`` also clears the JAX
caches (decision 169).

The bounds are set from the measured charge of one one-node table (``ONE``, 6.3 MB on this grid), so ``R``
follows from the bound: ``R = floor(bound / ONE)``. The S0 table of the walk is held by the build's
``StageMemo`` and does not take a table slot. Every comparison is an exact identity of integers or bytes.
"""

from __future__ import annotations

import jax
import numpy as np
import pytest

from syncem.engine import _disk
from syncem.engine import cache as cache_module
from syncem.engine._cache_store import Store, nbytes_of
from syncem.engine._stack_series import stack_cell, stack_lattice, stack_stage
from syncem.engine._stage_memo import StageMemo, local_memo, s1_arrays
from syncem.engine.average import AveragedKernel, uniform_rule
from syncem.engine.basis import build_from_averaged
from syncem.engine.cache import Cache, default_cache
from syncem.engine.field_average import FieldAverage
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from tests.v080.test_v080_cache import K
from tests.v080.test_v080_routes import eta4
from tests.v080.test_v080_stack import small_channels

ONE = nbytes_of(s1_arrays(FieldAverage.lookups([5.0], [1.0])))
M_PER_NODE = 16  # the distinct tables of distinct_case, the same in each cell
CELLS = 3


def distinct_case():
    """``B`` on 4 nodes given ``eta`` on 4 nodes: 16 distinct one-node tables, each read once per cell.

    Both ends of the field law move with ``eta``, so no two nodes coincide. The ``per_node_case`` of
    ``test_v080_cache.py`` shares nodes between ``eta`` nodes, and its walk reads 11 distinct tables.
    """
    lo, hi = (lambda p, s=s: s * (1 + p["eta"]) for s in (4e-6, 8e-6))
    b = uniform_rule(lo, hi, 4, parents=("eta",))
    avg = AveragedKernel(K, [("eta", uniform_rule(0.0, 0.5, 4)), ("B", b)])
    return avg, Lattice(gamma=Cells.log(300, 3e6, CELLS, order=2)), 0.4


def fixed_b_case():
    """``B`` fixed at 5 uG: one one-node table, read once per cell."""
    avg = AveragedKernel(K, [eta4()], fixed={"B": 5e-6})
    return avg, Lattice(gamma=Cells.log(300, 3e6, CELLS, order=2)), 0.4


def build(case, **kwargs):
    avg, lattice, factor = case
    ch = small_channels(3, 8)
    return build_from_averaged(avg, ch, lattice, azimuth_factor=factor, **kwargs)


def node_counts(basis):
    ev = dict(basis.record.events)
    return (
        ev["one-node tables built"],
        ev["one-node rebuilds"],
        ev["one-node tables not kept"],
    )


def predicted(m, a, r):
    """Builds, rebuilds and refusals of a cyclic walk (decision 168)."""
    if r >= m:
        return m, 0, 0
    return m + (a - 1) * (m - r), (a - 1) * (m - r), a * (m - r)


def same_bits(a, b):
    for x, y in ((a.spectra0, b.spectra0), (a.spectra2, b.spectra2)):
        assert (x is None) == (y is None)
        if x is not None:
            assert np.asarray(x).tobytes() == np.asarray(y).tobytes()


# ------------------------------------------------------------------ Store.admit


def _filled(entries):
    store = Store(sum(size for _, size in entries))
    for key, size in entries:
        store.put(key, key, size)
    return store


def mine(key):
    return key[:2] == ("node", 7)


def test_admit_evicts_unprotected_entries_in_lru_order_until_it_fits():
    store = _filled([(("s0", "a"), 1), (("node", 7, "p"), 1), (("s1", "b"), 1)])
    assert store.admit(("node", 7, "q"), "q", 2, mine) == (True, 2)
    assert ("s0", "a") not in store and ("s1", "b") not in store
    assert ("node", 7, "p") in store and store.nbytes == 3


def test_admit_takes_only_as_many_victims_as_it_needs():
    store = _filled([(("s0", "a"), 1), (("node", 8, "x"), 1), (("s1", "b"), 1)])
    assert store.admit(("node", 7, "q"), "q", 2, mine) == (True, 2)
    assert ("s1", "b") in store  # the most recent unprotected entry stays
    assert ("node", 8, "x") not in store  # another build's table is unprotected


def test_admit_stores_and_evicts_nothing_when_protected_entries_leave_no_room():
    store = _filled([(("node", 7, "p"), 2), (("s0", "a"), 1)])
    assert store.admit(("node", 7, "q"), "q", 2, mine) == (False, 0)
    assert ("s0", "a") in store and ("node", 7, "p") in store and store.nbytes == 3
    assert store.admit(("node", 7, "big"), "big", 4, mine) == (False, 0)
    assert store.admit(("node", 7, "r"), "r", 1, mine) == (True, 1)


# ------------------------------------------------------------------ NodeStore


def _table(seed):
    return {"lny": np.full(1000, float(seed))}


def test_a_table_that_needs_an_own_slot_is_used_and_not_kept(tmp_path):
    size = nbytes_of(_table(0))
    cache = Cache(directory=tmp_path, max_bytes=0, mem_bytes=2 * size)
    built = []

    def make(seed):
        built.append(seed)
        return _table(seed)

    with cache.node_store() as nodes:
        nodes.get(0, lambda: make(0))
        nodes.get(1, lambda: make(1))
        third = nodes.get(2, lambda: make(2))  # does not evict 0 or 1
        assert third["lny"].tobytes() == _table(2)["lny"].tobytes()
        assert not third["lny"].flags.writeable
        for seed in (0, 1, 2):
            nodes.get(seed, lambda s=seed: make(s))
        assert built == [0, 1, 2, 2]
        assert (nodes.rebuilds, nodes.not_kept, cache.events["rebuilds"]) == (1, 2, 1)
        assert nodes.distinct_bytes == 3 * size and cache.nbytes == 2 * size
    assert cache.nbytes == 0


def test_a_one_node_table_evicts_other_kinds_and_other_builds(tmp_path):
    size = nbytes_of(_table(0))
    cache = Cache(directory=tmp_path, max_bytes=0, mem_bytes=2 * size)
    cache.put("memo", "m", _table(9))
    with cache.node_store() as old, cache.node_store() as new:
        old.get("x", lambda: _table(1))
        new.get("y", lambda: _table(2))
        new.get("z", lambda: _table(3))
        assert cache.peek("memo", "m") is None and new.not_kept == 0
        assert cache.nbytes == 2 * size


# ------------------------------------------------------------------ Cache.local and clear


def test_a_local_cache_never_looks_up_the_directory(monkeypatch):
    def refuse(*args, **kwargs):
        raise AssertionError("cache_dir called")

    monkeypatch.setattr(_disk, "cache_dir", refuse)
    monkeypatch.setenv("SYNCEM_CACHE_MEM", "12kB")
    local = Cache.local()
    assert (local.directory, local.disk_enabled, local.mem_bytes) == (
        None,
        False,
        12_000,
    )
    assert Cache.local(mem_bytes=5).mem_bytes == 5
    assert ("cache_directory", None) in local.settings
    key = "a" * 64
    local.put("s1", key, {"x": np.arange(3.0)})
    assert local.peek("s1", key) == "memory"
    assert local.get("s1", key)["x"].tobytes() == np.arange(3.0).tobytes()
    with pytest.raises(ValueError, match="64 lowercase hex"):
        local.put("s0", "short", {"x": np.zeros(1)})
    local.clear(disk=True)
    assert local.nbytes == 0
    assert isinstance(local_memo().cache, Cache) and local_memo().cache is not local


def test_clear_with_jax_also_clears_the_jax_caches(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(jax, "clear_caches", lambda: calls.append(1))
    cache = Cache(directory=tmp_path, max_bytes=0)
    cache.clear()
    assert calls == []
    cache.clear(jax=True)
    assert calls == [1]


# ------------------------------------------------------------------ the dry walk


def test_the_dry_walk_keys_the_one_node_tables_and_builds_none():
    avg, lattice, _ = distinct_case()
    ch = small_channels(3, 8)
    memo = local_memo()
    assert stack_stage(avg, ch, lattice, 0, memo=memo) == "per-node"
    dry = memo.take_reads()
    assert len(dry) == M_PER_NODE and memo.counts["one-node tables built"] == 0
    stack_cell(avg, ch, lattice, 0, memo=memo)
    assert memo.take_reads() == dry
    assert memo.counts["one-node tables built"] == M_PER_NODE
    assert memo.close()["one-node tables not kept"] == 0


# ------------------------------------------------------------------ scan resistance in a build


BOUNDS = {
    "all": M_PER_NODE,
    "one short": M_PER_NODE - 1,
    "half": M_PER_NODE // 2,
    "none": 0,
}


def _bound(r):
    return r * ONE + ONE // 2


@pytest.mark.parametrize("cache", [False, True])
@pytest.mark.parametrize("name", list(BOUNDS))
def test_a_cyclic_walk_makes_the_predicted_builds(name, cache, monkeypatch):
    r = BOUNDS[name]
    monkeypatch.setenv("SYNCEM_CACHE_MEM", str(_bound(r)))
    bounded = build(distinct_case(), cache=cache)
    assert node_counts(bounded) == predicted(M_PER_NODE, CELLS, r)
    assert {x.mode for x in bounded.record.routes} == {"per-node"}
    monkeypatch.setenv("SYNCEM_CACHE_MEM", "100GB")
    unbounded = build(distinct_case(), cache=False)
    assert node_counts(unbounded) == (M_PER_NODE, 0, 0)
    same_bits(bounded, unbounded)


@pytest.mark.parametrize("r", [0, 1])
def test_a_single_table_read_once_per_cell(r, monkeypatch):
    monkeypatch.setenv("SYNCEM_CACHE_MEM", str(_bound(r)))
    bounded = build(fixed_b_case(), cache=False)
    assert node_counts(bounded) == predicted(1, CELLS, r)
    monkeypatch.setenv("SYNCEM_CACHE_MEM", "100GB")
    same_bits(bounded, build(fixed_b_case(), cache=False))


def test_tables_not_kept_give_a_note_with_the_setting_that_keeps_them(monkeypatch):
    monkeypatch.setenv("SYNCEM_CACHE_MEM", str(_bound(0)))
    notes = build(distinct_case(), cache=False).record.notes
    m = M_PER_NODE
    head = f"one-node tables: {CELLS * m} not kept, {(CELLS - 1) * m} rebuilt."
    assert notes[0] == "not cached: cache=False"
    assert len(notes) == 2 and notes[1].startswith(head)
    assert f"mem_bytes of {_bound(0) / 1e9:.3g} GB" in notes[1]
    setting = notes[1].rsplit("SYNCEM_CACHE_MEM=", 1)[1].split()[0]
    assert _disk.parse_size(setting) >= m * ONE
    monkeypatch.setenv("SYNCEM_CACHE_MEM", setting)
    again = build(distinct_case(), cache=False)
    assert node_counts(again) == (m, 0, 0)
    assert again.record.notes == ("not cached: cache=False",)


# ------------------------------------------------------------------ uncached builds stay local


def test_an_uncached_build_keeps_its_tables_under_the_bound_and_out_of_the_process(
    monkeypatch, tmp_path
):
    seen = []
    original = cache_module.NodeStore.get

    def watched(self, key, make):
        out = original(self, key, make)
        seen.append((self._cache, self._cache.nbytes, self._cache.mem_bytes))
        return out

    monkeypatch.setattr(cache_module.NodeStore, "get", watched)
    monkeypatch.setenv("SYNCEM_CACHE_MEM", str(_bound(5)))
    process = default_cache()
    before = (process.nbytes, process.events)
    build(distinct_case(), cache=False)
    assert (process.nbytes, process.events) == before
    assert seen and all(c is not process and c.directory is None for c, _, _ in seen)
    assert max(n for _, n, _ in seen) <= _bound(5) == seen[0][2]
    assert not process.directory.exists() or not any(process.directory.rglob("*.npz"))


def test_the_local_memo_of_stack_lattice_equals_the_plain_dict_memo(monkeypatch):
    avg, lattice, _ = distinct_case()
    ch = small_channels(3, 8)
    monkeypatch.setenv("SYNCEM_CACHE_MEM", str(_bound(2)))
    local = stack_lattice(avg, ch, lattice)
    memo: dict = {}
    parts = [stack_cell(avg, ch, lattice, c, memo=memo) for c in range(lattice.n_cells)]
    for k in range(2):
        plain = np.concatenate([p[k] for p in parts])
        assert np.asarray(local[k]).tobytes() == plain.tobytes()
    assert isinstance(local_memo(), StageMemo)
