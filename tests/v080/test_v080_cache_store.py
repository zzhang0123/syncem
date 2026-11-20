"""Step 5b-2 of design/0.8/SPEC.md: the process ``Cache`` of ``syncem.engine.cache`` (decisions 160 to 162).

These are unit tests of the store itself, without a build: the byte-bounded least-recently-used
memory, ``peek``, the disk stages, the jet routines, the trial LRU, the per-build one-node store and
``clear``. The S2 columns and their rule chains are in ``test_v080_cache_columns.py``. Every
dispatcher is tested on both sides: memory hit against disk hit against miss, disk enabled against
disabled, an equal token against a fresh one. Every comparison is an exact identity of bytes,
integers or object identity, so no tolerance enters.

Every directory is under ``tmp_path``. The autouse fixtures of ``tests/conftest.py`` point
``SYNCEM_CACHE`` at a per-test directory and give each test a fresh default cache.
"""

from __future__ import annotations

import os
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.engine import _disk
from syncem.engine._cache_store import memo_bytes
from syncem.engine.cache import (
    DEFAULT_MEM_BYTES,
    ROUTINE_SLOTS,
    TRIAL_SLOTS,
    Cache,
    default_cache,
)

KEYS = [c * 64 for c in "abcdef"]
N = 1000  # float64 values per test array, 8000 bytes
OLD = 1_000_000_000  # a fixed modification time (2001)


def _table(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"lny": rng.standard_normal(N), "order": np.array([4], dtype=np.int64)}


def _nbytes(table: dict[str, np.ndarray]) -> int:
    return sum(a.nbytes for a in table.values())


def _same(a, b) -> bool:
    return sorted(a) == sorted(b) and all(
        np.asarray(a[k]).dtype == np.asarray(b[k]).dtype
        and np.asarray(a[k]).shape == np.asarray(b[k]).shape
        and np.asarray(a[k]).tobytes() == np.asarray(b[k]).tobytes()
        for k in a
    )


def _memory(directory: Path, mem_bytes: int) -> Cache:
    """A cache without disk, so ``peek`` reports the memory alone."""
    return Cache(directory=directory, max_bytes=0, mem_bytes=mem_bytes)


def _disk_files(directory: Path) -> list[str]:
    return sorted(p.name for p in directory.rglob("*") if p.is_file())


# Settings ---------------------------------------------------------------------------------------


def test_defaults_follow_the_environment(monkeypatch):
    cache = Cache()
    assert cache.directory == Path(os.environ["SYNCEM_CACHE"])
    assert cache.mem_bytes == DEFAULT_MEM_BYTES == 1_000_000_000
    assert cache.max_bytes == _disk.DEFAULT_MAX_BYTES
    monkeypatch.setenv("SYNCEM_CACHE_MEM", "10kB")
    monkeypatch.setenv("SYNCEM_CACHE_MAX", "0")
    cache = Cache()
    assert (cache.mem_bytes, cache.max_bytes, cache.disk_enabled) == (10_000, 0, False)
    assert Cache(mem_bytes=5, max_bytes=7).mem_bytes == 5


@pytest.mark.parametrize("field", ["mem_bytes", "max_bytes"])
def test_negative_bounds_raise(field):
    with pytest.raises(ValueError, match="non-negative"):
        Cache(**{field: -1})


def test_bad_memory_size_raises(monkeypatch):
    monkeypatch.setenv("SYNCEM_CACHE_MEM", "lots")
    with pytest.raises(ValueError, match="SYNCEM_CACHE_MEM"):
        Cache()


def test_unknown_stage_and_bad_disk_key_raise(tmp_path):
    cache = Cache(directory=tmp_path)
    with pytest.raises(ValueError, match="stage"):
        cache.put("s3", KEYS[0], _table(0))
    with pytest.raises(ValueError, match="64 lowercase hex"):
        cache.put("s0", "short", _table(0))


# Byte-bounded memory ----------------------------------------------------------------------------


def test_eviction_removes_the_least_recently_used(tmp_path):
    cache = _memory(tmp_path, mem_bytes=2 * _nbytes(_table(0)))
    cache.put("s0", KEYS[0], _table(0))
    cache.put("s0", KEYS[1], _table(1))
    assert cache.nbytes == 2 * _nbytes(_table(0))
    assert cache.get("s0", KEYS[0]) is not None  # KEYS[0] is now the most recent
    cache.put("s0", KEYS[2], _table(2))
    assert [cache.peek("s0", k) for k in KEYS[:3]] == ["memory", None, "memory"]
    assert cache.events["evictions"] == 1


def test_peek_does_not_change_recency(tmp_path):
    cache = _memory(tmp_path, mem_bytes=2 * _nbytes(_table(0)))
    cache.put("s0", KEYS[0], _table(0))
    cache.put("s0", KEYS[1], _table(1))
    assert cache.peek("s0", KEYS[0]) == "memory"
    cache.put("s0", KEYS[2], _table(2))
    assert [cache.peek("s0", k) for k in KEYS[:3]] == [None, "memory", "memory"]


def test_stages_share_one_bound(tmp_path):
    size, memo = _nbytes(_table(0)), {"spectra0": np.zeros(N)}
    cache = _memory(tmp_path, mem_bytes=size + memo_bytes(memo))
    cache.put("s0", KEYS[0], _table(0))
    cache.put("memo", "m", memo)
    cache.put("s1", KEYS[1], _table(1))
    assert cache.peek("s0", KEYS[0]) is None
    assert cache.peek("memo", "m") == "memory"
    assert cache.nbytes == memo_bytes(memo) + size


def test_an_entry_above_the_bound_is_not_kept_and_evicts_nothing(tmp_path):
    cache = _memory(tmp_path, mem_bytes=_nbytes(_table(0)))
    cache.put("s0", KEYS[0], _table(0))
    cache.put("memo", "big", {"x": np.zeros(2 * N)})
    assert cache.peek("memo", "big") is None
    assert cache.peek("s0", KEYS[0]) == "memory"


def test_mem_zero_keeps_nothing_in_memory_and_still_writes_disk(tmp_path):
    cache = Cache(directory=tmp_path, mem_bytes=0)
    table = _table(0)
    cache.put("s0", KEYS[0], table)
    assert cache.nbytes == 0
    assert cache.peek("s0", KEYS[0]) == "disk"
    assert _same(cache.get("s0", KEYS[0]), table)
    assert cache.events["disk_hits"] == 1


def test_memory_hit_equals_the_stored_value_and_is_read_only(tmp_path):
    cache = _memory(tmp_path, mem_bytes=10**6)
    table = _table(0)
    cache.put("s0", KEYS[0], table)
    table["lny"][0] = 99.0  # the cache holds its own copy
    hit = cache.get("s0", KEYS[0])
    assert _same(hit, _table(0))
    with pytest.raises(ValueError):
        hit["lny"][0] = 1.0
    assert cache.events["hits"] == 1 and cache.get("s0", KEYS[1]) is None
    assert cache.events["misses"] == 1


def test_memo_counts_its_jax_and_numpy_leaves_and_containers(tmp_path):
    cache = _memory(tmp_path, mem_bytes=10**6)
    value = {
        "spectra0": jnp.zeros((4, 5), jnp.float32),
        "columns": (np.zeros(3), "label"),
    }
    cache.put("memo", "m", value)
    assert cache.nbytes == memo_bytes(value) > 4 * 5 * 4 + 3 * 8 + len("label")
    hit = cache.get("memo", "m")
    assert hit["columns"][1] == "label"
    assert np.asarray(hit["spectra0"]).tobytes() == np.zeros((4, 5), "f4").tobytes()


# Disk stages ------------------------------------------------------------------------------------


@pytest.mark.parametrize("stage", ["s0", "s1"])
def test_disk_hit_in_a_fresh_cache_equals_the_stored_arrays(tmp_path, stage):
    table = _table(3)
    Cache(directory=tmp_path).put(stage, KEYS[0], table, description={"n": N})
    fresh = Cache(directory=tmp_path)
    assert fresh.peek(stage, KEYS[0]) == "disk"
    assert _same(fresh.get(stage, KEYS[0]), table)
    assert fresh.peek(stage, KEYS[0]) == "memory"
    assert fresh.events["disk_hits"] == 1


def test_the_memo_never_goes_to_disk(tmp_path):
    cache = Cache(directory=tmp_path)
    cache.put("memo", KEYS[0], {"x": np.zeros(3)})
    assert _disk_files(tmp_path) == []
    assert Cache(directory=tmp_path).get("memo", KEYS[0]) is None


def test_cache_max_zero_disables_the_disk(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNCEM_CACHE_MAX", "0")
    Cache(directory=tmp_path).put("s0", KEYS[0], _table(0))
    assert _disk_files(tmp_path) == []
    assert Cache(directory=tmp_path).get("s0", KEYS[0]) is None


def test_disk_bound_evicts_the_oldest_entry(tmp_path):
    cache = Cache(directory=tmp_path)
    cache.put("s0", KEYS[0], _table(0))
    first = _disk.entry_path(tmp_path, "s0", KEYS[0])
    size = first.stat().st_size + first.with_suffix(".json").stat().st_size
    for path in (first, first.with_suffix(".json")):
        os.utime(path, (OLD, OLD))
    small = Cache(directory=tmp_path, max_bytes=size)
    small.put("s0", KEYS[1], _table(1))
    assert Cache(directory=tmp_path).peek("s0", KEYS[0]) is None
    assert Cache(directory=tmp_path).peek("s0", KEYS[1]) == "disk"


def test_a_failed_write_is_skipped_and_counted(tmp_path):
    blocker = tmp_path / "file"
    blocker.write_text("not a directory")
    cache = Cache(directory=blocker)
    cache.put("s0", KEYS[0], _table(0))
    assert cache.events["disk_write_errors"] == 1
    assert cache.peek("s0", KEYS[0]) == "memory"


def test_a_corrupt_entry_is_a_miss(tmp_path):
    Cache(directory=tmp_path).put("s0", KEYS[0], _table(0))
    _disk.entry_path(tmp_path, "s0", KEYS[0]).write_bytes(b"garbage")
    fresh = Cache(directory=tmp_path)
    assert fresh.get("s0", KEYS[0]) is None
    assert fresh.events["misses"] == 1


# Column ids for ``clear``. The S2 cases are in test_v080_cache_columns.py. ---------------------

IDS = tuple(((0,), (n,)) for n in range(5))
CHAIN = tuple(f"d{n}" for n in range(5))


# Jet routines and the trial LRU -----------------------------------------------------------------

SIGNATURE = (("B",), ("log",), ((0,), (1,)))


def _maker():
    calls = []

    def make(names, charts, exponents):
        calls.append((names, charts, exponents))
        return object()

    return make, calls


def test_routine_is_reused_for_an_equal_token(tmp_path):
    cache = _memory(tmp_path, 0)
    make, calls = _maker()
    first = cache.routine(*SIGNATURE, ("fp", 1), make=make)
    assert cache.routine(*SIGNATURE, ("fp", 1), make=make) is first
    assert calls == [SIGNATURE]
    assert (
        cache.routine(("B",), ("lin",), ((0,), (1,)), ("fp", 1), make=make) is not first
    )


def test_routine_is_fresh_for_fresh_tokens_and_another_maker(tmp_path):
    cache = _memory(tmp_path, 0)
    make, calls = _maker()
    a = cache.routine(*SIGNATURE, object(), make=make)
    b = cache.routine(*SIGNATURE, object(), make=make)
    other, _ = _maker()
    c = cache.routine(*SIGNATURE, "t", make=make)
    assert a is not b and cache.routine(*SIGNATURE, "t", make=other) is not c
    assert len(calls) == 3


def test_routines_keep_the_last_slots(tmp_path):
    cache = _memory(tmp_path, 0)
    make, calls = _maker()
    for token in range(ROUTINE_SLOTS + 1):
        cache.routine(*SIGNATURE, token, make=make)
    cache.routine(*SIGNATURE, ROUTINE_SLOTS, make=make)
    cache.routine(*SIGNATURE, 0, make=make)
    assert len(calls) == ROUTINE_SLOTS + 2


def test_trial_lru_keeps_eight_entries_and_never_writes_disk(tmp_path):
    assert TRIAL_SLOTS == 8
    cache = Cache(directory=tmp_path)
    computed = []

    def compute(key):
        computed.append(key)
        return _table(key)

    for key in range(TRIAL_SLOTS):
        cache.trial(key, lambda k=key: compute(k))
    cache.trial(0, lambda: compute(0))  # a hit makes 0 the most recent
    cache.trial(TRIAL_SLOTS, lambda: compute(TRIAL_SLOTS))  # evicts 1
    cache.trial(0, lambda: compute(0))
    cache.trial(1, lambda: compute(1))
    assert computed == [*range(TRIAL_SLOTS), TRIAL_SLOTS, 1]
    assert _same(cache.trial(1, lambda: None), _table(1))
    assert _disk_files(tmp_path) == []


# One-node store ---------------------------------------------------------------------------------


def test_node_store_rebuilds_a_table_it_did_not_keep_and_counts_it(tmp_path):
    size = _nbytes(_table(0))
    cache = _memory(tmp_path, mem_bytes=2 * size)
    built = []

    def build(seed):
        built.append(seed)
        return _table(seed)

    with cache.node_store() as nodes:
        first = nodes.get(("node", 0), lambda: build(0))
        assert _same(nodes.get(("node", 0), lambda: build(0)), first)
        nodes.get(("node", 1), lambda: build(1))
        third = nodes.get(("node", 2), lambda: build(2))  # not kept (decision 168)
        assert _same(nodes.get(("node", 0), lambda: build(0)), first)
        again = nodes.get(("node", 2), lambda: build(2))
        assert nodes.rebuilds == 1 and built == [0, 1, 2, 2]
        assert _same(again, third) and nodes.not_kept == 2
    assert cache.nbytes == 0 and cache.events["rebuilds"] == 1
    assert _disk_files(tmp_path) == []


def test_node_stores_of_two_builds_are_separate(tmp_path):
    cache = _memory(tmp_path, mem_bytes=10**6)
    with cache.node_store() as a, cache.node_store() as b:
        a.get("n", lambda: _table(0))
        assert _same(b.get("n", lambda: _table(1)), _table(1))
        assert (a.rebuilds, b.rebuilds) == (0, 0)


# clear and the default cache --------------------------------------------------------------------


def test_clear_empties_memory_and_keeps_disk_unless_asked(tmp_path):
    cache = Cache(directory=tmp_path)
    make, calls = _maker()
    cache.put("s0", KEYS[0], _table(0))
    cache.put("memo", "m", {"x": np.zeros(3)})
    cache.put_columns(("k",), IDS, np.ones((1, len(IDS), 3)), (CHAIN,))
    cache.routine(*SIGNATURE, "t", make=make)
    cache.trial("t", lambda: 1)
    cache.clear()
    assert cache.nbytes == 0 and cache.peek("memo", "m") is None
    assert cache.peek("s0", KEYS[0]) == "disk"
    assert cache.columns(("k",), IDS, (CHAIN,))[1] is None
    cache.routine(*SIGNATURE, "t", make=make)
    assert len(calls) == 2 and cache.trial("t", lambda: 2) == 2
    cache.clear(disk=True)
    assert cache.peek("s0", KEYS[0]) is None and _disk_files(tmp_path) == []


def test_default_cache_is_one_object_in_the_test_directory():
    cache = default_cache()
    assert default_cache() is cache
    assert cache.directory == Path(os.environ["SYNCEM_CACHE"])
    cache.put("memo", "left over", {"x": np.zeros(1)})


def test_default_cache_is_fresh_in_each_test():
    assert default_cache().peek("memo", "left over") is None
