"""Step 5b-2 of design/0.8/SPEC.md: the S2 columns of the process ``Cache`` (decisions 159 and 160).

An S2 entry belongs to one context key and holds one column per ``(Legendre degrees, exponent)``
with the rule chain ``(d_0, ..., d_N)`` it was computed with. The dispatcher between a stored column
and a computed one is tested on both sides: a chain that first differs at degree ``k`` serves the
columns of total degree below ``k`` and no column at or above it, for every ``k`` from 0 to the chain
length. Every comparison is an exact identity of bytes or booleans, so no tolerance enters.
"""

from __future__ import annotations

import numpy as np
import pytest

from syncem.engine._cache_store import columns_bytes
from syncem.engine.cache import Cache


def _charge(cache: Cache, key: str) -> int:
    """The charge of the S2 entry of ``key``: values, index, chain, key and store slot (decision 162)."""
    return columns_bytes(cache._store.value(("s2", key)), key)


def _memory(directory, mem_bytes: int) -> Cache:
    return Cache(directory=directory, max_bytes=0, mem_bytes=mem_bytes)


def _disk_files(directory) -> list[str]:
    return sorted(p.name for p in directory.rglob("*") if p.is_file())


IDS = tuple(((0,), (n,)) for n in range(5))  # Legendre degree 0, Taylor exponent n
CHAIN = tuple(f"d{n}" for n in range(5))


def _values(rows: int, ids=IDS, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).standard_normal((rows, len(ids), 3))


def test_columns_found_on_the_stored_context_keys_only(tmp_path):
    cache = _memory(tmp_path, mem_bytes=10**6)
    values = _values(2)
    cache.put_columns(("k0", "k1"), IDS, values, (CHAIN, CHAIN))
    found, got = cache.columns(("k1", "k2", "k0"), IDS, (CHAIN,) * 3)
    assert found.tolist() == [[True] * 5, [False] * 5, [True] * 5]
    assert got[0].tobytes() == values[1].tobytes()
    assert got[2].tobytes() == values[0].tobytes()
    assert got[1].tobytes() == np.zeros((5, 3)).tobytes()
    assert cache.nbytes == _charge(cache, "k0") + _charge(cache, "k1")
    assert _charge(cache, "k0") > values[0].nbytes


def test_columns_of_nothing_stored_are_none(tmp_path):
    found, got = _memory(tmp_path, 10**6).columns(("k0",), IDS, (CHAIN,))
    assert found.tolist() == [[False] * 5] and got is None


@pytest.mark.parametrize("differ", range(6))
def test_chains_agree_up_to_the_total_degree(tmp_path, differ):
    """A chain that first differs at degree ``differ`` serves the columns of total degree below it."""
    cache = _memory(tmp_path, mem_bytes=10**6)
    cache.put_columns(("k",), IDS, _values(1), (CHAIN,))
    other = tuple(d if n < differ else d + "x" for n, d in enumerate(CHAIN))
    found, _ = cache.columns(("k",), IDS, (other,))
    assert found[0].tolist() == [n < differ for n in range(5)]


def test_total_degree_sums_the_exponent_and_skips_legendre_degrees(tmp_path):
    ids = (((4,), (0, 0)), ((0,), (1, 1)), ((2,), (2, 0)), ((0,), (0, 1)))
    cache = _memory(tmp_path, mem_bytes=10**6)
    cache.put_columns(("k",), ids, _values(1, ids), (CHAIN[:3],))
    found, _ = cache.columns(("k",), ids, (("d0", "d1", "dX"),))
    assert found[0].tolist() == [True, False, False, True]


def test_a_shorter_request_chain_cannot_serve_higher_degrees(tmp_path):
    cache = _memory(tmp_path, mem_bytes=10**6)
    cache.put_columns(("k",), IDS, _values(1), (CHAIN,))
    found, _ = cache.columns(("k",), IDS, (CHAIN[:3],))
    assert found[0].tolist() == [True, True, True, False, False]


def test_order_two_then_four_adds_the_missing_columns(tmp_path):
    cache = _memory(tmp_path, mem_bytes=10**6)
    full = _values(1)
    cache.put_columns(("k",), IDS[:3], full[:, :3], (CHAIN[:3],))
    found, _ = cache.columns(("k",), IDS, (CHAIN,))
    assert found[0].tolist() == [True, True, True, False, False]
    cache.put_columns(("k",), IDS[3:], full[:, 3:], (CHAIN,))
    found, got = cache.columns(("k",), IDS, (CHAIN,))
    assert found.all() and got.tobytes() == full.tobytes()
    found, got = cache.columns(("k",), IDS[:3], (CHAIN[:3],))
    assert found.all() and got.tobytes() == full[:, :3].tobytes()


def test_a_put_with_another_chain_drops_the_columns_it_contradicts(tmp_path):
    cache = _memory(tmp_path, mem_bytes=10**6)
    cache.put_columns(("k",), IDS, _values(1), (CHAIN,))
    other = CHAIN[:2] + ("e2", "e3", "e4")
    new = _values(1, IDS[2:3], seed=5)
    cache.put_columns(("k",), IDS[2:3], new, (other,))
    found, got = cache.columns(("k",), IDS, (other,))
    assert found[0].tolist() == [True, True, True, False, False]
    assert got[0, 2].tobytes() == new[0, 0].tobytes()
    assert not cache.columns(("k",), IDS, (CHAIN,))[0][0, 2:].any()


def test_put_columns_refuses_a_chain_shorter_than_a_column_degree(tmp_path):
    with pytest.raises(ValueError, match="chain"):
        _memory(tmp_path, 10**6).put_columns(("k",), IDS, _values(1), (CHAIN[:2],))


def test_s2_entries_share_the_byte_bound_and_never_go_to_disk(tmp_path):
    values = _values(1)
    probe = _memory(tmp_path, mem_bytes=10**6)
    probe.put_columns(("k0",), IDS, values, (CHAIN,))
    cache = Cache(directory=tmp_path, mem_bytes=_charge(probe, "k0"))
    cache.put_columns(("k0",), IDS, values, (CHAIN,))
    cache.put_columns(("k1",), IDS, values, (CHAIN,))
    assert cache.columns(("k0",), IDS, (CHAIN,))[1] is None
    assert cache.columns(("k1",), IDS, (CHAIN,))[0].all()
    assert _disk_files(tmp_path) == []


def test_a_put_with_another_dtype_or_value_shape_raises(tmp_path):
    cache = _memory(tmp_path, mem_bytes=10**6)
    cache.put_columns(("k",), IDS[:1], _values(1, IDS[:1]), (CHAIN,))
    with pytest.raises(ValueError, match="value shape and dtype"):
        cache.put_columns(("k",), IDS[1:2], _values(1, IDS[1:2]).astype("f4"), (CHAIN,))
    with pytest.raises(ValueError, match="value shape and dtype"):
        cache.put_columns(("k",), IDS[1:2], np.zeros((1, 1, 2)), (CHAIN,))


def test_a_request_over_entries_of_two_value_shapes_raises(tmp_path):
    cache = _memory(tmp_path, mem_bytes=10**6)
    cache.put_columns(("k0",), IDS[:1], np.zeros((1, 1, 3)), (CHAIN,))
    cache.put_columns(("k1",), IDS[:1], np.zeros((1, 1, 2)), (CHAIN,))
    assert cache.columns(("k0",), IDS[:1], (CHAIN,))[0].all()
    with pytest.raises(ValueError, match="one request"):
        cache.columns(("k0", "k1"), IDS[:1], (CHAIN, CHAIN))


def test_a_put_that_replaces_a_column_keeps_one_copy(tmp_path):
    cache = _memory(tmp_path, mem_bytes=10**6)
    cache.put_columns(("k",), IDS, _values(1), (CHAIN,))
    new = _values(1, IDS[:1], seed=9)
    cache.put_columns(("k",), IDS[:1], new, (CHAIN,))
    found, got = cache.columns(("k",), IDS[:1], (CHAIN,))
    assert found.all() and got.tobytes() == new.tobytes()
    assert cache._store.value(("s2", "k")).values.shape[0] == len(IDS)
    assert cache.nbytes == _charge(cache, "k")
