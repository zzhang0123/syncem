"""Private parts of ``syncem.engine.cache`` (design/0.8/SPEC.md decisions 160 to 162).

``Store`` is a least-recently-used store bounded by a total size, and a count bound gives each entry
size 1. ``freeze`` copies the NumPy leaves of a value as read-only arrays, ``shallow`` gives new
containers around the same leaves and ``nbytes_of`` sums the bytes of the array leaves, the charge of an
S0, S1 or one-node table. ``memo_bytes`` and ``columns_bytes`` charge a memo entry and an S2 entry their
arrays, the ``sys.getsizeof`` of their containers, strings and objects, and the store slot. ``Columns``
is one S2 entry. ``merge`` adds columns to an entry and drops the stored columns that a new rule
chain contradicts. A column id is ``(Legendre degrees, exponent)`` and its total degree is the sum
of the exponent.
"""

from __future__ import annotations

import dataclasses
import sys
import types
from collections import OrderedDict
from collections.abc import Callable, Hashable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import jax
import numpy as np

MISSING = object()


def frozen(leaf: Any) -> Any:
    """A read-only copy of a NumPy array. Any other leaf is returned as it is."""
    if isinstance(leaf, np.ndarray):
        copy = np.array(leaf, copy=True)
        copy.setflags(write=False)
        return copy
    return leaf


def freeze(value: Any) -> Any:
    """``value`` with every NumPy leaf replaced by a read-only copy."""
    return jax.tree_util.tree_map(frozen, value)


def shallow(value: Any) -> Any:
    """New containers around the same leaves, so a caller cannot change a stored container."""
    return jax.tree_util.tree_map(lambda leaf: leaf, value)


def nbytes_of(value: Any) -> int:
    """The bytes of the array leaves of ``value``. Other leaves count 0."""
    return sum(int(getattr(x, "nbytes", 0)) for x in jax.tree_util.tree_leaves(value))


_SHARED = (
    types.ModuleType,
    types.FunctionType,
    type,
)  # program objects, not entry data
SLOT_BYTES = 200  # the store's own entry: the key tuple, the (value, size) tuple and the dict slot


def deep_bytes(value: Any, seen: set[int] | None = None) -> int:
    """Array bytes plus ``sys.getsizeof`` of every container, string and object reached from ``value``.

    Containers are tuples, lists, sets, dicts, dataclass fields and instance ``__dict__``. An object
    reached twice counts once. Modules, functions and classes count 0.
    """
    seen = set() if seen is None else seen
    if id(value) in seen or isinstance(value, _SHARED):
        return 0
    seen.add(id(value))
    if isinstance(value, (np.ndarray, jax.Array)):
        return int(value.nbytes)
    size = sys.getsizeof(value)
    if isinstance(value, Mapping):
        items = [x for kv in value.items() for x in kv]
    elif isinstance(value, (tuple, list, set, frozenset)):
        items = list(value)
    elif dataclasses.is_dataclass(value):
        items = [getattr(value, f.name) for f in dataclasses.fields(value)]
    else:
        items = list(getattr(value, "__dict__", {}).values())
    return size + sum(deep_bytes(x, seen) for x in items)


def memo_bytes(value: Any) -> int:
    """The charge of a memo entry: ``deep_bytes`` and the store slot."""
    return deep_bytes(value) + SLOT_BYTES


def table(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    """An S0 or S1 table as a dict of read-only contiguous copies."""
    if not isinstance(arrays, Mapping):
        raise TypeError("an S0 or S1 table is a mapping of names to arrays")
    return {str(name): frozen(np.ascontiguousarray(a)) for name, a in arrays.items()}


def bound(value: int | None, read: Callable[[], int]) -> int:
    """``value``, or ``read()`` when it is ``None``, checked as a non-negative integer."""
    size = read() if value is None else value
    if isinstance(size, bool) or not isinstance(size, int) or size < 0:
        raise ValueError(f"bound {size!r}: give a non-negative number of bytes")
    return size


class Store:
    """A least-recently-used store bounded by a total size. A count bound gives each entry size 1."""

    def __init__(self, bound: int) -> None:
        self.bound = bound
        self.nbytes = 0
        self._entries: OrderedDict[Hashable, tuple[Any, int]] = OrderedDict()

    def get(self, key: Hashable) -> Any:
        if key not in self._entries:
            return MISSING
        self._entries.move_to_end(key)
        return self._entries[key][0]

    def value(self, key: Hashable) -> Any:
        """The value without a change of recency."""
        return self._entries.get(key, (MISSING, 0))[0]

    def __contains__(self, key: Hashable) -> bool:
        return key in self._entries

    def pop(self, key: Hashable) -> None:
        if key in self._entries:
            self.nbytes -= self._entries.pop(key)[1]

    def put(self, key: Hashable, value: Any, size: int) -> int:
        """Store ``value`` and return the number of entries evicted."""
        self.pop(key)
        if size > self.bound:
            return 0
        evicted = 0
        while self.nbytes + size > self.bound:
            self.nbytes -= self._entries.popitem(last=False)[1][1]
            evicted += 1
        self._entries[key] = (value, size)
        self.nbytes += size
        return evicted

    def clear(self) -> None:
        self._entries.clear()
        self.nbytes = 0


# S2 columns -------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Columns:
    """One S2 entry: column ids by row of ``values`` (read-only), and the chain they agree with."""

    index: Mapping[tuple, int]
    values: np.ndarray
    chain: tuple[str, ...]


def columns_bytes(entry: Columns, key: str) -> int:
    """The charge of an S2 entry: its values, index, chain, context key and store slot."""
    return deep_bytes(entry) + sys.getsizeof(key) + SLOT_BYTES


def column_id(column: Any) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """A column id ``(Legendre degrees, exponent)`` as two tuples of ints."""
    degrees, exponent = column
    return tuple(int(d) for d in degrees), tuple(int(e) for e in exponent)


def degree(column: tuple) -> int:
    """The total degree of a column, the sum of its exponent."""
    return sum(column[1])


def agreement(a: Sequence[str], b: Sequence[str]) -> int:
    """The number of leading entries in which the chains agree."""
    count = 0
    for x, y in zip(a, b):
        if x != y:
            break
        count += 1
    return count


def merge(old: Columns | None, ids: tuple, row: np.ndarray, chain: tuple) -> Columns:
    """The entry ``old`` with the columns ``ids`` of ``row`` (shape ``(n_ids, ...)``) for ``chain``.

    A new chain that is a prefix of the stored chain keeps every stored column and the stored chain.
    Otherwise the stored columns of total degree at or above the first disagreement are dropped.
    """
    kept: dict[tuple, np.ndarray] = {}
    new = set(ids)
    if old is not None:
        agree = agreement(old.chain, chain)
        if agree == len(chain) and len(old.chain) >= len(chain):
            chain, limit = old.chain, len(old.chain)
        else:
            limit = agree
        kept = {
            i: old.values[k]
            for i, k in old.index.items()
            if degree(i) < limit and i not in new
        }
        if kept and (old.values.shape[1:], old.values.dtype) != (
            row.shape[1:],
            row.dtype,
        ):
            raise ValueError(
                "S2 columns of one context key need one value shape and dtype"
            )
    order = (*kept, *ids)
    stacked = np.concatenate([np.stack(list(kept.values())), row]) if kept else row
    return Columns(dict(zip(order, range(len(order)))), frozen(stacked), chain)


def check_columns(
    keys: Sequence, ids: tuple, values: np.ndarray, chains: Sequence
) -> None:
    """Shapes, distinct keys and ids, and chains long enough for the highest column degree."""
    if values.shape[:2] != (len(keys), len(ids)) or len(chains) != len(keys):
        raise ValueError("values need shape (n_rows, n_ids, ...) and one chain per row")
    if len(set(keys)) != len(keys) or len(set(ids)) != len(ids):
        raise ValueError("context keys and column ids of one put must be distinct")
    top = max((degree(i) for i in ids), default=-1)
    if any(len(c) <= top for c in chains):
        raise ValueError(
            f"a rule chain covers the degrees 0 to len(chain) - 1, and a column has degree {top}"
        )
