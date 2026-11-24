"""The process cache of the engine (design/0.8/SPEC.md Sec. 8.5, decisions 160 to 162 and 167 to 169).

LABEL ``[extension]``. One ``Cache`` holds the basis memo (stage ``memo``), the S0 and S1 tables
(stages ``s0`` and ``s1``), the S2 columns, the jet routines, the trial LRU and the one-node tables
of each build. ``default_cache()`` returns the process ``Cache``, the only module-level state of the
engine apart from ``lru_cache`` on ``derivative_polynomials`` and ``_gauss`` in ``engine/primitive.py``.

The memo, S0, S1 and S2 entries and the one-node tables share one least-recently-used bound of
``mem_bytes`` bytes, 1000000000 (1GB) by default (``SYNCEM_CACHE_MEM`` in the size syntax of
``SYNCEM_CACHE_MAX``, decision 162). An S0, S1 or one-node table counts the bytes of its array leaves.
A memo entry and an S2 entry also count the ``sys.getsizeof`` of their containers, strings and objects
and a store slot, since an S2 entry holds a few values and many small objects. An entry above the bound
is not kept and evicts nothing. ``get`` refreshes the
recency of a hit and ``peek`` does not. S0 and S1 tables also go to disk through
``engine/_disk.py`` (decision 161), at most ``max_bytes`` (``SYNCEM_CACHE_MAX``, 2 GB by default, 0
disables the disk). A failed write is skipped and counted. The memo, the S2 columns, the routines,
the trial LRU and the one-node tables stay in memory. Stored arrays are read-only copies.

An S2 entry belongs to one context key (cell, block, channel). It holds one column per column id
``(Legendre degrees, exponent)`` and the rule chain ``(d_0, ..., d_N)`` it was computed with. The
total degree of a column is the sum of its exponent. A request reuses a column when the id is stored
and the two chains agree up to the column's total degree. A put whose chain contradicts the stored
chain drops the stored columns above the last agreeing degree.

``routine(names, charts, exponents, token, make=...)`` keeps the last 64 jet routines (the bound of
the former ``lru_cache`` of ``provider_jet._routine``), keyed by the maker, the signature and the
token. A fresh token per build gives a fresh routine. ``trial`` keeps 8 entries by count and never
writes to disk. ``node_store()`` opens the one-node store of one build. Its entries share the byte
bound. A new table is kept only when it fits without evicting a one-node table of the same build, and
otherwise it is returned and not kept (decision 168). A table read again after it was evicted or not
kept is rebuilt, the rebuilds are counted, and closing the store removes its entries. ``Cache.local()``
is a cache of one uncached build: its own byte bound, no disk and no directory (decision 167). Compiled
jet programs lie outside the byte bound (decision 169).

Units: none of its own. The cache stores what the providers give, in their units. Array shapes: an
S0 or S1 table is a mapping of names to arrays of any shape. ``columns`` returns a found mask of
shape ``(n_rows, n_wanted)`` and values of shape ``(n_rows, n_wanted, *value_shape)``. Not
certified: the stored values, which the providers compute and their tests check. The cache returns
them bit for bit and checks no numbers.
"""

from __future__ import annotations

import itertools
import threading
from collections.abc import Callable, Hashable, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from pathlib import Path
from typing import Any

import jax as _jax
import numpy as np

from . import _disk
from ._cache_store import (
    MISSING,
    NodeStore,
    Store,
    agreement,
    bound,
    check_columns,
    column_id,
    degree,
    freeze,
    merge,
    columns_bytes,
    memo_bytes,
    nbytes_of,
    shallow,
    table,
)

__all__ = [
    "DEFAULT_MEM_BYTES",
    "EVENTS",
    "MEM_ENV",
    "ROUTINE_SLOTS",
    "STAGES",
    "TRIAL_SLOTS",
    "Cache",
    "NodeStore",
    "default_cache",
]

DEFAULT_MEM_BYTES = 1_000_000_000
MEM_ENV = "SYNCEM_CACHE_MEM"
STAGES = ("s0", "s1", "memo")
ROUTINE_SLOTS = 64
TRIAL_SLOTS = 8
_TRIAL_S1 = ContextVar("syncem_trial_s1", default=False)
EVENTS = (
    "hits",
    "misses",
    "disk_hits",
    "disk_writes",
    "disk_write_errors",
    "evictions",
    "rebuilds",
    "routines",
    "columns_found",
    "columns_missing",
)


# The cache --------------------------------------------------------------------------------------


@contextmanager
def trial_tables():
    """Keep variable S1 tables in the trial LRU and leave fixed stages available."""
    token = _TRIAL_S1.set(True)
    try:
        yield
    finally:
        _TRIAL_S1.reset(token)


class Cache:
    """The stage cache of one process (decisions 160 to 162). See the module docstring.

    ``directory`` defaults to ``_disk.cache_dir()``, ``max_bytes`` to ``SYNCEM_CACHE_MAX`` or 2 GB and
    ``mem_bytes`` to ``SYNCEM_CACHE_MEM`` or 1GB. Not certified: the stored values.
    """

    def __init__(
        self,
        *,
        directory: Path | str | None = None,
        max_bytes: int | None = None,
        mem_bytes: int | None = None,
    ) -> None:
        root = Path(directory) if directory is not None else _disk.cache_dir()
        self._setup(root, bound(max_bytes, _disk.disk_limit), mem_bytes)

    @classmethod
    def local(cls, mem_bytes: int | None = None) -> Cache:
        """The cache of one uncached build: ``mem_bytes`` or ``SYNCEM_CACHE_MEM``, no disk, no directory."""
        local = cls.__new__(cls)
        local._setup(None, 0, mem_bytes)
        return local

    def _setup(
        self, directory: Path | None, max_bytes: int, mem_bytes: int | None
    ) -> None:
        self._directory, self._max_bytes = directory, max_bytes
        self._mem_bytes = bound(
            mem_bytes, lambda: _disk.env_bytes(MEM_ENV, DEFAULT_MEM_BYTES)
        )
        self._store = Store(self._mem_bytes)
        self._routines = Store(ROUTINE_SLOTS)
        self._trials = Store(TRIAL_SLOTS)
        self._events = dict.fromkeys(EVENTS, 0)
        self._lock = threading.RLock()
        self._builds = itertools.count()

    directory = property(lambda self: self._directory)
    max_bytes = property(lambda self: self._max_bytes)
    mem_bytes = property(lambda self: self._mem_bytes)
    disk_enabled = property(lambda self: self._max_bytes > 0)
    nbytes = property(lambda self: self._store.nbytes)
    events = property(
        lambda self: dict(self._events), doc="A copy of the event counts."
    )

    @property
    def settings(self) -> tuple[tuple[str, Any], ...]:
        """The bounds and the directory, for ``BuildRecord.defaults``."""
        return (
            (
                "cache_directory",
                None if self._directory is None else str(self._directory),
            ),
            ("cache_max_bytes", self._max_bytes),
            ("cache_mem_bytes", self._mem_bytes),
        )

    def _count(self, name: str, n: int = 1) -> None:
        with self._lock:
            self._events[name] += n

    def _path(self, stage: str, key: str) -> Path | None:
        if stage not in STAGES:
            raise ValueError(f"stage {stage!r}: the stages are {', '.join(STAGES)}")
        if not isinstance(key, str):
            raise TypeError(f"key {key!r}: a cache key is a string")
        if stage in _disk.STAGES:
            path = _disk.entry_path(self._directory or Path(), stage, key)
            return path if self._directory is not None else None
        return None

    def _keep(self, key: Hashable, value: Any) -> None:
        size = memo_bytes(value) if key[0] == "memo" else nbytes_of(value)
        with self._lock:
            self._count("evictions", self._store.put(key, value, size))

    def peek(self, stage: str, key: str) -> str | None:
        """``"memory"``, ``"disk"`` or ``None``. Neither the recency nor a modification time changes."""
        path = self._path(stage, key)
        if stage == "s1" and _TRIAL_S1.get():
            with self._lock:
                return "memory" if (stage, key) in self._trials else None
        with self._lock:
            if (stage, key) in self._store:
                return "memory"
        if path is not None and self.disk_enabled:
            if path.is_file() and path.with_suffix(".json").is_file():
                return "disk"
        return None

    def get(self, stage: str, key: str) -> Any:
        """The stored value or ``None``. A disk hit is kept in memory as well."""
        path = self._path(stage, key)
        if stage == "s1" and _TRIAL_S1.get():
            with self._lock:
                value = self._trials.get((stage, key))
                self._count("misses" if value is MISSING else "hits")
            return None if value is MISSING else shallow(value)
        with self._lock:
            value = self._store.get((stage, key))
            if value is not MISSING:
                self._count("hits")
                return shallow(value)
        entry = _disk.read_entry(path) if path and self.disk_enabled else None
        if entry is None:
            self._count("misses")
            return None
        arrays = table(entry["arrays"])
        self._count("disk_hits")
        self._keep((stage, key), arrays)
        return dict(arrays)

    def put(self, stage: str, key: str, value: Any, *, description: Any = None) -> Any:
        """Store ``value`` and return the read-only copy kept. An S0 or S1 table is written to disk as well.

        ``description`` is the strict-JSON description of a disk entry. A description that is not strict
        JSON raises ``TypeError`` or ``ValueError`` before anything is kept.
        """
        path = self._path(stage, key)
        if stage == "s1" and _TRIAL_S1.get():
            value = table(value)
            if nbytes_of(value) <= self._mem_bytes // TRIAL_SLOTS:
                with self._lock:
                    self._trials.put((stage, key), value, 1)
            return shallow(value)
        if path is not None:
            description = _disk.plain_description(description)
        value = table(value) if stage in _disk.STAGES else freeze(value)
        self._keep((stage, key), value)
        if path is None or not self.disk_enabled:
            return value
        try:
            _disk.write_entry(path, value, description)
            _disk.evict(self._directory, self._max_bytes)
        except OSError:
            self._count("disk_write_errors")
            return value
        self._count("disk_writes")
        return value

    def columns(
        self, context_keys: Sequence[str], wanted: Sequence, chains: Sequence
    ) -> tuple[np.ndarray, np.ndarray | None]:
        """The stored S2 columns ``wanted`` of each context key, valid for the row's chain.

        Returns the found mask ``(n_rows, n_wanted)`` and the values ``(n_rows, n_wanted, ...)``,
        zero where not found, or ``None`` when nothing is found.
        """
        ids = tuple(column_id(c) for c in wanted)
        found = np.zeros((len(context_keys), len(ids)), dtype=bool)
        values = None
        for r, (key, chain) in enumerate(zip(context_keys, chains, strict=True)):
            with self._lock:
                entry = self._store.get(("s2", key))
            if entry is MISSING:
                continue
            agree = agreement(entry.chain, chain)
            for c, i in enumerate(ids):
                k = entry.index.get(i)
                if k is None or degree(i) >= agree:
                    continue
                if values is None:
                    shape = (len(context_keys), len(ids), *entry.values.shape[1:])
                    values = np.zeros(shape, dtype=entry.values.dtype)
                elif (values.shape[2:], values.dtype) != (
                    entry.values.shape[1:],
                    entry.values.dtype,
                ):
                    raise ValueError(
                        "S2 columns of one request need one value shape and dtype"
                    )
                values[r, c] = entry.values[k]
                found[r, c] = True
        self._count("columns_found", int(found.sum()))
        self._count("columns_missing", int(found.size - found.sum()))
        return found, values

    def put_columns(
        self, context_keys: Sequence[str], ids: Sequence, values: Any, chains: Sequence
    ) -> None:
        """Store S2 columns: ``values`` has shape ``(n_rows, n_ids, ...)`` and ``chains`` one chain per row."""
        ids = tuple(column_id(c) for c in ids)
        values = np.asarray(values)
        check_columns(context_keys, ids, values, chains)
        for r, key in enumerate(context_keys):
            chain = tuple(chains[r])
            with self._lock:
                old = self._store.value(("s2", key))
                entry = merge(None if old is MISSING else old, ids, values[r], chain)
                self._count(
                    "evictions",
                    self._store.put(("s2", key), entry, columns_bytes(entry, key)),
                )

    def routine(
        self,
        names: tuple,
        charts: tuple,
        exponents: tuple,
        token: Hashable,
        *,
        make: Callable[[tuple, tuple, tuple], Any],
    ) -> Any:
        """The jet routine ``make(names, charts, exponents)`` for ``token``, built once per key."""
        key = (make, names, charts, exponents, token)
        with self._lock:
            found = self._routines.get(key)
            if found is not MISSING:
                return found
            built = make(names, charts, exponents)
            self._routines.put(key, built, 1)
            self._count("routines")
            return built

    def trial(self, key: Hashable, compute: Callable[[], Any]) -> Any:
        """The trial entry of ``key``, computed on a miss. The last 8 keys are kept, never on disk."""
        with self._lock:
            found = self._trials.get(key)
            if found is not MISSING:
                return shallow(found)
        value = freeze(compute())
        with self._lock:
            self._trials.put(key, value, 1)
        return shallow(value)

    def node_store(self) -> NodeStore:
        """A one-node store for one build, to be closed (or used as a context manager) at its end."""
        return NodeStore(self, next(self._builds))

    def clear(self, *, disk: bool = False, jax: bool = False) -> None:
        """Empty the memory stages, the routines and the trial LRU. ``disk=True`` also removes the disk entries.

        Compiled jet programs lie outside the byte bound, at about 16 MB of resident memory per retrace
        (decision 169). Dropping a routine releases its programs but does not return that memory, and
        ``jax=True`` also calls ``jax.clear_caches()``, which clears every JAX cache of the process and
        returned about 20% of it in the memory audit.
        """
        with self._lock:
            self._store.clear()
            self._routines.clear()
            self._trials.clear()
        if disk and self._directory is not None:
            _disk.evict(self._directory, 0)
        if jax:
            _jax.clear_caches()


_DEFAULT: Cache | None = None


def default_cache() -> Cache:
    """The process ``Cache``, created at the first call from the environment of that call."""
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = Cache()
    return _DEFAULT
