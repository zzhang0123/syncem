"""Memory of the one-node store and of compiled jet programs at full size (design/0.8/SPEC.md decisions 167 to 169).

Every test here is slow. The per-node cases are those of the memory audit: ``B`` on 96 nodes given ``eta`` on
12 or 2 nodes, 6.3 MB per one-node table. Before step 5b-3 the ``cache=False`` build of 96 by 12 on one cell
kept its 1152 tables in a dict and peaked at 7.24 GB, and the cached build of 96 by 2 over 4 cells made 1536
table builds with 1344 rebuilds (fcfe5cf, 1-minute load 18 to 25, loaded). The counts asserted are those of
decision 168: ``R = floor(mem_bytes / table)`` tables stay resident and the others are built at every read.

The trace bound and the resident growth over builds with fresh tokens are those of decision 169. The growth is
measured and printed (run with ``-s``), and only the bound on programs and routine slots is asserted.
"""

from __future__ import annotations

import gc
import os
import subprocess
import types

import numpy as np
import pytest

from syncem.engine import cache as cache_module
from syncem.engine._cache_store import nbytes_of
from syncem.engine._stage_memo import s1_arrays
from syncem.engine.average import AveragedKernel, NodeRule, uniform_rule
from syncem.engine.basis import build_from_averaged
from syncem.engine.cache import DEFAULT_MEM_BYTES, ROUTINE_SLOTS, default_cache
from syncem.engine.field_average import FieldAverage
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from tests.v080.test_v080_cache import K
from tests.v080.test_v080_stack import small_channels

pytestmark = pytest.mark.slow

ONE = nbytes_of(s1_arrays(FieldAverage.lookups([5.0], [1.0])))
R = DEFAULT_MEM_BYTES // ONE


def per_node(n_b, n_eta, cells):
    b = uniform_rule(lambda p: 4e-6 * (1 + p["eta"]), 6e-6, n_b, parents=("eta",))
    avg = AveragedKernel(K, [("eta", uniform_rule(0.0, 0.5, n_eta)), ("B", b)])
    return avg, Lattice(gamma=Cells.log(300, 3e6, cells, order=2))


def build(case, **kwargs):
    avg, lattice = case
    ch = small_channels(3, 8)
    return build_from_averaged(avg, ch, lattice, azimuth_factor=0.4, **kwargs)


def node_counts(basis):
    ev = dict(basis.record.events)
    keys = ("one-node tables built", "one-node rebuilds", "one-node tables not kept")
    return tuple(ev[k] for k in keys)


def watch(monkeypatch):
    """The largest charge of the store of each ``NodeStore.get``, with its bound."""
    seen = []
    original = cache_module.NodeStore.get

    def watched(self, key, make):
        out = original(self, key, make)
        seen.append((self._cache.nbytes, self._cache.mem_bytes))
        return out

    monkeypatch.setattr(cache_module.NodeStore, "get", watched)
    return seen


def test_96_by_12_without_the_cache_stays_under_the_bound(monkeypatch):
    seen = watch(monkeypatch)
    basis = build(per_node(96, 12, 1), cache=False)
    m = 96 * 12
    assert node_counts(basis) == (m, 0, m - R)
    assert max(n for n, _ in seen) <= DEFAULT_MEM_BYTES == seen[0][1]
    print(f"\n96x12 cache=False: peak charge {max(n for n, _ in seen)} bytes, R = {R}")


def test_96_by_2_over_4_cells_makes_the_predicted_builds(monkeypatch):
    seen = watch(monkeypatch)
    basis = build(per_node(96, 2, 4))
    m, a = 96 * 2, 4
    assert node_counts(basis) == (m + (a - 1) * (m - R), (a - 1) * (m - R), a * (m - R))
    assert (m + (a - 1) * (m - R), (a - 1) * (m - R)) == (294, 102)
    assert max(n for n, _ in seen) <= DEFAULT_MEM_BYTES
    assert any(n.startswith("one-node tables: ") for n in basis.record.notes)


# ------------------------------------------------------------------ compiled programs


def rss_mb():
    out = subprocess.run(
        ["ps", "-o", "rss=", "-p", str(os.getpid())], capture_output=True, text=True
    )
    return int(out.stdout.strip()) / 1e3


def fresh_rule(k):
    """An averaged kernel whose rule is a new function, so the jet retraces."""
    scale = 1.0 + 1e-3 * k
    rule = NodeRule(
        lambda p, s=scale: (np.array([0.1 * s, 0.4]), np.array([0.5, 0.5])),
        parents=(),
    )
    return AveragedKernel(K, [("eta", rule)], fixed={"B": 5e-6})


_SKIP = (
    type,
    types.ModuleType,
    types.FunctionType,
    types.BuiltinFunctionType,
    types.MethodType,
)


def compiled_reached(routine):
    """The ``jax.stages.Compiled`` objects reachable from ``routine``, and the traces of jitted functions.

    Functions, methods, modules and classes are not followed, so module globals are not walked.
    """
    import jax.stages  # noqa: PLC0415

    count, seen, todo = 0, set(), [(routine, 0)]
    while todo and len(seen) < 200_000:
        obj, depth = todo.pop()
        if id(obj) in seen or depth > 12:
            continue
        seen.add(id(obj))
        if isinstance(obj, jax.stages.Compiled):
            count += 1
            continue
        size = getattr(obj, "_cache_size", None)
        if callable(size):
            count += int(size())
            continue
        todo.extend(
            (x, depth + 1) for x in gc.get_referents(obj) if not isinstance(x, _SKIP)
        )
    return count


def test_forty_direct_calls_with_fresh_rules_keep_at_most_the_trace_slots():
    from syncem.engine._jet_routine import TRACE_SLOTS  # noqa: PLC0415
    from syncem.engine.provider_jet import jet_columns  # noqa: PLC0415

    ch = small_channels(2, 8)
    start = rss_mb()
    for k in range(40):
        jet_columns(fresh_rule(k), ch, {"gamma": ("log", 3e3, 0.5)}, [(0,), (1,), (2,)])
    slots = [value for value, _ in default_cache()._routines._entries.values()]
    held = [(r.programs, compiled_reached(r)) for r in slots]
    print(f"\n40 direct calls: programs {held}, RSS {start:.0f} -> {rss_mb():.0f} MB")
    assert len(slots) == 1 and 1 <= held[0][0] == held[0][1] <= TRACE_SLOTS


def test_resident_growth_over_forty_uncached_builds_is_reported():
    ch = small_channels(2, 8)
    lattice = Lattice(gamma=Cells.log(300, 3e5, 1, order=2))
    rss = []
    for k in range(40):
        build_from_averaged(
            fresh_rule(k), ch, lattice, routes="jet", azimuth_factor=0.4, cache=False
        )
        rss.append(rss_mb())
    growth = (rss[-1] - rss[4]) / 35
    print(
        f"\n40 cache=False jet builds: RSS {rss[0]:.0f} -> {rss[-1]:.0f} MB, {growth:.1f} MB per build"
    )
    assert len(default_cache()._routines._entries) == 0 <= ROUTINE_SLOTS
    default_cache().clear(jax=True)
