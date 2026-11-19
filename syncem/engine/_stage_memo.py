"""The S0 and S1 tables of one basis build through the process cache (design/0.8/SPEC.md decisions 158 to 162).

LABEL ``eq: directional continuum``. ``StageMemo(cache)`` is the memo that the stack walk of a cached build
reads. An S0 table is keyed by ``_keys.s0_key`` over its quantised span and goes to memory and disk. A
cached-law S1 table is keyed by ``_keys.s1_key`` and goes to memory and disk. A one-node S1 table has the same
key and lives in the build's ``NodeStore``, under the byte bound and never on disk, and a table evicted during
the build is rebuilt and counted. The tables are stored as mappings of arrays (``s0_arrays``, ``s1_arrays``)
and rebuilt from them as objects, bit for bit. ``take_reads`` returns the S1 keys read since its last call,
which are the stage of the S2 context keys of a stack cell. A build without the cache passes a plain dict as
the memo, as in step 5a. ``primitive_table`` and ``s1_table`` read either memo.

The S1 key of the quadrature path has no S0 table. Its S0 part is ``QUADRATURE_S0``, a digest of the format, the
source of ``primitive.py`` (the quadrature of ``scaled_log_derivatives``), the environment, the node count and
the cut-off.

Units: field nodes in uG, spans in e-folds of ``ln x``. Shapes: those of ``PrimitiveTable`` and
``FieldAverage``. Not certified: the tables, which the step-5 tests compare with the quadrature. The memo
returns them unchanged.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..base._fingerprint import digest
from ._stack_checks import StackRefused
from ._keys import ENVIRONMENT, FMT, S0_SOURCE, s0_key, s1_key
from .cache import Cache
from .field_average import TABLE_ORDER, FieldAverage
from .primitive import DEFAULT_CUTOFF, DEFAULT_NODES, PrimitiveTable, quantised_span

__all__ = [
    "PRIMITIVE_STEP",
    "QUADRATURE_S0",
    "StageMemo",
    "primitive_from",
    "primitive_table",
    "s1_table",
    "s0_arrays",
    "s1_arrays",
    "s1_from",
]

PRIMITIVE_STEP = 0.0025  # the knot spacing of PrimitiveTable.build
QUADRATURE_S0 = digest(
    ("S0 quadrature", FMT, S0_SOURCE, ENVIRONMENT, DEFAULT_NODES, DEFAULT_CUTOFF)
)
_EVENTS = (
    "S0 tables built",
    "S0 tables read",
    "S1 tables built",
    "S1 tables read",
    "one-node tables built",
)


def _scalar(value: Any) -> Any:
    """The one element of a stored scalar, which the cache keeps as an array of shape ``(1,)``."""
    return np.asarray(value).reshape(-1)[0]


def s0_arrays(table: PrimitiveTable) -> dict[str, np.ndarray]:
    """The arrays of an S0 table."""
    return {
        "lnx0": np.float64(table.lnx0),
        "step": np.float64(table.step),
        "n": np.int64(table.n),
        "order": np.int64(table.order),
        "coef_F": table.coef["F"],
        "coef_G": table.coef["G"],
    }


def primitive_from(arrays: dict) -> PrimitiveTable:
    """The S0 table of ``s0_arrays``."""
    coef = {"F": arrays["coef_F"], "G": arrays["coef_G"]}
    return PrimitiveTable(
        float(_scalar(arrays["lnx0"])),
        float(_scalar(arrays["step"])),
        int(_scalar(arrays["n"])),
        int(_scalar(arrays["order"])),
        coef,
    )


def s1_arrays(table: FieldAverage) -> dict[str, np.ndarray]:
    """The arrays of an S1 table. The method is part of its key and is not stored."""
    return {
        "nodes_uG": table.nodes_uG,
        "weights": table.weights,
        "order": np.int64(table.order),
        "coef_F": table.coef["F"],
        "coef_G": table.coef["G"],
        "lny": table.lny,
    }


def s1_from(arrays: dict, method: str) -> FieldAverage:
    """The S1 table of ``s1_arrays`` for ``method``."""
    coef = {"F": arrays["coef_F"], "G": arrays["coef_G"]}
    return FieldAverage(
        arrays["nodes_uG"],
        arrays["weights"],
        int(_scalar(arrays["order"])),
        method,
        coef,
        arrays["lny"],
    )


def primitive_table(memo: Any, nodes_ug: np.ndarray) -> PrimitiveTable:
    """The S0 table of the law's quantised span, built once per span in the memo (decision 158)."""
    span = quantised_span(*FieldAverage.needed_lnx(nodes_ug))
    if isinstance(memo, StageMemo):
        return memo.primitive(span, TABLE_ORDER)
    key = ("S0", span, TABLE_ORDER)
    if key not in memo:
        memo[key] = PrimitiveTable.build(*span, order=TABLE_ORDER)
    return memo[key]


def s1_table(
    memo: Any, method: str, nodes_ug: np.ndarray, probs: np.ndarray, kind: str = "law"
) -> FieldAverage:
    """An S1 table of the field law. Nodes at ``B = 0`` and zero weights are dropped: ``B W(y / B) -> 0``.

    With a ``StageMemo`` the table goes through the process cache, as a cached law (``kind="law"``) or as a
    one-node table (``kind="node"``).
    """
    nodes_ug, probs = np.asarray(nodes_ug, float), np.asarray(probs, float)
    if (
        nodes_ug.shape != probs.shape
        or not np.all(np.isfinite(nodes_ug) & (nodes_ug >= 0))
        or not np.all(np.isfinite(probs) & (probs >= 0))
    ):
        raise StackRefused(
            "the field law needs finite non-negative nodes and weights of equal length: the jet takes the cell"
        )
    keep = (nodes_ug > 0) & (probs > 0)
    if not np.any(keep):
        raise StackRefused(
            "the field law puts all its weight at B = 0: the jet takes the cell"
        )
    nodes_ug, probs = nodes_ug[keep], probs[keep]

    def build() -> FieldAverage:
        if method == "quadrature":
            return FieldAverage.quadrature(nodes_ug, probs)
        primitive = primitive_table(memo, nodes_ug)
        return FieldAverage.lookups(nodes_ug, probs, primitive=primitive)

    if isinstance(memo, StageMemo):
        return getattr(memo, kind)(method, nodes_ug, probs, build)
    return build()


class StageMemo:
    """S0 and S1 tables of one build through ``cache`` (see the module docstring).

    ``close()`` removes the one-node tables and returns the event counts of the build, with the rebuilds of
    evicted one-node tables. Not certified: the tables.
    """

    def __init__(self, cache: Cache) -> None:
        self.cache = cache
        self.nodes = cache.node_store()
        self.counts = dict.fromkeys(_EVENTS, 0)
        self._reads: list[str] = []

    def primitive(self, span: tuple[float, float], order: int) -> PrimitiveTable:
        """The S0 table over the quantised ``span``, read or built and stored."""
        key = s0_key(span, order, PRIMITIVE_STEP)
        found = self.cache.get("s0", key)
        if found is not None:
            self.counts["S0 tables read"] += 1
            return primitive_from(found)
        table = PrimitiveTable.build(*span, order=order, step=PRIMITIVE_STEP)
        self.counts["S0 tables built"] += 1
        about = {"stage": "S0", "span": list(span), "order": order}
        self.cache.put("s0", key, s0_arrays(table), description=about)
        return table

    def s1_key(self, method: str, nodes_ug: np.ndarray, probs: np.ndarray) -> str:
        """The S1 key of the law ``(nodes_ug, probs)`` for ``method``, after the zero nodes are dropped."""
        if method == "quadrature":
            return s1_key(QUADRATURE_S0, method, nodes_ug, probs)
        span = quantised_span(*FieldAverage.needed_lnx(nodes_ug))
        s0 = s0_key(span, TABLE_ORDER, PRIMITIVE_STEP)
        return s1_key(s0, method, nodes_ug, probs)

    def law(
        self, method: str, nodes_ug: Any, probs: Any, build: Callable[[], FieldAverage]
    ) -> FieldAverage:
        """A cached-law S1 table: memory, then disk, then ``build()``, which is stored."""
        key = self.s1_key(method, nodes_ug, probs)
        self._reads.append(key)
        found = self.cache.get("s1", key)
        if found is not None:
            self.counts["S1 tables read"] += 1
            return s1_from(found, method)
        table = build()
        self.counts["S1 tables built"] += 1
        about = {"stage": "S1", "method": method, "n_nodes": int(np.size(nodes_ug))}
        self.cache.put("s1", key, s1_arrays(table), description=about)
        return table

    def node(
        self, method: str, nodes_ug: Any, probs: Any, build: Callable[[], FieldAverage]
    ) -> FieldAverage:
        """A one-node S1 table from the build's ``NodeStore``, rebuilt when it was evicted."""
        key = self.s1_key(method, nodes_ug, probs)
        self._reads.append(key)

        def fresh() -> dict:
            self.counts["one-node tables built"] += 1
            return s1_arrays(build())

        return s1_from(self.nodes.get(key, fresh), method)

    def take_reads(self) -> tuple[str, ...]:
        """The distinct S1 keys read since the last call, sorted."""
        out = tuple(sorted(set(self._reads)))
        self._reads.clear()
        return out

    def close(self) -> dict[str, int]:
        """Remove the one-node tables. The counts of the build, with "one-node rebuilds"."""
        self.nodes.close()
        return {**self.counts, "one-node rebuilds": self.nodes.rebuilds}
