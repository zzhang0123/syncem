"""The jet routine: compiled programs of one signature, the memory budget and the order limit (design/0.8/SPEC.md
decisions 165, 166 and 169).

LABEL ``eq: channel basis expansion``. ``JetRoutine(names, charts, exponents, one)`` evaluates
``one(averaged, view, ref, hw)``, the Taylor columns of one cell (``ref`` and ``hw`` of shape ``(d,)``) on one
node chunk of one channel. It compiles each program ahead of time with ``eqx.filter_jit(one).lower(...)
.compile()`` and keeps at most ``TRACE_SLOTS`` programs, least recently used first. A program is keyed by the
static part of its arguments and the shapes and dtypes of the array part, so two calls share a program when
``eqx.filter_jit`` would reuse its trace, and only then.

Before the first execution of a program (not at its compile) the routine reads ``memory_analysis()`` and the budget
``SYNCEM_JET_MEM`` (default 4 GB, the size syntax of decision 161, 0 disables the check), and raises
``JetMemoryError`` when the temporary, argument and output bytes together exceed it. The budget decides whether
a program runs, never its bits. When the analysis is ``None`` or raises, the program runs unchecked, counted in
"jet memory unchecked", with one ``RuntimeWarning`` per routine. A routine whose nesting depth (the largest total
degree of its exponents) exceeds ``SYNCEM_JET_MAX_ORDER`` (default 12) raises before tracing. Nested forward mode
is used through order 5.

``chunk_plan(n_nu)`` gives the ``K = ceil(n_nu / 64)`` chunks of ``C = ceil(n_nu / K)`` nodes of a channel, and
``chunk_views(channels, j)`` the ``ResponseView`` chunks of channel ``j``, the last padded with copies of its last
node at response weight 0. A view holds the nodes, the weights and the response weights of the channel row as
array leaves, evaluated on the host, and the chunk size as its only static field, so the channels of every family
(including the per-channel static tables and callables of ``from_bandpass``, ``from_table`` and
``from_callable``) share one program per chunk size (decision 171).

Memory: the budget bounds the buffers of one execution. Compile memory is not read before compiling, and the
order limit is its only guard (an order-5 compile at ``d = 2`` peaked at 5.3 GB). An evicted program leaves its
executable in the JAX caches until ``jax.clear_caches()`` (``Cache.clear(jax=True)``). Units: none here, the
columns carry those of the kernel. Not certified: the estimate is what XLA reports for the compiled program.
"""

from __future__ import annotations

import dataclasses
import math
import os
import threading
import warnings
from collections import OrderedDict
from collections.abc import Callable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ._disk_settings import env_bytes

__all__ = [
    "DEFAULT_JET_MAX_ORDER",
    "DEFAULT_JET_MEM_BYTES",
    "JET_MAX_ORDER_ENV",
    "JET_MEM_ENV",
    "JET_NODE_CHUNK",
    "TRACE_SLOTS",
    "JetMemoryError",
    "JetRoutine",
    "ResponseView",
    "check_order",
    "chunk_plan",
    "chunk_views",
    "jet_max_order",
    "jet_mem_bytes",
    "order_error",
]

JET_NODE_CHUNK = 64
TRACE_SLOTS = 4
DEFAULT_JET_MEM_BYTES = 4_000_000_000
DEFAULT_JET_MAX_ORDER = 12
NESTED_MAX_ORDER = 5
JET_MEM_ENV = "SYNCEM_JET_MEM"
JET_MAX_ORDER_ENV = "SYNCEM_JET_MAX_ORDER"


class JetMemoryError(MemoryError):
    """A jet program above the budget (``kind`` "budget") or a routine above the order limit ("order")."""

    def __init__(
        self,
        message: str,
        *,
        kind: str,
        order: int | None = None,
        cell: int | None = None,
    ):
        super().__init__(message)
        self.kind, self.order, self.cell = kind, order, cell


# ------------------------------------------------------------------ settings


def jet_mem_bytes(environ: Mapping[str, str] | None = None) -> int:
    """The budget per compiled program, ``SYNCEM_JET_MEM`` or 4 GB. The value 0 disables the check."""
    return env_bytes(JET_MEM_ENV, DEFAULT_JET_MEM_BYTES, environ)


def jet_max_order(environ: Mapping[str, str] | None = None) -> int:
    """The overall order ceiling, ``SYNCEM_JET_MAX_ORDER`` (a positive integer) or 12."""
    value = (os.environ if environ is None else environ).get(JET_MAX_ORDER_ENV, "")
    value = value.strip()
    if not value:
        return DEFAULT_JET_MAX_ORDER
    if not value.isdigit() or int(value) < 1:
        raise ValueError(
            f"{JET_MAX_ORDER_ENV} must be a positive integer, got {value!r}"
        )
    return int(value)


def _size(n: int) -> str:
    return f"{n} bytes" if n < 10**6 else f"{n / 1e9:.3g} GB"


def order_error(
    names: tuple, order: int, cell: int | None = None, reason: str | None = None
) -> JetMemoryError:
    """The refusal of a routine of nesting depth ``order`` above the limit, with the stack's reason if any."""
    limit = jet_max_order()
    where = f"cell {cell}" if cell is not None else f"the jet routine for {names}"
    why = f" (the stack refused it: {reason})" if reason else ""
    return JetMemoryError(
        f"{where} needs the jet at order {order}{why}, and the jet takes orders up to {limit}. "
        f"Lower the order, route the cell to the stack where its precondition holds (the continuum kernel "
        f"to order 6), or set {JET_MAX_ORDER_ENV}={order}.",
        kind="order",
        order=order,
        cell=cell,
    )


def check_order(names: tuple, exponents: Any, cell: int | None = None) -> int:
    """The highest derivative order. Raises ``JetMemoryError`` above the order limit."""
    order = max((sum(int(x) for x in e) for e in exponents), default=0)
    if order > jet_max_order():
        raise order_error(names, order, cell)
    return order


# ------------------------------------------------------------------ node chunks


def chunk_plan(n_nu: int) -> tuple[int, int]:
    """``(K, C)``: ``K = ceil(n_nu / 64)`` chunks of ``C = ceil(n_nu / K)`` nodes."""
    if isinstance(n_nu, bool) or not isinstance(n_nu, (int, np.integer)) or n_nu < 1:
        raise ValueError(f"n_nu must be a positive int, got {n_nu!r}")
    k = math.ceil(int(n_nu) / JET_NODE_CHUNK)
    return k, math.ceil(int(n_nu) / k)


def _row(channels: Any, j: int) -> Any:
    return (
        channels[j : j + 1] if hasattr(channels, "__getitem__") else channels.select(j)
    )


class ResponseView(eqx.Module):
    """One node chunk of one channel: ``nodes``, ``weights`` and ``measure`` of shape ``(1, C)`` (decision 171).

    ``measure`` is the channel's ``response_weights()`` at the chunk nodes, evaluated on the host from the
    channel row, and ``response_weights()`` returns it. The static part is ``n_nu = C`` only, so every channel
    of every family shares the program of its chunk size. A kernel reads the nodes and the response weights
    (the protocol of Sec. 8.1).
    """

    nodes: jax.Array
    weights: jax.Array
    measure: jax.Array
    n_nu: int = eqx.field(static=True)

    @property
    def n_ch(self) -> int:
        return 1

    def response_weights(self) -> jax.Array:
        return self.measure


def _measure(row: Any) -> np.ndarray | None:
    """The response weights of a one-channel row on the host, or ``None`` for a row without them."""
    if hasattr(row, "response_weights"):
        return np.asarray(row.response_weights())[0]
    if hasattr(row, "node_values"):
        return np.asarray(row.weights * row.node_values())[0]
    return None


def chunk_views(channels: Any, j: int) -> tuple[Any, ...]:
    """The node chunks of channel ``j`` as ``ResponseView`` objects of shape ``(1, C)``, in chunk order.

    A channel object without ``response_weights`` or ``node_values`` gives one-channel slices instead.
    """
    row = _row(channels, j)
    nodes, weights = np.asarray(row.nodes)[0], np.asarray(row.weights)[0]
    measure = _measure(row)
    k, c = chunk_plan(nodes.shape[0])
    pad = k * c - nodes.shape[0]
    nodes = np.concatenate([nodes, np.full(pad, nodes[-1])])
    weights = np.concatenate([weights, np.zeros(pad, weights.dtype)])
    if measure is None:
        return tuple(
            dataclasses.replace(
                row,
                nodes=jnp.asarray(nodes[None, i * c : (i + 1) * c]),
                weights=jnp.asarray(weights[None, i * c : (i + 1) * c]),
                n_nu=c,
            )
            for i in range(k)
        )
    measure = np.concatenate([measure, np.zeros(pad, measure.dtype)])
    return tuple(
        ResponseView(
            *(
                jnp.asarray(a[None, i * c : (i + 1) * c])
                for a in (nodes, weights, measure)
            ),
            c,
        )
        for i in range(k)
    )


# ------------------------------------------------------------------ the routine


@dataclasses.dataclass
class _Program:
    run: Any
    parts: tuple[int, int, int] | None
    checked: bool = False


def _key(args: tuple) -> tuple:
    """The static part of ``args`` and the shapes and dtypes of the array part, as ``eqx.filter_jit`` keys."""
    dynamic, static = eqx.partition(args, eqx.is_array)
    leaves, tree = jax.tree_util.tree_flatten(dynamic)
    avals = tuple(
        (np.shape(x), np.dtype(x.dtype).str, bool(getattr(x, "weak_type", False)))
        for x in leaves
    )
    fixed, layout = jax.tree_util.tree_flatten(static)
    return tree, avals, layout, tuple(fixed)


def _parts(compiled: Any) -> tuple[int, int, int] | None:
    """Temporary, argument and output bytes of a compiled program, or ``None`` when not reported."""
    try:
        stats = compiled.compiled.memory_analysis()
    except (
        Exception
    ):  # noqa: BLE001  (any backend failure leaves the program unchecked)
        return None
    if stats is None:
        return None
    names = ("temp_size_in_bytes", "argument_size_in_bytes", "output_size_in_bytes")
    return tuple(int(getattr(stats, n)) for n in names)


def _bump(events: dict | None, name: str) -> None:
    if events is not None:
        events[name] = events.get(name, 0) + 1


class JetRoutine:
    """The compiled programs of one (names, charts, exponents) signature, at most ``TRACE_SLOTS`` of them."""

    def __init__(
        self, names: tuple, charts: tuple, exponents: tuple, one: Callable
    ) -> None:
        self.names, self.charts, self.exponents = names, charts, exponents
        self.order = check_order(names, exponents)
        self._one = one
        self._programs: OrderedDict = OrderedDict()
        self._lock = threading.RLock()
        self._warned = False

    @property
    def programs(self) -> int:
        """The number of compiled programs held."""
        return len(self._programs)

    def __call__(
        self,
        averaged: Any,
        view: Any,
        ref: Any,
        hw: Any,
        *,
        events: dict | None = None,
        request: tuple | None = None,
    ) -> tuple[Any, Any, Any]:
        """``(I, P, V)`` of one cell on one view, ``(n_rows, n_view_channels, *L)``."""
        args = (averaged, view, jnp.asarray(ref, float), jnp.asarray(hw, float))
        with self._lock:
            key, program = self._program(args, events)
            if not program.checked:
                self._check(key, program, view, request, events)
        return program.run(*args)

    def estimate(
        self, averaged: Any, view: Any, ref: Any, hw: Any
    ) -> tuple[int, int, int] | None:
        """Temporary, argument and output bytes of the program of these arguments, compiled, not run or checked."""
        args = (averaged, view, jnp.asarray(ref, float), jnp.asarray(hw, float))
        with self._lock:
            return self._program(args, None)[1].parts

    def _program(self, args: tuple, events: dict | None) -> tuple[tuple, _Program]:
        key = _key(args)
        found = self._programs.get(key)
        if found is not None:
            self._programs.move_to_end(key)
            return key, found
        compiled = eqx.filter_jit(self._one).lower(*args).compile()
        program = _Program(compiled, _parts(compiled))
        _bump(events, "jet compiles")
        self._programs[key] = program
        while len(self._programs) > TRACE_SLOTS:
            self._programs.popitem(last=False)
        return key, program

    def _check(self, key, program, view, request, events) -> None:
        budget = jet_mem_bytes()
        if budget and program.parts is None:
            _bump(events, "jet memory unchecked")
            if not self._warned:
                self._warned = True
                warnings.warn(
                    f"memory_analysis() is not available on the {jax.default_backend()} backend: the jet "
                    f"program for {self.names} at order {self.order} runs without the {JET_MEM_ENV} check",
                    RuntimeWarning,
                    stacklevel=4,
                )
        elif budget and sum(program.parts) > budget:
            del self._programs[key]
            raise JetMemoryError(
                self._message(program.parts, view, request, budget), kind="budget"
            )
        program.checked = True

    def _message(self, parts, view, request, budget) -> str:
        temp, arg, out = parts
        nodes = getattr(view, "nodes", None)
        per_call = (
            "the whole channel set per call"
            if nodes is None
            else f"one channel of {np.shape(nodes)[-1]} nodes per call"
        )
        n_ch, n_cells = request if request is not None else ("?", "?")
        return (
            f"the jet program for {self.names} at order {self.order} ({len(self.exponents)} exponents, "
            f"{per_call}, {n_ch} channels, {n_cells} cells) needs about {_size(temp + arg + out)} per call "
            f"(temporary {_size(temp)}, arguments {_size(arg)}, outputs {_size(out)}), above {JET_MEM_ENV} = "
            f"{_size(budget)}. Lower the order, average fewer nodes, route the cells to the stack where its "
            f"precondition holds, or raise {JET_MEM_ENV}."
        )
