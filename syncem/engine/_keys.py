"""Stage keys of the basis build (design/0.8/SPEC.md decision 159, Sec. 8.2 and 8.5).

The keys are sha256 digests through ``syncem.base._fingerprint.digest``, so they hold no ``repr``, ``hash()``
or ``id()`` and are equal across processes. This module takes every input as an argument: chains, phase
fingerprints, channel row keys, geometry digests and the rules record arrive as strings or bytes, so it
calls no provider and no jet.

* ``s0_key(span, order, step)`` is ``H("S0", FMT, src(S0_SOURCES), environment, quantised span, order,
  step)``. The primitive table computes ``F`` and ``G`` in ``primitive.py`` itself, so ``S0_SOURCES`` is that
  module.
* ``s1_key(s0, method, nodes_ug, weights)`` is ``H("S1", K_S0, src(S1_SOURCES), method, field nodes in uG,
  weights)``.
* ``s2_context_keys(...)`` gives one context key per channel of one (cell, block): the provider and mode, the
  S1 key or the kernel record, the block, the averaged-kernel record, ``d_0`` of each rule, the cell geometry
  digest and the rules record are shared, and the channel row key and the phase fingerprint of that channel
  are its own. A channel's key does not depend on the other channels.
* ``memo_key(...)`` hashes the column-map key, the kernel record, the averaged-kernel record, the ordered
  phase fingerprints, the rule fingerprints ``d_N``, the ordered channel row keys, the provider settings and
  the engine source digest. It takes no moments and no weights.
* ``design_key(memo, lattice, factor)`` hashes the memo key, ``lattice.key`` and the azimuth factor.

``kernel_record`` is ``("class", module.qualname)`` of the kernel followed by ``describe()`` without the
entries the kernel class names in ``guards``. A class outside syncem that inherits ``describe()`` from a
syncem class has no record, so a subclass that changes the kernel values is not cached. A class that
defines its own ``describe()`` is keyed by it. ``averaged_record`` holds the averaged names and families, the fixed values, ``legendre`` and
``outer_batch``. ``column_record`` is the averaged-kernel record of the S2 context key. It holds each Legendre
axis by name and node count, since column ``l`` of a projection on ``n`` Gauss nodes does not depend on ``L``
and the column id carries the degrees. A key whose inputs hold ``None`` (no fingerprint) is ``None``, and a traced azimuth factor
gives a ``None`` design key.

``environment`` (the Python minor version and the NumPy and SciPy versions) enters the S0 key, and through
it the S1 key, because the disk entries outlive an upgrade of the libraries that compute them. The source
digests ``S0_SOURCE``, ``S1_SOURCE`` and ``ENGINE_SOURCE`` are read once, when this module is imported.

Units: field nodes in uG, spans in e-folds of ``ln x``. Shapes: ``s2_context_keys`` returns a ``(n_ch,)``
array of 64-character strings. Not certified: a source digest covers the named files and not the code they
import, and a key is as complete as the fingerprints passed in.
"""

from __future__ import annotations

import importlib.metadata
import importlib.util
import math
import pkgutil
import sys
import types
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ..base._fingerprint import NotCacheable, digest, encode
from ._disk import FORMAT
from .average import NodeRule

__all__ = [
    "ENGINE_SOURCE",
    "ENVIRONMENT",
    "FMT",
    "S0_SOURCE",
    "S0_SOURCES",
    "S1_SOURCE",
    "S1_SOURCES",
    "averaged_record",
    "column_record",
    "design_key",
    "engine_modules",
    "environment",
    "kernel_class",
    "kernel_record",
    "kernel_record_with_reason",
    "memo_key",
    "s0_key",
    "s1_key",
    "s2_context_keys",
    "source_digest",
]

FMT = FORMAT
S0_SOURCES = ("syncem.engine.primitive",)
S1_SOURCES = ("syncem.engine.field_average",)
_ENGINE = "syncem.engine"


# ------------------------------------------------------------------ source digests and the environment


def _module_name(module: Any) -> str:
    if isinstance(module, types.ModuleType):
        return module.__name__
    if isinstance(module, str):
        return module
    raise ValueError(
        f"a module is a module object or a dotted name, not {type(module).__name__}"
    )


def _source_path(name: str) -> Path:
    try:
        spec = importlib.util.find_spec(name)
    except (ImportError, ValueError) as err:
        raise ValueError(f"module {name} has no source file: {err}") from err
    origin = None if spec is None else spec.origin
    if origin is None or not Path(origin).is_file():
        raise ValueError(f"module {name} has no source file (origin {origin})")
    return Path(origin)


def source_digest(*modules: Any) -> str:
    """sha256 of the names and file bytes of ``modules``, read now, in sorted order and without repeats."""
    names = sorted({_module_name(m) for m in modules})
    if not names:
        raise ValueError("source_digest needs at least one module")
    return digest(("source", tuple((n, _source_path(n).read_bytes()) for n in names)))


def engine_modules() -> tuple[str, ...]:
    """The package ``syncem.engine`` and its modules, by name. File copies with invalid names are left out."""
    path = Path(_source_path(_ENGINE)).parent
    found = (m.name for m in pkgutil.iter_modules([str(path)]) if m.name.isidentifier())
    return (_ENGINE, *sorted(f"{_ENGINE}.{name}" for name in found))


def _version(distribution: str) -> str | None:
    try:
        return importlib.metadata.version(distribution)
    except importlib.metadata.PackageNotFoundError:
        return None


def environment() -> tuple[tuple[str, Any], ...]:
    """The Python minor version and the installed NumPy and SciPy versions."""
    python = f"{sys.version_info[0]}.{sys.version_info[1]}"
    return (
        ("python", python),
        ("numpy", _version("numpy")),
        ("scipy", _version("scipy")),
    )


S0_SOURCE = source_digest(*S0_SOURCES)
S1_SOURCE = source_digest(*S1_SOURCES)
ENGINE_SOURCE = source_digest(*engine_modules())
ENVIRONMENT = environment()


# ------------------------------------------------------------------ records


_OWN = "syncem"


def _own(cls: type) -> bool:
    return cls.__module__ == _OWN or cls.__module__.startswith(_OWN + ".")


def kernel_class(kernel: Any) -> tuple[str, str]:
    """``("class", module.qualname)`` of the kernel.

    Raises ``NotCacheable`` for a class outside syncem that inherits ``describe()`` from a syncem class, since
    that record does not cover what the subclass overrides.
    """
    cls = type(kernel)
    owner = next((c for c in cls.__mro__ if "describe" in vars(c)), None)
    if owner is not None and _own(owner) and not _own(cls):
        raise NotCacheable(
            f"kernel class {cls.__qualname__} inherits describe() from {owner.__qualname__}, which does "
            f"not record what the subclass changes. Define describe() on {cls.__qualname__}"
        )
    return ("class", f"{cls.__module__}.{cls.__qualname__}")


def kernel_record_with_reason(kernel: Any) -> tuple[tuple | None, tuple[str, ...]]:
    """The kernel record and ``()``, or ``None`` and the reasons starting "not cached: "."""
    try:
        head = kernel_class(kernel)
        entries = tuple(kernel.describe())
    except NotCacheable as err:
        return None, (f"not cached: {err.reason}",)
    if not all(
        isinstance(e, tuple) and len(e) == 2 and isinstance(e[0], str) for e in entries
    ):
        raise TypeError("describe() must return (name, value) pairs")
    why = dict(entries).get("not_cached")
    if why:
        return None, tuple(why)
    guards = set(getattr(type(kernel), "guards", ()))
    kept = tuple((n, v) for n, v in entries if n not in guards and n != "not_cached")
    record = (head, *kept)
    try:
        encode(record)
    except NotCacheable as err:
        return None, (f"not cached: the kernel record has no encoding: {err.reason}",)
    return record, ()


def kernel_record(kernel: Any) -> tuple | None:
    """The kernel class and ``kernel.describe()`` without the entries named in ``guards``, or ``None``."""
    return kernel_record_with_reason(kernel)[0]


def _family(rule: Any) -> str:
    return "NodeRule" if isinstance(rule, NodeRule) else "callable"


def averaged_record(averaged: Any) -> tuple:
    """Averaged names and families, fixed values as floats, ``legendre`` (name, L, nodes) and ``outer_batch``."""
    return (
        ("averaged", tuple((str(n), _family(r)) for n, r in averaged.averaged)),
        ("fixed", tuple((str(n), float(v)) for n, v in averaged.fixed)),
        ("legendre", tuple((str(n), int(L), int(k)) for n, L, k in averaged.legendre)),
        ("outer_batch", averaged.outer_batch),
    )


def column_record(averaged: Any) -> tuple:
    """``averaged_record`` with each Legendre axis as (name, node count), the record of the S2 context key."""
    record = dict(averaged_record(averaged))
    record["legendre"] = tuple((str(n), int(k)) for n, _, k in averaged.legendre)
    return tuple(record.items())


# ------------------------------------------------------------------ S0 and S1


def _span(span: Any) -> tuple[float, float]:
    values = np.asarray(span, dtype=float)
    if (
        values.shape != (2,)
        or not np.all(np.isfinite(values))
        or not values[0] < values[1]
    ):
        raise ValueError(f"an S0 span is a finite pair lo < hi, not {span!r}")
    return float(values[0]), float(values[1])


def _count(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 0:
        raise ValueError(f"{what} must be a non-negative int, not {value!r}")
    return int(value)


def s0_key(
    span: Any,
    order: Any,
    step: Any,
    *,
    fmt: int = FMT,
    source: str = S0_SOURCE,
    environment: Any = ENVIRONMENT,
) -> str:
    """Key of the S0 table over a quantised ``span`` in ``ln x``, to ``order``, with knot spacing ``step``."""
    step = float(step)
    if not (math.isfinite(step) and step > 0):
        raise ValueError(f"step must be finite and positive, not {step!r}")
    parts = ("S0", _count(fmt, "fmt"), str(source), environment, _span(span))
    return digest((*parts, _count(order, "order"), step))


def _finite_vector(values: Any, what: str) -> np.ndarray:
    out = np.ascontiguousarray(values, dtype=float)
    if out.ndim != 1 or out.size == 0 or not np.all(np.isfinite(out)):
        raise ValueError(f"{what} must be a non-empty finite vector, not {values!r}")
    return out


def s1_key(
    s0: str | None, method: str, nodes_ug: Any, weights: Any, *, source: str = S1_SOURCE
) -> str | None:
    """Key of the S1 table of one field law read from the S0 table ``s0``. ``None`` when ``s0`` is ``None``."""
    if s0 is None:
        return None
    nodes = _finite_vector(nodes_ug, "nodes_ug")
    probs = _finite_vector(weights, "weights")
    if nodes.shape != probs.shape:
        raise ValueError(f"{nodes.size} field nodes and {probs.size} weights")
    return digest(("S1", str(s0), str(source), str(method), nodes, probs))


# ------------------------------------------------------------------ S2, memo and design


def _strings(values: Any, what: str) -> tuple[str, ...] | None:
    if values is None:
        return None
    out = tuple(values)
    if any(v is None for v in out):
        return None
    if not all(isinstance(v, str) for v in out):
        raise TypeError(f"{what} must be strings")
    return tuple(str(v) for v in out)


def _digests(values: Any, what: str) -> dict | None:
    if values is None:
        return None
    if not isinstance(values, Mapping):
        raise TypeError(f"{what} must map rule names to digests")
    if any(v is None for v in values.values()):
        return None
    return {str(k): str(v) for k, v in values.items()}


def _per_channel(row_keys: Any, phase: Any) -> tuple[tuple, tuple] | None:
    rows, phases = _strings(row_keys, "row_keys"), _strings(phase, "phase")
    if rows is None or phases is None:
        return None
    if len(rows) != len(phases):
        raise ValueError(
            f"{len(rows)} channel row keys and {len(phases)} phase fingerprints"
        )
    return rows, phases


def s2_context_keys(
    *,
    provider: str,
    mode: str,
    stage: Any,
    block: Any,
    row_keys: Any,
    phase: Any,
    averaged: Any,
    d0: Any,
    geometry: str | None,
    rules: Any,
) -> np.ndarray | None:
    """One S2 context key per channel of one (cell, block). ``None`` when an input has no fingerprint.

    ``stage`` is the S1 key of a stack entry or the kernel record of a jet or core entry. ``d0`` maps each rule
    to its ``d_0``, ``geometry`` is the cell geometry digest and ``rules`` the rules record.
    """
    channels, chains = _per_channel(row_keys, phase), _digests(d0, "d0")
    if channels is None or chains is None or None in (stage, averaged, geometry, rules):
        return None
    shared = digest(
        (
            "S2",
            str(provider),
            str(mode),
            stage,
            block,
            averaged,
            chains,
            str(geometry),
            rules,
        )
    )
    out = [digest(("S2 channel", shared, r, p)) for r, p in zip(*channels)]
    return np.array(out, dtype="<U64")


def memo_key(
    *,
    column_map: str | None,
    kernel: Any,
    averaged: Any,
    phase: Any,
    rules: Any,
    row_keys: Any,
    provider: Any,
    engine: str,
) -> str | None:
    """Basis memo key over the ordered channels. ``None`` when an input has no fingerprint."""
    channels, chains = _per_channel(row_keys, phase), _digests(rules, "rules")
    if channels is None or chains is None or None in (column_map, kernel, averaged):
        return None
    rows, phases = channels
    parts = (str(column_map), kernel, averaged, phases, chains, rows)
    return digest(("basis memo", *parts, provider, str(engine)))


def design_key(memo: str | None, lattice: str | None, factor: Any) -> str | None:
    """``H(memo key, lattice.key, factor)``. ``None`` when a key is ``None`` or the factor is traced."""
    if memo is None or lattice is None:
        return None
    try:
        return digest(("design", str(memo), str(lattice), factor))
    except NotCacheable:
        return None
