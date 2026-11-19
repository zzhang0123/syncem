"""Canonical byte encoding and sha256 digests of plain data (design/0.8/SPEC.md decision 153).

``encode(value)`` writes a value as bytes that depend on its type and content only. It uses no
``repr``, ``hash()`` or ``id()``, so the bytes are equal across processes and ``PYTHONHASHSEED`` values.
``digest(value)`` is the sha256 hex digest of a format tag and ``encode(value)``.

Encoding rules:

* every value starts with a one-byte type tag, and strings, bytes and containers carry a length prefix,
* ``None``, ``bool``, ``int``, ``float``, ``complex``, ``str`` and ``bytes`` are distinct types,
* floats are written as ``'<f8'`` of ``x + 0.0``, so ``-0.0`` equals ``0.0``, and every NaN is written as
  the one quiet NaN,
* NumPy scalars are written as the Python scalar of the same value,
* tuples and lists are distinct and keep their order, dicts, sets and frozensets are sorted by the
  encoding of their entries,
* arrays (NumPy, and concrete JAX arrays) are written as dtype string, shape and C-order little-endian
  bytes, with ``-0.0`` and NaN normalised as for floats,
* dataclass instances, ``eqx.Module`` included, are written as module, qualname and every field by name.

Anything else raises ``NotCacheable`` with a reason: functions and other callables, classes, traced JAX
arrays, object and structured arrays. Functions enter keys through ``function_fingerprint`` of
``syncem.base._function_fingerprint`` (decision 156). A caller that records a function field, such as
``UserKernel.describe``, writes ``("function", function_fingerprint(f))`` in its place before encoding. The
module imports NumPy and the standard library only and reads JAX from ``sys.modules`` when a JAX value is
passed.
"""

from __future__ import annotations

import dataclasses
import hashlib
import struct
import sys
from typing import Any

import numpy as np

__all__ = ["NotCacheable", "digest", "encode"]

_FORMAT = b"syncem-encode-1"
_NAN = struct.pack("<d", float("nan"))


class NotCacheable(Exception):
    """A value without a canonical encoding. ``reason`` names the value and why."""

    def __init__(self, reason: str) -> None:
        super().__init__(f"not cacheable: {reason}")
        self.reason = reason


def _length(n: int) -> bytes:
    return struct.pack("<Q", n)


def _float(x: float) -> bytes:
    x = float(x) + 0.0
    return _NAN if x != x else struct.pack("<d", x)


def _text(tag: bytes, data: bytes) -> bytes:
    return tag + _length(len(data)) + data


def _sequence(tag: bytes, items: Any) -> bytes:
    parts = [encode(v) for v in items]
    return tag + _length(len(parts)) + b"".join(parts)


def _unordered(tag: bytes, parts: list[bytes]) -> bytes:
    return tag + _length(len(parts)) + b"".join(sorted(parts))


def _array(value: np.ndarray) -> bytes:
    if value.dtype.hasobject or value.dtype.fields is not None:
        raise NotCacheable(f"array of dtype {value.dtype} has no canonical bytes")
    a = np.ascontiguousarray(value).astype(value.dtype.newbyteorder("<"), copy=False)
    if a.dtype.kind in "fc":
        a = a + np.zeros((), a.dtype)
        a = np.where(np.isnan(a), np.array(np.nan, a.dtype), a)
    head = _text(b"", a.dtype.str.encode()) + _length(a.ndim)
    head += b"".join(_length(n) for n in a.shape)
    return b"A" + head + _text(b"", np.ascontiguousarray(a).tobytes())


def _jax_value(value: Any) -> np.ndarray | None:
    """The concrete NumPy array of a JAX array, ``None`` for other values."""
    jax = sys.modules.get("jax")
    if jax is None:
        return None
    if isinstance(value, jax.core.Tracer):
        raise NotCacheable(f"traced array of shape {value.shape} has no value")
    return np.asarray(value) if isinstance(value, jax.Array) else None


def _dataclass(value: Any) -> bytes:
    cls = type(value)
    name = f"{cls.__module__}.{cls.__qualname__}".encode()
    fields = [
        _text(b"", f.name.encode()) + encode(getattr(value, f.name))
        for f in dataclasses.fields(value)
    ]
    return _text(b"K", name) + _length(len(fields)) + b"".join(fields)


def _scalar(value: Any) -> bytes | None:
    if value is None:
        return b"N"
    if isinstance(value, (bool, np.bool_)):
        return b"B" + (b"\x01" if value else b"\x00")
    if isinstance(value, (int, np.integer)):
        return _text(b"I", str(int(value)).encode())
    if isinstance(value, (float, np.floating)):
        return b"F" + _float(value)
    if isinstance(value, (complex, np.complexfloating)):
        return b"C" + _float(value.real) + _float(value.imag)
    if isinstance(value, str):
        return _text(b"S", value.encode("utf-8"))
    if isinstance(value, (bytes, bytearray)):
        return _text(b"Y", bytes(value))
    return None


def _container(value: Any) -> bytes | None:
    if isinstance(value, tuple):
        return _sequence(b"T", value)
    if isinstance(value, list):
        return _sequence(b"L", value)
    if isinstance(value, dict):
        return _unordered(b"D", [encode(k) + encode(v) for k, v in value.items()])
    if isinstance(value, frozenset):
        return _unordered(b"Z", [encode(v) for v in value])
    if isinstance(value, set):
        return _unordered(b"E", [encode(v) for v in value])
    return None


def encode(value: Any) -> bytes:
    """Canonical bytes of ``value``. Raises ``NotCacheable`` for a value without an encoding."""
    found = _scalar(value)
    if found is None:
        found = _container(value)
    if found is not None:
        return found
    if isinstance(value, np.ndarray):
        return _array(value)
    concrete = _jax_value(value)
    if concrete is not None:
        return _array(concrete)
    if isinstance(value, type):
        raise NotCacheable(f"class {value.__qualname__} has no encoding")
    if dataclasses.is_dataclass(value):
        return _dataclass(value)
    if callable(value):
        name = getattr(value, "__qualname__", type(value).__qualname__)
        raise NotCacheable(
            f"function {name} has no encoding, its keys use function_fingerprint"
        )
    raise NotCacheable(f"no encoding for type {type(value).__qualname__}")


def digest(value: Any) -> str:
    """sha256 hex digest of the format tag and ``encode(value)``."""
    return hashlib.sha256(_FORMAT + encode(value)).hexdigest()
