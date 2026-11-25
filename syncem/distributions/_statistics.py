"""Identity of placed distributions for grouping specified populations."""

from dataclasses import dataclass, fields, is_dataclass
from numbers import Number

import jax
import numpy as np

from ..base._fingerprint import NotCacheable, digest
from ..base._function_fingerprint import function_fingerprint
from ._common import Law
from .azimuth import AzimuthFactor
from .parameters import Conditional


@dataclass(frozen=True, eq=False)
class _Identity:
    """An opaque object's local identity remains valid while the key is held."""

    obj: object

    def __eq__(self, other):
        return isinstance(other, _Identity) and self.obj is other.obj

    def __hash__(self):
        return id(self.obj)


def _local_key(value):
    if callable(value):
        return _Identity(value)
    if is_dataclass(value):
        return type(value), tuple(
            (f.name, _local_key(getattr(value, f.name))) for f in fields(value)
        )
    if isinstance(value, dict):
        return type(value), frozenset((k, _local_key(v)) for k, v in value.items())
    if isinstance(value, (tuple, list)):
        return type(value), tuple(_local_key(v) for v in value)
    try:
        if isinstance(value, (Number, np.ndarray, jax.Array)):
            value = np.asarray(value)
        return digest(value)
    except NotCacheable:
        return _Identity(value)


@dataclass(frozen=True)
class _LocalStatistic:
    """Compare parameters by value and opaque callables by identity across placement."""

    key: object

    @classmethod
    def from_declaration(cls, declaration):
        return cls(_local_key(declaration))


def factor_record(factor):
    """Compare placed declarations without built-in display-unit metadata."""
    if isinstance(factor, AzimuthFactor) and isinstance(factor.c, Conditional):
        if factor.c.record()["fingerprint"] is None:
            return _LocalStatistic.from_declaration(factor)
    if hasattr(factor, "record"):
        record = factor.record()
        if "fingerprint" in record and record["fingerprint"] is None:
            return _LocalStatistic.from_declaration(factor)
        kind = type(factor)
        if (
            kind.__module__.startswith("syncem.distributions.")
            and kind.record is Law.record
        ):
            record = {
                name: value
                for name, value in record.items()
                if name not in ("given_unit", "fingerprint")
            }
            record["parameters"] = {
                name: value
                for name, value in record["parameters"].items()
                if name != "given_unit"
            }
            return {**record, "fingerprint": digest(record)}
        return record
    if callable(factor):
        try:
            return ("function", function_fingerprint(factor))
        except NotCacheable:
            return ("declaration", id(factor))
    return factor


def factor_key(factor):
    """Hash a placed declaration or keep its identity when it cannot be fingerprinted."""
    try:
        record = factor_record(factor)
        if isinstance(record, _LocalStatistic):
            return (type(factor), record)
        return (type(factor), digest(record))
    except (NotCacheable, TypeError, ValueError):
        return ("identity", id(factor))
