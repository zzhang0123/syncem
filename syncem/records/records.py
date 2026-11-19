"""Records of a basis build (design/0.8/SPEC.md Sec. 3.4, decisions 149 and 150).

LABEL ``[extension]``. ``SpecifiedRecord`` names an averaged statistic of a basis: its variable, its family
("NodeRule", "callable" or "fixed"), its parents and its fingerprint as static fields, and its parameter values
as an array leaf of shape ``(n_params,)``, empty in step 5. ``RouteRecord`` gives the realised provider of one
product cell ("stack", "jet" or "core"), the stack mode, the S1 method and the reason of a refusal.
``BuildRecord`` holds tuples of routes, the Stokes request, the S1 method, the sizes ``(n_rows, n, n0, n2)``,
event counts, timings in seconds with the 1, 5 and 15 minute load averages, versions, keys, defaults and the
raw agreement numbers of step 6. Its other fields are tuples of ``(key, value)`` pairs with scalar values,
after ``Provenance`` of ``syncem.model``.

Units: seconds for timings, bytes for ``mem_bytes``. Counts and sizes are dimensionless. Every ``to_dict`` is
strict JSON through ``strict_json`` of ``syncem.model``: string keys, lists for tuples and ``None`` for a
non-finite float. Not certified: a record documents a build and certifies nothing about its accuracy.
"""

from __future__ import annotations

import dataclasses
import operator
import os
from collections.abc import Mapping
from typing import Any, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from syncem import __version__
from syncem.model._provenance import strict_json

__all__ = [
    "FAMILIES",
    "PROVIDERS",
    "S1_METHODS",
    "SIZE_KEYS",
    "STACK_MODES",
    "STOKES_BLOCKS",
    "BuildRecord",
    "RouteRecord",
    "SpecifiedRecord",
]

FAMILIES = ("NodeRule", "callable", "fixed")
PROVIDERS = ("stack", "jet", "core")
STACK_MODES = ("cached-S1", "per-node", "field-scaling")
S1_METHODS = ("lookup", "quadrature")
SIZE_KEYS = ("n_rows", "n", "n0", "n2")
STOKES_BLOCKS = ("I", "P", "Q", "U", "V")
_PAIR_FIELDS = (
    "sizes",
    "events",
    "timings",
    "versions",
    "keys",
    "defaults",
    "agreement",
)


def _optional_str(value: Any, name: str) -> str | None:
    if value is not None and not isinstance(value, str):
        raise ValueError(f"{name} must be a str or None, got {value!r}")
    return value


def _count(value: Any, name: str) -> int:
    """A non-negative int. Bools and floats are refused, NumPy integers become ``int``."""
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    try:
        value = operator.index(value)
    except TypeError:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}") from None
    if value < 0:
        raise ValueError(f"{name} must be a non-negative int, got {value!r}")
    return value


def _strs(values: Any, name: str) -> tuple[str, ...]:
    if not isinstance(values, tuple) or not all(isinstance(v, str) for v in values):
        raise ValueError(f"{name} must be a tuple of str, got {values!r}")
    return values


def _scalar(value: Any, name: str) -> Any:
    """A str, int, float, bool or None. NumPy scalars become Python scalars, arrays are refused."""
    if isinstance(value, np.generic) and np.ndim(value) == 0:
        value = value.item()
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"{name} values must be scalars, got {value!r}")


def _pairs(value: Any, name: str) -> tuple[tuple[str, Any], ...]:
    """Tuple of ``(str, scalar)`` pairs with distinct keys, from a mapping or a tuple of pairs."""
    items = tuple(value.items()) if isinstance(value, Mapping) else value
    if not isinstance(items, tuple):
        raise ValueError(f"{name} must be a tuple of (key, value) pairs or a mapping")
    pairs = []
    for item in items:
        if not (
            isinstance(item, tuple) and len(item) == 2 and isinstance(item[0], str)
        ):
            raise ValueError(f"{name} entries must be (str, value) pairs, got {item!r}")
        pairs.append((item[0], _scalar(item[1], name)))
    keys = [k for k, _ in pairs]
    if len(set(keys)) != len(keys):
        raise ValueError(f"{name} has repeated keys {keys}")
    return tuple(pairs)


def _real(value: Any, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} values must be real numbers, got {value!r}")


def _load_average() -> tuple[float, ...]:
    """The 1, 5 and 15 minute load averages, or ``()`` where the platform does not report them."""
    try:
        return tuple(float(v) for v in os.getloadavg())
    except (AttributeError, OSError):
        return ()


def _versions() -> tuple[tuple[str, str], ...]:
    return (
        ("syncem", __version__),
        ("jax", jax.__version__),
        ("numpy", np.__version__),
        ("equinox", eqx.__version__),
    )


class SpecifiedRecord(eqx.Module):
    """An averaged statistic of a basis: variable, family, parents and fingerprint (static), and parameters.

    ``params`` is a real array leaf of shape ``(n_params,)``, empty by default. ``fingerprint`` is ``None``
    until step 5b-2 computes the rule fingerprint.
    """

    LABEL: ClassVar[str] = "[extension] specified statistic record"
    variable: str = eqx.field(static=True)
    family: str = eqx.field(static=True)
    parents: tuple[str, ...] = eqx.field(static=True)
    fingerprint: str | None = eqx.field(static=True)
    params: Any

    def __init__(
        self,
        variable: str,
        family: str,
        *,
        parents: Any = (),
        fingerprint: str | None = None,
        params: Any = None,
    ):
        if not isinstance(variable, str):
            raise ValueError(f"variable must be a str, got {variable!r}")
        if family not in FAMILIES:
            raise ValueError(f"family must be one of {FAMILIES}, got {family!r}")
        if isinstance(parents, str):
            raise ValueError(
                f"parents must be a sequence of str, got the str {parents!r}"
            )
        parents = tuple(parents)
        _strs(parents, "parents")
        params = jnp.zeros((0,)) if params is None else jnp.asarray(params)
        if params.ndim != 1 or jnp.iscomplexobj(params):
            raise ValueError(
                f"params must be a real array of shape (n_params,), got shape {params.shape} "
                f"and dtype {params.dtype}"
            )
        self.variable, self.family, self.parents = variable, family, parents
        self.fingerprint = _optional_str(fingerprint, "fingerprint")
        self.params = params

    def to_dict(self) -> dict:
        """Strict-JSON dict. ``params`` must be concrete, and a non-finite value becomes ``None``."""
        return strict_json(
            {
                "variable": self.variable,
                "family": self.family,
                "parents": self.parents,
                "fingerprint": self.fingerprint,
                "params": self.params,
            }
        )


@dataclasses.dataclass(frozen=True)
class RouteRecord:
    """The realised provider of one product cell (decision 150).

    ``cell`` is the product-cell index. ``provider`` is one of ``PROVIDERS``. ``mode`` is the stack mode, one of
    ``STACK_MODES`` for the stack and a free label or ``None`` for the jet and the core. ``s1`` is one of
    ``S1_METHODS`` or ``None``. ``reason`` is the refusal text that sent the cell to its provider, or ``None``.
    """

    LABEL: ClassVar[str] = "[extension] route record"
    cell: int
    provider: str
    mode: str | None = None
    s1: str | None = None
    reason: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "cell", _count(self.cell, "cell"))
        if self.provider not in PROVIDERS:
            raise ValueError(
                f"provider must be one of {PROVIDERS}, got {self.provider!r}"
            )
        _optional_str(self.mode, "mode")
        if self.provider == "stack" and self.mode not in STACK_MODES:
            raise ValueError(
                f"a stack route needs a mode of {STACK_MODES}, got {self.mode!r}"
            )
        if self.s1 is not None and self.s1 not in S1_METHODS:
            raise ValueError(f"s1 must be one of {S1_METHODS} or None, got {self.s1!r}")
        _optional_str(self.reason, "reason")

    def to_dict(self) -> dict:
        """Strict-JSON dict of the five fields."""
        return strict_json(dataclasses.asdict(self))


def _check_routes(routes: Any) -> None:
    if not isinstance(routes, tuple) or not all(
        isinstance(r, RouteRecord) for r in routes
    ):
        raise ValueError("routes must be a tuple of RouteRecord")


def _check_stokes(stokes: Any) -> None:
    _strs(stokes, "stokes")
    if len(set(stokes)) != len(stokes) or not set(stokes) <= set(STOKES_BLOCKS):
        raise ValueError(
            f"stokes must hold distinct entries of {STOKES_BLOCKS}, got {stokes!r}"
        )


def _sizes(sizes: Any) -> tuple[tuple[str, int], ...]:
    found = dict(_pairs(sizes, "sizes"))
    if set(found) != set(SIZE_KEYS):
        raise ValueError(f"sizes must have the keys {SIZE_KEYS}, got {tuple(found)}")
    return tuple((k, _count(found[k], f"sizes[{k}]")) for k in SIZE_KEYS)


def _load(load: Any) -> tuple[float, ...]:
    if not isinstance(load, tuple) or len(load) not in (0, 3):
        raise ValueError(
            f"load must be a tuple of three load averages or (), got {load!r}"
        )
    for v in load:
        _real(v, "load")
        if not (np.isfinite(v) and v >= 0):
            raise ValueError(
                f"load averages must be finite and non-negative, got {load!r}"
            )
    return tuple(float(v) for v in load)


@dataclasses.dataclass(frozen=True, kw_only=True)
class BuildRecord:
    """Hashable record of what one basis build did (decisions 149 and 150).

    ``routes`` holds one ``RouteRecord`` per product cell. ``stokes`` is the Stokes request and ``s1`` the S1
    method. ``sizes`` has the keys of ``SIZE_KEYS``. ``events`` counts S2 columns computed and reused, provider
    calls and jet compiles. ``timings`` are seconds, measured at the ``load`` averages taken when the record is
    made unless given. ``keys`` holds the column-map and index keys in step 5b-1. ``defaults`` holds chosen
    defaults such as ``mem_bytes`` and the cache directory. ``agreement`` is empty in step 5. ``notes`` carries
    statements such as "not cached: no fingerprints before 5b-2".
    """

    LABEL: ClassVar[str] = "[extension] build record"
    routes: tuple[RouteRecord, ...]
    stokes: tuple[str, ...]
    s1: str
    sizes: tuple[tuple[str, int], ...]
    events: tuple = ()
    timings: tuple = ()
    load: tuple[float, ...] = dataclasses.field(default_factory=_load_average)
    versions: tuple = dataclasses.field(default_factory=_versions)
    keys: tuple = ()
    defaults: tuple = ()
    agreement: tuple = ()
    notes: tuple[str, ...] = ()

    def __post_init__(self):
        _check_routes(self.routes)
        _check_stokes(self.stokes)
        if self.s1 not in S1_METHODS:
            raise ValueError(f"s1 must be one of {S1_METHODS}, got {self.s1!r}")
        put = object.__setattr__
        put(self, "sizes", _sizes(self.sizes))
        events = _pairs(self.events, "events")
        put(self, "events", tuple((k, _count(v, f"events[{k}]")) for k, v in events))
        timings = _pairs(self.timings, "timings")
        for _, v in timings:
            _real(v, "timings")
        put(self, "timings", timings)
        put(self, "load", _load(self.load))
        versions = _pairs(self.versions, "versions")
        for _, v in versions:
            _optional_str(v, "versions")
        put(self, "versions", versions)
        keys = _pairs(self.keys, "keys")
        for _, v in keys:
            _optional_str(v, "keys")
        put(self, "keys", keys)
        put(self, "defaults", _pairs(self.defaults, "defaults"))
        put(self, "agreement", _pairs(self.agreement, "agreement"))
        _strs(self.notes, "notes")

    def to_dict(self) -> dict:
        """Strict-JSON dict. Pair tuples become dicts, routes a list of dicts, non-finite floats ``None``."""
        out = {}
        for field in dataclasses.fields(self):
            value = getattr(self, field.name)
            if field.name == "routes":
                out[field.name] = [r.to_dict() for r in value]
            elif field.name in _PAIR_FIELDS:
                out[field.name] = strict_json(dict(value))
            else:
                out[field.name] = strict_json(value)
        return out

    def with_notes(self, *notes: str) -> "BuildRecord":
        """New record with ``notes`` appended."""
        return dataclasses.replace(
            self, notes=self.notes + _strs(tuple(notes), "notes")
        )
