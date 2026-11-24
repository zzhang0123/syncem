"""Strict record and channel codec for the spectral basis archive (SPEC Sec. 8.5).

Only the record types named here are constructed from metadata. Callable
responses are stored as quadrature snapshots with their original row keys.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import fields, replace
from types import MappingProxyType

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..base.channels import Channels
from ..base.variables import Variable
from ..lattice.cells import Cells
from ..lattice.representations import AzimuthFourier, AzimuthHarmonic, Legendre
from ..lattice.table import CellTable
from ..model._provenance import AssumptionRecord, Provenance
from ..records.ledger import ErrorTerm, Ledger
from ..records.records import BuildRecord, RouteRecord, SpecifiedRecord
from ..values.errors import DataError
from ._hybrid_basis import HybridRouteRecord
from .columns import ColumnMap, ColumnRules

_RECORD_TYPES = (
    AssumptionRecord,
    AzimuthFourier,
    AzimuthHarmonic,
    BuildRecord,
    Cells,
    CellTable,
    ColumnRules,
    ColumnMap,
    ErrorTerm,
    HybridRouteRecord,
    Ledger,
    Legendre,
    Provenance,
    RouteRecord,
    SpecifiedRecord,
    Variable,
)
_TAGS = {cls.__name__: cls for cls in _RECORD_TYPES}
_TYPES = {cls: tag for tag, cls in _TAGS.items()}


class StoredChannels(Channels):
    """Stored nodes and responses of a channel family with live functions."""

    saved_measure: jax.Array = eqx.field(kw_only=True)
    saved_values: jax.Array = eqx.field(kw_only=True)
    saved_peak: jax.Array = eqx.field(kw_only=True)
    saved_keys: tuple[str, ...] = eqx.field(static=True, kw_only=True)
    saved_edge: float = eqx.field(static=True, kw_only=True)

    def __call__(self, nu_hz):
        """Refuse continuous evaluation when the response function is absent."""
        raise DataError(
            "loaded channels have only their stored quadrature response. "
            "Use the original channel declaration to evaluate other frequencies."
        )

    def node_values(self):
        """Stored response samples at the original nodes."""
        return self.saved_values

    def response_weights(self):
        """Stored channel integration weights, including the response."""
        return self.saved_measure

    def row_keys(self):
        """The original row keys of the complete response quadrature."""
        return self.saved_keys

    def peak_scale(self):
        """The original channel normalisation factors."""
        return self.saved_peak

    def edge_jump(self):
        """The archived edge jump of the response."""
        return self.saved_edge

    def derivative_bound_reason(self, order):
        """A quadrature snapshot cannot verify a derivative bound."""
        return "response derivatives need the original callable channel declaration"

    def __getitem__(self, rows):
        index = np.atleast_1d(np.arange(self.n_ch)[rows])
        base = super().__getitem__(index)
        return replace(
            base,
            saved_measure=self.saved_measure[index],
            saved_values=self.saved_values[index],
            saved_peak=self.saved_peak[index],
            saved_keys=tuple(self.saved_keys[i] for i in index),
        )

    def select(self, j):
        """Keep one snapshot row with its archived quadrature."""
        return self[j]


def _analytic(channels):
    return (
        type(channels) is Channels
        and channels.response_fn is None
        and (
            channels.reference_channels is None
            or _analytic(channels.reference_channels)
        )
    )


def _pack_channels(value, arrays):
    attrs = {field.name: getattr(value, field.name) for field in fields(Channels)}
    tag = "Channels"
    if not _analytic(value):
        tag = "StoredChannels"
        attrs.update(
            reference_channels=None,
            response_fn=None,
            derivative_bound=None,
            table=None,
            coefficients=None,
            saved_measure=value.response_weights(),
            saved_values=value.node_values(),
            saved_peak=value.peak_scale(),
            saved_keys=value.row_keys(),
            saved_edge=value.edge_jump(),
        )
    return {"record": tag, "fields": pack(attrs, arrays)}


def _pack_array(value, arrays):
    a = np.asarray(value)
    if a.dtype.kind not in "biufc":
        raise DataError(f"basis arrays must have numerical dtype, got {a.dtype}.")
    name = f"a{len(arrays)}"
    arrays[name] = a if a.flags.c_contiguous else np.ascontiguousarray(a)
    return {"array": name}


def pack(value, arrays):
    """Encode safe metadata and record references into plain JSON types."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, np.generic):
        return pack(value.item(), arrays)
    if isinstance(value, float):
        return value if math.isfinite(value) else {"float": repr(value)}
    if isinstance(value, complex):
        return {"complex": [pack(value.real, arrays), pack(value.imag, arrays)]}
    if isinstance(value, (np.ndarray, jax.Array)):
        return _pack_array(value, arrays)
    if isinstance(value, tuple):
        return {"tuple": [pack(item, arrays) for item in value]}
    if isinstance(value, list):
        return [pack(item, arrays) for item in value]
    if isinstance(value, Mapping):
        if not all(isinstance(k, str) for k in value):
            raise DataError("basis metadata mapping keys must be strings.")
        return {"map": {k: pack(v, arrays) for k, v in value.items()}}
    if isinstance(value, Channels):
        return _pack_channels(value, arrays)
    cls = type(value)
    if cls in _TYPES:
        attrs = {field.name: getattr(value, field.name) for field in fields(cls)}
        return {"record": _TYPES[cls], "fields": pack(attrs, arrays)}
    raise DataError(f"basis archive cannot encode {cls.__qualname__}.")


def unpack(value, arrays):
    """Decode metadata without imports, executable code or object arrays."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataError("basis metadata numbers must be finite.")
        return value
    if isinstance(value, list):
        return [unpack(item, arrays) for item in value]
    if not isinstance(value, dict):
        raise DataError("invalid basis metadata value.")
    if set(value) == {"record", "fields"}:
        tag = value["record"]
        if tag == "Channels":
            cls = Channels
        elif tag == "StoredChannels":
            cls = StoredChannels
        else:
            cls = _TAGS[tag]
        attrs = unpack(value["fields"], arrays)
        if cls is ColumnMap:
            attrs["uses"] = MappingProxyType(attrs["uses"])
        if issubclass(cls, Channels):
            attrs = {
                k: jnp.asarray(v) if isinstance(v, np.ndarray) else v
                for k, v in attrs.items()
            }
        return cls(**attrs)
    if len(value) != 1:
        raise DataError("invalid basis metadata tag.")
    tag, content = next(iter(value.items()))
    if tag == "array":
        return arrays[content]
    if tag == "tuple":
        return tuple(unpack(item, arrays) for item in content)
    if tag == "map":
        return {key: unpack(item, arrays) for key, item in content.items()}
    if tag == "complex":
        real, imag = (unpack(i, arrays) for i in content)
        return complex(real, imag)
    if tag == "float" and content in ("inf", "-inf", "nan"):
        return float(content)
    raise DataError(f"unrecognised metadata tag {tag!r}.")
