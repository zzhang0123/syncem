"""Store and restore a 0.8 spectral basis (SPEC Sec. 8.5).

The NPZ carries a strict JSON manifest and numerical arrays. Saved kernels
are identified by an explicit compatibility fingerprint. A loaded basis
without live declarations can predict supplied Moments, fit and select
saved channel rows.

LABEL [extension].

Stored columns retain their original units and array shapes. A content
digest checks archive integrity. It has no certified information about the
accuracy of the saved response.
"""

from __future__ import annotations

import numpy as np

from ..base.capabilities import Parity, ReferenceDensity
from ..distributions.azimuth import UniformAzimuth
from ..distributions.densities import Isotropic
from ..distributions.screens import NoRotation
from ..lattice.lattice import Lattice
from ..values.errors import DataError
from ._basis_archive import read_archive, write_archive
from ._basis_io_codec import pack, unpack
from ._basis_io_source import (
    ArchivedBasis,
    _KERNEL_FIELDS,
    _source,
    basis_source_key,
    require_rebuild,
)
from ._conditional_archive import describe_conditional, restore_conditional
from ._hybrid_archive import ArchivedHybridBasis, _hybrid_layout, _save_hybrid
from ._keys import design_key
from .assemble import spin0_blocks, stokes_axes
from .basis import SpectralBasis
from .provider_fourier import FourierBasis, fourier_axis
from .single_reference import SingleReferenceBasis

__all__ = ["basis_source_key", "load_basis", "require_rebuild", "save_basis"]


class ArchivedFourierBasis(ArchivedBasis, FourierBasis):
    """Stored angular modes with the archive's declaration checks."""


def _save_lattice(basis):
    lattice = basis.lattice
    return {
        "cells": lattice.cells,
        "weights": lattice.weights,
        "order": lattice.order,
        "depth_in_total": lattice.depth_in_total,
        "explicit_order": lattice.explicit_order,
        "explicit_weights": lattice.explicit_weights,
        "cell_axes": lattice.cell_axes,
        "reps": lattice.reps,
        "variables": lattice._variables,
        "depth_variable": lattice._depth_variable,
        "reference_names": (
            basis.reference_names
            if isinstance(basis, ArchivedBasis)
            else tuple(name for name, _ in lattice._references)
        ),
        "key": lattice.key,
        "index_key": lattice.index.key,
    }


def _load_lattice(meta, averaged):
    requested = set(meta["reference_names"])
    references = tuple(
        (x.variable, x.density)
        for x in averaged.kernel.capabilities
        if isinstance(x, ReferenceDensity) and x.variable in requested
    )
    variables = meta["variables"]
    table = meta["cells"]
    table.validate({v.name: v.kind for v in variables})
    lattice = Lattice(
        order=meta["order"] if meta["explicit_order"] else None,
        weights=meta["weights"],
        depth_in_total=meta["depth_in_total"],
        _table=table,
        _reps=dict(meta["reps"]),
        _variables=variables,
        _depth_variable=meta["depth_variable"],
        _references=references,
    )
    if lattice.key != meta["key"] or lattice.index.key != meta["index_key"]:
        raise DataError("the stored lattice and moment index do not match their keys.")
    # The public constructor correctly regards the supplied numeric weights
    # as explicit. Restore the original declaration metadata after building.
    if type(meta["explicit_weights"]) is not bool:
        raise DataError("invalid stored explicit-weight flag.")
    object.__setattr__(lattice, "explicit_weights", meta["explicit_weights"])
    object.__setattr__(lattice, "cell_axes", meta["cell_axes"])
    return lattice


def _defaults(values):
    """Keep default-law metadata as hashable static records."""

    def record(value):
        if isinstance(value, (str, int, float, bool, tuple)):
            return value
        name = type(value).__name__
        if type(value) in (UniformAzimuth, Isotropic):
            return name, ("nodes", value.nodes)
        if type(value) is NoRotation:
            return name, ("depth", 0.0)
        return name

    return tuple((name, record(value)) for name, value in values)


def _describe_basis(basis):
    kernel = basis.averaged.kernel
    legacy = None
    if isinstance(basis, SingleReferenceBasis):
        legacy = {
            "legacy_provenance": basis.legacy_provenance,
            "legacy_reference": basis.legacy_reference,
            "parity": tuple(
                (p.even, p.odd) for p in kernel.capabilities if isinstance(p, Parity)
            ),
        }
    return {
        "legacy": legacy,
        "hybrid": _save_hybrid(basis),
        "fourier": (
            {"m_max": basis.m_max, "refined_delta": basis.refined_delta}
            if isinstance(basis, FourierBasis)
            else None
        ),
        "key": basis.key,
        "source_key": basis_source_key(basis.averaged),
        "kernel": {field: getattr(kernel, field, False) for field in _KERNEL_FIELDS},
        "lattice": _save_lattice(basis),
        "channels": basis.channels,
        "channel_keys": basis.channels.row_keys(),
        "columns": basis.columns,
        "spectra0": basis.spectra0,
        "spectra2": basis.spectra2,
        "azimuth_factor": basis.azimuth_factor,
        "specified": basis.specified,
        "rules": basis.rules,
        "record": basis.record,
        "ledger": basis.ledger,
        "defaults": _defaults(basis.forward_defaults),
        "depth_tol": basis.forward_depth_tol,
    }


def save_basis(basis, path):
    """Save a prebuilt spectral basis to the requested NPZ path."""
    if not isinstance(basis, SpectralBasis):
        raise TypeError("save_basis expects a SpectralBasis.")
    if getattr(basis, "_archive_kind", None) == "conditional":
        body = {"conditional": describe_conditional(basis)}
    else:
        body = _describe_basis(basis)
    arrays = {}
    write_archive(path, pack(body, arrays), arrays)


def _indices(values, length, title, *, shape=None, sentinel=False):
    if (
        values.dtype.kind not in "iu"
        or values.ndim != 1
        or (shape is not None and values.shape != shape)
        or np.any(values < (-1 if sentinel else 0))
        or np.any(values >= length)
    ):
        raise DataError(f"invalid stored {title} indices.")


def _column_layout(columns, lattice):
    n, rows = columns.n, lattice.index.n
    pointers = columns.col_start
    members = columns.col_rows
    if (
        pointers.ndim != 1
        or pointers.dtype.kind not in "iu"
        or not len(pointers)
        or pointers[0] != 0
        or pointers[-1] != members.size
        or np.any(pointers[1:] <= pointers[:-1])
    ):
        raise DataError("invalid stored column pointers.")
    _indices(columns.col_of_row, n, "lattice-to-column", shape=(rows,), sentinel=True)
    _indices(members, rows, "column membership")
    if (
        np.unique(members).size != members.size
        or np.count_nonzero(columns.col_of_row >= 0) != members.size
        or not np.array_equal(
            columns.col_of_row[members], np.repeat(np.arange(n), np.diff(pointers))
        )
    ):
        raise DataError("stored column membership disagrees with the lattice row map.")
    index = lattice.index
    _indices(columns.column_block, len(index.blocks), "column block", shape=(n,))
    _indices(columns.column_cell, lattice.cells.n_cells, "column cell", shape=(n,))
    _indices(columns.column_base, index.block_size, "column base", shape=(n,))
    for block, size in (("spin0", columns.n0), ("spin2", columns.n2)):
        _indices(getattr(columns, f"{block}_base"), index.block_size, block + " base")
        _indices(
            getattr(columns, f"{block}_from"),
            size,
            block + " source",
            shape=(n,),
            sentinel=True,
        )
    for block, used in columns.uses.items():
        _indices(used, n, f"{block} use")
    for reason, affected in columns.reasons:
        _indices(affected, rows, reason)


def _check_layout(body, lattice):
    channels, columns, record = body["channels"], body["columns"], body["record"]
    _hybrid_layout(body, lattice)
    if channels.row_keys() != body["channel_keys"]:
        raise DataError("the stored channels do not match their row keys.")
    geometry = (channels.n_ch, channels.n_nu)
    if channels.nodes.shape != geometry or channels.weights.shape != geometry:
        raise DataError("the stored channel quadrature shapes are inconsistent.")
    _column_layout(columns, lattice)
    sizes = dict(record.sizes)
    if sizes != dict(n=columns.n, n_rows=lattice.index.n, n0=columns.n0, n2=columns.n2):
        raise DataError("the stored basis sizes disagree with the column map.")
    if record.stokes != stokes_axes(columns.stokes):
        raise DataError("the stored Stokes axes do not match the column map.")
    keys = dict(record.keys)
    expected = {
        "column map": columns.key,
        "index": lattice.index.key,
        "lattice": lattice.key,
    }
    if any(keys.get(k) != value for k, value in expected.items()):
        raise DataError(
            "the stored record keys disagree with the lattice or column map."
        )
    key = design_key(keys.get("memo"), lattice.key, body["azimuth_factor"])
    if key != body["key"] or keys.get("design") != key:
        raise DataError("the stored design key does not match the basis metadata.")
    i, p = body["spectra0"], body["spectra2"]
    fourier = body["fourier"]
    width, dtype = (), "f"
    angular = fourier_axis(lattice, body["kernel"]["azimuth"])
    if (angular is None) != (fourier is None):
        raise DataError("Fourier metadata needs the corresponding lattice axis.")
    if angular is not None:
        order = fourier["m_max"]
        if type(order) is not int or angular.m_max != order:
            raise DataError("stored Fourier degree does not match the lattice.")
        width, dtype = (2 * order + 1,), "c"
        delta = fourier["refined_delta"]
        spectra = (i,) if p is None else (i, p)
        if len(delta) != len(spectra) or any(
            d.shape != s.shape or d.dtype != s.dtype for d, s in zip(delta, spectra)
        ):
            raise DataError("invalid Fourier refinement array shape or dtype.")
    if (
        i.shape
        != (channels.n_ch, len(spin0_blocks(columns.stokes)), columns.n0, *width)
        or i.dtype.kind != dtype
    ):
        raise DataError("invalid spin-0 spectral array shape or dtype.")
    if "P" in columns.stokes and (
        p is None
        or p.shape != (channels.n_ch, columns.n2, *width)
        or p.dtype.kind != "c"
    ):
        raise DataError("invalid spin-2 spectral array shape or dtype.")
    if "P" not in columns.stokes and p is not None:
        raise DataError("unexpected spin-2 spectra in the stored basis.")


def load_basis(path, *, averaged=None, registry=None):
    """Load an archived basis, optionally attaching an explicitly verified live source."""
    payload, arrays = read_archive(path)
    try:
        body = unpack(payload, arrays)
        if set(body) == {"conditional"}:
            if averaged is not None or registry is not None:
                raise DataError(
                    "a conditional archive supports only stored channels. Rebuild from the original conditional declaration."
                )
            return restore_conditional(body["conditional"])
        return _restore_basis(body, averaged, registry)
    except DataError:
        raise
    except (AttributeError, KeyError, ValueError, TypeError) as exc:
        raise DataError(f"invalid stored spectral basis metadata. {exc}") from exc


def _restore_basis(body, averaged=None, registry=None):
    checked = body.get("hybrid") is not None and (
        averaged is not None or registry is not None
    )
    if checked:
        # Validate the saved geometry and routes before a caller-supplied
        # assignment function can inspect the cells.
        stored = _load_lattice(body["lattice"], _source(body, None, None))
        _check_layout(body, stored)
        del stored
    source = _source(body, averaged, registry)
    lattice = _load_lattice(body["lattice"], source)
    if not checked:
        _check_layout(body, lattice)
    fields = {
        key: body[key]
        for key in (
            "channels",
            "columns",
            "spectra0",
            "spectra2",
            "azimuth_factor",
            "specified",
            "record",
            "ledger",
            "key",
            "rules",
        )
    }
    legacy = body["legacy"]
    cls = ArchivedBasis if legacy is None else SingleReferenceBasis
    if legacy is None:
        fields["reference_names"] = body["lattice"]["reference_names"]
        if body["fourier"] is not None:
            cls = ArchivedFourierBasis
            fields.update(body["fourier"])
        elif body.get("hybrid") is not None:
            cls = ArchivedHybridBasis
            fields["hybrid_metadata"] = tuple(body["hybrid"].items())
    else:
        fields.update(
            {name: legacy[name] for name in ("legacy_provenance", "legacy_reference")}
        )
    return cls(
        lattice=lattice,
        averaged=source,
        forward_defaults=body["defaults"],
        forward_depth_tol=body["depth_tol"],
        **fields,
    )
