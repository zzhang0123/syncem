"""Stored parent-to-joint inverse bases without portable Python declarations."""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import equinox as eqx
import numpy as np

from ..values.errors import DataError
from ._basis_io_source import ArchivedBasis
from ._basis_ledger import channel_ledger
from ._basis_update import match_channels, record_sizes
from ._moment_bound import BOUND_BYTES, bound_term, fitted_bound_term
from ._moment_map import (
    ConditionalMap,
    ResidualSample,
    mapped_design_key,
    residual_estimate,
)
from ._spectral_basis import SpectralBasis
from .assemble import stokes_axes

_STORED = "a conditional archive supports only stored channels and order. Rebuild from the original conditional declaration."


class ArchivedConditionalBasis(ArchivedBasis):
    """A saved sparse inverse basis for supplied parent Moments."""

    _archive_kind: ClassVar[str] = "conditional"

    joint: SpectralBasis = eqx.field(kw_only=True)
    mapping: ConditionalMap = eqx.field(kw_only=True)
    residuals: tuple = eqx.field(kw_only=True)
    child: str = eqx.field(static=True, kw_only=True)
    nodes: int = eqx.field(static=True, kw_only=True)

    def design(self, data=None, *, stokes=None):
        if data is not None:
            return self._observed_design(data, stokes)
        joint = self.joint.columns.lift(self.joint.design(stokes=stokes))
        return self.mapping.design(joint)

    def at(self, channels):
        found, missing = match_channels(self.channels.row_keys(), channels.row_keys())
        if missing:
            raise DataError(_STORED)
        joint = self.joint.at(channels)
        keys = {
            **dict(self.record.keys),
            "design": mapped_design_key(joint, self.lattice, self.mapping),
        }
        record = replace(
            self.record,
            keys=keys,
            routes=joint.record.routes,
            events=joint.record.events,
            timings=joint.record.timings,
            load=joint.record.load,
        )
        return replace(
            self,
            joint=joint,
            channels=channels,
            key=keys["design"],
            record=record,
            ledger=channel_ledger(self, channels, found),
        )

    def with_order(self, order):
        raise DataError(_STORED)

    def conditional_truncation(self):
        """Per-cell estimates evaluated with the archived joint spectra."""
        return residual_estimate(self.joint, self.lattice.n_cells, self.residuals)

    def conditional_truncation_bound(self, *, chunk_bytes=BOUND_BYTES):
        """Bound the conditional map error per unit parent column in each cell."""
        return bound_term(self, chunk_bytes)

    def conditional_error(self, moments, *, chunk_bytes=BOUND_BYTES):
        """Estimate the conditional error at supplied fitted cell masses."""
        return fitted_bound_term(self, moments, chunk_bytes)


def describe_conditional(basis):
    from .basis_io import _defaults, _describe_basis, _save_lattice

    mapping = basis.mapping
    return {
        "joint": _describe_basis(basis.joint),
        "lattice": _save_lattice(basis),
        "columns": basis.columns,
        "rules": basis.rules,
        "key": basis.key,
        "record": basis.record,
        "ledger": basis.ledger,
        "defaults": _defaults(basis.forward_defaults),
        "child": basis.child,
        "nodes": basis.nodes,
        "map_key": mapping.key,
        "joint_rows": mapping.joint_rows,
        "parent_rows": mapping.parent_rows,
        "coefficients": mapping.coefficients,
        "residuals": tuple(
            (r.parent_cell, r.first_row, r.values) for r in basis.residuals
        ),
    }


def _validated_map(body, joint, parent):
    from .basis_io import _indices

    indices = body["joint_rows"], body["parent_rows"]
    coefficients = body["coefficients"]
    _indices(indices[0], joint.n, "conditional joint row")
    _indices(indices[1], parent.n, "conditional parent row", shape=indices[0].shape)
    if (
        coefficients.shape != indices[0].shape
        or coefficients.dtype.kind != "f"
        or not np.all(np.isfinite(coefficients))
    ):
        raise DataError("invalid stored conditional coefficients.")
    mapping = ConditionalMap(*indices, coefficients, joint.n, parent.n)
    if mapping.key != body["map_key"]:
        raise DataError("the stored conditional moment map does not match its key.")
    return mapping


def _validated_residuals(body, joint, parent):
    starts = {
        r.start: len(r)
        for cell in range(joint.n_cells)
        for r in (joint.index.cell_rows(cell),)
    }
    residuals, counts = [], {}
    for cell, start, values in body["residuals"]:
        if (
            type(cell) is not int
            or not 0 <= cell < parent.n_cells
            or type(start) is not int
            or start not in starts
            or values.ndim != 2
            or values.dtype.kind != "f"
            or values.shape[0] < 1
            or values.shape[1] != starts[start]
            or not np.all(np.isfinite(values))
            or counts.get(cell, values.shape[0]) != values.shape[0]
        ):
            raise DataError("invalid stored conditional residual samples.")
        counts[cell] = values.shape[0]
        residuals.append(ResidualSample(values, cell, start))
    return tuple(residuals)


def restore_conditional(body):
    from .basis_io import _column_layout, _load_lattice, _restore_basis

    joint = _restore_basis(body["joint"])
    lattice = _load_lattice(body["lattice"], joint.averaged)
    columns, record = body["columns"], body["record"]
    _column_layout(columns, lattice)
    child, nodes = body["child"], body["nodes"]
    if (
        not isinstance(child, str)
        or child not in joint.lattice.cells.variables
        or set(joint.lattice.cells.variables) - {child} != set(lattice.cells.variables)
        or joint.lattice.reps != lattice.reps
        or type(nodes) is not int
        or nodes < 1
        or not np.array_equal(columns.col_of_row, np.arange(lattice.n))
        or dict(record.sizes) != record_sizes(columns)
        or record.stokes != stokes_axes(columns.stokes)
        or record.stokes != joint.record.stokes
        or record.routes != joint.record.routes
    ):
        raise DataError(
            "the stored conditional record disagrees with its parent and joint layouts."
        )
    mapping = _validated_map(body, joint.lattice, lattice)
    residuals = _validated_residuals(body, joint.lattice, lattice)
    key = mapped_design_key(joint, lattice, mapping)
    identity = {"lattice": lattice.key, "R3a map": mapping.key, "design": key}
    if body["key"] != key or dict(record.keys) != identity:
        raise DataError(
            "the stored conditional lattice and map disagree with the design key."
        )
    return ArchivedConditionalBasis(
        lattice=lattice,
        channels=joint.channels,
        columns=columns,
        spectra0=None,
        spectra2=None,
        azimuth_factor=joint.azimuth_factor,
        specified=joint.specified,
        record=record,
        ledger=body["ledger"],
        key=key,
        averaged=joint.averaged,
        rules=body["rules"],
        forward_defaults=body["defaults"],
        forward_depth_tol=joint.forward_depth_tol,
        reference_names=body["lattice"]["reference_names"],
        joint=joint,
        mapping=mapping,
        residuals=residuals,
        child=child,
        nodes=nodes,
    )
