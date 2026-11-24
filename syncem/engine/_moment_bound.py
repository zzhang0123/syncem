"""Pre-data column-space envelopes for a sparse conditional moment map.

LABEL [extension]. All quantities use the canonical Stokes units of the
spectral basis. The per-cell envelope assumes a positive parent measure
and a child probability measure on its full declared support.
"""

from math import prod
from numbers import Integral

import jax
import numpy as np

from ..records.ledger import ErrorTerm
from ..values.errors import SizeError

BOUND_BYTES = 128 * 2**20
_EPS = np.finfo(float).eps
_TINY = np.finfo(float).tiny
_REFERENCE = "unit parent column in each cell"


def _checked_budget(chunk_bytes):
    if (
        isinstance(chunk_bytes, bool)
        or not isinstance(chunk_bytes, Integral)
        or chunk_bytes <= 0
    ):
        raise ValueError("chunk_bytes must be a positive integer")
    return int(chunk_bytes)


def _workspace(basis, chunk_bytes):
    chunk_bytes = _checked_budget(chunk_bytes)
    n_out = basis.channels.n_ch * len(basis.record.stokes)
    n_map = basis.mapping.coefficients.size
    n_cols = basis.joint.columns.n
    # Charge the unlifted design and its assembly buffers, host sparse
    # arrays and stable sort, plus the returned envelope.
    fixed = 8 * (6 * n_out * n_cols + 6 * n_map + 2 * basis.lattice.n_cells * n_out)
    dimensions = len(basis.joint.lattice.cells.variables)
    per_row = 8 * (16 * n_out + 5 * dimensions + 16)
    count = (chunk_bytes - fixed) // per_row
    if count < 1:
        raise SizeError(
            f"conditional bound needs at least {fixed + per_row:,} bytes of workspace. "
            "Increase chunk_bytes."
        )
    return min(count, 4096)


def _radii(table, cell):
    frames = np.stack((table.lo[cell], table.hi[cell], table.reference[cell]))
    if not np.all(np.isfinite(frames)):
        return None
    distances = np.abs(frames[:2] - frames[2])
    for i, chart in enumerate(table.chart[cell]):
        if chart == "log":
            face, ref = frames[:2, i], frames[2, i]
            if min(face.min(), ref) <= 0:
                return None
            gap = distances[:, i] / np.minimum(face, ref)
            with np.errstate(over="ignore"):
                wide = np.abs(np.log(face) - np.log(ref))
                distances[:, i] = np.where(gap < 0.5, np.log1p(gap), wide)
                # Away from the reference, include rounding of both logs.
                distances[:, i] += np.where(
                    gap < 0.5,
                    0,
                    8 * _EPS * (np.abs(np.log(face)) + abs(np.log(ref))),
                )
    radius = distances.max(axis=0) / table.half_width[cell]
    return np.nextafter(radius * (1 + 8 * _EPS), np.inf)


def _blocks(start, stop, step):
    for lo in range(start, stop, step):
        yield lo, min(lo + step, stop)


def _upper_sum(values, scale):
    # Both arguments are nonnegative. Allow roundoff in the row products
    # and the positive reduction.
    gain = 1 - (4 * len(scale) + 4) * _EPS
    out = np.sum(values * scale[:, None], axis=0)
    return np.nextafter((out + _TINY * len(scale)) / gain, np.inf)


def _suprema(lattice, cell, start, stop, radii):
    index = lattice.index
    pattern = index.patterns[int(index.cell_pattern[cell])]
    first = index.cell_rows(cell).start
    powers = np.asarray(
        [pattern[(row - first) % len(pattern)] for row in range(start, stop)],
        dtype=int,
    )
    power = np.prod(radii[None, :] ** powers, axis=-1)
    # Each unnormalised Legendre factor has absolute value at most one.
    scale = power / lattice.weights[cell]
    gain = 1 + (int(np.sum(index.table.order[cell])) + 8) * _EPS
    return np.nextafter(scale * gain, np.inf)


def _stored_rows(design, column_map, rows):
    columns = column_map[rows]
    if design.shape[-1] == 0:
        return np.zeros((columns.size, design.shape[0]))
    out = design[:, np.maximum(columns, 0)].T
    out[columns < 0] = 0
    return out


def _mapped_rows(design, column_map, sparse, lo, hi, step):
    order, keys, source, coefficients = sparse
    low, high = np.searchsorted(keys, (lo, hi))
    mapped = np.zeros((hi - lo, design.shape[0]))
    magnitude = np.zeros_like(mapped)
    for first, last in _blocks(low, high, step):
        ids = order[first:last]
        offsets = keys[first:last] - lo
        values = _stored_rows(design, column_map, source[ids])
        values *= coefficients[ids, None]
        np.add.at(mapped, offsets, values)
        np.add.at(magnitude, offsets, np.abs(values))
    # Enclose the signed sum even when its final value has cancellation.
    eps = 4 * (high - low + 1) * _EPS
    correction = (eps / (1 - eps)) * magnitude + _TINY * (high - low)
    return np.nextafter(np.abs(mapped) + correction, np.inf)


def _direct_max(joint, parent, design, columns, step, result):
    for c in range(joint.n_cells):
        coordinates = joint.cells.cell_numbers(c)
        b = int(
            np.ravel_multi_index(
                tuple(coordinates[name] for name in parent.cells.variables),
                parent.cells.tensor,
            )
        )
        radii = _radii(joint.cells, c)
        if radii is None:
            result[b] = np.inf
            continue
        cell_rows = joint.index.cell_rows(c)
        ceiling = np.zeros(design.shape[0])
        for lo, hi in _blocks(cell_rows.start, cell_rows.stop, step):
            scale = _suprema(joint, c, lo, hi, radii)
            partial = _upper_sum(
                np.abs(_stored_rows(design, columns, np.arange(lo, hi))), scale
            )
            ceiling = np.nextafter(ceiling + partial, np.inf)
        # Child boxes form a partition and their combined probability is
        # at most one. The bound on their weighted sum is their maximum.
        np.maximum(result[b], ceiling, out=result[b])


def _envelope(basis, chunk_bytes):
    step = _workspace(basis, chunk_bytes)
    out_shape = basis.channels.n_ch, len(basis.record.stokes)
    joint, parent = basis.joint.lattice, basis.lattice
    child_axis = dict(joint.cell_axes)[basis.child]
    if child_axis.edges is None and child_axis.n > 1:
        # Several unpartitioned charts cannot share a probability budget.
        return np.full((parent.n_cells, *out_shape), np.inf)
    design = np.asarray(basis.joint.design(), dtype=float).reshape(
        out_shape[0] * out_shape[1], basis.joint.columns.n
    )
    columns = basis.joint.columns.col_of_row
    destinations = np.asarray(basis.mapping.parent_rows)
    order = np.argsort(destinations, kind="stable")
    sparse = (
        order,
        destinations[order],
        np.asarray(basis.mapping.joint_rows),
        np.asarray(basis.mapping.coefficients),
    )
    out = np.zeros((parent.n_cells, design.shape[0]))
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        _direct_max(joint, parent, design, columns, step, out)
        for b in range(parent.n_cells):
            radii = _radii(parent.cells, b)
            if radii is None:
                out[b] = np.inf
                continue
            cell_rows = parent.index.cell_rows(b)
            for lo, hi in _blocks(cell_rows.start, cell_rows.stop, step):
                scale = _suprema(parent, b, lo, hi, radii)
                mapped = _mapped_rows(design, columns, sparse, lo, hi, step)
                out[b] = np.nextafter(out[b] + _upper_sum(mapped, scale), np.inf)
    out[~np.isfinite(out)] = np.inf
    return out.reshape(parent.n_cells, *out_shape)


def _term_name(basis):
    return next(
        t.name
        for t in basis.ledger.terms
        if t.name.startswith("conditional_truncation:")
    )


def bound_term(basis, chunk_bytes):
    """Return a relative, pre-data bound for a unit parent column per cell."""
    name = _term_name(basis)
    try:
        value = _envelope(basis, chunk_bytes)
    except SizeError as exc:
        return ErrorTerm(None, "unbounded", str(exc), name=name, relative_to=_REFERENCE)
    if not np.all(np.isfinite(value)):
        return ErrorTerm(
            None,
            "unbounded",
            "a parent or joint cell has no finite envelope",
            name=name,
            relative_to=_REFERENCE,
        )
    return ErrorTerm(
        value,
        "bound",
        "A bound on the stored basis polynomials with a positive parent and "
        "a child probability measure on its full support. It applies on the "
        "child cell boxes.",
        name=name,
        relative_to=_REFERENCE,
    )


def fitted_bound_term(basis, moments, chunk_bytes):
    """Estimate a channel error by plugging fitted masses into the bound."""
    basis.lattice._own(moments)
    name = _term_name(basis)
    if isinstance(moments.values, jax.core.Tracer):
        return ErrorTerm.unbounded(
            "the conditional bound is evaluated on the host", name=name
        )
    lattice, values = basis.lattice, moments.values
    shape = np.shape(values)
    pixels = prod(shape[:-1])
    output = basis.channels.n_ch * len(basis.record.stokes)
    # Count the gathered zeroth rows, their host weight copy, and the
    # channel result in both NumPy and the final ErrorTerm.
    reserved = 8 * (4 * pixels * (lattice.n_cells + output) + lattice.n_cells)
    if not hasattr(values, "shape"):
        reserved += 8 * prod(shape)
    available = _checked_budget(chunk_bytes) - reserved
    if available < 1:
        return ErrorTerm.unbounded(
            f"the fitted-mass diagnostic needs {reserved:,} bytes of batch workspace",
            name=name,
        )
    try:
        envelope = _envelope(basis, available)
    except SizeError as exc:
        return ErrorTerm.unbounded(str(exc), name=name)
    rows = lattice.index.n_pairs * lattice.index.offsets[:-1]
    if not hasattr(values, "shape"):
        values = np.asarray(values)
    weights = np.abs(np.asarray(values[..., rows], dtype=float))
    weights *= lattice.weights
    bad = ~np.all(np.isfinite(envelope), axis=(1, 2))
    if not np.all(np.isfinite(weights)) or np.any(weights[..., bad] != 0):
        return ErrorTerm.unbounded(
            "a participating fitted cell has no finite envelope", name=name
        )
    with np.errstate(over="ignore", invalid="ignore"):
        value = np.einsum("...b,bcs->...cs", weights[..., ~bad], envelope[~bad])
    if not np.all(np.isfinite(value)):
        return ErrorTerm.unbounded(
            "the fitted-mass error estimate is nonfinite", name=name
        )
    return ErrorTerm(
        value,
        "estimate",
        "Per-unit conditional bounds weighted by the fitted zeroth moments. "
        "The estimate refers to fitted masses.",
        name=name,
    )
