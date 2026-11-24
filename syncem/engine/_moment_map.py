"""Numerical R3a contraction shared by live and stored inverse bases."""

from __future__ import annotations

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..base._fingerprint import digest
from ..values.errors import SizeError

_DESIGN_BYTES = 32 * 2**20


class ConditionalMap(eqx.Module):
    """A sparse map from scaled parent moments to scaled joint moments."""

    joint_rows: jax.Array
    parent_rows: jax.Array
    coefficients: jax.Array
    n_joint: int = eqx.field(static=True)
    n_parent: int = eqx.field(static=True)
    key: str = eqx.field(static=True)

    def __init__(self, rows, columns, values, n_joint, n_parent):
        rows, columns, values = (
            np.asarray(rows, dtype=int),
            np.asarray(columns, dtype=int),
            np.asarray(values, dtype=float),
        )
        self.key = digest(("R3a map", rows, columns, values, n_joint, n_parent))
        self.joint_rows, self.parent_rows, self.coefficients = map(
            jnp.asarray, (rows, columns, values)
        )
        self.n_joint, self.n_parent = n_joint, n_parent

    def apply(self, parent):
        """Apply the moment map to one parent vector or a batch of vectors."""
        parent = jnp.asarray(parent)
        if parent.shape[-1:] != (self.n_parent,):
            raise ValueError(f"parent moments must end in {self.n_parent} rows")
        return self._contract(parent, self.parent_rows, self.joint_rows, self.n_joint)

    def design(self, joint):
        """Contract a joint design into parent rows without a dense moment matrix."""
        if joint.shape[-1:] != (self.n_joint,):
            raise ValueError(f"the joint design must end in {self.n_joint} rows")
        return self._contract(joint, self.joint_rows, self.parent_rows, self.n_parent)

    def _contract(self, values, src, dest, width):
        dtype = jnp.result_type(values, self.coefficients)
        leading = values.shape[:-1]
        result = jnp.zeros((*leading, width), dtype=dtype)
        count = self.coefficients.size
        if count == 0:
            return result
        bytes_per_pair = 2 * max(1, math.prod(leading)) * dtype.itemsize
        step = min(count, _DESIGN_BYTES // bytes_per_pair)
        if not step:
            raise SizeError("R3a contraction budget cannot hold one sparse coefficient")
        n = math.ceil(count / step)
        tail = n * step - count
        rows, columns, coeffs = (
            jnp.pad(a, (0, tail)) for a in (src, dest, self.coefficients)
        )

        def add(i, acc):
            start = i * step
            r, c, coefficients = (
                jax.lax.dynamic_slice_in_dim(a, start, step)
                for a in (rows, columns, coeffs)
            )
            return acc.at[..., c].add(values[..., r] * coefficients)

        return jax.lax.fori_loop(0, n, add, result)


class ResidualSample(eqx.Module):
    """Joint moment errors sampled within one parent cell."""

    values: np.ndarray
    parent_cell: int = eqx.field(static=True)
    first_row: int = eqx.field(static=True)


def residual_estimate(joint, parent_cells, residuals):
    """Channel residual estimate per unit parent column at sampled points."""
    design = np.asarray(joint.columns.lift(joint.design()))
    sampled = {}
    for part in residuals:
        first, values = part.first_row, np.asarray(part.values)
        sub = design[..., first : first + values.shape[1]]
        estimate = np.tensordot(values, sub, axes=([-1], [-1]))
        prior = sampled.get(part.parent_cell)
        sampled[part.parent_cell] = estimate if prior is None else prior + estimate
    out = np.full((parent_cells, *design.shape[:-1]), np.nan)
    for cell, err in sampled.items():
        out[cell] = np.max(np.abs(err), axis=0)
    return out


def mapped_design_key(joint, lattice, mapping):
    return (
        None
        if joint.key is None
        else digest(("R3a basis", joint.key, lattice.key, mapping.key))
    )
