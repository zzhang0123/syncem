"""A fixed inverse basis for a bounded conditional child of unknown cell moments.

The sparse R3a map sends parent moment rows to joint rows before applying
the physical columns. Its shifted terms are kept when the resulting parent
exponent is available in the parent lattice. Error on parent sample grids
is an estimate per unit parent column, not a bound on the inferred source.

LABEL [extension].

The design has shape (n_ch, n_stokes, n_parent_row). The map carries the
column and reference-weight units of the joint basis.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import equinox as eqx
import numpy as np

from ..base._fingerprint import digest
from ..engine._basis_update import record_sizes
from ..engine._moment_bound import BOUND_BYTES, bound_term, fitted_bound_term
from ..engine._moment_map import ResidualSample, mapped_design_key, residual_estimate
from ..engine.basis import SpectralBasis
from ..engine.columns import ColumnRules, column_map
from ..lattice.lattice import Lattice
from ..lattice.moments import Moments
from ..lattice.representations import Legendre
from ..population.declaration import parents
from ..records.ledger import ErrorTerm, Ledger
from ..values.errors import RoleError, SizeError
from ._conditional_legendre import build_legendre_map
from ._conditional_map import ConditionalMap, _moving_child_axis, build_map

__all__ = ["ConditionalBasis", "conditional_basis"]


def _project_lattice(joint, child, variables, weights):
    if (
        joint.cell_axes is None
        or len(joint.reps) > 1
        or any(not isinstance(rep, Legendre) for _, rep in joint.reps)
    ):
        raise RoleError("R3a needs tensor cells with at most one Legendre parent")
    axes = {name: axis for name, axis in joint.cell_axes if name != child}
    reps = dict(joint.reps)
    if not variables or set(axes).union(reps) != set(variables):
        raise RoleError("R3a needs one conditional child and one unknown parent block")
    non_depth = [
        max(axis.order)
        for name, axis in axes.items()
        if name not in joint.cells.excluded
    ]
    order = (
        min(joint.order, sum(non_depth))
        if joint.order is not None and non_depth
        else None
    )
    return Lattice(
        order=order,
        weights=weights,
        depth_in_total=joint.depth_in_total,
        **joint._config(),
        **axes,
        **reps,
    )


def conditional_basis(
    joint_basis: SpectralBasis,
    child: str,
    declaration,
    *,
    parent_variables=None,
    parent_weights=None,
    nodes=24,
) -> ConditionalBasis:
    """Compile a joint basis into a linear basis over its unknown parent cells.

    Build the joint basis with both the parent and the conditional child
    resolved, and supply the other kernel statistics when building it.
    The returned moments and fit observables use the projected parent
    lattice and its reference weights. Explicit parent weights can replace
    the kernel's default metric without changing predictions.
    """
    if not isinstance(joint_basis, SpectralBasis) or isinstance(
        joint_basis, ConditionalBasis
    ):
        raise TypeError("conditional_basis needs an unmapped SpectralBasis")
    table = joint_basis.lattice.cells
    if child not in table.variables:
        raise RoleError(f"R3a child {child!r} needs a lattice cell axis")
    variables = (
        tuple(name for name in table.variables if name != child)
        + tuple(joint_basis.lattice.representations)
        if parent_variables is None
        else tuple(parent_variables)
    )
    if len(variables) != len(set(variables)):
        raise RoleError("R3a parent variables must be distinct")
    dependencies = set(parents(declaration))
    if not dependencies or dependencies - set(variables):
        raise RoleError(
            "R3a needs a conditional law whose parents belong to the unknown cell block"
        )
    d = table.variables.index(child)
    moving = _moving_child_axis(joint_basis.lattice, child)
    if not moving and not np.isfinite((table.lo[:, d], table.hi[:, d])).all():
        raise RoleError(
            f"R3a child {child} needs finite cells or one cell without edges"
        )
    if isinstance(nodes, bool) or not isinstance(nodes, (int, np.integer)) or nodes < 1:
        raise ValueError("R3a panel nodes must be a positive integer")
    if not moving and 8 * 6 * int(nodes) ** 2 > 32 * 2**20:
        raise SizeError(
            "the requested R3a Gauss rule exceeds the coefficient work budget"
        )
    lattice = _project_lattice(joint_basis.lattice, child, variables, parent_weights)
    kernel = joint_basis.averaged.kernel
    definitions = {v.name: v for v in kernel.variables}
    builder = build_legendre_map if lattice.index.legendre else build_map
    mapping, residual = builder(
        joint_basis.lattice, lattice, child, declaration, definitions, int(nodes)
    )
    return ConditionalBasis(
        joint_basis, lattice, mapping, residual, child, declaration, int(nodes)
    )


class ConditionalBasis(SpectralBasis):
    """A joint basis contracted to its independently fitted parent moments."""

    _archive_kind: ClassVar[str] = "conditional"

    joint: SpectralBasis
    mapping: ConditionalMap
    residuals: tuple
    child: str = eqx.field(static=True)
    declaration: object
    nodes: int = eqx.field(static=True)

    def __init__(self, joint, lattice, mapping, residuals, child, declaration, nodes):
        self.joint, self.lattice, self.mapping = joint, lattice, mapping
        self.child, self.declaration, self.nodes = child, declaration, nodes
        # Turn the records' NumPy diagnostics into an ordinary Equinox tree
        # of array leaves. The integer row offsets stay static.
        self.residuals = tuple(
            ResidualSample(part.values, part.parent_cell, part.first_row)
            for part in residuals
        )
        self.channels, self.averaged = joint.channels, joint.averaged
        self.spectra0 = self.spectra2 = None
        self.specified, self.azimuth_factor = joint.specified, joint.azimuth_factor
        self.rules = ColumnRules(stokes=joint.columns.stokes)
        columns = column_map(
            lattice,
            joint.averaged.kernel,
            self.rules,
            max_columns=dict(joint.record.defaults).get("max_columns", 20_000),
        )
        self.columns = replace(columns, key=digest((columns.key, mapping.key)))
        self.key = mapped_design_key(joint, lattice, mapping)
        self.forward_population = None
        self.forward_phase = joint.forward_phase
        self.forward_defaults = joint.forward_defaults
        self.structural_plan = joint.structural_plan
        if hasattr(joint, "forward_depth_tol"):
            self.forward_depth_tol = joint.forward_depth_tol
        name = f"conditional_truncation:{child}|{','.join(parents(declaration))}"
        residual = ErrorTerm.unbounded(
            "a sampled per-unit-column residual does not bound the error of unknown parent moments",
            name=name,
        )
        self.ledger = Ledger(
            (*joint.ledger.terms, residual), assumptions=joint.ledger.assumptions
        )
        self.record = replace(
            joint.record,
            sizes=record_sizes(self.columns),
            keys=(
                ("lattice", lattice.key),
                ("R3a map", mapping.key),
                ("design", self.key),
            ),
        ).with_notes(
            f"R3a child {child} contracted into parent rows. Provider routes describe the joint basis."
        )

    def design(self, data=None, *, stokes=None):
        """Channel design in the scaled parent rows."""
        if data is not None:
            return self._observed_design(data, stokes)
        joint = self.joint.columns.lift(self.joint.design(stokes=stokes))
        return self.mapping.design(joint)

    def at(self, channels):
        """Build joint rows at new channels and preserve the moment map."""
        joint = self.joint.at(channels)
        return ConditionalBasis(
            joint,
            self.lattice,
            self.mapping,
            self.residuals,
            self.child,
            self.declaration,
            self.nodes,
        )

    def select(self, **cells):
        """A smaller parent domain requires a joint basis on the selected tensor grid."""
        if self.lattice.index.cells_where(**cells).size == self.lattice.n_cells:
            return self
        raise RoleError(
            "build the conditional basis on a tensor lattice with the selected parent cells"
        )

    def with_order(self, order):
        """Compile the shifted correlations at a new total Taylor order."""
        joint = self.joint.with_order(order)
        return conditional_basis(
            joint,
            self.child,
            self.declaration,
            parent_weights=self.lattice.weights,
            nodes=self.nodes,
        )

    def moments(self, population, *, allow_mismatch=False):
        if not isinstance(population, Moments):
            raise RoleError("supply parent Moments to the conditional inverse basis")
        self.lattice._own(population)
        return population

    def direct(self, population, *, within="support"):
        """Integrate a specified joint population for a reference spectrum."""
        return self.joint.direct(population, within=within)

    def conditional_truncation(self):
        """Sampled per-unit-column residual per parent cell and channel Stokes.

        Parent cells without finite faces return NaN because the diagnostic
        has no finite sampling domain there. The extrema are estimates.
        """
        return residual_estimate(self.joint, self.lattice.n_cells, self.residuals)

    def conditional_truncation_bound(self, *, chunk_bytes=BOUND_BYTES):
        """Bound the conditional map error per unit parent column in each cell."""
        return bound_term(self, chunk_bytes)

    def conditional_error(self, moments, *, chunk_bytes=BOUND_BYTES):
        """Estimate the conditional error at supplied fitted cell masses."""
        return fitted_bound_term(self, moments, chunk_bytes)
