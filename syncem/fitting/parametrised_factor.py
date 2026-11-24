"""Fit a resolved family with one free block on a frozen joint basis.

LABEL [extension].

Free lattice moments and family parameters keep their native units. The
combined design has shape (n_data, n_row) after whitening. The profiled
minimum has no certified global optimum.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields, replace
from numbers import Integral
from typing import ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..distributions.parameters import Free
from ..engine._basis_update import build_record
from ..engine.assemble import spin0_blocks
from ..engine.basis import SpectralBasis
from ..engine.columns import ColumnRules, column_map
from ..lattice.lattice import Lattice
from ..lattice.moments import Moments
from ..population.declaration import Population, parents
from ..population.pairing import pair
from ..values.errors import DataError, RoleError
from ._free_fit_arrays import memory_chunk
from .basis_parametric import BasisParametricFit, fit_basis_parametric
from .data import StokesData
from .family_fit import free_parameters
from .free_fit import _design, _fit_basis

__all__ = ["ParametrisedFactorFit", "fit_parametrised_factor"]

_BASIS_NOTE = "The fitted family changes the moment map of a fixed joint basis."


def _axes_subset(joint, names):
    axes = {name: joint.axes[name] for name in names}
    maxima = [
        max(axis.order)
        for name, axis in axes.items()
        if name not in joint.cells.excluded
    ]
    order = min(joint.order, sum(maxima)) if maxima else None
    return Lattice(
        order=order,
        depth_in_total=joint.depth_in_total,
        **joint._config(),
        **axes,
    )


def _row_projection(joint, sub):
    source = joint.cells
    target = sub.cells
    select = [source.variables.index(name) for name in target.variables]
    grid = np.unravel_index(np.arange(source.n_cells), source.tensor)
    if target.tensor != tuple(source.tensor[i] for i in select):
        raise RoleError("the parent lattice must have the joint basis cell geometry")
    cells = np.ravel_multi_index(tuple(grid[i] for i in select), target.tensor)
    if any(
        not np.array_equal(
            getattr(source, name)[:, select], getattr(target, name)[cells]
        )
        for name in ("lo", "hi", "reference", "half_width")
    ) or any(
        tuple(chart[i] for i in select) != target.chart[cell]
        for chart, cell in zip(source.chart, cells, strict=True)
    ):
        raise RoleError("the parent lattice must have the joint basis cell geometry")

    sub_rows = sub.index.arrays
    lookup = {
        (cell, tuple(powers)): i
        for i, (cell, powers) in enumerate(
            zip(sub_rows.cell, sub_rows.powers, strict=True)
        )
    }
    joint_rows = joint.index.arrays
    try:
        return np.asarray(
            [
                lookup[cell, tuple(powers)]
                for cell, powers in zip(
                    cells[joint_rows.cell], joint_rows.powers[:, select], strict=True
                )
            ],
            dtype=np.intp,
        )
    except KeyError as exc:
        raise RoleError(
            "the parent lattice order omits joint basis moment rows"
        ) from exc


@dataclass(frozen=True, eq=False)
class _Projection:
    """One nonzero per joint row in the map from parent to joint moments."""

    parent: Lattice
    factor: Lattice
    parent_rows: np.ndarray
    factor_rows: np.ndarray
    weight_ratio: np.ndarray

    @classmethod
    def build(cls, joint, variable, parent):
        all_names = joint.cells.variables
        known = tuple(name for name in all_names if name != variable)
        if (
            joint.cell_axes is None
            or joint.cells.tensor is None
            or joint.reps
            or variable not in all_names
            or not known
        ):
            raise RoleError(
                "a parametrised factor needs its own axis in an all-celled tensor lattice"
            )
        parent = _axes_subset(joint, known) if parent is None else parent
        if (
            not isinstance(parent, Lattice)
            or parent.cells.tensor is None
            or parent.reps
            or set(parent.cells.variables) != set(known)
        ):
            raise RoleError("the parent lattice must resolve the remaining joint axes")
        factor = _axes_subset(joint, (variable,))
        p = _row_projection(joint, parent)
        f = _row_projection(joint, factor)
        ratio = np.asarray(parent._row_weights())[p] / np.asarray(joint._row_weights())
        return cls(parent, factor, p, f, ratio)

    def coefficients(self, family, variable):
        name = variable.name
        law = variable.place(family)
        factors = Population(**{name: law})
        units = {name: variable.unit} if variable.unit else {}
        moments = pair(factors, self.factor, units=units, variables={name: variable})
        raw = np.asarray(self.factor.raw(moments))
        return self.weight_ratio * raw[self.factor_rows]

    def contract(self, design, coefficients):
        from scipy.sparse import csr_matrix

        n = self.parent_rows.size
        lift = csr_matrix(
            (coefficients, (np.arange(n), self.parent_rows)), shape=(n, self.parent.n)
        )
        return (design.reshape(-1, n) @ lift).reshape(*design.shape[:-1], self.parent.n)


class _ContractedBasis(SpectralBasis):
    """A parent-basis view of one specified factor on the unchanged joint basis."""

    joint_basis: SpectralBasis = eqx.field(kw_only=True)
    projection: _Projection = eqx.field(static=True, kw_only=True)
    coefficients: object = eqx.field(kw_only=True)
    family: object = eqx.field(kw_only=True)

    def at(self, channels):
        joint = _fit_basis(self.joint_basis, channels)
        return self.with_joint(joint)

    def with_joint(self, joint):
        design = np.asarray(_design(joint))
        reduced = self.projection.contract(design, np.asarray(self.coefficients))
        s0, s2 = _spectra(
            reduced, self.lattice, self.columns.stokes, self.record.stokes
        )
        return replace(
            self,
            channels=joint.channels,
            spectra0=s0,
            spectra2=s2,
            ledger=joint.ledger,
            joint_basis=joint,
        )

    def moments(self, population, *, allow_mismatch=False):
        if not isinstance(population, Moments):
            raise RoleError("a fitted-factor basis needs moments in its parent rows")
        self.lattice._own(population)
        return population

    def with_order(self, order):
        raise RoleError("changing this order needs a new joint basis and factor fit")

    def direct(self, population, *, within="support"):
        raise RoleError("direct integration needs a specified joint population")

    def save(self, path):
        raise RoleError("archive the joint basis used by a fitted-factor basis")


def _spectra(design, parent, blocks, axes):
    """Undo the parent assembly weights for the stored spectral representation."""
    scaled = design / np.asarray(parent._row_weights())
    labels = {name: scaled[:, i] for i, name in enumerate(axes)}
    spin = spin0_blocks(blocks)
    s0 = (
        np.stack([labels[name] for name in spin], axis=1)
        if spin
        else np.zeros((design.shape[0], 0, parent.n))
    )
    s2 = labels["Q"] + 1j * labels["U"] if "P" in blocks else None
    return s0, s2


def _trial_geometry(basis, projection):
    parent = projection.parent
    rules = ColumnRules(stokes=basis.columns.stokes)
    columns = column_map(
        parent,
        basis.averaged.kernel,
        rules,
        max_columns=dict(basis.record.defaults).get("max_columns", 20_000),
    )
    if not np.array_equal(columns.col_of_row, np.arange(parent.n)):
        raise RoleError("this factor fit needs distinct parent moment columns")
    record = build_record(
        (),
        columns.stokes,
        basis.record.s1,
        columns,
        parent,
        notes=(_BASIS_NOTE,),
        events={"joint kernel rebuilds during fit": 0},
    )
    return rules, columns, record


def _family_parameters(variable, family):
    if parents(family) or not hasattr(family, "rule"):
        raise RoleError("the resolved factor needs an independent quadrature family")
    source = Population(**{variable: family})
    declared = free_parameters(source)
    arguments = getattr(family, "params", None)
    if isinstance(arguments, Mapping):
        declared.update(
            (f"{variable}.{name}", value)
            for name, value in arguments.items()
            if isinstance(value, Free)
        )
    if not declared:
        raise RoleError("the resolved factor needs at least one Free shape parameter")
    if any(key.endswith((".column", ".amplitude")) for key in declared):
        raise RoleError("the resolved factor normalisation must be fixed")
    return source, declared


def _family_at(source, variable, theta):
    family = source.at(theta).factors[variable]
    arguments = getattr(family, "params", None)
    if not isinstance(arguments, Mapping):
        return family
    updates = {
        name: theta.get(f"{variable}.{name}", value)
        for name, value in arguments.items()
    }
    return eqx.tree_at(lambda law: law.params, family, updates)


class ParametrisedFactorFit(BasisParametricFit):
    """Free parent moments with a fitted independent factor on another axis."""

    kind: ClassVar[str] = "bilinear (parametrised factor)"
    estimator: ClassVar[str] = "variable projection on a fixed joint spectral basis"

    @property
    def factor(self):
        """The resolved family at the fitted parameters."""
        return self.problem.basis.family

    @property
    def joint_moments(self):
        """Lift the fitted parent moments into the original joint rows."""
        basis = self.problem.basis
        parent = self.moments.values[..., basis.projection.parent_rows]
        values = parent * jnp.asarray(basis.coefficients)
        return Moments(basis.joint_basis.lattice, values, origin=self.moments.origin)

    @property
    def defaults(self):
        return (
            *super().defaults,
            ("joint basis builds during fit", 0),
            ("factor map derivatives", "finite differences of the profiled residual"),
        )


def fit_parametrised_factor(
    basis: SpectralBasis,
    data: StokesData,
    variable: str,
    family,
    *,
    parent_lattice: Lattice | None = None,
    **solver_options,
) -> ParametrisedFactorFit:
    """Profile one resolved parametric factor and free moments of the other axes.

    Each trial pairs the factor on its frozen one-dimensional grid and
    contracts the existing joint design. The inner SVD uses scaled parent
    moments. Noise covariance in the prediction conditions on the fitted
    factor. The outer covariance uses the profiled finite-difference Jacobian.
    """
    if not isinstance(basis, SpectralBasis) or not isinstance(data, StokesData):
        raise TypeError(
            "fit_parametrised_factor expects a SpectralBasis and StokesData"
        )
    if data.batch_shape:
        raise DataError("the parametrised factor fit accepts one sightline")
    projection = _Projection.build(basis.lattice, variable, parent_lattice)
    definition = next(v for v in basis.averaged.kernel.variables if v.name == variable)
    source, declared = _family_parameters(variable, family)
    joint = _fit_basis(basis, data.channels)
    chunk_bytes = solver_options.get("chunk_bytes", 2**31)
    if (
        not isinstance(chunk_bytes, Integral)
        or isinstance(chunk_bytes, (bool, np.bool_))
        or chunk_bytes < 1
    ):
        raise ValueError("chunk_bytes must be a positive integer")
    memory_chunk(data, joint.lattice.n, chunk_bytes, 8)
    design = np.asarray(_design(joint))
    rules, columns, record = _trial_geometry(joint, projection)
    parent = projection.parent

    def contracted(theta):
        law = _family_at(source, variable, theta)
        coefficients = projection.coefficients(law, definition)
        reduced = projection.contract(design, coefficients)
        s0, s2 = _spectra(reduced, parent, columns.stokes, record.stokes)
        return _ContractedBasis(
            lattice=parent,
            channels=joint.channels,
            columns=columns,
            spectra0=s0,
            spectra2=s2,
            azimuth_factor=None,
            specified=joint.specified,
            record=record,
            ledger=joint.ledger,
            key=None,
            averaged=joint.averaged,
            rules=rules,
            joint_basis=joint,
            projection=projection,
            coefficients=jnp.asarray(coefficients),
            family=law,
        )

    result = fit_basis_parametric(contracted, data, declared, **solver_options)
    if joint is not basis:
        prediction = result.problem.basis.with_joint(basis)
        result = replace(result, problem=replace(result.problem, basis=prediction))
    return ParametrisedFactorFit(
        **{
            field.name: getattr(result, field.name)
            for field in fields(BasisParametricFit)
        }
    )
