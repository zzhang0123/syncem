"""R3a joint moments for Lobatto atoms of the nonnegative parent measure."""

from __future__ import annotations

from dataclasses import fields, replace
from numbers import Integral
from types import SimpleNamespace

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..engine.basis import SpectralBasis
from ..population.blocks import NonNegativeCells
from ..population.declaration import law_at, parents, placed_law
from ..records.ledger import ErrorTerm, Ledger
from ..values.errors import RoleError, SizeError
from ..values.observable import Observable
from ._conditional_map import (
    ChildPanel,
    _cell_layout,
    _moving_child_axis,
    _validate_samples,
)
from ._penalized_result import NonnegativeFit
from ._point_mass_fit import PointMassMap, _box, point_mass_map
from .conditional_basis import ConditionalBasis, _project_lattice

__all__ = ["conditioned_point_mass_map", "conditioned_result"]


def _integer(value, label):
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Integral)
        or value < 1
    ):
        raise ValueError(f"{label} must be a positive integer")
    return int(value)


def _workspace(joint, parent, marker, definitions, nodes, budget):
    names = parent.cells.variables
    hints = tuple(marker.support_for(definitions[name]) for name in names)
    selected = parent.index.cells_where(**marker.selections())
    active = sum(_box(parent.cells, int(c), hints) is not None for c in selected)
    per_cell = marker.point_masses ** len(names)
    points = active * per_cell
    if not points:
        raise RoleError("the parent marker support does not overlap any cell")
    local_rows = max(len(pattern) for pattern in joint.index.patterns)
    # Retain the parent and joint maps simultaneously while filling local
    # row blocks. The eventual NNLS solver separately budgets its own work.
    maps = 8 * points * (2 * parent.n + joint.n + 2 * len(names) + 1)
    scratch = 8 * (4 * local_rows * per_cell + 6 * nodes**2)
    if maps + scratch >= budget:
        raise SizeError(
            f"the {points:,} R3a parent masses need about {maps + scratch:,} "
            f"bytes of joint map workspace. chunk_bytes={budget:,}. "
            "Use fewer parent masses or a larger budget."
        )
    return budget - maps - scratch


def _spans(cells):
    ids, first, counts = np.unique(cells, return_index=True, return_counts=True)
    return {
        int(cell): slice(int(start), int(start + count))
        for cell, start, count in zip(ids, first, counts, strict=True)
    }


def _child_at(panel, coordinates, budget):
    law = placed_law(
        law_at(panel.declaration, coordinates, panel.units), panel.variable
    )
    bound = replace(panel, declaration=law)
    if panel.moving_nodes:
        return np.asarray(
            replace(bound, workspace_bytes=min(budget, panel.workspace_bytes))({})
        )
    if not getattr(law, "is_point", False) and callable(getattr(law, "density", None)):
        lo, hi = law.support
        if hi <= panel.lo or lo >= panel.hi:
            return np.r_[np.zeros(len(panel.powers)), lo, hi]
    if not hasattr(law, "panel_rule"):
        count = len(getattr(law, "kinks", ())) + 1
        need = 8 * count * panel.x.size * (len(panel.powers) + 7)
        if need > budget:
            raise SizeError(
                "the split child quadrature exceeds the R3a map workspace budget"
            )
        return np.asarray(bound({}))
    knots = sorted(
        {
            float(x)
            for x in getattr(law, "kinks", ())
            if panel.lo < x < panel.hi and np.isfinite(x)
        }
    )
    if not knots:
        return np.asarray(bound({}))
    # A special panel rule (for example an electron family) knows its own
    # normalisation. Evaluate it on each side of every interior kink.
    edges = (panel.lo, *knots, panel.hi)
    output = None
    for left, right in zip(edges[:-1], edges[1:], strict=True):
        value = np.asarray(replace(bound, lo=left, hi=right)({}))
        if output is None:
            output = value.copy()
        else:
            output[:-2] += value[:-2]
    return output


def _joint_rows(
    joint, parent, points, child, declaration, definitions, nodes, budget, *, moving
):
    table, index = joint.cells, joint.index
    d = table.variables.index(child)
    x, w = (
        (np.empty(0), np.empty(0)) if moving else np.polynomial.legendre.leggauss(nodes)
    )
    units = {name: variable.unit for name, variable in definitions.items()}
    active = _spans(points.cells)
    full = np.zeros((joint.n, points.cells.size))
    for joint_cell in range(table.n_cells):
        parent_cell, parent_powers, powers, output_rows = _cell_layout(
            joint, parent, joint_cell, d
        )
        if parent_cell not in active:
            continue
        span = active[parent_cell]
        panel = ChildPanel(
            declaration,
            definitions[child],
            units,
            table.chart[joint_cell][d],
            table.lo[joint_cell, d],
            table.hi[joint_cell, d],
            table.reference[joint_cell, d],
            table.half_width[joint_cell, d],
            powers,
            x,
            w,
            moving,
        )
        # Each atom contributes only to its own parent cell and the child
        # cells under it. The dense map is the existing NNLS solver's input.
        size = span.stop - span.start
        integrals = np.empty((size, len(powers) + 2))
        for row, q in enumerate(range(span.start, span.stop)):
            integrals[row] = _child_at(
                panel,
                {name: values[q] for name, values in points.coordinates.items()},
                budget,
            )
        _validate_samples(integrals, child, full_probability=moving)
        start = int(parent.index.offsets[parent_cell])
        lookup = {power: start + i for i, power in enumerate(parent_powers)}
        parent_rows = [lookup[power] for power, _ in output_rows]
        child_rows = [degree for _, degree in output_rows]
        weight = parent.weights[parent_cell] / joint.weights[joint_cell]
        rows = index.cell_rows(joint_cell)
        full[rows.start : rows.stop, span] = (
            points.moments[parent_rows, span] * integrals[:, child_rows].T * weight
        )
    return full


def conditioned_point_mass_map(basis, marker, child, declaration, budget, *, nodes=24):
    """Lift parent masses into joint cells with direct conditional quadrature.

    Electron child families use unit column. Other child densities supply
    probabilities. The returned coordinates and cell indices are the parent
    marginal. The moment matrix uses the original joint lattice.
    """
    if not isinstance(basis, SpectralBasis) or not isinstance(marker, NonNegativeCells):
        raise TypeError("conditioned points need a SpectralBasis and NonNegativeCells")
    if isinstance(basis, ConditionalBasis):
        raise RoleError("conditioned point masses need the original joint basis")
    budget, nodes = _integer(budget, "chunk_bytes"), _integer(nodes, "nodes")
    joint = basis.lattice
    variables = tuple(name for name in joint.cells.variables if name != child)
    if child not in joint.cells.variables:
        raise RoleError(f"R3a child {child!r} needs a lattice cell axis")
    dependencies = set(parents(declaration))
    if not dependencies or dependencies - set(variables):
        raise RoleError("the R3a conditional must depend on its fitted parent axes")
    if not marker.variables and len(variables) == 1:
        marker = marker.bind(variables[0])
    if set(marker.variables) != set(variables):
        raise RoleError("the nonnegative marker needs all the R3a parent axes")
    parent = _project_lattice(joint, child, variables, None)
    d = joint.cells.variables.index(child)
    moving = _moving_child_axis(joint, child)
    if (
        not moving
        and not np.isfinite((joint.cells.lo[:, d], joint.cells.hi[:, d])).all()
    ):
        raise RoleError(
            f"R3a child {child} needs finite cells or one cell without edges"
        )
    definitions = {v.name: v for v in basis.averaged.kernel.variables}
    remaining = _workspace(
        joint, parent, marker, definitions, 0 if moving else nodes, budget
    )
    points = point_mass_map(
        SimpleNamespace(lattice=parent, averaged=basis.averaged), marker, budget
    )
    full = _joint_rows(
        joint,
        parent,
        points,
        child,
        declaration,
        definitions,
        nodes,
        remaining,
        moving=moving,
    )
    return PointMassMap(full, points.coordinates, points.cells)


class _ConditionedResult(NonnegativeFit):
    child: str = eqx.field(static=True, kw_only=True)
    dependencies: tuple[str, ...] = eqx.field(static=True, kw_only=True)

    @property
    def population(self):
        """The parent marginal only. The joint moment values include the child."""
        return super().population

    @property
    def source_column(self):
        return Observable(
            jnp.sum(self.weights, axis=-1),
            origin=self.moments.origin,
            label="full parent-marginal column in cm^-2",
        )

    @property
    def defaults(self):
        return (
            *super().defaults,
            (
                "point coordinates",
                f"parent marginal with conditional child {self.child}",
            ),
        )

    def resample(self, n, *, key):
        return conditioned_result(
            super().resample(n, key=key), self.child, dependencies=self.dependencies
        )

    def profile_max(self, channels=None, *, stokes="I", delta_chi2=1.0):
        answer = super().profile_max(channels, stokes=stokes, delta_chi2=delta_chi2)
        return answer._replace(label=f"{answer.label}, conditional on {self.child}")


def conditioned_result(result, child, *, dependencies=()):
    """Attach parent-marginal and conditional-profile provenance."""
    dependencies = tuple(dependencies) or tuple(result.points.coordinates)
    name = f"conditional_truncation:{child}|{','.join(dependencies)}"
    term = ErrorTerm.not_applicable(
        "child moments use direct quadrature at every parent atom", name=name
    )
    terms = tuple(t for t in result.ledger.terms if t.name != name)
    values = {
        field.name: getattr(result, field.name) for field in fields(NonnegativeFit)
    }
    values["ledger"] = Ledger((*terms, term), assumptions=result.ledger.assumptions)
    return _ConditionedResult(
        **values,
        child=child,
        dependencies=dependencies,
    )
