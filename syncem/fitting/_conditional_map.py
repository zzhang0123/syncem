"""Sparse shifted correlations for an R3a child on a tensor lattice."""

from __future__ import annotations

import math
from dataclasses import dataclass

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..distributions.electrons import ElectronFamily
from ..engine._jet_routine import check_order
from ..engine._moment_map import ConditionalMap as ConditionalMap
from ..engine.conditional_columns import _derivatives, conditional_contraction
from ..engine.provider_jet import _point
from ..population.declaration import law_at, placed_law
from ..values.errors import RoleError, SizeError

_MAP_BYTES = 128 * 2**20


def _moving_child_axis(lattice, child):
    axis = dict(lattice.cell_axes or ()).get(child)
    if axis is None or axis.edges is not None:
        return False
    if axis.n != 1:
        raise RoleError(
            f"R3a child {child} needs one cell without edges or disjoint finite cells"
        )
    return axis.support is None or not np.isfinite(axis.support).all()


@dataclass(frozen=True)
class ChildPanel:
    """Integrate in the child chart, using the law's own rule on a whole cell."""

    declaration: object
    variable: object
    units: dict
    chart: str
    lo: float
    hi: float
    reference: float
    half_width: float
    powers: tuple[int, ...]
    x: np.ndarray
    w: np.ndarray
    moving_nodes: bool = False
    workspace_bytes: int = _MAP_BYTES

    def __call__(self, point):
        law = placed_law(law_at(self.declaration, point, self.units), self.variable)
        if getattr(law, "is_point", False) or not callable(
            getattr(law, "density", None)
        ):
            raise RoleError(
                f"R3a child {self.variable.name} needs a conditional density. "
                "Resolve it by quadrature with a specified parent or average the dependence."
            )

        if isinstance(law, ElectronFamily):
            # The unknown parent block owns the column. A conditional child
            # contributes a probability on its full declared support.
            law = eqx.tree_at(
                lambda family: (family.column, family.amplitude),
                law,
                (1.0, None),
                is_leaf=lambda value: value is None,
            )
        bounds = jnp.asarray(law.support)
        if self.moving_nodes:
            child, weight, bounds = self._moving_rule(law, bounds)
        elif hasattr(law, "panel_rule"):
            child, weight = law.panel_rule(self.lo, self.hi, fitted=True)
        else:
            child, weight = self._rule(law)
        local = (
            jnp.log(child / self.reference)
            if self.chart == "log"
            else child - self.reference
        ) / self.half_width
        powers = jnp.stack([local**s for s in self.powers])
        moments = powers @ weight
        return jnp.concatenate((moments, bounds))

    def _moving_rule(self, law, bounds):
        count = getattr(law, "nodes", None)
        if (
            isinstance(count, (int, np.integer))
            and 48 * int(count) ** 2 > self.workspace_bytes
        ):
            raise SizeError(
                "the moving child quadrature exceeds the R3a workspace budget"
            )
        child, weight = (jnp.asarray(value) for value in law.rule())
        if (
            child.ndim != 1
            or not child.size
            or child.shape != weight.shape
            or jnp.iscomplexobj(child)
            or jnp.iscomplexobj(weight)
        ):
            raise RoleError(
                f"R3a child {self.variable.name} needs a real rule with one probability per node"
            )
        if 8 * child.size * (len(self.powers) + 7) > self.workspace_bytes:
            raise SizeError(
                "the moving child quadrature exceeds the R3a workspace budget"
            )
        valid = (bounds[0] >= self.lo) & (bounds[1] <= self.hi)
        valid &= jnp.all(jnp.isfinite(child) & (child >= self.lo) & (child <= self.hi))
        valid &= jnp.all(jnp.isfinite(weight) & (weight >= 0))
        if self.chart == "log":
            valid &= jnp.all(child > 0)
        # Permit normalisation roundoff in the probability rule's own dtype.
        precision = (
            jnp.finfo(weight.dtype).eps
            if jnp.issubdtype(weight.dtype, jnp.floating)
            else 0.0
        )
        mass = jnp.sum(weight, dtype=jnp.float64)
        rounded = jnp.abs(mass - 1.0) <= max(1e-11, 8 * precision)
        weight = weight / jnp.where(rounded, mass, 1.0)
        # Report invalid measures through the support diagnostics. The
        # differentiated monomial integral uses every rule node.
        return child, weight, jnp.where(valid, bounds, jnp.nan)

    def _rule(self, law):
        support_lo, support_hi = law.support
        lo, hi = jnp.maximum(self.lo, support_lo), jnp.minimum(self.hi, support_hi)
        valid = hi > lo
        a, b = jnp.where(valid, lo, self.lo), jnp.where(valid, hi, self.hi)
        knots = jnp.asarray(getattr(law, "kinks", ()), dtype=float)
        knots = jnp.sort(knots) if knots.size > 1 else knots
        edges = jnp.concatenate((a[None], jnp.clip(knots, a, b), b[None]))
        if self.chart == "log":
            edges = jnp.log(edges)
        half = jnp.diff(edges) / 2
        coordinate = edges[:-1, None] + half[:, None] * (1 + self.x[None, :])
        weights = half[:, None] * self.w[None, :]
        if self.chart == "log":
            child = jnp.exp(coordinate)
            weights = weights * child
        else:
            child = coordinate
        # A clipped empty panel carries no weight. Evaluate its density at
        # the midpoint of a nonempty support intersection when available.
        centre = jnp.where(valid, (lo + hi) / 2, (self.lo + self.hi) / 2)
        child = jnp.where((half > 0)[:, None], child, centre)
        density = jnp.broadcast_to(law.density(child.reshape(-1)), (child.size,))
        weights = jnp.where(
            valid & (half > 0)[:, None], weights * density.reshape(child.shape), 0.0
        )
        return child.reshape(-1), weights.reshape(-1)


def _cell_layout(joint, parent, c, d):
    indices = joint.cells.cell_numbers(c)
    parent_cell = int(
        np.ravel_multi_index(
            tuple(indices[name] for name in parent.cells.variables),
            parent.cells.tensor,
        )
    )
    a, b = joint.index, parent.index
    joint_powers = a.patterns[int(a.cell_pattern[c])]
    parent_powers = b.patterns[int(b.cell_pattern[parent_cell])]
    children = tuple(sorted({p[d] for p in joint_powers}))
    rows = tuple(
        (tuple(k for i, k in enumerate(p) if i != d), children.index(p[d]))
        for p in joint_powers
    )
    return parent_cell, parent_powers, children, rows


def _parent_coordinates(lattice, cell):
    table = lattice.cells
    if np.any(~np.isfinite(table.lo[cell])) or np.any(~np.isfinite(table.hi[cell])):
        return None
    charts, ref, half = table.chart[cell], table.reference[cell], table.half_width[cell]
    low, high = table.lo[cell].copy(), table.hi[cell].copy()
    for i, chart in enumerate(charts):
        if chart == "log":
            low[i], high[i] = math.log(low[i] / ref[i]), math.log(high[i] / ref[i])
        else:
            low[i], high[i] = low[i] - ref[i], high[i] - ref[i]
    low, high = low / half, high / half
    n = len(low)
    if n == 1:
        return np.linspace(low, high, 65)
    if n <= 3:
        axes = [np.linspace(a, b, 5) for a, b in zip(low, high)]
        return np.stack(np.meshgrid(*axes, indexing="ij"), axis=-1).reshape(-1, n)
    return np.concatenate((np.zeros((1, n)), np.diag(low), np.diag(high)))


def _residual(node, points, actual, base_rows):
    z = points[:, None, :]
    parent_powers, bases = np.asarray(node.parent_powers), np.asarray(base_rows)
    parent = np.prod(z**parent_powers, axis=-1) / node.parent_weight
    indices = [i for _, i in node.output_rows]
    direct = np.prod(z**bases, axis=-1) * actual[:, indices] / node.joint_weight
    return np.asarray(node.apply(parent)) - direct


@dataclass(frozen=True)
class ResidualRows:
    """Sampled joint-moment residuals per unit parent column."""

    parent_cell: int
    first_row: int
    values: np.ndarray


def _validate_samples(values, child, *, full_probability=False):
    a, b = values[..., -2], values[..., -1]
    if np.any(~(b > a)) or np.any(~np.isfinite(values[..., :-2])):
        raise RoleError(
            f"R3a conditional child {child} has invalid support or density on the parent grid"
        )
    mass = values[..., 0]
    if np.any(mass < 0):
        raise RoleError(
            f"R3a conditional child {child} has negative mass on the parent grid"
        )
    if full_probability and not np.allclose(mass, 1.0, rtol=1e-11, atol=0):
        raise RoleError(f"R3a conditional child {child} needs a unit-probability rule")


def _check_moving_face(coefficients, shifts, panel):
    zero = shifts.index((0,) * len(shifts[0]))
    boundaries = coefficients[zero, -2:]
    _validate_samples(
        coefficients[zero], panel.variable.name, full_probability=panel.moving_nodes
    )
    if panel.moving_nodes:
        return
    close = np.isclose(
        boundaries[:, None], (panel.lo, panel.hi), rtol=8 * np.finfo(float).eps, atol=0
    )
    moves = np.any(np.delete(coefficients[:, -2:], zero, axis=0) != 0.0, axis=0)
    if np.any(close & moves[:, None]):
        raise RoleError(
            f"R3a child {panel.variable.name} has moving support on a child cell face at the "
            "parent Taylor reference. Shift the reference or refine the parent cells."
        )


def build_map(joint, parent, child, declaration, definitions, nodes):
    """Compile one sparse index and its per-parent diagnostic samples."""
    d = joint.cells.variables.index(child)
    table = joint.cells
    moving = _moving_child_axis(joint, child)
    x, w = (
        (np.empty(0), np.empty(0)) if moving else np.polynomial.legendre.leggauss(nodes)
    )
    parts = []
    residuals = []
    checked_parents = set()
    sizes = 0
    units = {name: v.unit for name, v in definitions.items()}
    for c in range(table.n_cells):
        b, shifts, child_powers, output_rows = _cell_layout(joint, parent, c, d)
        frames = parent.cells
        charts, reference, half = (
            frames.chart[b],
            frames.reference[b],
            frames.half_width[b],
        )
        names = frames.variables
        evaluator = ChildPanel(
            declaration,
            definitions[child],
            units,
            table.chart[c][d],
            table.lo[c, d],
            table.hi[c, d],
            table.reference[c, d],
            table.half_width[c, d],
            child_powers,
            x,
            w,
            moving,
        )
        if 16 * len(shifts) * len(child_powers) > _MAP_BYTES:
            raise SizeError("one R3a cell exceeds the coefficient memory budget")
        order = check_order(names, shifts)
        derivatives = _derivatives(order)

        def integrals(z):
            return evaluator(_point(names, charts, reference, half, z))

        if b not in checked_parents:
            _validate_samples(
                np.asarray(integrals(jnp.zeros(len(names)))),
                child,
                full_probability=moving,
            )
            checked_parents.add(b)
        try:
            h = np.asarray(
                eqx.filter_jit(lambda: derivatives(integrals, len(names), shifts))()
            )
        except KeyError as error:
            if order <= 5:
                raise
            raise RoleError(
                f"R3a child {child} has an unsupported Taylor primitive {error}. "
                "Use order five or lower and refine the parent cells."
            ) from error
        _check_moving_face(h, shifts, evaluator)
        node = conditional_contraction(
            h[:, :-2],
            shifts,
            shifts,
            output_rows,
            parent_weight=parent.weights[b],
            joint_weight=joint.weights[c],
        )
        start = int(joint.index.offsets[c])
        sizes += 24 * len(node.retained_rows)
        points = _parent_coordinates(parent, b)
        if points is not None:
            sizes += 8 * len(points) * len(output_rows)
        if sizes > _MAP_BYTES:
            raise SizeError("R3a sparse map and diagnostics exceed the memory budget")
        parts.append(
            (
                node.retained_rows + start,
                node.retained_parents + int(parent.index.offsets[b]),
                node.retained_coefficients,
            )
        )
        if points is not None:

            def evaluate(coordinates):
                if moving:
                    return jax.lax.map(integrals, coordinates, batch_size=1)
                return jax.vmap(integrals)(coordinates)

            actual = np.asarray(eqx.filter_jit(evaluate)(jnp.asarray(points)))
            _validate_samples(actual, child, full_probability=moving)
            actual = actual[:, :-2]
            residual = _residual(
                node, points, np.asarray(actual), [r for r, _ in output_rows]
            )
            residuals.append(ResidualRows(b, start, residual))
    rows, columns, values = (np.concatenate(p) for p in zip(*parts))
    return ConditionalMap(rows, columns, values, joint.n, parent.n), tuple(residuals)
