"""Legendre-product contractions for one angular parent of an R3a child."""

from __future__ import annotations

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss, legvander

from ..engine.conditional_columns import _derivatives
from ..engine.provider_jet import _point
from ..engine._jet_routine import check_order
from ..values.errors import RoleError, SizeError
from ._conditional_map import (
    _MAP_BYTES,
    ChildPanel,
    ConditionalMap,
    ResidualRows,
    _cell_layout,
    _check_moving_face,
    _moving_child_axis,
    _parent_coordinates,
    _validate_samples,
)


def _layout(joint, parent, cell, d):
    if parent.cells.variables:
        return _cell_layout(joint, parent, cell, d)
    powers = joint.index.patterns[int(joint.index.cell_pattern[cell])]
    degrees = tuple(sorted({p[d] for p in powers}))
    return 0, ((),), degrees, tuple(((), degrees.index(p[d])) for p in powers)


def _spatial(parent, cell):
    if not parent.cells.variables:
        return np.empty((1, 0))
    sample = _parent_coordinates(parent, cell)
    if sample is None:
        return None
    sample = np.asarray(sample).reshape(-1, len(parent.cells.variables))
    if len(sample) > 17:
        sample = sample[np.rint(np.linspace(0, len(sample) - 1, 17)).astype(int)]
    return sample


def _basis_values(z, theta, powers, degree):
    monomials = np.prod(z[:, None, :] ** np.asarray(powers)[None, :, :], axis=-1)
    angular = legvander(theta, degree)
    return (angular[:, :, None] * monomials[:, None, :]).reshape(z.shape[0], -1)


def _local_map(h, theta, weights, degree, shifts, rows, ratio):
    angular = legvander(theta, degree)
    dual = angular * ((2 * np.arange(degree + 1) + 1) / 2)
    # Project P_l h directly against the dual P_j. This includes all terms
    # of h that can enter a retained product, including degrees above L.
    product = np.einsum("ql,qj,qks,q->ljks", angular, dual, h, weights, optimize=True)
    n_joint, n_parent = len(rows), len(shifts)
    matrix = np.zeros(((degree + 1) * n_joint, (degree + 1) * n_parent))
    lookup = {power: i for i, power in enumerate(shifts)}
    for output, (base, child) in enumerate(rows):
        for k, shift in enumerate(shifts):
            source = lookup.get(tuple(a + b for a, b in zip(base, shift, strict=True)))
            if source is not None:
                matrix[output::n_joint, source::n_parent] += (
                    ratio * product[:, :, k, child]
                )
    return matrix


def _sample(
    panel_at, parent, cell, angular, degree, shifts, rows, local, joint_weight, panel
):
    spatial = _spatial(parent, cell)
    if spatial is None:
        return None
    z = np.repeat(spatial, len(angular), axis=0)
    theta = np.tile(angular, len(spatial))

    batch = 1 if panel.moving_nodes else 8

    @eqx.filter_jit
    def evaluate(coordinates, angles):
        return jax.lax.map(
            lambda pair: panel_at(*pair), (coordinates, angles), batch_size=batch
        )

    actual = np.asarray(evaluate(jnp.asarray(z), jnp.asarray(theta)))
    _validate_samples(actual, panel.variable.name, full_probability=panel.moving_nodes)
    parent_basis = _basis_values(z, theta, shifts, degree) / parent.weights[cell]
    r, s = zip(*rows, strict=True)
    exact = _basis_values(z, theta, r, degree).reshape(len(z), degree + 1, -1)
    exact = (exact * actual[:, None, s]).reshape(len(z), -1) / joint_weight
    return parent_basis @ local.T - exact


def _coefficients(panel_at, shifts, theta, panel, names):
    ndim = len(names)
    z = jnp.zeros(ndim)
    batch = 1 if panel.moving_nodes else 8

    def evaluate(fn):
        return eqx.filter_jit(lambda angles: jax.lax.map(fn, angles, batch_size=batch))(
            theta
        )

    if not ndim:
        values = evaluate(lambda eta: panel_at(z, eta))
        h = np.asarray(values)[:, None, :]
        _validate_samples(
            h[:, 0], panel.variable.name, full_probability=panel.moving_nodes
        )
        return h
    total = check_order(names, shifts)
    derivative = _derivatives(total)

    def at_angle(eta):
        return derivative(lambda value: panel_at(value, eta), ndim, shifts)

    try:
        h = np.asarray(evaluate(at_angle))
    except KeyError as error:
        if total <= 5:
            raise
        raise RoleError(
            f"R3a child {panel.variable.name} has an unsupported Taylor primitive {error}. "
            "Use order five or lower and refine the celled parent axes."
        ) from error
    for row in h:
        _check_moving_face(row, shifts, panel)
    return h


def build_legendre_map(joint, parent, child, declaration, definitions, nodes):
    """Project products of the child moments and retained Legendre modes."""
    ((name, degree),) = parent.index.legendre
    n_angle = max(24, 2 * degree + 1)
    moving = _moving_child_axis(joint, child)
    quadrature = 8 * 6 * (n_angle**2 + (0 if moving else nodes**2))
    if quadrature > _MAP_BYTES:
        raise SizeError("R3a Legendre quadrature exceeds the workspace budget")
    theta, weights = leggauss(n_angle)
    probe = np.linspace(-1.0, 1.0, max(17, 2 * degree + 1))
    x, w = (np.empty(0), np.empty(0)) if moving else leggauss(nodes)
    theta_jax = jnp.asarray(theta)
    table = joint.cells
    d = table.variables.index(child)
    units = {n: variable.unit for n, variable in definitions.items()}
    parts, residuals = [], []
    used = 0
    for cell in range(table.n_cells):
        b, shifts, powers, rows = _layout(joint, parent, cell, d)
        count = (degree + 1) ** 2 * len(rows) * len(shifts)
        product = (degree + 1) ** 2 * len(powers) * len(shifts)
        sample = _spatial(parent, b)
        samples = 0 if sample is None else len(sample) * len(probe)
        values = samples * (degree + 1) * len(rows)
        h_size = n_angle * len(shifts) * (len(powers) + 2)
        workspace = 8 * (5 * count + 3 * product + 4 * values + 4 * h_size)
        if used + max(quadrature, workspace) > _MAP_BYTES:
            raise SizeError(
                "R3a Legendre map and sampled residuals exceed the workspace budget"
            )
        panel = ChildPanel(
            declaration,
            definitions[child],
            units,
            table.chart[cell][d],
            table.lo[cell, d],
            table.hi[cell, d],
            table.reference[cell, d],
            table.half_width[cell, d],
            powers,
            x,
            w,
            moving,
        )
        frames = parent.cells

        def panel_at(z, eta):
            point = _point(
                frames.variables,
                frames.chart[b],
                frames.reference[b],
                frames.half_width[b],
                z,
            )
            return panel({**point, name: eta})

        h = _coefficients(panel_at, shifts, theta_jax, panel, frames.variables)
        local = _local_map(
            h[:, :, :-2],
            theta,
            weights,
            degree,
            shifts,
            rows,
            parent.weights[b] / joint.weights[cell],
        )
        r, c = np.nonzero(local)
        j_start = joint.index.cell_rows(cell).start
        p_start = parent.index.cell_rows(b).start
        parts.append((r + j_start, c + p_start, local[r, c]))
        error = _sample(
            panel_at,
            parent,
            b,
            probe,
            degree,
            shifts,
            rows,
            local,
            joint.weights[cell],
            panel,
        )
        if error is not None:
            residuals.append(ResidualRows(b, j_start, error))
        used += 24 * len(r) + (0 if error is None else error.nbytes)
    arrays = tuple(np.concatenate(group) for group in zip(*parts, strict=True))
    return ConditionalMap(*arrays, joint.n, parent.n), tuple(residuals)
