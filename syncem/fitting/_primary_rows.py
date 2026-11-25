"""Differentiable moments of an electron law on finite primary cells."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..distributions.user import UserFamily


def primary_geometry(lattice, primary):
    table, index = lattice.cells, lattice.index
    if table.variables != (primary,) or lattice.representations:
        return None
    if not np.isfinite(table.lo).all() or not np.isfinite(table.hi).all():
        return None
    return tuple(
        (
            float(lo[0]),
            float(hi[0]),
            float(ref[0]),
            float(half[0]),
            chart[0],
            tuple(r[0] for r in index.patterns[index.cell_pattern[i]]),
            float(lattice.weights[i]),
        )
        for i, (lo, hi, ref, half, chart) in enumerate(
            zip(table.lo, table.hi, table.reference, table.half_width, table.chart)
        )
    )


def _integrate(geometry, knots, panel_values, trailing_shape=()):
    """Integrate a vector of polynomial moments on the declared primary cells."""
    groups, size = {}, 0
    for lo, hi, ref, half, chart, powers, weight in geometry:
        faces, indices = groups.setdefault((chart, powers), ([], []))
        faces.append((lo, hi, ref, half, weight))
        indices.extend(range(size, size + len(powers)))
        size += len(powers)
    output = jnp.zeros((size, *trailing_shape))
    for (chart, powers), (faces, indices) in groups.items():
        powers = jnp.asarray(powers)

        def panel(cell):
            lo, hi, ref, half, weight = cell
            edges = jnp.sort(jnp.clip(jnp.asarray((lo, *knots, hi)), lo, hi))
            row = 0.0
            for left, right in zip(edges[:-1], edges[1:]):
                row += panel_values(left, right, ref, half, chart, powers)
            return row / weight

        values = jax.lax.map(panel, jnp.asarray(faces), batch_size=16).reshape(
            (-1, *trailing_shape)
        )
        if len(groups) == 1:
            return values
        output = output.at[jnp.asarray(indices)].set(values)
    return output


def _polynomials(x, ref, half, chart, powers):
    u = (jnp.log(x / ref) if chart == "log" else x - ref) / half
    return u[:, None] ** powers


def primary_rows(law, geometry):
    """Keep the derivative of each moving panel edge in the moment integral."""

    def panel_values(left, right, ref, half, chart, powers):
        x, w = law.panel_rule(left, right, fitted=True)
        return jnp.sum(w[:, None] * _polynomials(x, ref, half, chart, powers), axis=0)

    return _integrate(geometry, tuple(getattr(law, "kinks", ())), panel_values)


def _at_fields(law, fields, values):
    if not fields:
        return law
    updated = dict(zip(fields, values))
    if isinstance(law, UserFamily):
        return eqx.tree_at(lambda item: item.params, law, law.params | updated)
    return law._at(updated)


def _user_moment_response(law, geometry, fields, values):
    """Differentiate normalized node weights before projecting into observables."""

    def total_mass(x):
        mass = _at_fields(law, fields, x)._mass()
        return mass, mass

    dmass, mass = jax.jacfwd(total_mass, has_aux=True)(values)

    def panel_values(left, right, ref, half, chart, powers):
        def rule(x):
            local = _at_fields(law, fields, x)
            nodes, weights = local._raw_panel(left, right)
            basis = _polynomials(nodes, ref, half, chart, powers)
            return (weights, basis), (weights, basis)

        (draw, dbasis), (raw, basis) = jax.jacfwd(rule, has_aux=True)(values)
        w = raw / mass
        normalizer = w[:, None] * (dmass / mass)
        dw = draw / mass - normalizer
        # d(raw / mass) subtracts w * dmass / mass from draw / mass.
        # A multiplicative density coefficient cancels at each node. Judge
        # roundoff against that subtraction in the units of each derivative.
        roundoff = 16 * jnp.finfo(w.dtype).eps * jnp.abs(normalizer)
        dw = jnp.where(jnp.abs(dw) <= roundoff, 0.0, dw)
        derivative = basis.T @ dw + jnp.einsum("n,nkp->kp", w, dbasis)
        return jnp.column_stack((basis.T @ w, derivative))

    combined = _integrate(geometry, (), panel_values, (len(fields) + 1,))
    return combined[:, 1:], combined[:, 0]


@eqx.filter_jit
def primary_moment_jacobian(law, geometry, fields, values):
    if isinstance(law, UserFamily):
        return _user_moment_response(law, geometry, fields, values)

    def evaluate(x):
        moments = primary_rows(_at_fields(law, fields, x), geometry)
        return moments, moments

    return jax.jacfwd(evaluate, has_aux=True)(values)


@eqx.filter_jit
def primary_curve(law, geometry, matrix, fields, log, coordinates):
    """Return the whitened unit-amplitude model and its exact JAX Jacobian."""

    if isinstance(law, UserFamily):
        values = jnp.where(log, jnp.exp(jnp.where(log, coordinates, 0.0)), coordinates)
        derivative, moments = _user_moment_response(law, geometry, fields, values)
        return matrix @ moments, (matrix @ derivative) * jnp.where(log, values, 1.0)

    def evaluate(z):
        x = jnp.where(log, jnp.exp(jnp.where(log, z, 0.0)), z)
        trial = _at_fields(law, fields, x)
        result = matrix @ primary_rows(trial, geometry)
        return result, result

    jacobian, value = jax.jacfwd(evaluate, has_aux=True)(coordinates)
    return value, jacobian
