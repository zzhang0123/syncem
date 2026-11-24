"""Differentiable moments of an electron law on finite primary cells."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np


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


def primary_rows(law, geometry):
    """Keep the derivative of each moving panel edge in the moment integral."""
    groups, size = {}, 0
    for lo, hi, ref, half, chart, powers, weight in geometry:
        faces, indices = groups.setdefault((chart, powers), ([], []))
        faces.append((lo, hi, ref, half, weight))
        indices.extend(range(size, size + len(powers)))
        size += len(powers)
    output = jnp.zeros(size)
    knots = tuple(getattr(law, "kinks", ()))
    for (chart, powers), (faces, indices) in groups.items():
        powers = jnp.asarray(powers)

        def panel(cell):
            lo, hi, ref, half, weight = cell
            edges = jnp.sort(jnp.clip(jnp.asarray((lo, *knots, hi)), lo, hi))
            row = 0.0
            for left, right in zip(edges[:-1], edges[1:]):
                x, w = law.panel_rule(left, right, fitted=True)
                u = (jnp.log(x / ref) if chart == "log" else x - ref) / half
                row += jnp.sum(w[:, None] * u[:, None] ** powers, axis=0)
            return row / weight

        values = jax.lax.map(panel, jnp.asarray(faces), batch_size=16).reshape(-1)
        if len(groups) == 1:
            return values
        output = output.at[jnp.asarray(indices)].set(values)
    return output


@eqx.filter_jit
def primary_curve(law, geometry, matrix, fields, log, coordinates):
    """Return the whitened unit-amplitude model and its exact JAX Jacobian."""

    def evaluate(z):
        x = jnp.where(log, jnp.exp(jnp.where(log, z, 0.0)), z)
        trial = law._at(dict(zip(fields, x))) if fields else law
        result = matrix @ primary_rows(trial, geometry)
        return result, result

    jacobian, value = jax.jacfwd(evaluate, has_aux=True)(coordinates)
    return value, jacobian
