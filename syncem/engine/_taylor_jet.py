"""Taylor-mode coefficients of a smooth function on cell coordinates.

For each degree, a univariate jet along v gives the homogeneous
polynomial evaluated at v. Interpolate on its full monomial basis, then
gather the requested rows. Derivatives outside the requested lower set
can contribute to these directional data.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
from jax._src import core
from jax.experimental.jet import jet2

from ._taylor_rules import is_equinox_guard, taylor_rules

_MAX_MATRIX_ELEMENTS = 1_000_000


def homogeneous_powers(d, degree):
    """The weak compositions of a degree, in descending lexicographic order."""
    if d == 0:
        if degree == 0:
            yield ()
    elif d == 1:
        yield (degree,)
    else:
        for i in range(degree, -1, -1):
            for rest in homogeneous_powers(d - 1, degree - i):
                yield (i, *rest)


def _directions(d, order):
    count = math.comb(order + d - 1, d - 1)
    if d == 2:
        angles = np.pi * np.arange(count) / count
        return np.column_stack([np.cos(angles), np.sin(angles)])
    # Specify the bit generator and seed since the direction set enters the
    # numerical definition of a Taylor-mode column.
    rng = np.random.Generator(np.random.PCG64(0))
    points = rng.standard_normal((2 * count, d))
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def _fits_matrix(d, order):
    columns = math.comb(order + d - 1, d - 1)
    return (2 if d > 2 else 1) * columns**2 <= _MAX_MATRIX_ELEMENTS


def _check_derivatives(graph):
    """Reject custom derivatives that the JAX jet would silently ignore."""
    if isinstance(graph, core.ClosedJaxpr):
        for equation in graph.eqns:
            if is_equinox_guard(equation.params):
                continue
            if equation.primitive.name in ("custom_jvp_call", "custom_vjp_call"):
                raise TypeError(
                    "Taylor mode cannot use a custom JVP or VJP. "
                    "Use order 5 or lower or express the kernel and laws using Taylor-compatible primitives."
                )
            for value in equation.params.values():
                _check_derivatives(value)
    elif isinstance(graph, (tuple, list)):
        for item in graph:
            _check_derivatives(item)


def _reconstruction(vectors, converted, exponents):
    """Selected rows of the inverses for each requested homogeneous degree."""
    for degree in sorted({sum(ex) for ex in exponents} - {0}):
        selected = [i for i, ex in enumerate(exponents) if sum(ex) == degree]
        powers = tuple(homogeneous_powers(vectors.shape[1], degree))
        positions = {ex: i for i, ex in enumerate(powers)}
        indices = [positions[converted[i]] for i in selected]
        matrix = np.ones((len(vectors), len(powers)))
        for axis in range(vectors.shape[1]):
            matrix *= vectors[:, axis, None] ** np.asarray(powers)[:, axis]
        inverse = np.linalg.pinv(matrix)[indices]
        # Refuse to use a numerically unresolved interpolation.
        error = inverse @ matrix
        error[np.arange(len(indices)), indices] -= 1
        if np.max(np.abs(error)) > 1e-10:
            raise TypeError(
                f"Taylor direction reconstruction is ill-conditioned at degree {degree} "
                f"in {vectors.shape[1]} dimensions"
            )
        yield degree, np.asarray(selected), inverse


def taylor_mode_coefficients(f, d, exponents):
    """Rows of normalized partial derivatives at zero for a pytree-valued function."""
    exponents = tuple(tuple(int(p) for p in ex) for ex in exponents)
    origin = jnp.zeros(d)
    order = max(map(sum, exponents))
    if order:
        _check_derivatives(jax.make_jaxpr(f)(origin))
    try:
        return _evaluate(f, origin, exponents, order)
    except KeyError as err:
        if len(err.args) != 1 or not isinstance(err.args[0], core.Primitive):
            raise
        raise TypeError(
            f"Taylor mode has no rule for JAX primitive {err.args[0]}"
        ) from err


def _evaluate(f, origin, exponents, order):
    if order == 0:
        return jax.tree.map(
            lambda x: jnp.repeat(jnp.asarray(x)[None], len(exponents), 0), f(origin)
        )
    axes = tuple(i for i in range(origin.size) if any(ex[i] for ex in exponents))
    if len(axes) == 1:
        tangent = origin.at[axes[0]].set(1)
        with taylor_rules():
            values, coefficients = jet2(
                f, (origin,), ((tangent,) + (origin,) * (order - 1),)
            )
        structure = jax.tree.structure(values)
        terms = structure.flatten_up_to(coefficients)
        degrees = tuple(map(sum, exponents))
        return structure.unflatten(
            jnp.stack([(v, *s)[degree] for degree in degrees])
            for v, s in zip(jax.tree.leaves(values), terms)
        )

    if not _fits_matrix(len(axes), order):
        by_support = {}
        for row, ex in enumerate(exponents):
            support = tuple(i for i, power in enumerate(ex) if power)
            by_support.setdefault(support, []).append(row)
        if len(by_support) == 1:
            raise TypeError(
                f"Taylor reconstruction at degree {order} spans {len(axes)} variables. "
                "Lower the cell order or the number of mixed derivatives."
            )
        pieces = []
        rows = []
        for group in by_support.values():
            selected = tuple(exponents[i] for i in group)
            pieces.append(_evaluate(f, origin, selected, max(map(sum, selected))))
            rows.extend(group)
        pos = jnp.asarray(np.argsort(rows))
        return jax.tree.map(lambda *a: jnp.concatenate(a)[pos], *pieces)

    vectors = _directions(len(axes), order)
    converted = tuple(tuple(ex[i] for i in axes) for ex in exponents)
    plan = tuple(_reconstruction(vectors, converted, exponents))
    axes = jnp.asarray(axes)

    def along(direction):
        tangent = origin.at[axes].set(direction)
        return jet2(f, (origin,), ((tangent,) + (origin,) * (order - 1),))

    with taylor_rules():
        values, coefficients = jax.vmap(along, out_axes=(None, 0))(
            jnp.asarray(vectors, dtype=origin.dtype)
        )
    structure = jax.tree.structure(values)
    terms = structure.flatten_up_to(coefficients)

    def recover(value, series):
        value = jnp.asarray(value)
        result = jnp.broadcast_to(value, (len(exponents), *value.shape))
        for degree, rows, inverse in plan:
            directional = series[degree - 1]
            selected = jnp.tensordot(
                jnp.asarray(inverse, dtype=directional.dtype), directional, axes=1
            )
            result = result.at[rows].set(selected)
        return result

    return structure.unflatten(
        recover(x, term) for x, term in zip(jax.tree.leaves(values), terms)
    )
