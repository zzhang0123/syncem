"""Rounding displacements of deterministic nodes on population panels."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from .declaration import law_at, parents, placed_law


def sum_with_residual(a, b):
    """Return the rounded sum and its residual (Knuth's TwoSum)."""
    total = a + b
    part = total - a
    return total, (a - (total - part)) + (b - part)


def _value(declaration, variable, units, values):
    law = law_at(declaration, values, units)
    if variable is not None:
        law = placed_law(law, variable)
    if not hasattr(law, "rule"):
        return law
    return law.rule()[0][0] if getattr(law, "is_point", False) else jnp.nan


@eqx.filter_jit
def _point_gradient(declaration, variable, units, values):
    def coordinate(point):
        return _value(declaration, variable, units, point)

    return jax.grad(coordinate)(values)


def _level_chart(evaluate, base, level, slope):
    """Centre a locally monotone rounded level set on its representable root."""
    # A rounded conditional can be equal to a face at several successive
    # parent floats. Bracket the two changes of level in physical coordinates.
    # Only the local smooth neighbourhood is used. A truly flat branch keeps
    # its declared point. A one-sided domain need not support both probes.
    sides = []
    radius = np.sqrt(np.finfo(float).eps) * max(abs(base), np.spacing(1.0))
    for direction in (-np.inf, np.inf):
        inner = base
        ulp = abs(np.nextafter(base, direction) - base)
        step = max(ulp, abs(np.spacing(level) / slope) / 2)
        outer = base + np.copysign(step, direction)
        while abs(outer - base) <= radius:
            try:
                ordinate = evaluate(outer)
            except (ValueError, ArithmeticError):
                return None
            if not np.isfinite(ordinate):
                return None
            if ordinate != level:
                break
            inner = outer
            step *= 2
            outer = base + np.copysign(step, direction)
        else:
            return None
        while True:
            middle = inner + (outer - inner) / 2
            if middle == inner or middle == outer:
                break
            if evaluate(middle) == level:
                inner = middle
            else:
                outer = middle
        sides.append((inner - base, outer - base, ordinate - level))
    left, right = sides
    if np.sign(left[2]) == np.sign(right[2]):
        return None
    # Each transition lies between the last equal and the first unequal float.
    # Averaging their relative coordinates avoids rounding each boundary first.
    lo = (left[0] + left[1]) / 2
    hi = (right[0] + right[1]) / 2
    # The half-ULPs on the two sides of a power of two are unequal.
    down = abs(np.nextafter(level, np.copysign(np.inf, left[2])) - level)
    up = abs(np.nextafter(level, np.copysign(np.inf, right[2])) - level)
    return base + (lo + (hi - lo) * (down / (down + up)))


def point_displacement(declaration, variable, units, context, value, faces, charts):
    """Transport continuous coordinates through a deterministic conditional node."""
    names = parents(declaration)
    continuous = [name for name in names if (name,) in context]
    if not continuous:
        return None
    values = {name: float(context[name]) for name in names}
    offsets = {name: float(context.get((name,), 0.0)) for name in names}

    def evaluate(name, at):
        return float(_value(declaration, variable, units, {**values, name: at}))

    # Cache the local chart at the last representable parent of each law. The
    # continuous nodes around it differ only in their sub-ulp offsets.
    identifier = id(declaration)
    key = (tuple(values.values()), tuple(continuous), value)
    saved = charts.get(identifier)
    if saved is not None and saved[0] == key:
        gradient, anchor = saved[1:]
    else:
        try:
            primals = {name: jnp.asarray(x) for name, x in values.items()}
            gradient = _point_gradient(declaration, variable, units, primals)
        except (
            jax.errors.ConcretizationTypeError,
            jax.errors.TracerArrayConversionError,
        ):
            gradient = {}
        gradient = {name: float(gradient.get(name, np.nan)) for name in continuous}
        for name in continuous:
            if not np.isfinite(gradient[name]):
                # Host callbacks can use Python control flow. Resolve their
                # local slope beyond a single quantized child output.
                base = values[name]
                sign = offsets[name] if offsets[name] else 1.0
                span = np.sqrt(np.finfo(float).eps) * max(
                    abs(base), abs(offsets[name]), np.spacing(1.0)
                )
                other = base + np.copysign(span, sign)
                try:
                    moved = evaluate(name, other)
                except (ValueError, ArithmeticError):
                    moved = np.nan
                if not np.isfinite(moved):
                    other = np.nextafter(base, np.copysign(np.inf, sign))
                    moved = evaluate(name, other)
                gradient[name] = (moved - value) / (other - base)
        anchor = None
        if value in faces:
            for name in continuous:
                slope = gradient[name]
                if slope and np.isfinite(slope):
                    root = _level_chart(
                        lambda at: evaluate(name, at), values[name], value, slope
                    )
                    if root is not None:
                        anchor = name, root
                        break
        charts[identifier] = key, gradient, anchor

    # A child rounded to the face at several distinct parent values needs one
    # common crossing as its local origin. Its tangent at each rounded parent
    # alone would omit the rounding in the child's own value.
    if anchor is not None:
        name, root = anchor
        offsets[name] += values[name] - root
    return sum(gradient[name] * offsets[name] for name in continuous)
