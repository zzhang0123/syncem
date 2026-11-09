"""Private helpers of ``Channels.from_basis`` and ``Channels.from_callable``.

``from_basis``: an instrument response ``R_j = sum_k c_jk phi_k`` written in
reference responses ``phi_k`` (another ``Channels``). ``basis_parts`` checks
the coefficients ``(n_ch, n_ref)``, flattens a reference that is itself a
basis (``c -> c c'``) and returns each row's support: the hull of the
supports of the references with a nonzero coefficient, outside which ``R_j``
is exactly zero.

``from_callable``: a user function ``R(nu)`` (Hz in, response out, elementwise
and JAX-differentiable) on a declared support in ``(0, inf)``. The eager
checks evaluate it on the Gauss-Legendre nodes, on ``N_CHECK`` interior
points and one floating-point step inside each edge: finite real values of
the input's shape, a nonzero peak, a first derivative that JAX can trace,
vanishing edge values of ``R^(k)`` for ``k = 0 .. min(smoothness,
EDGE_ORDERS)``, and every declared ``derivative_bound``
(``{order: sup|R^(order)|}`` in Hz^-order times the response units) at
least the sampled ``|R^(order)|``. An edge value counts as vanishing when
``|R^(k)| w^k`` is at most ``EDGE_TOLERANCE`` of the sampled peak, with
``w`` the support half-width. The extension by zero of a ``C^k`` response
has these derivatives zero at the edges. These checks can refute a
declaration. Passing them does not verify it, because a sampled maximum is
not a supremum and an edge value one step inside is not the one-sided
limit.
"""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from ._channel_shapes import legendre_rule

EDGE_TOLERANCE = 1e-6
EDGE_ORDERS = 4
BOUND_SLACK = 1e-9
N_CHECK = 2048


# -- from_basis --------------------------------------------------------------------


def basis_parts(reference, coefficients):
    """``(reference, c (n_ch, n_ref), lo, hi)``, or ``ValueError`` on bad input."""
    try:
        c = np.asarray(coefficients)
    except Exception as exc:  # a tracer cannot fix the static support
        raise ValueError("from_basis needs concrete coefficients") from exc
    if np.iscomplexobj(c) or not np.issubdtype(c.dtype, np.number):
        raise ValueError("coefficients must be a real array")
    c = c.astype(float)
    if c.ndim == 1:
        c = c[None, :]
    if c.ndim != 2:
        raise ValueError("coefficients must be (n_ref,) or (n_ch, n_ref)")
    if reference.family == "basis":
        inner = np.asarray(reference.coefficients, dtype=float)
        if c.shape[1] != inner.shape[0]:
            raise ValueError(
                f"coefficients need {inner.shape[0]} columns, got {c.shape[1]}"
            )
        c = c @ inner
        reference = reference.reference_channels
    if c.shape[1] != reference.n_ch:
        raise ValueError(
            f"coefficients need one column per reference channel "
            f"({reference.n_ch}), got {c.shape[1]}"
        )
    if not np.all(np.isfinite(c)):
        raise ValueError("coefficients must be finite")
    used = c != 0.0
    if not np.all(np.any(used, axis=1)):
        raise ValueError("every row of coefficients needs a nonzero entry")
    support = np.asarray(reference.support, dtype=float)
    lo = np.array([support[row, 0].min() for row in used])
    hi = np.array([support[row, 1].max() for row in used])
    return reference, c, lo, hi


# -- from_callable -----------------------------------------------------------------


def callable_parts(fn, support):
    """``(fns, lo, hi)``: one callable per channel and the declared support."""
    fns = (fn,) if callable(fn) else tuple(fn) if _iterable(fn) else None
    if not fns or not all(callable(f) for f in fns):
        raise ValueError("fn must be a callable or a sequence of callables")
    if np.iscomplexobj(np.asarray(support)):
        raise ValueError("support must be real")
    edges = np.asarray(support, dtype=float)
    edges = edges[None, :] if edges.ndim == 1 else edges
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("support must have shape (2,) or (n_ch, 2)")
    lo, hi = edges[:, 0], edges[:, 1]
    if not (np.all(np.isfinite(edges)) and np.all(lo > 0) and np.all(hi > lo)):
        raise ValueError(
            "support must be finite with 0 < nu_lo < nu_hi (positive frequency)"
        )
    if len(fns) == 1:
        fns = fns * lo.size
    if len(fns) != lo.size:
        raise ValueError(f"{len(fns)} callables for {lo.size} support intervals")
    return fns, lo, hi


def _iterable(value):
    try:
        iter(value)
    except TypeError:
        return False
    return True


def derivative_bounds(derivative_bound, n_ch):
    """Static ``((order, (D_1, ..., D_n_ch)), ...)`` sorted by order, or ``None``."""
    if derivative_bound is None:
        return None
    if not isinstance(derivative_bound, Mapping):
        raise ValueError(
            "derivative_bound must be a mapping {order: sup|R^(order)|} or None"
        )
    out = []
    for order, value in derivative_bound.items():
        if not isinstance(order, int) or isinstance(order, bool) or order < 0:
            raise ValueError("derivative_bound orders must be integers >= 0")
        value = np.asarray(value)
        if np.iscomplexobj(value) or not np.issubdtype(value.dtype, np.number):
            raise ValueError("derivative_bound values must be real numbers")
        value = value.astype(float)
        if value.ndim == 0:
            value = np.full(n_ch, float(value))
        if (
            value.shape != (n_ch,)
            or not np.all(np.isfinite(value))
            or np.any(value < 0)
        ):
            raise ValueError(
                "derivative_bound values must be finite, nonnegative scalars or "
                f"({n_ch},) arrays"
            )
        out.append((order, tuple(float(v) for v in value)))
    return tuple(sorted(out))


def evaluate(fns, points):
    """``(n_ch, n)`` NumPy values of each callable at its row of ``points``."""
    rows = []
    for f, x in zip(fns, points):
        x = jnp.asarray(x)
        value = jnp.asarray(f(x))
        if value.shape != x.shape:
            raise ValueError(
                f"fn must return an array of the input's shape {x.shape}, "
                f"got {value.shape}"
            )
        if jnp.iscomplexobj(value):
            raise ValueError("fn must return real values")
        rows.append(np.asarray(value, dtype=float))
    out = np.stack(rows)
    if not np.all(np.isfinite(out)):
        raise ValueError("fn must return finite values on its support")
    return out


def _derivative(f, order):
    g = f
    for _ in range(order):
        g = jax.grad(g)
    return g


def _sampled_derivative(f, x, order):
    values = jax.vmap(_derivative(f, order))(jnp.asarray(x))
    return np.abs(np.asarray(values, dtype=float))


def _check_traceable(fns, lo, hi):
    for j, f in enumerate(fns):
        try:
            jax.grad(f)(jnp.asarray((lo[j] + hi[j]) / 2))
        except Exception as exc:
            raise ValueError(
                f"fn of channel {j} must be traceable and differentiable by JAX "
                f"(use jax.numpy): {type(exc).__name__}"
            ) from exc


def _check_edges(fns, inner, half, peak, smoothness):
    """Relative edge value, or ``ValueError`` when the edges refute ``smoothness``.

    ``|R^(k)|`` is evaluated one step inside each edge. A ``C^s`` response
    extended by zero has ``R^(k) = 0`` at the edges for ``k = 0 .. s``.
    Orders ``0 .. min(s, EDGE_ORDERS)`` are checked when ``s >= 1``.
    ``smoothness = 0`` allows a jump and only records it.
    """
    edge = 0.0
    for order in range(min(smoothness, EDGE_ORDERS) + 1):
        for j, f in enumerate(fns):
            values = _sampled_derivative(f, inner[j], order)
            if not np.all(np.isfinite(values)):
                raise ValueError(
                    f"R^({order}) of channel {j} is not finite at a support edge; "
                    "floor the argument near the edges (as the built-in bump "
                    "does) or declare a lower smoothness"
                )
            relative = float(np.max(values)) * half[j] ** order / peak[j]
            if order == 0:
                edge = max(edge, relative)
            if smoothness >= 1 and relative > EDGE_TOLERANCE:
                raise ValueError(
                    f"R^({order}) of channel {j} at a support edge is {relative:.3g} "
                    f"of the sampled peak (times the half-width^{order}); the "
                    f"extension by zero is then not C^{order}, so declare "
                    f"smoothness={max(order - 1, 0)}"
                )
    return edge


def check_points(lo, hi, n_nu):
    """``(n_ch, n_nu + N_CHECK)`` interior points: the nodes and a uniform grid."""
    x, _ = legendre_rule(n_nu)
    mid, half = (hi + lo) / 2, (hi - lo) / 2
    nodes = mid[:, None] + half[:, None] * np.asarray(x)[None, :]
    t = (np.arange(N_CHECK) + 0.5) / N_CHECK
    uniform = lo[:, None] + (hi - lo)[:, None] * t[None, :]
    return np.concatenate([nodes, uniform], axis=1)


def check_callable(fns, lo, hi, smoothness, bounds, n_nu) -> float:
    """Eager refutation checks (module docstring). Returns the relative edge value."""
    points = check_points(lo, hi, n_nu)
    values = evaluate(fns, points)
    peak = np.max(np.abs(values), axis=1)
    if np.any(peak == 0):
        raise ValueError("fn is zero at every check point of a support")
    _check_traceable(fns, lo, hi)
    inner = np.stack(
        [np.nextafter(lo, np.inf), np.nextafter(hi, -np.inf)], axis=1
    )  # (n_ch, 2)
    edge = _check_edges(fns, inner, (hi - lo) / 2, peak, smoothness)
    for order, declared in bounds or ():
        for j, (f, x) in enumerate(zip(fns, points)):
            sampled = float(np.max(_sampled_derivative(f, x, order)))
            if not np.isfinite(sampled):
                raise ValueError(
                    f"derivative_bound: R^({order}) is not finite on the support "
                    f"of channel {j}"
                )
            if sampled > declared[j] * (1.0 + BOUND_SLACK):
                raise ValueError(
                    f"derivative_bound[{order}] = {declared[j]:.6g} for channel {j} "
                    f"is below the sampled max |R^({order})| = {sampled:.6g}"
                )
    return edge


def callable_response(fns, nu, inside, centres):
    """Masked ``(n_ch, *shape)`` values of the callables (exactly zero outside)."""
    safe = jnp.where(inside, nu, centres.reshape((-1,) + (1,) * (nu.ndim - 1)))
    rows = []
    for j, f in enumerate(fns):
        value = jnp.asarray(f(safe[j]))
        if value.shape != safe[j].shape:
            raise ValueError(
                f"fn must return an array of the input's shape {safe[j].shape}, "
                f"got {value.shape}"
            )
        rows.append(value)
    return jnp.where(inside, jnp.stack(rows), 0.0)


def describe_fn(f) -> str:
    return getattr(f, "__qualname__", None) or type(f).__name__


__all__ = [
    "EDGE_TOLERANCE",
    "N_CHECK",
    "basis_parts",
    "callable_parts",
    "derivative_bounds",
    "check_callable",
    "check_points",
    "callable_response",
    "evaluate",
    "describe_fn",
]
