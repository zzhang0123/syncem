"""Support and density knots restricted to a beaming diagonal.

At a fixed value of the other parents, a boundary b(q) changes the
integrand at q = b(q). A finite sign scan can resolve bracketed crossings.
It supplies no bound for unresolved oscillations of a user function.
"""

from dataclasses import dataclass, replace
from functools import lru_cache

import jax
import jax.numpy as jnp
import numpy as np

from .declaration import ancestors, law_at, ordered, parents

# Bracket each boundary separately, so two close parallel boundaries can
# define an arbitrarily narrow emitting interval between distinct roots.
_BRACKETS = 32


def conditional_crossings(factors, pair):
    """Whether a folded density may have knots depending on the driving cosine."""
    a, b = pair
    return a in ancestors(factors, b) or b in ancestors(factors, a)


def _knots(law):
    if not hasattr(law, "density") or getattr(law, "support", None) is None:
        return jnp.empty(0)
    return jnp.asarray((*law.support, *getattr(law, "kinks", ())))


@dataclass(frozen=True)
class KnotPaths:
    """Folded density knots at each quadrature node of its intermediate parents."""

    law: object
    chain: tuple
    dependencies: tuple
    rule_nodes: object
    host: bool

    def evaluate(self, point, units):
        def walk(i, values):
            if i == len(self.chain):
                return _knots(law_at(self.law, values, units))
            name, declaration = self.chain[i]
            local = law_at(declaration, values, units)
            nodes = self.rule_nodes(name, local, values)

            def next_level(node):
                return walk(i + 1, {**values, name: node})

            if self.host:
                rows = [np.asarray(next_level(node)) for node in nodes]
                return np.concatenate(rows) if rows else np.empty(0)
            return jax.vmap(next_level)(nodes).ravel()

        return walk(0, point)


def knot_paths(factors, source, driver, rule_nodes, *, given=(), host=False):
    """Follow the driver through the conditional graph at fixed quadrature indices.

    Independent parents stay outside the angular integral. The moving parents
    are sampled by the same rules as the integral whose panels are being split.
    """
    law = factors[source]
    chain = {
        name: factors[name]
        for name in ancestors(factors, source, given=given)
        if name in factors and driver in ancestors(factors, name, given=given)
    }
    if not chain:
        return law
    inputs = set(parents(law)).union(*(parents(decl) for decl in chain.values()))
    dependencies = tuple(sorted(inputs.difference(chain)))
    return KnotPaths(
        law, ordered(chain, available=dependencies), dependencies, rule_nodes, host
    )


def uses_panel_nodes(law, integrated):
    return isinstance(law, KnotPaths) and any(
        name in integrated for name, _ in law.chain
    )


def on_panel_nodes(law, integrated, panel_nodes):
    if not uses_panel_nodes(law, integrated):
        return law

    def nodes(name, local, point):
        rule = panel_nodes if name in integrated else law.rule_nodes
        return rule(name, local, point)

    return replace(law, rule_nodes=nodes, host=True)


def knot_parents(law):
    return law.dependencies if isinstance(law, KnotPaths) else parents(law)


def _at(law, point, units):
    if isinstance(law, KnotPaths):
        return law.evaluate(point, units)
    return _knots(law_at(law, point, units))


def fixed_angular_knots(law, units):
    """Return knots independent of every parent, or None for moving geometry.

    Abstract parents distinguish constant coordinates from a density whose
    support merely agrees at a few sampled points. No parent values are
    chosen and the density itself may vary arbitrarily.
    """
    names = parents(law)
    found = []

    def inspect(*values):
        with jax.ensure_compile_time_eval():
            edges = _knots(law_at(law, dict(zip(names, values)), units))
            if not isinstance(edges, jax.core.Tracer):
                found.append(tuple(np.unique(np.asarray(edges))))
        return ()

    try:
        jax.eval_shape(inspect, *(jax.ShapeDtypeStruct((), float) for _ in names))
    except (jax.errors.ConcretizationTypeError, jax.errors.TracerArrayConversionError):
        return None
    return found[0] if found else None


def host_diagonal_knots(law, driver, point, units, interval=(-1.0, 1.0)):
    """Return interior integration breaks at the current external parents."""
    lower, upper = interval
    if lower >= upper:
        return ()

    @lru_cache(maxsize=128)
    def knots(q):
        return _at(law, {**point, driver: q}, units)

    if driver not in knot_parents(law):
        values = np.asarray(knots(lower))
    else:
        from scipy.optimize import brentq, minimize_scalar

        grid = np.linspace(lower, upper, _BRACKETS + 1)
        sampled = [np.asarray(knots(q)) for q in grid]
        # Node-dependent atom/density or table choices can change the number
        # of breaks. A missing break has no bracket on that interval.
        size = max(map(len, sampled))
        residual = np.full((grid.size, size), np.nan)
        for i, row in enumerate(sampled):
            residual[i, : len(row)] = grid[i] - row
        roots = []
        for k, column in enumerate(residual.T):
            if np.all(column == 0):
                continue

            def boundary(q):
                row = knots(q)
                return q - float(row[k]) if k < len(row) else np.nan

            # Insert sampled turning points before bracketing. This separates
            # two crossings of the same boundary around a narrow extremum.
            delta = np.diff(column)
            turns = np.flatnonzero(delta[:-1] * delta[1:] < 0) + 1
            candidates = list(zip(grid, column))
            for i in turns:
                sign = np.sign(delta[i])
                extremum = minimize_scalar(
                    lambda q: sign * boundary(q),
                    bounds=(grid[i - 1], grid[i + 1]),
                    method="bounded",
                    options={"xatol": 1e-14},
                )
                if extremum.success:
                    candidates.append((extremum.x, sign * extremum.fun))
            q, f = np.array(sorted(candidates)).T
            roots.extend(q[f == 0])
            for i in np.flatnonzero(
                (f[:-1] != 0)
                & (f[1:] != 0)
                & np.isfinite(f[:-1])
                & np.isfinite(f[1:])
                & (np.signbit(f[:-1]) != np.signbit(f[1:]))
            ):
                roots.append(
                    brentq(boundary, q[i], q[i + 1], xtol=np.finfo(float).tiny)
                )
        values = np.array(roots)
    return tuple(map(float, np.unique(values[(lower < values) & (values < upper)])))


def _bisect(fn, left, right, f_left):
    def step(_, ends):
        a, b = ends
        mid = (a + b) / 2
        same = jnp.signbit(fn(mid)) == jnp.signbit(f_left)
        return jnp.where(same, mid, a), jnp.where(same, b, mid)

    a, b = jax.lax.fori_loop(0, 55, step, (left, right))
    return (a + b) / 2


def traced_diagonal_knots(law, driver, point, units, lo, hi):
    """Panel edges as functions of the external coordinates for a jet."""
    if driver not in knot_parents(law):
        return _at(law, point, units)

    def boundaries(q):
        return _at(law, {**point, driver: q}, units)

    grid = jnp.linspace(lo, hi, _BRACKETS + 1)
    residual = grid[:, None] - jax.vmap(boundaries)(grid)
    count = residual.shape[1]
    if not count:
        return jnp.empty(0)
    k = jnp.arange(count)
    slope_fn = jax.jacfwd(boundaries)
    slopes = 1.0 - jax.vmap(slope_fn)(grid)
    turns = jnp.signbit(slopes[:-1]) != jnp.signbit(slopes[1:])

    # Insert a stationary point between adjacent crossings of a single
    # boundary. These extra coordinates only bracket the final roots.
    mid = jnp.broadcast_to((grid[:-1, None] + grid[1:, None]) / 2, turns.shape)

    def extremum(left, right, f_left, index):
        return _bisect(lambda q: 1.0 - slope_fn(q)[index], left, right, f_left)

    extrema = jax.vmap(
        jax.vmap(extremum, in_axes=(None, None, 0, 0)), in_axes=(0, 0, 0, None)
    )
    middle = jax.lax.cond(
        jnp.any(turns),
        lambda: jnp.where(turns, extrema(grid[:-1], grid[1:], slopes[:-1], k), mid),
        lambda: mid,
    )
    middle = jax.lax.stop_gradient(middle)
    edge = jnp.broadcast_to(grid[:, None], residual.shape)
    q = jnp.stack((edge[:-1], middle), axis=1).reshape(-1, count)
    q = jnp.concatenate((q, edge[-1:]))
    middle_residual = jax.vmap(
        jax.vmap(lambda x, i: x - boundaries(x)[i]), in_axes=(0, None)
    )(middle, k)
    f = jnp.stack((residual[:-1], middle_residual), axis=1).reshape(-1, count)
    f = jnp.concatenate((f, residual[-1:]))

    def crossing(left, right, fl, fr, after, index):
        # Include an exact grid root only if its neighbours have opposite
        # signs. A tangent touch or an edge identical to q needs no split.
        opposite = jnp.signbit(fl) != jnp.signbit(fr)
        exact = (fr == 0) & (after != 0) & (jnp.signbit(fl) != jnp.signbit(after))
        active = (
            jnp.isfinite(fl)
            & jnp.isfinite(fr)
            & (fl != 0)
            & (((fr != 0) & opposite) | exact)
        )

        def equation(q):
            return jnp.where(active, q - boundaries(q)[index], q - lo)

        def solve(fn, _):
            root = _bisect(fn, left, right, fl)
            root = jnp.where(fr == 0, right, root)
            return jnp.where(active, root, lo)

        # Implicit differentiation includes motion of an isolated simple
        # crossing. At a tangency no such local Taylor expansion exists.
        return jax.lax.custom_root(equation, left, solve, lambda g, y: y / g(1.0))

    brackets = jax.vmap(
        jax.vmap(crossing),
        in_axes=(0, 0, 0, 0, 0, None),
    )
    return brackets(
        q[:-1],
        q[1:],
        f[:-1],
        f[1:],
        jnp.concatenate((f[2:], f[-1:])),
        k,
    ).ravel()
