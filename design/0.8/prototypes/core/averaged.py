"""AveragedKernel: the point kernel averaged over specified laws of the averaged variables.

A rule maps the current point (resolved values, Legendre values and the values of
averaged variables listed before it) to (nodes (n,), probability weights (n,)). Rules are
JAX functions, so a rule that depends on a resolved variable is differentiated by the jet
(the derivatives of the conditional enter the columns, PAPER:659). Averages nest in the
order given. Legendre axes (Spec Sec. 4.2) integrate the kernel against
(2l+1)/2 P_l(x) on [-1, 1] and append one trailing axis of length L+1 per axis.
The azimuth factor is not applied here (predict and direct apply it).
"""
from __future__ import annotations

from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy.special import eval_legendre

Rule = Callable[[dict], tuple]


def fixed(nodes, probs) -> Rule:
    """A rule with nodes and weights independent of the point."""
    n, w = jnp.asarray(nodes, float), jnp.asarray(probs, float)
    return lambda point: (n, w)


def uniform(lo, hi, n: int) -> Rule:
    """U[lo, hi] by n Gauss-Legendre nodes. lo, hi may be callables of the point."""
    t, w = leggauss(n)
    t, w = jnp.asarray(t), jnp.asarray(w)

    def rule(point):
        a = lo(point) if callable(lo) else lo
        b = hi(point) if callable(hi) else hi
        return 0.5 * (b - a) * t + 0.5 * (b + a), 0.5 * w

    return rule


def legendre_rule(L: int, n: int) -> tuple:
    """Nodes (n,) on [-1, 1] and weights (n, L+1) = w_i (2l+1)/2 P_l(x_i)."""
    x, w = leggauss(n)
    P = np.stack([eval_legendre(l, x) for l in range(L + 1)], axis=1)
    return jnp.asarray(x), jnp.asarray(w[:, None] * P * (2 * np.arange(L + 1) + 1) / 2.0)


class AveragedKernel:
    """``avg(point) -> (I (n_ch, L...), P (n_ch, L...))`` at a resolved point.

    ``averaged``: sequence of (name, rule). ``legendre``: {name: (L, n_nodes)} projected by
    :meth:`__call__` and taken from the point by :meth:`at`. ``fixed``: values merged into
    every kernel point. ``outer_batch``: the outermost average runs as ``lax.map`` with this
    batch size, which bounds memory. None uses vmap at every level.
    """

    def __init__(self, kernel, averaged: Sequence[tuple[str, Rule]] = (), *, legendre=None,
                 fixed=None, outer_batch: int | None = 1):
        self.kernel = kernel
        self.averaged = tuple(averaged)
        self.legendre = dict(legendre or {})
        self.fixed = dict(fixed or {})
        self.outer_batch = outer_batch
        self.n_ch = kernel.n_ch
        self._leg = {v: legendre_rule(L, n) for v, (L, n) in self.legendre.items()}

    def _nest(self, point, levels, depth):
        if not levels:
            return self.kernel({**self.fixed, **point})
        (name, kind, rule), rest = levels[0], levels[1:]
        nodes, wts = rule(point)

        def inner(x):
            return self._nest({**point, name: x}, rest, depth + 1)

        if depth == 0 and self.outer_batch is not None:
            I, P = jax.lax.map(inner, nodes, batch_size=self.outer_batch)
        else:
            I, P = jax.vmap(inner)(nodes)
        if kind == "avg":
            return jnp.tensordot(wts, I, axes=1), jnp.tensordot(wts, P, axes=1)
        I = jnp.moveaxis(jnp.tensordot(wts, I, axes=([0], [0])), 0, 1)
        P = jnp.moveaxis(jnp.tensordot(wts, P, axes=([0], [0])), 0, 1)
        return I, P

    def __call__(self, point: dict):
        """Projected onto the Legendre axes, shapes (n_ch, L_1+1, ...)."""
        leg = [(v, "legendre", (lambda p, r=self._leg[v]: r)) for v in self.legendre]
        avg = [(v, "avg", r) for v, r in self.averaged]
        return self._nest(dict(point), tuple(leg + avg), 0)

    def at(self, point: dict):
        """At the Legendre variables' values in ``point``, shapes (n_ch,)."""
        missing = [v for v in self.legendre if v not in point]
        if missing:
            raise ValueError(f"point lacks Legendre variables {missing}")
        return self._nest(dict(point), tuple((v, "avg", r) for v, r in self.averaged), 0)
