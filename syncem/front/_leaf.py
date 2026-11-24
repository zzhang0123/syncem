"""Conditional polarisation and beaming factors at individual kernel nodes."""

import jax.numpy as jnp
import numpy as np

from ..base.capabilities import Beaming
from ..base.continuum import ContinuumKernel
from ..distributions import AzimuthFactor, Conditional, Isotropic
from ..population.declaration import law_at, parents
from ..values.errors import RoleError


def spin_factor(law):
    if hasattr(law, "circular"):
        return law.circular(2)
    if hasattr(law, "rule"):
        angles, weights = law.rule()
        return jnp.sum(weights * jnp.exp(2j * angles))
    return jnp.exp(2j * law)


def conditional_spin(law, units):
    def factor(point):
        if isinstance(law, AzimuthFactor) and isinstance(law.c, Conditional):
            value = law_at(law.c, point, units)
            return (
                value if hasattr(value, "circular") else AzimuthFactor(value)
            ).circular(2)
        return spin_factor(law_at(law, point, units))

    return factor


def beam(kernel, factors, resolved, units):
    beaming = next((c for c in kernel.capabilities if isinstance(c, Beaming)), None)
    if beaming is None:
        return (), ()
    mu, eta = beaming.variable, beaming.partner
    if mu not in factors or (mu in resolved and eta in resolved):
        return (), ()
    if (
        type(kernel) is ContinuumKernel
        and mu not in resolved
        and isinstance(factors[mu], Isotropic)
        and not any(mu in parents(law) for name, law in factors.items() if name != mu)
    ):
        return (), ()
    source, target = (eta, mu) if mu in resolved else (mu, eta)
    law = factors.get(source)
    if law is None or (not hasattr(law, "density") and not parents(law)):
        raise RoleError(f"Beaming needs a density on the averaged variable {source}")

    def evaluate(point):
        density = law_at(law, point, units).density
        return 2 * density(point[target])

    return ((source, target),), (("all", evaluate),)


def split_beaming_rule(law, knots, *, factor=1):
    """Rule on the intervals where the multiplier has one formula."""
    lo, hi = law.support
    edges = np.array([lo, *(x for x in knots if lo < x < hi), hi])
    n = max(24, getattr(law, "nodes", 24))
    nodes, weights = np.polynomial.legendre.leggauss(factor * n)
    radius = np.diff(edges) / 2
    x = (edges[:-1, None] + radius[:, None] * (nodes + 1)).reshape(-1)
    w = (radius[:, None] * weights).reshape(-1)
    return jnp.asarray(x), w * law.density(x)


def projection_panels(factors, ties, knots, legendre):
    """Exclude angular panels where the independent folded density vanishes."""
    result = []
    for source, target in ties:
        law = factors.get(source)
        if target not in legendre or parents(law) or not hasattr(law, "support"):
            continue
        lo, hi = (float(x) for x in law.support)
        lo, hi = max(-1.0, lo), min(1.0, hi)
        splits = [x for x in knots.get(target, ()) if lo < x < hi]
        if lo < hi and (lo > -1 or hi < 1 or splits):
            edges = (lo, *splits, hi)
            result.append((target, tuple(zip(edges[:-1], edges[1:]))))
    return tuple(result)
