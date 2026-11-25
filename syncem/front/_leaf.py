"""Conditional polarisation and beaming factors at individual kernel nodes."""

import jax.numpy as jnp
import numpy as np

from ..base.capabilities import Beaming
from ..base.continuum import ContinuumKernel
from ..distributions import AzimuthFactor, Conditional, Isotropic
from ..population._diagonal import traced_diagonal_knots
from ..population.declaration import ancestors, law_at, parents, placed_law
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
    if not {mu, eta}.intersection(resolved):
        # Keep the marginal as the integration variable. Eliminating an
        # ancestor of the other cosine would create a cycle after the tie.
        source_law = factors.get(source)
        if source in ancestors(factors, target) or (
            not parents(source_law) and not hasattr(source_law, "density")
        ):
            source, target = target, source
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
    breaks = (*getattr(law, "kinks", ()), *knots)
    edges = np.unique([lo, *(x for x in breaks if lo < x < hi), hi])
    n = max(24, getattr(law, "nodes", 24))
    nodes, weights = np.polynomial.legendre.leggauss(factor * n)
    radius = np.diff(edges) / 2
    x = (edges[:-1, None] + radius[:, None] * (nodes + 1)).reshape(-1)
    w = (radius[:, None] * weights).reshape(-1)
    return jnp.asarray(x), w * law.density(x)


def moving_beaming_rule(law, folded, target, units, variable, factor, point, edges):
    """Integrate the driver on panels that follow both conditional densities."""
    local = law_at(law, point, units)
    local = placed_law(local, variable)
    if getattr(local, "is_point", False) or not hasattr(local, "density"):
        return local.rule()
    n = factor * max(24, getattr(local, "nodes", 24))
    lo, hi = local.support
    if edges is None:
        edges = traced_diagonal_knots(folded, target, point, units, lo, hi)
    x, w = moving_panel_nodes(lo, hi, (*getattr(local, "kinks", ()), *edges), n)
    return x, w * local.density(x)


def moving_panel_nodes(lo, hi, breaks, n):
    """A fixed-shape Gaussian quadrature with differentiable panel edges."""
    edges = jnp.sort(jnp.clip(jnp.asarray((lo, hi, *breaks)), lo, hi))
    nodes, weights = np.polynomial.legendre.leggauss(n)
    widths = jnp.diff(edges) / 2
    x = edges[:-1, None] + widths[:, None] * (nodes + 1)
    # Collapsed panels may touch a pole where an otherwise unused kernel
    # derivative is singular. Evaluate zero-weight nodes in the interior.
    x = jnp.where(widths[:, None] > 0, x, (lo + hi) / 2)
    return x.ravel(), (widths[:, None] * weights).ravel()


def projection_panels(factors, ties, knots, legendre, moving, units, geometry):
    """Exclude angular panels where the independent folded density vanishes."""
    result = []
    for source, target in ties:
        law = factors.get(source)
        if target not in legendre:
            continue
        if target in moving:
            from ..engine.average import NodeRule

            def project(
                point, law=geometry[target], target=target, n=legendre[target][1]
            ):
                breaks = traced_diagonal_knots(law, target, point, units, -1.0, 1.0)
                return moving_panel_nodes(-1.0, 1.0, breaks, n)

            result.append((target, NodeRule(project, moving[target])))
            continue
        lo, hi = map(float, getattr(law, "support", (-1.0, 1.0)))
        lo, hi = max(-1.0, lo), min(1.0, hi)
        splits = [float(x) for x in knots.get(target, ()) if lo < x < hi]
        if lo < hi and (lo > -1 or hi < 1 or splits):
            edges = (lo, *splits, hi)
            result.append((target, tuple(zip(edges[:-1], edges[1:]))))
    return tuple(result)
