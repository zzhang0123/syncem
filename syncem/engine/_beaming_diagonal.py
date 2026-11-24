"""Legendre projection of the Beaming diagonal, which carries the kernel factor 2."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np


def diagonal_weight(point):
    """Convert the isotropic-pitch average into the beaming diagonal kernel."""
    return 2.0


def pair_weights(left, right, nodes):
    """One common Gaussian rule with an outer product of scaled Legendre values."""
    x, w = np.polynomial.legendre.leggauss(nodes)
    a = np.polynomial.legendre.legvander(x, left) * (2 * np.arange(left + 1) + 1) / 2
    b = np.polynomial.legendre.legvander(x, right) * (2 * np.arange(right + 1) + 1) / 2
    return jnp.asarray(x), jnp.asarray(w[:, None, None] * a[:, :, None] * b[:, None, :])


def project_pair(modes, weights, span):
    """Integrate both Legendre degrees and restore the requested axis order."""
    i, v, p = modes

    def spin0(values):
        projected = jnp.einsum("nc...,nlk->clk...", values, weights)
        return jnp.moveaxis(projected, 2, 1 + span)

    spin2 = jnp.einsum("nwc...,nlk->wclk...", p, weights)
    return spin0(i), spin0(v), jnp.moveaxis(spin2, 3, 2 + span)


def point_weights(x, degree):
    """Scaled Legendre values evaluated at a traced celled cosine."""
    terms = [jnp.ones_like(x)]
    if degree:
        terms.append(x)
    for k in range(1, degree):
        terms.append(((2 * k + 1) * x * terms[k] - k * terms[k - 1]) / (k + 1))
    return jnp.stack(terms) * (2 * jnp.arange(degree + 1) + 1) / 2


def project_point(modes, weights):
    """Insert the projected degree axis at the current celled coordinate."""
    i, v, p = modes
    return (
        jnp.einsum("c...,l->cl...", i, weights),
        jnp.einsum("c...,l->cl...", v, weights),
        jnp.einsum("wc...,l->wcl...", p, weights),
    )
