"""Joint measures with an explicit azimuth and direct integration over them.

The reference of every R5 test is direct integration of the full joint measure, with phi as
a measure variable: sum_i w_i exp(2i phi_i) K_P(x_i). No factor c is used on this side. A
law of phi given its parents enters only through its phi nodes.

* ``two_point_phi(c)``: phi = (arg c +- arccos |c|) / 2 with probability 1/2 each, so that
  E[exp(2i phi)] = c exactly. Valid for |c| <= 1, including 0 and 1.
* ``uniform_phi(centre, half, n)``: phi uniform on [centre - half, centre + half] by n
  Gauss-Legendre nodes, E[exp(2i phi)] = exp(2i centre) sin(2 half) / (2 half).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

from core.pair import Measure


def two_point_phi(c: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """phi nodes (M, 2) and probabilities (M, 2) with E[exp(2i phi)] = c."""
    c = np.asarray(c, complex)
    a = np.abs(c)
    if np.any(a > 1 + 1e-12):
        raise ValueError(f"|c| must be at most 1, got {a.max()}")
    psi, delta = np.angle(c), np.arccos(np.clip(a, 0.0, 1.0))
    phi = 0.5 * np.stack([psi + delta, psi - delta], axis=-1)
    return phi, np.full(phi.shape, 0.5)


def uniform_phi(centre: np.ndarray, half: np.ndarray, n: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """phi nodes (M, n) and probabilities (M, n) of U[centre - half, centre + half]."""
    t, w = leggauss(n)
    centre, half = np.asarray(centre, float)[:, None], np.asarray(half, float)[:, None]
    return centre + half * t[None, :], np.broadcast_to(0.5 * w[None, :], (centre.shape[0], n)).copy()


def uniform_phi_factor(centre, half):
    """Analytic E[exp(2i phi)] of ``uniform_phi``."""
    return np.exp(2j * np.asarray(centre)) * np.sinc(2 * np.asarray(half) / np.pi)


def with_phi(measure: Measure, law) -> Measure:
    """Joint measure: each node x_i is split over phi nodes of ``law(nodes) -> (phi, probs)``."""
    phi, probs = law(measure.nodes)
    k = phi.shape[1]
    nodes = {v: np.repeat(np.asarray(x), k) for v, x in measure.nodes.items()}
    nodes["phi"] = phi.ravel()
    return Measure(nodes, (measure.weights[:, None] * probs).ravel())


def factor_law(c_np):
    """Two-point law of phi given the parents, from c_np(nodes dict) -> (M,) complex."""
    return lambda nodes: two_point_phi(c_np(nodes))


_EVAL: dict = {}


def node_values(avg, nodes: dict, names: tuple, batch_size: int = 8):
    """avg.at at every node: I (M, n_ch), P (M, n_ch) complex."""
    key = (id(avg), names, batch_size)
    if key not in _EVAL:
        def run(xs):
            return jax.lax.map(lambda x: avg.at(dict(zip(names, x))), xs, batch_size=batch_size)

        _EVAL[key] = (avg, jax.jit(run))
    xs = tuple(jnp.asarray(nodes[v], float) for v in names)
    I, P = _EVAL[key][1](xs)
    return np.asarray(I), np.asarray(P)


def unique_nodes(measure: Measure, names: tuple) -> tuple[dict, np.ndarray]:
    """Distinct values of the non-phi coordinates and the inverse map of every node."""
    X = np.stack([np.asarray(measure.nodes[v], float) for v in names], axis=1)
    U, inv = np.unique(X, axis=0, return_inverse=True)
    return {v: U[:, i] for i, v in enumerate(names)}, inv.ravel()


def direct_joint(avg, measure: Measure, table=None, *, batch_size: int = 8, cache=None) -> np.ndarray:
    """(n_ch, 3) [I, Q, U] = sum_i w_i [K_I, exp(2i phi_i) K_P](x_i) over the joint measure.

    The kernel does not depend on phi, so it is evaluated once per distinct non-phi node
    (the sum is unchanged). ``cache``: a dict reused across calls with the same avg and nodes.
    With a table the nodes outside its cells are dropped (within="cells").
    """
    if table is not None:
        measure = measure.subset(table.locate(measure.nodes) >= 0)
    names = tuple(v for v in measure.nodes if v != "phi")
    uniq, inv = unique_nodes(measure, names)
    key = (id(avg), names, uniq[names[0]].size, float(np.sum(uniq[names[0]])))
    if cache is not None and key in cache:
        I, P = cache[key]
    else:
        I, P = node_values(avg, uniq, names, batch_size)
        if cache is not None:
            cache[key] = (I, P)
    w = measure.weights
    e2 = np.exp(2j * measure.nodes["phi"]) if "phi" in measure.nodes else np.ones_like(w)
    Itot = np.einsum("m,mc->c", w, I[inv])
    Ptot = np.einsum("m,mc->c", w * e2, P[inv])
    return np.stack([Itot.real, Ptot.real, Ptot.imag], axis=-1)


def mean_factor(measure: Measure) -> complex:
    """Population mean <exp(2i phi)> over a joint measure, the factorised <c>."""
    w = measure.weights
    return complex(np.sum(w * np.exp(2j * measure.nodes["phi"])) / np.sum(w))
