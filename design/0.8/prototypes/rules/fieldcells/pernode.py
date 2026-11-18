"""Field cells with gamma averaged: the per-node mode of Spec Sec. 8.3 (0.9) and a naive shortcut.

Field cells alone, with gamma averaged by a rule (nodes gamma_q, probabilities p_q):
    Kbar_avg(B) = sum_q p_q Kbar(gamma_q, B).
The 0.9 per-node mode computes, for a log field chart u_B = ln(B/B_b)/h_B,
    (1/n!) d^n/du_B^n Kbar_avg = (h_B^n/n!) sum_q p_q (1 + (1/2) d/dln gamma)^n Kbar(gamma_q, B_b)
from energy derivatives at each node. It holds when nodes and probabilities do not depend on B.
``homogeneous_columns`` is the shortcut that treats the averaged kernel as an infinite power
law, Kbar_avg(L B) = L^((s+1)/2) Kbar_avg(B). It ignores the range of the electron law.
"""
from __future__ import annotations

import math

import numpy as np

import core as C
from core.lattice import ProductTable


def node_energy_table(gamma_nodes, B_ref, nmax) -> ProductTable:
    """Two-variable table at (gamma_q, B_b) for every node and field cell, order nmax in ln gamma only."""
    g = np.repeat(np.asarray(gamma_nodes, float), len(B_ref))
    b = np.tile(np.asarray(B_ref, float), len(gamma_nodes))
    n = g.size
    return ProductTable(("gamma", "B"), np.zeros((n, 2), int), np.zeros((n, 2)), np.full((n, 2), np.inf),
                        np.c_[g, b], np.ones((n, 2)), tuple(("log", "log") for _ in range(n)),
                        tuple((nmax, 0) for _ in range(n)), nmax)


def pernode_columns(avg_point, B_cells: C.Cells, gamma_nodes, gamma_probs, N: int) -> dict:
    """Columns (n_B_cells * (N+1), n_ch) of log field cells from per-node energy derivatives.

    ``avg_point`` is the averaged kernel without the gamma average (point: gamma, B).
    The B and probabilities are frozen at each cell's reference, so no derivative of the
    electron law enters.
    """
    if B_cells.chart != "log":
        raise ValueError("the per-node identity is written for the log field chart")
    tab = node_energy_table(gamma_nodes, B_cells.reference, N)
    E = C.build_columns(avg_point, tab)  # rows: per (node, cell), n = 0..N, coefficients in t = ln gamma
    nB, nq = B_cells.n, len(gamma_nodes)
    p = np.asarray(gamma_probs, float)
    out = {}
    for key in ("I", "P"):
        G = E[key].reshape((nq, nB, N + 1) + E[key].shape[1:])  # g_k = (1/k!) d^k/dt^k
        cols = []
        for b in range(nB):
            h = B_cells.half_width[b]
            for n in range(N + 1):
                # (d/dlnB)^n = sum_k C(n,k) (1/2)^k d^k/dt^k, d^k/dt^k = k! g_k
                acc = sum(math.comb(n, k) * 0.5**k * math.factorial(k) * np.tensordot(p, G[:, b, k], axes=1)
                          for k in range(n + 1))
                cols.append(h**n / math.factorial(n) * acc)
        out[key] = np.stack(cols)
    return out


def homogeneous_columns(avg_gamma, B_cells: C.Cells, slope: float, N: int) -> dict:
    """The power-law shortcut: (h^n/n!) ((slope+1)/2)^n Kbar_avg(B_b). Wrong at the range edges."""
    k = 0.5 * (slope + 1.0)
    out = {"I": [], "P": []}
    for b in range(B_cells.n):
        I, P = (np.asarray(a) for a in avg_gamma({"B": B_cells.reference[b]}))
        h = B_cells.half_width[b]
        for n in range(N + 1):
            f = (h * k) ** n / math.factorial(n)
            out["I"].append(f * I)
            out["P"].append(f * P)
    return {k_: np.stack(v) for k_, v in out.items()}


_VALUES: dict = {}


def node_values(avg, measure: C.Measure, batch_size=4) -> tuple[np.ndarray, np.ndarray]:
    """Averaged kernel at every node of ``measure``: I (M, n_ch), P (M, n_ch). For direct sums on subsets."""
    import jax
    import jax.numpy as jnp

    names = tuple(measure.nodes)
    key = (id(avg), names, batch_size)
    if key not in _VALUES:
        def run(xs):
            return jax.lax.map(lambda x: avg.at(dict(zip(names, x))), xs, batch_size=batch_size)

        _VALUES[key] = (avg, jax.jit(run))
    xs = tuple(jnp.asarray(measure.nodes[v], float) for v in names)
    I, P = _VALUES[key][1](xs)
    return np.asarray(I), np.asarray(P)
