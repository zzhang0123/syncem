"""Averaged-kernel values at energy nodes, panel quadrature and the 17-node cell table.

Every direct integral of this prototype is a weighted sum of averaged-kernel values
Kbar(gamma_i) on a set of energy nodes. The values depend on the nodes only, so they are
computed once per node set and kept on disk (``cache/kbar_<hash>.npz``). Changing the
population changes only the weights.
"""
from __future__ import annotations

import hashlib
import os
import time

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial import chebyshev as cheb
from numpy.polynomial.legendre import leggauss

from . import setup as S

_RUN = {}


def _runner(batch=8):
    if batch not in _RUN:
        avg = S.kernel()
        _RUN[batch] = jax.jit(lambda xs: jax.lax.map(lambda g: avg.at({"gamma": g}), xs, batch_size=batch))
    return _RUN[batch]


def kbar(gammas: np.ndarray, *, label: str = "") -> tuple[np.ndarray, np.ndarray]:
    """(I (M, 68), P (M, 68) complex) of the averaged kernel at the energies ``gammas``."""
    g = np.ascontiguousarray(np.asarray(gammas, float))
    key = hashlib.sha256(g.tobytes()).hexdigest()[:20]
    path = S.CACHE / f"kbar_{key}.npz"
    if path.exists():
        d = np.load(path)
        return d["I"], d["P"]
    t0 = time.perf_counter()
    I, P = (np.asarray(a) for a in _runner()(jnp.asarray(g)))
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, gamma=g, I=I, P=P, seconds=time.perf_counter() - t0, label=label)
    os.replace(tmp, path)
    return I, P


def kbar_seconds(gammas) -> float:
    g = np.ascontiguousarray(np.asarray(gammas, float))
    key = hashlib.sha256(g.tobytes()).hexdigest()[:20]
    return float(np.load(S.CACHE / f"kbar_{key}.npz")["seconds"])


def log_panels(breaks, n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Gauss-Legendre panels in ln gamma between consecutive breaks.

    Returns nodes (M,), base weights (M,) with sum_i w_i f(g_i) = int f dgamma, and the
    panel number of each node.
    """
    t, w = leggauss(n)
    b = np.unique(np.asarray(breaks, float))
    s = np.log(b)
    xs, ws, ps = [], [], []
    for k, (a, c) in enumerate(zip(s[:-1], s[1:])):
        u = 0.5 * (c - a) * t + 0.5 * (c + a)
        xs.append(np.exp(u))
        ws.append(0.5 * (c - a) * w * np.exp(u))
        ps.append(np.full(n, k))
    return np.concatenate(xs), np.concatenate(ws), np.concatenate(ps)


def panel_quad(breaks, n: int):
    """Nodes, base weights and Kbar values (I, P) of :func:`log_panels`, cached panel by panel."""
    g, w, p = log_panels(breaks, n)
    Is, Ps = [], []
    for k in np.unique(p):
        I, P = kbar(g[p == k], label="panel")
        Is.append(I)
        Ps.append(P)
    return g, w, np.concatenate(Is), np.concatenate(Ps)


# ------------------------------------------------------------------ 17-node cell table
N_TABLE = 17


def lobatto(n: int = N_TABLE) -> np.ndarray:
    """Chebyshev-Lobatto points u_k = cos(pi k / (n - 1)) on [-1, 1]."""
    return np.cos(np.pi * np.arange(n) / (n - 1))


def table_values(reference: np.ndarray, half_width: np.ndarray, n: int = N_TABLE):
    """Kbar at n Chebyshev-Lobatto points of every log cell: (I, P) of shape (n_cells, n, 68)."""
    u = lobatto(n)
    g = np.asarray(reference)[:, None] * np.exp(np.asarray(half_width)[:, None] * u[None, :])
    vals = [kbar(row, label=f"table{n}") for row in g]  # cached cell by cell
    return np.stack([v[0] for v in vals]), np.stack([v[1] for v in vals])


def cheb_coefficients(values: np.ndarray) -> np.ndarray:
    """Chebyshev coefficients of the interpolant through Lobatto values, axis 1 of (cells, n, ...)."""
    n = values.shape[1]
    u = lobatto(n)
    V = cheb.chebvander(u, n - 1)  # (n, n)
    return np.einsum("kn,cn...->ck...", np.linalg.inv(V), values)


def table_columns(values: np.ndarray, order: int) -> np.ndarray:
    """(1/r!) d^r/du^r of the interpolant at u = 0, shape (cells, order + 1, ...)."""
    c = cheb_coefficients(values)
    out = []
    fact = 1.0
    d = c
    for r in range(order + 1):
        if r > 0:
            d = cheb.chebder(d, axis=1)
            fact *= r
        out.append(cheb.chebval(0.0, np.moveaxis(d, 1, 0)) / fact)
    return np.stack(out, axis=1)


def table_interpolate(values: np.ndarray, cell: np.ndarray, u: np.ndarray) -> np.ndarray:
    """Interpolant of cell ``cell[i]`` at local coordinate ``u[i]``, shape (M, ...)."""
    c = cheb_coefficients(values)  # (cells, n, ...)
    T = cheb.chebvander(np.asarray(u), values.shape[1] - 1)  # (M, n)
    return np.einsum("mn,mn...->m...", T, c[np.asarray(cell)])
