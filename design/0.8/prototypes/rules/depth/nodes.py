"""Kernel values and energy columns at every channel-quadrature frequency.

The closed-form depth route of Spec Sec. 4.3 multiplies the energy columns by the phase
weights exp(i tau d_c) (i tau h_c)^s / s! inside the channel integral. The weights depend
on the frequency, so the energy columns are needed per quadrature node (j, t) before the
sum over t. :func:`node_channels` turns J channels of T nodes into J*T unit channels of one
node each, so the core kernel returns the integrand at every node. Results are reshaped to
(..., J, T).
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.legendre import leggauss

import core as C
from syncem.constants import C_SI_M


def node_channels(ch: C.Channels) -> C.Channels:
    """J*T unit channels with one node each, in the order of ch.nu.ravel()."""
    return C.Channels(ch.nu.reshape(-1, 1), np.ones((ch.nu.size, 1)))


def tau(ch: C.Channels) -> np.ndarray:
    """tau = 2 lambda^2 in m^2 at every quadrature node, shape (J, T)."""
    return 2.0 * (C_SI_M / ch.nu) ** 2


def tau_max(ch: C.Channels) -> float:
    """tau at the lowest frequency of the channel supports (Spec Sec. 4.6 rule 4)."""
    return float(np.max(tau(ch)))


class PhaseFree:
    """A kernel with the depth value dropped before the call, so P carries no Faraday phase.

    Averaged laws that take depth as a parent still see it (AveragedKernel passes the full
    point to the rules), so this is the phase-free averaged kernel of Spec Sec. 4.3 bullet 3.
    """

    def __init__(self, kernel):
        self.kernel = kernel
        self.n_ch = kernel.n_ch

    def __call__(self, p):
        return self.kernel({k: v for k, v in p.items() if k != "depth"})


def eta_rule(n=12):
    """eta ~ U[-1/2, 1/2] on the half range [0, 1/2], as core.paper (the kernel is even in eta)."""
    e, we = leggauss(n)
    return C.fixed(0.25 * (e + 1.0), we * 0.5)


def node_kernel(ch: C.Channels, kind="e1", n_nodes_F=128, outer_batch=1, extra=()):
    """AveragedKernel whose outputs are (J*T,) node values without any Faraday phase.

    kind "e1": B ~ U[4, 6] uG (16 nodes) and eta on [0, 1/2] (12 nodes), the E1 laws.
    kind "cheap": B = 5 uG and eta on [0, 1/2] with 4 nodes.
    ``extra`` prepends averaged (name, rule) pairs, for example a depth-dependent B law.
    """
    K = C.ContinuumChannelKernel(node_channels(ch), n_nodes_F=n_nodes_F)
    if kind == "e1":
        laws = [("B", C.uniform(4.0, 6.0, 16)), ("eta", eta_rule(12))]
        fixed = {}
    elif kind == "cheap":
        laws, fixed = [("eta", eta_rule(4))], {"B": 5.0}
    else:
        raise ValueError(f"unknown kernel kind {kind!r}")
    return C.AveragedKernel(K, list(extra) + laws, fixed=fixed, outer_batch=outer_batch)


def energy_columns(avg, gamma_table: C.ProductTable, J: int) -> dict:
    """Jet energy columns at every node, {"I": (n_rows, J, T) real, "P": complex}."""
    cols = C.build_columns(avg, gamma_table)
    return {k: v.reshape(v.shape[0], J, -1) for k, v in cols.items()}


_VALUES: dict = {}


def values_at(avg, points: dict, J: int, batch_size=8):
    """Averaged kernel at each point (no expansion), shapes (M, J, T)."""
    names = tuple(points)
    key = (id(avg), names, batch_size)
    if key not in _VALUES:
        def run(xs):
            return jax.lax.map(lambda x: avg.at(dict(zip(names, x))), xs, batch_size=batch_size)

        _VALUES[key] = (avg, jax.jit(run))
    xs = tuple(jnp.asarray(points[v], float) for v in names)
    I, P = (np.asarray(a) for a in _VALUES[key][1](xs))
    return I.reshape(I.shape[0], J, -1), P.reshape(P.shape[0], J, -1)
