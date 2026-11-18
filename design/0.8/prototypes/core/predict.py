"""Forward prediction d = sum over rows of column * w_c * a, and direct integration.

Both return a (n_ch, 3) array [I, Q, U] with P = Q + iU. Without a spin-2 block the
constant azimuth factor multiplies P. ``direct`` evaluates the averaged kernel at every
measure node (``AveragedKernel.at``, so Legendre variables take their node values) with no
expansion. With a table it keeps the nodes inside the product cells (within="cells").
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from .lattice import ProductTable
from .pair import Measure


def _stokes(I, P) -> np.ndarray:
    return np.stack([np.real(I), np.real(P), np.imag(P)], axis=-1)


def predict(columns: dict, table: ProductTable, moments: dict, w_cells, azimuth=1.0) -> np.ndarray:
    wr = np.asarray(w_cells)[table.row_cell]
    a0 = moments["M0"]
    ax = "".join(chr(ord("k") + i) for i in range(a0.ndim - 1))
    spec = f"r,rc{ax},r{ax}->c"
    I = np.einsum(spec, wr, columns["I"], a0)
    if moments.get("M2") is not None:
        P = np.einsum(spec, wr, columns["P"], moments["M2"])
    else:
        P = azimuth * np.einsum(spec, wr, columns["P"], a0)
    return _stokes(I, P)


_DIRECT: dict = {}


def direct(avg, measure: Measure, table: ProductTable | None = None, *, azimuth=1.0,
           batch_size=8) -> np.ndarray:
    """sum_i weight_i Kbar(x_i), with exp(2i phi_i) on P when the measure has phi."""
    if table is not None:
        measure = measure.subset(table.locate(measure.nodes) >= 0)
    names = tuple(v for v in measure.nodes if v != "phi")
    key = (id(avg), names, batch_size)
    if key not in _DIRECT:
        def run(xs):
            return jax.lax.map(lambda x: avg.at(dict(zip(names, x))), xs, batch_size=batch_size)

        _DIRECT[key] = (avg, jax.jit(run))
    xs = tuple(jnp.asarray(measure.nodes[v], float) for v in names)
    I, P = (np.asarray(a) for a in _DIRECT[key][1](xs))
    w = measure.weights
    c = np.exp(2j * measure.nodes["phi"]) if "phi" in measure.nodes else azimuth * np.ones_like(w)
    return _stokes(w @ I, (w * c) @ P)


def representation_error(pred: np.ndarray, ref: np.ndarray) -> float:
    """max over channels and I, Q, U of |pred - ref| / I_ref."""
    return float(np.max(np.abs(pred - ref) / np.abs(ref[:, :1])))
