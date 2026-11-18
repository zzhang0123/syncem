"""Jet provider: Taylor coefficients (1/r!) d^r/du^r on the lower-set index of each product cell.

Nested ``jax.jacfwd`` in the active directions (variables with a positive exponent in the
index), with lower orders carried as auxiliary outputs, as ``taylor_tensors`` of
``syncem/model/_basis_taylor.py`` generalised to d directions. The cost per cell is about
(1 + d_active)^levels primal evaluations, levels = largest total degree. Cells with equal
signature (charts, exponents) share one jitted routine, mapped over cells by ``lax.map``.
"""
from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from .lattice import ProductTable, from_u


def taylor_coefficients(f, d: int, exponents):
    """Pytree with leading axis len(exponents): (1/alpha!) d^alpha f at z = 0, f: (d,) -> pytree."""
    exponents = tuple(tuple(int(v) for v in e) for e in exponents)
    active = [v for v in range(d) if any(e[v] > 0 for e in exponents)]
    levels = max(sum(e) for e in exponents)
    pos = jnp.asarray(active, dtype=int)

    def g(y):
        return f(jnp.zeros(d, dtype=y.dtype).at[pos].set(y) if active else jnp.zeros(d))

    def level0(y):
        v = g(y)
        return v, (v,)

    fn = level0
    for _ in range(levels):
        def fn(y, prev=fn):
            out, lower = jax.jacfwd(prev, has_aux=True)(y)
            return out, lower + (out,)

    tensors = fn(jnp.zeros(len(active)))[1]
    picks = []
    for e in exponents:
        idx = tuple(k for k, v in enumerate(active) for _ in range(e[v]))
        norm = math.prod(math.factorial(r) for r in e)
        picks.append(jax.tree.map(lambda a, i=idx, s=norm: a[(Ellipsis,) + i] / s, tensors[sum(e)]))
    return jax.tree.map(lambda *xs: jnp.stack(xs), *picks)


_COMPILED: dict = {}


def _cell_routine(avg, table: ProductTable, signature):
    key = (id(avg), table.variables, signature)
    if key not in _COMPILED:
        charts, exponents = signature
        names, d = table.variables, len(table.variables)

        def one(args):
            ref, hw = args

            def f(z):
                return avg({v: from_u(charts[i], z[i], ref[i], hw[i]) for i, v in enumerate(names)})

            return taylor_coefficients(f, d, exponents)

        # avg is stored with the routine so that its id stays valid
        _COMPILED[key] = (avg, jax.jit(lambda refs, hws: jax.lax.map(one, (refs, hws))))
    return _COMPILED[key][1]


def build_columns(avg, table: ProductTable) -> dict:
    """Basis columns in lattice rows: {"I": (n_rows, n_ch, L...), "P": complex, same shape}."""
    groups: dict = {}
    for c in range(table.n_cells):
        groups.setdefault(table.signature(c), []).append(c)
    offsets = table.offsets
    out_I, out_P = [None] * table.n_rows, [None] * table.n_rows
    for sig, cells in groups.items():
        I, P = _cell_routine(avg, table, sig)(jnp.asarray(table.reference[cells]),
                                              jnp.asarray(table.half_width[cells]))
        I, P = np.asarray(I), np.asarray(P)
        for k, c in enumerate(cells):
            for m in range(offsets[c + 1] - offsets[c]):
                out_I[offsets[c] + m], out_P[offsets[c] + m] = I[k, m], P[k, m]
    return {"I": np.stack(out_I), "P": np.stack(out_P)}
