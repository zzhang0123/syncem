"""The core column provider: the unchanged ``_build_core`` per product cell (design/0.8/SPEC.md Sec. 8.3).

LABEL ``eq: illustrated derivative spectra``. For product cells whose celled variables are ``gamma`` and
optionally ``B``, the averaged kernel goes through ``engine.average.core_adapter`` into the jitted
``syncem.model.basis._build_core``, which takes Taylor coefficients by nested forward mode in the affine
displacements ``z_v = (x_v - x_b) / s_v``. The scale is ``s_v = x_b`` for a log chart and ``s_v = h_v`` for an
affine chart, so the affine coefficients are already the chart coefficients and the log coefficients follow by
``engine.expand.log_from_affine`` (``z = exp(h u) - 1``). The result is the column of every exponent in the
cell's lower set, in the order of the lattice index.

This is the fast path and oracle of the jet at ``d <= 2`` (SPEC Sec. 8.3). The columns carry the units of the
kernel. Shapes:
``I`` ``(n_rows, n_ch)``, ``P`` ``(n_rows, n_ch)`` complex, one row per exponent. Not certified: the averages
and channel quadrature of the averaged kernel, validated against the study's analytic stack.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from ..model._population import Reference
from ..model.basis import _build_core
from ..model.index import MomentIndex as _ModelIndex
from ..model.index import Truncation
from .average import AveragedKernel, core_adapter
from .expand import log_from_affine

__all__ = ["CORE_VARIABLES", "core_cell", "core_columns"]

CORE_VARIABLES = ("gamma", "B")


def core_columns(
    averaged: AveragedKernel,
    channels: Any,
    axes: dict[str, tuple[str, float, float]],
    exponents: list[tuple[int, ...]],
) -> tuple[Any, Any]:
    """``(I, P)`` columns, one row per exponent tuple over ``axes`` (``{v: (chart, reference, half_width)}``).

    ``axes`` holds ``gamma`` and optionally ``B``. A variable without an axis must be averaged or fixed in the
    averaged kernel.
    """
    names = tuple(axes)
    if not names or names[0] != "gamma" or not set(names) <= set(CORE_VARIABLES):
        raise ValueError(
            f"the core provider takes cells on gamma and optionally B, not {names}"
        )
    caps = [max(e[i] for e in exponents) for i in range(len(names))]
    total = max(sum(e) for e in exponents)
    resolved = {n for n, _ in averaged.averaged} | {n for n, _ in averaged.fixed}
    clash = set(names) & resolved
    if clash:
        raise ValueError(
            f"{sorted(clash)} are celled and also averaged or fixed in the averaged kernel. Take each "
            "variable in one role"
        )
    if "B" not in names and "B" not in resolved:
        raise ValueError(
            "B is neither celled nor averaged nor fixed. Put B on the lattice, or pass it in averaged= or fixed= of "
            "the AveragedKernel"
        )
    chart_g, ref_g, h_g = axes["gamma"]
    chart_b, ref_b, h_b = axes.get("B", ("affine", 1.0, 1.0))
    scales = (
        ref_g if chart_g == "log" else h_g,
        ref_b if chart_b == "log" else h_b,
        1.0,
    )
    reference = Reference(ref_g, ref_b, 0.0, scales=scales)
    cap_b = caps[1] if len(names) == 2 else 0
    truncation = Truncation(
        0, 0, total, depth_degree=0, max_orders=(caps[0], cap_b, None)
    )
    index = _ModelIndex.build(truncation, components=("I", "Q"))
    i_basis, _, p_basis = _build_core(
        core_adapter(averaged), channels, reference, None, index=index, quadrature=None
    )
    rows0 = {(r, s): k for k, (l, kk, r, s, b) in enumerate(index.h0)}
    rows2 = {(r, s): k for k, (l, kk, r, s, b) in enumerate(index.h2)}
    t_g = log_from_affine(caps[0], h_g) if chart_g == "log" else np.eye(caps[0] + 1)
    t_b = log_from_affine(cap_b, h_b) if chart_b == "log" else np.eye(cap_b + 1)
    out_i, out_p = [], []
    for e in exponents:
        n, m = e[0], (e[1] if len(names) == 2 else 0)
        acc_i, acc_p = 0.0, 0.0
        for r in range(n + 1):
            for s in range(m + 1):
                coef = t_g[n, r] * t_b[m, s]
                if coef == 0.0:
                    continue
                acc_i = acc_i + coef * i_basis[:, rows0[(r, s)]]
                acc_p = acc_p + coef * p_basis[:, rows2[(r, s)]]
        out_i.append(acc_i)
        out_p.append(acc_p)
    return jnp.stack(out_i), jnp.stack(out_p)


def core_cell(
    averaged: AveragedKernel, channels: Any, lattice: Any, cell: int
) -> tuple[Any, Any]:
    """``(I, P)`` columns of one product cell of a lattice, rows in the lattice index order.

    The lattice may list ``B`` before ``gamma``. Lattices with ``Legendre`` axes go to the jet (decision 131).
    """
    table, index = lattice.cells, lattice.index
    if index.legendre:
        raise ValueError(
            "the core provider takes no Legendre axes. Use the jet provider"
        )
    order = sorted(
        range(len(table.variables)),
        key=lambda i: (
            CORE_VARIABLES.index(table.variables[i])
            if table.variables[i] in CORE_VARIABLES
            else len(CORE_VARIABLES)
        ),
    )
    names = [table.variables[i] for i in order]
    axes = {
        v: (
            table.chart[cell][i],
            float(table.reference[cell, i]),
            float(table.half_width[cell, i]),
        )
        for v, i in zip(names, order)
    }
    pattern = list(index.patterns[int(index.cell_pattern[cell])])
    exponents = [tuple(e[i] for i in order) for e in pattern]
    return core_columns(averaged, channels, axes, exponents)
