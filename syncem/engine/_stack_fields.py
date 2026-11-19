"""Field cells of the stack by the field-scaling relation, and the row layout of a cell (design/0.8/SPEC.md Sec. 8.3).

LABEL ``eq: illustrated derivative spectra``. A ``(gamma, B)`` cell takes the energy series at the field
reference ``B_b`` and turns it into the ``(gamma, B)`` columns by ``expand.field_scaling_columns`` when the kernel
declares ``FieldScalingRelation``, the energy chart is the log chart and no law depends on ``gamma`` or ``B``.
Other field cells raise ``StackRefused``, and the caller takes the jet. ``_to_rows`` writes per-exponent arrays
in the lattice order (Legendre pair, then monomial). Units: those of the kernel. Shapes: per-exponent arrays
``(J, *L)``, cell rows ``(n_rows, J)``. Not certified: the energy series of the walk, validated against the jet.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import numpy as np

from ..base.capabilities import FieldScalingRelation
from ._stack_checks import StackRefused
from .average import AveragedKernel, NodeRule
from .expand import field_scaling_columns


def _to_rows(series: Any, pattern: list[tuple[int, ...]], picker: Any) -> Any:
    """Rows in lattice order (Legendre pair, then monomial) from per-exponent arrays ``(J, *L)``."""
    cols = np.stack([picker(e) for e in pattern])  # (n_m, J, *L)
    n_l = cols.ndim - 2
    if n_l == 0:
        return cols
    moved = np.moveaxis(cols, tuple(range(2, cols.ndim)), tuple(range(n_l)))
    return moved.reshape((-1, cols.shape[1]))


def _check_field_laws(averaged: AveragedKernel) -> None:
    """The field-scaling precondition on the averaged laws: declared parents, none of them ``gamma`` or ``B``."""
    for name, rule in averaged.averaged:
        if not isinstance(rule, NodeRule):
            raise StackRefused(
                f"the law of {name} is a bare function: the field-scaling precondition needs its parents"
            )
        if rule.parents is None:
            raise StackRefused(
                f"the law of {name} has undeclared parents: the field-scaling precondition needs them"
            )
        if {"gamma", "B"} & set(rule.parents):
            raise StackRefused(
                f"the law of {name} depends on gamma or B: the field-scaling precondition fails"
            )


def _field_cell(
    averaged: AveragedKernel,
    table: Any,
    cell: int,
    pattern: list,
    names: tuple,
    energy: tuple,
    series: Callable[[int, dict, tuple[str, ...]], tuple[Any, Any]],
) -> tuple[Any, Any]:
    """``(I, P)`` rows of a ``(gamma, B)`` cell. ``series(order, fixed, probe)`` gives the log-chart energy series."""
    chart, ref, h = energy
    if chart != "log":
        raise StackRefused(
            "field cells with an affine energy chart: the scaling relation needs the log chart"
        )
    if not any(
        isinstance(c, FieldScalingRelation) for c in averaged.kernel.capabilities
    ):
        raise StackRefused("the kernel declares no FieldScalingRelation")
    _check_field_laws(averaged)
    b = names.index("B")
    b_chart, b_ref, b_h = (
        table.chart[cell][b],
        float(table.reference[cell, b]),
        float(table.half_width[cell, b]),
    )
    need = max(sum(e) for e in pattern)
    si, sp = series(need, {"B": b_ref}, ("gamma", "B"))
    exps = [(e[names.index("gamma")], e[b]) for e in pattern]
    kw = dict(
        h_gamma=h,
        field_chart=b_chart,
        field_ref=b_ref,
        field_half_width=b_h,
        field_basis=b_ref,
    )
    ci = field_scaling_columns({n: si[n] for n in range(need + 1)}, exps, **kw)
    cp = field_scaling_columns({n: sp[n] for n in range(need + 1)}, exps, **kw)
    return (
        _to_rows(ci, exps, lambda e: np.asarray(ci[e])),
        _to_rows(cp, exps, lambda e: np.asarray(cp[e])),
    )
