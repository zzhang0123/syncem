"""Numerical estimates at the continuum kernel's small-x guard."""

import numpy as np

from ..records.ledger import ErrorTerm
from ..ultrarel import F, G
from .cache import default_cache
from .primitive import scaled_log_derivatives


def _probe(x, nodes, cutoff):
    values = []
    for kind, kernel in (("F", F), ("G", G)):
        exponents, _ = scaled_log_derivatives(x, which=kind, nodes=256, cutoff=90)
        reference = np.exp(np.log(x) - x + exponents)
        actual = float(kernel(x, n_nodes=nodes, tail_cutoff=cutoff))
        if not (np.isfinite(reference) and reference > 0 and np.isfinite(actual)):
            return None
        values.append(abs(actual - reference) / reference)
    return max(values)


def small_x_term(kernel):
    x = kernel.model.x_min
    args = x, kernel.n_nodes_F, kernel.tail_cutoff
    error = (
        default_cache().trial(("continuum small-x probe", *args), lambda: _probe(*args))
        if 1e-22 <= x <= 1
        else None
    )
    if error is None:
        return ErrorTerm.unbounded(
            "no convergent lower-edge quadrature comparison", name="small_x"
        )
    return ErrorTerm(
        error,
        "estimate",
        f"{kernel.n_nodes_F}-node kernel against 256-node quadrature at x={x:.4g}",
        name="small_x",
        relative_to="continuum F and G at the lower guard",
    )
