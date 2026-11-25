"""Small-x reachability on the physical support of direct comparisons."""

import numpy as np

from ..base.continuum import ContinuumKernel
from ..base.hybrid import HybridKernel
from ..engine.direct import continuum_children, continuum_tail_threshold
from ..planning.defaults import support_for
from ..population.quadrature import cell_boxes, complement
from ..records.ledger import ErrorTerm
from ._sample_beaming import _SampledCosine


def _intersection(support, box):
    clipped = support.copy()
    for axis, (lo, hi) in box.items():
        a, b = clipped[axis]
        interval = max(lo, a), min(hi, b)
        if interval[0] > interval[1]:
            return None
        clipped[axis] = interval
    return clipped


def _contained(support, box):
    return all(
        lo <= support[k][0] and support[k][1] <= hi for k, (lo, hi) in box.items()
    )


def _regions(kernel, table, support, within, cells):
    """Yield continuum children with the occupied bounds of their assigned region."""
    if type(kernel) is ContinuumKernel and within == "support":
        yield kernel, support
        return
    boxes = cell_boxes(table)
    if within != "outside":
        for i in range(len(boxes)) if cells is None else cells:
            box = boxes[i]
            child = (
                kernel.kernels[kernel.assignment[i]]
                if isinstance(kernel, HybridKernel)
                else kernel
            )
            if type(child) is not ContinuumKernel:
                continue
            clipped = _intersection(support, box)
            if clipped is None:
                continue
            if isinstance(kernel, HybridKernel) and any(
                _contained(clipped, prev) for prev in boxes[:i]
            ):
                # A measure on a shared face belongs to its first containing cell.
                continue
            yield child, clipped
    if within == "cells" or (
        isinstance(kernel, HybridKernel) and kernel.switch is None
    ):
        return
    for box in complement(table):
        clipped = _intersection(support, box)
        if clipped is None or any(_contained(clipped, cell) for cell in boxes):
            continue
        if type(kernel) is ContinuumKernel:
            yield kernel, clipped
            continue
        axis, edge = kernel.switch
        lo, hi = clipped[axis]
        for child, bounds, present in zip(
            kernel.kernels,
            ((lo, min(hi, edge)), (max(lo, edge), hi)),
            (lo <= edge, hi > edge),
            strict=True,
        ):
            if present and type(child) is ContinuumKernel:
                yield child, {**clipped, axis: bounds}


def _sampled_abs_cosine(pop, name):
    """The smallest absolute cosine in an empirical angular margin."""
    law = pop.factors.get(name)
    if isinstance(law, _SampledCosine):
        nodes, weights = law.rule()
    else:
        block = next((b for b in pop.blocks if name in b.variables), None)
        if block is None:
            return 0.0
        nodes, weights = block.nodes[name], block.weights
    return float(np.min(np.abs(np.asarray(nodes)[np.asarray(weights) > 0])))


def _reaches_tail(child, channels, support, tied, cosine_floor):
    # The maximum sine occurs at the cosine closest to zero. At either
    # pole the transverse field vanishes even for an unbounded field law.
    eta_lo, eta_hi = support["eta"]
    if tied:
        mu_lo, mu_hi = support["mu"]
        eta_lo, eta_hi = max(eta_lo, mu_lo), min(eta_hi, mu_hi)
        if eta_lo > eta_hi:
            return False
    near = min(1.0, max(eta_lo, -eta_hi, cosine_floor))
    if near == 1:
        return False
    bound = support["B"][1] * np.sqrt((1 - near) * (1 + near))
    if np.isposinf(bound):
        return True
    cutoff = continuum_tail_threshold(child, channels, bound)
    return cutoff is not None and support["gamma"][1] > cutoff


class DirectTail:
    """Accumulate the continuation error across the direct integrals that ran."""

    def __init__(self):
        self.terms = ()

    def record(self, setup, table, channels, within, cells=None):
        kernel = setup.averaged.kernel
        children = continuum_children(kernel)
        if self.terms or not children:
            return
        axes = {"gamma", "B", "eta", *table.variables}
        tied = any(
            {source, target} == {"eta", "mu"} for source, target in setup.averaged.ties
        )
        cosines = ("eta", "mu") if tied else ("eta",)
        axes.update(cosines)
        cosine_floor = max(
            _sampled_abs_cosine(setup.population, name) for name in cosines
        )
        support = {name: support_for(setup.population, name) for name in axes}
        if not any(
            _reaches_tail(child, channels, support, tied, cosine_floor)
            for child in children
        ):
            return
        if any(
            _reaches_tail(child, channels, bounds, tied, cosine_floor)
            for child, bounds in _regions(kernel, table, support, within, cells)
        ):
            self.terms = (
                ErrorTerm.unbounded(
                    "the integrated support bounds can reach the small-x continuation",
                    name="small_x:tail",
                ),
            )
