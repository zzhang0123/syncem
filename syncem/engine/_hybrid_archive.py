"""Verified assignment metadata for hybrid basis archives."""

from dataclasses import replace

import equinox as eqx
import numpy as np

from ..base.hybrid import HybridKernel
from ..values.errors import DataError
from ._basis_io_source import ArchivedBasis, hybrid_child_keys
from ._hybrid_basis import HybridBasis, HybridRouteRecord


class ArchivedHybridBasis(ArchivedBasis, HybridBasis):
    """Saved hybrid columns with a verified or unavailable child source."""

    hybrid_metadata: tuple = eqx.field(static=True, kw_only=True)

    def select(self, **cells):
        selection = super().select(**cells)
        if selection is self:
            return self
        return replace(
            selection, hybrid_metadata=tuple(_save_hybrid(selection).items())
        )


def _save_hybrid(basis):
    if not isinstance(basis, HybridBasis):
        return None
    kernel = basis.averaged.kernel
    if isinstance(basis, ArchivedHybridBasis) and not isinstance(kernel, HybridKernel):
        return dict(basis.hybrid_metadata)
    return {
        "assignment": kernel.assignment,
        "child_keys": hybrid_child_keys(kernel),
        "switch": kernel.switch,
        "switch_tol": kernel.switch_tol,
    }


def _hybrid_layout(body, lattice):
    hybrid = body.get("hybrid")
    record = body["record"]
    routes = tuple(isinstance(r, HybridRouteRecord) for r in record.routes)
    if hybrid is None:
        if any(routes):
            raise DataError("the hybrid basis is missing its frozen kernel assignment.")
        return
    if (
        set(hybrid) != {"assignment", "child_keys", "switch", "switch_tol"}
        or body["kernel"]["name"] != "hybrid"
        or body["legacy"] is not None
        or body["fourier"] is not None
    ):
        raise DataError("invalid hybrid archive subtype metadata.")
    assignment, keys = hybrid["assignment"], hybrid["child_keys"]
    count = lattice.cells.n_cells
    if (
        not isinstance(keys, tuple)
        or len(keys) < 2
        or any(
            key is not None
            and (
                not isinstance(key, str)
                or len(key) != 64
                or any(c not in "0123456789abcdef" for c in key)
            )
            for key in keys
        )
        or not isinstance(assignment, tuple)
        or len(assignment) != count
        or any(type(i) is not int or not 0 <= i < len(keys) for i in assignment)
    ):
        raise DataError("invalid hybrid assignment or child-source fingerprints.")
    if (
        len(routes) != count
        or not all(routes)
        or any(
            route.cell != i
            or route.provider != "jet"
            or route.mode != "hybrid"
            or type(route.kernel_index) is not int
            or route.kernel_index != child
            for i, (route, child) in enumerate(zip(record.routes, assignment))
        )
    ):
        raise DataError("stored hybrid routes disagree with the frozen assignment.")
    tol = hybrid["switch_tol"]
    if (
        type(tol) not in (int, float)
        or not np.isfinite(tol)
        or tol <= 0
        or dict(record.defaults).get("switch_tol") != tol
    ):
        raise DataError("invalid stored hybrid switch tolerance.")
    switch = hybrid["switch"]
    if switch is not None:
        if not isinstance(switch, tuple) or len(switch) != 2 or len(keys) != 2:
            raise DataError("invalid stored hybrid switch declaration.")
        axis, edge = switch
        if (
            axis not in lattice.cells.variables
            or type(edge) not in (int, float)
            or not np.isfinite(edge)
        ):
            raise DataError("invalid stored hybrid switch boundary.")
        index = lattice.cells.variables.index(axis)
        left, right = (
            lattice.cells.hi[:, index] <= edge,
            lattice.cells.lo[:, index] >= edge,
        )
        if not np.all(left | right) or tuple((~left).astype(int)) != assignment:
            raise DataError(
                "the hybrid switch boundary disagrees with the frozen assignment."
            )
