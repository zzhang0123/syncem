"""Phase expansion bounds used by the forward planner (SPEC Sec. 4.6).

LABEL [extension].

Depth faces are in rad/m^2 and channel frequencies are in Hz. The per-
channel shape is (n_ch, 1). The bound covers the specified affine phase law
and carries no certification of that law.
"""

from __future__ import annotations

import math

import numpy as np

from ..values.errors import DepthExpansionError
from .phase import FaradayRotation, PhaseLaw, phase_coordinate


DEPTH_TOL = 1e-3


def tolerance(value):
    result = float(value)
    if not np.isfinite(result) or result <= 0:
        raise ValueError("depth_tol must be positive and finite")
    return result


def phase_extent(phase, channels):
    """Maximum phase-coordinate magnitude per band before multiplying depth."""
    if phase is None:
        phase = FaradayRotation()
    if phase is False:
        return np.zeros(channels.n_ch), "no depth phase"
    if not isinstance(phase, PhaseLaw):
        raise TypeError("phase must be PhaseLaw, False or None")
    if phase.tau is phase_coordinate:
        return np.asarray(phase.tau(channels.support[:, 0])), "full frequency support"
    tau = np.broadcast_to(np.asarray(phase.tau(channels.nodes)), channels.nodes.shape)
    if np.iscomplexobj(tau) or np.any(~np.isfinite(tau)):
        raise DepthExpansionError(
            "depth phase needs finite real values on response nodes"
        )
    return np.max(np.abs(tau), axis=-1), "channel quadrature nodes"


def cell_count(lo, hi, order, phase, channels, depth_tol):
    """Smallest equal-cell count for an affine depth axis."""
    tau, _ = phase_extent(phase, channels)
    permitted = math.exp(
        (math.log(tolerance(depth_tol)) + math.lgamma(order + 2)) / (order + 1)
    )
    count = float(np.max(tau)) * (hi - lo) / (2 * permitted)
    return max(1, math.ceil(count)) if math.isfinite(count) else math.inf


def depth_bound(lattice, variable, phase, channels, depth_tol=DEPTH_TOL):
    """Maximum polynomial phase remainder on the admitted depth cells."""
    tol = tolerance(depth_tol)
    table = lattice.cells
    if phase is False or variable not in table.variables:
        return None
    i = table.variables.index(variable)
    if variable not in table.excluded and len(table.variables) > 1:
        raise DepthExpansionError("the depth bound needs depth outside the total order")
    if any(chart[i] != "affine" for chart in table.chart):
        raise DepthExpansionError("the phase bound needs affine depth cells")
    radius = np.maximum(
        table.reference[:, i] - table.lo[:, i], table.hi[:, i] - table.reference[:, i]
    )
    if np.any(~np.isfinite(radius)):
        raise DepthExpansionError(
            "give finite depth cell edges or average the depth law"
        )
    tau, reference = phase_extent(phase, channels)
    bound = np.zeros(channels.n_ch)
    orders = np.asarray(table.order)[:, i]
    for order in np.unique(orders):
        radial = radius[orders == order].max()
        power = int(order) + 1
        x_max = tau * radial
        with np.errstate(divide="ignore", over="ignore", under="ignore"):
            error = np.exp(power * np.log(x_max) - math.lgamma(power + 1))
        bound = np.maximum(bound, error)
    peak = bound.max()
    if peak > tol:
        raise DepthExpansionError(
            f"depth Taylor remainder {peak:.4g} exceeds depth_tol={tol:g}. "
            "Use narrower affine depth cells, higher depth order or average the screen."
        )
    return bound[:, None], reference
