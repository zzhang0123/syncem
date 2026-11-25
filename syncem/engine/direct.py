"""Direct integration on population quadrature panels (SPEC Sec. 7.2 and 8.3).

Panels are prepared in canonical units and split at cell faces before they
reach this module. A panel is a mapping of one-dimensional coordinate
arrays and a vector of column weights. The returned modes share the
channel response and averaged statistics of the evaluator.

LABEL ``eq: channel kernel``. Integrated modes have shape ``(n_ch, *L)``
for I and V and ``(n_phase, n_ch, *L)`` for P. The units are those of the
kernel times column density. The panel quadrature error is not certified.
"""

from __future__ import annotations

import dataclasses
import operator
from collections.abc import Iterable, Mapping
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..base.continuum import ContinuumKernel
from ..base.hybrid import HybridKernel
from ..base.kernel import Modes
from ..base.window import A1_HZ_PER_G
from .average import AveragedKernel, NodeRule

__all__ = [
    "TAIL_X",
    "continuum_tail_threshold",
    "continuum_tail_thresholds",
    "integrate",
    "node_modes",
    "sum_modes",
]

Panel = tuple[Mapping[str, Any], Any]


# Largest x at which the direct path joins the small-x asymptote of F and G
# (SPEC Sec. 8.3, decision 198). Its relative error there is about
# 0.844 x^(2/3) = 8.4e-7 in F.
TAIL_X = 1e-9
# Leave room for the rounding of B_perp and a_B gamma**2 in the kernel.
_MARGIN = 1 - 8 * np.finfo(float).eps


def continuum_children(kernel):
    """Continuum children whose direct integrals use the local continuation."""
    children = kernel.kernels if isinstance(kernel, HybridKernel) else (kernel,)
    return tuple(child for child in children if type(child) is ContinuumKernel)


def _tail_x(kernel: ContinuumKernel) -> float:
    return min(float(kernel.model.x_min), TAIL_X)


def _channel_lows(channels: Any) -> np.ndarray:
    return np.min(np.asarray(channels.nodes), axis=-1)


def continuum_tail_thresholds(
    kernel: Any, channels: Any, b_perp_max: float | None
) -> np.ndarray | None:
    """Give the earliest node transition in each channel for a bound on B_perp in G."""
    if type(kernel) is not ContinuumKernel or b_perp_max is None:
        return None
    b = float(b_perp_max)
    if np.isnan(b) or b < 0:
        raise ValueError("b_perp_max must be a nonnegative field in G")
    if b == 0 or np.isposinf(b):
        return None
    scale = A1_HZ_PER_G * b * _tail_x(kernel)
    return np.sqrt(_channel_lows(channels) / scale) * _MARGIN


def continuum_tail_threshold(
    kernel: Any, channels: Any, b_perp_max: float | None
) -> float | None:
    """Give the lowest asymptote transition over channels for a bound on B_perp in G."""
    thresholds = continuum_tail_thresholds(kernel, channels, b_perp_max)
    return None if thresholds is None else float(np.min(thresholds))


class _ContinuumTail(eqx.Module):
    """Continue each frequency node at its own small-x threshold before band integration."""

    continuum: ContinuumKernel
    x_tail: float = eqx.field(static=True)

    variables = ContinuumKernel.variables
    azimuth = ContinuumKernel.azimuth
    depth = ContinuumKernel.depth
    traceable = True

    def _threshold(self, nu, point):
        # Use the same transverse field and critical frequency scale as the kernel.
        positive, _, scale = self.continuum.model._geometry(
            1.0, point["B"], point.get("eta", 0.0)
        )
        threshold = (
            jnp.sqrt(jnp.asarray(nu, dtype=float) / scale / self.x_tail) * _MARGIN
        )
        return jnp.where(positive, threshold, jnp.inf)

    def channel(self, channels, point, *, phase, per_node=False):
        gamma = point["gamma"]
        probe = jnp.minimum(gamma, self._threshold(jnp.asarray(channels.nodes), point))
        weighted = self.continuum.channel(
            channels, {**point, "gamma": probe}, phase=phase, per_node=True
        )
        scale = (probe / gamma) ** (2.0 / 3.0)
        nodes = Modes(*(value * scale for value in weighted))
        if per_node:
            return nodes
        return Modes(*(jnp.sum(value, axis=-1) for value in nodes))


def _continued(
    averaged: AveragedKernel, channels: Any, b_perp_max: float | None
) -> AveragedKernel:
    if not continuum_children(averaged.kernel) or b_perp_max is None:
        return averaged
    b = float(b_perp_max)
    if np.isnan(b) or b < 0:
        raise ValueError("b_perp_max must be a nonnegative field in G")
    if b == 0:
        return averaged
    kernel = averaged.kernel

    def adapt(child):
        if type(child) is not ContinuumKernel:
            return child
        x_tail = _tail_x(child)
        return _ContinuumTail(dataclasses.replace(child, x_min=x_tail), x_tail)

    evaluator = (
        kernel._with_evaluators(adapt(child) for child in kernel.kernels)
        if isinstance(kernel, HybridKernel)
        else adapt(kernel)
    )
    return eqx.tree_at(lambda a: a.kernel, averaged, evaluator)


def _positive_batch(batch: int) -> int:
    if isinstance(batch, bool):
        raise ValueError("batch must be a positive integer")
    batch = operator.index(batch)
    if batch < 1:
        raise ValueError("batch must be a positive integer")
    return batch


def _runner(averaged: AveragedKernel, channels: Any, batch: int):
    if not averaged.kernel.traceable:
        return None

    # A batched switch evaluates every child at all points. Hybrid children
    # may guard distinct validity domains, so keep branch selection scalar.
    mapped_batch = None if isinstance(averaged.kernel, HybridKernel) else batch

    @eqx.filter_jit
    def run(coordinates):
        return jax.lax.map(
            lambda point: averaged(channels, point),
            coordinates,
            batch_size=mapped_batch,
        )

    return run


def _columns(points: Mapping[str, Any], size: int) -> dict[str, np.ndarray]:
    values = {name: np.asarray(a, dtype=float) for name, a in points.items()}
    for name, column in values.items():
        if column.shape != (size,) or not np.all(np.isfinite(column)):
            raise ValueError(f"direct nodes for {name} need {size} finite values")
    return values


def _zeros(averaged: AveragedKernel, channels: Any, leading=()) -> Modes:
    shape = (int(channels.n_ch), *(n + 1 for _, n, _ in averaged.legendre))
    return Modes(
        np.zeros((*leading, *shape)),
        np.zeros((*leading, *shape)),
        np.zeros((*leading, 1, *shape), dtype=complex),
    )


def _azimuth_independent(averaged: AveragedKernel, azimuth: str) -> bool:
    kernel = averaged.kernel
    if isinstance(kernel, HybridKernel) and (
        (kernel.switch is not None and kernel.switch[0] == azimuth)
        or (
            kernel.geometry
            and any(axis == azimuth for axis, _, _ in kernel.geometry[0])
        )
    ):
        return False
    return not averaged.leaf_factors and all(
        isinstance(rule, NodeRule)
        and rule.parents is not None
        and azimuth not in rule.parents
        for _, rule in averaged.averaged
    )


def _evaluate(averaged, channels, run, values, size, azimuth_factor, apply_azimuth):
    name = averaged.kernel.azimuth
    azimuth = values.get(name, dict(averaged.fixed).get(name))
    if name in values and _azimuth_independent(averaged, name):
        values = {key: value for key, value in values.items() if key != name}
    if size == 0:
        return _zeros(averaged, channels, leading=(0,))
    if values:
        rows = np.ascontiguousarray(np.column_stack(tuple(values.values())))
        # Compare bitwise so signed zero remains distinct for a user kernel.
        _, unique, inverse = np.unique(
            rows.view(np.uint64), axis=0, return_index=True, return_inverse=True
        )
    else:
        unique, inverse = np.array([0]), np.zeros(size, dtype=int)
    distinct = {k: v[unique] for k, v in values.items()}
    if not distinct:
        sample = averaged(channels, {})
        out = Modes(*(np.expand_dims(a, 0) for a in sample))
    elif run is None:
        result = [
            averaged(channels, dict(zip(distinct, x, strict=True)))
            for x in zip(*distinct.values(), strict=True)
        ]
        out = Modes(*(np.stack(a) for a in zip(*result, strict=True)))
    else:
        out = run({n: jnp.asarray(v) for n, v in distinct.items()})
    i, v, p = (np.asarray(a)[inverse] for a in out)
    if apply_azimuth:
        if azimuth_factor is not None and azimuth is not None:
            raise ValueError("provide the azimuth factor or the azimuth nodes")
        if azimuth is not None:
            factor = np.asarray(np.exp(2j * azimuth))
            p = p * factor.reshape(factor.shape + (1,) * (p.ndim - factor.ndim))
        elif azimuth_factor is not None:
            p = p * complex(azimuth_factor)
    return Modes(i, v, p)


def node_modes(
    averaged: AveragedKernel,
    channels: Any,
    points: Mapping[str, Any],
    *,
    size: int | None = None,
    b_perp_max: float | None = None,
    azimuth_factor: complex | None = None,
    apply_azimuth: bool = True,
    batch: int = 4,
) -> Modes:
    """Evaluate modes at the population nodes in their input order.

    The output has one leading axis for the population nodes. An empty
    coordinate mapping evaluates a constant kernel at ``size`` points.
    With ``apply_azimuth=False`` the caller receives the natural-basis
    polarization used for the cell table.
    """
    if size is None:
        size = len(next(iter(points.values()))) if points else 1
    continued = _continued(averaged, channels, b_perp_max)
    run = _runner(continued, channels, _positive_batch(batch))
    return _evaluate(
        continued,
        channels,
        run,
        _columns(points, size),
        size,
        azimuth_factor,
        apply_azimuth,
    )


def sum_modes(nodes: Modes, weights: Any, *, mask: Any = None) -> Modes:
    """Pair evaluated modes with column weights and an optional Boolean mask."""
    weights = np.asarray(weights, dtype=float)
    if weights.shape != (nodes.I.shape[0],) or not np.all(np.isfinite(weights)):
        raise ValueError("direct integration needs one finite weight per node")
    if mask is not None:
        mask = np.asarray(mask)
        if mask.shape != weights.shape or mask.dtype != np.bool_:
            raise ValueError("direct integration needs one Boolean mask entry per node")
        weights = weights[mask]
        nodes = Modes(*(a[mask] for a in nodes))
    return Modes(*(np.tensordot(weights, a, axes=1) for a in nodes))


def integrate(
    averaged: AveragedKernel,
    channels: Any,
    panels: Iterable[Panel],
    *,
    b_perp_max: float | None = None,
    azimuth_factor: complex | None = None,
    batch: int = 4,
) -> Modes:
    """Integrate the same averaged kernel on each already split panel.

    A declared field bound enables the continuum tail. Each frequency node
    joins the asymptote at its own small-x threshold for the local field.
    The asymptote changes the gamma used by the continuum kernel leaf.
    Conditional laws and phase weights still receive the actual point.
    """
    continued = _continued(averaged, channels, b_perp_max)
    run = _runner(continued, channels, _positive_batch(batch))
    total = None
    for points, weights in panels:
        weights = np.asarray(weights, dtype=float)
        if weights.ndim != 1 or not np.all(np.isfinite(weights)):
            raise ValueError("direct integration needs a finite weight vector")
        values = _columns(points, weights.size)
        keep = weights != 0.0
        if not np.any(keep):
            continue
        values = {k: a[keep] for k, a in values.items()}
        nodes = _evaluate(
            continued, channels, run, values, int(keep.sum()), azimuth_factor, True
        )
        part = sum_modes(nodes, weights[keep])
        total = part if total is None else Modes(*(a + b for a, b in zip(total, part)))
    return _zeros(continued, channels) if total is None else total
