"""Default lattice geometry without an engine build (SPEC Sec. 4.6).

LABEL [extension].

Default cells use each variable's canonical units. Their support has the
shape of one interval per resolved axis. Reference-law rules have no
certified source error.
"""

from __future__ import annotations

import math

import numpy as np

from ..base.capabilities import (
    Beaming,
    EnergyWindow,
    FieldScalingRelation,
    ReferenceDensity,
)
from ..base.channels import Channels
from ..base.continuum import ContinuumKernel
from ..base.depth import DEPTH_TOL, cell_count
from ..distributions import (
    GammaScreen,
    GaussianScreen,
    LaplaceScreen,
    LogNormal,
    Normal,
    Uniform,
)
from ..engine.columns import column_map
from ..lattice.cells import Cells
from ..lattice.lattice import Lattice
from ..lattice.representations import AzimuthFourier, AzimuthHarmonic, Legendre
from ..population.declaration import Population, parents
from ..values.errors import DepthExpansionError, RoleError, SizeError
from .roles import compile_roles


def _cap(kernel, type_):
    return next((item for item in kernel.capabilities if isinstance(item, type_)), None)


def support(law):
    if hasattr(law, "support"):
        return tuple(float(x) for x in getattr(law, "quadrature_support", law.support))
    if not parents(law) and not callable(law):
        value = float(law)
        return value, value
    return -np.inf, np.inf


def support_for(pop, name):
    if name in pop.factors:
        return support(pop.factors[name])
    return next(
        tuple(float(x) for x in block.support[name])
        for block in pop.blocks
        if name in block.variables
    )


def beaming_knots(law):
    """Concrete support edges and interior density breaks for a cosine law."""
    if law is None or parents(law) or not hasattr(law, "support"):
        return ()
    return tuple(
        sorted(
            {
                float(x)
                for x in (*law.support, *getattr(law, "kinks", ()))
                if np.isfinite(x) and -1 < x < 1
            }
        )
    )


def _finite_support(source, variable):
    law = source.declaration
    bounds = source.support_for(variable)
    if bounds is None or np.all(np.isfinite(bounds)):
        return bounds
    tail = 1e-3
    if isinstance(law, (Normal, LogNormal, GaussianScreen)):
        from scipy.special import ndtri

        z = ndtri([tail, 1 - tail])
        if isinstance(law, LogNormal):
            return float(law.median) * np.exp(float(law.width) * z)
        return float(law.mean) + float(law.sigma) * z
    if isinstance(law, LaplaceScreen):
        delta = -float(law.sigma) * np.log(2 * tail) / np.sqrt(2)
        return float(law.mean) - delta, float(law.mean) + delta
    if isinstance(law, GammaScreen):
        from scipy.special import gammaincinv

        shape = float(law.shape)
        return float(law.mean) + float(law.sigma) / np.sqrt(shape) * (
            gammaincinv(shape, [tail, 1 - tail]) - shape
        )
    quantile = getattr(law, "quantile", None)
    return bounds if quantile is None else (quantile(tail), quantile(1 - tail))


def primary_lattice(pop, kernel, channels):
    """The primary-variable log lattice of SPEC Sec. 4.6."""
    name = kernel.primary
    if name is None:
        raise RoleError("this kernel has no declared primary variable. Pass lattice=.")
    lo, hi = (
        support_for(pop, name)
        if name in pop.factors or any(name in block.variables for block in pop.blocks)
        else (0.0, np.inf)
    )
    window = _cap(kernel, EnergyWindow)
    if name not in pop.factors and not any(name in b.variables for b in pop.blocks):
        if type(kernel) is not ContinuumKernel or window is None:
            raise RoleError(f"{name} needs support= or an explicit lattice=")
    scaling = _cap(kernel, FieldScalingRelation)
    if window is not None and scaling is not None:
        from ..base.window import continuum_mixture_window, energy_range

        nlo, nhi = (
            np.asarray(channels.support).min(),
            np.asarray(channels.support).max(),
        )
        field = pop.factors.get(scaling.variable)
        blo, bhi = support_for(pop, scaling.variable)
        eta = pop.factors.get("eta")
        if eta is not None and not parents(eta):
            inclination = (
                eta.rule() if hasattr(eta, "rule") else (np.asarray([eta]), np.ones(1))
            )
            laws = {"eta": inclination}
        else:
            laws = {}
        if 0 < blo <= bhi < np.inf:
            interval, field_range = window.window(laws, channels), (blo, bhi)
        elif (
            type(kernel) is ContinuumKernel
            and blo == 0 < bhi < np.inf
            and hasattr(field, "rule")
            and not parents(field)
            and eta is not None
            and not parents(eta)
        ):
            interval, ref = continuum_mixture_window(laws, field.rule())
            field_range = (ref, ref)
        elif not 0 < lo < hi < np.inf:
            raise RoleError(
                "the energy window needs bounded field quadrature or an explicit lattice="
            )
        else:
            interval = None
        if interval is not None:
            left, right = energy_range(interval, (nlo, nhi), field_range)
            lo, hi = max(lo, left), min(hi, right)
    if type(kernel) is ContinuumKernel:
        lo = max(lo, kernel.gamma_min)
    if not 0 < lo < hi < np.inf:
        raise RoleError(f"give bounded positive support for {name} or pass lattice=")
    n = max(1, math.ceil(math.log(hi / lo) / 0.484))
    variable = next(v for v in kernel.variables if v.name == name)
    return Lattice(
        _variables=tuple(kernel.variables),
        _depth_variable=kernel.depth,
        **{name: Cells.log(lo, hi, n, unit=variable.unit or None)},
    )


def _default_axis(layout, variable, primary, depth_axis=None):
    name = variable.name
    kind = layout.representations[name]
    if kind == "Legendre":
        return Legendre(8)
    if kind == "AzimuthHarmonic":
        return AzimuthHarmonic()
    if kind == "AzimuthFourier":
        return AzimuthFourier(4)
    if kind != "Cells":
        raise RoleError(
            f"default construction of {kind} for {name} needs an explicit representation"
        )
    sources = tuple(component.source_for(name) for component in layout.components)
    first = sources[0].declaration
    if hasattr(first, "cells_for"):
        hint = first.cells_for(variable)
        if hint is not None:
            return hint
    supports = tuple(_finite_support(source, variable) for source in sources)
    if any(bounds is None for bounds in supports):
        raise RoleError(f"{name} needs support= or an explicit lattice=")
    lo = min(float(bounds[0]) for bounds in supports)
    hi = max(float(bounds[1]) for bounds in supports)
    unit = variable.unit or None
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise RoleError(f"{name} needs a finite support hint or an explicit lattice=")
    if lo == hi:
        step = max(abs(lo), 1.0) * 0.1
        return Cells.single(lo, step, unit=unit)
    if variable.kind == "positive" and lo > 0 and name == primary:
        return Cells.log(
            lo, hi, max(1, math.ceil(math.log(hi / lo) / 0.484)), unit=unit
        )
    count = 1
    if depth_axis is not None:
        phase, channels, depth_tol, max_columns = depth_axis
        count = cell_count(lo, hi, 2, phase, channels, depth_tol)
        if count > max_columns:
            raise DepthExpansionError(
                f"depth requires {count:g} cells for depth_tol={depth_tol:g}. "
                "Increase max_columns or supply higher-order depth cells."
            )
    return Cells.linear(lo, hi, count, unit=unit)


def _split_beaming(kernel, axes, roles):
    for capability in kernel.capabilities:
        if not isinstance(capability, Beaming):
            continue
        pair = capability.variable, capability.partner
        if sum(name in axes for name in pair) != 1:
            continue
        target, source = pair if pair[0] in axes else pair[::-1]
        axis = axes[target]
        if not isinstance(axis, Cells) or axis.edges is None:
            continue
        knots = {
            x
            for component in roles.components
            for x in beaming_knots(component.source_for(source).declaration)
            if axis.edges[0] < x < axis.edges[-1]
        }
        if not knots - set(axis.edges):
            continue
        edges = np.unique(np.r_[axis.edges, list(knots)])
        parts = np.searchsorted(axis.edges[1:], (edges[1:] + edges[:-1]) / 2)
        unit = next(v.unit for v in kernel.variables if v.name == target)
        axes[target] = Cells.from_edges(
            edges,
            chart=axis.chart,
            order=tuple(axis.order[i] for i in parts),
            unit=unit or None,
        )


def planned_lattice(
    kernel, channels, roles, *, phase=None, depth_tol=DEPTH_TOL, max_columns=20_000
):
    """Choose default geometry from support and role declarations."""
    names = {v.name: v for v in kernel.variables}
    axes = {}
    if kernel.primary in roles.resolved:
        axis = names[kernel.primary]
        source = roles.components[0].source_for(axis.name)
        marker = source.kind in ("FreeMoments", "NonNegativeCells")
        if marker and source.declaration.cells_for(axis) is not None:
            axes[axis.name] = source.declaration.cells_for(axis)
        elif not (
            marker and source.support_for(axis) is not None
        ) and source.kind not in ("Samples", "Moments"):
            bounds = []
            scaling = _cap(kernel, FieldScalingRelation)
            for component in roles.components:
                factors = {
                    src.variables[0]: src.declaration
                    for src in component.sources
                    if len(src.variables) == 1 and src.kind in ("point", "distribution")
                }
                if scaling is not None and scaling.variable not in factors:
                    field = names[scaling.variable]
                    limit = component.source_for(field.name).support_for(field)
                    if limit is not None and np.all(np.isfinite(limit)):
                        lo, hi = map(float, limit)
                        if lo < hi:
                            factors[field.name] = Uniform(
                                lo, hi, unit=field.unit or None
                            )
                        elif lo == hi:
                            factors[field.name] = lo
                if axis.name not in factors and not marker:
                    break
                try:
                    cells = primary_lattice(
                        Population(**factors), kernel, channels
                    ).axes[axis.name]
                except (RoleError, StopIteration):
                    break
                bounds.append((float(cells.edges[0]), float(cells.edges[-1])))
            if len(bounds) == len(roles.components):
                lo = min(x[0] for x in bounds)
                hi = max(x[1] for x in bounds)
                axes[axis.name] = Cells.log(
                    lo,
                    hi,
                    max(1, math.ceil(math.log(hi / lo) / 0.484)),
                    unit=axis.unit or None,
                )
    for name in roles.resolved:
        if name not in axes:
            depth = (
                (phase, channels, depth_tol, max_columns)
                if name == kernel.depth and "P" in kernel.stokes
                else None
            )
            axes[name] = _default_axis(roles, names[name], kernel.primary, depth)
    _split_beaming(kernel, axes, roles)
    return Lattice(
        _variables=tuple(kernel.variables),
        _depth_variable=kernel.depth,
        **{name: axes[name] for name in roles.resolved},
    )


def default_lattice(
    population,
    channels,
    *,
    kernel=None,
    phase=None,
    resolve=(),
    average=(),
    max_columns=20_000,
    depth_tol=DEPTH_TOL,
):
    """Build default cell and representation geometry without spectral columns."""
    model = ContinuumKernel() if kernel is None else kernel
    ch = (
        channels
        if isinstance(channels, Channels)
        else Channels.at(channels[0], unit=channels[1])
    )
    roles = compile_roles(population, model, resolve=resolve, average=average)
    if roles.direct:
        raise RoleError("this source has a direct prediction without a default lattice")
    lattice = planned_lattice(
        model, ch, roles, phase=phase, depth_tol=depth_tol, max_columns=max_columns
    )
    references = tuple(
        (cap.variable, cap.density)
        for cap in model.capabilities
        if isinstance(cap, ReferenceDensity)
    )
    result = lattice._for_kernel(model.variables, model.depth, references)
    try:
        column_map(result, model, max_columns=max_columns)
    except SizeError as exc:
        if model.depth in result.cells.variables:
            raise DepthExpansionError(
                f"{exc} Use higher-order depth cells or increase max_columns."
            ) from exc
        raise
    return result
