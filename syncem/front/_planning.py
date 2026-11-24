"""Access the structural role compiler without running an engine stage."""

from __future__ import annotations

from ..base.continuum import ContinuumKernel
from ..base.depth import DEPTH_TOL, depth_bound
from ..base.hybrid import HybridKernel
from ..engine._basis_update import stokes_blocks
from ..engine.columns import ColumnRules, column_map
from ..fitting._data_join import _join_channels
from ..lattice.representations import AzimuthFourier
from ..planning.defaults import planned_lattice
from ..planning.plan import Plan, PlanRecord
from ..planning.roles import compile_roles
from ..values.errors import DepthExpansionError, RoleError, SizeError
from ._beaming_layout import merge_beaming, resolved_beaming
from ._predictions import channels_from
from ._shared import shared_azimuth
from ._setup import bind_lattice


def plan(
    population,
    channels=None,
    *,
    lattice=None,
    kernel=None,
    phase=None,
    stokes=None,
    predict=None,
    resolve=(),
    average=(),
    max_columns=20_000,
    strict=False,
    depth_tol=DEPTH_TOL,
):
    """Compile population roles and column sizes before computing spectra."""
    model = ContinuumKernel() if kernel is None else kernel
    layout = compile_roles(
        population, model, lattice=lattice, resolve=resolve, average=average
    )
    if strict and layout.components[0].defaults:
        raise RoleError(
            f"strict planning requires explicit defaults: {layout.components[0].defaults}"
        )
    if lattice is None and not layout.direct:
        if channels is None:
            raise RoleError("give channels or an explicit lattice for the plan")
        ch = channels_from(channels)
        if predict is not None:
            ch = _join_channels((ch, channels_from(predict)))
        lattice = planned_lattice(
            model, ch, layout, phase=phase, depth_tol=depth_tol, max_columns=max_columns
        )
    else:
        ch = None if channels is None else channels_from(channels)
        if predict is not None and ch is not None:
            ch = _join_channels((ch, channels_from(predict)))
    lat = None if lattice is None else bind_lattice(lattice, model)
    if lat is not None and isinstance(model, HybridKernel):
        model = model.bind(lat.cells)
    blocks = stokes_blocks(model, ColumnRules(), stokes)
    bound = None
    if lat is not None and ch is not None and "P" in blocks:
        bound = depth_bound(lat, model.depth, phase, ch, depth_tol)
    if lat is not None:
        for component in layout.components:
            resolved_beaming(model, component.population, lat)
    try:
        columns = (
            None
            if lat is None
            else column_map(
                lat,
                model,
                ColumnRules(stokes=blocks, merge=merge_beaming(model, lat)),
                max_columns=max_columns,
            )
        )
    except SizeError as exc:
        if bound is not None:
            raise DepthExpansionError(
                f"{exc} Use higher-order depth cells or increase max_columns."
            ) from exc
        raise
    kinds = {
        source.kind for component in layout.components for source in component.sources
    }
    kind = (
        "non-negative linear"
        if "NonNegativeCells" in kinds
        else "bilinear"
        if len(layout.components) == 1
        and (
            shared_azimuth(model, layout.components[0].population)
            or sum(
                source.kind == "FreeMoments" for source in layout.components[0].sources
            )
            > 1
        )
        else "basis-parametric"
        if any(
            role.role == "basis parameter"
            for group in layout.components
            for role in group.roles.values()
        )
        else "linear"
        if "FreeMoments" in kinds
        else "fixed"
    )
    angular = None if lat is None else lat.representations.get(model.azimuth)
    return Plan(
        PlanRecord(
            layout,
            model.name,
            None if lat is None else lat.key,
            None if lat is None else lat.index,
            columns,
            kind,
            fourier_order=angular.m_max
            if isinstance(angular, AzimuthFourier)
            else None,
            depth_truncation=None if bound is None else float(bound[0].max()),
            kernel_assignment=model.assignment
            if isinstance(model, HybridKernel)
            else (),
        ),
        lat,
    )
