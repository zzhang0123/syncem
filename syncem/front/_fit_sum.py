"""Compile a population sum into one linear fit on the observation rows."""

from dataclasses import replace

import equinox as eqx
import numpy as np

from ..base.continuum import ContinuumKernel
from ..base.depth import DEPTH_TOL
from ..engine.basis_sum import BasisSum
from ..fitting.family_fit import free_parameters
from ..fitting.problem import Problem
from ..fitting.sum_fit import SumFitInput, solve_sum, sum_fit_workspace
from ..population import FreeMoments
from ..values.errors import RoleError
from ._declared import check_declared, effective_phase, same_kernel
from ._inverse import _moments_basis
from ._planning import plan


def _held_source(basis, kernel, phase):
    if basis is None:
        return ContinuumKernel() if kernel is None else kernel, phase
    check_declared(basis, kernel=kernel, phase=phase)
    parts = basis.parts if isinstance(basis, BasisSum) else (basis,)
    first = parts[0]
    model, original = first.averaged.kernel, first.forward_phase
    if any(not same_kernel(model, part.averaged.kernel) for part in parts[1:]):
        raise RoleError("the population sum needs one kernel matching the stored bases")

    stored_phase = effective_phase(original, model)
    for part in parts[1:]:
        phase_of_part = effective_phase(part.forward_phase, part.averaged.kernel)
        if not eqx.tree_equal(stored_phase, phase_of_part, typematch=True):
            raise RoleError("the stored bases of a population sum need the same phase")
    return model, original


def _check_members(parts, layout, lattice, *, strict):
    resolved = set(layout.resolved)
    common_selection = None
    for i, (part, component) in enumerate(
        zip(parts, layout.components, strict=True), start=1
    ):
        if strict and component.defaults:
            raise RoleError(
                f"sum component {i} needs explicit defaults: {component.defaults}"
            )
        unknown = [s for s in component.sources if s.kind == "FreeMoments"]
        if (
            len(unknown) != 1
            or set(unknown[0].variables) != resolved
            or free_parameters(part)
        ):
            raise RoleError(
                f"sum component {i} needs one joint FreeMoments block over "
                "the common resolved axes and concrete averaged factors"
            )
        selection = unknown[0].declaration.selections()
        if common_selection is None:
            common_selection = selection
        elif selection != common_selection and not np.array_equal(
            lattice.index.cells_where(**selection),
            lattice.index.cells_where(**common_selection),
        ):
            raise RoleError(
                "a sum of free moments needs the same cell selection for each component"
            )
        if unknown[0].declaration.prior is not None:
            raise RoleError(
                "use a prior on the concatenated moments of se.problem for a sum"
            )
        if float(part.column_factor) == 0:
            raise RoleError(f"remove zero-weight sum component {i} before fitting")
    for group in layout.basis_groups:
        symmetries = {frozenset(parts[i].symmetries) for i in group}
        if len(symmetries) != 1:
            raise RoleError(
                "components with tied moments need the same declared symmetries"
            )
    return lattice.select(**common_selection) if common_selection else lattice


def _stored_parts(basis, groups):
    if basis is None:
        return (None,) * len(groups)
    if isinstance(basis, BasisSum):
        if basis.member_indices is not None and basis.member_indices != groups:
            raise RoleError("the supplied BasisSum has different component groups")
        available = basis.parts
    else:
        available = (basis,)
    if len(available) != len(groups):
        raise RoleError("give one stored basis per distinct set of averaged statistics")
    return available


def _group_plan(plan_record):
    layout = plan_record.layout
    notes = tuple(
        "Components "
        + ", ".join(str(i + 1) for i in group)
        + " share basis statistics. Only their summed moments are fitted."
        for group in layout.basis_groups
        if len(group) > 1
    )
    return replace(
        plan_record, layout=replace(layout, caveats=(*layout.caveats, *notes))
    )


def _sum_basis(
    population,
    data,
    *,
    basis=None,
    lattice=None,
    kernel=None,
    phase=None,
    predict=None,
    check=True,
    resolve=(),
    average=(),
    allow_mismatch=False,
    max_columns=20_000,
    strict=False,
    cache=True,
    depth_tol=DEPTH_TOL,
    chunk_bytes=2**31,
):
    """Compile one fitted moment vector for each set of specified statistics."""
    if average:
        raise RoleError("fitting sums with average= needs a shared inverse moment map")
    first = basis.parts[0] if isinstance(basis, BasisSum) else basis
    model, phase = _held_source(basis, kernel, phase)
    if first is not None:
        if lattice is not None and lattice.key != first.lattice.key:
            raise RoleError("lattice= and basis= describe different coordinates")
        lattice = first.lattice
    summarized = plan(
        population,
        data.channels,
        lattice=lattice,
        kernel=model,
        phase=phase,
        predict=predict,
        resolve=resolve,
        strict=strict,
        max_columns=max_columns,
        depth_tol=depth_tol,
    )
    parts, layout = population.parts, summarized.record.layout
    selected = _check_members(parts, layout, summarized.lattice, strict=strict)
    groups = layout.basis_groups
    stored = _stored_parts(basis, groups)
    width = selected.n * len(groups)
    sum_fit_workspace(data, width, chunk_bytes)
    compiled = tuple(
        _moments_basis(
            parts[group[0]],
            data,
            FreeMoments,
            basis=prebuilt,
            lattice=summarized.lattice,
            kernel=model,
            phase=phase,
            check=check,
            strict=strict,
            cache=cache,
            max_columns=max_columns,
            depth_tol=depth_tol,
            allow_mismatch=allow_mismatch,
        )
        for group, prebuilt in zip(groups, stored, strict=True)
    )
    record = _group_plan(summarized.record)
    names = (
        basis.components
        if isinstance(basis, BasisSum)
        else tuple(f"component {group[0] + 1}" for group in groups)
    )
    return BasisSum(
        tuple(replace(part, structural_plan=record) for part in compiled),
        names,
        member_indices=groups,
    )


def sum_problem(population, data, *, prior=None, **options):
    """Compile the explicit Gaussian likelihood of a population sum."""
    return Problem(_sum_basis(population, data, **options), data, prior=prior)


def fit_sum(population, data, *, rank, keep_directions, chunk_bytes, **options):
    """Fit a population sum without compiling a full batched likelihood first."""
    basis = _sum_basis(population, data, chunk_bytes=chunk_bytes, **options)
    return solve_sum(
        SumFitInput(basis, data), rank, chunk_bytes, keep_directions=keep_directions
    )
