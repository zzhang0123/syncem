"""Prepare a family fit and reuse the basis when its averaged statistics agree."""

from dataclasses import replace

from ..engine.cache import trial_tables
from ..fitting._family_batch import FamilyBatch
from ..fitting.family_fit import fit_family, free_parameters, start_values
from ..fitting.moment_form import fit_moment_form
from ..values.errors import RoleError
from ._accuracy import check_policy
from ._planning import plan
from ._predictions import _same_basis_stats, build_one


def separable_fit(
    component,
    data,
    *,
    basis,
    lattice,
    model,
    phase,
    predict,
    starts,
    rank,
    moment_form,
    chunk_bytes,
    check,
    resolve,
    strict,
    max_columns,
    cache,
    depth_tol,
    assumptions,
    allow_mismatch,
):
    defaults = free_parameters(component)
    beginning = start_values(component, starts)
    initial = component.at(beginning)
    if basis is not None and lattice is not None and lattice.key != basis.lattice.key:
        raise RoleError("the supplied lattice must match the basis lattice")
    geometry = lattice if basis is None else basis.lattice
    layout = plan(
        initial,
        data.channels,
        lattice=geometry,
        kernel=model,
        phase=phase,
        predict=predict,
        resolve=resolve,
        strict=strict,
        max_columns=max_columns,
        depth_tol=depth_tol,
    )
    resolved = set(layout.record.layout.resolved)
    changed = tuple(name for name in defaults if name.split(".", 1)[0] not in resolved)
    primary = component.factors.get(model.primary)
    kind = (
        "linear"
        if moment_form
        else (
            "basis-parametric"
            if changed
            else (
                "separable"
                if defaults
                or (
                    hasattr(primary, "amplitude")
                    and primary.amplitude is None
                    and primary.column is None
                )
                else "fixed"
            )
        )
    )
    structural = replace(layout.record, kind=kind)

    def build(pop, *, accuracy):
        built = build_one(
            pop,
            data.channels,
            layout.lattice,
            kernel=model,
            phase=phase,
            check=accuracy,
            strict=strict,
            max_columns=max_columns,
            cache=cache,
            depth_tol=depth_tol,
        )
        return replace(
            built,
            structural_plan=structural,
            ledger=built.ledger.assume_independence(assumptions),
        )

    if basis is None:
        basis = build(initial, accuracy=False if changed else check)
    elif changed:
        raise RoleError(
            "the averaged family fit needs a rebuildable lattice declaration"
        )
    else:
        _same_basis_stats(initial, basis, allow_mismatch)
        basis = replace(basis, forward_accuracy=check_policy(check))
    if moment_form:
        return fit_moment_form(
            basis, data, component, rank=rank, chunk_bytes=chunk_bytes
        )
    if not changed:
        return fit_family(
            basis, data, component, starts=starts, allow_mismatch=allow_mismatch
        )

    saved = tuple(float(beginning[name]) for name in changed), basis

    def build_trial(pop, params):
        nonlocal saved
        identity = tuple(float(params[name]) for name in changed)
        if identity != saved[0]:
            with trial_tables():
                saved = identity, build(pop, accuracy=False)
        return saved[1]

    result = fit_family(basis, data, component, starts=starts, rebuild=build_trial)
    if check is False:
        return result

    def checked(fit):
        with trial_tables():
            return replace(fit, basis=build(fit.population, accuracy=check))

    if isinstance(result, FamilyBatch):
        return replace(result, pixels=tuple(checked(pixel) for pixel in result.pixels))
    return checked(result)
