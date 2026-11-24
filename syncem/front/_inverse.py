"""Prepare the basis and row map for inverse fits."""

from __future__ import annotations

from dataclasses import replace

from ..base.continuum import ContinuumKernel
from ..base.depth import DEPTH_TOL
from ..engine.basis import SpectralBasis
from ..engine.cache import trial_tables
from ..fitting.basis_parametric import fit_basis_parametric
from ..fitting.conditional_basis import ConditionalBasis, conditional_basis
from ..fitting.family_fit import start_values
from ..fitting.parametrised_factor import fit_parametrised_factor
from ..fitting.family_problem import FamilyProblem, family_template
from ..population import FreeMoments, NonNegativeCells
from ..population._marginal import marginalize
from ..values.errors import RoleError
from ._planning import plan
from ._predictions import (
    _basis_population,
    _factor_record,
    _same_basis_stats,
    build_one,
)


def _moments_basis(
    component,
    data,
    marker_type,
    *,
    basis=None,
    lattice=None,
    kernel=None,
    phase=None,
    predict=None,
    check=True,
    resolve=(),
    max_columns=20_000,
    strict=False,
    cache=True,
    depth_tol=DEPTH_TOL,
    allow_mismatch=False,
    assumptions=(),
    fit_kind=None,
    independent=False,
    extra_resolved=(),
):
    model = ContinuumKernel() if kernel is None else kernel
    joint = None
    if basis is not None:
        joint = basis.joint if isinstance(basis, ConditionalBasis) else basis
        model = joint.averaged.kernel
        if lattice is not None and lattice.key != joint.lattice.key:
            raise RoleError("lattice= and basis= describe different coordinates")
        lattice = joint.lattice
    layout = plan(
        component,
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
    specified = {
        name: law
        for name, law in component.factors.items()
        if not isinstance(law, marker_type)
    }
    roles = layout.record.layout.components[0]
    unknown = [
        source.variables
        for source in roles.sources
        if source.kind == marker_type.__name__
    ]
    if (not independent and len(unknown) != 1) or (independent and len(unknown) < 2):
        raise RoleError("this inverse solver needs one joint unknown moment block")
    children = tuple(name for name, role in roles.roles.items() if role.rule == "R3a")
    if len(children) > 1:
        raise RoleError("R3a fitting needs one conditional child per fitted block")
    if children and independent:
        raise RoleError(
            "conditional children of several fitted factors need an inverse moment map"
        )
    variables = {name for group in unknown for name in group}
    if set(layout.record.layout.resolved) != variables.union(children, extra_resolved):
        raise RoleError("an extra resolved factor needs its own inverse moment map")
    if joint is None:
        joint = SpectralBasis.build(
            data.channels,
            layout.lattice,
            specified=specified,
            symmetries=component.symmetries,
            kernel=model,
            phase=phase,
            check=check,
            max_columns=max_columns,
            depth_tol=depth_tol,
            cache=cache,
        )
        structural = (
            layout.record if fit_kind is None else replace(layout.record, kind=fit_kind)
        )
        joint = replace(joint, structural_plan=structural)
    else:
        source = _basis_population(lattice, specified, component.symmetries, model)
        _same_basis_stats(source, joint, allow_mismatch)
    if assumptions:
        joint = replace(joint, ledger=joint.ledger.assume_independence(assumptions))
    if marker_type is NonNegativeCells:
        if isinstance(basis, ConditionalBasis):
            raise RoleError("nonnegative conditional fitting needs the joint basis")
        child = children[0] if children else None
        declaration = None if child is None else roles.source_for(child).declaration
        return joint, (child, declaration) if child is not None else None
    result = joint
    if children:
        child = children[0]
        declaration = roles.source_for(child).declaration
        if isinstance(basis, ConditionalBasis):
            if basis.child != child or _factor_record(
                basis.declaration
            ) != _factor_record(declaration):
                raise RoleError(
                    "the supplied basis holds a different conditional child"
                )
            result = basis
        else:
            result = conditional_basis(
                joint, child, declaration, parent_variables=unknown[0]
            )
    selections = {
        name: selection
        for source in roles.sources
        if source.kind == "FreeMoments"
        for name, selection in source.declaration.selections().items()
    }
    return result.select(**selections) if selections else result


def _inverse_marginal(component, model, average):
    names = (average,) if isinstance(average, str) else tuple(average)
    transformed, assumptions = marginalize(component, model, names)
    if len(transformed.parts) != 1:
        raise RoleError(
            "fitting this marginal mixture needs a shared inverse moment map"
        )
    return transformed, assumptions


def family_problem(
    population,
    data,
    *,
    basis=None,
    kernel=None,
    lattice=None,
    phase=None,
    predict=None,
    starts=None,
    check=True,
    resolve=(),
    strict=False,
    max_columns=20_000,
    cache=True,
    depth_tol=DEPTH_TOL,
    assumptions=(),
    prior=None,
):
    model = kernel or (
        basis.averaged.kernel if basis is not None else ContinuumKernel()
    )
    template = family_template(population, model, starts)
    initial = template[0]
    if basis is None:
        chosen = plan(
            initial,
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
        basis = build_one(
            initial,
            data.channels,
            chosen.lattice,
            kernel=model,
            phase=phase,
            check=check,
            strict=strict,
            max_columns=max_columns,
            cache=cache,
            depth_tol=depth_tol,
        )
        basis = replace(basis, structural_plan=replace(chosen.record, kind="separable"))
    elif lattice is not None and lattice.key != basis.lattice.key:
        raise RoleError("lattice= and basis= describe different coordinates")
    _same_basis_stats(initial, basis, False)
    if assumptions:
        basis = replace(basis, ledger=basis.ledger.assume_independence(assumptions))
    return FamilyProblem(basis, data, population, template, prior=prior)


def parametric_fit(
    component,
    data,
    parameters,
    *,
    basis,
    lattice,
    model,
    phase,
    predict,
    starts,
    rank,
    penalty,
    solver,
    keep_directions,
    chunk_bytes,
    resolve,
    strict,
    max_columns,
    cache,
    depth_tol,
    assumptions,
):
    if penalty is not None or solver is not None:
        raise RoleError("the profiled fit takes an SVD rank policy")
    initial = component.at(start_values(component, starts))
    explicit = lattice if basis is None else basis.lattice
    layout = plan(
        initial,
        data.channels,
        lattice=explicit,
        kernel=model,
        phase=phase,
        predict=predict,
        resolve=resolve,
        strict=strict,
        max_columns=max_columns,
        depth_tol=depth_tol,
    )
    changed = {name.split(".", 1)[0] for name in parameters}
    resolved = changed & set(layout.record.layout.resolved)
    if resolved:
        if len(resolved) != 1 or resolved != changed:
            raise RoleError(
                "a resolved factor fit currently needs one parameterised family axis"
            )
        name = next(iter(resolved))
        joint = _moments_basis(
            initial,
            data,
            FreeMoments,
            basis=basis,
            lattice=layout.lattice,
            kernel=model,
            phase=phase,
            check=False,
            cache=cache,
            max_columns=max_columns,
            strict=strict,
            depth_tol=depth_tol,
            assumptions=assumptions,
            extra_resolved=(name,),
            fit_kind="bilinear (parametrised factor)",
        )
        return fit_parametrised_factor(
            joint,
            data,
            name,
            component.factors[name],
            starts=starts,
            rank=rank,
            keep_directions=keep_directions,
            chunk_bytes=chunk_bytes,
        )
    if basis is not None:
        raise RoleError("a basis-parametric fit rebuilds the basis")

    def build_trial(values):
        with trial_tables():
            return _moments_basis(
                component.at(values),
                data,
                FreeMoments,
                lattice=layout.lattice,
                kernel=model,
                phase=phase,
                check=False,
                cache=cache,
                max_columns=max_columns,
                strict=strict,
                depth_tol=depth_tol,
                assumptions=assumptions,
                fit_kind="basis-parametric",
            )

    return fit_basis_parametric(
        build_trial,
        data,
        parameters,
        starts=starts,
        rank=rank,
        keep_directions=keep_directions,
        chunk_bytes=chunk_bytes,
    )
