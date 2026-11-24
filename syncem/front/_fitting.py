"""Fit entry-point preparation for free moments on a planned lattice."""

from __future__ import annotations


import jax
import numpy as np

from ..base.continuum import ContinuumKernel
from ..base.depth import DEPTH_TOL
from ..engine.basis_sum import BasisSum
from ..fitting.data import StokesData
from ..fitting.factor_fit import fit_factors
from ..fitting.family_fit import fit_family, free_parameters
from ..fitting.free_fit import RetainedCombinations, fit_free
from ..fitting.penalized_fit import fit_nonnegative, fit_penalized
from ..fitting.penalties import Penalty
from ..fitting.problem import Problem, _solve_penalty
from ..fitting.shared_azimuth import fit_shared_azimuth
from ..population import FreeMoments, NonNegativeCells, Population
from ..values.errors import RoleError
from ._declared import check_azimuth_declared, check_declared
from ._fit_sum import fit_sum, sum_problem
from ._inverse import (
    _inverse_marginal,
    _moments_basis,
    family_problem,
    parametric_fit,
)
from ._regime_physical import PaperEnergyBasis
from ._regimes import RegimeBasis
from ._separable import separable_fit
from ._shared import shared_azimuth, shared_source


def _population(value):
    return value if isinstance(value, Population) else Population(value)


def _model_of(kernel, basis):
    if kernel is not None:
        return kernel
    if basis is None:
        return ContinuumKernel()
    first = basis.parts[0] if isinstance(basis, BasisSum) else basis
    averaged = getattr(getattr(first, "joint", first), "averaged", None)
    return None if averaged is None else averaged.kernel


def problem(population, data, *, basis=None, prior=None, **options):
    """Compile the fixed-basis likelihood for a free-moment population."""
    if isinstance(basis, RegimeBasis):
        raise RoleError(
            "use basis.problem(data) for a likelihood in independent branch moments"
        )
    if isinstance(basis, PaperEnergyBasis) and any(
        options.get(name) is not None for name in ("kernel", "phase")
    ):
        raise RoleError(
            "the projected paper basis already defines its kernel and phase"
        )
    if not isinstance(data, StokesData):
        raise TypeError("problem needs StokesData")
    component = _population(population)
    check_azimuth_declared(
        component, data, _model_of(options.get("kernel"), basis)
    )
    if component.components:
        return sum_problem(component, data, basis=basis, prior=prior, **options)
    if isinstance(basis, BasisSum):
        raise RoleError("a BasisSum needs a population sum")
    if basis is not None:
        check_declared(
            basis, kernel=options.get("kernel"), phase=options.get("phase")
        )
    average = options.pop("average", ())
    assumptions = ()
    if average:
        model = options.get("kernel") or (
            basis.averaged.kernel if basis is not None else ContinuumKernel()
        )
        component, assumptions = _inverse_marginal(component, model, average)
    declarations = [
        block
        for block in (*component.blocks, *component.factors.values())
        if isinstance(block, FreeMoments)
    ]
    if not declarations:
        return family_problem(
            component,
            data,
            basis=basis,
            assumptions=assumptions,
            prior=prior,
            **options,
        )
    declared = [block.prior for block in declarations if block.prior is not None]
    if prior is None and declared:
        if len(declarations) != 1:
            raise RoleError("pass a prior on the full concatenated lattice rows")
        prior = declared[0]
    built = _moments_basis(
        component, data, FreeMoments, basis=basis, assumptions=assumptions, **options
    )
    return Problem(built, data, prior=prior)


def fit(
    population,
    data,
    *,
    lattice=None,
    kernel=None,
    phase=None,
    basis=None,
    predict=None,
    rank=None,
    penalty=None,
    starts=None,
    solver=None,
    keep_directions=False,
    chunk_bytes=2**31,
    check=True,
    resolve=(),
    average=(),
    allow_mismatch=False,
    max_columns=20_000,
    strict=False,
    cache=True,
    key=None,
    depth_tol=DEPTH_TOL,
):
    """Fit declared lattice moments or a specified family to channel Stokes data."""
    if isinstance(basis, PaperEnergyBasis) and (
        kernel is not None or phase is not None
    ):
        raise RoleError(
            "the projected paper basis already defines its kernel and phase"
        )
    if isinstance(basis, RegimeBasis):
        component = _population(population)
        if component.blocks:
            raise RoleError(
                "use basis.fit(data) for free branch moments or se.paper.energy_basis for physical cells"
            )
        if (
            any(
                x is not None
                for x in (lattice, kernel, phase, rank, penalty, solver, key)
            )
            or resolve
            or average
            or allow_mismatch
        ):
            raise RoleError(
                "the stored RegimeBasis defines the physical family response and its coordinates"
            )
        result = basis.fit_family(component, data, starts=starts)
        if predict is not None:
            from dataclasses import replace

            destination = basis.at(predict)
            result = replace(
                result,
                basis=destination,
                moments=destination.moments(result.population),
            )
        return result
    if isinstance(data, RetainedCombinations):
        if (
            data.basis is None
            or any(x is not None for x in (phase, predict, rank, penalty, solver))
            or average
            or resolve
        ):
            raise RoleError(
                "the compressed family fit uses the original observational basis and a fixed retained rank"
            )
        original = data.basis
        if basis is not None and basis is not original:
            raise RoleError("the compressed family fit uses the observational basis")
        if lattice is not None and lattice.key != original.lattice.key:
            raise RoleError("the compressed family fit needs the original lattice key")
        if kernel is not None and kernel is not original.averaged.kernel:
            raise RoleError("the compressed family fit needs the original kernel")
        return fit_family(
            original,
            data,
            _population(population),
            starts=starts,
            allow_mismatch=allow_mismatch,
        )
    if not isinstance(data, StokesData):
        raise TypeError("fit needs StokesData")
    component = _population(population)
    if basis is not None and not component.components:
        check_declared(basis, kernel=kernel, phase=phase)
    check_azimuth_declared(component, data, _model_of(kernel, basis))
    if component.components:
        if any(value is not None for value in (penalty, starts, solver, key)):
            raise RoleError("a sum of free moments takes a linear rank policy")
        return fit_sum(
            component,
            data,
            basis=basis,
            lattice=lattice,
            kernel=kernel,
            phase=phase,
            predict=predict,
            check=check,
            resolve=resolve,
            average=average,
            allow_mismatch=allow_mismatch,
            max_columns=max_columns,
            strict=strict,
            cache=cache,
            depth_tol=depth_tol,
            chunk_bytes=chunk_bytes,
            rank=rank,
            keep_directions=keep_directions,
        )
    if isinstance(basis, BasisSum):
        raise RoleError("a BasisSum needs a population sum")
    model = kernel or (
        basis.averaged.kernel if basis is not None else ContinuumKernel()
    )
    assumptions = ()
    if average:
        component, assumptions = _inverse_marginal(component, model, average)
    markers = tuple(
        block for block in component.blocks if isinstance(block, FreeMoments)
    )
    named = {
        name: law
        for name, law in component.factors.items()
        if isinstance(law, FreeMoments)
    }
    declarations = (*markers, *named.values())
    priors = tuple(block.prior for block in declarations if block.prior is not None)
    nonnegative = [
        block for block in component.blocks if isinstance(block, NonNegativeCells)
    ]
    nonnegative.extend(
        law.bind(name)
        for name, law in component.factors.items()
        if isinstance(law, NonNegativeCells)
    )
    if shared_azimuth(model, component):
        if priors:
            raise RoleError(
                "the shared-azimuth fit does not support a free-moment prior"
            )
        if (
            nonnegative
            or penalty is not None
            or starts is not None
            or solver is not None
        ):
            raise RoleError("the shared-azimuth route takes an SVD rank policy")
        component, lattice = shared_source(model, component, lattice)
        built = _moments_basis(
            component,
            data,
            FreeMoments,
            basis=basis,
            lattice=lattice,
            kernel=model,
            phase=phase,
            predict=predict,
            check=check,
            resolve=resolve,
            cache=cache,
            max_columns=max_columns,
            strict=strict,
            depth_tol=depth_tol,
            allow_mismatch=allow_mismatch,
            assumptions=assumptions,
            fit_kind="bilinear",
        )
        return fit_shared_azimuth(
            built,
            data,
            rank=rank,
            keep_directions=keep_directions,
            chunk_bytes=chunk_bytes,
        )
    if nonnegative:
        if len(nonnegative) != 1 or markers or named:
            raise RoleError(
                "the nonnegative route needs one joint NonNegativeCells block"
            )
        if rank is not None or starts is not None or solver is not None:
            raise RoleError("the nonnegative route takes a penalty and uses NNLS")
        basis, conditional = _moments_basis(
            component,
            data,
            NonNegativeCells,
            basis=basis,
            lattice=lattice,
            kernel=model,
            phase=phase,
            predict=predict,
            check=check,
            resolve=resolve,
            max_columns=max_columns,
            depth_tol=depth_tol,
            strict=strict,
            cache=cache,
            allow_mismatch=allow_mismatch,
            assumptions=assumptions,
        )
        return fit_nonnegative(
            basis,
            data,
            nonnegative[0],
            Penalty() if penalty is None else penalty,
            chunk_bytes=chunk_bytes,
            conditional=conditional,
        )
    if not (markers or named):
        moment_form = hasattr(component.factors.get(model.primary), "basis_density")
        if (
            penalty is not None
            or solver is not None
            or (rank is not None and not moment_form)
        ):
            raise RoleError(
                "the separable route takes the trust-region solver without a rank policy"
            )
        return separable_fit(
            component,
            data,
            basis=basis,
            lattice=lattice,
            model=model,
            phase=phase,
            predict=predict,
            starts=starts,
            rank=rank,
            moment_form=moment_form,
            chunk_bytes=chunk_bytes,
            check=check,
            resolve=resolve,
            strict=strict,
            max_columns=max_columns,
            cache=cache,
            depth_tol=depth_tol,
            assumptions=assumptions,
            allow_mismatch=allow_mismatch,
        )
    parameters = free_parameters(component)
    independent = tuple(block.variables for block in markers) + tuple(
        (name,) for name in named
    )
    if len(independent) > 1:
        if priors:
            raise RoleError(
                "independent moment factor fits do not support free-moment priors"
            )
        if parameters or penalty is not None or solver is not None:
            raise RoleError(
                "independent free moment factors take a rank policy without a penalty"
            )
        groups = sorted(independent, key=lambda group: model.primary not in group)
        seed = (
            0
            if key is None
            else int.from_bytes(
                np.asarray(jax.random.key_data(key), dtype=np.uint32).tobytes(),
                "little",
            )
        )
        built = _moments_basis(
            component,
            data,
            FreeMoments,
            basis=basis,
            lattice=lattice,
            kernel=model,
            phase=phase,
            predict=predict,
            check=check,
            resolve=resolve,
            cache=cache,
            strict=strict,
            max_columns=max_columns,
            depth_tol=depth_tol,
            allow_mismatch=allow_mismatch,
            assumptions=assumptions,
            fit_kind="bilinear",
            independent=True,
        )
        return fit_factors(
            built,
            data,
            groups,
            rank=rank,
            starts=3 if starts is None else starts,
            seed=seed,
            chunk_bytes=chunk_bytes,
        )
    if parameters:
        if priors:
            raise RoleError("parameterised fits do not support a free-moment prior")
        return parametric_fit(
            component,
            data,
            parameters,
            basis=basis,
            lattice=lattice,
            model=model,
            phase=phase,
            predict=predict,
            starts=starts,
            rank=rank,
            penalty=penalty,
            solver=solver,
            keep_directions=keep_directions,
            chunk_bytes=chunk_bytes,
            resolve=resolve,
            strict=strict,
            max_columns=max_columns,
            cache=cache,
            depth_tol=depth_tol,
            assumptions=assumptions,
        )
    if (
        starts is not None
        or solver is not None
        or (penalty is not None and rank is not None)
    ):
        raise RoleError("the linear fit needs a rank policy or a ridge penalty")
    penalty = _solve_penalty(priors[0] if priors else None, rank=rank, penalty=penalty)
    basis = _moments_basis(
        component,
        data,
        FreeMoments,
        basis=basis,
        lattice=lattice,
        kernel=model,
        phase=phase,
        predict=predict,
        check=check,
        resolve=resolve,
        max_columns=max_columns,
        depth_tol=depth_tol,
        strict=strict,
        cache=cache,
        allow_mismatch=allow_mismatch,
        assumptions=assumptions,
    )
    if penalty is not None:
        return fit_penalized(
            basis,
            data,
            penalty,
            keep_directions=keep_directions,
            chunk_bytes=chunk_bytes,
        )
    return fit_free(
        basis,
        data,
        rank=rank,
        keep_directions=keep_directions,
        chunk_bytes=chunk_bytes,
    )
