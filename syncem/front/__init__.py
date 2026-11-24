"""Forward entry points for the 0.8 model.

LABEL ``eq: channel basis expansion``.

Channels and population factors use their declared units. Predicted Stokes
have shape (..., n_ch, n_stokes). The ledger retains assumptions and finite
checks. These carry no certified source error.
"""

from __future__ import annotations

import dataclasses

from ..base.continuum import ContinuumKernel
from ..base.depth import DEPTH_TOL
from ..base.harmonic import HarmonicKernel
from ..engine.basis_sum import BasisSum, share_channels
from ..lattice.moments import Moments
from ..population import Population
from ..population._marginal import marginalize
from ..records.ledger import combine_ledgers
from ..values.errors import RoleError
from ._declared import check_declared
from ._harmonic import build_harmonic, is_harmonic_basis, predict_harmonic
from ._harmonic_shape import HarmonicShapeBasis
from ._independence import independence_check
from ._planning import plan
from ._rays import as_ray, predict_ray, predict_ray_batch, ray_population
from ._regime_physical import PaperEnergyBasis
from ._regimes import RegimeBasis
from ._starts import prediction_source
from ._predictions import _records, build_one, channels_from, predict_one


def _role_options(lattice, resolve, average):
    if lattice is not None and resolve:
        raise RoleError("a supplied lattice already declares the resolved variables")
    target = (resolve,) if isinstance(resolve, str) else tuple(resolve)
    if set(target) & set(average):
        raise RoleError("resolve= and average= cannot name the same variable")


def _stat_key(basis):
    pop = basis.forward_population
    names = set(basis.lattice.cells.variables) | set(basis.lattice.representations)
    return _records(pop, set(pop.factors) - names)


def spectral_basis(
    population,
    channels,
    *,
    lattice=None,
    kernel=None,
    phase=None,
    stokes=None,
    check=True,
    resolve=(),
    average=(),
    max_columns=20_000,
    strict=False,
    cache=True,
    depth_tol=DEPTH_TOL,
):
    """Build columns for the resolved variables of a specified population."""
    average = (average,) if isinstance(average, str) else tuple(average)
    _role_options(lattice, resolve, average)
    population, starting = prediction_source(population, strict)
    assumptions = ()
    if average:
        model = ContinuumKernel() if kernel is None else kernel
        population, assumptions = marginalize(population, model, average)
    ray = as_ray(population)
    if ray is not None:
        if average:
            raise RoleError("Ray geometry does not take average=")
        if lattice is None:
            raise RoleError("a Ray without a lattice is evaluated through predict")
        plan(ray, channels, lattice=lattice, kernel=kernel, stokes=stokes)
        if ray.shape:
            raise RoleError("build a Ray basis for one sightline at a time")
        population = ray_population(ray)
    if isinstance(kernel, HarmonicKernel):
        return build_harmonic(
            population, channels, lattice, kernel, phase=phase, check=check
        )
    layout = plan(
        population,
        channels,
        lattice=lattice,
        kernel=kernel,
        phase=phase,
        stokes=stokes,
        resolve=resolve,
        strict=strict,
        max_columns=max_columns,
        depth_tol=depth_tol,
    )
    lattice = layout.lattice
    parts = population.parts if isinstance(population, Population) else (population,)
    bases, stats, member_indices = [], [], []
    for i, part in enumerate(parts):
        basis = build_one(
            part,
            channels,
            lattice,
            kernel=kernel,
            phase=phase,
            stokes=stokes,
            check=check,
            max_columns=max_columns,
            depth_tol=depth_tol,
            cache=cache,
            strict=strict,
        )
        basis = dataclasses.replace(basis, structural_plan=layout.record)
        if starting:
            basis = dataclasses.replace(
                basis,
                record=basis.record.with_notes(
                    f"Starting values used for {', '.join(starting)}"
                ),
            )
        if average:
            basis = dataclasses.replace(
                basis,
                forward_average=average,
                ledger=basis.ledger.assume_independence(assumptions),
            )
        lattice = basis.lattice
        key = _stat_key(basis)
        if key in stats:
            member_indices[stats.index(key)].append(i)
        else:
            stats.append(key)
            bases.append(basis)
            member_indices.append([i])
    if len(bases) == 1:
        return bases[0]
    keys = tuple(f"component {i + 1}" for i in range(len(bases)))
    return BasisSum(bases, keys, member_indices=member_indices)


def _sum_results(results):
    first = results[0]
    return dataclasses.replace(
        first,
        stokes=sum(result.stokes for result in results),
        moments=tuple(result.moments for result in results),
        ledger=combine_ledgers(result.ledger for result in results),
    )


def predict(
    population,
    channels,
    *,
    lattice=None,
    kernel=None,
    phase=None,
    basis=None,
    stokes=None,
    check=True,
    resolve=(),
    average=(),
    allow_mismatch=False,
    max_columns=20_000,
    strict=False,
    cache=True,
    depth_tol=DEPTH_TOL,
):
    """Predict channel Stokes from a specified population or supplied moments."""
    if isinstance(basis, HarmonicShapeBasis):
        if (
            lattice is not None
            or kernel is not None
            or phase is not None
            or resolve
            or average
        ):
            raise RoleError(
                "the supplied HarmonicShapeBasis defines the harmonic coordinates and phase"
            )
        return basis.at(channels).predict(
            population, check=check, stokes=stokes, strict=strict
        )
    if isinstance(basis, RegimeBasis):
        if (
            lattice is not None
            or kernel is not None
            or phase is not None
            or resolve
            or average
        ):
            raise RoleError(
                "the supplied RegimeBasis already defines both model branches"
            )
        if stokes not in (None, "IQU", ("I", "Q", "U")):
            raise RoleError("RegimeBasis returns I, Q and U")
        if allow_mismatch:
            raise RoleError("the stored RegimeBasis fixes its averaged statistics")
        return basis.at(channels).predict(population, check=check)
    if isinstance(basis, PaperEnergyBasis):
        if (
            lattice is not None
            or kernel is not None
            or phase is not None
            or resolve
            or average
        ):
            raise RoleError(
                "the projected paper basis already defines its physical coordinates"
            )
        if stokes not in (None, "IQU", ("I", "Q", "U")):
            raise RoleError("the projected paper basis returns I, Q and U")
        return basis.at(channels).predict(
            population, check=check, allow_mismatch=allow_mismatch
        )
    if basis is not None:
        check_declared(basis, kernel=kernel, phase=phase)
    population, starting = prediction_source(population, strict)
    original = population
    average = (average,) if isinstance(average, str) else tuple(average)
    _role_options(lattice, resolve, average)
    ray = as_ray(population)
    if ray is not None and average:
        raise RoleError("Ray geometry does not take average=")
    if ray is not None and resolve:
        raise RoleError(
            "Ray geometry is explicit. Use lattice= for a resolved Ray prediction."
        )
    assumptions = ()
    building = basis is None
    if building and average:
        model = ContinuumKernel() if kernel is None else kernel
        population, assumptions = marginalize(population, model, average)
    if ray is not None:
        if lattice is None and basis is None:
            return predict_ray(ray, channels, kernel=kernel, phase=phase, stokes=stokes)
        if basis is not None:
            one = basis.parts[0] if isinstance(basis, BasisSum) else basis
            plan(ray, channels, lattice=one.lattice, kernel=one.averaged.kernel)
        if ray.shape:
            return predict_ray_batch(
                ray,
                channels,
                lambda line: predict(
                    line,
                    channels,
                    lattice=lattice,
                    basis=basis,
                    kernel=kernel,
                    phase=phase,
                    stokes=stokes,
                    check=check,
                    allow_mismatch=allow_mismatch,
                    max_columns=max_columns,
                    strict=strict,
                    cache=cache,
                    depth_tol=depth_tol,
                ),
            )
    if basis is None:
        # Q and U share the stored P block. The output selection is applied below.
        blocks = (
            None
            if stokes is None
            else tuple(
                dict.fromkeys("P" if axis in ("Q", "U") else axis for axis in stokes)
            )
        )
        basis = spectral_basis(
            population,
            channels,
            lattice=lattice,
            kernel=kernel,
            phase=phase,
            stokes=blocks,
            check=check,
            resolve=resolve,
            max_columns=max_columns,
            depth_tol=depth_tol,
            strict=strict,
            cache=cache,
        )
    if ray is not None:
        population = ray_population(ray)
    parts = basis.parts if isinstance(basis, BasisSum) else (basis,)
    first = parts[0]
    declared = first.forward_average
    if not building and average and declared != average:
        raise RoleError("average= must match the supplied basis declaration")
    if declared and not isinstance(population, Moments):
        population, assumptions = marginalize(
            population, first.averaged.kernel, declared
        )
    ch = channels_from(channels)
    if not share_channels(parts, ch):
        basis = basis.at(ch)
    if is_harmonic_basis(basis) and not isinstance(population, Moments):
        return predict_harmonic(basis, population, check=check, stokes=stokes)
    if isinstance(basis, BasisSum):
        if not isinstance(population, Population):
            raise RoleError(
                "predict with a BasisSum needs declared population components"
            )
        sources = population.parts
        membership = basis.member_indices or tuple(
            (i,) for i in range(len(basis.parts))
        )
        if len(sources) != sum(len(group) for group in membership):
            raise RoleError(
                "population components do not match the planned basis groups"
            )
        parts = []
        for group in membership:
            component = sources[group[0]]
            for member in group[1:]:
                component = component + sources[member]
            parts.append(component)
        result = _sum_results(
            [
                predict_harmonic(one, part, check=check)
                if is_harmonic_basis(one)
                else predict_one(one, part, check=check, allow_mismatch=allow_mismatch)
                for one, part in zip(basis.parts, parts, strict=True)
            ]
        )
    else:
        result = predict_one(
            basis, population, check=check, allow_mismatch=allow_mismatch
        )
    if assumptions:
        reference = basis.parts[0] if isinstance(basis, BasisSum) else basis
        result = independence_check(result, reference, original, assumptions, check)
    if starting:
        result = dataclasses.replace(
            result, defaults=(*result.defaults, ("starting values", starting))
        )
    return result.select(stokes)
