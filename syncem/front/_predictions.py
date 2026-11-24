"""Forward predictions and independent integrals on cell and support panels."""

from __future__ import annotations

import dataclasses

import equinox as eqx
import jax.numpy as jnp

from ..base.channels import Channels
from ..base.continuum import ContinuumKernel
from ..base.depth import DEPTH_TOL, tolerance
from ..base.hybrid import HybridKernel
from ..distributions._statistics import factor_record as _factor_record
from ..engine._basis_ledger import depth_term
from ..engine._basis_update import stokes_blocks
from ..engine._small_x import small_x_term
from ..engine.assemble import predict_values
from ..engine._remainder import remainder_term
from ..engine.basis import SpectralBasis, build_from_averaged
from ..engine.direct import continuum_tail_threshold
from ..engine.provider_fourier import FourierBasis
from ..lattice.moments import Moments
from ..population import Population, pair
from ..records.ledger import ErrorTerm, Ledger
from ..values.errors import SpecifiedMismatch
from ._accuracy import check_policy, estimate_direct, provider_checks
from ._direct import (
    UnassignedSupport,
    _direct_values,
    _projected_cell_values,
    _spectrum,
)
from ._quadrature_accuracy import quadrature_terms
from ._specified import comparison_plan, replace_terms, specified_terms
from ._setup import (
    bind_lattice,
    column_rules,
    configure,
    default_lattice,
    placed,
    screen_truncation,
    support_for,
)


def channels_from(channels):
    """Keep response bands or interpret the frequency pair as monochromatic."""
    if isinstance(channels, Channels):
        return channels
    if isinstance(channels, tuple) and len(channels) == 2:
        return Channels.at(channels[0], unit=channels[1])
    raise TypeError("channels must be Channels or (frequencies, unit)")


def _records(pop, names):
    return {
        name: _factor_record(pop.factors[name]) for name in names if name in pop.factors
    }


def _same_basis_stats(population, basis, allow_mismatch):
    source = basis.forward_population
    if source is None:
        return
    kernel = basis.averaged.kernel
    pop, _ = placed(population, kernel)
    resolved = set(basis.lattice.cells.variables) | set(basis.lattice.representations)
    averaged = set(pop.factors) - resolved
    match = _records(pop, averaged) == _records(source, averaged)
    if not match and not allow_mismatch:
        raise SpecifiedMismatch(
            "population statistics differ from those used for this basis"
        )


def _physical(kernel, channels):
    return ErrorTerm.from_model("physical_kernel", kernel.physical_error(channels))


def build_one(
    population,
    channels,
    lattice=None,
    *,
    kernel=None,
    phase=None,
    stokes=None,
    check=True,
    max_columns=20_000,
    cache=True,
    strict=False,
    depth_tol=DEPTH_TOL,
):
    ch = channels_from(channels)
    kernel = ContinuumKernel() if kernel is None else kernel
    lat = default_lattice(population, kernel, ch) if lattice is None else lattice
    lat = bind_lattice(lat, kernel)
    if isinstance(kernel, HybridKernel):
        kernel = kernel.bind(lat.cells)
    setup = configure(population, kernel, ch, lat, phase, strict=strict)
    rules = column_rules(setup, lat)
    blocks = stokes_blocks(setup.averaged.kernel, rules, stokes)
    tol = tolerance(depth_tol)
    depth = (
        depth_term(lat, kernel, setup.averaged.phase, ch, tol)
        if "P" in blocks
        else None
    )
    factor = setup.azimuth if "P" in blocks else None
    basis = build_from_averaged(
        setup.averaged,
        ch,
        lat,
        rules=rules,
        stokes=stokes,
        azimuth_factor=factor,
        max_columns=max_columns,
        cache=cache,
    )
    policy = check_policy(check)
    if policy is not None and policy.providers and isinstance(basis, FourierBasis):
        record = basis.record.with_notes("Fourier columns have one provider.")
    elif policy is not None and policy.providers and isinstance(kernel, HybridKernel):
        record = basis.record.with_notes(
            "Each assigned hybrid cell uses the child jet provider."
        )
    elif policy is not None and policy.providers:
        agreement, notes = provider_checks(basis, policy)
        record = dataclasses.replace(basis.record, agreement=agreement).with_notes(
            *notes
        )
    else:
        record = basis.record
    terms = (*basis.ledger.terms, _physical(setup.averaged.kernel, ch))
    if depth is not None:
        terms = (*terms, depth)
    if "P" in blocks:
        omitted = screen_truncation(
            setup.population,
            phase,
            conditional=any(
                block == "depth" for block, _ in setup.averaged.leaf_factors
            ),
        )
        if omitted is not None:
            terms = (*terms, omitted)
    result = dataclasses.replace(
        basis,
        record=record,
        ledger=Ledger(terms),
        forward_population=setup.population,
        forward_phase=phase,
        forward_defaults=setup.defaults,
        forward_depth_tol=tol if depth is not None else None,
        forward_accuracy=policy,
    )
    return dataclasses.replace(
        result, ledger=replace_terms(result.ledger, specified_terms(result))
    )


def _basis_population(basis, specified, symmetries, kernel):
    if isinstance(specified, Population):
        return specified
    facts = dict(specified or {})
    for variable in kernel.variables:
        name = variable.name
        if name in facts or variable.default is not None:
            continue
        if name in basis.cells.variables:
            cell = basis.cells.variables.index(name)
            value = basis.cells.reference[0, cell]
            facts[name] = (value, variable.unit) if variable.unit else value
        elif name in basis.representations:
            facts[name] = (0.0, variable.unit) if variable.unit else 0.0
    return Population(symmetries=symmetries, **facts)


def build_specified(
    channels, lattice, *, specified=None, symmetries=(), kernel=None, **options
):
    model = ContinuumKernel() if kernel is None else kernel
    population = _basis_population(lattice, specified, symmetries, model)
    return build_one(population, channels, lattice, kernel=model, **options)


def basis_moments(basis, population, *, allow_mismatch=False, nodes=24):
    if isinstance(population, Moments):
        return pair(population, basis.lattice)
    kernel, lattice = basis.averaged.kernel, basis.lattice
    parts = population.parts if isinstance(population, Population) else (population,)
    output = []
    for part in parts:
        _same_basis_stats(part, basis, allow_mismatch)
        pop, _ = placed(part, kernel)
        resolved = set(lattice.cells.variables) | set(lattice.representations)
        if kernel.primary is not None:
            resolved.add(
                kernel.primary
            )  # preserve the column even when the primary shape is averaged
        factors = {name: val for name, val in pop.factors.items() if name in resolved}
        measure = eqx.tree_at(lambda p: p.factors, pop, factors)
        variables = {v.name: v for v in kernel.variables}
        units = {name: v.unit for name, v in variables.items()}
        face = getattr(basis, "excluded_upper_gamma", None)
        open_upper = None if face is None else {"gamma": face}
        output.append(
            pair(
                measure,
                lattice,
                units=units,
                variables=variables,
                n=nodes,
                open_upper=open_upper,
            ).values
        )
    return Moments(lattice, sum(output))


def predict_one(basis: SpectralBasis, population, *, check=True, allow_mismatch=False):
    """Contract the spectral basis and measure the two independent residual terms."""
    moments = basis.moments(population, allow_mismatch=allow_mismatch)
    values = predict_values(basis, moments)
    ledger = basis.ledger
    if isinstance(basis, FourierBasis):
        sampling = ErrorTerm(
            basis.sampling_error(moments),
            "estimate",
            "Stokes change after doubling the angular projection grid.",
            name="quadrature:azimuth",
        )
        ledger = Ledger((*ledger.terms, sampling), assumptions=ledger.assumptions)
    policy = check_policy(check)
    if policy is not None:
        ledger = replace_terms(ledger, (remainder_term(basis, moments),))
    if policy is not None and isinstance(population, Population):
        comparison = (
            comparison_plan(basis, population, place=placed, record=_factor_record)
            if allow_mismatch
            else None
        )
        if comparison is not None and comparison.unavailable is not None:
            ledger = replace_terms(ledger, comparison.unbounded())
            return _spectrum(basis, values, ledger=ledger, moments=moments)
        baseline = population.parts if comparison is None else comparison.baseline_parts
        direct_cost = estimate_direct(basis)
        estimate = direct_cost
        if comparison is not None and comparison.changed:
            estimate *= comparison.evaluation_count(
                projected=isinstance(basis, FourierBasis)
            )
        if estimate > policy.max_seconds:
            note = f"not measured. Estimated direct cost {estimate:.2g} s exceeds {policy.max_seconds:g} s"
            if comparison is not None and comparison.changed:
                ledger = replace_terms(ledger, comparison.unbounded(note))
            terms = (*ledger.terms, ErrorTerm.unbounded(note, name="representation"))
            ledger = Ledger(terms, assumptions=ledger.assumptions)
            return _spectrum(basis, values, ledger=ledger, moments=moments)
        cell_checks = [_direct_values(basis, part, "cells") for part in baseline]
        cells = sum(value for value, _ in cell_checks)
        if (
            isinstance(basis, FourierBasis)
            and ledger["azimuth_truncation"].kind == "unbounded"
        ):
            projected = sum(
                _projected_cell_values(basis, setup) for _, setup in cell_checks
            )
            term = ErrorTerm(
                projected - cells,
                "measured",
                "Omitted angular orders on the in-cell panels of this population.",
                name="azimuth_truncation",
                signed=True,
            )
            ledger = Ledger(
                tuple(term if t.name == term.name else t for t in ledger.terms),
                assumptions=ledger.assumptions,
            )
            cells = projected
        try:
            uncovered = [_direct_values(basis, p, "outside")[0] for p in baseline]
        except UnassignedSupport as exc:
            terms = (
                *ledger.terms,
                ErrorTerm(values - cells, "measured", name="representation"),
                ErrorTerm.unbounded(str(exc), name="outside_lattice"),
            )
            return _spectrum(
                basis,
                values,
                ledger=Ledger(terms, assumptions=ledger.assumptions),
                moments=moments,
            )
        outside = sum(uncovered)
        if comparison is not None and comparison.changed:
            totals = tuple(
                value + extra
                for (value, _), extra in zip(cell_checks, uncovered, strict=True)
            )
            diagnostics = comparison.measure(
                totals, lambda part: _direct_values(basis, part, "support")[0]
            )
            ledger = replace_terms(ledger, diagnostics)
        note = ""
        kernel = basis.averaged.kernel
        if type(kernel) is ContinuumKernel:
            if any(
                support_for(p, kernel.primary)[0] < kernel.gamma_min
                for p in population.parts
            ):
                note = "declared support extends below the continuum kernel regime"
        terms = (
            *ledger.terms,
            ErrorTerm(values - cells, "measured", name="representation"),
            ErrorTerm(-outside, "measured", note, name="outside_lattice"),
        )
        quadrature, *other = quadrature_terms(
            basis,
            cell_checks,
            baseline,
            max(0.0, policy.max_seconds - estimate),
            direct_cost,
        )
        if quadrature.value is not None:
            refined = basis_moments(
                basis, population, allow_mismatch=allow_mismatch, nodes=48
            )
            delta = jnp.abs(predict_values(basis, refined) - values)
            quadrature = dataclasses.replace(
                quadrature,
                value=quadrature.value + delta,
                note=f"{quadrature.note} Includes doubled resolved-moment quadrature.",
            )
        terms = (*terms, quadrature, *other)
        if type(kernel) is ContinuumKernel:
            terms = (*terms, small_x_term(kernel))
            for _, setup in cell_checks:
                cutoff = continuum_tail_threshold(kernel, basis.channels, setup.b_max)
                if (
                    cutoff is not None
                    and support_for(setup.population, kernel.primary)[1] > cutoff
                ):
                    terms = (
                        *terms,
                        ErrorTerm.unbounded(
                            "the population support can reach the small-x continuation",
                            name="small_x:tail",
                        ),
                    )
                    break
        ledger = Ledger(terms, assumptions=ledger.assumptions)
    return _spectrum(basis, values, ledger=ledger, moments=moments)
