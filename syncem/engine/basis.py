"""The step-5 spectral basis and its engine entry ``build_from_averaged`` (design/0.8/SPEC.md Sec. 3.4, decisions 149
to 153 and 159 to 162).

LABEL ``eq: channel basis expansion``. ``build_from_averaged(averaged, channels, lattice, ...)`` checks the Stokes
request, computes the column map and its size check before any provider call, routes every product cell to the
stack, the jet or the core (decision 150), and stores one spin-0 spectrum per ``spin0_base`` row and one spin-2
spectrum per ``spin2_base`` row. ``design`` assembles the columns at S3 (``engine.assemble``), so ``w`` and the
azimuth factor enter no stored spectrum. ``SpectralBasis.build`` with the signature of Sec. 3.4 arrives in step 6
and calls this entry. Until then the class is reached as ``syncem.engine.basis``.

With ``cache=True`` the build computes the basis memo key (``engine._basis_cache``). A memo hit builds a new
basis around the requested lattice and azimuth factor, without a provider call. A miss routes the cells through
the S2 store (``engine._s2_columns``): a column of a cell is read when its (Legendre degrees, exponent) is stored
under the cell's context key with a rule chain that agrees up to its total degree, and the providers compute
the other columns only. The S0 and S1 tables, the one-node tables and the jet routines come from the process
cache. The build stores the new S2 columns and the memo entry. ``key`` is the design key, the hash of the memo
key, ``lattice.key`` and the factor. A build whose inputs lack a fingerprint has the key ``None``, the notes
"not cached: <reason>", computes every column and stores nothing. ``cache=False`` bypasses the memo, the S2
store, the disk and the routines and stores nothing.

``at(channels)`` gathers the stored rows whose channel row keys match, bitwise and without a provider call. It
computes the other rows through ``channels[missing]`` with the recorded route of each cell, through the S2
store when the basis was cached. A stack cell that the new channels refuse takes the jet on every channel, so
one provider serves each cell. ``with_order(N')`` at or below the built order caps each per-cell order of the
variables inside the total at ``N'``, sets the total to ``N'``, gathers the spectra of the matching (cell,
pair, powers) rows, keeps the parent's routes and calls no provider. Above the built order it builds on
``lattice.with_order(N')`` with the parent's routes ("auto" where the parent took the stack), so a cached
parent's columns come from the S2 store and the providers compute the higher exponents only. A lattice without
a total order raises. A basis from ``at`` or from ``with_order`` at or below the built order has the key
``None``.

Units: those of the kernel per unit scaled moment. Array shapes: ``spectra0`` ``(n_ch, n_s0, n0)`` real with the
spin-0 blocks I then V, ``spectra2`` ``(n_ch, n2)`` complex or ``None``, ``design()`` ``(n_ch, n_stokes, n)``.
Not certified: the providers, which the tests compare with the study oracles and with each other (check (a) of
Sec. 8.3), and the column rules, which the caller states until the plan of step 7.
"""

from __future__ import annotations

import dataclasses
import time
from typing import Any

import numpy as np

from ..records.ledger import Ledger
from ..base.hybrid import HybridKernel
from ..records.records import SpecifiedRecord
from ._basis_cache import HIT_EVENTS, BuildRun
from ._jet_routine import jet_max_order, jet_mem_bytes
from ._basis_update import (
    S2_EVENTS,
    build_record,
    check_arguments,
    check_factor,
    derived_record,
    raised_routes,
    stokes_blocks,
)
from ._routes import route_cells
from ._s2_columns import reuse_cells
from ._rule_fingerprint import fixed_fingerprint
from .assemble import gather_spectra
from .average import AveragedKernel, NodeRule
from .cache import default_cache
from .columns import ColumnMap, ColumnRules, column_map
from ._spectral_basis import SpectralBasis

__all__ = ["SpectralBasis", "build_from_averaged", "specified_records"]


def specified_records(
    averaged: AveragedKernel, fingerprints: dict | None = None
) -> tuple[SpecifiedRecord, ...]:
    """One ``SpecifiedRecord`` per averaged or fixed variable (decision 157).

    A rule's fingerprint is ``fingerprints[name]``, the digest of its ``d_N`` on every cell, or ``None``. A fixed
    value has the digest of its canonical float.
    """
    prints = fingerprints or {}
    out = []
    for name, rule in averaged.averaged:
        if isinstance(rule, NodeRule):
            parents = rule.parents or ()
            fingerprint = prints.get(name)
            out.append(
                SpecifiedRecord(
                    name, "NodeRule", parents=parents, fingerprint=fingerprint
                )
            )
        else:
            out.append(SpecifiedRecord(name, "callable"))
    out += [
        SpecifiedRecord(name, "fixed", fingerprint=fixed_fingerprint(value))
        for name, value in averaged.fixed
    ]
    return tuple(out)


def build_from_averaged(
    averaged: AveragedKernel,
    channels: Any,
    lattice: Any,
    *,
    rules: ColumnRules = ColumnRules(),
    routes: Any = "auto",
    s1: str = "lookup",
    stokes: Any = None,
    azimuth_factor: Any = None,
    max_columns: int = 20_000,
    cache: bool = True,
) -> SpectralBasis:
    """The basis of ``averaged`` on ``channels`` and ``lattice`` (decisions 149 and 159 to 162).

    ``cache=True`` reads and stores the basis memo, the S0, S1 and S2 stages and the jet routines of the process
    cache when every key has a fingerprint. ``cache=False`` bypasses them and stores nothing. A build whose rules
    have no fingerprint and a ``cache=False`` build compile fresh jet routines, so a session of many such builds
    grows by about 16 MB of resident memory per build and signature (decision 169).
    """
    start = time.perf_counter()
    check_arguments(rules, cache)
    kernel = averaged.kernel
    if isinstance(kernel, HybridKernel):
        from ._hybrid_basis import build_hybrid_from_averaged

        return build_hybrid_from_averaged(
            averaged,
            channels,
            lattice,
            rules=rules,
            routes=routes,
            s1=s1,
            stokes=stokes,
            azimuth_factor=azimuth_factor,
            max_columns=max_columns,
            cache=cache,
        )
    blocks = stokes_blocks(kernel, rules, stokes)
    rules = dataclasses.replace(rules, stokes=blocks)
    from .provider_fourier import build_fourier, fourier_axis

    if fourier_axis(lattice, kernel.azimuth) is not None:
        if azimuth_factor is not None:
            raise ValueError("an explicit azimuth factor conflicts with AzimuthFourier")
        if not isinstance(routes, str) or routes not in ("auto", "jet"):
            raise ValueError("AzimuthFourier builds through the jet provider")
        return build_fourier(
            averaged, channels, lattice, rules=rules, max_columns=max_columns
        )
    factor = check_factor(kernel, lattice, blocks, azimuth_factor)
    columns = column_map(lattice, kernel, rules, max_columns=max_columns)
    if averaged.leaf_factors:
        cache = False
    run = BuildRun.open(averaged, channels, lattice, columns, rules, routes, s1, cache)
    request = (averaged, channels, lattice, factor, rules)
    found = run.lookup()
    if found is not None:
        return _rebound(found, run, request, start)
    rows, records, counts = _provide(run, request, routes, s1, blocks)
    defaults = {
        "max_columns": int(max_columns),
        "cache": cache,
        "jet_mem_bytes": jet_mem_bytes(),
        "jet_max_order": jet_max_order(),
    }
    route = (records, blocks, s1)
    basis = _fresh(run, request, columns, route, rows, counts, defaults, start)
    run.store(basis)
    return basis


# ------------------------------------------------------------------ build helpers


def _fresh(
    run: BuildRun,
    request: tuple,
    columns: ColumnMap,
    route: tuple,
    rows: dict,
    counts: dict,
    defaults: dict,
    start: float,
) -> SpectralBasis:
    """The basis of a build that missed the memo, with its record."""
    averaged, channels, lattice, factor, rules = request
    records, blocks, s1 = route
    spectra0, spectra2 = gather_spectra(rows["I"], rows["P"], rows.get("V"), columns)
    if defaults["cache"]:
        defaults = {**defaults, **dict(default_cache().settings)}
    key = run.design_key(lattice, factor)
    record = build_record(
        records,
        blocks,
        s1,
        columns,
        lattice,
        events=counts,
        timings={"build": time.perf_counter() - start},
        keys={"memo": run.memo_key, "design": key},
        defaults=defaults,
        notes=run.notes,
    )
    return SpectralBasis(
        lattice=lattice,
        channels=channels,
        columns=columns,
        spectra0=spectra0,
        spectra2=spectra2,
        azimuth_factor=factor,
        specified=specified_records(averaged, run.fingerprints),
        record=record,
        ledger=Ledger(),
        key=key,
        averaged=averaged,
        rules=rules,
    )


def _provide(
    run: BuildRun, request: tuple, routes: Any, s1: str, blocks: tuple
) -> tuple[dict, tuple, dict]:
    """Provider rows per block, the route records and the event counts of a build that missed the memo.

    A build with keys reads the S2 store column by column (``reuse_cells``). Any other build computes every
    column with ``route_cells`` and stores none.
    """
    averaged, channels, lattice = request[:3]
    counts: dict = {}
    common = dict(s1=s1, with_v="V" in blocks, events=counts)
    try:
        if run.reuse is not None:
            out = reuse_cells(run.reuse, averaged, channels, lattice, routes, **common)
        else:
            out = route_cells(
                averaged,
                channels,
                lattice,
                routes,
                memo=run.memo,
                jet=run.jet,
                **common,
            )
            size = int(sum(np.size(x) for x in out[:3] if x is not None))
            counts.update(dict.fromkeys(S2_EVENTS, 0), **{S2_EVENTS[0]: size})
        counts.update(run.finish())
    finally:
        run.close()
    I, P, V, records = out
    rows = {"I": I, "P": P, **({"V": V} if V is not None else {})}
    counts.update(
        {"channel rows reused": 0, "channel rows computed": int(channels.n_ch)}
    )
    return rows, records, counts


def _rebound(entry: dict, run: BuildRun, request: tuple, start: float) -> SpectralBasis:
    """A memo hit around the requested lattice and azimuth factor. No provider runs and no S2 column is computed."""
    averaged, channels, lattice, factor, rules = request
    events = {**HIT_EVENTS, **run.close(), "channel rows reused": int(channels.n_ch)}
    key = run.design_key(lattice, factor)
    stored = entry["record"]
    record = derived_record(stored, stored.routes, events, start)
    keys = {**dict(stored.keys), "lattice": lattice.key, "design": key}
    return SpectralBasis(
        lattice=lattice,
        channels=channels,
        columns=entry["columns"],
        spectra0=entry["spectra0"],
        spectra2=entry["spectra2"],
        azimuth_factor=factor,
        specified=entry["specified"],
        record=dataclasses.replace(record, keys=keys),
        ledger=Ledger(),
        key=key,
        averaged=averaged,
        rules=rules,
    )


def _raised(basis: SpectralBasis, order: int) -> SpectralBasis:
    """``with_order(N')`` above the built order: a build on ``lattice.with_order(N')`` with the parent's routes.

    The columns of the built order come from the S2 store when the parent was cached, so the providers compute
    the exponents above it only. A cell the parent put on the stack takes "auto", so a stack refusal at ``N'``
    sends it to the jet.
    """
    defaults = dict(basis.record.defaults)
    out = build_from_averaged(
        basis.averaged,
        basis.channels,
        basis.lattice.with_order(order),
        rules=basis.rules,
        routes=raised_routes(basis.record.routes),
        s1=basis.record.s1,
        azimuth_factor=basis.azimuth_factor,
        max_columns=defaults.get("max_columns", 20_000),
        cache=bool(defaults.get("cache", True)),
    )
    note = f"routes of the order-{basis.lattice.order} build"
    return dataclasses.replace(
        out,
        record=out.record.with_notes(note),
        ledger=basis.ledger,
        forward_population=basis.forward_population,
        forward_phase=basis.forward_phase,
        forward_defaults=basis.forward_defaults,
        forward_depth_tol=basis.forward_depth_tol,
        forward_average=basis.forward_average,
        structural_plan=basis.structural_plan,
        forward_accuracy=basis.forward_accuracy,
    )
