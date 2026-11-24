"""Static child-kernel columns and reusable channel rows for a hybrid basis."""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
import time

import equinox as eqx
import numpy as np

from ..base.hybrid import HybridKernel
from ..records.ledger import ErrorTerm, Ledger
from ..records.records import RouteRecord
from ..values.errors import RoleError
from ._basis_cache import BuildRun, fresh_jet
from ._basis_ledger import channel_ledger
from ._basis_update import (
    S2_EVENTS,
    check_arguments,
    check_factor,
    derived_record,
    match_channels,
    stokes_blocks,
    uncached_record,
)
from ._hybrid_switch import switch_measurement
from ._spectral_basis import SpectralBasis
from .basis import _fresh, _rebound
from .columns import ColumnRules, column_map
from .provider_fourier import fourier_axis
from .provider_jet import jet_lattice

__all__ = ["HybridBasis", "build_hybrid_from_averaged"]


@dataclass(frozen=True, kw_only=True)
class HybridRouteRecord(RouteRecord):
    """The child kernel chosen by the eager cell assignment."""

    kernel_index: int


def _with_switch(ledger, term):
    others = tuple(t for t in ledger.terms if t.name != "kernel_switch")
    return Ledger((*others, term), assumptions=ledger.assumptions)


def _hybrid_basis(basis, term):
    return HybridBasis(
        **{
            **{f.name: getattr(basis, f.name) for f in fields(SpectralBasis)},
            "ledger": _with_switch(basis.ledger, term),
        }
    )


class HybridBasis(SpectralBasis):
    """Stored columns whose per-cell kernel choice is made before Taylor expansion."""

    def _on_grid(self, channels, lattice):
        return build_hybrid_from_averaged(
            self.averaged,
            channels,
            lattice,
            rules=self.rules,
            s1=self.record.s1,
            azimuth_factor=self.azimuth_factor,
            max_columns=dict(self.record.defaults)["max_columns"],
            cache=dict(self.record.defaults)["cache"],
        )

    def at(self, channels):
        """Reuse all stored rows and build only genuinely new response bands."""
        start = time.perf_counter()
        keys = channels.row_keys()
        found, missing = match_channels(self.channels.row_keys(), keys)
        if not missing:
            return super().at(channels)
        positions = tuple(missing.values())
        fresh = self._on_grid(channels[np.asarray(positions)], self.lattice)
        new_row = {key: i for i, key in enumerate(fresh.channels.row_keys())}

        def gather(old, new):
            if old is None:
                return None
            return np.stack(
                [old[i] if i >= 0 else new[new_row[key]] for i, key in zip(found, keys)]
            )

        old_term, new_term = self.ledger["kernel_switch"], fresh.ledger["kernel_switch"]
        ledger = channel_ledger(self, channels, found)
        if old_term.value is not None and new_term.value is not None:
            switch = replace(new_term, value=gather(old_term.value, new_term.value))
        else:
            switch = ErrorTerm.unbounded(
                "the hybrid switch could not be compared on every requested channel grid",
                name="kernel_switch",
            )
        counts = {
            **dict.fromkeys(S2_EVENTS, 0),
            **dict(fresh.record.events),
            "channel rows reused": int(np.count_nonzero(found >= 0)),
            "channel rows computed": len(missing),
        }
        record = uncached_record(
            derived_record(self.record, fresh.record.routes, counts, start)
        )
        agreement = dict(fresh.record.agreement)
        if switch.value is not None:
            agreement["maximum kernel switch / I"] = float(
                np.max(np.asarray(switch.value))
            )
        record = replace(record, agreement=agreement)
        return replace(
            self,
            channels=channels,
            spectra0=gather(self.spectra0, fresh.spectra0),
            spectra2=gather(self.spectra2, fresh.spectra2),
            ledger=_with_switch(ledger, switch),
            record=record,
            key=None,
        )

    def with_order(self, order):
        """Lower the order from stored rows or rebuild with the same assignment."""
        built = self.lattice.order
        if built is None or order <= built:
            return super().with_order(order)
        raised = self._on_grid(self.channels, self.lattice.with_order(order))
        return replace(
            self,
            lattice=raised.lattice,
            columns=raised.columns,
            spectra0=raised.spectra0,
            spectra2=raised.spectra2,
            record=raised.record,
            key=raised.key,
            averaged=raised.averaged,
            specified=raised.specified,
            ledger=_with_switch(self.ledger, raised.ledger["kernel_switch"]),
        )


def _static_columns(averaged, channels, lattice, blocks, run):
    hybrid = averaged.kernel
    index = lattice.index
    shape = (index.block_size, channels.n_ch)
    i = np.zeros(shape)
    p = np.zeros(shape, complex)
    with_v = "V" in blocks
    v = np.zeros(shape) if with_v else None
    records = {}
    events = {"jet cells": lattice.n_cells}
    for branch, kernel in enumerate(hybrid.kernels):
        cells = tuple(
            c for c, assigned in enumerate(hybrid.assignment) if assigned == branch
        )
        if not cells:
            continue
        child = eqx.tree_at(lambda a: a.kernel, averaged, kernel)
        # Use the actual child, without a traced kernel selector in the jet.
        jet = fresh_jet(child, run.fingerprints)
        extra = {}
        made = jet_lattice(
            jet.averaged,
            channels,
            lattice,
            cells=cells,
            with_v=with_v,
            routine=jet.routine,
            events=extra,
        )
        for key, count in extra.items():
            events[key] = events.get(key, 0) + count
        rows = np.concatenate(
            [np.fromiter(index.cell_rows(c), dtype=int) for c in cells]
        )
        i[rows], p[rows] = made[:2]
        if with_v:
            v[rows] = made[2]
        for cell in cells:
            records[cell] = HybridRouteRecord(
                cell,
                "jet",
                mode="hybrid",
                reason=f"hybrid kernel {branch}: {kernel.name}",
                kernel_index=branch,
            )
    return (
        {"I": i, "P": p, "V": v},
        tuple(records[c] for c in range(lattice.n_cells)),
        events,
    )


def build_hybrid_from_averaged(
    averaged,
    channels,
    lattice,
    *,
    rules=ColumnRules(),
    routes="auto",
    s1="lookup",
    stokes=None,
    azimuth_factor=None,
    max_columns=20_000,
    cache=True,
):
    """Build statically grouped child columns and directly measure switch agreement."""
    start = time.perf_counter()
    if not isinstance(averaged.kernel, HybridKernel):
        raise TypeError("build_hybrid_from_averaged expects an averaged HybridKernel")
    check_arguments(rules, cache)
    if not isinstance(routes, str) or routes not in ("auto", "jet"):
        raise RoleError("hybrid cells are built with one static jet provider per child")
    hybrid = averaged.kernel.bind(lattice.cells)
    if fourier_axis(lattice, hybrid.azimuth) is not None:
        raise RoleError(
            "hybrid kernels with AzimuthFourier need a joint angular projection"
        )
    averaged = eqx.tree_at(lambda a: a.kernel, averaged, hybrid)
    blocks = stokes_blocks(hybrid, rules, stokes)
    rules = replace(rules, stokes=blocks)
    factor = check_factor(hybrid, lattice, blocks, azimuth_factor)
    columns = column_map(lattice, hybrid, rules, max_columns=max_columns)
    switch, tested = switch_measurement(averaged, lattice, channels, blocks)
    cache = cache and not averaged.leaf_factors
    run = BuildRun.open(
        averaged, channels, lattice, columns, rules, "hybrid jet", s1, cache
    )
    request = averaged, channels, lattice, factor, rules
    maximum = None if switch.value is None else float(np.max(np.asarray(switch.value)))
    agreement = {"hybrid comparisons": tested, "maximum kernel switch / I": maximum}
    found = run.lookup()
    if found is not None:
        base = _rebound(found, run, request, start)
        defaults = {**dict(base.record.defaults), "switch_tol": hybrid.switch_tol}
        return _hybrid_basis(
            replace(
                base,
                record=replace(base.record, defaults=defaults, agreement=agreement),
            ),
            switch,
        )
    rows, routes, counts = _static_columns(averaged, channels, lattice, blocks, run)
    counts.update(run.finish())
    defaults = {
        "max_columns": int(max_columns),
        "cache": cache,
        "switch_tol": hybrid.switch_tol,
    }
    base = _fresh(
        run, request, columns, (routes, blocks, s1), rows, counts, defaults, start
    )
    record = replace(base.record, agreement=agreement)
    basis = _hybrid_basis(replace(base, record=record), switch)
    run.store(basis)
    return basis
