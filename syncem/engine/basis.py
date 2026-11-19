"""The step-5 spectral basis and its engine entry ``build_from_averaged`` (design/0.8/SPEC.md Sec. 3.4, decisions 149
to 153).

LABEL ``eq: channel basis expansion``. ``build_from_averaged(averaged, channels, lattice, ...)`` checks the Stokes
request, computes the column map and its size check before any provider call, routes every product cell to the
stack, the jet or the core (decision 150), and stores one spin-0 spectrum per ``spin0_base`` row and one spin-2
spectrum per ``spin2_base`` row. ``design`` assembles the columns at S3 (``engine.assemble``), so ``w`` and the
azimuth factor enter no stored spectrum. ``SpectralBasis.build`` with the signature of Sec. 3.4 arrives in step 6
and calls this entry. Until then the class is reached as ``syncem.engine.basis``.

``at(channels)`` gathers the stored rows whose channel row keys match, bitwise and without a provider call. It
computes the other rows through ``channels[missing]`` with the recorded route of each cell. A stack cell that the
new channels refuse takes the jet on every channel, so one provider serves each cell. ``with_order(N')`` at or
below the built order caps each per-cell order of the variables inside the total at ``N'``, sets the total to
``N'``, gathers the spectra of the matching (cell, pair, powers) rows, keeps the parent's routes and calls no
provider. A higher order raises until step 5b-2, and so does a lattice without a total order. In step 5b-1 no basis is cached and every ``key`` is
``None``, and ``cache=True`` records "not cached: no fingerprints before 5b-2".

Units: those of the kernel per unit scaled moment. Array shapes: ``spectra0`` ``(n_ch, n_s0, n0)`` real with the
spin-0 blocks I then V, ``spectra2`` ``(n_ch, n2)`` complex or ``None``, ``design()`` ``(n_ch, n_stokes, n)``.
Not certified: the providers, which the tests compare with the study oracles and with each other (check (a) of
Sec. 8.3), and the column rules, which the caller states until the plan of step 7.
"""

from __future__ import annotations

import dataclasses
import os
import time
from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..lattice.representations import AzimuthHarmonic
from ..records.ledger import Ledger
from ..records.records import BuildRecord, SpecifiedRecord
from ._basis_update import capped_lattice, fill_new, match_channels, order_indices
from ._routes import check_stokes, route_cells
from .assemble import design_array, gather_spectra, select_stokes, stokes_axes
from .average import AveragedKernel, NodeRule
from .columns import ColumnMap, ColumnRules, column_map

__all__ = ["NOT_CACHED", "SpectralBasis", "build_from_averaged", "specified_records"]

NOT_CACHED = "not cached: no fingerprints before 5b-2"
_NO_KEY = "not cached: lattice key is None"
_OFF = "not cached: cache=False"


class SpectralBasis(eqx.Module):
    """Stored spectra, the column map and the build record of one basis (SPEC Sec. 3.4, decision 149).

    ``averaged`` and ``rules`` are the sources that ``at`` and ``with_order`` reuse. ``key`` is ``None`` in step
    5b-1. Not certified: the spectra are provider output, compared with oracles by the tests.
    """

    LABEL: ClassVar[str] = "eq: channel basis expansion"
    lattice: Any
    channels: Any
    columns: ColumnMap = eqx.field(static=True)
    spectra0: Any
    spectra2: Any
    azimuth_factor: Any
    specified: tuple
    record: BuildRecord = eqx.field(static=True)
    ledger: Ledger
    key: str | None = eqx.field(static=True)
    averaged: AveragedKernel
    rules: ColumnRules = eqx.field(static=True)

    def design(self, data: Any = None, *, stokes: Any = None) -> Any:
        """``(n_ch, n_stokes, n)`` real with ``w`` applied. ``stokes`` selects rows of the built Stokes axes."""
        if data is not None:
            raise NotImplementedError(
                "design(data=...) gives whitened rows of the data, which arrive in step 8"
            )
        full = design_array(
            self.spectra0,
            self.spectra2,
            self.columns,
            self.lattice.weights,
            block_names=self.lattice.index.blocks,
            azimuth_factor=self.azimuth_factor,
        )
        return select_stokes(full, stokes_axes(self.columns.stokes), stokes)

    def at(self, channels: Any) -> "SpectralBasis":
        """The same lattice and statistics on ``channels``. Stored rows are gathered, new rows computed."""
        start = time.perf_counter()
        keys = channels.row_keys()
        found, compute = match_channels(self.channels.row_keys(), keys)
        s0 = np.zeros((found.size,) + np.shape(self.spectra0)[1:])
        s2 = (
            None
            if self.spectra2 is None
            else np.zeros((found.size, self.columns.n2), complex)
        )
        hit = found >= 0
        s0[hit] = np.asarray(self.spectra0)[found[hit]]
        if s2 is not None:
            s2[hit] = np.asarray(self.spectra2)[found[hit]]
        counts = {
            "channel rows reused": int(hit.sum()),
            "channel rows computed": len(compute),
        }
        routes = self.record.routes
        if compute:
            routes = fill_new(self, channels, compute, s0, s2, counts)
            again = np.flatnonzero(~hit)
            first = np.array([compute[keys[j]] for j in again], dtype=np.int64)
            s0[again] = s0[first]
            if s2 is not None:
                s2[again] = s2[first]
        record = _derived_record(self.record, routes, counts, start)
        return dataclasses.replace(
            self, channels=channels, spectra0=s0, spectra2=s2, record=record
        )

    def with_order(self, order: int) -> "SpectralBasis":
        """The basis on ``capped_lattice(lattice, order)`` at or below the built order, from the stored spectra."""
        start = time.perf_counter()
        built = self.lattice.order
        if (
            isinstance(order, bool)
            or not isinstance(order, (int, np.integer))
            or order < 0
        ):
            raise ValueError(f"order must be a non-negative int, not {order!r}")
        if built is None:
            raise ValueError(
                "the lattice has no total order: every celled variable lies outside the total (depth), so "
                "with_order has no order to lower"
            )
        if order > built:
            raise ValueError(
                f"with_order({order}) is above the built order {built}. Step 5b-2 computes the missing exponents "
                f"only. Until then build again at order {order}"
            )
        lattice = capped_lattice(self.lattice, int(order))
        max_columns = dict(self.record.defaults).get("max_columns", 20_000)
        columns = column_map(
            lattice, self.averaged.kernel, self.rules, max_columns=max_columns
        )
        idx0, idx2 = order_indices(self, lattice, columns)
        s0 = np.asarray(self.spectra0)[:, :, idx0]
        s2 = None if self.spectra2 is None else np.asarray(self.spectra2)[:, idx2]
        counts = {
            "channel rows reused": int(np.shape(s0)[0]),
            "channel rows computed": 0,
        }
        record = _derived_record(self.record, self.record.routes, counts, start)
        record = dataclasses.replace(
            record,
            sizes=_sizes(columns),
            keys=_keys(columns, lattice),
            notes=record.notes + (f"routes of the order-{built} build",),
        )
        return dataclasses.replace(
            self,
            lattice=lattice,
            columns=columns,
            spectra0=s0,
            spectra2=s2,
            record=record,
        )


def specified_records(averaged: AveragedKernel) -> tuple[SpecifiedRecord, ...]:
    """One ``SpecifiedRecord`` per averaged or fixed variable. Fingerprints are ``None`` until step 5b-2."""
    out = []
    for name, rule in averaged.averaged:
        if isinstance(rule, NodeRule):
            out.append(SpecifiedRecord(name, "NodeRule", parents=rule.parents or ()))
        else:
            out.append(SpecifiedRecord(name, "callable"))
    out += [SpecifiedRecord(name, "fixed") for name, _ in averaged.fixed]
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
    """The basis of ``averaged`` on ``channels`` and ``lattice`` (decision 149). ``cache`` takes effect in 5b-2."""
    start = time.perf_counter()
    _check_arguments(rules, cache)
    kernel = averaged.kernel
    blocks = _blocks(kernel, rules, stokes)
    rules = dataclasses.replace(rules, stokes=blocks)
    factor = _factor(kernel, lattice, blocks, azimuth_factor)
    columns = column_map(lattice, kernel, rules, max_columns=max_columns)
    counts: dict = {}
    I, P, V, records = route_cells(
        averaged, channels, lattice, routes, s1=s1, with_v="V" in blocks, events=counts
    )
    spectra0, spectra2 = gather_spectra(I, P, V, columns)
    counts.update(
        {"channel rows reused": 0, "channel rows computed": int(channels.n_ch)}
    )
    defaults = {"max_columns": int(max_columns), "cache": cache}
    timing = {"build": time.perf_counter() - start}
    record = _build_record(
        records, blocks, s1, columns, lattice, counts, timing, defaults
    )
    return SpectralBasis(
        lattice=lattice,
        channels=channels,
        columns=columns,
        spectra0=spectra0,
        spectra2=spectra2,
        azimuth_factor=factor,
        specified=specified_records(averaged),
        record=record,
        ledger=Ledger(),
        key=None,
        averaged=averaged,
        rules=rules,
    )


# ------------------------------------------------------------------ build helpers


def _build_record(
    routes: tuple,
    blocks: tuple[str, ...],
    s1: str,
    columns: ColumnMap,
    lattice: Any,
    counts: dict,
    timings: dict,
    defaults: dict,
) -> BuildRecord:
    """The record of a fresh build. Every basis of step 5b-1 is uncached, and the note says why."""
    cache = defaults["cache"]
    note = _NO_KEY if lattice.key is None else NOT_CACHED if cache else _OFF
    return BuildRecord(
        routes=routes,
        stokes=stokes_axes(blocks),
        s1=s1,
        sizes=_sizes(columns),
        events=counts,
        timings=timings,
        keys=_keys(columns, lattice),
        defaults=defaults,
        notes=(note,),
    )


def _check_arguments(rules: Any, cache: Any) -> None:
    if not isinstance(cache, bool):
        raise ValueError(f"cache must be a bool, not {cache!r}")
    if not isinstance(rules, ColumnRules):
        raise TypeError(f"rules must be ColumnRules, not {type(rules).__name__}")


def _blocks(kernel: Any, rules: ColumnRules, stokes: Any) -> tuple[str, ...]:
    if stokes is not None and rules.stokes is not None:
        if check_stokes(kernel, stokes) != check_stokes(kernel, rules.stokes):
            raise ValueError(
                f"stokes={stokes!r} differs from rules.stokes={rules.stokes!r}"
            )
    request = stokes if stokes is not None else rules.stokes
    return check_stokes(kernel, kernel.stokes if request is None else request)


def _factor(kernel: Any, lattice: Any, blocks: tuple[str, ...], factor: Any) -> Any:
    """The R5 (i) stand-in: required only for P with an azimuth that no ``AzimuthHarmonic`` resolves."""
    harmonic = any(isinstance(r, AzimuthHarmonic) for _, r in lattice.reps)
    if factor is None:
        if "P" in blocks and kernel.azimuth is not None and not harmonic:
            raise ValueError(
                f"P needs the factor of the azimuth {kernel.azimuth}: pass azimuth_factor (route R5 (i)) or "
                f"resolve {kernel.azimuth} with AzimuthHarmonic"
            )
        return None
    if "P" not in blocks or kernel.azimuth is None or harmonic:
        raise ValueError(
            "azimuth_factor multiplies P for an azimuth the lattice leaves unresolved. Here P is not requested, "
            "the kernel has no azimuth, or AzimuthHarmonic resolves it"
        )
    value = jnp.asarray(factor, dtype=jnp.complex128)
    if value.shape != ():
        raise ValueError(
            f"azimuth_factor must be a complex scalar, not shape {value.shape}"
        )
    return value


def _sizes(columns: ColumnMap) -> dict:
    return {
        "n_rows": columns.n_rows,
        "n": columns.n,
        "n0": columns.n0,
        "n2": columns.n2,
    }


def _keys(columns: ColumnMap, lattice: Any) -> dict:
    return {
        "column map": columns.key,
        "index": lattice.index.key,
        "lattice": lattice.key,
    }


def _load() -> tuple[float, ...]:
    try:
        return tuple(float(v) for v in os.getloadavg())
    except (AttributeError, OSError):
        return ()


def _derived_record(
    record: BuildRecord, routes: tuple, counts: dict, start: float
) -> BuildRecord:
    providers = [("stack cells", 0), ("jet cells", 0), ("core cells", 0)]
    return dataclasses.replace(
        record,
        routes=routes,
        events={**dict(providers), **counts},
        timings={"build": time.perf_counter() - start},
        load=_load(),
    )
