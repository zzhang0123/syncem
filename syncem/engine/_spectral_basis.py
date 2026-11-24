"""Stored spectra and operations on one lattice (SPEC Sec. 3.4)."""

from __future__ import annotations

import dataclasses
import time
from typing import Any, ClassVar

import equinox as eqx
import numpy as np

from ..records.ledger import ErrorTerm, Ledger
from ..records.records import BuildRecord
from ..values.errors import RoleError
from ._basis_ledger import channel_ledger, coverage_ledger
from ._basis_update import (
    S2_EVENTS,
    capped_lattice,
    derived_record,
    fill_new,
    match_channels,
    order_indices,
    record_keys,
    record_sizes,
    uncached_record,
)
from .assemble import design_array, select_stokes, selected_axes, stokes_axes
from .average import AveragedKernel
from .columns import ColumnMap, ColumnRules, column_map


class SpectralBasis(eqx.Module):
    """Stored spectra, the column map and the build record of one basis (SPEC Sec. 3.4, decision 149).

    ``averaged`` and ``rules`` are the sources that ``at`` and ``with_order`` reuse. ``key`` is the design key, or
    ``None`` when the basis is not cacheable. Not certified: the spectra are provider output, compared with
    oracles by the tests.
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

    forward_population: Any = None
    forward_phase: Any = None
    forward_defaults: tuple = eqx.field(static=True, default=())
    forward_depth_tol: float | None = eqx.field(static=True, default=None)
    forward_average: tuple[str, ...] = eqx.field(static=True, default=())
    structural_plan: Any = eqx.field(static=True, default=None)
    forward_accuracy: Any = eqx.field(static=True, default=None)

    def result_plan(self, record=None):
        record = self.record if record is None else record
        if self.structural_plan is None:
            return record
        return dataclasses.replace(self.structural_plan, build=record)

    @property
    def plan(self):
        return self.result_plan()

    @property
    def reference_cases(self):
        """Measure the standard reference populations for the current response grid."""
        from ..front._reference_cases import reference_cases

        return reference_cases(self)

    @property
    def remainder(self):
        """Sampled next Taylor coefficient in channel Stokes per unit column."""
        from ._remainder import sampled_remainder

        return sampled_remainder(self)

    @classmethod
    def from_single_reference(cls, old_basis):
        """Import a frozen model basis in unit-area, physical-column conventions."""
        from .single_reference import from_single_reference

        return from_single_reference(old_basis)

    def save(self, path):
        """Save the numeric basis and its records in a versioned archive."""
        from .basis_io import save_basis

        save_basis(self, path)

    @classmethod
    def load(cls, path, *, averaged=None, registry=None):
        """Load saved columns and optionally verify a live averaged kernel."""
        from .basis_io import load_basis

        return load_basis(path, averaged=averaged, registry=registry)

    @classmethod
    def build(
        cls,
        channels,
        lattice,
        *,
        specified=None,
        symmetries=(),
        kernel=None,
        phase=None,
        stokes=None,
        check=True,
        max_columns=20_000,
        cache=True,
        depth_tol=1e-3,
    ):
        """Build from declared averaged statistics and a lattice."""
        from ..front._predictions import build_specified

        return build_specified(
            channels,
            lattice,
            specified=specified,
            symmetries=symmetries,
            kernel=kernel,
            phase=phase,
            stokes=stokes,
            check=check,
            max_columns=max_columns,
            cache=cache,
            depth_tol=depth_tol,
        )

    def moments(self, population, *, allow_mismatch=False):
        """Pair the declared measure with the rows of this basis."""
        from ..front._predictions import basis_moments

        return basis_moments(self, population, allow_mismatch=allow_mismatch)

    def predict(self, population_or_moments, *, check=True, allow_mismatch=False):
        """Predict channel Stokes on the channels of this basis."""
        from ..front import predict

        return predict(
            population_or_moments,
            self.channels,
            basis=self,
            check=check,
            allow_mismatch=allow_mismatch,
        )

    def direct(self, population, *, within="support"):
        """Integrate using separate panels over the support or the cell boxes."""
        from ..front._direct import direct

        return direct(self, population, within=within)

    def fit(self, population, data, *, allow_mismatch=False, **policies):
        """Fit on these columns after matching the observation channels."""
        from ..front._fitting import fit

        return fit(
            population, data, basis=self, allow_mismatch=allow_mismatch, **policies
        )

    def _observed_design(self, data, stokes):
        if not hasattr(data, "whitened") or not hasattr(data, "channels"):
            raise TypeError("design(data=...) needs StokesData")
        basis = self if self.channels is data.channels else self.at(data.channels)
        axes = selected_axes(basis.record.stokes, stokes)
        return data.whitened(basis.design(stokes=stokes), components=axes).design

    def design(self, data: Any = None, *, stokes: Any = None) -> Any:
        """Return weighted spectral columns or their whitened observation rows."""
        if data is not None:
            return self._observed_design(data, stokes)
        full = design_array(
            self.spectra0,
            self.spectra2,
            self.columns,
            self.lattice.weights,
            block_names=self.lattice.index.blocks,
            azimuth_factor=self.azimuth_factor,
        )
        return select_stokes(full, stokes_axes(self.columns.stokes), stokes)

    def _match_channel_rows(self, channels, keys):
        return match_channels(self.channels.row_keys(), keys)

    def at(self, channels: Any) -> "SpectralBasis":
        """The same lattice and statistics on ``channels``. Stored rows are gathered, new rows computed."""
        start = time.perf_counter()
        keys = channels.row_keys()
        found, compute = self._match_channel_rows(channels, keys)
        ledger = channel_ledger(self, channels, found)
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
            **dict.fromkeys(S2_EVENTS, 0),
            "channel rows reused": int(hit.sum()),
            "channel rows computed": len(compute),
        }
        routes, notes = self.record.routes, ()
        if compute:
            routes, notes = fill_new(self, channels, compute, s0, s2, counts)
            again = np.flatnonzero(~hit)
            first = np.array([compute[keys[j]] for j in again], dtype=np.int64)
            s0[again] = s0[first]
            if s2 is not None:
                s2[again] = s2[first]
        record = uncached_record(derived_record(self.record, routes, counts, start))
        record = record.with_notes(*notes)
        if compute and record.agreement:
            record = dataclasses.replace(record, agreement={}).with_notes(
                "Provider agreement has no measurement on the new channel grid."
            )
        return dataclasses.replace(
            self,
            channels=channels,
            spectra0=s0,
            spectra2=s2,
            record=record,
            ledger=ledger,
            key=None,
        )

    def _selection_spectra(self, idx0, idx2):
        return {
            "spectra0": np.asarray(self.spectra0)[:, :, idx0],
            "spectra2": (
                None if self.spectra2 is None else np.asarray(self.spectra2)[:, idx2]
            ),
        }

    def select(self, **cells):
        """Use the stored columns of selected product cells."""
        chosen = self.lattice.index.cells_where(**cells)
        if chosen.size == self.lattice.n_cells:
            return self
        if self.spectra0 is None or getattr(self.averaged.kernel, "stored_only", False):
            raise RoleError("cell selection needs a live spectral column map")
        start = time.perf_counter()
        lattice = self.lattice.select(**cells)
        averaged = self.averaged
        kernel = averaged.kernel
        from ..base.hybrid import HybridKernel

        if isinstance(kernel, HybridKernel):
            kernel = kernel.bind(lattice.cells)
            averaged = eqx.tree_at(lambda adapter: adapter.kernel, averaged, kernel)
        columns = column_map(
            lattice,
            kernel,
            self.rules,
            max_columns=dict(self.record.defaults).get("max_columns", 20_000),
        )
        idx0, idx2 = order_indices(self, lattice, columns, cells=chosen)
        old_routes = {part.cell: part for part in self.record.routes}
        routes = tuple(
            dataclasses.replace(old_routes[int(c)], cell=i)
            for i, c in enumerate(chosen)
            if int(c) in old_routes
        )
        record = derived_record(
            self.record,
            routes,
            {"channel rows reused": self.channels.n_ch, "channel rows computed": 0},
            start,
        )
        record = dataclasses.replace(
            uncached_record(record, record_keys(columns, lattice)),
            sizes=record_sizes(columns),
        )
        count = self.lattice.n_cells - chosen.size
        omitted = ErrorTerm.not_applicable(
            f"The selected basis omits {count} original product cells.",
            name="cell_selection",
        )
        terms = tuple(t for t in self.ledger.terms if t.name != omitted.name)
        ledger = Ledger((*terms, omitted), assumptions=self.ledger.assumptions)
        selected = dataclasses.replace(
            self,
            lattice=lattice,
            columns=columns,
            averaged=averaged,
            record=record,
            key=None,
            ledger=ledger,
            **self._selection_spectra(idx0, idx2),
        )
        return dataclasses.replace(selected, ledger=coverage_ledger(selected))

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
            from .basis import _raised

            return _raised(self, int(order))
        lattice = capped_lattice(self.lattice, int(order))
        max_columns = dict(self.record.defaults).get("max_columns", 20_000)
        columns = column_map(
            lattice, self.averaged.kernel, self.rules, max_columns=max_columns
        )
        idx0, idx2 = order_indices(self, lattice, columns)
        s0 = np.asarray(self.spectra0)[:, :, idx0]
        s2 = None if self.spectra2 is None else np.asarray(self.spectra2)[:, idx2]
        counts = {
            **dict.fromkeys(S2_EVENTS, 0),
            "channel rows reused": int(np.shape(s0)[0]),
            "channel rows computed": 0,
        }
        record = derived_record(self.record, self.record.routes, counts, start)
        record = dataclasses.replace(
            uncached_record(record, record_keys(columns, lattice)),
            sizes=record_sizes(columns),
        )
        record = record.with_notes(f"routes of the order-{built} build")
        return dataclasses.replace(
            self,
            lattice=lattice,
            columns=columns,
            spectra0=s0,
            spectra2=s2,
            record=record,
            key=None,
        )
