"""Cost policies and provider checks for the specified forward route."""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np

from ..engine.provider_fourier import fourier_axis, sample_count
from ..engine._routes import route_cells
from ..engine.assemble import gather_spectra
from ..engine.provider_table import TableRefused, build_cell_table, check_tables


@dataclass(frozen=True)
class AccuracyCheck:
    """Optional direct check budget and an explicit request to compare providers."""

    max_seconds: float = 5.0
    providers: bool = False
    own_peak_tol: float | None = None

    def __post_init__(self):
        if not math.isfinite(self.max_seconds) or self.max_seconds < 0:
            raise ValueError("max_seconds must be finite and nonnegative")
        if self.own_peak_tol is not None and (
            not math.isfinite(self.own_peak_tol) or self.own_peak_tol < 0
        ):
            raise ValueError("own_peak_tol must be finite and nonnegative")


def check_policy(check):
    """Normalize the public check argument."""
    if check is True:
        return AccuracyCheck()
    if check is False:
        return None
    if not isinstance(check, AccuracyCheck):
        raise TypeError("check must be a bool or an AccuracyCheck")
    return check


def estimate_direct(basis):
    """Return a coarse wall-time estimate for a specified population integral."""
    averaged, lattice, channels = basis.averaged, basis.lattice, basis.channels
    initial = dict(averaged.fixed)
    initial.update(zip(lattice.cells.variables, lattice.cells.reference[0]))
    phi = averaged.kernel.azimuth
    fourier = fourier_axis(lattice, phi)
    if fourier is not None:
        initial[phi] = 0.0
    n_inner = 1
    for name, _, count in averaged.legendre:
        initial[name] = 0.0
        n_inner *= count
    for name, rule in averaged.averaged:
        nodes, _ = rule(initial)
        initial[name] = nodes[len(nodes) // 2]
        n_inner *= len(nodes)
    n_panel_terms = lattice.n_cells * 24 ** len(lattice.cells.variables)
    if fourier is not None:
        n_panel_terms *= fourier.moment_nodes * (1 + sample_count(fourier.m_max))
    n_kernel_terms = channels.n_ch * channels.n_nu * n_inner * n_panel_terms
    # This is an operation-count estimate for the continuum quadrature and
    # does not assume a measured bandwidth or compile time for the user's host.
    return float(n_kernel_terms * 1e-8)


def _channel_scale(values):
    return np.max(np.abs(values), axis=tuple(range(1, values.ndim)), keepdims=True)


def _fraction(gap, scale):
    return float(
        np.max(
            np.divide(
                gap,
                scale,
                out=np.where(gap == 0, 0.0, np.inf),
                where=scale != 0,
            )
        )
    )


def _probe_jet(basis, cells, policy):
    wanted = basis.columns.stokes
    values = route_cells(
        basis.averaged,
        basis.channels,
        basis.lattice,
        "jet",
        with_v="V" in wanted,
        cells=cells,
    )
    ref = gather_spectra(*values[:3], basis.columns)
    stored = basis.spectra0, basis.spectra2
    index = basis.lattice.index.arrays.cell
    keep = (
        np.isin(index[basis.columns.spin0_base], cells),
        np.isin(index[basis.columns.spin2_base], cells),
    )
    peak_gap = 0.0
    differences = []
    for current, exact, select in zip(stored, ref, keep):
        if current is None or not np.any(select):
            continue
        want = np.asarray(exact)[..., select]
        error = np.abs(np.asarray(current)[..., select] - want)
        scale = _channel_scale(want)
        differences.append((error, scale))
        for cell in cells:
            row = np.isin(
                index[
                    (
                        basis.columns.spin2_base
                        if np.asarray(current).ndim == 2
                        else basis.columns.spin0_base
                    )
                ][select],
                cell,
            )
            if np.any(row):
                peak_gap = max(
                    peak_gap, _fraction(error[..., row], _channel_scale(want[..., row]))
                )
    worst = max((_fraction(error, scale) for error, scale in differences), default=0.0)
    if worst > 1e-10:
        raise ValueError(f"column providers differ by {worst:g} of the channel maximum")
    if policy.own_peak_tol is not None and peak_gap > policy.own_peak_tol:
        raise ValueError(f"column providers differ by {peak_gap:g} of the cell peak")
    if peak_gap > 1e-8:
        warnings.warn(
            f"column providers differ by {peak_gap:g} of the cell peak",
            RuntimeWarning,
            stacklevel=3,
        )
    return worst, peak_gap


def provider_checks(basis, policy):
    """Check sampled derivative columns and the independent table route."""
    if not policy.providers:
        return {}, ()
    probe = (
        basis.with_order(2)
        if basis.lattice.order and basis.lattice.order > 2
        else basis
    )
    n = probe.lattice.n_cells
    cells = tuple(dict.fromkeys((0, n // 2, n - 1)))
    record, notes = {}, []
    if any(probe.record.routes[cell].provider != "jet" for cell in cells):
        worst, peak = _probe_jet(probe, cells, policy)
        record.update(provider_channel_max=worst, provider_cell_peak=peak)
    try:
        tables = tuple(
            build_cell_table(probe.averaged, probe.channels, probe.lattice, c)
            for c in cells
        )
        agreement = check_tables(probe.averaged, probe.channels, probe.lattice, tables)
        record["reference_table"] = agreement.relative
        if not agreement.passed:
            notes.append("reference table check failed. Use direct kernel calls")
    except TableRefused as err:
        notes.append(f"reference table unavailable. Use direct kernel calls. {err}")
    return record, tuple(notes)
