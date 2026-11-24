"""Cell columns for the two physical regimes of the 0.9 response."""

from dataclasses import dataclass, replace
from itertools import product
from math import factorial
from time import perf_counter
from typing import Any

import equinox as eqx
import numpy as np
from numpy.polynomial import chebyshev

from ..base._fingerprint import digest
from ..base.channels import Channels
from ..base.variables import CRITICAL_FREQUENCY, SYNCHROTRON_VARIABLES
from ..constants import C_CGS, C_SI_M, E_ESU, M_E
from ..distributions.screens import GaussianScreen
from ..model.channels import Channels as ModelChannels
from ..model.phase import phase_coordinate
from ..records.ledger import ErrorTerm, Ledger
from ..records.records import BuildRecord
from .assemble import design_array, gather_spectra
from ._basis_update import match_channels
from ._stage_memo import StageMemo
from .cache import default_cache
from .columns import ColumnRules, column_map
from .primitive import quantised_span

AMPLITUDE_PER_GAUSS = np.sqrt(3.0) * E_ESU**3 / (4 * np.pi * M_E * C_CGS**2)


def _analytic_response_keys(channels, *, host_bump=False):
    """Identify the known response formulas independently of quadrature."""
    family = channels.family
    if not host_bump and type(channels) not in (Channels, ModelChannels):
        return (None,) * channels.n_ch
    if family == "callable":
        return (None,) * channels.n_ch
    geometry = tuple(
        np.asarray(values)
        for values in (channels.centres_hz, channels.widths_hz, channels.support)
    )
    labels = (
        "line response",
        family,
        channels.normalisation,
        channels.taper,
        channels.support_sigma,
    )
    if family == "table":
        grid, samples = channels.table
        labels = (*labels, grid)
        geometry = (*geometry, np.asarray(samples))
    if family == "basis":
        reference = _analytic_response_keys(channels.reference_channels)
        if None in reference:
            return (None,) * channels.n_ch
        labels = (*labels, reference)
        geometry = (*geometry, np.asarray(channels.coefficients))
    return tuple(digest((*labels, *row)) for row in zip(*geometry, strict=True))


def _harmonic_response_keys(channels):
    if type(channels) is Channels or channels.family == "bump":
        return _analytic_response_keys(channels, host_bump=channels.family == "bump")
    # The integer driver evaluates select(j), with line limits taken from
    # the original support. This also identifies rows of instrument unions.
    keys = []
    for j, support in enumerate(np.asarray(channels.support)):
        single = channels.select(j)
        keys.append(
            _analytic_response_keys(single)[0]
            if single.n_ch == 1 and np.array_equal(single.support[0], support)
            else None
        )
    return tuple(keys)


def regime_channel_keys(channels, *, harmonic=True, continuum=True):
    """Return ordered response identities, with None for opaque line responses."""
    h = _harmonic_response_keys(channels) if harmonic else ("",) * channels.n_ch
    c = channels.row_keys() if continuum else ("",) * channels.n_ch
    support = np.asarray(channels.support)
    return tuple(
        None if response is None else (response, rule, tuple(bounds))
        for response, rule, bounds in zip(h, c, support, strict=True)
    )


def regime_channel_indices(stored, wanted, *, harmonic=True, continuum=True):
    """Return the stored row index, or -1 for a new or opaque response."""
    options = dict(harmonic=harmonic, continuum=continuum)
    new = regime_channel_keys(wanted, **options)
    found, _ = match_channels(regime_channel_keys(stored, **options), new)
    return np.where([key is not None for key in new], found, -1)


class _Geometry(eqx.Module):
    name: str = eqx.field(static=True)
    stokes = ("I", "P")
    depth = "depth"
    azimuth = "phi"
    capabilities = ()

    @property
    def variables(self):
        if self.name == "harmonic":
            return tuple(SYNCHROTRON_VARIABLES.values())
        return (
            CRITICAL_FREQUENCY,
            SYNCHROTRON_VARIABLES["depth"],
            SYNCHROTRON_VARIABLES["phi"],
        )


@dataclass(frozen=True)
class _Axis:
    centre: float
    radius: float
    ref: float
    half: float
    chart: str

    def point(self, u):
        value = self.centre + self.radius * u
        return np.exp(value) if self.chart == "log" else value

    def coefficients(self, degree, count):
        coordinate = (self.ref - self.centre) / self.radius
        polynomials = chebyshev.chebder(np.eye(count), m=degree, axis=0)
        return (
            chebyshev.chebval(coordinate, polynomials)
            @ _cheb_rule(count)[1]
            * (self.half / self.radius) ** degree
            / factorial(degree)
        )


def _cheb_rule(count):
    cache = default_cache()
    key = f"regime-chebyshev:{count}"
    rule = cache.get("memo", key)
    if rule is not None:
        return rule
    nodes = np.cos(np.pi * np.arange(count) / (count - 1))
    return cache.put(
        "memo", key, (nodes, np.linalg.inv(chebyshev.chebvander(nodes, count - 1)))
    )


def polynomial_columns(table, cell, powers, value, extra, *, dimensions=None):
    """Evaluate a local interpolant into the requested scaled Taylor rows."""
    axes, counts = [], []
    dimensions = range(len(table.variables)) if dimensions is None else dimensions
    names = [table.variables[i] for i in dimensions]
    for i in dimensions:
        chart = table.chart[cell][i]
        lo, hi, ref = table.lo[cell, i], table.hi[cell, i], table.reference[cell, i]
        if chart == "log":
            lo, hi, ref = np.log([lo, hi, ref])
        axes.append(
            _Axis((lo + hi) / 2, (hi - lo) / 2, ref, table.half_width[cell, i], chart)
        )
        order = np.max(powers[:, i])
        counts.append(int(order + extra) if order or extra > 3 else 3)
    grids = [axis.point(_cheb_rule(n)[0]) for axis, n in zip(axes, counts, strict=True)]
    values = [value(dict(zip(names, point, strict=True))) for point in product(*grids)]
    table_values = np.asarray(values).reshape(*counts, *np.shape(values[0]))
    rows = []
    for power in powers:
        row = table_values
        for axis, degree, count in zip(
            axes, power[list(dimensions)], counts, strict=True
        ):
            row = np.tensordot(axis.coefficients(int(degree), count), row, axes=1)
        rows.append(row)
    return np.asarray(rows)


def continuum_columns(channels, lattice, phase, field):
    """Differentiate F and G in ln(nu_c), with node-wise phase derivatives."""
    nodes = np.asarray(channels.nodes)
    weights = np.asarray(channels.response_weights())
    index, table = lattice.index, lattice.cells
    n_axis = table.variables.index("nu_c")
    depth_axis = table.variables.index("depth") if "depth" in table.variables else None
    order = int(np.max(index.arrays.powers[:, n_axis]))
    log_nu = np.log(nodes)
    lnx = log_nu[None] - np.log(table.reference[:, n_axis, None, None])
    lo, hi = quantised_span(float(lnx.min()) - 1e-6, float(lnx.max()) + 1e-6)
    memo = StageMemo(default_cache())
    try:
        primitive = memo.primitive((lo, hi), order)
    finally:
        memo.close()
    rows = np.empty((index.block_size, channels.n_ch, 2), dtype=complex)
    tau = np.broadcast_to(np.asarray(phase.tau(nodes)), nodes.shape)
    for cell in range(table.n_cells):
        span = index.cell_rows(cell)
        powers = index.arrays.powers[span]
        r = powers[:, n_axis]
        u = lnx[cell]
        amp = AMPLITUDE_PER_GAUSS * field * weights
        log_f, f = primitive.lookup(u, "F", order)
        log_g, g = primitive.lookup(u, "G", order)
        scale = table.half_width[cell, n_axis] ** r / [factorial(int(p)) for p in r]
        intensity = (
            amp[..., None] * np.exp(u - np.exp(u) + log_f)[..., None] * f[..., r]
        )
        pol = -amp[..., None] * np.exp(u - np.exp(u) + log_g)[..., None] * g[..., r]
        depth = 0.0 if depth_axis is None else table.reference[cell, depth_axis]
        rotation = np.asarray(phase.weights(nodes, depth))[..., None]
        if depth_axis is not None:
            s = powers[:, depth_axis]
            rotation = (
                rotation
                * (1j * tau[..., None] * table.half_width[cell, depth_axis]) ** s
            )
            rotation = rotation / [factorial(int(p)) for p in s]
            intensity[..., s > 0] = 0.0
        rows[span, :, 0] = (intensity.sum(axis=1) * scale).T
        rows[span, :, 1] = ((pol * rotation).sum(axis=1) * scale).T
    return rows


def _phase_derivative(phase, depth, half, power):
    screen = getattr(phase.screen, "__self__", None)
    gaussian = isinstance(screen, GaussianScreen)
    standard = phase.tau is phase_coordinate and (phase.screen is None or gaussian)
    mean = float(screen.mean) if standard and gaussian else 0.0
    sigma = float(screen.sigma) if standard and gaussian else 0.0

    def weights(nu):
        if standard:
            tau = 2 * (C_SI_M / nu) ** 2
            base = np.exp(1j * tau * (depth + mean) - 0.5 * (tau * sigma) ** 2)
        else:
            base = np.asarray(phase.weights(nu, depth))
            tau = np.asarray(phase.tau(nu)) if power else None
        if power:
            base = base * (1j * tau * half) ** power / factorial(power)
        return base

    return weights


def harmonic_columns(
    channels, lattice, phase, rule, *, extra, refine, conditional_screen=None
):
    """Interpolate the host harmonic kernel within each physical cell."""
    from ._regime_harmonic import harmonic_channel, prepare_harmonic_channels

    table, index = lattice.cells, lattice.index
    degrees = dict(index.legendre)
    pitch_degrees = tuple(range(degrees["mu"] + 1)) if "mu" in degrees else None
    eta_order = degrees.get("eta")
    fields = rule.field_points(refine)
    inclinations = rule.inclination_points(eta_order, refine)
    pitches = rule.pitch_points(refine)
    eta_size = 1 if eta_order is None else eta_order + 1
    mu_size = 1 if pitch_degrees is None else len(pitch_degrees)
    rows = np.empty((index.block_size, channels.n_ch, 2), dtype=complex)
    prepared_depth = {}
    depth_axis = table.variables.index("depth") if "depth" in table.variables else None
    dimensions = tuple(i for i in range(len(table.variables)) if i != depth_axis)

    def value(point, depth_params):
        depth, half, power = depth_params
        if depth_params not in prepared_depth:
            prepared_depth[depth_params] = prepare_harmonic_channels(
                channels,
                phase=_phase_derivative(phase, depth, half, power),
                frequency_nodes=rule.frequency_nodes * refine,
            )
        prepared = prepared_depth[depth_params]
        result = np.zeros((mu_size, eta_size, channels.n_ch, 2), dtype=complex)
        for b, bw in fields if "B" not in point else ((point["B"], 1.0),):
            for eta, ew in inclinations:
                rotation = (
                    None
                    if conditional_screen is None
                    else conditional_screen.at_geometry(point["gamma"], b, eta)
                )
                for pitch, pw in pitches:
                    modes = harmonic_channel(
                        channels,
                        point["gamma"],
                        b,
                        eta,
                        pitch,
                        angular_nodes=rule.angular_nodes * refine,
                        degrees=pitch_degrees,
                        pitch_support=rule.pitch_support,
                        pitch_breaks=rule.pitch_breaks,
                        prepared=prepared,
                        phase_at_pitch=rotation,
                    ).reshape(mu_size, channels.n_ch, 2)
                    result += (bw * pw) * modes[:, None] * ew[None, :, None, None]
        if power:
            result[..., 0] = 0.0
        return result

    for cell in range(table.n_cells):
        span = index.cell_rows(cell)
        powers = index.arrays.powers[span]
        depth_powers = (
            np.zeros(len(powers), dtype=int)
            if depth_axis is None
            else powers[:, depth_axis]
        )
        for order in np.unique(depth_powers):
            active = depth_powers == order
            exponents, inverse = np.unique(powers[active], axis=0, return_inverse=True)
            params = (
                (0.0, 0.0, 0)
                if depth_axis is None
                else (
                    table.reference[cell, depth_axis],
                    table.half_width[cell, depth_axis],
                    int(order),
                )
            )
            coeff = polynomial_columns(
                table,
                cell,
                exponents,
                lambda point: value(point, params),
                extra,
                dimensions=dimensions,
            )
            degrees = index.arrays.degrees[span][active].T
            angles = dict(zip(dict(index.legendre), degrees, strict=True))
            rows[np.asarray(span)[active]] = coeff[
                inverse, angles.get("mu", 0), angles.get("eta", 0)
            ]
    return rows


class RegimePart(eqx.Module):
    """A stored branch with a provider for building new channels."""

    lattice: Any
    channels: Any
    columns: Any = eqx.field(static=True)
    spectra0: Any
    spectra2: Any
    record: BuildRecord = eqx.field(static=True)
    ledger: Ledger
    rebuild: Any = eqx.field(static=True)

    @property
    def source_column_identifiable(self):
        return "nu_c" not in self.lattice.cells.variables

    @property
    def plan(self):
        return self.record

    def design(self, data=None):
        """Return stored basis columns in channel and Stokes order."""
        if data is not None:
            basis = self.at(data.channels)
            return data.whitened(basis.design(), components=self.record.stokes).design
        return design_array(
            self.spectra0,
            self.spectra2,
            self.columns,
            self.lattice.weights,
            block_names=self.lattice.index.blocks,
            azimuth_factor=None,
        )

    def at(self, channels):
        """Select stored channels or build this branch on new responses."""
        if channels is self.channels:
            return self
        harmonic = self.source_column_identifiable
        indices = regime_channel_indices(
            self.channels, channels, harmonic=harmonic, continuum=not harmonic
        )
        if np.array_equal(indices, np.arange(self.channels.n_ch)):
            return replace(self, channels=channels)
        if np.all(indices >= 0):
            terms = tuple(
                (
                    replace(term, value=term.value[indices])
                    if term.value is not None and np.ndim(term.value) > 0
                    else term
                )
                for term in self.ledger.terms
            )
            return replace(
                self,
                channels=channels,
                spectra0=self.spectra0[indices],
                spectra2=self.spectra2[indices],
                ledger=Ledger(terms, assumptions=self.ledger.assumptions),
            )
        return self.rebuild(channels)


def build_part(channels, lattice, provider, name, refine, *, max_columns, rebuild):
    start = perf_counter()
    geometry = _Geometry(name)
    rules = ColumnRules(depth_free_spin0=True)
    columns = column_map(lattice, geometry, rules, max_columns=max_columns)
    coarse = provider(1)
    rows = provider(2) if refine else coarse
    i, p = gather_spectra(rows[..., 0].real, rows[..., 1], None, columns)
    terms = [
        ErrorTerm.unbounded("no certified population error", name="physical_kernel")
    ]
    if name == "harmonic":
        terms.append(
            ErrorTerm.unbounded(
                "real-order evaluation and line-sum approximation",
                name="harmonic_numerical",
            )
        )
        if refine:
            difference = np.max(np.abs(rows - coarse), axis=0)
            terms.append(
                ErrorTerm(
                    np.column_stack(
                        [difference[:, 0], difference[:, 1], difference[:, 1]]
                    ),
                    "estimate",
                    "maximum change of an unweighted response column under rule refinement",
                    name="harmonic_column_refinement",
                    relative_to="unweighted cell coefficients",
                )
            )
    record = BuildRecord(
        routes=(),
        stokes=("I", "Q", "U"),
        s1="lookup",
        sizes=dict(n_rows=lattice.n, n=columns.n, n0=columns.n0, n2=columns.n2),
        keys=dict(lattice=lattice.key, columns=columns.key),
        timings=(("build_seconds", perf_counter() - start),),
        notes=(name,),
    )
    return RegimePart(
        lattice,
        channels,
        columns,
        i,
        p,
        record,
        Ledger(tuple(terms)),
        rebuild,
    )
