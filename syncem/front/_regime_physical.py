"""Fit physical energy cells through the two-regime paper response."""

from dataclasses import replace
from math import comb
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..base.continuum import ContinuumKernel
from ..base.variables import SYNCHROTRON_VARIABLES
from ..distributions import AzimuthFactor, Delta
from ..engine._basis_ledger import channel_ledger
from ..engine._basis_update import record_keys, record_sizes
from ..engine._regime_columns import polynomial_columns, regime_channel_indices
from ..engine.assemble import gather_spectra, predict_values
from ..engine.average import AveragedKernel
from ..engine.basis import SpectralBasis
from ..engine.columns import ColumnRules, column_map
from ..fitting._point_mass_fit import point_mass_map
from ..lattice._cell_values import log_displacement
from ..lattice.cells import Cells
from ..lattice.lattice import Lattice
from ..lattice.moments import Moments
from ..lattice.representations import AzimuthHarmonic
from ..population import Population
from ..population.declaration import parents
from ..population.quadrature import ownership
from ..records.ledger import ErrorTerm, Ledger
from ..values.errors import RegimeError, RoleError, SpecifiedMismatch
from ..values.spectrum import Spectrum
from ._conditional_regime import ConditionalRegimeBasis
from ._leaf import spin_factor
from ._predictions import channels_from
from ._regime_measure import physical_panels, projected_continuum_width
from ._beaming_layout import crossing_terms
from ._regimes import _check_statistics
from ._setup import placed


def _parts(regime):
    return regime.harmonic, regime.continuum


def _check_projection_width(channels, minimum_width):
    width = np.min(np.diff(np.asarray(channels.support), axis=1))
    if width < minimum_width:
        raise RegimeError(
            "the narrowest channel contains fewer than 32 harmonic spacings "
            "in the physical fit cells at the sampled geometry"
        )


def _sample_atom(source, point):
    return source.replace(
        **{
            name: Delta(float(value), unit=SYNCHROTRON_VARIABLES[name].unit)
            for name, value in point.items()
        }
    )


def _sample_moments(regime, source, point):
    return tuple(
        np.asarray(moment.values)
        for moment in regime.moments(_sample_atom(source, point))
    )


def _sample_conditional(regime, source, point):
    return np.asarray(regime.predict(_sample_atom(source, point), check=False).stokes)


def _log_energy_pullback(regime, source, gamma, half_width, orders):
    """Differentiate the field-weighted moment map within each effective cell."""
    part = regime.continuum
    index, table = part.lattice.index, part.lattice.cells
    population, _ = placed(source.replace(gamma=Delta(gamma)), ContinuumKernel())
    weight_scale = np.asarray(part.lattice.weights)
    result = np.zeros((len(orders), len(index.blocks), index.block_size))
    width = float(np.min(np.diff(np.asarray(regime.channels.support), axis=1)))
    for point, spin, mass, shifts in physical_panels(
        population,
        part.lattice,
        "continuum",
        regime.gamma_switch,
        regime.field,
        ("depth",),
        nodes=regime.moment_nodes,
        min_width=width,
        return_offsets=True,
    ):
        cell_number = ownership(point, table, mass.size, offsets=shifts)
        spin = np.broadcast_to(spin, mass.shape)
        for cell in np.unique(cell_number[cell_number >= 0]):
            keep = cell_number == cell
            span = index.cell_rows(cell)
            power = index.arrays.powers[span, 0]
            radius = table.half_width[cell, 0]
            shift = shifts.get("nu_c")
            z = (
                log_displacement(
                    point["nu_c"][keep],
                    table.reference[cell, 0],
                    offset=None if shift is None else shift[keep],
                )
                / radius
            )
            stride = 2 * half_width / radius
            weights = mass[keep] / weight_scale[cell]
            polar = weights * spin[keep]
            for i, order in enumerate(orders):
                valid = power >= order
                if np.any(valid):
                    locations = np.asarray(span)[valid]
                    degrees = power[valid]
                    coefficient = np.asarray(
                        [comb(int(s), int(order)) for s in degrees]
                    )
                    values = (
                        z[:, None] ** (degrees - order) * coefficient * stride**order
                    )
                    result[i, 0, locations] += weights @ values
                    weighted_p = polar @ values
                    result[i, 1, locations] += weighted_p.real
                    result[i, 2, locations] += weighted_p.imag
    return result.reshape(len(orders), -1)


def _transfer(regime, source, lattice, extra):
    table, index = lattice.cells, lattice.index
    shapes = tuple(part.lattice.n for part in _parts(regime))
    conditional = isinstance(regime, ConditionalRegimeBasis)
    result = np.empty(
        (index.block_size, regime.channels.n_ch, 3)
        if conditional
        else (index.block_size, sum(shapes))
    )
    sampled = {}
    switch = regime.gamma_switch
    axis = table.variables.index("gamma")
    unit_spin = source.replace(phi=AzimuthFactor(1))
    for cell, (lo, hi) in enumerate(
        zip(table.lo[:, axis], table.hi[:, axis], strict=True)
    ):
        rows = index.cell_rows(cell)
        powers = index.arrays.powers[rows]
        analytic = (
            not conditional
            and lo >= switch
            and table.variables == ("gamma",)
            and table.chart[cell][axis] == "log"
            and regime.continuum.lattice.cells.variables == ("nu_c",)
            and max(powers[:, axis]) <= np.min(regime.continuum.lattice.cells.order)
        )
        if analytic:
            result[rows, : shapes[0]] = 0.0
            result[rows, shapes[0] :] = _log_energy_pullback(
                regime,
                unit_spin,
                table.reference[cell, axis],
                table.half_width[cell, axis],
                powers[:, axis],
            )
            continue

        def point_response(point):
            key = tuple(point.values())
            if key not in sampled:
                sampled[key] = (
                    _sample_conditional(regime, unit_spin, point)
                    if conditional
                    else np.concatenate(_sample_moments(regime, unit_spin, point))
                )
            return sampled[key]

        result[rows] = polynomial_columns(
            table,
            cell,
            powers,
            point_response,
            extra,
            open_upper="gamma" if hi == switch else None,
        )
    if conditional:
        return result.transpose(1, 2, 0)
    return tuple(jnp.asarray(m.T) for m in np.split(result, [shapes[0]], axis=1))


class PaperEnergyBasis(SpectralBasis):
    """Physical gamma Taylor columns projected into the two reduced responses.

    Each projected coefficient is from an interpolant of the response to unit
    gamma point masses. Unknowns are the paper's scaled physical moments.
    The interpolation and truncation have no certified source error.
    """

    regime: object = eqx.field(kw_only=True)
    source: Population = eqx.field(kw_only=True)
    maps: tuple = eqx.field(kw_only=True)
    extra_nodes: int = eqx.field(kw_only=True, static=True)
    minimum_width: float = eqx.field(kw_only=True, static=True)

    @classmethod
    def _from_projection(
        cls, regime, source, lattice, maps, extra_nodes, minimum_width
    ):
        kernel = ContinuumKernel(gamma_min=regime.gamma_switch)
        rules = ColumnRules(stokes=("I", "P"))
        columns = column_map(lattice, kernel, rules, max_columns=lattice.n)
        forward_population, _ = placed(source, kernel)
        azimuth_factor = (
            None
            if "phi" in lattice.representations
            else spin_factor(forward_population.factors["phi"])
        )
        conditional = isinstance(regime, ConditionalRegimeBasis)
        if conditional:
            coefficients = _transfer(regime, source, lattice, extra_nodes)
        else:
            coefficients = sum(
                np.asarray(part.columns.lift(part.design())) @ np.asarray(mapping)
                for part, mapping in zip(_parts(regime), maps, strict=True)
            )
        spin = coefficients[:, 1, :] + 1j * coefficients[:, 2, :]
        spectra0, spectra2 = gather_spectra(
            coefficients[:, 0, :].T, spin.T, None, columns
        )
        ledger = Ledger(
            (
                *regime.harmonic.ledger.terms,
                *crossing_terms((forward_population,)),
                ErrorTerm.unbounded(
                    "physical energy collocation and Taylor truncation",
                    name="physical_energy_projection",
                ),
            )
        )
        record = replace(
            regime.harmonic.record,
            sizes=record_sizes(columns),
            keys=record_keys(columns, lattice),
        ).with_notes(
            (
                "physical energy columns by collocation of the conditional screen"
                if conditional
                else "physical energy columns by projection through both regimes"
            ),
            "continuum line spacing checked on fit cells at the sampled geometry",
        )
        return cls(
            lattice=lattice,
            channels=regime.channels,
            columns=columns,
            spectra0=spectra0,
            spectra2=spectra2,
            azimuth_factor=azimuth_factor,
            specified=(),
            record=record,
            ledger=ledger,
            key=None,
            averaged=AveragedKernel(kernel),
            rules=rules,
            forward_population=forward_population,
            regime=regime,
            source=source,
            maps=maps,
            extra_nodes=extra_nodes,
            minimum_width=minimum_width,
        )

    def at(self, channels):
        """Reuse the physical moment maps on another finite-width channel grid."""
        channels = channels_from(channels)
        _check_projection_width(channels, self.minimum_width)
        indices = regime_channel_indices(self.channels, channels)
        if np.array_equal(indices, np.arange(self.channels.n_ch)):
            return self
        regime = self.regime.at(channels)
        if np.all(indices >= 0):
            return replace(
                self,
                channels=channels,
                regime=regime,
                spectra0=self.spectra0[indices],
                spectra2=self.spectra2[indices],
                ledger=channel_ledger(self, channels, indices),
            )
        updated = self._from_projection(
            regime,
            self.source,
            self.lattice,
            self.maps,
            self.extra_nodes,
            self.minimum_width,
        )
        return replace(updated, azimuth_factor=self.azimuth_factor)

    def with_order(self, order):
        """Rebuild the physical interpolation at the requested Taylor order."""
        updated = paper_energy_basis(
            self.regime,
            self.source,
            self.lattice.with_order(order),
            extra_nodes=self.extra_nodes,
        )
        return replace(updated, azimuth_factor=self.azimuth_factor)

    def select(self, **cells):
        """Project the requested physical cells with their reference weights."""
        updated = paper_energy_basis(
            self.regime,
            self.source,
            self.lattice.select(**cells),
            extra_nodes=self.extra_nodes,
        )
        return replace(updated, azimuth_factor=self.azimuth_factor)

    @property
    def excluded_upper_gamma(self):
        """The upper edge whose atom belongs only to the continuum side."""
        return self.regime.gamma_switch

    def predict(self, population_or_moments, *, check=True, allow_mismatch=False):
        """Predict through the physical cells and optionally compare both reduced bases."""
        moments = self.moments(population_or_moments, allow_mismatch=allow_mismatch)
        values = predict_values(self, moments)
        ledger = self.ledger
        if check and isinstance(population_or_moments, Population):
            try:
                if isinstance(self.regime, ConditionalRegimeBasis):
                    reference = self.regime.predict(
                        population_or_moments, check=False
                    ).stokes
                else:
                    reference = self.regime._predict(
                        self.regime.moments(population_or_moments)
                    )
            except SpecifiedMismatch:
                if not allow_mismatch:
                    raise
                difference = ErrorTerm.unbounded(
                    "the population needs different branch statistics",
                    name="physical_energy_difference",
                )
            else:
                difference = ErrorTerm(
                    jnp.abs(values - reference),
                    "estimate",
                    "difference from evaluating the unrestricted physical population",
                    name="physical_energy_difference",
                )
            ledger = Ledger((*ledger.terms, difference))
        return Spectrum(
            values,
            self.channels,
            components=self.record.stokes,
            moments=moments,
            ledger=ledger,
            plan=self.plan,
        )

    def direct(self, population, *, within="support"):
        """The projected fit basis supplies no direct integration route."""
        raise RoleError(
            "use the two-regime response as the physical projection reference"
        )

    def save(self, path):
        """Require live projected bases for extrapolation and physical fit cuts."""
        raise RoleError(
            "save the fitted result or rebuild the projected paper basis from its declarations"
        )

    def project(self, moments):
        """Map scaled physical gamma moments into the two reduced lattices."""
        if isinstance(self.regime, ConditionalRegimeBasis):
            raise RoleError("conditional phases cannot project to reduced moments")
        raw = self.lattice.raw(moments)
        if "phi" in self.lattice.representations:
            zero, real, imag = jnp.split(raw, 3, axis=-1)
        else:
            zero = raw
            real = raw * jnp.real(self.azimuth_factor)
            imag = raw * jnp.imag(self.azimuth_factor)
        output = []
        for name, part, mapping in zip(
            ("harmonic", "continuum"), _parts(self.regime), self.maps, strict=True
        ):
            spin0, spin2, _ = jnp.split(mapping, 3, axis=0)
            values = jnp.concatenate(
                (zero @ spin0.T, real @ spin2.T, imag @ spin2.T), axis=-1
            )
            output.append(Moments(part.lattice, values, component=name))
        return tuple(output)

    def point_columns(self, marker, *, chunk_bytes=2**31):
        """Return physical Lobatto coordinates and their pre-collocation responses.

        A point on the switch belongs to the continuum measure. A fit using
        the Taylor columns instead contracts them with the returned moment map.
        """
        points = point_mass_map(self, marker, chunk_bytes)
        coordinates = points.coordinates
        source = self.source.replace(phi=AzimuthFactor(self.azimuth_factor))
        if isinstance(self.regime, ConditionalRegimeBasis):
            values = [
                _sample_conditional(
                    self.regime, source, dict(zip(coordinates, point, strict=True))
                )
                for point in zip(*coordinates.values(), strict=True)
            ]
            return points, jnp.stack(values, axis=-1)
        sampled = [
            _sample_moments(
                self.regime, source, dict(zip(coordinates, values, strict=True))
            )
            for values in zip(*coordinates.values(), strict=True)
        ]
        reduced = tuple(
            Moments(part.lattice, jnp.asarray(np.stack(values)))
            for part, values in zip(
                _parts(self.regime), zip(*sampled, strict=True), strict=True
            )
        )
        return points, jnp.moveaxis(self.regime._predict(reduced), 0, -1)


def _projection_source(population, resolved, switch, *, conditional=False):
    if not isinstance(population, Population):
        raise TypeError("the physical projection needs a Population")
    parts = population.parts
    if len(parts) != 1 and "phi" not in resolved:
        raise RoleError(
            "resolve phi for the joint azimuth projection of a population sum"
        )
    for part in parts:
        if part.blocks or "gamma" not in part.factors:
            raise RoleError(
                "the physical projection needs a declared primary gamma law"
            )
        if any(
            parents(law)
            and name not in resolved
            and not (conditional and name == "depth")
            for name, law in part.factors.items()
        ):
            raise RoleError(
                "resolve conditional laws on the physical projection lattice"
            )
    kernel = ContinuumKernel(gamma_min=switch)
    fixed = tuple(
        {
            name: law
            for name, law in placed(part, kernel)[0].factors.items()
            if name not in resolved
        }
        for part in parts
    )
    if any(not eqx.tree_equal(fixed[0], laws, typematch=True) for laws in fixed[1:]):
        raise SpecifiedMismatch(
            "physical projection components need common specified laws"
        )
    return Population(**parts[0].factors, symmetries=parts[0].symmetries)


def paper_energy_basis(regime, population, cells, *, weights=None, extra_nodes=3):
    """Build paper fit columns on physical energy and optional field cells.

    The lattice can also resolve azimuth with AzimuthHarmonic. Pass a Cells
    axis or a physical lattice. Use explicit reference weights before
    selecting cells. Place a face at the switch if it lies in the fitted range.
    """
    if isinstance(cells, Cells):
        lattice = Lattice(gamma=cells, weights=weights)
    elif isinstance(cells, Lattice) and weights is None:
        lattice = cells
    else:
        raise TypeError("pass gamma Cells with optional weights or a physical Lattice")
    table = lattice.cells
    variables = set(table.variables)
    if not (
        "gamma" in variables <= {"gamma", "B"}
        and all(
            name == "phi" and isinstance(rep, AzimuthHarmonic)
            for name, rep in lattice.reps
        )
    ):
        raise RoleError(
            "the physical projection takes gamma, optional B and AzimuthHarmonic"
        )
    if "B" in variables and "B" not in regime.harmonic.lattice.cells.variables:
        raise RoleError("physical B cells require B cells in the harmonic response")
    axis = table.variables.index("gamma")
    lo, hi = table.lo[:, axis], table.hi[:, axis]
    if not np.all(np.isfinite((table.lo, table.hi))) or np.any(lo < 1):
        raise RoleError("physical fit cells need finite edges with gamma >= 1")
    if np.any((lo < regime.gamma_switch) & (regime.gamma_switch < hi)):
        raise RoleError("put a physical energy cell edge at the regime switch")
    if (
        isinstance(extra_nodes, bool)
        or not isinstance(extra_nodes, Integral)
        or extra_nodes < 2
    ):
        raise ValueError("extra_nodes must be an integer of at least two")
    conditional = isinstance(regime, ConditionalRegimeBasis)
    source = _projection_source(
        population,
        variables | set(lattice.representations),
        regime.gamma_switch,
        conditional=conditional,
    )
    physical = placed(source, ContinuumKernel())[0]
    _check_statistics(
        (physical,),
        regime.averaged,
        conditional_screen=regime.screen.law if conditional else None,
    )
    minimum_width = projected_continuum_width(
        physical.factors, lattice, regime.gamma_switch, nodes=regime.moment_nodes
    )
    _check_projection_width(regime.channels, minimum_width)
    maps = () if conditional else _transfer(regime, source, lattice, int(extra_nodes))
    return PaperEnergyBasis._from_projection(
        regime, source, lattice, maps, int(extra_nodes), minimum_width
    )
