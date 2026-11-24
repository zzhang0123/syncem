"""Opt-in harmonic prediction in three spectral-shape variables."""

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ..base.channels import Channels
from ..base.harmonic import HarmonicKernel
from ..base.phase import FaradayRotation, PhaseLaw
from ..base.variables import SYNCHROTRON_VARIABLES
from ..distributions import (
    AzimuthFactor,
    Conditional,
    CumulantScreen,
    Delta,
    NoRotation,
)
from ..engine._harmonic_shape import (
    ShapeTable,
    _positive_int,
    point_stokes,
    shape_coordinates,
)
from ..engine._harmonic_shape_statistics import DepthRule, HarmonicShapeStatistics
from ..population import Population, Samples
from ..population.declaration import law_at, ordered, parents
from ..population.quadrature import panels
from ..records.ledger import ErrorTerm, Ledger
from ..values.errors import RoleError, SizeError
from ..values.spectrum import Spectrum, _components
from ._leaf import spin_factor
from ._predictions import channels_from
from ._regime_measure import _spin_law
from ._setup import placed
from ._starts import prediction_source

_UNITS = {name: var.unit for name, var in SYNCHROTRON_VARIABLES.items()}
_STOKES = ("I", "Q", "U", "V")
_DEFAULT_Z_EDGES = (0.0, 0.5, 0.8, 0.95, 1.0)


def _transmission(phase, depth, screen):
    if phase is False or (depth == 0 and screen is None and phase.screen is None):
        return None
    if screen is not None:
        phase = phase.with_screen(screen)
    return lambda nu: np.asarray(phase.weights(nu, depth))


def _population_parts(population):
    if isinstance(population, Samples):
        population = Population(population)
    if not isinstance(population, Population):
        raise RoleError("HarmonicShapeBasis needs a specified Population or Samples")
    for part in population.parts:
        yield placed(part, HarmonicKernel())


def _placed_spin(factors, dependencies):
    law = factors.get("phi")
    names = parents(law)
    if not names or "phi" in dependencies or isinstance(law, AzimuthFactor):
        return _spin_law(factors)
    factors.pop("phi")
    variable = SYNCHROTRON_VARIABLES["phi"]

    def spin(points):
        return np.asarray(
            [
                spin_factor(
                    variable.place(
                        law_at(law, dict(zip(names, row, strict=True)), _UNITS)
                    )
                )
                for row in zip(*(points[name] for name in names), strict=True)
            ]
        )

    return spin


def _depth_mark(law):
    if law is None or isinstance(law, NoRotation):
        return 0.0, None
    if hasattr(law, "characteristic"):
        return 0.0, law
    if isinstance(law, Delta):
        return float(law.rule()[0][0]), None
    if hasattr(law, "rule"):
        depths, weights = law.rule()
        weights = np.asarray(weights)
        mass = np.sum(weights)
        if not (np.isfinite(mass) and mass > 0):
            raise RoleError("a leaf depth rule needs positive finite mass")
        return 0.0, DepthRule(depths, weights / mass)
    return float(law), None


@dataclass(frozen=True, slots=True)
class _Atom:
    coordinates: tuple | np.ndarray
    weight: float
    spin: complex
    depth: float
    screen: object
    physical: tuple | None = None


def _project_part(pop, nodes, *, reference=False, retain_leaf_depth=True):
    column = float(pop.column_factor)
    if column == 0:
        return
    factors = {}
    for name, law in pop.factors.items():
        if isinstance(law, AzimuthFactor) and isinstance(law.c, Conditional):
            if not parents(law):
                value = law_at(law.c, {})
                law = value if hasattr(value, "circular") else AzimuthFactor(value)
        else:
            if not parents(law):
                value = law_at(law, {})
                if value is not law:
                    law = SYNCHROTRON_VARIABLES[name].place(value)
        factors[name] = law
    supplied = tuple(name for block in pop.blocks for name in block.variables)
    ordered(factors, available=supplied)
    dependencies = {parent for law in factors.values() for parent in parents(law)}
    spin = _placed_spin(factors, dependencies)
    depth = factors.get("depth")
    depth_parents = parents(depth)
    conditional = None
    fixed_mark = 0.0, None
    if "depth" not in dependencies:
        factors.pop("depth", None)
        if retain_leaf_depth:
            if depth_parents:
                conditional = depth
            else:
                fixed_mark = _depth_mark(depth)

    for points, weights in panels(
        factors,
        {},
        blocks=pop.blocks,
        units=_UNITS,
        variables=SYNCHROTRON_VARIABLES,
        charts={"gamma": "log"},
        n=nodes,
    ):
        if spin is None:
            spins = np.exp(2j * points["phi"])
        else:
            spins = np.broadcast_to(spin(points), weights.shape)
        for index in np.flatnonzero(weights):
            geometry = tuple(
                float(points[name][index]) for name in ("gamma", "B", "mu", "eta")
            )
            reduced = shape_coordinates(*geometry)
            if reduced is None:
                continue
            coordinates, power = reduced
            mass = column * float(weights[index])
            node_spin = complex(spins[index]) if abs(geometry[-1]) < 1 else 0j
            node_depth, screen = fixed_mark if node_spin else (0.0, None)
            if node_spin and conditional is not None:
                where = {name: points[name][index] for name in depth_parents}
                law = law_at(conditional, where, _UNITS)
                law = SYNCHROTRON_VARIABLES["depth"].place(law)
                node_depth, screen = _depth_mark(law)
            if node_spin and "depth" in points:
                node_depth = float(points["depth"][index])
            weighted = mass * power
            yield _Atom(
                coordinates,
                weighted,
                weighted * node_spin,
                node_depth,
                screen,
                (geometry, mass, node_spin) if reference else None,
            )


def _shape_atoms(statistics):
    for row, power, spin, depth, screen_index in zip(
        statistics.coordinates,
        statistics.power_weights,
        statistics.spin_weights,
        statistics.depths,
        statistics.screen_indices,
        strict=True,
    ):
        screen = None if screen_index < 0 else statistics.screens[screen_index]
        yield _Atom(row, power, spin, depth, screen)


def _truncates(screen):
    law = getattr(screen, "__self__", screen)
    return isinstance(law, CumulantScreen) and law.truncates


def _physical_atoms(population, nodes, defaults, **options):
    for part, declared in _population_parts(population):
        defaults.extend(declared)
        yield from _project_part(part, nodes, **options)


@dataclass(frozen=True, eq=False)
class HarmonicShapeBasis:
    """Reusable finite basis for absolute channel Stokes of a physical population.

    The spacing factor is the actual response at each harmonic frequency.
    The Bessel amplitudes are interpolated in the argument ratio. The signed
    polarisation ratio enters algebraically. The approximation is unbounded.
    """

    channels: Channels
    _table: ShapeTable = field(repr=False)
    phase: PhaseLaw | Literal[False]
    order: int
    nodes: int
    max_orders: int

    @classmethod
    def build(
        cls,
        channels,
        *,
        spacing,
        z_edges=_DEFAULT_Z_EDGES,
        order=12,
        phase: PhaseLaw | Literal[False] | None = None,
        nodes=24,
        max_orders=4096,
    ) -> "HarmonicShapeBasis":
        """Build for an inclusive spacing interval in Hz and argument-ratio cells.

        `order` is the Bessel-amplitude degree in each ratio cell. `nodes` is
        the physical density quadrature count per panel. `max_orders`
        caps the distinct integer orders needed over all passbands. Use
        `phase=False` to disable Faraday rotation or pass a PhaseLaw.
        """
        channels = channels_from(channels)
        phase = FaradayRotation() if phase is None else phase
        if phase is not False and not isinstance(phase, PhaseLaw):
            raise TypeError("phase must be a PhaseLaw, False or None")
        nodes = _positive_int(nodes, "nodes")
        table = ShapeTable.build(channels, spacing, z_edges, order, max_orders)
        return cls(channels, table, phase, int(order), nodes, int(max_orders))

    @property
    def spacing(self) -> tuple[float, float]:
        """Declared interval of fundamental line spacings in Hz."""
        return self._table.spacing

    @property
    def z_edges(self) -> tuple[float, ...]:
        """Edges of the Bessel argument-ratio cells."""
        return self._table.edges

    def at(self, channels) -> "HarmonicShapeBasis":
        """Use the same basis settings on other finite passbands."""
        channels = channels_from(channels)
        if channels is self.channels:
            return self
        return type(self).build(
            channels,
            spacing=self.spacing,
            z_edges=self.z_edges,
            order=self.order,
            phase=self.phase,
            nodes=self.nodes,
            max_orders=self.max_orders,
        )

    def moments(
        self, population, *, strict=False, max_atoms=200_000
    ) -> HarmonicShapeStatistics:
        """Materialise the phase-independent weighted shape rule.

        The basis supplies the population quadrature count. The returned rule
        can be applied with new passbands, cell edges or a different phase.
        """
        pop, starting = prediction_source(population, strict)
        max_atoms = _positive_int(max_atoms, "max_atoms")
        defaults: list[tuple] = []
        coordinates: list[tuple | np.ndarray] = []
        weights: list[float] = []
        spins: list[complex] = []
        depths: list[float] = []
        indices: list[int] = []
        screens: list[object] = []
        screen_ids: dict[int, int] = {}
        for atom in _physical_atoms(pop, self.nodes, defaults):
            if len(weights) == max_atoms:
                raise SizeError(f"harmonic statistics exceed max_atoms={max_atoms}")
            index = -1
            if atom.screen is not None:
                key = id(atom.screen)
                index = screen_ids.get(key, len(screens))
                if index == len(screens):
                    screen_ids[key] = index
                    screens.append(atom.screen)
            coordinates.append(atom.coordinates)
            weights.append(atom.weight)
            spins.append(atom.spin)
            depths.append(atom.depth)
            indices.append(index)
        if starting:
            defaults.append(("starting values", starting))
        return HarmonicShapeStatistics(
            np.asarray(coordinates).reshape(-1, 3),
            np.asarray(weights),
            np.asarray(spins),
            np.asarray(depths),
            np.asarray(indices, dtype=np.int64),
            tuple(screens),
            tuple(dict.fromkeys(defaults)),
        )

    def predict(self, population, *, check=True, stokes=None, strict=False) -> Spectrum:
        """Contract a weighted shape rule or stream the transform of a population.

        Checking compares finite against direct reduced lines. With a physical
        population it also compares against the physical harmonic calculation.
        The quadrature and numerical line errors are unbounded.
        """
        statistics = (
            population if isinstance(population, HarmonicShapeStatistics) else None
        )
        check = bool(check)
        reference = check and statistics is None
        if statistics is None:
            pop, starting = prediction_source(population, strict)
            defaults: list[tuple] = []
            atoms = _physical_atoms(
                pop,
                self.nodes,
                defaults,
                reference=reference,
                retain_leaf_depth=self.phase is not False,
            )
        else:
            starting = ()
            defaults = list(statistics.defaults)
            atoms = _shape_atoms(statistics)
        labels = (
            _STOKES
            if stokes is None
            else tuple(stokes.replace("P", "QU") if isinstance(stokes, str) else stokes)
        )
        labels = _components(labels, len(labels))
        selected = [_STOKES.index(label) for label in labels]
        values = np.zeros((1 + check + reference, self.channels.n_ch, 4))
        terms = [
            ErrorTerm.unbounded("physical population quadrature", name="quadrature"),
            ErrorTerm.unbounded(
                "Bessel evaluations, frequencies, phase and sums use float64",
                name="harmonic_numerics",
            ),
        ]
        truncated = self.phase is not False and any(
            _truncates(screen) for screen in self.phase.screens
        )
        for atom in atoms:
            values += point_stokes(
                self._table,
                atom.coordinates,
                atom.weight,
                atom.spin,
                _transmission(self.phase, atom.depth, atom.screen),
                check=check,
                physical=atom.physical,
            )
            if self.phase is not False and not truncated and _truncates(atom.screen):
                truncated = True
        if truncated:
            terms.append(
                ErrorTerm.unbounded(
                    "omitted higher screen cumulants", name="depth_cumulant_truncation"
                )
            )
        if not np.all(np.isfinite(values)):
            raise ArithmeticError("nonfinite harmonic population spectrum")
        values = values[..., selected]
        if check:
            terms.append(
                ErrorTerm(
                    values[0] - values[1],
                    "measured",
                    "finite basis minus direct transformed integer sum on the paired rule",
                    name="representation",
                )
            )
        else:
            terms.append(
                ErrorTerm.unbounded(
                    "Bessel-amplitude interpolation", name="representation"
                )
            )
        if reference:
            terms.append(
                ErrorTerm(
                    values[1] - values[2],
                    "measured",
                    "direct transformed minus physical harmonic line powers on the paired rule",
                    name="coordinate_transform",
                    signed=True,
                )
            )
        elif statistics is not None:
            terms.append(
                ErrorTerm.unbounded(
                    "physical quadrature nodes are unavailable in the reduced statistics",
                    name="coordinate_transform",
                )
            )
        if starting:
            defaults.append(("starting values", starting))
        return Spectrum(
            values[0],
            self.channels,
            components=labels,
            ledger=Ledger(tuple(terms)),
            moments=statistics,
            defaults=tuple(dict.fromkeys(defaults)),
        )
