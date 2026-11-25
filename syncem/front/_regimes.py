"""Public two-part response for finite-channel synchrotron populations.

LABEL ``eq: finite joint response``, ``eq: continuum joint response`` and
``eq: cell sum``.
"""

from dataclasses import dataclass, replace
from functools import partial
from numbers import Integral

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..base.continuum import ContinuumKernel
from ..base.phase import FaradayRotation, PhaseLaw
from ..base.variables import CRITICAL_FREQUENCY, SYNCHROTRON_VARIABLES
from ..base.depth import DEPTH_TOL, tolerance
from ..distributions import Delta, NoRotation
from ..engine._regime_columns import (
    build_part,
    continuum_columns,
    harmonic_columns,
)
from ..engine.assemble import predict_values
from ..engine._basis_ledger import depth_term
from ..fitting._combined import IndependentBases
from ..fitting.data import StokesData
from ..fitting.problem import Problem
from ..fitting.sum_fit import SumFitInput, solve_sum
from ..lattice.moments import Moments
from ..lattice.representations import AzimuthHarmonic, Legendre
from ..model.phase import phase_coordinate
from ..population import Population
from ..population.declaration import parents
from ..population.quadrature import _axis, _legendre
from ..records.ledger import ErrorTerm, Ledger
from ..units import to_canonical
from ..values.errors import RegimeError, RoleError, SpecifiedMismatch
from ..values.spectrum import Spectrum
from ._beaming_layout import crossing_terms
from ._predictions import channels_from
from ._regime_measure import pair_branch, source_columns
from ._setup import placed, screen_truncation

_PHYSICAL_VARIABLES = tuple(SYNCHROTRON_VARIABLES.values())


def _canonical(lattice, continuum):
    variables = (
        (
            CRITICAL_FREQUENCY,
            SYNCHROTRON_VARIABLES["depth"],
            SYNCHROTRON_VARIABLES["phi"],
        )
        if continuum
        else _PHYSICAL_VARIABLES
    )
    names = set(lattice.cells.variables)
    primary = "nu_c" if continuum else "gamma"
    allowed = {primary, "depth"} if continuum else {primary, "B", "depth"}
    if not names <= allowed or primary not in names:
        raise RoleError(
            f"{primary} cells are required and allowed cell variables are {sorted(allowed)}"
        )
    if not isinstance(lattice.representations.get("phi"), AzimuthHarmonic):
        raise RoleError("both regime lattices need phi=AzimuthHarmonic()")
    for name, rep in lattice.reps:
        if name == "phi":
            continue
        if continuum or name not in {"mu", "eta"} or not isinstance(rep, Legendre):
            raise RoleError(
                f"{name} is absent from the regime's angular representations"
            )
    for chart in lattice.cells.chart:
        if continuum and chart[lattice.cells.variables.index("nu_c")] != "log":
            raise RoleError("continuum cells must use the log critical-frequency chart")
    return lattice._for_kernel(variables, "depth")


def _as_point(law):
    return float(law.rule()[0][0]) if isinstance(law, Delta) else float(law)


def _independent(pop, name, *, no_children=False, ignore_depth=False):
    law = pop.factors.get(name)
    if (
        law is None
        or parents(law)
        or (
            no_children
            and any(
                name in parents(other)
                for child, other in pop.factors.items()
                if not (ignore_depth and child == "depth")
            )
        )
    ):
        raise RoleError(f"resolve {name} on the lattice to use its joint statistics")
    return law


def _nodes(law, count):
    if law is None:
        return ()
    pts = list(_axis(law, (-np.inf, np.inf), "affine", count, ()))
    return tuple(
        (float(x), float(w))
        for xx, ww in pts
        for x, w in zip(xx, ww, strict=True)
        if w != 0
    )


@dataclass(frozen=True)
class _HarmonicRule:
    field: object
    inclination: object
    pitch_law: object
    angular_nodes: int
    frequency_nodes: int
    field_nodes: int
    inclination_nodes: int

    def pitch_points(self, refine):
        law = self.pitch_law
        if law is None:
            return ((None, 1.0),)
        density = getattr(law, "density", None)
        if density is not None and not getattr(law, "is_point", False):
            return ((density, 1.0),)
        return _nodes(law, self.angular_nodes * refine)

    @property
    def pitch_support(self):
        if getattr(self.pitch_law, "is_point", False):
            return (-1.0, 1.0)
        return getattr(self.pitch_law, "support", (-1.0, 1.0))

    @property
    def pitch_breaks(self):
        return getattr(self.pitch_law, "kinks", ())

    def field_points(self, refine):
        return _nodes(self.field, self.field_nodes * refine)

    def inclination_points(self, order, refine):
        count = self.inclination_nodes * refine
        if order is None:
            return tuple((x, np.array([w])) for x, w in _nodes(self.inclination, count))
        nodes, weights = _legendre(max(count, order + 1))
        coefficients = (
            np.polynomial.legendre.legvander(nodes, order)
            * (2 * np.arange(order + 1) + 1)
            / 2
        )
        return tuple(zip(nodes, coefficients * weights[:, None], strict=True))


def _phase(phase, screen):
    base = FaradayRotation() if phase is None else phase
    if not isinstance(base, PhaseLaw):
        raise TypeError("phase must be a PhaseLaw")
    if screen is None or isinstance(screen, NoRotation):
        return base
    if hasattr(screen, "characteristic"):
        transmission = screen.characteristic
    elif hasattr(screen, "rule"):
        depths, mass = screen.rule()

        def transmission(tau):
            return jnp.sum(
                jnp.exp(1j * jnp.asarray(tau)[..., None] * depths) * mass, axis=-1
            )

    else:
        depth = _as_point(screen)

        def transmission(tau):
            return jnp.exp(1j * tau * depth)

    return base.with_screen(transmission)


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


@dataclass(frozen=True)
class RegimePlan:
    """Build records and optional physical column estimates for each branch."""

    gamma_switch: float
    reference_field: float
    harmonic: object
    continuum: object
    source_columns: tuple[float, float] | None = None

    def to_dict(self):
        """Represent both build records and the measured physical columns."""
        return {
            "gamma_switch": self.gamma_switch,
            "reference_field_G": self.reference_field,
            "harmonic": self.harmonic.to_dict(),
            "continuum": self.continuum.to_dict(),
            "source_columns_cm^-2": self.source_columns,
        }


@dataclass(frozen=True)
class RegimeBasis:
    """Reusable harmonic and critical-frequency bases sharing one observation.

    Build from physical population statistics and two lattices. Both carry
    azimuth harmonic rows. Intensity and Q/U are in channel cgs units. The
    upper-branch zeroth moment is a field-weighted column. Refinement in
    this response is an estimate and the physical approximation is unbounded.
    """

    harmonic: object
    continuum: object
    gamma_switch: float
    field: float
    averaged: tuple
    moment_nodes: int
    refine: bool

    @classmethod
    def build(
        cls,
        population,
        channels,
        *,
        harmonic,
        continuum,
        gamma_switch=300.0,
        reference_field="5 uG",
        phase=None,
        angular_nodes=12,
        frequency_nodes=64,
        field_nodes=12,
        inclination_nodes=12,
        moment_nodes=12,
        check=True,
        max_columns=20_000,
        depth_tol=DEPTH_TOL,
    ):
        """Build branch columns with specified statistics on unresolved axes."""
        if not isinstance(population, Population):
            raise TypeError("RegimeBasis.build needs a physical Population")
        if not np.isfinite(gamma_switch) or gamma_switch <= 1:
            raise ValueError("gamma_switch must be finite and exceed 1")
        field = float(to_canonical(reference_field, "field", name="reference_field"))
        if not np.isfinite(field) or field <= 0:
            raise ValueError("reference_field must be finite and positive")
        settings = {
            name: _positive_int(value, name)
            for name, value in (
                ("angular_nodes", angular_nodes),
                ("frequency_nodes", frequency_nodes),
                ("field_nodes", field_nodes),
                ("inclination_nodes", inclination_nodes),
                ("moment_nodes", moment_nodes),
            )
        }
        harmonic, continuum = _canonical(harmonic, False), _canonical(continuum, True)
        if ("depth" in harmonic.cells.variables) != (
            "depth" in continuum.cells.variables
        ):
            raise RoleError(
                "resolve depth on both branches or use one independent screen"
            )
        g = harmonic.cells.variables.index("gamma")
        if (
            not np.isclose(
                np.max(harmonic.cells.hi[:, g]), gamma_switch, rtol=1e-14, atol=0
            )
            or np.min(harmonic.cells.lo[:, g]) <= 1
        ):
            raise RegimeError(
                "harmonic energy cells must start above 1 and end at gamma_switch"
            )
        ch = channels_from(channels)
        parts = tuple(placed(part, ContinuumKernel())[0] for part in population.parts)
        avg = tuple(
            name
            for name in ("B", "eta", "mu")
            if name not in harmonic.cells.variables
            and name not in harmonic.representations
        )
        from ._conditional_regime import (
            ConditionalRegimeBasis,
            ConditionalScreen,
            conditional_depth,
        )

        conditional = conditional_depth(parts[0].factors.get("depth"), parts, harmonic)
        if "depth" not in harmonic.cells.variables and conditional is None:
            avg = (*avg, "depth")
        laws = tuple(
            (
                name,
                _independent(
                    parts[0],
                    name,
                    no_children=True,
                    ignore_depth=conditional is not None,
                ),
            )
            for name in avg
        )
        _check_statistics(parts, laws, conditional_screen=conditional)
        fixed = dict(laws)
        phase = _phase(phase, fixed.get("depth"))
        screen = None if conditional is None else ConditionalScreen(conditional, phase)
        depth_tol = tolerance(depth_tol)
        rule = _HarmonicRule(
            fixed.get("B"),
            fixed.get("eta"),
            fixed.get("mu"),
            settings["angular_nodes"],
            settings["frequency_nodes"],
            settings["field_nodes"],
            settings["inclination_nodes"],
        )
        low = partial(
            harmonic_columns,
            lattice=harmonic,
            phase=phase,
            rule=rule,
            conditional_screen=screen,
        )
        high = partial(continuum_columns, lattice=continuum, phase=phase, field=field)

        omitted = screen_truncation(parts[0], phase, conditional=screen is not None)

        def low_part(grid):
            part = _build_checked_depth(
                grid,
                harmonic,
                lambda k: low(grid, extra=3 if k == 1 else 5, refine=k),
                "harmonic",
                check,
                max_columns=max_columns,
                rebuild=low_part,
                phase=phase,
                depth_tol=depth_tol,
            )
            if omitted is None:
                return part
            return replace(part, ledger=Ledger((*part.ledger.terms, omitted)))

        def high_part(grid):
            _check_continuum_channels(grid)
            return _build_checked_depth(
                grid,
                continuum,
                lambda _: high(grid),
                "continuum",
                False,
                max_columns=max_columns,
                rebuild=high_part,
                phase=phase,
                depth_tol=depth_tol,
            )

        upper = high_part(ch)
        basis_class = cls if screen is None else ConditionalRegimeBasis
        return basis_class(
            low_part(ch),
            upper,
            float(gamma_switch),
            field,
            laws,
            settings["moment_nodes"],
            check,
            **({} if screen is None else {"screen": screen}),
        )

    @property
    def channels(self):
        return self.harmonic.channels

    @property
    def plan(self):
        return RegimePlan(
            self.gamma_switch, self.field, self.harmonic.record, self.continuum.record
        )

    def at(self, channels):
        """Rebuild both bases on another observation grid."""
        from dataclasses import replace

        ch = channels_from(channels)
        return replace(
            self, harmonic=self.harmonic.at(ch), continuum=self.continuum.at(ch)
        )

    def design(self, data=None):
        """Concatenate both Stokes designs, optionally whitened by data."""
        return jnp.concatenate(
            (self.harmonic.design(data), self.continuum.design(data)), axis=-1
        )

    def _pair(self, population, nodes):
        parts = tuple(placed(part, ContinuumKernel())[0] for part in population.parts)
        _check_statistics(parts, self.averaged)
        averaged = tuple(name for name, _ in self.averaged)
        harmonic = pair_branch(
            parts,
            self.harmonic,
            "harmonic",
            self.gamma_switch,
            self.field,
            averaged,
            nodes=nodes,
        )
        screen = tuple(name for name in averaged if name == "depth")
        continuum = pair_branch(
            parts,
            self.continuum,
            "continuum",
            self.gamma_switch,
            self.field,
            screen,
            nodes=nodes,
            channels=self.channels,
        )
        return harmonic, continuum

    def moments(self, population):
        """Project a physical population into the two lattices in regime order."""
        return tuple(part[0] for part in self._pair(population, self.moment_nodes))

    def _predict(self, moments):
        return predict_values(self.harmonic, moments[0]) + predict_values(
            self.continuum, moments[1]
        )

    def predict(self, population_or_moments, *, check=None):
        """Contract both branches and record physical-pairing refinement."""
        check = self.refine if check is None else check
        terms = list(self.harmonic.ledger.terms)
        terms.extend(
            t for t in self.continuum.ledger.terms if t.name != "physical_kernel"
        )
        terms.append(
            ErrorTerm.unbounded(
                "a comparison with direct physical integration is required",
                name="representation",
            )
        )
        plan = self.plan
        if isinstance(population_or_moments, Population):
            terms.extend(crossing_terms(population_or_moments.parts))
            paired = self._pair(population_or_moments, self.moment_nodes)
            moments = tuple(part[0] for part in paired)
            if check:
                refined = self._pair(population_or_moments, 2 * self.moment_nodes)
                finer = tuple(part[0] for part in refined)
                error = jnp.abs(self._predict(finer) - self._predict(moments))
                terms.append(
                    ErrorTerm(
                        error,
                        "estimate",
                        "Stokes change under physical-panel node doubling",
                        name="pairing_refinement",
                    )
                )
                paired = refined
                moments = finer
                from dataclasses import replace

                parts = tuple(
                    placed(part, ContinuumKernel())[0]
                    for part in population_or_moments.parts
                )
                plan = replace(
                    plan,
                    source_columns=source_columns(
                        parts, self.gamma_switch, nodes=2 * self.moment_nodes
                    ),
                )
            for (moment, total, missed), description in zip(
                paired, ("physical harmonic", "field-weighted continuum"), strict=True
            ):
                if total > 0:
                    terms.append(
                        ErrorTerm(
                            missed / total,
                            "estimate",
                            f"fraction of {description} mass outside the cells",
                            name=f"outside_{moment.component}",
                            relative_to="paired mass",
                        )
                    )
        else:
            moments = tuple(population_or_moments)
            if len(moments) != 2 or not all(isinstance(m, Moments) for m in moments):
                raise TypeError(
                    "predict needs a Population or one Moments object per branch"
                )
        return Spectrum(
            self._predict(moments),
            self.channels,
            components=("I", "Q", "U"),
            moments=moments,
            ledger=Ledger(tuple(terms)),
            plan=plan,
        )

    def _independent(self):
        return IndependentBases(
            (self.harmonic, self.continuum), ("harmonic", "continuum")
        )

    def problem(self, data, *, prior=None):
        """Compile one likelihood for the two independent moment vectors."""
        return Problem(self._independent(), data, prior=prior)

    def fit(self, data, *, rank=None, keep_directions=False, chunk_bytes=2**31):
        """Fit the concatenated free-moment response with a whitened SVD."""
        if not isinstance(data, StokesData):
            raise TypeError("the free-moment regime fit needs StokesData")
        return solve_sum(
            SumFitInput(self._independent(), data),
            rank,
            chunk_bytes,
            keep_directions=keep_directions,
        )

    def fit_family(self, population, data, *, starts=None):
        """Fit physical electron parameters to full or retained observations."""
        from ._regime_family import fit_regime_family

        return fit_regime_family(self, population, data, starts=starts)


def _check_statistics(parts, laws, *, conditional_screen=None):
    if conditional_screen is not None:
        for part in parts:
            actual = part.factors.get("depth")
            if not eqx.tree_equal(conditional_screen, actual, typematch=True):
                raise SpecifiedMismatch(
                    "the conditional depth law differs from the built screen"
                )
            if any("depth" in parents(law) for law in part.factors.values()):
                raise RoleError("a characteristic-only screen cannot be a parent")
    for name, expected in laws:
        for part in parts:
            actual = _independent(
                part,
                name,
                no_children=True,
                ignore_depth=conditional_screen is not None,
            )
            if not eqx.tree_equal(expected, actual, typematch=True):
                raise SpecifiedMismatch(
                    f"the {name} law differs from that averaged into the basis"
                )


def _build_checked_depth(
    channels, lattice, provider, name, check, *, max_columns, rebuild, phase, depth_tol
):
    from dataclasses import replace

    geometry = _DepthVariable()
    term = depth_term(lattice, geometry, phase, channels, depth_tol)
    part = build_part(
        channels,
        lattice,
        provider,
        name,
        check,
        max_columns=max_columns,
        rebuild=rebuild,
    )
    if term is not None:
        if name == "harmonic" and phase.tau is not phase_coordinate:
            term = ErrorTerm.unbounded(
                "custom phase needs a support envelope at the emitted frequencies",
                name="harmonic_depth_truncation",
            )
        else:
            term = ErrorTerm(
                term.value,
                term.kind,
                term.note,
                name=f"{name}_depth_truncation",
                relative_to=term.relative_to,
            )
        return replace(part, ledger=Ledger((*part.ledger.terms, term)))
    return part


class _DepthVariable:
    depth = "depth"


def _check_continuum_channels(channels):
    widths = np.diff(np.asarray(channels.support), axis=1)
    if np.any(widths <= 0):
        raise RegimeError(
            "the two-part response needs a finite-width observing channel"
        )
