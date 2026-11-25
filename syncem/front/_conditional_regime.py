"""Physical evaluation of a leaf Faraday screen in the two-regime response."""

from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

from ..base.continuum import ContinuumKernel
from ..base.variables import SYNCHROTRON_VARIABLES
from ..engine._regime_columns import AMPLITUDE_PER_GAUSS
from ..engine._stage_memo import StageMemo
from ..engine.assemble import predict_values
from ..engine.cache import default_cache
from ..engine.primitive import quantised_span
from ..population import Population
from ..population.declaration import law_at, parents
from ..population.quadrature import ownership
from ..records.ledger import ErrorTerm, Ledger
from ..values.errors import RoleError
from ..values.spectrum import Spectrum
from ._beaming_layout import crossing_terms
from ._regime_measure import physical_panels, pair_branch, source_columns
from ._regimes import RegimeBasis, _check_statistics, _phase
from ._setup import placed

_UNITS = {name: variable.unit for name, variable in SYNCHROTRON_VARIABLES.items()}


def conditional_depth(law, parts, harmonic):
    """Validate the optional depth role before preparing any response columns."""
    dependencies = set(parents(law))
    if not dependencies:
        return None
    if "depth" in harmonic.cells.variables:
        return None
    if not dependencies <= {"gamma", "B", "eta", "mu"} or any(
        "depth" in parents(child) for part in parts for child in part.factors.values()
    ):
        raise RoleError(
            "an unresolved regime screen must be a leaf conditioned on gamma, B, eta or mu"
        )
    return law


@dataclass(frozen=True)
class ConditionalScreen:
    law: object
    phase: object

    def at(self, point):
        depth = law_at(self.law, point, _UNITS)
        if depth is not None:
            depth = SYNCHROTRON_VARIABLES["depth"].place(depth)
        return _phase(None, depth).screen

    def at_geometry(self, gamma, field, eta):
        point = {"gamma": gamma, "B": field, "eta": eta}

        def evaluate(transmission, nu):
            if transmission is None:
                return 1.0
            return transmission(self.phase.tau(nu))

        if "mu" not in parents(self.law):
            transmission = self.at(point)
            return lambda nu, _: evaluate(transmission, nu)

        def rotate(nu, mu):
            if np.ndim(mu) == 0:
                return evaluate(self.at({**point, "mu": mu}), nu)
            return np.stack(
                [
                    np.broadcast_to(
                        evaluate(self.at({**point, "mu": m}), row), row.shape
                    )
                    for row, m in zip(nu, mu, strict=True)
                ]
            )

        return rotate


def _continuum(regime, parts, count):
    channels, lattice = regime.channels, regime.continuum.lattice
    frequency = np.asarray(channels.nodes)
    weights = np.asarray(channels.response_weights())
    phase = regime.screen.phase
    tau = phase.tau(frequency)
    phase_weight = np.asarray(phase.weights(frequency, 0.0)) * weights
    width = float(np.min(np.diff(np.asarray(channels.support), axis=1)))
    bounds = lattice.cells
    log_frequency = np.log(frequency)
    axis = bounds.variables.index("nu_c")
    lnx = (
        float(log_frequency.min() - np.log(bounds.hi[:, axis].max())),
        float(log_frequency.max() - np.log(bounds.lo[:, axis].min())),
    )
    stokes = np.zeros((channels.n_ch, 3))
    total = outside = 0.0
    names = parents(regime.screen.law)
    memo = StageMemo(default_cache())
    try:
        primitive = memo.primitive(quantised_span(lnx[0] - 1e-6, lnx[1] + 1e-6), 0)
    finally:
        memo.close()
    batch_size = max(1, 2**17 // frequency.size)
    for pop in parts:
        for point, spin, mass, shifts in physical_panels(
            pop,
            lattice,
            "continuum",
            regime.gamma_switch,
            regime.field,
            ("depth",),
            nodes=count,
            min_width=width,
            return_offsets=True,
        ):
            cells = ownership(point, lattice.cells, mass.size, offsets=shifts)
            total += float(mass.sum())
            outside += float(mass[cells < 0].sum())
            inside = np.flatnonzero(cells >= 0)
            spin = np.broadcast_to(spin, mass.shape)
            for start in range(0, len(inside), batch_size):
                rows = inside[start : start + batch_size]
                flux = mass[rows, None] * AMPLITUDE_PER_GAUSS * regime.field
                u = log_frequency[None] - np.log(point["nu_c"][rows, None, None])
                base = u - np.exp(u)
                log_f, _ = primitive.lookup(u, "F", 0)
                log_g, _ = primitive.lookup(u, "G", 0)
                stokes[:, 0] += np.sum(
                    flux * (weights * np.exp(base + log_f)).sum(-1), axis=0
                )
                rotations = []
                for values in zip(*(point[name][rows] for name in names), strict=True):
                    transmission = regime.screen.at(
                        dict(zip(names, values, strict=True))
                    )
                    value = 1.0 if transmission is None else transmission(tau)
                    rotations.append(
                        np.broadcast_to(np.asarray(value), frequency.shape)
                    )
                pol = -(phase_weight * np.exp(base + log_g) * rotations).sum(-1)
                p = np.sum((flux * spin[rows, None]) * pol, axis=0)
                stokes[:, 1] += p.real
                stokes[:, 2] += p.imag
    return stokes, total, outside


@dataclass(frozen=True)
class ConditionalRegimeBasis(RegimeBasis):
    """Two-regime response with an unresolved leaf Faraday screen."""

    screen: ConditionalScreen

    def _physical_only(self, *args, **kwargs):
        raise RoleError(
            "a conditional screen needs a physical Population or paper.energy_basis for linear fitting"
        )

    design = moments = _independent = fit_family = _physical_only

    def _prediction(self, parts, count):
        low, low_total, low_outside = pair_branch(
            parts,
            self.harmonic,
            "harmonic",
            self.gamma_switch,
            self.field,
            (*tuple(name for name, _ in self.averaged), "depth"),
            nodes=count,
        )
        high, high_total, high_outside = _continuum(self, parts, count)
        return predict_values(self.harmonic, low) + high, (
            ("harmonic", low_total, low_outside),
            ("continuum", high_total, high_outside),
        )

    def predict(self, population_or_moments, *, check=None):
        """Integrate the conditional screen on physical and frequency nodes."""
        if not isinstance(population_or_moments, Population):
            self._physical_only()
        parts = tuple(
            placed(part, ContinuumKernel())[0] for part in population_or_moments.parts
        )
        _check_statistics(parts, self.averaged, conditional_screen=self.screen.law)
        check = self.refine if check is None else check
        terms = [*self.harmonic.ledger.terms, *crossing_terms(parts)]
        terms.append(
            ErrorTerm.unbounded(
                "harmonic cell representation and continuum physical approximation",
                name="representation",
            )
        )
        count = self.moment_nodes
        values, coverage = self._prediction(parts, count)
        plan = self.plan
        if check:
            count *= 2
            finer, coverage = self._prediction(parts, count)
            terms.append(
                ErrorTerm(
                    jnp.abs(finer - values),
                    "estimate",
                    "Stokes change under physical-panel node doubling",
                    name="pairing_refinement",
                )
            )
            values = finer
            plan = replace(
                plan,
                source_columns=source_columns(parts, self.gamma_switch, nodes=count),
            )
        for branch, total, outside in coverage:
            if total > 0:
                terms.append(
                    ErrorTerm(
                        outside / total,
                        "estimate",
                        "mass outside the branch cells",
                        name=f"outside_{branch}",
                        relative_to="paired mass",
                    )
                )
        return Spectrum(
            values,
            self.channels,
            components=("I", "Q", "U"),
            ledger=Ledger(tuple(terms)),
            plan=plan,
        )
