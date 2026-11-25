"""Sample measures on a beaming diagonal (SPEC 4.2, T19)."""

from dataclasses import dataclass
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp

from ..base.capabilities import Beaming
from ..distributions._common import Law
from ..engine._beaming_diagonal import point_weights
from ..engine.average import NodeRule
from ..model._provenance import AssumptionRecord
from ..population import Samples
from ..population.declaration import law_at, parents
from ..records.ledger import ErrorTerm, Ledger


class _SampledCosine(Law):
    """A one-dimensional empirical probability measure with its original atoms."""

    values: Any
    weights: Any

    def rule(self, n=None):
        return self.values, self.weights

    @property
    def support(self):
        active = self.weights > 0
        return (
            jnp.min(jnp.where(active, self.values, jnp.inf)),
            jnp.max(jnp.where(active, self.values, -jnp.inf)),
        )


def pitch_measure(pop, kernel):
    """Place a pitch-only sample block among the scalar laws, retaining its mass."""
    beam = next((cap for cap in kernel.capabilities if isinstance(cap, Beaming)), None)
    if beam is None:
        return pop
    for block in pop.blocks:
        if isinstance(block, Samples) and block.variables == (beam.variable,):
            blocks = tuple(other for other in pop.blocks if other is not block)
            factors = {
                **pop.factors,
                beam.variable: _SampledCosine(
                    block.nodes[beam.variable], block.weights
                ),
            }
            mass = pop.column_factor
            if block.column is not None:
                mass = mass * block.column
            return eqx.tree_at(
                lambda p: (p.factors, p.blocks, p.column_factor),
                pop,
                (factors, blocks, mass),
            )
    return pop


@dataclass(frozen=True)
class SampleBeaming:
    pitch: str
    look: str
    degree: int
    smooth: bool

    def weights(self, point):
        """The finite expansion of twice the cosine delta in the pitch variable."""
        pitch = point_weights(point[self.pitch], self.degree)
        look = point_weights(point[self.look], self.degree)
        return jnp.sum(pitch * look * (4 / (2 * jnp.arange(self.degree + 1) + 1)))

    def projection_rule(self):
        """Project the cosine delta at the actual viewing quadrature node."""

        def project(point):
            return jnp.atleast_1d(point[self.look]), jnp.ones(1)

        return NodeRule(project, parents=(self.look,))

    def routes(self, factors, units):
        if self.smooth:
            # The smoothing weights use the empirical pitch. Physical laws use the sightline.
            return (), (("sample", self.weights),), ((self.pitch, self.look),)
        law = factors[self.look]

        def density(point):
            return 2 * law_at(law, point, units).density(point[self.pitch])

        return ((self.look, self.pitch),), (("all", density),), ()

    def record(self, ledger):
        if not self.smooth:
            return ledger
        label = (
            f"samples smoothed by Legendre({self.degree}) on {self.pitch} = {self.look}"
        )
        assumption = AssumptionRecord(
            "sample pitch",
            label,
            ((self.pitch, self.look),),
            "Legendre smoothing",
            (float(self.degree),),
            0,
            "unbounded",
        )
        if assumption in ledger.assumptions:
            return ledger
        terms = ledger.terms
        if not any(term.name == "representation:angular" for term in terms):
            terms = (*terms, ErrorTerm.unbounded(label, name="representation:angular"))
        return Ledger(terms, assumptions=(*ledger.assumptions, assumption))


def _view_has_density(law, kernel):
    """A conditional factory may return a density, a point mass, or a scalar."""
    names = parents(law)
    units = {variable.name: variable.unit for variable in kernel.variables}

    def density_marker(point):
        local = law_at(law, point, units)
        return (
            ()
            if hasattr(local, "density") and not getattr(local, "is_point", False)
            else None
        )

    if not names:
        return density_marker({}) is not None
    # Infer the return structure without sampling a fictitious physical point.
    scalar = jax.ShapeDtypeStruct((), jnp.float64)
    return jax.eval_shape(density_marker, dict.fromkeys(names, scalar)) is not None


def sample_beaming(kernel, pop, lattice, *, projected=False):
    cap = next((cap for cap in kernel.capabilities if isinstance(cap, Beaming)), None)
    if cap is None:
        return None
    pitch, look = cap.variable, cap.partner
    sampled = {name for block in pop.blocks for name in block.variables}
    has_samples = pitch in sampled or isinstance(pop.factors.get(pitch), _SampledCosine)
    degrees = dict(lattice.index.legendre)
    if not has_samples and not (projected and pitch in degrees):
        return None
    has_density = _view_has_density(pop.factors.get(look), kernel)
    if not has_samples and has_density:
        return None
    return SampleBeaming(pitch, look, degrees.get(pitch, 8), not has_density)


def angular_ledger(ledger, kernel, lattice, populations):
    """Use the angular reference of the actual predicted population."""
    if not any(isinstance(cap, Beaming) for cap in kernel.capabilities):
        return ledger
    terms = tuple(t for t in ledger.terms if t.name != "representation:angular")
    assumptions = tuple(
        a for a in ledger.assumptions if a.closure_kind != "Legendre smoothing"
    )
    if len(terms) != len(ledger.terms) or len(assumptions) != len(ledger.assumptions):
        ledger = Ledger(terms, assumptions=assumptions)
    for pop in populations:
        projection = sample_beaming(kernel, pop, lattice)
        if projection is not None:
            ledger = projection.record(ledger)
    return ledger
