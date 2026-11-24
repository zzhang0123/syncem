"""Streaming ray integration and preparation of its individual emitting steps."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..base.continuum import ContinuumKernel
from ..engine._basis_update import stokes_blocks
from ..engine.assemble import stokes_axes
from ..engine.columns import ColumnRules
from ..engine.direct import integrate
from ..lattice.cells import Cells
from ..lattice.lattice import Lattice
from ..model.errors import WEAKEST_ORDER
from ..planning.defaults import support
from ..population import Population, Ray
from ..records.ledger import ErrorTerm, Ledger
from ..values.errors import DataError, NothingToPredict, RoleError
from ..values.spectrum import Spectrum
from ._direct import _panel_geometry
from ._predictions import channels_from
from ._setup import configure


def as_ray(population):
    if isinstance(population, Ray):
        return population
    if (
        isinstance(population, Population)
        and len(population.blocks) == 1
        and not population.factors
    ):
        block = population.blocks[0]
        if isinstance(block, Ray):
            return eqx.tree_at(
                lambda ray: ray.electron_density,
                block,
                block.electron_density * population.column_factor,
            )
    return None


def _unit_column(law):
    if hasattr(law, "column") and hasattr(law, "amplitude"):
        return eqx.tree_at(
            lambda density: (density.column, density.amplitude),
            law,
            (1.0, None),
            is_leaf=lambda x: x is None,
        )
    return law


def ray_population(ray):
    """One specified component per nonzero emitting step."""
    parts = tuple(_components(ray))
    if not parts:
        raise NothingToPredict("the ray has no nonzero emitting steps")
    if len(parts) == 1:
        return parts[0]
    return eqx.tree_at(lambda pop: pop.components, Population(), parts)


def _components(ray):
    for factors, law, column in ray.iter_steps():
        yield column * Population(gamma=_unit_column(law), **factors)


def _sightline(ray, index, offset):
    # Ray.__getitem__ supports array selections and constructs a full index
    # map. A scalar sightline only needs its contiguous profile slice.
    start = offset * ray.n_steps
    profiles = slice(start, start + ray.n_steps)
    out = object.__new__(type(ray))
    for name in ("step", "B_vector", "thermal_density", "electron_density"):
        object.__setattr__(out, name, getattr(ray, name)[index])
    for name in ("electrons", "pitch"):
        law = getattr(ray, name)
        object.__setattr__(out, name, law[profiles] if isinstance(law, tuple) else law)
    return out


def _has_emission(ray):
    return bool(jnp.any((ray.columns > 0) & jnp.any(ray.B_vector != 0, axis=-1)))


def _term_value(term, axes, shape):
    value = np.asarray(term.value)
    if term.space != "stokes" or term.relative_to is not None:
        return value
    if value.shape[-1:] == (4,) and shape[-1] < 4:
        value = value[..., [("I", "Q", "U", "V").index(axis) for axis in axes]]
    return np.broadcast_to(value, shape)


@dataclass
class _SightlineTerm:
    seen: int = 0
    exemplar: ErrorTerm | None = None
    values: np.ndarray | None = None
    unavailable: str = ""
    varied_note: bool = False

    def add(self, term, batch, index, axes, shape):
        self.seen += 1
        if self.exemplar is None:
            self.exemplar = term
        if self.unavailable:
            return
        if term.kind == "unbounded":
            self.exemplar = term
            self.unavailable = term.note or "an emitting sightline has no finite term"
            self.values = None
            return
        if term.value is None:
            return
        try:
            value = _term_value(term, axes, shape)
        except ValueError:
            self.unavailable = "the sightline diagnostic has incompatible Stokes axes"
            self.values = None
            return
        if self.values is None:
            self.values = np.zeros((*batch, *value.shape), dtype=float)
            self.exemplar = term
        elif not self._compatible(term, value, len(batch)):
            self.unavailable = "emitting sightlines have different diagnostic layouts"
            self.values = None
            return
        else:
            self.varied_note |= term.note != self.exemplar.note
            if WEAKEST_ORDER.index(term.kind) > WEAKEST_ORDER.index(self.exemplar.kind):
                self.exemplar = term
        self.values[index] = value

    def _compatible(self, term, value, batch_rank):
        metadata = ("signed", "declared", "space", "relative_to", "manuscript_term")
        return value.shape == self.values.shape[batch_rank:] and all(
            getattr(term, name) == getattr(self.exemplar, name) for name in metadata
        )

    def term(self, total):
        exemplar = self.exemplar
        missing = self.seen < total
        if missing or self.unavailable:
            return ErrorTerm(
                None,
                "unbounded",
                self.unavailable or "an emitting sightline does not report this term",
                name=exemplar.name,
                manuscript_term=exemplar.manuscript_term,
                relative_to=exemplar.relative_to,
                space=exemplar.space,
            )
        if self.values is None:
            return exemplar
        note = (
            "Diagnostic details vary across sightlines."
            if self.varied_note
            else exemplar.note
        )
        return replace(exemplar, value=jnp.asarray(self.values), note=note)


def predict_ray_batch(ray, channels, evaluate):
    """Evaluate one sightline at a time, retaining only the batch output arrays."""
    if not ray.shape:
        raise RoleError("a batched Ray needs a leading sightline axis")
    ch = channels_from(channels)
    keys = ch.row_keys()
    values, axes, unit = None, None, None
    terms, assumptions = {}, {}
    emitting = 0
    for offset, index in enumerate(np.ndindex(ray.shape)):
        sightline = _sightline(ray, index, offset)
        if not _has_emission(sightline):
            continue
        member = evaluate(sightline)
        stokes = np.asarray(member.stokes)
        if values is None:
            axes, unit = member.components, member.unit
            values = np.zeros((*ray.shape, ch.n_ch, len(axes)), dtype=float)
        if (
            stokes.shape != (ch.n_ch, len(axes))
            or member.unit != unit
            or member.components != axes
            or member.channels.row_keys() != keys
        ):
            raise DataError("Ray sightlines returned incompatible spectra")
        values[index] = stokes
        for position, term in enumerate(member.ledger.terms):
            name = term.name or ("unnamed", position)
            state = terms.setdefault(name, _SightlineTerm())
            state.add(term, ray.shape, index, axes, stokes.shape)
        assumptions.update(dict.fromkeys(member.ledger.assumptions))
        emitting += 1
        del member
    if values is None:
        raise NothingToPredict("the ray batch has no nonzero emitting sightlines")
    ledger = Ledger(
        tuple(state.term(emitting) for state in terms.values()),
        assumptions=tuple(assumptions),
    )
    return Spectrum(values, ch, ledger=ledger, components=axes, unit=unit)


def predict_ray(ray, channels, *, kernel=None, phase=None, stokes=None):
    """Integrate a ray without building a spectral basis."""
    ch = channels_from(channels)
    model = ContinuumKernel() if kernel is None else kernel
    if model.primary != "gamma":
        raise RoleError("direct Ray integration needs an electron kernel")
    if ray.shape:
        return predict_ray_batch(
            ray,
            ch,
            lambda line: predict_ray(
                line, ch, kernel=model, phase=phase, stokes=stokes
            ),
        )
    components = stokes_axes(stokes_blocks(model, ColumnRules(), None))
    values = None
    for part in _components(ray):
        lo, hi = support(part.factors["gamma"])
        if not (np.isfinite(lo) and lo > 0 and hi >= lo):
            raise RoleError("a ray electron law needs a positive lower support edge")
        if np.isfinite(hi) and hi > lo:
            count = max(1, math.ceil((math.log(hi) - math.log(lo)) / 0.484))
            cells = Cells.log(lo, hi, count, order=0)
        else:
            cells = Cells.single(lo, lo, chart="log", support=(lo, hi), order=0)
        latent = Lattice(gamma=cells)
        setup = configure(part, model, ch, latent, phase, direct=True)
        direct = integrate(
            setup.averaged,
            ch,
            _panel_geometry(setup, latent, ch, "support"),
            b_perp_max=setup.b_max,
            azimuth_factor=setup.azimuth,
        )
        p = direct.P[0]
        modes = {"I": direct.I, "Q": p.real, "U": p.imag, "V": direct.V}
        spectrum = jnp.stack([modes[axis] for axis in components], axis=-1)
        values = spectrum if values is None else values + spectrum
    if values is None:
        raise NothingToPredict("the ray has no nonzero emitting steps")
    ledger = Ledger(
        (
            ErrorTerm.from_model("physical_kernel", model.physical_error(ch)),
            ErrorTerm.unbounded(
                "the line-of-sight and electron quadratures have no measured error bound",
                name="quadrature",
            ),
        )
    )
    return Spectrum(values, ch, ledger=ledger, components=components).select(stokes)
