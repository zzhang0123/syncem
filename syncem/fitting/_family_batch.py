"""Host pixel loop for independent family likelihoods."""

from dataclasses import dataclass, replace
from functools import cached_property

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..records.ledger import Ledger
from ..results.printing import fit_summary
from ..values.errors import DataError
from ._family_result import Parameter
from ._fit_records import RetainedCombinations
from ._free_fit_arrays import Withheld


def family_pixel(data, index):
    shape = data.batch_shape

    def take(array, tail):
        if array is None:
            return None
        return jnp.broadcast_to(array, (*shape, *array.shape[-tail:]))[index]

    if isinstance(data, RetainedCombinations):
        return replace(
            data,
            observations=(
                family_pixel(data.observations, index)
                if data.observations is not None
                else None
            ),
            beta=take(data.beta, 1),
            sigma=take(data.sigma, 1),
            directions=take(data.directions, 2),
            rank=jnp.broadcast_to(data.rank, shape)[index],
        )
    return eqx.tree_at(
        lambda d: (d.stokes, d.sigma, d.cov, d.mask),
        data,
        (
            take(data.stokes, len(data._layout)),
            take(data.sigma, len(data._layout)),
            take(data.cov, 2),
            take(data.mask, 1),
        ),
        is_leaf=lambda a: a is None,
    )


@dataclass(frozen=True)
class FamilyBatch:
    """Independent fitted populations in row-major pixel order."""

    data: object
    pixels: tuple

    def _collect(self, values):
        values = tuple(values)
        if values[0] is None:
            return None
        return jnp.stack(values).reshape((*self.data.batch_shape, *np.shape(values[0])))

    def _attribute(self, name):
        return self._collect(getattr(pixel, name) for pixel in self.pixels)

    @property
    def kind(self):
        return self.pixels[0].kind

    @property
    def estimator(self):
        return self.pixels[0].estimator

    @property
    def basis(self):
        return self.pixels[0].basis

    @property
    def plan(self):
        return self.pixels[0].plan

    @property
    def defaults(self):
        return (
            ("starts reaching the minimum", (self.converged_starts, self.n_starts)),
        )

    @property
    def population(self):
        return tuple(pixel.population for pixel in self.pixels)

    @cached_property
    def params(self):
        return {
            name: Parameter(
                value=self._collect(pixel.params[name].value for pixel in self.pixels),
                sigma=self._collect(pixel.params[name].sigma for pixel in self.pixels),
                bounds=parameter.bounds,
                unit=parameter.unit,
                at_bound=self._collect(
                    pixel.params[name].at_bound for pixel in self.pixels
                ),
                identified=self._collect(
                    pixel.params[name].identified for pixel in self.pixels
                ),
            )
            for name, parameter in self.pixels[0].params.items()
        }

    @cached_property
    def moments(self):
        first = self.pixels[0].moments
        if isinstance(first, tuple):
            return tuple(
                eqx.tree_at(
                    lambda m: m.values,
                    m,
                    self._collect(p.moments[i].values for p in self.pixels),
                )
                for i, m in enumerate(first)
            )
        return eqx.tree_at(
            lambda m: m.values,
            first,
            self._collect(p.moments.values for p in self.pixels),
        )

    @property
    def withheld(self):
        if all(p.withheld is None for p in self.pixels):
            return None
        records = tuple(p.withheld or Withheld(0.0, 0, None, 0.0) for p in self.pixels)
        return Withheld(
            *(
                self._collect(getattr(item, key) for item in records)
                for key in Withheld._fields
            )
        )

    @property
    def ledger(self):
        return self._ledgers([p.ledger for p in self.pixels])

    def _ledgers(self, ledgers, stokes_shape=None):
        terms = []
        for i, item in enumerate(ledgers[0].terms):
            peers = [ledger.terms[i] for ledger in ledgers]
            if any((term.name, term.kind) != (item.name, item.kind) for term in peers):
                raise DataError("the pixel error ledgers use different terms")
            if item.value is None:
                terms.append(item)
                continue
            shape = np.broadcast_shapes(*(term.value.shape for term in peers))
            if item.space == "stokes" and item.relative_to is None:
                shape = np.broadcast_shapes(
                    shape, stokes_shape or self.pixels[0]._fit_spectrum.stokes.shape
                )
            values = self._collect(
                jnp.broadcast_to(term.value, shape) for term in peers
            )
            terms.append(replace(item, value=values))
        return Ledger(tuple(terms), assumptions=ledgers[0].assumptions)

    def predict(self, channels=None, *, cov=False):
        spectra = [p.predict(channels, cov=cov) for p in self.pixels]
        return replace(
            spectra[0],
            stokes=self._collect(s.stokes for s in spectra),
            sigma=self._collect(s.sigma for s in spectra),
            cov=self._collect(s.cov for s in spectra),
            ledger=self._ledgers([s.ledger for s in spectra], spectra[0].stokes.shape),
            moments=self.moments,
        )

    def _observable(self, values):
        values = tuple(values)
        return values[0]._replace(
            value=self._collect(v.value for v in values),
            sigma=self._collect(v.sigma for v in values),
        )

    def observable(self, c):
        return self._observable(p.observable(c) for p in self.pixels)

    def derived(self, name):
        return self._observable(p.derived(name) for p in self.pixels)

    @property
    def column(self):
        return self._observable(p.column for p in self.pixels)

    @property
    def source_column(self):
        return self._observable(p.source_column for p in self.pixels)

    def electron_spectrum(self, gamma=None):
        output = (p.electron_spectrum(gamma) for p in self.pixels)
        return tuple(output) if gamma is None else self._collect(output)

    @property
    def parameter_covariance(self):
        return self._attribute("parameter_covariance")

    @property
    def chi2(self):
        return self._attribute("chi2")

    @property
    def dof(self):
        return self._attribute("dof")

    @property
    def p_value(self):
        return self._attribute("p_value")

    @property
    def rank(self):
        return self._attribute("rank")

    @property
    def n_starts(self):
        return self._attribute("n_starts")

    @property
    def converged_starts(self):
        return self._attribute("converged_starts")

    def __str__(self):
        return fit_summary(self)

    def save(self, path):
        from ._fit_io import save_fit

        save_fit(self, path)


def fit_pixels(data, solve):
    if not np.prod(data.batch_shape):
        raise DataError("a batch of spectra needs at least one pixel")
    return FamilyBatch(
        data,
        tuple(
            solve(family_pixel(data, index)) for index in np.ndindex(data.batch_shape)
        ),
    )
