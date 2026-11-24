"""Linear fits of a sum of spectral bases to one observation vector.

LABEL [extension].

Each component keeps its original lattice units and coefficient shape.
Concatenated columns share the observed Stokes units. The inverse solution
has no certified uncertainty in discarded directions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from hashlib import sha256
from numbers import Integral

import jax
import jax.numpy as jnp
import numpy as np

from ..lattice.moments import Moments
from ..records.ledger import ErrorTerm, Ledger, combine_ledgers
from ..results.printing import fit_summary
from ..values.errors import DataError, SizeError
from ..values.observable import Observable
from ..values.spectrum import Spectrum
from ._combined import IndependentBases, combined_columns
from ._fit_records import RetainedCombinations
from ._free_fit_arrays import (
    factor_chunk,
    memory_chunk,
    on_pixels,
    pixel_chunks,
    project,
    shared_noise,
    solve_chunk,
    withheld_statistics,
)
from ._response_bias import bias_elements, bias_terms, response_bias
from .free_fit import _ledger
from .rank import warn_rank_margin


@dataclass(frozen=True)
class SumFitInput:
    """Unwhitened compiled inputs for direct deterministic fits."""

    basis: object
    data: object


def _white(data, matrix, *, held=False):
    return data.whitened(matrix, components=data.components, withheld=held)


def sum_fit_workspace(data, width, chunk_bytes, *, n_out=0, cov=False, recompute=True):
    """Budget one raw component design and the whitened chunk workspace."""
    if (
        isinstance(chunk_bytes, bool)
        or not isinstance(chunk_bytes, Integral)
        or chunk_bytes <= 0
    ):
        raise ValueError("chunk_bytes must be a positive integer.")
    if math.prod(data.batch_shape) == 0:
        raise DataError("there are no sightlines to fit.")
    shared = shared_noise(data)
    m, n = data.n_data, width
    k = min(m, n)
    is_fit = n_out == 0
    raw_rows = data.channels.n_ch * len(data.components)
    fixed = 2 * n_out * n
    if recompute:
        fixed += 2 * raw_rows * n
        if data.response is not None:
            fixed += m * n
    if recompute and shared:
        fixed += m * n + (m + n) * k
    if data.cov is not None and recompute and shared:
        fixed += 4 * m * m
    diagnostics, pixel = bias_elements(data, n, shared=shared) if is_fit else (0, 0)
    if data.cov is not None and recompute and not shared:
        pixel += 4 * m * m
    reserved = math.ceil(1.5 * 8 * (fixed + diagnostics))
    if reserved >= chunk_bytes:
        raise SizeError(
            f"the combined design needs {reserved:,} bytes of fixed workspace. "
            "Increase chunk_bytes."
        )
    return memory_chunk(
        data,
        n,
        chunk_bytes - reserved,
        8,
        held=is_fit and bool(np.any(np.asarray(data.withheld))),
        n_out=n_out,
        cov=cov,
        shared=shared or not recompute,
        extra_elements=pixel,
    )


@dataclass(frozen=True)
class SumFit:
    """A fit of independent component moments in the shared observation space."""

    problem: object
    retained: RetainedCombinations
    moments: tuple[Moments, ...]
    chi2: object
    dof: object
    p_value: object
    withheld: object
    ledger: Ledger
    chunk_bytes: int
    chunk_pixels: int

    kind = "linear"
    estimator = "noise-whitened SVD of the component designs"
    population = params = penalty = weights = effective_parameters = None

    @property
    def rank(self):
        return self.retained.rank

    @property
    def plan(self):
        return self.problem.basis.parts[0].plan

    @property
    def defaults(self):
        return (
            ("components", self.problem.basis.components),
            ("chunk bytes", self.chunk_bytes),
            ("pixels per fit chunk", self.chunk_pixels),
            (
                "retained directions",
                "stored" if self.retained.directions is not None else "recomputed",
            ),
        )

    @property
    def _width(self):
        return sum(m.values.shape[-1] for m in self.moments)

    def _project(self, rows, *, cov=False):
        data = self.problem.data
        n = self._width
        directions = self.retained.directions
        shared = shared_noise(data)
        size = sum_fit_workspace(
            data,
            n,
            self.chunk_bytes,
            n_out=rows.shape[0],
            cov=cov,
            recompute=directions is None,
        )
        fit_design = None
        if directions is None:
            fit_design = combined_columns(
                self.problem.basis, data.channels, data.components
            )
            if shared:
                sample = next(pixel_chunks(data, 1)) if data.batch_shape else data
                directions = factor_chunk(_white(sample, fit_design).design, None)

        batches = math.prod(data.batch_shape)
        kept = self.retained.sigma.reshape(batches, -1)
        per_pixel = directions is not None and directions.ndim > 2
        if per_pixel:
            directions = directions.reshape(batches, -1, n)

        def chunk(part, start):
            stop = start + math.prod(part.batch_shape)
            scale = kept[start:stop] if data.batch_shape else self.retained.sigma
            if per_pixel:
                vectors = directions[start:stop]
            elif directions is None:
                vectors = factor_chunk(_white(part, fit_design).design, None)
            else:
                vectors = directions
            return project(rows, vectors, scale, cov)

        return on_pixels(chunk, data, size)

    def _value(self, rows):
        start, total = 0, 0.0
        for moment in self.moments:
            stop = start + moment.values.shape[-1]
            total = total + moment.values @ rows[:, start:stop].T
            start = stop
        return total

    def predict(self, channels=None, *, cov=False):
        basis = self.problem.basis
        first = basis.parts[0]
        default = (
            self.problem.data.channels
            if isinstance(basis, IndependentBases)
            else first.channels
        )
        ch = default if channels is None else channels
        axes = tuple(
            axis
            for axis in first.record.stokes
            if all(axis in part.record.stokes for part in basis.parts)
        )
        count = ch.n_ch * len(axes)
        n = self._width
        data = self.problem.data
        sum_fit_workspace(
            data,
            n,
            self.chunk_bytes,
            n_out=count,
            cov=cov,
            recompute=self.retained.directions is None,
        )
        design = combined_columns(basis, ch, axes)
        rows = design.reshape(count, n)
        if cov:
            n_bytes = math.prod(data.batch_shape) * count**2 * rows.dtype.itemsize
            print(f"Spectrum covariance size {n_bytes:,} bytes")
        sigma, fraction, covariance = self._project(rows, cov=cov)
        shape = (*data.batch_shape, *design.shape[:2])
        return Spectrum(
            self._value(rows).reshape(shape),
            ch,
            components=axes,
            moments=self.moments,
            sigma=sigma.reshape(shape),
            cov=covariance,
            unconstrained=fraction.reshape(shape),
            ledger=_spectrum_ledger(basis.parts, sigma.reshape(shape)),
            plan=first.plan,
        ).to(self.problem.data.unit)

    def observable(self, c):
        c = jnp.asarray(c)
        n = self._width
        if c.shape[-1:] != (n,) or jnp.iscomplexobj(c):
            raise DataError(f"the functional must end in {n} real moment coefficients")
        rows = c.reshape(-1, n)
        sigma, unconstrained, _ = self._project(rows)
        shape = (*self.problem.data.batch_shape, *c.shape[:-1])
        return Observable(
            self._value(rows).reshape(shape),
            sigma.reshape(shape),
            unconstrained.reshape(shape),
            "minimum_norm",
            "linear functional of component moments",
        )

    @property
    def column(self):
        rows = []
        for part in self.problem.basis.parts:
            if not getattr(part, "source_column_identifiable", True):
                raise DataError(
                    "the critical-frequency moments carry a field-weighted column"
                )
            lattice = part.lattice
            idx = lattice.index.offsets[:-1] * lattice.index.n_pairs
            rows.append(jnp.zeros(lattice.n).at[idx].set(lattice.weights))
        return self.observable(jnp.concatenate(rows))._replace(
            label="total column in cm^-2"
        )

    @property
    def source_column(self):
        return self.column

    def __str__(self):
        return fit_summary(self)

    def save(self, path):
        from ._fit_io import save_fit

        save_fit(self, path)


def _spectrum_ledger(parts, sigma=None):
    ledger = combine_ledgers(_ledger(part) for part in parts)
    terms = tuple(
        (
            ErrorTerm.unbounded(
                "a physical envelope for the component sum is required", name=term.name
            )
            if term.name == "physical_kernel"
            else term
        )
        for term in ledger.terms
    )
    if sigma is not None:
        terms = (
            *terms,
            ErrorTerm(
                sigma,
                "estimate",
                "standard deviation from observation noise",
                name="noise",
            ),
        )
    return Ledger(terms, assumptions=ledger.assumptions)


def _solve_chunks(data, design, rank, size, keep_directions):
    shared = shared_noise(data)
    held = bool(np.any(np.asarray(data.withheld)))
    reduction = None
    if shared and data.batch_shape:
        sample = next(pixel_chunks(data, 1))
        _, reduction = solve_chunk(_white(sample, design), rank, None)

    def chunk(part, _):
        white = _white(part, design)
        solution, svd = solve_chunk(white, rank, None, reduction)
        test = None
        if held:
            validation = _white(part, design, held=True)
            test = withheld_statistics(part, validation, solution, svd)
        cut = svd.cut if reduction is None else None
        stored = svd.directions if keep_directions and reduction is None else None
        bias = response_bias(part, design, data.components, solution, svd)
        return solution, white.count, cut, stored, test, bias

    solution, count, cut, directions, withheld, bias = on_pixels(chunk, data, size)
    if reduction is not None:
        directions = reduction.directions if keep_directions else None
        cut = reduction.cut
    warn_rank_margin(cut)
    return solution, count, directions, withheld, bias


def _component_moments(basis, values):
    offset = 0
    moments = []
    for name, part in zip(basis.components, basis.parts, strict=True):
        end = offset + part.lattice.n
        moments.append(
            Moments(
                part.lattice,
                values[..., offset:end],
                origin="minimum_norm",
                component=name,
                masked=tuple(map(int, np.flatnonzero(part.columns.col_of_row < 0))),
            )
        )
        offset = end
    return tuple(moments)


def solve_sum(problem, rank, chunk_bytes, *, keep_directions=False):
    """Fit one combined design using shared or chunked per-pixel noise."""
    if not isinstance(keep_directions, bool):
        raise ValueError("keep_directions must be a bool.")
    data, basis = problem.data, problem.basis
    width = sum(part.lattice.n for part in basis.parts)
    size = sum_fit_workspace(data, width, chunk_bytes)
    design = combined_columns(basis, data.channels, data.components)
    solution, count, directions, withheld, bias = _solve_chunks(
        data, design, rank, size, keep_directions
    )
    key = sha256(
        repr(tuple((x.lattice.key, x.columns.key) for x in basis.parts)).encode()
    ).hexdigest()
    retained = RetainedCombinations(
        solution.beta, solution.sigma, directions, solution.rank, key, unit=data.unit
    )
    base = _spectrum_ledger(basis.parts)
    ledger = Ledger(
        (*base.terms, *bias_terms(data, bias)), assumptions=base.assumptions
    )
    dof = count - solution.rank
    value = jax.scipy.special.gammaincc(jnp.maximum(dof, 1) / 2, solution.chi2 / 2)
    return SumFit(
        problem,
        retained,
        _component_moments(basis, solution.moments),
        solution.chi2,
        dof,
        jnp.where(dof > 0, value, jnp.nan),
        withheld,
        ledger,
        chunk_bytes,
        size,
    )
