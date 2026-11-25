"""Linear electron families expressed as fixed density basis functions.

LABEL [extension].

The derived design has shape (n_data, n_coefficient). The
coefficients use the electron column units. The expansion has no certified
error with respect to a source index distribution.
"""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from ..engine.assemble import predict_values
from ..lattice.moments import Moments
from ..records.ledger import ErrorTerm, Ledger
from ..values.errors import DataError, RoleError, SizeError
from ..values.spectrum import Spectrum
from ._family_noise import FamilyNoise
from ._family_batch import fit_pixels
from ._fit_records import RetainedCombinations
from .data import Whitened
from .family_fit import FamilyFit, Parameter, _withheld
from .free_fit import _design, _fit_basis, family_combinations
from .rank import FullRank, reduce_svd, solve_svd


def _moment_matrix(law, lattice, primary, chunk_bytes):
    table, index = lattice.cells, lattice.index
    if table.variables != (primary,) or lattice.representations:
        raise RoleError(
            "the linear density form needs its primary cells on the lattice"
        )
    sample = np.asarray(law.basis_density(table.reference[:1, 0]))
    if sample.ndim != 2 or sample.shape[0] != 1 or sample.shape[1] < 1:
        raise RoleError("basis_density must supply one coefficient axis per point")
    n = sample.shape[1]
    need = 8 * n * (3 * index.n + 96)
    if need >= chunk_bytes:
        raise SizeError(
            f"the density moment map needs about {need:,} bytes. Increase chunk_bytes."
        )
    nodes, weights = np.polynomial.legendre.leggauss(48)
    support = tuple(float(x) for x in law.support)
    output = np.zeros((index.n, n))
    for cell in range(table.n_cells):
        lo = max(float(table.lo[cell, 0]), support[0])
        hi = min(float(table.hi[cell, 0]), support[1])
        if not lo < hi:
            continue
        if not np.isfinite(lo) or not np.isfinite(hi):
            raise RoleError("the linear density form needs finite integration panels")
        reference, scale = table.reference[cell, 0], table.half_width[cell, 0]
        log = table.chart[cell][0] == "log"
        a, b = (np.log(lo), np.log(hi)) if log else (lo, hi)
        x = (b - a) * (nodes + 1) / 2 + a
        w = weights * (b - a) / 2
        u = (x - (np.log(reference) if log else reference)) / scale
        if log:
            x = np.exp(x)
            w = w * x
        powers = index.patterns[index.cell_pattern[cell]]
        weighted = u[:, None] ** np.asarray(powers).reshape(1, -1) * w[:, None]
        rows = np.asarray(tuple(index.cell_rows(cell)))
        densities = np.asarray(law.basis_density(x))
        if densities.shape != (x.size, n) or np.any(~np.isfinite(densities)):
            raise DataError("linear density columns must be finite at each panel node")
        output[rows] = weighted.T @ densities / lattice.weights[cell]
    return output


@dataclass(frozen=True)
class MomentFormFit(FamilyFit):
    """An unconstrained least-squares fit of linear density coefficients."""

    coefficients: object
    kind = "linear"
    estimator = "noise-whitened least squares on a linear density form"

    def electron_spectrum(self, gamma=None):
        law = self.population.factors[self.basis.averaged.kernel.primary]

        def density(x):
            return law.basis_density(x) @ self.coefficients

        return density if gamma is None else density(gamma)

    def _predict(self, basis):
        """Contract the fitted moments. The signed form has no density to integrate directly."""
        ledger = basis.ledger
        names = {term.name for term in ledger.terms}
        if basis.forward_accuracy is not None and "representation" not in names:
            note = (
                "the linear density form has signed coefficients, so the direct "
                "check of the fitted population is not run"
            )
            ledger = Ledger(
                (*ledger.terms, ErrorTerm.unbounded(note, name="representation")),
                assumptions=ledger.assumptions,
            )
        return Spectrum(
            predict_values(basis, self.moments),
            basis.channels,
            components=basis.record.stokes,
            ledger=ledger,
            moments=self.moments,
            plan=basis.plan,
        ).to(self.data.unit)


def fit_moment_form(basis, data, population, *, rank=None, chunk_bytes=2**31):
    """Solve for the density coefficients after integrating the lattice moment map."""
    primary = basis.averaged.kernel.primary
    law = population.factors[primary]
    if data.batch_shape:
        return fit_pixels(
            data,
            lambda pixel: fit_moment_form(
                basis, pixel, population, rank=rank, chunk_bytes=chunk_bytes
            ),
        )
    prediction_basis = basis
    if not isinstance(data, RetainedCombinations):
        basis = _fit_basis(basis, data.channels)
    mapping = _moment_matrix(law, basis.lattice, primary, chunk_bytes)
    if isinstance(data, RetainedCombinations):
        data = family_combinations(data, basis, mapping.shape[1])
        target, operator, count = data.whitened_operator()
        white = Whitened(target, operator @ mapping, count)
    else:
        matrix = jnp.asarray(_design(basis) @ mapping)
        white = data.whitened(matrix, components=basis.record.stokes)
    svd = reduce_svd(white.design)
    solution = solve_svd(svd, white.data, FullRank() if rank is None else rank)
    coefficient = solution.moments
    used = np.arange(solution.sigma.size) < int(solution.rank)
    directions = np.asarray(svd.directions)[used]
    noise = FamilyNoise(
        directions, 1 / np.asarray(solution.sigma)[used], np.ones(coefficient.size)
    )
    covariance, _, _ = noise.covariance()
    sigma = np.sqrt(np.diag(covariance))
    identified = np.sum(directions**2, axis=0) >= 1 - 1e-10
    params = {
        f"{primary}.amplitude_m{i}": Parameter(
            float(value),
            float(error) if detail else np.nan,
            (-np.inf, np.inf),
            False,
            bool(detail),
        )
        for i, (value, error, detail) in enumerate(zip(coefficient, sigma, identified))
    }
    moments = Moments(basis.lattice, mapping @ coefficient, origin="family")
    dof = int(white.count - solution.rank)
    from scipy.special import gammaincc

    p = float(gammaincc(dof / 2, float(solution.chi2) / 2)) if dof > 0 else np.nan
    return MomentFormFit(
        prediction_basis,
        data,
        population,
        params,
        moments,
        float(solution.chi2),
        dof,
        p,
        int(solution.rank),
        coefficient,
        withheld_details=_withheld(data, basis, moments),
        noise=noise,
        moment_jacobian=lambda: mapping,
        parameter_covariance=covariance,
        spectrum_jacobian=lambda output: np.asarray(_design(output) @ mapping).reshape(
            -1, coefficient.size
        ),
    )
