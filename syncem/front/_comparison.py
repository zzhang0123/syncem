"""Compare fitted models and check changes with lattice resolution."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np

from ..engine.basis_sum import BasisSum
from ..fitting.data import StokesData
from ..results.comparison import Comparison
from ..values.errors import RoleError
from . import predict, spectral_basis
from ._predictions import channels_from


def _withheld_chi2(fit, withheld):
    if withheld is None:
        return None if fit.withheld is None else fit.withheld.chi2
    test = fit.predict(withheld.channels)
    selection = withheld.withheld if np.any(withheld.withheld) else None
    white = withheld.whitened_spectrum(test, selection=selection)
    return jnp.sum((white.data - white.design[..., 0]) ** 2, axis=-1)


def compare(*fits, withheld=None):
    """Compare goodness of fit and the residuals on held-out observations."""
    if not fits:
        raise ValueError("compare needs at least one fit")
    if withheld is not None and not isinstance(withheld, StokesData):
        raise TypeError("withheld must be StokesData")
    headers = (
        "fit",
        "rank",
        "chi2",
        "dof",
        "p",
        "effective parameters",
        "withheld chi2",
    )
    rows = tuple(
        (
            f"{i}: {fit.kind}",
            fit.rank,
            fit.chi2,
            fit.dof,
            fit.p_value,
            getattr(fit, "effective_parameters", None),
            _withheld_chi2(fit, withheld),
        )
        for i, fit in enumerate(fits, 1)
    )
    return Comparison(headers, rows)


def convergence(population, channels, *, order=None, refine=None, **predict_options):
    """Measure the effect of changing the polynomial order or splitting the cells."""
    if (order is None) == (refine is None):
        raise ValueError("supply one of order= or refine=")
    if refine is not None and (
        isinstance(refine, bool)
        or not isinstance(refine, (int, np.integer))
        or refine < 2
    ):
        raise ValueError("refine must be an integer of at least two")
    if predict_options.get("basis") is not None:
        raise RoleError("convergence rebuilds the basis. Supply lattice= or resolve=.")
    ch = channels_from(channels)
    build_options = {
        name: value
        for name, value in predict_options.items()
        if name not in ("basis", "allow_mismatch")
    }
    basis = spectral_basis(population, ch, **build_options)
    lattice = basis.parts[0].lattice if isinstance(basis, BasisSum) else basis.lattice
    if not lattice.cells.variables:
        raise RoleError("convergence needs a lattice with cells")
    changed = (
        lattice.with_order(order)
        if order is not None
        else lattice.refine(**dict.fromkeys(lattice.cells.variables, refine))
    )
    options = dict(predict_options)
    options.pop("resolve", None)
    options.pop("lattice", None)
    options.pop("basis", None)
    baseline = predict(population, ch, basis=basis, **options)
    candidate = predict(population, ch, lattice=changed, **options)
    delta = candidate.stokes - baseline.stokes
    rows = tuple(
        (float(nu), *(delta[..., i, j] for j in range(len(baseline.components))))
        for i, nu in enumerate(ch.centres_hz)
    )
    return Comparison(
        (
            "frequency [Hz]",
            *(f"Δ{axis} [{baseline.unit}]" for axis in baseline.components),
        ),
        rows,
        baseline,
        candidate,
    )
