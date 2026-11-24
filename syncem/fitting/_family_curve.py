"""Prepare one differentiable family curve in observation coordinates."""

import jax.numpy as jnp
import numpy as np

from ..distributions.electrons import ElectronFamily
from ._primary_rows import primary_curve, primary_geometry
from .free_fit import RetainedCombinations, _design, _white


def compiled_curve(basis, data, source, names, log, start, expected):
    primary = basis.averaged.kernel.primary
    law = source.factors.get(primary)
    geometry = primary_geometry(basis.lattice, primary)
    if (
        geometry is None
        or not isinstance(law, ElectronFamily)
        or any(not name.startswith(f"{primary}.") for name in names)
    ):
        return None
    fields = tuple(name.split(".", 1)[1] for name in names)
    if isinstance(data, RetainedCombinations):
        keep = jnp.arange(data.beta.shape[0]) < data.rank
        scale = jnp.where(keep, data.sigma, 1.0)
        matrix = jnp.where(keep[:, None], data.directions / scale[:, None], 0.0)
    else:
        matrix = _white(data, basis, _design(basis)).design
    matrix *= source.column_factor
    log = jnp.asarray(log)
    saved = None

    def evaluate(coords):
        nonlocal saved
        if saved is None or not np.array_equal(coords, saved[0]):
            value, derivative = primary_curve(
                law, geometry, matrix, fields, log, jnp.asarray(coords)
            )
            saved = (
                np.array(coords, copy=True),
                np.asarray(value),
                np.asarray(derivative),
            )
        return saved[1:]

    got, derivative = evaluate(start)
    tolerance = 2e-11 * np.max(np.abs(expected))
    if np.isfinite(derivative).all() and np.max(np.abs(got - expected)) <= tolerance:
        return evaluate
    return None
