"""syncem: moment expansion of synchrotron emission (JAX + Equinox).

Note: importing this package enables ``jax_enable_x64`` (float64) as a global
side effect, because the Bessel quadrature and the likelihood-style sums need
float64 for accuracy. Do this once at the top of any driver script, before any
array is created.

Modules
-------
bessel           : differentiable J_n, J_n', K_nu (integral representations)
stokes           : exact Schott harmonic Stokes (I, Q, V) with correct
                   normalisation
derivatives      : derivative spectra dS/dp, d^2S/dp^2 via autodiff
expansion         : eqx.Module for quadratic Taylor averaging (mean/covariance)
cumulants        : finite Bell moment contractions from supplied cumulants
ultrarel         : ultra-relativistic F(x), G(x) fast path
rm               : rotation measure (RM) constants + Burn depolarisation
faraday          : joint emission-depth averages and finite responses
transfer         : Mueller matrix + numerical slab/LOS matrix exponentials
solutions        : analytic limiting solutions (thin/rotation/self-absorbed)
kirchhoff        : emissivity/absorption moment expansion + Kirchhoff closure
conversion       : cold-plasma Faraday rotation/conversion coefficients + moments
magnus           : Magnus expansion (Omega1, Omega2) of the LOS transfer
los_moments      : explicit reduced-units and physical-CGS moment-driven slabs
sed              : frequency-domain SED (spectral index, curvature, abs. emissivity)
model            : finite joint response of channel Stokes spectra and spectral
                   fits (subpackage: index, errors, channels, bandpass,
                   phase, moments,
                   bounds, assumptions, kernels, harmonic,
                   high_order_harmonic, screens, basis, high_order_basis,
                   predict, adapters, fit.observation, fit.linear,
                   fit.nonlinear, fit.diagnostics, fit.result, fit.layout,
                   fit.reduction, fit.combinations, fit.combination_result).
                   Import its modules by full path.
"""

from __future__ import annotations

__version__ = "0.7.10"

import jax

# Enable float64 for accuracy of Bessel quadrature and likelihood-style sums.
jax.config.update("jax_enable_x64", True)

from . import (  # noqa: E402
    bessel,
    conversion,
    cumulants,
    derivatives,
    kirchhoff,
    los_moments,
    magnus,
    expansion,
    rm,
    faraday,
    sed,
    solutions,
    stokes,
    transfer,
    ultrarel,
)  # noqa: E402,F401
from .stokes import larmor_power, stokes_harmonic  # noqa: E402,F401
from .derivatives import derivative_spectra  # noqa: E402,F401
from .expansion import (  # noqa: E402,F401
    CumulantExpansion,
    QuadraticTaylorExpansion,
    build_expansion,
    mixed_moments,
)
from .ultrarel import F, G  # noqa: E402,F401

__all__ = [
    "__version__",
    "bessel",
    "stokes",
    "derivatives",
    "expansion",
    "cumulants",
    "ultrarel",
    "rm",
    "faraday",
    "transfer",
    "solutions",
    "kirchhoff",
    "conversion",
    "magnus",
    "los_moments",
    "sed",
    "stokes_harmonic",
    "larmor_power",
    "derivative_spectra",
    "CumulantExpansion",
    "QuadraticTaylorExpansion",
    "build_expansion",
    "mixed_moments",
    "F",
    "G",
]


# Lazy front door of the 0.8 layer (design/0.8/SPEC.md Sec. 9.1). Names load their module on first
# use, so ``import syncem`` loads neither ``syncem.model`` nor SciPy.
_FRONT_DOOR = {
    "Channels": "syncem.base.channels",
    "Variable": "syncem.base.variables",
    "Quantity": "syncem.units",
    "Cells": "syncem.lattice.cells",
    "CellTable": "syncem.lattice.table",
    "Legendre": "syncem.lattice.representations",
    "AzimuthHarmonic": "syncem.lattice.representations",
    "Lattice": "syncem.lattice.lattice",
    "MomentIndex": "syncem.lattice.index",
    "Moments": "syncem.lattice.moments",
    "ContinuumKernel": "syncem.base.continuum",
    "Kernel": "syncem.base.kernel",
    "Modes": "syncem.base.kernel",
    "Regime": "syncem.base.kernel",
    "PhaseLaw": "syncem.base.phase",
    "FaradayRotation": "syncem.base.phase",
    **{
        name: "syncem.base.capabilities"
        for name in (
            "Similarity",
            "FieldScalingRelation",
            "Beaming",
            "Parity",
            "EnergyWindow",
            "ReferenceDensity",
            "Lines",
            "AnalyticTaylor",
            "HostOnly",
        )
    },
    **{
        name: "syncem.values.errors"
        for name in (
            "UnitError",
            "RoleError",
            "RegimeError",
            "SpecifiedMismatch",
            "NothingToPredict",
            "DataError",
            "DepthExpansionError",
            "SizeError",
            "ImproperPosterior",
        )
    },
}


def __getattr__(name):
    module = _FRONT_DOOR.get(name)
    if module is None:
        raise AttributeError(f"module 'syncem' has no attribute {name!r}")
    import importlib

    value = getattr(importlib.import_module(module), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(_FRONT_DOOR))
