"""Throwaway JAX prototype of the SyncEM redesign core (lattice, jet provider, pairing, predict).

Import with PYTHONPATH=<prototypes>:<syncem root>. Importing enables float64 via syncem.
"""
from .averaged import AveragedKernel, fixed, legendre_rule, uniform
from .jet import build_columns, taylor_coefficients
from .kernels import Channels, ContinuumChannelKernel, PolynomialKernel, bump_channels
from .lattice import Cells, ProductTable, from_u, lower_set, to_u
from .pair import Measure, pair, panel_measure, product, reference_weights
from .predict import direct, predict, representation_error

__all__ = [
    "AveragedKernel", "fixed", "uniform", "legendre_rule", "build_columns", "taylor_coefficients",
    "Channels", "ContinuumChannelKernel", "PolynomialKernel", "bump_channels", "Cells", "ProductTable",
    "from_u", "to_u", "lower_set", "Measure", "pair", "panel_measure", "product", "reference_weights",
    "direct", "predict", "representation_error",
]
