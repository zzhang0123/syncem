"""The paper's Sec. 5 E1 configuration on the prototype core (Spec Sec. 9.6, G1).

19 equal ln-gamma cells on [300, 2.9626e6], B ~ U[4, 6] uG (16 Gauss-Legendre nodes),
eta ~ U[-1/2, 1/2] taken on the half range [0, 1/2] with 12 nodes (the kernel depends on
eta through sqrt(1 - eta^2) only, as in the study's config.py), isotropic pitch, azimuth
factor 0.4 e^{0.5i}, Gaussian screen (8, 0.5) rad m^-2 inside the channel integral.
"""
from __future__ import annotations

import numpy as np
from numpy.polynomial.legendre import leggauss

from .averaged import AveragedKernel, fixed, uniform
from .kernels import ContinuumChannelKernel, bump_channels
from .lattice import Cells, ProductTable
from .pair import panel_measure, reference_weights

GAMMA_LO = 300.0
GAMMA_HI = 300.0 * (1e5 / 300.0) ** (19 / 12)  # 2.9626e6, config.py gamma_hi
AZIMUTH = 0.4 * np.exp(0.5j)
SCREEN = (8.0, 0.5)


def e1_kernel(n_chan_nodes=64, n_nodes_F=128, outer_batch=1) -> AveragedKernel:
    e, we = leggauss(12)
    eta, w_eta = 0.25 * (e + 1.0), we * 0.25 / 0.5  # GL on [0, 0.5], probabilities summing to 1
    K = ContinuumChannelKernel(bump_channels(n_nodes=n_chan_nodes), screen=SCREEN, n_nodes_F=n_nodes_F)
    return AveragedKernel(K, [("B", uniform(4.0, 6.0, 16)), ("eta", fixed(eta, w_eta))],
                          outer_batch=outer_batch)


def e1_table(N: int) -> ProductTable:
    return ProductTable.tensor({"gamma": Cells.log(GAMMA_LO, GAMMA_HI, 19, order=N)}, total=N)


def power_law_measure(table: ProductTable, slope=2.6, n=24):
    """Unit column of gamma^-slope on [GAMMA_LO, GAMMA_HI], panels split at the cell edges."""
    k = 1.0 - slope
    norm = k / (GAMMA_HI**k - GAMMA_LO**k)
    edges = np.unique(np.concatenate([table.lo[:, 0], table.hi[:, 0]]))
    return panel_measure("gamma", lambda g: norm * g ** (-slope), edges, n=n, chart="log")


def e1_weights(table: ProductTable) -> np.ndarray:
    return reference_weights(table, 2.6, "gamma")
