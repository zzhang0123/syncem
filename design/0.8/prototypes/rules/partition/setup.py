"""Shared configuration of the partition prototype.

One averaged kernel serves every test. Its 68 channels are the paper's 41 E1 channels
(0.1 to 10 GHz), the 26 E2 channels of the 1 to 100.6 GHz band (Spec G3) and one channel
at 300 GHz, all bump channels of support half-width 0.2 nu_j with 64 Gauss-Legendre nodes,
as in the core's E1 configuration. Channels are independent, so slicing the first 41 gives
the E1 kernel of core/paper.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

import core as C
from core import paper as E
from core.kernels import Z1, Channels, rho1

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"
CACHE.mkdir(exist_ok=True)
STUDY_LIB = HERE.parents[4] / "studies/classical-models/lib"

N_NODES = 64
HALF = 0.2
E1_CH = slice(0, 41)
E2_CH = slice(41, 67)
C300 = 67
E2_CENTRES = np.geomspace(1e9, 1e9 * 40 ** (31 / 20), 32)[:26]  # 1 to 100.6 GHz (Spec G3)
CENTRES = np.concatenate([np.geomspace(1e8, 1e10, 41), E2_CENTRES, [3e11]])
GAMMA_TOP = 3e7  # largest gamma with x >= x_min = 1e-9 at every node (x = 3.5e-9 at 0.08 GHz, B_perp 6 uG)
GAMMA_FLOOR = 1.0  # lower end of the "unbounded" power law


def channels() -> Channels:
    t, w = leggauss(N_NODES)
    nu = CENTRES[:, None] * (1.0 + HALF * t[None, :])
    return Channels(nu, np.broadcast_to(w * rho1(t) / Z1, nu.shape).copy())


_AVG = {}


def kernel() -> C.AveragedKernel:
    """E1 averaged kernel (B ~ U[4, 6] uG, eta on [0, 1/2], Gaussian screen) on the 68 channels."""
    if "k" not in _AVG:
        e, we = leggauss(12)
        eta, w_eta = 0.25 * (e + 1.0), we * 0.25 / 0.5
        K = C.ContinuumChannelKernel(channels(), screen=E.SCREEN, n_nodes_F=128)
        _AVG["k"] = C.AveragedKernel(K, [("B", C.uniform(4.0, 6.0, 16)), ("eta", C.fixed(eta, w_eta))],
                                     outer_batch=1)
    return _AVG["k"]
