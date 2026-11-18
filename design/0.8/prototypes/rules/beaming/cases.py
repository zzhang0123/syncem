"""Measured cases of the beaming rule family. Each function returns plain numbers for the tests.

(a) pitch-Legendre columns against the study stack, (b) E4-KP forward errors,
(c) Cells on mu with eta on Legendre degrees or with an eta density (the factor 2),
(d) half range and parity masks under Beaming.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from math import factorial
from pathlib import Path

import jax
import numpy as np

import core as C
from core import paper as E

from . import beaming as BM
from . import store

STUDY_LIB = Path(__file__).resolve().parents[5] / "studies/classical-models/lib"
if str(STUDY_LIB) not in sys.path:
    sys.path.insert(0, str(STUDY_LIB))

GAMMA_FIXED = 1e4
B_RULE = ("B", C.uniform(4.0, 6.0, 16))


# ------------------------------------------------------------------ (a) study oracle
def study_columns(eta, probs, cells, N, ell):
    """Study stack D (n_cells, N+1, J) for I and P (no azimuth) with eta weights (2l+1) P_l(eta) q.

    ``basis.channel_kernels`` at the given eta nodes (the stack at arbitrary nodes), weighted as
    ``basis.eta_weights``. With the study's own nodes this is ``derivative_spectra(cfg, N, pitch_l)``.
    """
    import basis as SB
    from config import PAPER

    g = PAPER.gamma_b[list(cells)]
    KI, KQ = SB.channel_kernels(PAPER, g, eta_s=np.sqrt(1 - np.asarray(eta) ** 2), order=N)
    w = np.asarray(probs) * (2 * ell + 1) * BM.legendre_matrix(eta, ell)[:, ell]
    fac = np.array([PAPER.h**r / factorial(r) for r in range(N + 1)])
    DI = np.einsum("rgej,e->grj", KI, w) * fac[None, :, None]
    DQ = np.einsum("rgej,e->grj", KQ, w) * fac[None, :, None]
    return DI, DQ


def study_derivative_spectra(n_eta, N, ell):
    """The study's own call: derivative_spectra(replace(PAPER, n_eta=n_eta), N, pitch_l=ell)."""
    from dataclasses import replace

    import basis as SB
    from config import PAPER

    return SB.derivative_spectra(replace(PAPER, n_eta=n_eta), N, pitch_l=ell)


def prototype_columns(entry: dict, probs, L, factor=BM.BEAMING_FACTOR, mask_odd=False):
    """Contract a stored eta-resolved build: (n_cells, N+1, J, L+1) for I and P."""
    W = BM.pitch_weights(entry["eta"], probs, L, factor, mask_odd)
    cols = BM.contract({"I": entry["I"], "P": entry["P"]}, W)
    n_cells, N = entry["cells"].size, int(entry["N"])
    return (cols["I"].reshape(n_cells, N + 1, -1, L + 1), cols["P"].reshape(n_cells, N + 1, -1, L + 1))


# ------------------------------------------------------------------ (b) E4-KP forward
def kp_forward(entry: dict, probs, L: int, N: int, moments_M0: np.ndarray, table, factor=BM.BEAMING_FACTOR):
    """Prediction (41, 3) from pitch-Legendre columns at degree L and order N."""
    W = BM.pitch_weights(entry["eta"], probs, L, factor)
    full = BM.contract({"I": entry["I"], "P": entry["P"]}, W)
    Nb = int(entry["N"])
    keep = np.array([r for c in range(19) for r in range(Nb + 1)]) <= N
    cols = {k: v[keep] for k, v in full.items()}
    return C.predict(cols, table, {"M0": moments_M0[:, : L + 1], "M2": None}, E.e1_weights(table),
                     azimuth=E.AZIMUTH)


# ------------------------------------------------------------------ fixed-gamma kernels for (c), (d)
@lru_cache(maxsize=1)
def continuum_base():
    return C.ContinuumChannelKernel(C.bump_channels(), screen=E.SCREEN)


def kernel_at_eta(eta) -> tuple[np.ndarray, np.ndarray]:
    """B-averaged continuum kernel at gamma = 1e4 on eta nodes: (n_ch, n_eta) for I and P."""
    avg = C.AveragedKernel(BM.EtaNodesKernel(continuum_base(), eta), [B_RULE])
    I, P = jax.jit(lambda g: avg({"gamma": g}))(GAMMA_FIXED)
    return np.asarray(I), np.asarray(P)


def stokes(I, P, azimuth=E.AZIMUTH) -> np.ndarray:
    P = azimuth * P
    return np.stack([np.real(I), np.real(P), np.imag(P)], axis=-1)


# ------------------------------------------------------------------ pitch populations
def poly_pitch(coef: dict):
    """Pitch density sum_l coef[l] (2l+1)/2 P_l(mu) with exact Legendre moments n_l = coef[l]."""
    def density(mu):
        return sum(c * (2 * l + 1) / 2.0 * BM.legendre_matrix(mu, l)[..., l] for l, c in coef.items())

    def moments(L):
        return np.array([coef.get(l, 0.0) for l in range(L + 1)])

    return density, moments


ASYM = {0: 1.0, 1: 0.3, 2: 0.15, 3: -0.08}  # positive on [-1, 1], odd part present
EVEN = {0: 1.0, 2: 0.15, 4: 0.05}


def legendre_moments_numeric(density, L, n=400):
    x, w = BM.gl_panels([-1.0, 0.0, 1.0], n // 2)
    return (w * density(x)) @ BM.legendre_matrix(x, L)


def exp_pitch(k=2.0):
    """Smooth asymmetric pitch density k e^{k mu} / (2 sinh k) (all Legendre degrees present)."""
    return lambda mu: k * np.exp(k * np.asarray(mu)) / (2 * np.sinh(k))
