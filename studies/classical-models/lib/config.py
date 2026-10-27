"""Paper observing geometry as one frozen configuration object.

Everything an ensemble-independent quantity needs lives here: channels and their
quadrature, the field-strength and field-direction quadrature, the azimuth factor,
the independent Gaussian screen, the noise model and the energy bins.

Use ``PAPER`` for the paper geometry; derive variants with ``dataclasses.replace``
(e.g. ``replace(PAPER, b_kind="maxwell", n_b=96)`` for E1M, or
``replace(PAPER, screen_mean=0.0, screen_sd=0.0, azimuth_amp=2.0)`` for intrinsic
polarisation tests).  Derived arrays are cached per instance.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
from functools import cached_property

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate

from kernels import C_SI


def rho1(t):
    """Smooth bump rho_1(t) = exp[-t^2/(1-t^2)] for |t| < 1, else 0."""
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    ok = np.abs(t) < 1
    out[ok] = np.exp(-t[ok] ** 2 / (1 - t[ok] ** 2))
    return out


def _z1() -> float:
    val, _ = integrate.quad(lambda t: float(rho1(np.array(t))), -1.0, 1.0, epsabs=0, epsrel=1e-13, limit=400)
    return val


Z1 = _z1()


def gl_interval(n: int, lo: float, hi: float):
    """Gauss-Legendre nodes and (unnormalised) weights on [lo, hi]."""
    t, w = leggauss(n)
    return 0.5 * (hi - lo) * t + 0.5 * (hi + lo), 0.5 * (hi - lo) * w


@dataclass(frozen=True)
class Config:
    # channels
    nu_lo: float = 1e8
    nu_hi: float = 1e10
    n_channels: int = 41
    chan_half: float = 0.2  # h_j = chan_half * nu_j (support half-width of rho_1)
    n_chan_nodes: int = 400
    # field strength
    b_kind: str = "uniform"  # "uniform" on [b_lo, b_hi] or "maxwell" with b_rms
    b_lo: float = 4.0  # muG
    b_hi: float = 6.0
    b_rms: float = 5.0
    n_b: int = 16
    maxwell_cut: float = 8.0  # upper quadrature limit in units of b_rms
    # field-direction cosine eta ~ U[-eta_max, eta_max]; quadrature on [0, eta_max]
    # (every integrand used in this study is even in eta)
    eta_max: float = 0.5
    n_eta: int = 12
    # projected-field azimuth density [1 + amp cos 2(phi - phase)]/(2 pi)
    azimuth_amp: float = 0.8
    azimuth_phase: float = 0.25
    # independent Gaussian Faraday screen [rad m^-2]
    screen_mean: float = 8.0
    screen_sd: float = 0.5
    # noise
    noise_frac: float = 0.01
    seed: int = 20260928
    # energy bins and reference metric
    gamma_lo: float = 300.0
    gamma_hi: float = 1e5
    n_bins: int = 12
    ref_slope: float = 2.6

    # ------------------------------------------------------------ channels
    @cached_property
    def centres(self) -> np.ndarray:
        return np.geomspace(self.nu_lo, self.nu_hi, self.n_channels)

    @cached_property
    def fit_idx(self) -> np.ndarray:
        return np.arange(0, self.n_channels, 2)

    @cached_property
    def held_idx(self) -> np.ndarray:
        return np.arange(1, self.n_channels, 2)

    @cached_property
    def chan_t(self) -> np.ndarray:
        return leggauss(self.n_chan_nodes)[0]

    @cached_property
    def chan_w(self) -> np.ndarray:
        """Weights w_t with sum_t w_t f(nu_t) = int R_j(nu) f(nu) dnu (unit-area response)."""
        t, w = leggauss(self.n_chan_nodes)
        return w * rho1(t) / Z1

    @cached_property
    def chan_nu(self) -> np.ndarray:
        """(J, T) channel quadrature frequencies [Hz]."""
        return self.centres[:, None] * (1.0 + self.chan_half * self.chan_t[None, :])

    @cached_property
    def screen(self) -> np.ndarray:
        """(J, T) <exp(2 i lambda^2 Psi)> at each quadrature frequency (lambda in m)."""
        lam2 = (C_SI / self.chan_nu) ** 2
        return np.exp(2j * lam2 * self.screen_mean - 2.0 * lam2**2 * self.screen_sd**2)

    @cached_property
    def azimuth_factor(self) -> complex:
        """<e^{2 i phi}> = (amp/2) e^{2 i phase} (= 0.4 e^{0.5 i} for the paper)."""
        return 0.5 * self.azimuth_amp * np.exp(2j * self.azimuth_phase)

    # ------------------------------------------------------------ field quadrature
    @cached_property
    def b_nodes_weights(self):
        """(B nodes [muG], probability weights)."""
        if self.b_kind == "uniform":
            b, w = gl_interval(self.n_b, self.b_lo, self.b_hi)
            return b, w / (self.b_hi - self.b_lo)
        if self.b_kind == "maxwell":
            sig = self.b_rms / np.sqrt(3.0)
            b, w = gl_interval(self.n_b, 0.0, self.maxwell_cut * self.b_rms)
            pdf = np.sqrt(2.0 / np.pi) * b**2 / sig**3 * np.exp(-(b**2) / (2.0 * sig**2))
            return b, w * pdf  # analytic normalisation; truncated mass < 1e-40
        raise ValueError(f"unknown b_kind {self.b_kind}")

    @cached_property
    def mean_b2(self) -> float:
        b, w = self.b_nodes_weights
        return float(np.sum(w * b**2))

    @cached_property
    def eta_nodes_weights(self):
        """(eta nodes on [0, eta_max], weights summing to 1) for q(eta) = U[-eta_max, eta_max]."""
        e, w = gl_interval(self.n_eta, 0.0, self.eta_max)
        return e, w / self.eta_max

    @cached_property
    def eta_s(self) -> np.ndarray:
        """sqrt(1 - eta^2) at the eta nodes (B_perp = B * eta_s)."""
        e, _ = self.eta_nodes_weights
        return np.sqrt(1.0 - e**2)

    @cached_property
    def mean_bperp(self) -> float:
        b, wb = self.b_nodes_weights
        _, we = self.eta_nodes_weights
        return float(np.sum(wb * b) * np.sum(we * self.eta_s))

    # ------------------------------------------------------------ energy bins
    @cached_property
    def log_edges(self) -> np.ndarray:
        return np.linspace(np.log(self.gamma_lo), np.log(self.gamma_hi), self.n_bins + 1)

    @cached_property
    def edges(self) -> np.ndarray:
        return np.exp(self.log_edges)

    @cached_property
    def h(self) -> float:
        """Bin half-width h_b in ln gamma (equal bins)."""
        return float(0.5 * (self.log_edges[1] - self.log_edges[0]))

    @cached_property
    def gamma_b(self) -> np.ndarray:
        return np.exp(0.5 * (self.log_edges[1:] + self.log_edges[:-1]))

    @cached_property
    def w_b(self) -> np.ndarray:
        """Reference bin masses of p0 ~ gamma^-ref_slope on [gamma_lo, gamma_hi], sum 1."""
        k = 1.0 - self.ref_slope
        m = np.diff(self.edges**k) / k
        return m / m.sum()

    # ------------------------------------------------------------ noise model
    def noise_sigma(self, direct: np.ndarray) -> np.ndarray:
        """(J, 3) noise sd = noise_frac * I_j^direct for each of I, Q, U."""
        direct = np.asarray(direct)
        return self.noise_frac * direct[:, 0:1] * np.ones((1, 3))

    def draw_noise(self, n_rows: int, seed: int | None = None) -> np.ndarray:
        """Standard-normal draws, shape (n_rows, 3), from default_rng(seed).

        With n_rows = 21 (fitted channels) and the paper seed this reproduces the
        harmonic pipeline's draw (finalize.py: rng.normal(size=(21, 3))).
        """
        rng = np.random.default_rng(self.seed if seed is None else seed)
        return rng.normal(size=(n_rows, 3))

    def describe(self) -> dict:
        d = {k: getattr(self, k) for k in self.__dataclass_fields__}
        d.update(azimuth_factor=[self.azimuth_factor.real, self.azimuth_factor.imag],
                 Z1=Z1, h_b=self.h, mean_b2_muG2=self.mean_b2, mean_bperp_muG=self.mean_bperp)
        return d


PAPER = Config()
PAPER_MAXWELL = replace(PAPER, b_kind="maxwell", n_b=96)
