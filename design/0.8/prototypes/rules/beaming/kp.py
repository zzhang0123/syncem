"""The E4-KP population (Kardashev-Pacholczyk ageing, pitch density depending on gamma) as core measures.

The density, its normalisation and the per-mu energy quadrature come from the study oracle
(``ST/lib/ensembles.py`` KardashevPacholczyk, one common amplitude, open question 11).
Two measures are formed from it.
* ``pair_measure``: nodes (gamma, mu) over the full pitch range [-1, 1] (mirrored nodes,
  the density is even in mu), weights N(gamma, mu) dgamma dmu, so that core.pair gives
  <1_b u^r P_l(mu)> with odd l included and <1_{b,c} u^r v^s> on mu cells.
* ``direct_measure``: nodes (gamma, eta) with weights q(eta) 2 N(gamma, mu = eta) dgamma deta,
  the beaming-limit integrand of the study's ``KardashevPacholczyk.direct``.
"""
from __future__ import annotations

import sys
from functools import lru_cache
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

import core as C

STUDY = Path(__file__).resolve().parents[5] / "studies/classical-models"
if str(STUDY / "lib") not in sys.path:
    sys.path.insert(0, str(STUDY / "lib"))

import ensembles as ENS  # noqa: E402
from config import PAPER  # noqa: E402


@lru_cache(maxsize=1)
def population():
    return ENS.KardashevPacholczyk()


def _mu_nodes(mu_max: float, n_mu: int, extra_breaks=()) -> tuple[np.ndarray, np.ndarray]:
    """GL panels on [0, mu_max] split where gamma_c(mu) crosses a cell edge and at extra breaks."""
    kp = population()
    pts = {a for pair in kp._mu_panels(PAPER, mu_max) for a in pair}
    pts |= {float(b) for b in extra_breaks if 0.0 < b < mu_max}
    pts = np.array(sorted(pts))
    t, w = leggauss(n_mu)
    xs = [0.5 * (b - a) * t + 0.5 * (a + b) for a, b in zip(pts[:-1], pts[1:])]
    ws = [0.5 * (b - a) * w for a, b in zip(pts[:-1], pts[1:])]
    return np.concatenate(xs), np.concatenate(ws)


def pair_measure(n: int = 24, n_mu: int = 24, extra_breaks=()) -> C.Measure:
    """Nodes (gamma, mu) on the full pitch range, weights N dgamma dmu (total mass 1)."""
    kp = population()
    mu, wm = _mu_nodes(1.0, n_mu, extra_breaks)
    g, m, w = [], [], []
    for x, wx in zip(mu, wm):
        meas = kp.gamma_measure(PAPER, x, n, scale=wx)
        for sign in (-1.0, 1.0):
            g.append(meas.gamma)
            m.append(np.full(meas.gamma.size, sign * x))
            w.append(meas.w)
    return C.Measure({"gamma": np.concatenate(g), "mu": np.concatenate(m)}, np.concatenate(w))


def direct_measure(n: int = 24, n_eta: int = 16) -> C.Measure:
    """Nodes (gamma, eta) with weights q(eta) 2 N(gamma, mu = eta), eta ~ U[-1/2, 1/2] on [0, 1/2].

    The half range holds because q, N(gamma, .) and the kernel are even in eta.
    This is the node set of the study's KardashevPacholczyk.direct.
    """
    kp = population()
    eta, we = kp._outer(PAPER, PAPER.eta_max, n_eta)
    we = we / PAPER.eta_max
    g, e, w = [], [], []
    for x, wx in zip(eta, we):
        meas = kp.gamma_measure(PAPER, x, n, scale=2.0)
        g.append(meas.gamma)
        e.append(np.full(meas.gamma.size, x))
        w.append(wx * meas.w)
    return C.Measure({"gamma": np.concatenate(g), "eta": np.concatenate(e)}, np.concatenate(w))


def stored_truth() -> np.ndarray:
    """The study's direct E4-KP spectrum (41, 3) [I, Q, U] (forward_spectra.npz)."""
    return np.load(STUDY / "forward_spectra.npz")["E4_KP"]


def stored_forward() -> dict:
    """forward_pitch_legendre_vs_L of the study: {"N2": {"L4": [I, Q, U], ...}, "N4": ...}."""
    import json

    d = json.loads((STUDY / "forward_results.json").read_text())
    return d["ensembles"]["E4-KP"]["forward_pitch_legendre_vs_L"]
