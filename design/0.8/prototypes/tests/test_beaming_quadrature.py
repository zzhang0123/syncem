"""Quadrature of eta laws whose support reaches |eta| = 1 (the extreme laws of the beaming tests).

Kbar depends on eta through B_perp = B sqrt(1 - eta^2), which has a branch point at |eta| = 1.
Gauss-Legendre in eta then converges algebraically. In theta = arccos(eta) the integrand
f(cos t) sin t is smooth, so Gauss-Legendre in theta converges fast. Reference: graded panels.
Run: PYTHONPATH=.:../../.. python -m pytest tests/test_beaming_quadrature.py
"""
from __future__ import annotations

import numpy as np
import pytest

from rules.beaming import beaming as BM
from rules.beaming import cases as CS
from rules.beaming.record import record

from test_beaming_parity import beaming_sum, reference_rule, rel  # noqa: E402

LAWS = {"isotropic": (0.0, 1.0), "near1_|eta|>0.95": (0.95, 1.0)}


def theta_rule(a, b, n):
    """Probabilities of U on [a, b] (half range) by Gauss-Legendre in theta = arccos(eta)."""
    t, w = BM.gl_panels([np.arccos(b), np.arccos(a)], n)
    eta, wt = np.cos(t), w * np.sin(t)
    return eta, wt / (b - a)


@pytest.mark.parametrize("n", [6, 12, 24])
@pytest.mark.parametrize("law", list(LAWS))
def test_theta_rule_against_eta_rule(law, n):
    a, b = LAWS[law]
    dens = CS.poly_pitch({0: 1.0})[0]
    iv = [(-b, -a), (a, b)] if a > 0 else [(-b, b)]
    ref = beaming_sum(*reference_rule(iv), dens)
    out = {"gl_in_eta": rel(beaming_sum(*BM.mirror(*BM.uniform_law(a, b, n)), dens), ref),
           "gl_in_theta": rel(beaming_sum(*BM.mirror(*theta_rule(a, b, n)), dens), ref)}
    out = {k: [v, BM.verdict(v)] for k, v in out.items()}
    record(f"q_eta_rule|{law}|n{n}", out)
    if n >= 12:
        assert out["gl_in_theta"][0] < out["gl_in_eta"][0]
