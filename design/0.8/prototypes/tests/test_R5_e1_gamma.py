"""R5 (ii) with a resolved gamma parent on the E1 continuum kernel, against direct integration.

Correct route: per-cell Cauchy product of the bare spin-2 columns with the Taylor
coefficients of c(gamma), at assembly. Alternative correct route: phi resolved by the
harmonic, c on the moment side (M2). Wrong routes: one global <c>, and c at each cell's
reference point only. The reference integrates gamma (6 nodes per cell) and phi (two-point
law given gamma) jointly.

Run from the prototypes directory (about 4 min):
PYTHONPATH=.:../../.. python -m pytest tests/test_R5_e1_gamma.py -s
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import core as C
from core import paper as E
from rules.R5 import e1 as X
from rules.R5 import routes as R

RESULTS = Path(__file__).with_name("R5_e1_gamma_results.json")
LN = lambda g: jnp.log(g) - X.LN1E4

FACTORS = {
    "moderate": lambda p: 0.4 * jnp.exp(0.5j) * (1 + 0.2 * LN(p["gamma"]) / jnp.log(10.0)),
    "steep": lambda p: jax.nn.sigmoid(4 * LN(p["gamma"])),
    "phase": lambda p: 0.6 * jnp.exp(1j * LN(p["gamma"])),
    "near_zero": lambda p: 1e-3 * jnp.exp(0.5j) * (1 + 0.2 * LN(p["gamma"]) / jnp.log(10.0)),
    "near_one": lambda p: 0.999 * jnp.exp(0.3j * LN(p["gamma"])),
}
STATE: dict = {}


def _record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


@pytest.fixture(scope="module")
def e1():
    if not STATE:
        timer = X.Timer()
        avg = E.e1_kernel()
        big = X.table(4)
        STATE.update(avg=avg, big=big, cols=timer.run("cols_N4", C.build_columns, avg, big), cache={}, timer=timer)
        jet2 = C.build_columns(C.AveragedKernel(R.FactorKernel(avg.kernel, FACTORS["steep"]), avg.averaged), X.table(2))
        STATE["jet2_steep"] = jet2
    return STATE


@pytest.mark.parametrize("name", list(FACTORS))
def test_gamma_factor_cauchy_product(e1, name):
    jet = (2, e1["jet2_steep"]) if name == "steep" else None
    out = X.gamma_case(FACTORS[name], e1["cols"], e1["big"], (0, 2, 4), e1["cache"], e1["avg"], jet_check=jet)
    out["timings_s"] = e1["timer"].t
    _record(f"gamma_{name}", out)
    if jet is not None:
        assert out["N2"]["cauchy_vs_jet_of_factor_kernel"] < 1e-12
    for N in (2, 4):
        r = out[f"N{N}"]
        assert r["cauchy"]["max_rel_I"] < 1e-2  # converges with N, measured below
        assert r["wrong_global_mean"]["max_rel_P"] > 3 * r["cauchy"]["max_rel_P"]
    assert out["N4"]["cauchy"]["max_rel_I"] < out["N2"]["cauchy"]["max_rel_I"]
