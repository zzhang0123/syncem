"""R5 (iii) with an averaged depth parent on the E1 continuum kernel, against direct integration.

depth ~ N(8, 0.5) rad m^-2 by 60 Gauss-Hermite nodes, B = 5 uG fixed, eta on [0, 0.5]
(6 nodes). Correct route: c(depth) is a weight inside the depth average, with the Faraday
phase exp(2i lam^2 depth) at each node (R4: depth is then averaged by its nodes, not by the
characteristic function). Wrong route: <c> times the depth-averaged kernel. The reference
integrates gamma, depth (the same 60 nodes) and phi (two-point law given depth) jointly.
The lowest channel (0.1 GHz) is the Faraday-depolarised end.

Run from the prototypes directory (about 5 min):
PYTHONPATH=.:../../.. python -m pytest tests/test_R5_e1_depth.py -s
"""
from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import core as C
from rules.R5 import e1 as X
from rules.R5 import joint as J
from rules.R5 import routes as R

RESULTS = Path(__file__).with_name("R5_e1_depth_results.json")
FACTORS = {
    "phase_slow": lambda d: 0.4 * jnp.exp(0.5j * (d - 8)),
    "phase_fast": lambda d: 0.4 * jnp.exp(3j * (d - 8)),
    "amplitude": lambda d: 0.4 * (1 + 0.5 * jnp.tanh(2 * (d - 8))),
    "near_one": lambda d: 0.999 * jnp.exp(0.2j * (d - 8)),
    "near_zero": lambda d: 1e-4 * jnp.exp(0.5j * (d - 8)),
}
STATE: dict = {}


def _record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


@pytest.fixture(scope="module")
def setup():
    if not STATE:
        K = X.base_kernel(screen=None)
        Kscr = X.base_kernel()
        d, wd = X.gh_rule(8.0, 0.5, 60)
        rule_d, eta = C.fixed(d, wd), X.eta_half_rule(6)
        tab = X.table(2)
        timer = X.Timer()
        STATE.update(K=K, rule_d=rule_d, eta=eta, tab=tab, w=X.E.e1_weights(tab), gm=X.gamma_measure(tab),
                     dm=C.Measure({"depth": d}, wd), cache={}, timer=timer,
                     avg_d=C.AveragedKernel(K, [("eta", eta)], fixed={"B": 5.0}))
        STATE["S1_nodes"] = timer.run("cols_depth_nodes", C.build_columns, C.AveragedKernel(
            K, [("depth", rule_d), ("eta", eta)], fixed={"B": 5.0}), tab)
        STATE["S1_charfn"] = timer.run("cols_charfn", C.build_columns, C.AveragedKernel(
            Kscr, [("eta", eta)], fixed={"B": 5.0}), tab)
    return STATE


@pytest.mark.parametrize("name", list(FACTORS))
def test_depth_factor_inside_depth_average(setup, name):
    s, c = setup, FACTORS[name]
    tab, w = s["tab"], s["w"]
    a = C.pair(tab, s["gm"], w)
    m = C.product(s["gm"], s["dm"])
    joint = J.with_phi(m, J.factor_law(lambda nd: np.asarray(c(jnp.asarray(nd["depth"])), complex)))
    ref = J.direct_joint(s["avg_d"], joint, tab, cache=s["cache"])
    ok = C.build_columns(C.AveragedKernel(R.FactorKernel(s["K"], lambda p: c(p["depth"])),
                                          [("depth", s["rule_d"]), ("eta", s["eta"])], fixed={"B": 5.0}), tab)
    cbar = J.mean_factor(joint)
    out = {"cbar": [cbar.real, cbar.imag],
           "correct": X.per_channel(C.predict(ok, tab, a, w), ref),
           "wrong_mean_c_times_depth_average": X.per_channel(R.assemble_constant(s["S1_nodes"], tab, a, w, cbar), ref),
           "wrong_mean_c_times_characteristic_function": X.per_channel(
               R.assemble_constant(s["S1_charfn"], tab, a, w, cbar), ref)}
    if name == "phase_slow":  # depth nodes vs characteristic function, constant factor, relative to I
        out["nodes_vs_charfn_constant_c"] = X.per_channel(C.predict(s["S1_nodes"], tab, a, w, azimuth=0.4),
                                                          C.predict(s["S1_charfn"], tab, a, w, azimuth=0.4))
    out["timings_s"] = s["timer"].t
    _record(f"depth_{name}", out)
    assert out["correct"]["max_rel_I"] < 2e-3
    assert out["wrong_mean_c_times_depth_average"]["max_rel_P"] > 5 * out["correct"]["max_rel_P"]


@pytest.mark.parametrize("a", [0.5, 3.0], ids=["phase_slow", "phase_fast"])
def test_depth_exponential_factor_exact_reference(setup, a):
    """c(depth) = 0.4 exp(i a (depth - 8)) has the exact joint average
    0.4 exp(16 i lam^2) exp(-(a + 2 lam^2)^2 sd^2 / 2). Against it, the node route and the
    factorised route are measured at the Faraday-depolarised end, relative to |P|."""
    s = setup
    tab, w = s["tab"], s["w"]
    a_ = C.pair(tab, s["gm"], w)
    Kx = X.base_kernel(screen=None)
    Kx.phase = 0.4 * jnp.exp(16j * Kx.lam2 - 0.5 * (a + 2 * Kx.lam2) ** 2 * 0.25)
    exact = J.direct_joint(C.AveragedKernel(Kx, [("eta", s["eta"])], fixed={"B": 5.0}), s["gm"], tab)
    c = lambda d: 0.4 * jnp.exp(1j * a * (d - 8))
    ok = C.build_columns(C.AveragedKernel(R.FactorKernel(s["K"], lambda p: c(p["depth"])),
                                          [("depth", s["rule_d"]), ("eta", s["eta"])], fixed={"B": 5.0}), tab)
    cbar = 0.4 * np.exp(-0.5 * a**2 * 0.25)
    out = {"node_route_vs_exact": X.per_channel(C.predict(ok, tab, a_, w), exact),
           "wrong_mean_c_times_charfn_vs_exact": X.per_channel(R.assemble_constant(s["S1_charfn"], tab, a_, w, cbar),
                                                               exact)}
    _record(f"depth_exact_reference_a{a}", out)
    assert out["node_route_vs_exact"]["max_rel_I"] < 2e-3
    assert out["wrong_mean_c_times_charfn_vs_exact"]["max_rel_P"] > 0.1
