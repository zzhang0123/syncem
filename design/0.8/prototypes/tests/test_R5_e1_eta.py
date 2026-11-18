"""R5 with an inclination parent on the E1 continuum kernel: route (iii) with eta averaged and
route (iv) with eta on Legendre(L), against direct integration over (gamma, eta, phi).

Route (iii): c(eta) is a weight inside the eta average. The full range [-0.5, 0.5] is
correct for any c. The half range [0, 0.5] is correct only for c even in eta (Sec. 4.3 (b)).
Route (iv): c(eta) multiplies the kernel inside the Legendre projection. Wrong route in both:
<c> times the bare average or projection.

Run from the prototypes directory (about 8 min):
PYTHONPATH=.:../../.. python -m pytest tests/test_R5_e1_eta.py -s
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import core as C
from rules.R5 import e1 as X
from rules.R5 import joint as J
from rules.R5 import routes as R

RESULTS = Path(__file__).with_name("R5_e1_eta_results.json")
FACTORS = {
    "even": lambda e: 0.6 * (1 - e ** 2),
    "odd_part": lambda e: 0.4 * (1 + 0.8 * e),
    "steep": lambda e: jax.nn.sigmoid(20 * e) * jnp.exp(0.5j),
}
STATE: dict = {}
LMAX, NLEG = 16, 64  # Legendre degree and projection nodes (64 resolve the steep factor)


def _record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


@pytest.fixture(scope="module")
def setup():
    if not STATE:
        K = X.base_kernel()
        rule_B = C.uniform(4.0, 6.0, 16)
        tab = X.table(2)
        STATE.update(K=K, rule_B=rule_B, tab=tab, w=X.E.e1_weights(tab), gm=X.gamma_measure(tab), cache={},
                     avg_d=C.AveragedKernel(K, [("B", rule_B)]), timer=X.Timer())
        STATE["S1_full"] = STATE["timer"].run("cols_S1_full", C.build_columns,
                                              C.AveragedKernel(K, [("B", rule_B), ("eta", C.uniform(-0.5, 0.5, 24))]),
                                              tab)
        STATE["bare_leg"] = STATE["timer"].run("cols_bare_legendre8", C.build_columns,
                                               C.AveragedKernel(K, [("B", rule_B)], legendre={"eta": (LMAX, NLEG)}), tab)
    return STATE


def _joint(s, c, lo, hi, n):
    m = C.product(s["gm"], C.panel_measure("eta", lambda x: 0 * x + 1 / (hi - lo), [lo, hi], n=n, chart="affine"))
    return m, J.with_phi(m, J.factor_law(lambda nd: np.asarray(c(jnp.asarray(nd["eta"])), complex)))


@pytest.mark.parametrize("name", list(FACTORS))
def test_route_iii_eta_averaged(setup, name):
    s, c = setup, FACTORS[name]
    tab, w = s["tab"], s["w"]
    a = C.pair(tab, s["gm"], w)
    m, joint = _joint(s, c, -0.5, 0.5, 32)
    ref = J.direct_joint(s["avg_d"], joint, tab, cache=s["cache"])
    full = C.AveragedKernel(R.FactorKernel(s["K"], lambda p: c(p["eta"])),
                            [("B", s["rule_B"]), ("eta", C.uniform(-0.5, 0.5, 24))])
    half = C.AveragedKernel(R.FactorKernel(s["K"], lambda p: c(p["eta"])),
                            [("B", s["rule_B"]), ("eta", C.uniform(0.0, 0.5, 12))])
    cbar = J.mean_factor(joint)
    p_full = C.predict(C.build_columns(full, tab), tab, a, w)
    p_wrong = R.assemble_constant(s["S1_full"], tab, a, w, cbar)
    out = {"cbar": [cbar.real, cbar.imag],
           "full_range": X.per_channel(p_full, ref),
           "half_range": X.per_channel(C.predict(C.build_columns(half, tab), tab, a, w), ref),
           "wrong_mean_c_times_S1": X.per_channel(p_wrong, ref),
           "wrong_minus_full_range": X.per_channel(p_wrong, p_full)}
    _record(f"iii_eta_{name}", out)
    assert out["full_range"]["max_rel_I"] < 2e-3
    if name == "even":
        assert out["half_range"]["max_rel_I"] < 2e-3
        assert out["wrong_mean_c_times_S1"]["max_rel_P"] > 3 * out["full_range"]["max_rel_P"]
    else:
        # c - <c> is odd in eta and K_P is even, so <c K_P> = <c><K_P> over the symmetric law:
        # the factorised route is exact here, and only the half range is wrong.
        assert out["half_range"]["max_rel_P"] > 5 * out["full_range"]["max_rel_P"]
        assert out["wrong_minus_full_range"]["max_rel_P"] < 1e-10


@pytest.mark.parametrize("name", ["odd_part", "steep"])
def test_route_iv_eta_legendre(setup, name):
    s, c = setup, FACTORS[name]
    tab, w = s["tab"], s["w"]
    lo, hi = -0.2, 0.5
    m, joint = _joint(s, c, lo, hi, 16)
    ref = J.direct_joint(s["avg_d"], joint, tab, cache=s["cache"])
    ok = C.build_columns(C.AveragedKernel(R.FactorKernel(s["K"], lambda p: c(p["eta"])), [("B", s["rule_B"])],
                                          legendre={"eta": (LMAX, NLEG)}), tab)
    cbar = J.mean_factor(joint)
    out = {"cbar": [cbar.real, cbar.imag]}
    for L in (0, 2, 4, 8, 16):
        a = C.pair(tab, m, w, legendre={"eta": L})
        a2 = C.pair(tab, joint, w, legendre={"eta": L}, harmonic=True)  # M2 = <c(eta) P_l(eta) u^r>
        sl = {k: v[..., : L + 1] for k, v in ok.items()}
        bare = {k: v[..., : L + 1] for k, v in s["bare_leg"].items()}
        out[f"L{L}"] = {"correct": X.per_channel(C.predict(sl, tab, a, w), ref),
                        "moment_side_harmonic": X.per_channel(C.predict(bare, tab, a2, w), ref),
                        "wrong_mean_c_after_projection": X.per_channel(
                            R.assemble_constant(bare, tab, a, w, cbar), ref)}
    if name == "odd_part":  # the r2-02 statement on the even mask, eta ~ U[0, 1/2], L = 8
        m2 = C.product(s["gm"], C.panel_measure("eta", lambda x: 0 * x + 2.0, [0.0, 0.5], n=16, chart="affine"))
        a8 = C.pair(tab, m2, w, legendre={"eta": 8})
        ok8 = {k: v[..., :9] for k, v in ok.items()}
        full = C.predict(ok8, tab, a8, w)
        masked = {k: v.copy() for k, v in ok8.items()}
        masked["P"][..., 1::2] = 0.0
        pm = C.predict(masked, tab, a8, w)
        drop = 1 - np.abs(X.stokes_P(pm)) / np.abs(X.stokes_P(full))
        out["even_mask_U0_half_L8"] = {"P_drop_min": float(drop.min()), "P_drop_max": float(drop.max()),
                                       "P_drop_median": float(np.median(drop))}
    _record(f"iv_eta_{name}", out)
    assert out["L16"]["correct"]["max_rel_I"] < out["L2"]["correct"]["max_rel_I"]  # converges in L
    if name == "odd_part":
        assert out["L16"]["correct"]["max_rel_P"] < out["L16"]["wrong_mean_c_after_projection"]["max_rel_P"]
    # c on the moment side truncates K only. Route (iv) truncates c K, which converges slowly for a steep c.
    assert out["L8"]["moment_side_harmonic"]["max_rel_P"] <= out["L8"]["correct"]["max_rel_P"] * 1.5
