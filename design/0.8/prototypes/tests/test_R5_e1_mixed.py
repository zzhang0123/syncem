"""R5 with a celled parent and an averaged parent together: c(gamma, B), gamma on 19 cells,
B ~ U[4, 6] uG averaged, on the E1 continuum kernel, against direct integration.

Correct: c is a weight inside the field average and its gamma derivatives combine with the
kernel's per field node (the jet of the weighted average, equal to a per-node Cauchy product).
Wrong (a): c(gamma_b, B_q) frozen at the cell's reference energy. Wrong (b): Cauchy product of
the field-averaged columns with the Taylor coefficients of <c>_B(gamma). Wrong (c): one global <c>.

Run from the prototypes directory (about 3 min):
PYTHONPATH=.:../../.. python -m pytest tests/test_R5_e1_mixed.py -s
"""
from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np

import core as C
from rules.R5 import e1 as X
from rules.R5 import joint as J
from rules.R5 import routes as R

RESULTS = Path(__file__).with_name("R5_e1_mixed_results.json")


def c_gB(p):
    ln = jnp.log(p["gamma"]) - X.LN1E4  # phase turns by one radian per e-fold, faster at high B
    return 0.4 * (1 + 0.5 * (p["B"] - 5)) * jnp.exp(1j * ln * (1 + 0.3 * (p["B"] - 5)))


def cbar_g(p):  # <c(gamma, B)>_B for B ~ U[4, 6], by 32 Gauss-Legendre nodes
    t, wt = np.polynomial.legendre.leggauss(32)
    return sum(0.5 * wi * c_gB({"gamma": p["gamma"], "B": 5.0 + ti}) for ti, wi in zip(t, wt))


def test_mixed_gamma_cells_B_averaged():
    K = X.base_kernel()
    eta, rule = X.eta_half_rule(), C.uniform(4.0, 6.0, 16)
    tab = X.table(2)
    w, gm = X.E.e1_weights(tab), X.gamma_measure(tab)
    timer = X.Timer()
    ok = timer.run("cols_ok", C.build_columns, C.AveragedKernel(R.FactorKernel(K, c_gB), [("B", rule), ("eta", eta)]), tab)
    fz = timer.run("cols_frozen", C.build_columns, C.AveragedKernel(R.FactorKernel(K, c_gB, frozen=("gamma",)),
                                                                    [("B", rule), ("eta", eta)]), tab)
    s1 = timer.run("cols_S1", C.build_columns, C.AveragedKernel(K, [("B", rule), ("eta", eta)]), tab)
    m = C.product(gm, C.panel_measure("B", lambda x: 0 * x + 0.5, [4.0, 6.0], n=24, chart="affine"))
    joint = J.with_phi(m, J.factor_law(lambda nd: np.asarray(c_gB({k: jnp.asarray(v) for k, v in nd.items()}),
                                                             complex)))
    ref = timer.run("direct_joint", J.direct_joint, C.AveragedKernel(K, [("eta", eta)]), joint, tab)
    a = C.pair(tab, gm, w)
    fb = R.cauchy_product(s1["P"], tab, R.factor_taylor(tab, cbar_g))
    out = {"correct_jet_inside_average": X.per_channel(C.predict(ok, tab, a, w), ref),
           "wrong_a_frozen_gamma": X.per_channel(C.predict(fz, tab, a, w), ref),
           "wrong_b_cauchy_of_field_mean": X.per_channel(R.predict_with_P(s1, fb, tab, a, w), ref),
           "wrong_c_global_mean": X.per_channel(R.assemble_constant(s1, tab, a, w, J.mean_factor(joint)), ref),
           "timings_s": timer.t}
    out_path = RESULTS
    out_path.write_text(json.dumps(out, indent=1, default=float))
    e_ok = out["correct_jet_inside_average"]["max_rel_I"]
    assert e_ok < 2e-3
    for k in ("wrong_a_frozen_gamma", "wrong_b_cauchy_of_field_mean", "wrong_c_global_mean"):
        assert out[k]["max_rel_P"] > 3 * out["correct_jet_inside_average"]["max_rel_P"]
