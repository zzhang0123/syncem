"""Rule R5 routes (i) to (iv) on polynomial kernels, against direct integration of the joint measure.

The reference never uses the factor c. It integrates exp(2i phi) K_P over a joint measure in
which phi has its own nodes given the parents (rules/R5/joint.py). Kernels and factors are
polynomials, so the correct route is exact (to rounding) where the product c K lies in the
lower set or in the Legendre degrees, and every wrong route shows a finite difference.

Run from the prototypes directory (about 1 min):
PYTHONPATH=.:../../.. python -m pytest tests/test_R5_poly.py -s
"""
from __future__ import annotations

import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import core as C
from rules.R5 import joint as J
from rules.R5 import routes as R

RESULTS = Path(__file__).with_name("R5_poly_results.json")
EXACT = 1e-12  # threshold for routes that are exact by construction
DIFFERS = 1e-3  # a wrong route must differ by at least this much (relative to I)


def _record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


def _same(a, b):
    return np.max(np.abs(a - b)) <= 1e-14 * np.max(np.abs(b))


def _tg(g):  # chart variable of gamma used by every polynomial below
    return (jnp.log(g) - np.log(1e4)) / 2.0


def _gamma_measure(table, n=6):
    edges = np.unique(np.r_[table.lo[:, 0], table.hi[:, 0]])
    return C.panel_measure("gamma", lambda x: x ** -2.6, edges, n=n)


def _lower_poly(transforms, caps, total, seed, n_ch=5):
    rng = np.random.default_rng(seed)
    alphas = C.lower_set(caps, total)
    return C.PolynomialKernel(transforms, {a: rng.normal(size=n_ch) for a in alphas},
                              {a: rng.normal(size=n_ch) + 1j * rng.normal(size=n_ch) for a in alphas})


# ---------------------------------------------------------------- route (i)
@pytest.mark.parametrize("c", [0.0, 1e-9 * np.exp(0.3j), 0.4 * np.exp(0.5j), 0.999999 * np.exp(-2j), 1.0 + 0j],
                         ids=["zero", "1e-9", "paper", "near1", "one"])
def test_route_i_constant_at_assembly(c):
    tr = {"gamma": ("log", np.log(1e4), 2.0)}
    K = _lower_poly(tr, (2,), 2, seed=1)
    tab = C.ProductTable.tensor({"gamma": C.Cells.log(300, 3e6, 4, order=2)})
    avg = C.AveragedKernel(K)
    cols, w = C.build_columns(avg, tab), C.reference_weights(tab)
    m = _gamma_measure(tab)
    joint = J.with_phi(m, J.factor_law(lambda nd: np.full(nd["gamma"].size, c)))
    ref = J.direct_joint(avg, joint, tab)
    pred = R.assemble_constant(cols, tab, C.pair(tab, m, w), w, c)
    err = C.representation_error(pred, ref)
    _record(f"i_constant_{c!r}", err)
    assert err < EXACT


# ---------------------------------------------------------------- route (ii): Leibniz identity
TR3 = {"gamma": ("log", np.log(1e4), 2.0), "B": ("affine", 5.0, 2.0), "depth": ("affine", 8.0, 1.0)}

FACTORS = {
    "smooth": lambda p: 0.4 * jnp.exp(0.5j) * (1 + 0.3 * jnp.tanh(_tg(p["gamma"]) / 3))
    * jnp.exp(0.01j * p.get("depth", 0.0)) * (1 + 0.1 * jnp.tanh((p.get("B", 5.0) - 5) / 10)),
    "steep": lambda p: jax.nn.sigmoid(16 * _tg(p["gamma"]))  # 0.5 (1 + tanh(8 t)) without cancellation
    * jnp.exp(0.2j * p.get("depth", 0.0))
    * (1 + 0.0 * p.get("B", 5.0)),
    "tiny": lambda p: 1e-8 * jnp.exp(1j * _tg(p["gamma"]) + 0.05j * p.get("depth", 0.0) + 0.0 * p.get("B", 5.0)),
}


def _axes(variables, caps, extreme):
    if extreme:
        allax = {"gamma": C.Cells.log(1e7, 1e9, 2, order=caps[0]), "B": C.Cells.single(500.0, 50.0, order=caps[1]),
                 "depth": C.Cells.linear(-300, 300, 2, order=caps[2])}
    else:
        allax = {"gamma": C.Cells.log(300, 2.9626e6, 3, order=caps[0]), "B": C.Cells.linear(2, 8, 2, order=caps[1]),
                 "depth": C.Cells.from_edges([6.0, 10.0], chart="affine", order=caps[2])}
    return {v: allax[v] for v in variables}


@pytest.mark.parametrize("extreme", [False, True], ids=["moderate", "extreme"])
@pytest.mark.parametrize("factor", list(FACTORS))
@pytest.mark.parametrize("variables,caps,total", [
    (("gamma",), (4, 0, 0), 4),
    (("gamma", "depth"), (2, 0, 2), 2),
    (("gamma", "B", "depth"), (3, 3, 3), 3),
    (("gamma", "B", "depth"), (2, 1, 2), 2),
], ids=["g4", "gd2", "gBd3", "gBd_caps212"])
def test_route_ii_cauchy_equals_jet_of_factor_kernel(variables, caps, total, factor, extreme):
    """Leibniz: the Cauchy product of the bare columns and the Taylor coefficients of c equals the
    jet of c K_P on every row of the lower set, in 1 to 3 celled variables, depth included."""
    K = C.PolynomialKernel.random({v: TR3[v] for v in variables}, 4, 5, seed=len(variables))
    tab = C.ProductTable.tensor(_axes(variables, caps, extreme), total=total)
    c = FACTORS[factor]
    bare = C.build_columns(C.AveragedKernel(K), tab)
    jet = C.build_columns(C.AveragedKernel(R.FactorKernel(K, c)), tab)
    cau = R.cauchy_product(bare["P"], tab, R.factor_taylor(tab, c))
    o, worst = tab.offsets, 0.0
    for cell in range(tab.n_cells):
        s = slice(o[cell], o[cell + 1])
        scale = np.max(np.abs(jet["P"][s])) or 1.0
        worst = max(worst, np.max(np.abs(cau[s] - jet["P"][s])) / scale)
    assert _same(bare["I"], jet["I"])  # c never touches the spin-0 block
    _record(f"ii_leibniz_{'-'.join(variables)}_caps{caps}_t{total}_{factor}_{'ext' if extreme else 'mod'}", worst)
    assert worst < EXACT


# ---------------------------------------------------------------- route (ii): predict vs direct
TRGD = {"gamma": ("log", np.log(1e4), 2.0), "depth": ("affine", 0.0, 10.0)}


def _c_gd(p):
    tg, td = _tg(p["gamma"]), p["depth"] / 10.0
    return (0.3 + 0.1j) + 0.08 * tg + 0.05j * td + 0.02 * tg * td + 0.01 * tg ** 2


def test_route_ii_predict_equals_direct_and_wrong_routes_differ():
    """c(gamma, depth) of degree 2 times K_P of degree 1 lies in the lower set (caps 3, total 3)."""
    K = _lower_poly(TRGD, (1, 1), 1, seed=5)
    tab = C.ProductTable.tensor({"gamma": C.Cells.log(300, 3e6, 3, order=3),
                                 "depth": C.Cells.linear(-10, 10, 2, order=3)}, total=3)
    avg = C.AveragedKernel(K)
    cols, w = C.build_columns(avg, tab), C.reference_weights(tab)
    m = C.product(_gamma_measure(tab), C.panel_measure("depth", lambda x: 0 * x + 0.05, [-10, 0, 10], n=6,
                                                        chart="affine"))
    c_np = lambda nd: np.asarray(_c_gd({k: jnp.asarray(v) for k, v in nd.items()}))
    joint = J.with_phi(m, J.factor_law(c_np))
    ref = J.direct_joint(avg, joint, tab)
    a = C.pair(tab, m, w)
    ctay = R.factor_taylor(tab, _c_gd)
    out = {
        "cauchy": C.representation_error(R.predict_with_P(cols, R.cauchy_product(cols["P"], tab, ctay), tab, a, w), ref),
        "harmonic_M2": C.representation_error(C.predict(cols, tab, C.pair(tab, joint, w, harmonic=True), w), ref),
        "wrong_global_mean": C.representation_error(R.assemble_constant(cols, tab, a, w, J.mean_factor(joint)), ref),
        "wrong_zeroth_order": C.representation_error(
            R.predict_with_P(cols, R.zeroth_order_factor(cols["P"], tab, ctay), tab, a, w), ref),
    }
    _record("ii_predict_vs_direct_gamma_depth", out)
    assert out["cauchy"] < EXACT and out["harmonic_M2"] < EXACT
    assert out["wrong_global_mean"] > DIFFERS and out["wrong_zeroth_order"] > DIFFERS


# ---------------------------------------------------------------- route (iii): averaged B parent
TRGB = {"gamma": ("log", np.log(1e4), 2.0), "B": ("affine", 5.0, 1.0)}

C_OF_B = {
    "paper": lambda B: 0.4 * (1 + 0.5 * (B - 5)),
    "zero_mean": lambda B: 0.4 * (B - 5),
    "near_one": lambda B: 0.995 + 0.005 * (B - 5),
    "strong": lambda B: 0.5 + 0.5 * (B - 5),
    "phase": lambda B: 0.6 * jnp.exp(1j * jnp.pi * (B - 5)),
}


@pytest.mark.parametrize("name", list(C_OF_B))
def test_route_iii_factor_inside_field_average(name):
    cB = C_OF_B[name]
    K = _lower_poly(TRGB, (2, 3), 5, seed=7)  # degree 2 in gamma, 3 in B
    tab = C.ProductTable.tensor({"gamma": C.Cells.log(300, 3e6, 4, order=2)})
    rule = C.uniform(4.0, 6.0, 16)
    avg_ok = C.AveragedKernel(R.FactorKernel(K, lambda p: cB(p["B"])), [("B", rule)])
    avg_s1 = C.AveragedKernel(K, [("B", rule)])  # the cached field average S1, no factor
    cols_ok, cols_s1, w = C.build_columns(avg_ok, tab), C.build_columns(avg_s1, tab), C.reference_weights(tab)
    m = C.product(_gamma_measure(tab), C.panel_measure("B", lambda x: 0 * x + 0.5, [4, 6], n=24, chart="affine"))
    joint = J.with_phi(m, J.factor_law(lambda nd: np.asarray(cB(jnp.asarray(nd["B"])))))
    ref = J.direct_joint(C.AveragedKernel(K), joint, tab)
    a = C.pair(tab, _gamma_measure(tab), w)
    ok = C.representation_error(C.predict(cols_ok, tab, a, w), ref)
    wrong = C.representation_error(R.assemble_constant(cols_s1, tab, a, w, J.mean_factor(joint)), ref)
    _record(f"iii_B_{name}", {"correct": ok, "wrong_mean_c_times_S1": wrong})
    assert _same(cols_ok["I"], cols_s1["I"])  # the spin-0 block keeps the cached S1
    assert ok < EXACT
    assert wrong > DIFFERS


def test_route_iii_joint_block_B_phi():
    """Joint (B, phi) block: phi given B uniform on [psi(B) - h(B), psi(B) + h(B)]. The factor at
    each field node is E[exp(2i phi) | B_q] of that law."""
    K = _lower_poly(TRGB, (2, 3), 5, seed=8)
    psi = lambda B: 0.25 + 0.4 * (B - 5)
    half = lambda B: 0.6 + 0.3 * (B - 5)
    cB = lambda B: J.uniform_phi_factor(psi(B), half(B))
    tab = C.ProductTable.tensor({"gamma": C.Cells.log(300, 3e6, 4, order=2)})
    rule = C.uniform(4.0, 6.0, 16)
    cjnp = lambda p: jnp.exp(2j * psi(p["B"])) * jnp.sinc(2 * half(p["B"]) / jnp.pi)
    avg_ok = C.AveragedKernel(R.FactorKernel(K, cjnp), [("B", rule)])
    cols_ok, cols_s1 = C.build_columns(avg_ok, tab), C.build_columns(C.AveragedKernel(K, [("B", rule)]), tab)
    w = C.reference_weights(tab)
    m = C.product(_gamma_measure(tab), C.panel_measure("B", lambda x: 0 * x + 0.5, [4, 6], n=24, chart="affine"))
    joint = J.with_phi(m, lambda nd: J.uniform_phi(psi(nd["B"]), half(nd["B"]), n=12))
    ref = J.direct_joint(C.AveragedKernel(K), joint, tab)
    a = C.pair(tab, _gamma_measure(tab), w)
    ok = C.representation_error(C.predict(cols_ok, tab, a, w), ref)
    wrong = C.representation_error(R.assemble_constant(cols_s1, tab, a, w, J.mean_factor(joint)), ref)
    _record("iii_joint_block_B_phi", {"correct": ok, "wrong_mean_c_times_S1": wrong,
                                      "factor_from_rule_vs_analytic": float(np.max(np.abs(
                                          np.sum(np.exp(2j * J.uniform_phi(np.r_[4.3, 5.7], np.r_[0.5, 0.8], 12)[0])
                                                 * J.uniform_phi(np.r_[4.3, 5.7], np.r_[0.5, 0.8], 12)[1], axis=1)
                                          - J.uniform_phi_factor(np.r_[4.3, 5.7], np.r_[0.5, 0.8]))))})
    assert ok < 1e-10  # the 12-node phi rule integrates exp(2i phi) to about 1e-15
    assert wrong > DIFFERS


# ---------------------------------------------------------------- route (iii): averaged eta parent, parity
TRGE = {"gamma": ("log", np.log(1e4), 2.0), "eta": ("affine", 0.0, 1.0)}


@pytest.mark.parametrize("parity", ["even", "odd_part"])
def test_route_iii_eta_parent_full_and_half_range(parity):
    """K even in eta. c(eta) even keeps the half range exact. c(eta) with an odd part needs the
    full range: on the half range the result is wrong (Sec. 4.3 check (b))."""
    rng = np.random.default_rng(11)
    alphas = [(a, b) for a in range(3) for b in (0, 2, 4)]  # even in eta
    K = C.PolynomialKernel(TRGE, {a: rng.normal(size=4) for a in alphas},
                           {a: rng.normal(size=4) + 1j * rng.normal(size=4) for a in alphas})
    c = (lambda p: 0.6 * (1 - p["eta"] ** 2)) if parity == "even" else (lambda p: 0.4 * (1 + 0.8 * p["eta"]))
    tab = C.ProductTable.tensor({"gamma": C.Cells.log(300, 3e6, 4, order=2)})
    full = C.AveragedKernel(R.FactorKernel(K, c), [("eta", C.uniform(-0.5, 0.5, 16))])
    half = C.AveragedKernel(R.FactorKernel(K, c), [("eta", C.uniform(0.0, 0.5, 8))])
    w = C.reference_weights(tab)
    m = C.product(_gamma_measure(tab), C.panel_measure("eta", lambda x: 0 * x + 1.0, [-0.5, 0.5], n=16, chart="affine"))
    joint = J.with_phi(m, J.factor_law(lambda nd: np.asarray(c({"eta": nd["eta"]}), complex)))
    ref = J.direct_joint(C.AveragedKernel(K), joint, tab)
    a = C.pair(tab, _gamma_measure(tab), w)
    e_full = C.representation_error(C.predict(C.build_columns(full, tab), tab, a, w), ref)
    e_half = C.representation_error(C.predict(C.build_columns(half, tab), tab, a, w), ref)
    _record(f"iii_eta_{parity}", {"full_range": e_full, "half_range": e_half})
    assert e_full < EXACT
    assert (e_half < EXACT) if parity == "even" else (e_half > DIFFERS)


# ---------------------------------------------------------------- route (iv): Legendre parent
@pytest.mark.parametrize("L", [3, 4, 5, 8])
def test_route_iv_factor_inside_legendre_projection(L):
    """K of degree 3 in eta, c(eta) of degree 2: c K has degree 5, so L >= 5 is exact."""
    rng = np.random.default_rng(13)
    alphas = [(a, b) for a in range(3) for b in range(4)]
    K = C.PolynomialKernel(TRGE, {a: rng.normal(size=4) for a in alphas},
                           {a: rng.normal(size=4) + 1j * rng.normal(size=4) for a in alphas})
    c = lambda p: (0.35 + 0.1j) + 0.3 * p["eta"] - 0.2 * p["eta"] ** 2
    tab = C.ProductTable.tensor({"gamma": C.Cells.log(300, 3e6, 4, order=2)})
    w = C.reference_weights(tab)
    avg_ok = C.AveragedKernel(R.FactorKernel(K, c), legendre={"eta": (L, 12)})
    avg_bare = C.AveragedKernel(K, legendre={"eta": (L, 12)})
    m = C.product(_gamma_measure(tab), C.panel_measure("eta", lambda x: 0 * x + 1 / 0.7, [-0.2, 0.5], n=16,
                                                        chart="affine"))
    joint = J.with_phi(m, J.factor_law(lambda nd: np.asarray(c({"eta": nd["eta"]}), complex)))
    ref = J.direct_joint(avg_bare, joint, tab)
    a = C.pair(tab, m, w, legendre={"eta": L})
    ok = C.representation_error(C.predict(C.build_columns(avg_ok, tab), tab, a, w), ref)
    wrong = C.representation_error(R.assemble_constant(C.build_columns(avg_bare, tab), tab, a, w,
                                                       J.mean_factor(joint)), ref)
    _record(f"iv_legendre_L{L}", {"correct": ok, "wrong_mean_c_after_projection": wrong})
    assert (ok < EXACT) if L >= 5 else (ok > 1e-8)  # truncation at L < 5 is measured, not hidden
    assert wrong > DIFFERS


# ---------------------------------------------------------------- mixed parents: celled gamma, averaged B
def test_mixed_celled_and_averaged_parents():
    """c(gamma, B) with gamma on Cells and B averaged. Correct: c inside the field average with its
    gamma derivatives (per-node Cauchy product, here by the jet). Wrong (a): c frozen at the
    cell's reference gamma. Wrong (b): Cauchy product of the field-averaged columns with the
    Taylor coefficients of <c>_B(gamma). Wrong (c): one global <c>."""
    K = _lower_poly(TRGB, (1, 3), 4, seed=17)
    cgB = lambda p: 0.3 + 0.1 * _tg(p["gamma"]) + 0.1 * (p["B"] - 5) + 0.05j * _tg(p["gamma"]) * (p["B"] - 5)
    tab = C.ProductTable.tensor({"gamma": C.Cells.log(300, 3e6, 4, order=2)})
    rule = C.uniform(4.0, 6.0, 16)
    w = C.reference_weights(tab)
    cols_ok = C.build_columns(C.AveragedKernel(R.FactorKernel(K, cgB), [("B", rule)]), tab)
    cols_fz = C.build_columns(C.AveragedKernel(R.FactorKernel(K, cgB, frozen=("gamma",)), [("B", rule)]), tab)
    cols_s1 = C.build_columns(C.AveragedKernel(K, [("B", rule)]), tab)
    cbar_g = lambda p: 0.3 + 0.1 * _tg(p["gamma"])  # <c(gamma, B)>_B for B ~ U[4, 6]
    cols_fb = R.cauchy_product(cols_s1["P"], tab, R.factor_taylor(tab, cbar_g))
    m = C.product(_gamma_measure(tab), C.panel_measure("B", lambda x: 0 * x + 0.5, [4, 6], n=24, chart="affine"))
    c_np = lambda nd: np.asarray(cgB({k: jnp.asarray(v) for k, v in nd.items()}))
    joint = J.with_phi(m, J.factor_law(c_np))
    ref = J.direct_joint(C.AveragedKernel(K), joint, tab)
    a = C.pair(tab, _gamma_measure(tab), w)
    out = {
        "correct_jet_inside_average": C.representation_error(C.predict(cols_ok, tab, a, w), ref),
        "wrong_a_frozen_gamma": C.representation_error(C.predict(cols_fz, tab, a, w), ref),
        "wrong_b_cauchy_of_field_mean": C.representation_error(R.predict_with_P(cols_s1, cols_fb, tab, a, w), ref),
        "wrong_c_global_mean": C.representation_error(R.assemble_constant(cols_s1, tab, a, w,
                                                                          J.mean_factor(joint)), ref),
    }
    _record("mixed_gamma_cells_B_averaged", out)
    assert out["correct_jet_inside_average"] < EXACT
    assert min(out["wrong_a_frozen_gamma"], out["wrong_b_cauchy_of_field_mean"], out["wrong_c_global_mean"]) > DIFFERS
