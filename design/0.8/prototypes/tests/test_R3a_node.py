"""R3a node: ingredients and exactness cases (Spec Sec. 5.2 R3a, r3-01).

Run from the prototypes directory:
PYTHONPATH=.:rules:../../.. python -m pytest tests/test_R3a_node.py
Numbers go to tests/R3a_results.json.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import integrate

import core as C
from R3a import conditional as Q
from R3a import node as R
from R3a import setups as T

RESULTS = Path(__file__).with_name("R3a_results.json")


def _record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


# ------------------------------------------------------------------ g, the child-cell moments
@pytest.mark.parametrize("chart", ["affine", "log"])
@pytest.mark.parametrize("sigma", [0.01, 0.2, 3.0])
def test_g_exact_vs_adaptive_quadrature(chart, sigma):
    cond = Q.LogNormalConditional(5.0, 3e4, 0.3, sigma)
    cells = T.child_cells(chart, 2)
    lo, hi, ref, hw = Q.child_bounds(cells)
    pt = {"gamma": 4.4e4}  # median 5.6 uG
    G = np.asarray(cond.g(pt, chart, cells, 4))
    worst = 0.0
    for c in range(lo.size):
        for s in range(5):
            x = (lambda B: np.log(B / ref[c]) / hw[c]) if chart == "log" else (lambda B: (B - ref[c]) / hw[c])
            pts = [float(np.exp(cond.mu(pt)))] if lo[c] < np.exp(cond.mu(pt)) < hi[c] else None
            v = integrate.quad(lambda B: float(cond.density(B, pt)) * x(B) ** s, lo[c], hi[c], points=pts,
                               epsabs=1e-15, epsrel=1e-13, limit=500)[0]
            worst = max(worst, abs(v - G[c, s]))
    _record(f"g_exact_vs_quad_{chart}_sigma{sigma}", worst)
    assert worst < 1e-11


@pytest.mark.parametrize("chart", ["affine", "log"])
@pytest.mark.parametrize("sigma", [0.003, 0.2, 3.0])
def test_h_analytic_vs_jax_jet(chart, sigma):
    """Hermite-derivative h in NumPy against jax.experimental.jet of the closed-form g, order 3."""
    case = T.make_case(3, 2, chart, cells=(9,))  # median 5 uG at the cell reference, nonzero g
    cond = Q.LogNormalConditional(5.0, float(case.parent.reference[0, 0]), 0.3, sigma)
    ha, _ = R.taylor_h(cond, case.parent, ("gamma",), case.child_chart, case.child, 2, 3, deriv="analytic")
    hj, _ = R.taylor_h(cond, case.parent, ("gamma",), case.child_chart, case.child, 2, 3, deriv="jet")
    rel = np.abs(ha - hj).max() / max(np.abs(hj).max(), 1e-300)
    _record(f"h_analytic_vs_jet_{chart}_sigma{sigma}", rel)
    assert rel < 1e-10


def test_h_jacfwd_equals_jet_order2():
    """The Spec's route (nested jax.jacfwd, core.jet) gives the same h as Taylor mode."""
    case = T.make_case(2, 2, "log", cells=(9,))
    cond = Q.LogNormalConditional(5.0, float(case.parent.reference[0, 0]), 0.5, 0.2)
    hf, _ = R.taylor_h(cond, case.parent, ("gamma",), "log", case.child, 2, 2, deriv="jacfwd")
    hj, _ = R.taylor_h(cond, case.parent, ("gamma",), "log", case.child, 2, 2, deriv="jet")
    rel = np.abs(hf - hj).max() / np.abs(hj).max()
    _record("h_jacfwd_vs_jet_log_order2", rel)
    assert rel < 1e-12


@pytest.mark.parametrize("chart", ["affine", "log", "log_full"])
@pytest.mark.parametrize("sigma", [0.05, 0.3])
def test_h_quadrature_on_fixed_nodes_vs_moving_nodes(chart, sigma):
    """Which quadrature of g may be differentiated. Fixed nodes on the child cell with density
    weights give the exact h. The conditional's own moving nodes times the cell indicator drop
    the boundary flux at child edges (and the primal is a step-function quadrature). On a child
    cell without edges the moving nodes are exact."""
    case = T.make_case(2, 2, chart, cells=(9,))
    cond = Q.LogNormalConditional(5.0, float(case.parent.reference[0, 0]), 0.5, sigma)
    ha, _ = R.taylor_h(cond, case.parent, ("gamma",), case.child_chart, case.child, 2, 2, deriv="analytic")
    hm, _ = R.taylor_h(cond, case.parent, ("gamma",), case.child_chart, case.child, 2, 2, method="moving",
                       deriv="jet", n=64)
    sc = np.abs(ha[..., 0]).max()
    moving = [float(np.abs(hm[..., k] - ha[..., k]).max() / sc) for k in range(3)]
    rec = {"moving_by_k": moving}
    if chart != "log_full":
        hf, _ = R.taylor_h(cond, case.parent, ("gamma",), case.child_chart, case.child, 2, 2, method="fixed",
                           deriv="jet", n=64)
        rec["fixed"] = float(np.abs(hf - ha).max() / sc)
        assert rec["fixed"] < 1e-12
        assert max(moving[1:]) > 1e-3  # BAD or WARN by the boundary rule: the flux term is missing
    else:
        assert max(moving) < 1e-12
    _record(f"h_fixed_vs_moving_{chart}_sigma{sigma}", rec)


# ------------------------------------------------------------------ exactness cases of the node
@pytest.mark.parametrize("chart", ["affine", "log"])
@pytest.mark.parametrize("cell", [0, 18])
@pytest.mark.parametrize("sigma", [0.003, 3.0])
def test_independent_child_is_exact(chart, cell, sigma):
    """beta = 0: g does not depend on gamma, h_k = 0 for k > 0, and the node is exact."""
    case = T.make_case(4, 2, chart, cells=(cell,))
    cond = Q.LogNormalConditional(4.0, float(case.parent.reference[0, 0]), 0.0, sigma)
    out = T.node_vs_nested(cond, case, "pl2.6")
    _record(f"independent_exact_{chart}_cell{cell}_sigma{sigma}", float(out["err"].max()))
    assert out["err"].max() < 1e-13


@pytest.mark.parametrize("caps,total", [((1, 1), 1), ((2, 2), 2), ((4, 2), 4), ((6, 2), 6), ((6, 4), 6),
                                        ((1, 2), 2), ((2, 2), 3), ((3, 1), 3)])
@pytest.mark.parametrize("beta,sigma", [(0.1, 0.2), (2.0, 0.003), (2.0, 3.0)])
def test_polynomial_g_exact_rows(caps, total, beta, sigma):
    """Child cell without edges (log chart): g_s is a polynomial of degree s in u_gamma, so the
    node is exact exactly on the rows with r + s <= N_gamma (Spec: r <= N_b - K, K = s here)."""
    case = T.make_case(caps[0], caps[1], "log_full", total=total, cells=(0, 9, 18))
    cond = Q.LogNormalConditional(5.0, 3e4, beta, sigma)
    # Taylor mode (jet): the Hermite-integral h of taylor_u loses digits as (beta h_b / sigma)^k / k!
    out = T.node_vs_nested(cond, case, "pl2.6", deriv="jet")
    rows = out["rows"]
    exact = rows[:, 0] + rows[:, 1] <= caps[0]
    # x = ln(B / 5) / 0.3 reaches |x| ~ 30 when beta = 2, so rows reach 1e3 per unit mass
    err = np.abs(out["a_node"] - out["a_true"]) / np.maximum(out["scale"], np.abs(out["a_true"]))
    e_in = float(err[exact].max())
    e_out = float(err[~exact].max()) if np.any(~exact) else None
    _record(f"polynomial_g_caps{caps}_total{total}_beta{beta}_sigma{sigma}", {"exact_rows": e_in,
                                                                             "other_rows": e_out})
    assert e_in < 1e-12
    if e_out is not None:
        assert e_out > 1e-10


@pytest.mark.parametrize("kappa", [0.0, 0.3, 1.2])
@pytest.mark.parametrize("caps,total", [((2, 2), 2), ((4, 3), 4)])
def test_uniform_shift_inside_one_cell_exact(kappa, caps, total):
    """B ~ U[c(gamma) - 0.3, c(gamma) + 0.3] inside the middle affine cell over the parent cell:
    g_s is a polynomial of degree s, and with h from the Spec's jacfwd route the node is exact."""
    case = T.make_case(caps[0], caps[1], "affine", total=total, cells=(9,))
    gp = float(case.parent.reference[0, 0])
    hb = float(case.parent.half_width[0, 0])
    assert 5.0 + kappa * hb + 0.3 < 6.0  # support stays in [4, 6] over the parent cell
    cond = Q.UniformShiftConditional(5.0, gp, kappa, 0.3)
    out = T.node_vs_nested(cond, case, "pl2.6", deriv="jacfwd")
    _record(f"uniform_shift_kappa{kappa}_caps{caps}", float(out["err"].max()))
    assert out["err"].max() < 1e-12


def test_weight_ratio_direction():
    """Child-axis weight factors other than 1: the node needs w_b / w_{b,c} exactly."""
    case = T.make_case(2, 2, "affine", cells=(0, 9, 18))
    rng = np.random.default_rng(4)
    wj = case.w_joint * rng.uniform(0.2, 3.0, case.w_joint.size)
    case = T.Case(case.parent, case.joint, case.child, case.child_chart, case.w_parent, wj)
    cond = Q.LogNormalConditional(5.0, 3e4, 0.0, 0.3)  # exact node, so only the ratio is tested
    out = T.node_vs_nested(cond, case, "bpl")
    L0 = out["L"] * (wj / case.w_parent[np.zeros(wj.size, int)])[case.joint.row_cell][:, None]  # drop the ratio
    no_ratio = np.abs(L0 @ out["a_parent"] - out["a_true"]) / out["scale"]
    _record("weight_ratio", {"with_ratio": float(out["err"].max()), "without_ratio": float(no_ratio.max())})
    assert out["err"].max() < 1e-13 and no_ratio.max() > 0.1


def test_non_negative_cells_point_masses_exact():
    """Under NonNegativeCells the node evaluates g at the point masses, exact by construction."""
    case = T.make_case(2, 2, "affine", cells=(9,))
    cond = Q.LogNormalConditional(4.0, float(case.parent.reference[0, 0]), 2.0, 0.01)  # hardest case
    uq = np.cos(np.pi * (np.arange(17) + 0.5) / 17)
    wq = np.random.default_rng(2).uniform(0, 1, 17)
    gq = C.from_u("log", uq, case.parent.reference[0, 0], case.parent.half_width[0, 0])
    m = C.Measure({"gamma": np.asarray(gq)}, wq)
    a_true = R.joint_moments_nested(cond, m, case.parent, case.joint, "affine", case.child, case.w_joint)
    G = np.stack([np.asarray(cond.g({"gamma": g}, "affine", case.child, 2)) for g in np.asarray(gq)])
    a_pm = np.zeros(case.joint.n_rows)
    for cj in range(case.joint.n_cells):
        c = int(case.joint.index[cj, -1])
        for mm, (r, s) in enumerate(case.joint.exponents(cj)):
            a_pm[case.joint.offsets[cj] + mm] = (case.w_parent[0] / case.w_joint[cj]) * np.sum(
                uq**r * G[:, c, s] * wq) / case.w_parent[0]
    err = float(np.abs(a_pm - a_true).max() / np.abs(a_true).max())
    _record("non_negative_cells_point_masses", err)
    assert err < 1e-13


# ------------------------------------------------------------------ two parents, multi-index k
def test_two_parents_multi_index_polynomial_exact():
    """Parents (gamma, depth) on cells, B | gamma, depth uniform with an affine shift in both,
    support inside one B cell: g_s has total degree s, and the multi-index node is exact."""
    g = C.Cells.log(3e4, 1e5, 2, order=2)
    d = C.Cells.linear(6.0, 10.0, 2, order=2)
    child = C.Cells.linear(2.0, 8.0, 3, order=2)
    parent = C.ProductTable.tensor({"gamma": g, "depth": d}, total=2)
    joint = C.ProductTable.tensor({"gamma": g, "depth": d, "B": child}, total=2)
    wp, wj = C.reference_weights(parent), C.reference_weights(joint)

    class Shift2(Q.UniformShiftConditional):
        def centre(self, point):
            return self.c0 + 0.15 * C.to_u("log", point["gamma"], 5.4e4, 1.0) + 0.1 * (point["depth"] - 8.0)

    cond = Shift2(5.0, 5.4e4, 0.0, 0.2)
    h, ks = R.taylor_h(cond, parent, ("gamma", "depth"), "affine", child, 2, 2, deriv="jacfwd")
    L = R.node_matrix(parent, joint, ("gamma", "depth"), h, ks, wp, wj)
    mg = C.panel_measure("gamma", lambda x: x**-2.6, [3e4, np.sqrt(3e9), 1e5], n=8)
    md = C.panel_measure("depth", lambda x: 1.0 + 0.1 * (x - 8.0), [6, 8, 10], n=6, chart="affine")
    m = C.product(mg, md)
    ap = C.pair(parent, m, wp)["M0"]
    a_true = R.joint_moments_nested(cond, m, parent, joint, "affine", child, wj)
    err = float(np.abs(L @ ap - a_true).max() / np.abs(a_true).max())
    n_drop = R.dropped_terms(parent, joint, ("gamma", "depth"), h, ks)
    _record("two_parents_polynomial", {"err": err, "dropped_terms": n_drop})
    assert err < 1e-12 and n_drop == 0
