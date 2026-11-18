"""R3a at the prediction level: node route against exact joint moments, direct integration, and
the averaged-child route that the Spec names as the remedy.

conditional_truncation = predict(node) - predict(exact joint moments), same columns.
representation         = predict(exact joint moments) - direct, within the cells.
avg route              = B averaged with its conditional into the kernel on the gamma cells.

Run from the prototypes directory:
PYTHONPATH=.:rules:../../.. python -m pytest tests/test_R3a_predict.py
The continuum-kernel case is marked slow (about 3 min per order on a loaded machine).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import core as C
from core import paper as E
from R3a import conditional as Q
from R3a import predict_cases as PC
from R3a import probe_predict as PP
from R3a import setups as T

RESULTS = Path(__file__).with_name("R3a_results.json")


def _record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


def _poly(caps, total, Bchart, seed=1, n_ch=5):
    """Kernel polynomial in (u_gamma, x_B) with monomials in the lattice's lower set."""
    rng = np.random.default_rng(seed)
    tr = {"gamma": ("log", np.log(3e4), 2.0),
          "B": (Bchart, np.log(5.0) if Bchart == "log" else 5.0, 1.0 if Bchart == "log" else 3.0)}
    al = C.lower_set(caps, total)
    cI = {a: rng.normal(size=n_ch) * 0.3 + (5.0 if sum(a) == 0 else 0.0) for a in al}
    cP = {a: (rng.normal(size=n_ch) + 1j * rng.normal(size=n_ch)) * 0.3 + (2.0 if sum(a) == 0 else 0.0)
          for a in al}
    return C.PolynomialKernel(tr, cI, cP)


_POLY_AVG = {}


@pytest.mark.parametrize("chart,caps,total,beta,sigma,exact", [
    ("log_full", (2, 2), 2, 0.3, 0.2, True),
    ("log_full", (4, 2), 4, 0.3, 0.2, True),
    ("log_full", (2, 2), 2, 2.0, 3.0, True),
    ("log_full", (1, 2), 2, 0.3, 0.2, False),
    ("log_full", (1, 2), 2, 2.0, 0.03, False),
    ("affine", (2, 1), 2, 0.3, 0.2, False),
    ("affine", (2, 1), 2, 0.01, 0.3, False),
])
def test_polynomial_kernel_node_equals_averaged_route(chart, caps, total, beta, sigma, exact):
    """For a kernel polynomial in the lattice, the node prediction equals the averaged-child
    prediction (the Cauchy-product adjoint), whether or not the node is exact. So averaging the
    child does not remove the conditional truncation: both routes truncate the same function
    of u_gamma at order N_gamma."""
    case = T.make_case(caps[0], caps[1], chart, total=total)
    K = _poly(caps, total, "log" if chart.startswith("log") else "affine")
    cond = Q.LogNormalConditional(5.0, 3e4, beta, sigma)
    rule = cond.hermite_rule(40) if chart == "log_full" else cond.cell_rule(case.child_chart, case.child, 64)
    out = PC.run(C.AveragedKernel(K), cond, case, averaged_rule=rule)
    out.pop("seconds")
    _record(f"poly_pred_{chart}_caps{caps}_t{total}_b{beta}_s{sigma}", out)
    assert out["representation"] < 1e-12  # the kernel is in the lattice: only the node truncates
    assert out["avg_vs_node"] < 1e-12
    assert out["bound_holds"]
    if exact:
        assert out["cond_trunc"] < 1e-13
    else:
        assert out["cond_trunc"] > 1e-8
        assert abs(out["avg_vs_direct"] - out["cond_trunc"]) < 1e-12 + 1e-9 * out["cond_trunc"]


_TOY = {}


def _toy(chart, N, sigma, beta, where, cell, averaged=False):
    if "avg" not in _TOY:
        _TOY["avg"] = C.AveragedKernel(T.ToySynchrotron())
    m0 = {"affine": {"centre": 5.0, "edge": 4.0}, "log": {"centre": 4.0, "edge": 2.0 * 4.0 ** (2 / 3)}}[chart][where]
    case = T.make_case(N, min(2, N), chart, cells=(cell,))
    cond = Q.LogNormalConditional(m0, float(case.parent.reference[0, 0]), beta, sigma)
    rule = cond.cell_rule(case.child_chart, case.child, 48) if averaged else None
    out = PC.run(_TOY["avg"], cond, case, n_pop=12, n_child=8, averaged_rule=rule)
    out.pop("seconds")
    return out


@pytest.mark.parametrize("chart", ["affine", "log"])
@pytest.mark.parametrize("cell", [0, 18])
@pytest.mark.parametrize("sigma,beta", [(0.3, 0.1), (3.0, 0.5), (1.0, 2.0)])
def test_toy_conditional_truncation_converges_when_smooth(chart, cell, sigma, beta):
    """zeta = beta h_b / sigma <= 0.5: the contracted error falls with N although the rows with
    r near N_b do not improve (test_R3a_moments)."""
    errs = {N: _toy(chart, N, sigma, beta, "edge", cell) for N in (1, 3, 5)}
    _record(f"toy_converge_{chart}_cell{cell}_s{sigma}_b{beta}", errs)
    assert all(e["bound_holds"] for e in errs.values())
    assert errs[5]["cond_trunc"] < 0.05 * errs[1]["cond_trunc"]


@pytest.mark.parametrize("chart", ["affine", "log"])
def test_toy_narrow_conditional_across_edge_does_not_converge(chart):
    """sigma = 0.03, beta = 0.5 (zeta = 4), median on a child edge: no convergence with N.
    The bound holds and is of order the error, so it flags the case."""
    errs = {N: _toy(chart, N, 0.03, 0.5, "edge", 9) for N in (1, 3, 5)}
    _record(f"toy_narrow_edge_{chart}", errs)
    assert all(e["bound_holds"] for e in errs.values())
    assert errs[5]["cond_trunc"] > 0.2 * errs[3]["cond_trunc"]
    assert errs[5]["cond_trunc"] > 1e-3


@pytest.mark.parametrize("chart", ["affine", "log"])
@pytest.mark.parametrize("N", [1, 3])
def test_toy_averaged_route_keeps_the_truncation(chart, N):
    """The remedy of the Spec ('removing the child from the lattice ... removes this term') against
    the node: the averaged route's error against direct integration is of the same order as the
    node's conditional truncation, and the two routes differ by the B-direction representation."""
    out = _toy(chart, N, 0.3, 1.0, "centre", 9, averaged=True)
    _record(f"toy_averaged_route_{chart}_N{N}", out)
    assert out["avg_vs_direct"] > 0.1 * out["cond_trunc"]


@pytest.mark.slow
@pytest.mark.parametrize("chart", ["affine", "log"])
@pytest.mark.parametrize("N", [2, 4])
def test_continuum_kernel(chart, N):
    """E1 continuum kernel with B resolved on [2, 8] uG, eta averaged, screen and azimuth factor.
    B | gamma lognormal, median 5 (gamma / 3e4)^0.1 uG, sigma 0.2. Parent cells 0, 9, 18."""
    avg = _TOY.setdefault("continuum", PP.continuum_kernel())
    case = T.make_case(N, 2, chart, cells=(0, 9, 18))
    cond = Q.LogNormalConditional(5.0, 3.0e4, 0.1, 0.2)
    out = PC.run(avg, cond, case, n_pop=8, n_child=6, azimuth=E.AZIMUTH)
    _record(f"continuum_{chart}_N{N}", out)
    assert out["bound_holds"]
    assert out["cond_trunc"] < 1e-3 and out["node_vs_direct"] < 1e-2
