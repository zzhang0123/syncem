"""R3a node applied to true parent moments against nested-quadrature joint moments.

Two methods for the same joint moments, compared directly (boundary-validation rule): the R3a
node (Taylor shift of g, truncated at the parent lower set) and nested quadrature of the
specified parent with the exact child-cell moments g. Extremes: conditional width sigma from
0.003 to 3 in ln B, gamma dependence beta from 0 to 2, median at a child-cell centre or on an
edge, first and last E1 cells, orders 2 and 6, five parent populations.

Error per row: |a_node - a_true| / ((w_b / w_{b,c}) a_parent[b, 0]), per unit parent mass.
The governing number is zeta = beta h_b / sigma, the shift of the median across a parent
half-cell in units of the conditional width.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from R3a import conditional as Q
from R3a import node as R
from R3a import setups as T

RESULTS = Path(__file__).with_name("R3a_results.json")
MEDIAN = {"affine": {"centre": 5.0, "edge": 4.0}, "log": {"centre": 4.0, "edge": 2.0 * 4.0 ** (2 / 3)}}
H_B = T.gamma_cells(2)[0].half_width[0]


def _record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


def _run(chart, sigma, beta, where, cell, N, pop="pl2.6"):
    case = T.make_case(N, min(2, N), chart, cells=(cell,))
    cond = Q.LogNormalConditional(MEDIAN[chart][where], float(case.parent.reference[0, 0]), beta, sigma)
    out = T.node_vs_nested(cond, case, pop)
    Rb = R.truncation_sup(cond, case.parent, case.joint, ("gamma",), out["h"], out["ks"], case.child_chart,
                          case.child)
    rows = out["rows"]
    by_r = [float(out["err"][rows[:, 0] == r].max()) for r in range(N + 1)]
    holds = bool(np.all(out["err"] <= Rb * (1 + 1e-9) + 1e-13))
    return dict(err=float(out["err"].max()), by_r=by_r, bound=float(Rb.max()), bound_holds=holds)


@pytest.mark.parametrize("chart", ["affine", "log"])
@pytest.mark.parametrize("sigma", [0.003, 0.3, 3.0])
@pytest.mark.parametrize("beta", [0.0, 0.01, 0.5, 2.0])
@pytest.mark.parametrize("where", ["centre", "edge"])
@pytest.mark.parametrize("cell", [0, 18])
def test_node_vs_nested_grid(chart, sigma, beta, where, cell):
    r2 = _run(chart, sigma, beta, where, cell, 2)
    r6 = _run(chart, sigma, beta, where, cell, 6)
    zeta = beta * H_B / sigma
    _record(f"moments_{chart}_s{sigma}_b{beta}_{where}_cell{cell}", {"zeta": zeta, "N2": r2, "N6": r6})
    assert r2["bound_holds"] and r6["bound_holds"]
    if beta == 0.0:
        assert r6["err"] < 1e-13  # independent child: exact
    elif zeta <= 0.5:
        # smooth on the parent cell: the r = 0 row converges, the r = N row does not improve
        assert r6["by_r"][0] < 1e-2 * r2["by_r"][0] + 1e-14  # floor: roundoff of the reference
        if r2["by_r"][-1] > 1e-12:
            assert r6["by_r"][-1] > 0.1 * r2["by_r"][-1]


@pytest.mark.parametrize("chart", ["affine", "log"])
@pytest.mark.parametrize("beta", [0.5, 2.0])
def test_narrow_conditional_across_an_edge_diverges(chart, beta):
    """sigma = 0.003, median on a child edge: g is a near-step in u over the parent cell. The
    node error exceeds the parent mass and grows with N (Taylor series of a near-step). The sup
    bound flags it before any data are seen."""
    errs = {N: _run(chart, 0.003, beta, "edge", 18, N) for N in (2, 4, 6)}
    _record(f"narrow_edge_divergence_{chart}_b{beta}", errs)
    assert errs[2]["err"] > 1.0 and errs[6]["err"] > errs[2]["err"]
    assert all(e["bound_holds"] and e["bound"] > 0.1 for e in errs.values())


@pytest.mark.parametrize("pop", ["pl0", "pl6", "bpl", "edge"])
@pytest.mark.parametrize("chart", ["affine", "log"])
@pytest.mark.parametrize("sigma,beta", [(0.03, 0.1), (0.3, 1.0), (3.0, 1.0)])
def test_sup_bound_holds_for_other_parents(pop, chart, sigma, beta):
    """The per-unit-mass bound needs only the cell's parent mass: it holds for a flat, a steep, a
    broken and an edge-concentrated parent population, so it applies to a free parent."""
    r = _run(chart, sigma, beta, "edge", 9, 4, pop=pop)
    _record(f"bound_{pop}_{chart}_s{sigma}_b{beta}", r)
    assert r["bound_holds"]
