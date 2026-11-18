"""Probe: R3a node applied to true parent moments against nested-quadrature joint moments.

Grid over child chart, conditional width sigma, gamma dependence beta, median position m0
(centre of the middle B cell or on a B edge), parent cell (first, middle, last) and order N.
The median equals m0 at the reference point of the tested parent cell. Error per row is
|a_node - a_true| / ((w_b / w_{b,c}) a_parent[b, 0]), the error per unit parent mass.
Also records the sup bound R of node.truncation_sup and whether err <= R.

Run: PYTHONPATH=<prototypes>:<prototypes>/rules:<syncem> python -m R3a.probe_moments
Writes rules/R3a/results/moments_grid.json.
"""
from __future__ import annotations

import itertools
import json
import time
from pathlib import Path

import numpy as np

from . import conditional as Q
from . import node as R
from . import setups as T

OUT = Path(__file__).with_name("results")
SIGMAS = (0.003, 0.03, 0.1, 0.3, 1.0, 3.0)
BETAS = (0.0, 0.01, 0.1, 0.5, 1.0, 2.0)
M0S = (5.0, 4.0)
CELLS = (0, 9, 18)
ORDERS = (0, 1, 2, 4, 6)


def run_one(chart, sigma, beta, m0, cell, N, pop="pl2.6", with_bound=False):
    case = T.make_case(N, min(2, N), chart, cells=(cell,))
    cond = Q.LogNormalConditional(m0, float(case.parent.reference[0, 0]), beta, sigma)
    S = min(2, N)
    out = T.node_vs_nested(cond, case, pop, deriv="analytic")
    rows = out["rows"]
    per_r = [float(out["err"][rows[:, 0] == r].max()) for r in range(N + 1)]
    rec = dict(chart=chart, sigma=sigma, beta=beta, m0=m0, cell=cell, N=N, pop=pop,
               err=float(out["err"].max()), err_by_r=per_r)
    if with_bound:
        Rb = R.truncation_sup(cond, case.parent, case.joint, ("gamma",), out["h"], out["ks"],
                              case.child_chart, case.child)
        rec["bound"] = float(Rb.max())
        rec["bound_holds"] = bool(np.all(out["err"] <= Rb * (1 + 1e-9) + 1e-13))
    return rec


def main():
    OUT.mkdir(exist_ok=True)
    recs, t0 = [], time.time()
    for chart, sigma, beta, m0, cell, N in itertools.product(("affine", "log"), SIGMAS, BETAS, M0S, CELLS,
                                                             ORDERS):
        recs.append(run_one(chart, sigma, beta, m0, cell, N))
    for chart, sigma, beta, m0, N, pop in itertools.product(("affine", "log"), SIGMAS, (0.1, 1.0), M0S,
                                                            (2, 4), ("pl0", "pl6", "bpl", "edge")):
        recs.append(run_one(chart, sigma, beta, m0, 9, N, pop=pop, with_bound=True))
    (OUT / "moments_grid.json").write_text(json.dumps(dict(records=recs, seconds=time.time() - t0), indent=0))
    print(f"{len(recs)} records in {time.time() - t0:.1f} s")


if __name__ == "__main__":
    main()
