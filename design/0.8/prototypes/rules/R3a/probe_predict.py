"""Probe: R3a at the prediction level, order dependence, toy and continuum kernels.

toy: the cheap analytic kernel of setups.ToySynchrotron, one parent cell per run (first,
middle, last), median m0 at the cell's reference point, N = 1..5, B on affine or log cells.
continuum: the E1 continuum kernel with B resolved, eta averaged on [0, 0.5] (12 nodes), the
Gaussian screen and the azimuth factor 0.4 e^{0.5i}, N = 2 and 4, parent cells 0, 9, 18.

Run: PYTHONPATH=<prototypes>:<prototypes>/rules:<syncem> python -m R3a.probe_predict [toy|continuum]
Writes rules/R3a/results/predict_<which>.json.
"""
from __future__ import annotations

import itertools
import json
import sys
import time
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss

import core as C
from core import paper as E

from . import conditional as Q
from . import predict_cases as PC
from . import setups as T

OUT = Path(__file__).with_name("results")
TOY_CONDS = [(0.3, 0.1), (0.3, 1.0), (0.03, 0.5), (3.0, 0.5), (0.3, 0.0)]  # (sigma, beta)


def toy():
    avg = C.AveragedKernel(T.ToySynchrotron())
    recs = []
    for chart, N, (sigma, beta), m0, cell in itertools.product(("affine", "log"), (1, 2, 3, 4, 5), TOY_CONDS,
                                                               (5.0, 4.0), (0, 9, 18)):
        case = T.make_case(N, min(2, N), chart, cells=(cell,))
        cond = Q.LogNormalConditional(m0, float(case.parent.reference[0, 0]), beta, sigma)
        t0 = time.time()
        out = PC.run(avg, cond, case, n_pop=12, n_child=8)
        recs.append(dict(chart=chart, N=N, sigma=sigma, beta=beta, m0=m0, cell=cell, wall=time.time() - t0, **out))
        print(chart, N, sigma, beta, m0, cell, f"cond {out['cond_trunc']:.2e} repr {out['representation']:.2e}",
              f"bound {out['bound_rel_I']:.2e} holds {out['bound_holds']}", flush=True)
    return recs


def continuum_kernel():
    e, we = leggauss(12)
    eta, w_eta = 0.25 * (e + 1.0), we * 0.25 / 0.5
    K = C.ContinuumChannelKernel(C.bump_channels(n_nodes=64), screen=E.SCREEN, n_nodes_F=128)
    return C.AveragedKernel(K, [("eta", C.fixed(eta, w_eta))], outer_batch=None)


def continuum():
    avg = continuum_kernel()
    recs = []
    for chart, N in itertools.product(("affine", "log"), (2, 4)):
        case = T.make_case(N, 2, chart, cells=(0, 9, 18))
        cond = Q.LogNormalConditional(5.0, 3.0e4, 0.1, 0.2)
        t0 = time.time()
        out = PC.run(avg, cond, case, n_pop=8, n_child=6, azimuth=E.AZIMUTH,
                     averaged_rule=cond.cell_rule(case.child_chart, case.child, 16))
        recs.append(dict(chart=chart, N=N, sigma=0.2, beta=0.1, m0=5.0, gamma_p=3e4, cells=[0, 9, 18],
                         wall=time.time() - t0, **out))
        print(recs[-1], flush=True)
    return recs


def main(which):
    OUT.mkdir(exist_ok=True)
    t0 = time.time()
    recs = toy() if which == "toy" else continuum()
    (OUT / f"predict_{which}.json").write_text(json.dumps(dict(records=recs, seconds=time.time() - t0),
                                                          indent=0, default=float))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "toy")
