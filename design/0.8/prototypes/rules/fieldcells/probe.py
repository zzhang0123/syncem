"""Cached builds shared by the field-cell tests, and the results file.

Builds are memoised in the process so that parametrised tests reuse one jet build per
(kernel, table). Results go to tests/fieldcells_results.json.
"""
from __future__ import annotations

import json
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

import core as C
from core.paper import GAMMA_HI, GAMMA_LO

from . import models as M

RESULTS = Path(__file__).resolve().parents[2] / "tests" / "fieldcells_results.json"
GAMMA_EDGES = C.Cells.log(GAMMA_LO, GAMMA_HI, 19).edges
H_GAMMA = float(C.Cells.log(GAMMA_LO, GAMMA_HI, 19).half_width[0])


def record(key: str, value) -> None:
    """Merge one entry into the results file under an exclusive lock (test files may run in parallel)."""
    import fcntl

    with open(RESULTS.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        data[key] = value
        RESULTS.write_text(json.dumps(data, indent=1, default=float))


def json_get(key: str):
    return json.loads(RESULTS.read_text())[key]


def rel_err(a, b) -> float:
    """Boundary-rule rel_err: max |a - b| / max(|a|, |b|), maxima over channels, per row, then max."""
    a, b = np.asarray(a), np.asarray(b)
    num = np.max(np.abs(a - b), axis=tuple(range(1, a.ndim)))
    den = np.maximum(np.maximum(np.max(np.abs(a), axis=tuple(range(1, a.ndim))),
                                np.max(np.abs(b), axis=tuple(range(1, a.ndim)))), 1e-300)
    return float(np.max(num / den))


def verdict(e: float) -> str:
    return "OK" if e < 1e-3 else ("WARN" if e < 1e-1 else "BAD")


@lru_cache(maxsize=None)
def kernel():
    return M.continuum()


@lru_cache(maxsize=None)
def avg(kind: str = "E1"):
    """Averaged kernels with gamma and B resolved: eta on the E1 law, or eta given B."""
    rule = M.eta_rule() if kind == "E1" else M.eta_given_B()
    return C.AveragedKernel(kernel(), [("eta", rule)])


B_LATTICES = {
    "lin_2_8": lambda o: C.Cells.linear(2.0, 8.0, 3, order=o),
    "lin_ratio1.01": lambda o: C.Cells.linear(5.0, 5.05, 3, order=o),
    "lin_ratio100": lambda o: C.Cells.linear(0.5, 50.0, 3, order=o),
    "log_ratio100": lambda o: C.Cells.log(0.5, 50.0, 3, order=o),
    "log_ratio100_9": lambda o: C.Cells.log(0.5, 50.0, 9, order=o),
    "single_5_1": lambda o: C.Cells.single(5.0, 1.0, order=o),
    # log field cells with h_B = 2 h_gamma: B_{b+1}/B_b = (gamma_{g+1}/gamma_g)^2, so product cells coincide
    "log_aligned": lambda o: C.Cells.log(2.0, 2.0 * np.exp(12.0 * H_GAMMA), 3, order=o),
}


def table(blat: str, caps: tuple, total: int, gamma_cells=None) -> C.ProductTable:
    g = C.Cells.log(GAMMA_LO, GAMMA_HI, 19, order=caps[0])
    t = C.ProductTable.tensor({"gamma": g, "B": B_LATTICES[blat](caps[1])}, total=total)
    if gamma_cells is not None:
        t = t.select(np.isin(t.index[:, 0], gamma_cells))
    return t


_JET: dict = {}


def jet(kind: str, blat: str, caps: tuple, total: int, gamma_cells=None):
    key = (kind, blat, caps, total, None if gamma_cells is None else tuple(gamma_cells))
    if key not in _JET:
        tab = table(blat, caps, total, gamma_cells)
        t0 = time.perf_counter()
        cols = C.build_columns(avg(kind), tab)
        _JET[key] = (tab, cols, time.perf_counter() - t0)
    return _JET[key]
