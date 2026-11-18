"""(a) Forward on (gamma, B) lattices against direct integration over (gamma, B), Spec Sec. 4.2, 8.3, G8, T5.

Populations: the paper's E6 (energy cut-off from the local field, B ~ U[4, 6] uG) on one field
cell, an E6-like population with B ~ U[2, 8] uG on 3 affine field cells, the same law with a
log-uniform field on 3 log cells over [0.5, 50] uG (range ratio 100), and a power law with a
field law conditional on gamma. Columns come from the jet in (gamma, B) and from the
field-scaling relation. Direct integration uses the same nodes as the pairing, so the error
is the representation error of the expansion (Spec Sec. 7.2). Jet cost per product cell is
recorded at d = 2.

Run from the prototypes directory (about 10 min):
PYTHONPATH=.:../../.. python -m pytest tests/test_fieldcells_forward.py
"""
from __future__ import annotations

import hashlib
import time
from functools import lru_cache
from pathlib import Path

import numpy as np
import pytest

import core as C
from core.paper import AZIMUTH
from rules.fieldcells import models as M
from rules.fieldcells import probe as R
from rules.fieldcells import scaling as S

STUDY_E6 = 7.808321077944934e-3  # cost_results.json joint_gamma_B moment_route_72_max_rel_err

POPS = {
    "E6": ("single_5_1", lambda: M.e6_measure(R.GAMMA_EDGES, [4.0, 6.0], lambda b: 0 * b + 0.5, n_B=16)),
    "E6_B2_8": ("lin_2_8", lambda: M.e6_measure(R.GAMMA_EDGES, [2.0, 4.0, 6.0, 8.0], lambda b: 0 * b + 1 / 6)),
    "E6_logB": ("log_ratio100", lambda: M.e6_measure(
        R.GAMMA_EDGES, C.Cells.log(0.5, 50.0, 3).edges, lambda b: 1.0 / (b * np.log(100.0)))),
    "E6_logB9": ("log_ratio100_9", lambda: M.e6_measure(
        R.GAMMA_EDGES, C.Cells.log(0.5, 50.0, 9).edges, lambda b: 1.0 / (b * np.log(100.0)))),
    "B_given_gamma": ("lin_2_8", lambda: M.conditional_field_measure(
        R.GAMMA_EDGES, [2.0, 4.0, 6.0, 8.0], lambda g: 5.0 + 2.5 * np.tanh(np.log(g / 1e4) / 3.0))),
}


@lru_cache(maxsize=None)
def _measure(pop):
    return POPS[pop][1]()


CACHE = Path(__file__).with_name("_fieldcells_cache")


@lru_cache(maxsize=None)
def _direct(pop, kind="E1"):
    """Direct sum, cached on disk with the node hash (the measure is deterministic)."""
    m = _measure(pop)
    tag = hashlib.sha256(np.concatenate([m.nodes["gamma"], m.nodes["B"], m.weights]).tobytes()).hexdigest()[:16]
    f = CACHE / f"direct_{pop}_{kind}_{tag}.npy"
    if f.exists():
        return np.load(f)
    t0 = time.perf_counter()
    d = C.direct(R.avg(kind), m, azimuth=AZIMUTH)
    R.record(f"a_direct_cost[{pop},{kind}]", {"nodes": int(m.weights.size), "seconds": time.perf_counter() - t0})
    CACHE.mkdir(exist_ok=True)
    np.save(f, d)
    return d


def _forward(cols, tab, pop):
    w = C.reference_weights(tab)
    m = C.pair(tab, _measure(pop), w)
    return C.predict(cols, tab, m, w, azimuth=AZIMUTH)


CASES = [("E6", (2, 2), 2), ("E6", (4, 4), 4),
         ("E6_B2_8", (2, 1), 2), ("E6_B2_8", (2, 2), 2), ("E6_B2_8", (1, 1), 2), ("E6_B2_8", (4, 2), 4),
         ("E6_logB", (2, 2), 2), ("E6_logB9", (2, 2), 2), ("B_given_gamma", (2, 2), 2)]
# Representation errors far above the method agreement are expected where a field cell is wide
# against the field dependence of the cut-off: 3 log cells on [0.5, 50] uG (ratio 4.6 per cell).
LOOSE = {"E6_logB": 10.0}


@pytest.mark.parametrize("pop,caps,total", CASES)
def test_forward_jet_and_scaling(pop, caps, total):
    blat = POPS[pop][0]
    tab, jcols, t_jet = R.jet("E1", blat, caps, total)
    t0 = time.perf_counter()
    scols = S.scaling_columns(R.avg("E1"), tab, 5.0)
    t_sc = time.perf_counter() - t0
    ref = _direct(pop)
    in_cells = (tab.locate(_measure(pop).nodes) >= 0).all()
    pj, ps = _forward(jcols, tab, pop), _forward(scols, tab, pop)
    e_jet, e_sc = C.representation_error(pj, ref), C.representation_error(ps, ref)
    e_between = C.representation_error(ps, pj)
    out = {"rows": tab.n_rows, "cells": tab.n_cells, "jet_vs_direct": e_jet, "scaling_vs_direct": e_sc,
           "scaling_vs_jet_forward": e_between, "jet_build_s_first": t_jet, "scaling_build_s_first": t_sc}
    per_ch = np.max(np.abs(pj - ref) / np.abs(ref[:, :1]), axis=1)  # channels 0.1 to 10 GHz, log-spaced
    out.update({"jet_vs_direct_lowest_channel": float(per_ch[0]), "jet_vs_direct_highest_channel": float(per_ch[-1]),
                "jet_vs_direct_max_below_1GHz": float(per_ch[:21].max())})
    if pop == "E6" and total == 2:
        out["study_E6"] = STUDY_E6
        out["ratio_to_study"] = e_jet / STUDY_E6
    R.record(f"a_forward[{pop},caps={caps},total={total}]", out)
    assert in_cells
    assert np.isfinite(pj).all() and np.isfinite(ps).all()
    assert e_between < 1e-9
    assert e_jet < LOOSE.get(pop, 0.5)  # a representation error, recorded. The method checks are above and below
    if pop == "E6_logB9":
        assert e_jet < 0.1 * R.json_get("a_forward[E6_logB,caps=(2, 2),total=2]")["jet_vs_direct"]
    if pop == "E6" and total == 2:
        assert abs(e_jet / STUDY_E6 - 1) < 0.02


def test_forward_fallback_eta_given_B():
    """Precondition broken by eta given B: the jet still matches direct, the scaling columns do not."""
    tab, jcols, _ = R.jet("etaB", "lin_2_8", (2, 2), 2)
    scols = S.scaling_columns(R.avg("etaB"), tab, 5.0)
    ref = _direct("E6_B2_8", "etaB")
    e_jet = C.representation_error(_forward(jcols, tab, "E6_B2_8"), ref)
    e_sc = C.representation_error(_forward(scols, tab, "E6_B2_8"), ref)
    R.record("a_forward_eta_given_B", {"jet_vs_direct": e_jet, "forced_scaling_vs_direct": e_sc})
    assert e_sc > 2 * e_jet


def test_jet_cost_per_cell_d2():
    """Cached cost of the jet at d = 2 (gamma, B) and of the scaling route, 57 product cells."""
    out = {}
    for N in (2, 4):
        tab, _, first = R.jet("E1", "lin_2_8", (N, N), N)
        t0 = time.perf_counter()
        C.build_columns(R.avg("E1"), tab)
        cached = time.perf_counter() - t0
        S.scaling_columns(R.avg("E1"), tab, 5.0)
        t0 = time.perf_counter()
        S.scaling_columns(R.avg("E1"), tab, 5.0)
        sc = time.perf_counter() - t0
        g1 = C.ProductTable.tensor({"gamma": C.Cells.log(300, 2.9626e6, 19, order=N),
                                    "B": C.Cells.single(5.0, 1.0, order=0)}, total=N)
        C.build_columns(R.avg("E1"), g1)
        t0 = time.perf_counter()
        C.build_columns(R.avg("E1"), g1)
        d1 = time.perf_counter() - t0
        out[f"N={N}"] = {"cells": tab.n_cells, "rows": tab.n_rows, "jet_d2_first_s": first,
                         "jet_d2_cached_s": cached, "jet_d2_per_cell_s": cached / tab.n_cells,
                         "scaling_cached_s": sc, "scaling_per_cell_s": sc / tab.n_cells,
                         "jet_d1_energy_per_cell_s": d1 / g1.n_cells,
                         "jet_d2_over_d1": (cached / tab.n_cells) / (d1 / g1.n_cells)}
    R.record("a_jet_cost_d2", out)
    assert out["N=2"]["jet_d2_per_cell_s"] > 0
