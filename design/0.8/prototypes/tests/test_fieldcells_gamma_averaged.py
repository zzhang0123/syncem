"""(d) Field cells alone with gamma averaged against a specified spectrum (Spec Sec. 8.3, r3-07, T5).

Four log field cells on [0.5, 50] uG, gamma averaged on [300, 2.9626e6] by gamma^-2.6
(114 nodes), eta on the E1 law. The jet gives the field columns (the 0.8.0 route). Checks:
* jet forward against direct integration over (gamma, B) at both extreme field cells, at the
  lowest and highest channels, for a log-uniform field law.
* the 0.9 per-node identity (d/dln B)^n = (1 + (1/2) d/dln gamma)^n at each gamma node
  against the jet (both methods evaluated directly).
* the shortcut that treats the averaged kernel as an infinite power law,
  Kbar(L B) = L^((s+1)/2) Kbar(B), against the jet. It must not be used.
* an electron law whose cut-off depends on B (the E6 law with gamma averaged): the jet
  against direct, and the per-node identity with nodes and probabilities frozen at B_b.

Run from the prototypes directory (about 5 min):
PYTHONPATH=.:../../.. python -m pytest tests/test_fieldcells_gamma_averaged.py
"""
from __future__ import annotations

import time
from functools import lru_cache

import numpy as np
import pytest

import core as C
from core.paper import AZIMUTH
from rules.fieldcells import models as M
from rules.fieldcells import pernode as PN
from rules.fieldcells import probe as R

N = 2
SLOPE = 2.6


RULES = {"power": lambda: M.power_law_rule(SLOPE),
         "power_narrow": lambda: M.power_law_rule(SLOPE, lo=300.0, hi=3e4, n_panels=10),
         "cutoff": lambda: M.cutoff_rule()}


def _cells(n=4):
    return C.Cells.log(0.5, 50.0, n, order=N)


@lru_cache(maxsize=None)
def _avg(law):
    return C.AveragedKernel(R.kernel(), [("gamma", RULES[law]()), ("eta", M.eta_rule())])


@lru_cache(maxsize=None)
def _jet(law, n=4):
    tab = C.ProductTable.tensor({"B": _cells(n)})
    t0 = time.perf_counter()
    cols = C.build_columns(_avg(law), tab)
    R.record(f"d_jet_build_s[{law},cells={n}]", {"cells": tab.n_cells, "seconds": time.perf_counter() - t0})
    return tab, cols


@lru_cache(maxsize=None)
def _measure(n=4):
    b, w = M.field_nodes(_cells(n).edges, 5)
    return C.Measure({"B": b}, w / (b * np.log(100.0)))


@lru_cache(maxsize=None)
def _values(law, n=4):
    return PN.node_values(_avg(law), _measure(n))


def _forward_errors(law, n=4):
    tab, cols = _jet(law, n)
    m = _measure(n)
    w = np.ones(tab.n_cells)
    I, P = _values(law, n)
    cell = tab.locate(m.nodes)
    out = {}
    for name, cells in (("all", range(tab.n_cells)), ("lowest_cell", [0]), ("highest_cell", [tab.n_cells - 1])):
        sel = np.isin(cell, list(cells))
        sub = m.subset(sel)
        rows = np.isin(tab.row_cell, list(cells))
        a = C.pair(tab, sub, w)
        pred = C.predict({"I": cols["I"] * rows[:, None], "P": cols["P"] * rows[:, None]}, tab, a, w,
                         azimuth=AZIMUTH)
        ref = np.stack([sub.weights @ I[sel], (AZIMUTH * (sub.weights @ P[sel])).real,
                        (AZIMUTH * (sub.weights @ P[sel])).imag], axis=-1)
        e = np.abs(pred - ref) / np.abs(ref[:, :1])
        out[name] = {"max": float(e.max()), "lowest_channel": float(e[0].max()), "highest_channel": float(e[-1].max())}
    return out


def test_jet_forward_vs_direct():
    out = _forward_errors("power")
    R.record("d_forward_power_law", out)
    for v in out.values():
        assert np.isfinite(v["max"]) and v["max"] < 2e-2


@pytest.mark.parametrize("cell", [0, 3])
def test_pernode_identity_equals_jet(cell):
    tab, cols = _jet("power")
    nodes, probs = (np.asarray(a) for a in M.power_law_rule(SLOPE)({}))
    one = C.Cells.from_edges(_cells().edges[cell:cell + 2], chart="log", order=N)
    pn = PN.pernode_columns(R.avg("E1"), one, nodes, probs, N)
    rows = slice(tab.offsets[cell], tab.offsets[cell + 1])
    e = max(R.rel_err(cols[k][rows], pn[k]) for k in ("I", "P"))
    R.record(f"d_pernode_vs_jet[power,cell={cell}]", {"rel_err": e, "verdict": R.verdict(e)})
    assert e < 1e-8


@pytest.mark.parametrize("law,hi", [("power", 2.9626e6), ("power_narrow", 3e4)])
def test_homogeneous_shortcut(law, hi):
    """The infinite-power-law shortcut misses the range edges of the electron law."""
    tab, cols = _jet(law)
    hc = PN.homogeneous_columns(_avg(law), _cells(), SLOPE, N)
    per_order = {}
    for n in range(1, N + 1):
        rows = np.flatnonzero(tab.row_exp[:, 0] == n)
        per_order[f"n={n}"] = max(R.rel_err(cols[k][rows], hc[k][rows]) for k in ("I", "P"))
    e0 = max(R.rel_err(cols[k][tab.row_exp[:, 0] == 0], hc[k][tab.row_exp[:, 0] == 0]) for k in ("I", "P"))
    R.record(f"d_homogeneous_shortcut_vs_jet[gamma_hi={hi:g}]",
             {**per_order, "n=0": e0, "verdict": R.verdict(max(per_order.values()))})
    assert e0 < 1e-12
    assert max(per_order.values()) > 1e-4


def _fd_columns(avg, b_ref, h, d=0.02):
    """(1/n!) d^n/du^n of avg at B = b_ref e^{h u}, n = 1, 2, by five-point central differences."""
    f = {k: avg({"B": b_ref * np.exp(h * k * d)}) for k in (-2, -1, 0, 1, 2)}
    out = {}
    for j, key in enumerate(("I", "P")):
        v = {k: np.asarray(f[k][j]) for k in f}
        d1 = (-v[2] + 8 * v[1] - 8 * v[-1] + v[-2]) / (12 * d)
        d2 = (-v[2] + 16 * v[1] - 30 * v[0] + 16 * v[-1] - v[-2]) / (12 * d * d)
        out[key] = np.stack([v[0], d1, d2 / 2])
    return out


@pytest.mark.parametrize("cell", [0, 3])
def test_cutoff_jet_vs_finite_differences(cell):
    """The jet differentiates the moving support and weights of p(gamma | B): check against differences."""
    tab, cols = _jet("cutoff")
    fd = _fd_columns(_avg("cutoff"), _cells().reference[cell], _cells().half_width[cell])
    rows = slice(tab.offsets[cell], tab.offsets[cell + 1])
    e = max(R.rel_err(cols[k][rows], fd[k]) for k in ("I", "P"))
    R.record(f"d_cutoff_jet_vs_fd[cell={cell}]", {"rel_err": e, "verdict": R.verdict(e)})
    assert e < 1e-5


def test_cutoff_frozen_pernode_fails():
    """The 0.9 per-node identity without the law derivative is wrong when the law depends on B."""
    tab, cols = _jet("cutoff")
    frozen = {}
    for cell in (0, 3):
        b = _cells().reference[cell]
        nodes, probs = (np.asarray(a) for a in M.cutoff_rule()({"B": b}))
        one = C.Cells.from_edges(_cells().edges[cell:cell + 2], chart="log", order=N)
        pn = PN.pernode_columns(R.avg("E1"), one, nodes, probs, N)
        rows = slice(tab.offsets[cell], tab.offsets[cell + 1])
        frozen[f"cell={cell}"] = {f"n={n}": max(R.rel_err(cols[k][rows][n:n + 1], pn[k][n:n + 1])
                                                for k in ("I", "P")) for n in range(N + 1)}
    R.record("d_cutoff_frozen_pernode_vs_jet", frozen)
    assert frozen["cell=0"]["n=0"] < 1e-10
    assert max(v["n=1"] for v in frozen.values()) > 1e-2


@pytest.mark.parametrize("n_cells", [4, 16])
def test_cutoff_forward_vs_direct(n_cells):
    """Representation error of the jet columns for the cut-off law, 4 and 16 field cells."""
    out = _forward_errors("cutoff", n_cells)
    R.record(f"d_cutoff_forward[cells={n_cells}]", out)
    assert np.isfinite(out["all"]["max"])
    if n_cells == 16:
        assert out["all"]["max"] < 0.2 * R.json_get("d_cutoff_forward[cells=4]")["all"]["max"]
