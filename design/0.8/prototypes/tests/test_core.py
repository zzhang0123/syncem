"""Acceptance tests A1 to A3 of the prototype core.

Run from the prototypes directory:
PYTHONPATH=.:../../.. python -m pytest tests/test_core.py -s
The E1 tests take a few minutes (direct integration and the N = 4 jet build) and write
their numbers to tests/core_results.json.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pytest

import core as C
from core import paper as E

ROOT = Path(__file__).resolve().parents[4]  # syncem repository root
STUDY_LIB = ROOT / "studies/classical-models/lib"
STUDY_JSON = STUDY_LIB.parent / "forward_results.json"
RESULTS = Path(__file__).with_name("core_results.json")
STORED = {0: 2.56e-2, 2: 8.58e-4, 4: 8.76e-5}

TR3 = {"gamma": ("log", np.log(1e4), 2.0), "B": ("affine", 5.0, 2.0), "depth": ("affine", 8.0, 1.0)}


def _record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


def _axes3(caps, extreme=False):
    if extreme:  # far from the polynomial centres, and one unbounded cell
        return {"gamma": C.Cells.log(1e7, 1e9, 2, order=caps[0]),
                "B": C.Cells.single(500.0, 50.0, order=caps[1]),
                "depth": C.Cells.linear(-300, 300, 2, order=caps[2])}
    return {"gamma": C.Cells.log(300, 2.9626e6, 3, order=caps[0]), "B": C.Cells.linear(2, 8, 2, order=caps[1]),
            "depth": C.Cells.from_edges([6.0, 10.0], chart="affine", order=caps[2])}


# ------------------------------------------------------------------ lattice
def test_row_counts_and_charts():
    g = C.Cells.log(300, 2.9626e6, 19, order=2)
    assert C.ProductTable.tensor({"gamma": g}).n_rows == 57
    t = C.ProductTable.tensor({"gamma": g, "B": C.Cells.linear(2, 8, 3, order=1)}, total=2)
    assert (t.n_cells, t.n_rows) == (57, 285)
    assert np.isclose(g.half_width[0], 0.2420, atol=1e-4)
    for chart in ("log", "affine"):
        x = np.array([3.0, 4.5, 7.0])
        u = C.to_u(chart, x, 5.0, 0.3)
        assert np.allclose(C.from_u(chart, u, 5.0, 0.3), x, rtol=1e-14)
    assert C.lower_set((2, 1), 2) == ((0, 0), (1, 0), (0, 1), (2, 0), (1, 1))


# ------------------------------------------------------------------ A1
@pytest.mark.parametrize("extreme", [False, True])
@pytest.mark.parametrize("caps,total", [((2, 1, 2), 2), ((3, 3, 3), 3), ((0, 2, 1), 2), ((4, 0, 0), 4)])
def test_A1_jet_equals_exact_taylor(caps, total, extreme):
    K = C.PolynomialKernel.random(TR3, 5, 7, seed=sum(caps))
    tab = C.ProductTable.tensor(_axes3(caps, extreme), total=total)
    cols = C.build_columns(C.AveragedKernel(K), tab)
    o, worst = tab.offsets, 0.0
    for c in range(tab.n_cells):
        ref = dict(zip(tab.variables, tab.reference[c]))
        hw = dict(zip(tab.variables, tab.half_width[c]))
        eI, eP = K.taylor(ref, hw, tab.exponents(c))
        for got, want in ((cols["I"][o[c]:o[c + 1]], eI), (cols["P"][o[c]:o[c + 1]], eP)):
            worst = max(worst, np.max(np.abs(got - want)) / np.max(np.abs(want)))
    _record(f"A1_jet_vs_exact_caps{caps}_total{total}_extreme{extreme}", worst)
    assert worst < 1e-12


def _lower_set_poly(caps, total, seed=3, n_ch=6):
    rng = np.random.default_rng(seed)
    alphas = C.lower_set(caps, total)
    return C.PolynomialKernel(TR3, {a: rng.normal(size=n_ch) for a in alphas},
                              {a: rng.normal(size=n_ch) + 1j * rng.normal(size=n_ch) for a in alphas})


@pytest.mark.parametrize("extreme", [False, True])
def test_A1_predict_equals_direct_for_polynomial(extreme):
    caps, total = (2, 1, 2), 2
    K = _lower_set_poly(caps, total)
    tab = C.ProductTable.tensor(_axes3(caps, extreme), total=total)
    avg = C.AveragedKernel(K)
    edges = lambda v: np.unique(np.concatenate([tab.lo[:, tab.variables.index(v)], tab.hi[:, tab.variables.index(v)]]))
    g = C.panel_measure("gamma", lambda x: x**-2.6, edges("gamma"), n=6)
    b = (C.panel_measure("B", lambda x: np.exp(-0.5 * ((x - 500) / 40) ** 2), [300, 700], n=8, chart="affine")
         if extreme else C.panel_measure("B", lambda x: 1 + 0 * x, edges("B"), n=4, chart="affine"))
    d = C.panel_measure("depth", lambda x: np.exp(-0.5 * (x / 100) ** 2) if extreme else 1 + 0 * x,
                        edges("depth"), n=4, chart="affine")
    m = C.product(g, b, d)
    w = C.reference_weights(tab)
    pred = C.predict(C.build_columns(avg, tab), tab, C.pair(tab, m, w), w, azimuth=0.3 + 0.2j)
    ref = C.direct(avg, m, tab, azimuth=0.3 + 0.2j)
    err = np.max(np.abs(pred - ref)) / np.max(np.abs(ref))
    _record(f"A1_predict_vs_direct_extreme{extreme}", err)
    assert err < 1e-12


def test_predict_equals_direct_legendre_harmonic_and_conditional():
    """Legendre eta (L = 3), resolved azimuth (M2), and B averaged by U[lo(gamma), hi(gamma)]."""
    rng = np.random.default_rng(5)
    tr = {"gamma": ("log", np.log(1e4), 2.0), "eta": ("affine", 0.0, 1.0), "B": ("affine", 5.0, 2.0)}
    alphas = [(0, 0, 0), (1, 0, 0), (0, 3, 0), (1, 2, 0), (0, 0, 1), (1, 0, 1), (0, 1, 1)]
    K = C.PolynomialKernel(tr, {a: rng.normal(size=4) for a in alphas},
                           {a: rng.normal(size=4) + 1j * rng.normal(size=4) for a in alphas})
    def centre(p):  # the B law moves with ln gamma, so its derivatives must enter the columns
        return 5.0 + 0.5 * C.to_u("log", p["gamma"], 1e4, 2.0)

    avg = C.AveragedKernel(K, [("B", C.uniform(lambda p: centre(p) - 1, lambda p: centre(p) + 1, 4))],
                           legendre={"eta": (3, 6)})
    tab = C.ProductTable.tensor({"gamma": C.Cells.log(300, 3e6, 3, order=2)}, total=2)
    g = C.panel_measure("gamma", lambda x: x**-2.6, np.unique(np.r_[tab.lo[:, 0], tab.hi[:, 0]]), n=6)
    eta = C.panel_measure("eta", lambda x: 1 + 0.3 * x, [-1, 1], n=6, chart="affine")
    phi = C.Measure({"phi": np.linspace(0, np.pi, 5)}, np.array([0.1, 0.3, 0.2, 0.25, 0.15]))
    m = C.product(g, eta, phi)
    w = C.reference_weights(tab)
    pred = C.predict(C.build_columns(avg, tab), tab, C.pair(tab, m, w, legendre={"eta": 3}, harmonic=True), w)
    ref = C.direct(avg, m, tab)
    err = np.max(np.abs(pred - ref)) / np.max(np.abs(ref))
    _record("predict_vs_direct_legendre_harmonic_conditional", err)
    assert err < 1e-12


# ------------------------------------------------------------------ A2, A3
@pytest.fixture(scope="module")
def e1():
    avg = E.e1_kernel()
    out = {"avg": avg, "timing": {}}
    for N in (2, 4):
        tab = E.e1_table(N)
        t0 = time.perf_counter()
        cols = C.build_columns(avg, tab)
        t1 = time.perf_counter()
        cols = C.build_columns(avg, tab)
        t2 = time.perf_counter()
        out[N] = (tab, cols)
        out["timing"][f"jet_N{N}_first_s"] = t1 - t0
        out["timing"][f"jet_N{N}_cached_s"] = t2 - t1
        out["timing"][f"jet_N{N}_per_cell_cached_s"] = (t2 - t1) / tab.n_cells
    tab0 = E.e1_table(0)
    m = E.power_law_measure(tab0)
    t0 = time.perf_counter()
    out["direct"] = C.direct(avg, m, tab0, azimuth=E.AZIMUTH)
    out["timing"]["direct_456_nodes_s"] = time.perf_counter() - t0
    t0 = time.perf_counter()
    out["primal"] = [np.asarray(x) for x in avg({"gamma": 1e4})]
    out["timing"]["one_averaged_kernel_eval_s_incl_jit"] = time.perf_counter() - t0
    _record("A3_timings", out["timing"])
    return out


@pytest.mark.parametrize("N", [0, 2, 4])
def test_A2_e1_representation_error(e1, N):
    tab4, cols4 = e1[4]
    tab = E.e1_table(N)
    keep = np.array([sum(r) <= N for r in tab4.row_exp])  # 1-D lower sets nest, so slice the N = 4 rows
    cols = {k: v[keep] for k, v in cols4.items()}
    w = E.e1_weights(tab)
    a = C.pair(tab, E.power_law_measure(tab), w)
    assert np.allclose(a["M0"][tab.row_exp[:, 0] == 0], 1.0, rtol=1e-12)  # unit column of the w power law
    pred = C.predict(cols, tab, a, w, azimuth=E.AZIMUTH)
    err = np.max(np.abs(pred - e1["direct"]) / e1["direct"][:, :1], axis=0)
    stored = json.loads(STUDY_JSON.read_text())["ensembles"]["E1"]["forward"][f"N{N}"]["max_resid_over_I_IQU"]
    _record(f"A2_E1_N{N}", {"prototype_IQU": err.tolist(), "study_IQU": stored, "max": float(err.max()),
                            "stored_max": STORED[N], "ratio": float(err.max() / STORED[N])})
    if N == 2:  # the N = 2 build agrees with the N = 4 slice
        d = np.max(np.abs(e1[2][1]["I"] - cols["I"])) / np.max(np.abs(cols["I"]))
        _record("A2_N2_build_vs_N4_slice", d)
        assert d < 1e-12
    assert abs(err.max() / STORED[N] - 1.0) < 0.10


def test_A2_columns_against_study_stack(e1):
    sys.path.insert(0, str(STUDY_LIB))
    import basis as SB  # noqa: E402  (study code)
    from config import PAPER  # noqa: E402

    DI, DQ = SB.derivative_spectra(PAPER, 4)  # (19, 5, 41) analytic derivative stack
    tab, cols = e1[4]
    I = cols["I"].reshape(19, 5, -1)
    P = cols["P"].reshape(19, 5, -1)
    res = {}
    for r in range(5):
        for name, ours, study in (("I", I[:, r], DI[:, r]), ("P", P[:, r], DQ[:, r])):
            chan_max = np.max(np.abs(study), axis=0)  # per channel over cells
            big = np.abs(study) > 1e-8 * chan_max
            res[f"{name}_r{r}"] = {
                "max_over_channel_max": float(np.max(np.abs(ours - study) / chan_max)),
                "max_own_rel_where_above_1e-8": float(np.max(np.abs(ours - study)[big] / np.abs(study)[big])),
                "cells_1_10_19_own_rel": [float(np.max(np.abs(ours[b] - study[b]) / np.abs(study[b]).max()))
                                          for b in (0, 9, 18)]}
    _record("A2_columns_vs_study", res)
    sep = SB.separable_columns(PAPER, DI[:, :1], DQ[:, :1])  # (J, 3, 19) with w_b and azimuth factor
    ours = np.stack([I[:, 0], (E.AZIMUTH * P[:, 0]).real, (E.AZIMUTH * P[:, 0]).imag], -1) * E.e1_weights(E.e1_table(0))[:, None, None]
    d0 = float(np.max(np.abs(ours.transpose(1, 2, 0) - sep)) / np.max(np.abs(sep)))
    _record("A2_order0_separable_columns_max_rel", d0)
    assert res["I_r0"]["max_over_channel_max"] < 1e-6 and d0 < 1e-6


def test_A3_timings_recorded(e1):
    t = e1["timing"]
    assert all(np.isfinite(v) and v > 0 for v in t.values())
