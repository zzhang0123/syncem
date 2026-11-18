"""(b) E4-KP: a pitch density that depends on gamma, under Beaming, against direct integration.

Columns: the stored eta-resolved jet (19 E1 cells, N = 4, 32 half-range eta nodes of U[-1/2, 1/2])
contracted with (2l+1) P_l(eta). Moments: core.pair of the KP density over (gamma, mu) on the full
pitch range, <1_b u^r P_l(mu)>. Reference: the study's stored direct spectrum (forward_spectra.npz),
which integrates q(eta) 2 N(gamma, mu = eta) Kbar. The study's errors (forward_results.json,
forward_pitch_legendre_vs_L) are 7.84e-3 at L = 4 and 8.83e-3 at L = 8 for N = 2.
Also: the prototype direct against the stored truth, the factor 1, and Cells on mu (G9 comment).
Run: PYTHONPATH=.:../../.. python -m pytest tests/test_beaming_kp.py
"""
from __future__ import annotations

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
import pytest

import core as C
from core import paper as E
from rules.beaming import beaming as BM
from rules.beaming import cases as CS
from rules.beaming import kp
from rules.beaming import store
from rules.beaming.record import record

LMAX = 16


def rel3(S, P):
    """Study metric: max_j |S - P| / I_j per Stokes component [I, Q, U]."""
    return [float(v) for v in np.max(np.abs(S - P) / S[:, 0:1], axis=0)]


@lru_cache(maxsize=1)
def legendre_moments():
    tab = E.e1_table(4)
    return C.pair(tab, kp.pair_measure(), E.e1_weights(tab), legendre={"mu": LMAX})["M0"]


@lru_cache(maxsize=1)
def entry():
    return store.load("half32_all_N4")


def predict(L, N, *, mirrored=False, mask_odd=False, factor=2.0):
    d = entry()
    eta, q = BM.half_range(-0.5, 0.5, 32)
    I, P = d["I"], d["P"]
    if mirrored:  # full range by mirrored nodes. K is even in eta, so the stored values are reused
        eta, q = BM.mirror(eta, q)
        I = np.concatenate([I[..., ::-1], I], axis=-1)
        P = np.concatenate([P[..., ::-1], P], axis=-1)
    cols = BM.contract({"I": I, "P": P}, BM.pitch_weights(eta, q, L, factor, mask_odd))
    keep = np.tile(np.arange(5), 19) <= N
    tab = E.e1_table(N)
    mom = {"M0": legendre_moments()[keep][:, : L + 1], "M2": None}
    return C.predict({k: v[keep] for k, v in cols.items()}, tab, mom, E.e1_weights(tab), azimuth=E.AZIMUTH)


def test_moments_parity_and_mass():
    M = legendre_moments()
    w = E.e1_weights(E.e1_table(4))
    out = {"order0_l0_column": float(np.sum(w * M[::5, 0])),
           "max_abs_odd_l_over_l0": float(np.max(np.abs(M[:, 1::2])) / np.max(np.abs(M[:, 0])))}
    record("b_kp_moments", out)
    assert abs(out["order0_l0_column"] - 1.0) < 1e-6
    assert out["max_abs_odd_l_over_l0"] < 1e-14


@pytest.mark.parametrize("L", [0, 2, 4, 8, 16])
@pytest.mark.parametrize("N", [2, 4])
def test_kp_forward_error_matches_study(N, L):
    truth = kp.stored_truth()
    study = kp.stored_forward()[f"N{N}"][f"L{L}"]
    plan = predict(L, N, mask_odd=True)           # half range with the parity mask (Spec route)
    full = predict(L, N, mirrored=True)           # full range, all degrees
    out = {"prototype_IQU": rel3(truth, plan), "study_IQU": study,
           "half_mask_vs_full_all": C.representation_error(plan, full)}
    out["ratio_I"] = out["prototype_IQU"][0] / study[0]
    record(f"b_kp_forward|N{N}|L{L}", out)
    assert out["half_mask_vs_full_all"] < 1e-12
    assert abs(out["ratio_I"] - 1.0) < 1e-3


def test_kp_factor_one_gives_half():
    truth = kp.stored_truth()
    pred = predict(8, 4, mask_odd=True, factor=1.0)
    out = {"IQU": rel3(truth, pred), "I_ratio_range": [float(np.min(pred[:, 0] / truth[:, 0])),
                                                       float(np.max(pred[:, 0] / truth[:, 0]))]}
    record("b_kp_factor1", out)
    assert out["IQU"][0] > 0.49


def test_kp_direct_prototype_vs_stored_truth():
    """Prototype direct (core.direct, 64 channel nodes) with 12 gamma and 8 eta nodes per panel."""
    import ensembles as ENS  # study oracle, on sys.path through rules.beaming.kp
    from config import PAPER

    avg = C.AveragedKernel(store.base_kernel(), [CS.B_RULE])
    proto = C.direct(avg, kp.direct_measure(12, 8), azimuth=E.AZIMUTH)
    study_12_8 = kp.population().direct(PAPER, n=12, n_eta=8)
    truth = kp.stored_truth()
    out = {"proto_vs_study_same_nodes": C.representation_error(proto, study_12_8),
           "study_12_8_vs_truth_24_16": C.representation_error(study_12_8, truth),
           "proto_vs_truth": C.representation_error(proto, truth)}
    record("b_kp_direct", out)
    assert out["proto_vs_truth"] < 1e-6


# ------------------------------------------------------------------ Cells on mu with eta averaged
MU_EDGES = {"2cells": [-0.5, 0.0, 0.5], "4cells": [-0.5, -0.25, 0.0, 0.25, 0.5]}
_MU_AVG: dict = {}


def mu_cells_avg():
    if "avg" not in _MU_AVG:
        kern = BM.DiagonalDensityKernel(store.base_kernel(), lambda m: jnp.where(jnp.abs(m) <= 0.5, 1.0, 0.0))
        _MU_AVG["avg"] = C.AveragedKernel(kern, [CS.B_RULE])
    return _MU_AVG["avg"]


@pytest.mark.parametrize("N", [2, 4])
@pytest.mark.parametrize("cells", list(MU_EDGES))
def test_kp_mu_cells_eta_density(cells, N):
    """gamma x mu product cells, g = 2 q(mu) Kbar(gamma, eta = mu), q = U[-1/2, 1/2]. Cells |mu| > 1/2 are dropped (support)."""
    edges = MU_EDGES[cells]
    tab = C.ProductTable.tensor({"gamma": C.Cells.log(E.GAMMA_LO, E.GAMMA_HI, 19, order=N),
                                 "mu": C.Cells.from_edges(edges, chart="affine", order=N)}, total=N)
    cols = C.build_columns(mu_cells_avg(), tab)
    w = C.reference_weights(tab)
    meas = kp.pair_measure(extra_breaks=[abs(e) for e in edges])
    pred = C.predict(cols, tab, C.pair(tab, meas, w), w, azimuth=E.AZIMUTH)
    truth = kp.stored_truth()
    out = {"IQU": rel3(truth, pred), "max": C.representation_error(pred, truth), "n_rows": int(tab.n_rows)}
    out["verdict"] = BM.verdict(out["max"])
    record(f"b_kp_mu_cells|{cells}|N{N}", out)
    assert out["max"] < 5e-2


def test_mu_cell_outside_support_has_zero_columns():
    tab = C.ProductTable.tensor({"gamma": C.Cells.log(E.GAMMA_LO, E.GAMMA_HI, 19, order=2),
                                 "mu": C.Cells.from_edges([0.5, 1.0], chart="affine", order=2)}, total=2)
    tab = tab.select(np.arange(tab.n_cells) == 9)
    cols = C.build_columns(mu_cells_avg(), tab)
    out = float(np.max(np.abs(cols["I"])) + np.max(np.abs(cols["P"])))
    record("b_kp_mu_cell_outside_support_max_abs", out)
    assert out == 0.0
