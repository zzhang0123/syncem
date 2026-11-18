"""(a) Pitch-Legendre columns under Beaming as eta weights (2l+1) P_l(eta), against the study stack.

Prototype: core jet on E1 energy cells (64 channel nodes) with eta as a node axis, contracted with
pitch_weights. Oracle: ST/lib/basis.py channel_kernels (spline table, 400 channel nodes) with the
weights of basis.eta_weights. With the study's 32 half-range nodes the oracle equals
basis.derivative_spectra(cfg, N, pitch_l=l), which the first test checks.
Builds come from rules/beaming/store.py (about 7 min for 19 cells at N = 4 on the first run).
Run: PYTHONPATH=.:../../.. python -m pytest tests/test_beaming_columns.py
"""
from __future__ import annotations

from functools import lru_cache

import numpy as np
import pytest

import core as C
from rules.beaming import beaming as BM
from rules.beaming import cases as CS
from rules.beaming import store
from rules.beaming.record import record

LAWS = {  # store entry -> (nodes, probabilities) of the eta law on those nodes
    "half32_all_N4": lambda: BM.half_range(-0.5, 0.5, 32),
    "near0_N4": lambda: BM.uniform_law(-0.02, 0.02, 16),
    "iso_N4": lambda: BM.uniform_law(-1.0, 1.0, 32, panels=2),
    "near1_N4": lambda: BM.union_law([(-1.0, -0.95), (0.95, 1.0)], 16),
    "asym_N4": lambda: BM.uniform_law(0.0, 0.5, 32),
}
TOL = 1e-6  # relative to the channel maximum of the column block; 64 against 400 channel nodes


@lru_cache(maxsize=8)
def entry(name):
    return store.load(name)


@lru_cache(maxsize=64)
def oracle(name, ell):
    d = entry(name)
    eta, q = LAWS[name]()
    return CS.study_columns(eta, q, tuple(int(c) for c in d["cells"]), int(d["N"]), ell)


def compare(name, ell, L=16):
    eta, q = LAWS[name]()
    PI, PP = CS.prototype_columns(entry(name), q, L)
    SI, SQ = oracle(name, ell)
    pI, pP = PI[..., ell], PP[..., ell]
    out = {}
    for tag, a, b in (("I", pI, SI), ("P", pP, SQ)):
        scale = np.max(np.abs(b))
        out[f"{tag}_rel_to_max"] = float(np.max(np.abs(a - b)) / scale) if scale > 0 else float(np.max(np.abs(a)))
        per_r = [float(np.max(np.abs(a[:, r] - b[:, r])) / max(np.max(np.abs(b[:, r])), 1e-300))
                 for r in range(b.shape[1])]
        out[f"{tag}_rel_per_r"] = per_r
        out[f"{tag}_max_abs"] = float(scale)
    return out


def test_oracle_equals_study_derivative_spectra():
    """study_columns at the study's nodes is derivative_spectra(n_eta=32, N=4, pitch_l=l)."""
    out = {}
    for ell in (0, 1, 8, 16):
        DI, DQ = CS.study_derivative_spectra(32, 4, ell)
        SI, SQ = oracle("half32_all_N4", ell)
        out[f"l{ell}"] = max(float(np.max(np.abs(DI - SI)) / np.max(np.abs(DI))),
                             float(np.max(np.abs(DQ - SQ)) / np.max(np.abs(DQ))))
    record("a_oracle_vs_derivative_spectra", out)
    assert max(out.values()) < 1e-14


@pytest.mark.parametrize("ell", list(range(17)))
def test_pitch_columns_e1_law_all_cells(ell):
    """eta ~ U[-1/2, 1/2] on the study's half-range nodes, 19 cells, r = 0..4."""
    out = compare("half32_all_N4", ell)
    record(f"a_columns|U(-0.5,0.5)_half32|l{ell}", out)
    assert out["I_rel_to_max"] < TOL and out["P_rel_to_max"] < TOL


@pytest.mark.parametrize("ell", [0, 1, 2, 3, 8, 15, 16])
@pytest.mark.parametrize("name", ["near0_N4", "iso_N4", "near1_N4", "asym_N4"])
def test_pitch_columns_extreme_laws(name, ell):
    """Laws near 0, isotropic, near +-1 and asymmetric, cells 1, 10 and 19, r = 0..4."""
    out = compare(name, ell)
    if name in ("near0_N4", "iso_N4", "near1_N4") and ell % 2:
        # even law on a symmetric rule: odd columns of both sides vanish up to rounding
        eta, q = LAWS[name]()
        PI, _ = CS.prototype_columns(entry(name), q, 16)
        SI, _ = oracle(name, ell)
        l0 = np.max(np.abs(PI[..., 0]))
        out["proto_odd_over_l0"] = float(np.max(np.abs(PI[..., ell])) / l0)
        out["study_odd_over_l0"] = float(np.max(np.abs(SI)) / l0)
        record(f"a_columns|{name}|l{ell}", out)
        assert out["proto_odd_over_l0"] < 1e-12 and out["study_odd_over_l0"] < 1e-12
    else:
        record(f"a_columns|{name}|l{ell}", out)
        assert out["I_rel_to_max"] < TOL and out["P_rel_to_max"] < TOL


def test_odd_columns_of_even_law_full_range_vs_half_range():
    """Mirrored (full-range) nodes give zero odd columns. The study's half-range weights do not."""
    d = entry("half32_all_N4")
    eta, q = LAWS["half32_all_N4"]()
    full_eta, full_q = BM.mirror(eta, q)
    # K is even in eta (sqrt(1 - eta^2) only), so the mirrored columns reuse the stored kernel values
    I2 = np.concatenate([d["I"][..., ::-1], d["I"]], axis=-1)
    W = BM.pitch_weights(full_eta, full_q, 16)
    cI = np.einsum("rcn,nl->rcl", I2, W)
    hI = np.einsum("rcn,nl->rcl", d["I"], BM.pitch_weights(eta, q, 16))
    out = {"full_odd_over_l0": float(np.max(np.abs(cI[..., 1::2])) / np.max(np.abs(cI[..., 0]))),
           "full_even_vs_half_even": float(np.max(np.abs(cI[..., 0::2] - hI[..., 0::2])) / np.max(np.abs(hI[..., 0]))),
           "half_l1_over_l0_r0_max": float(np.max(np.abs(hI[::5, :, 1]) / np.max(np.abs(hI[::5, :, 0]))))}
    record("a_odd_columns_full_vs_half", out)
    assert out["full_odd_over_l0"] < 1e-13 and out["full_even_vs_half_even"] < 1e-13


def test_rule_route_equals_eta_node_contraction():
    """PitchLegendreKernel through AveragedKernel with the eta rule equals the stored contraction (cell 10, N = 2)."""
    eta, q = LAWS["half32_all_N4"]()
    avg = C.AveragedKernel(BM.PitchLegendreKernel(store.base_kernel(), 8),
                           [CS.B_RULE, ("eta", C.fixed(eta, q))])
    tab = store.table_for((9,), 2)
    cols = C.build_columns(avg, tab)
    PI, PP = CS.prototype_columns(entry("half32_all_N4"), q, 8)
    a, b = cols["I"], PI[9, :3]
    out = {"I": float(np.max(np.abs(a - b)) / np.max(np.abs(b))),
           "P": float(np.max(np.abs(cols["P"] - PP[9, :3])) / np.max(np.abs(PP[9, :3])))}
    record("a_rule_route_vs_contraction", out)
    assert max(out.values()) < 1e-12
