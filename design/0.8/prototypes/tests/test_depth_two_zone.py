"""Spec Sec. 4.3, 4.4 and stress row (d): a two-zone population with electron index correlated with depth.

Zone A: gamma^-2.2, column 0.3, depth uniform on one interval. Zone B: gamma^-3.0, column 0.7,
depth uniform on another. E1 kernel (B ~ U[4, 6] uG, eta on [0, 1/2]) without a screen, paper
channels 0.1 to 10 GHz, azimuth factor 0.4 e^{0.5i}. Prediction on 19 energy cells x n depth
cells with closed-form phase weights, against direct integration with the exact phase. The
"depth part" compares with the same energy expansion and the exact depth average, which
isolates the depth truncation. The first run builds the E1 node cache (about 5 min).
"""
from __future__ import annotations

import pytest

import core as C
from depth_common import record, verdict
from rules.depth import check as CK
from rules.depth import study as S

LOWERS = {"total2": ((2, 2), 2), "tensor2x2": ((2, 2), 4), "total4": ((4, 4), 4), "gamma2_depth4_total4": ((2, 4), 4)}
ND = (1, 2, 4, 8, 16, 32, 128)


@pytest.fixture(scope="module", params=["rotated", "near_zero"])
def study(request):
    return request.param, S.part_a(name=request.param, n_depth=ND, lowers=LOWERS)


def test_two_zone_forward(study):
    name, r = study
    rows = {}
    for k, v in r["cases"].items():
        rows[k] = {kk: v[kk] for kk in v if kk in ("x", "spec_passes", "proposed_bound", "I", "absP_over_I",
                                                     "depth_part_absP_over_I", "max_IQU")}
        if "depth_part_absP_over_I" in v:
            rows[k]["depth_part_verdict"] = verdict(v["depth_part_absP_over_I"])
    record(f"two_zone_{name}", {"P_over_I_max": r["P_over_I_max"], "tau_max": r["tau_max"], "cases": rows})
    base = {2: r["cases"]["char_gamma2"], 4: r["cases"]["char_gamma4"]}
    for k, v in r["cases"].items():
        if not k.startswith("nd"):
            continue
        caps, total = LOWERS[v["lower"]]
        dep = v["depth_part_absP_over_I"]
        assert dep < 0.1  # finite and no blow-up, even at x = 5.6
        if total >= caps[0] + caps[1]:  # depth outside the total: the static factor bounds the depth part
            assert dep <= 0.4 * CK.spec_factor(v["x"], caps[1]) + 1e-12  # |c| = 0.4
        if CK.spec_factor(v["x"], CK.effective_depth_order(C.lower_set(caps, total))) <= 1e-3:
            assert v["max_IQU"] <= base[caps[0]]["max_IQU"] + 1e-3
        if v["n_depth_cells"] >= 16 and total >= caps[0] + caps[1]:
            assert abs(v["max_IQU"] - base[caps[0]]["max_IQU"]) < 1e-6  # only the energy expansion remains


def test_spin0_rows_repeat_one_spectrum_per_energy_row():
    r = S.explicit_vs_factorised("cheap", nd=8)
    record("two_zone_spin0_repetition", r)
    assert r["stored_spin0_spectra"] == 57 and r["rel_diff"] < 1e-13
