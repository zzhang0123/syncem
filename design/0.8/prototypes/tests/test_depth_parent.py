"""Spec Sec. 4.3 bullet 3: an averaged law with depth as a parent.

B = 5 uG (1 + 0.05 (depth - 8)) as a one-node (Delta) law, and log B normal with that median
and width 0.1 by Gauss-Hermite nodes that move with depth. Depth cell [6, 10], the first and
last E1 energy cells, cheap channel set (6 channels, 0.1 to 10 GHz). Methods compared:
the full jet (phase and law differentiated together), the Cauchy product of the closed-form
phase weights with the depth Taylor coefficients of the phase-free averaged kernel, the
frozen-node reading with stop_gradient on the nodes, and frozen nodes with density
reweighting (the stack mode of Sec. 8.3).
"""
from __future__ import annotations

import pytest

from depth_common import record
from rules.depth import study as S


@pytest.fixture(scope="module")
def res():
    return S.part_d()


def test_cauchy_product_equals_jet(res):
    record("depth_parent", res)
    for law in ("delta", "lognormal"):
        assert res[law]["cauchy_vs_jet_P"] < 1e-12 and res[law]["cauchy_vs_jet_I"] < 1e-12
        lo, hi = res[law]["I_s1_over_s0_range"]
        assert lo > 0.01  # I has nonzero s = 1 columns when the field law depends on depth


def test_frozen_nodes_drop_the_dependence_of_a_delta_law(res):
    assert res["delta"]["frozen_I_s1_max_over_true"] == 0.0
    assert res["delta"]["frozen_vs_jet_I"] > 0.05


def test_reweighting_matches_jet_for_a_law_with_a_density(res):
    assert res["lognormal"]["reweighted_vs_jet_I"] < 1e-10 and res["lognormal"]["reweighted_vs_jet_P"] < 1e-10


def test_depth_parent_predict_equals_direct(res):
    assert res["delta_predict_vs_direct"]["max_IQU"] < 1e-3
