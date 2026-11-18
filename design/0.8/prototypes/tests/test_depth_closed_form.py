"""Spec Sec. 4.3 bullet 2: closed-form phase weights in the spin-2 block, I merged across depth cells.

Two methods for the same columns: the jet with depth as a differentiated direction (the
general provider) and the closed-form weights exp(i tau d_c) (i tau h_c)^s / s! applied to
the energy columns at every channel node. Run from the prototypes directory:
PYTHONPATH=.:../../.. python -m pytest tests/test_depth_closed_form.py
"""
from __future__ import annotations

import numpy as np
import pytest

import core as C
from core import paper as E
from rules.depth import closed_form as CF
from rules.depth import nodes as Nd
from rules.depth import study as S
from depth_common import record

CH = C.bump_channels(n=6, nu_lo=1e8, nu_hi=1e10, n_nodes=32)
G_EDGES = C.Cells.log(E.GAMMA_LO, E.GAMMA_HI, 19).edges
DEPTHS = {
    "moderate": [0.9, 1.0, 1.1],
    "wide": [6.0, 10.0],
    "extreme": [-300.1, -300.0, 299.95, 300.0],  # phases up to tau d = 8400 rad at 0.08 GHz
}


def _tables(caps, total, depth):
    g = C.Cells.from_edges(G_EDGES[[0, 1, -2, -1]], chart="log", order=caps[0])  # first, middle, last E1 cells
    d = C.Cells.from_edges(DEPTHS[depth], chart="affine", order=caps[1])
    tab = C.ProductTable.tensor({"gamma": g, "depth": d}, total=total)
    keep = tab.index[:, 0] != 1  # drop the wide middle gamma cell
    if depth == "extreme":
        keep &= tab.index[:, 1] != 1  # and the gap between -300 and 299.95
    return C.ProductTable.tensor({"gamma": g}, total=caps[0]), tab.select(keep)


@pytest.mark.parametrize("depth", list(DEPTHS))
@pytest.mark.parametrize("caps,total", [((2, 2), 2), ((2, 2), 4), ((0, 3), 3), ((4, 1), 4), ((1, 4), 4)])
def test_closed_form_equals_jet(caps, total, depth):
    gt, tab = _tables(caps, total, depth)
    avg_nodes = Nd.node_kernel(CH, "cheap")
    ec = Nd.energy_columns(avg_nodes, gt, CH.n_ch)
    cf = CF.assemble(ec, gt, tab, CH, Nd.tau(CH))
    avg = C.AveragedKernel(C.ContinuumChannelKernel(CH), [("eta", Nd.eta_rule(4))], fixed={"B": 5.0})
    jet = C.build_columns(avg, tab)
    s = tab.row_exp[:, 1]
    own = lambda a, b: float(np.max(np.abs(a - b).max(axis=1) / np.maximum(np.abs(b).max(axis=1), 1e-300)))
    errP = own(cf["P"], jet["P"])
    errI = own(cf["I"][s == 0], jet["I"][s == 0])
    zeroI = float(np.max(np.abs(jet["I"][s > 0]))) if np.any(s > 0) else 0.0
    # the s = 0 I rows of one gamma row are equal across depth cells
    spread = 0.0
    for g in set(cf["spin0_from"][s == 0].tolist()):
        rows = jet["I"][cf["spin0_from"] == g]
        spread = max(spread, float(np.max(np.abs(rows - rows[0])) / np.max(np.abs(rows[0]))))
    record(f"closed_form_vs_jet_caps{caps}_total{total}_{depth}",
           {"P_own_row_rel": errP, "I_own_row_rel": errI, "I_s_gt_0_max": zeroI, "I_spread_across_depth_cells": spread,
            "rows": tab.n_rows})
    assert errP < 1e-10 and errI < 1e-12 and zeroI == 0.0 and spread < 1e-14


def test_explicit_rows_equal_factorised_prediction():
    r = S.explicit_vs_factorised("cheap", nd=8)
    record("explicit_vs_factorised", r)
    assert r["rel_diff"] < 1e-13
    assert r["stored_spin0_spectra"] == 57 and r["I_rows_used"] == 19 * 8 * 3 and r["rows"] == 19 * 8 * 6


@pytest.mark.parametrize("d_zone", [(0.93, 1.05), (299.0, 299.4)])
def test_factorised_direct_equals_core_direct(d_zone):
    """direct_zones (exact characteristic function) against core.direct with the phase in the kernel."""
    ch = C.bump_channels(n=4, nu_lo=1e8, nu_hi=1e9, n_nodes=32)
    tau = Nd.tau(ch)
    g = C.Cells.log(300, 3e6, 3)
    d = C.Cells.linear(*d_zone, 2)
    z = CF.Zone(CF.power_law_gamma(2.2, 300, 3e6, g.edges, n=4), *CF.uniform_law(*d_zone, d, n=16))
    vals = Nd.values_at(Nd.node_kernel(ch, "cheap"), {"gamma": z.gamma.nodes["gamma"]}, ch.n_ch)
    r1 = CF.direct_zones([vals], [z], ch, tau, azimuth=S.AZ)
    avg = C.AveragedKernel(C.ContinuumChannelKernel(ch), [("eta", Nd.eta_rule(4))], fixed={"B": 5.0})
    r2 = C.direct(avg, CF.joint_measure([z]), None, azimuth=S.AZ)
    err = float(np.max(np.abs(r1 - r2)) / np.max(np.abs(r2)))
    record(f"direct_zones_vs_core_direct_{d_zone}", err)
    assert err < 1e-8  # Gauss-Legendre depth nodes against the exact characteristic function
