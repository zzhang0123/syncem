"""Spec Sec. 4.2 and R4: the independent-screen characteristic-function route against resolved depth cells.

Gaussian screen N(m, sd^2) independent of gamma ~ PowerLaw(2.6). Route 1: depth averaged,
C_scr(tau) = exp(i tau m - tau^2 sd^2 / 2) inside the channel integral (57 rows). Route 2:
depth cells on [m - 6 sd, m + 6 sd] with closed-form phase weights and Gaussian moments by
quadrature. Both use the same energy expansion (19 cells, N = 2), so their difference is the
depth truncation plus the 2e-9 tail mass beyond 6 sd. E1 kernel, paper channels, azimuth
0.4 e^{0.5i}. Extremes: sd 0.02 and 0.5, mean 0 and 300 rad m^-2, x up to 2.
"""
from __future__ import annotations

import pytest

from depth_common import record, verdict
from rules.depth import check as CK
from rules.depth import study as S

XS = (0.1, 0.95 * CK.x_limit(2), CK.x_limit(2), 1.05 * CK.x_limit(2), 0.3, 0.95, 1.0, 1.05, 2.0)


@pytest.mark.parametrize("mean", [0.0, 8.0, 300.0])
@pytest.mark.parametrize("sd", [0.02, 0.1, 0.5])
def test_screen_route_equals_depth_cells(sd, mean):
    r = S.part_c(sds=(sd,), means=(mean,), xs=XS, max_cells=6000)
    out = {}
    for row in r["rows"]:
        e = row["cells_vs_char"]["absP_over_I"]
        out[f"x{row['x']:.4g}_{row['lower']}"] = {"cells_vs_char": e, "verdict": verdict(e), "n_cells": row["n_cells"],
                                                 "char_vs_direct_max_IQU": row["char_vs_direct"]["max_IQU"]}
        assert e < 0.1
        if row["lower"] == "tensor2x2" and row["x"] <= CK.x_limit(2) * (1 + 1e-12):
            assert e < 1e-3  # the proposed rule admits this cell, and the routes agree
        if row["x"] <= 0.3:
            assert e < 1e-3  # both lower sets agree with the screen route well inside the spec threshold
    record(f"screen_sd{sd}_m{mean}", out)
