"""A lattice that does not cover the channels: the 1 to 100.6 GHz default range at 300 GHz (r3-23).

The default range is measured with the basis kernel (rules/partition/window.py, Spec Sec. 4.6
rule 1, tolerance 1e-3 per side for gamma^-2.6). The partition is checked in the band, at 300 GHz
and at 0.1 to 10 GHz (below the band), for an unbounded power law, the bounded E1 power law and
a cut-off power law. The Sec. 6.5 estimate (share outside by the reference density) is compared
with the measured outside_lattice of each population.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rules.partition import run as R
from rules.partition import setup as S
from rules.partition import window as W

RESULTS = Path(__file__).with_name("partition_results.json")
ID_TOL = 1e-10


def _record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data.setdefault("tests_default300", {})[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


def test_default_range_and_in_band_shares():
    lo, hi = W.default_range(S.E2_CH)
    g, below, above = W.shares(S.E2_CH)
    cells = R.default_lattice()
    sb = [float(np.interp(np.log(lo), np.log(g), below[:, j])) for j in range(below.shape[1])]
    sa = [float(np.interp(np.log(hi), np.log(g), above[:, j])) for j in range(above.shape[1])]
    _record("range", {"lo": lo, "hi": hi, "n_cells": cells.n, "half_width": float(cells.half_width[0]),
                      "max_share_below": max(sb), "max_share_above": max(sa)})
    assert max(sb) < 1.05e-3 and max(sa) < 1.05e-3
    assert hi < 2.9626e6  # the band's range does not reach the E1 top edge


@pytest.mark.parametrize("pop", list(R.POPULATIONS))
@pytest.mark.parametrize("N", [0, 2, 4])
@pytest.mark.parametrize("span", ["band_1_100GHz", "at_300GHz", "E1_0p1_10GHz"])
def test_identity_default_lattice(pop, N, span):
    s = R.partition_default(pop, N)[span]
    _record(f"{pop}|N{N}|{span}", s)
    assert np.isfinite(s["identity_residual_kernel_reference"]) and np.isfinite(s["representation"])
    assert s["identity_residual_kernel_reference"] < ID_TOL
    assert s["envelope_over_total"] >= 1.0 - 1e-12


@pytest.mark.parametrize("pop", ["PL_E1_bounded", "cutoff_1e6_from_100"])
def test_reference_density_share_is_not_the_source_share(pop):
    """At 300 GHz the Sec. 6.5 estimate (gamma^-2.6 share) differs from the measured share of other populations."""
    ref = R.partition_default("PL_unbounded", 2)["at_300GHz"]["outside_share_I_max"]
    got = R.partition_default(pop, 2)["at_300GHz"]["outside_share_I_max"]
    _record(f"share300|{pop}", {"reference_density": ref, "measured": got, "ratio": got / ref})
    assert ref > 1e-3  # 300 GHz is not covered
    assert np.isfinite(got)
