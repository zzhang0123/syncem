"""Partition of predict - direct(full support) into representation + outside_lattice (Spec Sec. 7.2).

Each piece comes from its own quadrature (see rules/partition/ledger.py), so the identity is
tested, not assumed. Lattices: the 19 E1 cells, a subset with uncovered support on both sides,
a lattice with an interior hole, the lowest and the highest cell alone, and one cell over the
whole E1 range (half-width 4.60) and five cells (0.92). Populations: an unbounded power law gamma^-2.6 on [1, inf),
a cut-off power law exp(-gamma/1e6) on [100, inf), and the bounded E1 power law. Channels:
the 41 E1 channels (0.1 to 10 GHz). All values are max over channels and I, Q, U of |dX| / I_j
with I_j the direct intensity over the full support.

Run from the prototypes directory (first run fills rules/partition/cache, about 15 min):
PYTHONPATH=.:../../.. python -m pytest tests/test_partition_identity.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rules.partition import run as R

RESULTS = Path(__file__).with_name("partition_results.json")
STORED = {0: 2.56e-2, 2: 8.58e-4, 4: 8.76e-5}  # Spec Sec. 9.6, E1 forward
ID_TOL = 1e-11  # identity residual with kernel-call references (independent quadratures)
TABLE_ID_TOL = 1e-13  # identity residual with the 17-node table as the representation reference, E1 cells


def _record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data.setdefault("tests_identity", {})[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


CASES = [(lat, pop, N) for lat in R.E1_LATTICES for pop in R.POPULATIONS for N in (0, 2, 4)]


@pytest.mark.parametrize("lattice,pop,N", CASES)
def test_identity_e1_lattices(lattice, pop, N):
    s = R.partition_e1(lattice, pop, N)
    _record(f"{lattice}|{pop}|N{N}", s)
    # I_cells is zero in channels where a lone low cell emits nothing, so only that ratio may be nan
    assert all(np.isfinite(v) for k, v in s.items() if k != "representation_over_I_cells")
    # predict - D_full = representation - outside_lattice, with outside_lattice = D_full - D_cells
    assert s["identity_residual_kernel_reference"] < ID_TOL
    assert s["identity_residual_table_reference"] < TABLE_ID_TOL
    # the sign of Spec Sec. 7.2 (representation + outside_lattice) misses by 2 |outside_lattice|
    if s["outside_lattice"] > 1e-12:
        assert abs(s["identity_residual_spec_sign"] / (2 * s["outside_lattice"]) - 1.0) < 1e-6
    # the scalar terms bound the total (triangle inequality), they do not add to it
    assert s["envelope_over_total"] >= 1.0 - 1e-12


@pytest.mark.parametrize("pop", list(R.POPULATIONS))
@pytest.mark.parametrize("N", [0, 2, 4])
def test_identity_one_wide_cell(pop, N):
    s = R.partition_one_cell(pop, N)
    _record(f"one_cell|{pop}|N{N}", s)
    assert s["identity_residual_kernel_reference"] < ID_TOL
    assert s["envelope_over_total"] >= 1.0 - 1e-12
    # at half-width 4.60 the 17-node table is not a reference for the cell integral
    assert s["table_vs_kernel_direct_in_cells"] > 1e-2


@pytest.mark.parametrize("N", [0, 2, 4])
def test_bounded_e1_reproduces_paper_numbers(N):
    """The E1 population lies inside the 19 cells: outside_lattice is zero and representation is A2."""
    s = R.partition_e1("E1_19", "PL_E1_bounded", N)
    assert s["outside_lattice"] == 0.0
    assert abs(s["representation"] / STORED[N] - 1.0) < 0.01


@pytest.mark.parametrize("N", [0, 2, 4])
def test_unbounded_power_law_outside_is_upper_tail(N):
    """For gamma^-2.6 on [1, inf) the 19 E1 cells miss the emission above 2.96e6 (and nothing below)."""
    s = R.partition_e1("E1_19", "PL_unbounded", N)
    b = R.partition_e1("E1_19", "PL_E1_bounded", N)
    assert 1e-6 < s["outside_lattice"] < 1e-2
    assert abs(s["representation"] - b["representation"]) <= 1e-3 * b["representation"]


@pytest.mark.parametrize("lattice", ["E1_cells_2_to_16", "E1_hole_at_9", "E1_cell_0_only", "E1_cell_18_only"])
def test_node_mask_is_not_within_cells(lattice):
    """Masking face-blind quadrature nodes is not direct(within cells). Pinned as a regression case."""
    s = R.partition_e1(lattice, "PL_unbounded", 2)
    _record(f"naive_mask|{lattice}", s["naive_mask_within_cells_error"])
    assert s["naive_mask_within_cells_error"] > 10 * s["identity_residual_kernel_reference"]


def test_small_x_tail_rule():
    """Tail above 3e7 by the x^(1/3) asymptote, checked against panels on [1e7, 3e7] plus the tail."""
    d = R.L.tail_check(R.POPULATIONS["PL_unbounded"], np.r_[0:68])
    _record("tail_check_PL_unbounded", d)
    assert d < 1e-7


@pytest.mark.parametrize("pop", list(R.POPULATIONS))
@pytest.mark.parametrize("N", [0, 2, 4])
def test_identity_five_cells(pop, N):
    s = R.partition_n_cells(5, pop, N)
    _record(f"five_cells|{pop}|N{N}", s)
    assert s["identity_residual_kernel_reference"] < ID_TOL
    assert s["envelope_over_total"] >= 1.0 - 1e-12
