"""Provider dispatch boundary (Spec Sec. 8.3, T2): jet, study stack and 17-node table at every cell.

Both providers are evaluated directly, without a dispatcher, on the E1 configuration with
1, 5, 19 and 200 equal ln-gamma cells (half-widths 4.60, 0.92, 0.242, 0.023), N = 0..4 in
all cells and N = 6 in the first and last cell. Scales: the channel column maximum per r,
the channel order-0 maximum, the cell's own peak per r and the cell's order-0 own peak, and
pointwise values at cell 1 / lowest channel and top cell / highest channel.
The jet uses 64 channel nodes, the stack 64 (same quadrature) and 400 (the study's).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from rules.partition import providers as Pv
from rules.partition import setup as S

RESULTS = Path(__file__).with_name("partition_results.json")
SPEC_CHAN = 1e-10  # Spec Sec. 8.3: build fails above this against the channel column maximum (r <= 2)
SPEC_OWN = 1e-5  # Spec Sec. 8.3: warning above this against the cell's own peak


def _record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data.setdefault("tests_providers", {})[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


def _cols(n, N, keep, which):
    cells = Pv.e1_cells(n, N)
    if which == "jet":
        I, P, _ = Pv.jet_columns(cells, N, keep=keep)
        return I[..., S.E1_CH], P[..., S.E1_CH]
    if which == "table":
        I, P = Pv.table_columns(cells, N, keep=keep)
        return I[..., S.E1_CH], P[..., S.E1_CH]
    I, P = Pv.stack_columns(n, N, int(which[5:]))
    k = np.arange(n) if keep is None else np.asarray(keep)
    return I[k], P[k]


LATTICES = [(19, 4, None), (19, 6, [0, 18]), (1, 4, [0]), (5, 4, [0, 4]), (200, 4, [0, 100, 199])]


@pytest.mark.parametrize("n,N,keep", LATTICES)
def test_jet_vs_stack_same_quadrature(n, N, keep):
    jI, jP = _cols(n, N, keep, "jet")
    sI, sP = _cols(n, N, keep, "stack64")
    res = {"I": Pv.compare(jI, sI, probes=[(0, 0), (len(sI) - 1, 40)]),
           "P": Pv.compare(jP, sP, probes=[(0, 0), (len(sI) - 1, 40)])}
    _record(f"jet_vs_stack64|n{n}|N{N}", res)
    for X in ("I", "P"):
        for r in range(N + 1):
            c = res[X][r]
            assert np.isfinite(c["chan_all"]) and np.isfinite(c["own"])
            assert c["own"] < 2e-9  # own peak of the same cell and order, every case
            assert max(c["probes"]) < 1e-9  # cell 1 at 0.1 GHz, top cell at 10 GHz
            if r <= 2:
                assert c["chan_all"] < SPEC_CHAN  # the build check of Spec Sec. 8.3


@pytest.mark.parametrize("n,N,keep", LATTICES)
def test_table17_vs_jet(n, N, keep):
    tI, tP = _cols(n, N, keep, "table")
    jI, jP = _cols(n, N, keep, "jet")
    res = {"I": Pv.compare(tI, jI), "P": Pv.compare(tP, jP)}
    _record(f"table17_vs_jet|n{n}|N{N}", res)
    worst = max(res[X][r]["chan_all"] for X in ("I", "P") for r in range(3))
    for X in ("I", "P"):
        assert res[X][0]["chan_all"] < 1e-14  # u = 0 is a Lobatto node, r = 0 is a kernel value
    if n in (19, 200):  # half-widths 0.242 and 0.023
        assert worst < SPEC_CHAN
    else:  # half-widths 0.92 and 4.60: the 17-node table fails the build check
        assert worst > 1e-4


def test_table17_own_peak_in_tail_cell_exceeds_spec_warning():
    """E1 cell 1 (exponential tail at every channel): the table misses the own-peak 1e-5 at r = 1, 2."""
    tI, tP = _cols(19, 4, None, "table")
    jI, jP = _cols(19, 4, None, "jet")
    cI, cP = Pv.compare(tI, jI), Pv.compare(tP, jP)
    _record("table17_own_peak_E1", {"I": [cI[r]["own"] for r in range(5)], "P": [cP[r]["own"] for r in range(5)],
                                    "worst_cell_I": [cI[r]["own_worst_cell"] for r in range(5)]})
    assert cI[1]["own"] > SPEC_OWN and cP[1]["own"] > SPEC_OWN
    assert cI[1]["own_worst_cell"] == 0


def test_channel_quadrature_dominates_jet_vs_stack400():
    """The study at 400 channel nodes differs from both 64-node providers by more than they differ."""
    jI, jP = _cols(19, 4, None, "jet")
    s64 = _cols(19, 4, None, "stack64")
    s400 = _cols(19, 4, None, "stack400")
    a = Pv.compare(jP, s400[1])[0]["chan0"]
    b = Pv.compare(jP, s64[1])[0]["chan0"]
    _record("P_r0_chan0_jet_vs_stack400_and_64", [a, b])
    assert a > 100 * b
