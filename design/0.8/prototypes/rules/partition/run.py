"""Computations of the partition family. Heavy parts are cached in ``cache/``.

python -m rules.partition.run   writes tests/partition_results.json
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

import core as C

from . import ledger as L
from . import nodes as Nd
from . import providers as Pv
from . import setup as S
from . import window as W

RESULTS = Path(__file__).resolve().parents[2] / "tests" / "partition_results.json"

POPULATIONS = {
    "PL_unbounded": L.power_law(),
    "cutoff_1e6_from_100": L.cutoff(1e6, 100.0),
    "PL_E1_bounded": L.power_law(300.0, float(L.E.GAMMA_HI)),
}

# E1 lattices: (description, kept cells of the 19)
E1_LATTICES = {
    "E1_19": np.arange(19),
    "E1_cells_2_to_16": np.arange(2, 17),
    "E1_hole_at_9": np.r_[0:9, 10:19],
    "E1_cell_0_only": np.array([0]),
    "E1_cell_18_only": np.array([18]),
}


def record(key: str, value) -> None:
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


def _cols(I, P, keep, N):
    return {"I": I[keep][:, :N + 1].reshape(-1, I.shape[-1]), "P": P[keep][:, :N + 1].reshape(-1, I.shape[-1])}


def _table(cells: C.Cells, keep, N) -> C.ProductTable:
    t = C.ProductTable.tensor({"gamma": cells.with_order(N)}, total=N)
    return t.select(np.isin(np.arange(cells.n), keep))


def partition_e1(lattice: str, pop: str, N: int) -> dict:
    keep = E1_LATTICES[lattice]
    cells = Pv.e1_cells(19, 4)
    I, P, _ = Pv.jet_columns(cells, 4)
    tab = _table(cells, keep, N)
    tvals = Nd.table_values(cells.reference, cells.half_width)
    tvals = (tvals[0][keep], tvals[1][keep])
    p = L.pieces(tab, _cols(I, P, keep, N), POPULATIONS[pop], table_values=tvals)
    return L.summary(p, S.E1_CH)


def partition_n_cells(n: int, pop: str, N: int) -> dict:
    """n equal cells on the E1 range (n = 1: half-width 4.60, n = 5: 0.92)."""
    cells = Pv.e1_cells(n, 4)
    I, P, _ = Pv.jet_columns(cells, 4)
    keep = np.arange(n)
    tab = _table(cells, keep, N)
    p = L.pieces(tab, _cols(I, P, keep, N), POPULATIONS[pop])
    return L.summary(p, S.E1_CH)


def partition_one_cell(pop: str, N: int) -> dict:
    return partition_n_cells(1, pop, N)


def default_lattice() -> C.Cells:
    return W.default_cells(S.E2_CH)


def partition_default(pop: str, N: int) -> dict:
    cells = default_lattice()
    I, P, _ = Pv.jet_columns(cells, 4)
    keep = np.arange(cells.n)
    tab = _table(cells, keep, N)
    p = L.pieces(tab, _cols(I, P, keep, N), POPULATIONS[pop], use_table=False)
    return {"band_1_100GHz": L.summary(p, S.E2_CH), "at_300GHz": L.summary(p, [S.C300]),
            "E1_0p1_10GHz": L.summary(p, S.E1_CH)}


def provider_cases() -> dict:
    """Jet against stack and table, per r, for the extreme lattices of T2."""
    out = {}
    lowest, highest = 0, 40
    # 19 cells, N = 4, all cells
    cells = Pv.e1_cells(19, 4)
    jI, jP, sec = Pv.jet_columns(cells, 4)
    jI, jP = jI[..., S.E1_CH], jP[..., S.E1_CH]
    probes = [(0, lowest), (18, highest), (0, highest)]
    sI64, sP64 = Pv.stack_columns(19, 4, 64)
    sI400, sP400 = Pv.stack_columns(19, 4, 400)
    tI, tP = Pv.table_columns(cells, 4)
    tI, tP = tI[..., S.E1_CH], tP[..., S.E1_CH]
    out["19_cells"] = {
        "jet_seconds_68ch": sec,
        "jet_vs_stack64": {"I": Pv.compare(jI, sI64, probes=probes), "P": Pv.compare(jP, sP64, probes=probes)},
        "jet_vs_stack400": {"I": Pv.compare(jI, sI400, probes=probes), "P": Pv.compare(jP, sP400, probes=probes)},
        "stack64_vs_stack400": {"I": Pv.compare(sI64, sI400), "P": Pv.compare(sP64, sP400)},
        "table17_vs_jet": {"I": Pv.compare(tI, jI, probes=probes), "P": Pv.compare(tP, jP, probes=probes)},
        "table17_vs_stack64": {"I": Pv.compare(tI, sI64), "P": Pv.compare(tP, sP64)},
    }
    # N = 6 in the first and last cell
    keep = [0, 18]
    jI6, jP6, sec6 = Pv.jet_columns(Pv.e1_cells(19, 6), 6, keep=keep)
    sI6, sP6 = Pv.stack_columns(19, 6, 64)
    tI6, tP6 = Pv.table_columns(Pv.e1_cells(19, 6), 6, keep=keep)
    pr = [(0, lowest), (1, highest)]
    out["19_cells_N6_first_last"] = {
        "jet_seconds_68ch": sec6,
        "jet_vs_stack64": {"I": Pv.compare(jI6[..., S.E1_CH], sI6[keep], probes=pr),
                           "P": Pv.compare(jP6[..., S.E1_CH], sP6[keep], probes=pr)},
        "table17_vs_jet": {"I": Pv.compare(tI6[..., S.E1_CH], jI6[..., S.E1_CH], probes=pr),
                           "P": Pv.compare(tP6[..., S.E1_CH], jP6[..., S.E1_CH], probes=pr)},
    }
    # 1 cell (half-width 2.30), 5 cells (0.92), 200 cells (0.023) at first, middle and last cells
    for n, keep in ((1, [0]), (5, [0, 4]), (200, [0, 100, 199])):
        cells = Pv.e1_cells(n, 4)
        jI, jP, sec = Pv.jet_columns(cells, 4, keep=keep)
        jI, jP = jI[..., S.E1_CH], jP[..., S.E1_CH]
        sI, sP = Pv.stack_columns(n, 4, 64)
        sI, sP = sI[keep], sP[keep]
        tI, tP = Pv.table_columns(cells, 4, keep=keep)
        tI, tP = tI[..., S.E1_CH], tP[..., S.E1_CH]
        pr = [(0, lowest), (len(keep) - 1, highest)]
        out[f"{n}_cells"] = {
            "half_width": float(cells.half_width[0]), "kept": list(map(int, keep)), "jet_seconds_68ch": sec,
            "jet_vs_stack64": {"I": Pv.compare(jI, sI, probes=pr), "P": Pv.compare(jP, sP, probes=pr)},
            "table17_vs_jet": {"I": Pv.compare(tI, jI, probes=pr), "P": Pv.compare(tP, jP, probes=pr)},
        }
    return out


def main(stages=("providers", "e1", "default")) -> None:
    t0 = time.perf_counter()
    if "providers" in stages:
        record("providers", provider_cases())
        print("providers", time.perf_counter() - t0, flush=True)
    if "e1" in stages:
        stage_e1(t0)
    if "default" in stages:
        stage_default(t0)


def stage_e1(t0) -> None:
    part = {}
    for lat in E1_LATTICES:
        for pop in POPULATIONS:
            for N in (0, 2, 4):
                part[f"{lat}|{pop}|N{N}"] = partition_e1(lat, pop, N)
    for pop in POPULATIONS:
        for N in (0, 2, 4):
            part[f"one_cell|{pop}|N{N}"] = partition_one_cell(pop, N)
    part["tail_check_PL_unbounded"] = L.tail_check(POPULATIONS["PL_unbounded"], np.r_[0:68])
    record("partition_e1", part)
    print("partition_e1", time.perf_counter() - t0, flush=True)


def stage_default(t0) -> None:
    cells = default_lattice()
    lo, hi = W.default_range(S.E2_CH)
    lo_e1, hi_e1 = W.default_range(S.E1_CH)
    lo_all, hi_all = W.default_range(np.r_[0:68])
    d = {"range_1_100GHz": [lo, hi], "n_cells": cells.n, "half_width": float(cells.half_width[0]),
         "range_0p1_10GHz": [lo_e1, hi_e1], "range_0p1_300GHz": [lo_all, hi_all]}
    for pop in POPULATIONS:
        for N in (0, 2, 4):
            d[f"{pop}|N{N}"] = partition_default(pop, N)
    record("default_1_100GHz_lattice", d)
    print("default", time.perf_counter() - t0, flush=True)


if __name__ == "__main__":
    import sys

    main(tuple(sys.argv[1:]) or ("providers", "e1", "default"))
