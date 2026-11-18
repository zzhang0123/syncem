"""Disk store of the expensive eta-resolved jet builds of the beaming tests.

Each entry is the E1 kernel (bump channels with 64 nodes, Gaussian screen (8, 0.5), B ~ U[4, 6] uG
by 16 Gauss-Legendre nodes) differentiated by the core jet on E1 energy cells, with eta kept as
a node axis. Any eta law on those nodes is then a contraction (beaming.contract).
Run ``python -m rules.beaming.store`` from the prototypes directory to build every entry.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

import core as C
from core import paper as E

from . import beaming as BM

HERE = Path(__file__).resolve().parent
CACHE = HERE / "cache"

# name -> (eta nodes, cells selected from the 19 E1 cells, order N)
EXTREME_CELLS = (0, 9, 18)


def _entries() -> dict:
    half32, _ = BM.half_range(-0.5, 0.5, 32)
    near0, _ = BM.uniform_law(-0.02, 0.02, 16)
    iso, _ = BM.uniform_law(-1.0, 1.0, 32, panels=2)
    near1, _ = BM.union_law([(-1.0, -0.95), (0.95, 1.0)], 16)
    asym, _ = BM.uniform_law(0.0, 0.5, 32)
    return {
        "half32_all_N4": (half32, tuple(range(19)), 4),
        "near0_N4": (near0, EXTREME_CELLS, 4),
        "iso_N4": (iso, EXTREME_CELLS, 4),
        "near1_N4": (near1, EXTREME_CELLS, 4),
        "asym_N4": (asym, EXTREME_CELLS, 4),
    }


def base_kernel():
    return C.ContinuumChannelKernel(C.bump_channels(), screen=E.SCREEN)


def table_for(cells, N):
    return E.e1_table(N).select(np.isin(np.arange(19), cells))


def load(name: str, build: bool = True) -> dict:
    """{"I", "P", "eta", "cells", "N", "seconds"} for a named entry, built on first use."""
    path = CACHE / f"{name}.npz"
    if path.exists():
        d = np.load(path)
        return {k: d[k] for k in d.files}
    if not build:
        raise FileNotFoundError(path)
    eta, cells, N = _entries()[name]
    t = time.time()
    cols = BM.eta_resolved_columns(base_kernel(), table_for(cells, N), eta, [("B", C.uniform(4.0, 6.0, 16))])
    out = {"I": cols["I"], "P": cols["P"], "eta": eta, "cells": np.array(cells), "N": np.array(N),
           "seconds": np.array(time.time() - t)}
    CACHE.mkdir(exist_ok=True)
    np.savez(path, **out)
    return out


if __name__ == "__main__":
    for name in (sys.argv[1:] or _entries()):
        d = load(name)
        print(name, d["I"].shape, float(d["seconds"]), flush=True)
