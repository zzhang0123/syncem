"""Column providers compared at every cell (Spec Sec. 8.3, test T2).

* jet: ``core.build_columns`` on the 68-channel averaged kernel (nested forward mode).
* stack: the study's analytic derivative stack ``basis.derivative_spectra`` (cached field-average
  spline table, NumPy), at the core's 64 channel nodes and at the study's 400.
* table: 17 Chebyshev-Lobatto values of the averaged kernel per cell, columns from the
  derivatives of the interpolant at the reference point.

All three return arrays (cells, r, channels) in the local coordinate u = ln(gamma/gamma_b)/h_b,
that is (1/r!) d^r/du^r Kbar. P carries the screen and no azimuth factor.
"""
from __future__ import annotations

import hashlib
import os
import sys
import time
from dataclasses import replace

import numpy as np

import core as C
from core import paper as E

from . import nodes as Nd
from . import setup as S


def e1_cells(n: int, N: int) -> C.Cells:
    return C.Cells.log(E.GAMMA_LO, E.GAMMA_HI, n, order=N)


def jet_columns(cells: C.Cells, N: int, keep=None) -> tuple[np.ndarray, np.ndarray, float]:
    """Jet columns (cells, N+1, 68) for the kept cells, and the build time (cached on disk)."""
    tab = C.ProductTable.tensor({"gamma": cells.with_order(N)}, total=N)
    keep = np.arange(cells.n) if keep is None else np.asarray(keep)
    tab = tab.select(np.isin(np.arange(cells.n), keep))
    key = hashlib.sha256(np.r_[tab.reference[:, 0], tab.half_width[:, 0], N].tobytes()).hexdigest()[:20]
    path = S.CACHE / f"jet_{key}.npz"
    if path.exists():
        d = np.load(path)
        return d["I"], d["P"], float(d["seconds"])
    t0 = time.perf_counter()
    cols = C.build_columns(S.kernel(), tab)
    sec = time.perf_counter() - t0
    I = cols["I"].reshape(tab.n_cells, N + 1, -1)
    P = cols["P"].reshape(tab.n_cells, N + 1, -1)
    tmp = path.with_suffix(f".{os.getpid()}.tmp.npz")
    np.savez(tmp, I=I, P=P, seconds=sec)
    os.replace(tmp, path)
    return I, P, sec


def stack_columns(n: int, N: int, n_chan_nodes=S.N_NODES) -> tuple[np.ndarray, np.ndarray]:
    """Study derivative stack (cells, N+1, 41) for n equal cells on the E1 range."""
    if str(S.STUDY_LIB) not in sys.path:
        sys.path.insert(0, str(S.STUDY_LIB))
    import basis as SB  # noqa: E402  (study oracle)
    from config import PAPER  # noqa: E402

    cfg = replace(PAPER, n_chan_nodes=n_chan_nodes, n_bins=n)
    return SB.derivative_spectra(cfg, N)


def table_columns(cells: C.Cells, N: int, keep=None, n_nodes=Nd.N_TABLE):
    keep = np.arange(cells.n) if keep is None else np.asarray(keep)
    I, P = Nd.table_values(cells.reference[keep], cells.half_width[keep], n_nodes)
    return Nd.table_columns(I, N), Nd.table_columns(P, N)


def compare(A: np.ndarray, B: np.ndarray, *, probes=()) -> dict:
    """A against the reference B, both (cells, r, ch). Per r:
    chan   max |A - B| / max over cells of |B| at that r and channel (channel column maximum),
    chan0  the same with the order-0 channel maximum as the scale,
    chan_all  the same with the maximum over all cells and orders of that channel,
    own    max over cells of max_j |A - B| / max_j |B| of that cell and r (own peak),
    own0   the same with the cell's order-0 own peak,
    probes pointwise |A - B| / |B| at the given (cell, channel) pairs.
    """
    D = np.abs(A - B)
    Bab = np.abs(B)
    out = {}
    chan_all = Bab.max(axis=(0, 1))
    for r in range(A.shape[1]):
        chan = Bab[:, r].max(axis=0)
        chan0 = Bab[:, 0].max(axis=0)
        own = Bab[:, r].max(axis=1)
        own0 = Bab[:, 0].max(axis=1)
        out[r] = {
            "chan": float(np.max(D[:, r] / chan)),
            "chan0": float(np.max(D[:, r] / chan0)),
            "chan_all": float(np.max(D[:, r] / chan_all)),
            "own": float(np.max(D[:, r].max(axis=1) / own)),
            "own_worst_cell": int(np.argmax(D[:, r].max(axis=1) / own)),
            "own0": float(np.max(D[:, r].max(axis=1) / own0)),
            "probes": [float(D[b, r, j] / Bab[b, r, j]) if Bab[b, r, j] > 0
                       else (0.0 if D[b, r, j] == 0 else float("inf")) for b, j in probes],
        }
    return out
