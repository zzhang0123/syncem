"""Disk cache of the expensive node arrays (energy columns and kernel values at gamma nodes).

Files live in rules/depth/cache/ (git-ignored with the prototypes). The key names the kernel
kind, the channel set and the lattice, so a change of any of them writes a new file.
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np

import core as C

from . import nodes as Nd

DIR = Path(__file__).with_name("cache")


def _key(*parts) -> str:
    h = hashlib.sha1()
    for p in parts:
        h.update(np.ascontiguousarray(np.asarray(p, dtype=float)).tobytes() if isinstance(p, np.ndarray)
                 else repr(p).encode())
    return h.hexdigest()[:16]


def energy_columns(kind: str, ch: C.Channels, gcells: C.Cells, N: int) -> dict:
    key = _key("ecols", kind, ch.nu, ch.w, gcells.edges, N)
    f = DIR / f"ecols_{kind}_{key}.npz"
    if f.exists():
        d = np.load(f)
        return {"I": d["I"], "P": d["P"]}
    DIR.mkdir(exist_ok=True)
    gt = C.ProductTable.tensor({"gamma": gcells.with_order(N)}, total=N)
    out = Nd.energy_columns(Nd.node_kernel(ch, kind), gt, ch.n_ch)
    np.savez(f, **out)
    return out


def values(kind: str, ch: C.Channels, gammas: np.ndarray) -> tuple:
    key = _key("values", kind, ch.nu, ch.w, gammas)
    f = DIR / f"values_{kind}_{key}.npz"
    if f.exists():
        d = np.load(f)
        return d["I"], d["P"]
    DIR.mkdir(exist_ok=True)
    I, P = Nd.values_at(Nd.node_kernel(ch, kind), {"gamma": gammas}, ch.n_ch)
    np.savez(f, I=I, P=P)
    return I, P
