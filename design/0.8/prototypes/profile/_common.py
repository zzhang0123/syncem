"""Shared helpers for the profiling scripts: paths, timers with CPU-time ratio, JSON output.

Every timer returns wall seconds and the ratio of process CPU time to wall time. A ratio near
1 means one core did the work. A ratio near the core count means the work ran on all cores.
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]  # syncem repository root
SYNCEM = ROOT
STUDY_LIB = SYNCEM / "studies" / "classical-models" / "lib"
PROTO = HERE.parent
RESULTS = HERE / "results"

for p in (str(PROTO), str(SYNCEM), str(STUDY_LIB)):
    if p not in sys.path:
        sys.path.insert(0, p)


def timed(fn, *args, repeat=1, **kw):
    """(result of the last call, best wall seconds, CPU/wall ratio of that call)."""
    best, ratio, out = float("inf"), 0.0, None
    for _ in range(repeat):
        c0, t0 = time.process_time(), time.perf_counter()
        out = fn(*args, **kw)
        out = _block(out)
        t1, c1 = time.perf_counter(), time.process_time()
        if t1 - t0 < best:
            best, ratio = t1 - t0, (c1 - c0) / max(t1 - t0, 1e-12)
    return out, best, ratio


def _block(x):
    try:
        import jax
        return jax.block_until_ready(x)
    except Exception:
        return x


def save(name: str, data: dict) -> Path:
    RESULTS.mkdir(exist_ok=True)
    path = RESULTS / f"{name}.json"
    meta = {"python": sys.version.split()[0], "cpu_count": os.cpu_count(),
            "xla_flags": os.environ.get("XLA_FLAGS", ""), "loadavg_at_save": os.getloadavg()}
    path.write_text(json.dumps({"meta": meta, **data}, indent=1, default=float))
    return path


def rel_err(a, b, floor=1e-300):
    import numpy as np
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.maximum(np.maximum(np.abs(a), np.abs(b)), floor)))
