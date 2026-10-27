"""Provenance block for results JSON files.

Paths are recorded relative to the study directory (the parent of lib/), never as absolute
paths. Versions are recorded for Python, numpy and scipy, and for matplotlib, jax and syncem
when the calling script has imported them.
"""
from __future__ import annotations

import hashlib
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import scipy

LIB = Path(__file__).resolve().parent
STUDY = LIB.parent
LIB_FILES = ("kernels.py", "config.py", "basis.py", "ensembles.py", "provenance.py")
OPTIONAL_VERSIONS = ("matplotlib", "jax", "syncem")


def sha256(path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def relative(path) -> str:
    """Path of a file inside the study directory, relative to it, in POSIX form."""
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(STUDY).as_posix()
    except ValueError:
        raise ValueError(f"{path} is not inside the study directory") from None


def versions() -> dict:
    out = {"python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__}
    for name in OPTIONAL_VERSIONS:
        module = sys.modules.get(name)
        if module is not None:
            out[name] = getattr(module, "__version__", "unknown")
    return out


def provenance(script, t_start: float, inputs=()) -> dict:
    lib = {name: sha256(LIB / name) for name in LIB_FILES if (LIB / name).exists()}
    return {
        "script": relative(script),
        "script_sha256": sha256(script),
        "lib_sha256": lib,
        "inputs_sha256": {relative(p): sha256(p) for p in inputs},
        "versions": versions(),
        "wall_clock_s": time.time() - t_start,
        "finished": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }


def _default(o):
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, complex):
        return [o.real, o.imag]
    raise TypeError(type(o))


def write_json(path, payload: dict) -> None:
    Path(path).write_text(json.dumps(payload, indent=1, default=_default) + "\n")
