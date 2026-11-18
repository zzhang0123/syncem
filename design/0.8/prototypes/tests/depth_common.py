"""Shared helpers of the depth rule tests: result recording in tests/depth_results.json."""
from __future__ import annotations

import json
from pathlib import Path

RESULTS = Path(__file__).with_name("depth_results.json")


def record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


def verdict(rel_err: float) -> str:
    """Boundary-validation verdict of the user's global rule."""
    return "OK" if rel_err < 1e-3 else ("WARN" if rel_err < 1e-1 else "BAD")
