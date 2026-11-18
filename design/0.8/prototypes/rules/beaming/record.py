"""Write measured numbers of the beaming tests to tests/beaming_results.json."""
from __future__ import annotations

import fcntl
import json
from pathlib import Path

RESULTS = Path(__file__).resolve().parents[2] / "tests" / "beaming_results.json"


def record(key: str, value) -> None:
    lock = RESULTS.with_suffix(".lock")
    with open(lock, "w") as fh:
        fcntl.flock(fh, fcntl.LOCK_EX)
        data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
        data[key] = value
        RESULTS.write_text(json.dumps(data, indent=1, default=float, sort_keys=True))
