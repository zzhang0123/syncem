"""Run the harmonic-pipeline modules in inputs/harmonic-radio-fit against syncem.

radio_numerics.py and stable_lines.py are copies of the harmonic pipeline that produced the
frozen harmonic matrix (inputs/README.md). They import the package by its earlier name,
`syncmoments`, and call syncem.constants and syncem.model.high_order_harmonic. The harmonic
pipeline pinned SyncMoments d51f9be. Since that commit these two modules have changed only by
the rename to syncem, which use_syncem() checks against their d51f9be SHA-256.

use_syncem() imports syncem from the package root of this repository (two levels above the
study directory), checks that it resolved there, registers syncem and its loaded submodules
under the earlier name, and puts inputs/harmonic-radio-fit on sys.path. Call it before
importing radio_numerics or stable_lines, in every process that imports them.
"""
from __future__ import annotations

import hashlib
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = STUDY.parents[1]
HARMONIC_INPUTS = STUDY / "inputs" / "harmonic-radio-fit"
EARLIER_NAME = "syncmoments"  # package name before SyncEM 0.6.0, used by the frozen modules
D51F9BE = "d51f9bef4fa744a536a60c3da821b1033818a232"
# SHA-256 of the modules the frozen pipeline calls, at SyncMoments d51f9be.
D51F9BE_SOURCES = {
    "constants.py": "551a658aca9d85d526dd5d17ae15f47662840289900ca3e581fc6b07309693a0",
    "model/high_order_harmonic.py": "e6eb18c4a8e1960f7bda653824343930edb8417e76b97cd95e1e26418b87a8f1",
}


def source_sha256_before_rename(relpath: str) -> str:
    """SHA-256 of syncem/<relpath> with the package name mapped back to the earlier name."""
    source = (PACKAGE_ROOT / "syncem" / relpath).read_bytes()
    source = source.replace(b"syncem", EARLIER_NAME.encode()).replace(b"SyncEM", b"SyncMoments")
    return hashlib.sha256(source).hexdigest()


def use_syncem():
    """Import syncem from PACKAGE_ROOT and register it under the earlier package name."""
    root = str(PACKAGE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import syncem  # noqa: PLC0415
    import syncem.constants  # noqa: F401, PLC0415
    import syncem.model.high_order_harmonic  # noqa: F401, PLC0415

    if Path(syncem.__file__).resolve().parents[1] != PACKAGE_ROOT:
        raise RuntimeError("syncem did not resolve to the package root of this repository")
    for relpath, expected in D51F9BE_SOURCES.items():
        if source_sha256_before_rename(relpath) != expected:
            raise RuntimeError(f"syncem/{relpath} differs from SyncMoments d51f9be beyond the rename")
    for name, module in list(sys.modules.items()):
        if name == "syncem" or name.startswith("syncem."):
            sys.modules[EARLIER_NAME + name[len("syncem"):]] = module
    path = str(HARMONIC_INPUTS)
    if path not in sys.path:
        sys.path.insert(0, path)
    return syncem


def check_record() -> dict:
    """What use_syncem() checked, for a results JSON."""
    import syncem  # noqa: PLC0415

    return {
        "package": "syncem",
        "version": syncem.__version__,
        "resolved_to_package_root_of_this_repository": True,
        "modules_equal_to_SyncMoments_d51f9be_up_to_the_rename": sorted(D51F9BE_SOURCES),
        "SyncMoments_pin": D51F9BE,
    }
