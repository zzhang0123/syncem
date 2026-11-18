"""Guards that keep the existing code fixed while the 0.8 layer is built (design/0.8/SPEC.md Sec. 9.7).

1. The two AST digests of ``studies/classical-models/lib/syncem_compat.py`` still hold. They pin the
   code that the frozen harmonic pipeline calls (``syncem.constants`` and two functions of
   ``syncem.model.high_order_harmonic``).
2. The five names that the study scripts import from ``syncem.model`` still exist.
3. Every file under ``syncem/`` at the v0.7.10 tag, except ``syncem/__init__.py`` (which holds the
   version and gains the front door), has its v0.7.10 SHA-256. The hashes are stored in
   ``guard_v0_7_10_hashes.json``, written once from ``git show v0.7.10:<path>``.

A deliberate change to existing code updates the stored hash in the same commit, with the reason in
the commit message. The manuscript cites v0.7.10, so such a change needs the owner's approval.
"""

import hashlib
import importlib
import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
STUDY_LIB = ROOT / "studies" / "classical-models" / "lib"
HASHES = Path(__file__).resolve().parent / "guard_v0_7_10_hashes.json"
STUDY_NAMES = (
    "Channels",
    "Reference",
    "Support",
    "Truncation",
    "build_high_order_basis",
)
EXCLUDED = ("syncem/__init__.py",)


def _compat():
    if str(STUDY_LIB) not in sys.path:
        sys.path.insert(0, str(STUDY_LIB))
    return importlib.import_module("syncem_compat")


def _stored() -> dict:
    return json.loads(HASHES.read_text())


def test_compat_digests_hold():
    compat = _compat()
    assert len(compat.D51F9BE_CODE) == 2
    for relpath, (roots, expected) in compat.D51F9BE_CODE.items():
        assert (
            compat.code_digest(compat.source_before_rename(relpath), roots) == expected
        ), relpath


def test_compat_digest_detects_a_code_change():
    compat = _compat()
    relpath, (roots, expected) = next(iter(compat.D51F9BE_CODE.items()))
    source = compat.source_before_rename(relpath) + "\n_GUARD_PROBE = 1\n"
    changed = compat.code_digest(source, roots)
    # A whole-module digest sees any new statement. A rooted digest sees only reachable names.
    assert (changed != expected) if roots is None else (changed == expected)


@pytest.mark.parametrize("name", STUDY_NAMES)
def test_study_names_import_from_syncem_model(name):
    model = importlib.import_module("syncem.model")
    assert hasattr(model, name), name


def test_hash_record_covers_the_tagged_files():
    stored = _stored()
    assert stored["tag"] == "v0.7.10"
    assert stored["excluded"] == list(EXCLUDED)
    files = stored["sha256"]
    assert len(files) == 99
    assert all(path.startswith("syncem/") and path not in EXCLUDED for path in files)


NEW_LAYER_DIRS = (
    "values",
    "base",
    "distributions",
    "lattice",
    "records",
    "engine",
    "population",
    "planning",
    "results",
    "fitting",
)
NEW_TOP_LEVEL = ("syncem/units.py", "syncem/paper.py", "syncem/front.py")


def _added_files(root: Path) -> list[str]:
    """Files under ``root/syncem`` that are neither in the v0.7.10 record nor in a new 0.8 place."""
    files = set(_stored()["sha256"]) | set(EXCLUDED)
    added = []
    for path in (root / "syncem").rglob("*"):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        rel = path.relative_to(root).as_posix()
        parts = rel.split("/")
        if (
            rel in files
            or rel in NEW_TOP_LEVEL
            or (len(parts) > 2 and parts[1] in NEW_LAYER_DIRS)
        ):
            continue
        added.append(rel)
    return sorted(added)


def test_no_file_added_outside_the_new_layers():
    assert _added_files(ROOT) == []


@pytest.mark.parametrize(
    "probe",
    [
        "syncem/model/newpkg/x.py",
        "syncem/model/fit/_new.py",
        "syncem/extra.py",
        "syncem/model/data.json",
    ],
)
def test_added_file_guard_detects(tmp_path, probe):
    for rel in _stored()["sha256"]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    allowed = ["syncem/__init__.py", "syncem/units.py", "syncem/lattice/cells.py"]
    for rel in [*allowed, probe]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).touch()
    assert _added_files(tmp_path) == [probe]


@pytest.mark.parametrize(
    "relpath",
    sorted(json.loads(HASHES.read_text())["sha256"]) if HASHES.exists() else [],
)
def test_existing_file_unchanged_since_v0_7_10(relpath):
    path = ROOT / relpath
    assert path.is_file(), f"{relpath} was removed"
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest == _stored()["sha256"][relpath], f"{relpath} differs from v0.7.10"
