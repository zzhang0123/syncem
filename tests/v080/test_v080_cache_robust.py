"""Edge cases of the disk entries and the byte bound (decisions 161 and 162, step-5b-2 review).

* The synced-folder check compares paths case-insensitively on macOS and Windows and case-sensitively on
  other platforms. ``~\\`` is home-relative on Windows.
* A ``.npz`` replaced by a plain ``.npy`` is a corrupt entry: deleted and a miss.
* ``evict`` reports only entries whose files are gone and moves on to the next entry after a failed removal.
* ``Cache.put`` writes NumPy scalars of a description as Python numbers and refuses a non-JSON description
  before it keeps anything.
* An S2 entry and a memo entry are charged their container bytes as well as their arrays. The bound is that
  ``Cache.nbytes`` is at least half the bytes ``tracemalloc`` traces for the stored entries. On one E1 build
  (41 channels, 19 cells) the review probe of 2026-11-19 found 47 times the charged bytes of the S2 entries
  in ``sys.getsizeof`` over their objects (1-minute load 18.7). After the change the store charged 8.66 MB
  against 8.06 MB traced (1-minute load 12.2), a ratio of 1.07, so the factor 0.5 has a margin of 2.1.

The cases create files only under ``tmp_path``. Every assertion but the byte bound is an exact identity.
"""

from __future__ import annotations

import os
import sys
import tracemalloc
import warnings
from pathlib import Path

import numpy as np
import pytest

from syncem.engine import _disk
from syncem.engine.cache import Cache

# Directory ---------------------------------------------------------------------------------------


def _synced(env, platform, monkeypatch):
    monkeypatch.setattr(sys, "platform", platform)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        path = _disk.cache_dir(env)
    count = sum(issubclass(w.category, _disk.SyncedFolderWarning) for w in seen)
    return path, count


@pytest.mark.parametrize("spelling", ["Documents", "documents", "DOCUMENTS"])
@pytest.mark.parametrize("platform", ["darwin", "win32", "linux"])
def test_the_synced_folder_check_follows_the_case_rule_of_the_platform(
    tmp_path, monkeypatch, spelling, platform
):
    home = tmp_path / "home"
    env = {"HOME": str(home), "USERPROFILE": str(home)}
    env["SYNCEM_CACHE"] = str(home / spelling / "cache")
    _, count = _synced(env, platform, monkeypatch)
    folds = platform in ("darwin", "win32")
    assert count == (1 if spelling == "Documents" or folds else 0)


def test_a_mobile_documents_spelling_in_lower_case_warns_on_macos(
    tmp_path, monkeypatch
):
    home = tmp_path / "home"
    env = {
        "HOME": str(home),
        "SYNCEM_CACHE": str(home / "library" / "mobile documents" / "c"),
    }
    assert _synced(env, "darwin", monkeypatch)[1] == 1


@pytest.mark.parametrize(
    ("platform", "value", "under_home"),
    [
        ("win32", "~\\syncem", True),
        ("win32", "~/syncem", True),
        ("linux", "~/syncem", True),
    ],
)
def test_a_home_relative_cache_value(
    tmp_path, monkeypatch, platform, value, under_home
):
    home = tmp_path / "home"
    env = {"HOME": str(home), "USERPROFILE": str(home), "SYNCEM_CACHE": value}
    path, _ = _synced(env, platform, monkeypatch)
    assert path.is_relative_to(home) == under_home and "~" not in path.parts


def test_a_tilde_backslash_on_posix_is_a_relative_name(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    env = {"HOME": str(tmp_path / "home"), "SYNCEM_CACHE": "~\\syncem"}
    path, _ = _synced(env, "linux", monkeypatch)
    assert path == tmp_path / "~\\syncem"


# Entries -----------------------------------------------------------------------------------------


def _entry(root, char):
    path = _disk.entry_path(root, "s1", char * 64)
    _disk.write_entry(path, {"x": np.arange(1000.0)}, {"n": 1})
    return path


def test_a_npz_replaced_by_a_plain_npy_is_a_corrupt_miss(tmp_path):
    path = _entry(tmp_path, "a")
    assert _disk.read_entry(path) is not None
    with open(path, "wb") as handle:
        np.save(handle, np.arange(3.0))
    assert _disk.read_entry(path) is None
    assert not path.exists() and not path.with_suffix(".json").exists()


@pytest.mark.skipif(sys.platform == "win32", reason="POSIX permission bits")
def test_evict_reports_only_removed_entries_and_moves_on(tmp_path, monkeypatch):
    kept, gone = _entry(tmp_path, "a"), _entry(tmp_path, "b")
    os.utime(kept, (1.0, 1.0))
    os.utime(kept.with_suffix(".json"), (1.0, 1.0))
    real = _disk._unlink

    def refuse(path):
        if Path(path).stem == "a" * 64:
            raise PermissionError(path)
        real(path)

    monkeypatch.setattr(_disk, "_unlink", refuse)
    assert _disk.evict(tmp_path, 0) == ("b" * 64,)
    assert kept.exists() and not gone.exists()


def test_put_writes_numpy_scalars_and_refuses_a_non_json_description(tmp_path):
    cache = Cache(directory=tmp_path)
    cache.put("s1", "c" * 64, {"x": np.arange(3.0)}, description={"n": np.int64(1)})
    found = _disk.read_entry(_disk.entry_path(tmp_path, "s1", "c" * 64))
    assert found["description"] == {"n": 1}
    for bad in ({"x": float("nan")}, {"x": object()}):
        with pytest.raises((TypeError, ValueError), match="description"):
            cache.put("s1", "d" * 64, {"x": np.arange(3.0)}, description=bad)
        assert cache.peek("s1", "d" * 64) is None


# The byte bound ----------------------------------------------------------------------------------


def test_s2_entries_are_charged_their_container_bytes():
    cache = Cache(directory=None, max_bytes=0)
    ids = [((), (i,)) for i in range(3)]
    keys = [f"{k:064x}" for k in range(400)]
    tracemalloc.start()
    try:
        start = tracemalloc.get_traced_memory()[0]
        cache.put_columns(
            keys, ids, np.ones((len(keys), 3)), [("d0", "d1", "d2", "d3")] * len(keys)
        )
        traced = tracemalloc.get_traced_memory()[0] - start
    finally:
        tracemalloc.stop()
    assert cache.nbytes >= 0.5 * traced
