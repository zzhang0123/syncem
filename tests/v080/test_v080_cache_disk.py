"""Step 5b-2 of design/0.8/SPEC.md: the disk layer ``syncem.engine._disk`` (decision 161).

These are the cases that need only ``_disk``: writing and reading an entry, corrupt and missing
files, eviction and two writers of one key. The directory, the synced-folder warning and the size
syntax are in ``test_v080_cache_disk_settings.py``. The cases through ``Cache`` and through a build
live in other files. Every comparison is an exact identity of bytes, paths or
integers, so no tolerance enters. Modification times are set with ``os.utime`` to fixed values, so no
test sleeps or depends on the clock resolution.

No test writes under the real home directory. Every entry goes under ``tmp_path``, and the autouse
fixture of ``tests/conftest.py`` points ``SYNCEM_CACHE`` at a per-test directory.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import time
from pathlib import Path

import numpy as np
import pytest

from syncem.engine import _disk

KEY_A = "a" * 64
KEY_B = "b" * 64
KEY_C = "c" * 64
OLD = 1_000_000_000  # a fixed modification time (2001), older than any write in a test
HOUR = 3600


def _arrays(seed: int = 0, n: int = 16) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {
        "table": rng.standard_normal((n, 3)),
        "grid": np.arange(n, dtype=np.int64),
        "flags": np.array([True, False]),
    }


def _write(root: Path, key: str, arrays=None, stage: str = "s0") -> Path:
    path = _disk.entry_path(root, stage, key)
    _disk.write_entry(
        path, arrays if arrays is not None else _arrays(), {"stage": stage}
    )
    return path


def _same(a: dict[str, np.ndarray], b: dict[str, np.ndarray]) -> bool:
    return sorted(a) == sorted(b) and all(
        a[k].dtype == b[k].dtype
        and a[k].shape == b[k].shape
        and a[k].tobytes() == b[k].tobytes()
        for k in a
    )


def _age(path: Path, mtime: float) -> None:
    os.utime(path, (mtime, mtime))


def _age_entry(path: Path, mtime: float) -> None:
    _age(path, mtime)
    _age(path.with_suffix(".json"), mtime)


# Writing and reading -----------------------------------------------------------------------------


def test_entry_path_layout(tmp_path):
    assert _disk.entry_path(tmp_path, "s1", KEY_A) == (
        tmp_path / f"v{_disk.FORMAT}" / "s1" / f"{KEY_A}.npz"
    )


@pytest.mark.parametrize("key", ["A" * 64, "a" * 63, "a" * 65, "g" * 64, KEY_A + " 2"])
def test_entry_path_refuses_other_keys(tmp_path, key):
    with pytest.raises(ValueError, match="64"):
        _disk.entry_path(tmp_path, "s0", key)


def test_entry_path_refuses_other_stages(tmp_path):
    with pytest.raises(ValueError, match="s0"):
        _disk.entry_path(tmp_path, "s2", KEY_A)


def test_write_then_read_is_bitwise(tmp_path):
    arrays = _arrays(1)
    path = _write(tmp_path, KEY_A, arrays)
    entry = _disk.read_entry(path)
    assert entry is not None
    assert _same(entry["arrays"], arrays)
    assert entry["description"] == {"stage": "s0"}
    assert sorted(p.name for p in path.parent.iterdir()) == [
        f"{KEY_A}.json",
        f"{KEY_A}.npz",
    ]


def test_equal_content_writes_equal_bytes(tmp_path):
    """Decision 161 has no locks because equal keys write equal bytes. The zip carries no clock time."""
    path = _write(tmp_path, KEY_A)
    first = (path.read_bytes(), path.with_suffix(".json").read_bytes())
    _age_entry(path, OLD)
    _write(tmp_path, KEY_A)
    assert (path.read_bytes(), path.with_suffix(".json").read_bytes()) == first


def test_write_refuses_object_arrays_and_bad_descriptions(tmp_path):
    path = _disk.entry_path(tmp_path, "s0", KEY_A)
    with pytest.raises(ValueError, match="object"):
        _disk.write_entry(path, {"x": np.array([object()])}, {})
    with pytest.raises(ValueError):
        _disk.write_entry(path, _arrays(), {"bad": float("nan")})
    with pytest.raises(ValueError, match="64"):
        _disk.write_entry(tmp_path / "name.npz", _arrays(), {})
    assert not path.exists()
    assert not path.parent.exists() or not any(path.parent.iterdir())


def test_read_of_an_absent_entry_is_a_miss(tmp_path):
    assert _disk.read_entry(_disk.entry_path(tmp_path, "s0", KEY_A)) is None


def test_hit_refreshes_modification_time(tmp_path):
    path = _write(tmp_path, KEY_A)
    _age_entry(path, OLD)
    assert _disk.read_entry(path) is not None
    assert path.stat().st_mtime > OLD
    assert path.with_suffix(".json").stat().st_mtime > OLD


def _edit_meta(path: Path, edit) -> None:
    json_path = path.with_suffix(".json")
    meta = json.loads(json_path.read_text())
    edit(meta)
    json_path.write_text(json.dumps(meta))


def _pickled(path: Path) -> None:
    with open(path, "wb") as f:
        np.savez(f, table=np.array([{"x": 1}], dtype=object))
    _edit_meta(
        path, lambda m: m.update(arrays={"table": {"dtype": "|O", "shape": [1]}})
    )


def _truncate(path: Path) -> None:
    text = path.with_suffix(".json").read_text()
    path.with_suffix(".json").write_text(text[: len(text) // 2])


def _meta_edit(edit):
    return lambda path: _edit_meta(path, edit)


CORRUPTIONS = {
    "npz_not_a_zip": lambda path: path.write_bytes(b"not a zip file"),
    "json_truncated": _truncate,
    "json_a_list": lambda path: path.with_suffix(".json").write_text("[1, 2]"),
    "pickled_object_array": _pickled,
    "wrong_shape": _meta_edit(lambda m: m["arrays"]["table"].update(shape=[1, 2])),
    "wrong_dtype": _meta_edit(lambda m: m["arrays"]["grid"].update(dtype="<f8")),
    "array_missing": _meta_edit(
        lambda m: m["arrays"].update(extra={"dtype": "<f8", "shape": [1]})
    ),
    "array_extra": _meta_edit(lambda m: m["arrays"].pop("flags")),
    "wrong_key": _meta_edit(lambda m: m.update(key=KEY_B)),
    "wrong_format": _meta_edit(lambda m: m.update(format=_disk.FORMAT + 1)),
}


@pytest.mark.parametrize("corrupt", CORRUPTIONS.values(), ids=CORRUPTIONS.keys())
def test_corrupt_entry_is_deleted_counted_as_a_miss_and_rebuilt_equal(
    tmp_path, corrupt
):
    arrays = _arrays(2)
    path = _write(tmp_path, KEY_A, arrays)
    corrupt(path)
    assert _disk.read_entry(path) is None
    assert not path.exists()
    assert not path.with_suffix(".json").exists()
    _write(tmp_path, KEY_A, arrays)
    entry = _disk.read_entry(path)
    assert entry is not None and _same(entry["arrays"], arrays)


def test_npz_deleted_between_check_and_load_is_a_miss(tmp_path, monkeypatch):
    """The JSON is read, then the ``.npz`` vanishes before ``np.load`` opens it."""
    path = _write(tmp_path, KEY_A)
    real_load = np.load

    def load_after_delete(file, *args, **kwargs):
        os.unlink(path)
        return real_load(file, *args, **kwargs)

    monkeypatch.setattr(_disk.np, "load", load_after_delete)
    assert _disk.read_entry(path) is None
    assert path.with_suffix(".json").exists()


def test_json_absent_is_a_miss_and_leaves_the_npz(tmp_path):
    path = _write(tmp_path, KEY_A)
    path.with_suffix(".json").unlink()
    assert _disk.read_entry(path) is None
    assert path.exists()


def test_unreadable_json_is_a_miss_and_deletes_nothing(tmp_path):
    path = _write(tmp_path, KEY_A)
    path.with_suffix(".json").unlink()
    path.with_suffix(".json").mkdir()
    assert _disk.read_entry(path) is None
    assert path.exists() and path.with_suffix(".json").is_dir()


# Eviction ----------------------------------------------------------------------------------------

MB4 = 500_000  # float64 values per 4 MB table


def _big(seed: int) -> dict[str, np.ndarray]:
    return {"table": np.random.default_rng(seed).standard_normal(MB4)}


def _keys_on_disk(root: Path) -> set[str]:
    stage = root / f"v{_disk.FORMAT}" / "s1"
    return {p.stem for p in stage.glob("*.npz")}


def test_eviction_at_10_mb_removes_the_oldest_and_a_hit_refreshes(tmp_path):
    """Three 4 MB entries over a 10 MB bound: one goes. The oldest was read last, so the next goes."""
    paths = [
        _write(tmp_path, k, _big(i), "s1") for i, k in enumerate((KEY_A, KEY_B, KEY_C))
    ]
    for i, path in enumerate(paths):
        _age_entry(path, OLD + i)
    assert _disk.read_entry(paths[0]) is not None
    removed = _disk.evict(tmp_path, 10_000_000)
    assert removed == (KEY_B,)
    assert _keys_on_disk(tmp_path) == {KEY_A, KEY_C}
    assert not paths[1].with_suffix(".json").exists()


def test_eviction_without_a_hit_removes_by_age(tmp_path):
    paths = [
        _write(tmp_path, k, _big(i), "s1") for i, k in enumerate((KEY_A, KEY_B, KEY_C))
    ]
    for path, mtime in zip(paths, (OLD + 2, OLD, OLD + 1)):
        _age_entry(path, mtime)
    assert _disk.evict(tmp_path, 5_000_000) == (KEY_B, KEY_C)
    assert _keys_on_disk(tmp_path) == {KEY_A}


def test_eviction_under_the_bound_removes_nothing(tmp_path):
    _write(tmp_path, KEY_A, _big(0), "s1")
    assert _disk.evict(tmp_path, 10_000_000) == ()
    assert _keys_on_disk(tmp_path) == {KEY_A}


def test_eviction_spans_both_stages(tmp_path):
    a = _write(tmp_path, KEY_A, _big(0), "s0")
    b = _write(tmp_path, KEY_B, _big(1), "s1")
    _age_entry(a, OLD)
    _age_entry(b, OLD + 1)
    assert _disk.evict(tmp_path, 5_000_000) == (KEY_A,)
    assert not a.exists() and b.exists()


def test_eviction_refuses_a_negative_bound(tmp_path):
    with pytest.raises(ValueError, match="max_bytes"):
        _disk.evict(tmp_path, -1)


def test_eviction_of_an_absent_directory_removes_nothing(tmp_path):
    assert _disk.evict(tmp_path / "absent", 0) == ()


def test_name_2_copies_are_ignored_by_size_and_removal(tmp_path):
    """iCloud leaves "name 2" copies. They are neither counted nor removed."""
    path = _write(tmp_path, KEY_A, _big(0), "s1")
    for suffix in (".npz", ".json"):
        copy = path.parent / f"{KEY_A} 2{suffix}"
        copy.write_bytes(path.with_suffix(suffix).read_bytes())
    assert _disk.evict(tmp_path, 5_000_000) == ()
    assert _disk.evict(tmp_path, 0) == (KEY_A,)
    assert sorted(p.name for p in path.parent.iterdir()) == [
        f"{KEY_A} 2.json",
        f"{KEY_A} 2.npz",
    ]


def test_orphan_temporary_files_older_than_one_hour_are_removed(tmp_path):
    path = _write(tmp_path, KEY_A)
    now = time.time()
    old_tmp = path.parent / _disk.temporary_name(f"{KEY_B}.npz")
    young_tmp = path.parent / _disk.temporary_name(f"{KEY_C}.json")
    other = path.parent / "notes.txt"
    for f, age in ((old_tmp, HOUR + 60), (young_tmp, HOUR - 60), (other, 10 * HOUR)):
        f.write_bytes(b"partial")
        _age(f, now - age)
    assert _disk.evict(tmp_path, 10_000_000) == ()
    assert not old_tmp.exists()
    assert young_tmp.exists() and other.exists() and path.exists()


def test_temporary_name_is_not_an_entry_name():
    name = _disk.temporary_name(f"{KEY_A}.npz")
    assert str(os.getpid()) in name
    assert not name.endswith((".npz", ".json"))


def test_lone_files_count_and_are_removed(tmp_path):
    a = _write(tmp_path, KEY_A, _big(0), "s1")
    b = _write(tmp_path, KEY_B, _big(1), "s1")
    a.with_suffix(".json").unlink()
    _age(a, OLD)
    _age_entry(b, OLD + 1)
    assert _disk.evict(tmp_path, 5_000_000) == (KEY_A,)
    assert not a.exists() and b.exists()


# Two processes on one key ------------------------------------------------------------------------


def _race_worker(root: str, start, rounds: int) -> None:
    start.wait()
    path = _disk.entry_path(Path(root), "s1", KEY_A)
    for _ in range(rounds):
        _disk.write_entry(path, _big(7), {"stage": "s1"})
        entry = _disk.read_entry(path)
        if entry is not None and not _same(entry["arrays"], _big(7)):
            raise AssertionError("a read returned different bytes")


@pytest.mark.slow
def test_two_processes_writing_one_key_both_succeed(tmp_path):
    """No locks (decision 161): concurrent writers of one key leave one valid entry."""
    ctx = multiprocessing.get_context("spawn")
    start = ctx.Event()
    procs = [
        ctx.Process(target=_race_worker, args=(str(tmp_path), start, 20))
        for _ in range(2)
    ]
    for p in procs:
        p.start()
    start.set()
    for p in procs:
        p.join(timeout=300)
    assert [p.exitcode for p in procs] == [0, 0]
    path = _disk.entry_path(tmp_path, "s1", KEY_A)
    entry = _disk.read_entry(path)
    assert entry is not None and _same(entry["arrays"], _big(7))
    assert sorted(p.name for p in path.parent.iterdir()) == [
        f"{KEY_A}.json",
        f"{KEY_A}.npz",
    ]
