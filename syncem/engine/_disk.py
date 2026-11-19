"""Disk entries of the stage cache (design/0.8/SPEC.md decision 161, Sec. 2.3 and 8.5).

The directory and the size settings (``cache_dir``, ``parse_size``, ``env_bytes``, ``disk_limit``,
``disk_enabled`` and ``SyncedFolderWarning``) live in ``engine/_disk_settings.py`` and are re-exported here.

An entry is ``<root>/v<FORMAT>/<stage>/<key>.npz`` with ``<key>.json``, where ``stage`` is ``s0`` or ``s1``
and ``key`` is 64 lowercase hex characters. The ``.npz`` holds the arrays. The JSON holds the format, the
key, the dtype string and shape of every array, and the caller's description. A write goes to a
temporary file named by the process and thread and then ``os.replace``. The zip members carry a fixed
time stamp, so equal content gives equal bytes and concurrent writers of one key need no lock. The
``.npz`` is replaced before the JSON.

A read returns ``None`` on a miss. A missing or unreadable file is a miss and deletes nothing. A
corrupt entry (bad JSON, a bad zip, a plain ``.npy`` in place of the ``.npz``, other arrays, another
shape or dtype, an object array) is deleted and is a miss. A hit refreshes the modification time of both files, so eviction by the
oldest modification time removes the least recently used entry. Eviction counts and removes only
its own ``<64 hex>.npz`` and ``.json`` names, so "name 2" copies of a synced folder are left alone.
It removes temporary files older than one hour. An entry whose files cannot be removed stays counted,
and eviction goes on to the next entry.

This module knows nothing of what the arrays mean. The keys, the stages that go to disk and the
choice to skip the disk come from ``engine/cache.py``. It imports the standard library and NumPy.
"""

from __future__ import annotations

import io
import json
import os
import re
import threading
import time
import zipfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

from ._disk_settings import (
    DEFAULT_MAX_BYTES,
    SyncedFolderWarning,
    cache_dir,
    disk_enabled,
    disk_limit,
    env_bytes,
    parse_size,
)

__all__ = [
    "DEFAULT_MAX_BYTES",
    "FORMAT",
    "STAGES",
    "SyncedFolderWarning",
    "cache_dir",
    "disk_enabled",
    "disk_limit",
    "entry_path",
    "env_bytes",
    "evict",
    "parse_size",
    "plain_description",
    "read_entry",
    "temporary_name",
    "write_entry",
]

FORMAT = 1
STAGES = ("s0", "s1")
TEMPORARY_AGE = 3600.0

_KEY = re.compile(r"[0-9a-f]{64}")
_ENTRY = re.compile(r"([0-9a-f]{64})\.(npz|json)")
_TEMPORARY = re.compile(r"[0-9a-f]{64}\.(npz|json)\.\d+-\d+\.tmp")
_ZIP_TIME = (1980, 1, 1, 0, 0, 0)


# Directory and settings live in engine/_disk_settings.py ----------------------------------------


def entry_path(root: Path | str, stage: str, key: str) -> Path:
    """The ``.npz`` path of one entry, ``<root>/v<FORMAT>/<stage>/<key>.npz``."""
    if stage not in STAGES:
        raise ValueError(
            f"stage {stage!r}: the disk holds the stages {', '.join(STAGES)}"
        )
    _check_key(key)
    return Path(root) / f"v{FORMAT}" / stage / f"{key}.npz"


def _check_key(key: str) -> None:
    if not _KEY.fullmatch(key):
        raise ValueError(f"key {key!r}: a disk key is 64 lowercase hex characters")


def temporary_name(name: str) -> str:
    """The temporary file name of ``name`` for this process and thread."""
    return f"{name}.{os.getpid()}-{threading.get_ident()}.tmp"


# Writing ---------------------------------------------------------------------------------------


def _check_arrays(arrays: Mapping[str, Any]) -> dict[str, np.ndarray]:
    checked = {}
    for name, value in arrays.items():
        array = np.ascontiguousarray(value)
        if array.dtype.hasobject or array.dtype.fields is not None:
            raise ValueError(
                f"array {name!r}: object and structured arrays are not written"
            )
        checked[str(name)] = array
    return checked


def _npz_bytes(arrays: Mapping[str, np.ndarray]) -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", zipfile.ZIP_STORED) as archive:
        for name in sorted(arrays):
            member = io.BytesIO()
            np.lib.format.write_array(member, arrays[name], allow_pickle=False)
            archive.writestr(
                zipfile.ZipInfo(f"{name}.npy", _ZIP_TIME), member.getvalue()
            )
    return buffer.getvalue()


def _json_bytes(key: str, arrays: Mapping[str, np.ndarray], description: Any) -> bytes:
    meta = {
        "format": FORMAT,
        "key": key,
        "arrays": {
            n: {"dtype": a.dtype.str, "shape": list(a.shape)} for n, a in arrays.items()
        },
        "description": description,
    }
    text = json.dumps(meta, sort_keys=True, allow_nan=False, separators=(",", ":"))
    return text.encode("ascii")


def _plain(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {k: _plain(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(v) for v in value]
    return value


def plain_description(description: Any) -> Any:
    """``description`` with NumPy scalars as Python numbers, checked as strict JSON.

    A value that is not strict JSON raises ``TypeError`` or ``ValueError`` with a message naming the
    description.
    """
    out = _plain(description)
    try:
        json.dumps(out, sort_keys=True, allow_nan=False)
    except (TypeError, ValueError) as err:
        raise type(err)(
            f"description {description!r} is not strict JSON: {err}"
        ) from None
    return out


def _replace(path: Path, data: bytes) -> None:
    temporary = path.with_name(temporary_name(path.name))
    try:
        with open(temporary, "wb") as handle:
            handle.write(data)
        os.replace(temporary, path)
    finally:
        _remove(temporary)


def write_entry(path: Path | str, arrays: Mapping[str, Any], description: Any) -> None:
    """Write one entry: the arrays to ``path`` (a ``<key>.npz``) and the JSON beside it.

    ``description`` is any strict-JSON value, NumPy scalars written as Python numbers. Object and
    structured arrays and a file name other than ``<64 hex>.npz`` raise ``ValueError``, and a non-JSON
    description raises ``TypeError`` or ``ValueError``, before anything is written.
    """
    path = Path(path)
    if path.suffix != ".npz":
        raise ValueError(f"path {path}: an entry is a <64 hex>.npz file")
    _check_key(path.stem)
    checked = _check_arrays(arrays)
    meta = _json_bytes(path.stem, checked, plain_description(description))
    path.parent.mkdir(parents=True, exist_ok=True)
    _replace(path, _npz_bytes(checked))
    _replace(path.with_suffix(".json"), meta)


# Reading ---------------------------------------------------------------------------------------


class _Corrupt(Exception):
    """An entry whose files exist and disagree with the format."""


def _unlink(path: Path) -> None:
    os.unlink(path)


def _remove(path: Path) -> bool:
    """Remove ``path``. True when the file is gone, also when another process removed it first."""
    try:
        _unlink(path)
    except FileNotFoundError:
        return True
    except OSError:
        return False
    return True


def _meta(path: Path) -> dict[str, Any]:
    text = path.with_suffix(".json").read_bytes()
    try:
        meta = json.loads(text)
    except ValueError as err:
        raise _Corrupt(f"JSON: {err}") from None
    if not isinstance(meta, dict) or not isinstance(meta.get("arrays"), dict):
        raise _Corrupt("JSON without an array table")
    if meta.get("format") != FORMAT or meta.get("key") != path.stem:
        raise _Corrupt("JSON of another format or key")
    return meta


def _load_arrays(path: Path, spec: Mapping[str, Any]) -> dict[str, np.ndarray]:
    try:
        loaded = np.load(path, allow_pickle=False)
        if not isinstance(loaded, np.lib.npyio.NpzFile):
            raise _Corrupt("a plain array in place of an npz archive")
        with loaded as npz:
            if sorted(npz.files) != sorted(spec):
                raise _Corrupt("arrays differ from the JSON")
            arrays = {name: npz[name] for name in npz.files}
    except (zipfile.BadZipFile, ValueError, EOFError, KeyError) as err:
        raise _Corrupt(f"npz: {err}") from None
    for name, array in arrays.items():
        want = spec[name]
        if not isinstance(want, dict):
            raise _Corrupt(f"array {name!r} without dtype and shape")
        if array.dtype.str != want.get("dtype") or list(array.shape) != want.get(
            "shape"
        ):
            raise _Corrupt(f"array {name!r} differs in dtype or shape from the JSON")
    return arrays


def _touch(path: Path) -> None:
    for file in (path, path.with_suffix(".json")):
        try:
            os.utime(file)
        except OSError:
            pass  # an entry evicted after the read still served this read


def read_entry(path: Path | str) -> dict[str, Any] | None:
    """The entry at ``path`` as ``{"arrays": {name: ndarray}, "description": value}``, or ``None``.

    A missing or unreadable file is a miss. A corrupt entry is deleted and is a miss. A hit refreshes
    the modification time of both files.
    """
    path = Path(path)
    try:
        meta = _meta(path)
        arrays = _load_arrays(path, meta["arrays"])
    except _Corrupt:
        _remove(path)
        _remove(path.with_suffix(".json"))
        return None
    except OSError:
        return None
    _touch(path)
    return {"arrays": arrays, "description": meta.get("description")}


# Eviction --------------------------------------------------------------------------------------


def _scan(directory: Path, now: float) -> dict[tuple[Path, str], list[os.stat_result]]:
    """Own entry files grouped by (stage directory, key). Old temporary files are removed."""
    groups: dict[tuple[Path, str], list[os.stat_result]] = {}
    root = directory / f"v{FORMAT}"
    for stage in STAGES:
        try:
            items = list(os.scandir(root / stage))
        except OSError:
            continue
        for item in items:
            try:
                stat = item.stat(follow_symlinks=False)
            except OSError:
                continue
            match = _ENTRY.fullmatch(item.name)
            if match:
                groups.setdefault((root / stage, match.group(1)), []).append(stat)
            elif (
                _TEMPORARY.fullmatch(item.name) and now - stat.st_mtime > TEMPORARY_AGE
            ):
                _remove(Path(item.path))
    return groups


def evict(directory: Path | str, max_bytes: int) -> tuple[str, ...]:
    """Remove the oldest entries under the cache root ``directory`` until they hold at most ``max_bytes``.

    Returns the removed keys in the order of removal. An entry's age is the newest modification time
    of its files, and ties go by key. An entry whose files cannot be removed is left out of the
    returned keys and still counts. Temporary files older than one hour are removed as well.
    """
    if max_bytes < 0:
        raise ValueError(
            f"max_bytes {max_bytes}: the bound is a non-negative number of bytes"
        )
    groups = _scan(Path(directory), time.time())
    total = sum(s.st_size for stats in groups.values() for s in stats)
    order = sorted(groups, key=lambda g: (max(s.st_mtime for s in groups[g]), g[1]))
    removed = []
    for stage_dir, key in order:
        if total <= max_bytes:
            break
        gone = [_remove(stage_dir / f"{key}.{ext}") for ext in ("npz", "json")]
        if all(gone):
            total -= sum(s.st_size for s in groups[(stage_dir, key)])
            removed.append(key)
    return tuple(removed)
