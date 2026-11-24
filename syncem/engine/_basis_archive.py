"""Atomic basis archive with strict JSON metadata and numeric NPY members."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import tempfile
from zipfile import BadZipFile

import numpy as np

from .. import __version__
from ..values.errors import DataError

_FORMAT = 1
_METADATA = "metadata_json"
_KIND = "SyncEM SpectralBasis"


def _json(value):
    return json.dumps(
        value,
        allow_nan=False,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _unique(pairs):
    result = dict(pairs)
    if len(result) != len(pairs):
        raise ValueError("duplicate JSON metadata key")
    return result


def _invalid_float(token):
    raise ValueError(f"invalid JSON number {token!r}")


def _array_info(array):
    data = np.ravel(array, order="C").view(np.uint8)
    return {
        "sha256": hashlib.sha256(data).hexdigest(),
        "shape": list(array.shape),
        "dtype": array.dtype.str,
    }


def write_archive(path, body, arrays, *, kind=_KIND):
    """Write an archive with one atomically replaced destination."""
    manifest = {
        "kind": kind,
        "format": _FORMAT,
        "package_version": __version__,
        "arrays": {key: _array_info(a) for key, a in arrays.items()},
        "basis": body,
    }
    signed = {
        "manifest": manifest,
        "sha256": hashlib.sha256(_json(manifest)).hexdigest(),
    }
    meta = np.frombuffer(_json(signed), dtype=np.uint8)
    target = Path(path)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            dir=target.parent, prefix=f".{target.name}.", delete=False
        ) as stream:
            temporary = Path(stream.name)
            np.savez(stream, allow_pickle=False, **{_METADATA: meta, **arrays})
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, target)
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)


def _read_payload(archive, kind):
    if len(set(archive.files)) != len(archive.files):
        raise ValueError("duplicate NPZ members")
    raw = archive[_METADATA]
    if raw.ndim != 1 or raw.dtype != np.dtype(np.uint8):
        raise ValueError("metadata must be a UTF-8 byte array")
    signed = json.loads(
        raw.tobytes().decode("utf-8"),
        object_pairs_hook=_unique,
        parse_constant=_invalid_float,
    )
    if set(signed) != {"manifest", "sha256"}:
        raise ValueError("invalid basis archive header")
    manifest = signed["manifest"]
    if hashlib.sha256(_json(manifest)).hexdigest() != signed["sha256"]:
        raise ValueError("basis metadata integrity mismatch")
    if manifest["kind"] != kind or manifest["format"] != _FORMAT:
        raise ValueError("unsupported spectral basis archive format")
    if manifest["package_version"] != __version__:
        raise ValueError(
            f"basis package version {manifest['package_version']!r} differs from {__version__!r}"
        )
    known = manifest["arrays"]
    if set(archive.files) != {_METADATA, *known}:
        raise ValueError("saved array names do not match the basis metadata")
    arrays = {}
    for name, info in known.items():
        array = archive[name]
        if array.dtype.kind not in "biufc" or _array_info(array) != info:
            raise ValueError(f"basis array {name!r} failed integrity validation")
        array.setflags(write=False)
        arrays[name] = array
    return manifest["basis"], arrays


def read_archive(path, *, kind=_KIND):
    """Read and validate the archive with pickle loading disabled."""
    try:
        with np.load(Path(path), allow_pickle=False) as archive:
            return _read_payload(archive, kind)
    except (
        BadZipFile,
        OSError,
        TypeError,
        ValueError,
        KeyError,
        AttributeError,
    ) as exc:
        raise DataError(f"cannot load the basis archive {Path(path)}. {exc}") from exc
