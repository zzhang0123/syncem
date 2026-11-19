"""The directory and the size settings of the disk cache (design/0.8/SPEC.md decisions 161 and 162).

The directory is ``$SYNCEM_CACHE`` or the platform cache directory, ``~/Library/Caches/syncem`` on macOS,
``$XDG_CACHE_HOME/syncem`` or ``~/.cache/syncem`` on Linux and other POSIX systems, and
``%LOCALAPPDATA%\\syncem`` on Windows. A leading ``~`` followed by ``/`` (or by ``\\`` on Windows) is the home
directory. A directory under ``~/Documents``, ``~/Desktop``, ``~/Library/Mobile Documents`` or
``~/Library/CloudStorage`` gives a ``SyncedFolderWarning`` and is used. The check compares the given and
resolved paths, without case on macOS and Windows, whose file systems ignore case by default.

A size is an integer number of bytes or a decimal number with the suffix kB, MB or GB (powers of 1000),
rounded down to whole bytes. ``1e9`` and lower-case suffixes are refused with the accepted syntax in the
message. ``engine/_disk.py`` imports and re-exports these names. The module imports the standard library.
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from collections.abc import Mapping
from decimal import ROUND_FLOOR, Decimal
from pathlib import Path

__all__ = [
    "CACHE_ENV",
    "DEFAULT_MAX_BYTES",
    "MAX_ENV",
    "SYNCED_FOLDERS",
    "SyncedFolderWarning",
    "cache_dir",
    "disk_enabled",
    "disk_limit",
    "env_bytes",
    "parse_size",
]

DEFAULT_MAX_BYTES = 2_000_000_000
CACHE_ENV = "SYNCEM_CACHE"
MAX_ENV = "SYNCEM_CACHE_MAX"
SYNCED_FOLDERS = (
    ("Documents",),
    ("Desktop",),
    ("Library", "Mobile Documents"),
    ("Library", "CloudStorage"),
)
_SIZE = re.compile(r"\s*(\d+(?:\.\d+)?)\s*(kB|MB|GB)?\s*")
_SUFFIX = {None: 1, "kB": 10**3, "MB": 10**6, "GB": 10**9}
_FOLDING = ("darwin", "win32")  # platforms whose default file systems ignore case


class SyncedFolderWarning(UserWarning):
    """The cache directory lies in a folder that a file-sync service may copy and duplicate."""


# Directory and settings ------------------------------------------------------------------------


def _environ(environ: Mapping[str, str] | None) -> Mapping[str, str]:
    return os.environ if environ is None else environ


def _home(environ: Mapping[str, str]) -> Path:
    names = ("USERPROFILE", "HOME") if sys.platform == "win32" else ("HOME",)
    for name in names:
        if environ.get(name):
            return Path(environ[name])
    return Path.home()


def _platform_dir(environ: Mapping[str, str], home: Path) -> Path:
    if sys.platform == "darwin":
        return home / "Library" / "Caches" / "syncem"
    if sys.platform == "win32":
        local = environ.get("LOCALAPPDATA")
        return (
            Path(local) / "syncem" if local else home / "AppData" / "Local" / "syncem"
        )
    xdg = environ.get("XDG_CACHE_HOME", "")
    # The XDG base-directory specification ignores a relative XDG_CACHE_HOME.
    return Path(xdg) / "syncem" if os.path.isabs(xdg) else home / ".cache" / "syncem"


def _folded(path: Path) -> Path:
    return Path(str(path).casefold()) if sys.platform in _FOLDING else path


def _within(path: Path, root: Path) -> bool:
    forms = {_folded(p) for p in (path, path.resolve())}
    roots = {_folded(r) for r in (root, root.resolve())}
    return any(p.is_relative_to(r) for p in forms for r in roots)


def _expand(value: str, home: Path) -> str:
    """``value`` with a leading ``~`` or ``~/`` (and ``~\\`` on Windows) as the home directory."""
    seps = ("/", "\\") if sys.platform == "win32" else ("/",)
    if value == "~":
        return str(home)
    if value[:1] == "~" and value[1:2] in seps:
        return str(home / value[2:])
    return value


def _warn_if_synced(path: Path, home: Path) -> None:
    for parts in SYNCED_FOLDERS:
        if _within(path, home.joinpath(*parts)):
            warnings.warn(
                f"the syncem cache directory {path} is in the synced folder "
                f"{home.joinpath(*parts)}, where sync copies can fill the disk. "
                f"Unset {CACHE_ENV} to use the platform cache directory.",
                SyncedFolderWarning,
                stacklevel=3,
            )
            return


def cache_dir(environ: Mapping[str, str] | None = None) -> Path:
    """The cache root: ``$SYNCEM_CACHE`` made absolute, or the platform cache directory.

    ``environ`` defaults to ``os.environ`` and the platform is ``sys.platform``, both read at the call.
    The directory is not created. A root in a synced folder gives a ``SyncedFolderWarning``.
    """
    env = _environ(environ)
    home = _home(env)
    value = env.get(CACHE_ENV, "")
    if value:
        path = Path(os.path.abspath(_expand(value, home)))
    else:
        path = _platform_dir(env, home)
    _warn_if_synced(path, home)
    return path


def parse_size(text: str) -> int:
    """Bytes from an integer or a decimal number with the suffix kB, MB or GB (powers of 1000).

    A number with a suffix is rounded down to whole bytes. Any other form raises ``ValueError``.
    """
    match = _SIZE.fullmatch(text)
    if match is None or (match.group(2) is None and "." in match.group(1)):
        raise ValueError(
            f"size {text!r}: give an integer number of bytes, or a number with kB, MB or GB"
        )
    value = Decimal(match.group(1)) * _SUFFIX[match.group(2)]
    return int(value.to_integral_value(rounding=ROUND_FLOOR))


def env_bytes(name: str, default: int, environ: Mapping[str, str] | None = None) -> int:
    """The size in the environment variable ``name`` by ``parse_size``, or ``default`` when unset or empty."""
    value = _environ(environ).get(name, "")
    if not value.strip():
        return default
    try:
        return parse_size(value)
    except ValueError as err:
        raise ValueError(f"{name}: {err}") from None


def disk_limit(environ: Mapping[str, str] | None = None) -> int:
    """The disk bound in bytes, ``SYNCEM_CACHE_MAX`` or 2 GB. The value 0 disables the disk."""
    return env_bytes(MAX_ENV, DEFAULT_MAX_BYTES, environ)


def disk_enabled(environ: Mapping[str, str] | None = None) -> bool:
    """True unless ``SYNCEM_CACHE_MAX`` is 0."""
    return disk_limit(environ) > 0
