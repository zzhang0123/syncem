"""Step 5b-2 of design/0.8/SPEC.md: the settings of ``syncem.engine._disk`` (decision 161).

The cache directory, its platform default, the synced-folder warning and the size syntax of
``SYNCEM_CACHE_MAX``. Entries and eviction are in ``test_v080_cache_disk.py``. Every comparison is an
exact identity of paths, integers or warning classes, so no tolerance enters. The synced-folder cases
use a fake ``HOME`` under ``tmp_path`` and create nothing.
"""

from __future__ import annotations

import os
import sys
import warnings
from pathlib import Path

import pytest

from syncem.engine import _disk

# Directory ---------------------------------------------------------------------------------------


def test_conftest_redirects_the_cache_to_a_test_directory(tmp_path):
    assert Path(os.environ["SYNCEM_CACHE"]).is_relative_to(tmp_path)
    assert _disk.cache_dir() == Path(os.environ["SYNCEM_CACHE"])


def test_syncem_cache_redirect_expands_user_and_makes_relative_paths_absolute(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HOME", str(tmp_path / "home"))
    monkeypatch.setenv("SYNCEM_CACHE", "~/elsewhere")
    assert _disk.cache_dir() == tmp_path / "home" / "elsewhere"
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("SYNCEM_CACHE", "relative/dir")
    assert _disk.cache_dir() == tmp_path / "relative" / "dir"


def test_cache_dir_creates_nothing(tmp_path, monkeypatch):
    monkeypatch.setenv("SYNCEM_CACHE", str(tmp_path / "not-yet"))
    _disk.cache_dir()
    assert not (tmp_path / "not-yet").exists()


@pytest.mark.parametrize(
    ("platform", "env", "expected"),
    [
        ("darwin", {}, ("home", "Library", "Caches", "syncem")),
        ("darwin", {"XDG_CACHE_HOME": "/xdg"}, ("home", "Library", "Caches", "syncem")),
        ("linux", {}, ("home", ".cache", "syncem")),
        ("linux", {"XDG_CACHE_HOME": "/xdg"}, ("/xdg", "syncem")),
        ("linux", {"XDG_CACHE_HOME": ""}, ("home", ".cache", "syncem")),
        ("linux", {"XDG_CACHE_HOME": "relative"}, ("home", ".cache", "syncem")),
        ("freebsd14", {}, ("home", ".cache", "syncem")),
        ("win32", {"LOCALAPPDATA": "/appdata"}, ("/appdata", "syncem")),
        ("win32", {}, ("home", "AppData", "Local", "syncem")),
    ],
)
def test_platform_directory(tmp_path, monkeypatch, platform, env, expected):
    """The default directory of SPEC decision 161, under a monkeypatched platform and environment."""
    for name in ("SYNCEM_CACHE", "XDG_CACHE_HOME", "LOCALAPPDATA"):
        monkeypatch.delenv(name, raising=False)
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("USERPROFILE", str(home))
    for name, value in env.items():
        monkeypatch.setenv(name, value)
    monkeypatch.setattr(sys, "platform", platform)
    head, *rest = expected
    base = home if head == "home" else Path(head)
    assert _disk.cache_dir() == base.joinpath(*rest)


@pytest.mark.parametrize("platform", ["darwin", "linux", "win32"])
def test_syncem_cache_overrides_every_platform(tmp_path, monkeypatch, platform):
    monkeypatch.setattr(sys, "platform", platform)
    monkeypatch.setenv("SYNCEM_CACHE", str(tmp_path / "mine"))
    assert _disk.cache_dir() == tmp_path / "mine"


def test_empty_syncem_cache_falls_back_to_the_platform_directory(tmp_path, monkeypatch):
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("XDG_CACHE_HOME", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setenv("SYNCEM_CACHE", "")
    assert _disk.cache_dir() == tmp_path / ".cache" / "syncem"


# Synced-folder warning ---------------------------------------------------------------------------

SYNCED = (
    ("Documents",),
    ("Desktop",),
    ("Library", "Mobile Documents"),
    ("Library", "CloudStorage"),
)


@pytest.mark.parametrize("parts", SYNCED)
@pytest.mark.parametrize("below", [(), ("project", "cache")])
def test_synced_folder_gives_a_warning_and_is_used(tmp_path, monkeypatch, parts, below):
    """Owner answer of 2026-11-19 (question 2): a warning, not an error. ``HOME`` is fake."""
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    target = home.joinpath(*parts, *below)
    monkeypatch.setenv("SYNCEM_CACHE", str(target))
    with pytest.warns(_disk.SyncedFolderWarning, match="synced"):
        assert _disk.cache_dir() == target
    assert not home.exists()


@pytest.mark.parametrize(
    "parts",
    [
        ("Documents2",),
        ("MyDocuments",),
        ("Library", "Caches"),
        ("Library", "Mobile"),
        (),
    ],
)
def test_neighbours_of_synced_folders_give_no_warning(tmp_path, monkeypatch, parts):
    home = tmp_path / "home"
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SYNCEM_CACHE", str(home.joinpath(*parts, "cache")))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        _disk.cache_dir()


def test_synced_folder_is_found_through_a_symlink(tmp_path, monkeypatch):
    home = tmp_path / "home"
    (home / "Documents").mkdir(parents=True)
    link = tmp_path / "link"
    link.symlink_to(home / "Documents")
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("SYNCEM_CACHE", str(link / "cache"))
    with pytest.warns(_disk.SyncedFolderWarning):
        _disk.cache_dir()


def test_platform_default_inside_a_synced_folder_also_warns(tmp_path, monkeypatch):
    home = tmp_path / "home"
    monkeypatch.setattr(sys, "platform", "linux")
    monkeypatch.delenv("SYNCEM_CACHE", raising=False)
    monkeypatch.setenv("HOME", str(home))
    monkeypatch.setenv("XDG_CACHE_HOME", str(home / "Desktop" / "xdg"))
    with pytest.warns(_disk.SyncedFolderWarning):
        _disk.cache_dir()


# Size syntax -------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("0", 0),
        ("12345", 12345),
        (" 7 ", 7),
        ("1kB", 1000),
        ("10 MB", 10_000_000),
        ("2GB", 2_000_000_000),
        ("1.5GB", 1_500_000_000),
        ("0.5kB", 500),
        ("1.2345kB", 1234),
        ("0MB", 0),
    ],
)
def test_parse_size(text, expected):
    """Bytes or a decimal suffix kB, MB, GB as powers of 1000 (SPEC decision 161), rounded down."""
    assert _disk.parse_size(text) == expected


@pytest.mark.parametrize(
    "text",
    ["", "-1", "1.5", "1e9", "1 KiB", "1 TB", "kB", "1 k B", "1,000", "nan", "1 gb"],
)
def test_parse_size_refuses_other_forms(text):
    with pytest.raises(ValueError, match="kB, MB or GB"):
        _disk.parse_size(text)


def test_disk_limit_default_override_and_zero_disables(monkeypatch):
    monkeypatch.delenv("SYNCEM_CACHE_MAX", raising=False)
    assert _disk.disk_limit() == _disk.DEFAULT_MAX_BYTES == 2_000_000_000
    assert _disk.disk_enabled()
    monkeypatch.setenv("SYNCEM_CACHE_MAX", "10MB")
    assert _disk.disk_limit() == 10_000_000
    assert _disk.disk_enabled()
    monkeypatch.setenv("SYNCEM_CACHE_MAX", "0")
    assert _disk.disk_limit() == 0
    assert not _disk.disk_enabled()
    monkeypatch.setenv("SYNCEM_CACHE_MAX", "lots")
    with pytest.raises(ValueError, match="SYNCEM_CACHE_MAX"):
        _disk.disk_limit()


def test_env_bytes_reads_any_variable(monkeypatch):
    monkeypatch.setenv("SYNCEM_CACHE_MEM", "3MB")
    assert _disk.env_bytes("SYNCEM_CACHE_MEM", 1) == 3_000_000
    monkeypatch.delenv("SYNCEM_CACHE_MEM")
    assert _disk.env_bytes("SYNCEM_CACHE_MEM", 1) == 1
