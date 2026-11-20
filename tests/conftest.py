"""Shared pytest settings live in pyproject.toml. This file gates ``slow`` tests and redirects the cache.

The cache goes to a session directory for module and class fixtures and to ``tmp_path`` for each test.

The ``slow`` marker is registered in ``pyproject.toml``. A test marked
``slow`` runs when it is selected explicitly (``-m slow``, or any ``-m``
expression naming ``slow`` without ``not slow``) or when the environment
sets ``SYNCEM_RUN_SLOW=1``. Otherwise it is skipped with that reason.
``-m 'not slow'`` deselects it.
"""

import os
import sys

import pytest

SLOW_ENV = "SYNCEM_RUN_SLOW"


def slow_requested(markexpr: str, environ=os.environ) -> bool:
    """True when the ``-m`` expression or ``SYNCEM_RUN_SLOW`` asks for slow tests."""
    if environ.get(SLOW_ENV, "").strip() not in ("", "0"):
        return True
    expr = " ".join(markexpr.split())
    return "slow" in expr and "not slow" not in expr


CACHE_ENV = "SYNCEM_CACHE"
CACHE_LIMIT_ENVS = (
    "SYNCEM_CACHE_MAX",
    "SYNCEM_CACHE_MEM",
    "SYNCEM_JET_MEM",
    "SYNCEM_JET_MAX_ORDER",
)


@pytest.fixture(scope="session", autouse=True)
def _syncem_cache_for_the_session(tmp_path_factory):
    """Point ``SYNCEM_CACHE`` at a session directory before any module or class fixture runs.

    pytest sets up higher-scoped fixtures before function-scoped ones, so a module fixture that builds
    with ``cache=True`` runs before ``_syncem_cache_in_tmp_path``. This fixture is the first of the
    session, and it removes the size variables as well.
    """
    patch = pytest.MonkeyPatch()
    patch.setenv(CACHE_ENV, str(tmp_path_factory.mktemp("syncem_session_cache")))
    for name in CACHE_LIMIT_ENVS:
        patch.delenv(name, raising=False)
    yield
    patch.undo()


@pytest.fixture(autouse=True)
def _syncem_cache_in_tmp_path(tmp_path, monkeypatch):
    """Point ``SYNCEM_CACHE`` at a directory under ``tmp_path`` for every test.

    No test writes a cache under the home directory, which may be a synced folder. The size
    variables of design/0.8/SPEC.md decisions 161 and 162 are removed, so every test starts from
    the defaults. A test that needs another setting sets it with ``monkeypatch``.
    """
    monkeypatch.setenv(CACHE_ENV, str(tmp_path / "syncem_cache"))
    for name in CACHE_LIMIT_ENVS:
        monkeypatch.delenv(name, raising=False)


CACHE_MODULE = "syncem.engine.cache"


def _drop_default_cache() -> None:
    module = sys.modules.get(CACHE_MODULE)
    if module is not None:
        module._DEFAULT = None


@pytest.fixture(autouse=True)
def _fresh_default_cache(_syncem_cache_in_tmp_path):
    """Give each test a fresh process ``Cache`` (design/0.8/SPEC.md decision 160).

    The default cache is dropped before and after the test, so the next ``default_cache()`` reads
    the ``SYNCEM_CACHE`` of this test and no entry passes from one test to another. The fixture
    imports nothing: before ``syncem.engine.cache`` is loaded there is no default cache to drop.
    """
    _drop_default_cache()
    yield
    _drop_default_cache()


def pytest_collection_modifyitems(config, items):
    if slow_requested(config.getoption("markexpr") or ""):
        return
    skip = pytest.mark.skip(reason=f"slow: select with -m slow or set {SLOW_ENV}=1")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
