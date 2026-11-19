"""Function fingerprints loaded outside the package and under other hash seeds (decision 156, step 5b-2).

``_fingerprint.py`` and ``_function_fingerprint.py`` load from their files under a package of their own, so
they need NumPy and the standard library only. One subprocess per ``PYTHONHASHSEED`` fingerprints a frozenset
constant. Every assertion is an exact identity of strings, so no tolerance enters.
"""

import importlib
import os
import subprocess
import sys
import types
from pathlib import Path

import pytest

from syncem.base import _function_fingerprint as ffm
from tests.v080.test_v080_fingerprint import _is_hex, _outer

ffp = ffm.function_fingerprint
BASE_DIR = Path(ffm.__file__).resolve().parent


_ALONE = (
    "import importlib, sys, types\n"
    "pkg = types.ModuleType('fp_alone')\n"
    "pkg.__path__ = [sys.argv[1]]\n"
    "sys.modules['fp_alone'] = pkg\n"
    "fp = importlib.import_module('fp_alone._function_fingerprint')\n"
)
_SEED_SCRIPT = (
    _ALONE + "TABLE = {'k%d' % i: i for i in range(40)}\n"
    "def f(x):\n"
    "    return x in {'a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j'} and TABLE\n"
    "print(fp.function_fingerprint(f))\n"
)


@pytest.mark.parametrize(
    "seeds", [("1", "2"), pytest.param(("1", "2", "3", "4"), marks=pytest.mark.slow)]
)
def test_fingerprints_are_equal_across_hash_seeds(seeds):
    """One subprocess per seed loads the module from its file and fingerprints a frozenset constant."""
    outputs = set()
    for seed in seeds:
        done = subprocess.run(
            [sys.executable, "-c", _SEED_SCRIPT, str(BASE_DIR)],
            env={**os.environ, "PYTHONHASHSEED": seed},
            capture_output=True,
            text=True,
            check=True,
        )
        outputs.add(done.stdout.strip())
    assert len(outputs) == 1 and _is_hex(outputs.pop())


def test_fingerprint_module_loads_without_the_package(monkeypatch):
    """``_fingerprint`` and ``_function_fingerprint`` load from their files under a package of their own."""
    package = types.ModuleType("fp_alone")
    package.__path__ = [str(BASE_DIR)]
    monkeypatch.setitem(sys.modules, "fp_alone", package)
    for name in ("fp_alone._fingerprint", "fp_alone._function_fingerprint"):
        monkeypatch.delitem(sys.modules, name, raising=False)
    module = importlib.import_module("fp_alone._function_fingerprint")
    assert module.function_fingerprint(_outer) == ffp(_outer)
    for name in ("fp_alone._fingerprint", "fp_alone._function_fingerprint"):
        sys.modules.pop(name, None)
