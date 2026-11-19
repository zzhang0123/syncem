"""No test writes a cache under the home directory, module-scoped fixtures included (``tests/conftest.py``).

The module fixture ``parts`` of ``test_v080_basis_sum.py`` builds with the default ``cache=True`` before any
function-scoped fixture runs. The guard runs one of its tests in a subprocess with ``HOME`` under
``tmp_path`` and ``SYNCEM_CACHE``, ``SYNCEM_CACHE_MAX``, ``SYNCEM_CACHE_MEM`` and ``XDG_CACHE_HOME`` unset,
and asserts that no file appears under that ``HOME``. The check is an exact identity on the file list.
"""

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TARGET = (
    "tests/v080/test_v080_basis_sum.py::test_design_is_the_horizontal_concatenation"
)
UNSET = ("SYNCEM_CACHE", "SYNCEM_CACHE_MAX", "SYNCEM_CACHE_MEM", "XDG_CACHE_HOME")


def test_a_module_scoped_build_writes_nothing_under_home(tmp_path):
    home = tmp_path / "home"
    home.mkdir()
    env = {k: v for k, v in os.environ.items() if k not in UNSET}
    env.update(HOME=str(home), PYTHONPATH=str(ROOT))
    args = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", TARGET]
    args += ["--basetemp", str(tmp_path / "basetemp")]
    done = subprocess.run(args, cwd=ROOT, env=env, capture_output=True, text=True)
    assert done.returncode == 0, done.stdout + done.stderr
    assert [p for p in home.rglob("*") if p.is_file()] == []
