"""What ``function_fingerprint`` reaches into (design/0.8/SPEC.md decision 156, review findings of step 5b-2).

Library code is the standard library, the package ``syncem`` and packages whose files lie under a
site-packages directory. Its functions without closure cells are written by module, name and version. Every
other function is user code and is traced, a versioned package outside site-packages included, and so is any
function with closure cells. A user module that a function names as a global enters through the attributes
the bytecode reads from it. A user module used in any other way raises ``NotCacheable``. The user packages
of these tests are written under ``tmp_path`` and removed from ``sys.modules`` afterwards. Every assertion is
an exact identity of strings, so no tolerance enters.
"""

import importlib
import sys
import textwrap

import numpy as np
import pytest

import syncem
from syncem.base import _fingerprint as enc
from syncem.base import _function_fingerprint as fp
from syncem.engine.average import fixed_rule

ffp = fp.function_fingerprint


def _load(tmp_path, monkeypatch, files):
    """Write ``files`` (module name to source) under ``tmp_path`` and import them as user modules."""
    root = tmp_path / "user_modules"
    root.mkdir(exist_ok=True)
    for name, source in files.items():
        (root / f"{name}.py").write_text(textwrap.dedent(source))
    monkeypatch.syspath_prepend(str(root))
    importlib.invalidate_caches()
    loaded = {}
    for name in files:
        module = importlib.import_module(name)
        del sys.modules[name]  # so that the undo of setitem removes the module again
        monkeypatch.setitem(sys.modules, name, module)
        loaded[name] = module
    return loaded


VERSIONED = """
__version__ = "1.0"
SCALE = 1.0


def make(alpha):
    def law(a):
        return alpha * a

    return law


def scaled(a):
    return SCALE * a
"""


@pytest.fixture
def versioned(tmp_path, monkeypatch):
    return _load(tmp_path, monkeypatch, {"reach_versioned": VERSIONED})[
        "reach_versioned"
    ]


# ------------------------------------------------- library code against user code


def test_library_version_names_the_library_and_not_a_versioned_user_package(
    versioned,
):
    assert fp._library_version("json") == ""
    assert fp._library_version("numpy.linalg") == np.__version__
    assert fp._library_version("syncem.engine.average") == syncem.__version__
    assert versioned.__version__ == "1.0"
    assert fp._library_version("reach_versioned") is None
    assert fp._library_version("__main__") is None


def test_a_closure_of_a_versioned_user_package_is_traced(versioned):
    assert ffp(versioned.make(1.0)) != ffp(versioned.make(3.0))
    assert ffp(versioned.make(1.0)) == ffp(versioned.make(1.0))


def test_a_global_of_a_versioned_user_package_is_traced(versioned, monkeypatch):
    before = ffp(versioned.scaled)
    assert ffp(versioned.scaled) == before
    monkeypatch.setattr(versioned, "SCALE", 3.0)
    assert ffp(versioned.scaled) != before


def test_a_closure_of_syncem_is_traced_and_a_library_function_is_named():
    one, two = (fixed_rule([x], [1.0]).fn for x in (1.0, 2.0))
    assert ffp(one) != ffp(two) and ffp(one) == ffp(fixed_rule([1.0], [1.0]).fn)
    assert ffp(np.linalg.norm) == ffp(np.linalg.norm)


# ------------------------------------------------- attributes of user modules

HELPERS = """
import numpy as np

TABLE = np.array([1.0, 2.0])


def helper(a):
    return 2.0 * a
"""

READER = """
import reach_helpers as rh


def reads_table(a):
    return rh.TABLE[0] * a


def calls_helper(a):
    return rh.helper(a)


def nested(a):
    return (lambda: rh.TABLE[0])() * a


def passes_module(a):
    return len(vars(rh)) * a


def make_holder():
    held = rh

    def law(a):
        return held.TABLE[0] * a

    return law
"""


@pytest.fixture
def reader(tmp_path, monkeypatch):
    found = _load(
        tmp_path, monkeypatch, {"reach_helpers": HELPERS, "reach_reader": READER}
    )
    return found["reach_helpers"], found["reach_reader"]


@pytest.mark.parametrize("name", ["reads_table", "nested"])
def test_an_in_place_edit_of_a_module_attribute_changes_the_fingerprint(reader, name):
    helpers, module = reader
    fn = getattr(module, name)
    before = ffp(fn)
    assert ffp(fn) == before
    helpers.TABLE[0] = 10.0
    assert ffp(fn) != before


def test_a_replaced_module_function_changes_the_fingerprint(reader, monkeypatch):
    helpers, module = reader
    before = ffp(module.calls_helper)
    monkeypatch.setattr(helpers, "helper", lambda a: 3.0 * a)
    assert ffp(module.calls_helper) != before


def test_a_user_module_used_other_than_by_attribute_reads_is_not_cacheable(reader):
    _, module = reader
    with pytest.raises(enc.NotCacheable, match="reach_helpers"):
        ffp(module.passes_module)
    with pytest.raises(enc.NotCacheable, match="reach_helpers"):
        ffp(module.make_holder())
