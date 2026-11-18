"""Import-graph guard for the 0.8 layers (design/0.8/SPEC.md Sec. 3.1 and 9.7).

The checker in ``_import_graph.py`` walks every import statement of every module
under ``syncem/``, function-local imports included. A module-level import must
follow the layer order, with ``if TYPE_CHECKING:`` blocks exempt. A function-local
import of a later layer must sit in a method of the delegation table of Sec. 3.1.
The first test applies it to the package. The others check the checker on small
synthetic sources, so that a vacuous pass is not mistaken for enforcement.
"""

from pathlib import Path

import pytest

from _import_graph import check_package, check_source, layer_of

PACKAGE = Path(__file__).resolve().parents[1] / "syncem"


def test_package_follows_the_layer_order():
    assert check_package(PACKAGE) == []


@pytest.mark.parametrize(
    ("module", "layer"),
    [
        ("syncem.units", "units"),
        ("syncem.values.spectrum", "values"),
        ("syncem.base.kernel", "base"),
        ("syncem.lattice.cells", "lattice"),
        ("syncem.engine.provider_jet", "engine"),
        ("syncem.fitting.fit", "fitting"),
        ("syncem.paper", "front"),
        ("syncem", "front"),
        ("syncem.model.basis", "existing"),
        ("syncem.model", "existing"),
        ("syncem.constants", "existing"),
        ("numpy", None),
        ("jax.numpy", None),
    ],
)
def test_layer_of(module, layer):
    assert layer_of(module) == layer


def _violations(module, source):
    return check_source(module, source)


def test_earlier_layer_import_passes():
    assert (
        _violations("syncem.engine.basis", "from syncem.lattice.cells import Cells\n")
        == []
    )


def test_later_layer_import_at_module_level_fails():
    found = _violations(
        "syncem.lattice.lattice",
        "from syncem.planning.defaults import default_lattice\n",
    )
    assert len(found) == 1
    assert "syncem.planning.defaults" in found[0]


def test_relative_import_is_resolved():
    found = _violations("syncem.lattice.lattice", "from ..planning import defaults\n")
    assert len(found) == 1
    assert "syncem.planning" in found[0]


def test_relative_import_within_a_layer_passes():
    assert (
        _violations(
            "syncem.lattice.lattice", "from .cells import Cells\nfrom . import index\n"
        )
        == []
    )


def test_type_checking_block_is_exempt():
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    from syncem.lattice.moments import Moments\n"
    )
    assert _violations("syncem.values.spectrum", source) == []


def test_else_branch_of_type_checking_is_checked_once():
    source = (
        "from typing import TYPE_CHECKING\n"
        "if TYPE_CHECKING:\n"
        "    pass\n"
        "else:\n"
        "    from syncem.planning.plan import plan\n"
    )
    assert len(_violations("syncem.lattice.lattice", source)) == 1


def test_typing_attribute_type_checking_is_exempt():
    source = "import typing\nif typing.TYPE_CHECKING:\n    from syncem.records.ledger import Ledger\n"
    assert _violations("syncem.values.spectrum", source) == []


def test_lattice_may_not_import_a_kernel():
    found = _violations(
        "syncem.lattice.table", "from syncem.base.kernel import Kernel\n"
    )
    assert len(found) == 1


def test_lattice_may_import_variables():
    assert (
        _violations(
            "syncem.lattice.table", "from syncem.base.variables import Variable\n"
        )
        == []
    )


def test_population_may_not_import_engine_or_base_kernel():
    assert (
        len(
            _violations(
                "syncem.population.pair", "from syncem.engine.direct import direct\n"
            )
        )
        == 1
    )
    assert (
        len(_violations("syncem.population.pair", "import syncem.base.kernel\n")) == 1
    )


def test_planning_may_import_only_the_engine_column_map():
    assert (
        _violations(
            "syncem.planning.plan", "from syncem.engine.columns import ColumnMap\n"
        )
        == []
    )
    assert (
        len(
            _violations(
                "syncem.planning.plan", "from syncem.engine.provider_jet import build\n"
            )
        )
        == 1
    )


def test_existing_code_may_not_import_new_layers():
    found = _violations(
        "syncem.model.basis", "from syncem.lattice.cells import Cells\n"
    )
    assert len(found) == 1


def test_engine_may_call_into_existing_code():
    assert (
        _violations(
            "syncem.engine.provider_core", "from syncem.model import ContinuumKernel\n"
        )
        == []
    )


def test_listed_delegation_passes():
    source = (
        "class SpectralBasis:\n"
        "    def moments(self, population):\n"
        "        from syncem.population.pair import pair\n"
        "        return pair(self, population)\n"
    )
    assert _violations("syncem.engine.basis", source) == []


def test_delegation_to_the_wrong_layer_fails():
    source = (
        "class SpectralBasis:\n"
        "    def moments(self, population):\n"
        "        from syncem.fitting.fit import fit\n"
    )
    assert len(_violations("syncem.engine.basis", source)) == 1


def test_unlisted_function_local_import_fails():
    source = "def helper():\n    from syncem.planning.plan import plan\n"
    found = _violations("syncem.engine.assemble", source)
    assert len(found) == 1
    assert "function-local" in found[0]


def test_unlisted_method_local_import_fails():
    source = "class SpectralBasis:\n    def summary(self):\n        from syncem.fitting.fit import fit\n"
    assert len(_violations("syncem.engine.basis", source)) == 1


def test_function_local_import_of_an_earlier_layer_passes():
    source = "def helper():\n    from syncem.lattice.cells import Cells\n"
    assert _violations("syncem.engine.assemble", source) == []


def test_lattice_delegations_pass():
    source = (
        "class Lattice:\n"
        "    @classmethod\n"
        "    def default(cls, population):\n"
        "        from syncem.planning.defaults import default_lattice\n"
        "    @classmethod\n"
        "    def single(cls, about):\n"
        "        from syncem.population.population import single_lattice\n"
    )
    assert _violations("syncem.lattice.lattice", source) == []


def test_front_door_may_import_anything_lazily_and_at_module_level():
    source = "from syncem.fitting.fit import fit\ndef __getattr__(name):\n    import syncem.planning.plan\n"
    assert _violations("syncem", source) == []


def test_module_outside_the_layers_is_reported():
    found = _violations("syncem.unknown_module", "import numpy\n")
    assert len(found) == 1
    assert "no layer" in found[0]


def test_front_door_function_delegation_passes():
    source = (
        "class SpectralBasis:\n"
        "    def predict(self, channels):\n"
        "        from syncem import predict\n"
        "        from .. import predict as front_predict\n"
    )
    assert _violations("syncem.engine.basis", source) == []


def test_front_door_function_outside_a_delegation_fails():
    found = _violations(
        "syncem.engine.assemble", "def helper():\n    from syncem import fit\n"
    )
    assert len(found) == 1
    assert "front" in found[0]


def test_dynamic_import_of_a_later_layer_fails():
    source = "import importlib\nmod = importlib.import_module('syncem.planning.plan')\n"
    assert len(_violations("syncem.lattice.lattice", source)) == 1
    assert (
        len(_violations("syncem.lattice.lattice", "m = __import__('syncem.fitting')\n"))
        == 1
    )


def test_dynamic_import_with_a_computed_name_fails():
    source = "from importlib import import_module\ndef load(name):\n    return import_module(name)\n"
    found = _violations("syncem.engine.cache", source)
    assert len(found) == 1
    assert "computed" in found[0]


def test_dynamic_import_of_an_earlier_layer_passes():
    source = "import importlib\nmod = importlib.import_module('syncem.lattice.cells')\n"
    assert _violations("syncem.engine.basis", source) == []


def test_type_checking_alias_is_exempt():
    source = (
        "from typing import TYPE_CHECKING as TC\n"
        "import typing as t\n"
        "if TC:\n"
        "    from syncem.lattice.moments import Moments\n"
        "if t.TYPE_CHECKING:\n"
        "    from syncem.records.ledger import Ledger\n"
    )
    assert _violations("syncem.values.spectrum", source) == []


def test_local_name_type_checking_is_not_exempt():
    source = "TYPE_CHECKING = True\nif TYPE_CHECKING:\n    from syncem.planning.plan import plan\n"
    assert len(_violations("syncem.lattice.lattice", source)) == 1


def test_front_door_may_use_computed_dynamic_imports():
    source = "import importlib\ndef __getattr__(name):\n    return importlib.import_module(TABLE[name])\n"
    assert _violations("syncem", source) == []
    assert len(_violations("syncem.engine.cache", source)) == 1
