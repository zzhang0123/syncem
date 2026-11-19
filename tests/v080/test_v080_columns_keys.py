"""Step 5b-1 of design/0.8/SPEC.md: column-map keys, refusals and the module contract (decisions 146 to 148, 155).

The column capability record and ``ColumnMap.key`` (decision 146), the literal depth name of decision 155,
the validation of ``ColumnRules`` and the refusals of ``column_map``, and the import contract of the column
modules. The brute-force map comes from ``tests/v080/_column_cases.py``.
"""

from __future__ import annotations

import ast
import importlib.util
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.capabilities import EnergyWindow, Parity
from syncem.base.continuum import ContinuumKernel
from syncem.base.user import kernel
from syncem.base.variables import SYNCHROTRON_VARIABLES as SV
from syncem.base.variables import Variable
from syncem.engine.columns import ColumnRules, column_capability_record, column_map
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import AzimuthHarmonic, Legendre
from tests.v080._column_cases import LATTICES, ROOT, assert_matches_brute, energy

# ------------------------------------------------------------------ 7. the column capability record and the key


def user_kernel(window, parity=Parity(("I", "P"), ("V",))):
    @kernel(
        variables=[Variable("a", kind="positive"), SV["mu"], SV["eta"]],
        stokes=("I", "P"),
        capabilities=(parity, EnergyWindow(window)),
    )
    def keyed(nu, a, mu, eta):
        return {"I": a * jnp.ones_like(nu)}

    return keyed


def _window_a(*args, **kwargs):
    return (1.0, 2.0)


def _window_b(*args, **kwargs):
    return (3.0, 4.0)


def test_column_capability_record_holds_data_only():
    """Decision 146: Parity, Beaming, depth, azimuth and Stokes. EnergyWindow holds a function and is left out."""
    assert column_capability_record(ContinuumKernel()) == column_capability_record(
        ContinuumKernel(x_min=1e-9, n_nodes_F=64)
    )
    assert column_capability_record(user_kernel(_window_a)) == column_capability_record(
        user_kernel(_window_b)
    )
    assert isinstance(column_capability_record(ContinuumKernel()), bytes)
    assert column_capability_record(user_kernel(_window_a)) != column_capability_record(
        user_kernel(_window_a, Parity(("I",), ("P", "V")))
    )


def test_column_map_key():
    lat = LATTICES["E4-KP"]()
    rules = ColumnRules(masks=(("mu", "single"),))
    key = column_map(lat, ContinuumKernel(), rules).key
    assert isinstance(key, str) and len(key) == 64
    assert column_map(lat, ContinuumKernel(x_min=1e-9), rules).key == key
    assert column_map(lat, ContinuumKernel(), ColumnRules()).key != key
    assert column_map(lat, ContinuumKernel(), rules).key == key
    reweighted = Lattice(gamma=energy(), mu=Legendre(8), weights=np.full(19, 3.0))
    assert reweighted.key != lat.key
    assert column_map(reweighted, ContinuumKernel(), rules).key == key
    order4 = Lattice(gamma=energy().with_order(4), mu=Legendre(8))
    assert column_map(order4, ContinuumKernel(), rules).key != key
    stokes_i = ColumnRules(stokes=("I",), masks=rules.masks)
    assert column_map(lat, ContinuumKernel(), stokes_i).key != key
    explicit = ColumnRules(stokes=("P", "I"), masks=rules.masks)
    assert column_map(lat, ContinuumKernel(), explicit).key == key
    assert column_map(lat.with_key(None), ContinuumKernel(), rules).key is None


def test_user_kernel_keys_ignore_the_window_and_follow_parity():
    lat = Lattice(a=Cells.log(1.0, 10.0, 2), mu=Legendre(4))
    rules = ColumnRules(masks=(("mu", "single"),))
    a = column_map(lat, user_kernel(_window_a), rules)
    b = column_map(lat, user_kernel(_window_b), rules)
    c = column_map(lat, user_kernel(_window_a, Parity(("I",), ("P", "V"))), rules)
    assert a.key == b.key != c.key
    assert a.n == 2 * 3 * 3 and c.n == 2 * 3 * 5


def test_column_maps_compare_by_key():
    lat = LATTICES["E1"]()
    assert column_map(lat, ContinuumKernel()) == column_map(lat, ContinuumKernel())
    assert hash(column_map(lat, ContinuumKernel())) == hash(
        column_map(lat, ContinuumKernel())
    )
    assert column_map(lat, ContinuumKernel()) != column_map(
        lat, ContinuumKernel(), ColumnRules(stokes=("I",))
    )


# ------------------------------------------------------------------ 8. literal depth name in the lattice (decision 155)


def _depth_kernel(name):
    @kernel(
        variables=[Variable("a", kind="positive"), Variable(name, unit="rad m^-2")],
        stokes=("I", "P"),
        depth=name,
    )
    def depthy(nu, a):
        return {"I": a * jnp.ones_like(nu), "P": 0.3 * a * jnp.ones_like(nu)}

    return depthy


def test_depth_variable_with_another_name_pins_step_5_behaviour():
    """Decision 155: the lattice excludes the literal "depth" from the total, the map reads ``kernel.depth``.

    With the depth variable named "rm" the lattice keeps it inside the total degree (6 monomials per cell),
    while the column map treats it as the depth. Step 6 takes the excluded set from the kernel and changes
    the "rm" counts to those of "depth". This test pins the step-5 counts and is the case step 6 changes.
    """
    rules = ColumnRules(depth_free_spin0=True)
    counts = {}
    for name in ("rm", "depth"):
        lat = Lattice(
            a=Cells.log(1.0, 10.0, 2), **{name: Cells.linear(-1, 1, 3)}, order=2
        )
        cm = column_map(lat, _depth_kernel(name), rules)
        counts[name] = (lat.n, cm.n, len(cm.uses["I"]), cm.n0, cm.n2)
        assert_matches_brute(cm, lat, _depth_kernel(name), rules)
    assert counts["rm"] == (36, 36, 18, 6, 36)
    assert counts["depth"] == (54, 54, 18, 6, 54)


# ------------------------------------------------------------------ refusals


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (dict(stokes=("Q",)), "Stokes"),
        (dict(stokes=("I", "I")), "Stokes"),
        (dict(stokes=()), "Stokes"),
        (dict(masks=(("mu", "half"),)), "joint"),
        (dict(masks=(("mu", "single"), ("mu", "single"))), "mask"),
        (dict(masks=(("mu", "single"), ("eta", "joint"))), "mask"),
        (dict(masks=(("mu", "joint"),)), "joint"),
        (dict(merge=("mu", "mu")), "merge"),
        (dict(depth_free_spin0=1), "depth_free_spin0"),
        (dict(symmetries=("time_reversal",)), "symmetr"),
    ],
)
def test_rules_validate(kwargs, match):
    with pytest.raises(ValueError, match=match):
        ColumnRules(**kwargs)


def test_rules_record_is_bytes_and_distinguishes_rules():
    a, b = ColumnRules(), ColumnRules(depth_free_spin0=True)
    assert isinstance(a.record(), bytes) and a.record() != b.record()
    assert ColumnRules(masks=[["mu", "single"]]) == ColumnRules(
        masks=(("mu", "single"),)
    )


def test_refusals_name_the_cause():
    ck = ContinuumKernel()
    with pytest.raises(ValueError, match=r"\('I', 'P'\)"):
        column_map(LATTICES["E1"](), ck, ColumnRules(stokes=("I", "V")))
    with pytest.raises(ValueError, match="Legendre"):
        column_map(LATTICES["E1"](), ck, ColumnRules(masks=(("mu", "single"),)))
    with pytest.raises(ValueError, match="Parity"):
        column_map(
            Lattice(a=Cells.log(1.0, 10.0, 2), mu=Legendre(2)),
            _no_parity_kernel(),
            ColumnRules(masks=(("mu", "single"),)),
        )
    both = Lattice(gamma=Cells.log(300, 3e5, 2), mu=Legendre(2), eta=Legendre(2))
    with pytest.raises(ValueError, match="joint"):
        column_map(both, ck, ColumnRules(masks=(("mu", "single"),)))
    with pytest.raises(ValueError, match="Beaming"):
        column_map(both, ck, ColumnRules(merge=("eta", "mu")))
    with pytest.raises(ValueError, match="degree"):
        unequal = Lattice(gamma=Cells.log(300, 3e5, 2), mu=Legendre(2), eta=Legendre(4))
        column_map(unequal, ck, ColumnRules(merge=("mu", "eta")))
    with pytest.raises(ValueError, match="depth"):
        column_map(
            Lattice(a=Cells.log(1.0, 10.0, 2), mu=Legendre(2)),
            user_kernel(_window_a),
            ColumnRules(depth_free_spin0=True),
        )
    with pytest.raises(ValueError, match="variables of the kernel"):
        column_map(Lattice(x=Cells.log(1.0, 10.0, 2)), ck)
    with pytest.raises(ValueError, match="azimuth"):
        column_map(
            Lattice(a=Cells.log(1.0, 10.0, 2), phi=AzimuthHarmonic()),
            _no_parity_kernel(),
        )
    with pytest.raises(ValueError, match="Beaming"):
        column_map(
            Lattice(a=Cells.log(1.0, 10.0, 2), mu=Legendre(2)),
            user_kernel(_window_a),
            ColumnRules(symmetries=("pitch_reversal",)),
        )


def _no_parity_kernel():
    @kernel(
        variables=[Variable("a", kind="positive"), SV["mu"], SV["phi"]],
        stokes=("I", "P"),
    )
    def plain(nu, a, mu, phi):
        return {"I": a * jnp.ones_like(nu)}

    return plain


# ------------------------------------------------------------------ module contract


def _graph():
    spec = importlib.util.spec_from_file_location(
        "_import_graph_columns", ROOT / "tests" / "_import_graph.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


COLUMN_MODULES = ("columns", "_column_passes")


@pytest.mark.parametrize("stem", COLUMN_MODULES)
def test_column_modules_import_no_jax_at_module_level(stem):
    """Planning imports ``engine.columns`` (``tests/_import_graph.py``), so it names no JAX at module level."""
    path = ROOT / "syncem" / "engine" / f"{stem}.py"
    source = path.read_text()
    assert len(source.splitlines()) <= 400
    names = []
    for node in ast.parse(source).body:
        if isinstance(node, ast.Import):
            names += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom) and node.level == 0:
            names.append(node.module)
    assert not [n for n in names if n.split(".")[0] in ("jax", "equinox", "scipy")]
    assert _graph().check_source(f"syncem.engine.{stem}", source) == []


def test_columns_docstring_contract():
    import syncem.engine.columns as columns

    doc = columns.__doc__
    for needle in ("LABEL", "Units", "shape", "Not certified"):
        assert needle in doc
