"""Step 5b-2 of design/0.8/SPEC.md: the S2 context, memo and design keys of decision 159 (``engine/_keys.py``).

Each key changes with each of its named components, one test case per component with the others held fixed,
and with nothing else. ``x_min`` enters no key, a permuted channel tuple changes the memo key and permutes
the S2 context keys, moments and ``w`` enter no memo key, and a traced azimuth factor gives no design key.
The source digests, records and S0 and S1 keys are in ``test_v080_keys.py``. Every assertion is an exact
identity of strings, so no tolerance enters.
"""

from __future__ import annotations

import ast
import inspect

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.engine import _keys as keys
from syncem.engine.columns import ColumnRules, column_map
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import Legendre
from tests._import_graph import check_source
from tests.v080._column_cases import LATTICES, energy

K = ContinuumKernel()
S1 = "1" * 64
ROW = tuple(f"{j:064x}" for j in range(1, 4))
PHASE = tuple(f"{j:064x}" for j in range(11, 14))


def _changes(fn, base, cases, name):
    """The key of ``base`` with ``name`` set to ``cases[name]`` differs from the key of ``base``."""
    assert fn(**{**base, name: cases[name]}) != fn(**base)


# ------------------------------------------------------------------ S2 context keys

S2_BASE = {
    "provider": "jet",
    "mode": "core",
    "stage": S1,
    "block": 0,
    "row_keys": ROW,
    "phase": PHASE,
    "averaged": (("averaged", ()),),
    "d0": {"B": "2" * 64, "eta": "3" * 64},
    "geometry": "4" * 64,
    "rules": b"rules",
}
S2_CASES = {
    "provider": "stack",
    "mode": "per-node",
    "stage": (("name", "continuum"),),
    "block": 1,
    "averaged": (("averaged", (("B", "NodeRule"),)),),
    "d0": {"B": "2" * 64, "eta": "5" * 64},
    "geometry": "6" * 64,
    "rules": b"other rules",
}


@pytest.mark.parametrize("name", sorted(S2_CASES))
def test_s2_context_keys_change_with_each_shared_component(name):
    base = keys.s2_context_keys(**S2_BASE)
    other = keys.s2_context_keys(**{**S2_BASE, name: S2_CASES[name]})
    assert np.all(other != base)


@pytest.mark.parametrize("name", ["row_keys", "phase"])
def test_s2_context_keys_change_only_in_the_channel_that_changes(name):
    base = keys.s2_context_keys(**S2_BASE)
    edited = list(S2_BASE[name])
    edited[1] = "e" * 64
    other = keys.s2_context_keys(**{**S2_BASE, name: tuple(edited)})
    assert list(other != base) == [False, True, False]


def test_s2_context_keys_follow_each_channel_and_nothing_else():
    base = keys.s2_context_keys(**S2_BASE)
    assert base.shape == (3,) and base.dtype.kind == "U" and len(set(base)) == 3
    order = [2, 0, 1]
    permuted = {
        **S2_BASE,
        "row_keys": tuple(ROW[j] for j in order),
        "phase": tuple(PHASE[j] for j in order),
    }
    assert list(keys.s2_context_keys(**permuted)) == list(base[order])
    subset = {**S2_BASE, "row_keys": ROW[1:], "phase": PHASE[1:]}
    assert list(keys.s2_context_keys(**subset)) == list(base[1:])
    reordered_d0 = {
        **S2_BASE,
        "d0": {"eta": "3" * 64, "B": "2" * 64},
        "block": np.int64(0),
    }
    assert list(keys.s2_context_keys(**reordered_d0)) == list(base)


@pytest.mark.parametrize(
    "name,value",
    [
        ("stage", None),
        ("d0", {"B": None, "eta": "3" * 64}),
        ("phase", None),
        ("row_keys", (ROW[0], None, ROW[2])),
    ],
)
def test_s2_context_keys_are_none_without_a_component(name, value):
    assert keys.s2_context_keys(**{**S2_BASE, name: value}) is None


def test_s2_context_keys_refuse_unequal_channel_counts():
    with pytest.raises(ValueError):
        keys.s2_context_keys(**{**S2_BASE, "phase": PHASE[:2]})


def test_x_min_enters_no_s2_context_key():
    ch = Channels.log(0.1, 10, 5, unit="GHz", half_width=0.2, n_nu=16)
    resolved = keys.kernel_record(ContinuumKernel().resolved(ch, 1e5, 1e-4))
    plain = keys.kernel_record(ContinuumKernel())
    assert list(keys.s2_context_keys(**{**S2_BASE, "stage": resolved})) == list(
        keys.s2_context_keys(**{**S2_BASE, "stage": plain})
    )


# ------------------------------------------------------------------ memo and design keys

MEMO_BASE = {
    "column_map": "7" * 64,
    "kernel": (("name", "continuum"),),
    "averaged": (("averaged", ()),),
    "phase": PHASE,
    "rules": {"B": "8" * 64},
    "row_keys": ROW,
    "provider": {"routes": "auto"},
    "engine": "9" * 64,
}
MEMO_CASES = {
    "column_map": "a" * 64,
    "kernel": (("name", "user"),),
    "averaged": (("averaged", ("B",)),),
    "phase": (PHASE[0], PHASE[1], "e" * 64),
    "rules": {"B": "b" * 64},
    "row_keys": (ROW[0], ROW[1], "e" * 64),
    "provider": {"routes": "jet"},
    "engine": "c" * 64,
}


@pytest.mark.parametrize("name", sorted(MEMO_CASES))
def test_memo_key_changes_with_each_component(name):
    _changes(keys.memo_key, MEMO_BASE, MEMO_CASES, name)


def test_memo_key_keeps_the_channel_order():
    base = keys.memo_key(**MEMO_BASE)
    assert keys.memo_key(**{**MEMO_BASE, "row_keys": ROW[::-1]}) != base
    assert (
        keys.memo_key(**{**MEMO_BASE, "row_keys": ROW[::-1], "phase": PHASE[::-1]})
        != base
    )
    assert (
        keys.memo_key(**{**MEMO_BASE, "row_keys": list(ROW), "phase": np.array(PHASE)})
        == base
    )


def test_memo_key_holds_no_moments_and_no_weights():
    assert set(inspect.signature(keys.memo_key).parameters) == set(MEMO_BASE)
    lat = LATTICES["E4-KP"]()
    reweighted = Lattice(gamma=energy(), mu=Legendre(8), weights=np.full(19, 3.0))
    assert reweighted.key != lat.key
    rules = ColumnRules(masks=(("mu", "single"),))
    a = keys.memo_key(**{**MEMO_BASE, "column_map": column_map(lat, K, rules).key})
    b = keys.memo_key(
        **{**MEMO_BASE, "column_map": column_map(reweighted, K, rules).key}
    )
    assert a == b


@pytest.mark.parametrize(
    "name,value",
    [
        ("column_map", None),
        ("kernel", None),
        ("rules", {"B": None}),
        ("phase", None),
        ("row_keys", (ROW[0], None, ROW[2])),
    ],
)
def test_memo_key_is_none_without_a_component(name, value):
    assert keys.memo_key(**{**MEMO_BASE, name: value}) is None


def test_design_key_changes_with_memo_lattice_and_factor():
    base = keys.design_key("1" * 64, "2" * 64, 0.4)
    assert keys.design_key("3" * 64, "2" * 64, 0.4) != base
    assert keys.design_key("1" * 64, "3" * 64, 0.4) != base
    assert keys.design_key("1" * 64, "2" * 64, 0.5) != base
    assert keys.design_key("1" * 64, "2" * 64, None) != base
    assert keys.design_key("1" * 64, "2" * 64, np.float64(0.4)) == base
    assert keys.design_key(
        "1" * 64, "2" * 64, jnp.asarray([0.4, 0.5])
    ) == keys.design_key("1" * 64, "2" * 64, np.array([0.4, 0.5]))
    assert keys.design_key(None, "2" * 64, 0.4) is None
    assert keys.design_key("1" * 64, None, 0.4) is None


def test_design_key_is_none_for_a_traced_factor():
    seen = []

    @jax.jit
    def run(factor):
        seen.append(keys.design_key("1" * 64, "2" * 64, factor))
        return factor

    run(jnp.asarray(0.4))
    assert seen == [None]


# ------------------------------------------------------------------ the module contract


def test_keys_module_uses_no_repr_hash_or_id_and_keeps_the_import_graph():
    path = inspect.getsourcefile(keys)
    source = open(path, encoding="utf-8").read()
    called = {
        n.func.id
        for n in ast.walk(ast.parse(source))
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
    }
    assert not called & {"repr", "hash", "id"}
    assert check_source("syncem.engine._keys", source) == []
    assert len(source.splitlines()) < 400


def test_s2_and_design_keys_take_their_named_components_only():
    assert set(inspect.signature(keys.s2_context_keys).parameters) == {*S2_BASE}
    assert set(inspect.signature(keys.design_key).parameters) == {
        "memo",
        "lattice",
        "factor",
    }


def test_x_min_enters_no_memo_key():
    ch = Channels.log(0.1, 10, 5, unit="GHz", half_width=0.2, n_nu=16)
    resolved = keys.kernel_record(ContinuumKernel().resolved(ch, 1e5, 1e-4))
    assert keys.memo_key(**{**MEMO_BASE, "kernel": resolved}) == keys.memo_key(
        **{**MEMO_BASE, "kernel": keys.kernel_record(ContinuumKernel())}
    )
