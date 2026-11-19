"""Step 5b-1 records of design/0.8/SPEC.md (decisions 149 and 150, Sec. 3.4, 9.1).

``SpecifiedRecord``, ``RouteRecord``, ``BuildRecord`` and ``Ledger`` are frozen. Each ``to_dict()`` is strict
JSON: it survives ``json.dumps(..., allow_nan=False)`` and ``json.loads`` unchanged, and every key is a string.
All checks are exact (identities of the record types), so no tolerance appears here.
"""

import dataclasses
import importlib.util
import json
import math
import os
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

import syncem
from syncem.model.errors import ErrorTerm
from syncem.records.ledger import Ledger
from syncem.records.records import (
    FAMILIES,
    PROVIDERS,
    S1_METHODS,
    SIZE_KEYS,
    STACK_MODES,
    BuildRecord,
    RouteRecord,
    SpecifiedRecord,
)

ROOT = Path(__file__).resolve().parents[2]


def _graph():
    spec = importlib.util.spec_from_file_location(
        "_import_graph_records", ROOT / "tests" / "_import_graph.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses of the checker look up their module
    spec.loader.exec_module(module)
    return module


def _keys_are_strings(value):
    if isinstance(value, dict):
        return all(
            isinstance(k, str) and _keys_are_strings(v) for k, v in value.items()
        )
    if isinstance(value, list):
        return all(_keys_are_strings(v) for v in value)
    return True


def _assert_strict_json(d):
    assert isinstance(d, dict)
    assert _keys_are_strings(d)
    assert json.loads(json.dumps(d, allow_nan=False)) == d


def _route(cell=0, provider="stack", mode="cached-S1", s1="lookup", reason=None):
    return RouteRecord(cell=cell, provider=provider, mode=mode, s1=s1, reason=reason)


def _build(**kw):
    base = dict(
        routes=(
            _route(0),
            _route(
                1,
                "jet",
                None,
                None,
                "gamma averaged: Similarity variable not celled -> jet",
            ),
        ),
        stokes=("I", "P"),
        s1="lookup",
        sizes=(("n_rows", 57), ("n", 57), ("n0", 57), ("n2", 57)),
    )
    base.update(kw)
    return BuildRecord(**base)


# RouteRecord -------------------------------------------------------------------------------------------------


def test_route_providers_are_stack_jet_core():
    assert PROVIDERS == ("stack", "jet", "core")
    assert STACK_MODES == ("cached-S1", "per-node", "field-scaling")
    assert S1_METHODS == ("lookup", "quadrature")
    for provider in PROVIDERS:
        mode = "per-node" if provider == "stack" else None
        assert _route(provider=provider, mode=mode).provider == provider


@pytest.mark.parametrize(
    "kw",
    [
        {"provider": "table"},
        {"provider": "Stack"},
        {"cell": -1},
        {"cell": True},
        {"cell": 1.0},
        {"mode": "lookup"},
        {"mode": 3},
        {"s1": "cached"},
        {"reason": 1},
    ],
)
def test_route_fields_are_validated(kw):
    with pytest.raises(ValueError):
        _route(**kw)


def test_route_numpy_cell_becomes_int_and_record_is_frozen_and_hashable():
    route = _route(cell=np.int64(3))
    assert type(route.cell) is int and route.cell == 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        route.provider = "jet"
    assert hash(route) == hash(_route(cell=3))


def test_route_to_dict_is_strict_json():
    route = _route(1, "jet", None, None, "the stack refused: grid reach")
    d = route.to_dict()
    _assert_strict_json(d)
    assert d == {
        "cell": 1,
        "provider": "jet",
        "mode": None,
        "s1": None,
        "reason": "the stack refused: grid reach",
    }


# BuildRecord -------------------------------------------------------------------------------------------------


def test_build_record_has_the_plan_fields():
    names = tuple(f.name for f in dataclasses.fields(BuildRecord))
    assert names == (
        "routes",
        "stokes",
        "s1",
        "sizes",
        "events",
        "timings",
        "load",
        "versions",
        "keys",
        "defaults",
        "agreement",
        "notes",
    )
    assert SIZE_KEYS == ("n_rows", "n", "n0", "n2")


def test_build_record_holds_the_load_average_of_construction():
    rec = _build()
    if hasattr(os, "getloadavg"):
        assert len(rec.load) == 3
        assert all(isinstance(v, float) and v >= 0 for v in rec.load)
    else:
        assert rec.load == ()
    assert _build(load=(1.0, 2.0, 3.0)).load == (1.0, 2.0, 3.0)
    with pytest.raises(ValueError):
        _build(load=(1.0, -2.0, 3.0))
    with pytest.raises(ValueError):
        _build(load=(1.0, 2.0))


def test_build_record_versions_name_syncem():
    versions = dict(_build().versions)
    assert versions["syncem"] == syncem.__version__
    assert {"jax", "numpy", "equinox"} <= set(versions)


def test_build_record_to_dict_is_strict_json():
    rec = _build(
        events=(("provider calls", np.int64(2)), ("jet compiles", 1)),
        timings=(("build s", np.float64(0.25)), ("routes s", math.nan)),
        keys=(("column map", None), ("index", "abc")),
        defaults=(("mem_bytes", 1e9), ("cache directory", None)),
        notes=("not cached: no fingerprints before 5b-2",),
    )
    d = rec.to_dict()
    _assert_strict_json(d)
    assert d["sizes"] == {"n_rows": 57, "n": 57, "n0": 57, "n2": 57}
    assert d["timings"] == {"build s": 0.25, "routes s": None}
    assert d["events"] == {"provider calls": 2, "jet compiles": 1}
    assert d["routes"][1]["provider"] == "jet"
    assert d["agreement"] == {}
    assert d["notes"] == ["not cached: no fingerprints before 5b-2"]
    assert d["load"] == [None if not math.isfinite(v) else v for v in rec.load]


def test_build_record_is_frozen_and_hashable():
    rec = _build()
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.s1 = "quadrature"
    assert hash(rec) == hash(dataclasses.replace(rec))


def test_build_record_pairs_accept_a_mapping_in_order():
    rec = _build(events={"provider calls": 2, "jet compiles": 1})
    assert rec.events == (("provider calls", 2), ("jet compiles", 1))


@pytest.mark.parametrize(
    "kw",
    [
        {"routes": ({"cell": 0},)},
        {"routes": [_route(0)]},
        {"stokes": ("I", "X")},
        {"stokes": ("I", "I")},
        {"s1": "cached"},
        {"sizes": (("n_rows", 57), ("n", 57), ("n0", 57))},
        {"sizes": (("n_rows", 57), ("n", -1), ("n0", 57), ("n2", 57))},
        {"sizes": (("n_rows", 57.0), ("n", 57), ("n0", 57), ("n2", 57))},
        {"events": ((1, 2),)},
        {"events": (("calls", np.zeros(2)),)},
        {"events": (("calls", 1), ("calls", 2))},
        {"keys": (("index", ["a"]),)},
        {"notes": ("ok", 3)},
    ],
)
def test_build_record_fields_are_validated(kw):
    with pytest.raises(ValueError):
        _build(**kw)


def test_build_record_with_notes_returns_a_new_record():
    rec = _build(notes=("a",))
    new = rec.with_notes("routes of the order-4 build")
    assert new.notes == ("a", "routes of the order-4 build")
    assert rec.notes == ("a",)
    with pytest.raises(ValueError):
        rec.with_notes(1)


# SpecifiedRecord ---------------------------------------------------------------------------------------------


def test_specified_record_static_fields_and_empty_params_leaf():
    rec = SpecifiedRecord("B", "NodeRule", parents=["eta"], fingerprint=None)
    assert FAMILIES == ("NodeRule", "callable", "fixed")
    assert rec.parents == ("eta",)
    assert rec.params.shape == (0,)
    leaves = jax.tree_util.tree_leaves(rec)
    assert len(leaves) == 1 and leaves[0] is rec.params
    doubled = eqx.filter_jit(lambda r: r.params * 2.0)(
        SpecifiedRecord("B", "callable", params=jnp.array([1.0, 2.0]))
    )
    assert np.array_equal(np.asarray(doubled), [2.0, 4.0])


def test_specified_record_is_frozen():
    rec = SpecifiedRecord("B", "fixed")
    with pytest.raises(dataclasses.FrozenInstanceError):
        rec.variable = "eta"


@pytest.mark.parametrize(
    ("args", "kw"),
    [
        (("B", "rule"), {}),
        ((1, "fixed"), {}),
        (("B", "fixed"), {"parents": ("eta", 2)}),
        (("B", "fixed"), {"parents": "eta"}),
        (("B", "fixed"), {"fingerprint": 5}),
        (("B", "fixed"), {"params": jnp.zeros((2, 2))}),
        (("B", "fixed"), {"params": jnp.array([1.0 + 1.0j])}),
    ],
)
def test_specified_record_fields_are_validated(args, kw):
    with pytest.raises(ValueError):
        SpecifiedRecord(*args, **kw)


def test_specified_record_to_dict_is_strict_json():
    rec = SpecifiedRecord(
        "B",
        "NodeRule",
        parents=("eta",),
        fingerprint="ab12",
        params=np.array([0.5, np.nan]),
    )
    d = rec.to_dict()
    _assert_strict_json(d)
    assert d == {
        "variable": "B",
        "family": "NodeRule",
        "parents": ["eta"],
        "fingerprint": "ab12",
        "params": [0.5, None],
    }


# Ledger ------------------------------------------------------------------------------------------------------


def test_empty_ledger_round_trips():
    ledger = Ledger()
    assert ledger.terms == ()
    d = ledger.to_dict()
    _assert_strict_json(d)
    assert d == {"terms": []}
    assert jax.tree_util.tree_leaves(ledger) == []


def test_ledger_is_frozen_and_checks_its_terms():
    ledger = Ledger()
    with pytest.raises(dataclasses.FrozenInstanceError):
        ledger.terms = ()
    with pytest.raises(ValueError):
        Ledger(terms=[])
    with pytest.raises(ValueError):
        Ledger(terms=("bound",))


def test_ledger_with_a_term_is_strict_json():
    term = ErrorTerm(jnp.array([[1e-9, 0.0, 0.0, 0.0]]), "measured", note="probe")
    d = Ledger(terms=(term, ErrorTerm.unbounded(note="remainder"))).to_dict()
    _assert_strict_json(d)
    assert [t["kind"] for t in d["terms"]] == ["measured", "unbounded"]


# Layers and docstrings ---------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("relpath", "module", "is_package"),
    [
        ("syncem/records/__init__.py", "syncem.records", True),
        ("syncem/records/records.py", "syncem.records.records", False),
        ("syncem/records/ledger.py", "syncem.records.ledger", False),
    ],
)
def test_records_modules_follow_the_layer_order(relpath, module, is_package):
    graph = _graph()
    assert graph.layer_of(module) == "records"
    source = (ROOT / relpath).read_text()
    assert graph.check_source(module, source, is_package=is_package) == []
