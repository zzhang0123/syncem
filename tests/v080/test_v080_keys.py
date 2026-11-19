"""Step 5b-2 of design/0.8/SPEC.md: source digests, records and S0 and S1 keys of decision 159 (``engine/_keys.py``).

Each key changes with each of its named components, one test case per component with the others held fixed,
and with nothing else: equal values written in other Python or NumPy types give the same key. ``x_min``
enters no kernel record, and a source digest changes with one byte of a module file. The S2 context, memo
and design keys are in ``test_v080_keys_basis.py``. Every assertion is an exact identity of strings, so no
tolerance enters.
"""

from __future__ import annotations

import importlib
import inspect
import sys

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.user import kernel
from syncem.base.variables import SYNCHROTRON_VARIABLES as SV
from syncem.engine import _keys as keys
from syncem.engine._disk import FORMAT
from syncem.engine.average import AveragedKernel, NodeRule, fixed_rule, uniform_rule


def _changes(fn, base, cases, name):
    """The key of ``base`` with ``name`` set to ``cases[name]`` differs from the key of ``base``."""
    assert fn(**{**base, name: cases[name]}) != fn(**base)


# ------------------------------------------------------------------ source digests and the format tag


def test_format_tag_is_the_disk_format():
    assert keys.FMT == FORMAT


def _temporary_module(tmp_path, monkeypatch, text):
    path = tmp_path / "keys_probe_module.py"
    path.write_text(text)
    monkeypatch.syspath_prepend(str(tmp_path))
    sys.modules.pop("keys_probe_module", None)
    return path, importlib.import_module("keys_probe_module")


def test_source_digest_changes_with_one_source_byte(tmp_path, monkeypatch):
    path, module = _temporary_module(tmp_path, monkeypatch, "X = 1\n")
    first = keys.source_digest("keys_probe_module")
    assert keys.source_digest(module) == first
    assert keys.source_digest("keys_probe_module") == first
    path.write_text("X = 2\n")
    assert keys.source_digest("keys_probe_module") != first
    path.write_text("X = 1\n")
    assert keys.source_digest("keys_probe_module") == first


def test_source_digest_names_its_modules_and_ignores_their_order():
    a, b = "syncem.engine.primitive", "syncem.engine.field_average"
    assert (
        keys.source_digest(a, b)
        == keys.source_digest(b, a)
        == keys.source_digest(a, b, a)
    )
    assert keys.source_digest(a) != keys.source_digest(a, b)
    assert len(keys.source_digest(a)) == 64


@pytest.mark.parametrize("bad", ["sys", "syncem.no_such_module", ()])
def test_source_digest_refuses_modules_without_a_source_file(bad):
    with pytest.raises(ValueError):
        keys.source_digest(*((bad,) if isinstance(bad, str) else bad))


def test_engine_source_covers_the_engine_modules():
    names = keys.engine_modules()
    assert {
        "syncem.engine.primitive",
        "syncem.engine.field_average",
        "syncem.engine._keys",
    } <= set(names)
    assert all(n.startswith("syncem.engine") for n in names)
    assert names[0] == "syncem.engine" and len(set(names)) == len(names)
    assert keys.ENGINE_SOURCE == keys.source_digest(*names)
    assert keys.S0_SOURCE == keys.source_digest(*keys.S0_SOURCES)
    assert keys.S1_SOURCE == keys.source_digest(*keys.S1_SOURCES)


# ------------------------------------------------------------------ the kernel record


def test_x_min_enters_no_kernel_record():
    ch = Channels.log(0.1, 10, 5, unit="GHz", half_width=0.2, n_nu=16)
    base = keys.kernel_record(ContinuumKernel())
    assert all(name != "x_min" for name, _ in base)
    assert keys.kernel_record(ContinuumKernel(x_min=1e-9)) == base
    assert keys.kernel_record(ContinuumKernel().resolved(ch, 1e5, 1e-4)) == base
    assert keys.kernel_record(ContinuumKernel(n_nodes_F=64)) != base
    assert keys.kernel_record(ContinuumKernel(gamma_min=2.0)) != base


class _Guarded:
    guards = ("tolerance",)

    def __init__(self, scale, tolerance):
        self.scale, self.tolerance = scale, tolerance

    def describe(self):
        return (
            ("name", "guarded"),
            ("scale", self.scale),
            ("tolerance", self.tolerance),
        )


def test_kernel_record_drops_the_guards_of_any_kernel():
    assert keys.kernel_record(_Guarded(1.0, 1e-3)) == keys.kernel_record(
        _Guarded(1.0, 1e-9)
    )
    assert keys.kernel_record(_Guarded(2.0, 1e-3)) != keys.kernel_record(
        _Guarded(1.0, 1e-3)
    )
    assert keys.kernel_record(_Guarded(1.0, 1e-3)) == (
        ("class", f"{__name__}._Guarded"),
        ("name", "guarded"),
        ("scale", 1.0),
    )


class _Opaque:
    pass


_OPAQUE = _Opaque()


def _user(scale):
    @kernel(variables=[SV["gamma"], SV["mu"]], stokes=("I",))
    def k(nu, gamma, mu):
        return {"I": scale * gamma * jnp.ones_like(nu)}

    return k


@kernel(variables=[SV["gamma"], SV["mu"]], stokes=("I",))
def _opaque_kernel(nu, gamma, mu):
    return {"I": float(_OPAQUE is not None) * jnp.ones_like(nu)}


def test_user_kernel_record_follows_the_function_and_is_none_without_a_fingerprint():
    assert keys.kernel_record(_user(1.0)) == keys.kernel_record(_user(1.0))
    assert keys.kernel_record(_user(2.0)) != keys.kernel_record(_user(1.0))
    assert all(name != "not_cached" for name, _ in keys.kernel_record(_user(1.0)))
    assert keys.kernel_record(_opaque_kernel) is None


# ------------------------------------------------------------------ the averaged-kernel record

K = ContinuumKernel()


def _averaged(**change):
    args = {
        "averaged": [
            ("B", fixed_rule([1e-6, 2e-6], [0.5, 0.5])),
            ("eta", uniform_rule(0.0, 0.5, 4)),
        ],
        "legendre": {"mu": (4, 8)},
        "fixed": {"phi": 0.25},
        "outer_batch": 1,
    }
    return AveragedKernel(K, **{**args, **change})


AVERAGED_CASES = {
    "names": {
        "averaged": [
            ("B", fixed_rule([1e-6], [1.0])),
            ("depth", uniform_rule(0.0, 0.5, 4)),
        ]
    },
    "family": {
        "averaged": [
            ("B", lambda point: (jnp.ones(1), jnp.ones(1))),
            ("eta", uniform_rule(0.0, 0.5, 4)),
        ]
    },
    "fixed value": {"fixed": {"phi": 0.5}},
    "fixed name": {"fixed": {"depth": 0.25}},
    "legendre degree": {"legendre": {"mu": (6, 8)}},
    "legendre nodes": {"legendre": {"mu": (4, 12)}},
    "outer batch": {"outer_batch": None},
}


@pytest.mark.parametrize("case", sorted(AVERAGED_CASES))
def test_averaged_record_changes_with_each_component(case):
    assert keys.averaged_record(
        _averaged(**AVERAGED_CASES[case])
    ) != keys.averaged_record(_averaged())


def test_averaged_record_holds_no_rule_function_and_canonical_fixed_values():
    base = keys.averaged_record(_averaged())
    other_law = [
        ("B", fixed_rule([3e-6], [1.0])),
        ("eta", NodeRule(lambda p: (jnp.zeros(2), jnp.ones(2)))),
    ]
    assert keys.averaged_record(_averaged(averaged=other_law)) == base
    assert keys.averaged_record(_averaged(fixed={"phi": np.float32(0.25)})) == base
    assert keys.averaged_record(_averaged(fixed={"phi": 0.0})) == keys.averaged_record(
        _averaged(fixed={"phi": -0.0})
    )
    assert dict(base)["averaged"] == (("B", "NodeRule"), ("eta", "NodeRule"))


# ------------------------------------------------------------------ S0 and S1 keys

S0_BASE = {
    "span": (-30.0, 20.0),
    "order": 6,
    "step": 0.0025,
    "fmt": 1,
    "source": "a" * 64,
    "environment": (("python", "3.12"),),
}
S0_CASES = {
    "span": (-40.0, 20.0),
    "order": 4,
    "step": 0.005,
    "fmt": 2,
    "source": "b" * 64,
    "environment": (("python", "3.13"),),
}


@pytest.mark.parametrize("name", sorted(S0_CASES))
def test_s0_key_changes_with_each_component(name):
    _changes(keys.s0_key, S0_BASE, S0_CASES, name)


def test_s0_key_changes_with_the_upper_end_and_nothing_else():
    base = keys.s0_key(**S0_BASE)
    assert keys.s0_key(**{**S0_BASE, "span": (-30.0, 30.0)}) != base
    same = {
        **S0_BASE,
        "span": np.array([-30, 20]),
        "order": np.int64(6),
        "step": np.float64(0.0025),
    }
    assert keys.s0_key(**same) == base
    assert keys.s0_key((-30.0, 20.0), 6, 0.0025) == keys.s0_key(
        (-30.0, 20.0), 6, 0.0025
    )
    default = keys.s0_key((-30.0, 20.0), 6, 0.0025)
    assert default == keys.s0_key(
        (-30.0, 20.0),
        6,
        0.0025,
        fmt=keys.FMT,
        source=keys.S0_SOURCE,
        environment=keys.environment(),
    )
    assert (
        dict(keys.ENVIRONMENT)["python"]
        == f"{sys.version_info[0]}.{sys.version_info[1]}"
    )


@pytest.mark.parametrize(
    "span", [(20.0, -30.0), (0.0, 0.0), (-np.inf, 20.0), (np.nan, 1.0), (1.0, 2.0, 3.0)]
)
def test_s0_key_refuses_a_bad_span(span):
    with pytest.raises(ValueError):
        keys.s0_key(span, 6, 0.0025)


S1_BASE = {
    "s0": "0" * 64,
    "method": "lookup",
    "nodes_ug": (1.0, 4.0),
    "weights": (0.25, 0.75),
    "source": "c" * 64,
}
S1_CASES = {
    "s0": "f" * 64,
    "method": "quadrature",
    "nodes_ug": (1.0, 5.0),
    "weights": (0.5, 0.5),
    "source": "d" * 64,
}


@pytest.mark.parametrize("name", sorted(S1_CASES))
def test_s1_key_changes_with_each_component(name):
    _changes(keys.s1_key, S1_BASE, S1_CASES, name)


def test_s1_key_changes_with_the_node_count_and_nothing_else():
    base = keys.s1_key(**S1_BASE)
    assert (
        keys.s1_key(
            **{**S1_BASE, "nodes_ug": (1.0, 4.0, 9.0), "weights": (0.2, 0.3, 0.5)}
        )
        != base
    )
    assert (
        keys.s1_key(
            **{**S1_BASE, "nodes_ug": np.array([1.0, 4.0]), "weights": [0.25, 0.75]}
        )
        == base
    )
    assert keys.s1_key(**{**S1_BASE, "s0": None}) is None
    with pytest.raises(ValueError):
        keys.s1_key(**{**S1_BASE, "weights": (1.0,)})
    with pytest.raises(ValueError):
        keys.s1_key(**{**S1_BASE, "nodes_ug": (1.0, np.nan)})


def test_s0_and_s1_keys_take_their_named_components_only():
    """The S1 key takes no kernel record, and neither key takes x_min, moments or weights."""
    assert set(inspect.signature(keys.s0_key).parameters) == {*S0_BASE}
    assert set(inspect.signature(keys.s1_key).parameters) == {*S1_BASE}
