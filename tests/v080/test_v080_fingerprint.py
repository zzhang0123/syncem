"""Step 5b-2 of design/0.8/SPEC.md: function fingerprints, kernel records and phase fingerprints.

``function_fingerprint`` (decision 156) hashes code objects, defaults, closure cells, the globals that the
code names, ``functools.partial`` and the Python version. ``UserKernel.describe`` carries it and writes the
capabilities through ``encode``. ``ContinuumKernel.describe`` carries the source digest of the kernel
modules. ``PhaseLaw.channel_fingerprints`` hashes ``tau`` and the screen at each channel's nodes (decision
159). Every assertion is an exact identity of strings or bytes, so no tolerance enters. The rule-chain part
of the step-5b plan's fingerprint list is tested with the rule chains, not here. Loading outside the package
and the hash seeds are in ``test_v080_fingerprint_alone.py``.
"""

import functools
import hashlib
import importlib
import importlib.util
import math
import types
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base import _fingerprint as fp
from syncem.base import _function_fingerprint as ffm
from syncem.base import continuum as continuum_module
from syncem.base.capabilities import EnergyWindow, Parity, Similarity
from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.phase import FaradayRotation, PhaseLaw, phase_fingerprints
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.model.phase import GaussianScreen, phase_coordinate

ffp = ffm.function_fingerprint
NotCacheable = fp.NotCacheable

_TABLE = {"k": 1.0}
_ARRAY = np.ones(4)


def _helper(x):
    return _TABLE["k"] * x


def _other_helper(x):
    return 3.0 * x


def _outer(x):
    return _helper(x) + 1.0


def _reads_array(x):
    return _ARRAY[2] * x


def _is_hex(value):
    return isinstance(value, str) and len(value) == 64 and int(value, 16) >= 0


def _compile(source, name="f"):
    """A function compiled from ``source`` in a fresh namespace, as if defined in another file."""
    namespace = {"__name__": "user_namespace"}
    exec(compile(source, "<user>", "exec"), namespace)
    return namespace[name]


def _fp(source, name="f"):
    return ffp(_compile(source, name))


# ------------------------------------------------------------ fingerprint
_SET = "{" + ", ".join(repr(f"s{i}") for i in range(30)) + "}"
_SET_REVERSED = "{" + ", ".join(repr(f"s{i}") for i in reversed(range(30))) + "}"
_SET_FEWER = "{" + ", ".join(repr(f"s{i}") for i in range(29)) + "}"
_RECURSIVE = (
    "def even(n):\n    return True if n == 0 else odd(n - 1)\n"
    "def odd(n):\n    return False if n == 0 else even(n - 1)\n"
    "def fact(n):\n    return 1 if n == 0 else n * fact(n - 1)\n"
)
_MEMBER = "def f(x):\n    return x in {}\n"
_DIFFER = {
    "default": (
        "def f(x, k=1.0):\n    return k * x\n",
        "def f(x, k=2.0):\n    return k * x\n",
    ),
    "kwdefault": (
        "def f(*, k=1.0):\n    return k\n",
        "def f(*, k=2.0):\n    return k\n",
    ),
    "nested body": (
        "def f(x):\n    return (lambda y: y + 1)(x)\n",
        "def f(x):\n    return (lambda y: y + 2)(x)\n",
    ),
    "frozenset constant": (_MEMBER.format(_SET), _MEMBER.format(_SET_FEWER)),
    "module": (
        "import numpy as np\ndef f(x):\n    return np.sin(x)\n",
        "import math as np\ndef f(x):\n    return np.sin(x)\n",
    ),
}
_EQUAL = {
    "defined twice": ("import numpy as np\ndef f(x):\n    return np.sin(x) * 2.0\n",)
    * 2,
    "frozenset order": (_MEMBER.format(_SET), _MEMBER.format(_SET_REVERSED)),
    "recursion": (_RECURSIVE + "f = even\n",) * 2,
}


@pytest.mark.parametrize("case", sorted(_DIFFER))
def test_fingerprint_changes_with_the_source(case):
    one, two = _DIFFER[case]
    assert _fp(one) != _fp(two) and _fp(one) == _fp(one)


@pytest.mark.parametrize("case", sorted(_EQUAL))
def test_fingerprint_is_equal_for_equal_functions(case):
    one, two = _EQUAL[case]
    assert _is_hex(_fp(one)) and _fp(one) == _fp(two)


def test_cases_compile_as_intended():
    f = _compile(_EQUAL["frozenset order"][0])
    assert any(isinstance(c, frozenset) for c in f.__code__.co_consts)
    assert _fp(_RECURSIVE, "even") != _fp(_RECURSIVE, "odd") != _fp(_RECURSIVE, "fact")
    first = lambda x: 2 * x  # noqa: E731

    second = lambda x: 2 * x  # noqa: E731
    assert ffp(first) == ffp(second)  # line numbers do not enter


def test_fingerprint_changes_with_an_edited_code_object():
    def f(x):
        return x + 1.0

    def g(x):
        return x + 2.0

    copy = types.FunctionType(f.__code__, f.__globals__, "f")
    before = ffp(copy)
    copy.__code__ = g.__code__
    assert ffp(copy) != before


def test_fingerprint_changes_with_globals(monkeypatch):
    """A global dict read only by a helper, a rebound helper and a global array edited in place."""
    before, array = ffp(_outer), ffp(_reads_array)
    monkeypatch.setitem(_TABLE, "k", 2.0)
    assert ffp(_outer) != before
    monkeypatch.undo()
    assert ffp(_outer) == before
    monkeypatch.setitem(_outer.__globals__, "_helper", _other_helper)
    assert ffp(_outer) != before
    monkeypatch.undo()
    _ARRAY[2] = 5.0
    try:
        assert ffp(_reads_array) != array
    finally:
        _ARRAY[2] = 1.0
    assert ffp(_reads_array) == array


def test_fingerprint_changes_with_closure_values_and_helpers():
    def make(scale, helper):
        def f(x):
            return scale * helper(x)

        return f

    assert ffp(make(1.0, _helper)) == ffp(make(1.0, _helper))
    assert ffp(make(1.0, _helper)) != ffp(make(2.0, _helper))
    assert ffp(make(1.0, _helper)) != ffp(make(1.0, _other_helper))


def test_fingerprint_changes_with_partial_arguments():
    partial = functools.partial
    assert ffp(partial(_helper, 1.0)) == ffp(partial(_helper, 1.0))
    assert ffp(partial(_helper, 1.0)) != ffp(partial(_helper, 2.0))
    assert ffp(partial(_helper, 1.0)) != ffp(partial(_other_helper, 1.0))
    assert ffp(partial(_outer, x=1.0)) != ffp(partial(_outer, x=2.0))


def test_fingerprint_includes_the_python_version(monkeypatch):
    before = ffp(_outer)
    monkeypatch.setattr(fp.sys, "version_info", (3, 99, 0, "final", 0))
    assert ffp(_outer) != before


# ------------------------------------- library callables against user functions
def test_library_callables_and_modules_are_named_by_version(monkeypatch):
    """A library callable is not traced into. Its version changes the fingerprint."""
    for library in (np.sin, jnp.exp, math.sqrt, np.linspace, len):
        assert _is_hex(ffp(library))
    assert ffp(np.sin) != ffp(np.cos)
    before, module = ffp(np.sin), _fp(_DIFFER["module"][0])
    monkeypatch.setattr(np, "__version__", "0.0.0")
    assert ffp(np.sin) != before and _fp(_DIFFER["module"][0]) != module


def test_wrapped_user_functions_and_bound_methods():
    """A jitted user function is hashed through ``__wrapped__``. A bound method carries its instance."""
    one, two = (jax.jit(_compile(f"def f(x):\n    return x + {c}\n")) for c in "12")
    assert ffp(one) != ffp(two)
    assert ffp(one) == ffp(jax.jit(_compile("def f(x):\n    return x + 1\n")))
    assert ffp(Similarity(("gamma",)).__eq__) != ffp(Similarity(("B",)).__eq__)


# ------------------------------------------------------- not cacheable
_REFUSED = {
    "closure": (
        "def make():\n    t = object()\n    return lambda x: t\nf = make()\n",
        "object",
    ),
    "global": ("TOKEN = object()\ndef f(x):\n    return TOKEN\n", "TOKEN"),
    "user class": ("class Box:\n    pass\ndef f(x):\n    return Box()\n", "Box"),
}


@pytest.mark.parametrize("case", sorted(_REFUSED))
def test_a_value_without_an_encoding_raises_not_cacheable_with_a_reason(case):
    source, named = _REFUSED[case]
    with pytest.raises(NotCacheable, match=named):
        _fp(source)


def test_non_callables_and_the_encoder_refuse():
    """``encode`` keeps the decision-153 refusal. Functions enter keys through ``function_fingerprint``."""
    for refused in (lambda: ffp(object()), lambda: fp.encode(_outer)):
        with pytest.raises(NotCacheable):
            refused()


# ----------------------------------------------------- user kernel records
def _make_kernel(scale=1.0, capabilities=()):
    @kernel(
        variables=[Variable("a", kind="positive")],
        stokes=("I",),
        capabilities=capabilities,
    )
    def k(nu, a):
        return {"I": scale * _helper(a) * jnp.ones_like(nu)}

    return k


def _window_one(averaged, channels, *, density, eps):
    return 1.0, 2.0


def _window_two(averaged, channels, *, density, eps):
    return 1.0, 3.0


def test_user_kernel_record_holds_no_address():
    k = _make_kernel(capabilities=(EnergyWindow(_window_one), Parity(("I",), ())))
    record = dict(k.describe())
    assert "0x" not in str(k.describe())
    assert record["function"] == ffp(k.fn)
    assert _is_hex(record["capabilities"]) and record["not_cached"] == ()
    hash(k.describe())


def test_user_kernel_record_follows_functions_closures_and_helpers(monkeypatch):
    def record(window, scale=1.0):
        return _make_kernel(scale, (EnergyWindow(window),)).describe()

    assert record(_window_one) != record(_window_two)
    assert record(_window_one) == record(_window_one)
    assert record(_window_one, 1.0) != record(_window_one, 5.0)
    before = record(_window_one)
    monkeypatch.setitem(_TABLE, "k", 4.0)
    assert record(_window_one) != before


def test_user_kernel_record_without_fingerprints_names_the_reasons():
    token = object()

    @kernel(variables=[Variable("a", kind="positive")], stokes=("I",))
    def k(nu, a):
        return {"I": a * jnp.ones_like(nu) + 0 * len(str(token))}

    record = dict(k.describe())
    assert record["function"] is None and "0x" not in str(k.describe())
    assert len(record["not_cached"]) == 1 and "object" in record["not_cached"][0]
    hash(k.describe())
    record = dict(
        _make_kernel(capabilities=(EnergyWindow(lambda *a, **kw: token),)).describe()
    )
    assert record["capabilities"] is None and record["function"] is not None
    assert "capabilit" in record["not_cached"][0]


# --------------------------------------------------- continuum kernel record
_KERNEL_SOURCES = (
    "syncem.base.continuum",
    "syncem.model._continuum",
    "syncem.ultrarel",
    "syncem.bessel",
    "syncem.constants",
)


def test_continuum_record_carries_the_source_digest_of_the_kernel_modules():
    h = hashlib.sha256()
    for name in _KERNEL_SOURCES:
        h.update(
            name.encode()
            + b"\x00"
            + Path(importlib.util.find_spec(name).origin).read_bytes()
        )
    k = ContinuumKernel(x_min=1e-9)
    names = [name for name, _ in k.describe()]
    assert names == ["name", "gamma_min", "x_min", "n_nodes_F", "tail_cutoff", "source"]
    assert (
        dict(k.describe())["source"]
        == h.hexdigest()
        == dict(ContinuumKernel().describe())["source"]
    )
    assert continuum_module._KERNEL_SOURCES == _KERNEL_SOURCES
    hash(k.describe())


# ------------------------------------------------------- phase fingerprints
@pytest.fixture(scope="module")
def channels():
    return Channels.log(0.5, 10, 5, unit="GHz", half_width=0.2, n_nu=16)


def test_phase_fingerprints_are_one_digest_per_channel(channels):
    law = FaradayRotation(screen=GaussianScreen(5.0, 3.0))
    keys = law.channel_fingerprints(channels)
    assert isinstance(keys, tuple) and len(keys) == channels.n_ch
    assert all(_is_hex(k) for k in keys) and len(set(keys)) == channels.n_ch
    assert keys == FaradayRotation(
        screen=GaussianScreen(5.0, 3.0)
    ).channel_fingerprints(channels)
    assert law.channel_fingerprints(channels[[3, 1]]) == (keys[3], keys[1])


def test_phase_fingerprints_hash_tau_and_the_screen_at_the_nodes(channels):
    tau = np.asarray(phase_coordinate(np.asarray(channels.nodes)[1]))
    assert FaradayRotation().channel_fingerprints(channels)[1] == fp.digest(
        ("phase", tau, None)
    )
    value = np.asarray(GaussianScreen(5.0, 3.0)(jnp.asarray(tau)))[0]
    screened = FaradayRotation(screen=GaussianScreen(5.0, 3.0))
    assert screened.channel_fingerprints(channels)[1] == fp.digest(
        ("phase", tau, value)
    )


def test_phase_fingerprints_change_with_tau_and_screen(channels):
    def keys(law):
        return law.channel_fingerprints(channels)

    base = keys(FaradayRotation())
    assert keys(FaradayRotation(screen=GaussianScreen(5.0, 3.0))) != base
    assert keys(FaradayRotation(screen=GaussianScreen(5.0, 3.0))) != keys(
        FaradayRotation(screen=GaussianScreen(5.0, 4.0))
    )
    assert keys(PhaseLaw(lambda nu: 2.0 * phase_coordinate(nu))) != base
    flat = PhaseLaw(phase_coordinate, screen=lambda tau: jnp.exp(-(tau**2)))
    lifted = PhaseLaw(phase_coordinate, screen=lambda tau: jnp.exp(-(tau**2))[None])
    assert keys(flat) == keys(lifted)  # the screen values enter, not their leading axis


def test_no_phase_has_one_tag_per_channel(channels):
    for phase in (None, False):
        keys = phase_fingerprints(phase, channels)
        assert len(keys) == channels.n_ch and len(set(keys)) == 1
        assert keys[0] not in FaradayRotation().channel_fingerprints(channels)
    law = FaradayRotation()
    assert phase_fingerprints(law, channels) == law.channel_fingerprints(channels)
    with pytest.raises(TypeError, match="PhaseLaw"):
        phase_fingerprints("faraday", channels)


def test_phase_fingerprints_raise_not_cacheable_under_trace(channels):
    def traced(width):
        PhaseLaw(
            phase_coordinate, screen=lambda tau: jnp.exp(-width * tau)
        ).channel_fingerprints(channels)
        return width

    with pytest.raises(NotCacheable):
        jax.jit(traced)(1.0)
