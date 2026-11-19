"""The basis memo, its keys and the process cache in the build (design/0.8/SPEC.md decisions 156 to 162, T14).

T14 items 1, 2, 3, 5, 8 and 10 of the step-5b plan (item 7 is in ``test_v080_cache_build_disk.py``). A memo hit equals the rebuild by ``tobytes()``. The key
cases are asserted on the basis memo key and on the S2 context keys that the build stores, with an unchanged
rebuild as the control. The cache is the process ``Cache``, which ``tests/conftest.py`` makes fresh for every
test with ``SYNCEM_CACHE`` under ``tmp_path``. Every assertion is an exact identity except the stale-trace ratio
of item 5, whose tolerance 1e-15 is that of the plan (row 160).
"""

import ast
import math
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import _study
import syncem.engine as engine_package
from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.phase import FaradayRotation
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine import _routes
from syncem.engine._stage_memo import s1_arrays
from syncem.engine.average import AveragedKernel, NodeRule, fixed_rule, uniform_rule
from syncem.engine.basis import build_from_averaged
from syncem.engine.cache import default_cache
from syncem.engine.field_average import FieldAverage
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.model.phase import GaussianScreen
from tests.v080.test_v080_stack import small_channels

K = ContinuumKernel(x_min=1e-12)
AB = [Variable("a", kind="positive"), Variable("b", kind="positive")]
ABE = [*AB, Variable("eta", kind="cosine")]
GAMMA2 = Cells.log(300, 3e5, 2, order=2)
A2 = Cells.log(1.0, 30.0, 2, order=2)


def events(basis):
    return dict(basis.record.events)


def memo_key(basis):
    return dict(basis.record.keys)["memo"]


def same_bits(a, b):
    """``tobytes()`` equality of the stored spectra and of the design."""
    for x, y in ((a.spectra0, b.spectra0), (a.spectra2, b.spectra2)):
        assert (x is None) == (y is None)
        if x is not None:
            assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
    assert np.asarray(a.design()).tobytes() == np.asarray(b.design()).tobytes()


def build(case, ch, **kwargs):
    avg, lattice, factor = case
    return build_from_averaged(avg, ch, lattice, azimuth_factor=factor, **kwargs)


# ------------------------------------------------------------------ 1. a memo hit equals the rebuild


def paper_case(kind):
    cfg = _study.paper()
    gamma = Cells.log(300, cfg.gamma_hi, 19, order=2)
    if kind == "E1":
        b, wb = cfg.b_nodes_weights
        lattice, levels = Lattice(gamma=gamma), [("B", fixed_rule(b * 1e-6, wb))]
    else:
        single = Cells.single("5 uG", scale="1 uG", order=2)
        lattice, levels = Lattice(gamma=gamma, B=single, order=2), []
    levels.append(("eta", fixed_rule(*cfg.eta_nodes_weights)))
    screen = GaussianScreen(cfg.screen_mean, cfg.screen_sd)
    avg = AveragedKernel(K, levels, phase=FaradayRotation(screen=screen))
    return avg, lattice, cfg.azimuth_factor


def paper_channels(n):
    return Channels.log(0.1, 10, n, unit="GHz", half_width=0.2, n_nu=64)


def per_node_case():
    """``B`` on 4 nodes given ``eta`` on 4 nodes, 3 cells: 16 one-node tables per cell."""
    b = uniform_rule(lambda p: 4e-6 * (1 + p["eta"]), 6e-6, 4, parents=("eta",))
    avg = AveragedKernel(K, [("eta", uniform_rule(0.0, 0.5, 4)), ("B", b)])
    return avg, Lattice(gamma=Cells.log(300, 3e6, 3, order=2)), 0.4


@pytest.mark.parametrize("kind", ["E1", "E6", "per-node"])
def test_a_memo_hit_equals_the_uncached_build_bitwise(kind):
    case = per_node_case() if kind == "per-node" else paper_case(kind)
    ch = small_channels(8) if kind == "per-node" else paper_channels(8)
    first = build(case, ch)
    with pytest.MonkeyPatch.context() as mp:
        for name in ("stack_cell", "core_cell", "jet_lattice"):
            mp.setattr(_routes, name, lambda *a, **k: pytest.fail("provider call"))
        hit = build(case, ch)
    assert (events(first)["memo misses"], events(hit)["memo hits"]) == (1, 1)
    assert (
        events(first)["S2 columns computed"] > 0 == events(hit)["S2 columns computed"]
    )
    assert memo_key(hit) == memo_key(first) and hit.key == first.key is not None
    rebuilt = build(case, ch, cache=False)
    same_bits(hit, rebuilt)
    same_bits(first, rebuilt)
    assert hit.record.routes == rebuilt.record.routes
    assert rebuilt.key is None and rebuilt.record.notes == ("not cached: cache=False",)


# ------------------------------------------------------------------ 2. the key cases of T14

_SCALE = {"k": 1.0}
_SCALE2 = {"k": 1.0}


def _helper(a):
    return _SCALE["k"] * a


def _helper_b(a):
    return _SCALE["k"] * a * a


def _scaled(a):
    return _SCALE2["k"] * a


@kernel(variables=ABE, stokes=("I",))
def helper_kernel(nu, a, b, eta):
    return {"I": _helper(a) * b * (1 + eta) * jnp.ones_like(nu)}


def _factory(fn):
    @kernel(variables=ABE, stokes=("I",))
    def made(nu, a, b, eta):
        return {"I": fn(a) * b * (1 + eta) * jnp.ones_like(nu)}

    return made


FACTORY_KERNEL = _factory(_scaled)


def _lambda_rule(c):
    return uniform_rule(
        lambda p: c * p["a"], lambda p: 2 * c * p["a"], 3, parents=("a",)
    )


def _named_rule(c):
    def law(p):
        return c * p["a"] * jnp.asarray([1.25, 1.75]), jnp.asarray([0.5, 0.5])

    return NodeRule(law, parents=("a",))


def _eta_rule(c):
    return uniform_rule(
        lambda p: 1 + c * p["eta"], lambda p: 2 + c * p["eta"], 3, parents=("eta",)
    )


def _user(kern, b_rule):
    eta = ("eta", uniform_rule(0.0, 0.5, 3))
    return AveragedKernel(kern, [eta, ("b", b_rule)]), Lattice(a=A2), None


def _stack(fixed_b=5e-6, mean=8.0):
    screen = FaradayRotation(screen=GaussianScreen(mean, 0.5))
    eta = ("eta", uniform_rule(0.0, 0.5, 3))
    avg = AveragedKernel(K, [eta], fixed={"B": fixed_b}, phase=screen)
    return avg, Lattice(gamma=GAMMA2), 0.4


PAIRS = {
    "unnamed conditional": lambda c: _user(helper_kernel, _lambda_rule(c)),
    "named closure": lambda c: _user(helper_kernel, _named_rule(c)),
    "eta dependence": lambda c: _user(helper_kernel, _eta_rule(0.1 * c)),
    "fixed B": lambda c: _stack(fixed_b=5e-6 * c),
    "phase screen": lambda c: _stack(mean=8.0 * c),
    "unchanged": lambda c: _user(helper_kernel, _eta_rule(0.1)),
}


def _case(monkeypatch, name):
    """``(base, edit, variant)``: makers of ``(avg, lattice, factor)`` and the edit made between the builds."""
    edits = {
        "helper": lambda: monkeypatch.setitem(globals(), "_helper", _helper_b),
        "global dict": lambda: monkeypatch.setitem(_SCALE, "k", 2.0),
        "factory closure": lambda: monkeypatch.setitem(_SCALE2, "k", 3.0),
    }
    if name in edits:
        kern = FACTORY_KERNEL if name == "factory closure" else helper_kernel
        return (
            (lambda: _user(kern, _eta_rule(0.1))),
            edits[name],
            (lambda: _user(kern, _eta_rule(0.1))),
        )
    other = 1.0 if name == "unchanged" else 1.2
    return (lambda: PAIRS[name](1.0)), (lambda: None), (lambda: PAIRS[name](other))


def _keys_of(make, ch, monkeypatch):
    """Memo key and the set of S2 context keys that one build stores. The cache is cleared after it."""
    cache, seen = default_cache(), []
    real = cache.put_columns

    def spy(context_keys, ids, values, chains):
        seen.extend(str(k) for k in context_keys)
        return real(context_keys, ids, values, chains)

    monkeypatch.setattr(cache, "put_columns", spy)
    basis = build(make(), ch)
    monkeypatch.setattr(cache, "put_columns", real)
    cache.clear()
    return memo_key(basis), set(seen)


CASES = ["helper", "global dict", "factory closure", *PAIRS]


@pytest.mark.parametrize("name", CASES)
def test_key_cases_change_the_memo_key_and_every_s2_context_key(name, monkeypatch):
    ch = small_channels(3, 8)
    base, edit, variant = _case(monkeypatch, name)
    key0, s2_0 = _keys_of(base, ch, monkeypatch)
    edit()
    key1, s2_1 = _keys_of(variant, ch, monkeypatch)
    assert key0 is not None and key1 is not None and s2_0 and s2_1
    if name == "unchanged":
        assert key0 == key1 and s2_0 == s2_1
    else:
        assert key0 != key1 and not s2_0 & s2_1


# ------------------------------------------------------------------ 3. rebinding w and the azimuth factor


def test_a_hit_rebinds_the_lattice_weights_and_the_azimuth_factor():
    avg, lat, _ = _stack()
    ch = small_channels(3, 8)
    heavy = Lattice(gamma=GAMMA2, weights=lat.weights * np.array([1e-3, 1e2]))
    first = build((avg, lat, 0.4), ch)
    for case in ((avg, heavy, 0.4), (avg, lat, 0.3 + 0.2j)):
        hit = build(case, ch)
        assert events(hit)["memo hits"] == 1 and events(hit)["S2 columns computed"] == 0
        assert memo_key(hit) == memo_key(first) and hit.key not in (None, first.key)
        assert hit.lattice is case[1]
        same_bits(hit, build(case, ch, cache=False))


# ------------------------------------------------------------------ 5. stale traces (plan row 160)

NODES = np.array([1.5])


@kernel(variables=AB, stokes=("I",))
def ab_kernel(nu, a, b):
    return {"I": a * b * jnp.ones_like(nu)}


class _Holder:
    def __init__(self):
        self.k = 1.0


HOLDER = _Holder()


@kernel(variables=AB, stokes=("I",))
def held_kernel(nu, a, b):
    return {"I": HOLDER.k * a * b * jnp.ones_like(nu)}


def _global_rule():
    return NodeRule(lambda p: (jnp.asarray(NODES), jnp.ones(1)), parents=())


@pytest.mark.parametrize(
    "kern", [ab_kernel, held_kernel], ids=["keyed", "not cacheable"]
)
def test_an_edited_global_array_gives_the_new_numbers(kern):
    """Plan row 160: [1.5, 0.3] before the edit and [15, 3] after it, times the channel response of a constant
    (1.00026 on channel 0). On the step-5b-1 code the rebuild gave [1.50039, 0.30008] both times (probe of
    2026-11-19, ``probe_stale_red.py`` in the session scratchpad). The rebuilt columns equal ten times the first
    to 1e-15 (the plan's tolerance) and a ``cache=False`` build bitwise (a fresh trace).
    """
    avg = AveragedKernel(kern, [("b", _global_rule())])
    lattice = Lattice(a=Cells.log(math.exp(-0.2), math.exp(0.2), 1, order=1))
    ch = small_channels(2, 8)
    before = build_from_averaged(avg, ch, lattice)
    old = np.asarray(before.spectra0).copy()
    np.testing.assert_allclose(old[:, 0, 1], 0.2 * old[:, 0, 0], rtol=1e-15)
    try:
        np.multiply(NODES, 10.0, out=NODES)
        after = build_from_averaged(avg, ch, lattice)
        fresh = build_from_averaged(avg, ch, lattice, cache=False)
    finally:
        np.divide(NODES, 10.0, out=NODES)
    np.testing.assert_allclose(
        np.asarray(after.spectra0), 10.0 * old, rtol=1e-15, atol=0
    )
    same_bits(after, fresh)
    if kern is held_kernel:
        assert after.key is None and memo_key(after) is None
        assert any(n.startswith("not cached: ") for n in after.record.notes)
    else:
        assert after.key is not None and memo_key(after) != memo_key(before)


# ------------------------------------------------------------------ 8. memory bound


def test_a_bound_below_two_s1_tables_rebuilds_one_node_tables_bitwise(monkeypatch):
    one = sum(a.nbytes for a in s1_arrays(FieldAverage.lookups([5.0], [1.0])).values())
    bound = int(1.5 * one)
    monkeypatch.setenv("SYNCEM_CACHE_MEM", str(bound))
    case, ch = per_node_case(), small_channels(3, 8)
    bounded = build(case, ch)
    assert (
        default_cache().mem_bytes
        == bound
        == dict(bounded.record.defaults)["cache_mem_bytes"]
    )
    assert events(bounded)["one-node rebuilds"] > 0
    assert {r.mode for r in bounded.record.routes} == {"per-node"}
    same_bits(bounded, build(case, ch, cache=False))


# ------------------------------------------------------------------ 10. no module-level lru_cache

ALLOWED = {("primitive", "derivative_polynomials"), ("primitive", "_gauss")}
CACHING = ("lru_cache", "cache")


def _cached_functions(path):
    found = set()
    for node in ast.walk(ast.parse(path.read_text())):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        for dec in node.decorator_list:
            target = dec.func if isinstance(dec, ast.Call) else dec
            name = (
                target.attr
                if isinstance(target, ast.Attribute)
                else getattr(target, "id", "")
            )
            if name in CACHING:
                found.add((path.stem, node.name))
    return found


def test_no_module_level_lru_cache_in_the_engine_but_the_named_pure_functions():
    root = Path(engine_package.__file__).parent
    files = [p for p in root.glob("*.py") if p.stem.isidentifier()]
    assert set().union(*(_cached_functions(p) for p in files)) == ALLOWED
