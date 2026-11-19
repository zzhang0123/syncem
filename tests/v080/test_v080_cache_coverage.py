"""Components of the memo key and of the S2 context key that no rule chain carries (decision 159, T14 item 2).

Each case changes one input that no averaged rule reads, so neither ``d_0`` nor an S1 key moves with it:

* memo key: the channel order without a phase law, ``routes`` and ``s1``,
* S2 context key, with the memo entry dropped between the builds: a fixed value of a jet cell, the Legendre
  node count, the cell geometry and the provider (jet after core).

The second build must miss and reuse no column, and it equals the ``cache=False`` build by ``tobytes()``. A
control case with the same inputs hits. The RED runs are mutants of ``_basis_cache.build_keys``,
``_s2_columns._keyed`` and ``_keys.column_record`` that drop one component each (step-5b-2 review).
"""

import jax.numpy as jnp
import pytest

from syncem.base.continuum import ContinuumKernel
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine.average import AveragedKernel, uniform_rule
from syncem.engine.basis import build_from_averaged
from syncem.engine.cache import default_cache
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import Legendre
from tests.v080.test_v080_cache import memo_key, same_bits
from tests.v080.test_v080_stack import small_channels

K = ContinuumKernel(x_min=1e-12)
GAMMA2 = Cells.log(300, 3e5, 2, order=2)


def events(basis):
    return dict(basis.record.events)


def eta_avg():
    return AveragedKernel(K, [("eta", uniform_rule(0.0, 0.5, 3))], fixed={"B": 5e-6})


def drop_memo():
    """Remove the memo entries of the process cache and keep the S2 store."""
    store = default_cache()._store
    for key in [k for k in list(store._entries) if k[0] == "memo"]:
        store.pop(key)


def check_miss(first, second, fresh, reused=0):
    assert None not in (memo_key(first), memo_key(second))
    assert memo_key(first) != memo_key(second)
    assert events(second).get("memo hits") in (None, 0)
    assert events(second)["S2 columns reused"] == reused
    same_bits(second, fresh)


# ------------------------------------------------------------------ the memo key


def test_a_reversed_channel_set_without_a_phase_law_misses_the_memo():
    ch, lat = small_channels(4, 8), Lattice(gamma=GAMMA2)
    first = build_from_averaged(eta_avg(), ch, lat, azimuth_factor=0.4)
    again = build_from_averaged(eta_avg(), ch, lat, azimuth_factor=0.4)
    assert events(again)["memo hits"] == 1
    rev = ch[::-1]
    second = build_from_averaged(eta_avg(), rev, lat, azimuth_factor=0.4)
    fresh = build_from_averaged(eta_avg(), rev, lat, azimuth_factor=0.4, cache=False)
    # every S2 entry of a channel is its own, so the reversed set reuses every column
    check_miss(first, second, fresh, reused=events(first)["S2 columns computed"])


@pytest.mark.parametrize(
    "a, b",
    [({"routes": "auto"}, {"routes": "jet"}), ({"s1": "lookup"}, {"s1": "quadrature"})],
    ids=["routes", "s1"],
)
def test_the_provider_settings_enter_the_memo_key(a, b):
    ch = small_channels(3, 8)
    b_law = uniform_rule(4e-6, 6e-6, 4)
    avg = AveragedKernel(K, [("B", b_law), ("eta", uniform_rule(0.0, 0.5, 3))])
    lat = Lattice(gamma=GAMMA2)
    first = build_from_averaged(avg, ch, lat, azimuth_factor=0.4, **a)
    second = build_from_averaged(avg, ch, lat, azimuth_factor=0.4, **b)
    fresh = build_from_averaged(avg, ch, lat, azimuth_factor=0.4, cache=False, **b)
    assert None not in (memo_key(first), memo_key(second))
    assert memo_key(first) != memo_key(second)
    assert events(second)["memo misses"] == 1
    same_bits(second, fresh)


# ------------------------------------------------------------------ the S2 context key

AB = [Variable("a", kind="positive"), Variable("b", kind="positive")]


@kernel(variables=AB, stokes=("I",))
def ab_kernel(nu, a, b):
    return {"I": a * b * jnp.ones_like(nu)}


def _user(b, lo=1.0, hi=30.0):
    lat = Lattice(a=Cells.log(lo, hi, 2, order=2))
    return AveragedKernel(ab_kernel, fixed={"b": b}), lat, {}


def _legendre(nodes):
    avg = AveragedKernel(K, [], legendre={"eta": (4, nodes)}, fixed={"B": 5e-6})
    lat = Lattice(gamma=GAMMA2, eta=Legendre(4))
    return avg, lat, {"routes": "jet", "azimuth_factor": 0.4}


def _provider(routes):
    return eta_avg(), Lattice(gamma=GAMMA2), {"routes": routes, "azimuth_factor": 0.4}


S2_CASES = {
    "fixed value": (lambda: _user(2.0), lambda: _user(3.0)),
    "legendre nodes": (lambda: _legendre(24), lambda: _legendre(6)),
    "geometry": (lambda: _user(2.0), lambda: _user(2.0, 2.0, 60.0)),
    "provider": (lambda: _provider("core"), lambda: _provider("jet")),
}


@pytest.mark.parametrize("name", sorted(S2_CASES))
def test_s2_context_key_components_without_a_rule(name):
    ch = small_channels(4, 8)
    make_first, make_second = S2_CASES[name]
    avg, lat, kw = make_first()
    first = build_from_averaged(avg, ch, lat, **kw)
    drop_memo()
    avg, lat, kw = make_second()
    second = build_from_averaged(avg, ch, lat, **kw)
    check_miss(first, second, build_from_averaged(avg, ch, lat, cache=False, **kw))


def test_s2_control_with_equal_inputs_reuses_every_column():
    ch = small_channels(4, 8)
    avg, lat, kw = _user(2.0)
    first = build_from_averaged(avg, ch, lat, **kw)
    drop_memo()
    second = build_from_averaged(avg, ch, lat, **kw)
    assert events(second)["S2 columns computed"] == 0
    assert events(second)["S2 columns reused"] == events(first)["S2 columns computed"]
    same_bits(second, first)
