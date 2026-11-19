"""Per-column reuse of S2 entries in a basis build (design/0.8/SPEC.md decision 159 part 3, T14 items 4 and 9).

A build that misses the memo reads the S2 store per (cell, block, channel) context key. A column is reused when
its (Legendre degrees, exponent) is stored and the rule chains agree up to its total degree. The providers run for
the other columns only, and "S2 columns computed" counts the (row, channel, block) values taken from a provider
call. Both providers give the I and P blocks for every kernel, so a build without V counts two blocks. The
counts are exact identities of the lattice index.

A reused column equals the stored one by ``tobytes()``, checked through ``with_order``, which gathers stored
spectra. A first build through the store on distinct channels equals the ``cache=False`` build by
``tobytes()``. A build that reuses columns of another order or channel set differs from the ``cache=False``
build by the batch dependence of an uncached build, within ``REUSE_TOL`` (decision 164, ``test_v080_cache_s2_bound.py``). The cache is the
process ``Cache`` of ``tests/conftest.py``, fresh for every test with ``SYNCEM_CACHE`` under ``tmp_path``.
"""

import dataclasses
import math

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine import _routes
from syncem.engine._stack_series import StackRefused, stack_cell, stack_stage
from syncem.engine._stage_memo import StageMemo
from syncem.engine.average import AveragedKernel, NodeRule
from syncem.engine.basis import build_from_averaged
from syncem.engine.cache import default_cache
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import Legendre
from tests.v080.test_v080_basis import counts
from tests.v080.test_v080_cache import per_node_case
from tests.v080.test_v080_cache_s2_bound import REUSE_TOL
from tests.v080.test_v080_routes import (
    REACH,
    cached_avg,
    eta4,
    field_lattice,
    fixed_b_avg,
)
from tests.v080.test_v080_stack import screen, small_channels
from tests.v080.test_v080_stack_boundaries import _lognormal_avg, gamma_lattice

K = ContinuumKernel(x_min=1e-12)
MIXED = ("stack", "jet")
PROVIDERS = ("stack_cell", "jet_lattice", "core_cell")


def events(basis):
    return dict(basis.record.events)


def computed(basis):
    return events(basis)["S2 columns computed"]


def n_blocks(basis):
    """The provider blocks: I and P, and V when it is built."""
    return 3 if "V" in basis.columns.stokes else 2


def n_rows(lattice, keep):
    """Rows of one block whose (degrees, powers) pass ``keep``."""
    a = lattice.index.arrays
    return sum(
        bool(keep(tuple(a.degrees[k]), tuple(a.powers[k])))
        for k in range(lattice.index.block_size)
    )


def _pairs(a, b):
    for x, y in ((a.spectra0, b.spectra0), (a.spectra2, b.spectra2)):
        assert (x is None) == (y is None)
        if x is not None:
            yield np.asarray(x), np.asarray(y)


def same_spectra(a, b):
    for x, y in _pairs(a, b):
        assert x.tobytes() == y.tobytes()


def close_spectra(a, b, tol=REUSE_TOL):
    """Largest difference over the largest stored spectrum of each channel of ``b``."""
    for x, y in _pairs(a, b):
        scale = np.abs(y).reshape(y.shape[0], -1).max(axis=1)
        assert np.max(np.abs(x - y).reshape(y.shape[0], -1).max(axis=1) / scale) < tol


def mixed(ch, order=2, **kwargs):
    """One stack cell and one jet cell on the gamma lattice of the basis tests."""
    return build_from_averaged(
        cached_avg(),
        ch,
        gamma_lattice(2, order),
        routes=MIXED,
        azimuth_factor=0.4,
        **kwargs,
    )


def legendre(ch, degree, **kwargs):
    """``eta`` resolved by ``Legendre(degree)`` on 24 Gauss nodes, ``B`` fixed, one stack and one jet cell."""
    avg = AveragedKernel(K, [], legendre={"eta": (degree, 24)}, fixed={"B": 5e-6})
    lat = Lattice(gamma=Cells.log(300, 3e5, 2, order=2), eta=Legendre(degree))
    return build_from_averaged(avg, ch, lat, routes=MIXED, azimuth_factor=0.4, **kwargs)


@pytest.fixture
def no_providers(monkeypatch):
    for name in PROVIDERS:
        monkeypatch.setattr(_routes, name, lambda *a, **k: pytest.fail("provider call"))


# ------------------------------------------------------------------ the reuse path against the plain path


@pytest.mark.parametrize("make", [mixed, lambda ch, **k: legendre(ch, 4, **k)])
def test_a_first_build_with_the_store_equals_the_uncached_build_bitwise(make):
    ch = small_channels(3, 8)
    first = make(ch)
    assert events(first)["S2 columns reused"] == 0
    assert computed(first) == first.lattice.index.block_size * ch.n_ch * n_blocks(first)
    same_spectra(first, make(ch, cache=False))


# ------------------------------------------------------------------ T14 item 4: orders and degrees


def test_order_2_then_4_computes_only_the_columns_of_degree_3_and_4(monkeypatch):
    ch = small_channels(3, 8)
    low = mixed(ch, 2)
    exponents = []
    real = _routes.jet_lattice

    def spy(*a, **k):
        exponents.append(k.get("exponents"))
        return real(*a, **k)

    monkeypatch.setattr(_routes, "jet_lattice", spy)
    high = mixed(ch, 4)
    lat, scale = high.lattice, ch.n_ch * n_blocks(high)
    assert computed(high) == n_rows(lat, lambda d, p: sum(p) > 2) * scale
    assert (
        events(high)["S2 columns reused"]
        == n_rows(lat, lambda d, p: sum(p) <= 2) * scale
    )
    assert exponents == [((3,), (4,))]
    close_spectra(high, mixed(ch, 4, cache=False))
    same_spectra(high.with_order(2), low)


def test_order_4_then_2_computes_nothing(request):
    ch = small_channels(3, 8)
    high = mixed(ch, 4)
    request.getfixturevalue("no_providers")
    low = mixed(ch, 2)
    assert computed(low) == 0 and events(low)["memo misses"] == 1
    assert (
        events(low)["S2 columns reused"] == low.lattice.index.block_size * ch.n_ch * 2
    )
    same_spectra(low, high.with_order(2))
    request.getfixturevalue("monkeypatch").undo()
    close_spectra(low, mixed(ch, 2, cache=False))


def test_legendre_4_then_8_computes_only_the_new_degrees():
    """The node count is 24 in both builds, so column ``l`` does not depend on ``L`` (``average.py:157-178``)."""
    ch = small_channels(3, 8)
    legendre(ch, 4)
    wide = legendre(ch, 8)
    lat, scale = wide.lattice, ch.n_ch * n_blocks(wide)
    assert computed(wide) == n_rows(lat, lambda d, p: max(d) > 4) * scale
    assert (
        events(wide)["S2 columns reused"]
        == n_rows(lat, lambda d, p: max(d) <= 4) * scale
    )
    close_spectra(wide, legendre(ch, 8, cache=False))


def test_with_order_above_the_built_order_computes_only_the_new_exponents():
    ch = small_channels(3, 8)
    low = mixed(ch, 2)
    up = low.with_order(4)
    lat, scale = up.lattice, ch.n_ch * n_blocks(up)
    assert up.lattice.order == 4
    assert computed(up) == n_rows(lat, lambda d, p: sum(p) > 2) * scale
    assert [r.provider for r in up.record.routes] == list(MIXED)
    assert "routes of the order-2 build" in up.record.notes
    close_spectra(up, mixed(ch, 4, cache=False))
    same_spectra(up.with_order(2), low)


def test_with_order_above_an_uncached_build_computes_every_column():
    ch = small_channels(3, 8)
    up = mixed(ch, 2, cache=False).with_order(4)
    assert computed(up) == up.lattice.index.block_size * ch.n_ch * n_blocks(up)
    assert "not cached: cache=False" in up.record.notes
    same_spectra(up, mixed(ch, 4, cache=False))


XY = [Variable("x", kind="positive"), Variable("y", kind="real")]
T12, W12 = (jnp.asarray(a) for a in np.polynomial.legendre.leggauss(12))


@kernel(variables=XY, stokes=("I",))
def xy_kernel(nu, x, y):
    return {"I": x * (1.0 + 0.1 * y) * jnp.ones_like(nu)}


def _y_rule(shift):
    """``y`` given ``x``: 12 nodes around ``0.3 ln x`` plus ``shift(u)``, ``u`` the chart coordinate of cell 0."""

    def law(point):
        u = jnp.log(point["x"]) / 0.5
        return 0.3 * jnp.log(point["x"]) + shift(u) + 0.5 * T12, 0.5 * W12

    return NodeRule(law, parents=("x",))


def test_a_rule_change_seen_first_at_degree_1_reuses_the_degree_0_columns():
    """The shift is zero for ``|u| < 0.05`` and is ``0.2 u`` there, so every ``d_0`` point (``|u| >= 0.18``, see
    ``test_v080_rule_chain.py``) sees the base law and ``d_1`` sees the slope. Degree 0 is reused, degrees 1 and
    2 are computed."""
    lat = Lattice(x=Cells.log(math.exp(-0.5), math.exp(0.5), 1, order=2))
    ch = small_channels(3, 8)
    base = _y_rule(lambda u: 0.0 * u)
    sloped = _y_rule(lambda u: jnp.where(jnp.abs(u) < 0.05, 0.2 * u, 0.0))
    build_from_averaged(AveragedKernel(xy_kernel, [("y", base)]), ch, lat)
    avg = AveragedKernel(xy_kernel, [("y", sloped)])
    second = build_from_averaged(avg, ch, lat)
    scale = ch.n_ch * n_blocks(second)
    assert computed(second) == n_rows(lat, lambda d, p: sum(p) >= 1) * scale
    assert events(second)["S2 columns reused"] == scale
    same_spectra(second, build_from_averaged(avg, ch, lat, cache=False))


# ------------------------------------------------------------------ T14 item 9: channel rows


def test_a_build_on_overlapping_channels_computes_only_the_new_keys():
    every = small_channels(6, 8)
    parent = mixed(every[[0, 1, 2, 3]])
    block, scale = parent.lattice.index.block_size, n_blocks(parent)
    second = mixed(every[[2, 3, 4, 5]])
    assert computed(second) == 2 * block * scale
    assert events(second)["S2 columns reused"] == 2 * block * scale
    child = parent.at(every[[2, 3, 4, 5]])
    assert counts(child) == (2, 2) and computed(child) == 0
    fresh = mixed(every[[2, 3, 4, 5]], cache=False)
    close_spectra(second, fresh)
    same_spectra(child, second)


def test_at_on_overlapping_channels_computes_only_the_missing_keys(monkeypatch):
    every = small_channels(6, 8)
    parent = mixed(every[[0, 1, 2, 3]])
    seen = []
    for name in ("stack_cell", "jet_lattice"):
        real = getattr(_routes, name)

        def spy(*a, real=real, name=name, **k):
            seen.append((name, a[1].n_ch))
            return real(*a, **k)

        monkeypatch.setattr(_routes, name, spy)
    child = parent.at(every[[2, 3, 4, 5]])
    assert sorted(seen) == [("jet_lattice", 2), ("stack_cell", 2)]
    assert computed(child) == 2 * parent.lattice.index.block_size * n_blocks(parent)
    close_spectra(child, mixed(every[[2, 3, 4, 5]], cache=False))


def _stored_keys(make, monkeypatch):
    cache, seen = default_cache(), []
    real = cache.put_columns

    def spy(context_keys, ids, values, chains):
        seen.extend(str(k) for k in context_keys)
        return real(context_keys, ids, values, chains)

    monkeypatch.setattr(cache, "put_columns", spy)
    basis = make()
    monkeypatch.setattr(cache, "put_columns", real)
    return basis, set(seen)


def test_a_one_bit_change_of_one_channel_weight_changes_one_context_key(monkeypatch):
    ch = small_channels(4, 8)
    w = np.asarray(ch.weights).copy()
    w[1, 3] = np.nextafter(w[1, 3], np.inf)
    other = dataclasses.replace(ch, weights=jnp.asarray(w))
    changed = [a != b for a, b in zip(ch.row_keys(), other.row_keys())]
    assert changed == [False, True, False, False]
    first, keys0 = _stored_keys(lambda: mixed(ch), monkeypatch)
    second, keys1 = _stored_keys(lambda: mixed(other), monkeypatch)
    cells, scale = first.lattice.n_cells, n_blocks(first)
    assert len(keys0) == cells * scale * ch.n_ch
    assert len(keys1) == cells * scale and not keys0 & keys1
    assert computed(second) == first.lattice.index.block_size * scale


def _resolved(ch, lat):
    kern = ContinuumKernel().resolved(ch, float(lat.cells.hi.max()), 5e-6)
    base = fixed_b_avg()
    return AveragedKernel(kern, base.averaged, fixed=dict(base.fixed), phase=base.phase)


def test_a_channel_subset_on_a_jet_cell_computes_zero_columns(request):
    """``x_min`` follows the lowest channel and is a guard only, so it is in no key (decision 159 part 4)."""
    ch, lat = small_channels(5, 8), gamma_lattice(2, 2)
    sub = ch[[3, 4]]
    full, part = _resolved(ch, lat), _resolved(sub, lat)
    assert full.kernel.x_min < part.kernel.x_min
    parent = build_from_averaged(full, ch, lat, routes="jet", azimuth_factor=0.4)
    request.getfixturevalue("no_providers")
    child = build_from_averaged(part, sub, lat, routes="jet", azimuth_factor=0.4)
    assert computed(child) == 0 and events(child)["memo misses"] == 1
    for a, b in ((child.spectra0, parent.spectra0), (child.spectra2, parent.spectra2)):
        assert np.asarray(a).tobytes() == np.asarray(b)[[3, 4]].tobytes()


def test_a_build_on_reversed_channels_gives_the_reversed_design(request):
    ch = small_channels(4, 8)
    first = mixed(ch)
    request.getfixturevalue("no_providers")
    second = mixed(ch[::-1])
    assert computed(second) == 0 and second.key != first.key
    assert (
        np.asarray(second.design()).tobytes()
        == np.asarray(first.design())[::-1].tobytes()
    )


def test_new_channels_the_stack_refuses_move_the_cell_to_the_jet_on_every_channel():
    """The cell of ``test_at_moves_a_cell_the_stack_refuses_on_new_channels_to_the_jet``: the stored stack columns
    of the first build are not read, and every channel of the second build takes the jet.
    """
    lat = Lattice(gamma=Cells.log(1e8 / 1.2, 1e8 * 1.2, 1, order=1), order=1)
    low = Channels.log(0.02, 10, 6, unit="GHz", half_width=0.2, n_nu=32)
    build_from_averaged(fixed_b_avg(), small_channels(), lat, azimuth_factor=0.4)
    second = build_from_averaged(fixed_b_avg(), low, lat, azimuth_factor=0.4)
    route = second.record.routes[0]
    assert route.provider == "jet" and REACH in route.reason
    assert computed(second) == lat.index.block_size * low.n_ch * n_blocks(second)
    plain = build_from_averaged(
        fixed_b_avg(), low, lat, azimuth_factor=0.4, cache=False
    )
    assert plain.record.routes == second.record.routes
    same_spectra(second, plain)


# ------------------------------------------------------------------ the dry walk against the stack walk


def _stage_cases():
    per_node = per_node_case()
    return {
        "cached-S1": (cached_avg(), gamma_lattice(2, 2)),
        "per-node": per_node[:2],
        "field-scaling": (AveragedKernel(K, [eta4()], phase=screen()), field_lattice()),
    }


@pytest.mark.parametrize("mode", ["cached-S1", "per-node", "field-scaling"])
def test_the_dry_walk_reads_the_tables_of_the_stack_walk(mode):
    avg, lat = _stage_cases()[mode]
    ch, memo = small_channels(3, 8), StageMemo(default_cache())
    try:
        for c in range(lat.n_cells):
            assert stack_stage(avg, ch, lat, c, s1="lookup", memo=memo) == mode
            dry = memo.take_reads()
            record = []
            stack_cell(avg, ch, lat, c, memo=memo, record=record)
            assert record == [mode] and memo.take_reads() == dry and dry
    finally:
        memo.close()


def test_the_dry_walk_refuses_where_the_stack_walk_refuses():
    """The amplification refusal at N = 4 of ``test_v080_basis.py``, on both walks with the same reason."""
    avg, lat, ch = _lognormal_avg(0.01, 0.1), gamma_lattice(1, 4), small_channels(3, 8)
    with pytest.raises(StackRefused) as dry:
        stack_stage(avg, ch, lat, 0, s1="lookup", memo={})
    with pytest.raises(StackRefused) as walk:
        stack_cell(avg, ch, lat, 0)
    assert str(dry.value) == str(walk.value) and "amplifies" in str(dry.value)
