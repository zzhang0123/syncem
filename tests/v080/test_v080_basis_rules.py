"""Bases built with column rules, ``with_order`` on uneven orders, ``at`` on repeated new channels (decisions 148
to 153).

A basis built with ``ColumnRules`` and the same basis built with ``ColumnRules()`` call the providers on the
same lattice with the same routes, so each rules-on design column equals, bitwise, the rules-off column of its
representative row (exact identity of S3 assembly, which is selection and scaling). ``with_order(N')`` caps
every per-cell order of the non-depth variables at ``N'`` and sets the total to ``N'``, so the lower-order rows
are a subset of the built rows and need no provider call (decision 153). Probes ran on 2026-11-19 at a 1-minute
load of 2.3 to 4.1 (``probe_rules_cases.py`` and ``probe_basis_rules.py`` in the session scratchpad). Each
tolerance names its probe value and margin, a SPEC number or an exact identity.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.capabilities import Beaming, Parity
from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.user import kernel
from syncem.base.variables import SYNCHROTRON_VARIABLES as SV
from syncem.base.variables import Variable
from syncem.engine import _routes
from syncem.engine.assemble import predict_values
from syncem.engine.average import AveragedKernel, uniform_rule
from syncem.engine.basis import build_from_averaged
from syncem.engine.columns import ColumnRules
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.moments import Moments
from syncem.lattice.representations import Legendre
from tests.v080._column_cases import N_MU, cosine_kernel
from tests.v080.test_v080_basis import counts, mixed_basis
from tests.v080.test_v080_routes import cached_avg
from tests.v080.test_v080_stack import small_channels

K = ContinuumKernel(x_min=1e-12)


def channels():
    return Channels.log(0.1, 10, 4, unit="GHz", half_width=0.2, n_nu=16)


@pytest.fixture
def no_providers(monkeypatch):
    for name in ("stack_cell", "core_cell", "jet_lattice"):
        monkeypatch.setattr(_routes, name, lambda *a, **k: pytest.fail("provider call"))


@kernel(
    variables=[Variable("a", kind="positive"), SV["mu"], SV["eta"]],
    stokes=("I", "P"),
    capabilities=(Parity(("I", "P"), ("V",)), Beaming("mu", "eta")),
)
def beamed(nu, a, mu, eta):
    """Symmetric under mu <-> eta, so the (l, k) and (k, l) projections agree up to summation order."""
    s = (nu / 1e9) ** -0.7
    return {
        "I": a * s * (1 + 0.5 * mu * eta + 0.3 * (mu**2 + eta**2)),
        "P": 0.4 * a * s * (1 + 0.2 * mu * eta),
    }


@kernel(
    variables=[Variable("a", kind="positive"), Variable("b", kind="positive")],
    stokes=("I", "P"),
)
def ab(nu, a, b):
    s = (nu / 1e9) ** -0.7
    return {"I": a * b**2 * s, "P": (0.3 + 0.1j) * a**2 * b * s}


def mask_case(order=None):
    """The cosine kernel of the column tests with an even ``eta`` law and one masked cosine."""
    a = (
        Cells.log(1.0, 10.0, 2)
        if order is None
        else Cells.log(1.0, 10.0, 2, order=order)
    )
    avg = AveragedKernel(
        cosine_kernel(),
        [("eta", uniform_rule(-1.0, 1.0, 8))],
        legendre={"mu": (4, N_MU)},
    )
    lat = Lattice(a=a, mu=Legendre(4), order=order)
    return avg, lat, ColumnRules(masks=(("mu", "single"),)), None


def depth_case(order=1):
    """Two energy cells by two depth cells at depth order 1, outside the total. I does not read depth, so its
    spectra are shared across depth."""
    avg = AveragedKernel(
        K, [("B", uniform_rule(4e-6, 6e-6, 4)), ("eta", uniform_rule(0.0, 0.5, 4))]
    )
    lat = Lattice(
        gamma=Cells.log(300, 3e5, 2, order=order),
        depth=Cells.linear(6, 10, 2, order=1),
        order=order,
    )
    return avg, lat, ColumnRules(depth_free_spin0=True), 0.4


def merge_case(order=None):
    """Two resolved cosines with a joint mask and the Beaming merge of (l, k) with (k, l)."""
    avg = AveragedKernel(beamed, [], legendre={"mu": (2, 8), "eta": (2, 8)})
    lat = Lattice(a=Cells.log(1.0, 10.0, 2), mu=Legendre(2), eta=Legendre(2))
    rules = ColumnRules(masks=(("mu", "joint"), ("eta", "joint")), merge=("mu", "eta"))
    return avg, lat, rules, None


CASES = {"mask": mask_case, "depth": depth_case, "merge": merge_case}


def pair(case):
    avg, lat, rules, factor = case
    common = dict(routes="jet", azimuth_factor=factor)
    off = build_from_averaged(avg, channels(), lat, **common)
    on = build_from_averaged(avg, channels(), lat, rules=rules, **common)
    return lat, on, off


# ------------------------------------------------------------------ rules on against rules off


@pytest.mark.parametrize("name", sorted(CASES))
def test_rules_on_design_columns_equal_the_rules_off_columns_of_their_rows(name):
    """Design: exact identity. Prediction with moments in ``U[0.5, 1.5]``: the masked rows of the mask case
    are zero up to the quadrature rounding of ``test_mask_dispatcher_rules_on_against_off``, and merged rows
    agree up to summation order. Probe: 1.2e-16 (mask), 0.0 (depth) and 1.7e-16 (merge) of the per-channel
    maximum of ``sum |D| |a|``. Bound 1e-14, margin 60.
    """
    lat, on, off = pair(CASES[name]())
    cm = on.columns
    assert cm.n < off.columns.n or cm.n0 < off.columns.n0  # the rules change the map
    reps = cm.col_rows[cm.col_start[:-1]]
    d_on, d_off = np.asarray(on.design()), np.asarray(off.design())
    np.testing.assert_array_equal(d_on, d_off[:, :, off.columns.col_of_row[reps]])
    a = np.random.default_rng(3).uniform(0.5, 1.5, lat.n)
    p_on = np.asarray(predict_values(on, Moments(lat, a)))
    p_off = np.asarray(predict_values(off, Moments(lat, a)))
    scale = np.einsum("csn,n->cs", np.abs(d_off), a).max(axis=1, keepdims=True)
    assert np.max(np.abs(p_on - p_off) / scale) < 1e-14


def test_depth_sharing_stores_one_spin0_spectrum_per_energy_monomial():
    """2 energy cells by 2 monomials keep 4 spin-0 spectra for 16 columns (Sec. 4.3)."""
    _, on, off = pair(depth_case())
    assert (on.columns.n, on.columns.n0, off.columns.n0) == (16, 4, 16)
    assert np.shape(on.spectra0) == (4, 1, 4) and np.shape(on.spectra2) == (4, 16)


# ------------------------------------------------------------------ with_order on rules-on parents


@pytest.mark.parametrize("name", ["mask", "depth"])
def test_with_order_on_a_rules_on_parent_equals_a_fresh_build(name, request):
    """Order 2 to 1. Same provider, same routes and the stored spectra of the matching rows. Probe: 0.0 for
    both cases. Bound 1e-15 of the largest column per channel and Stokes parameter, as ``at`` and a fresh
    build need not agree bitwise (decision 153)."""
    make = CASES[name]
    avg, lat, rules, factor = make(2)
    common = dict(routes="jet", rules=rules, azimuth_factor=factor)
    parent = build_from_averaged(avg, channels(), lat, **common)
    fresh = build_from_averaged(avg, channels(), make(1)[1], **common)
    request.getfixturevalue("no_providers")
    child = parent.with_order(1)
    assert child.columns == fresh.columns
    assert (child.columns.n0, child.columns.n2) == (fresh.columns.n0, fresh.columns.n2)
    d, ref = np.asarray(child.design()), np.asarray(fresh.design())
    assert np.max(np.abs(d - ref) / np.abs(ref).max(axis=(1, 2), keepdims=True)) < 1e-15


# ------------------------------------------------------------------ with_order on uneven orders


def uneven(a_order=4, b_order=1, order=4):
    return Lattice(
        a=Cells.log(1.0, 10.0, 2, order=a_order),
        b=Cells.log(1.0, 3.0, 2, order=b_order),
        order=order,
    )


@pytest.mark.parametrize("N", [4, 3, 2, 1, 0])
def test_with_order_caps_uneven_axis_orders(N, request):
    """``a`` at order 4, ``b`` at order 1, total 4. The child equals a fresh build on ``a`` at ``min(4, N)``,
    ``b`` at ``min(1, N)`` and total ``N``. Probe: 0.0 for every N. Bound 1e-15 as above (decision 153).
    """
    avg = AveragedKernel(ab)
    parent = build_from_averaged(avg, channels(), uneven())
    capped = uneven(min(4, N), min(1, N), N)
    fresh = build_from_averaged(avg, channels(), capped)
    request.getfixturevalue("no_providers")
    child = parent.with_order(N)
    assert child.lattice.key == capped.key and child.lattice.order == N
    d, ref = np.asarray(child.design()), np.asarray(fresh.design())
    assert d.shape == ref.shape
    assert np.max(np.abs(d - ref) / np.abs(ref).max(axis=(1, 2), keepdims=True)) < 1e-15


def _row_keys(lattice):
    a, n = lattice.index.arrays, lattice.index.n
    return [
        (int(a.cell[k]), int(a.pair[k]), tuple(map(int, a.powers[k]))) for k in range(n)
    ]


@pytest.mark.parametrize("N", [3, 2])
def test_with_order_on_cells_of_different_orders_selects_the_capped_rows(N, request):
    """A corner split of cells at orders (4, 2) in ``a`` and 1 in ``b``. The child holds exactly the parent rows
    whose powers are at most ``min(own order, N)`` per variable with total at most ``N``, and each child column
    is the parent column of the same row (exact identity of a gather)."""
    lat = Lattice(
        a=Cells.log(1.0, 10.0, 2, order=(4, 2)),
        b=Cells.log(1.0, 3.0, 2, order=1),
        order=4,
    ).refine(where={"a": 0, "b": 0}, a=2)
    parent = build_from_averaged(AveragedKernel(ab), channels(), lat)
    request.getfixturevalue("no_providers")
    child = parent.with_order(N)
    t = lat.cells
    old = _row_keys(lat)

    def kept(key):
        cell, _, powers = key
        caps = [min(o, N) for o in t.order[cell]]
        return all(p <= c for p, c in zip(powers, caps)) and sum(powers) <= N

    new = _row_keys(child.lattice)
    assert sorted(new) == sorted(k for k in old if kept(k))
    d, d0 = np.asarray(child.design()), np.asarray(parent.design())
    np.testing.assert_array_equal(d, d0[:, :, [old.index(k) for k in new]])


def test_with_order_on_a_lattice_without_a_total_order_names_the_cause():
    @kernel(
        variables=[Variable("a", kind="positive"), Variable("depth", unit="rad m^-2")],
        stokes=("I", "P"),
        depth="depth",
    )
    def depthy(nu, a):
        return {"I": a * jnp.ones_like(nu), "P": 0.3 * a * jnp.ones_like(nu)}

    lat = Lattice(depth=Cells.linear(-1.0, 1.0, 2))
    assert lat.order is None
    basis = build_from_averaged(
        AveragedKernel(depthy, fixed={"a": 1.0}), channels(), lat
    )
    with pytest.raises(ValueError, match="no total order"):
        basis.with_order(0)


# ------------------------------------------------------------------ at on repeated new channels


def test_at_on_repeated_new_channels_fills_every_repeat():
    """Channels 1 and 3 of the wide set are new and each appears twice. Probe: 0.0 against a fresh build for I
    and for P. Bound 1e-15 as in ``test_at_on_new_channels_computes_only_the_new_rows``.
    """
    narrow, wide = (
        Channels.log(0.1, 10, n, unit="GHz", half_width=0.2, n_nu=32) for n in (5, 9)
    )
    target = wide[[1, 1, 0, 3, 3]]
    child, fresh = mixed_basis(narrow).at(target), mixed_basis(target)
    assert counts(child) == (2, 1)
    d, ref = np.asarray(child.design()), np.asarray(fresh.design())
    for s in range(d.shape[1]):  # I, then Q and U
        assert np.all(np.abs(d[:, s]).max(axis=1) > 0)
        err = np.abs(d[:, s] - ref[:, s]) / np.abs(ref[:, s]).max(axis=1, keepdims=True)
        assert np.max(err) < 1e-15


# ------------------------------------------------------------------ the cache argument in step 5b-1


def test_cache_false_records_its_note_and_a_non_bool_cache_raises():
    ch, avg, lat = small_channels(), cached_avg(), Lattice(gamma=Cells.log(300, 3e5, 2))
    basis = build_from_averaged(avg, ch, lat, stokes="I", cache=False)
    assert basis.record.notes == ("not cached: cache=False",)
    assert basis.key is None and dict(basis.record.defaults)["cache"] is False
    for bad in (0, 1, "yes", None):
        with pytest.raises(ValueError, match="cache must be a bool"):
            build_from_averaged(avg, ch, lat, stokes="I", cache=bad)
