"""The bound on builds with S2 column reuse (design/0.8/SPEC.md decision 164, T14).

A reused column carries the bits of the provider call that stored it, which ran on another channel set or
exponent set. A ``cache=False`` build depends on its batch in the same way, because XLA compiles one program
per channel count and per exponent set of the jet, and the stack's field-scaling mode and its one-channel
calls change bits too. The distance of a reused build from the ``cache=False`` build is therefore the batch
dependence of an uncached build, measured over the channel maximum.

``REUSE_TOL`` bounds that gap for builds to order 5 on the jet and order 6 on the stack, the range of the sweep.
It is a tested ceiling, not a derived bound. Measured worst gaps of the step-5b-2 sweep and its review over the
highest order ``N`` of the builds that computed the columns (1-minute load 2.1 to 28, loaded), with the margin of
``REUSE_TOL``. "256x1" is 256 channels of 0.1 to 10 GHz with one node each:

    N <= 2   2.24e-15   jet, P block, 5 channels after channel 2 (28 ulps)    margin 45
    N  = 3   6.30e-15   jet, B 0.5 uG, 256x1 then the top 8                   margin 16
    N  = 4   1.81e-14   jet, B 0.5 uG, 256x1 then the top 8                   margin 5.5
    N  = 5   3.85e-14   jet, B 0.5 uG, 256x1 then the top 8 (173 eps)         margin 2.6

The gap scatters by more than a factor 10 between nearby configurations (2.5e-15 to 3.9e-14 at order 5), so
these are the worst of the cases run. The jet does not compile at order 6 under 12 GB, so order 6 is measured
on the stack only, where every gap is below 1.1 eps. A channel whose stored spectra are all zero (the P block
below 50 MHz behind the screen) is compared bitwise. What stays bitwise: a memo hit equals the build that stored
it, a build without reuse on distinct channels equals the ``cache=False`` build, and a reused row equals the
row of the build that stored it. A repeated channel is computed once, so a build on a channel set with repeats
equals the ``cache=False`` build on its distinct channels, gathered.
"""

import numpy as np
import pytest

import _study
from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.phase import FaradayRotation
from syncem.engine.average import AveragedKernel, fixed_rule
from syncem.engine.basis import build_from_averaged
from syncem.engine.cache import default_cache
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import AzimuthHarmonic
from syncem.model.phase import GaussianScreen
from tests.v080.test_v080_cache import per_node_case
from tests.v080.test_v080_routes import cached_avg, eta4, field_lattice
from tests.v080.test_v080_stack import screen
from tests.v080.test_v080_stack_boundaries import gamma_lattice

K = ContinuumKernel(x_min=1e-12)
REUSE_TOL = 1e-13  # decision 164, over the channel maximum


def chans(n, lo=0.1, hi=10.0, n_nu=8):
    return Channels.log(lo, hi, n, unit="GHz", half_width=0.2, n_nu=n_nu)


def fixed_b(b):
    return AveragedKernel(K, [eta4()], fixed={"B": b}, phase=screen())


def eta_only():
    return AveragedKernel(K, [eta4()], phase=screen())


def field_2d(order, lo="2 uG", hi="8 uG"):
    """One gamma cell and one ``B`` cell, both celled, so the jet takes the 2-d cell."""
    b = Cells.log(lo, hi, 1, order=order)
    return Lattice(gamma=Cells.log(300, 3e5, 1, order=order), B=b, order=order)


def spectra(basis):
    return [np.asarray(s) for s in (basis.spectra0, basis.spectra2) if s is not None]


def gap(a, b):
    """Largest difference over the largest stored spectrum of each channel of ``b``.

    A channel whose spectra in ``b`` are all zero must be zero in ``a`` too."""
    out = 0.0
    for x, y in zip(spectra(a), spectra(b), strict=True):
        assert np.all(np.isfinite(x)) and np.all(np.isfinite(y))
        scale = np.abs(y).reshape(y.shape[0], -1).max(axis=1)
        diff = np.abs(x - y).reshape(y.shape[0], -1).max(axis=1)
        assert np.all(diff[scale == 0] == 0)
        out = max(out, float(np.max(diff[scale > 0] / scale[scale > 0], initial=0.0)))
    return out


def bitwise(a, b, rows=slice(None)):
    return all(
        x.tobytes() == y[rows].tobytes()
        for x, y in zip(spectra(a), spectra(b), strict=True)
    )


def events(basis):
    return dict(basis.record.events)


# ------------------------------------------------------------------ what stays bitwise


def _case(name):
    """(averaged kernel, lattice of order N, routes) of the routes in the bitwise tests."""
    return {
        "jet": lambda n: (fixed_b(5e-6), gamma_lattice(2, n), "jet"),
        "cached-S1": lambda n: (cached_avg(), gamma_lattice(2, n), "auto"),
        "field-scaling": lambda n: (eta_only(), field_lattice(order=n), "auto"),
    }[name]


def _build(name, ch, order, cache=True):
    avg, lat, routes = _case(name)(order)
    return build_from_averaged(
        avg, ch, lat, routes=routes, azimuth_factor=0.4, cache=cache
    )


ROUTES = ["jet", "cached-S1", "field-scaling"]


@pytest.mark.parametrize("name", ROUTES)
def test_a_memo_hit_replays_the_bits_of_a_build_that_reused_columns(name):
    ch = chans(5)
    _build(name, ch[[2]], 2)
    reused = _build(name, ch, 2)
    assert events(reused)["S2 columns reused"] > 0
    hit = _build(name, ch, 2)
    assert events(hit)["memo hits"] == 1 and bitwise(hit, reused)
    assert gap(reused, _build(name, ch, 2, cache=False)) < REUSE_TOL


@pytest.mark.parametrize("name", ROUTES)
@pytest.mark.parametrize("lo, hi", [(0.005, 0.05), (100.0, 1000.0)])
def test_a_build_without_reuse_equals_the_uncached_build_at_the_frequency_ends(
    name, lo, hi
):
    ch = chans(3, lo, hi)
    first = _build(name, ch, 2)
    assert events(first)["S2 columns reused"] == 0
    assert bitwise(first, _build(name, ch, 2, cache=False))


@pytest.mark.parametrize("name", ROUTES)
def test_reused_rows_equal_the_rows_of_the_uncached_build_on_the_stored_channels(name):
    """A subset is a row selection of the stored build. Its gap to the subset's own build is batch dependence."""
    ch, rows = chans(6), [1, 2, 4]
    _build(name, ch, 2)
    child = _build(name, ch[rows], 2)
    assert events(child)["S2 columns computed"] == 0
    assert bitwise(child, _build(name, ch, 2, cache=False), rows)
    assert gap(child, _build(name, ch[rows], 2, cache=False)) < REUSE_TOL


def test_a_repeated_channel_is_computed_once_and_equals_the_uncached_distinct_build():
    """Review probe: on the jet, ``[0, 1, 1, 2, 3]`` reuses no column and is 2.2e-15 from ``cache=False``."""
    ch, rows = chans(5), [0, 1, 1, 2, 3]
    first = _build("jet", ch[rows], 2)
    assert events(first)["S2 columns reused"] == 0
    assert bitwise(
        first, _build("jet", ch[[0, 1, 2, 3]], 2, cache=False), [0, 1, 1, 2, 3]
    )
    assert gap(first, _build("jet", ch[rows], 2, cache=False)) < REUSE_TOL


def test_a_lower_order_after_a_higher_one_equals_the_uncached_higher_build_gathered():
    ch = chans(3)
    _build("jet", ch, 3)
    low = _build("jet", ch, 2)
    assert events(low)["S2 columns computed"] == 0
    assert bitwise(low, _build("jet", ch, 3, cache=False).with_order(2))


# ------------------------------------------------------------------ the bound, swept


def j(b=5e-6):
    """Two gamma cells on the jet with ``B`` fixed at ``b`` gauss."""
    return lambda n: (fixed_b(b), gamma_lattice(2, n), "jet")


def _sweep():
    """name: (setup, seeds [(channels, order)], target (channels, order), slow)."""
    c5, c6, c41, c256 = chans(5), chans(6), chans(41), chans(256)
    c256x1 = chans(256, n_nu=1)
    lo, hi = chans(6, 0.005, 0.05), chans(6, 100.0, 1000.0)
    return {
        "jet order 0 then 2": (j(), [(c6[:3], 0)], (c6[:3], 2), False),
        "jet order 2 then 1": (j(), [(c6[:3], 2)], (c6[:3], 1), False),
        "jet 6 channels then 2": (j(), [(c6, 2)], (c6[[3, 4]], 2), False),
        "jet 1 channel then 6": (j(), [(c6[[0]], 2)], (c6, 2), False),
        "jet 1 channel then 5, the order-2 worst": (
            j(),
            [(c5[[2]], 2)],
            (c5, 2),
            False,
        ),
        "jet 5 MHz to 50 MHz, 6 then 2": (j(), [(lo, 2)], (lo[[3, 4]], 2), False),
        "jet 100 to 1000 GHz, 6 then 2": (j(), [(hi, 2)], (hi[[3, 4]], 2), False),
        "jet 0.5 uG, 6 then 2": (j(5e-7), [(c6, 2)], (c6[[3, 4]], 2), False),
        "jet 5 mG, 6 then 2": (j(5e-3), [(c6, 2)], (c6[[3, 4]], 2), False),
        "stack cached-S1, 1 then 6": (
            _case("cached-S1"),
            [(c6[[2]], 2)],
            (c6, 2),
            False,
        ),
        "stack field-scaling, order 3 then 2": (
            _case("field-scaling"),
            [(c6[:3], 3)],
            (c6[:3], 2),
            False,
        ),
        "stack per-node, 1 then 6": (
            lambda n: (per_node_case()[0], gamma_lattice(3, n), "auto"),
            [(c6[[2]], 2)],
            (c6, 2),
            False,
        ),
        "jet order 4 then 2, 50 uG": (j(5e-5), [(c6, 4)], (c6, 2), True),
        "jet order 4 then 2, 0.5 uG": (j(5e-7), [(c6, 4)], (c6, 2), True),
        "jet order 4 then 2, 5 MHz to 50 MHz": (j(), [(lo, 4)], (lo, 2), True),
        "jet order 4 then 2, 100 to 1000 GHz": (j(), [(hi, 4)], (hi, 2), True),
        "jet order 2 then 4": (j(), [(c6[:3], 2)], (c6[:3], 4), True),
        "jet 41 channels then the top 5, order 4": (
            lambda n: (
                cached_avg(),
                Lattice(gamma=Cells.log(300, 3e6, 2, order=n), order=n),
                "jet",
            ),
            [(c41, 4)],
            (c41[36:], 4),
            True,
        ),
        "jet 41 channels then 37, order 4": (
            lambda n: (
                cached_avg(),
                Lattice(gamma=Cells.log(300, 3e6, 2, order=n), order=n),
                "jet",
            ),
            [(c41, 4)],
            (c41[2:39], 4),
            True,
        ),
        "jet 256 channels then the top one": (j(), [(c256, 2)], (c256[[255]], 2), True),
        "jet 256x1 then the top 8, order 3, 0.5 uG": (
            j(5e-7),
            [(c256x1, 3)],
            (c256x1[248:], 3),
            True,
        ),
        "jet 256x1 then the top 8, order 4, 0.5 uG": (
            j(5e-7),
            [(c256x1, 4)],
            (c256x1[248:], 4),
            True,
        ),
        "jet 256x1 then the top 5, order 4": (
            lambda n: (
                cached_avg(),
                Lattice(gamma=Cells.log(300, 3e6, 2, order=n), order=n),
                "jet",
            ),
            [(c256x1, 4)],
            (c256x1[251:], 4),
            True,
        ),
        "jet 2-d, 2 mG to 2 G, 1 channel then 6, order 3": (
            lambda n: (eta_only(), field_2d(n, "2 mG", "2 G"), "jet"),
            [(c6[[2]], 3)],
            (c6, 3),
            True,
        ),
        "jet 2-d, 2 to 8 uG, 1 channel then 6": (
            lambda n: (eta_only(), field_2d(n), "jet"),
            [(c6[[2]], 2)],
            (c6, 2),
            True,
        ),
    }


SWEEP = [
    pytest.param(name, marks=[pytest.mark.slow] if case[3] else [])
    for name, case in _sweep().items()
]


@pytest.mark.parametrize("name", SWEEP)
def test_a_reused_build_is_within_the_bound_of_the_uncached_build(name):
    setup, seeds, (ch, order), _ = _sweep()[name]

    def build(c, n, cache=True):
        avg, lat, routes = setup(n)
        return build_from_averaged(
            avg, c, lat, routes=routes, azimuth_factor=0.4, cache=cache
        )

    for c, n in seeds:
        build(c, n)
    reused = build(ch, order)
    assert events(reused)["S2 columns reused"] > 0
    assert gap(reused, build(ch, order, cache=False)) < REUSE_TOL


ORDER_5 = {
    # name: (averaged kernel, channels, rows of the subset). Probe gaps and peak RSS in the comments.
    "41x8 then the top 5": (cached_avg, lambda: chans(41), 36),  # 8.02e-15, 6.2 GB
    "256x1 then the top 8, 0.5 uG": (
        lambda: fixed_b(5e-7),
        lambda: chans(256, n_nu=1),
        248,
    ),  # 3.85e-14 (173 eps, margin 2.6), the largest measured gap, 5.3 GB
}


@pytest.mark.slow
@pytest.mark.parametrize("name", list(ORDER_5))
def test_order_5_on_the_jet_is_within_the_bound(name):
    """Order 5 is the end of the measured jet range. The test asserts the bound only, not the size of the gap."""
    avg, channels, start = ORDER_5[name]
    ch = channels()
    lat = Lattice(gamma=Cells.log(300, 3e6, 2, order=5), order=5)

    def build(c, cache=True):
        return build_from_averaged(
            avg(), c, lat, routes="jet", azimuth_factor=0.4, cache=cache
        )

    full = build(ch, cache=False)
    build(ch)
    child = build(ch[start:])
    assert events(child)["S2 columns reused"] > 0
    assert bitwise(child, full, slice(start, None))
    assert gap(child, build(ch[start:], cache=False)) < REUSE_TOL


# ------------------------------------------------------------------ the paper ensembles (auto routes, stack)


def _paper(ens, order, rows=None, cache=True):
    cfg = _study.paper()
    b, wb = cfg.b_nodes_weights
    levels = [("eta", fixed_rule(*cfg.eta_nodes_weights))]
    gamma = Cells.log(300, cfg.gamma_hi, 19, order=order)
    lattice = {
        "E1": lambda: Lattice(gamma=gamma),
        "E5": lambda: Lattice(gamma=gamma, phi=AzimuthHarmonic()),
        "E6": lambda: Lattice(
            gamma=gamma, B=Cells.single("5 uG", scale="1 uG", order=order), order=order
        ),
    }[ens]()
    if ens != "E6":
        levels.insert(0, ("B", fixed_rule(b * 1e-6, wb)))
    avg = AveragedKernel(
        K,
        levels,
        phase=FaradayRotation(screen=GaussianScreen(cfg.screen_mean, cfg.screen_sd)),
    )
    ch = Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2, n_nu=400)
    ch = ch if rows is None else ch[[int(r) for r in rows]]
    factor = None if ens == "E5" else cfg.azimuth_factor
    return build_from_averaged(avg, ch, lattice, azimuth_factor=factor, cache=cache)


@pytest.mark.slow
@pytest.mark.parametrize("ens", ["E1", "E5", "E6"])
def test_the_paper_ensembles_after_order_6_and_after_the_fitted_channels(ens):
    """The step-5b-2 probe seeded order 4 and gave E1 and E5 bitwise and E6 (field-scaling) 6.5e-18. The test
    seeds order 6."""
    fit = np.asarray(_study.paper().fit_idx)
    _paper(ens, 6)
    low = _paper(ens, 2)
    assert events(low)["S2 columns computed"] == 0
    assert gap(low, _paper(ens, 2, cache=False)) < REUSE_TOL
    default_cache().clear()
    _paper(ens, 2, rows=fit)
    every = _paper(ens, 2)
    assert events(every)["S2 columns reused"] > 0
    assert gap(every, _paper(ens, 2, cache=False)) < REUSE_TOL
    assert bitwise(_paper(ens, 2, rows=fit), every, fit)
