"""Bits of the one-channel jet calls (design/0.8/SPEC.md decision 165, step 5b-3).

A channel's columns depend on its nodes and weights, the cell, the exponent set and the averaged kernel. The
tests check bit for bit that they do not depend on the channel set or the cell set: one channel, the last
channel, the reversed set, the two ends, repeats and a set of 256 channels, at 1, 64, 65 and 400 nodes and
orders 0, 2 and 4.

``batch_columns`` is the formulation of commit fcfe5cf: one program over every channel and every channel node
(``make_routine`` before step 5b-3, without the map over cells). The current columns lie within ``BATCH_TOL``
of it over the channel maximum, at E1, at orders 0 to 5, on 1 and 256 channels, at the frequency ends, at the
field ends and on a 2-d cell. The measured gaps of the 15 cases lie between 0 and 1.33e-15 (6.0 eps, ``B`` fixed
at 0.5 uG on 6 channels of 65 nodes), and E1 at order 4 on 5 channels of 32 nodes, one chunk per channel, is
2.9e-19 (2026-11-19, jax 0.10.0 on CPU, 1-minute load 7.7 to 10.1).
"""

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.engine import provider_jet
from syncem.engine.average import AveragedKernel
from syncem.engine.provider_jet import jet_columns, jet_lattice
from tests.v080.test_v080_jet_reference import e1_kernel, e1_lattice
from tests.v080.test_v080_routes import eta4, fixed_b_avg
from tests.v080.test_v080_stack import screen
from tests.v080.test_v080_stack_boundaries import gamma_lattice

K = ContinuumKernel(x_min=1e-12)
BATCH_TOL = 1e-13  # decision 165, over the channel maximum
SUBSETS = {
    "one channel": [2],
    "the last channel": [5],
    "reversed": [5, 4, 3, 2, 1, 0],
    "the ends": [0, 5],
    "repeats": [1, 1, 4, 1],
}


def chans(n, n_nu, lo=0.1, hi=10.0):
    return Channels.log(lo, hi, n, unit="GHz", half_width=0.2, n_nu=n_nu)


def same(a, b):
    return np.asarray(a).tobytes() == np.asarray(b).tobytes()


def channel_gap(new, old):
    new, old = np.asarray(new), np.asarray(old)
    scale = np.abs(old).max(axis=0)
    diff = np.abs(new - old).max(axis=0)
    assert np.all(diff[scale == 0] == 0)
    return float(np.max(diff[scale > 0] / scale[scale > 0], initial=0.0))


# ------------------------------------------------------------------ independence from the batch


@pytest.mark.parametrize("n_nu", [1, 64, 65, 400])
@pytest.mark.parametrize("order", [0, 2, 4])
def test_columns_do_not_depend_on_the_channel_set_or_the_cell_set(n_nu, order):
    ch, avg, lat = chans(6, n_nu), fixed_b_avg(), gamma_lattice(2, order)
    full = jet_lattice(avg, ch, lat)
    for name, rows in SUBSETS.items():
        part = jet_lattice(avg, ch[rows], lat)
        assert all(same(a, np.asarray(b)[:, rows]) for a, b in zip(part, full)), name
    n = order + 1
    for cells, take in (((1,), [1]), ((1, 0, 1), [1, 0, 1])):
        rows = np.concatenate([np.arange(c * n, c * n + n) for c in take])
        part = jet_lattice(avg, ch, lat, cells=cells)
        assert all(same(a, np.asarray(b)[rows]) for a, b in zip(part, full)), cells


def test_columns_on_256_channels_equal_those_of_any_subset():
    ch, avg, lat = chans(256, 1), fixed_b_avg(), gamma_lattice(1, 2)
    full = jet_lattice(avg, ch, lat)
    for rows in ([255], [0, 128, 255], list(range(250, 256))):
        part = jet_lattice(avg, ch[rows], lat)
        assert all(same(a, np.asarray(b)[:, rows]) for a, b in zip(part, full)), rows


# ------------------------------------------------------------------ against one program over every channel


def batch_columns(avg, ch, axes, exponents):
    """The columns of one program over every channel and channel node, as at commit fcfe5cf."""
    names = tuple(axes)
    charts = tuple(axes[v][0] for v in names)
    ref = jnp.asarray([axes[v][1] for v in names], dtype=float)
    hw = jnp.asarray([axes[v][2] for v in names], dtype=float)
    exps = tuple(tuple(e) for e in exponents)

    @eqx.filter_jit
    def run(avg, ch, ref, hw):
        def f(z):
            modes = avg(ch, provider_jet._point(names, charts, ref, hw, z))
            return modes.I, modes.P[0], modes.V

        return provider_jet.taylor_coefficients(f, len(names), exps)

    return run(avg, ch, ref, hw)


def b_fixed(b):
    return AveragedKernel(K, [eta4()], fixed={"B": b}, phase=screen())


def eta_only():
    return AveragedKernel(K, [eta4()], phase=screen())


def one_d(order, gamma=3e3):
    return {"gamma": ("log", gamma, 0.24)}, [(n,) for n in range(order + 1)]


def two_d(order):
    axes = {"gamma": ("log", 3e3, 0.24), "B": ("log", 4e-6, 0.35)}
    return axes, [(r, s) for r in range(order + 1) for s in range(order + 1 - r)]


def e1_cell_9(order):
    t = e1_lattice(order).cells
    axes = {"gamma": ("log", float(t.reference[9, 0]), float(t.half_width[9, 0]))}
    return axes, [(n,) for n in range(order + 1)]


BATCH_CASES = {
    # name: (averaged kernel, channels, (axes, exponents), slow)
    "E1 cell 9, order 4, 5x32": (e1_kernel, lambda: chans(5, 32), e1_cell_9(4), False),
    "E1 cell 9, order 2, 3x65": (e1_kernel, lambda: chans(3, 65), e1_cell_9(2), False),
    "order 0, 4x400": (fixed_b_avg, lambda: chans(4, 400), one_d(0), False),
    "order 1, 4x65": (fixed_b_avg, lambda: chans(4, 65), one_d(1), False),
    "order 3, 4x130": (fixed_b_avg, lambda: chans(4, 130), one_d(3), False),
    "order 4, 4x400": (fixed_b_avg, lambda: chans(4, 400), one_d(4), False),
    "order 5, 2x65": (fixed_b_avg, lambda: chans(2, 65), one_d(5), True),
    "1 channel of 400 nodes, order 2": (
        fixed_b_avg,
        lambda: chans(1, 400, 1.0, 1.0),
        one_d(2),
        False,
    ),
    "256x1, order 2": (fixed_b_avg, lambda: chans(256, 1), one_d(2), False),
    "5 to 50 MHz, 6x65, order 2": (
        fixed_b_avg,
        lambda: chans(6, 65, 0.005, 0.05),
        one_d(2),
        False,
    ),
    "100 to 1000 GHz, 6x65, order 2, gamma 1e5": (
        fixed_b_avg,
        lambda: chans(6, 65, 100.0, 1000.0),
        one_d(2, 1e5),
        False,
    ),
    "B 0.5 uG, 6x65, order 2": (
        lambda: b_fixed(5e-7),
        lambda: chans(6, 65),
        one_d(2),
        False,
    ),
    "B 5 mG, 6x65, order 2": (
        lambda: b_fixed(5e-3),
        lambda: chans(6, 65),
        one_d(2),
        False,
    ),
    "2-d, 4x65, order 2": (eta_only, lambda: chans(4, 65), two_d(2), False),
    "2-d, 3x65, order 4": (eta_only, lambda: chans(3, 65), two_d(4), True),
}
BATCH_PARAMS = [
    pytest.param(name, marks=[pytest.mark.slow] if case[3] else [])
    for name, case in BATCH_CASES.items()
]


@pytest.mark.parametrize("name", BATCH_PARAMS)
def test_columns_lie_within_the_bound_of_one_program_over_every_channel(name):
    avg_of, ch_of, (axes, exponents), _ = BATCH_CASES[name]
    avg, ch = avg_of(), ch_of()
    got = jet_columns(avg, ch, axes, exponents, with_v=True)
    ref = batch_columns(avg, ch, axes, exponents)
    for a, b in zip(got, ref):
        assert np.all(np.isfinite(np.asarray(a)))
        assert channel_gap(a, b) < BATCH_TOL
