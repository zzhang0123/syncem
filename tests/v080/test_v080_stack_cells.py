"""Step 5a of design/0.8/SPEC.md Sec. 10.2, second part: frozen reweighting of ``B`` given celled ``gamma``, the
moving-support refusal, field cells (T5), the stack refusals and the S1 build cost (Sec. 8.2, 8.3).

Split from ``test_v080_stack.py`` in the step-5b-3 review, with the test bodies unchanged (decision 170 keeps
files under 400 lines). The helpers stay in ``test_v080_stack.py``, which other files import.
"""

import time

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.continuum import ContinuumKernel
from syncem.engine._stack_series import StackRefused, stack_cell, stack_lattice
from syncem.engine.average import AveragedKernel, NodeRule, fixed_rule, uniform_rule
from syncem.engine.field_average import FieldAverage
from syncem.engine.provider_jet import jet_cell
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from tests.v080.test_v080_stack import (
    channel_max,
    lognormal_given_gamma,
    rel,
    screen,
    small_channels,
)


@pytest.mark.parametrize(
    ("width", "slope", "n_cells"), [(0.4, 0.02, 3), (0.1, 0.08, 12), (0.02, 0.5, 3)]
)
def test_b_given_celled_gamma_by_frozen_reweighting(width, slope, n_cells):
    """Boundary probe of decision 139: reweighting equals the jet whether the nodes move by 0.03 or 19 of their gaps.

    Errors are measured against each channel's largest value over the lattice (the design-matrix scale). Where a
    cell contributes below about 1e-8 of that scale, fixed and moving Gauss-Hermite nodes resolve the exponential
    tail differently (6e-4 relative at 16 nodes, 8e-11 at 64), a quadrature property of the law that both methods share.
    """
    ch = small_channels()
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        [
            ("B", lognormal_given_gamma(width, slope)),
            ("eta", uniform_rule(0.0, 0.5, 4)),
        ],
        phase=screen(),
    )
    lattice = Lattice(gamma=Cells.log(300, 3e6, n_cells))
    li, lp = stack_lattice(avg, ch, lattice)
    scale_i, scale_p = channel_max(li), channel_max(lp)
    for c in (0, n_cells // 2):
        si, sp = stack_cell(avg, ch, lattice, c)
        ji, jp = jet_cell(avg, ch, lattice, c)
        err_i = np.max(np.abs(np.asarray(si) - np.asarray(ji)) / scale_i)
        err_p = np.max(np.abs(np.asarray(sp) - np.asarray(jp)) / scale_p)
        # probe at 16 law nodes: I 5.0e-10 and P 1.7e-9 at cell 0 of the first case, below 1e-11 elsewhere
        assert err_i < 5e-9 and err_p < 5e-9, (width, slope, c, err_i, err_p)
    naive = stack_cell(avg, ch, lattice, n_cells // 2, reweight=False)[0]
    jet = np.asarray(jet_cell(avg, ch, lattice, n_cells // 2)[0])
    assert (
        np.max(np.abs(np.asarray(naive) - jet) / scale_i) > 1e-6
    )  # frozen weights without reweighting


def test_frozen_reweighting_converges_with_the_law_nodes_in_the_tail():
    def worst(n):
        avg = AveragedKernel(
            ContinuumKernel(x_min=1e-12),
            [
                ("B", lognormal_given_gamma(0.4, 0.02, n)),
                ("eta", uniform_rule(0.0, 0.5, 4)),
            ],
            phase=screen(),
        )
        lattice = Lattice(gamma=Cells.log(300, 3e6, 3))
        return rel(
            stack_cell(avg, small_channels(), lattice, 0)[0],
            jet_cell(avg, small_channels(), lattice, 0)[0],
        )

    assert worst(16) > 1e-5 and worst(64) < 1e-9  # probe: 6.4e-4 and 7.6e-11


def test_moving_support_goes_to_the_jet():
    """A uniform law whose ends move with gamma, with its (correct) density, fails the moving-node check (decision 139)."""
    lo = lambda p: 4e-6 * (p["gamma"] / 1e4) ** 0.1  # noqa: E731
    hi = lambda p: 6e-6 * (p["gamma"] / 1e4) ** 0.1  # noqa: E731
    base = uniform_rule(lo, hi, 8, parents=("gamma",))
    moving = NodeRule(
        base.fn,
        parents=("gamma",),
        density=lambda x, p: jnp.ones_like(x) / (hi(p) - lo(p)),
    )
    ch = small_channels()
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        [("B", moving), ("eta", uniform_rule(0.0, 0.5, 4))],
        phase=screen(),
    )
    lattice = Lattice(gamma=Cells.log(300, 3e6, 3))
    with pytest.raises(StackRefused, match="does not reproduce the moving nodes"):
        stack_cell(avg, ch, lattice, 1)
    with pytest.raises(StackRefused, match="without a density"):
        stack_cell(
            AveragedKernel(
                avg.kernel,
                [("B", base), ("eta", uniform_rule(0.0, 0.5, 4))],
                phase=screen(),
            ),
            ch,
            lattice,
            1,
        )


# ------------------------------------------------------------------ T5: field cells


@pytest.mark.parametrize("chart", ["log", "affine"])
@pytest.mark.parametrize("order", [2, 4])
def test_field_cells_by_the_scaling_relation_equal_the_jet(chart, order):
    ch = small_channels()
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        [("eta", uniform_rule(0.0, 0.5, 6))],
        phase=screen(),
    )
    field = (
        Cells.log("2 uG", "8 uG", 3, order=order)
        if chart == "log"
        else Cells.single("5 uG", scale="1 uG", order=order)
    )
    lattice = Lattice(gamma=Cells.log(300, 3e6, 3, order=order), B=field, order=order)
    for c in (0, lattice.n_cells - 1):
        si, sp = stack_cell(avg, ch, lattice, c)
        ji, jp = jet_cell(avg, ch, lattice, c)
        assert rel(si, ji) < 1e-8 and rel(sp, jp) < 1e-8, (chart, order, c)


def test_field_cells_refuse_a_b_dependent_law():
    eta = uniform_rule(0.0, lambda p: 0.3 + 4e4 * p["B"], 4, parents=("B",))
    avg = AveragedKernel(ContinuumKernel(x_min=1e-12), [("eta", eta)], phase=screen())
    lattice = Lattice(
        gamma=Cells.log(300, 3e6, 2), B=Cells.log("2 uG", "8 uG", 2), order=2
    )
    with pytest.raises(StackRefused, match="field-scaling precondition"):
        stack_cell(avg, small_channels(), lattice, 0)


def test_stack_refusals():
    ch = small_channels(3, 16)
    k = ContinuumKernel(x_min=1e-12)
    with pytest.raises(StackRefused, match="bare function"):
        stack_cell(
            AveragedKernel(
                k,
                [
                    ("B", lambda p: (jnp.array([5e-6]), jnp.array([1.0]))),
                    ("eta", fixed_rule([0.0], [1.0])),
                ],
            ),
            ch,
            Lattice(gamma=Cells.log(300, 3e5, 2)),
            0,
        )
    with pytest.raises(StackRefused, match="celled variables"):
        stack_cell(
            AveragedKernel(k, fixed={"B": 5e-6}),
            ch,
            Lattice(gamma=Cells.log(300, 3e5, 2), eta=Cells.linear(0, 0.5, 2), order=2),
            0,
        )
    with pytest.raises(StackRefused, match="undeclared"):
        stack_cell(
            AveragedKernel(
                k,
                [
                    ("B", uniform_rule(lambda p: 4e-6, 6e-6, 2)),
                    ("eta", fixed_rule([0.0], [1.0])),
                ],
            ),
            ch,
            Lattice(gamma=Cells.log(300, 3e5, 2)),
            0,
        )


# ------------------------------------------------------------------ S1 build cost (SPEC Sec. 8.7)


@pytest.mark.slow
def test_s1_build_cost_by_lookups():
    import os

    from syncem.engine.primitive import PrimitiveTable

    load = os.getloadavg()[0]
    s0 = PrimitiveTable.build(
        *FieldAverage.needed_lnx([4.0, 6.0])
    )  # covers every node set below
    times = {}
    for n in (1, 16, 96):
        t, w = np.polynomial.legendre.leggauss(n)
        start = time.perf_counter()
        FieldAverage.lookups(5.0 + t, 0.5 * w, primitive=s0)
        times[n] = time.perf_counter() - start
    print(f"load {load:.1f}: S1 by lookups {times}")
    assert times[96] < 3 * (0.006 * 96 + 0.5) * max(1.0, load / 4)
