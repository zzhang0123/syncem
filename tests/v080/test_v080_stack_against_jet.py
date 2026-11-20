"""The stack provider against the jet on the affine chart, at orders 0, 5 and 6, for eta given B, on field cells and at strong fields.

The cases follow design/0.8/SPEC.md Sec. 8.3 and 8.8. Tolerances come from probes on 2026-11-18 (load 16 to
24), and the measured value is in a comment or docstring next to each tolerance. Above order 4 the reference is
the quadrature S1 (decision 141). Split from ``test_v080_stack_boundaries.py`` for the memory target of decision
170.
"""

import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.engine._stack_series import stack_cell, stack_lattice
from syncem.engine.average import AveragedKernel, uniform_rule
from syncem.engine.provider_jet import jet_cell, jet_lattice
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from tests.v080.test_v080_stack import (
    channel_max,
    lognormal_given_gamma,
    rel,
    screen,
    small_channels,
)
from tests.v080.test_v080_stack_boundaries import (
    K,
    eta4,
    gamma_lattice,
)

# ------------------------------------------------------------------ 4. affine energy chart


@pytest.mark.parametrize("mode", ["cached", "per-node"])
@pytest.mark.parametrize("order", [2, 4])
def test_affine_energy_chart_equals_the_jet(mode, order):
    """``Cells.linear(1e3, 1e5, 3)``: cached S1 (uniform B) and per-node (log-normal B given gamma, width 0.4).

    Probe: cached 9.5e-12 / 1.0e-11 (N = 2), 9.3e-12 / 7.2e-12 (N = 4), per-node 6.4e-12 / 6.5e-12 (N = 2),
    3.4e-11 / 4.4e-11 (N = 4). Per-node at N = 4: a flipped ``eps`` in ``log_to_affine`` gives 1.8, a dropped top
    log coefficient 0.24, no reweighting 0.17.
    """
    b = (
        uniform_rule(4e-6, 6e-6, 6)
        if mode == "cached"
        else lognormal_given_gamma(0.4, 0.1)
    )
    avg = AveragedKernel(K, [("B", b), eta4()], phase=screen())
    ch, lattice = small_channels(), gamma_lattice(3, order, 1e3, 1e5, chart="affine")
    si, sp = stack_lattice(avg, ch, lattice)
    ji, jp = jet_lattice(avg, ch, lattice)
    assert rel(si, ji) < 5e-10 and rel(sp, jp) < 5e-10, (rel(si, ji), rel(sp, jp))


# ------------------------------------------------------------------ 5. orders 0, 5 and 6


@pytest.mark.parametrize("mode", ["cached", "per-node"])
def test_order_0_equals_the_jet(mode):
    b = (
        uniform_rule(4e-6, 6e-6, 6)
        if mode == "cached"
        else lognormal_given_gamma(0.4, 0.1)
    )
    avg = AveragedKernel(K, [("B", b), eta4()], phase=screen())
    ch, lattice = small_channels(), gamma_lattice(4, 0)
    si, sp = stack_lattice(avg, ch, lattice)
    ji, jp = jet_lattice(avg, ch, lattice)
    # probe: cached 4.1e-12 / 5.2e-12, per-node 3.2e-12 / 3.8e-12
    assert rel(si, ji) < 5e-11 and rel(sp, jp) < 5e-11


@pytest.mark.parametrize(
    "mode", ["cached", pytest.param("per-node", marks=pytest.mark.slow)]
)
@pytest.mark.parametrize("order", [5, 6])
def test_orders_5_and_6_by_lookups_equal_quadrature(mode, order):
    """Decision 141: above N = 4 the reference is the quadrature S1 (the oracle path), not the jet.

    The rows 0..4 of each cell equal the order-4 stack, which the tests above compare with the jet.
    Probe (3 cells): lookup against quadrature 3.0e-10 / 2.2e-10 (cached, N = 5), 2.4e-10 / 5.2e-10 (cached, N = 6),
    2.6e-10 / 2.2e-10 (per-node, N = 5), 2.8e-10 / 3.4e-10 (per-node, N = 6). Rows 0..4: equal to the order-4
    stack bit for bit. Per-node quadrature takes 13 to 16 s at load 20.
    """
    b = (
        uniform_rule(4e-6, 6e-6, 6)
        if mode == "cached"
        else lognormal_given_gamma(0.4, 0.1)
    )
    avg = AveragedKernel(K, [("B", b), eta4()], phase=screen())
    ch, lattice = small_channels(), gamma_lattice(3, order)
    li, lp = stack_lattice(avg, ch, lattice)
    qi, qp = stack_lattice(avg, ch, lattice, s1="quadrature")
    assert rel(li, qi) < 1e-9 and rel(lp, qp) < 1e-9, (rel(li, qi), rel(lp, qp))
    fi, fp = stack_lattice(avg, ch, gamma_lattice(3, 4))
    low = np.concatenate(
        [np.arange(c * (order + 1), c * (order + 1) + 5) for c in range(3)]
    )
    assert rel(li[low], fi) < 1e-14 and rel(lp[low], fp) < 1e-14  # measured 0.0


# ------------------------------------------------------------------ 6. eta given B, per-node route


@pytest.mark.parametrize("order", [2, 4])
def test_eta_given_b_per_node_equals_the_jet(order):
    eta = uniform_rule(0.0, lambda p: 0.3 + 4e4 * p["B"], 6, parents=("B",))
    avg = AveragedKernel(
        K, [("B", uniform_rule(4e-6, 6e-6, 4)), ("eta", eta)], phase=screen()
    )
    ch, lattice = small_channels(), gamma_lattice(3, order)
    si, sp = stack_lattice(avg, ch, lattice)
    ji, jp = jet_lattice(avg, ch, lattice)
    # probe: 1.2e-11 / 1.3e-11 (N = 2), 7.8e-11 / 5.6e-11 (N = 4)
    assert rel(si, ji) < 5e-10 and rel(sp, jp) < 5e-10


# ------------------------------------------------------------------ 7. field cells


@pytest.mark.parametrize("ratio", [1.01, 100.0])
@pytest.mark.parametrize("order", [2, 4])
def test_field_cells_over_narrow_and_wide_field_ranges(ratio, order):
    """``B`` cells on ``[4, 4 ratio] uG`` (3 cells) times 2 energy cells, every cell against the jet.

    Probe (against each cell's channel maxima): largest 2.6e-10 (ratio 1.01, N = 2), 2.3e-10 (1.01, N = 4),
    1.2e-10 (100, N = 2), 2.1e-10 (100, N = 4), all in the low energy cell.
    """
    avg = AveragedKernel(K, [("eta", uniform_rule(0.0, 0.5, 6))], phase=screen())
    lattice = Lattice(
        gamma=Cells.log(300, 3e6, 2, order=order),
        B=Cells.log("4 uG", f"{4 * ratio} uG", 3, order=order),
        order=order,
    )
    ch = small_channels()
    for c in range(lattice.n_cells):
        si, sp = stack_cell(avg, ch, lattice, c)
        ji, jp = jet_cell(avg, ch, lattice, c)
        assert rel(si, ji) < 1e-9 and rel(sp, jp) < 1e-9, (ratio, order, c, rel(si, ji))


# ------------------------------------------------------------------ 8. strong fields


@pytest.mark.parametrize("order", [2, 4])
def test_strong_field_equals_the_jet(order):
    """``B = 1 G``, ``gamma`` in ``[30, 100]``, channels 10 to 1000 GHz: the S1 grid extends to ``ln(700e6)``.

    Before the grid extension the high channels gave zero rows. Probe: 1.8e-10 / 1.8e-10 (N = 2), 1.5e-10 /
    1.5e-10 (N = 4). The 1000 GHz channel maximum is 2.7e-35 (x near 300), and no stack row is zero.
    """
    ch = Channels.log(10, 1000, 5, unit="GHz", half_width=0.2, n_nu=32)
    avg = AveragedKernel(
        ContinuumKernel(gamma_min=10, x_min=1e-12),
        [eta4()],
        fixed={"B": 1.0},
        phase=screen(),
    )
    lattice = gamma_lattice(3, order, 30, 100)
    si, sp = stack_lattice(avg, ch, lattice)
    ji, jp = jet_lattice(avg, ch, lattice)
    assert np.all(channel_max(ji) > 0) and not np.any(
        np.all(np.asarray(si) == 0, axis=1)
    )
    assert rel(si, ji) < 1e-9 and rel(sp, jp) < 1e-9
