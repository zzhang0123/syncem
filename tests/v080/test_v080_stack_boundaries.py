"""Boundary tests of the stack provider (design/0.8/SPEC.md Sec. 8.3, 8.8, step 5a of Sec. 10.2).

Each dispatch condition of ``_stack_series`` is tested on both sides: where the stack applies it is compared with
the jet (``provider_jet``) at the same cell, at extreme parameter values, and where it refuses the test asserts
``StackRefused`` and checks that the jet computes the cell. Tolerances come from probes on 2026-11-18 (machine load
average 16 to 24, which changes timings, not values). The measured value is in a comment next to each tolerance.
A sign error or a dropped term changes the compared columns by 1e-4 or more in every case measured (frozen weights
without reweighting, a flipped affine ``eps``, a dropped log coefficient).
"""

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.capabilities import FieldScalingRelation, Similarity
from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.user import kernel
from syncem.base.variables import SYNCHROTRON_VARIABLES
from syncem.engine._stack_series import StackRefused, stack_cell, stack_lattice
from syncem.engine.average import AveragedKernel, NodeRule, fixed_rule, uniform_rule
from syncem.engine.field_average import FieldAverage
from syncem.engine.primitive import PrimitiveTable
from syncem.engine.provider_jet import jet_cell, jet_lattice
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import Legendre
from tests.v080.test_v080_stack import (
    channel_max,
    lognormal_given_gamma,
    rel,
    screen,
    small_channels,
)

K = ContinuumKernel(x_min=1e-12)


def eta4():
    return ("eta", uniform_rule(0.0, 0.5, 4))


def gamma_lattice(n_cells, order, lo=300, hi=3e6, chart="log"):
    cells = (Cells.log if chart == "log" else Cells.linear)(
        lo, hi, n_cells, order=order
    )
    return Lattice(gamma=cells, order=order)


def scaled_error(stack, jet, scale):
    return float(np.max(np.abs(np.asarray(stack) - np.asarray(jet)) / scale))


def finite(a):
    return bool(np.all(np.isfinite(np.asarray(a))))


# ------------------------------------------------------------------ 1. frozen-node reweighting at N = 4


def _lognormal_avg(width, slope, n=16, k=K):
    return AveragedKernel(
        k, [("B", lognormal_given_gamma(width, slope, n)), eta4()], phase=screen()
    )


def _lattice_scale(avg, ch, n_cells):
    """Each channel's largest value over the lattice.

    Up to 19 cells: over every row of the stack lattice, as in ``test_b_given_celled_gamma_by_frozen_reweighting``.
    At 200 cells: over the order-0 rows of the jet at the 200 references (2 s instead of 35 s). That is a lower bound
    of the lattice maximum, so the test is stricter there.
    """
    if n_cells <= 19:
        li, lp = stack_lattice(avg, ch, gamma_lattice(n_cells, 4))
    else:
        li, lp = jet_lattice(avg, ch, gamma_lattice(n_cells, 0))
    return channel_max(li), channel_max(lp)


_REWEIGHT_CASES = [
    (w, s, n)
    for w in (0.01, 0.4)
    for s in (0.02, 0.1, 0.5)
    for n in (1, 19, 200)
    if not (
        w == 0.01 and s >= 0.1 and n == 1
    )  # the two cases of the amplification test below
]


@pytest.mark.parametrize(("width", "slope", "n_cells"), _REWEIGHT_CASES)
def test_reweighting_equals_the_jet_at_order_4(width, slope, n_cells):
    """``ln B ~ N(ln 5 uG + slope ln(gamma / 1e4), width^2)`` on 16 Gauss-Hermite nodes, order 4, cells first/mid/last."""
    ch, avg = small_channels(), _lognormal_avg(width, slope)
    lattice = gamma_lattice(n_cells, 4)
    scale_i, scale_p = _lattice_scale(avg, ch, n_cells)
    memo: dict = {}
    for c in sorted({0, n_cells // 2, n_cells - 1}):
        si, sp = stack_cell(avg, ch, lattice, c, memo=memo)
        ji, jp = jet_cell(avg, ch, lattice, c)
        err_i, err_p = scaled_error(si, ji, scale_i), scaled_error(sp, jp, scale_p)
        # probe: largest 1.5e-10 (I and P, width 0.01, slope 0.5, 19 cells, cell 9), 1.7e-11 on one cell at
        # width 0.01 and slope 0.02, below 1e-10 elsewhere
        assert err_i < 1e-9 and err_p < 1e-9, (width, slope, n_cells, c, err_i, err_p)
    mid = n_cells // 2
    naive = stack_cell(avg, ch, lattice, mid, reweight=False, memo=memo)[0]
    jet = jet_cell(avg, ch, lattice, mid)[0]
    # frozen weights without reweighting: 5.1e-4 (200 cells, width 0.4, slope 0.02) to 0.93 (one cell, slope 0.5)
    assert scaled_error(naive, jet, scale_i) > 1e-4


@pytest.mark.parametrize("slope", [0.1, 0.5])
def test_reweighting_of_a_narrow_law_across_one_wide_cell_refuses(slope):
    """One cell over ``[300, 3e6]`` (h = 4.6), width 0.01: the weights are amplified beyond the bound.

    Probe (2026-11-18, load 16 to 18): amplification ``max_k sum_q |w_q^(k)|`` 928, 5.8e5 and 3.6e8 at slopes 0.02,
    0.1 and 0.5, with stack-jet errors 1.7e-11, 8.5e-9 and 9.8e-6 of the lattice scale. The error is 1.5e-14 to
    5.4e-14 of the amplification wherever cancellation dominates, and the bound 1e4 keeps it below about 5e-10. The
    largest amplification of the accepted cases above is 2.8e3 (width 0.01, slope 0.5, 19 cells).
    """
    ch, avg = small_channels(), _lognormal_avg(0.01, slope)
    lattice = gamma_lattice(1, 4)
    with pytest.raises(StackRefused, match="amplifies the weights"):
        stack_cell(avg, ch, lattice, 0)
    assert finite(jet_cell(avg, ch, lattice, 0)[0])


def test_reweighting_of_a_narrow_law_across_one_wide_cell_at_a_small_slope_equals_the_jet():
    ch, avg = small_channels(), _lognormal_avg(0.01, 0.02)
    lattice = gamma_lattice(1, 4)
    scale_i, scale_p = _lattice_scale(avg, ch, 1)
    si, sp = stack_cell(avg, ch, lattice, 0)
    ji, jp = jet_cell(avg, ch, lattice, 0)
    # probe: 1.7e-11 (amplification 928)
    assert scaled_error(si, ji, scale_i) < 1e-9 and scaled_error(sp, jp, scale_p) < 1e-9


@pytest.mark.parametrize("slope", [0.02, 0.5])
def test_reweighting_refuses_an_unresolved_wide_law(slope):
    """Width 1.5 on 16 nodes: the reweighted moments of ``t^2`` or ``t^3`` (``t`` linear in ``B``) differ from the
    moving-node moments (probe: order 1, power 3, 0.491339 against 0.491528 at slope 0.02), since ``E[B^3]`` of a
    log-normal of width 1.5 grows as ``exp(10.1)`` and 16 nodes do not resolve it. Every cell refuses.
    """
    ch, avg = small_channels(), _lognormal_avg(1.5, slope)
    for n_cells in (1, 19, 200):
        lattice = gamma_lattice(n_cells, 4)
        for c in sorted({0, n_cells // 2, n_cells - 1}):
            with pytest.raises(
                StackRefused, match="does not reproduce the moving nodes"
            ):
                stack_cell(avg, ch, lattice, c)
    assert finite(jet_cell(avg, ch, gamma_lattice(19, 4), 9)[0])


@pytest.mark.slow
def test_reweighting_of_a_wide_law_on_32_nodes_matches_the_jet_to_its_quadrature_error():
    """Width 1.5, slope 0.1, 32 nodes: the check accepts, and the stack differs from the jet by the law quadrature.

    Probe (mid cell of 3, order 4, against the cell's channel maxima): stack - jet 2.2e-4. Against the 64-node jet
    the 32-node jet is off by 8.8e-4 and the 32-node stack by 6.6e-4, so the difference is the law quadrature,
    which both methods share, not the reweighting. ``x_min = 1e-30`` admits the 64-node field nodes up to 1e5 G.
    """
    k = ContinuumKernel(x_min=1e-30)
    ch, lattice = small_channels(), gamma_lattice(3, 4)
    si = stack_cell(_lognormal_avg(1.5, 0.1, 32, k), ch, lattice, 1)[0]
    j32 = jet_cell(_lognormal_avg(1.5, 0.1, 32, k), ch, lattice, 1)[0]
    j64 = jet_cell(_lognormal_avg(1.5, 0.1, 64, k), ch, lattice, 1)[0]
    assert rel(si, j32) < 5e-4  # measured 2.2e-4
    assert rel(si, j64) < rel(j32, j64)  # measured 6.6e-4 < 8.8e-4


# ------------------------------------------------------------------ 2. refusals that must not return numbers


def test_translating_uniform_support_refuses():
    """``B ~ U(lo, lo + 2 uG)`` with ``lo = 4 uG + 0.5 uG ln(gamma / 1e4)``: constant width, moving support."""

    def lo(p):
        return 4e-6 + 0.5e-6 * jnp.log(p["gamma"] / 1e4)

    def hi(p):
        return lo(p) + 2e-6

    def density(x, p):
        return jnp.where((x >= lo(p)) & (x <= hi(p)), 1.0 / (hi(p) - lo(p)), 0.0)

    rule = NodeRule(
        uniform_rule(lo, hi, 8, parents=("gamma",)).fn,
        parents=("gamma",),
        density=density,
    )
    ch, lattice = small_channels(), gamma_lattice(3, 4)
    avg = AveragedKernel(K, [("B", rule), eta4()], phase=screen())
    # probe: order 1, test power 1, 0 against 1.3294 (the density ratio is 1 at the interior nodes)
    with pytest.raises(StackRefused, match="does not reproduce the moving nodes"):
        stack_cell(avg, ch, lattice, 1)
    assert finite(jet_cell(avg, ch, lattice, 1)[0])


def test_density_with_another_gamma_slope_refuses():
    rule, wrong = lognormal_given_gamma(0.4, 0.1), lognormal_given_gamma(0.4, 0.2)
    avg = AveragedKernel(
        K,
        [("B", NodeRule(rule.fn, parents=("gamma",), density=wrong.density)), eta4()],
        phase=screen(),
    )
    ch, lattice = small_channels(), gamma_lattice(3, 4)
    # probe: order 1, test power 0, -0.210804 against 0 (the reweighted weights no longer sum to one)
    with pytest.raises(StackRefused, match="does not reproduce the moving nodes"):
        stack_cell(avg, ch, lattice, 1)
    assert finite(jet_cell(avg, ch, lattice, 1)[0])


def test_density_zero_at_a_node_refuses():
    """Simpson nodes on ``[4, 6] uG`` with an open-support mask: the density is zero at the end nodes."""
    nodes, probs = jnp.array([4e-6, 5e-6, 6e-6]), jnp.array([1 / 6, 2 / 3, 1 / 6])
    rule = NodeRule(
        lambda p: (nodes, probs),
        parents=("gamma",),
        density=lambda x, p: jnp.where((x > 4e-6) & (x < 6e-6), 0.5e6, 0.0),
    )
    avg = AveragedKernel(K, [("B", rule), eta4()], phase=screen())
    ch, lattice = small_channels(), gamma_lattice(3, 4)
    with pytest.raises(StackRefused, match="not positive and finite at its own nodes"):
        stack_cell(avg, ch, lattice, 1)
    assert finite(jet_cell(avg, ch, lattice, 1)[0])


def test_untraceable_rule_refuses():
    t, w = np.polynomial.hermite.hermgauss(16)

    def fn(p):
        median = np.log(5e-6) + 0.1 * np.log(float(p["gamma"]) / 1e4)
        return np.exp(median + 0.4 * np.sqrt(2.0) * t), w / np.sqrt(np.pi)

    rule = NodeRule(
        fn, parents=("gamma",), density=lognormal_given_gamma(0.4, 0.1).density
    )
    avg = AveragedKernel(K, [("B", rule), eta4()], phase=screen())
    with pytest.raises(StackRefused, match="cannot be differentiated"):
        stack_cell(avg, small_channels(), gamma_lattice(3, 4), 1)


_V = SYNCHROTRON_VARIABLES


@kernel(
    variables=[_V["gamma"], _V["B"], _V["eta"]],
    stokes=("I", "P"),
    capabilities=(Similarity(("gamma",)), FieldScalingRelation("B", "gamma")),
)
def _similar_toy(nu, gamma, B, eta):
    x = nu / (4.2e6 * B * jnp.sqrt(1 - eta**2) * gamma**2)
    return {
        "I": B * x ** (1 / 3) * jnp.exp(-x),
        "P": 0.7 * B * x ** (1 / 3) * jnp.exp(-x),
    }


class _NoSimilarity(ContinuumKernel):
    capabilities: ClassVar[tuple] = (FieldScalingRelation("B", "gamma"),)


class _NoScaling(ContinuumKernel):
    capabilities: ClassVar[tuple] = (Similarity(("gamma",)),)


class _Doubled(ContinuumKernel):
    """A subclass that changes the physics: twice the continuum."""

    def _node_values(self, nu, point, phase):
        i, p = super()._node_values(nu, point, phase)
        return 2.0 * i, 2.0 * p


def test_user_kernel_with_similarity_refuses():
    avg = AveragedKernel(
        _similar_toy, [("B", uniform_rule(4e-6, 6e-6, 4)), eta4()], phase=screen()
    )
    ch, lattice = small_channels(), gamma_lattice(3, 4)
    with pytest.raises(StackRefused, match="continuum F and G only, not UserKernel"):
        stack_cell(avg, ch, lattice, 1)
    assert finite(jet_cell(avg, ch, lattice, 1)[0])


def test_continuum_subclass_without_similarity_refuses():
    avg = AveragedKernel(_NoSimilarity(x_min=1e-12), [eta4()], fixed={"B": 5e-6})
    with pytest.raises(StackRefused, match="continuum F and G only, not _NoSimilarity"):
        stack_cell(avg, small_channels(3, 16), gamma_lattice(2, 2, hi=3e5), 0)


def test_continuum_subclass_with_other_physics_refuses():
    """Before the exact type check the stack returned the plain continuum, half of the subclass's jet."""
    avg = AveragedKernel(
        _Doubled(x_min=1e-12),
        [("B", uniform_rule(4e-6, 6e-6, 4)), eta4()],
        phase=screen(),
    )
    with pytest.raises(StackRefused, match="continuum F and G only, not _Doubled"):
        stack_cell(avg, small_channels(), gamma_lattice(3, 2), 1)


def test_order_7_refuses():
    """The jet is not run here: its order-7 compile did not finish in 9 minutes at load 19 (one cell, no averages)."""
    avg = AveragedKernel(K, fixed={"B": 5e-6, "eta": 0.0}, phase=screen())
    with pytest.raises(StackRefused, match="order 7 exceeds the S1 table order 6"):
        stack_cell(avg, small_channels(3, 16), gamma_lattice(1, 7, 3e3, 3e4), 0)


def test_energy_below_the_s1_grid_refuses_and_its_neighbour_does_not():
    """At 0.1 GHz (channel nodes from 0.08 GHz) and 5 uG the S1 grid ends at ``gamma`` near 1.1e8.

    Probe: ``[1e8, 1e9]`` gives ``ln y = -22.38 < -20.73``. Of ``[3e7, 3e8]`` in two cells, cell 0 (reference
    5.3e7) is inside and equals the jet to 3.9e-13 (I and P), cell 1 (reference 1.7e8) gives ``ln y = -21.12``.
    """
    ch = small_channels()
    avg = AveragedKernel(K, [eta4()], fixed={"B": 5e-6}, phase=screen())
    high = gamma_lattice(1, 2, 1e8, 1e9)
    with pytest.raises(StackRefused, match="below the field-average table"):
        stack_cell(avg, ch, high, 0)
    assert finite(jet_cell(avg, ch, high, 0)[0])
    pair = gamma_lattice(2, 4, 3e7, 3e8)
    si, sp = stack_cell(avg, ch, pair, 0)
    ji, jp = jet_cell(avg, ch, pair, 0)
    assert rel(si, ji) < 1e-11 and rel(sp, jp) < 1e-11  # measured 3.9e-13
    with pytest.raises(StackRefused, match="below the field-average table"):
        stack_cell(avg, ch, pair, 1)
    assert finite(jet_cell(avg, ch, pair, 1)[0])


# ------------------------------------------------------------------ 3. every other raise of _stack_series


def _field_lattice():
    return Lattice(
        gamma=Cells.log(300, 3e5, 2, order=2),
        B=Cells.log("2 uG", "8 uG", 2, order=2),
        order=2,
    )


_ETA0 = ("eta", fixed_rule([0.0], [1.0]))


def _raise_cases():
    g2 = gamma_lattice(2, 2, hi=3e5)
    b_uniform = ("B", uniform_rule(4e-6, 6e-6, 2))
    return {
        "undeclared parents": (
            StackRefused,
            "undeclared",
            AveragedKernel(K, [("B", uniform_rule(lambda p: 4e-6, 6e-6, 2)), _ETA0]),
            g2,
            {},
        ),
        "no density": (
            StackRefused,
            "without a density",
            AveragedKernel(
                K,
                [
                    ("B", NodeRule(lognormal_given_gamma(0.4, 0.1).fn, ("gamma",))),
                    _ETA0,
                ],
            ),
            g2,
            {},
        ),
        "bare callable": (
            StackRefused,
            "bare function",
            AveragedKernel(
                K, [("B", lambda p: (jnp.array([5e-6]), jnp.array([1.0]))), _ETA0]
            ),
            g2,
            {},
        ),
        "legendre on mu": (
            StackRefused,
            "a Legendre axis on mu: the stack projects only eta",
            AveragedKernel(K, [b_uniform], legendre={"mu": (2, 8)}, fixed={"eta": 0.0}),
            Lattice(gamma=Cells.log(300, 3e5, 2, order=2), mu=Legendre(2), order=2),
            {},
        ),
        "no B": (
            StackRefused,
            "B is neither celled nor averaged nor fixed",
            AveragedKernel(K, [_ETA0]),
            g2,
            {},
        ),
        "bad s1": (
            ValueError,
            "s1 must be 'lookup' or 'quadrature'",
            AveragedKernel(K, [_ETA0], fixed={"B": 5e-6}),
            g2,
            {"s1": "spline"},
        ),
        "celled variable eta": (
            StackRefused,
            "celled variables",
            AveragedKernel(K, fixed={"B": 5e-6}),
            Lattice(gamma=Cells.log(300, 3e5, 2), eta=Cells.linear(0, 0.5, 2), order=2),
            {},
        ),
        "celled and fixed": (
            ValueError,
            r"\['B'\] are celled and also averaged or fixed",
            AveragedKernel(K, [_ETA0], fixed={"B": 5e-6}),
            _field_lattice(),
            {},
        ),
        "celled and averaged": (
            ValueError,
            r"\['B'\] are celled and also averaged or fixed",
            AveragedKernel(K, [b_uniform, _ETA0]),
            _field_lattice(),
            {},
        ),
        "legendre axes differ": (
            ValueError,
            "Legendre axes differ",
            AveragedKernel(K, [b_uniform], legendre={"eta": (2, 8)}),
            g2,
            {},
        ),
        "affine energy chart in a field cell": (
            StackRefused,
            "field cells with an affine energy chart",
            AveragedKernel(K, [_ETA0]),
            Lattice(
                gamma=Cells.linear(300, 3e5, 2, order=2),
                B=Cells.log("2 uG", "8 uG", 2, order=2),
                order=2,
            ),
            {},
        ),
        "no field-scaling relation": (
            StackRefused,
            "declares no FieldScalingRelation",
            AveragedKernel(_NoScaling(x_min=1e-12), [_ETA0]),
            _field_lattice(),
            {},
        ),
        "gamma parent in a field cell": (
            StackRefused,
            "the law of eta depends on gamma or B",
            AveragedKernel(
                K,
                [
                    (
                        "eta",
                        uniform_rule(
                            0.0,
                            lambda p: 0.3 + 0.01 * jnp.log(p["gamma"]),
                            3,
                            parents=("gamma",),
                        ),
                    )
                ],
            ),
            _field_lattice(),
            {},
        ),
        "B parent in a field cell": (
            StackRefused,
            "the law of eta depends on gamma or B",
            AveragedKernel(
                K,
                [
                    (
                        "eta",
                        uniform_rule(
                            0.0, lambda p: 0.3 + 4e4 * p["B"], 3, parents=("B",)
                        ),
                    )
                ],
            ),
            _field_lattice(),
            {},
        ),
        "undeclared parents in a field cell": (
            StackRefused,
            "the law of eta has undeclared parents",
            AveragedKernel(K, [("eta", uniform_rule(0.0, lambda p: 0.4, 3))]),
            _field_lattice(),
            {},
        ),
        "bare callable in a field cell": (
            StackRefused,
            "the law of eta is a bare function",
            AveragedKernel(
                K, [("eta", lambda p: (jnp.array([0.0]), jnp.array([1.0])))]
            ),
            _field_lattice(),
            {},
        ),
    }


@pytest.mark.parametrize("case", list(_raise_cases()))
def test_every_raise_of_the_stack(case):
    """Each raise with its message."""
    error, match, avg, lattice, kw = _raise_cases()[case]
    with pytest.raises(error, match=match):
        stack_cell(avg, small_channels(3, 16), lattice, 0, **kw)


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


# ------------------------------------------------------------------ 9. S0 coverage and sharing


def test_lookups_refuse_a_short_primitive_table():
    """For one node at 5 uG, ``needed_lnx`` is about ``(-22.35, 12.21)``."""
    lo, hi = FieldAverage.needed_lnx([5.0])
    for table in (
        PrimitiveTable.build(lo + 1.0, hi + 1.0),
        PrimitiveTable.build(lo - 1.0, hi - 1.0),
        PrimitiveTable.build(lo - 1.0, hi + 1.0, order=2),
    ):
        with pytest.raises(ValueError, match="does not cover the lookups"):
            FieldAverage.lookups([5.0], [1.0], primitive=table)
    table = PrimitiveTable.build(-1.0, 1.0, order=2)
    for lnx in (-1.5, 1.5):
        with pytest.raises(ValueError, match="leaves the primitive table"):
            table.lookup(np.array([lnx]), "F", 2)


def _count_builds(monkeypatch):
    calls = []
    build = PrimitiveTable.build.__func__

    def counted(cls, *args, **kwargs):
        calls.append(args)
        return build(cls, *args, **kwargs)

    monkeypatch.setattr(PrimitiveTable, "build", classmethod(counted))
    return calls


def test_one_s0_serves_every_s1_table_of_a_lattice(monkeypatch):
    """Per-node mode builds one S1 table per field node and cell (16 nodes on 3 cells), all from one S0."""
    calls = _count_builds(monkeypatch)
    avg = _lognormal_avg(0.4, 0.1)
    stack_lattice(avg, small_channels(), gamma_lattice(3, 2))
    assert len(calls) == 1


def test_s0_is_rebuilt_over_the_union_when_a_law_needs_more(monkeypatch):
    """Decision 144: a 1 G field reads ``ln x`` down to about -34.6, below the first S0 (from ln 1e-12)."""
    calls = _count_builds(monkeypatch)
    ch, lattice, memo = small_channels(), gamma_lattice(3, 2), {}
    weak = AveragedKernel(
        K, [("B", uniform_rule(4e-6, 6e-6, 4)), eta4()], phase=screen()
    )
    strong = AveragedKernel(K, [eta4()], fixed={"B": 1.0}, phase=screen())
    stack_cell(weak, ch, lattice, 1, memo=memo)
    first = memo["S0"]
    stack_cell(strong, ch, lattice, 1, memo=memo)
    union = memo["S0"]
    assert len(calls) == 2
    assert union.lnx0 <= min(first.lnx0, FieldAverage.needed_lnx([1e6])[0])
    assert union.lnx_hi >= first.lnx_hi
    other = AveragedKernel(
        K, [("B", uniform_rule(3e-6, 7e-6, 4)), eta4()], phase=screen()
    )
    stack_cell(other, ch, lattice, 2, memo=memo)
    assert len(calls) == 2 and memo["S0"] is union


def test_s0_grows_with_headroom_over_a_strong_field_lattice(monkeypatch):
    """Six ``B`` cells from 2 mG to 2 G in ascending order: one S0 build, not one per cell (decision 144).

    Probe before the headroom: 6 builds (lnx0 from -28.91 to -34.67), 3.4 s at load 17.
    """
    calls = _count_builds(monkeypatch)
    avg = AveragedKernel(K, [("eta", uniform_rule(0.0, 0.5, 6))], phase=screen())
    lattice = Lattice(
        gamma=Cells.log(30, 300, 1), B=Cells.log("2 mG", "2 G", 6), order=2
    )
    ch = Channels.log(0.1, 10, 5, unit="GHz", half_width=0.2, n_nu=32)
    stack_lattice(avg, ch, lattice)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "nodes", [np.geomspace(2e3, 1e7, 8), np.geomspace(0.1, 1e-3, 8)]
)
def test_s0_grows_with_headroom_over_monotone_field_nodes(monkeypatch, nodes):
    """One-node tables in ascending strong or descending weak fields (8 builds each before the headroom)."""
    from syncem.engine._stack_series import _s1_table

    calls, memo = _count_builds(monkeypatch), {}
    for b in nodes:
        _s1_table(memo, "lookup", np.array([b]), np.ones(1))
    assert len(calls) == 1


# ------------------------------------------------------------------ 10. findings of the step-5a re-review


def _fixed_node_rule(c):
    """Fixed Gauss-Legendre nodes on [2, 10] uG with weights moving with gamma, and a density off by ``c P4``.

    Over 8 Gauss-Legendre nodes ``P4`` is orthogonal to ``t^0..t^3``, so the moment check alone accepts ``c != 0``.
    """
    s, w = np.polynomial.legendre.leggauss(8)
    lo, hi = 2e-6, 10e-6
    x = 0.5 * (hi - lo) * s + 0.5 * (lo + hi)
    p1, p4 = (np.polynomial.legendre.Legendre.basis(k) for k in (1, 4))

    def ell(p):
        return jnp.log(p["gamma"] / 1e4)

    def std(xx):
        return (2 * xx - (lo + hi)) / (hi - lo)

    def density(xx, p):
        return (
            1 + 0.05 * ell(p) * p1(std(xx)) + c * (ell(p) - np.log(3.0)) * p4(std(xx))
        ) / (hi - lo)

    def fn(p):
        return jnp.asarray(x), jnp.asarray(0.5 * w) * (
            1 + 0.05 * ell(p) * jnp.asarray(p1(s))
        )

    return NodeRule(fn, parents=("gamma",), density=density)


def test_fixed_nodes_compare_weights_node_by_node():
    """Probe (re-review): with ``c = 0.05`` the moment check accepted and the stack was 4.0e-4 from the jet."""
    ch, lattice = small_channels(), gamma_lattice(3, 4)
    right = AveragedKernel(K, [("B", _fixed_node_rule(0.0)), eta4()], phase=screen())
    si, sp = stack_cell(right, ch, lattice, 1)
    ji, jp = jet_cell(right, ch, lattice, 1)
    assert rel(si, ji) < 1e-10 and rel(sp, jp) < 1e-10  # probe 4.2e-12 / 5.4e-12
    for c in (0.05, 0.3):
        wrong = AveragedKernel(K, [("B", _fixed_node_rule(c)), eta4()], phase=screen())
        with pytest.raises(StackRefused, match="at its fixed nodes"):
            stack_cell(wrong, ch, lattice, 1)


def test_untraceable_density_refuses():
    good = lognormal_given_gamma(0.4, 0.1)
    rule = NodeRule(
        good.fn,
        parents=("gamma",),
        density=lambda x, p: good.density(x, p) * (1.0 + 0.0 * float(p["gamma"])),
    )
    avg = AveragedKernel(K, [("B", rule), eta4()], phase=screen())
    ch, lattice = small_channels(), gamma_lattice(3, 4)
    with pytest.raises(
        StackRefused, match="density of the law of B cannot be differentiated"
    ):
        stack_cell(avg, ch, lattice, 1)
    assert finite(jet_cell(avg, ch, lattice, 1)[0])


def _eta_reading_gamma(p):
    b = 0.3 + 0.1 * jnp.tanh(jnp.log(p["gamma"] / 1e4))
    s, w = np.polynomial.legendre.leggauss(4)
    return 0.5 * b * (jnp.asarray(s) + 1), jnp.asarray(0.5 * w)


@pytest.mark.parametrize("parents", [(), ("B",)])
def test_a_law_reading_gamma_without_declaring_it_refuses(parents):
    """Probe (re-review): both declarations were frozen silently, 1.7e-2 from the jet."""
    avg = AveragedKernel(
        K,
        [
            ("B", lognormal_given_gamma(0.4, 0.1)),
            ("eta", NodeRule(_eta_reading_gamma, parents=parents)),
        ],
        phase=screen(),
    )
    ch, lattice = small_channels(), gamma_lattice(3, 4)
    with pytest.raises(StackRefused, match="the law of eta depends on gamma"):
        stack_cell(avg, ch, lattice, 1)
    assert finite(jet_cell(avg, ch, lattice, 1)[0])


def test_a_law_reading_b_without_declaring_it_in_a_field_cell_refuses():
    eta = NodeRule(
        lambda p: (jnp.array([0.0, 0.3 + 4e4 * p["B"]]), jnp.array([0.5, 0.5])),
        parents=(),
    )
    avg = AveragedKernel(K, [("eta", eta)], phase=screen())
    with pytest.raises(StackRefused, match="the law of eta depends on B"):
        stack_cell(avg, small_channels(3, 16), _field_lattice(), 0)


@pytest.mark.parametrize("phase", [None, "screen"])
def test_inclination_nodes_at_one_add_zero(phase):
    """Endpoint rules on eta and a fixed eta = 1: ``s = 0`` gives no emission, as in the jet."""
    ph = screen() if phase else None
    ch, lattice = small_channels(3, 16), gamma_lattice(1, 2, 3e3, 3e4)
    for avg in (
        AveragedKernel(
            K,
            [("eta", fixed_rule([0.0, 1.0], [0.5, 0.5]))],
            fixed={"B": 5e-6},
            phase=ph,
        ),
        AveragedKernel(
            K,
            [("eta", fixed_rule([-1.0, 0.0, 1.0], [1 / 6, 2 / 3, 1 / 6]))],
            fixed={"B": 5e-6},
            phase=ph,
        ),
        AveragedKernel(K, fixed={"B": 5e-6, "eta": 1.0}, phase=ph),
    ):
        si, sp = stack_cell(avg, ch, lattice, 0)
        ji, jp = jet_cell(avg, ch, lattice, 0)
        assert np.allclose(
            si, ji, rtol=1e-9, atol=1e-9 * float(np.max(np.abs(ji)) or 1e-300)
        )
        assert np.allclose(
            sp, jp, rtol=1e-9, atol=1e-9 * float(np.max(np.abs(jp)) or 1e-300)
        )


@pytest.mark.parametrize("s1", ["lookup", "quadrature"])
@pytest.mark.parametrize("weights", [[0.5, 0.5], [0.0, 1.0]])
def test_a_field_node_at_zero_adds_zero(s1, weights):
    """Probe (re-review): OverflowError (lookup) and ValueError (quadrature) before the law was filtered."""
    avg = AveragedKernel(
        K, [("B", fixed_rule([0.0, 5e-6], weights))], fixed={"eta": 0.0}, phase=screen()
    )
    ch, lattice = small_channels(3, 16), gamma_lattice(1, 2, 3e3, 3e4)
    si, sp = stack_cell(avg, ch, lattice, 0, s1=s1)
    ji, jp = jet_cell(avg, ch, lattice, 0)
    assert (
        rel(si, ji) < 1e-9 and rel(sp, jp) < 1e-9
    )  # probe without the zero node: 7.2e-12


def test_a_per_node_field_node_at_zero_adds_zero():
    """Per-node mode: a field node at 0 of a law conditional on eta gives a zero leaf."""
    b = NodeRule(
        lambda p: (jnp.array([0.0, 5e-6 * (1 + p["eta"])]), jnp.array([0.5, 0.5])),
        parents=("eta",),
    )
    avg = AveragedKernel(
        K, [("eta", uniform_rule(0.0, 0.5, 3)), ("B", b)], phase=screen()
    )
    ch, lattice = small_channels(3, 16), gamma_lattice(1, 2, 3e3, 3e4)
    si, sp = stack_cell(avg, ch, lattice, 0)
    ji, jp = jet_cell(avg, ch, lattice, 0)
    assert rel(si, ji) < 1e-9 and rel(sp, jp) < 1e-9
