"""Refusals of the stack provider (design/0.8/SPEC.md Sec. 8.3, 8.8): each refusal and each raise of ``_stack_series``.

A refusal raises ``StackRefused`` with its message, and the jet computes the refused cell. Split from
``test_v080_stack_boundaries.py`` for the memory target of decision 170.
"""

from typing import ClassVar

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.capabilities import FieldScalingRelation, Similarity
from syncem.base.continuum import ContinuumKernel
from syncem.base.user import kernel
from syncem.base.variables import SYNCHROTRON_VARIABLES
from syncem.engine._stack_series import StackRefused, stack_cell
from syncem.engine.average import AveragedKernel, NodeRule, fixed_rule, uniform_rule
from syncem.engine.provider_jet import jet_cell
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import Legendre
from tests.v080.test_v080_stack import (
    lognormal_given_gamma,
    rel,
    screen,
    small_channels,
)
from tests.v080.test_v080_stack_boundaries import (
    K,
    eta4,
    finite,
    gamma_lattice,
)

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
