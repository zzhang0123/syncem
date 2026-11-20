"""Findings of the step-5a re-review of the stack provider (design/0.8/SPEC.md Sec. 8.3, 8.8).

Fixed nodes with moving weights, untraceable densities, laws that read undeclared parents, and nodes at the ends
of the inclination and field ranges. Split from ``test_v080_stack_boundaries.py`` for the memory target of
decision 170.
"""

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.engine._stack_series import StackRefused, stack_cell
from syncem.engine.average import AveragedKernel, NodeRule, fixed_rule, uniform_rule
from syncem.engine.provider_jet import jet_cell
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
from tests.v080.test_v080_stack_refusals import _field_lattice

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
