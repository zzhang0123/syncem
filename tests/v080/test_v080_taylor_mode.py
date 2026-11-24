"""Taylor-mode route checks for SPEC decision 172."""

import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.experimental import jet

from syncem.base.channels import Channels
from syncem.engine._jet_routine import JET_MAX_ORDER_ENV, chunk_views
from syncem.engine._taylor_jet import taylor_mode_coefficients
from syncem.engine._taylor_rules import taylor_rules
from syncem.engine.average import AveragedKernel, uniform_rule
from syncem.engine.basis import build_from_averaged
from syncem.engine.provider_jet import jet_columns, make_routine, taylor_coefficients
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import Legendre
from tests.v080.test_v080_jet_reference import (
    cosine_kernel,
    e1_kernel,
    e1_lattice,
    three_kernel,
)


def _full(d, order):
    if d == 1:
        return tuple((n,) for n in range(order + 1))
    return tuple((n, *e) for n in range(order + 1) for e in _full(d - 1, order - n))


def _compare(averaged, names, charts, references, widths, n_nu=8):
    exponents = _full(len(names), 5)
    nested = make_routine(names, charts, exponents, method="nested")
    taylor = make_routine(names, charts, exponents, method="taylor")
    view = chunk_views(Channels.log(1, 1, 1, unit="GHz", n_nu=n_nu), 0)[0]
    for ref in references:
        for left, right in zip(
            nested(averaged, view, ref, widths), taylor(averaged, view, ref, widths)
        ):
            scale = np.max(np.abs(left))
            assert np.max(np.abs(left - right)) <= 1e-13 * scale


@pytest.mark.slow
def test_both_methods_agree_on_continuum_at_the_corner_cells():
    cells = e1_lattice(5).cells
    refs = cells.reference[np.asarray([0, cells.n_cells - 1])]
    _compare(e1_kernel(), ("gamma",), ("log",), refs, cells.half_width[0])


@pytest.mark.parametrize("d", (2, 3))
def test_both_methods_agree_on_user_kernel(d):
    kernel = three_kernel().kernel
    fixed = {"c": 0.1} if d == 2 else {}
    names = ("a", "b", "c")[:d]
    _compare(
        AveragedKernel(kernel, fixed=fixed),
        names,
        ("log", "log", "affine")[:d],
        ((1.0, 0.5, 0.1)[:d], (4.0, 2.0, 0.9)[:d]),
        (0.3, 0.2, 0.3)[:d],
    )


def test_both_methods_agree_on_legendre_with_circular_polarization():
    _compare(cosine_kernel(), ("g",), ("log",), ((1.0,), (100.0,)), (0.4,))


def test_cases_of_order_zero_one_and_twelve():
    def f(z):
        return {"I": [jnp.exp(z[0]), jnp.cos(z[0])], "constant": 2.0}

    exponents = ((12,), (1,), (0,), (0,), (6,))
    tree = taylor_mode_coefficients(f, 1, exponents)
    np.testing.assert_array_equal(tree["constant"], [0, 0, 2, 2, 0])
    out = tree["I"]
    np.testing.assert_allclose(
        out[0], [1 / math.factorial(e[0]) for e in exponents], rtol=2e-14
    )
    np.testing.assert_allclose(
        out[1], [1 / math.factorial(12), 0, 1, 1, -1 / math.factorial(6)], rtol=2e-14
    )
    for ex in (((0,),), ((0,), (1,))):
        nested = taylor_coefficients(lambda z: jax.tree.map(jnp.asarray, f(z)), 1, ex)
        taylor = taylor_mode_coefficients(f, 1, ex)
        for a, b in zip(jax.tree.leaves(nested), jax.tree.leaves(taylor)):
            np.testing.assert_array_equal(a, b)


def test_sparse_sixth_order_rows_use_all_monomials_in_the_solve():
    def f(z):
        a, b, c = z
        return 3 * a**3 * b**2 * c - 5 * a * b**2 * c**3 + 10 * b**6 + a**6

    exponents = ((0, 6, 0), (3, 2, 1), (0, 0, 0), (1, 2, 3), (3, 2, 1))
    np.testing.assert_allclose(
        taylor_mode_coefficients(f, 3, exponents), [10, 3, 0, -5, 3], atol=2e-13
    )


def test_moving_nodes_inside_the_scanned_average():
    k = three_kernel().kernel
    avg = AveragedKernel(
        k,
        [("c", uniform_rule(0.0, lambda pt: pt["a"], 5, parents=("a",)))],
        fixed={"b": 1.5},
    )
    ch = Channels.log(1, 1, 1, unit="GHz", n_nu=8)
    axes = {"a": ("affine", 1.0, 0.3)}
    exps = [(n,) for n in range(7)]
    got = jet_columns(avg, ch, axes, exps)
    expected = jet_columns(
        avg,
        ch,
        axes,
        exps,
        routine=lambda n, c, e: make_routine(n, c, e, method="nested"),
    )
    for a, b in zip(got, expected):
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-13 * np.max(np.abs(b)))


def test_error_if_is_live_when_only_derivatives_are_requested():
    def f(z):
        return eqx.error_if(jnp.exp(z[0]), z[0] >= 0, "bad Taylor primal")

    # Use a compiled run because Equinox dispatches the check through a callback.
    with pytest.raises(Exception, match="bad Taylor primal"):
        jax.jit(lambda: taylor_mode_coefficients(f, 1, ((6,),)))().block_until_ready()


def test_scoped_rules_restore_the_registry_after_an_error():
    before = dict(jet.jet_rules)
    with pytest.raises(RuntimeError, match="check restoration"):
        with taylor_rules(), taylor_rules():
            raise RuntimeError("check restoration")
    assert jet.jet_rules == before


def test_order_boundary_s2_keys_and_channel_reuse(monkeypatch):
    avg = cosine_kernel()
    lattice = Lattice(g=Cells.log(1, 4, 1, order=6), mu=Legendre(4))
    ch = Channels.log(1, 2, 2, unit="GHz", n_nu=4)
    lower = build_from_averaged(avg, ch, lattice.with_order(5), routes="jet")
    high = build_from_averaged(avg, ch, lattice, routes="jet")
    assert dict(high.record.events)["S2 columns reused"] == 0
    subset = build_from_averaged(avg, ch[[0]], lattice, routes="jet")
    assert dict(subset.record.events)["S2 columns computed"] == 0
    np.testing.assert_array_equal(subset.spectra0, high.spectra0[:1])
    assert dict(lower.record.events)["S2 columns computed"] > 0
    monkeypatch.setenv(JET_MAX_ORDER_ENV, "5")
    with pytest.raises(MemoryError, match="order 6"):
        build_from_averaged(avg, ch, lattice, routes="jet")


def test_constant_equinox_predicate_is_a_pass_through():
    def f(z):
        return eqx.error_if(jnp.exp(z[0]), False, "not taken")

    got = jax.jit(lambda: taylor_mode_coefficients(f, 1, ((6,),)))()
    np.testing.assert_allclose(got, [1.0 / 720.0], rtol=2e-14)


def test_unhandled_custom_derivative_is_refused_instead_of_dropped():
    with pytest.raises(TypeError, match="custom JVP or VJP"):
        taylor_mode_coefficients(lambda z: jnp.sinc(z[0]), 1, ((6,),))


def test_sparse_axes_avoid_a_large_dense_solve():
    axis_rows = tuple(tuple(12 if j == i else 0 for j in range(7)) for i in range(7))
    got = taylor_mode_coefficients(
        lambda z: jnp.sum(z**12), 7, (axis_rows[0], (0,) * 7, *axis_rows[1:])
    )
    np.testing.assert_allclose(got, [1, 0, 1, 1, 1, 1, 1, 1], rtol=2e-14)
