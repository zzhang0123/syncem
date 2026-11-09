"""Composed channel responses at extreme widths, centres and orders.

Each check bypasses the ``build_basis`` family dispatch and compares two
methods directly:

(a) ``combine_basis`` on a reference basis built once against ``build_basis``
    on a ``Channels.from_callable`` channel whose ``fn`` is the same
    combination ``sum_k c_jk phi_k``;
(b) ``fit_bandpass`` on a table whose linear interpolant lies in the span of
    tent references (``Channels.from_table`` of the identity), which must
    return the table values as coefficients and a zero residual;
(c) ``Channels.from_callable`` with the bump formula against ``Channels.bump``.

Configurations (centre ``c``, half-width ``w``): moderate ``w/c = 0.3``,
narrow ``w/c = 1e-3``, a support edge near zero frequency
(``w/c = 1 - 1e-6``, so ``nu_lo ~ 1e-6 c``) and wide ``w/c = 0.9``.
Truncation orders ``N = 0`` and ``N = 3`` (the highest of
``CERTIFIED_ORDERS``) on the tensor route, ``N = 0, 1`` on the product route,
and ``N = 3`` on the product route in the ``slow`` tier.

On the tensor route the angular nodes do not depend on the channel support,
so (a) holds to rounding for any combination. The narrow channel is left out
there: no tensor node places a line inside a channel of relative width 2e-3,
and both methods return zero columns. On the product route the cells are
bounded by the channel support, so (a) holds to rounding for rows with one
reference each. For general rows the direct build integrates one cell over
the hull, across the flat zeros of the inner references; it converges to the
``combine_basis`` columns as the nodes are refined (the ``slow`` test below,
measured 9e-5 for the moderate case at N = 2 and 3e-6 for the narrow case at
N = 1, with 128 outer and inner nodes).
"""

from functools import lru_cache

import jax.numpy as jnp
import numpy as np
from numpy.testing import assert_allclose
import pytest

from syncem.model.bandpass import fit_bandpass
from syncem.model.basis import build_basis, combine_basis
from syncem.model.channels import Channels
from syncem.model.harmonic import HarmonicKernel
from syncem.model.index import Truncation
from syncem.model.kernels import required_m_max
from syncem.model.moments import Reference

from _basis_fixtures import SMALL_SUPPORT
from _harmonic_oracles import B0, GAMMA0, NU_STAR, S_DEPTH

REFERENCE = Reference(GAMMA0, B0, depth_ref=3.0 * S_DEPTH, scales=(GAMMA0, B0, S_DEPTH))
# name -> (centre / NU_STAR, half-width / centre)
CONFIGS = {
    "moderate": (2.0, 0.3),
    "narrow": (2.0, 1e-3),
    "near_zero_edge": (1.2, 1.0 - 1e-6),
    "wide": (2.0, 0.9),
}
MIXED = np.array([[0.5, 1.0, 0.25], [0.0, -0.3, 0.8]])
SINGLE = np.array([[0.0, 1.0, 0.0], [0.0, 0.0, 2.0]])
ROUNDING = 1e-12
FAST_N = {"tensor": (0, 3), "product": (0, 1)}


def bump_formula(c, w):
    def fn(nu):
        t = (nu - c) / w
        return jnp.exp(1.0 - 1.0 / jnp.maximum(1.0 - t * t, 1e-3))

    return fn


def combination(ref, coefficients):
    """One callable per row, ``nu -> sum_k c_jk phi_k(nu)``, on the hull support."""
    combined = Channels.from_basis(ref, coefficients)
    fns = [
        (lambda nu, row=jnp.asarray(row): jnp.tensordot(row, ref(nu), axes=1))
        for row in coefficients
    ]
    return combined, Channels.from_callable(
        fns, np.asarray(combined.support), smoothness=99
    )


@lru_cache(maxsize=None)
def channel_set(name):
    """References, the two combinations and the bump pair of one configuration."""
    c_ratio, r = CONFIGS[name]
    c = c_ratio * NU_STAR
    w = r * c
    centres = c + np.array([-0.15, 0.0, 0.15]) * w
    # Keep every reference edge at positive frequency.
    widths = np.minimum(w, centres * (1.0 - 1e-6))
    ref = Channels.bump(centres, widths)
    bump = Channels.bump([c], [w])
    call = Channels.from_callable(
        bump_formula(c, w), np.asarray(bump.support)[0], smoothness=99
    )
    return {
        "ref": ref,
        "mixed": combination(ref, MIXED),
        "single": combination(ref, SINGLE),
        "bump": bump,
        "callable_bump": call,
    }


def kernel_for(channels, n_nodes=16):
    m_max = required_m_max(SMALL_SUPPORT, channels) + 2
    return HarmonicKernel(m_max, n_outer=n_nodes, n_inner=n_nodes, n_mu=24, n_eta=24)


@lru_cache(maxsize=None)
def bases(name, N, route, rows, n_nodes=16):
    """``(combine_basis, direct callable basis)`` of the ``rows`` combination."""
    ch = channel_set(name)
    combined, direct = ch[rows]
    kernel = kernel_for(combined, n_nodes)
    kw = dict(support=SMALL_SUPPORT, quadrature=route, convergence=False)
    truncation = Truncation(1, 1, N)
    by_reference = build_basis(kernel, ch["ref"], truncation, REFERENCE, **kw)
    by_direct = build_basis(kernel, direct, truncation, REFERENCE, **kw)
    return combine_basis(by_reference, combined), by_direct


@lru_cache(maxsize=None)
def bump_bases(name, N, route):
    ch = channel_set(name)
    kernel = kernel_for(ch["bump"])
    kw = dict(support=SMALL_SUPPORT, quadrature=route, convergence=False)
    truncation = Truncation(1, 1, N)
    return (
        build_basis(kernel, ch["callable_bump"], truncation, REFERENCE, **kw),
        build_basis(kernel, ch["bump"], truncation, REFERENCE, **kw),
    )


def block_rel_err(a, b):
    """``max_X max |C_a - C_b| / max(|C_a|, |C_b|)`` over the rows of Stokes ``X``."""
    a, b = np.asarray(a.response_matrix()), np.asarray(b.response_matrix())
    assert np.all(np.isfinite(a)) and np.all(np.isfinite(b))
    errors = []
    for x in range(4):
        scale = max(np.max(np.abs(a[x::4])), np.max(np.abs(b[x::4])))
        errors.append(np.max(np.abs(a[x::4] - b[x::4])) / scale if scale > 0 else 0.0)
    assert max(np.max(np.abs(a)), np.max(np.abs(b))) > 0, "degenerate: zero columns"
    return float(max(errors))


def cells(route, names):
    fast = [(name, N) for name in names for N in FAST_N[route]]
    slow = [
        pytest.param(name, 3, marks=pytest.mark.slow, id=f"{name}-3")
        for name in names
        if 3 not in FAST_N[route]
    ]
    return fast + slow


TENSOR_NAMES = ("moderate", "near_zero_edge", "wide")


# -- (a) from_basis against the direct callable ------------------------------------


@pytest.mark.parametrize("name, N", cells("tensor", TENSOR_NAMES))
def test_combine_equals_direct_callable_on_the_tensor_route(name, N):
    by_combine, by_direct = bases(name, N, "tensor", "mixed")
    assert block_rel_err(by_combine, by_direct) < ROUNDING


@pytest.mark.parametrize("name, N", cells("product", tuple(CONFIGS)))
def test_combine_equals_direct_callable_for_single_reference_rows(name, N):
    by_combine, by_direct = bases(name, N, "product", "single")
    assert block_rel_err(by_combine, by_direct) < ROUNDING


@pytest.mark.slow
@pytest.mark.parametrize("name, N", [("moderate", 2), ("narrow", 1)])
def test_product_route_methods_converge_to_the_same_columns(name, N):
    """The direct hull cell converges slowly toward the ``combine_basis`` columns.

    Measured (largest Stokes-block discrepancy against ``combine_basis`` at
    128 nodes): moderate, N = 2: direct 0.36 / 2.3e-2 / 4.6e-3 / 9.3e-5 at
    16 / 32 / 64 / 128 nodes, combine 1.4e-2 / 2.8e-4 / 4.6e-7 at 16 / 32 /
    64; narrow, N = 1: direct 0.13 / 5.9e-3 / 4.7e-5 / 3.5e-6, combine
    4.9e-3 / 3.0e-4 / 2.0e-7.
    """
    reference_route, _ = bases(name, N, "product", "mixed", n_nodes=64)
    coarse, _ = bases(name, N, "product", "mixed", n_nodes=32)
    _, direct = bases(name, N, "product", "mixed", n_nodes=128)
    assert block_rel_err(coarse, reference_route) < 1e-3
    assert block_rel_err(direct, reference_route) < 1e-3


# -- (c) the bump formula as a callable ----------------------------------------------


@pytest.mark.parametrize("name, N", cells("tensor", TENSOR_NAMES))
def test_callable_bump_equals_bump_on_the_tensor_route(name, N):
    assert block_rel_err(*bump_bases(name, N, "tensor")) < ROUNDING


@pytest.mark.parametrize("name, N", cells("product", tuple(CONFIGS)))
def test_callable_bump_equals_bump_on_the_product_route(name, N):
    assert block_rel_err(*bump_bases(name, N, "product")) < ROUNDING


@pytest.mark.parametrize("name", tuple(CONFIGS))
def test_callable_bump_response_equals_bump_at_and_near_the_edges(name):
    ch = channel_set(name)
    bump, call = ch["bump"], ch["callable_bump"]
    lo, hi = np.asarray(bump.support)[0]
    c, w = float(bump.centres_hz[0]), float(bump.widths_hz[0])
    nu = np.concatenate(
        [
            c + w * np.linspace(-1.2, 1.2, 241),
            [lo, hi, np.nextafter(lo, np.inf), np.nextafter(hi, -np.inf)],
        ]
    )
    nu = nu[nu > 0]
    assert_allclose(np.asarray(call(nu)), np.asarray(bump(nu)), rtol=0, atol=0)


# -- (b) a table in the span of the references --------------------------------------


def asymmetric_table(grid, lo, hi):
    """A skewed bandpass with a ripple, zero at both ends of the grid."""
    t = (grid - lo) / (hi - lo)
    inside = (t > 0) & (t < 1)
    shape = np.zeros_like(grid)
    tt = t[inside]
    shape[inside] = np.sqrt(tt) * (1 - tt) ** 2 * (1 + 0.4 * np.sin(7 * tt))
    return shape


@pytest.mark.parametrize("n_pts", [9, 65])
@pytest.mark.parametrize("name", tuple(CONFIGS))
def test_fit_bandpass_in_the_span_of_tents_is_exact(name, n_pts):
    c_ratio, r = CONFIGS[name]
    c = c_ratio * NU_STAR
    lo, hi = c * (1 - r), c * (1 + r)
    grid = np.linspace(lo, hi, n_pts)
    tents = Channels.from_table(grid, np.eye(n_pts)[1:-1], smoothness=0)
    table = np.stack(
        [
            asymmetric_table(grid, lo, hi),
            np.where(np.abs(grid - c) < 0.5 * r * c, 1.0, 0.0),
        ]
    )
    table[:, [0, -1]] = 0.0
    fit = fit_bandpass(tents, grid, table)
    # Rounding of the least-squares solve, relative to the table peak.
    assert_allclose(
        np.asarray(fit.coefficients), table[:, 1:-1], rtol=0, atol=1e-14 * table.max()
    )
    assert float(np.max(fit.relative_residual)) < 1e-13
    assert float(np.max(fit.uncovered)) == 0.0
    assert fit.open_edges == ()
    term = fit.response_uncertainty()
    assert term.kind == "estimate"
    assert float(np.max(term.value)) < 1e-12
