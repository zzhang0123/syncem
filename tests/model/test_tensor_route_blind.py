"""Tensor angular route at narrow channels: blind grids, the refusal and the notes.

The tensor route of ``HarmonicKernel`` evaluates the lines on fixed
``n_mu x n_eta`` Gauss-Legendre nodes. For a ``bump`` of centre ``c`` the
columns vanish identically when the half-width ``w`` is at most
``w0 = min_{m, i, k} |nu_m(mu_i, eta_k) - c|``, because ``Channels`` masks the
response outside the open support. ``w0`` comes from a NumPy oracle below,
independent of ``_basis_core.tensor_node_lines``.

Each check compares the two routes directly through ``_build_core``, bypassing
the ``build_basis`` refusal: on both sides of ``w0`` (``w0 (1 - 1e-6)`` and
``1.05 w0``) for square, rectangular and coincidence grids (``128 x 128`` has a
node line at ``|nu/c - 1| = 2.7e-6``), at the extreme widths ``w/c = 1e-6`` and
``0.9``, and at the 24 x 24 configuration with ``w/c = 1e-3`` that first showed
the zeros, for ``N = 0..3``. The narrow-channel note is checked on both sides
of ``TENSOR_MIN_WIDTH / min(n_mu, n_eta)``. The node-line count is checked
against a brute-force NumPy count, and the refusal against several channels
and the reference channels of ``Channels.from_basis``. The measured
thresholds are in ``docs/DESIGN.md`` (sampled, not bounds).
"""

import numpy as np
import pytest

from syncem.model import ContinuumKernel
from syncem.model import _basis_core as core
from syncem.model.basis import _build_core, build_basis
from syncem.model.channels import Channels
from syncem.model.harmonic import HarmonicKernel
from syncem.model.index import MomentIndex, Truncation
from syncem.model.kernels import required_m_max
from syncem.model.moments import Reference
from syncem.model.phase import TaylorPhase

from _basis_fixtures import SMALL_SUPPORT
from _harmonic_oracles import B0, GAMMA0, NU_STAR, S_DEPTH

REFERENCE = Reference(GAMMA0, B0, depth_ref=3.0 * S_DEPTH, scales=(GAMMA0, B0, S_DEPTH))
CENTRE = 2.0 * NU_STAR  # the m = 2 line crosses the channel centre at mu eta = 0
GRIDS = [(8, 8), (24, 24), (64, 64), (128, 128), (24, 48), (12, 96)]


def kernel_for(channels, n_mu=24, n_eta=24):
    m_max = required_m_max(SMALL_SUPPORT, channels) + 2
    return HarmonicKernel(m_max, n_outer=16, n_inner=16, n_mu=n_mu, n_eta=n_eta)


def channel(half_width):
    return Channels.bump([CENTRE], [half_width])


def blind_half_width(n_mu, n_eta, m_max):
    """``min |nu_m(mu_i, eta_k) - c|`` over harmonics and tensor nodes (NumPy oracle)."""
    x, _ = np.polynomial.legendre.leggauss(n_mu)
    y, _ = np.polynomial.legendre.leggauss(n_eta)
    beta = np.sqrt(1.0 - 1.0 / GAMMA0**2)
    m = np.arange(1, m_max + 1, dtype=float)[:, None, None]
    nu = m * NU_STAR / (1.0 - beta * x[None, :, None] * y[None, None, :])
    return float(np.min(np.abs(nu - CENTRE)))


def route_columns(kernel, channels, N, route):
    """``C`` of one route through ``_build_core`` (``build_basis`` dispatch bypassed)."""
    kernel = kernel.for_support(SMALL_SUPPORT, channels, reference=REFERENCE)
    truncation = Truncation(1, 1, N)
    index = MomentIndex.build(truncation, components=tuple(kernel.components))
    phase = TaylorPhase(truncation.max_b())
    blocks = _build_core(
        kernel, channels, REFERENCE, phase, index=index, quadrature=route
    )
    return np.asarray(core.response_matrix(index, channels.n_ch, *blocks))


def build(kernel, channels, N=0, route="tensor", **kw):
    kw.setdefault("convergence", False)
    return build_basis(
        kernel,
        channels,
        Truncation(1, 1, N),
        REFERENCE,
        support=SMALL_SUPPORT,
        quadrature=route,
        **kw,
    )


def mass_error(tensor, product):
    return abs(tensor[0, 0] - product[0, 0]) / abs(product[0, 0])


# -- the configuration that first showed the zeros ----------------------------------------


@pytest.mark.parametrize("N", [0, 1, 2, pytest.param(3, marks=pytest.mark.slow)])
def test_narrow_channel_tensor_columns_zero_where_product_resolves(N):
    channels = channel(1e-3 * CENTRE)
    kernel = kernel_for(channels)
    tensor = route_columns(kernel, channels, N, "tensor")
    product = route_columns(kernel, channels, N, "product")
    assert np.all(tensor == 0.0)
    assert np.all(np.isfinite(product))
    assert np.all(np.max(np.abs(product), axis=0) > 0.0)
    assert core.tensor_blind_channels(kernel, channels, REFERENCE) == (0,)
    with pytest.raises(ValueError, match="tensor route: no node of the 24 x 24"):
        build(kernel, channels, N)


# -- both sides of the blind half-width w0 -------------------------------------------------


@pytest.mark.parametrize("n_mu, n_eta", GRIDS)
def test_blind_threshold_both_sides(n_mu, n_eta):
    probe = kernel_for(channel(1e-3 * CENTRE))
    w0 = blind_half_width(n_mu, n_eta, probe.m_max)
    # Precondition of the noted "seen" side: 1.05 w0 lies below the note threshold.
    assert 2.1 * w0 / CENTRE < core.TENSOR_MIN_WIDTH / min(n_mu, n_eta)
    for side, half_width in (("blind", w0 * (1 - 1e-6)), ("seen", 1.05 * w0)):
        channels = channel(half_width)
        kernel = kernel_for(channels, n_mu, n_eta)
        tensor = route_columns(kernel, channels, 0, "tensor")
        product = route_columns(kernel, channels, 0, "product")
        assert np.all(np.isfinite(tensor)) and np.all(np.isfinite(product))
        assert np.max(np.abs(product)) > 0.0
        blind = core.tensor_blind_channels(kernel, channels, REFERENCE)
        if side == "blind":
            assert blind == (0,)
            assert np.all(tensor == 0.0)
            with pytest.raises(ValueError, match="tensor route: no node"):
                build(kernel, channels)
        else:
            assert blind == ()
            assert np.max(np.abs(tensor)) > 0.0
            # Seen but far below the resolution threshold: still BAD, and noted.
            assert mass_error(tensor, product) > 0.1
            notes = build(kernel, channels).provenance.notes
            assert any("sampled, not a bound" in n for n in notes)


# -- extreme widths ------------------------------------------------------------------------


@pytest.mark.parametrize("n_mu, n_eta", GRIDS)
def test_extreme_narrow_width_is_blind_and_refused(n_mu, n_eta):
    channels = channel(1e-6 * CENTRE)
    kernel = kernel_for(channels, n_mu, n_eta)
    assert np.all(route_columns(kernel, channels, 0, "tensor") == 0.0)
    assert np.max(np.abs(route_columns(kernel, channels, 0, "product"))) > 0.0
    with pytest.raises(ValueError, match="tensor route: no node"):
        build(kernel, channels)


@pytest.mark.parametrize("n_mu, n_eta", GRIDS)
def test_extreme_wide_width_resolved_without_note(n_mu, n_eta):
    channels = channel(0.9 * CENTRE)
    kernel = kernel_for(channels, n_mu, n_eta)
    tensor = route_columns(kernel, channels, 0, "tensor")
    product = route_columns(kernel, channels, 0, "product")
    # Sampled pin of the wide side (DESIGN.md): mass column within 10%.
    assert mass_error(tensor, product) < 0.1
    basis = build(kernel, channels)
    assert not any("tensor route" in n for n in basis.provenance.notes)


# -- the narrow-channel note ---------------------------------------------------------------


def test_narrow_note_both_sides_of_threshold():
    limit = core.TENSOR_MIN_WIDTH / 24  # (nu_hi - nu_lo)/centre at 24 x 24
    for factor, expected in ((1 - 1e-3, True), (1 + 1e-3, False)):
        channels = channel(0.5 * limit * factor * CENTRE)
        kernel = kernel_for(channels)
        assert core.tensor_narrow_channels(kernel, channels) == (
            (0,) if expected else ()
        )
        notes = build(kernel, channels).provenance.notes
        assert any("tensor route: Channel(s) [0]" in n for n in notes) is expected


# -- the route check and column_ratio ------------------------------------------------------


def test_route_check_of_product_basis_names_blind_tensor_grid():
    channels = channel(1e-3 * CENTRE)
    kernel = kernel_for(channels)
    basis = build(
        kernel, channels, route="product", convergence="angular", cross_route=True
    )
    (check,) = [c for c in basis.provenance.finite_checks if c.startswith("route")]
    assert "= 1;" in check
    assert "The 48 x 48 tensor nodes place no line inside channel(s) [0]" in check


def test_column_ratio_reports_a_zero_basis_column_as_infinite():
    C = np.array([[1.0, 0.0], [2.0, 0.0]])
    assert core.column_ratio(np.array([[0.1, 0.0], [0.0, 0.0]]), C) == 0.05
    assert core.column_ratio(np.array([[0.1, 1e-30], [0.0, 0.0]]), C) == np.inf
    assert core.column_ratio(np.zeros((2, 2)), np.zeros((2, 2))) == 0.0
    assert core.column_ratio(np.array([[0.0, np.nan], [0.0, 0.0]]), C) == np.inf


# -- the node-line count, several channels, from_basis and other kernels -------------------


@pytest.mark.parametrize("n_mu, n_eta", [(8, 8), (24, 24), (24, 48), (12, 96), (9, 9)])
def test_node_line_count_matches_brute_force(n_mu, n_eta):
    channels = Channels.bump(
        [1.3 * NU_STAR, CENTRE, CENTRE, 3.1 * NU_STAR],
        [0.2 * NU_STAR, 1e-3 * CENTRE, 0.3 * CENTRE, 0.05 * NU_STAR],
    )
    kernel = kernel_for(channels, n_mu, n_eta)
    counts, active = core.tensor_node_lines(kernel, channels, REFERENCE)
    x, _ = np.polynomial.legendre.leggauss(n_mu)
    y, _ = np.polynomial.legendre.leggauss(n_eta)
    beta = np.sqrt(1.0 - 1.0 / GAMMA0**2)
    m = np.asarray(kernel.harmonics())[:, None, None]
    nu = (m * NU_STAR / (1.0 - beta * x[None, :, None] * y[None, None, :])).ravel()
    lo, hi = np.asarray(channels.support).T
    expected = [int(np.sum((nu > a) & (nu < b))) for a, b in zip(lo, hi)]
    assert counts.tolist() == expected
    assert active.tolist() == [True] * 4


def test_refusal_names_only_the_blind_channel():
    channels = Channels.bump([CENTRE, CENTRE], [0.3 * CENTRE, 1e-3 * CENTRE])
    kernel = kernel_for(channels)
    assert core.tensor_blind_channels(kernel, channels, REFERENCE) == (1,)
    with pytest.raises(ValueError, match=r"support of channel\(s\) \[1\] at"):
        build(kernel, channels)


def test_from_basis_refusal_names_reference_channels():
    ref = Channels.bump([CENTRE, CENTRE], [0.3 * CENTRE, 1e-6 * CENTRE])
    unused = Channels.from_basis(ref, np.array([[1.0, 0.0]]))
    used = Channels.from_basis(ref, np.array([[1.0, 1e-3]]))
    kernel = kernel_for(ref)
    with pytest.raises(ValueError, match=r"reference channel\(s\) \[1\] of") as info:
        build(kernel, unused)
    assert "can be dropped from the reference" in str(info.value)
    with pytest.raises(ValueError, match=r"reference channel\(s\) \[1\] of") as info:
        build(kernel, used)
    assert "can be dropped" not in str(info.value)
    assert np.all(
        np.isfinite(np.asarray(build(kernel, unused, route="product").I_basis))
    )


def test_other_kernels_are_never_blind_or_narrow():
    channels = channel(1e-6 * CENTRE)
    kernel = ContinuumKernel()
    assert core.tensor_blind_channels(kernel, channels, REFERENCE) == ()
    assert core.tensor_narrow_channels(kernel, channels) == ()
    assert core.tensor_route_note(kernel, channels, REFERENCE, "tensor") == ""
