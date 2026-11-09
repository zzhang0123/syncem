"""Boundary of the ``from_basis`` route at reference width = channel width.

An instrument channel ``bump(c, W)`` is fitted by ``2K + 1`` bump references
of width ``w = W (1 + delta)`` centred on ``c + k s``. At ``delta = 0`` the
channel is the ``k = 0`` reference and ``from_basis`` must equal the directly
integrated channel to rounding. For ``delta > 0`` wider references cannot
represent the narrower channel: the fitted value residual is about ``delta``
of the peak, and the column discrepancy follows it (measured 1.5 to 1.9 times
the residual for ``W/c = 0.05`` and 6 to 14 times for ``W/c = 0.4``, per
Stokes block; the columns carry ``R'`` and ``R''`` through the ``z``
derivatives, so no bound relates the two). Both methods are built directly
(``build_basis`` on ``Channels.bump`` and on the fitted ``Channels.from_basis``),
not through a dispatcher. The width ratios include a narrow channel
(``W/c = 0.05``) and a broad one (``W/c = 0.4``). The discrepancy is
measured per Stokes block (largest ``|dC|`` over the largest ``|C|`` of that
Stokes component); a per-column ratio inflates columns that nearly vanish.

The population check uses ``direct_channel_average``, which is exactly linear
in the response at every line, so ``S(R_table) - S(sum_k c_k phi_k)`` is the
response error ``int Delta R dS`` of ``eq: chromatic beam response``; the
fitted covering must dominate it in every Stokes component.
"""

import numpy as np
from numpy.testing import assert_allclose
import pytest

from syncem.model.bandpass import fit_bandpass
from syncem.model.basis import build_basis
from syncem.model.channels import Channels
from syncem.model.harmonic import HarmonicKernel
from syncem.model.index import Truncation
from syncem.model.kernels import required_m_max
from syncem.model.moments import PopulationSamples, Reference
from syncem.model.predict import direct_channel_average

from _basis_fixtures import SMALL_SUPPORT
from _harmonic_oracles import B0, GAMMA0, NU_STAR, S_DEPTH

CENTRE = 2.0 * NU_STAR
K, SPACING = 2, 0.2  # references at c + k SPACING w, k = -K..K
WIDTH_RATIOS = (0.05, 0.4)
DELTAS = (0.3, 0.1, 0.03, 0.01, 0.0)


def _references(W, delta):
    w = W * (1.0 + delta)
    centres = CENTRE + SPACING * w * np.arange(-K, K + 1)
    return Channels.bump(centres, np.full(centres.size, w))


def _table(W, n=4001):
    """The channel sampled on a grid symmetric about c (so its peak is sampled)."""
    # Covers the hull of the widest reference set, c +/- 1.3 W (1 + K SPACING).
    grid = CENTRE + np.linspace(-2.0 * W, 2.0 * W, n)
    return grid, np.asarray(Channels.bump([CENTRE], [W])(grid))[0]


def _kernel(channels):
    m_max = required_m_max(SMALL_SUPPORT, channels) + 2
    return HarmonicKernel(m_max, n_outer=16, n_inner=16, n_mu=24, n_eta=24)


def _rel_err(a, b):
    """``max_X max |C_a - C_b| / max |C_b|`` over the rows of Stokes ``X``."""
    a, b = np.asarray(a.response_matrix()), np.asarray(b.response_matrix())
    return max(
        float(np.max(np.abs(a[x::4] - b[x::4])) / np.max(np.abs(b[x::4])))
        for x in range(4)
    )


def _column_rel_err(a, b):
    """``max_a max_j |dC_ja| / max_j |C_ja|`` over nonzero columns."""
    a, b = np.asarray(a.response_matrix()), np.asarray(b.response_matrix())
    scale = np.max(np.abs(np.stack([a, b])), axis=(0, 1))
    keep = scale > 0
    return float(np.max(np.abs(a - b)[:, keep] / scale[keep]))


@pytest.fixture(scope="module", params=WIDTH_RATIOS, ids=lambda r: f"W/c={r}")
def sweep(request):
    W = request.param * CENTRE
    reference = Reference(
        GAMMA0, B0, depth_ref=3.0 * S_DEPTH, scales=(GAMMA0, B0, S_DEPTH)
    )
    widest = _references(W, max(DELTAS))
    kernel = _kernel(widest)
    truncation = Truncation(1, 1, 2)
    kw = dict(support=SMALL_SUPPORT, convergence=False)
    direct = build_basis(
        kernel, Channels.bump([CENTRE], [W]), truncation, reference, **kw
    )
    grid, table = _table(W)
    rows = []
    for delta in DELTAS:
        fit = fit_bandpass(_references(W, delta), grid, table)
        basis = build_basis(kernel, fit.channels, truncation, reference, **kw)
        rows.append((delta, fit, basis))
    return W, kernel, grid, table, rows, direct


def test_methods_agree_at_the_boundary(sweep):
    *_, rows, direct = sweep
    delta, fit, basis = rows[-1]
    assert delta == 0.0
    assert_allclose(np.asarray(fit.coefficients)[0, K], 1.0, rtol=1e-10)
    # The other least-squares coefficients fall below the 1e-13 floor.
    assert np.count_nonzero(np.asarray(fit.coefficients)) == 1
    assert _rel_err(basis, direct) < 1e-12
    assert _column_rel_err(basis, direct) < 1e-10


def test_agreement_improves_toward_the_boundary(sweep):
    *_, rows, direct = sweep
    errors = [_rel_err(basis, direct) for _, _, basis in rows]
    residuals = [float(np.max(fit.relative_residual)) for _, fit, _ in rows]
    assert all(np.isfinite(errors))
    assert all(b < a for a, b in zip(errors, errors[1:])), errors
    assert all(b < a for a, b in zip(residuals, residuals[1:])), residuals
    for delta, error, residual in zip(DELTAS[:-1], errors, residuals):
        # The residual is about delta; the ratio is measured, not derived.
        assert 0.5 * delta < residual < 2.0 * delta, (delta, residual)
        assert error < 20.0 * residual, (delta, error, residual)
    # delta = 0.01 lies in the warn band (1e-3, 1e-1) of the boundary
    # methodology, as its representation residual (1e-2) requires.
    assert errors[DELTAS.index(0.01)] < 1e-1, errors


def _samples(n=12, seed=7):
    rng = np.random.default_rng(seed)
    g_lo, g_hi = SMALL_SUPPORT.gamma
    b_lo, b_hi = SMALL_SUPPORT.B
    return PopulationSamples(
        rng.uniform(float(g_lo), float(g_hi), n),
        rng.uniform(float(b_lo), float(b_hi), n),
        rng.uniform(-0.95, 0.95, n),
        rng.uniform(-0.95, 0.95, n),
        rng.uniform(0.0, np.pi, n),
        rng.uniform(0.0, 5.0 * S_DEPTH, n),
        weights=rng.uniform(0.5, 2.0, n),
    )


@pytest.mark.parametrize("delta", [0.3, 0.03, 0.0])
def test_response_error_covering_bounds_the_population_difference(sweep, delta):
    W, kernel, grid, table, rows, _ = sweep
    fit = next(f for d, f, _ in rows if d == delta)
    samples = _samples()
    measured = Channels.from_table(grid, table, smoothness=0)
    kw = dict(amplitude=1.0)
    s_true = np.asarray(direct_channel_average(samples, kernel, measured, **kw).stokes)
    s_fit = np.asarray(
        direct_channel_average(samples, kernel, fit.channels, **kw).stokes
    )
    s_ref = np.asarray(
        direct_channel_average(samples, kernel, fit.reference, **kw).stokes
    )
    # The direct average is linear in the response at every line.
    assert_allclose(
        np.asarray(fit.operator()) @ s_ref.ravel(),
        s_fit.ravel(),
        rtol=1e-11,
        atol=1e-13 * np.max(np.abs(s_ref)),
    )
    bound = np.asarray(fit.response_uncertainty().value) @ np.abs(s_ref.ravel())
    difference = np.abs(s_true - s_fit).ravel()
    assert np.all(difference <= bound), (difference, bound)
