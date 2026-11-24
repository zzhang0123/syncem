"""Integer inputs and the on-axis Doppler factor of the legacy kernels (2026-11-23 review).

The three modules match v0.7.10 except for these numerical-range corrections.
References: float evaluation for the integer cases, and a 60-digit mpmath
evaluation of the Schott formulas of ``syncem.stokes`` for the harmonic.
"""

import itertools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syncem.constants import C_CGS, E_ESU, M_E
from syncem.conversion import (
    conversion_coefficient,
    conversion_rotation_ratio,
    mueller_conversion,
    mueller_rotation,
)
from syncem.rm import burn_depolarisation, rotation_angle
from syncem.stokes import larmor_power, stokes_harmonic

jax.config.update("jax_enable_x64", True)


@pytest.mark.parametrize("nu", [1_000_000_000, 5_000_000_000, 300_000_000_000])
def test_conversion_rates_take_integer_frequencies(nu):
    for transform in (jnp.asarray, np.asarray, int, jax.jit(lambda x: x)):
        value = transform(nu)
        np.testing.assert_allclose(
            mueller_conversion(value, 1.0, 5e-6),
            mueller_conversion(float(nu), 1.0, 5e-6),
            rtol=1e-15,
        )
        np.testing.assert_allclose(
            mueller_rotation(value, 1.0, 5e-6),
            mueller_rotation(float(nu), 1.0, 5e-6),
            rtol=1e-15,
        )
        np.testing.assert_allclose(
            conversion_rotation_ratio(value, 5e-6, 2e-6),
            conversion_rotation_ratio(float(nu), 5e-6, 2e-6),
            rtol=1e-15,
        )
    jitted = jax.jit(conversion_coefficient)(jnp.asarray(nu), 1.0, 5e-6)
    assert float(jitted) < 0


def test_conversion_pins_the_signed_value():
    got = mueller_conversion(jnp.asarray(1_000_000_000), 1.0, 5e-6)
    np.testing.assert_allclose(got, -1.654914500031242e-27, rtol=1e-12)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.int64])
@pytest.mark.parametrize("lam", [1, 216, 5000])
def test_gaussian_screen_takes_integer_wavelengths(dtype, lam):
    got = burn_depolarisation(1.0, 0.3, 1e-10, jnp.asarray(lam, dtype=dtype))
    want = burn_depolarisation(1.0, 0.3, 1e-10, float(lam))
    np.testing.assert_allclose(got, want, rtol=1e-12)
    assert abs(got) <= 1.0


def test_gaussian_screen_pins_the_review_case():
    got = abs(burn_depolarisation(1.0, 0.0, 1e-10, jnp.int32(216)))
    np.testing.assert_allclose(got, 0.6470339795732095, rtol=1e-12)


def test_rotation_angle_takes_integer_wavelengths():
    np.testing.assert_allclose(
        rotation_angle(jnp.int32(50_000), 1e-6),
        rotation_angle(50_000.0, 1e-6),
        rtol=1e-15,
    )


def _reference(n, gamma, alpha, theta, B=1.0):
    mp = pytest.importorskip("mpmath")
    mp.mp.dps = 60
    g, a, t = mp.mpf(gamma), mp.mpf(alpha), mp.mpf(theta)
    beta = mp.sqrt(1 - 1 / g**2)
    b_par, b_perp = beta * mp.cos(a), beta * mp.sin(a)
    D = 1 - b_par * mp.cos(t)
    x = n * b_perp * mp.sin(t) / D
    neighbours = mp.besselj(n - 1, x) + mp.besselj(n + 1, x)
    par = (mp.cos(t) - b_par) * b_perp * neighbours / (2 * D)
    perp = b_perp * mp.besselj(n, x, derivative=1)
    factor = mp.mpf(n) ** 2 / D**3
    wB = mp.mpf(E_ESU) * B / (g * mp.mpf(M_E) * mp.mpf(C_CGS))
    pref = mp.mpf(E_ESU) ** 2 * wB**2 / (2 * mp.pi * mp.mpf(C_CGS))
    return tuple(
        float(pref * factor * v)
        for v in (par**2 + perp**2, par**2 - perp**2, 2 * par * perp)
    )


@pytest.mark.parametrize(
    ("gamma", "want"), [(1e8, 0.147095804907), (2e8, 2.15568040071)]
)
def test_on_axis_harmonic_at_high_gamma(gamma, want):
    I, Q, V = stokes_harmonic(1, gamma, 1e-9, 0.0, 1.0)
    np.testing.assert_allclose(I, want, rtol=1e-10)
    np.testing.assert_allclose(I, _reference(1, gamma, 1e-9, 0.0)[0], rtol=1e-11)


@pytest.mark.parametrize("dtype", [jnp.int32, jnp.int64])
def test_integer_gamma_keeps_the_physical_power(dtype):
    gamma = dtype(200_000_000)
    stokes = jax.jit(lambda g: stokes_harmonic(1, g, 1e-9, 0.0, 1.0))(gamma)
    np.testing.assert_allclose(
        stokes, [2.15568040071, 0, 2.15568040071], rtol=1e-10, atol=1e-14
    )
    np.testing.assert_allclose(
        larmor_power(gamma, 0.3, 1.0), larmor_power(2e8, 0.3, 1.0), rtol=1e-13
    )


@pytest.mark.parametrize(
    ("gamma", "alpha", "theta", "field", "reference"),
    [
        (
            jnp.int32(200_000_000),
            np.float32(1e-9),
            np.float32(0),
            1.0,
            (2.1556802928458945, 0.0, 2.1556802928458945),
        ),
        (
            1e10,
            np.nextafter(np.pi - 1e-10, 0.0),
            np.pi,
            1.0,
            (18944.010464954805, -2.130843544692605e-08, -18944.010464954805),
        ),
        (
            1e15,
            np.pi - 5 * np.spacing(np.pi),
            np.pi - 2 * np.spacing(np.pi),
            1.0,
            (11883985342857.875, -2469535522190.6772, -11624564591153.479),
        ),
        (
            1e15,
            np.pi - 5 * np.spacing(np.pi),
            np.pi - 3 * np.spacing(np.pi),
            np.float32(1),
            (5355767425353.167, -2480245810123.1226, -4746854267390.226),
        ),
    ],
)
def test_harmonic_at_axis_and_dtype_boundaries(gamma, alpha, theta, field, reference):
    # Schott formula at 90 decimal digits, using the supplied angles after dtype rounding.
    want = np.asarray(reference)
    evaluate = jax.jit(lambda g, a, t, b: stokes_harmonic(1, g, a, t, b))
    got = np.array(evaluate(gamma, alpha, theta, field))
    np.testing.assert_allclose(got / want[0], want / want[0], rtol=1e-12, atol=1e-12)


_GAMMA = (1.001, 2.0, 100.0, 1e4, 1e8, 1e10)
_ALPHA = (1e-9, 0.3, np.pi / 2, np.pi - 1e-9)
_THETA_OFFSETS = (0.0, 1e-9, 1e-4)


@pytest.mark.parametrize("n", [1, 2, 5])
@pytest.mark.parametrize(("gamma", "alpha"), list(itertools.product(_GAMMA, _ALPHA)))
def test_harmonic_matches_high_precision_on_a_grid(n, gamma, alpha):
    # Views on and near the cone (theta = alpha), on the axis and broadside.
    thetas = (
        *(alpha + d for d in _THETA_OFFSETS if alpha + d <= np.pi),
        0.0,
        np.pi / 2,
        np.pi,
    )
    for theta in thetas:
        got = np.asarray(stokes_harmonic(n, gamma, alpha, theta, 1.0))
        want = np.asarray(_reference(n, gamma, alpha, theta))
        assert np.all(np.isfinite(got))
        scale = max(abs(want[0]), 1e-300)
        # Absolute error relative to I: Q and V may vanish where I does not.
        assert np.max(np.abs(got - want)) / scale < 1e-8, (theta, got, want)
