"""Two resolved Beaming cosines share one diagonal quadrature (SPEC Sec. 4.2)."""

import jax
import numpy as np
import pytest

import syncem as se
from syncem.engine._tabulated_rule import with_tabulated_rules
from syncem.engine.average import AveragedKernel

_PITCH = se.Variable("pitch", kind="cosine")
_LOOK = se.Variable("look", kind="cosine")
_NUISANCE = se.Variable("nuisance", kind="cosine")
_BEAM = (se.Beaming("pitch", "look"),)


@se.kernel(
    variables=(_PITCH, _NUISANCE, _LOOK), capabilities=_BEAM, stokes=("I", "P", "V")
)
def _projected(nu, pitch, nuisance, look):
    t = nu / 1e9
    i = t * (3 + look + 0.5 * look**2 + nuisance**2 + (pitch - look) ** 2)
    return {"I": i, "P": (0.4 + 0.1j) * i, "V": t * look * nuisance}


@se.kernel(variables=(_PITCH, _LOOK), capabilities=_BEAM, stokes=("I", "P", "V"))
def _simple(nu, pitch, look):
    t = nu / 1e9
    i = t * (2 + look + 0.5 * look**2 + (pitch - look) ** 2)
    return {"I": i, "P": (0.3 - 0.15j) * i, "V": t * look}


@pytest.fixture(scope="module")
def channels():
    return se.Channels.at([0.5, 1.0, 2.0], unit="GHz")


def test_joint_legendre_projection_preserves_order_and_all_stokes(channels):
    legendre = {"pitch": (1, 12), "nuisance": (2, 12), "look": (3, 12)}
    averages = AveragedKernel(_projected, legendre=legendre, diagonal=("pitch", "look"))
    got = averages(channels, {})
    nodes, weights = np.polynomial.legendre.leggauss(16)
    values = jax.vmap(
        lambda z: jax.vmap(
            lambda q: _projected.channel(
                channels, {"pitch": z, "look": z, "nuisance": q}, phase=None
            )
        )(nodes)
    )(nodes)
    matrices = tuple(
        np.polynomial.legendre.legvander(nodes, d) * (2 * np.arange(d + 1) + 1) / 2
        for d in (1, 2, 3)
    )
    a, b, c = matrices

    def check(spin, actual):
        expected = 2 * np.einsum(
            "xy...,x,y,xa,yb,xc->...abc", spin, weights, weights, a, b, c
        )
        np.testing.assert_allclose(actual, expected, atol=5e-14, rtol=0)

    check(values.I, got.I)
    check(values.V, got.V)
    check(values.P, got.P)
    wrapped = with_tabulated_rules(averages, {})
    np.testing.assert_array_equal(wrapped(channels, {}).I, got.I)


def test_public_two_legendre_axes_and_direct_reference(channels):
    pop = se.Population(pitch=se.Isotropic(), look=se.Isotropic())
    lat = se.Lattice(pitch=se.Legendre(2), look=se.Legendre(2))
    basis = se.spectral_basis(pop, channels, kernel=_simple, lattice=lat, check=False)
    direct = basis.direct(pop)
    want = (2 + 1 / 6) * channels.centres_hz / 1e9
    np.testing.assert_allclose(direct.I, want, atol=5e-14, rtol=0)
    np.testing.assert_allclose(
        basis.predict(pop, check=False).stokes, direct.stokes, atol=5e-14, rtol=0
    )
    assert all(route.provider == "jet" for route in basis.record.routes)
    assert basis.key is None


@pytest.mark.parametrize("cell_axis", ["pitch", "look"])
def test_cells_and_legendre_use_the_celled_cosine_in_taylor_columns(
    channels, cell_axis
):
    other = "look" if cell_axis == "pitch" else "pitch"
    lat = se.Lattice(
        **{cell_axis: se.Cells.linear(-1, 1, 2, order=2), other: se.Legendre(3)}
    )
    pop = se.Population(pitch=se.Isotropic(), look=se.Isotropic())
    basis = se.spectral_basis(pop, channels, kernel=_simple, lattice=lat, check=False)
    assert basis.averaged.diagonal == ("pitch", "look")
    value = basis.predict(pop, check=False)
    direct = basis.direct(pop, within="cells")
    np.testing.assert_allclose(value.stokes, direct.stokes, atol=5e-14, rtol=0)
    np.testing.assert_allclose(
        direct.I, (2 + 1 / 6) * channels.centres_hz / 1e9, atol=5e-14, rtol=0
    )


def test_direct_reference_with_one_way_conditional_uses_its_diagonal_density(channels):
    # The pitch density is 1 / (1 + look/2) on its range. Restrict look
    # to [-0.4,0.4] so every diagonal point is inside that range.
    def pitch(look):
        return se.Uniform(-0.5, 0.5 + 0.5 * look)

    pop = se.Population(look=se.Uniform(-0.4, 0.4), pitch=se.Conditional(pitch))
    lat = se.Lattice(pitch=se.Legendre(3), look=se.Legendre(3))
    basis = se.spectral_basis(pop, channels, kernel=_simple, lattice=lat, check=False)
    x, w = np.polynomial.legendre.leggauss(48)
    q = x * 0.4
    expected = np.dot(w, (2 + q + 0.5 * q**2) / (1 + 0.5 * q))
    direct = basis.direct(pop)
    np.testing.assert_allclose(
        direct.I, expected * channels.centres_hz / 1e9, rtol=2e-14
    )


def test_mixed_direct_reference_splits_cells_at_folded_density_jumps(channels):
    pop = se.Population(pitch=se.Isotropic(), look=se.Uniform(-0.4, 0.4))
    lat = se.Lattice(pitch=se.Cells.linear(-1, 1, 2, order=2), look=se.Legendre(3))
    basis = se.spectral_basis(pop, channels, kernel=_simple, lattice=lat, check=False)
    direct = basis.direct(pop, within="cells")
    expected = (2 + 0.4**2 / 6) * channels.centres_hz / 1e9
    np.testing.assert_allclose(direct.I, expected, atol=5e-14, rtol=0)


def test_doubly_celled_and_joint_point_routes_refuse(channels):
    pop = se.Population(pitch=se.Isotropic(), look=se.Isotropic())
    axis = se.Cells.linear(-1, 1, 2, order=1)
    with pytest.raises(se.RoleError, match="Beaming"):
        se.spectral_basis(
            pop,
            channels,
            kernel=_simple,
            lattice=se.Lattice(pitch=axis, look=axis),
            check=False,
        )
    points = se.Population(se.Samples(pitch=[-0.3, 0.3], look=[-0.2, 0.6]))
    lat = se.Lattice(pitch=se.Legendre(2), look=se.Legendre(2))
    with pytest.raises(se.RoleError, match="joint point samples"):
        se.spectral_basis(points, channels, kernel=_simple, lattice=lat, check=False)


def test_celled_pitch_axis_needs_all_cosines_with_a_legendre_partner(channels):
    pop = se.Population(pitch=se.Isotropic(), look=se.Isotropic())
    lat = se.Lattice(pitch=se.Cells.linear(-0.8, 1, 2, order=1), look=se.Legendre(3))
    with pytest.raises(se.RoleError, match=r"cover \[-1, 1\]"):
        se.spectral_basis(pop, channels, kernel=_simple, lattice=lat, check=False)
