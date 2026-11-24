"""Explicit kernel and phase declarations against a supplied basis (review P1-3)."""

import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se
from syncem.distributions.screens import GaussianScreen


@pytest.fixture(scope="module")
def setup():
    channels = se.Channels.log(0.5, 2.0, 3, unit="GHz", half_width=0.2, n_nu=8)
    population = se.paper.e1()
    lattice = se.Lattice(gamma=se.paper.cells(order=1))
    basis = se.spectral_basis(population, channels, lattice=lattice, check=False)
    return channels, population, lattice, basis


def test_explicit_phase_differing_from_the_basis_is_refused(setup):
    channels, population, lattice, basis = setup
    screen = se.FaradayRotation(screen=GaussianScreen(8.0, 0.5))
    rebuilt = se.spectral_basis(
        population, channels, lattice=lattice, phase=screen, check=False
    )
    held = se.predict(population, channels, basis=basis, check=False)
    wanted = se.predict(population, channels, basis=rebuilt, check=False)
    # The value the stored basis returned before the check differs from the rebuild.
    assert not np.allclose(held.stokes[:, 1:], wanted.stokes[:, 1:], rtol=1e-3, atol=0)
    with pytest.raises(se.RoleError, match="phase= differs"):
        se.predict(population, channels, basis=basis, phase=screen, check=False)
    same = se.predict(
        population, channels, basis=rebuilt, phase=screen, check=False
    )
    np.testing.assert_array_equal(same.stokes, wanted.stokes)
    default = se.predict(
        population, channels, basis=basis, phase=se.FaradayRotation(), check=False
    )
    np.testing.assert_array_equal(default.stokes, held.stokes)


def test_loaded_basis_refuses_an_unverifiable_phase(setup, tmp_path):
    channels, population, _, basis = setup
    path = tmp_path / "basis.npz"
    basis.save(path)
    loaded = se.SpectralBasis.load(path)
    moments = basis.moments(population)
    with pytest.raises(se.RoleError, match="does not record the phase"):
        se.predict(moments, channels, basis=loaded, phase=se.FaradayRotation())
    se.predict(moments, channels, basis=loaded, check=False)


@se.kernel(variables=[se.Variable("c", kind="positive")], stokes=("I",))
def _one(nu, c):
    return {"I": c * jnp.ones_like(nu)}


@se.kernel(variables=[se.Variable("c", kind="positive")], stokes=("I",))
def _ten(nu, c):
    return {"I": 10.0 * c * jnp.ones_like(nu)}


def test_fits_and_predictions_refuse_a_kernel_differing_from_the_basis():
    channels = se.Channels.log(0.5, 2.0, 3, unit="GHz", half_width=0.2, n_nu=4)
    lattice = se.Lattice(c=se.Cells.linear(1, 2, 1, order=0))
    unknown = se.Population(c=se.FreeMoments())
    data = se.StokesData(channels, I=np.full(3, 10.0), sigma=1.0, unit="cgs")
    basis = se.problem(unknown, data, lattice=lattice, kernel=_one, check=False).basis
    with pytest.raises(se.RoleError, match="kernel= differs"):
        se.fit(unknown, data, basis=basis, kernel=_ten, check=False)
    with pytest.raises(se.RoleError, match="kernel= differs"):
        se.problem(unknown, data, basis=basis, kernel=_ten, check=False)
    moments = se.fit(unknown, data, basis=basis, kernel=_one, check=False).moments
    with pytest.raises(se.RoleError, match="kernel= differs"):
        se.predict(moments, channels, basis=basis, kernel=_ten, check=False)
    rebuilt = se.fit(unknown, data, lattice=lattice, kernel=_ten, check=False)
    held = se.fit(unknown, data, basis=basis, check=False)
    # The two kernels need columns differing by their factor of ten.
    np.testing.assert_allclose(
        np.asarray(held.moments.values), 10 * np.asarray(rebuilt.moments.values)
    )


def test_native_harmonic_declaration_compares_the_harmonic_count():
    samples = se.Samples(
        gamma=[19.5, 20.5],
        B=([0.98, 1.02], "G"),
        mu=[-0.05, 0.05],
        eta=[-0.05, 0.05],
        phi=([0.05, 0.1], "rad"),
        depth=[1e-8, 2e-8],
        column="1 cm^-2",
    )
    ch = se.Channels.at([279924.89872333036], unit="Hz", half_width=0.65, n_nu=8)
    basis = se.spectral_basis(
        samples,
        ch,
        lattice=se.Lattice.single(about=samples, order=0, degree=1),
        kernel=se.HarmonicKernel(m_max=10),
    )
    wanted = se.predict(samples, ch, basis=basis, check=False)
    got = se.predict(
        samples, ch, basis=basis, kernel=se.HarmonicKernel(m_max=10), check=False
    )
    np.testing.assert_array_equal(got.stokes, wanted.stokes)
    with pytest.raises(se.RoleError, match="kernel"):
        se.predict(samples, ch, basis=basis, kernel=se.HarmonicKernel(m_max=11))


def test_sum_fit_refuses_an_unverifiable_archived_phase(tmp_path):
    ch = se.Channels.at([1, 1.5], unit="GHz")
    pop = se.Population(
        gamma=se.PowerLaw(2.6, lo=300, hi=2e4, column=1),
        B="5 uG",
        phi=se.AzimuthFactor(1 + 0j),
        depth="8 rad m^-2",
    )
    basis = se.spectral_basis(
        pop,
        ch,
        lattice=se.Lattice(gamma=se.Cells.log(300, 2e4, 1)),
        phase=False,
        check=False,
    )
    free = pop.replace(gamma=se.FreeMoments())
    data = se.StokesData(
        ch, I=[1.0, 2.0], Q=[1.0, 0.0], U=[0.0, 1.0], sigma=1.0, unit="cgs"
    )
    path = tmp_path / "disabled.npz"
    basis.save(path)
    for averaged in (None, basis.averaged):
        stored = se.SpectralBasis.load(path, averaged=averaged)
        for method in (se.fit, se.problem):
            with pytest.raises(se.RoleError, match="does not record the phase"):
                method(
                    free + free,
                    data,
                    basis=stored,
                    phase=se.FaradayRotation(),
                    check=False,
                )


def test_explicit_hybrid_compares_the_assignment_on_the_basis_cells():
    ch = se.Channels.at([1], unit="GHz")
    lat = se.Lattice(c=se.Cells.linear(1, 3, 2, order=1))
    p = se.Population(c=se.Uniform(1, 3))
    model = se.HybridKernel(_one, _one, switch={"c": 2})
    basis = se.spectral_basis(
        p, ch, lattice=lat, kernel=model, check=False, cache=False
    )
    want = se.predict(p, ch, basis=basis, check=False)
    got = se.predict(p, ch, basis=basis, kernel=model, check=False)
    np.testing.assert_array_equal(got.stokes, want.stokes)
    other = se.HybridKernel(_ten, _one, switch={"c": 2})
    with pytest.raises(se.RoleError, match="kernel"):
        se.predict(p, ch, basis=basis, kernel=other, check=False)
