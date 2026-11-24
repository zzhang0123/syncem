"""Full-turn Fourier columns with shared population moments, SPEC R4a."""

import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se
from syncem.engine.provider_fourier import sample_count

_CHANNELS = ([0.8, 2.0], "GHz")


@se.kernel(
    variables=(
        se.Variable("x", kind="positive"),
        se.Variable("B", kind="positive", unit="G", bare=False),
        se.Variable("phi", kind="angle", unit="rad", bare=False),
    ),
    stokes=("I", "P", "V"),
    azimuth="phi",
)
def _kernel(nu, x, B):
    b = B * 1e6
    one = jnp.ones_like(nu)
    return {
        "I": (x**2 + b) * one,
        "P": (2 + (0.3 - 0.2j) * b) * one,
        "V": (x - b * 0.1) * one,
    }


def _field(x, phi):
    a = phi.value
    centre = (
        5.0 + 0.2 * x * jnp.cos(a) + 0.4 * jnp.sin(2 * a) + 0.7 * jnp.cos(4 * a + 0.3)
    )
    return se.Uniform(centre - 0.3, centre + 0.3, unit="uG", nodes=2)


def _population():
    return se.Population(
        x=se.Uniform(1.0, 2.0),
        B=_field,
        phi=se.AzimuthDensity(0.8, 0.51, unit="rad", nodes=24),
    )


def _lattice(degree, order=2):
    return se.Lattice(
        x=se.Cells.linear(1.0, 2.0, 2, order=order),
        phi=se.AzimuthFourier(degree),
        weights=[0.25, 0.75],
    )


def test_shared_fourier_rows_and_default_degree():
    pop = _population()
    index = _lattice(2).index
    pos = index.position(block="Re M3", x=0, powers={"x": 1})
    assert index.blocks == (
        "M0",
        "Re M1",
        "Im M1",
        "Re M2",
        "Im M2",
        "Re M3",
        "Im M3",
        "Re M4",
        "Im M4",
    )
    assert index.row(pos).pair == pos + index.block_size
    assert index.row(pos + index.block_size).pair == pos
    planned = se.plan(pop, _CHANNELS, kernel=_kernel, resolve=("phi",))
    assert planned.lattice.representations["phi"].m_max == 4
    assert "AzimuthFourier(m_max=4)" in str(planned)
    assert planned.to_dict()["azimuth_fourier_order"] == 4
    with pytest.raises(se.RoleError, match="phi.*AzimuthFourier"):
        wrong = _lattice(2).replace(phi=se.AzimuthHarmonic())
        se.spectral_basis(pop, _CHANNELS, lattice=wrong, kernel=_kernel, cache=False)
    with pytest.raises(ValueError, match="non-negative"):
        se.AzimuthFourier(-1)
    assert [sample_count(m) for m in (0, 15, 16, 30)] == [32, 32, 33, 61]


def test_resolve_is_forwarded_and_fourier_archive_keeps_design(tmp_path):
    pop = _population()
    result = se.predict(pop, _CHANNELS, kernel=_kernel, resolve=("phi",), check=False)
    assert "Re M6" in result.moments.index.blocks
    basis = se.spectral_basis(
        pop, _CHANNELS, kernel=_kernel, lattice=_lattice(2), check=False
    )
    path = tmp_path / "fourier.npz"
    basis.save(path)
    restored = se.SpectralBasis.load(path)
    np.testing.assert_array_equal(restored.design(), basis.design())
    np.testing.assert_array_equal(
        restored.at(restored.channels[[1]]).design(), basis.design()[1:]
    )
    moments = basis.moments(pop)
    np.testing.assert_array_equal(
        restored.sampling_error(moments), basis.sampling_error(moments)
    )


@pytest.mark.parametrize("degree", [0, 2, 4])
def test_fourier_b_given_phi_against_direct_and_known_signed_mode_result(degree):
    pop = _population()
    basis = se.spectral_basis(
        pop,
        _CHANNELS,
        lattice=_lattice(degree),
        kernel=_kernel,
        check=False,
        cache=False,
    )
    pred = basis.predict(pop)
    direct = basis.direct(pop)
    # Independently pair the known polynomial Fourier content with the azimuth law.
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    density = 1 + 0.8 * np.cos(2 * (angles - 0.51))
    bbar = np.full(angles.shape, 5.0)
    if degree >= 2:
        bbar += 0.3 * np.cos(angles) + 0.4 * np.sin(2 * angles)
    if degree >= 4:
        bbar += 0.7 * np.cos(4 * angles + 0.3)
    target_i = np.mean(density * (7 / 3 + bbar))
    target_p = np.mean(density * np.exp(2j * angles) * (2 + (0.3 - 0.2j) * bbar))
    target_v = np.mean(density * (1.5 - 0.1 * bbar))
    target = np.tile([target_i, target_p.real, target_p.imag, target_v], (2, 1))
    np.testing.assert_allclose(pred.stokes, target, atol=8e-14, rtol=2e-14)
    if degree == 4:
        np.testing.assert_allclose(pred.stokes, direct.stokes, atol=8e-14, rtol=2e-14)
    elif degree == 2:
        assert np.max(np.abs(np.asarray(pred.P - direct.P))) > 0.04
    else:
        assert np.max(np.abs(np.asarray(pred.I - direct.I))) > 0.1
    assert pred.ledger["quadrature:azimuth"].kind == "estimate"
    np.testing.assert_allclose(pred.ledger["quadrature:azimuth"].value, 0.0, atol=8e-14)
    truncation = pred.ledger["azimuth_truncation"]
    assert truncation.kind == "measured"
    np.testing.assert_allclose(
        truncation.value, pred.stokes - direct.stokes, atol=8e-14
    )
    np.testing.assert_allclose(pred.ledger["representation"].value, 0.0, atol=8e-14)
    assert basis.record.routes[0].mode == "azimuth_fourier"
    assert basis.record.events and dict(basis.record.events)["jet compiles"] <= 2
    if degree == 4:
        lower = basis.with_order(0)
        assert lower.lattice.order == 0
        assert lower.design().shape[-1] < basis.design().shape[-1]
        new = basis.at(se.Channels.at([1.0], unit="GHz"))
        np.testing.assert_allclose(
            new.predict(pop, check=False).stokes, target[:1], atol=8e-14
        )
        swapped = basis.at(basis.channels[[1, 0]])
        np.testing.assert_array_equal(
            swapped.design(), np.asarray(basis.design())[[1, 0]]
        )


def test_phi_child_and_resolved_joint_legendre_axis():
    @se.kernel(
        variables=(
            se.Variable("x", kind="positive"),
            se.Variable("u", kind="cosine"),
            se.Variable("eta", kind="cosine"),
            se.Variable("phi", kind="angle", unit="rad"),
        ),
        stokes=("P",),
        azimuth="phi",
    )
    def model(nu, x, u, eta):
        return {"P": x * (1 + u**2) * (2 - 0.5j + eta) * jnp.ones_like(nu)}

    def moving(azimuth):
        angle = azimuth.value
        centre = 0.5 * jnp.cos(2 * angle) + 0.2 * jnp.sin(2 * angle)
        return se.Uniform(centre - 0.1, centre + 0.1, nodes=2)

    def eta(phi):
        return moving(phi)

    samples = se.Samples(
        x=[1.2, 1.7],
        u=[-0.7, 0.1],
        phi=([0.38, 2.63], "rad"),
        weights=[1, 3],
        column=8,
    )
    pop = se.Population(samples, eta=eta)
    lattice = se.Lattice(
        x=se.Cells.linear(1.0, 2.0, 2, order=1),
        u=se.Legendre(2),
        phi=se.AzimuthFourier(2),
    )
    basis = se.spectral_basis(
        pop, _CHANNELS, lattice=lattice, kernel=model, cache=False
    )
    result = basis.predict(pop)
    target = basis.direct(pop)
    np.testing.assert_allclose(result.P, target.P, atol=8e-14)
    np.testing.assert_allclose(
        result.ledger["azimuth_truncation"].value, 0.0, atol=8e-14
    )
    assert basis.spectra0.shape[1:3] == (0, 0)


def test_doubled_grid_detects_aliasing():
    def field(x, phi):
        del x
        return ((5 + jnp.cos(32 * phi.value)) * 1e-6, "G")

    pop = _population().replace(B=field)
    basis = se.spectral_basis(
        pop, _CHANNELS, lattice=_lattice(0), kernel=_kernel, check=False, cache=False
    )
    result = basis.predict(pop, check=False)
    np.testing.assert_allclose(result.ledger["quadrature:azimuth"].value[:, 0], 1.0)
    assert result.ledger["azimuth_truncation"].kind == "unbounded"


@pytest.mark.parametrize("degree", [15, 16])
def test_projection_grid_dispatch_boundary(degree):
    def field(x, phi):
        del x
        return ((5 + jnp.cos(16 * phi.value)) * 1e-6, "G")

    angle = 0.15
    pop = _population().replace(
        x=se.Delta(1.2), B=field, phi=se.Delta(angle, unit="rad")
    )
    basis = se.spectral_basis(
        pop,
        _CHANNELS,
        lattice=_lattice(degree),
        kernel=_kernel,
        check=False,
        cache=False,
    )
    result = basis.predict(pop, check=False)
    expected_i = 1.2**2 + 5 + (np.cos(16 * angle) if degree == 16 else 0)
    np.testing.assert_allclose(result.I, expected_i, atol=2e-13)
    np.testing.assert_allclose(
        result.ledger["quadrature:azimuth"].value, 0.0, atol=2e-13
    )


def test_free_phi_block_with_an_averaged_child_outside_the_block():
    lattice = _lattice(4)
    truth_basis = se.spectral_basis(
        _population(),
        _CHANNELS,
        lattice=lattice,
        kernel=_kernel,
        check=False,
        cache=False,
    )
    truth = truth_basis.direct(_population())
    data = se.StokesData(
        truth.channels, stokes=truth.stokes, unit=truth.unit, sigma=np.ones((2, 4))
    )
    unknown = se.Population(se.FreeMoments("x", "phi"), B=_field)
    fit = se.fit(
        unknown,
        data,
        lattice=lattice,
        kernel=_kernel,
        rank=se.FullRank(),
        check=False,
        cache=False,
    )
    np.testing.assert_allclose(fit.predict().stokes, truth.stokes, atol=4e-12)
    assert fit.problem.basis.lattice.representations["phi"].m_max == 4


def test_signed_fourier_partition_with_spatial_and_outside_residuals():
    population = _population()
    lattice = _lattice(2, order=0).select(x=0)
    basis = se.spectral_basis(
        population,
        _CHANNELS,
        lattice=lattice,
        kernel=_kernel,
        check=False,
        cache=False,
    )
    predicted = basis.predict(population)
    direct = basis.direct(population)
    signed = [
        predicted.ledger[name].value
        for name in (
            "representation",
            "azimuth_truncation",
            "outside_lattice",
        )
    ]
    assert all(np.max(np.abs(v)) > 1e-3 for v in signed)
    np.testing.assert_allclose(
        sum(signed), predicted.stokes - direct.stokes, atol=5e-13
    )
