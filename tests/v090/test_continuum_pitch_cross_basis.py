"""The physical and critical-frequency bases integrate the same joint pitch law."""

import numpy as np
import pytest
from scipy.integrate import quad_vec
from scipy.special import kv

import syncem as se
from syncem import model
from syncem.constants import C_CGS, E_ESU, M_E

GAMMA = 3000.0
FIELD = 5e-6  # G
SPIN = 0.3 + 0.4j
CRITICAL = 3 * E_ESU * FIELD * GAMMA**2 / (4 * np.pi * M_E * C_CGS)


@se.family(variable="mu", support=(-1, 1))
def pitch_density(mu, slope):
    return 1 + slope * mu


def population(correlation):
    # p(mu|eta) = (1 + correlation*eta*mu)/2, with both marginals uniform.
    def pitch(eta):
        return pitch_density(slope=correlation * eta.value, nodes=4)

    return se.Population(
        gamma=se.Delta(GAMMA),
        B=(FIELD, "G"),
        eta=se.Isotropic(),
        mu=se.Conditional(pitch) if correlation else se.Isotropic(),
        phi=se.AzimuthFactor(SPIN),
        depth=se.NoRotation(),
    )


@pytest.fixture(scope="module")
def channels():
    return se.Channels.at([0.1, 0.3], unit="GHz", half_width=0.2, n_nu=16)


@pytest.fixture(scope="module")
def critical_basis(channels):
    phi = se.AzimuthHarmonic()
    return se.RegimeBasis.build(
        population(0),
        channels,
        harmonic=se.Lattice(
            gamma=se.Cells.linear(299, 300, 1, order=0),
            mu=se.Legendre(1),
            eta=se.Legendre(1),
            phi=phi,
        ),
        continuum=se.Lattice(
            nu_c=se.Cells.log(0.002 * CRITICAL, 1.002 * CRITICAL, 48, order=4), phi=phi
        ),
        angular_nodes=2,
        field_nodes=2,
        inclination_nodes=2,
        frequency_nodes=12,
        moment_nodes=64,
        check=False,
    )


@pytest.fixture(scope="module")
def physical_basis(channels):
    return model.build_basis(
        model.ContinuumKernel(pitch="joint", n_eta=128),
        channels,
        model.Truncation(1, 1, 0),
        model.Reference(GAMMA, FIELD),
        support=model.Support(
            (GAMMA - 1, GAMMA + 1), (0.99 * FIELD, 1.01 * FIELD), (0, 0)
        ),
        convergence=False,
    )


@pytest.fixture(scope="module")
def integrated_terms(channels):
    # Even angular integrand. Integrate eta from zero to one using eta=cos(theta).
    x, w = np.polynomial.legendre.leggauss(80)
    theta = (np.pi / 4) * (1 + x)
    sine = np.sin(theta)
    weights = (np.pi / 4) * w * sine
    x_nu = np.asarray(channels.nodes) / (CRITICAL * sine[:, None, None])
    # F(x) = x int_x^infty K_{5/3}(z) dz, G(x) = x K_{2/3}(x).
    f = x_nu * quad_vec(lambda t: kv(5 / 3, t + x_nu), 0, np.inf, epsrel=1e-11)[0]
    g = x_nu * kv(2 / 3, x_nu)
    amplitude = np.sqrt(3) * E_ESU**3 * FIELD / (4 * np.pi * M_E * C_CGS**2)
    measure = amplitude * sine[:, None, None] * np.asarray(channels.response_weights())
    i = np.sum(measure * f, axis=-1)
    q = -np.sum(measure * g, axis=-1)
    stokes = np.stack((i, SPIN.real * q, SPIN.imag * q), axis=-1)
    # 2 p(mu=eta,eta) = (1 + correlation*eta**2)/2.
    return np.einsum("n,ncs->cs", weights, stokes), np.einsum(
        "n,ncs->cs", weights * np.cos(theta) ** 2, stokes
    )


@pytest.mark.parametrize("correlation", [-1.0, -0.48, 0.0, 0.48, 1.0])
def test_continuum_pitch_cross_basis(
    correlation, physical_basis, critical_basis, integrated_terms
):
    expected = integrated_terms[0] + correlation * integrated_terms[1]
    # At angular degree one only M00 and M11 survive parity. Integrating
    # mu*eta against (1 + correlation*mu*eta)/4 gives M11 = correlation/9.
    moments = {(0, 0): 1.0, (1, 1): correlation / 9}
    idx = physical_basis.index
    joint = model.JointMoments(
        index=idx,
        m0=np.array([moments[row[:2]] for row in idx.h0]),
        m2=SPIN * np.array([moments[row[:2]] for row in idx.h2]),
        reference=physical_basis.reference,
    )
    physical = np.asarray(model.predict(physical_basis, joint, amplitude=1).stokes)
    critical = np.asarray(
        critical_basis.predict(population(correlation), check=False).stokes
    )
    np.testing.assert_array_equal(physical[:, 3], 0)
    np.testing.assert_allclose(physical[:, :3], expected, rtol=2e-7, atol=0)
    np.testing.assert_allclose(critical, expected, rtol=2e-7, atol=0)
    if correlation == -0.48:
        # Independent SciPy integration in eta with 160 Gauss-Legendre nodes.
        np.testing.assert_allclose(
            expected[:, 0],
            [5.241998901616238e-29, 2.243373616478023e-29],
            rtol=5e-11,
            atol=0,
        )
        # Independent anisotropic laws yield the same even joint moments.
        # Their diagonal is (1 + 0.2*eta - 0.48*eta**2)/2. The odd term
        # integrates to zero because the continuum I and Q are even in eta.
        product = population(0).replace(
            mu=pitch_density(slope=0.8, nodes=4),
            eta=se.Tabulated([-1, 1], [1.6, 0.4]),
        )
        np.testing.assert_allclose(
            critical_basis.predict(product, check=False).stokes,
            physical[:, :3],
            rtol=2e-7,
            atol=0,
        )


def test_pitch_correlated_with_azimuth(
    physical_basis, critical_basis, integrated_terms
):
    # Two equal groups at sky angles separated by pi/2. Their spin factors
    # cancel, but opposite pitch slopes couple to the sign of each spin.
    slope = 0.8
    spin = 2 * SPIN
    idx = physical_basis.index
    joint = model.JointMoments(
        index=idx,
        m0=np.array([float(row[:2] == (0, 0)) for row in idx.h0]),
        m2=np.array([spin * slope / 9 if row[:2] == (1, 1) else 0j for row in idx.h2]),
        reference=physical_basis.reference,
    )
    physical = np.asarray(model.predict(physical_basis, joint, amplitude=1).stokes)
    phi0 = np.angle(spin) / 2

    def pitch(eta, phi):
        sign = np.cos(2 * (phi.value - phi0))
        return pitch_density(slope=slope * eta.value * sign, nodes=4)

    source = se.Population(
        se.Samples(phi=([phi0, phi0 + np.pi / 2], "rad"), weights=[0.5, 0.5], column=1),
        gamma=se.Delta(GAMMA),
        B=(FIELD, "G"),
        eta=se.Isotropic(),
        mu=se.Conditional(pitch, units={"phi": "rad"}),
        depth=se.NoRotation(),
    )
    critical = critical_basis.predict(source, check=False).stokes
    expected = integrated_terms[0].copy()
    expected[:, 1:] = 2 * slope * integrated_terms[1][:, 1:]
    np.testing.assert_allclose(physical[:, :3], expected, rtol=2e-7, atol=0)
    np.testing.assert_array_equal(physical[:, 3], 0)
    np.testing.assert_allclose(critical, expected, rtol=2e-7, atol=0)
