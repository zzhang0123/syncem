"""Check absolute column normalization before and after conditional averaging."""

import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se


@se.kernel(
    variables=(se.Variable("gamma", kind="positive"), se.Variable("y")),
    stokes=("I", "P", "V"),
)
def constant(nu, gamma, y):
    return {"I": 1 + 0 * nu, "P": -0.2 + 0.4j, "V": 0.3}


@pytest.mark.parametrize("column", [0.0, 1.0, 7.0, 19.0])
def test_averaged_primary_preserves_absolute_stokes_and_ledger(column):
    channels = se.Channels.at([1.0], unit="GHz")
    lattice = se.Lattice(y=se.Cells.linear(-0.5, 0.5, 2, order=1))
    base = se.Population(
        gamma=se.PowerLaw(2.2, lo=300, hi=2000, column=1), y=se.Uniform(-1, 1)
    )
    population = base.replace(gamma=se.PowerLaw(2.2, lo=300, hi=2000, column=column))
    expected = column * np.array([[1.0, -0.2, 0.4, 0.3]])
    for part in (population, column * base):
        basis = se.spectral_basis(
            part,
            channels,
            lattice=lattice,
            kernel=constant,
            stokes="IQUV",
            check=False,
            cache=False,
        )
        pred = basis.predict(part, check=True)
        np.testing.assert_allclose(basis.direct(part).stokes, expected, atol=1e-13)
        np.testing.assert_allclose(
            basis.direct(part, within="cells").stokes, expected / 2, atol=1e-13
        )
        np.testing.assert_allclose(pred.stokes, expected / 2, atol=1e-13)
        np.testing.assert_allclose(pred.ledger["representation"].value, 0, atol=1e-13)
        np.testing.assert_allclose(
            pred.ledger["outside_lattice"].value, -expected / 2, atol=1e-13
        )


@se.kernel(
    variables=(
        se.Variable("gamma", kind="positive"),
        se.Variable("z"),
        se.Variable("y"),
    ),
    stokes=("I",),
)
def tracer(nu, gamma, z, y):
    return {"I": (1 + 0.5 * z) * (1 + y) + 0 * nu}


def column_given_parent(z):
    # At each z, integrate dN/dgamma = 1 + z/2 on [2, 4].
    return se.PowerLaw(0, lo=2, hi=4, amplitude=1 + z.value / 2)


def test_averaged_parent_is_weighted_by_its_electron_column():
    pop = se.Population(
        gamma=se.Conditional(column_given_parent),
        z=se.Uniform(-1, 1, nodes=4),
        y=se.Delta(0.3),
    )
    lattice = se.Lattice(y=se.Cells.linear(-1, 1, 1, order=1), weights=[1])
    basis = se.spectral_basis(
        pop, ([1.0], "GHz"), kernel=tracer, lattice=lattice, check=False
    )
    # E[N(z)] = 2, E[N(z) K(z,y)] = 2 (1 + y) (1 + E[z^2]/4).
    np.testing.assert_allclose(basis.moments(pop).values, [2, 0.6], rtol=1e-13)
    for value in (basis.predict(pop, check=False).I, basis.direct(pop).I):
        np.testing.assert_allclose(value, 2 * 1.3 * (1 + 1 / 12), rtol=1e-13)


def pitch_given_depth(depth):
    coefficient = 0.4 * depth.to("rad m^-2").value
    return se.Tabulated(
        jnp.array([-1.0, 1.0]), jnp.array([1 - coefficient, 1 + coefficient])
    )


def test_intensity_keeps_depth_columns_when_the_diagonal_density_uses_depth():
    channels = se.Channels.at([1.0], unit="GHz")
    lattice = se.Lattice(depth=se.Cells.linear(0.0, 2.0, 2, order=1))
    uniform = se.Population(
        gamma=se.Delta(1e4), B="5 uG", mu=se.Isotropic(), eta=0.5, depth=se.Delta(0.3)
    )
    iso = se.spectral_basis(uniform, channels, lattice=lattice, stokes="I", check=False)
    scale = np.asarray(iso.direct(uniform).I)
    np.testing.assert_equal(np.unique(iso.columns.spin0_from).size, 1)
    pop = uniform.replace(mu=se.Conditional(pitch_given_depth))
    basis = se.spectral_basis(pop, channels, lattice=lattice, stokes="I", check=False)
    for depth in (0.0, 0.2, 0.7, 1.3, 1.8, 2.0):
        local = pop.replace(depth=se.Delta(depth))
        for value in (basis.predict(local, check=False).I, basis.direct(local).I):
            np.testing.assert_allclose(value / scale, 1 + 0.2 * depth, rtol=1e-12)
