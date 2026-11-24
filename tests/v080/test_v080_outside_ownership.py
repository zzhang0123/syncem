"""Ownership of complement faces in the outside-lattice term (review P2-2)."""

import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se

_X = se.Variable("x", kind="real")
_Y = se.Variable("y", kind="real")


@se.kernel(variables=(_X, _Y), stokes=("I",))
def _flat(nu, x, y):
    return {"I": jnp.ones_like(nu) + 0.0 * (x + y)}


@pytest.mark.parametrize(
    "atom",
    [
        (0.0, -1.0),  # on the face shared by two complement boxes
        (-1.0, -1.0),  # in a corner region
        (1.0, 2.0),  # on the upper face of another complement pair
        (-1.0, 0.5),  # beside a cell face
        (1.5, 0.5),  # in the gap beside the second cell
        (0.5, 0.5),  # inside the lattice
    ],
)
def test_full_support_is_cells_plus_outside(atom):
    channels = se.Channels.at([1.0], unit="GHz", half_width=0.1, n_nu=2)
    population = se.Population(
        se.Samples(x=[atom[0]], y=[atom[1]], column="1 cm^-2")
    )
    lattice = se.Lattice(x=se.Cells.linear(0, 1, 1, order=0), y=se.Cells.linear(0, 1, 1, order=0))
    basis = se.spectral_basis(population, channels, lattice=lattice, kernel=_flat, check=False)
    full = float(basis.direct(population).I[0])
    cells = float(basis.direct(population, within="cells").I[0])
    got = se.predict(population, channels, basis=basis)
    inside = 1.0 if (0 <= atom[0] <= 1 and 0 <= atom[1] <= 1) else 0.0
    unit = float(_flat.channel(channels, {"x": 0.0, "y": 0.0}, phase=None).I[0])
    np.testing.assert_allclose([full, cells], [unit, inside * unit], rtol=1e-12)
    # outside_lattice is the in-cell minus the full-support emission (SPEC Sec. 7.2).
    np.testing.assert_allclose(
        np.ravel(got.ledger["outside_lattice"].value), [cells - full], atol=1e-12
    )
