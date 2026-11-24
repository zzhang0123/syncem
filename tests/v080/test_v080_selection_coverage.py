"""Coverage estimate after cell selection, in either order of operations (review P2-3)."""

import numpy as np

import syncem as se


def test_selection_recomputes_the_coverage_estimate():
    population = se.paper.e1()
    lattice = se.Lattice(gamma=se.Cells.log(300, 3e6, 6, order=1))
    one = se.Channels.at([1.0], unit="GHz", half_width=0.2, n_nu=8)
    two = se.Channels.at([2.0], unit="GHz", half_width=0.2, n_nu=8)
    basis = se.spectral_basis(population, one, lattice=lattice, check=False)
    first_at = basis.at(two).select(gamma=[0])
    first_select = basis.select(gamma=[0]).at(two)
    assert first_at.lattice.key == first_select.lattice.key
    a = np.asarray(first_at.ledger["outside_lattice_reference"].value)
    b = np.asarray(first_select.ledger["outside_lattice_reference"].value)
    np.testing.assert_allclose(a, b, rtol=1e-12)
    # One of six cells covers little of the reference power-law emission at 2 GHz.
    assert a.ravel()[0] > 0.5
