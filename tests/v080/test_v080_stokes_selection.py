"""Explicit Stokes selection on supplied bases (review P2-4)."""

import numpy as np

import syncem as se


def test_prebuilt_basis_selects_and_orders_components():
    channels = se.Channels.log(0.5, 2.0, 3, unit="GHz", half_width=0.2, n_nu=8)
    population = se.paper.e1()
    lattice = se.Lattice(gamma=se.paper.cells(order=1))
    basis = se.spectral_basis(population, channels, lattice=lattice, check=False)
    full = se.predict(population, channels, basis=basis, check=False)
    only = se.predict(population, channels, basis=basis, stokes="I", check=False)
    assert only.components == ("I",)
    np.testing.assert_array_equal(only.I, full.I)
    order = se.predict(population, channels, basis=basis, stokes="UQI", check=False)
    assert order.components == ("U", "Q", "I")
    np.testing.assert_array_equal(order.stokes, full.stokes[:, [2, 1, 0]])


def test_harmonic_selection_keeps_the_representation_term_in_order():
    samples = se.Samples(
        gamma=[19.5, 20.5],
        B=([0.98, 1.02], "G"),
        mu=[-0.05, 0.05],
        eta=[-0.05, 0.05],
        phi=([0.05, 0.1], "rad"),
        depth=[1e-8, 2e-8],
        column="1 cm^-2",
    )
    nu = 139962.44936166518
    channels = se.Channels.at([2 * nu], unit="Hz", half_width=0.65, n_nu=8)
    lattice = se.Lattice.single(about=samples, order=0, degree=1)
    kernel = se.HarmonicKernel(m_max=10)
    basis = se.spectral_basis(samples, channels, lattice=lattice, kernel=kernel)
    full = se.predict(samples, channels, basis=basis)
    reference = np.asarray(full.ledger["representation"].value)
    for stokes, keep in (("I", [0]), ("VUQI", [3, 2, 1, 0])):
        got = se.predict(samples, channels, basis=basis, stokes=stokes)
        assert got.components == tuple(stokes)
        np.testing.assert_array_equal(got.stokes, full.stokes[:, keep])
        np.testing.assert_array_equal(
            got.ledger["representation"].value, reference[:, keep]
        )


def test_selection_of_broadcast_errors_and_noise():
    from syncem.records.ledger import ErrorTerm, Ledger
    from syncem.values.spectrum import Spectrum

    channels = se.Channels.at([1, 4], unit="GHz")
    ledger = Ledger(
        (
            ErrorTerm([[0.1, 0.2, 0.3]], "bound", name="band"),
            ErrorTerm([0.4, 0.5, 0.6], "bound", name="shared"),
            ErrorTerm([[0.0, 0.1, 0.2, 0.3]], "bound", name="canonical"),
        )
    )
    full = Spectrum(
        np.arange(12).reshape(2, 2, 3),
        channels,
        ledger=ledger,
        sigma=1.0,
        unconstrained=[[0.2], [0.3]],
        components="IQU",
    )
    for names, ids in (("Q", [1]), ("UQI", [2, 1, 0])):
        picked = full.select(names)
        np.testing.assert_array_equal(picked.stokes, np.asarray(full.stokes)[..., ids])
        expected = np.array([0.5, 0.8, 1.1])[ids]
        np.testing.assert_allclose(
            np.broadcast_to(picked.ledger.envelope().value, picked.stokes.shape),
            np.broadcast_to(expected, picked.stokes.shape),
        )
        for attr in ("sigma", "unconstrained"):
            expected = np.broadcast_to(getattr(full, attr), full.stokes.shape)[..., ids]
            np.testing.assert_array_equal(
                np.broadcast_to(getattr(picked, attr), picked.stokes.shape), expected
            )


def test_predict_builds_polarization_columns_for_a_single_output_component():
    channels = se.Channels.at([1, 2], unit="GHz")
    population = se.paper.e1()
    lattice = se.Lattice(gamma=se.Cells.log(300, 3e4, 2, order=0))
    full = se.predict(population, channels, lattice=lattice, check=False)
    picked = se.predict(population, channels, lattice=lattice, stokes="U", check=False)
    assert picked.components == ("U",)
    np.testing.assert_array_equal(picked.U, full.U)
