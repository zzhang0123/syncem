"""Physical Taylor interpolation and point masses on either side of a cell face."""

import numpy as np
import pytest

import syncem as se
from syncem.engine._regime_columns import polynomial_columns


@pytest.fixture(scope="module")
def narrow_source():
    switch = 300.0
    step = np.spacing(switch)
    population = se.Population(
        gamma=se.Uniform(switch - 2 * step, switch),
        B=(5e-6, "G"),
        eta=0.2,
        mu=se.Isotropic(),
        phi=se.AzimuthFactor(0.4),
        depth=se.NoRotation(),
    )
    regime = se.RegimeBasis.build(
        population,
        se.Channels.at([1.0, 2.0], unit="MHz", half_width=0.12, n_nu=8),
        harmonic=se.Lattice(
            gamma=se.Cells.linear(290, switch, 1, order=0),
            phi=se.AzimuthHarmonic(),
        ),
        continuum=se.Lattice(
            nu_c=se.Cells.log(1e5, 1e7, 1, order=1), phi=se.AzimuthHarmonic()
        ),
        gamma_switch=switch,
        field_nodes=2,
        inclination_nodes=2,
        angular_nodes=2,
        frequency_nodes=8,
        moment_nodes=8,
        check=False,
    )
    return regime, population


@pytest.mark.parametrize("chart", ["log", "affine"])
def test_physical_fit_preserves_both_sides_of_a_narrow_switch(narrow_source, chart):
    regime, population = narrow_source
    s = regime.gamma_switch
    u = np.spacing(s)
    cells = se.Cells.from_edges([s - 2 * u, s, s + 2 * u], chart=chart, order=1)
    basis = se.paper.energy_basis(regime, population, cells)
    np.testing.assert_allclose(
        basis.predict(population).stokes,
        regime.predict(population).stokes,
        rtol=3e-14,
        atol=0,
    )
    points, columns = basis.point_columns(se.NonNegativeCells(point_masses=3))
    expected = s + u * np.arange(-2, 3)
    np.testing.assert_array_equal(points.coordinates["gamma"], expected)
    np.testing.assert_array_equal(points.cells, [0, 0, 1, 1, 1])
    for index, gamma in enumerate(expected):
        atom = population.replace(gamma=se.Delta(gamma))
        np.testing.assert_allclose(
            columns[..., index], regime.predict(atom).stokes, rtol=2e-14, atol=0
        )
        np.testing.assert_allclose(
            points.moments[:, index], basis.moments(atom).values, rtol=2e-14, atol=0
        )


@pytest.mark.parametrize("chart", ["affine", "log"])
def test_collocation_reproduces_a_polynomial_from_representable_nodes(chart):
    # The requested Lobatto grid rounds to five physical samples on this axis.
    # Use a shifted reference and a mixed term to exercise a tensor grid.
    lo = 300.0
    step = np.spacing(lo)
    ref = lo + step
    lattice = se.Lattice(
        gamma=se.Cells.from_edges(
            [lo, lo + 4 * step], chart=chart, order=2, reference=[ref]
        ),
        B=se.Cells.linear(1, 3, 1, order=1, unit="G"),
    )
    table = lattice.cells
    gamma = table.variables.index("gamma")
    width = table.half_width[0, gamma]
    powers = np.array([[0, 0], [1, 0], [2, 0], [0, 1], [1, 1]])

    def value(point):
        dx = point["gamma"] - ref
        x = (np.log1p(dx / ref) if chart == "log" else dx) / width
        y = point["B"] - 2
        return 0.2 + x + 2 * x**2 - 3 * y + x * y

    ordered = ["gamma", "B"]
    dims = tuple(table.variables.index(name) for name in ordered)
    powers = powers[:, [ordered.index(name) for name in table.variables]]
    rows = polynomial_columns(table, 0, powers, value, 7, dimensions=dims)
    np.testing.assert_allclose(rows, [0.2, 1, 2, -3, 1], rtol=1e-13, atol=1e-14)


@pytest.mark.parametrize("chart", ["log", "affine"])
def test_single_float_harmonic_cell_retains_its_only_atom(narrow_source, chart):
    regime, source = narrow_source
    switch = regime.gamma_switch
    low = np.nextafter(switch, -np.inf)
    high = np.nextafter(switch, np.inf)
    cells = se.Cells.from_edges([low, switch, high], chart=chart, order=2)
    basis = se.paper.energy_basis(regime, source, cells)
    atom = source.replace(gamma=se.Delta(low))
    np.testing.assert_allclose(
        basis.predict(atom).stokes, regime.predict(atom).stokes, rtol=3e-14, atol=0
    )
    masses, _ = basis.point_columns(se.NonNegativeCells(point_masses=1))
    np.testing.assert_array_equal(masses.coordinates["gamma"][masses.cells == 0], [low])


@pytest.mark.parametrize("chart", ["log", "affine"])
def test_clipped_mass_support_keeps_its_endpoint_below_switch(narrow_source, chart):
    regime, source = narrow_source
    s, step = regime.gamma_switch, np.spacing(regime.gamma_switch)
    cells = se.Cells.from_edges([s - 4 * step, s, s + 4 * step], chart=chart, order=1)
    basis = se.paper.energy_basis(regime, source, cells)
    marker = se.NonNegativeCells(
        "gamma", point_masses=3, support=(s - 3 * step, s - step)
    )
    points, columns = basis.point_columns(marker)
    np.testing.assert_array_equal(
        points.coordinates["gamma"], s + step * np.arange(-3, 0)
    )
    np.testing.assert_array_equal(points.cells, [0, 0, 0])
    np.testing.assert_allclose(
        columns[..., -1],
        regime.predict(source.replace(gamma=se.Delta(s - step))).stokes,
        rtol=3e-14,
        atol=0,
    )
