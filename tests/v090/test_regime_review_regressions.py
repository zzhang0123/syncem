"""Regression checks for channel identity, projected coverage, and angle laws."""

from types import SimpleNamespace

import numpy as np
import pytest

import syncem as se
from syncem.base.window import A1_HZ_PER_G
from syncem.engine._regime_columns import regime_channel_indices
from syncem.engine._regime_harmonic import harmonic_channel
from syncem.front._regime_physical import _check_projection_width
from syncem.front._regimes import RegimeBasis


def _regime(source, channels, gamma, critical, *, switch=300, moments=3, **angular):
    phi = se.AzimuthHarmonic()
    return RegimeBasis.build(
        source,
        channels,
        harmonic=se.Lattice(gamma=gamma, phi=phi, **angular),
        continuum=se.Lattice(nu_c=critical, phi=phi),
        gamma_switch=switch,
        field_nodes=2,
        inclination_nodes=2,
        angular_nodes=2,
        frequency_nodes=16,
        moment_nodes=moments,
        check=False,
    )


def _population(gamma, *, field=5e-6, eta=0.0, mu=None):
    return se.Population(
        gamma=gamma,
        B=(field, "G"),
        eta=eta,
        mu=se.Isotropic() if mu is None else mu,
        phi=se.AzimuthFactor(0.2 + 0.1j),
        depth=se.NoRotation(),
    )


@pytest.fixture
def matching_quadrature():
    frequencies = np.arange(300, 701, 100) * 1e3
    tables = [[0, 1, 0.5, 0.25, 0], [0, 0.25, 0.5, 1, 0]]
    grids = tuple(
        se.Channels.from_bandpass([(frequencies, row)], unit="Hz", n_nu=1)
        for row in tables
    )
    assert grids[0].row_keys() == grids[1].row_keys()
    field = 200e3 * 10 / ((2 / 3) * A1_HZ_PER_G)
    source = _population(se.Delta(10.0), field=field, mu=0.2)
    gamma = se.Cells.linear(9.0, 11.0, 1, order=0)
    critical = se.Cells.log(1e6, 1e7, 1)
    regimes = tuple(_regime(source, grid, gamma, critical, switch=11) for grid in grids)
    return grids, regimes, source, gamma


def test_integer_line_response_controls_rebuild_and_physical_fit(matching_quadrature):
    (first, second), (old, fresh), source, gamma = matching_quadrature
    direct = fresh.predict(source, check=False)
    assert float(old.predict(source, check=False).I[0] / direct.I[0]) < 0.9
    np.testing.assert_allclose(
        old.at(second).predict(source).stokes, direct.stokes, atol=0
    )
    field = 200e3 * 10 / ((2 / 3) * A1_HZ_PER_G)
    reference = harmonic_channel(second, 10.0, field, 0.0, 0.2)
    np.testing.assert_allclose(direct.I, reference[:, 0].real, rtol=1e-14, atol=0)

    energy = se.paper.energy_basis(old, source, gamma, order=0)
    rebuilt = se.paper.energy_basis(fresh, source, gamma, order=0)
    assert energy.at(first) is energy
    np.testing.assert_allclose(energy.at(second).design(), rebuilt.design(), atol=0)
    assert not np.allclose(energy.design(), rebuilt.design(), rtol=0.1, atol=0)

    target = rebuilt.predict(source, check=False)
    data = se.StokesData(
        second, stokes=target.stokes, sigma=target.I[:, None], unit="cgs"
    )
    free = source.replace(gamma=se.FreeMoments())
    expected = se.fit(free, data, basis=rebuilt, rank=se.FullRank(), check=False)
    actual = se.fit(free, data, basis=energy, rank=se.FullRank(), check=False)
    np.testing.assert_allclose(actual.moments.values, expected.moments.values, atol=0)
    np.testing.assert_allclose(energy.design(data), rebuilt.design(data), atol=0)


def test_known_response_selection_and_opaque_response_keys():
    grid = se.Channels.at([1.2, 1.5], unit="MHz", half_width=0.2, n_nu=8)
    np.testing.assert_array_equal(regime_channel_indices(grid, grid[[1, 0]]), [1, 0])

    def response(frequency):
        return 1 + (frequency - 0.5e6) / 0.4e6

    first = se.Channels.from_callable(response, [0.3e6, 0.7e6], 0, n_nu=1)
    second = se.Channels.from_callable(
        lambda frequency: 2 - response(frequency), [0.3e6, 0.7e6], 0, n_nu=1
    )
    assert first.row_keys() == second.row_keys()
    np.testing.assert_array_equal(regime_channel_indices(first, second), [-1])
    np.testing.assert_array_equal(
        regime_channel_indices(first, second, harmonic=False), [0]
    )


def test_same_line_response_and_new_quadrature_preserves_new_grid():
    source = _population(se.Delta(390), eta=0.5)
    first, second = (
        se.Channels.at([1.2], unit="MHz", half_width=0.2, n_nu=n) for n in (8, 16)
    )
    regime = _regime(
        source,
        first,
        se.Cells.from_edges([299.0, 300.0]),
        se.Cells.log(1e4, 1e7, 4),
    )
    selected = regime.at(second)
    assert selected.channels is second
    assert selected.harmonic.spectra0 is regime.harmonic.spectra0
    assert selected.continuum.spectra0 is not regime.continuum.spectra0


def test_narrow_band_is_checked_against_fitted_domains():
    broad = se.Channels.at([0.4, 0.8], unit="GHz", half_width=0.2, n_nu=8)
    narrow = se.Channels.at([0.4], unit="GHz", half_width=1e-6, n_nu=8)
    source = _population(se.Delta(3000), field=5e-3, eta=0.5)
    gamma = se.Cells.from_edges([290.0, 300.0, 340.0])
    critical = se.Cells.log(1e9, 1e12, 1)
    regime = _regime(source, broad, se.Cells.from_edges([290.0, 300.0]), critical)
    energy = se.paper.energy_basis(regime, source, gamma, order=0)
    assert np.isfinite(regime.at(narrow).predict(source, check=False).I[0])
    with pytest.raises(se.RegimeError, match="32"):
        regime.at(narrow).predict(source.replace(gamma=se.Delta(310)), check=False)
    with pytest.raises(se.RegimeError, match="32"):
        energy.at(narrow)
    with pytest.raises(se.RegimeError, match="32"):
        se.paper.energy_basis(regime.at(narrow), source, gamma, order=0)

    width = energy.minimum_width
    assert width > (narrow.support[0, 1] - narrow.support[0, 0])
    band = SimpleNamespace(support=np.array([[width, 2 * width]]))
    _check_projection_width(band, width)
    band.support[0, 1] = np.nextafter(2 * width, -np.inf)
    with pytest.raises(se.RegimeError, match="32"):
        _check_projection_width(band, width)


def test_refined_spectrum_reports_the_refined_coverage():
    source = _population(se.Delta(400), eta=se.Uniform(-0.5, 0.5))
    centre = A1_HZ_PER_G * 5e-6 * 400**2
    channels = se.Channels.at([1.2], unit="MHz", half_width=0.2, n_nu=8)
    regime = _regime(
        source,
        channels,
        se.Cells.from_edges([299.0, 300.0]),
        se.Cells.log(0.98 * centre, 1.02 * centre, 1),
        moments=1,
    )
    coarse = regime.predict(source, check=False)
    refined = regime.predict(source, check=True)
    q = np.sqrt((1 - 0.98) * (1 + 0.98))
    faces = np.array([-0.5, -q, q, 0.5])

    def outside(n):
        x, w = np.polynomial.legendre.leggauss(n)
        half = np.diff(faces) / 2
        eta = faces[:-1, None] + half[:, None] * (1 + x)
        mass = half * (np.sqrt(1 - eta**2) @ w)
        return (mass[0] + mass[-1]) / mass.sum()

    for result, order in ((coarse, 1), (refined, 2)):
        assert result.I[0] > 0
        np.testing.assert_allclose(
            result.ledger["outside_continuum"].value, outside(order), rtol=2e-14
        )

    def primitive(z):
        return 0.5 * (np.arcsin(z) + z * np.sqrt(1 - z**2))

    exact = 1 - primitive(q) / primitive(0.5)
    assert abs(outside(2) - exact) < abs(outside(1) - exact)


def test_conditional_point_angle_and_mixed_law_dispatch():
    def pitch(gamma):
        return se.Delta(0.2) if gamma.value < 400 else se.Isotropic()

    source = _population(se.Delta(390), eta=se.Isotropic(), mu=se.Conditional(pitch))
    channels = se.Channels.at([1.2], unit="MHz", half_width=0.2, n_nu=8)
    regime = _regime(
        source,
        channels,
        se.Cells.from_edges([299.0, 300.0]),
        se.Cells.log(1e4, 1e7, 4),
        mu=se.Legendre(2),
    )
    point = regime.predict(source.replace(mu=se.Delta(0.2)), check=False)
    conditional = regime.predict(source, check=False)
    assert point.I[0] > 0
    np.testing.assert_allclose(conditional.stokes, point.stokes, rtol=1e-14, atol=0)

    mixture = se.Population(
        se.Samples(gamma=[390, 410], weights=[0.4, 0.6], column=1.0),
        **{name: law for name, law in source.factors.items() if name != "gamma"},
    )
    continuous = regime.predict(
        source.replace(gamma=se.Delta(410), mu=se.Isotropic()), check=False
    )
    expected = 0.4 * point.stokes + 0.6 * continuous.stokes
    actual = regime.predict(mixture, check=False)
    np.testing.assert_allclose(actual.stokes, expected, rtol=1e-14, atol=0)


def test_conditional_point_angle_follows_nested_parents():
    def field(gamma):
        return se.Delta(4.5e-6 if gamma.value < 400 else 5.5e-6, unit="G")

    def pitch(B):
        return se.Delta(0.2) if B.value < 5e-6 else se.Isotropic()

    source = _population(
        se.Delta(390), eta=se.Isotropic(), mu=se.Conditional(pitch, units={"B": "G"})
    ).replace(B=se.Conditional(field))
    channels = se.Channels.at([1.2], unit="MHz", half_width=0.2, n_nu=8)
    regime = _regime(
        source,
        channels,
        se.Cells.from_edges([299.0, 300.0]),
        se.Cells.log(1e4, 1e7, 4),
        B=se.Cells.linear(4, 6, 1, unit="uG"),
        mu=se.Legendre(2),
        eta=se.Legendre(2),
    )
    point = regime.predict(source.replace(mu=se.Delta(0.2)), check=False)
    actual = regime.predict(source, check=False)
    assert point.I[0] > 0
    np.testing.assert_allclose(actual.stokes, point.stokes, rtol=1e-14, atol=0)

    def field_with_tied_ancestor(eta):
        return se.Delta(4.5e-6, unit="G")

    cyclic_reverse = source.replace(B=se.Conditional(field_with_tied_ancestor))
    with pytest.raises(se.RoleError, match="continuous density for mu"):
        regime.predict(cyclic_reverse, check=False)


def test_folded_pitch_knots_preserve_mass_and_absolute_spectrum():
    from scipy.integrate import quad
    from scipy.special import kv
    from syncem.constants import C_CGS, E_ESU, M_E

    gamma, field = 30_000.0, 5e-6
    critical = A1_HZ_PER_G * field * gamma**2
    channels = se.Channels.at([0.55 * critical], unit="Hz", half_width=0.05, n_nu=8)
    source = _population(se.Delta(gamma), field=field, eta=se.Isotropic())
    basis = _regime(
        source,
        channels,
        se.Cells.linear(299, 300, 1, order=0),
        se.Cells.log(0.985 * critical, critical, 2, order=4),
        moments=12,
        mu=se.Legendre(1),
        eta=se.Legendre(1),
    )
    lo, peak, hi = 0.128, 0.132, 0.136
    shape = [0, 2 / (hi - lo), 0]
    nodes, response = np.asarray(channels.nodes[0]), np.asarray(
        channels.response_weights()[0]
    )

    def density(x):
        return np.interp(x, [lo, peak, hi], shape)

    def integrand(x):
        sine = np.sqrt(1 - x * x)
        argument = nodes / (critical * sine)
        intensity = [
            t * quad(lambda u: kv(5 / 3, u), t, np.inf, epsabs=1e-12)[0]
            for t in argument
        ]
        return sine * density(x) * np.dot(response, intensity)

    reference = (
        quad(integrand, lo, hi, points=[peak])[0]
        * np.sqrt(3)
        * E_ESU**3
        * field
        / (4 * np.pi * M_E * C_CGS**2)
    )
    effective_column = quad(
        lambda x: np.sqrt(1 - x * x) * density(x), lo, hi, points=[peak]
    )[0]
    laws = (
        se.Tabulated([lo, peak, hi], [0, 1, 0]),
        se.Tabulated([-1, lo, peak, hi, 1], [0, 0, 1, 0, 0]),
    )

    def conditioned(gamma):
        # Conditioning on energy leaves this same density at the selected energy.
        shift = (gamma.value - 30_000) / 100_000
        return se.Tabulated(np.array([-1, lo, peak, hi, 1]) + shift, [0, 0, 1, 0, 0])

    for law in (*laws, se.Conditional(conditioned)):
        population = source.replace(mu=law)
        harmonic, continuum = basis.moments(population)
        np.testing.assert_array_equal(harmonic.values, 0)
        np.testing.assert_allclose(
            basis.continuum.lattice.column(continuum), effective_column, rtol=1e-12
        )
        for refine in (False, True):
            result = basis.predict(population, check=refine)
            np.testing.assert_allclose(result.I, reference, rtol=2e-9, atol=0)
        np.testing.assert_allclose(
            basis.predict((harmonic, continuum)).I, reference, rtol=2e-9, atol=0
        )


def test_narrow_harmonic_cells_and_switch_boundary_preserve_mass_and_ownership():
    from syncem.front._regime_measure import diagonal_energy_moments, source_columns

    g0 = 200.0
    dg = 2.0**-42
    switch = g0 + 2 * dg
    harmonic_lattice = se.Lattice(
        gamma=se.Cells.linear(g0, switch, 2, order=1), phi=se.AzimuthHarmonic()
    )
    continuum_lattice = se.Lattice(
        nu_c=se.Cells.log("1e5 Hz", "1e11 Hz", 16, order=4), phi=se.AzimuthHarmonic()
    )
    pop_cell1 = se.Population(
        gamma=se.Uniform(g0 + dg, switch),
        B=(5e-6, "G"),
        eta=0.2,
        mu=se.Isotropic(),
        phi=se.AzimuthFactor(0.4),
        depth=se.NoRotation(),
    )
    basis = se.RegimeBasis.build(
        pop_cell1,
        se.Channels.at([10.0], unit="MHz", half_width=0.1, n_nu=8),
        harmonic=harmonic_lattice,
        continuum=continuum_lattice,
        gamma_switch=switch,
        check=False,
        moment_nodes=48,
    )
    m_h, m_c = basis.moments(pop_cell1)
    idx = basis.harmonic.lattice.index
    weights = basis.harmonic.lattice.weights
    cell0_mass = float(m_h.values[idx.cell_rows(0).start] * weights[0])
    cell1_mass = float(m_h.values[idx.cell_rows(1).start] * weights[1])
    cell1_first = float(m_h.values[idx.cell_rows(1).start + 1] * weights[1])
    assert cell0_mass == 0.0
    np.testing.assert_allclose(cell1_mass, 1.0, rtol=1e-14, atol=0)
    np.testing.assert_allclose(cell1_first, 0.0, atol=1e-13)
    np.testing.assert_array_equal(m_c.values, 0.0)

    # An atom placed directly at gamma_switch still belongs to the continuum branch.
    pop_atom = pop_cell1.replace(gamma=se.Delta(switch))
    atom_h, atom_c = basis.moments(pop_atom)
    np.testing.assert_array_equal(atom_h.values, 0.0)
    assert float(np.sum(np.abs(atom_c.values))) > 0.0
    low_atom, high_atom = source_columns(pop_atom.parts, switch, nodes=48)
    assert low_atom == 0.0 and high_atom == 1.0

    # Continuous harmonic densities ending at gamma_switch=300 stay below the switch.
    for mult in (8, 64, 512):
        width = mult * np.spacing(300.0)
        pop_below = pop_cell1.replace(gamma=se.Uniform(300.0 - width, 300.0))
        low_col, high_col = source_columns(pop_below.parts, 300.0, nodes=48)
        np.testing.assert_allclose(low_col, 1.0, rtol=1e-14, atol=0)
        assert high_col == 0.0

    # diagonal_energy_moments also preserves narrow adjacent cell ownership.
    gamma_only = se.Lattice(gamma=se.Cells.linear(g0, switch, 2, order=1))
    diag = diagonal_energy_moments(pop_cell1.parts, gamma_only, nodes=48)
    d_idx, d_w = gamma_only.index, gamma_only.weights
    assert float(diag.values[d_idx.cell_rows(0).start] * d_w[0]) == 0.0
    np.testing.assert_allclose(
        float(diag.values[d_idx.cell_rows(1).start] * d_w[1]), 1.0, rtol=1e-14, atol=0
    )
