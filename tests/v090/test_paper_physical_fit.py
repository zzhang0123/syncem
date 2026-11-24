"""Physical moment fits with columns projected through both reduced lattices."""

import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se
from syncem.base.window import A1_HZ_PER_G
from syncem.fitting.family_fit import fit_family
from syncem.fitting.free_fit import fit_free
from syncem.fitting.penalized_fit import fit_nonnegative, fit_penalized
from syncem.front._regime_physical import paper_energy_basis
from syncem.front._regimes import RegimeBasis


@pytest.fixture(scope="module")
def example():
    channels = se.Channels.at([0.4, 0.8, 1.6, 3.0], unit="GHz", half_width=0.2, n_nu=32)
    gamma_min, switch, gamma_max = 280, 300, 321
    source = se.Population(
        gamma=se.PowerLaw(2.6, lo=gamma_min, hi=gamma_max, column=1.0),
        B=(0.005, "G"),
        eta=0.5,
        mu=se.Isotropic(),
        phi=se.AzimuthFactor(0.3 + 0.1j),
        depth=se.GaussianScreen(3.0, 0.4),
    )
    gamma = se.Cells.from_edges([gamma_min, switch, gamma_max])
    high_scale = A1_HZ_PER_G * 0.005 * np.sqrt(0.75)
    regime = RegimeBasis.build(
        source,
        channels,
        harmonic=se.Lattice(
            gamma=se.Cells.from_edges([gamma_min, switch]),
            phi=se.AzimuthHarmonic(),
        ),
        continuum=se.Lattice(
            nu_c=se.Cells.log(high_scale * switch**2, high_scale * gamma_max**2, 1),
            phi=se.AzimuthHarmonic(),
        ),
        gamma_switch=switch,
        field_nodes=2,
        inclination_nodes=2,
        angular_nodes=4,
        frequency_nodes=32,
        moment_nodes=8,
        check=False,
    )
    energy = paper_energy_basis(regime, source, gamma, weights=(0.43, 0.57))
    return regime, energy, source


def test_physical_projection_retains_scaled_coordinates(example):
    regime, energy, source = example
    moments = energy.moments(source)
    expected = regime._predict(energy.project(moments))
    np.testing.assert_allclose(
        energy.predict(moments).stokes, expected, rtol=4e-15, atol=0
    )
    result = energy.predict(source)
    assert result.ledger["physical_energy_difference"].kind == "estimate"
    np.testing.assert_allclose(
        result.stokes, regime.predict(source).stokes, rtol=3e-6, atol=0
    )
    assert float(energy.lattice.column(moments)) == pytest.approx(1)
    assert list(energy.lattice.weights) == [0.43, 0.57]


def test_atom_transfer_at_the_switch_uses_continuum(example):
    regime, energy, source = example
    points, columns = energy.point_columns(se.NonNegativeCells(point_masses=5))
    idx = np.flatnonzero(np.isclose(points.coordinates["gamma"], 300, rtol=1e-14))
    assert idx.size == 1
    expected = regime.predict(source.replace(gamma=se.Delta(300.0))).stokes
    np.testing.assert_allclose(columns[..., idx[0]], expected, rtol=1e-14, atol=0)
    assert points.moments[3, idx[0]] == pytest.approx(1 / 0.57)
    assert points.cells[idx[0]] == 1
    point = source.replace(gamma=se.Delta(300.0))
    low, high = energy.project(energy.moments(point))
    assert not np.any(low.values)
    assert np.any(high.values)
    np.testing.assert_allclose(
        se.predict(point, energy.channels, basis=energy).stokes, expected, rtol=1e-12
    )


def test_fits_accept_the_projected_basis(example):
    _, energy, source = example
    truth = energy.predict(source, check=False)
    data = se.StokesData(
        energy.channels,
        stokes=truth.stokes,
        sigma=truth.I[:, None] * 0.01,
        unit="cgs",
    )
    model = source.replace(
        gamma=se.PowerLaw(se.Free(2.4, bounds=(1.0, 4.0)), lo=280, hi=321)
    )
    for fit in (
        fit_free(energy, data, rank=se.FullRank()),
        fit_penalized(energy, data, se.Penalty(0)),
        fit_family(energy, data, model),
        fit_nonnegative(
            energy, data, se.NonNegativeCells(point_masses=5), se.Penalty(0)
        ),
    ):
        assert np.isfinite(fit.chi2)
        assert float(fit.chi2) < 1e-8
        assert (
            np.max(jnp.abs(fit.predict().stokes - truth.stokes) / truth.I[:, None])
            < 1e-6
        )


def test_projected_family_fit_reports_withheld_stokes(example):
    _, energy, source = example
    truth = np.asarray(energy.predict(source, check=False).stokes)
    sigma = truth[:, :1] * 0.01
    observed = truth.copy()
    observed[-1] += sigma[-1] * np.array([1.0, -2.0, 3.0])
    data = se.StokesData(
        energy.channels, stokes=observed, sigma=sigma, unit="cgs"
    ).withhold([-1])

    fit = se.fit(source, data, basis=energy, check=False)

    assert fit.chi2 < 1e-10
    assert fit.withheld.n == 3
    assert fit.withheld.chi2 == pytest.approx(14.0)
    assert fit.withheld.max_residual == pytest.approx(3.0)


def test_crossing_physical_cells_require_a_switch_face(example):
    regime, _, source = example
    with pytest.raises(se.RoleError, match="cell edge at the regime switch"):
        paper_energy_basis(regime, source, se.Cells.log(280, 321, 1))


def test_joint_azimuth_and_shared_factor_fits(example):
    regime, energy, source = example
    zone_a = source.replace(
        gamma=se.PowerLaw(2.1, lo=280, hi=321, column=0.4),
        phi=se.AzimuthDensity(0.8, 0.25, unit="rad"),
    )
    zone_b = source.replace(
        gamma=se.PowerLaw(3.1, lo=280, hi=321, column=0.6),
        phi=se.AzimuthDensity(0.8, 0.25 + np.pi / 4, unit="rad"),
    )
    lattice = se.Lattice(gamma=energy.lattice.cell_axes[0][1], phi=se.AzimuthHarmonic())
    joint = paper_energy_basis(regime, zone_a + zone_b, lattice)
    moments = joint.moments(zone_a + zone_b)
    expected = regime._predict(joint.project(moments))
    truth = joint.predict(moments, check=False)
    np.testing.assert_allclose(truth.stokes, expected, rtol=5e-15, atol=0)
    assert joint.design().shape[-1] == 3 * energy.design().shape[-1]
    specified = {k: v for k, v in source.factors.items() if k not in ("gamma", "phi")}
    free = se.Population(se.FreeMoments("gamma", "phi"), **specified)
    data = se.StokesData(
        joint.channels, stokes=truth.stokes, sigma=truth.I[:, None] * 0.01, unit="cgs"
    )
    fit = se.fit(free, data, basis=joint, rank=se.FullRank(), check=False)
    np.testing.assert_allclose(fit.predict().stokes, truth.stokes, rtol=1e-9, atol=0)

    shared = paper_energy_basis(
        regime, source.replace(phi=se.AzimuthFactor(1)), energy.lattice
    )
    single = energy.predict(source, check=False)
    data = se.StokesData(
        energy.channels,
        stokes=single.stokes,
        sigma=single.I[:, None] * 0.01,
        unit="cgs",
    )
    free = se.Population(gamma=se.FreeMoments(), phi=se.FreeMoments(), **specified)
    fit = se.fit(free, data, basis=shared, rank=se.Rank(6), check=False)
    assert fit.factor == pytest.approx(0.3 + 0.1j, rel=2e-6)
    np.testing.assert_allclose(fit.predict().stokes, single.stokes, rtol=2e-6, atol=0)
    np.testing.assert_allclose(
        fit.predict(energy.channels[:2]).stokes, single.stokes[:2], rtol=2e-6, atol=0
    )


def test_joint_energy_and_field_projection():
    channels = se.Channels.at([0.4, 0.8, 1.6], unit="GHz", half_width=0.2, n_nu=32)

    def electron(B):
        return se.AgedPowerLaw(
            2.6,
            lo=280,
            gamma_cut=310 + B.value / 0.002,
            amplitude=4e-7,
        )

    source = se.Population(
        gamma=se.Conditional(electron, units={"B": "G"}),
        B=se.Uniform(4, 6, unit="mG"),
        eta=0.5,
        mu=se.Isotropic(),
        phi=se.AzimuthFactor(0.3 + 0.2j),
        depth=se.GaussianScreen(3.0, 0.4),
    )
    gamma = se.Cells.from_edges([280, 300, 325], order=1)
    field = se.Cells.log(4, 6, 1, unit="mG", order=1)
    regime = RegimeBasis.build(
        source,
        channels,
        harmonic=se.Lattice(
            gamma=se.Cells.from_edges([280, 300], order=1),
            B=field,
            phi=se.AzimuthHarmonic(),
        ),
        continuum=se.Lattice(
            nu_c=se.Cells.log(1e9, 3e9, 2, order=2),
            phi=se.AzimuthHarmonic(),
        ),
        gamma_switch=300,
        angular_nodes=4,
        frequency_nodes=32,
        moment_nodes=8,
        check=False,
    )
    energy = paper_energy_basis(regime, source, se.Lattice(gamma=gamma, B=field))
    truth = energy.predict(source, check=False)
    np.testing.assert_allclose(
        truth.stokes, regime._predict(energy.project(truth.moments)), rtol=1e-14, atol=0
    )
    data = se.StokesData(
        channels, stokes=truth.stokes, sigma=truth.I[:, None] * 0.01, unit="cgs"
    )
    fixed = {k: v for k, v in source.factors.items() if k not in ("gamma", "B")}
    free = se.Population(se.FreeMoments("gamma", "B"), **fixed)
    fit = se.fit(free, data, basis=energy, rank=se.FullRank(), check=False)
    np.testing.assert_allclose(fit.predict().stokes, truth.stokes, rtol=1e-9, atol=0)
