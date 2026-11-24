"""Absolute Stokes checks for the optional harmonic shape basis."""

from dataclasses import replace

import numpy as np
import pytest

import syncem as se
from syncem.constants import C_CGS, E_ESU, M_E
from syncem.model.high_order_harmonic import (
    high_order_channel_modes,
    high_order_line_powers,
)
from syncem.model.phase import TaylorPhase

_GYRO = E_ESU / (2 * np.pi * M_E * C_CGS)


def _reference(channels, gamma, field, mu, eta, phi, depth):
    modes = high_order_channel_modes(
        channels, gamma, field, mu, eta, phase=TaylorPhase(0), depth_ref=depth
    ).modes
    pol = np.exp(2j * phi) * modes.P[0]
    return np.column_stack((modes.I, pol.real, pol.imag, modes.V))


def _source(gamma, field, mu, eta, phi=0.0, depth=0.0):
    return se.Population(
        gamma=gamma, B=(field, "G"), mu=mu, eta=eta, phi=(phi, "rad"), depth=depth
    )


@pytest.mark.parametrize("sign", [1, -1])
def test_absolute_signed_stokes_and_faraday_at_lines(sign):
    channels = se.Channels.at([1.19, 2.6, 4.3], unit="GHz", half_width=0.28)
    basis = se.HarmonicShapeBasis.build(channels, spacing=(0.6e9, 2e9))
    point = 3.0, 1350.0, sign * 0.25, sign * 0.65, 0.28, 12.0
    population = _source(*point)
    result = se.predict(population, channels, basis=basis)
    expected = _reference(channels, *point)
    pin = [
        2.9011631604553816e-20,
        -8.391670784749491e-21,
        -8.704443471733344e-21,
        sign * 2.6372093084283746e-20,
    ]
    np.testing.assert_allclose(result.stokes[1], pin, rtol=2e-12, atol=0)
    np.testing.assert_allclose(result.stokes, expected, rtol=2e-12, atol=0)
    rep, transform = (
        result.ledger[name] for name in ("representation", "coordinate_transform")
    )
    assert rep.kind == transform.kind == "measured"
    assert rep.signed and transform.signed
    np.testing.assert_allclose(
        result.stokes - expected, rep.value + transform.value, atol=1e-34, rtol=0
    )
    assert result.ledger["quadrature"].kind == "unbounded"
    assert result.ledger["harmonic_numerics"].kind == "unbounded"

    intrinsic = _reference(channels, *point[:-1], 0.0)
    centre_rotation = (intrinsic[:, 1] + 1j * intrinsic[:, 2]) * np.exp(
        1j * 2 * (299792458 / np.asarray(channels.centres_hz)) ** 2 * point[-1]
    )
    assert np.max(np.abs(result.P - centre_rotation)) > 0.05 * np.max(np.abs(result.P))
    selected = se.predict(
        population, channels[[2, 0]], basis=basis, stokes="VI", check=False
    )
    np.testing.assert_allclose(selected.stokes, expected[[2, 0]][:, [3, 0]], rtol=2e-12)
    assert selected.components == ("V", "I")
    assert selected.ledger["representation"].kind == "unbounded"


def test_joint_power_scale_and_conditional_depth_and_azimuth(tmp_path):
    channels = se.Channels.at([0.8, 1.25, 2.15], unit="MHz", half_width=0.47)
    spacing, z, s = 0.9e6, 0.60, 0.32
    eta = np.array([-0.55, 0.65])
    d = (1 - eta**2) / (1 - s * eta)
    parallel = (eta - s) / (1 - s * eta)
    perp = z * np.sqrt(1 - eta**2) / (1 - s * eta)
    gamma = 1 / np.sqrt(1 - parallel**2 - perp**2)
    mu = parallel / np.hypot(parallel, perp)
    field = spacing * gamma * d / _GYRO
    phi = 0.4 * eta + 0.1
    depth = (2 + eta) * 1e-6
    mass = np.array([1.0, 4.0])
    block = se.Samples(
        gamma=gamma, B=(field, "G"), mu=mu, eta=eta, weights=mass, column=mass.sum()
    )
    population = se.Population(
        block,
        phi=se.Conditional(lambda eta: se.Quantity(0.4 * eta.value + 0.1, "rad")),
        depth=se.Conditional(lambda eta: f"{(2 + eta.value) * 1e-6} rad m^-2"),
    )
    basis = se.HarmonicShapeBasis.build(channels, spacing=(0.89e6, 0.91e6))
    result = basis.predict(2 * population)
    expected = sum(
        2 * w * _reference(channels, g, b, p, e, angle, rm)
        for g, b, p, e, angle, rm, w in zip(
            gamma, field, mu, eta, phi, depth, mass, strict=True
        )
    )
    assert perp[0] ** 2 / d[0] != perp[1] ** 2 / d[1]
    np.testing.assert_allclose(result.stokes, expected, rtol=2e-12)
    np.testing.assert_allclose(basis.predict(block).P, 0)
    assert not any(
        term.name == "depth_cumulant_truncation" for term in result.ledger.terms
    )
    moments = basis.moments(2 * population)
    np.testing.assert_allclose(
        moments.coordinates, np.broadcast_to([spacing, z, s], (2, 3)), rtol=2e-14
    )
    np.testing.assert_allclose(
        moments.power_weights / moments.power_weights[0],
        mass / mass[0] * (perp**2 / d) / (perp[0] ** 2 / d[0]),
        rtol=2e-14,
    )
    np.testing.assert_allclose(
        moments.spin_weights / moments.power_weights, np.exp(2j * phi)
    )
    np.testing.assert_allclose(moments.depths, depth)
    file = tmp_path / "joint-shape.npz"
    moments.save(file)
    restored = se.HarmonicShapeStatistics.load(file)
    changed = se.Channels.at([1.05, 1.65], unit="MHz", half_width=0.24)
    replay = se.predict(restored, changed, basis=basis)
    target = basis.at(changed).predict(2 * population)
    np.testing.assert_array_equal(replay.stokes, target.stokes)
    assert replay.moments is restored
    assert replay.ledger["coordinate_transform"].kind == "unbounded"
    with pytest.raises(se.SizeError, match="max_atoms=1"):
        basis.moments(population, max_atoms=1)


def test_depth_conditioned_on_azimuth_and_zero_parent_field_factory():
    channels = se.Channels.at([1.19, 2.6], unit="GHz", half_width=0.28)
    basis = se.HarmonicShapeBasis.build(channels, spacing=(0.6e9, 2e9))
    population = se.Population(
        se.Samples(phi=([15, 75], "deg"), weights=[1, 3]),
        gamma=3,
        B=se.Conditional(lambda field: (field, "G"), params={"field": 1350}),
        mu=0.25,
        eta=0.65,
        depth=se.Conditional(
            lambda phi: f"{12*phi.value} rad m^-2", units={"phi": "rad"}
        ),
    )
    angles = np.deg2rad([15, 75])
    expected = sum(
        w * _reference(channels, 3, 1350, 0.25, 0.65, phi, 12 * phi)
        for w, phi in zip([0.25, 0.75], angles, strict=True)
    )
    np.testing.assert_allclose(basis.predict(population).stokes, expected, rtol=2e-12)


def test_density_and_independent_and_conditional_depth_characteristics():
    channels = se.Channels.at([1.19, 2.6], unit="GHz", half_width=0.28)
    population = se.Population(
        gamma=se.Uniform(2.9, 3.1),
        B="1350 G",
        mu=0.25,
        eta=0.65,
        phi=se.Conditional(lambda angle: f"{angle} deg", params={"angle": 15}),
        depth=se.Conditional(
            lambda gamma: se.EmpiricalScreen([0, 4 * gamma.value], [1, 3])
        ),
    )
    foreground = se.EmpiricalScreen([-1, 6], [3, 2])
    basis = se.HarmonicShapeBasis.build(
        channels, spacing=(0.6e9, 2e9), phase=se.FaradayRotation(screen=foreground)
    )
    x, w = np.polynomial.legendre.leggauss(basis.nodes)
    lo, hi = np.log([2.9, 3.1])
    gamma = np.exp((lo + hi) / 2 + (hi - lo) / 2 * x)
    weight = w * gamma * (hi - lo) / (2 * (3.1 - 2.9))
    expected = sum(
        w_g
        * w_depth
        * w_fg
        * _reference(channels, g, 1350, 0.25, 0.65, np.pi / 12, depth + fg)
        for g, w_g in zip(gamma, weight, strict=True)
        for depth, w_depth in ((0, 0.25), (4 * g, 0.75))
        for fg, w_fg in ((-1, 0.6), (6, 0.4))
    )
    np.testing.assert_allclose(basis.predict(population).stokes, expected, rtol=2e-12)


def test_leaf_depth_rule_keeps_intrinsic_weights_and_can_change_phase(tmp_path):
    channels = se.Channels.at([1.19, 2.6], unit="GHz", half_width=0.28)
    options = {"spacing": (0.6e9, 2e9), "nodes": 2}
    bare = se.HarmonicShapeBasis.build(channels, **options, phase=False)
    point = 3.0, 1350.0, 0.25, 0.65, np.pi / 12
    law = se.Normal(12, 2.5, unit="rad m^-2")
    population = _source(*point).replace(depth=law)
    statistics = bare.moments(population)
    assert statistics.coordinates.shape == (1, 3)
    file = tmp_path / "depth-rule.npz"
    statistics.save(file)
    loaded = se.HarmonicShapeStatistics.load(file)
    np.testing.assert_allclose(
        bare.predict(population).stokes, _reference(channels, *point, 0)
    )
    np.testing.assert_array_equal(
        bare.predict(loaded).stokes, bare.predict(population).stokes
    )
    depth, weight = law.rule()
    expected = sum(
        w * _reference(channels, *point, rm)
        for rm, w in zip(np.asarray(depth), np.asarray(weight), strict=True)
    )
    rotated = se.HarmonicShapeBasis.build(channels, **options)
    np.testing.assert_allclose(
        rotated.predict(loaded).stokes, expected, rtol=2e-12, atol=0
    )


def test_analytic_screen_and_empty_statistics_archive(tmp_path):
    channels = se.Channels.at([1.19, 2.6], unit="GHz", half_width=0.28)
    basis = se.HarmonicShapeBasis.build(channels, spacing=(0.6e9, 2e9))
    source = _source(3, 1350, 0.25, 0.65, 0.3).replace(
        depth=se.Conditional(
            lambda mu: se.CumulantScreen([4 * mu.value, 0.6, 0.1, 0], tau_max=1.0)
        )
    )
    for pop in (source, 0 * source):
        original = basis.moments(pop)
        file = tmp_path / "statistics.npz"
        original.save(file)
        loaded = se.HarmonicShapeStatistics.load(file)
        result = basis.predict(loaded)
        np.testing.assert_array_equal(result.stokes, basis.predict(original).stokes)
        if loaded.coordinates.size:
            assert result.ledger["depth_cumulant_truncation"].kind == "unbounded"
        else:
            np.testing.assert_array_equal(result.stokes, 0.0)


def test_custom_characteristic_replays_in_memory_and_refuses_archive(tmp_path):
    class Screen:
        given_unit = "rad m^-2"
        is_point = False

        def characteristic(self, tau):
            return np.exp(2j * np.asarray(tau))

    channels = se.Channels.at([1.19, 2.6], unit="GHz", half_width=0.28)
    basis = se.HarmonicShapeBasis.build(channels, spacing=(0.6e9, 2e9))
    point = 3.0, 1350.0, 0.25, 0.65, 0.28
    statistics = basis.moments(_source(*point).replace(depth=Screen()))
    np.testing.assert_allclose(
        basis.predict(statistics).stokes,
        _reference(channels, *point, 2),
        rtol=2e-12,
        atol=0,
    )
    with pytest.raises(se.DataError, match="custom depth characteristic"):
        statistics.save(tmp_path / "custom.npz")
    with pytest.raises(se.DataError, match="screen index"):
        replace(
            statistics,
            screen_indices=np.array([np.iinfo(np.uint64).max], dtype=np.uint64),
        )


@pytest.mark.parametrize("eta", [1.0, -1.0, 1.0 - 1e-12, -1.0 + 1e-12])
def test_both_sides_of_the_viewing_axis_limit(eta):
    channels = se.Channels.at([0.8, 1.25, 2.15], unit="MHz", half_width=0.47)
    basis = se.HarmonicShapeBasis.build(channels, spacing=(0.5e6, 1e6))
    point = 2.0, 0.5, 0.10, eta, 0.3, 1.2e-6
    result = basis.predict(_source(*point))
    expected = _reference(channels, *point)
    np.testing.assert_allclose(
        result.stokes, expected, atol=2e-13 * expected[0, 0], rtol=2e-12
    )
    pin = 2.521544182283037e-23 if eta > 0 else 1.226673149621942e-23
    np.testing.assert_allclose(result.I[0], pin, rtol=5e-11)
    if abs(eta) == 1:
        np.testing.assert_array_equal(result.P, 0.0)
        np.testing.assert_array_equal(result.V, eta * result.I)


@pytest.mark.parametrize("z", [0.954, 0.998])
def test_integer_orders_above_one_thousand_and_finite_basis_refinement(z):
    channels = se.Channels.at([1.1], unit="GHz", half_width=0.05)
    gamma = 1 / np.sqrt(1 - z**2)
    point = gamma, 1e6 * gamma / _GYRO, 0.0, 0.0, 0.0, 0.0
    population = _source(*point)
    spacing = (0.99e6, 1.01e6)
    coarse = se.HarmonicShapeBasis.build(channels, spacing=spacing, order=4).predict(
        population
    )
    fine = se.HarmonicShapeBasis.build(
        channels, spacing=spacing, z_edges=(0.95, 0.96, 0.98, 0.99, 1.0), order=16
    ).predict(population)
    expected = _reference(channels, *point)
    pin = 2.333666531759647e-30 if z == 0.954 else 7.929271907699315e-22
    np.testing.assert_allclose(fine.I[0], pin, rtol=3e-11)
    np.testing.assert_allclose(fine.stokes, expected, rtol=3e-11, atol=0)
    assert coarse.I[0] > 0.0
    assert abs(fine.ledger["representation"].value[0, 0]) < 1e-8 * abs(
        coarse.ledger["representation"].value[0, 0]
    )


def test_sharp_passband_edges_and_declared_spacing_boundary():
    # Exact edge hits should be zero for the open top-hat support.
    channels = se.Channels.tophat([2e6], [1e6], n_nu=8)
    field = 1e6 / _GYRO * 2
    spacing = float(high_order_line_powers([1], 2, field, 0, 0)[3][0])
    basis = se.HarmonicShapeBasis.build(channels, spacing=(spacing, 1.1 * spacing))
    for b in (field, field * (1 + 1e-12)):
        point = 2, b, 0, 0, 0, 0
        np.testing.assert_allclose(
            basis.predict(_source(*point)).stokes,
            _reference(channels, *point),
            rtol=2e-12,
            atol=0,
        )
    with pytest.raises(se.RegimeError, match="outside spacing"):
        basis.predict(_source(2, field * (1 - 1e-12), 0, 0))


def test_coincident_reduced_shapes_at_sharp_edge(tmp_path):
    channels = se.Channels.tophat([3.05e6, 2.95e6], [0.05e6, 0.05e6], n_nu=8)
    basis = se.HarmonicShapeBasis.build(channels, spacing=(0.9999e6, 1.0001e6))
    points = (
        (2.7035081649188113, 0.28994150513270905, -0.9416911761436771, -0.79985),
        (2.6135547040887226, 0.29901781226486557, -0.936747057728619, -0.7854),
    )
    first, second = (_source(*p) for p in points)
    stats = basis.moments(first + second)
    np.testing.assert_array_equal(stats.coordinates[0], stats.coordinates[1])
    expected = _reference(channels, *points[1], 0, 0)
    np.testing.assert_allclose(expected[0, 0], 3.4095753362414362e-18, rtol=3e-13)
    result = basis.predict(second)
    np.testing.assert_array_equal(result.stokes, 0)
    np.testing.assert_allclose(
        result.ledger["coordinate_transform"].value, -expected, rtol=3e-13, atol=0
    )
    file = tmp_path / "edge.npz"
    stats.save(file)
    loaded = se.HarmonicShapeStatistics.load(file)
    np.testing.assert_array_equal(basis.predict(loaded).stokes, 0)
    for toward, expected_active in [(0, [False, True]), (np.inf, [True, False])]:
        coordinates = loaded.coordinates.copy()
        coordinates[:, 0] = np.nextafter(coordinates[:, 0], toward)
        adjacent = replace(loaded, coordinates=coordinates)
        np.testing.assert_array_equal(basis.predict(adjacent).I > 0, expected_active)


def test_declared_ratio_support_zero_emission_and_order_budget():
    channels = se.Channels.tophat([2e6], [1e6], n_nu=8)
    options = {"spacing": (1e6, 1.5e6), "z_edges": (0.95, 1)}
    with pytest.raises(se.SizeError, match="5 line orders"):
        se.HarmonicShapeBasis.build(channels, **options, max_orders=4)
    basis = se.HarmonicShapeBasis.build(channels, **options, max_orders=5)
    with pytest.raises(se.RegimeError, match="outside z_edges"):
        basis.predict(_source(2, 1e6 * 2.2 / _GYRO, 0, 0))
    axis = _source(2, 1e6 * 2.2 / _GYRO, 0, 1)
    assert basis.predict(axis).I[0] > 0
    no_field = axis.replace(B=se.Delta(0, unit="G"))
    for zero in (axis.replace(gamma=1), axis.replace(mu=1), no_field):
        np.testing.assert_array_equal(basis.predict(zero).stokes, 0.0)
    np.testing.assert_array_equal(
        basis.predict(axis + 0 * axis.replace(B="1 G")).stokes,
        basis.predict(axis).stokes,
    )
    with pytest.raises(ValueError, match="finite positive-width"):
        se.HarmonicShapeBasis.build(se.Channels.at([1.0], unit="GHz"), **options)
