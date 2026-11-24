"""Check the one-cell bridge against the frozen response and column conventions."""

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.engine.single_reference import from_single_reference
from syncem.lattice.moments import Moments
from syncem.model._basis_checks import UNITS
from syncem.model.basis import KernelTerms, SpectralBasis, build_basis
from syncem.model.channels import Channels
from syncem.model.errors import ErrorTerm, Provenance
from syncem.model.harmonic import HarmonicKernel
from syncem.model.index import MomentIndex, Truncation
from syncem.model.moments import JointMoments, Reference, Support
from syncem.values.errors import RoleError, UnitError


def _frozen(
    *,
    channels=None,
    truncation=None,
    parity=True,
    components=("I", "Q", "V"),
    depth=(2, 12),
):
    if channels is None:
        channels = Channels.tophat([1e8, 2e8], [1e7, 4e7], n_nu=8)
    if truncation is None:
        truncation = Truncation(8, 8, 2)
    index = MomentIndex(truncation, parity=parity, components=components)
    ref = Reference(300.0, 5e-6, depth_ref=8.0, scales=(80, 2e-6, 3))
    support = Support((200, 450), (2e-6, 9e-6), depth)
    rng = np.random.default_rng(21)
    i = rng.normal(size=(channels.n_ch, index.n0)) * index.parity_I()
    v = rng.normal(size=i.shape) * index.parity_V() * ("V" in components)
    p = rng.normal(size=(channels.n_ch, index.n2, 2))
    term = ErrorTerm.unbounded("fixture")
    provenance = Provenance(
        package_version="0.7.10",
        kernel=(("name", "fixture"),),
        channels=channels.describe(),
        truncation=(),
        reference=ref.describe(),
        support=support.describe(),
        phase_route=(),
        quadrature=(),
        assumptions=(),
        units=UNITS,
        numerics=(),
        certified_orders=(),
        finite_checks=(),
        notes=(),
    )
    return SpectralBasis(
        index,
        truncation,
        channels,
        ref,
        support,
        jnp.array(i),
        jnp.array(v),
        jnp.array(p[..., 0] + 1j * p[..., 1]),
        KernelTerms(term, term, term, term),
        provenance,
    )


def _moments(old):
    values = np.random.default_rng(41).normal(size=old.index.n_real)
    values[0] = 1.0
    return JointMoments.from_vector(old.index, values, old.reference)


def _compare(old, basis, channel_scale):
    moments = _moments(old)
    column = 7.0
    new = basis.from_legacy_moments(moments, column="70000 m^-2")
    expected = (
        np.asarray(old.response_matrix()) @ np.asarray(moments.to_vector())
    ).reshape(old.n_ch, 4)
    axes = ["IQUV".index(name) for name in basis.record.stokes]
    expected = expected[:, axes] * np.asarray(channel_scale)[:, None] * column
    np.testing.assert_allclose(
        basis.predict(new).stokes, expected, rtol=3e-13, atol=1e-28
    )
    assert new.values[0] == pytest.approx(column)
    assert basis.predict(new).unit == "cgs"
    assert basis.ledger.unbounded() == ("legacy_error",)


def test_import_benchmark_layout_and_band_area():
    old = _frozen()
    basis = from_single_reference(old)
    assert basis.lattice.cells.n_cells == 1
    assert basis.lattice.depth_in_total
    assert (
        basis.columns.n_rows,
        basis.columns.n,
        basis.columns.n0,
        basis.columns.n2,
    ) == (2430, 1306, 486, 410)
    np.testing.assert_array_equal(basis.lattice.cells.reference[0], [300.0, 5e-6, 8.0])
    np.testing.assert_array_equal(basis.lattice.cells.half_width[0], [80.0, 2e-6, 3.0])
    assert basis.channels.normalisation == "unit_integral"
    assert old.channels.normalisation == "unit_peak"
    _compare(old, basis, [1 / 2e7, 1 / 8e7])


@pytest.mark.parametrize("components,parity", [(("I", "Q"), False), (("I", "V"), True)])
def test_import_separate_depth_cap_and_stokes(components, parity):
    old = _frozen(
        channels=Channels.raised_cosine([1e8], [1e7], normalisation="unit_integral"),
        truncation=Truncation(1, 2, 3, 1, max_orders=(1, 2, 1)),
        parity=parity,
        components=components,
    )
    basis = from_single_reference(old)
    assert not basis.lattice.depth_in_total
    _compare(old, basis, [1.0])


def test_existing_channel_rows_and_lower_order_use_stored_columns():
    old = _frozen()
    basis = from_single_reference(old)
    select = basis.at(basis.channels[[1, 0, 1]])
    np.testing.assert_array_equal(
        select.design(), np.asarray(basis.design())[[1, 0, 1]]
    )
    lower = basis.with_order(0)
    reduced = lower.from_legacy_moments(_moments(old), column=1)
    full = basis.from_legacy_moments(_moments(old), column=1)
    nonconstant = np.any(basis.lattice.index.arrays.powers, axis=1)
    constant = Moments(basis.lattice, np.where(nonconstant, 0.0, full.values))
    np.testing.assert_allclose(
        lower.predict(reduced).stokes, basis.predict(constant).stokes, rtol=1e-14
    )
    with pytest.raises(RoleError, match="new channel rows"):
        basis.at(basis.channels.__class__.at("900 MHz"))
    with pytest.raises(RoleError, match="extend the Taylor order"):
        basis.with_order(3)
    with pytest.raises(RoleError, match="integrate a population directly"):
        basis.direct(None)
    with pytest.raises(RoleError, match="explicit|from_legacy_moments"):
        basis.predict(_moments(old))


def test_import_refuses_inconsistent_metadata():
    old = _frozen()
    with pytest.raises(UnitError, match="affine coordinate units"):
        from_single_reference(
            dataclasses.replace(
                old, provenance=dataclasses.replace(old.provenance, units="")
            )
        )
    with pytest.raises(UnitError, match="same channel normalisation"):
        from_single_reference(
            dataclasses.replace(
                old, provenance=dataclasses.replace(old.provenance, channels=())
            )
        )
    arbitrary = Channels.from_callable(
        jnp.ones_like, (1e8, 2e8), smoothness=0, normalisation="unit_integral"
    )
    with pytest.raises(UnitError, match="without a band area"):
        from_single_reference(_frozen(channels=arbitrary))
    with pytest.raises(RoleError, match="contain its reference"):
        from_single_reference(_frozen(depth=(0, 1)))
    with pytest.raises(RoleError, match="reference and scales"):
        moments = dataclasses.replace(_moments(old), reference=Reference(300, 5e-6))
        from_single_reference(old).from_legacy_moments(moments, column=1.0)


@pytest.mark.parametrize(
    "normalisation, reference_areas",
    [
        ("unit_peak", [2e7, 8e7]),
        ("unit_integral", [1.0, 1.0]),
    ],
)
def test_import_composed_channels_normalises_the_combined_profile(
    normalisation, reference_areas
):
    ref = Channels.tophat([1e8, 2e8], [1e7, 4e7], n_nu=8, normalisation=normalisation)
    coeff = np.array([[0.3, 0.7], [0.4, 0.9]])
    old = _frozen(channels=Channels.from_basis(ref, coeff))
    basis = from_single_reference(old)
    factors = 1.0 / (coeff @ reference_areas)
    np.testing.assert_allclose(np.sum(basis.channels.coefficients, axis=1), 1.0)
    frequencies = np.linspace(9e7, 2.4e8, 21)
    np.testing.assert_allclose(
        basis.channels(frequencies), old.channels(frequencies) * factors[:, None]
    )
    _compare(old, basis, factors)
    invalid = Channels.from_basis(ref, [-1.0, 0.0])
    with pytest.raises(UnitError, match="composed channel areas"):
        from_single_reference(_frozen(channels=invalid))


def test_degenerate_depth_support_with_zero_retained_power():
    old = _frozen(truncation=Truncation(1, 2, 2, 0), depth=(8, 8))
    basis = from_single_reference(old)
    assert basis.lattice.cells.variables == ("gamma", "B")
    _compare(old, basis, [1 / 2e7, 1 / 8e7])
    with pytest.raises(RoleError, match="legacy depth support"):
        from_single_reference(_frozen(depth=(8, 8)))


def test_small_native_harmonic_source():
    nu = 139962.44936166518
    depth = 0.5 * (nu / 299792458.0) ** 2
    source = build_basis(
        HarmonicKernel(10, n_outer=8, n_inner=8, n_mu=12, n_eta=12),
        Channels.bump([2 * nu], [1.3 * nu], n_nu=8),
        Truncation(1, 1, 0),
        Reference(20, 1.0, depth_ref=2 * depth, scales=(20, 1.0, depth)),
        support=Support((19.5, 20.5), (0.98, 1.02), (0, 5 * depth)),
        convergence=False,
    )
    converted = from_single_reference(source)
    # Divide by the area of the original unit-peak band profile.
    scale = source.channels.__class__.bump(
        [2 * nu], [1.3 * nu], n_nu=8, normalisation="unit_integral"
    ).peak_scale()
    _compare(source, converted, scale)


def test_public_native_harmonic_prediction_about_samples():
    import syncem as se

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
    got = se.predict(samples, channels, lattice=lattice, kernel=kernel)
    assert got.stokes.shape == (1, 4)
    assert np.all(np.isfinite(got.stokes))
    assert got.ledger["representation"].kind == "measured"


@pytest.mark.parametrize("scale", [0.0, 1.0, 3.0])
def test_harmonic_prediction_carries_the_population_scale(scale):
    """Moments, absolute Stokes and the direct reference scale with the column."""
    import syncem as se

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
    basis = se.spectral_basis(
        samples, channels, lattice=lattice, kernel=se.HarmonicKernel(m_max=10)
    )
    unit = se.predict(se.Population(samples), channels, basis=basis)
    got = se.predict(scale * se.Population(samples), channels, basis=basis)
    np.testing.assert_allclose(
        np.asarray(got.moments.values), scale * np.asarray(unit.moments.values)
    )
    np.testing.assert_allclose(got.stokes, scale * unit.stokes, rtol=1e-12, atol=0)
    np.testing.assert_allclose(
        got.ledger["representation"].value,
        scale * np.asarray(unit.ledger["representation"].value),
        rtol=1e-12,
        atol=0,
    )
    # Pinned unit-column intensity of this two-atom source (single-reference bridge).
    np.testing.assert_allclose(unit.stokes[0, 0], 1.485425677565482e-24, rtol=1e-10)


def test_harmonic_basis_refuses_a_different_line_transmission_on_equal_rows():
    """Equal quadrature rows with different line transmission are a different response."""
    import syncem as se

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
    grid = nu * np.array([1.0, 1.5, 2.0, 2.5, 3.0])

    def band(values):
        return se.Channels.from_table(
            grid, values, smoothness=1, n_nu=1, normalisation="unit_integral"
        )

    first, second = band([0, 0.8, 1, 0.2, 0]), band([0, 0.2, 1, 0.8, 0])
    assert first.row_keys() == second.row_keys()
    lattice = se.Lattice.single(about=samples, order=0, degree=1)
    kernel = se.HarmonicKernel(m_max=10)
    own = se.spectral_basis(samples, second, lattice=lattice, kernel=kernel)
    reused = se.spectral_basis(samples, first, lattice=lattice, kernel=kernel)
    got = se.predict(samples, second, basis=own)
    # Reusing the first band's rows gave 1.4463052112780868e-24 before the fix.
    np.testing.assert_allclose(got.stokes[0, 0], 1.5731204948579984e-24, rtol=1e-10)
    assert got.channels.row_keys() == second.row_keys()
    np.testing.assert_allclose(
        se.predict(samples, first, basis=reused).stokes[0, 0],
        1.4463052112780868e-24,
        rtol=1e-10,
    )
    with pytest.raises(RoleError, match="line transmission"):
        se.predict(samples, second, basis=reused)
    with pytest.raises(RoleError, match="line transmission"):
        reused.at(second)


def test_stored_harmonic_rows_follow_line_transmission_in_memory_and_on_disk(tmp_path):
    import syncem as se

    grid = np.array([1, 1.5, 2, 2.5, 3]) * 139962.44936166518
    channels = se.Channels.from_table(
        grid,
        [[0, 0.8, 1, 0.2, 0], [0, 0.2, 1, 0.8, 0]],
        smoothness=1,
        n_nu=1,
        normalisation="unit_integral",
    )
    assert channels.row_keys()[0] == channels.row_keys()[1]
    old = _frozen(channels=channels)
    basis = from_single_reference(old)
    moments = basis.from_legacy_moments(_moments(old), column=1.0)
    expected = basis.predict(moments).stokes
    assert not np.allclose(expected[0], expected[1], atol=0)
    path = tmp_path / "lines.npz"
    basis.save(path)
    for stored in (basis, se.SpectralBasis.load(path)):
        rows = np.array([1, 0, 1])
        selected = se.predict(moments, channels[rows], basis=stored, check=False)
        np.testing.assert_array_equal(selected.stokes, np.asarray(expected)[rows])
        np.testing.assert_array_equal(selected.channels(grid), channels(grid)[rows])


def test_sum_requires_the_line_transmission_of_every_harmonic_part(tmp_path):
    import syncem as se

    grid = 139962.44936166518 * np.array([1, 1.5, 2, 2.5, 3])
    ch = se.Channels.from_table(
        grid,
        [[0, 0.8, 1, 0.2, 0], [0, 0.2, 1, 0.8, 0]],
        smoothness=1,
        n_nu=1,
        normalisation="unit_integral",
    )
    old = _frozen(channels=ch)
    all_rows = from_single_reference(old)
    path = tmp_path / "two-line-bands.npz"
    all_rows.save(path)
    m = all_rows.from_legacy_moments(_moments(old), column=1)
    for basis in (all_rows, se.SpectralBasis.load(path)):
        a, b = basis.at(ch[0]), basis.at(ch[1])
        for first, second in ((a, b), (b, a)):
            with pytest.raises(ValueError, match="channels"):
                se.BasisSum((first, second), ("a", "b"))
            equal = se.BasisSum((first, first), ("a", "b"))
            np.testing.assert_array_equal(
                equal.predict((m, m)), 2 * first.predict(m).stokes
            )
            with pytest.raises(RoleError, match="line transmission"):
                equal.at(second.channels)

        # A plain quadrature basis and a harmonic basis may share the stored row.
        plain = se.SpectralBasis(
            **{
                field.name: getattr(a, field.name)
                for field in dataclasses.fields(se.SpectralBasis)
            }
        )
        mixed = se.BasisSum((plain, a), ("quadrature", "line"))
        with pytest.raises(RoleError, match="line transmission"):
            se.predict(
                se.Population(gamma=se.Delta(20)), b.channels, basis=mixed, check=False
            )


def test_live_harmonic_sum_predicts_each_component_on_requested_rows():
    import syncem as se

    samples = se.Samples(
        gamma=[19.5, 20.5],
        B=([0.98, 1.02], "G"),
        mu=[-0.05, 0.05],
        eta=[-0.05, 0.05],
        phi=([0.05, 0.1], "rad"),
        depth=[1e-8, 2e-8],
        column="2 cm^-2",
        weights=[0.21, 0.79],
    )
    grid = np.array([1, 1.5, 2, 2.5, 3]) * 139962.44936166518
    channels = se.Channels.from_table(
        grid,
        [[0, 0.8, 1, 0.2, 0], [0, 0.2, 1, 0.8, 0]],
        smoothness=1,
        n_nu=1,
        normalisation="unit_integral",
    )
    basis = se.spectral_basis(
        samples,
        channels,
        lattice=se.Lattice.single(about=samples, order=0, degree=1),
        kernel=se.HarmonicKernel(m_max=10),
    )
    source = se.Population(samples)
    unit = se.predict(source, channels, basis=basis)
    summed = se.BasisSum((basis, basis), ("first", "second"))
    rows = np.array([1, 0, 1])
    got = se.predict(3 * source + 0.5 * source, channels[rows], basis=summed)
    np.testing.assert_allclose(
        got.stokes, 3.5 * np.asarray(unit.stokes)[rows], atol=0, rtol=1e-14
    )
    np.testing.assert_allclose(got.moments[0].values, 3 * unit.moments.values, atol=0)
    np.testing.assert_allclose(got.moments[1].values, 0.5 * unit.moments.values, atol=0)
    delta = np.asarray(unit.ledger["representation"].value)[rows]
    np.testing.assert_allclose(
        got.ledger["representation"].value, 3.5 * delta, atol=0, rtol=1e-11
    )
