"""Forward entry points, panel partition and kernel-declared lattice metadata."""

import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se
import syncem.base.window as reference_window


@se.kernel(variables=[se.Variable("x", kind="positive")], stokes=("I",))
def polynomial(nu, x):
    return {"I": (1 + x**2) * jnp.ones_like(nu)}


def _lattice(weights=None):
    rows = se.Lattice(x=se.Cells.linear(0, 3, 3, order=2), weights=weights)
    return rows.select(x=(1, 2))


def test_reference_window_bounds_the_bessel_input_slab(monkeypatch):
    seen = []

    def mock_F(x):
        seen.append(x.size)
        return jnp.exp(-x)

    monkeypatch.setattr(reference_window, "_REFERENCE_BLOCK", 64)
    monkeypatch.setattr(reference_window, "F", mock_F)
    y = np.geomspace(1e-4, 10.0, 9)
    sine = np.linspace(0.2, 1.0, 145)
    mass = np.full(sine.size, 1.0 / sine.size)
    got = reference_window._average_F(y, sine, mass)
    truth = np.exp(-y[:, None] / sine) @ (mass * sine)
    np.testing.assert_allclose(got, truth, rtol=1e-14)
    assert len(seen) > 1 and max(seen) <= 64


def test_independent_box_partition_and_direct_truth():
    pop = se.Population(x=se.Uniform(0, 3))
    basis = se.spectral_basis(
        pop, ([1.0, 2.0], "GHz"), lattice=_lattice(), kernel=polynomial
    )
    result = basis.predict(pop)
    within = basis.direct(pop, within="cells")
    truth = basis.direct(pop)
    np.testing.assert_allclose(within.I, (1 + (2**3 - 1) / 3) / 3, rtol=1e-13)
    np.testing.assert_allclose(truth.I, 4, rtol=1e-13)
    np.testing.assert_allclose(result.I, within.I, rtol=1e-13)
    rep = result.ledger["representation"].value
    outside = result.ledger["outside_lattice"].value
    np.testing.assert_allclose(rep + outside, result.stokes - truth.stokes, rtol=1e-13)
    assert np.all(np.asarray(outside) < 0)


def test_point_on_shared_cell_face_counts_once():
    pop = se.Population(x=se.Delta(1.0))
    lat = se.Lattice(x=se.Cells.linear(0, 3, 3, order=2))
    basis = se.spectral_basis(pop, ([1.0], "GHz"), lattice=lat, kernel=polynomial)
    result = basis.predict(pop)
    np.testing.assert_allclose(result.I, [2])
    np.testing.assert_allclose(basis.direct(pop, within="cells").I, [2])
    np.testing.assert_allclose(result.ledger["outside_lattice"].value, 0)


def test_convergence_compares_on_an_explicit_lattice():
    pop = se.Population(x=se.Uniform(0, 3))
    lattice = se.Lattice(x=se.Cells.linear(0, 3, 3, order=1))
    result = se.convergence(
        pop, ([1, 2], "GHz"), lattice=lattice, order=2, kernel=polynomial
    )
    np.testing.assert_allclose(result.difference, 1 / 12, rtol=1e-13)
    assert "frequency [Hz]" in result.table()


def test_default_free_lattice_covers_prediction_band():
    low = se.Channels.at([1.0, 2.0], unit="GHz")
    high = se.Channels.at([300.0], unit="GHz")
    pop = se.Population(se.FreeMoments("gamma"), B="5 uG")
    plan = se.plan(pop, low, predict=high)
    base = se.Lattice.default(pop, low)
    assert plan.lattice.cells.hi.max() > 10 * base.cells.hi.max()


def test_reference_tail_at_300_ghz_is_an_estimate_in_reference_units():
    pop = se.paper.e1()
    fit_band = se.Channels.log(1, 100.6, 4, unit="GHz", half_width=0.2)
    high = se.Channels.at([300.0], unit="GHz", half_width=0.2)
    lat = se.Lattice(gamma=se.Cells.log(2455, 1.502e6, 14))
    basis = se.spectral_basis(pop, fit_band, lattice=lat, check=False)
    reference = basis.at(high).ledger["outside_lattice_reference"]
    assert reference.kind == "estimate"
    np.testing.assert_allclose(reference.value, 3.6e-3, rtol=0.2)
    assert "reference power-law" in reference.relative_to
    high_basis = basis.at(high)
    prediction = high_basis.predict(pop)
    direct = high_basis.direct(pop)
    error = (
        prediction.ledger["representation"].value
        + prediction.ledger["outside_lattice"].value
    )
    np.testing.assert_allclose(
        error / direct.I[:, None],
        (prediction.stokes - direct.stokes) / direct.I[:, None],
        atol=1e-11,
    )


def test_direct_ray_agrees_with_separate_emitting_steps():
    ch = se.Channels.at([1.0, 2.0], unit="GHz")
    law = se.PowerLaw(2.6, lo=300, hi=1e5, column=1)
    ray = se.Ray(
        step=([1.0, 2.0], "pc"),
        B_vector=([[3, 4, 2], [1, 3, -2]], "uG"),
        thermal_density=([1.0, 2.0], "cm^-3"),
        electron_density=([3.0, 2.0], "cm^-3"),
        electrons=law,
    )
    lattice = se.Lattice(gamma=se.Cells.log(300, 1e5, 4, order=1))
    expected = []
    for step in ray.iter_steps():
        component = step.column * se.Population(gamma=law, **step.factors)
        basis = se.spectral_basis(component, ch, lattice=lattice, check=False)
        expected.append(basis.direct(component).stokes)
    result = se.predict(ray, ch)
    assert result.moments is None
    np.testing.assert_allclose(result.stokes, sum(expected), rtol=2e-10)
    with pytest.raises(se.RoleError, match="Ray"):
        se.predict(ray, ch, lattice=lattice)


def test_depth_gate_uses_band_edges_and_preserves_dimensionless_bound():
    @se.kernel(
        variables=[se.Variable("x"), se.Variable("depth", unit="rad m^-2")],
        depth="depth",
        stokes=("I", "P"),
    )
    def model(nu, x):
        return {"I": x + 0 * nu, "P": 0.5 * x + 0 * nu}

    channels = se.Channels.at([1], unit="GHz", half_width=0.2)
    pop = se.Population(x=se.Uniform(1, 2), depth=se.Uniform(-2, 2))
    with pytest.raises(se.DepthExpansionError, match="depth_tol"):
        se.spectral_basis(
            pop,
            channels,
            kernel=model,
            lattice=se.Lattice(
                x=se.Cells.linear(1, 2, 1), depth=se.Cells.linear(-2, 2, 1)
            ),
        )
    plan = se.plan(pop, channels, kernel=model, resolve=("depth",))
    assert plan.lattice.axes["depth"].n == 4
    basis = se.spectral_basis(pop, channels, kernel=model, lattice=plan.lattice)
    result = basis.predict(pop, check=False)
    term = result.ledger["depth_truncation"]
    tau = 2 * (299792458 / 8e8) ** 2
    np.testing.assert_allclose(term.value, (tau * 0.5) ** 3 / 6, rtol=1e-10)
    np.testing.assert_array_equal(
        result.to("Jy/sr").ledger["depth_truncation"].value, term.value
    )
    with pytest.raises(se.DepthExpansionError, match="depth_tol"):
        basis.at(se.Channels.at([0.1], unit="GHz"))


def test_reference_density_and_nonstandard_depth_resolve_before_pairing():
    variables = [se.Variable("x", kind="positive"), se.Variable("z")]

    @se.kernel(
        variables=variables,
        stokes=("I",),
        depth="z",
        capabilities=(se.ReferenceDensity("x", lambda x: x),),
    )
    def model(nu, x):
        return {"I": x**2 + 0 * nu}

    pop = se.Population(x=se.Uniform(1, 3), z=se.Uniform(-1, 1))
    lat = se.Lattice(x=se.Cells.linear(1, 3, 2), z=se.Cells.linear(-1, 1, 2), order=2)
    basis = se.spectral_basis(pop, ([1.0], "GHz"), lattice=lat, kernel=model)
    assert basis.lattice.cells.excluded == ("z",)
    assert basis.lattice.key != lat.key
    np.testing.assert_allclose(basis.lattice.weights, [3 / 8, 3 / 8, 5 / 8, 5 / 8])
    result = basis.predict(pop)
    np.testing.assert_allclose(result.I, [13 / 3], rtol=1e-13)
    custom = se.Lattice(x=lat.axes["x"], z=lat.axes["z"], order=2, weights=[1, 2, 3, 4])
    weighted = se.spectral_basis(pop, ([1.0], "GHz"), lattice=custom, kernel=model)
    np.testing.assert_array_equal(weighted.lattice.weights, custom.weights)
    np.testing.assert_allclose(weighted.predict(pop, check=False).I, result.I)
    with pytest.raises(ValueError, match="lattice key"):
        weighted.predict(result.moments, check=False)


def test_continuum_powerlaw_and_phase_smoke():
    ch = se.Channels.at([1.0, 2.0], unit="GHz", half_width=0.2, n_nu=8)
    lat = se.Lattice(gamma=se.Cells.log(300.0, 1e5, 8, order=2))
    pop = se.Population(
        gamma=se.PowerLaw(2.6, lo=300.0, hi=1e5, amplitude=1),
        B="5 uG",
        depth=se.GaussianScreen(8.0, 0.5),
        phi=se.AzimuthFactor(0.4),
    )
    basis = se.spectral_basis(pop, ch, lattice=lat)
    measured = basis.predict(pop)
    truth = basis.direct(pop)
    assert np.all(np.asarray(truth.I) > 0)
    assert np.any(np.asarray(truth.U) != 0)
    np.testing.assert_allclose(measured.I, truth.I, rtol=2e-2)
    np.testing.assert_allclose(
        measured.ledger["representation"].value
        + measured.ledger["outside_lattice"].value,
        measured.stokes - truth.stokes,
        rtol=1e-10,
        atol=1e-39,
    )
    other = pop.replace(B="8 uG")
    with pytest.raises(se.SpecifiedMismatch):
        basis.predict(other)


def test_conditionals_place_moving_point_and_refuse_bare_field():
    @se.kernel(
        variables=[
            se.Variable("x", kind="positive"),
            se.Variable("B", kind="positive", unit="G", bare=False),
        ],
        stokes=("I",),
    )
    def field_kernel(nu, x, B):
        return {"I": (x + B * 1e6) * jnp.ones_like(nu)}

    lat = se.Lattice(x=se.Cells.linear(1.0, 2.0, 1, order=2))
    channels = ([1.0], "GHz")
    valid = se.Population(x=se.Uniform(1, 2), B=lambda x: (2e-6 * x, "G"))
    out = se.predict(valid, channels, lattice=lat, kernel=field_kernel)
    np.testing.assert_allclose(out.I, [4.5], rtol=1e-13)
    invalid = valid.replace(B=lambda x: 2e-6 * x)
    with pytest.raises(se.UnitError, match="no unit"):
        se.predict(invalid, channels, lattice=lat, kernel=field_kernel)


def test_shifted_gaussian_depth_uses_the_same_faces_in_its_partition():
    @se.kernel(
        variables=[
            se.Variable("x", kind="positive"),
            se.Variable("rm", unit="rad m^-2"),
        ],
        stokes=("I", "P"),
        depth="rm",
    )
    def source(nu, x):
        i = x * jnp.ones_like(nu)
        return {"I": i, "P": i}

    ch = ([1.0], "GHz")
    lat = se.Lattice(
        x=se.Cells.linear(1, 2, 1, order=1),
        rm=se.Cells.linear(99.5, 100.5, 2, unit="rad m^-2", order=2),
    )
    pop = se.Population(x=se.Uniform(1, 2), rm=se.GaussianScreen(100, 0.5))
    basis = se.spectral_basis(pop, ch, lattice=lat, kernel=source)
    got = basis.predict(pop)
    truth = basis.direct(pop)
    np.testing.assert_allclose(truth.I, [1.5], rtol=2e-13)
    np.testing.assert_allclose(
        got.ledger["representation"].value + got.ledger["outside_lattice"].value,
        got.stokes - truth.stokes,
        rtol=1e-12,
        atol=5e-14,
    )


def test_maxwellian_default_lattice_integrates_over_positive_field_nodes():
    from syncem.front._setup import default_lattice

    pop = se.Population(gamma=se.PowerLaw(2.6), B=se.Maxwellian("5 uG"))
    ch = se.Channels.at([1.0, 10.0], unit="GHz")
    lattice = default_lattice(pop, se.ContinuumKernel(), ch)
    assert 300 <= lattice.cells.lo[0, 0] < lattice.cells.hi[-1, 0] < 1e8
    assert lattice.n_cells > 1


def test_joint_samples_keep_their_correlation_on_an_arbitrary_lattice():
    @se.kernel(
        variables=[se.Variable("a", kind="positive"), se.Variable("b")],
        stokes=("I",),
    )
    def correlated(nu, a, b):
        return {"I": a * b * jnp.ones_like(nu)}

    samples = se.Samples(a=[1, 2], b=[1, 2], weights=[1, 1], column="3 cm^-2")
    pop = se.Population(samples)
    lat = se.Lattice(
        a=se.Cells.linear(1, 2, 1, order=2),
        b=se.Cells.linear(1, 2, 1, order=2),
        order=2,
    )
    basis = se.spectral_basis(pop, ([1.0], "GHz"), lattice=lat, kernel=correlated)
    assert basis.predict(pop).I.item() == pytest.approx(7.5)
    assert basis.direct(pop).I.item() == pytest.approx(7.5)
    with pytest.raises(se.RoleError, match="joint block.*resolved and averaged"):
        se.spectral_basis(
            pop, ([1.0], "GHz"), lattice=se.Lattice(a=lat.axes["a"]), kernel=correlated
        )


def test_requested_provider_check_and_cost_budget():
    pop = se.Population(x=se.Uniform(0, 3))
    check = se.AccuracyCheck(max_seconds=0, providers=True)
    basis = se.spectral_basis(
        pop, ([1.0], "GHz"), lattice=_lattice(), kernel=polynomial, check=check
    )
    assert dict(basis.record.agreement)["reference_table"] < 1e-10
    result = basis.predict(pop, check=check)
    assert result.ledger["representation"].kind == "unbounded"
    assert "Estimated direct cost" in result.ledger["representation"].note


def test_azimuth_conditional_inside_field_average():
    @se.kernel(
        variables=[
            se.Variable("a"),
            se.Variable("b"),
            se.Variable("az", kind="angle", unit="rad"),
        ],
        stokes=("I", "P"),
        azimuth="az",
    )
    def amplitude(nu, a, b):
        v = (a + b) * jnp.ones_like(nu)
        return {"I": v, "P": v}

    pop = se.Population(
        a=se.Uniform(0, 2),
        b=se.Uniform(1, 2),
        az=lambda b: se.AzimuthFactor(0.2 * b),
    )
    lat = se.Lattice(a=se.Cells.linear(0, 2, 2, order=1))
    basis = se.spectral_basis(pop, ([1.0], "GHz"), lattice=lat, kernel=amplitude)
    expected = [2.5, 0.2 * (1.5 + 7 / 3), 0.0]
    np.testing.assert_allclose(basis.predict(pop).stokes[0], expected, atol=1e-14)
    np.testing.assert_allclose(basis.direct(pop).stokes[0], expected, atol=1e-14)


def test_beaming_density_is_evaluated_on_the_partner_nodes():
    @se.kernel(
        variables=[
            se.Variable("a"),
            se.Variable("pitch", kind="cosine"),
            se.Variable("view", kind="cosine"),
        ],
        stokes=("I",),
        capabilities=(se.Beaming("pitch", "view"),),
    )
    def amplitude(nu, a, pitch, view):
        return {"I": (a + pitch * view) * jnp.ones_like(nu)}

    pop = se.Population(
        a=se.Uniform(0, 2),
        pitch=se.Uniform(-0.5, 0.5),
        view=se.Uniform(-0.5, 0.5),
    )
    lat = se.Lattice(a=se.Cells.linear(0, 2, 2, order=1))
    basis = se.spectral_basis(pop, ([1.0], "GHz"), lattice=lat, kernel=amplitude)
    expected = [2 * (1 + 1 / 12)]
    np.testing.assert_allclose(basis.predict(pop).I, expected, rtol=1e-13)
    np.testing.assert_allclose(basis.direct(pop).I, expected, rtol=1e-13)


def test_public_plan_and_free_moment_fit_on_user_lattice():
    @se.kernel(variables=[se.Variable("x", kind="positive")], stokes=("I",))
    def model(nu, x):
        return {"I": 1 + x * nu / 1e9}

    ch = se.Channels.at([1.0, 2.0, 3.0], unit="GHz")
    lat = se.Lattice(x=se.Cells.linear(1, 2, 1, order=1))
    unknown = se.Population(se.FreeMoments("x"))
    compiled = se.plan(unknown, ch, lattice=lat, kernel=model)
    assert compiled.kind == "linear"
    assert compiled.record.to_dict()["roles"]["x"]["role"] == "resolved"
    data = se.StokesData(ch, I=[4.0, 5.5, 7.0], unit="cgs", sigma=0.1)
    result = se.fit(
        unknown, data, lattice=lat, kernel=model, rank=se.FullRank(), check=False
    )
    assert result.moments.origin == "minimum_norm"
    assert result.rank == 2
    np.testing.assert_allclose(result.predict().I, data.stokes[:, 0], rtol=1e-13)
