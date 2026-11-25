"""Analytic integrals on a conditional beaming diagonal."""

import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se

_PITCH = se.Variable("pitch", kind="cosine")
_LOOK = se.Variable("look", kind="cosine")
_BEAMING = (se.Beaming("pitch", "look"),)
_CHANNELS = se.Channels.at([1.0], unit="GHz")


@se.kernel(variables=(_PITCH, _LOOK), capabilities=_BEAMING, stokes=("I",))
def _polynomial(nu, pitch, look):
    return {"I": 2 + look + look**2 / 2 + 0 * nu}


@se.kernel(
    variables=(se.Variable("x"), _PITCH, _LOOK), capabilities=_BEAMING, stokes=("I",)
)
def _constant(nu, x, pitch, look):
    return {"I": jnp.ones_like(nu)}


@pytest.mark.parametrize("slope", [-0.25, 0.25])
@pytest.mark.parametrize("width", [0.002, 0.32])
def test_intersections_of_each_conditional_edge_on_the_diagonal(slope, width):
    centre = 0.24

    def conditional(look):
        shift = centre + slope * look
        return se.Uniform(shift - width / 2, shift + width / 2)

    pop = se.Population(look=se.Isotropic(), pitch=se.Conditional(conditional))
    # On two resolved angles the direct reference must use the actual law.
    # With both angles averaged the forward columns integrate the same density.
    left, right = (centre + np.array([-0.5, 0.5]) * width) / (1 - slope)
    average = 2 + (left + right) / 2 + (left**2 + left * right + right**2) / 6
    expected = average / (1 - slope)
    for lattice in (
        se.Lattice(),
        se.Lattice(look=se.Legendre(3), pitch=se.Legendre(3)),
    ):
        basis = se.spectral_basis(
            pop, _CHANNELS, kernel=_polynomial, lattice=lattice, check=False
        )
        direct = basis.direct(pop)
        assert direct.ledger["quadrature:beaming_crossings"].kind == "unbounded"
        np.testing.assert_allclose(direct.I, expected, rtol=1e-12)
        if not lattice.representations:
            np.testing.assert_allclose(
                basis.predict(pop, check=False).I, expected, rtol=1e-12
            )


def test_moving_support_contributes_to_the_jet_and_to_direct():
    def conditional(x):
        width = 0.5 + 0.2 * x.value
        return se.Uniform(-width, width)

    source = se.Population(
        x=0.0, pitch=se.Conditional(conditional), look=se.Isotropic()
    )
    lattice = se.Lattice(x=se.Cells.linear(-1, 1, 1, order=2))
    basis = se.spectral_basis(
        source, _CHANNELS, kernel=_constant, lattice=lattice, check=False
    )
    # Every angular integral is the unit mass of the conditional for all x.
    np.testing.assert_allclose(
        basis.design().ravel(), [1.0, 0.0, 0.0], rtol=0, atol=2e-14
    )
    for x in [-0.9, 0.0, 0.9]:
        pop = source.replace(x=se.Delta(x))
        np.testing.assert_allclose(basis.direct(pop).I, 1.0, atol=2e-14, rtol=0)
        np.testing.assert_allclose(
            basis.predict(pop, check=False).I, 1.0, atol=2e-14, rtol=0
        )


def test_equivalent_angular_factorizations_and_true_cycle():
    def pitch(eta):
        slope = 0.8 * eta.value
        return se.Tabulated([-1.0, 1.0], [1.0 - slope, 1.0 + slope])

    def view(mu):
        slope = 0.8 * mu.value
        return se.Tabulated([-1.0, 1.0], [1.0 - slope, 1.0 + slope])

    common = dict(gamma=3000.0, B="5 uG")
    lat = se.Lattice(gamma=se.Cells.log(2999, 3001, 1, order=0))
    channels = se.Channels.at([0.1], unit="GHz")
    directions = [
        dict(mu=se.Conditional(pitch), eta=se.Isotropic(nodes=64)),
        dict(eta=se.Conditional(view), mu=se.Isotropic(nodes=64)),
    ]
    values = []
    for angles in directions:
        pop = se.Population(**common, **angles)
        basis = se.spectral_basis(pop, channels, lattice=lat, check=False)
        values.append(
            [float(basis.direct(pop).I[0]), float(basis.predict(pop, check=False).I[0])]
        )
    np.testing.assert_allclose(values[0], values[1], rtol=1e-13, atol=0)
    # Independent quadrature of the Bessel kernel with 1+0.8 eta^2.
    np.testing.assert_allclose(values[0], 6.917404771101726e-29, rtol=1e-7, atol=0)
    with pytest.raises(se.RoleError, match="cyclic"):
        se.spectral_basis(
            se.Population(**common, mu=se.Conditional(pitch), eta=se.Conditional(view)),
            channels,
            lattice=lat,
        )


@pytest.mark.parametrize("project_driver", [False, True])
def test_joint_dependence_of_moving_boundaries_and_their_derivatives(project_driver):
    # The two diagonal intersections are (.17 + .08 x + [.0,.002]) / .75.
    # After folding the integral of the constant kernel is exactly 1/.75.
    def pitch(x, look):
        base = 0.17 + 0.08 * x.value + 0.25 * look.value
        return se.Uniform(base, base + 0.002)

    source = se.Population(x=0.0, look=se.Isotropic(), pitch=se.Conditional(pitch))
    axes = {"x": se.Cells.linear(-1, 1, 1, order=1)}
    if project_driver:
        axes["look"] = se.Legendre(2)
    basis = se.spectral_basis(
        source, _CHANNELS, kernel=_constant, lattice=se.Lattice(**axes), check=False
    )
    for x in [-0.7, 0.7]:
        p = source.replace(x=se.Delta(x))
        np.testing.assert_allclose(
            basis.predict(p, check=False).I, 1 / 0.75, rtol=1e-12
        )
        np.testing.assert_allclose(basis.direct(p).I, 1 / 0.75, rtol=1e-12)


def test_boundary_collinear_with_the_beaming_diagonal():
    # The lower edge is the diagonal throughout the interior of the driver.
    # Only the changing density 1/w(x) contributes to the derivative.
    def pitch(x, look):
        return se.Uniform(look.value, look.value + 0.2 + 0.1 * x.value)

    source = se.Population(
        x=0.0, look=se.Uniform(-0.25, 0.25), pitch=se.Conditional(pitch)
    )
    lattice = se.Lattice(x=se.Cells.linear(-1, 1, 1, order=1))
    basis = se.spectral_basis(
        source, _CHANNELS, kernel=_constant, lattice=lattice, check=False
    )
    np.testing.assert_allclose(basis.design().ravel(), [10.0, -5.0], rtol=5e-13)


def test_host_panels_separate_nearby_roots_of_one_boundary():
    from syncem.population._diagonal import host_diagonal_knots

    centre, half = 0.27843, 0.0001

    def law(look):
        z = look.value
        difference = (z - centre) ** 2 - half**2
        return se.Uniform(z + difference, z + difference + 0.1)

    edges = host_diagonal_knots(
        se.Conditional(law), "look", {}, {"look": ""}, (-0.1, 0.5)
    )
    np.testing.assert_allclose(
        edges, [centre - half, centre + half], atol=1e-12, rtol=0
    )


def test_projected_driver_and_a_continuous_averaged_parent():
    def pitch(x):
        width = 0.001 * (1 + 0.4 * x.value)
        return se.Uniform(0.327 - width, 0.327 + width)

    source = se.Population(
        x=se.Uniform(-1.0, 1.0, nodes=8),
        look=se.Isotropic(),
        pitch=se.Conditional(pitch),
    )
    lattice = se.Lattice(look=se.Legendre(3))
    b = se.spectral_basis(
        source, _CHANNELS, kernel=_constant, lattice=lattice, check=False
    )
    np.testing.assert_allclose(b.predict(source, check=False).I, 1.0, rtol=1e-12)
    np.testing.assert_allclose(b.direct(source).I, 1.0, rtol=1e-12)


def test_traced_nearby_intersections_and_their_velocities():
    import jax
    from syncem.population._diagonal import traced_diagonal_knots

    half = 0.0001

    def law(x, look):
        q = look.value
        gap = (q - 0.27843 - 0.1 * x.value) ** 2 - half**2
        return se.Uniform(q + gap, q + gap + 0.1)

    declared = se.Conditional(law)

    def edges(x):
        return traced_diagonal_knots(
            declared, "look", {"x": x}, {"x": "", "look": ""}, -0.1, 0.5
        )

    roots, derivative = jax.jvp(edges, (0.0,), (1.0,))
    interior = (np.asarray(roots) > -0.1) & (np.asarray(roots) < 0.5)
    np.testing.assert_allclose(
        sorted(np.asarray(roots)[interior]),
        [0.27843 - half, 0.27843 + half],
        atol=2e-12,
        rtol=0,
    )
    np.testing.assert_allclose(np.asarray(derivative)[interior], 0.1, rtol=1e-10)


def test_moving_conditional_knots_enter_the_projected_legendre_coefficients():
    def pitch(x):
        peak = 0.15 + 0.2 * x.value
        return se.Tabulated(
            jnp.array([peak - 0.005, peak, peak + 0.005]), [0.0, 1.0, 0.0]
        )

    # The coefficients are the Legendre moments of this translated triangular
    # profile, with 2p multiplying the projected kernel.
    source = se.Population(x=0.0, pitch=se.Conditional(pitch), look=se.Isotropic())
    lattice = se.Lattice(x=se.Cells.linear(-1, 1, 1, order=1), look=se.Legendre(1))
    basis = se.spectral_basis(
        source, _CHANNELS, kernel=_constant, lattice=lattice, check=False
    )
    values = np.asarray(basis.averaged(_CHANNELS, {"x": 0.2}).I)
    np.testing.assert_allclose(
        values, [[1.0, 3 * (0.15 + 0.2 * 0.2)]], atol=5e-14, rtol=0
    )


def test_fixed_knots_keep_the_conditional_density_derivative_in_projection():
    import warnings

    import jax
    from scipy.integrate import quad

    knots = [-0.7, -0.2, 0.3, 0.7]

    def pitch(x):
        return se.Tabulated(knots, [0.0, 1.0 + x.value / 5, 0.8, 0.0])

    source = se.Population(x=0.0, pitch=se.Conditional(pitch), look=se.Isotropic())
    lattice = se.Lattice(x=se.Cells.linear(-1, 1, 1, order=1), look=se.Legendre(2))
    with warnings.catch_warnings():
        warnings.filterwarnings("error", message=".*JAX array is being set as static.*")
        basis = se.spectral_basis(
            source,
            _CHANNELS,
            kernel=_constant,
            lattice=lattice,
            check=False,
            cache=False,
        )
        values, derivative = jax.jvp(
            lambda x: basis.averaged(_CHANNELS, {"x": x}).I[0], (0.2,), (1.0,)
        )

    # Project the unnormalised piecewise-linear density, then differentiate
    # its normalisation. The density and its derivative have the same knots.
    def project(heights):
        moments = []
        for degree in range(3):
            polynomial = np.polynomial.legendre.Legendre.basis(degree)
            integral, _ = quad(
                lambda q: polynomial(q) * np.interp(q, knots, heights),
                knots[0],
                knots[-1],
                points=knots[1:-1],
            )
            moments.append((2 * degree + 1) * integral)
        return np.array(moments)

    raw = project([0.0, 1.04, 0.8, 0.0])
    slope = project([0.0, 0.2, 0.0, 0.0])
    expected = raw / raw[0]
    np.testing.assert_allclose(values, expected, rtol=0, atol=2e-13)
    np.testing.assert_allclose(
        derivative, (slope - expected * slope[0]) / raw[0], rtol=0, atol=2e-13
    )


@se.kernel(
    variables=(se.Variable("z"), se.Variable("x"), _PITCH, _LOOK),
    capabilities=_BEAMING,
    stokes=("I",),
)
def _chain_kernel(nu, z, x, pitch, look):
    return {"I": (1 + look + 0.5 * x) * jnp.ones_like(nu)}


def _law_at_offset(q, width):
    return se.Uniform(q - width, q + width, nodes=3) if width else se.Delta(q)


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("width", [0.0, 0.03])
def test_multistep_conditional_density_on_the_diagonal(reverse, width):
    def x_from_look(look):
        return _law_at_offset(0.25 * look.value, width)

    def x_from_pitch(pitch):
        return _law_at_offset(0.25 * pitch.value, width)

    def folded(x):
        return se.Uniform(0.239 + x.value, 0.241 + x.value)

    source, driver = ("look", "pitch") if reverse else ("pitch", "look")
    pop = se.Population(
        z=0.0,
        x=se.Conditional(x_from_pitch if reverse else x_from_look),
        **{driver: se.Isotropic(), source: se.Conditional(folded)},
    )
    # At fixed intermediate quadrature index the diagonal has slope 1/4.
    # The intermediate offsets and the uniform pitch offsets both have zero mean.
    expected = (1 + 1.125 * 0.24 / 0.75) / 0.75
    b = se.spectral_basis(
        pop,
        _CHANNELS,
        kernel=_chain_kernel,
        lattice=se.Lattice(),
        check=False,
        cache=False,
    )
    np.testing.assert_allclose(b.predict(pop, check=False).I, expected, rtol=2e-12)
    np.testing.assert_allclose(b.direct(pop).I, expected, rtol=2e-12)


def test_external_coordinate_moves_every_intermediate_node_in_jet():
    def x_given(look, z):
        return _law_at_offset(0.25 * look.value + 0.07 * z.value, 0.03)

    def pitch(x):
        return se.Uniform(0.239 + x.value, 0.241 + x.value)

    pop = se.Population(
        z=0.0,
        look=se.Isotropic(),
        x=se.Conditional(x_given),
        pitch=se.Conditional(pitch),
    )
    b = se.spectral_basis(
        pop,
        _CHANNELS,
        kernel=_chain_kernel,
        lattice=se.Lattice(z=se.Cells.linear(-1, 1, 1, order=1)),
        check=False,
        cache=False,
    )
    np.testing.assert_allclose(
        b.design().ravel(),
        [(1 + 1.125 * 0.24 / 0.75) / 0.75, 0.07 * (1.125 / 0.75 + 0.5) / 0.75],
        rtol=3e-12,
    )


def test_direct_alias_is_available_to_resolved_children_of_both_cosines():
    def x_from_pitch(pitch):
        return se.Delta(0.25 * pitch.value)

    def z_from_look(look):
        return se.Delta(0.125 * look.value)

    pop = se.Population(
        pitch=se.Isotropic(nodes=6),
        look=se.Isotropic(nodes=6),
        x=se.Conditional(x_from_pitch),
        z=se.Conditional(z_from_look),
    )
    lattice = se.Lattice(
        x=se.Cells.linear(-0.5, 0.5, 1, order=1),
        z=se.Cells.linear(-0.5, 0.5, 1, order=1),
        pitch=se.Legendre(1),
        look=se.Legendre(1),
    )
    b = se.spectral_basis(
        pop, _CHANNELS, lattice=lattice, kernel=_chain_kernel, check=False, cache=False
    )
    np.testing.assert_allclose(b.predict(pop, check=True).I, 1.0, atol=1e-13)
    np.testing.assert_allclose(b.direct(pop).I, 1.0, atol=1e-13)


def test_direct_panel_paths_use_the_actual_intermediate_integration_nodes():
    from syncem.population.quadrature import panels

    def x_given(look):
        return se.Uniform(0.25 * look.value - 0.025, 0.25 * look.value + 0.025, nodes=2)

    def pitch(x):
        return se.Uniform(0.239 + x.value, 0.241 + x.value)

    pop = se.Population(
        look=se.Isotropic(), x=se.Conditional(x_given), pitch=se.Conditional(pitch)
    )
    total = sum(
        np.dot(weight, 2 * (1 + point["look"] + 0.5 * point["x"]))
        for point, weight in panels(
            pop.factors,
            {},
            units={"look": "", "x": ""},
            diagonal=("pitch", "look"),
            n=4,
            axis_nodes={"x": 5},
        )
    )
    np.testing.assert_allclose(total, (1 + 1.125 * 0.24 / 0.75) / 0.75, rtol=2e-12)


def test_host_diagonal_knots_returns_python_floats_without_static_array_warning():
    import warnings
    from syncem.population._diagonal import host_diagonal_knots

    def x_given(look):
        return se.Uniform(0.25 * look.value - 0.025, 0.25 * look.value + 0.025, nodes=4)

    def pitch(x):
        return se.Uniform(0.239 + x.value, 0.241 + x.value)

    edges = host_diagonal_knots(
        se.Conditional(
            lambda look: se.Uniform(0.2 + 0.25 * look.value, 0.3 + 0.25 * look.value)
        ),
        "look",
        {},
        {"look": ""},
        (-1.0, 1.0),
    )
    assert edges and all(type(value) is float for value in edges)

    pop = se.Population(
        z=0.0,
        look=se.Uniform(-0.6, 0.6),
        x=se.Conditional(x_given),
        pitch=se.Conditional(pitch),
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        se.spectral_basis(
            pop,
            _CHANNELS,
            kernel=_chain_kernel,
            lattice=se.Lattice(look=se.Legendre(2)),
            check=False,
            cache=False,
        )
    static_warnings = [
        item
        for item in caught
        if "JAX array is being set as static" in str(item.message)
    ]
    assert not static_warnings
