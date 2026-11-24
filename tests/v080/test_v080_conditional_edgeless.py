"""Moving-node R3a quadrature on an unpartitioned child axis."""

import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import build_from_averaged
from syncem.fitting._conditioned_points import conditioned_point_mass_map
from syncem.fitting.conditional_basis import conditional_basis
from syncem.population import pair

_FREQ = se.Channels.at([0.5, 1, 2], unit="GHz")
_CHILD = se.Variable("y", kind="positive", unit="G", bare=False)


@se.kernel(variables=(se.Variable("x", kind="positive"), _CHILD), stokes=("I",))
def _log_kernel(nu, x, y):
    t = nu / 1e9
    z = jnp.log(y / 5e-6)
    return {"I": 10 + t * jnp.log(x / 2) + t**2 * (z + z**2)}


def _law(x):
    return se.LogNormal((5e-6 * (x / 2) ** 0.3, "G"), 0.22)


def _lattice(order=2, child=None):
    if child is None:
        child = se.Cells.single("5 uG", 0.3, chart="log", order=order)
    return se.Lattice(
        x=se.Cells.log(1.0, 4.0, 2, order=order),
        y=child,
        weights=[0.3, 0.8],
        order=order,
    )


def _reference(x):
    # Independent integration of the law in standard-normal coordinates.
    u, w = np.polynomial.legendre.leggauss(64)
    w = w * np.exp(-((6 * u) ** 2) / 2)
    w /= w.sum()
    z = 0.3 * np.log(x / 2) + 6 * 0.22 * u
    t = np.asarray(_FREQ.centres_hz) / 1e9
    return 10 + t * np.log(x / 2) + t**2 * (w @ (z + z**2))


@pytest.mark.parametrize("order", [2, 6])
def test_edgeless_lognormal_child_moves_with_parent_and_preserves_weights(order):
    lat = _lattice(order)
    joint = build_from_averaged(AveragedKernel(_log_kernel), _FREQ, lat, cache=False)
    basis = conditional_basis(joint, "y", _law, parent_weights=[0.6, 0.4])
    for point in (1.1, 1.8, 2.5, 3.7):
        parent = se.Population(x=se.Delta(point))
        moments = pair(parent, basis.lattice)
        result = basis.predict(moments, check=False)
        np.testing.assert_allclose(result.I, _reference(point), rtol=3e-12, atol=2e-13)


@se.kernel(
    variables=(se.Variable("eta", kind="cosine"), se.Variable("u")), stokes=("I",)
)
def _angular_kernel(nu, eta, u):
    return {"I": (3 + eta * u) * nu / 1e9}


def test_edgeless_child_of_legendre_parent_uses_its_own_rule():
    lat = se.Lattice(
        eta=se.Legendre(2),
        u=se.Cells.single(0.0, 0.4, order=1),
        weights=[0.3],
    )
    joint = build_from_averaged(
        AveragedKernel(_angular_kernel, legendre={"eta": (2, 24)}),
        _FREQ,
        lat,
        cache=False,
    )

    def law(eta):
        centre = 0.3 + 0.2 * eta
        return se.Uniform(centre - 0.1, centre + 0.1)

    basis = conditional_basis(joint, "u", law, parent_weights=[0.6])
    eta = -0.7
    parent = pair(se.Population(eta=se.Delta(eta)), basis.lattice)
    expected = (3 + eta * (0.3 + 0.2 * eta)) * _FREQ.centres_hz / 1e9
    np.testing.assert_allclose(
        basis.predict(parent, check=False).I, expected, atol=2e-13
    )


def test_edgeless_child_rejects_point_law_and_overlapping_edgeless_charts():
    lat = _lattice()
    joint = build_from_averaged(AveragedKernel(_log_kernel), _FREQ, lat, cache=False)
    with pytest.raises(se.RoleError, match="needs a conditional density"):
        conditional_basis(joint, "y", lambda x: se.Delta((x * 4e-6, "G")))
    charts = se.Cells.at_points(["4 uG", "6 uG"], [0.2, 0.2], chart="log", order=2)
    duplicate = se.Lattice(x=lat.axes["x"], y=charts, order=2)
    joint = build_from_averaged(
        AveragedKernel(_log_kernel), _FREQ, duplicate, cache=False
    )
    with pytest.raises(se.RoleError, match="one cell without edges"):
        conditional_basis(joint, "y", _law)


def test_edgeless_nonnegative_parent_maps_whole_probability_per_atom():
    lattice = _lattice()
    basis = build_from_averaged(
        AveragedKernel(_log_kernel), _FREQ, lattice, cache=False
    )
    marker = se.NonNegativeCells("x", point_masses=3)
    with pytest.raises(se.SizeError, match="map workspace"):
        conditioned_point_mass_map(basis, marker, "y", _law, 256)
    with pytest.raises(se.SizeError, match="moving child quadrature"):
        conditioned_point_mass_map(basis, marker, "y", _law, 2**14)
    mapping = conditioned_point_mass_map(basis, marker, "y", _law, 2**24)
    design = basis.columns.lift(basis.design())
    expected = np.column_stack([_reference(x) for x in mapping.coordinates["x"]])
    np.testing.assert_allclose(design[:, 0] @ mapping.moments, expected, atol=2e-13)
    mass_rows = [lattice.index.cell_rows(cell).start for cell in range(lattice.n_cells)]
    scaled = mapping.moments[mass_rows] * lattice.weights[:, None]
    np.testing.assert_allclose(scaled.sum(axis=0), 1.0, atol=1e-14)

    def outside_chart(x):
        return se.Normal((x * 4e-6, "G"), "0.2 uG")

    with pytest.raises(se.RoleError, match="invalid support or density"):
        conditional_basis(basis, "y", outside_chart)
    with pytest.raises(se.RoleError, match="invalid support or density"):
        conditioned_point_mass_map(basis, marker, "y", outside_chart, 2**24)


def test_explicit_finite_child_support_keeps_conditioned_panel_mass():
    lat = se.Lattice(
        eta=se.Legendre(2),
        u=se.Cells.single(0.75, 0.25, order=1, support=(0.5, 1.0)),
        weights=[0.7],
    )
    joint = build_from_averaged(
        AveragedKernel(_angular_kernel, legendre={"eta": (2, 24)}),
        _FREQ,
        lat,
        cache=False,
    )

    def law(eta):
        return se.Uniform(0.2 + 0.1 * eta, 1.2 + 0.1 * eta)

    basis = conditional_basis(joint, "u", law)
    eta = -0.7
    parent = pair(se.Population(eta=se.Delta(eta)), basis.lattice)
    expected = 0.5 * (3 + eta * 0.75) * _FREQ.centres_hz / 1e9
    np.testing.assert_allclose(
        basis.predict(parent, check=False).I, expected, atol=2e-13
    )


def test_edgeless_child_requires_probability_rule():
    class HalfMass(se.Uniform):
        def rule(self, n=None, *, parents=None):
            points, weights = super().rule(n, parents=parents)
            return points, weights * 0.5

    joint = build_from_averaged(
        AveragedKernel(_log_kernel), _FREQ, _lattice(), cache=False
    )

    def law(x):
        return HalfMass((3e-6 * x, "G"), (5e-6 * x, "G"))

    with pytest.raises(se.RoleError, match="unit-probability rule"):
        conditional_basis(joint, "y", law)


def test_edgeless_single_precision_quadrature_preserves_mass():
    class RoundedRule(se.Uniform):
        def rule(self, n=None, *, parents=None):
            points = self.lo + (self.hi - self.lo) * jnp.array([0, 0.5, 1])
            return points, jnp.full(3, 1 / 3, dtype=jnp.float32)

    lat = _lattice()
    joint = build_from_averaged(AveragedKernel(_log_kernel), _FREQ, lat, cache=False)

    def law(x):
        return RoundedRule((3e-6 * x, "G"), (5e-6 * x, "G"))

    basis = conditional_basis(joint, "y", law)
    parent = pair(se.Population(x=se.Delta(1.5)), basis.lattice)
    joint_moments = basis.mapping.apply(parent.values)
    mass_rows = [lat.index.cell_rows(cell).start for cell in range(lat.n_cells)]
    np.testing.assert_allclose(
        np.asarray(joint_moments)[mass_rows] @ lat.weights, 1.0, atol=1e-14
    )
