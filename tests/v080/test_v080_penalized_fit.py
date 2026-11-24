"""Prebuilt-basis fits in scaled rows, with batched noise and profiles."""

from types import SimpleNamespace

import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import build_from_averaged
from syncem.fitting._point_mass_fit import point_mass_map
from syncem.fitting.data import StokesData
from syncem.fitting.penalized_fit import fit_nonnegative, fit_penalized
from syncem.fitting.penalties import Penalty
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.population.blocks import NonNegativeCells
from syncem.values.errors import DataError, RoleError, SizeError


@kernel(variables=[Variable("x", kind="positive")], stokes=("I",))
def _toy(nu, x):
    f = nu / 1e9
    return {"I": 1 + f * x}


@pytest.fixture(scope="module")
def basis():
    channels = Channels.at([1.0, 2.0, 3.0, 4.0], unit="GHz")
    lattice = Lattice(x=Cells.linear(1.0, 3.0, 1, order=1), weights=[2.5])
    return build_from_averaged(AveragedKernel(_toy), channels, lattice, cache=False)


def _lifted(basis):
    return np.asarray(basis.columns.lift(basis.design())[:, 0, :])


def _oracle(matrix, y, alpha):
    estimator = np.linalg.solve(
        matrix.T @ matrix + alpha * np.eye(matrix.shape[1]), matrix.T
    )
    return estimator @ y, estimator


def test_prebuilt_ridge_uses_scaled_rows_and_noise_only_covariance(basis):
    matrix = _lifted(basis)
    truth = np.array([1.2, 0.5])
    alpha = 0.7
    sigma = 2.0
    data = StokesData(
        basis.channels,
        I=(matrix @ truth) * 1e23,
        sigma=np.full((4, 1), sigma * 1e23),
        unit="Jy/sr",
    ).withhold([3])
    fit = fit_penalized(basis, data, Penalty(alpha))
    expected, estimate = _oracle(
        matrix[:3] / sigma, (matrix @ truth)[:3] / sigma, alpha
    )
    assert fit.moments.origin == "penalised"
    assert fit.retained.rank == 2 and fit.alpha == alpha
    np.testing.assert_allclose(fit.moments.values, expected)
    assert fit.withheld.n == 1
    assert fit.withheld.chi2 == pytest.approx(
        ((matrix[3] @ (expected - truth)) / sigma) ** 2
    )

    predicted = fit.predict(cov=True)
    assert predicted.channels is basis.channels
    np.testing.assert_allclose(predicted.I / 1e23, matrix @ expected)
    gain = matrix @ estimate
    np.testing.assert_allclose(predicted.cov / 1e46, gain @ gain.T, rtol=1e-13)
    np.testing.assert_allclose(
        predicted.sigma[:, 0] / 1e23, np.linalg.norm(gain, axis=1)
    )
    retain = estimate @ (matrix[:3] / sigma)
    fraction = np.linalg.norm(matrix - matrix @ retain, axis=-1) / np.linalg.norm(
        matrix, axis=-1
    )
    np.testing.assert_allclose(predicted.unconstrained[:, 0], fraction)
    assert predicted.ledger["remainder"].kind == "unbounded"
    assert fit.column.origin == "penalised"


@pytest.mark.parametrize("nonnegative", [False, True])
def test_penalized_goodness_uses_fractional_degrees_of_freedom(basis, nonnegative):
    from scipy.stats import chi2

    one = basis.at(basis.channels[[0]]).with_order(0)
    scale = float(_lifted(one)[0, 0])
    data = StokesData(one.channels, I=[scale], sigma=scale, unit="cgs")
    penalty = Penalty(1.0 / 3.0)
    if nonnegative:
        fit = fit_nonnegative(one, data, NonNegativeCells("x", point_masses=1), penalty)
    else:
        fit = fit_penalized(one, data, penalty)

    assert fit.dof == pytest.approx(0.25)
    assert fit.chi2 == pytest.approx(0.0625)
    assert fit.p_value == pytest.approx(chi2.sf(0.0625, 0.25))


def test_ridge_per_pixel_chunks_and_channel_projection(basis):
    matrix = _lifted(basis)
    y = np.array(
        [
            [[4.0, 5.0, 7.0, 8.0], [1.0, 2.0, 3.0, 4.0]],
            [[3.0, 6.0, 7.0, 9.0], [6.0, 9.0, 15.0, 20.0]],
        ]
    )
    noise = np.linspace(0.5, 3.0, y.size).reshape(y.shape)
    data = StokesData(basis.channels, I=y, sigma=noise[..., None], unit="cgs")
    small = fit_penalized(basis, data, Penalty(0.5), chunk_bytes=1100)
    stored = fit_penalized(basis, data, Penalty(0.5), keep_directions=True)
    assert small.chunk_pixels < np.prod(data.batch_shape)
    assert small.retained.directions is None
    for where in np.ndindex(data.batch_shape):
        expected, _ = _oracle(
            matrix / noise[where][:, None], y[where] / noise[where], 0.5
        )
        np.testing.assert_allclose(small.moments.values[where], expected, rtol=5e-13)
    channels = Channels.at([2.0, 5.0], unit="GHz")
    a, b = small.predict(channels, cov=True), stored.predict(channels, cov=True)
    assert a.stokes.shape == (2, 2, 2, 1)
    for field in ("stokes", "sigma", "cov", "unconstrained"):
        np.testing.assert_allclose(getattr(a, field), getattr(b, field), atol=1e-12)

    with pytest.raises(SizeError, match="ridge pixel"):
        fit_penalized(basis, data, Penalty(0.5), chunk_bytes=10)


def test_nonnegative_result_profiles_the_whitened_unpenalized_domain(basis):
    design = _lifted(basis)
    # A single atom lies at the cell's chart midpoint x = 2. Its scaled
    # zeroth moment is its column divided by the lattice weight 2.5.
    response = design[:, 0] / 2.5
    truth = 3.0
    sigma = np.array([1.0, 2.0, 3.0, 4.0])
    data = StokesData(
        basis.channels, I=response * truth, sigma=sigma[:, None], unit="cgs"
    ).withhold([3])
    fit = fit_nonnegative(
        basis, data, NonNegativeCells("x", point_masses=1), Penalty(3.0)
    )
    whitened = response[:3] / sigma[:3]
    norm = whitened @ whitened
    atom = truth * norm / (norm + 3.0 / 2.5**2)
    assert float(fit.weights[0]) == pytest.approx(atom)
    assert fit.moments.origin == "nonnegative_penalised"
    assert fit.negative_intensity_columns == 0
    assert fit.withheld.n == 1 and fit.withheld.chi2_predictive is None
    pred = fit.predict()
    np.testing.assert_allclose(pred.I, atom * response)
    assert pred.sigma is pred.cov is pred.unconstrained is None
    with pytest.raises(DataError, match="active constraints"):
        fit.predict(cov=True)

    profile = fit.profile_max(basis.channels[[1, 3]], delta_chi2=1.0)
    assert profile.origin == "nonnegative_profile" and profile.sigma is None
    np.testing.assert_allclose(
        profile.value, (truth + 1 / np.sqrt(norm)) * response[[1, 3]], rtol=1e-10
    )
    coords, weights = fit.electron_spectrum()
    np.testing.assert_allclose(coords, [2.0])
    np.testing.assert_allclose(weights, [atom])
    assert float(fit.population.column) == pytest.approx(atom)

    from jax.random import key

    draws = fit.resample(3, key=key(19))
    assert draws.points is fit.points
    assert draws.weights.shape == (3, 1)
    assert draws.predict().stokes.shape == (3, 4, 1)
    np.testing.assert_array_equal(draws.withheld.n, [1, 1, 1])
    np.testing.assert_array_equal(draws.weights, fit.resample(3, key=key(19)).weights)


def test_nonnegative_host_chunking_and_unit_projection(basis):
    response = _lifted(basis)[:, 0] / 2.5
    n = np.arange(1, 5) + 0.5
    data = StokesData(
        basis.channels,
        I=n[:, None] * response * 1e23,
        sigma=np.ones((4, 1)) * 1e23,
        unit="Jy/sr",
    )
    fit = fit_nonnegative(
        basis,
        data,
        NonNegativeCells("x", point_masses=1),
        Penalty(0),
        chunk_bytes=450,
    )
    assert fit.chunk_pixels == 1
    np.testing.assert_allclose(fit.weights[:, 0], n)
    np.testing.assert_allclose(fit.predict().I / 1e23, n[:, None] * response)
    prof = fit.profile_max(basis.channels[[1]], delta_chi2=0.0)
    np.testing.assert_allclose(prof.value[:, 0] / 1e23, n * response[1])


def test_mixed_chart_mass_map_uses_clipped_faces_and_shifted_references():
    # The map is geometric. No provider is needed to compare it with the
    # defining monomials for these two variables.
    variables = (Variable("x", kind="positive"), Variable("z"))
    lattice = Lattice(
        x=Cells.log(1.0, 4.0, 1, order=2, reference=2.2),
        z=Cells.linear(-2.0, 2.0, 1, order=2, reference=0.4),
        order=2,
        weights=[3.0],
    )
    source = SimpleNamespace(
        lattice=lattice,
        averaged=SimpleNamespace(kernel=SimpleNamespace(variables=variables)),
    )
    marker = NonNegativeCells("x", "z", point_masses=3, support={"x": (1.2, 3.0)})
    mp = point_mass_map(source, marker, 2**20)
    x, z = mp.coordinates["x"], mp.coordinates["z"]
    np.testing.assert_allclose(np.unique(x), [1.2, np.sqrt(3.6), 3.0])
    np.testing.assert_allclose(np.unique(z), [-2.0, 0.0, 2.0], atol=1e-15)
    u = np.log(x / 2.2) / lattice.cells.half_width[0, 0]
    v = (z - 0.4) / lattice.cells.half_width[0, 1]
    expected = np.array([u**a * v**b / 3 for a, b in lattice.index.arrays.powers])
    np.testing.assert_allclose(mp.moments, expected)
    with pytest.raises(RoleError, match="all resolved axes"):
        point_mass_map(source, NonNegativeCells("x"), 2**20)
    with pytest.raises(SizeError, match="map workspace"):
        point_mass_map(source, marker, 10)
