"""Batched combined estimators use one observed design with bounded reductions."""

import math

import numpy as np

import syncem as se
from syncem.fitting import sum_fit
from syncem.fitting._combined import IndependentBases
from tests.v080.test_v080_r7_fit import _kernel, _setup


def _compare_pixels(fit, singles, output, *, held=False):
    spectrum = fit.predict(output, cov=True)
    want = [single.predict(output, cov=True) for single in singles]
    batch = fit.rank.shape
    for name in ("stokes", "sigma", "cov", "unconstrained"):
        value = getattr(spectrum, name)
        expected = np.stack([getattr(item, name) for item in want])
        np.testing.assert_allclose(
            value,
            expected.reshape(*batch, *expected.shape[1:]),
            atol=2e-11,
            rtol=2e-11,
        )
    for name in ("rank", "dof", "chi2"):
        expected = np.asarray([getattr(item, name) for item in singles]).reshape(batch)
        np.testing.assert_allclose(getattr(fit, name), expected, atol=2e-11)
    for i, part in enumerate(fit.moments):
        expected = np.stack([single.moments[i].values for single in singles])
        np.testing.assert_allclose(
            part.values, expected.reshape(*batch, -1), atol=2e-11, rtol=2e-11
        )
    coordinate = np.arange(sum(part.values.shape[-1] for part in fit.moments))
    expected = np.asarray(
        [single.observable(coordinate).value for single in singles]
    ).reshape(batch)
    np.testing.assert_allclose(fit.observable(coordinate).value, expected, atol=2e-11)
    if held:
        expected = np.asarray(
            [single.withheld.chi2_predictive for single in singles]
        ).reshape(batch)
        np.testing.assert_allclose(fit.withheld.chi2_predictive, expected, atol=2e-11)


def test_shared_noise_batches_reuse_one_reduction_and_recompute_on_request(monkeypatch):
    first, second, lat, example = _setup()
    source = first + second
    band = example.channels
    values = np.arange(1, 5).reshape(2, 2, 1) * np.asarray(example.stokes)[:, 0]
    data = se.StokesData(band, I=values, sigma=0.7, unit="cgs")
    options = dict(lattice=lat, kernel=_kernel, check=False, rank=se.FullRank())
    original = sum_fit.solve_chunk
    reductions = []

    def solve(white, rank, pairs, reduction=None):
        if reduction is None:
            reductions.append(white.design.shape)
        return original(white, rank, pairs, reduction)

    with monkeypatch.context() as spy:
        spy.setattr(sum_fit, "solve_chunk", solve)
        fit = se.fit(source, data, **options)
    assert reductions == [(8, 4)]
    assert fit.retained.directions is None
    singles = [
        se.fit(source, se.StokesData(band, I=value, sigma=0.7, unit="cgs"), **options)
        for value in values.reshape(-1, band.n_ch)
    ]
    _compare_pixels(fit, singles, band[[0, 2, 4]])

    stored = se.fit(source, data, keep_directions=True, **options)
    assert stored.retained.directions.shape == (4, 4)
    np.testing.assert_allclose(stored.predict().sigma, fit.predict().sigma, rtol=2e-13)


def test_per_pixel_covariance_and_masks_bound_the_front_whitening(monkeypatch):
    first, second, lat, example = _setup()
    source = first + second
    band = example.channels
    values = np.arange(1, 5).reshape(4, 1) * np.asarray(example.stokes)[:, 0]
    coordinates = np.arange(band.n_ch)
    covariance = np.array([0.7, 0.8, 1.0, 1.2])[:, None, None] ** 2
    covariance = covariance * 0.15 ** np.abs(coordinates[:, None] - coordinates)
    mask = np.ones_like(values, dtype=bool)
    mask[1::2, -1] = False

    def observations(value, cov, selected):
        return se.StokesData(
            band, I=value, cov=cov, mask=selected, unit="cgs"
        ).withhold([2])

    data = observations(values, covariance, mask)
    options = dict(lattice=lat, kernel=_kernel, check=False, rank=se.FullRank())
    white = se.StokesData.whitened
    observed = []

    def whitened(self, *args, **kwargs):
        observed.append(math.prod(self.batch_shape))
        return white(self, *args, **kwargs)

    with monkeypatch.context() as spy:
        spy.setattr(se.StokesData, "whitened", whitened)
        fit = se.fit(source, data, chunk_bytes=16_000, **options)
    assert 1 <= fit.chunk_pixels < values.shape[0]
    assert max(observed) == fit.chunk_pixels
    singles = [
        se.fit(source, observations(value, matrix, selected), **options)
        for value, matrix, selected in zip(values, covariance, mask, strict=True)
    ]
    _compare_pixels(fit, singles, band[[0, 2, 4]], held=True)

    stored = se.fit(source, data, chunk_bytes=16_000, keep_directions=True, **options)
    assert stored.retained.directions.shape == (4, 4, 4)
    np.testing.assert_allclose(stored.predict().sigma, fit.predict().sigma, rtol=2e-13)


@se.kernel(variables=[se.Variable("c", kind="positive")], stokes=("I",))
def _grain(nu, c):
    return {"I": c * ((nu / 1e9) ** 4 + 1)}


def test_independent_mechanisms_preserve_lattices_in_a_batch():
    first, _, lat, example = _setup()
    grain = se.Population(c=se.FreeMoments())
    grain_lat = se.Lattice(c=se.Cells.linear(1, 2, 1, order=0))
    band = example.channels
    values = np.array([0.6, 1.4])[:, None] * np.asarray(example.stokes)[:, 0]

    def fit(value):
        data = se.StokesData(band, I=value, sigma=0.7, unit="cgs")
        a = se.problem(first, data, lattice=lat, kernel=_kernel, check=False)
        b = se.problem(grain, data, lattice=grain_lat, kernel=_grain, check=False)
        return (a + b).solve(rank=se.FullRank())

    result = fit(values)
    assert isinstance(result.problem.basis, IndependentBases)
    assert result.moments[0].index.table.variables == ("a",)
    assert result.moments[1].index.table.variables == ("c",)
    _compare_pixels(result, [fit(value) for value in values], band[[0, 2, 4]])
