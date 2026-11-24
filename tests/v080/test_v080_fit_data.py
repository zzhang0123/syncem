"""Unit and whitening tests for batched Stokes observations."""

import equinox as eqx
import jax
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.fitting.data import StokesData
from syncem.records.ledger import ErrorTerm, Ledger
from syncem.units import UnitError
from syncem.values.errors import DataError
from syncem.values.spectrum import Spectrum


def test_missing_unit_and_masked_noise():
    channels = Channels.at([1.0, 2.0], unit="GHz")
    with pytest.raises(UnitError, match="StokesData has no unit"):
        StokesData(channels, I=[1.0, 2.0], sigma=1.0)
    data = StokesData(
        channels,
        I=[np.nan, 1.0],
        sigma=[np.nan, 0.5],
        mask=[False, True],
        unit="Jy/sr",
    )
    model = np.array([[[2e-23, 0]], [[0, 3e-23]]])
    white = data.whitened(model)
    np.testing.assert_allclose(white.data, [0.0, 2.0])
    np.testing.assert_allclose(white.design, [[0.0, 0.0], [0.0, 6.0]])
    assert white.count == 1
    with pytest.raises(DataError, match="positive"):
        StokesData(channels, I=[1.0, 1.0], sigma=[1.0, -1.0], unit="cgs").whitened(
            model
        )


def test_channel_selection_and_withholding_keep_the_same_covariance_block():
    channels = Channels.at([1.0, 2.0, 3.0], unit="GHz")
    cov = np.array([[1.0, 0.3, -0.1], [0.3, 2.0, 0.2], [-0.1, 0.2, 2.0]])
    data = StokesData(channels, I=[4.0, 5.0, 6.0], cov=cov, unit="cgs")
    design = np.eye(3).reshape(3, 1, 3)
    all_fit = eqx.filter_jit(lambda d: d.withhold([1]).whitened(design))(data)
    short_fit = data[[0, 2]].whitened(design[[0, 2]])
    np.testing.assert_allclose(np.asarray(all_fit.data)[[0, 2]], short_fit.data)
    np.testing.assert_allclose(np.asarray(all_fit.design)[[0, 2]], short_fit.design)
    assert all_fit.count == short_fit.count == 2
    np.testing.assert_array_equal(data[[0, 2]].cov, cov[[0, 2]][:, [0, 2]])
    held = data.withhold([1]).whitened(design, withheld=True)
    np.testing.assert_allclose(held.data, [0.0, 5.0 / np.sqrt(2.0), 0.0])


def test_shared_noise_with_pixels_and_response_space():
    channels = Channels.at([1.0, 2.0], unit="GHz")
    design = np.array([[[2e-23]], [[3e-23]]])
    response = np.array([[1.0, 2.0], [-1.0, 0.0], [0.0, 3.0]])
    cov = 4 * np.eye(3)
    data = StokesData(
        channels,
        stokes=[[4.0, -2.0, 2.0], [8.0, -3.0, 4.0]],
        unit="Jy/sr",
        cov=cov,
        response=response,
        response_uncertainty=np.zeros_like(response),
        discrepancy=ErrorTerm(0.1, "bound", name="input"),
    )
    out = eqx.filter_jit(lambda d: d.whitened(design))(data)
    np.testing.assert_allclose(out.data, [[2.0, -1.0, 1.0], [4.0, -1.5, 2.0]])
    np.testing.assert_allclose(out.design, (response @ [[2.0], [3.0]]) / 2)
    np.testing.assert_array_equal(out.count, [3, 3])
    assert data.discrepancy.name == "discrepancy" and data.discrepancy.declared


def test_simulate_fractional_intensity_noise_under_jit():
    channels = Channels.at([1.0, 2.0], unit="GHz")
    truth = Spectrum(
        [[[2.0, 0.1], [6.0, 0.2]], [[4.0, 0.5], [10.0, 0.6]]], channels, ledger=Ledger()
    )
    key = jax.random.key(13)
    draw = eqx.filter_jit(lambda s, k: StokesData.simulate(s, noise=0.1, key=k))(
        truth, key
    )
    expected = np.asarray(truth.I)[..., None] * 0.1
    np.testing.assert_allclose(
        draw.sigma, np.broadcast_to(expected, truth.stokes.shape)
    )
    np.testing.assert_allclose(
        draw.stokes - truth.stokes,
        draw.sigma * jax.random.normal(key, truth.stokes.shape),
    )


def test_whitened_spectrum_checks_the_tabulated_response_between_nodes():
    grid = np.array([0.5, 0.7, 1.0, 1.3, 1.5]) * 1e9

    def channels(shoulder):
        return Channels.from_table(
            grid, [0, shoulder, 1, shoulder, 0], smoothness=0, n_nu=1
        )

    original = channels(0.25)
    different = channels(0.75)
    assert original.row_keys() == different.row_keys()
    data = StokesData(original, I=[2.0], sigma=1.0, unit="cgs")
    with pytest.raises(DataError, match="share response bands"):
        data.whitened_spectrum(Spectrum([[3.0]], different, ledger=Ledger()))
    matched = data.whitened_spectrum(Spectrum([[3.0]], channels(0.25), ledger=Ledger()))
    np.testing.assert_allclose(matched.design, [[3.0]])


def test_whitened_spectrum_requires_identity_for_opaque_responses():
    first = Channels.from_callable(lambda nu: 1.0 + 0 * nu, [1e9, 2e9], 0, n_nu=1)
    second = Channels.from_callable(lambda nu: nu / 1.5e9, [1e9, 2e9], 0, n_nu=1)
    assert first.row_keys() == second.row_keys()
    data = StokesData(first, I=[2.0], sigma=1.0, unit="cgs")
    with pytest.raises(DataError, match="share response bands"):
        data.whitened_spectrum(Spectrum([[3.0]], second, ledger=Ledger()))
    matched = data.whitened_spectrum(Spectrum([[3.0]], first, ledger=Ledger()))
    np.testing.assert_allclose(matched.design, [[3.0]])

    mono = Channels.at([1.5], unit="GHz")
    data = StokesData(mono, I=[2.0], sigma=1.0, unit="cgs")
    equivalent = Channels.at([1500.0], unit="MHz")
    matched = data.whitened_spectrum(Spectrum([[3.0]], equivalent, ledger=Ledger()))
    np.testing.assert_allclose(matched.design, [[3.0]])


def test_join_preserves_mixed_quadratures_masks_and_noise_blocks():
    first = Channels.at([20.0, 40.0], unit="GHz")
    second = Channels.at([150.0], unit="GHz", half_width=0.2, n_nu=16)
    a = StokesData(first, I=[2.0, 3.0], sigma=[0.5, 1.0], unit="K_RJ")
    b = StokesData(second, I=[4.0], cov=[[9.0]], unit="K_RJ")
    joined = a.withhold([1]) + b
    assert joined.channels.row_keys() == (*first.row_keys(), *second.row_keys())
    assert joined.channels[[2, 0, 2]].row_keys() == (
        second.row_keys()[0],
        first.row_keys()[0],
        second.row_keys()[0],
    )
    np.testing.assert_array_equal(joined.channels.response_weights()[:2, 1:], 0)
    np.testing.assert_array_equal(
        joined.channels.response_weights()[2:3], second.response_weights()
    )
    np.testing.assert_array_equal(joined.cov, np.diag([0.25, 1.0, 9.0]))
    columns = np.arange(3.0).reshape(3, 1, 1) + 1
    out = joined.whitened(columns, unit="K_RJ")
    np.testing.assert_allclose(out.data, [4.0, 0.0, 4 / 3])
    np.testing.assert_allclose(out.design[:, 0], [2.0, 0.0, 1.0])
    assert out.count == 2
    np.testing.assert_allclose(
        joined.whitened(columns, unit="K_RJ", withheld=True).data, [0.0, 3.0, 0.0]
    )


def test_join_embeds_each_instruments_stokes_axes_and_declared_envelopes():
    first = Channels.at([20.0], unit="GHz")
    second = Channels.at([150.0], unit="GHz")
    a = StokesData(
        first,
        I=[4.0],
        sigma=1.0,
        unit="Jy/sr",
        discrepancy=ErrorTerm(0.2, "bound", name="front"),
    )
    b = StokesData(
        second,
        stokes=[9.0],
        components=("Q",),
        sigma=2.0,
        unit="Jy/sr",
        response=[[-3.0]],
        response_uncertainty=[[0.5]],
        discrepancy=ErrorTerm([[0.4]], "estimate", name="front"),
    )
    joined = a + b
    assert joined.components == ("I", "Q")
    np.testing.assert_allclose(joined.response, [[1.0, 0.0, 0.0, 0.0], [0, 0, 0, -3]])
    out = joined.whitened(np.array([[[4.0], [8.0]], [[6.0], [2.0]]]), unit="Jy/sr")
    np.testing.assert_allclose(out.data, [4.0, 4.5])
    np.testing.assert_allclose(out.design[:, 0], [4.0, -3.0])
    assert joined.discrepancy.kind == "estimate" and joined.discrepancy.declared
    np.testing.assert_allclose(joined.discrepancy.value, [0.2, 1.4])


@pytest.mark.parametrize("metadata", [{"relative_to": "I"}, {"space": "retained"}])
def test_join_refuses_incompatible_discrepancy_coordinates(metadata):
    channels = Channels.at([20.0], unit="GHz")
    a = StokesData(
        channels,
        I=[4.0],
        sigma=1.0,
        unit="Jy/sr",
        discrepancy=ErrorTerm(0.1, "bound", **metadata),
    )
    b = StokesData(
        channels,
        I=[3.0],
        sigma=1.0,
        unit="Jy/sr",
        discrepancy=ErrorTerm(0.2, "bound"),
    )
    with pytest.raises(DataError, match="absolute envelope in the observation unit"):
        _ = a + b


def test_mixed_noise_join_checks_unmasked_standard_deviations():
    first = Channels.at([20.0, 40.0], unit="GHz")
    second = Channels.at([150.0], unit="GHz")
    covariance = StokesData(second, I=[3.0], cov=[[9.0]], unit="cgs")
    bad = StokesData(first, I=[2.0, 4.0], sigma=[0.5, -1.0], unit="cgs")
    for standard_deviation in (bad, bad.withhold([1])):
        with pytest.raises(DataError, match="standard deviations.*positive"):
            _ = standard_deviation + covariance

    masked = StokesData(
        first, I=[2.0, np.nan], sigma=[0.5, np.nan], mask=[True, False], unit="cgs"
    )
    joined = masked + covariance
    white = joined.whitened(np.ones((3, 1, 1)))
    np.testing.assert_allclose(white.data, [4.0, 0.0, 1.0])
    np.testing.assert_allclose(white.design[:, 0], [2.0, 0.0, 1 / 3])
    assert white.count == 2
