"""Retained response and discrepancy propagation against an explicit estimator."""

from types import SimpleNamespace

import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.fitting._free_fit_arrays import solve_chunk
from syncem.fitting._response_bias import bias_elements, bias_terms, response_bias
from syncem.fitting.data import StokesData
from syncem.fitting.rank import Rank
from syncem.records.ledger import ErrorTerm, Ledger
from syncem.values.errors import DataError


def _run(data, design, components, rank):
    white = data.whitened(design, components=components)
    solved, reduction = solve_chunk(white, Rank(rank), None)
    result = response_bias(data, design, components, solved, reduction)
    return result, bias_terms(data, result)


def _estimator(design, cov, active, rank):
    kept_cov = cov[np.ix_(active, active)]
    lower = np.linalg.cholesky(kept_cov)
    left, singular, vt = np.linalg.svd(
        np.linalg.solve(lower, design[active]), full_matrices=False
    )
    estimator = np.linalg.solve(lower.T, left[:, :rank]).T / singular[:rank, None]
    return estimator, vt[:rank]


def test_batched_correlated_noise_models_response_and_discrepancy():
    ch = Channels.at([4, 12], unit="GHz")
    design = np.array(
        [[[0.8, 1.1, 0.3], [-0.6, 0.7, 0.1]], [[-1.0, 0.2, 0.4], [0.9, 1.2, -0.7]]]
    )
    response = np.array(
        [
            [1, 0.5, -0.2, 0.7],
            [0, 0, 1, 0],
            [1, 1, 0, -1],
            [0.8, 0.1, 1, -0.3],
            [-0.4, 0.5, -1, -0.5],
        ]
    )
    width = np.abs(response) * 0.03 + 0.01
    source = design[:, [1, 0], :].reshape(4, 3)
    data_design = response @ source
    observed = np.stack(
        [data_design @ [1.0, -0.3, 0.1], data_design @ [0.1, 1.0, -0.3]]
    )
    cross = np.array([0.8, -0.4, 0.2, 0.6, -1.0])
    covariance = np.stack(
        [np.eye(5) + np.outer(cross, cross), 2 * np.eye(5) + np.outer(cross, cross)]
    )
    mask = [[True, True, True, False, True], [True, True, False, True, True]]
    model_error = np.array([[0.1, 0.2], [0.4, 0.05]])
    data = StokesData(
        ch,
        stokes=observed,
        components=("I", "Q"),
        cov=covariance,
        mask=mask,
        unit="Jy/sr",
        response=response,
        response_uncertainty=width,
        discrepancy=ErrorTerm(model_error, "bound", declared=True),
    ).withhold([1])

    result, terms = _run(data, design * 1e-23, ("Q", "I"), 2)

    for i, keep in enumerate(np.asarray(data.mask & ~data.withheld)):
        estimator, directions = _estimator(data_design, covariance[i], keep, 2)
        fitted = source @ (directions.T @ (estimator @ observed[i, keep]))
        expected_response = np.abs(estimator) @ (width @ np.abs(fitted))[keep]
        expected_discrepancy = (
            np.abs(estimator) @ ((np.abs(response) + width) @ model_error.ravel())[keep]
        )
        np.testing.assert_allclose(result.response[i], [*expected_response, 0.0])
        np.testing.assert_allclose(result.discrepancy[i], [*expected_discrepancy, 0.0])
    assert [term.name for term in terms] == ["response", "discrepancy"]
    assert [term.kind for term in terms] == ["estimate", "bound"]
    assert all(term.declared and term.space == "retained" for term in terms)
    stokes_error = ErrorTerm(np.ones((2, 2, 2)), "bound", name="physical_kernel")
    envelope = Ledger((*terms, stokes_error)).envelope()
    assert envelope.kind == "bound"
    np.testing.assert_array_equal(envelope.value, stokes_error.value)


def test_diagonal_noise_data_space_discrepancy_and_active_mask():
    ch = Channels.at([1, 2, 3, 4], unit="GHz")
    design = np.arange(1.0, 9).reshape(4, 1, 2)
    errors = np.array([0.1, 0.2, 1e100, 0.3])
    data = StokesData(
        ch,
        I=[[1.0, -1.0, np.nan, 0.5], [-1.0, 0.7, np.nan, 2.0]],
        sigma=[0.4, 0.6, np.nan, 0.3],
        mask=[True, True, False, True],
        unit="cgs",
        discrepancy=ErrorTerm(errors, "estimate"),
    ).withhold([1])

    result, terms = _run(data, design, ("I",), 1)

    active = np.array([True, False, False, True])
    cov = np.diag([0.4**2, 0.6**2, 1, 0.3**2])
    estimator, _ = _estimator(design[:, 0], cov, active, 1)
    expected = (np.abs(estimator) @ errors[active])[0]
    np.testing.assert_allclose(result.discrepancy, [[expected, 0], [expected, 0]])
    assert result.response is None
    assert terms[0].kind == "not_applicable"
    assert terms[1].kind == "estimate" and terms[1].declared


@pytest.mark.parametrize(
    "width, response_kind", [(None, "unbounded"), (np.zeros((2, 2)), "bound")]
)
def test_response_declaration_distinguishes_zero_from_unspecified(width, response_kind):
    ch = Channels.at([1, 2], unit="GHz")
    data = StokesData(
        ch,
        I=[1.5, 2.5],
        sigma=0.1,
        unit="cgs",
        response=np.eye(2),
        response_uncertainty=width,
    )
    _, terms = _run(data, np.ones((2, 1, 1)), ("I",), 1)
    assert terms[0].kind == response_kind
    assert terms[1].kind == "unbounded"
    if width is not None:
        np.testing.assert_array_equal(terms[0].value, [0.0])
        assert terms[0].declared


def test_diagonal_workspace_does_not_require_square_noise_storage():
    m = 100_000
    data = SimpleNamespace(
        n_data=m,
        channels=SimpleNamespace(n_ch=m),
        components=("I",),
        cov=None,
        response=None,
        response_uncertainty=None,
        discrepancy=ErrorTerm(0.2, "bound"),
    )
    fixed, per_pixel = bias_elements(data, 1, shared=True)
    assert fixed + per_pixel < 6 * m


def test_relative_discrepancy_needs_an_explicit_absolute_envelope():
    ch = Channels.at([1, 2], unit="GHz")
    data = StokesData(
        ch,
        I=[4.0, 8.0],
        sigma=0.5,
        unit="cgs",
        discrepancy=ErrorTerm(0.1, "bound", relative_to="I"),
    )
    with pytest.raises(DataError, match="absolute envelope"):
        _run(data, np.ones((2, 1, 1)), ("I",), 1)
