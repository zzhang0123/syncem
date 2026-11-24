"""Profile a nonnegative noise contour after a line-search status."""

from types import SimpleNamespace

import numpy as np
import pytest
from scipy import optimize

from syncem.fitting.penalties import profile_max


def _stop_at(monkeypatch, weights):
    def stopped(*args, **kwargs):
        return SimpleNamespace(
            x=np.asarray(weights),
            success=False,
            message="Positive directional derivative for linesearch",
        )

    monkeypatch.setattr(optimize, "minimize", stopped)


def test_feasible_profile_recovered_with_primal_dual_check(monkeypatch):
    _stop_at(monkeypatch, [1.25 + 1e-9, 0.5])
    identity = np.eye(2)
    result = profile_max(identity, [0.25, 0.5], identity, [1.0, 0.0])
    assert result.value == pytest.approx(1.25, abs=1e-13)
    np.testing.assert_allclose(result.weights, [1.25, 0.5], atol=1e-13)
    assert result.chi2 == pytest.approx(1.0, abs=1e-13)


def test_feasibility_alone_does_not_accept_failed_profile(monkeypatch):
    _stop_at(monkeypatch, [0.0, 0.6])
    identity = np.eye(2)
    with pytest.raises(RuntimeError, match="Positive directional derivative"):
        profile_max(identity, [0.25, 0.5], identity, [1.0, 0.0])


def test_recover_a_rank_deficient_positive_support(monkeypatch):
    _stop_at(monkeypatch, [0.7, 0.54, 0.6])
    design = np.array([[1.0, 1.0, 0.0], [0.0, 0.0, 1.0]])
    objective = np.array([1.0, 1.0, 0.0])
    result = profile_max(design, [0.25, 0.5], np.eye(3), objective)
    assert result.value == pytest.approx(1.25, abs=1e-13)
    assert result.chi2 == pytest.approx(1.0, abs=1e-13)
    assert np.all(result.weights >= 0.0)
    np.testing.assert_allclose(design @ result.weights, [1.25, 0.5], atol=1e-13)


def test_no_false_dual_agreement_with_a_large_residual(monkeypatch):
    _stop_at(monkeypatch, [1.0])
    result = profile_max([[1.0], [0.0]], [0.125, -26345179.905070756], np.eye(1), [1.0])
    assert result.value == pytest.approx(1.125, abs=1e-13)
    np.testing.assert_allclose(result.weights, [1.125], atol=1e-13)


def test_numerical_residue_does_not_keep_an_inactive_atom(monkeypatch):
    _stop_at(monkeypatch, [1.0, 1e-9])
    result = profile_max(np.eye(2), [0.125, 0.5], np.eye(2), [1.0, -1.0])
    expected = 0.125 + np.sqrt(0.75)
    assert result.value == pytest.approx(expected, abs=1e-13)
    assert result.chi2 == pytest.approx(1.0, abs=1e-13)
    np.testing.assert_allclose(result.weights, [expected, 0.0], atol=1e-13)
