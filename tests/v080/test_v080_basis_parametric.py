"""A fixed-rank outer solve across a change in the grid UPRE choice (T18)."""

from dataclasses import replace

import numpy as np
import pytest
import syncem as se

from syncem.base.channels import Channels
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.distributions.parameters import Free
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import build_from_averaged
from syncem.fitting.basis_parametric import _jacobian, fit_basis_parametric
from syncem.fitting.data import StokesData
from syncem.fitting.rank import FullRank, Rank
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.values.errors import RoleError


@kernel(variables=(Variable("x"),), stokes=("I",))
def template(nu, x):
    return {"I": (1 + x) * nu / 1e9}


@pytest.fixture(scope="module")
def baseline():
    lattice = Lattice(x=Cells.linear(0.5, 2.5, 1, order=1), weights=[1.0])
    channels = Channels.at([0.2, 0.3, 0.4, 0.8, 1.2, 2.0], unit="GHz")
    return build_from_averaged(AveragedKernel(template), channels, lattice, cache=False)


def columns(angle):
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0], [s, 0], [0, 0.4 * c], [0, 0.4 * s], [0.2, 0], [0, 0]])


def experiment(baseline, truth, *, log=False):
    visited = []

    def build(params):
        value = params["B.width"]
        visited.append(value)
        angle = np.log(value) if log else value
        design = columns(angle)
        return replace(baseline, spectra0=design[:, None, :])

    moments = np.array([4.0, np.sqrt(3) / 0.4])
    observed = columns(truth) @ moments
    observed[-1] += 0.4
    sigma = np.ones(len(observed))
    data = StokesData(baseline.channels, I=observed, sigma=sigma, unit="cgs")
    return build, data, moments, visited


def test_rank_selection_stays_fixed_across_a_upre_transition(baseline):
    truth, bounds = 0.75, (-0.8, 1.2)
    build, data, moments, _ = experiment(baseline, truth)
    output = []
    for start in (-0.6, 0.7):
        fit = fit_basis_parametric(build, data, {"B.width": Free(start, bounds=bounds)})
        output.append(fit)
        assert fit.rank == 2
        assert fit.dof == 3
        assert fit.chi2 == pytest.approx(0.16, abs=2e-14)
        assert fit.params["B.width"].value == pytest.approx(truth, abs=5e-9)
        assert not fit.params["B.width"].at_bound
        assert fit.params["B.width"].sigma == pytest.approx(1 / np.sqrt(19), rel=1e-6)
        np.testing.assert_allclose(fit.moments.values, moments, rtol=1e-8)
        assert {row.rank for row in fit.grid} == {1, 2}
        assert len(fit.grid) <= 27
        assert abs(fit.finite_difference_steps[0][1]) == pytest.approx(1e-4)
        assert 1 <= fit.converged_starts <= 3
        assert fit.converged
        assert any("conditional" in note for note in fit.predict().plan.notes)
    assert output[0].grid[0].rank == 1
    assert output[1].grid[0].rank == 2
    np.testing.assert_allclose(
        output[0].predict().stokes, output[1].predict().stokes, atol=1e-8
    )


@pytest.mark.parametrize("target", [-0.8, 1.2])
def test_log_parameter_with_boundary_refinement(baseline, target):
    bounds = tuple(map(float, np.exp([-0.8, 1.2])))
    build, data, _, visited = experiment(baseline, target, log=True)
    fit = fit_basis_parametric(
        build, data, {"B.width": Free(1.0, bounds=bounds, log=True)}, rank=Rank(2)
    )
    parameter = fit.params["B.width"]
    assert parameter.value == pytest.approx(np.exp(target), rel=1e-6)
    assert parameter.at_bound and parameter.identified
    assert parameter.sigma == pytest.approx(np.exp(target) / np.sqrt(19), rel=1e-5)
    assert np.sign(fit.finite_difference_steps[0][1]) == np.sign(-target)
    assert min(visited) >= bounds[0] - 1e-15 and max(visited) <= bounds[1] + 1e-15
    assert fit.basis_evaluations == len(visited)


def test_tied_amplitude_reports_profile_instead_of_sigma(baseline):
    build, data, _, _ = experiment(baseline, 0.75)
    fitted_basis = build({"B.width": 0.75})

    def scaled(params):
        return replace(fitted_basis, spectra0=fitted_basis.spectra0 * params["scale"])

    result = fit_basis_parametric(
        scaled,
        data,
        {"scale": Free(1.0, bounds=(0.4, 1.4))},
        rank=FullRank(),
        max_starts=1,
    )
    assert result.rank == 2
    parameter = result.params["scale"]
    assert not parameter.identified and np.isnan(parameter.sigma)
    assert not np.isfinite(result.parameter_covariance).any()
    _, profile = zip(*result.profiles["scale"])
    np.testing.assert_allclose(profile, 0.16, atol=5e-15)


def test_changes_to_moment_coordinates_raise(baseline):
    _, data, _, _ = experiment(baseline, 0.75)

    def rebuild(params):
        return baseline if params["width"] == 1 else baseline.with_order(0)

    with pytest.raises(RoleError, match="preserve the moment coordinates"):
        fit_basis_parametric(rebuild, data, {"width": Free(1.0, bounds=(0.8, 1.2))})


def test_narrow_physical_bounds_do_not_mark_an_interior_field_as_at_bound(baseline):
    _, data, _, _ = experiment(baseline, 0.75)

    def rebuild(params):
        angle = params["B.width"] / 1e-6 - 5.0 + 0.75
        return replace(baseline, spectra0=columns(angle)[:, None, :])

    result = fit_basis_parametric(
        rebuild, data, {"B.width": Free(4.7e-6, bounds=(3e-6, 7e-6))}, rank=Rank(2)
    )
    estimate = result.params["B.width"]
    assert estimate.value == pytest.approx(5e-6, abs=1e-12)
    assert estimate.sigma == pytest.approx(1e-6 / np.sqrt(19), rel=1e-6)
    assert estimate.identified and not estimate.at_bound
    assert result.finite_difference_steps[0][1] == pytest.approx(4e-10)


def test_finite_difference_backs_off_from_rank_boundary():
    def evaluate(x):
        return np.array([4 * x[0]]) if abs(x[0]) < 2e-5 else None

    jac, step = _jacobian(
        evaluate, np.array([0.0]), np.array([-1.0]), np.array([1.0]), 1e-4
    )
    np.testing.assert_allclose(jac, [[4]])
    assert step[0] == pytest.approx(1.25e-5)


def test_public_builder_rebuilds_an_averaged_basis_factor():
    @se.kernel(variables=(se.Variable("x"), se.Variable("y")), stokes=("I",))
    def model(nu, x, y):
        return {"I": x * (1 + y * nu / 1e9)}

    channels = se.Channels.at([0.5, 1.0, 1.5, 3.0, 4.0], unit="GHz")
    lattice = se.Lattice(x=se.Cells.linear(1.0, 2.0, 1, order=0))
    pop = se.Population(se.FreeMoments("x"), y=se.Free(0.2, bounds=(0.1, 0.6)))
    data = se.StokesData(
        channels, I=2 * (1 + 0.3 * channels.centres_hz / 1e9), unit="cgs", sigma=0.1
    )
    fit = se.fit(pop, data, lattice=lattice, kernel=model, rank=se.Rank(1), check=False)
    assert fit.kind == "basis-parametric"
    assert fit.params["y"].value == pytest.approx(0.3, abs=1e-8)
    np.testing.assert_allclose(fit.predict().I, data.stokes[:, 0], rtol=1e-8)
