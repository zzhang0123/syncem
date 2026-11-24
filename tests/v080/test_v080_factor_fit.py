"""Independent factors on joint cell rows with a total-degree cap."""

from dataclasses import replace

import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import build_from_averaged
from syncem.fitting.data import StokesData
from syncem.fitting.factor_fit import fit_factors
from syncem.fitting.rank import FullRank, Rank
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.values.errors import RoleError


@kernel(variables=tuple(Variable(name) for name in ("x", "b", "d")), stokes=("I",))
def template(nu, x, b, d):
    return {"I": (1.0 + nu / 1e9) * x * b * d}


def example(groups, raw, order):
    axes = {name: Cells.linear(1, 3, 2, order=order) for name in groups}
    weights = np.linspace(0.7, 1.6, 2 ** len(axes))
    lattice = Lattice(**axes, order=order, weights=weights)
    channels = Channels.at(np.linspace(1.0, 2.0, lattice.n + 2), unit="GHz")
    fixed = {} if "d" in axes else {"d": 1.0}
    basis = build_from_averaged(
        AveragedKernel(template, fixed=fixed), channels, lattice, cache=False
    )
    index = basis.lattice.index.arrays
    cell = np.unravel_index(index.cell, basis.lattice.cells.tensor)
    rows = [cell[i] * (order + 1) + index.powers[:, i] for i in range(len(axes))]
    values = np.prod([coeff[row] for coeff, row in zip(raw, rows)], axis=0)
    design = np.vstack(
        (np.eye(lattice.n), np.ones(lattice.n), np.linspace(-0.3, 0.9, lattice.n))
    )
    design[-2:] /= lattice.n
    basis = replace(basis, spectra0=design[:, None, :])
    # The assembler multiplies stored spectra by the joint reference weights.
    np.testing.assert_allclose(basis.design()[:, 0, :], design * weights[index.cell])
    estimate = design @ values
    covariance = 0.015 * np.eye(len(estimate)) + 0.0005 * np.ones(
        (len(estimate), len(estimate))
    )
    data = StokesData(channels, I=estimate, cov=covariance, unit="cgs").withhold(
        [len(estimate) - 1]
    )
    return basis, data, values, rows


def test_singular_pair_start_fits_truncated_weighted_tensor():
    a = np.array([2.3, -0.65, 1.1, 0.1])
    b = np.array([0.8, -0.4, 0.5, 0.3])
    basis, data, expected, rows = example(("x", "b"), (a, b), 1)
    result = fit_factors(
        basis, data, (("x",), ("b",)), rank=Rank(4), starts=1, max_iterations=100
    )

    assert result.rank == 4 and result.outer_dimension == 3 and result.outer_rank == 3
    assert result.outer_identified and result.dof == 6
    assert result.chi2 < 1e-9 and result.withheld.chi2 < 1e-9
    assert result.initial_joint_rank > 0
    np.testing.assert_allclose(
        result.factors[0].values, a * np.linalg.norm(b), atol=3e-6
    )
    np.testing.assert_allclose(
        result.factors[1].values, b / np.linalg.norm(b), atol=3e-6
    )
    np.testing.assert_allclose(basis.lattice.raw(result.moments), expected, atol=3e-6)
    np.testing.assert_allclose(
        result.factors[0].values[rows[0]] * result.factors[1].values[rows[1]],
        expected,
        atol=3e-6,
    )
    np.testing.assert_allclose(
        result.retained.directions @ result.retained.directions.T, np.eye(4), atol=1e-13
    )
    predicted = result.predict(cov=True)
    np.testing.assert_allclose(predicted.I, data.stokes[:, 0], atol=3e-6)
    np.testing.assert_allclose(
        predicted.sigma[:, 0] ** 2, np.diag(predicted.cov), atol=1e-13
    )
    assert any("conditional" in note for note in predicted.plan.notes)


def test_three_groups_and_nontrivial_gauge():
    factors = tuple(
        np.array(value) for value in ([2.2, 1.7], [-0.91, 0.5], [0.65, 0.7])
    )
    names = ("x", "b", "d")
    basis, data, expected, _ = example(names, factors, 0)
    result = fit_factors(basis, data, names, rank=FullRank(), seed=43, starts=2)
    assert result.rank == 2 and result.outer_dimension == result.outer_rank == 2
    assert result.effective_parameters == 4
    np.testing.assert_allclose(basis.lattice.raw(result.moments), expected, atol=3e-6)
    for factor in result.factors[1:]:
        assert np.linalg.norm(factor.values) == pytest.approx(1.0)
        assert factor.values[np.argmax(abs(factor.values))] > 0
    with pytest.raises(RoleError, match="partition"):
        fit_factors(basis, data, (("x",), ("d",)))
    coupled = fit_factors(basis, data, (("d", "x"), ("b",)), rank=FullRank(), starts=1)
    assert coupled.factors[0].variables == ("d", "x")
    assert coupled.factors[0].coordinates[1] == ((0, 1), (0, 0))
    np.testing.assert_allclose(basis.lattice.raw(coupled.moments), expected, atol=3e-6)


def test_continuum_field_and_gamma_fit_from_singular_pair():
    channels = Channels.log(0.1, 5.0, 16, unit="GHz")
    lattice = Lattice(
        gamma=Cells.log(1200, 12000, 3, order=0),
        B=Cells.log(3, 15, 2, order=0, unit="uG"),
    )
    basis = build_from_averaged(
        AveragedKernel(ContinuumKernel(), fixed={"eta": 0.0, "mu": 0.0, "depth": 0.0}),
        channels,
        lattice,
        stokes=("I",),
        cache=False,
    )
    unscaled = np.outer([2e11, 8e10, 4e10], [0.6, 0.4]).ravel()
    truth = basis.design()[:, 0, :] @ (unscaled / basis.lattice.weights)
    data = StokesData(channels, I=truth, sigma=0.005 * truth, unit="cgs")
    fit = fit_factors(basis, data, (("gamma",), ("B",)), rank=Rank(3), starts=1)
    assert fit.rank == 3 and fit.initial_joint_rank > 0
    assert any("scaling relation" in note for note in fit.plan.notes)
    np.testing.assert_allclose(fit.predict().I, truth, rtol=1e-3)
