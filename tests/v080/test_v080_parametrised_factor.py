"""A resolved G14-type family with a free parent moment block."""

from dataclasses import replace

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.distributions import Free, family
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import SpectralBasis, build_from_averaged
from syncem.fitting.data import StokesData
from syncem.fitting.parametrised_factor import _Projection, fit_parametrised_factor
from syncem.fitting.rank import FullRank
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.population import Population, pair
from syncem.values.errors import RoleError


@family(variable="B", unit="uG", support=(0.5, 50.0))
def field_law(B, median=5.0, width=0.3):
    return jnp.exp(-0.5 * (jnp.log(B / median) / width) ** 2) / B


@kernel(
    variables=(
        Variable("gamma", kind="positive"),
        Variable("B", kind="positive", unit="G"),
    ),
    stokes=("I", "P"),
)
def template(nu, gamma, B):
    intensity = gamma * B + nu / 1e9
    return {"I": intensity, "P": 0.5j * intensity}


@pytest.fixture(scope="module")
def example():
    energy = Cells.log(500, 3000, 2, order=1)
    field = Cells.log(0.5, 50, 3, order=1, unit="uG")
    joint = Lattice(gamma=energy, B=field, order=1, weights=np.linspace(0.45, 1.46, 6))
    channels = Channels.at(np.linspace(1.0, 2.0, joint.n + 3), unit="GHz")
    basis = build_from_averaged(AveragedKernel(template), channels, joint, cache=False)
    n = joint.n
    synthetic = np.vstack((np.diag(np.linspace(0.5, 1.5, n)), np.full((3, n), 1.0 / n)))
    basis = replace(
        basis,
        spectra0=synthetic[:, None, :],
        spectra2=synthetic * (0.1 + 0.2j),
    )
    factor_lattice = Lattice(B=field, order=1)
    known = Population(B=field_law(width=0.42))
    field_raw = np.asarray(factor_lattice.raw(pair(known, factor_lattice)))
    rows = joint.index.arrays
    g, b = np.unravel_index(rows.cell, joint.cells.tensor)
    parent_raw = np.array([1.8, -0.14, 1.3, 0.3])
    joint_raw = (
        parent_raw[2 * g + rows.powers[:, 0]] * field_raw[2 * b + rows.powers[:, 1]]
    )
    I = synthetic @ joint_raw
    observed = I[:, None] * [1.0, 0.1, 0.2]
    parent = Lattice(gamma=energy, weights=[0.8, 2.3], order=1)
    return basis, observed, parent, parent_raw, joint_raw


def test_recovers_weighted_parent_rows_factor_and_joint_spectra(example, monkeypatch):
    basis, observed, parent, parent_raw, joint_raw = example
    active_channels = basis.channels[:-1]
    data = StokesData(
        active_channels, stokes=observed[:-1], sigma=0.025, unit="cgs"
    ).withhold([0])
    original_at = SpectralBasis.at
    rows = []

    def at(self, channels):
        rows.append(channels.row_keys())
        return original_at(self, channels)

    monkeypatch.setattr(SpectralBasis, "at", at)
    model = field_law(width=Free(0.31, bounds=(0.22, 0.75)))
    result = fit_parametrised_factor(
        basis, data, "B", model, rank=FullRank(), parent_lattice=parent
    )

    assert rows == [active_channels.row_keys()]
    assert result.kind == "bilinear (parametrised factor)"
    assert result.params["B.width"].value == pytest.approx(0.42, abs=1e-7)
    assert result.factor.params["width"] == pytest.approx(0.42, abs=1e-7)
    assert result.parameter_covariance.shape == (1, 1)
    assert result.finite_difference_steps[0][1] != 0.0
    assert result.moments.key == parent.key
    assert result.joint_moments.key == basis.lattice.key
    assert result.moments.origin == result.joint_moments.origin == "minimum_norm"
    np.testing.assert_allclose(parent.raw(result.moments), parent_raw, atol=1e-7)
    np.testing.assert_allclose(
        basis.lattice.raw(result.joint_moments), joint_raw, atol=1e-7
    )
    prediction = result.predict(cov=True)
    assert prediction.channels.row_keys() == basis.channels.row_keys()
    np.testing.assert_allclose(prediction.stokes, observed, atol=1e-7)
    np.testing.assert_allclose(prediction.sigma.ravel() ** 2, np.diag(prediction.cov))
    assert result.withheld.chi2 < 1e-10


def test_invalid_parent_geometry_and_fixed_factor_are_refused(example):
    basis, observed, _, _, _ = example
    data = StokesData(basis.channels, I=observed[:, 0], sigma=0.02, unit="cgs")
    model = field_law(width=Free(0.31, bounds=(0.22, 0.75)))
    bad = Lattice(gamma=Cells.log(550, 3000, 2, order=1), order=1)
    with pytest.raises(RoleError, match="joint basis cell geometry"):
        fit_parametrised_factor(basis, data, "B", model, parent_lattice=bad)
    with pytest.raises(RoleError, match="Free shape parameter"):
        fit_parametrised_factor(basis, data, "B", field_law(width=0.42))


def test_projects_an_order_above_an_axis_cap_and_preserves_parent_cross_terms():
    gamma = Cells.log(500, 3000, 1, order=2)
    field = Cells.log(0.5, 50, 1, order=2, unit="uG")
    joint = Lattice(gamma=gamma, B=field, order=3)
    fit_map = _Projection.build(joint, "B", None)
    assert fit_map.parent.order == fit_map.factor.order == 2

    eta = Cells.linear(-0.8, 0.8, 1, order=2)
    joint = Lattice(gamma=gamma, B=field.with_order(3), eta=eta)
    fit_map = _Projection.build(joint, "B", None)
    assert fit_map.parent.order == 3
    assert np.any(np.sum(fit_map.parent.index.arrays.powers, axis=1) == 3)
