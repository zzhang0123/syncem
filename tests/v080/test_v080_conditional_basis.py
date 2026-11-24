"""R3a inverse design on a tensor lattice with a bounded moving-support child."""

import numpy as np
import pytest
import syncem as se

from syncem.base.channels import Channels
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.distributions import Conditional, Delta, PowerLaw, Uniform
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import build_from_averaged
from syncem.engine.assemble import predict_values
from syncem.fitting.conditional_basis import conditional_basis
from syncem.fitting.data import StokesData
from syncem.fitting.free_fit import fit_free
from syncem.fitting.rank import FullRank
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.moments import Moments
from syncem.values.errors import RoleError


@kernel(variables=(Variable("x"), Variable("y")), stokes=("I",))
def _toy(nu, x, y):
    t = nu / 1e9
    return {"I": 10 + t * x + t**2 * y**2}


@kernel(variables=(Variable("x"), Variable("z"), Variable("y")), stokes=("I",))
def _two_parents(nu, x, z, y):
    t = nu / 1e9
    return {"I": 10 + t * x + t**2 * y**2 + t**3 * z}


def _child(x):
    return Uniform(0.0, 0.6 + 0.12 * x)


def _cubic(x):
    return Uniform(0.0, 0.6 + 0.12 * x + 0.11 * x**2)


def _both(x, z):
    return Uniform(0.0, 0.6 + 0.12 * x + 0.08 * z)


def _point(x):
    return Delta(0.6 + 0.1 * x)


def _joint(parent_cells=1, order=2):
    channels = Channels.at([0.5, 1, 2, 3], unit="GHz")
    x = Cells.linear(-1, 1, parent_cells, order=order)
    y = Cells.from_edges([0, 0.3, 0.9], chart="affine", order=order)
    weights = np.arange(1, 2 * parent_cells + 1) / 7
    lattice = Lattice(x=x, y=y, order=order, weights=weights)
    return build_from_averaged(AveragedKernel(_toy), channels, lattice, cache=False)


def _spectrum(nu, x, upper):
    t = np.asarray(nu) / 1e9
    return 10 + t * x + t**2 * upper**2 / 3


def _scaled_point(lattice, cell, x):
    table = lattice.cells
    z = (x - table.reference[cell, 0]) / table.half_width[cell, 0]
    a = np.zeros(lattice.n)
    rows = tuple(lattice.index.cell_rows(cell))
    a[list(rows)] = z ** np.arange(len(rows)) / lattice.weights[cell]
    return Moments(lattice, a)


def test_moving_child_faces_and_weight_ratio_are_in_the_inverse_basis():
    joint = _joint(parent_cells=2)
    mapped = conditional_basis(
        joint, "y", Conditional(_child), parent_weights=[0.3, 0.7]
    )
    assert mapped.lattice.cells.variables == ("x",)
    assert mapped.design().shape == (4, 1, 6)
    for b, x in enumerate((-0.6, 0.9)):
        moments = _scaled_point(mapped.lattice, b, x)
        expected = _spectrum(mapped.channels.centres_hz, x, 0.6 + 0.12 * x)
        np.testing.assert_allclose(
            predict_values(mapped, moments)[:, 0], expected, atol=5e-13, rtol=0
        )
        expanded = mapped.mapping.apply(moments.values)
        np.testing.assert_allclose(
            predict_values(joint, Moments(joint.lattice, expanded))[:, 0],
            expected,
            atol=5e-13,
            rtol=0,
        )
    np.testing.assert_allclose(mapped.conditional_truncation(), 0, atol=6e-13)
    assert mapped.ledger["conditional_truncation:y|x"].kind == "unbounded"


def test_public_fit_and_problem_use_only_the_parent_moments():
    joint = _joint()
    mapped = conditional_basis(joint, "y", Conditional(_child))
    truth = np.array([1.5, 0.12, 0.3])
    data = StokesData(
        joint.channels, I=mapped.design()[:, 0] @ truth, sigma=0.1, unit="cgs"
    )
    pop = se.Population(se.FreeMoments("x"), y=Conditional(_child))
    result = se.fit(
        pop, data, lattice=joint.lattice, kernel=_toy, rank=FullRank(), check=False
    )
    np.testing.assert_allclose(result.moments.values, truth, rtol=3e-10)
    assert se.problem(pop, data, basis=result.problem.basis).params.a.size == 3
    name = "conditional_truncation:y|x"
    assert result.ledger[name].kind == "unbounded"
    assert result.ledger[f"{name}:fitted_mass"].kind == "estimate"
    assert result.predict().ledger[f"{name}:fitted_mass"].value.shape == (4, 1)


def test_conditional_parent_slice_requires_a_matching_tensor_grid():
    basis = conditional_basis(_joint(parent_cells=2), "y", Conditional(_child))
    data = StokesData(
        basis.channels, I=np.ones(basis.channels.n_ch), sigma=1, unit="cgs"
    )
    source = se.Population(
        x=se.FreeMoments(cells=slice(1, None)), y=Conditional(_child)
    )
    for method in (se.fit, se.problem):
        with pytest.raises(RoleError, match="selected parent cells"):
            method(source, data, basis=basis, check=False)


def test_r3a_sparse_operator_preserves_fit_observables_and_new_channels():
    mapped = conditional_basis(_joint(), "y", Conditional(_child), parent_weights=[0.8])
    truth = np.array([1.5, 0.12, 0.3])
    original = np.asarray(mapped.design())[:, 0, :]
    sigma = np.array([0.01, 0.03, 0.02, 0.04])
    data = StokesData(
        mapped.channels, I=original @ truth, sigma=sigma[:, None], unit="cgs"
    )
    fit = fit_free(mapped, data, rank=FullRank())
    np.testing.assert_allclose(fit.moments.values, truth, rtol=3e-10)
    assert fit.column.value == pytest.approx(0.8 * truth[0], rel=1e-11)
    new = Channels.at([1.0, 4.0], unit="GHz")
    rebuilt = mapped.at(new)
    assert rebuilt.mapping.key == mapped.mapping.key
    output = fit.predict(new, cov=True)
    design = np.asarray(rebuilt.design())[:, 0, :]
    whitened = original / sigma[:, None]
    _, s, vt = np.linalg.svd(whitened, full_matrices=False)
    covariance = (vt.T / s**2) @ vt
    np.testing.assert_allclose(output.I, design @ truth, rtol=1e-11)
    np.testing.assert_allclose(output.cov, design @ covariance @ design.T, rtol=3e-9)


def test_truncation_diagnostic_estimates_omitted_polynomial_terms():
    mapped = conditional_basis(_joint(), "y", Conditional(_cubic))
    estimate = mapped.conditional_truncation()[0, :, 0]
    for x in np.linspace(-1, 1, 65):
        moments = _scaled_point(mapped.lattice, 0, x)
        expected = _spectrum(
            mapped.channels.centres_hz, x, 0.6 + 0.12 * x + 0.11 * x**2
        )
        residual = np.abs(predict_values(mapped, moments)[:, 0] - expected)
        assert np.all(residual <= estimate + 5e-13)
    assert np.all(estimate > 0)
    assert mapped.with_order(1).lattice.index.n == 2


def test_r3a_supports_more_than_one_parent_direction():
    ch = Channels.at([0.5, 2.0], unit="GHz")
    axis = Cells.linear(-1, 1, 1, order=2)
    lattice = Lattice(x=axis, z=axis, y=Cells.linear(0, 1, 1, order=2), order=2)
    joint = build_from_averaged(AveragedKernel(_two_parents), ch, lattice, cache=False)
    result = conditional_basis(joint, "y", Conditional(_both))
    assert result.lattice.cells.variables == ("x", "z")
    assert result.design().shape == (2, 1, 6)
    x, z = 0.2, -0.6
    powers = result.lattice.index.patterns[0]
    point = np.array([x**r * z**s for r, s in powers])
    nu = ch.centres_hz
    wanted = _spectrum(nu, x, 0.6 + 0.12 * x + 0.08 * z) + (nu / 1e9) ** 3 * z
    pred = predict_values(result, Moments(result.lattice, point))
    np.testing.assert_allclose(pred[:, 0], wanted, rtol=1e-13)


def test_r3a_rejects_point_child_and_missing_dependency():
    joint = _joint()
    with pytest.raises(RoleError, match="needs a conditional density"):
        conditional_basis(joint, "y", Conditional(_point))
    with pytest.raises(RoleError, match="belong to the unknown"):
        conditional_basis(joint, "y", Conditional(lambda z: Uniform(0, 0.5 + z)))


def test_moving_face_at_parent_reference_raises_instead_of_losing_flux():
    def edge(x):
        return Uniform(0.0, 0.3 + 0.1 * x)

    with pytest.raises(RoleError, match="moving support on a child cell face"):
        conditional_basis(_joint(), "y", Conditional(edge))


def test_invalid_law_at_parent_face_raises_before_fit():
    def late(x):
        return Uniform(0.0, 0.2 + 0.4 * x)

    with pytest.raises(
        RoleError, match="invalid support or density on the parent grid"
    ):
        conditional_basis(_joint(), "y", Conditional(late))


@kernel(variables=(Variable("x"), Variable("gamma", kind="positive")), stokes=("I",))
def _normalisation_kernel(nu, x, gamma):
    return {"I": 10 + nu / 1e9 * x + 0.0 * gamma}


def test_electron_child_column_is_owned_by_the_unknown_parent():
    ch = Channels.at([1.0, 2.0], unit="GHz")
    joint = Lattice(
        x=Cells.linear(-1, 1, 1, order=1),
        gamma=Cells.log(300, 1000, 2, order=1),
    )
    basis = build_from_averaged(
        AveragedKernel(_normalisation_kernel), ch, joint, cache=False
    )

    def amplitude(x):
        return PowerLaw(2.6 + 0.1 * x.value, lo=300, hi=1000, amplitude=35.0)

    def column(x):
        return PowerLaw(2.6 + 0.1 * x.value, lo=300, hi=1000, column=0.1)

    for law in (amplitude, column):
        result = conditional_basis(basis, "gamma", Conditional(law))
        predicted = predict_values(result, _scaled_point(result.lattice, 0, 0.4))
        np.testing.assert_allclose(predicted[:, 0], [10.4, 10.8], atol=2e-13, rtol=0)


def test_order_six_jet_supports_negative_child_coordinates():
    for order in (5, 6):
        result = conditional_basis(_joint(order=order), "y", Conditional(_child))
        for x in (-0.7, 0.8):
            pred = predict_values(result, _scaled_point(result.lattice, 0, x))
            expected = _spectrum(result.channels.centres_hz, x, 0.6 + 0.12 * x)
            np.testing.assert_allclose(pred[:, 0], expected, atol=1e-10, rtol=0)
