"""Population sums use one observation likelihood and tie shared statistics."""

import numpy as np
import pytest

import syncem as se
from syncem.engine.basis_sum import BasisSum
from syncem.values.errors import RoleError, SpecifiedMismatch


@se.kernel(
    variables=[se.Variable("a", kind="positive"), se.Variable("b", kind="positive")],
    stokes=("I",),
)
def _kernel(nu, a, b):
    x = nu / 1e9
    return {"I": 1 + b * x + a * (x**2 + b * x**3)}


def _setup():
    abscissa = np.linspace(0.6, 2.5, 8)
    channels = se.Channels.at(abscissa, unit="GHz")
    lattice = se.Lattice(a=se.Cells.linear(1, 2, 1, order=1))
    first = se.Population(a=se.FreeMoments(), b=1.0)
    second = se.Population(a=se.FreeMoments(), b=2.0)

    def intensity(a, b):
        return _kernel.fn(abscissa * 1e9, a, b)["I"]

    data = se.StokesData(
        channels,
        I=2 * intensity(1.25, 1.0) + 3 * intensity(1.75, 2.0),
        sigma=0.7,
        unit="cgs",
    )
    return first, second, lattice, data


def test_r7_fits_distinct_statistics_as_one_linear_design():
    first, second, lattice, data = _setup()
    options = dict(lattice=lattice, kernel=_kernel, check=False)
    fit = se.fit(first + second, data, rank=se.FullRank(), **options)
    problem = se.problem(first + second, data, **options)
    solo = [se.problem(part, data, **options) for part in (first, second)]
    expected = np.concatenate([one.design for one in solo], axis=-1)

    np.testing.assert_array_equal(problem.design, expected)
    np.testing.assert_allclose(fit.predict().I, data.stokes[:, 0], rtol=2e-13)
    assert int(fit.rank) == 4
    assert int(fit.dof) == 4
    assert fit.plan.layout.basis_groups == ((0,), (1,))
    assert fit.problem.basis.member_indices == ((0,), (1,))
    np.testing.assert_allclose(
        [moments.values[0] for moments in fit.moments], [2, 3], rtol=2e-12
    )


def test_r7_ties_members_with_equal_statistics_on_prebuilt_basis():
    first, second, lattice, data = _setup()
    law = se.Uniform(1.0, 2.0)
    first_concrete = first.replace(a=law)
    second_concrete = second.replace(a=law)
    basis = se.spectral_basis(
        first_concrete + second_concrete + first_concrete,
        data.channels,
        lattice=lattice,
        kernel=_kernel,
        check=False,
    )
    basis = BasisSum(
        basis.parts,
        ("first statistic", "second statistic"),
        member_indices=basis.member_indices,
    )
    tied = se.fit(
        first + second + first, data, basis=basis, rank=se.FullRank(), check=False
    )
    assert tuple(item.component for item in tied.moments) == basis.components
    assert tied.problem.basis.member_indices == ((0, 2), (1,))
    assert any("Only their summed moments" in note for note in tied.plan.caveats)
    np.testing.assert_allclose(tied.predict().I, data.stokes[:, 0], rtol=2e-13)


def test_r7_rejects_a_mixed_specified_and_free_inverse_sum():
    first, second, lattice, data = _setup()
    with pytest.raises(RoleError, match="sum component 2 needs one joint FreeMoments"):
        se.fit(
            first + second.replace(a=se.Uniform(1.0, 2.0)),
            data,
            lattice=lattice,
            kernel=_kernel,
            check=False,
        )


def test_r7_prebuilt_statistics_mismatch_requires_opt_in():
    first, second, lattice, data = _setup()
    law = se.Uniform(1.0, 2.0)
    basis = se.spectral_basis(
        first.replace(a=law) + second.replace(a=law),
        data.channels,
        lattice=lattice,
        kernel=_kernel,
        check=False,
    )
    changed = first.replace(b=3.0) + second
    with pytest.raises(SpecifiedMismatch):
        se.fit(changed, data, basis=basis, check=False)
    applied = se.fit(
        changed, data, basis=basis, rank=se.FullRank(), check=False, allow_mismatch=True
    )
    assert applied.ledger["specified:b"].kind == "unbounded"
    np.testing.assert_allclose(applied.predict().I, data.stokes[:, 0], rtol=2e-13)


def test_r7_preserves_the_phase_of_prebuilt_columns():
    channels = se.Channels.at([1.0, 1.5, 2.0], unit="GHz")
    lattice = se.Lattice(gamma=se.Cells.log(300, 2e4, 2, order=0))
    law = se.PowerLaw(2.6, lo=300, hi=2e4, column=1)
    first = se.Population(
        gamma=law, B="5 uG", depth="2 rad m^-2", phi=se.AzimuthFactor(1 + 0j)
    )
    second = first.replace(B="6 uG")
    basis = se.spectral_basis(
        first + second, channels, lattice=lattice, phase=False, check=False
    )
    data = se.StokesData(
        channels,
        I=[1.0, 2.0, 3.0],
        Q=[-0.5, -1.0, -1.5],
        sigma=0.1,
        unit="cgs",
    )
    free = first.replace(gamma=se.FreeMoments()) + second.replace(
        gamma=se.FreeMoments()
    )
    with pytest.raises(RoleError, match="phase= differs"):
        se.problem(free, data, basis=basis, phase=se.FaradayRotation(), check=False)
    compiled = se.problem(free, data, basis=basis, check=False)
    assert all(part.forward_phase is False for part in compiled.basis.parts)


def test_r7_ties_equivalent_units_and_keeps_quadrature_settings():
    channels = se.Channels.at([1.0, 2.0, 4.0], unit="GHz")
    lattice = se.Lattice(gamma=se.Cells.log(300, 2e4, 2, order=0))
    law = se.PowerLaw(2.6, lo=300, hi=2e4, column=1)
    first = se.Population(gamma=law, B=se.Uniform(4, 6, unit="uG", nodes=3))
    second = first.replace(B=se.Uniform(4, 6, unit="µG", nodes=3))
    third = first.replace(B=se.Uniform(4, 6, unit="µG", nodes=4))
    basis = se.spectral_basis(
        first + second + third, channels, lattice=lattice, check=False
    )
    assert basis.member_indices == ((0, 1), (2,))
    data = se.StokesData(channels, I=[1.0, 2.0, 3.0], sigma=0.1, unit="cgs")
    first, second, third = (
        part.replace(gamma=se.FreeMoments()) for part in (first, second, third)
    )
    problem = se.problem(first + second + third, data, basis=basis, check=False)
    assert problem.record.layout.basis_groups == basis.member_indices
