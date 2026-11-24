"""Cellwise hybrid columns, physical switch probes and prediction (SPEC Sec. 8.4)."""

import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.harmonic import HarmonicKernel
from syncem.base.hybrid import HybridKernel
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.distributions import Uniform
from syncem.engine._hybrid_basis import HybridBasis, build_hybrid_from_averaged
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import build_from_averaged
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.population import Population
from syncem.values.errors import RegimeError, RoleError

_VARIABLES = (Variable("x"), Variable("z"))


@kernel(variables=_VARIABLES, stokes=("I", "P", "V"))
def left(nu, x, z):
    intensity = nu / 1e9 * (2 + x + z)
    return {"I": intensity, "P": intensity * (0.2 + 0.1j), "V": intensity * 0.05}


def other(fraction, profile="constant"):
    @kernel(variables=_VARIABLES, stokes=("I", "P"))
    def right(nu, x, z):
        response = {
            "constant": 1.0,
            "face": 1 - (x - 2) ** 2,
            "ends": (x - 2) ** 2,
        }[profile]
        intensity = (1 + fraction * response) * nu / 1e9 * (2 + x + z)
        return {"I": intensity, "P": intensity * (0.2 + 0.1j)}

    return right


def build(hybrid, lattice, channels, *, fixed=None, cache=False):
    averaged = AveragedKernel(hybrid, fixed={"z": 0.4} if fixed is None else fixed)
    return build_hybrid_from_averaged(averaged, channels, lattice, cache=cache)


@pytest.fixture(scope="module")
def experiment():
    channels = Channels.at([1.0, 2.0, 3.0], unit="GHz")
    lattice = Lattice(x=Cells.linear(1, 3, 2, order=1), weights=[0.9, 1.3])
    return channels, lattice


def test_static_columns_match_each_child_and_predict_fixed_population(experiment):
    channels, lattice = experiment
    right = other(5e-4)
    hybrid = HybridKernel(left, right, switch={"x": 2})
    joined = build(hybrid, lattice, channels)
    assert isinstance(joined, HybridBasis)
    assert joined.lattice.key == lattice.key
    assert joined.record.stokes == ("I", "Q", "U")
    assert joined.averaged.kernel.assignment == (0, 1)
    assert tuple(r.kernel_index for r in joined.record.routes) == (0, 1)
    assert all(route.mode == "hybrid" for route in joined.record.routes)
    assert dict(joined.record.agreement)["hybrid comparisons"] == 3

    children = [
        build_from_averaged(
            AveragedKernel(k, fixed={"z": 0.4}),
            channels,
            lattice,
            stokes=("I", "P"),
            routes="jet",
            cache=False,
        )
        for k in (left, right)
    ]
    for cell, child in enumerate(children):
        rows = list(lattice.index.cell_rows(cell))
        np.testing.assert_array_equal(
            np.asarray(joined.design())[..., rows],
            np.asarray(child.design())[..., rows],
        )
    switch = joined.ledger["kernel_switch"]
    ratio = 5e-4 / 1.0005
    assert switch.kind == "measured" and switch.relative_to
    np.testing.assert_allclose(switch.value[:, 0], ratio, atol=1e-15)
    np.testing.assert_allclose(
        switch.value[:, 1:], np.tile([0.2 * ratio, 0.1 * ratio], (3, 1)), atol=1e-15
    )

    population = Population(x=Uniform(1, 3), z=0.4)
    pred = joined.predict(population, check=False)
    direct = joined.direct(population, within="cells")
    expected = np.array([1.0, 2.0, 3.0]) * (3.9 + 4.9 * 1.0005) / 2
    np.testing.assert_allclose(pred.I, expected, rtol=1e-13)
    np.testing.assert_allclose(direct.stokes, pred.stokes, rtol=1e-13)


def test_switch_probes_the_face_and_extreme_cells(experiment):
    channels, lattice = experiment
    for profile in ("face", "ends"):
        hybrid = HybridKernel(left, other(0.02, profile), switch={"x": 2})
        with pytest.warns(RuntimeWarning, match="above switch_tol"):
            made = build(hybrid, lattice, channels)
        assert dict(made.record.agreement)[
            "maximum kernel switch / I"
        ] == pytest.approx(0.02 / 1.02)
    forbidden = HybridKernel(left, other(0.2, "face"), switch={"x": 2})
    with pytest.raises(RegimeError, match="hybrid kernel switch"):
        build(forbidden, lattice, channels)


def test_reuse_channels_order_and_assignment_in_key(experiment):
    channels, lattice = experiment
    right = other(5e-4)
    hybrid = HybridKernel(left, right, assign=lambda box: int(box["x"][0] >= 2))
    original = build(hybrid, lattice, channels, cache=True)
    repeat = build(hybrid, lattice, channels, cache=True)
    swapped = build(
        HybridKernel(left, right, assign=lambda box: int(box["x"][1] <= 2)),
        lattice,
        channels,
        cache=True,
    )
    assert original.key and original.key == repeat.key and original.key != swapped.key
    assert original.lattice.key == swapped.lattice.key
    assert dict(repeat.record.events)["memo hits"] == 1
    changed_tol = build(
        HybridKernel(left, right, assign=hybrid.assign, switch_tol=0.02),
        lattice,
        channels,
        cache=True,
    )
    assert changed_tol.key == original.key
    assert dict(changed_tol.record.defaults)["switch_tol"] == 0.02

    requested = Channels.at([3.0, 4.0, 1.0, 4.0], unit="GHz")
    changed = original.at(requested)
    assert isinstance(changed, HybridBasis)
    assert dict(changed.record.events)["channel rows computed"] == 1
    np.testing.assert_array_equal(
        np.asarray(changed.design())[[0, 2], ...],
        np.asarray(original.design())[[2, 0], ...],
    )
    np.testing.assert_array_equal(changed.design()[1, ...], changed.design()[3, ...])
    np.testing.assert_allclose(
        changed.ledger["kernel_switch"].value[:, 0], 5e-4 / 1.0005
    )
    reduced = changed.with_order(0)
    assert isinstance(reduced, HybridBasis)
    np.testing.assert_array_equal(
        reduced.design(), np.asarray(changed.design())[..., [0, 2]]
    )


def test_general_cell_table_and_invalid_assignment(experiment):
    channels, _ = experiment
    tensor = Lattice(x=Cells.linear(1, 3, 2, order=0), z=Cells.linear(0, 1, 2, order=0))
    general = Lattice.from_cells(tensor.cells.rows([2, 1, 0]))
    made = build(
        HybridKernel(left, other(5e-4), assign=lambda box: int(box["x"][0] >= 2)),
        general,
        channels,
        fixed={},
    )
    assert made.averaged.kernel.assignment == (1, 0, 0)
    assert dict(made.record.agreement)["hybrid comparisons"] >= 3
    crossed = HybridKernel(left, other(0), switch={"x": 1.8})
    with pytest.raises(RoleError, match="split the cell"):
        build(crossed, general, channels, fixed={})
    with pytest.raises(RoleError, match="must return an index"):
        build(
            HybridKernel(left, other(0), assign=lambda box: 2),
            general,
            channels,
            fixed={},
        )
    with pytest.raises(RoleError, match="traceable non-harmonic"):
        HybridKernel(HarmonicKernel(), left, assign=lambda box: 0)


def test_public_prediction_and_unassigned_support(experiment):
    import syncem as se

    channels, lattice = experiment
    model = se.HybridKernel(left, other(0.0), assign=lambda box: int(box["x"][0] >= 2))
    population = Population(x=Uniform(0.9, 3), z=0.4)
    basis = se.spectral_basis(
        population, channels, lattice=lattice, kernel=model, cache=False
    )
    assert isinstance(basis, HybridBasis)
    assert tuple(route.kernel_index for route in basis.record.routes) == (0, 1)
    prediction = se.predict(population, channels, basis=basis)
    direct = basis.direct(population, within="cells")
    np.testing.assert_allclose(prediction.stokes, direct.stokes, atol=1e-13)
    assert prediction.ledger["kernel_switch"].kind == "measured"
    assert prediction.ledger["outside_lattice"].kind == "unbounded"
    assert "outside its cells" in prediction.ledger["outside_lattice"].note
