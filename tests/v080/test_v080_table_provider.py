"""The 17-node table, derivative check and field-scaling reference."""

import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se
from syncem.base.capabilities import FieldScalingRelation
from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine.average import AveragedKernel, NodeRule
from syncem.engine.cache import default_cache
from syncem.engine.direct import node_modes
from syncem.engine.provider_table import (
    TableRefused,
    build_cell_table,
    check_tables,
    scaling_reference,
)
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice


def _channels():
    return Channels.at([0.6, 1.0], unit="GHz", half_width=0.1, n_nu=8)


def _gain(channels):
    return np.sum(channels.response_weights(), axis=-1)


def test_sampled_remainder_uses_the_highest_known_even_moment():
    @se.kernel(variables=(se.Variable("x"),), stokes=("I",))
    def cubic(nu, x):
        return {"I": (nu / 1e9) * (3 + (x - 1) ** 3)}

    ch = se.Channels.at([1, 2], unit="GHz")
    lattice = se.Lattice(x=se.Cells.linear(0, 2, 1, order=2))
    source = se.Population(x=se.Uniform(0, 2))
    check = se.AccuracyCheck(max_seconds=30, providers=True)
    basis = se.spectral_basis(source, ch, lattice=lattice, kernel=cubic, check=check)
    np.testing.assert_allclose(basis.remainder[:, 0, 0], [1, 2], rtol=1e-7)
    result = basis.predict(source, check=check)
    assert result.ledger["remainder"].kind == "estimate"
    np.testing.assert_allclose(
        result.ledger["remainder"].value[:, 0], np.array([1, 2]) / 3, rtol=1e-7
    )
    supplied = se.Moments(basis.lattice, [1.0, 0.0, 0.0], origin="minimum_norm")
    assert basis.predict(supplied, check=check).ledger["remainder"].kind == "unbounded"
    assert basis.at(ch[:1]).remainder is not None
    fresh_channel = se.Channels.at([3], unit="GHz")
    assert basis.at(fresh_channel).remainder is None


def test_two_dimensional_table_uses_faces_and_derivatives_at_shifted_references():
    @kernel(variables=[Variable("a"), Variable("b")], stokes=("I", "P"))
    def model(nu, a, b):
        coord = jnp.log(a)
        return {"I": coord * (1 + b**2), "P": (coord + 2j) * b}

    cell_a = Cells.log(1.0, 3.0, 1, order=2).shift(0.25)
    cell_b = Cells.linear(-2.0, 2.0, 1, order=2).shift(-0.25)
    lattice = Lattice(a=cell_a, b=cell_b)
    channels = _channels()
    avg = AveragedKernel(model)
    table = build_cell_table(avg, channels, lattice, 0, cache=False)

    a = np.array([1.0, 1.8, 3.0])
    b = np.array([2.0, -0.6, -2.0])
    values = table.evaluate({"a": a, "b": b})
    gain = _gain(channels)
    np.testing.assert_allclose(
        values.I, (np.log(a) * (1 + b**2))[:, None] * gain, atol=3e-14
    )
    np.testing.assert_allclose(
        values.P[:, 0], ((np.log(a) + 2j) * b)[:, None] * gain, atol=3e-14
    )
    ac, bc = np.log(cell_a.reference[0]), cell_b.reference[0]
    ha, hb = cell_a.half_width[0], cell_b.half_width[0]
    coefficients = table.columns(((0, 0), (1, 1), (0, 2)))
    expected = np.array([ac * (1 + bc**2), 2 * ha * hb * bc, ac * hb**2])
    np.testing.assert_allclose(coefficients.I, expected[:, None] * gain, atol=2e-12)
    assert check_tables(avg, channels, lattice, [table]).passed
    with pytest.raises(TableRefused, match="inside"):
        table.evaluate({"a": [0.9], "b": [0.0]})


def test_table_uses_process_cache_and_rechecks_moving_rule_values():
    @kernel(variables=[Variable("x"), Variable("y")], stokes=("I",))
    def model(nu, x, y):
        return {"I": x * y}

    scale = [1.0]

    def moving(q):
        return jnp.atleast_1d(scale[0] + q["x"]), jnp.ones(1)

    avg = AveragedKernel(model, [("y", NodeRule(moving, ("x",)))])
    lattice = Lattice(x=Cells.linear(1.0, 2.0, 1, order=2))
    channels = _channels()
    first = build_cell_table(avg, channels, lattice, 0)
    again = build_cell_table(avg, channels, lattice, 0)
    cache = default_cache()
    assert cache.events["misses"] == 1
    assert cache.events["hits"] == 1
    assert cache.events["disk_writes"] == 0
    np.testing.assert_array_equal(first.coefficients.I, again.coefficients.I)

    scale[0] = 2.0
    edited = build_cell_table(avg, channels, lattice, 0)
    assert cache.events["misses"] == 2
    np.testing.assert_allclose(
        edited.evaluate({"x": [1.5]}).I,
        (1.5 * 3.5 * _gain(channels))[None, :],
        rtol=4e-15,
    )
    assert check_tables(avg, channels, lattice, [edited]).passed


@pytest.mark.parametrize(
    "half_width, ref, expected",
    [(0.242, 4e3, True), (0.92, 4e3, False), (4.60, 5e4, False)],
)
def test_table_agreement_on_narrow_and_wide_continuum_cells(half_width, ref, expected):
    lattice = Lattice(
        gamma=Cells.log(
            float(ref * np.exp(-half_width)),
            float(ref * np.exp(half_width)),
            1,
            order=2,
        )
    )
    channels = _channels()
    avg = AveragedKernel(ContinuumKernel(x_min=1e-12), fixed={"B": 5e-6, "eta": 0.4})
    cell = build_cell_table(avg, channels, lattice, 0, cache=False)
    agreement = check_tables(avg, channels, lattice, [cell])
    assert agreement.passed == expected, agreement
    at = cell.evaluate({"gamma": np.array([ref])})
    target = node_modes(avg, channels, {"gamma": np.array([ref])})
    np.testing.assert_allclose(at.I, target.I, rtol=6e-14)
    np.testing.assert_allclose(at.P, target.P, rtol=6e-14)


def test_scaling_tables_match_directly_evaluated_field_nodes():
    variables = [Variable("g", kind="positive"), Variable("b", kind="positive")]

    @kernel(
        variables=variables,
        stokes=("I", "P"),
        capabilities=(FieldScalingRelation("b", "g"),),
    )
    def model(nu, g, b):
        return {"I": b * (1 + g * jnp.sqrt(b)), "P": 0.2j * b * g * jnp.sqrt(b)}

    b0 = 1.0
    fixed = AveragedKernel(model, fixed={"b": b0})
    lattice = Lattice(g=Cells.linear(1.0, 5.0, 4, order=2))
    channels = _channels()
    cells = [
        build_cell_table(fixed, channels, lattice, i, cache=False) for i in range(4)
    ]
    assert check_tables(fixed, channels, lattice, cells).passed
    reference = scaling_reference(fixed, cells)
    points = {"g": np.array([2.0, 2.0, 2.5, 2.0]), "b": np.array([0.25, 1.0, 4.0, 0.0])}
    scaled = reference.evaluate(points)
    direct = node_modes(AveragedKernel(model), channels, points)
    for observed, known in zip(scaled, direct, strict=True):
        np.testing.assert_allclose(observed, known, atol=2e-13)
    np.testing.assert_allclose(
        scaled.I[:, 0] / _gain(channels)[0], [0.5, 3.0, 24.0, 0.0]
    )
    with pytest.raises(TableRefused, match="covered span"):
        reference.evaluate({"g": [3.0], "b": [4.0]})
    with pytest.raises(TableRefused, match="azimuth laws"):
        scaling_reference(fixed, cells, azimuth_parents=("b",))
    with pytest.raises(TableRefused, match="same fixed-field"):
        scaling_reference(AveragedKernel(model, fixed={"b": 3.0}), cells)
