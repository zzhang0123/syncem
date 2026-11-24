"""R3a envelopes before data, with a separately labelled fitted-mass estimate."""

import numpy as np
import pytest

import syncem as se
from syncem.engine import _moment_bound
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import build_from_averaged
from syncem.fitting.conditional_basis import conditional_basis
from syncem.lattice.moments import Moments
from syncem.population import pair
from tests.v080.test_v080_conditional_basis import _toy


def _law(x):
    return se.Uniform(0.05, 0.35 + 0.12 * x + 0.015 * x**2)


@pytest.fixture(scope="module")
def mapped():
    lattice = se.Lattice(
        x=se.Cells.log(1, 4, 2, order=2).shift(0.7),
        y=se.Cells.linear(0.1, 0.95, 2, order=2).shift(-0.4),
        weights=[0.2, 0.4, 0.5, 0.8],
    )
    channels = se.Channels.at([0.5, 2, 4], unit="GHz")
    joint = build_from_averaged(AveragedKernel(_toy), channels, lattice, cache=False)
    return conditional_basis(joint, "y", _law, parent_weights=[0.3, 0.6])


def _dense_triangle(lattice, values, cell):
    table = lattice.cells
    faces = np.stack([table.lo[cell], table.hi[cell]]) - table.reference[cell]
    for i, chart in enumerate(table.chart[cell]):
        if chart == "log":
            faces[:, i] = np.log(
                [
                    table.lo[cell, i] / table.reference[cell, i],
                    table.hi[cell, i] / table.reference[cell, i],
                ]
            )
    radius = abs(faces).max(axis=0) / table.half_width[cell]
    rows = lattice.index.cell_rows(cell)
    exponents = lattice.index.arrays.powers[rows]
    scale = np.prod(radius**exponents, axis=-1) / lattice.weights[cell]
    return np.sum(abs(values[..., rows]) * scale, axis=-1)


def test_bound_matches_triangle_formula_and_contains_independent_pairing(mapped):
    parent, joint = mapped.lattice, mapped.joint
    jt = joint.lattice
    bound = mapped.conditional_truncation_bound()
    assert bound.kind == "bound"
    assert bound.relative_to == "unit parent column in each cell"
    assert mapped.ledger[bound.name].kind == "unbounded"
    assert bound.value.shape == (2, 3, 1)
    direct = np.asarray(joint.columns.lift(joint.design()))
    contracted = np.asarray(mapped.design())
    for b in range(parent.n_cells):
        child_cells = [
            c for c in range(jt.n_cells) if jt.cells.cell_numbers(c)["x"] == b
        ]
        expected = _dense_triangle(parent, contracted, b) + np.max(
            [_dense_triangle(jt, direct, c) for c in child_cells], axis=0
        )
        np.testing.assert_allclose(bound.value[b], expected, rtol=2e-12)
        for x in np.linspace(parent.cells.lo[b, 0], parent.cells.hi[b, 0], 13)[1:-1]:
            population = se.Population(x=se.Delta(x))
            fitted = mapped.predict(pair(population, parent), check=False)
            exact = joint.predict(
                pair(population.replace(y=_law), jt, n=32), check=False
            )
            assert np.all(abs(fitted.stokes - exact.stokes) <= bound.value[b])


def test_bound_chunks_and_fitted_mass_estimate(mapped, monkeypatch):
    bound = mapped.conditional_truncation_bound()
    assert mapped.conditional_truncation_bound(chunk_bytes=1).kind == "unbounded"
    monkeypatch.setattr(_moment_bound, "_workspace", lambda *_: 1)
    tiny = mapped.conditional_truncation_bound()
    np.testing.assert_allclose(tiny.value, bound.value, rtol=2e-12)

    lat = mapped.lattice
    rows = lat.index.n_pairs * lat.index.offsets[:-1]
    vectors = np.zeros((2, lat.n))
    weights = np.array([[1.5, -2.5], [0, 4.0]])
    vectors[:, rows] = weights / lat.weights
    moments = Moments(lat, vectors, origin="minimum_norm")
    estimate = mapped.conditional_error(moments)
    assert estimate.kind == "estimate" and estimate.relative_to is None
    np.testing.assert_allclose(
        estimate.value,
        np.einsum("pb,bcs->pcs", abs(weights), bound.value),
        rtol=2e-12,
    )


def test_fitted_mass_diagnostic_accounts_for_batch_output(mapped):
    budget = 24_000
    assert mapped.conditional_truncation_bound(chunk_bytes=budget).kind == "bound"
    values = np.broadcast_to(0.0, (2_000, mapped.lattice.n))
    moments = Moments(mapped.lattice, values, origin="minimum_norm")
    term = mapped.conditional_error(moments, chunk_bytes=budget)
    assert term.kind == "unbounded" and "batch workspace" in term.note


def test_bound_and_fitted_mass_survive_archive_without_declarations(tmp_path, mapped):
    path = tmp_path / "conditional-bound.npz"
    mapped.save(path)
    loaded = se.SpectralBasis.load(path)
    before, after = (basis.conditional_truncation_bound() for basis in (mapped, loaded))
    assert before.kind == after.kind == "bound"
    np.testing.assert_array_equal(before.value, after.value)
    values = Moments(mapped.lattice, np.arange(mapped.lattice.n), origin="minimum_norm")
    np.testing.assert_array_equal(
        mapped.conditional_error(values).value, loaded.conditional_error(values).value
    )
    subset = loaded.at(loaded.channels[[2, 0, 2]])
    np.testing.assert_array_equal(
        subset.conditional_truncation_bound().value,
        np.asarray(after.value)[:, [2, 0, 2]],
    )


def test_unbounded_parent_geometry_stays_unbounded_for_participating_mass():
    lattice = se.Lattice(
        x=se.Cells.single(0.5, 0.3, order=2),
        y=se.Cells.linear(0.1, 0.9, 2, order=2),
    )
    channels = se.Channels.at([1, 3], unit="GHz")
    joint = build_from_averaged(AveragedKernel(_toy), channels, lattice, cache=False)
    basis = conditional_basis(joint, "y", _law)
    assert basis.conditional_truncation_bound().kind == "unbounded"
    active = Moments(basis.lattice, [1.0, 0.0, 0.0], origin="minimum_norm")
    assert basis.conditional_error(active).kind == "unbounded"
    inactive = Moments(basis.lattice, [0.0, 0.0, 0.0], origin="minimum_norm")
    np.testing.assert_array_equal(basis.conditional_error(inactive).value, 0)
