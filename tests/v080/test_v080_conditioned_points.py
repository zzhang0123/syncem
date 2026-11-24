"""R3a point masses integrate the moving child support on each cell."""

from dataclasses import replace

import numpy as np
import pytest

import syncem as se
from syncem.fitting._conditioned_points import (
    conditioned_point_mass_map,
)
from syncem.values.errors import SizeError


@se.kernel(variables=(se.Variable("x"), se.Variable("y")), stokes=("I",))
def toy(nu, x, y):
    return {"I": nu / 1e9 * (1 + x + y)}


def test_moving_support_joint_rows_and_nonnegative_fit():
    lattice = se.Lattice(
        x=se.Cells.linear(0.2, 1.4, 2, order=2),
        y=se.Cells.from_edges([0, 0.8, 1.55], chart="affine", order=2),
        order=2,
        weights=[0.5, 0.7, 1.2, 3.0],
    )
    ch = se.Channels.at(np.arange(1, lattice.n + 1), unit="GHz")
    basis = se.SpectralBasis.build(ch, lattice, kernel=toy, check=False, cache=False)
    basis = replace(basis, spectra0=np.eye(lattice.n)[:, None, :])
    marker = se.NonNegativeCells(
        "x", cells=slice(1, None), support=(0.9, 1.2), point_masses=3
    )

    def child(x):
        return se.Uniform(x, x + 1)

    with pytest.raises(SizeError, match="map workspace"):
        conditioned_point_mass_map(basis, marker, "y", child, 1024)
    points = conditioned_point_mass_map(basis, marker, "y", child, 2**28)
    x = points.coordinates["x"]
    np.testing.assert_allclose(x, [0.9, 1.05, 1.2])
    raw = points.moments * lattice._row_weights()[:, None]
    for i, (cell, (rx, ry)) in enumerate(
        zip(lattice.index.arrays.cell, lattice.index.arrays.powers)
    ):
        if cell != 3:
            np.testing.assert_array_equal(raw[i], 0)
            continue
        ref, half = lattice.cells.reference[cell], lattice.cells.half_width[cell]
        lo, hi = (x - ref[1]) / half[1], (1.55 - ref[1]) / half[1]
        expected = (
            ((x - ref[0]) / half[0]) ** rx
            * half[1]
            * (hi ** (ry + 1) - lo ** (ry + 1))
            / (ry + 1)
        )
        np.testing.assert_allclose(raw[i], expected, atol=1e-14)

    truth = raw @ [1.0, 2.0, 1.0]
    data = se.StokesData(ch, I=truth, sigma=0.01, unit="cgs")
    fit = se.fit(
        se.Population(marker, y=se.Conditional(child)),
        data,
        basis=basis,
        penalty=se.Penalty(0),
        check=False,
    )
    np.testing.assert_allclose(fit.predict().I, truth, atol=1e-12)
    assert fit.population.variables == ("x",)
    assert fit.source_column.value == pytest.approx(4.0)
    assert fit.column.value < fit.source_column.value
    assert fit.ledger["conditional_truncation:y|x"].kind == "not_applicable"


def test_electron_family_contributes_unit_probability():
    lattice = se.Lattice(
        x=se.Cells.linear(1, 2, 1, order=0),
        y=se.Cells.log(300, 2000, 2, order=0),
        order=0,
    )
    basis = se.SpectralBasis.build(
        se.Channels.at([1.0], unit="GHz"), lattice, kernel=toy, check=False, cache=False
    )
    marker = se.NonNegativeCells("x", point_masses=2)

    def electrons(x):
        return se.PowerLaw(2.6, lo=400, hi=1900, column=50 * x)

    points = conditioned_point_mass_map(basis, marker, "y", electrons, 2**28)
    raw = points.moments * lattice._row_weights()[:, None]
    np.testing.assert_allclose(raw.sum(axis=0), [1, 1], atol=1e-13)
