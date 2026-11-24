"""R3a Legendre products with and without a celled free parent."""

import numpy as np
import pytest

import syncem as se
from syncem.engine.average import AveragedKernel
from syncem.engine.basis import build_from_averaged
from syncem.fitting.conditional_basis import conditional_basis
from syncem.lattice.moments import Moments
from syncem.population import pair

_ETA = se.Variable("eta", kind="cosine")
_X = se.Variable("x")
_Y = se.Variable("y")


@se.family(variable="y", support=(-1, 1))
def skew(y, beta=0.0):
    return 1 + beta * y


@se.kernel(variables=(_X, _ETA, _Y), stokes=("I",))
def template(nu, x, eta, y):
    return {"I": (nu / 1e9) * (3 + x * y * eta)}


def _build(lattice, fixed):
    ch = se.Channels.at([1.0, 3.0], unit="GHz")
    angles = {n: (rep.degree, 25) for n, rep in lattice.reps}
    return build_from_averaged(
        AveragedKernel(template, fixed=fixed, legendre=angles), ch, lattice, cache=False
    )


def _row(index, *, degree, powers):
    return next(
        r
        for r in range(index.n)
        if index.row(r).degrees["eta"] == degree and index.row(r).powers == powers
    )


def test_pure_legendre_keeps_products_with_child_modes_above_l():
    lattice = se.Lattice(
        eta=se.Legendre(2), y=se.Cells.linear(-1, 1, 1, order=1), weights=[0.37]
    )
    joint = _build(lattice, {"x": 1.0})

    def child(eta):
        p4 = (35 * eta**4 - 30 * eta**2 + 3) / 8
        return skew(beta=0.3 * p4)

    basis = conditional_basis(joint, "y", child)
    assert basis.lattice.cells.variables == ()
    assert basis.lattice.index.legendre == (("eta", 2),)
    input_values = [0.0, 0.0, 1.0]
    output = basis.mapping.apply(input_values)
    row = _row(lattice.index, degree=2, powers={"y": 1})
    assert float(output[row] * lattice.weights[0]) == pytest.approx(1 / 35, abs=1e-14)
    estimate = basis.conditional_truncation()
    assert estimate.shape == (1, 2, 1)
    assert np.all(np.isfinite(estimate))
    assert basis.ledger["conditional_truncation:y|eta"].kind == "unbounded"


def test_mixed_legendre_and_taylor_states_match_nested_quadrature():
    lattice = se.Lattice(
        x=se.Cells.linear(1, 3, 1, order=2),
        eta=se.Legendre(3),
        y=se.Cells.linear(-1, 1, 1, order=2),
        order=2,
        weights=[0.41],
    )
    joint = _build(lattice, {})

    def child(x, eta):
        return skew(beta=0.2 * x * eta)

    basis = conditional_basis(joint, "y", child, parent_weights=[0.8])
    source = se.Population(x=se.Uniform(1.1, 2.7), eta=se.Uniform(-0.4, 0.7))
    parent = pair(source, basis.lattice, n=6)
    full = pair(source.replace(y=child), lattice, n=6)
    contracted = basis.mapping.apply(parent.values)
    exact_rows = np.array(
        [
            _row(lattice.index, degree=degree, powers={"x": power, "y": 1})
            for degree in range(3)
            for power in range(2)
        ]
    )
    np.testing.assert_allclose(
        contracted[exact_rows], full.values[exact_rows], atol=4e-14
    )
    reduced = basis.predict(parent, check=False)
    mapped = joint.predict(Moments(lattice, contracted), check=False)
    np.testing.assert_allclose(reduced.stokes, mapped.stokes, rtol=1e-14)
    np.testing.assert_allclose(
        reduced.stokes, joint.predict(full, check=False).stokes, rtol=1e-14
    )
    data = se.StokesData(joint.channels, I=reduced.I, sigma=0.1, unit="cgs")
    fitted = se.fit(
        se.Population(se.FreeMoments("x", "eta"), y=child),
        data,
        basis=joint,
        rank=se.FullRank(),
        check=False,
    )
    assert fitted.moments.origin == "minimum_norm"
    np.testing.assert_allclose(fitted.predict().I, reduced.I, rtol=1e-14)
