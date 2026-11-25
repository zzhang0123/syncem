"""Identifiable combinations of otherwise degenerate physical parameters."""

import numpy as np
import pytest

import syncem as se


@se.family(variable="x", support=(-1.0, 1.0))
def polynomial(x, a, b=0.0):
    return 1 + a * x + b * (3 * x**2 - 1) / 2


def intensity(nu, x):
    return 2 + 3 * x + 7.5 * (nu / 1e9) * (x * x - 1 / 3)


@se.kernel(variables=(se.Variable("x"),), stokes=("I",))
def primary(nu, x):
    return {"I": intensity(nu, x)}


@se.kernel(variables=(se.Variable("x"), se.Variable("y")), stokes=("I",))
def coupled(nu, x, y):
    return {"I": intensity(nu, x) + (nu / 1e9) * y}


@pytest.mark.parametrize(
    "route, compressed",
    [
        ("primary", False),
        ("primary", True),
        ("joint", False),
        ("joint", True),
        ("rebuilt", False),
    ],
)
def test_identification_uses_the_same_response_in_likelihood_and_forecast(
    route, compressed
):
    # At the measured frequency Stokes I is 2+a+b (or 2+a+y).
    # Each other frequency weights the two free coefficients differently.
    free = se.Free(0.1, bounds=(-0.4, 0.4))
    y = (
        {"y": free if route == "rebuilt" else se.Uniform(-1.0, 1.0)}
        if route != "primary"
        else {}
    )
    pop = se.Population(
        x=polynomial(a=free, b=0.0 if route == "rebuilt" else free), **y
    )
    cells = {"x": se.Cells.linear(-1, 1, 1, order=2)}
    if route == "joint":
        cells["y"] = se.Cells.linear(-1, 1, 1, order=0)
    lattice = se.Lattice(**cells, weights=[1])
    kernel = primary if route == "primary" else coupled
    channels = se.Channels.at([1.0], unit="GHz")
    sigma = 1e-9
    data = se.StokesData(channels, I=[2.3], sigma=sigma, unit="cgs")
    target = data
    options = {"lattice": lattice, "kernel": kernel}
    if compressed:
        fixed = pop.at({"x.a": 0.1, "x.b": 0.1})
        basis = se.spectral_basis(
            fixed, channels, lattice=lattice, kernel=kernel, check=False
        )
        target = se.fit(
            se.Population(se.FreeMoments(*cells)),
            data,
            basis=basis,
            rank=se.FullRank(),
            keep_directions=True,
            check=False,
        ).retained
        options = {"basis": basis}
    fit = se.fit(pop, target, check=False, **options)
    assert fit.rank == 1
    assert not any(p.identified for p in fit.params.values())
    out = fit.predict(se.Channels.at([1.0, 1.00005, 1.002], unit="GHz"), cov=True)
    np.testing.assert_allclose(out.sigma[0, 0], sigma, rtol=3e-10)
    np.testing.assert_allclose(out.cov[0, 0], sigma**2, rtol=3e-10)
    assert np.isnan(out.sigma[1:]).all()
    assert np.isnan(out.cov[1:]).all()
    if route == "rebuilt":
        # y enters through the basis columns. Its response is essential even
        # though the physical moments depend only on a.
        assert fit.kind == "basis-parametric"
        assert np.isnan(fit.observable([0, 3, 0]).sigma)
    else:
        t = np.array([1.0, 1.00005, 1.002])
        rows = np.stack((-2.5 * t, 3 * np.ones_like(t), 7.5 * t), axis=-1)
        observable = fit.observable(rows)
        np.testing.assert_allclose(observable.sigma, out.sigma[:, 0], equal_nan=True)


def test_estimability_is_invariant_under_extreme_physical_parameter_units():
    from syncem.fitting._family_noise import FamilyNoise

    # The two parameter units differ by 360 decades. Their effects are
    # indistinguishable in the single observed combination.
    row = np.array([[1e180, 1e-180]])
    noise = FamilyNoise.from_response(row, np.ones(2))
    sigma, _ = noise.propagate(np.vstack((row, row * [1, 1.00005])), (2,))
    np.testing.assert_allclose(sigma[0], 1.0, rtol=2e-14)
    assert np.isnan(sigma[1])
