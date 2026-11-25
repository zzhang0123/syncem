"""Rank and noise tests against normalized probability measures."""

import numpy as np
import pytest

import syncem as se


@se.family(variable="x", support=(-1.0, 1.0))
def normalized_density(x, scale, slope, unit, raw_scale):
    return raw_scale * (1 + scale) * (1 + (slope / unit) * x)


@se.kernel(variables=(se.Variable("x"),), stokes=("I", "V"))
def weak_stokes_signal(nu, x):
    return _stokes_values(nu, x)


@se.kernel(variables=(se.Variable("x"), se.Variable("h")), stokes=("I", "V"))
def joint_weak_stokes_signal(nu, x, h):
    return _stokes_values(nu, x + 0 * h)


def _stokes_values(nu, x):
    t = nu / 1e9
    return {"I": 1e-26 * (2 + (0.3 + 0.25 * x) * t), "V": 1e-26 * (0.25 * t - 1.5 * x)}


@pytest.mark.parametrize("compressed", [False, True])
@pytest.mark.parametrize(
    "unit, joint, raw_scale",
    [(1.0, False, 1.0), (1e-24, False, 1e-200), (1e24, False, 1e200), (1.0, True, 1.0)],
)
def test_density_normalization_cancels_in_fisher_matrix_and_predictions(
    compressed, unit, joint, raw_scale
):
    t = np.array([0.6, 0.75, 1.2, 2.0, 3.4])
    ch = se.Channels.at(t, unit="GHz")
    slope = 0.65
    baseline = np.stack((2 + 0.3 * t, 0.25 * t), axis=-1)
    # For the normalized linear law E[x] = slope / 3 in these units.
    shape_response = np.stack((0.25 * t, np.full_like(t, -1.5)), axis=-1) / 3
    seed = np.random.default_rng(759)
    mixing = seed.normal(size=(2 * len(t), 2 * len(t)))
    cov = 1e-56 * (np.eye(2 * len(t)) + mixing @ mixing.T)
    response = np.eye(2 * len(t)) + 0.08 * seed.normal(size=cov.shape)
    y = response @ (1e-26 * (baseline + slope * shape_response)).ravel()
    data = se.StokesData(
        ch,
        stokes=y.reshape(-1, 2),
        components=("I", "V"),
        cov=cov,
        response=response,
        unit="cgs",
    )
    cells = {"x": se.Cells.linear(-1, 1, 2, order=1)}
    if joint:
        cells["h"] = se.Cells.linear(0, 1, 1, order=0)
    lattice = se.Lattice(**cells, weights=[1, 1])
    source = se.Population(
        x=normalized_density(
            scale=se.Free(0.19, bounds=(-0.9, 0.9)),
            slope=se.Free(0.2 * unit, bounds=(0.01 * unit, 0.9 * unit), log=True),
            unit=unit,
            raw_scale=raw_scale,
        )
    )
    if joint:
        source = source.replace(h=se.Uniform(0, 1))
    basis = se.spectral_basis(
        source.at({"x.scale": 0.19, "x.slope": 0.2 * unit}),
        ch,
        lattice=lattice,
        kernel=joint_weak_stokes_signal if joint else weak_stokes_signal,
        check=False,
    )
    target = data
    if compressed:
        target = se.fit(
            se.Population(se.FreeMoments(*cells)),
            data,
            basis=basis,
            rank=se.FullRank(),
            keep_directions=True,
            check=False,
        ).retained
    result = se.fit(source, target, basis=basis, check=False)
    assert result.rank == 1
    assert not result.params["x.scale"].identified
    assert np.isnan(result.params["x.scale"].sigma)
    b = result.params["x.slope"]
    assert b.identified
    np.testing.assert_allclose(b.value / unit, slope, rtol=1e-8)
    signal = 1e-26 * shape_response.ravel()
    observed = response @ signal
    variance = 1 / (observed @ np.linalg.solve(cov, observed))
    np.testing.assert_allclose(b.sigma / unit, np.sqrt(variance), rtol=1e-8)
    np.testing.assert_allclose(
        result.predict(cov=True).cov,
        variance * np.outer(signal, signal),
        rtol=1e-8,
        atol=0,
    )


@se.family(variable="x", support=(0, 1), cutoff=lambda edge: edge)
def scaled_uniform(x, scale, edge):
    return (1 + scale) + 0 * x


def test_density_normalization_preserves_motion_of_a_boundary():
    @se.kernel(variables=(se.Variable("x"),), stokes=("I",))
    def boundary_probe(nu, x):
        return {"I": 2 + (nu / 1e9) * x}

    t = np.array([0.5, 1, 2])
    ch = se.Channels.at(t, unit="GHz")
    source = se.Population(
        x=scaled_uniform(
            scale=se.Free(0.3, bounds=(0, 1)), edge=se.Free(0.4, bounds=(0.1, 0.9))
        )
    )
    data = se.StokesData(ch, I=2 + t * 0.7 / 2, sigma=0.01, unit="cgs")
    lattice = se.Lattice(x=se.Cells.linear(0, 1, 2, order=1), weights=[1, 1])
    fit = se.fit(source, data, lattice=lattice, kernel=boundary_probe, check=False)
    assert fit.rank == 1
    assert not fit.params["x.scale"].identified
    np.testing.assert_allclose(fit.params["x.edge"].value, 0.7)
    np.testing.assert_allclose(fit.params["x.edge"].sigma, 0.02 / np.linalg.norm(t))


@pytest.mark.parametrize("x0", [0.0, -1.0])
def test_small_resolvable_slope_survives_host_difference_roundoff(x0):
    from syncem.fitting._finite_difference import _jacobian

    coord = np.array([x0])

    def response(p):
        return np.array([1 + 3e-10 * p[0], 1e-26 * (1 + 3e-10 * p[0])])

    jac, _ = _jacobian(
        response, coord, np.array([-1.0]), np.array([1.0]), np.cbrt(np.finfo(float).eps)
    )
    np.testing.assert_allclose(jac[:, 0], [3e-10, 3e-36], rtol=0.01, atol=0)
