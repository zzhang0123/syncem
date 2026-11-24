"""Direct panels, moving dependencies and the continuum continuation."""

import jax.numpy as jnp
import numpy as np
import pytest

import syncem as se
from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.kernel import Modes
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.base.window import A1_HZ_PER_G
from syncem.engine.average import AveragedKernel, NodeRule
from syncem.engine.direct import (
    TAIL_X,
    continuum_tail_thresholds,
    continuum_tail_threshold,
    integrate,
    node_modes,
    sum_modes,
)


def _channels():
    return Channels.at([0.55, 1.37], unit="GHz", half_width=0.1, n_nu=6)


def test_probe_quadrature_reports_the_doubled_band_and_law_difference():
    @se.kernel(variables=(se.Variable("x"), se.Variable("y")), stokes=("I",))
    def model(nu, x, y):
        return {"I": (1 + x) * (1 + y**6) * (nu / 1e9) ** 4}

    ch = se.Channels.at([1, 2], unit="GHz", half_width=0.2, n_nu=2)
    high = se.Channels.at([1, 2], unit="GHz", half_width=0.2, n_nu=4)
    single = se.StokesData(
        se.Channels.at([3], unit="GHz"), I=[1.0], sigma=0.1, unit="cgs"
    )
    ch, high = (
        (se.StokesData(bands, I=[1.0, 1.0], sigma=0.1, unit="cgs") + single).channels
        for bands in (ch, high)
    )
    source = se.Population(x=se.Uniform(0, 1), y=se.Uniform(0, 1, nodes=2))
    lattice = se.Lattice(x=se.Cells.linear(0, 1, 3, order=1))
    basis = se.spectral_basis(source, ch, kernel=model, lattice=lattice, check=False)
    pred = basis.predict(source, check=se.AccuracyCheck(max_seconds=30))
    term = pred.ledger["quadrature"]
    assert term.kind == "estimate"
    refined = source.replace(y=se.Uniform(0, 1, nodes=4))
    difference = basis.at(high).direct(refined).I - basis.direct(source).I
    np.testing.assert_allclose(term.value[:, 0], abs(difference), rtol=1e-10)
    assert np.all(term.value > 0)


def test_basis_cell_selection_reuses_spectra_and_whitens_its_design():
    @se.kernel(variables=(se.Variable("x"),), stokes=("I",))
    def model(nu, x):
        return {"I": 2 + x + (nu / 1e9) * x**2}

    ch = se.Channels.at([1, 2], unit="GHz")
    pop = se.Population(x=se.Uniform(0, 1))
    lattice = se.Lattice(x=se.Cells.linear(0, 1, 3, order=2))
    parent = se.spectral_basis(pop, ch, lattice=lattice, kernel=model, check=False)
    selected = parent.select(x=slice(1, None))
    assert selected.lattice.n_cells == 2
    assert "1 original product cells" in selected.ledger["cell_selection"].note
    np.testing.assert_array_equal(selected.design(), parent.design()[..., 3:])
    data = se.StokesData(ch[1:], I=[3.0], sigma=0.25, unit="cgs")
    np.testing.assert_array_equal(selected.design(data), selected.design()[1, :1] * 4)
    fit = selected.fit(se.Population(x=se.FreeMoments()), data, check=False)
    assert fit.moments.key == selected.lattice.key
    checked = selected.predict(pop, check=True)
    assert float(checked.ledger["outside_lattice"].value[0, 0]) < 0


def test_direct_values_include_node_azimuth_and_a_moving_child():
    @kernel(
        variables=[Variable(n) for n in ("x", "y", "a")],
        stokes=("I", "P", "V"),
        azimuth="a",
    )
    def model(nu, x, y):
        return {"I": x + 2 * y, "P": y + 0j, "V": x - y}

    def moving_child(point):
        values = jnp.stack((point["x"], 2 * point["x"]))
        return values, jnp.array([0.75, 0.25])

    avg = AveragedKernel(model, [("y", NodeRule(moving_child, ("x",)))])
    x = np.array([1.0, 2.0, 1.0])
    a = np.array([0.0, 0.2, np.pi / 2])
    weights = np.array([1.0, 3.0, 2.0])
    points = {"x": x, "a": a}
    channels = _channels()
    values = node_modes(avg, channels, points)
    gain = np.sum(channels.response_weights(), axis=-1)
    np.testing.assert_allclose(values.I[:, 0], 3.5 * x * gain[0], rtol=2e-15)
    np.testing.assert_allclose(values.V[:, 0], -0.25 * x * gain[0], rtol=2e-15)
    np.testing.assert_allclose(
        values.P[:, 0, 0], 1.25 * x * np.exp(2j * a) * gain[0], rtol=2e-15
    )

    left = {k: v[:1] for k, v in points.items()}
    right = {k: v[1:] for k, v in points.items()}
    actual = integrate(avg, channels, [(left, weights[:1]), (right, weights[1:])])
    expected = sum_modes(values, weights)
    np.testing.assert_allclose(actual.I, 31.5 * gain, rtol=2e-15)
    for measured, target in zip(actual, expected, strict=True):
        np.testing.assert_allclose(measured, target, rtol=2e-15)

    masked = sum_modes(values, weights, mask=[True, False, True])
    np.testing.assert_allclose(masked.I, 10.5 * gain, rtol=2e-15)
    np.testing.assert_allclose(masked.P, (-1.25 * gain)[None, :], rtol=2e-15)


def test_direct_keeps_azimuth_when_a_rule_depends_on_it():
    @kernel(
        variables=[Variable(n) for n in ("y", "a")],
        stokes=("I", "P"),
        azimuth="a",
    )
    def model(nu, y):
        return {"I": y, "P": y}

    avg = AveragedKernel(
        model,
        [("y", NodeRule(lambda q: (jnp.atleast_1d(q["a"]), jnp.ones(1)), ("a",)))],
    )
    a = np.array([0.4, 0.9])
    channels = _channels()
    values = node_modes(avg, channels, {"a": a})
    gain = np.sum(channels.response_weights()[0])
    np.testing.assert_allclose(values.I[:, 0], a * gain, rtol=2e-15)
    np.testing.assert_allclose(values.P[:, 0, 0], a * np.exp(2j * a) * gain, rtol=2e-15)


def test_empty_and_zero_weight_panels():
    @kernel(variables=[Variable("x")], stokes=("I",))
    def model(nu, x):
        return {"I": x}

    averaged = AveragedKernel(model)
    channels = _channels()
    for panels in ([], [({"x": np.array([2.0])}, np.zeros(1))]):
        result = integrate(averaged, channels, panels)
        assert result.I.shape == (2,)
        assert result.P.shape == (1, 2)
        for value in result:
            np.testing.assert_array_equal(value, 0)
    original = Modes(np.array([[3.0], [np.nan]]), np.zeros((2, 1)), np.zeros((2, 1, 1)))
    np.testing.assert_array_equal(
        sum_modes(original, [1, 1], mask=[True, False]).I, [3]
    )


def test_continuum_tail_at_transition_and_in_the_far_tail():
    """Frequency-node thresholds at the local B_perp (SPEC Sec. 8.3, decision 198)."""
    avg = AveragedKernel(ContinuumKernel(x_min=1e-7), fixed={"B": 6e-6, "eta": 0.0})
    channels = _channels()
    bound = 6e-6
    # The kernel is lowered to x_tail = min(x_min, 1e-9) on the direct path.
    lowered = AveragedKernel(
        ContinuumKernel(x_min=TAIL_X), fixed={"B": 6e-6, "eta": 0.0}
    )
    thresholds = continuum_tail_thresholds(avg.kernel, channels, bound)
    assert continuum_tail_threshold(avg.kernel, channels, bound) == thresholds.min()
    below = np.array([0.5, 0.9]) * thresholds.min()
    raw = node_modes(lowered, channels, {"gamma": below})
    kept = node_modes(avg, channels, {"gamma": below}, b_perp_max=bound)
    for a, b in zip(raw, kept, strict=True):
        np.testing.assert_allclose(b, a, rtol=1e-15, atol=0)
    multiples = np.array([10.0, 1e5])
    far = node_modes(
        avg, channels, {"gamma": multiples * thresholds.max()}, b_perp_max=bound
    )
    # Each node continues its own value at the join. Include the response
    # weights before summing the independent node references across the band.
    node_threshold = np.sqrt(
        np.asarray(channels.nodes) / (A1_HZ_PER_G * bound * TAIL_X)
    )
    reference = ContinuumKernel(x_min=1e-12).channel(
        channels,
        {"gamma": node_threshold, "B": bound, "eta": 0.0},
        phase=None,
        per_node=True,
    )
    for n, factor in enumerate(multiples):
        scale = (node_threshold / (factor * thresholds.max())) ** (2 / 3)
        np.testing.assert_allclose(
            far.I[n], np.sum(reference.I * scale, axis=-1), rtol=4e-15
        )
        np.testing.assert_allclose(
            far.P[n], np.sum(reference.P * scale, axis=-1), rtol=4e-15
        )
    # Check the approximation against the callable kernel on a common range.
    smaller = AveragedKernel(
        ContinuumKernel(x_min=1e-12), fixed={"B": 6e-6, "eta": 0.0}
    )
    gamma = np.array([10 * thresholds.max()])
    direct = node_modes(smaller, channels, {"gamma": gamma})
    tail = node_modes(avg, channels, {"gamma": gamma}, b_perp_max=bound)
    np.testing.assert_allclose(direct.I[0], tail.I[0], rtol=2e-6)
    np.testing.assert_allclose(direct.P[0], tail.P[0], rtol=2e-6)
    assert continuum_tail_threshold(avg.kernel, channels, 0) is None
    with pytest.raises(ValueError, match="field in G"):
        continuum_tail_threshold(avg.kernel, channels, -1)
