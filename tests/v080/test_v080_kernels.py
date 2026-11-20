"""Step 3 of design/0.8/SPEC.md Sec. 10.2: kernel protocol, capabilities, continuum adapter, ``AveragedKernel``
and re-expansion (Sec. 3.2, 4.6, 8.1, 8.3, T3). The core provider and the review findings of step 3 are in
``test_v080_kernels_core.py``, split from this file in the step-5b-3 review with the test bodies unchanged.

Memory (decision 170): before the split this file peaked at 5.83 and 5.99 GB of resident memory in two fast runs
and at 3.50 GB with ``-m slow`` (2026-11-19, 1-minute load 6.4 to 8.5), and before step 5b-3 at 30.9 GB. After
the split it peaked at 3.41 GB in its fast run (load 7.0 to 6.0) and has no slow tests.
"""

import math
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base import capabilities as C
from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.kernel import Kernel, Modes
from syncem.base.phase import FaradayRotation, PhaseLaw
from syncem.base.window import A1_HZ_PER_G, continuum_window, energy_range
from syncem.engine.average import AveragedKernel, fixed_rule, uniform_rule
from syncem.engine.expand import field_scaling_columns, log_from_affine, stirling2
from syncem.model import ContinuumKernel as ModelContinuum
from syncem.model.phase import GaussianScreen

STUDY_LIB = Path(__file__).resolve().parents[2] / "studies" / "classical-models" / "lib"


@pytest.fixture(scope="module")
def study():
    if str(STUDY_LIB) not in sys.path:
        sys.path.insert(0, str(STUDY_LIB))
    import basis as SB  # noqa: PLC0415
    from config import PAPER  # noqa: PLC0415

    return SB, PAPER


def small_channels(n_nu=64):
    return Channels.log(0.1, 10, 7, unit="GHz", half_width=0.2, n_nu=n_nu)


# ------------------------------------------------------------------ protocol and capabilities


def test_continuum_kernel_meets_the_protocol():
    k = ContinuumKernel()
    assert isinstance(k, Kernel)
    assert tuple(v.name for v in k.variables) == (
        "gamma",
        "B",
        "mu",
        "eta",
        "phi",
        "depth",
    )
    assert (k.primary, k.stokes, k.azimuth, k.depth, k.traceable, k.lines) == (
        "gamma",
        ("I", "P"),
        "phi",
        "depth",
        True,
        False,
    )
    kinds = {type(c) for c in k.capabilities}
    assert kinds == {
        C.Similarity,
        C.FieldScalingRelation,
        C.Beaming,
        C.Parity,
        C.EnergyWindow,
    }
    hash(k.describe())


def test_regime():
    k = ContinuumKernel()
    assert k.regime({"gamma": (300.0, 500.0)}).ok
    refused = k.regime({"gamma": (100.0, 300.0)})
    assert not refused.ok and "gamma_min" in refused.reason
    assert ContinuumKernel(gamma_min=50.0).regime({"gamma": (100.0, 300.0)}).ok


def test_channel_equals_the_model_kernel():
    ch = small_channels()
    point = dict(gamma=3e3, B=5e-6, eta=0.3)
    modes = ContinuumKernel(x_min=1e-12).channel(ch, point)
    ref = ModelContinuum(x_min=1e-12).channel_modes(ch, 3e3, 5e-6, 0.0, 0.3)
    np.testing.assert_allclose(np.asarray(modes.I), np.asarray(ref.I), rtol=1e-14)
    np.testing.assert_allclose(np.asarray(modes.P), np.asarray(ref.P), rtol=1e-14)
    assert isinstance(modes, Modes) and modes.P.shape == (1, 7)


def test_per_node_output_and_phase():
    ch = small_channels()
    k = ContinuumKernel(x_min=1e-12)
    point = dict(gamma=3e3, B=5e-6, eta=0.3, depth=8.0)
    per = k.channel(ch, point, phase=False, per_node=True)
    np.testing.assert_allclose(
        np.asarray(per.I.sum(-1)),
        np.asarray(k.channel(ch, point, phase=False).I),
        rtol=1e-14,
    )
    rotated = k.channel(ch, point, phase=FaradayRotation())
    tau = 2.0 * (299792458.0 / np.asarray(ch.nodes)) ** 2
    manual = np.sum(np.asarray(per.P[0]) * np.exp(1j * tau * 8.0), axis=-1)
    np.testing.assert_allclose(np.asarray(rotated.P[0]), manual, rtol=1e-12)
    screened = k.channel(
        ch,
        dict(point, depth=0.0),
        phase=FaradayRotation(screen=GaussianScreen(8.0, 0.5)),
    )
    burn = np.exp(1j * tau * 8.0 - tau**2 * 0.25 / 2)
    np.testing.assert_allclose(
        np.asarray(screened.P[0]),
        np.sum(np.asarray(per.P[0]) * burn, axis=-1),
        rtol=1e-12,
    )
    with pytest.raises(TypeError):
        k.channel(ch, point, phase="faraday")


def test_spectral_equals_the_model_kernels():
    nu = np.geomspace(1e8, 1e10, 5)
    modes = ContinuumKernel(x_min=1e-12).spectral(nu, dict(gamma=3e3, B=5e-6, eta=0.2))
    k_i, k_q = ModelContinuum(x_min=1e-12).kernels(jnp.asarray(nu), 3e3, 5e-6, 0.2)
    np.testing.assert_allclose(np.asarray(modes.I), np.asarray(k_i), rtol=1e-15)
    np.testing.assert_allclose(np.asarray(modes.P[0]).real, np.asarray(k_q), rtol=1e-15)


def test_phase_law_with_any_tau():
    law = PhaseLaw(lambda nu: 1.0 / nu)
    np.testing.assert_allclose(
        np.asarray(law.weights(jnp.array([2.0]), 3.0)), np.exp(1.5j)
    )


# ------------------------------------------------------------------ x_min (T3)


def test_resolved_x_min_admits_the_last_cell(study):
    _, paper = study
    ch = Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2, n_nu=64)
    top = paper.gamma_hi
    point = dict(gamma=top, B=6e-6, eta=0.0)
    with pytest.raises(Exception, match="x_min"):
        jax.block_until_ready(ContinuumKernel().channel(ch, point).I)
    k = ContinuumKernel().resolved(ch, gamma_max=top, b_perp_max=6e-6)
    x_lo = float(np.min(np.asarray(ch.nodes))) / (A1_HZ_PER_G * 6e-6 * top**2)
    assert k.x_min == pytest.approx(0.5 * x_lo)
    assert ("x_min", k.x_min) in k.describe()
    assert np.all(np.isfinite(np.asarray(k.channel(ch, point).I)))
    assert x_lo < 1e-6  # below the model default, which refuses this cell (r2-36)


# ------------------------------------------------------------------ energy window (synth/window.py pins)

WINDOW_PINS = {  # design/0.8/prototypes/synth/window_output.txt
    ("isotropic", 1e-3): (1.729e-03, 6.729),
    ("isotropic", 5e-4): (9.359e-04, 7.410),
    ("half", 1e-3): (1.982e-03, 7.043),
    ("half", 5e-4): (1.073e-03, 7.734),
    ("zero", 1e-3): (2.069e-03, 7.312),
    ("zero", 5e-4): (1.120e-03, 8.024),
}


def _eta(case):
    t, w = np.polynomial.legendre.leggauss(48)
    if case == "zero":
        return np.array([0.0]), np.array([1.0])
    hi = 1.0 if case == "isotropic" else 0.5
    return 0.5 * hi * t + 0.5 * hi, 0.5 * w


@pytest.mark.parametrize(("case", "eps"), sorted(WINDOW_PINS))
def test_energy_window_reproduces_the_reference(case, eps):
    y_lo, y_hi = continuum_window({"eta": _eta(case)}, density=2.6, eps=eps)
    ref_lo, ref_hi = WINDOW_PINS[(case, eps)]
    # The reference tabulates F by scipy quad on 1400 log points, which errs near 1e-3 at small x, so the lower
    # edge is held to 2e-3 and the upper edge to 1e-3.
    assert y_lo == pytest.approx(ref_lo, rel=2e-3)
    assert y_hi == pytest.approx(ref_hi, rel=1e-3)


def test_energy_range_of_the_paper_statistics():
    window = continuum_window({"eta": _eta("half")}, density=2.6, eps=1e-3)
    lo, hi = energy_range(window, nu=(1e8, 1e10), field=(4e-6, 6e-6))
    assert lo == pytest.approx(750.71, rel=2e-3) and hi == pytest.approx(
        548140.0, rel=2e-3
    )
    assert energy_range(window, nu=(1e8, 3e11), field=(4e-6, 6e-6))[1] == pytest.approx(
        3.0023e6, rel=2e-3
    )
    assert C.EnergyWindow(continuum_window) in ContinuumKernel.capabilities


# ------------------------------------------------------------------ averaged kernel


def test_averaged_kernel_equals_the_nested_sum():
    ch = small_channels(32)
    k = ContinuumKernel(x_min=1e-12)
    b_rule, e_rule = uniform_rule(4e-6, 6e-6, 4), uniform_rule(0.0, 0.5, 3)
    avg = AveragedKernel(
        k,
        [("B", b_rule), ("eta", e_rule)],
        phase=FaradayRotation(screen=GaussianScreen(8.0, 0.5)),
    )
    got = avg(ch, {"gamma": 2e3})
    (bn, bw), (en, ew) = b_rule({}), e_rule({})
    i_ref, p_ref = 0.0, 0.0
    for b, wb in zip(np.asarray(bn), np.asarray(bw)):
        for e, we in zip(np.asarray(en), np.asarray(ew)):
            m = k.channel(ch, dict(gamma=2e3, B=b, eta=e), phase=avg.phase)
            i_ref, p_ref = i_ref + wb * we * m.I, p_ref + wb * we * m.P
    np.testing.assert_allclose(np.asarray(got.I), np.asarray(i_ref), rtol=1e-13)
    np.testing.assert_allclose(np.asarray(got.P), np.asarray(p_ref), rtol=1e-13)
    for batch in (None, 2):
        other = AveragedKernel(
            k, [("B", b_rule), ("eta", e_rule)], phase=avg.phase, outer_batch=batch
        )
        np.testing.assert_allclose(
            np.asarray(other(ch, {"gamma": 2e3}).I), np.asarray(got.I), rtol=1e-14
        )


def test_conditional_law_and_fixed_values():
    ch = small_channels(32)
    k = ContinuumKernel(x_min=1e-12)
    rule = uniform_rule(lambda p: 4e-6 * (p["gamma"] / 1e3) ** 0.1, 6e-6, 4)
    avg = AveragedKernel(k, [("B", rule)], fixed={"eta": 0.2})
    nodes, probs = rule({"gamma": 2e3})
    ref = sum(
        float(w) * k.channel(ch, dict(gamma=2e3, B=float(b), eta=0.2)).I
        for b, w in zip(nodes, probs)
    )
    np.testing.assert_allclose(
        np.asarray(avg(ch, {"gamma": 2e3}).I), np.asarray(ref), rtol=1e-13
    )


def test_legendre_projection():
    ch = small_channels(32)
    k = ContinuumKernel(x_min=1e-12)
    avg = AveragedKernel(k, legendre={"eta": (2, 24)}, fixed={"B": 5e-6})
    out = avg(ch, {"gamma": 2e3})
    assert out.I.shape == (7, 3) and out.P.shape == (1, 7, 3)
    x, w = np.polynomial.legendre.leggauss(24)
    ref = sum(
        wi
        * 2.5
        * 0.5
        * (3 * xi**2 - 1)
        * k.channel(ch, dict(gamma=2e3, B=5e-6, eta=xi)).I
        for xi, wi in zip(x, w)
    )
    np.testing.assert_allclose(np.asarray(out.I[:, 2]), np.asarray(ref), rtol=1e-12)


def test_averaged_kernel_refusals():
    k = ContinuumKernel()
    with pytest.raises(ValueError, match="role"):
        AveragedKernel(k, [("B", fixed_rule([5e-6], [1.0]))], fixed={"B": 5e-6})
    with pytest.raises(ValueError, match="not variables"):
        AveragedKernel(k, [("Bfield", fixed_rule([5e-6], [1.0]))])


# ------------------------------------------------------------------ re-expansion


def test_stirling_and_log_from_affine():
    assert stirling2(4)[4].tolist() == [0, 1, 7, 6, 1]
    h, xb, k, n_max = 0.3, 2.0, 3, 5
    affine = np.array(
        [xb**k * math.comb(k, r) for r in range(n_max + 1)]
    )  # x^k at x_b, z = (x - x_b) / x_b
    log = log_from_affine(n_max, h) @ affine
    np.testing.assert_allclose(
        log,
        [xb**k * (k * h) ** n / math.factorial(n) for n in range(n_max + 1)],
        rtol=1e-13,
    )


@pytest.mark.parametrize("chart", ["log", "affine"])
def test_field_scaling_columns_on_a_power_law_kernel(chart):
    # Kbar(gamma, B) = B (gamma^2 B)^alpha obeys Kbar(gamma, L B) = L Kbar(L^(1/2) gamma, B).
    alpha, h, g_b, b_b, h_b, b0 = (
        -0.8,
        0.24,
        2e3,
        6e-6,
        (0.35 if chart == "log" else 1e-6),
        5e-6,
    )

    def kbar(gamma, b):
        return b * (gamma**2 * b) ** alpha

    def cell(u, v):
        b = b_b * jnp.exp(h_b * v) if chart == "log" else b_b + h_b * v
        return kbar(g_b * jnp.exp(h * u), b)

    g_s = g_b * math.sqrt(b_b / b0)
    energy = {
        n: kbar(g_s, b0) * (2 * alpha * h) ** n / math.factorial(n) for n in range(5)
    }
    exps = [(r, s) for r in range(3) for s in range(3) if r + s <= 3]
    got = field_scaling_columns(
        energy,
        exps,
        h_gamma=h,
        field_chart=chart,
        field_ref=b_b,
        field_half_width=h_b,
        field_basis=b0,
    )
    for r, s in exps:
        d = cell
        for _ in range(r):
            d = jax.grad(d, argnums=0)
        for _ in range(s):
            d = jax.grad(d, argnums=1)
        exact = float(d(0.0, 0.0)) / (math.factorial(r) * math.factorial(s))
        assert float(got[(r, s)]) == pytest.approx(exact, rel=1e-12, abs=1e-30)
