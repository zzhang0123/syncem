"""Step 3 of design/0.8/SPEC.md Sec. 10.2: kernel protocol, capabilities, continuum adapter, ``AveragedKernel``,
core provider and re-expansion (Sec. 3.2, 4.6, 8.1, 8.3, T3)."""

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
from syncem.engine.provider_core import core_cell, core_columns
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
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


# ------------------------------------------------------------------ core provider


def paper_kernel(paper, n_nu=400):
    ch = Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2, n_nu=n_nu)
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        [
            ("B", uniform_rule(paper.b_lo * 1e-6, paper.b_hi * 1e-6, paper.n_b)),
            ("eta", uniform_rule(0.0, paper.eta_max, paper.n_eta)),
        ],
        phase=FaradayRotation(
            screen=GaussianScreen(paper.screen_mean, paper.screen_sd)
        ),
    )
    return ch, avg


@pytest.mark.parametrize(
    "cell",
    [
        0,
        pytest.param(9, marks=pytest.mark.slow),
        pytest.param(18, marks=pytest.mark.slow),
    ],
)
def test_core_provider_equals_the_study_stack(study, cell):
    sb, paper = study
    ch, avg = paper_kernel(paper)
    lattice = Lattice(gamma=Cells.log(300, paper.gamma_hi, 19, order=4))
    i, p = core_cell(avg, ch, lattice, cell)
    di, dq = sb.derivative_spectra(paper, 4)
    scale = np.abs(di[cell, 0]).max()
    assert np.max(np.abs(np.asarray(i) - di[cell])) / scale < 1e-6
    assert np.max(np.abs(np.asarray(p) - dq[cell])) / scale < 1e-6
    if cell in (9, 18):
        assert np.max(np.abs(np.asarray(i) - di[cell])) / scale < 1e-10


@pytest.mark.parametrize("chart", ["log", "affine"])
def test_joint_energy_field_columns_equal_the_scaling_relation(chart):
    ch = small_channels(64)
    k = ContinuumKernel(x_min=1e-12)
    eta = ("eta", uniform_rule(0.0, 0.5, 12))
    phase = FaradayRotation(screen=GaussianScreen(8.0, 0.5))
    field = (
        Cells.log("2 uG", "8 uG", 3)
        if chart == "log"
        else Cells.single("5.5 uG", scale="1 uG")
    )
    lattice = Lattice(gamma=Cells.log(300, 3e5, 4), B=field, order=2)
    b0 = 5e-6
    cell = (
        lattice.index.cells_where(gamma=1, B=min(1, field.n - 1))[0]
        if field.n > 1
        else 1
    )
    i_joint, p_joint = core_cell(
        AveragedKernel(k, [eta], phase=phase), ch, lattice, int(cell)
    )
    t = lattice.cells
    g_b, h, b_b, h_b = (
        float(t.reference[cell, 0]),
        float(t.half_width[cell, 0]),
        float(t.reference[cell, 1]),
        float(t.half_width[cell, 1]),
    )
    exps = list(lattice.index.patterns[int(lattice.index.cell_pattern[cell])])
    g_s = g_b * math.sqrt(b_b / b0)
    fixed = AveragedKernel(k, [eta], fixed={"B": b0}, phase=phase)
    e_i, e_p = core_columns(
        fixed, ch, {"gamma": ("log", g_s, h)}, [(n,) for n in range(3)]
    )
    kw = dict(
        h_gamma=h,
        field_chart=chart,
        field_ref=b_b,
        field_half_width=h_b,
        field_basis=b0,
    )
    s_i = field_scaling_columns({n: e_i[n] for n in range(3)}, exps, **kw)
    s_p = field_scaling_columns({n: e_p[n] for n in range(3)}, exps, **kw)
    scale_i, scale_p = float(np.abs(np.asarray(i_joint)).max()), float(
        np.abs(np.asarray(p_joint)).max()
    )
    for row, e in enumerate(exps):
        assert (
            np.max(np.abs(np.asarray(i_joint[row]) - np.asarray(s_i[e]))) / scale_i
            < 1e-10
        ), e
        assert (
            np.max(np.abs(np.asarray(p_joint[row]) - np.asarray(s_p[e]))) / scale_p
            < 1e-10
        ), e


def test_core_provider_refusals():
    ch = small_channels(16)
    k = ContinuumKernel(x_min=1e-12)
    with pytest.raises(ValueError, match="gamma"):
        core_columns(
            AveragedKernel(k, fixed={"B": 5e-6, "eta": 0.0}),
            ch,
            {"eta": ("affine", 0.0, 0.5)},
            [(0,)],
        )
    with pytest.raises(ValueError, match="B is neither"):
        core_columns(
            AveragedKernel(k, fixed={"eta": 0.0}),
            ch,
            {"gamma": ("log", 1e3, 0.2)},
            [(0,)],
        )
    with pytest.raises(ValueError, match="Legendre"):
        core_columns(
            AveragedKernel(k, legendre={"eta": (2, 8)}, fixed={"B": 5e-6}),
            ch,
            {"gamma": ("log", 1e3, 0.2)},
            [(0,)],
        )


# ------------------------------------------------------------------ review findings


def test_core_refuses_a_celled_variable_that_is_also_averaged_or_fixed():
    ch = small_channels(16)
    k = ContinuumKernel(x_min=1e-12)
    eta = ("eta", uniform_rule(0.0, 0.5, 3))
    axes = {"gamma": ("log", 3e3, 0.24), "B": ("affine", 5e-6, 1e-6)}
    for avg in (
        AveragedKernel(k, [("B", uniform_rule(4e-6, 6e-6, 3)), eta]),
        AveragedKernel(k, [eta], fixed={"B": 9e-6}),
    ):
        with pytest.raises(ValueError, match="celled"):
            core_columns(avg, ch, axes, [(0, 0), (0, 1)])
    with pytest.raises(ValueError, match="celled"):
        core_columns(
            AveragedKernel(
                k, [("gamma", uniform_rule(2e3, 4e3, 2)), eta], fixed={"B": 5e-6}
            ),
            ch,
            {"gamma": ("log", 3e3, 0.24)},
            [(0,)],
        )


def test_two_legendre_axes_keep_their_order():
    class Toy:
        name = "toy"
        from syncem.base.variables import Variable

        variables = (
            Variable("x", "x", "cosine", "", True, None, None, ""),
            Variable("y", "y", "cosine", "", True, None, None, ""),
        )

        def channel(self, channels, point, *, phase=None, per_node=False):
            v = jnp.asarray([1.0, 2.0, 3.0]) * (
                1 + point["x"] + point["x"] * point["y"] ** 2
            )
            return Modes(I=v, V=jnp.zeros_like(v), P=v[None].astype(complex))

    out = AveragedKernel(Toy(), legendre={"x": (2, 20), "y": (3, 20)})(None, {})
    assert out.I.shape == (3, 3, 4) and out.P.shape == (1, 3, 3, 4)
    # f = 1 + x + x y^2: the (l=1, k=0) coefficient is 1 + 1/3 and (l=1, k=2) is 2/3.
    np.testing.assert_allclose(np.asarray(out.I[0, 1, 0]), 4 / 3, rtol=1e-13)
    np.testing.assert_allclose(np.asarray(out.I[0, 1, 2]), 2 / 3, rtol=1e-13)


def test_core_cell_refuses_legendre_axes_and_reorders_b_first():
    ch = small_channels(16)
    k = ContinuumKernel(x_min=1e-12)
    from syncem.lattice.representations import Legendre

    eta = ("eta", uniform_rule(0.0, 0.5, 3))
    with pytest.raises(ValueError, match="Legendre"):
        core_cell(
            AveragedKernel(k, [eta], fixed={"B": 5e-6}),
            ch,
            Lattice(gamma=Cells.log(300, 3e5, 3), eta=Legendre(2)),
            0,
        )
    avg = AveragedKernel(k, [eta])
    b_first = Lattice(
        B=Cells.log("2 uG", "8 uG", 2), gamma=Cells.log(300, 3e5, 3), order=2
    )
    g_first = Lattice(
        gamma=Cells.log(300, 3e5, 3), B=Cells.log("2 uG", "8 uG", 2), order=2
    )
    i_b, _ = core_cell(avg, ch, b_first, 0)  # cell (B=0, gamma=0)
    i_g, _ = core_cell(avg, ch, g_first, 0)
    exps_b = list(b_first.index.patterns[0])
    exps_g = list(g_first.index.patterns[0])
    for row, (sb, rg) in enumerate(exps_b):
        np.testing.assert_allclose(
            np.asarray(i_b[row]), np.asarray(i_g[exps_g.index((rg, sb))]), rtol=1e-12
        )


def test_averaged_kernel_refuses_roles_in_the_point_and_bad_batches():
    k = ContinuumKernel(x_min=1e-12)
    ch = small_channels(16)
    avg = AveragedKernel(k, [("B", uniform_rule(4e-6, 6e-6, 2))], fixed={"eta": 0.2})
    for point in (
        {"gamma": 2e3, "B": 1e-3},
        {"gamma": 2e3, "eta": 0.5},
        {"gamma": 2e3, "zzz": 1.0},
    ):
        with pytest.raises(ValueError):
            avg(ch, point)
    for batch in (0, -1, 1.5):
        with pytest.raises(ValueError):
            AveragedKernel(k, [("B", uniform_rule(4e-6, 6e-6, 2))], outer_batch=batch)
    with pytest.raises(ValueError, match="cosine"):
        AveragedKernel(k, legendre={"gamma": (2, 8)}, fixed={"B": 5e-6})


def test_window_low_tail_and_slope_checks():
    eta = _eta("isotropic")
    y15, _ = continuum_window({"eta": eta}, density=1.5, eps=1e-3)
    # Below y = 1e-8 the weight is c y^((p - 1)/2 + 1/3), whose integral the window adds in closed form.
    lo_grid = np.exp(np.linspace(np.log(1e-16), np.log(200.0), 20000))
    from syncem.ultrarel import F

    s = np.sqrt(1 - eta[0] ** 2)
    fbar = np.asarray(
        jnp.sum(
            jnp.asarray(eta[1] * s)[None]
            * F(jnp.minimum(lo_grid[:, None] / s[None], 1e4)),
            axis=1,
        )
    )
    weight = fbar * lo_grid ** ((1.5 - 3) / 2) * lo_grid
    ly = np.log(lo_grid)
    cum = np.concatenate(
        [[0.0], np.cumsum(0.5 * (weight[1:] + weight[:-1]) * np.diff(ly))]
    )
    ref = np.exp(np.interp(1e-3 * cum[-1], cum, ly))
    assert y15 == pytest.approx(ref, rel=2e-3)
    with pytest.raises(ValueError, match="density"):
        continuum_window({"eta": eta}, density=0.2)


def test_window_takes_the_averaged_laws_and_defaults_to_isotropic():
    iso = continuum_window({}, None, density=2.6, eps=1e-3)
    assert iso[0] == pytest.approx(1.729e-3, rel=1e-3) and iso[1] == pytest.approx(
        6.729, rel=1e-3
    )


def test_x_min_for_returns_a_float_from_arrays():
    ch = small_channels(16)
    value = ContinuumKernel.x_min_for(ch, jnp.asarray(3e6), jnp.asarray(6e-6))
    assert type(value) is float


def test_keyword_only_constructors():
    with pytest.raises(TypeError):
        ContinuumKernel(500.0)
    with pytest.raises(TypeError):
        FaradayRotation(None)
    with pytest.raises(TypeError):
        PhaseLaw(lambda nu: nu, None)


def test_protocol_requires_spectral():
    class NoSpectral:
        name, variables, primary, stokes, azimuth, depth = (
            "x",
            (),
            None,
            ("I",),
            None,
            None,
        )
        traceable, lines, capabilities = True, False, ()

        def channel(self, channels, point, *, phase, per_node=False): ...

        def regime(self, box, channels): ...

        def physical_error(self, channels): ...

        def describe(self): ...

    assert not isinstance(NoSpectral(), Kernel)


@pytest.mark.parametrize("order", [0, 1, 2, 3])
def test_core_provider_lower_orders_in_cell_1(study, order):
    sb, paper = study
    ch, avg = paper_kernel(paper, n_nu=400)
    i, p = core_cell(
        avg, ch, Lattice(gamma=Cells.log(300, paper.gamma_hi, 19, order=order)), 0
    )
    di, dq = sb.derivative_spectra(paper, order)
    scale = np.abs(di[0, 0]).max()
    assert np.max(np.abs(np.asarray(i) - di[0])) / scale < 1e-6
    assert np.max(np.abs(np.asarray(p) - dq[0])) / scale < 1e-6


def test_last_paper_cell_with_the_resolved_kernel(study):
    _, paper = study
    ch = Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2, n_nu=64)
    kernel = ContinuumKernel().resolved(
        ch, gamma_max=paper.gamma_hi, b_perp_max=paper.b_hi * 1e-6
    )
    avg = AveragedKernel(
        kernel, [("B", uniform_rule(4e-6, 6e-6, 4)), ("eta", uniform_rule(0.0, 0.5, 4))]
    )
    i, p = core_cell(avg, ch, Lattice(gamma=Cells.log(300, paper.gamma_hi, 19)), 18)
    assert np.all(np.isfinite(np.asarray(i))) and np.all(np.isfinite(np.asarray(p)))


def test_window_names_the_expected_form():
    with pytest.raises(TypeError, match="eta"):
        continuum_window(_eta("half"))
