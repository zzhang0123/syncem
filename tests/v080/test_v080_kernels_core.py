"""Step 3 of design/0.8/SPEC.md Sec. 10.2, second part: the core provider against the study stack and the
re-expansion, and the review findings of step 3 (Sec. 4.6, 8.1, 8.3, T3).

Split from ``test_v080_kernels.py`` in the step-5b-3 review, with the test bodies unchanged (decision 170 keeps
files under 400 lines). The core comparisons take channels 0, 20 and 40 of the 41 paper channels, and the
low-tail window check evaluates its grid in 10 calls.

Memory (decision 170): this file alone peaked at 4.90 GB of resident memory in its fast run (2026-11-19, 1-minute
load 6.0 to 5.5, loaded).
"""

import math
import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.kernel import Kernel, Modes
from syncem.base.phase import FaradayRotation, PhaseLaw
from syncem.base.window import continuum_window
from syncem.engine.average import AveragedKernel, uniform_rule
from syncem.engine.expand import field_scaling_columns
from syncem.engine.provider_core import core_cell, core_columns
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.model.phase import GaussianScreen
from tests.v080.test_v080_kernels import _eta

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


# ------------------------------------------------------------------ core provider


PAPER_ROWS = [
    0,
    20,
    40,
]  # the lowest, middle and highest of the 41 paper channels (decision 170)


def paper_kernel(paper, n_nu=400):
    """The paper kernel on channels 0, 20 and 40 of the 41 paper channels of ``n_nu`` nodes.

    The core provider evaluates every channel in one program, whose XLA buffer grows with channels times nodes:
    on all 41 channels of 400 nodes the file peaked at 30.9 GB, so the comparisons take three channels.
    """
    ch = Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2, n_nu=n_nu)[PAPER_ROWS]
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
    di, dq = (a[:, :, PAPER_ROWS] for a in sb.derivative_spectra(paper, 4))
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

    def mean_f(
        y,
    ):  # 2000 grid points per call: one call over 20000 peaked at 7.7 GB (decision 170)
        x = jnp.minimum(y[:, None] / s[None], 1e4)
        return np.asarray(jnp.sum(jnp.asarray(eta[1] * s)[None] * F(x), axis=1))

    fbar = np.concatenate([mean_f(y) for y in np.array_split(lo_grid, 10)])
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
    di, dq = (a[:, :, PAPER_ROWS] for a in sb.derivative_spectra(paper, order))
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
