"""Step 4 of design/0.8/SPEC.md Sec. 10.2: the jet provider in ``d`` directions (Sec. 8.3, T16, G16, Sec. 12 (n))."""

import math

import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import binom

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.kernel import Modes
from syncem.base.phase import FaradayRotation
from syncem.base.user import UserKernel, kernel
from syncem.base.variables import Variable
from syncem.engine.average import AveragedKernel, fixed_rule, uniform_rule
from syncem.engine.provider_core import core_columns
from syncem.engine.provider_jet import (
    cost,
    jet_cell,
    jet_columns,
    jet_lattice,
    level_plan,
)
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import Legendre
from syncem.model.phase import GaussianScreen


def channels(n=5, n_nu=32):
    return Channels.log(0.1, 10, n, unit="GHz", half_width=0.2, n_nu=n_nu)


# ------------------------------------------------------------------ level plan


def test_level_plan_is_cap_aware():
    exps = [(r, s) for r in range(5) for s in range(2) if r + s <= 4]
    assert level_plan(exps) == ((0, 1), (0,), (0,), (0,))
    assert cost(exps) == 3 * 2 * 2 * 2
    three = [
        e for e in np.ndindex(3, 2, 4) if sum(e[:2]) <= 2
    ]  # depth-like third variable outside the total
    plan = level_plan(three)
    assert set(plan[0]) == {0, 1, 2} and len(plan) == 5
    assert level_plan([(0, 0, 0)]) == ()


# ------------------------------------------------------------------ T16: polynomial kernel in three variables

POWERS = [(1.5, 2.0, 1), (-0.7, 0.0, 3), (2.2, -1.3, 0), (0.0, 0.5, 2)]
COEF = np.array(
    [[1.0, -0.5, 0.25], [0.3, 0.8, -1.1], [-0.6, 0.2, 0.9], [0.4, -0.3, 0.05]]
)


class PowerKernel:
    """``I_j = sum_k c_kj x^p y^q w^m`` (``x``, ``y`` positive, ``w`` real), ``P = (1 + 0.5i) I``."""

    name = "power"
    variables = (
        Variable("x", "x", "positive", "", True, None, None, ""),
        Variable("y", "y", "positive", "", True, None, None, ""),
        Variable("w", "w", "real", "", True, None, None, ""),
    )

    def channel(self, ch, point, *, phase=None, per_node=False):
        total = 0.0
        for (p, q, m), c in zip(POWERS, COEF):
            total = (
                total
                + jnp.asarray(c) * point["x"] ** p * point["y"] ** q * point["w"] ** m
            )
        return Modes(I=total, V=jnp.zeros_like(total), P=((1 + 0.5j) * total)[None])


def _one(chart, power, ref, h, n):
    if chart == "log":
        return ref**power * (power * h) ** n / math.factorial(n)
    return binom(power, n) * ref ** (power - n) * h**n


def closed_form(axes, exps):
    out = []
    for e in exps:
        total = 0.0
        for (p, q, m), c in zip(POWERS, COEF):
            powers = {"x": p, "y": q, "w": m}
            term = c.copy()
            for i, (v, (chart, ref, h)) in enumerate(axes.items()):
                term = term * _one(chart, powers[v], ref, h, e[i])
            for v in ("x", "y", "w"):
                if v not in axes:
                    term = term * FIXED[v] ** powers[v]
            total = total + term
        out.append(total)
    return np.array(out)


FIXED = {"x": 2.0, "y": 0.7, "w": 1.3}


def lower(caps, total):
    return [e for e in np.ndindex(*[c + 1 for c in caps]) if sum(e) <= total]


@pytest.mark.parametrize(
    ("axes", "caps", "total"),
    [
        ({"x": ("log", 3.0, 0.2)}, (4,), 4),
        ({"x": ("log", 3.0, 0.2), "y": ("affine", 0.9, 0.1)}, (3, 2), 3),
        (
            {
                "x": ("log", 3.0, 0.2),
                "y": ("log", 0.9, 0.15),
                "w": ("affine", 1.1, 0.3),
            },
            (2, 2, 2),
            2,
        ),
        (
            {
                "x": ("log", 3.0, 0.2),
                "y": ("log", 0.9, 0.15),
                "w": ("affine", 1.1, 0.3),
            },
            (4, 1, 3),
            4,
        ),
        ({"w": ("affine", -1.4, 0.2), "x": ("affine", 5.0, 0.5)}, (3, 0), 3),
        ({"x": ("log", 3.0, 0.2), "y": ("affine", 0.9, 0.1)}, (4, 4), 4),
    ],
)
def test_polynomial_kernel_against_closed_form(axes, caps, total):
    fixed = {v: FIXED[v] for v in ("x", "y", "w") if v not in axes}
    avg = AveragedKernel(PowerKernel(), fixed=fixed)
    exps = lower(caps, total)
    i, p = jet_columns(avg, None, axes, exps)
    ref = closed_form(axes, exps)
    np.testing.assert_allclose(
        np.asarray(i), ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max()
    )
    np.testing.assert_allclose(
        np.asarray(p), (1 + 0.5j) * ref, rtol=1e-12, atol=1e-12 * np.abs(ref).max()
    )


@pytest.mark.parametrize(
    "axes",
    [
        dict(x=Cells.log(1.0, 30.0, 6, order=4)),
        dict(x=Cells.log(1.0, 30.0, 6, order=3), y=Cells.linear(0.5, 1.5, 4, order=2)),
        dict(
            x=Cells.log(1.0, 30.0, 3, order=4),
            y=Cells.log(0.5, 1.5, 2, order=2),
            w=Cells.linear(0.5, 2.0, 2, order=3),
        ),
    ],
)
def test_polynomial_kernel_first_and_last_cell_of_a_lattice(axes):
    lat = Lattice(**axes, order=max(c.order[0] for c in axes.values()))
    avg = AveragedKernel(
        PowerKernel(), fixed={v: FIXED[v] for v in ("x", "y", "w") if v not in axes}
    )
    all_i, _ = jet_lattice(avg, None, lat)
    assert all_i.shape[0] == lat.index.block_size
    for c in (0, lat.n_cells - 1):
        t = lat.cells
        axes = {
            v: (t.chart[c][k], float(t.reference[c, k]), float(t.half_width[c, k]))
            for k, v in enumerate(t.variables)
        }
        exps = list(lat.index.patterns[int(lat.index.cell_pattern[c])])
        ref = closed_form(axes, exps)
        rows = lat.index.cell_rows(c)
        np.testing.assert_allclose(
            np.asarray(all_i[rows.start : rows.stop]),
            ref,
            rtol=1e-12,
            atol=1e-12 * np.abs(ref).max(),
        )
        np.testing.assert_allclose(
            np.asarray(jet_cell(avg, None, lat, c)[0]),
            ref,
            rtol=1e-12,
            atol=1e-12 * np.abs(ref).max(),
        )


# ------------------------------------------------------------------ jet equals core at d = 2


@pytest.mark.parametrize("chart", ["log", "affine"])
def test_jet_equals_core(chart):
    ch = channels(7, 32)
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        [("eta", uniform_rule(0.0, 0.5, 6))],
        phase=FaradayRotation(screen=GaussianScreen(8.0, 0.5)),
    )
    axes = {
        "gamma": ("log", 3e3, 0.24),
        "B": (chart, 5e-6, 0.3 if chart == "log" else 1e-6),
    }
    exps = lower((4, 4), 4)
    ji, jp = jet_columns(avg, ch, axes, exps)
    ci, cp = core_columns(avg, ch, axes, exps)
    assert (
        np.max(np.abs(np.asarray(ji) - np.asarray(ci))) / np.abs(np.asarray(ci)).max()
        < 1e-12
    )
    assert (
        np.max(np.abs(np.asarray(jp) - np.asarray(cp))) / np.abs(np.asarray(cp)).max()
        < 1e-12
    )


def test_jet_equals_core_with_b_averaged_on_a_lattice():
    ch = channels(5, 32)
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        [("B", uniform_rule(4e-6, 6e-6, 4)), ("eta", uniform_rule(0.0, 0.5, 4))],
    )
    lat = Lattice(gamma=Cells.log(300, 3e5, 3, order=4))
    from syncem.engine.provider_core import core_cell

    for c in (0, 2):
        ji, _ = jet_cell(avg, ch, lat, c)
        ci, _ = core_cell(avg, ch, lat, c)
        assert (
            np.max(np.abs(np.asarray(ji) - np.asarray(ci)))
            / np.abs(np.asarray(ci)).max()
            < 1e-12
        )


# ------------------------------------------------------------------ finite-difference checks

STENCIL1 = (np.array([1, -8, 0, 8, -1]) / 12.0, np.arange(-2, 3))
STENCIL2 = (np.array([-1, 16, -30, 16, -1]) / 12.0, np.arange(-2, 3))


def finite_difference(avg, ch, axes, alpha, step):
    """``(1/alpha!) d^alpha f(0)`` for ``|alpha| <= 2`` by central stencils in the chart coordinates."""
    names = tuple(axes)

    def f(z):
        point = {
            v: (r * np.exp(h * z[k]) if c == "log" else r + h * z[k])
            for k, (v, (c, r, h)) in enumerate(axes.items())
        }
        m = avg(ch, point)
        return np.asarray(m.I), np.asarray(m.P[0])

    d = len(names)
    active = [k for k in range(d) if alpha[k]]
    if not active:
        return f(np.zeros(d))
    if len(active) == 1:
        k = active[0]
        weights, offsets = STENCIL1 if alpha[k] == 1 else STENCIL2
        acc_i, acc_p = 0.0, 0.0
        for w, o in zip(weights, offsets):
            z = np.zeros(d)
            z[k] = o * step
            fi, fp = f(z)
            acc_i, acc_p = acc_i + w * fi, acc_p + w * fp
        scale = step ** alpha[k] * math.factorial(alpha[k])
        return acc_i / scale, acc_p / scale
    a, b = active

    def mixed(h):
        acc_i, acc_p = 0.0, 0.0
        for sa, sb, sign in ((1, 1, 1), (1, -1, -1), (-1, 1, -1), (-1, -1, 1)):
            z = np.zeros(d)
            z[a], z[b] = sa * h, sb * h
            fi, fp = f(z)
            acc_i, acc_p = acc_i + sign * fi, acc_p + sign * fp
        return acc_i / (4 * h**2), acc_p / (4 * h**2)

    # Richardson extrapolation of the O(h^2) four-point stencil to O(h^4).
    (i1, p1), (i2, p2) = mixed(step), mixed(step / 2)
    return (4 * i2 - i1) / 3, (4 * p2 - p1) / 3


def grain_kernel():
    @kernel(
        variables=[
            Variable("a", kind="positive", unit="cm", meaning="grain radius"),
            Variable("n_H", kind="positive", unit="cm-3", meaning="gas density"),
        ],
        stokes=("I",),
    )
    def grain_emission(nu, a, n_H):  # noqa: N803  (G16 spelling)
        nu_peak = 30e9 * (a / 5e-8) ** -2.0 * (n_H / 30.0) ** 0.1
        return {"I": 1e-40 * (nu / nu_peak) ** 2 * jnp.exp(1.0 - (nu / nu_peak) ** 2)}

    return grain_emission


def lognormal_rule(median, width, n=16):
    t, w = np.polynomial.hermite.hermgauss(n)
    return fixed_rule(median * np.exp(width * np.sqrt(2.0) * t), w / np.sqrt(np.pi))


def test_g16_custom_kernel_columns():
    grain = grain_kernel()
    assert isinstance(grain, UserKernel) and grain.name == "grain_emission"
    ch = Channels.log(10, 300, 9, unit="GHz", half_width=0.2, n_nu=32)
    avg = AveragedKernel(grain, [("n_H", lognormal_rule(30.0, 0.5))])
    lat = Lattice(a=Cells.log(3.5e-8, 1e-6, 8, order=2, unit="cm"))
    assert np.all(np.asarray(lat.reference_weights) == 1.0)
    cols_i, _ = jet_lattice(avg, ch, lat)
    assert cols_i.shape == (24, 9)
    scale = float(np.abs(np.asarray(cols_i)).max())
    for c in (0, 7):
        t = lat.cells
        axes = {
            "a": (t.chart[c][0], float(t.reference[c, 0]), float(t.half_width[c, 0]))
        }
        for r in range(3):
            fd, _ = finite_difference(avg, ch, axes, (r,), 1e-2)
            got = np.asarray(cols_i[3 * c + r])
            assert np.max(np.abs(got - fd)) / scale < 1e-7, (c, r)


def test_cells_on_three_variables_one_a_cosine():
    # Sec. 12 row (n): gamma, B and eta cells with the continuum kernel go to the jet in (u_gamma, u_B, u_eta).
    ch = channels(5, 32)
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        phase=FaradayRotation(screen=GaussianScreen(8.0, 0.5)),
    )
    lat = Lattice(
        gamma=Cells.log(300, 3e5, 2),
        B=Cells.linear("2 uG", "8 uG", 3),
        eta=Cells.linear(0, 0.5, 2),
        order=2,
    )
    t = lat.cells
    for c in (0, lat.n_cells - 1):
        axes = {
            v: (t.chart[c][k], float(t.reference[c, k]), float(t.half_width[c, k]))
            for k, v in enumerate(t.variables)
        }
        i, p = jet_cell(avg, ch, lat, c)
        scale_i, scale_p = float(np.abs(np.asarray(i)).max()), float(
            np.abs(np.asarray(p)).max()
        )
        for row, alpha in enumerate(lat.index.patterns[int(lat.index.cell_pattern[c])]):
            fi, fp = finite_difference(avg, ch, axes, alpha, 2e-3)
            assert np.max(np.abs(np.asarray(i[row]) - fi)) / scale_i < 1e-6, (c, alpha)
            assert np.max(np.abs(np.asarray(p[row]) - fp)) / scale_p < 1e-6, (c, alpha)


# ------------------------------------------------------------------ Legendre axes and refusals


def test_jet_with_a_legendre_axis():
    ch = channels(5, 32)
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12), legendre={"eta": (2, 24)}, fixed={"B": 5e-6}
    )
    lat = Lattice(gamma=Cells.log(300, 3e5, 2), eta=Legendre(2))
    i, p = jet_cell(avg, ch, lat, 1)
    assert i.shape == (9, 5) and p.shape == (9, 5)
    ref = avg(ch, {"gamma": float(lat.cells.reference[1, 0])})
    for l in range(3):
        row = (
            lat.index.position(cell=1, degrees={"eta": l})
            - lat.index.cell_rows(1).start
        )
        # l = 1 vanishes by parity (an even kernel in eta), so the tolerance is set by the l = 0 scale.
        np.testing.assert_allclose(
            np.asarray(i[row]),
            np.asarray(ref.I[:, l]),
            rtol=1e-13,
            atol=1e-13 * float(np.abs(np.asarray(ref.I)).max()),
        )


def test_jet_refusals():
    ch = channels(3, 16)
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        [("B", uniform_rule(4e-6, 6e-6, 2))],
        fixed={"eta": 0.0},
    )
    with pytest.raises(ValueError, match="celled"):
        jet_columns(
            avg, ch, {"gamma": ("log", 1e3, 0.2), "B": ("affine", 5e-6, 1e-6)}, [(0, 0)]
        )
    with pytest.raises(ValueError, match="Legendre"):
        jet_cell(
            AveragedKernel(ContinuumKernel(x_min=1e-12), fixed={"B": 5e-6, "eta": 0.0}),
            ch,
            Lattice(gamma=Cells.log(300, 3e5, 2), eta=Legendre(2)),
            0,
        )


def test_user_kernel_contract():
    grain = grain_kernel()
    ch = Channels.log(10, 300, 3, unit="GHz", half_width=0.2, n_nu=16)
    with pytest.raises(ValueError, match="needs values"):
        grain.channel(ch, {"a": 1e-7})
    with pytest.raises(ValueError, match="not variables"):
        grain.channel(ch, {"a": 1e-7, "n_H": 30.0, "T": 10.0})
    per = grain.channel(ch, {"a": 1e-7, "n_H": 30.0}, per_node=True)
    np.testing.assert_allclose(
        np.asarray(per.I.sum(-1)),
        np.asarray(grain.channel(ch, {"a": 1e-7, "n_H": 30.0}).I),
    )
    with pytest.raises(ValueError):
        kernel(variables=[], stokes=("I",))(lambda nu: {"I": nu})
    with pytest.raises(ValueError):
        kernel(variables=[Variable("a", unit="cm")], stokes=("X",))(
            lambda nu, a: {"I": nu}
        )


# ------------------------------------------------------------------ review findings


def test_v_block_is_carried():
    @kernel(variables=[Variable("a", kind="positive")], stokes=("I", "V"))
    def iv(nu, a):
        return {"I": a * jnp.ones_like(nu), "V": 0.5 * a**2 * jnp.ones_like(nu)}

    ch = channels(3, 8)
    avg = AveragedKernel(iv)
    np.testing.assert_allclose(
        np.asarray(avg(ch, {"a": 2.0}).V), np.asarray(iv.channel(ch, {"a": 2.0}).V)
    )
    i, p, v = jet_columns(
        avg, ch, {"a": ("log", 2.0, 0.1)}, [(0,), (1,), (2,)], with_v=True
    )
    np.testing.assert_allclose(
        np.asarray(v[:, 0]) / float(np.asarray(iv.channel(ch, {"a": 2.0}).V)[0]),
        [1.0, 0.2, 0.02],
        rtol=1e-12,
    )


def test_user_kernel_depth_enters_only_through_the_phase():
    @kernel(
        variables=[Variable("a", kind="positive"), Variable("depth", unit="rad m^-2")],
        stokes=("I", "P"),
        depth="depth",
    )
    def dk(nu, a):
        return {"I": a * jnp.ones_like(nu), "P": 0.3 * a * jnp.ones_like(nu)}

    ch = channels(3, 8)
    shallow = dk.channel(ch, {"a": 1.0, "depth": 0.0}, phase=FaradayRotation())
    deep = dk.channel(ch, {"a": 1.0, "depth": 10.0}, phase=FaradayRotation())
    np.testing.assert_allclose(np.asarray(shallow.I), np.asarray(deep.I))
    assert not np.allclose(np.asarray(shallow.P), np.asarray(deep.P))
    dk.channel(ch, {"a": 1.0})  # depth may be absent


def test_describe_distinguishes_closures():
    def make(scale):
        @kernel(variables=[Variable("a", kind="positive")], stokes=("I",))
        def k(nu, a):
            return {"I": scale * a * jnp.ones_like(nu)}

        return k

    assert make(1.0).describe() != make(5.0).describe()
    assert make(1.0).describe() == make(1.0).describe()


@pytest.mark.parametrize("exps", [[(0, 0, 0)], [(1,)], [], [(-1, 0)]])
def test_jet_columns_checks_exponents(exps):
    avg = AveragedKernel(PowerKernel(), fixed={"w": 1.0})
    with pytest.raises(ValueError, match="exponents"):
        jet_columns(avg, None, {"x": ("log", 2.0, 0.1), "y": ("log", 1.5, 0.2)}, exps)


def test_verification_round():
    avg = AveragedKernel(PowerKernel(), fixed={"w": 1.0, "y": 1.0})
    for exps in ([(1.5,)], [(True,)], [np.int64(1)]):
        with pytest.raises(ValueError, match="exponents"):
            jet_columns(avg, None, {"x": ("log", 2.0, 0.1)}, exps)
    i, _ = jet_columns(avg, None, {"x": ("log", 2.0, 0.1)}, [(np.int64(1),)])
    with pytest.raises(ValueError, match="phase law"):
        kernel(
            variables=[
                Variable("a", kind="positive"),
                Variable("depth", unit="rad m^-2"),
            ],
            stokes=("I", "P"),
            depth="depth",
        )(lambda nu, a, depth: {"I": nu})

    def make():
        helper = lambda x: 2 * x  # noqa: E731

        @kernel(variables=[Variable("a", kind="positive")], stokes=("I",))
        def k(nu, a):
            return {"I": helper(a) * jnp.ones_like(nu)}

        return k

    assert make().describe() == make().describe()

    def arr_kernel(values):
        @kernel(variables=[Variable("a", kind="positive")], stokes=("I",))
        def k(nu, a):
            return {"I": values[500] * a * jnp.ones_like(nu)}

        return k

    base = np.ones(2000)
    other = base.copy()
    other[500] = 7.0
    assert arr_kernel(base).describe() != arr_kernel(other).describe()

    class NoneV:
        name = "nonev"
        variables = (Variable("x", "x", "positive", "", True, None, None, ""),)

        def channel(self, ch, point, *, phase=None, per_node=False):
            v = jnp.ones(2) * point["x"]
            return Modes(I=v, V=None, P=v[None].astype(complex))

    out = AveragedKernel(NoneV(), [("x", fixed_rule([1.0, 2.0], [0.5, 0.5]))])(None, {})
    np.testing.assert_array_equal(np.asarray(out.V), [0.0, 0.0])
