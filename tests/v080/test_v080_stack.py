"""Step 5a of design/0.8/SPEC.md Sec. 10.2: S0 and S1 tables and the stack provider (Sec. 8.2, 8.3, T2).

The frozen reweighting of ``B`` given celled ``gamma``, the field cells (T5), the refusals and the S1 build cost
are in ``test_v080_stack_cells.py``, split from this file in the step-5b-3 review with the test bodies
unchanged. The helpers stay here, since other files import them.

Memory (decision 170): before the split the fast run of this file alone peaked at 3.98 GB of resident memory
with the jet of step 5b-3 (1-minute load 7.5 to 7.3). After the split it peaked at 1.92 GB (load 5.5 to 5.8).
All loaded.
"""

import sys
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.phase import FaradayRotation
from syncem.engine._stack_series import stack_lattice
from syncem.engine.average import AveragedKernel, NodeRule, uniform_rule
from syncem.engine.field_average import TABLE_LNY, FieldAverage
from syncem.engine.provider_jet import jet_lattice
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import Legendre
from syncem.model.phase import GaussianScreen

STUDY_LIB = Path(__file__).resolve().parents[2] / "studies" / "classical-models" / "lib"


@pytest.fixture(scope="module")
def study():
    if str(STUDY_LIB) not in sys.path:
        sys.path.insert(0, str(STUDY_LIB))
    import basis as SB  # noqa: PLC0415
    from config import PAPER  # noqa: PLC0415

    return SB, PAPER


def small_channels(n=5, n_nu=32):
    return Channels.log(0.1, 10, n, unit="GHz", half_width=0.2, n_nu=n_nu)


def screen():
    return FaradayRotation(screen=GaussianScreen(8.0, 0.5))


def channel_max(a):
    return np.abs(np.asarray(a)).max(axis=0)


def rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)) / channel_max(b)))


# ------------------------------------------------------------------ S1: lookups against quadrature


def _s1_error(table, nodes, weights, which, order, seed=7):
    """Pointwise errors of ln T and of T^(1), T^(2), and the error of every T^(r) against the peak of |T^(r)|.

    High orders change sign, so a pointwise relative error near a zero measures the distance to the root,
    not the table. The peak-normalised error is the scale that enters the channel integrals.
    """
    lny = np.sort(np.random.default_rng(seed).uniform(TABLE_LNY[0], np.log(5e4), 1500))
    ln_t, tau = FieldAverage.direct(
        np.exp(lny), np.asarray(nodes, float), np.asarray(weights, float), which, order
    )
    keep = ln_t > -650.0
    ev = table.evaluate(lny[keep], which, order)
    t = np.exp(ln_t[keep])
    point = [float(np.max(np.abs(np.log(ev[0]) - ln_t[keep])))]
    for r in (1, 2):
        point.append(
            float(
                np.max(
                    np.abs(ev[r] - t * tau[keep, r])
                    / (t * np.maximum(1.0, np.abs(tau[keep, r])))
                )
            )
        )
    peak = [
        float(
            np.max(np.abs(ev[r] - t * tau[keep, r])) / np.max(np.abs(t * tau[keep, r]))
        )
        for r in range(order + 1)
    ]
    return point, peak


@pytest.mark.parametrize("b_ug", [0.1, 1.0, 4.0, 6.0, 50.0, 100.0])
def test_s1_lookups_against_quadrature_one_node(b_ug):
    table = FieldAverage.lookups([b_ug], [1.0])
    for which in ("F", "G"):
        point, peak = _s1_error(table, [b_ug], [1.0], which, 6)
        assert max(point) < 1e-8 and max(peak) < 1e-8, (
            which,
            point,
            peak,
        )  # S0 spline floor 3.4e-9


@pytest.mark.parametrize("n", [2, 8, 16])
def test_s1_lookups_against_quadrature_averages(n):
    t, w = np.polynomial.legendre.leggauss(n)
    nodes, weights = 5.0 + t, 0.5 * w
    table = FieldAverage.lookups(nodes, weights)
    for which in ("F", "G"):
        point, peak = _s1_error(table, nodes, weights, which, 6)
        assert max(point) < 1e-8 and max(peak) < 1e-8, (which, point, peak)


@pytest.mark.parametrize("b_ug", [0.1, 4.0, 100.0])
def test_s1_lookup_table_equals_the_quadrature_table(b_ug):
    """The lookup error alone: both tables share the S1 spline in ln y (3.4e-9 in ln T between knots)."""
    lny = np.sort(np.random.default_rng(3).uniform(TABLE_LNY[0], np.log(5e4), 1500))
    quad, look = FieldAverage.quadrature([b_ug], [1.0]), FieldAverage.lookups(
        [b_ug], [1.0]
    )
    for which in ("F", "G"):
        q, lk = quad.evaluate(lny, which, 6), look.evaluate(lny, which, 6)
        keep = q[0] > np.exp(-650.0)
        assert (
            np.max(np.abs(np.log(lk[0][keep]) - np.log(q[0][keep]))) < 1e-12
        )  # measured 1.7e-13 at 4 uG
        for r in range(1, 7):
            peak = np.max(np.abs(q[r][keep]))
            assert (
                np.max(np.abs(lk[r][keep] - q[r][keep])) < 1e-9 * peak
            ), r  # measured 3.3e-10 at r = 6


def test_s1_refusals():
    with pytest.raises(ValueError):
        FieldAverage.lookups([-1.0], [1.0])
    with pytest.raises(ValueError):
        FieldAverage.lookups([1.0, 2.0], [0.0, 0.0])
    table = FieldAverage.lookups([5.0], [1.0])
    with pytest.raises(ValueError, match="below"):
        table.evaluate(np.array([TABLE_LNY[0] - 1.0]), "F", 2)
    assert float(table.evaluate(np.array([TABLE_LNY[1] + 1.0]), "F", 0)[0][0]) == 0.0


# ------------------------------------------------------------------ columns against the study oracle


def paper_setup(paper, n_nu=400, order=4):
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
    lattice = Lattice(gamma=Cells.log(300, paper.gamma_hi, 19, order=order))
    return ch, avg, lattice


@pytest.mark.parametrize(("s1", "tol"), [("quadrature", 1e-10), ("lookup", 1e-7)])
def test_stack_equals_the_study_oracle(study, s1, tol):
    sb, paper = study
    ch, avg, lattice = paper_setup(paper)
    i, p = stack_lattice(avg, ch, lattice, s1=s1)
    di, dq = sb.derivative_spectra(paper, 4)
    di, dq = di.reshape(-1, 41), dq.reshape(-1, 41)  # rows b (N + 1) + r
    assert rel(i, di) < tol and rel(p, dq) < tol


# ------------------------------------------------------------------ T2: stack against the jet


def test_cached_stack_equals_the_jet():
    ch = small_channels()
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        [("B", uniform_rule(4e-6, 6e-6, 6)), ("eta", uniform_rule(0.0, 0.5, 6))],
        phase=screen(),
    )
    lattice = Lattice(gamma=Cells.log(300, 3e6, 4))
    si, sp = stack_lattice(avg, ch, lattice)
    ji, jp = jet_lattice(avg, ch, lattice)
    assert rel(si, ji) < 1e-10 and rel(sp, jp) < 1e-10


def _per_node_case(levels, phase):
    ch = small_channels()
    avg = AveragedKernel(ContinuumKernel(x_min=1e-12), levels, phase=phase)
    lattice = Lattice(gamma=Cells.log(300, 3e6, 3))
    return ch, avg, lattice


def test_eta_given_averaged_b():
    eta = uniform_rule(0.0, lambda p: 0.3 + 4e4 * p["B"], 6, parents=("B",))
    ch, avg, lattice = _per_node_case(
        [("B", uniform_rule(4e-6, 6e-6, 4)), ("eta", eta)], screen()
    )
    assert (
        rel(stack_lattice(avg, ch, lattice)[0], jet_lattice(avg, ch, lattice)[0])
        < 1e-10
    )


def test_b_given_averaged_eta():
    b = uniform_rule(lambda p: 4e-6 * (1 + p["eta"]), 6e-6, 4, parents=("eta",))
    ch, avg, lattice = _per_node_case(
        [("eta", uniform_rule(0.0, 0.5, 4)), ("B", b)], screen()
    )
    si, sp = stack_lattice(avg, ch, lattice)
    ji, jp = jet_lattice(avg, ch, lattice)
    assert rel(si, ji) < 1e-10 and rel(sp, jp) < 1e-10


def test_depth_given_averaged_b():
    depth = uniform_rule(
        lambda p: 6.0 + 4e5 * p["B"], lambda p: 10.0 + 4e5 * p["B"], 4, parents=("B",)
    )
    ch, avg, lattice = _per_node_case(
        [
            ("B", uniform_rule(4e-6, 6e-6, 3)),
            ("eta", uniform_rule(0.0, 0.5, 3)),
            ("depth", depth),
        ],
        FaradayRotation(),
    )
    si, sp = stack_lattice(avg, ch, lattice)
    ji, jp = jet_lattice(avg, ch, lattice)
    assert rel(si, ji) < 1e-10 and rel(sp, jp) < 1e-10


def test_b_given_a_legendre_eta():
    b = uniform_rule(
        lambda p: 4e-6 * (1 + 0.5 * p["eta"] ** 2), 6e-6, 4, parents=("eta",)
    )
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        [("B", b)],
        legendre={"eta": (2, 12)},
        phase=screen(),
    )
    ch, lattice = small_channels(), Lattice(
        gamma=Cells.log(300, 3e6, 3), eta=Legendre(2)
    )
    si, sp = stack_lattice(avg, ch, lattice)
    ji, jp = jet_lattice(avg, ch, lattice)
    assert rel(si, ji) < 1e-10 and rel(sp, jp) < 1e-10


def test_depth_given_celled_gamma_by_frozen_reweighting():
    """A conditional on gamma of a variable other than B: depth ~ N(8 + 0.02 ln(gamma / 1e4), 0.05^2).

    The width keeps the Faraday phase across the law (about 18 rad per unit depth at 0.1 GHz) resolved by 12 nodes.
    """
    t, w = np.polynomial.hermite.hermgauss(12)

    def mean(p):
        return 8.0 + 0.02 * jnp.log(p["gamma"] / 1e4)

    def density(x, p):
        return jnp.exp(-0.5 * ((x - mean(p)) / 0.05) ** 2) / (0.05 * np.sqrt(2 * np.pi))

    depth = NodeRule(
        lambda p: (mean(p) + 0.05 * np.sqrt(2.0) * t, jnp.asarray(w / np.sqrt(np.pi))),
        parents=("gamma",),
        density=density,
    )
    ch, avg, lattice = _per_node_case(
        [
            ("B", uniform_rule(4e-6, 6e-6, 3)),
            ("eta", uniform_rule(0.0, 0.5, 3)),
            ("depth", depth),
        ],
        FaradayRotation(),
    )
    si, sp = stack_lattice(avg, ch, lattice)
    ji, jp = jet_lattice(avg, ch, lattice)
    assert rel(si, ji) < 1e-10 and rel(sp, jp) < 1e-9


def lognormal_given_gamma(width, slope, n=16):
    """``ln B ~ N(ln 5 uG + slope ln(gamma / 1e4), width^2)`` by Gauss-Hermite, with its density."""
    t, w = np.polynomial.hermite.hermgauss(n)

    def median(p):
        return jnp.log(5e-6) + slope * jnp.log(p["gamma"] / 1e4)

    def fn(p):
        return jnp.exp(median(p) + width * np.sqrt(2.0) * t), jnp.asarray(
            w / np.sqrt(np.pi)
        )

    def density(x, p):
        z = (jnp.log(x) - median(p)) / width
        return jnp.exp(-0.5 * z**2) / (x * width * np.sqrt(2 * np.pi))

    return NodeRule(fn, parents=("gamma",), density=density)
