"""Basis build, design assembly, ``at`` and ``with_order`` of step 5b-1 (design/0.8/SPEC.md decisions 149, 151, 153).

E1, E5 and E6 designs equal the study oracles of ``studies/classical-models``, read in place through ``_study``
(decision 154), at 1e-10 with the quadrature S1 and 1e-7 with lookups (SPEC Sec. 9.3). The errors are taken per
channel and Stokes parameter against the largest column of the oracle. Probes ran on 2026-11-19 before
``basis.py`` existed, by hand assembly from ``route_cells`` and ``column_map``, at a 1-minute load of 9.6 to 13.6
(scripts ``probe_basis.py`` to ``probe_basis4.py`` in the session scratchpad). Each tolerance names its probe value
and margin, a SPEC number or an exact identity.
"""

import dataclasses
import inspect
import re

import numpy as np
import pytest

import _study
from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.phase import FaradayRotation
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine import _routes
from syncem.engine.assemble import predict_values
from syncem.engine.average import AveragedKernel, fixed_rule, uniform_rule
from syncem.engine.basis import build_from_averaged
from syncem.engine.provider_jet import jet_lattice
from syncem.fitting.data import StokesData
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.moments import Moments
from syncem.lattice.representations import AzimuthHarmonic
from syncem.model.phase import GaussianScreen
from tests.v080.test_v080_maps import direct_moments, field_density, gamma_density
from tests.v080.test_v080_routes import REACH, cached_avg, fixed_b_avg, only_i
from tests.v080.test_v080_stack import screen, small_channels
from tests.v080.test_v080_stack_boundaries import _lognormal_avg, gamma_lattice

K = ContinuumKernel(x_min=1e-12)
TOL = {"quadrature": 1e-10, "lookup": 1e-7}  # SPEC Sec. 9.3 against the study oracle
HEX64 = r"[0-9a-f]{64}"


def rel(a, b):
    """Largest error per channel and Stokes parameter over the largest column of ``b`` there."""
    a, b = np.asarray(a), np.asarray(b)
    return float(np.max(np.abs(a - b) / np.abs(b).max(axis=2, keepdims=True)))


def counts(basis):
    """(channel rows computed, channel rows reused) of the last build or update."""
    found = dict(basis.record.events)
    return found["channel rows computed"], found["channel rows reused"]


@pytest.fixture(scope="module")
def paper():
    cfg = _study.paper()
    return cfg, _study.kernel_table(cfg)


def paper_build(cfg, lattice, s1, factor=None):
    """The study's B and eta nodes as fixed rules, B averaged unless the lattice cells it."""
    b, wb = cfg.b_nodes_weights
    eta = fixed_rule(*cfg.eta_nodes_weights)
    levels = [("B", fixed_rule(b * 1e-6, wb)), ("eta", eta)]
    avg = AveragedKernel(
        K,
        [lv for lv in levels if lv[0] not in lattice.cells.variables],
        phase=FaradayRotation(screen=GaussianScreen(cfg.screen_mean, cfg.screen_sd)),
    )
    ch = Channels.log(0.1, 10, 41, unit="GHz", half_width=0.2, n_nu=400)
    return build_from_averaged(avg, ch, lattice, s1=s1, azimuth_factor=factor)


@pytest.fixture
def no_providers(monkeypatch):
    for name in ("stack_cell", "core_cell", "jet_lattice"):
        monkeypatch.setattr(_routes, name, lambda *a, **k: pytest.fail("provider call"))


# ------------------------------------------------------------------ 1-4. study oracles


@pytest.mark.parametrize("s1", ["quadrature", "lookup"])
@pytest.mark.parametrize("N", [2, 4])
def test_e1_and_e5_designs_equal_the_study_oracles(paper, N, s1):
    """Probe: E1 5.8e-14 (quadrature) and 5.8e-13 / 2.6e-12 (lookups at N = 2 / 4), E5 5.3e-14 and 4.4e-13 /
    1.6e-12. Builds of 4.8 to 5.9 s with the quadrature S1 and 0.7 to 0.9 s with lookups.
    """
    cfg, table = paper
    di, dq = _study.derivative_spectra(cfg, N, table)
    gamma = Cells.log(300, cfg.gamma_hi, 19, order=N)
    e1 = paper_build(cfg, Lattice(gamma=gamma), s1, cfg.azimuth_factor)
    e5 = paper_build(cfg, Lattice(gamma=gamma, phi=AzimuthHarmonic()), s1)
    assert e1.design().shape == (41, 3, 19 * (N + 1))
    assert e5.design().shape == (41, 3, 57 * (N + 1))
    assert rel(e1.design(), _study.separable_columns(cfg, di, dq)) < TOL[s1]
    assert rel(e5.design(), _study.joint_columns(cfg, di, dq)) < TOL[s1]
    routes = {(r.provider, r.mode, r.s1) for r in e5.record.routes}
    assert routes == {("stack", "cached-S1", s1)}
    assert re.fullmatch(HEX64, e1.key) and e1.record.notes == ()  # cached in step 5b-2
    assert e5.spectra2.shape == (41, 19 * (N + 1)) and e5.columns.n2 == 19 * (N + 1)


@pytest.mark.parametrize("s1", ["quadrature", "lookup"])
def test_e6_design_equals_the_study_oracle(paper, s1):
    """Columns permuted to the study's order ``6 cell + monomial`` through ``MomentIndex.arrays``. Probe: 5.8e-14
    (quadrature) and 8.1e-13 (lookups), field-scaling mode, builds of 0.6 to 0.7 s."""
    cfg, _ = paper
    lattice = Lattice(
        gamma=Cells.log(300, cfg.gamma_hi, 19, order=2),
        B=Cells.single("5 uG", scale="1 uG", order=2),
        order=2,
    )
    basis = paper_build(cfg, lattice, s1, cfg.azimuth_factor)
    rows, mons = lattice.index.arrays, _study.e6_monomials(2)
    perm = [
        6 * rows.cell[k] + mons.index(tuple(map(int, rows.powers[k])))
        for k in basis.columns.column_base
    ]
    assert basis.design().shape == (41, 3, 114)
    assert {r.mode for r in basis.record.routes} == {"field-scaling"}
    assert rel(basis.design(), _study.e6_joint_columns(cfg, 2)[:, :, perm]) < TOL[s1]


SLOW = [pytest.param(s, 96, marks=pytest.mark.slow) for s in ("lookup", "quadrature")]


@pytest.mark.parametrize("s1, n_b", [("lookup", 16), *SLOW])
def test_e1m_maxwell_design_equals_the_study_oracle(s1, n_b):
    """Maxwellian field, N = 4. With the paper's 96 nodes the study table takes 25.6 s and the quadrature S1
    build 26.4 s, the lookup build 2.0 s. Probe: 1.9e-12 (lookups) and 5.9e-14 (quadrature). The fast case
    uses 16 nodes of the same rule on both sides: table 3.9 s, build 0.7 s, probe 2.2e-12 at a 1-minute load of
    1.9 (``probe_e1m16.py`` in the session scratchpad)."""
    cfg = dataclasses.replace(_study.paper_maxwell(), n_b=n_b)
    di, dq = _study.derivative_spectra(cfg, 4, _study.kernel_table(cfg))
    lattice = Lattice(gamma=Cells.log(300, cfg.gamma_hi, 19, order=4))
    basis = paper_build(cfg, lattice, s1, cfg.azimuth_factor)
    assert rel(basis.design(), _study.separable_columns(cfg, di, dq)) < TOL[s1]


# ------------------------------------------------------------------ 5. Stokes layout


@kernel(variables=[Variable("a", kind="positive")], stokes=("I", "P"))
def ip_kernel(nu, a):
    return {"I": a * nu, "P": (0.3 + 0.1j) * a**2 * nu}


def test_stokes_axes_and_the_re_im_map_are_exact():
    lat = Lattice(gamma=Cells.log(300, 3e5, 2), phi=AzimuthHarmonic())
    basis = build_from_averaged(cached_avg(), small_channels(), lat)
    d = np.asarray(basis.design())
    blocks = np.asarray(lat.index.blocks)[basis.columns.column_block]
    assert basis.record.stokes == ("I", "Q", "U")
    np.testing.assert_array_equal(np.asarray(basis.design(stokes="I")), d[:, :1])
    np.testing.assert_array_equal(np.asarray(basis.design(stokes="P")), d[:, 1:])
    np.testing.assert_array_equal(np.asarray(basis.design(stokes="QU")), d[:, 1:])
    re, im = np.flatnonzero(blocks == "Re M2"), np.flatnonzero(blocks == "Im M2")
    np.testing.assert_array_equal(d[:, 1, im], -d[:, 2, re])  # Q = -b, U = a on Im M2
    np.testing.assert_array_equal(d[:, 2, im], d[:, 1, re])
    assert not d[:, 0, re].any() and not d[:, 1:, blocks == "M0"].any()
    with pytest.raises(ValueError, match="V"):
        basis.design(stokes="V")
    sigma = np.arange(1.0, 16.0).reshape(5, 3)
    data = StokesData(
        basis.channels, stokes=np.zeros_like(sigma), sigma=sigma, unit="cgs"
    )
    expected = (d / sigma[..., None]).reshape(-1, d.shape[-1])
    np.testing.assert_array_equal(basis.design(data=data), expected)
    with pytest.raises(TypeError, match="StokesData"):
        basis.design(data=np.zeros((5, 3)))


def test_azimuth_factor_rules():
    ch, avg, lat = small_channels(), cached_avg(), Lattice(gamma=Cells.log(300, 3e5, 2))
    with pytest.raises(ValueError, match="azimuth_factor"):
        build_from_averaged(avg, ch, lat)
    with pytest.raises(ValueError, match="AzimuthHarmonic"):
        harmonic = Lattice(gamma=Cells.log(300, 3e5, 2), phi=AzimuthHarmonic())
        build_from_averaged(avg, ch, harmonic, azimuth_factor=0.4)
    only = build_from_averaged(avg, ch, lat, stokes="I")
    assert only.spectra2 is None and only.design().shape == (5, 1, 6)
    c = 0.4 * np.exp(0.5j)
    full = build_from_averaged(avg, ch, lat, azimuth_factor=c)
    w = lat.weights[full.columns.column_cell]
    p = np.asarray(full.spectra2)[:, full.columns.spin2_from] * c * w
    np.testing.assert_array_equal(
        np.asarray(full.design())[:, 1:], np.stack([p.real, p.imag], axis=1)
    )


def test_a_kernel_without_azimuth_gives_p_with_factor_one_equal_to_the_jet():
    ch = small_channels()
    avg = AveragedKernel(ip_kernel)
    lat = Lattice(a=Cells.log(1.0, 30.0, 3, order=2))
    basis = build_from_averaged(avg, ch, lat)
    _, p = (np.asarray(x) for x in jet_lattice(avg, ch, lat))
    w = lat.weights[lat.index.arrays.cell]
    d = np.asarray(basis.design())
    np.testing.assert_array_equal(d[:, 1], (p.real * w[:, None]).T)
    np.testing.assert_array_equal(d[:, 2], (p.imag * w[:, None]).T)


def test_stokes_blocks_outside_the_kernel_raise():
    ch, gamma, a = small_channels(), Cells.log(300, 3e5, 2), Cells.log(1.0, 3.0, 2)
    with pytest.raises(ValueError, match=r"\('I', 'P'\)"):
        build_from_averaged(cached_avg(), ch, Lattice(gamma=gamma), stokes="IV")
    with pytest.raises(ValueError, match=r"\('I',\)"):
        build_from_averaged(AveragedKernel(only_i), ch, Lattice(a=a), stokes="IQU")


# ------------------------------------------------------------------ 6-8. w, keys, corner split


def test_predictions_do_not_depend_on_w():
    """w scaled by 1e-3, 1, 1e2 and 1e5 per cell, moments by the inverse. Probe: 4.9e-17 of the channel maximum
    (bound 1e-15, margin 20). Rounding of two products per term."""
    ch, avg, cells = small_channels(), cached_avg(), Cells.log(300, 3e5, 4, order=2)
    lat0 = Lattice(gamma=cells)
    scale = 10.0 ** np.array([-3.0, 0.0, 2.0, 5.0])
    lat1 = Lattice(gamma=cells, weights=lat0.weights * scale)
    m0 = direct_moments(lat0, {"gamma": gamma_density}, {})
    m1 = m0 / np.repeat(scale, np.diff(np.asarray(lat0.index.offsets)))
    b0 = build_from_averaged(avg, ch, lat0, azimuth_factor=0.4)
    b1 = build_from_averaged(avg, ch, lat1, azimuth_factor=0.4)
    p0 = np.asarray(predict_values(b0, Moments(lat0, m0)))
    p1 = np.asarray(predict_values(b1, Moments(lat1, m1)))
    assert np.max(np.abs(p0 - p1) / np.abs(p0).max(axis=0)) < 1e-15


def test_key_rule():
    ch, avg, lat = small_channels(), cached_avg(), Lattice(gamma=Cells.log(300, 3e5, 2))
    basis = build_from_averaged(avg, ch, lat, stokes="I")
    values = np.ones(lat.n)
    assert np.asarray(predict_values(basis, Moments(lat, values))).shape == (5, 1)
    with pytest.raises(ValueError, match="with_key"):
        predict_values(basis, Moments(lat.with_key(None), values))
    other = Lattice(gamma=Cells.log(300, 3e5, 2, order=1))
    with pytest.raises(ValueError, match="key"):
        predict_values(basis, Moments(other, np.ones(4)))
    keyed = build_from_averaged(avg, ch, lat.with_key("carried"), stokes="I")
    np.testing.assert_array_equal(
        np.asarray(predict_values(keyed, Moments(lat.with_key("carried"), values))),
        np.asarray(predict_values(basis, Moments(lat, values))),
    )
    assert tuple(inspect.signature(predict_values).parameters) == ("basis", "moments")
    unkeyed = build_from_averaged(avg, ch, lat.with_key(None), stokes="I")
    assert unkeyed.columns.key is None
    assert "not cached: lattice key is None" in unkeyed.record.notes


def test_corner_split_predicts_like_the_uniform_refinement_on_shared_cells():
    """Deferred from step 2 (``test_v080_maps.py``). The corner cells carry 0.85 per cent of I. SPEC Sec. 10.2 bound
    1e-12 of I_j. Probe: 0.0 (equal cells, references and weights give equal spectra), field-scaling stack.
    """
    parent = Lattice(
        gamma=Cells.log(300, 3e5, 4), B=Cells.linear("2 uG", "8 uG", 3), order=2
    )
    split = parent.refine(where={"gamma": 0, "B": 0}, gamma=2, B=2)
    uniform = Lattice(
        gamma=Cells.log(300, 3e5, 8), B=Cells.linear("2 uG", "8 uG", 6), order=2
    )
    avg = AveragedKernel(K, [("eta", uniform_rule(0, 0.5, 4))], phase=screen())
    hi = float(np.asarray(parent.cells.hi)[0, 0])
    corner = {"gamma": (300.0, hi), "B": (2e-6, 4e-6)}
    out = []
    for lat in (split, uniform):
        values = direct_moments(lat, {"gamma": gamma_density, "B": field_density}, {})
        shared = np.zeros_like(values)
        rows = list(lat.index.select(**corner))
        shared[rows] = values[rows]
        basis = build_from_averaged(avg, small_channels(), lat, azimuth_factor=0.4)
        out.append(
            [
                np.asarray(predict_values(basis, Moments(lat, v)))
                for v in (shared, values)
            ]
        )
    i_j = np.abs(out[1][1][:, :1])
    assert np.max(np.abs(out[0][0] - out[1][0]) / i_j) < 1e-12
    assert np.max(np.abs(out[1][0][:, 0]) / i_j[:, 0]) > 1e-3  # the corner contributes


# ------------------------------------------------------------------ 9. at


def mixed_basis(ch, order=2, cache=True):
    lat, routes = gamma_lattice(2, order), ("stack", "jet")
    return build_from_averaged(
        cached_avg(), ch, lat, routes=routes, azimuth_factor=0.4, cache=cache
    )


def test_at_on_a_channel_subset_gathers_rows_without_provider_calls(request):
    parent = mixed_basis(small_channels())
    request.getfixturevalue("no_providers")
    rows = [3, 0, 0, 4]
    child = parent.at(parent.channels[rows])
    np.testing.assert_array_equal(
        np.asarray(child.design()), np.asarray(parent.design())[rows]
    )
    assert child.record.routes == parent.record.routes
    assert counts(child) == (0, 4)


def test_at_on_new_channels_computes_only_the_new_rows(monkeypatch):
    """5 of the 9 channels share row keys with the parent. Review probe: 0.0 against the ``cache=False`` build
    (bound 1e-15, as the two need not agree bitwise, decisions 153 and 164)."""
    narrow, wide = (
        Channels.log(0.1, 10, n, unit="GHz", half_width=0.2, n_nu=32) for n in (5, 9)
    )
    parent = mixed_basis(narrow)
    seen = []
    for name in ("stack_cell", "jet_lattice"):
        real = getattr(_routes, name)

        def spy(*a, real=real, name=name, **k):
            seen.append((name, a[1].n_ch))
            return real(*a, **k)

        monkeypatch.setattr(_routes, name, spy)
    child = parent.at(wide)
    assert sorted(seen) == [("jet_lattice", 4), ("stack_cell", 4)]
    assert counts(child) == (4, 5)
    fresh = mixed_basis(
        wide, cache=False
    )  # independent of the S2 store that at() wrote
    assert rel(child.design(), fresh.design()) < 1e-15


def test_at_moves_a_cell_the_stack_refuses_on_new_channels_to_the_jet():
    """One cell around gamma = 1e8 at 5 uG: inside the S1 grid from 0.08 GHz, outside it from 0.016 GHz (probe)."""
    lat = Lattice(gamma=Cells.log(1e8 / 1.2, 1e8 * 1.2, 1, order=1), order=1)
    low = Channels.log(0.02, 10, 6, unit="GHz", half_width=0.2, n_nu=32)
    parent = build_from_averaged(
        fixed_b_avg(), small_channels(), lat, azimuth_factor=0.4
    )
    assert parent.record.routes[0].provider == "stack"
    child = parent.at(low)
    route = child.record.routes[0]
    assert route.provider == "jet" and REACH in route.reason
    fresh = build_from_averaged(
        fixed_b_avg(), low, lat, routes="jet", azimuth_factor=0.4
    )
    np.testing.assert_array_equal(
        np.asarray(child.design()), np.asarray(fresh.design())
    )


def test_at_below_the_kernel_x_min_names_resolved():
    ch, avg = small_channels(), fixed_b_avg()
    lat = Lattice(gamma=Cells.log(300, 3e5, 2, order=1), order=1)
    resolved = ContinuumKernel().resolved(ch, float(lat.cells.reference[1, 0]), 5e-6)
    avg = AveragedKernel(resolved, avg.averaged, fixed=dict(avg.fixed), phase=avg.phase)
    parent = build_from_averaged(avg, ch, lat, routes="jet", azimuth_factor=0.4)
    with pytest.raises(ValueError, match=r"ContinuumKernel\.resolved"):
        parent.at(Channels.log(0.02, 10, 6, unit="GHz", half_width=0.2, n_nu=32))


# ------------------------------------------------------------------ 10. with_order


def test_with_order_below_the_built_order_gathers_without_provider_calls(request):
    """Against a fresh order-2 build with the parent's routes. Probe: 0.0 for I and 2.0e-17 of the channel maximum
    for P (bound 1e-15, margin 50)."""
    ch = small_channels()
    parent, fresh = mixed_basis(ch, 4), mixed_basis(ch, 2)
    request.getfixturevalue("no_providers")
    child = parent.with_order(2)
    assert child.columns.n == fresh.columns.n == 6
    assert rel(child.design(), fresh.design()) < 1e-15
    assert child.record.routes == parent.record.routes
    assert "routes of the order-4 build" in child.record.notes
    same = parent.with_order(4)
    np.testing.assert_array_equal(
        np.asarray(same.design()), np.asarray(parent.design())
    )


def test_with_order_keeps_the_jet_route_at_the_amplification_boundary():
    """``ln B`` of width 0.01 moving with slope 0.1 across one wide cell: the stack refuses at N = 4 (amplification
    5.8e5 above the bound 1e4) and accepts at N = 2. Check (a) of SPEC Sec. 8.3 between the kept jet route and the
    fresh stack route: probe 1.3e-11 of the channel maximum (bound 1e-10) and 1.1e-11 of the own peak (bound 1e-8).
    """
    ch, avg = small_channels(), _lognormal_avg(0.01, 0.1)
    parent = build_from_averaged(avg, ch, gamma_lattice(1, 4), azimuth_factor=0.4)
    route = parent.record.routes[0]
    assert route.provider == "jet" and "amplifies" in route.reason
    child = parent.with_order(2)
    assert child.record.routes == parent.record.routes
    fresh = build_from_averaged(
        avg, ch, gamma_lattice(1, 2), s1="quadrature", azimuth_factor=0.4
    )
    assert fresh.record.routes[0].provider == "stack"
    pairs = (
        (child.spectra0[:, 0], fresh.spectra0[:, 0]),
        (child.spectra2, fresh.spectra2),
    )
    for jet, stack in ((np.asarray(a), np.asarray(b)) for a, b in pairs):
        err, scale = np.abs(stack - jet), np.abs(jet)  # (n_ch, stored spectra)
        assert np.max(err / scale.max(axis=1, keepdims=True)) < 1e-10  # channel maximum
        assert np.max(err / scale.max(axis=0, keepdims=True)) < 1e-8  # own peak
