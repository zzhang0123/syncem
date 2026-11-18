"""Spec Sec. 4.6 rule 4: the depth-expansion check tau_max h, tested at its threshold.

The two methods at the dispatch are the depth-cell expansion with closed-form phase weights
(below the threshold) and direct integration with the exact Faraday phase (the remedy above
it). Both are evaluated directly at the spec threshold x = tau_max h = 1, at the proposed
threshold x^(N_d+1)/(N_d+1)! = 1e-3, at +-5 per cent around each, at orders N_d = 0 to 4,
lowest channel centres 0.1 and 10 GHz and depth references 0 and 300 rad m^-2.
"""
from __future__ import annotations

import numpy as np
import pytest

import core as C
from depth_common import record, verdict
from rules.depth import cache
from rules.depth import check as CK
from rules.depth import nodes as Nd
from rules.depth import study as S

NUS = (1e8, 1e10)
D0S = (0.0, 300.0)
NS = (0, 1, 2, 3, 4)


def _err(nu_c, d0, pop, N, x):
    ch = S.sweep_channels(nu_c)
    I, P = cache.values("cheap", ch, np.array([1e4]))
    tau = Nd.tau(ch)
    return S.depth_only_error((I[0], P[0]), ch, tau, *S.depth_population(pop, d0, x / Nd.tau_max(ch)), N)


@pytest.mark.parametrize("pop", S.POPULATIONS)
@pytest.mark.parametrize("d0", D0S)
@pytest.mark.parametrize("nu_c", NUS)
@pytest.mark.parametrize("N", NS)
def test_static_factor_bounds_depth_error(N, nu_c, d0, pop):
    """x^(N+1)/(N+1)! bounds |dP|/I on a depth-only lattice, from x = 1e-3 to 4."""
    xs = np.geomspace(1e-3, 4.0, 25)
    errs = np.array([_err(nu_c, d0, pop, N, x) for x in xs])
    bound = np.array([CK.spec_factor(x, N) for x in xs])
    ratio = float(np.max(errs / (bound + 1e-12)))
    record(f"bound_ratio_N{N}_nu{nu_c:.0e}_d{d0}_{pop}", ratio)
    assert np.all(np.isfinite(errs)) and ratio <= 1.0


@pytest.mark.parametrize("d0", D0S)
@pytest.mark.parametrize("nu_c", NUS)
@pytest.mark.parametrize("N", NS)
def test_boundary_spec_and_proposed_thresholds(N, nu_c, d0):
    """Both thresholds, both sides: worst-case population (point mass at the depth-cell edge)."""
    cells = {}
    for name, x0 in (("spec", 1.0), ("proposed", CK.x_limit(N))):
        for f in (0.95, 1.0, 1.05):
            e = max(_err(nu_c, d0, pop, N, f * x0) for pop in S.POPULATIONS)
            cells[f"{name}_x{f * x0:.4g}"] = {"err": e, "verdict": verdict(e)}
    record(f"threshold_N{N}_nu{nu_c:.0e}_d{d0}", cells)
    # the proposed rule admits only OK cells at and below its threshold
    x_p = CK.x_limit(N)
    assert all(cells[f"proposed_x{f * x_p:.4g}"]["err"] < 1e-3 for f in (0.95, 1.0))


@pytest.mark.parametrize("N", NS)
def test_spec_threshold_admits_large_errors(N):
    """At x = 1 with no bandwidth depolarisation (10 GHz, d0 = 0) the worst case exceeds 1e-3 at every N <= 4."""
    e = _err(1e10, 0.0, "edge", N, 1.0)
    expected_floor = {0: 0.3, 1: 0.1, 2: 0.03, 3: 5e-3, 4: 1e-3}[N]
    record(f"spec_threshold_worst_N{N}", {"err": e, "verdict": verdict(e)})
    assert e > expected_floor


@pytest.mark.parametrize("u_g", [-1.0, 0.0, 1.0])
@pytest.mark.parametrize("x", [0.05, 0.1, 0.182, 0.4, 1.0])
def test_total_degree_cross_terms(u_g, x):
    """r + s <= 2 drops (r, s) = (1, 2), (2, 1), (2, 2): the depth error is O(x) at the energy-cell edges.

    Depth outside the total (r <= 2, s <= 2) keeps the static bound x^3/6.
    """
    total = S.coupled_worst_case(u_g, x, C.lower_set((2, 2), 2))
    tensor = S.coupled_worst_case(u_g, x, C.lower_set((2, 2), 4))
    record(f"cross_terms_ug{u_g}_x{x}", {"total2": total, "tensor2x2": tensor, "x3_over_6": x**3 / 6})
    assert tensor <= x**3 / 6
    if u_g != 0.0 and x <= 0.4:
        assert total > x**3 / 6  # the static factor is not a bound for a total-degree lower set
