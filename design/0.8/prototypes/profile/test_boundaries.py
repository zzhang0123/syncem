"""Method-against-method checks for the alternatives this profile proposes, and the harmonic switch.

Boundary-validation rule: both methods are evaluated directly (no dispatcher) at the threshold,
next to it and at extreme parameter values. rel_err < 1e-3 passes. Tighter pins record the
measured agreement so that drift shows.

1. S1 field average: S0 table in ln x plus lookups (p1_s1_table.s1_from_s0) against the study
   quadrature (KernelTable.direct), single field nodes from 0.1 to 100 uG, orders 0 to 6.
2. Tabulated F, G (p6_ray.TabulatedKernel) against syncem.ultrarel F, G (128-node quadrature)
   from x = 1e-13 to the table's upper end, on both sides of the end where the table returns 0.
3. syncem high_order_channel_modes: direct line sum against the corrected integral on the same
   channel, with the candidate count at, below and above direct_limit = 4096, gamma 2 to 3000.

Run from prototypes/profile: python -m pytest test_boundaries.py -q   (about 1 minute)
"""
from __future__ import annotations

import math

import numpy as np
import pytest

import _common as U  # noqa: F401  (paths)

import syncem  # noqa: F401
import jax.numpy as jnp

import basis as SB
import p1_s1_table as P1

OK = 1e-3
RNG = np.random.default_rng(7)
LNY = np.sort(RNG.uniform(SB.TABLE_LNY[0], np.log(5e4), 1500))


@pytest.fixture(scope="module")
def s0_wide():
    lo = SB.TABLE_LNY[0] - np.log(100.0) - 0.01
    hi = np.log(5e4) - np.log(0.1) + 0.01
    return P1.s0_table(lo, hi)


@pytest.mark.parametrize("which", ["F", "G"])
@pytest.mark.parametrize("b_node", [0.1, 1.0, 4.0, 6.0, 50.0, 100.0])
def test_s1_lookup_equals_quadrature(s0_wide, which, b_node):
    b, w = np.array([b_node]), np.array([1.0])
    tab = SB.KernelTable.__new__(SB.KernelTable)
    tab.max_order, tab.b_nodes, tab.b_weights = P1.ORDER, b, w
    ln_ref, tau_ref = tab.direct(np.exp(LNY), which)
    ln_t, tau = P1.s1_from_s0(s0_wide, LNY, b, w, which)
    err = P1.table_error(ln_t, tau, ln_ref, tau_ref)
    assert max(err.values()) < OK, err
    assert err["lnT_max_abs"] < 1e-10 and err["r2"] < 1e-9 and err["r6"] < 1e-6, err


@pytest.fixture(scope="module")
def tabulated():
    import core as C
    import p6_ray as P6

    return P6.TabulatedKernel(C.ContinuumChannelKernel(C.bump_channels(n=3, n_nodes=8)))


def _edge(tab):
    return tab.x0 + tab.h * (tab.n - 1)


@pytest.mark.parametrize("which", ["F", "G"])
@pytest.mark.parametrize("where", ["x=1e-13", "x=1e-6", "x=0.3", "x=10", "x=40",
                                   "edge-0.05", "edge-1e-6", "edge+1e-6", "edge+0.05"])
def test_tabulated_kernel_equals_quadrature(tabulated, which, where):
    from syncem.ultrarel import F, G

    edge = _edge(tabulated)
    if where.startswith("x="):
        lnx = math.log(float(where[2:]))
    else:
        lnx = edge + float(where[4:])
    quad = float((F if which == "F" else G)(jnp.exp(lnx)))
    above = lnx > edge
    tab = 0.0 if above else float(jnp.exp(tabulated._lnW(which, jnp.asarray(lnx))))
    if above:
        # beyond the table both values are below e^-80 of the order-unity peak
        assert quad < 1e-34 and tab == 0.0
        return
    rel = abs(tab - quad) / max(abs(tab), abs(quad), 1e-300)
    assert rel < OK, (where, tab, quad, rel)
    if lnx < edge - 40 * tabulated.h:
        assert rel < 1e-8, (where, rel)


def _harmonic_channel(gamma, factor):
    """A bump channel whose candidate count is about factor * 4096 (direct_limit), centred at
    harmonic number max(0.4 gamma^3, 3 * 4096 * factor), near the emission peak where possible.
    At gamma = 5 the centre lies in the exponential tail (about e^-65 of the peak)."""
    from syncem.model import Channels
    from syncem.model.high_order_harmonic import _geometry
    from syncem.constants import C_CGS, E_ESU, M_E

    D = _geometry(gamma, 0.2, 0.2)[2]
    spacing = E_ESU * 5e-6 / (2.0 * math.pi * gamma * M_E * C_CGS * D)
    m_c = max(0.4 * gamma**3, 3 * 4096 * factor)
    width = 0.5 * factor * 4096 * spacing
    return Channels.bump([m_c * spacing], [width])


def harmonic_rel(gamma, factor):
    from syncem.model.high_order_harmonic import high_order_channel_modes

    ch = _harmonic_channel(gamma, factor)
    direct = high_order_channel_modes(ch, gamma, 5e-6, 0.2, 0.2, direct_limit=10**6)
    integ = high_order_channel_modes(ch, gamma, 5e-6, 0.2, 0.2, direct_limit=1, max_quad_order=2048)
    assert direct.routes == ("direct",) and integ.routes == ("corrected_integral",)
    a = np.concatenate([direct.modes.I, direct.modes.V, np.abs(direct.modes.P).ravel()])
    b = np.concatenate([integ.modes.I, integ.modes.V, np.abs(integ.modes.P).ravel()])
    scale = float(np.max(np.abs(a)))
    return float(np.max(np.abs(a - b)) / scale) if scale > 0 else float("nan"), scale


@pytest.mark.parametrize("gamma", [5.0, 30.0, 300.0, 3000.0])
@pytest.mark.parametrize("factor", [0.5, 1.0, 2.0])
def test_harmonic_direct_equals_integral_at_switch(gamma, factor):
    rel, scale = harmonic_rel(gamma, factor)
    assert scale > 0, "both routes zero: the comparison would be vacuous"
    assert rel < OK, (gamma, factor, rel)
