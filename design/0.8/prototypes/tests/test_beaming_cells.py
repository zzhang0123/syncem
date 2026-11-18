"""(c) Cells on the pitch cosine under Beaming: the factor 2 of K_nat -> 2 K_iso delta(mu - eta).

Two constructions of Spec Sec. 4.2, both against the beaming integral on the diagonal mu = eta.
* eta on Legendre(L): mu-cell columns (1/r!) d^r/du^r [(2k+1) P_k(mu) K(eta = mu)], moments
  <1_b(mu) u^r P_k(eta)>. Population p(mu) q(eta), direct = int 2 p(x) q(x) K(x) dx.
* eta averaged with density q: mu-cell columns of g(mu) = 2 q(mu) K(eta = mu), moments <1_b u^r>.
The factor 1 (the v3 normalisation of r3-03) is run alongside and gives half the emission.
Run: PYTHONPATH=.:../../.. python -m pytest tests/test_beaming_cells.py
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import core as C
from core import paper as E
from rules.beaming import beaming as BM
from rules.beaming import cases as CS
from rules.beaming.record import record

AZ = E.AZIMUTH
_AVG: dict = {}


def q_poly(x):  # asymmetric eta density of degree 2 on [-1, 1], positive, unit mass
    return 0.5 * (1.0 + 0.4 * x + 0.3 * 0.5 * (3 * x**2 - 1))


def q_parabola(x):  # even, degree 2
    return 0.75 * (1.0 - x**2)


def q_box(x):  # U[-1/2, 1/2], jumps at +-1/2
    return np.where(np.abs(x) <= 0.5, 1.0, 0.0)


P_EXP = CS.exp_pitch(2.0)


def diag_direct(avg_plain, dens_p, dens_q, breaks, n=48):
    x, w = BM.gl_panels(breaks, n)
    m = C.Measure({"eta": x}, w * 2.0 * dens_p(x) * dens_q(x))
    return C.direct(avg_plain, m, azimuth=AZ)


def legendre_cells_predict(avg, n_cells, N, L, dens_p, dens_q, q_breaks):
    tab = C.ProductTable.tensor({"mu": C.Cells.linear(-1.0, 1.0, n_cells, order=N)})
    cols = C.build_columns(avg, tab)
    mu = C.panel_measure("mu", dens_p, np.linspace(-1, 1, n_cells + 1), n=24, chart="affine")
    eta = C.panel_measure("eta", dens_q, q_breaks, n=24, chart="affine")
    w = np.ones(tab.n_cells)
    mom = C.pair(tab, C.product(mu, eta), w, legendre={"eta": L})
    return C.predict(cols, tab, mom, w, azimuth=AZ)


def check_factor(out):
    """When the factor-2 construction is accurate, the factor 1 misses half of the emission."""
    e2, e1 = out["factor2"][0], out["factor1"][0]
    if e2 < 1e-2:
        assert abs(e1 - 0.5) < 2 * e2 + 1e-9


# ------------------------------------------------------------------ polynomial kernel: exact
POLY = C.PolynomialKernel.random({"eta": ("affine", 0.0, 1.0)}, degree=3, n_ch=4, seed=3)


@pytest.mark.parametrize("L", [2, 4, 16])
@pytest.mark.parametrize("n_cells", [1, 2, 4])
def test_polynomial_kernel_eta_legendre_mu_cells_exact(n_cells, L):
    """K cubic in eta, q of degree 2: exact when N >= deg q + 3. The factor 1 gives exactly half."""
    N = 5
    plain = C.AveragedKernel(POLY, [])
    ref = diag_direct(plain, P_EXP, q_poly, [-1, 0, 1])
    out = {}
    for factor in (2.0, 1.0):
        avg = C.AveragedKernel(BM.DiagonalLegendreKernel(POLY, L, factor), [])
        pred = legendre_cells_predict(avg, n_cells, N, L, P_EXP, q_poly, [-1, 0, 1])
        out[f"factor{factor:g}"] = C.representation_error(pred, ref)
        out[f"factor{factor:g}_I_ratio"] = float(np.max(np.abs(pred[:, 0] / ref[:, 0])))
    record(f"c_poly_eta_legendre|cells{n_cells}|L{L}", out)
    assert out["factor2"] < 1e-11
    assert abs(out["factor1_I_ratio"] - 0.5) < 1e-11


# ------------------------------------------------------------------ continuum kernel
def continuum_avgs(L=None, factor=2.0, density=None):
    key = (L, factor, density)
    if key not in _AVG:
        base = CS.continuum_base()
        kern = (BM.DiagonalLegendreKernel(base, L, factor) if density is None
                else BM.DiagonalDensityKernel(base, density, factor))
        _AVG[key] = C.AveragedKernel(kern, [CS.B_RULE], fixed={"gamma": CS.GAMMA_FIXED})
    return _AVG[key]


@pytest.fixture(scope="module")
def plain():
    return C.AveragedKernel(CS.continuum_base(), [CS.B_RULE], fixed={"gamma": CS.GAMMA_FIXED})


@pytest.mark.parametrize("N", [2, 4])
@pytest.mark.parametrize("n_cells", [1, 2, 4, 8])
def test_continuum_eta_legendre_mu_cells(plain, n_cells, N):
    """q even of degree 2 on Legendre(2): only the mu-cell Taylor truncation remains."""
    ref = diag_direct(plain, P_EXP, q_parabola, [-1, -0.5, 0, 0.5, 1])
    out = {}
    for factor in (2.0, 1.0):
        pred = legendre_cells_predict(continuum_avgs(2, factor), n_cells, N, 2, P_EXP, q_parabola, [-1, 0, 1])
        out[f"factor{factor:g}"] = C.representation_error(pred, ref)
    out = {k: [v, BM.verdict(v)] for k, v in out.items()}
    record(f"c_continuum_eta_legendre|cells{n_cells}|N{N}", out)
    check_factor(out)
    if n_cells >= 4:
        assert out["factor2"][0] < 3e-2


@pytest.mark.parametrize("L", [2, 4, 8, 16])
def test_continuum_eta_legendre_box_law_convergence(plain, L):
    """q = U[-1/2, 1/2] on Legendre(L): the Legendre truncation of a jump law, 8 mu cells, N = 4."""
    ref = diag_direct(plain, P_EXP, q_box, [-0.5, -0.25, 0, 0.25, 0.5])
    pred = legendre_cells_predict(continuum_avgs(L, 2.0), 8, 4, L, P_EXP, q_box, [-0.5, 0, 0.5])
    err = C.representation_error(pred, ref)
    record(f"c_continuum_eta_legendre_box|L{L}", [err, BM.verdict(err)])
    assert np.isfinite(err)


DENSITY_CASES = {  # name -> (q numpy, q jnp, support breaks of q, mu cell edges)
    "U(-0.5,0.5)_1cell": (q_box, lambda m: jnp.where(jnp.abs(m) <= 0.5, 1.0, 0.0), [-0.5, 0.5], [-0.5, 0.5]),
    "U(-0.5,0.5)_2cells": (q_box, lambda m: jnp.where(jnp.abs(m) <= 0.5, 1.0, 0.0), [-0.5, 0, 0.5], [-0.5, 0, 0.5]),
    "U(-0.5,0.5)_4cells": (q_box, lambda m: jnp.where(jnp.abs(m) <= 0.5, 1.0, 0.0), [-0.5, 0, 0.5],
                           [-0.5, -0.25, 0, 0.25, 0.5]),
    "U(-0.5,0.5)_jump_inside_2cells": (q_box, lambda m: jnp.where(jnp.abs(m) <= 0.5, 1.0, 0.0), [-0.5, 0, 0.5],
                                       [-1.0, 0.0, 1.0]),
    "U(-0.02,0.02)_1cell": (lambda x: np.where(np.abs(x) <= 0.02, 25.0, 0.0),
                            lambda m: jnp.where(jnp.abs(m) <= 0.02, 25.0, 0.0), [-0.02, 0.02], [-0.02, 0.02]),
    "|eta|>0.95_2cells": (lambda x: np.where(np.abs(x) >= 0.95, 10.0, 0.0),
                          lambda m: jnp.where(jnp.abs(m) >= 0.95, 10.0, 0.0), [-1, -0.95, 0.95, 1],
                          [-1.0, -0.95, 0.95, 1.0]),
    "parabola_4cells": (q_parabola, lambda m: 0.75 * (1 - m**2), [-1, 0, 1], [-1, -0.5, 0, 0.5, 1]),
}


@pytest.mark.parametrize("N", [2, 4])
@pytest.mark.parametrize("case", list(DENSITY_CASES))
def test_continuum_eta_density_mu_cells(plain, case, N):
    """g(mu) = 2 q(mu) K(eta = mu) on mu cells, against the beaming integral. Middle cells outside the support are kept."""
    qn, qj, qbreaks, edges = DENSITY_CASES[case]
    ref = diag_direct(plain, P_EXP, qn, qbreaks)
    out = {}
    for factor in (2.0, 1.0):
        avg = continuum_avgs(None, factor, qj)
        cells = C.Cells.from_edges(edges, chart="affine", order=N)
        tab = C.ProductTable.tensor({"mu": cells})
        cols = C.build_columns(avg, tab)
        breaks = np.unique(np.r_[edges, [b for b in qbreaks if edges[0] < b < edges[-1]]])
        m = C.panel_measure("mu", P_EXP, breaks, n=24, chart="affine")
        w = np.ones(tab.n_cells)
        pred = C.predict(cols, tab, C.pair(tab, m, w), w, azimuth=AZ)
        out[f"factor{factor:g}"] = C.representation_error(pred, ref)
    out = {k: [v, BM.verdict(v)] for k, v in out.items()}
    record(f"c_continuum_eta_density|{case}|N{N}", out)
    if "jump_inside" in case:
        assert out["factor2"][0] > 1e-1   # a jump of q inside a mu cell: open question 17 (a)
    elif "0.95" in case:
        assert out["factor2"][0] < 0.6   # cells touching |mu| = 1: slow, recorded
    else:
        check_factor(out)
        assert out["factor2"][0] < 3e-2


@pytest.mark.parametrize("L,n_cells,N", [(8, 8, 4), (8, 32, 4), (16, 16, 4), (16, 32, 4), (16, 64, 4)])
def test_continuum_eta_legendre_box_law_mu_resolution(plain, L, n_cells, N):
    """The mu cells must resolve (2k+1) P_k(mu) K(mu) up to k = L: error against the cell count at N = 4.

    Orders above 4 are not run: the N = 8 case (8 nested jacfwd levels) was stopped unfinished.
    """
    ref = diag_direct(plain, P_EXP, q_box, [-0.5, -0.25, 0, 0.25, 0.5])
    pred = legendre_cells_predict(continuum_avgs(L, 2.0), n_cells, N, L, P_EXP, q_box, [-0.5, 0, 0.5])
    err = C.representation_error(pred, ref)
    record(f"c_continuum_eta_legendre_box_resolution|L{L}|cells{n_cells}|N{N}", [err, BM.verdict(err)])
    assert np.isfinite(err)
