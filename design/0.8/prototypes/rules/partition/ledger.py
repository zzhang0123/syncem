"""The measured terms representation and outside_lattice of Spec Sec. 7.2 on the prototype core.

Definitions tested (Spec Sec. 7.2, r3-25):
  representation  = predict - direct(within cells)
  outside_lattice = direct(full support) - direct(within cells)
and the identity predict - direct(full) = representation + outside_lattice.

Each piece is computed by its own quadrature so that the identity is not true by
construction:
  * D_cells   kernel calls on Gauss-Legendre panels that break at every cell face,
  * D_table   the same nodes with Kbar from the 17-node Chebyshev cell table (the
              reference that Spec Sec. 7.2 names for the representation term),
  * D_out     kernel calls on panels over the uncovered support only, plus the
              small-x tail above GAMMA_TOP,
  * D_full    kernel calls on panels of fixed width in ln gamma that ignore the cell faces,
              plus the same tail.
The tail above GAMMA_TOP uses F, G ~ x^(1/3), so Kbar(gamma) = Kbar(g_top) (gamma/g_top)^(-2/3)
for every channel node and every field and inclination node. The continuum kernel raises
below x_min = 1e-9, so an unbounded support cannot be integrated by kernel calls alone.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from scipy import integrate

import core as C
from core import paper as E

from . import nodes as Nd
from . import setup as S

SLOPE = 2.6
AMP = (1.0 - SLOPE) / (E.GAMMA_HI ** (1.0 - SLOPE) - E.GAMMA_LO ** (1.0 - SLOPE))  # unit column on E1 range


@dataclass(frozen=True)
class Population:
    """dN/dgamma on the support [lo, hi] (hi may be inf)."""

    name: str
    density: Callable[[np.ndarray], np.ndarray]
    lo: float
    hi: float


def power_law(lo=S.GAMMA_FLOOR, hi=np.inf) -> Population:
    return Population(f"PL2.6[{lo:g},{hi:g}]", lambda g: AMP * g ** (-SLOPE), lo, hi)


def cutoff(gc=1e6, lo=100.0) -> Population:
    return Population(f"PL2.6*exp(-g/{gc:g})[{lo:g},inf]", lambda g: AMP * g ** (-SLOPE) * np.exp(-g / gc), lo, np.inf)


# ------------------------------------------------------------------ quadratures
def _weights(pop: Population, g, w):
    inside = (g >= pop.lo) & (g <= pop.hi)
    return np.where(inside, w * pop.density(np.where(inside, g, 1.0)), 0.0)


def tail(pop: Population, g_top=S.GAMMA_TOP) -> tuple[np.ndarray, np.ndarray]:
    """Emission of the support above g_top from the small-x asymptote, (I, P) of shape (68,)."""
    if pop.hi <= g_top:
        return np.zeros(68), np.zeros(68, complex)
    t_hi = np.log(pop.hi / g_top) if np.isfinite(pop.hi) else 200.0  # in t = ln(gamma / g_top)
    s = integrate.quad(lambda t: np.exp(t / 3.0) * g_top * float(pop.density(np.array(g_top * np.exp(t)))),
                       0.0, t_hi, epsabs=0, epsrel=1e-12, limit=400)[0]
    I, P = Nd.kbar(np.array([g_top]), label="tail")
    return I[0] * s, P[0] * s


def stokes(I, P, azimuth=E.AZIMUTH) -> np.ndarray:
    """(68, 3) [I, Q, U] with the constant azimuth factor on P."""
    p = azimuth * P
    return np.stack([np.real(I), p.real, p.imag], axis=-1)


E1_EDGES = np.exp(np.linspace(np.log(E.GAMMA_LO), np.log(E.GAMMA_HI), 20))


def aligned_breaks(table: C.ProductTable, pop: Population, extra=()) -> np.ndarray:
    """Panel breaks at every cell face, the support ends and fixed outer breaks (E1 faces always)."""
    b = [S.GAMMA_FLOOR, 30.0, 100.0, 300.0, 1e7, S.GAMMA_TOP, *E1_EDGES, *extra, *table.lo[:, 0], *table.hi[:, 0]]
    if np.isfinite(pop.hi):
        b.append(pop.hi)
    b.append(pop.lo)
    b = np.unique(np.asarray(b, float))
    return b[(b >= S.GAMMA_FLOOR) & (b <= S.GAMMA_TOP)]


@dataclass(frozen=True)
class Pieces:
    """(68, 3) Stokes arrays of every piece, before the azimuth factor is split off."""

    pred: np.ndarray
    cells: np.ndarray
    table: np.ndarray
    out: np.ndarray
    full: np.ndarray
    naive_cells: np.ndarray
    tail: np.ndarray


def full_quadrature(pop: Population, width=0.45, n=12):
    """Panels of fixed ln-width from 100 to GAMMA_TOP, two panels below 100, breaks at the
    support ends. Cell faces are ignored. Returns g, w, I, P."""
    lo = np.log(100.0)
    k = int(np.ceil((np.log(S.GAMMA_TOP) - lo) / width))
    br = np.r_[S.GAMMA_FLOOR, 30.0, np.exp(lo + width * np.arange(k)), S.GAMMA_TOP, pop.lo,
               pop.hi if np.isfinite(pop.hi) else S.GAMMA_TOP]
    br = br[(br >= S.GAMMA_FLOOR) & (br <= S.GAMMA_TOP)]
    return Nd.panel_quad(br, n)


def pieces(table: C.ProductTable, cols: dict, pop: Population, *, n=12, table_values=None,
           use_table=True) -> Pieces:
    """Every piece for one lattice (gamma cells, log chart), its columns and one population."""
    g, w0, I, P = Nd.panel_quad(aligned_breaks(table, pop), n)
    w = _weights(pop, g, w0)
    cell = table.locate({"gamma": g})
    inn = cell >= 0
    tI, tP = tail(pop)
    D_cells = stokes(w[inn] @ I[inn], w[inn] @ P[inn])
    D_out = stokes(w[~inn] @ I[~inn] + tI, w[~inn] @ P[~inn] + tP)
    # representation reference from the 17-node table on the same nodes
    if use_table:
        if table_values is None:
            table_values = Nd.table_values(table.reference[:, 0], table.half_width[:, 0])
        u = np.asarray(C.to_u("log", g[inn], table.reference[cell[inn], 0], table.half_width[cell[inn], 0]))
        TI = Nd.table_interpolate(table_values[0], cell[inn], u)
        TP = Nd.table_interpolate(table_values[1], cell[inn], u)
        D_table = stokes(w[inn] @ TI, w[inn] @ TP)
    else:
        D_table = np.full_like(D_cells, np.nan)
    # moments on the aligned nodes, then predict with the jet columns (all 68 channels)
    wc = C.reference_weights(table)
    m = C.Measure({"gamma": g[inn]}, w[inn])
    a = C.pair(table, m, wc)
    pred = C.predict(cols, table, a, wc, azimuth=E.AZIMUTH)
    # full support on panels that ignore the faces
    gf, wf0, If, Pf = full_quadrature(pop, n=n)
    wf = _weights(pop, gf, wf0)
    D_full = stokes(wf @ If + tI, wf @ Pf + tP)
    # a node-mask implementation of within="cells" on the face-blind panels
    mf = table.locate({"gamma": gf}) >= 0
    D_naive = stokes(wf[mf] @ If[mf], wf[mf] @ Pf[mf])
    return Pieces(pred, D_cells, D_table, D_out, D_full, D_naive, stokes(tI, tP))


def summary(p: Pieces, ch) -> dict:
    """Identity residuals and the scalar terms, each as max over channels and I, Q, U of |dX| / I_full."""
    sl = lambda a: a[ch]
    Ifull = sl(p.full)[:, :1]
    Icells = sl(p.cells)[:, :1]
    m = lambda a, den=Ifull: float(np.max(np.abs(sl(a)) / den))
    rep = p.pred - p.cells
    rep_t = p.pred - p.table
    total = p.pred - p.full
    return {
        "representation": m(rep),
        "representation_over_I_cells": m(rep, Icells),
        "representation_table_reference": m(rep_t),
        "outside_lattice": m(p.out),
        "outside_share_I_max": float(np.max(sl(p.out)[:, 0] / Ifull[:, 0])),
        "total_predict_minus_full": m(total),
        "envelope_over_total": (m(rep) + m(p.out)) / m(total) if m(total) > 0 else float("nan"),
        # Spec Sec. 7.2 as written: outside_lattice = D_full - D_cells and total = rep + outside
        "identity_residual_spec_sign": m(total - (rep + p.out)),
        # the identity that holds: total = rep - outside_lattice (outside_lattice = D_full - D_cells)
        "identity_residual_kernel_reference": m(total - (rep - p.out)),
        "identity_residual_table_reference": m(total - (rep_t - p.out)),
        "table_vs_kernel_direct_in_cells": m(p.table - p.cells),
        "naive_mask_within_cells_error": m(p.naive_cells - p.cells),
        "tail_above_gamma_top": m(p.tail),
    }


def tail_check(pop: Population, ch) -> float:
    """Asymptotic tail from 3e7 plus panels on [1e7, 3e7] against the asymptote from 1e7, over I_full."""
    t3I, t3P = tail(pop, S.GAMMA_TOP)
    t1I, t1P = tail(pop, 1e7)
    g, w0, I, P = Nd.panel_quad([1e7, S.GAMMA_TOP], 12)
    w = _weights(pop, g, w0)
    a = stokes(w @ I + t3I, w @ P + t3P)
    b = stokes(t1I, t1P)
    gf, wf0, If, _ = full_quadrature(pop)
    Ifull = _weights(pop, gf, wf0) @ If
    return float(np.max(np.abs(a - b)[ch] / Ifull[ch, None]))
