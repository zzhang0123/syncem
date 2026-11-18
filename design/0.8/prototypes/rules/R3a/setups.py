"""Test configurations for R3a: parent populations, lattices, a cheap analytic kernel, and runners.

The parent lattice is the paper's 19 equal ln-gamma cells on [300, 2.9626e6]. Each case
uses a subset of parent cells (first, middle, last) to keep direct integration cheap.
"""
from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

import core as C
from core import paper as E

from . import node as R

LN10 = np.log(10.0)


# ------------------------------------------------------------------ parent populations
def parent_density(name: str):
    """Unnormalised dN/dgamma. 'edge' concentrates the mass at the upper edge of cell 9."""
    edges = C.Cells.log(E.GAMMA_LO, E.GAMMA_HI, 19).edges
    if name.startswith("pl"):
        p = float(name[2:])
        return lambda g: g ** (-p)
    if name == "bpl":  # the E2 shape: 2.0 below 7800, 3.2 above, smoothness 2
        return lambda g: g ** -2.0 * (1.0 + (g / 7800.0) ** 2.0) ** (-(3.2 - 2.0) / 2.0)
    if name == "edge":
        ge = edges[10]
        return lambda g: np.exp(-0.5 * (np.log(g / ge) / 0.01) ** 2) / g + 1e-30 * g ** -2.6
    raise ValueError(name)


def parent_measure(table: C.ProductTable, name: str, n=24) -> C.Measure:
    br = np.unique(np.concatenate([table.lo[:, 0], table.hi[:, 0]]))
    if name == "edge":
        e = C.Cells.log(E.GAMMA_LO, E.GAMMA_HI, 19).edges[10]
        br = np.unique(np.concatenate([br, e * np.exp(0.01 * np.arange(-8, 8.5, 0.5))]))
        br = br[(br >= table.lo[:, 0].min()) & (br <= table.hi[:, 0].max())]
    return C.panel_measure("gamma", parent_density(name), br, n=n, chart="log")


def gamma_cells(N: int, cells=(0, 9, 18)) -> tuple[C.Cells, np.ndarray]:
    """E1 cells at order N and the selected cell numbers."""
    return C.Cells.log(E.GAMMA_LO, E.GAMMA_HI, 19, order=N), np.asarray(cells)


def child_cells(chart: str, NB: int):
    if chart == "affine":
        return C.Cells.linear(2.0, 8.0, 3, order=NB)
    if chart == "log":
        return C.Cells.log(2.0, 8.0, 3, order=NB)
    if chart == "log_full":  # one unbounded log cell: no child edge
        return C.Cells.single(5.0, 0.3, order=NB, chart="log")
    raise ValueError(chart)


@dataclass
class Case:
    parent: C.ProductTable
    joint: C.ProductTable
    child: C.Cells
    child_chart: str
    w_parent: np.ndarray
    w_joint: np.ndarray


def make_case(N: int, NB: int, chart: str, total=None, cells=(0, 9, 18)) -> Case:
    g, sel = gamma_cells(N, cells)
    ch = child_cells(chart, NB)
    total = N if total is None else total
    full_p = C.ProductTable.tensor({"gamma": g}, total=N)
    full_j = C.ProductTable.tensor({"gamma": g, "B": ch}, total=total)
    parent = full_p.select(np.isin(full_p.index[:, 0], sel))
    joint = full_j.select(np.isin(full_j.index[:, 0], sel))
    # weights of the full 19-cell lattice: gamma^-2.6 masses, factor 1 on B
    wp = C.reference_weights(full_p)[np.isin(full_p.index[:, 0], sel)]
    wj = C.reference_weights(full_j)[np.isin(full_j.index[:, 0], sel)]
    cchart = "log" if chart.startswith("log") else "affine"
    return Case(parent, joint, ch, cchart, wp, wj)


# ------------------------------------------------------------------ moment comparison
def node_vs_nested(cond, case: Case, pop: str, method="exact", n_pop=24, deriv="analytic", **kw):
    """Returns dict with a_node, a_true, a_parent, the error per unit parent mass per row, the
    row exponents and the sup bound."""
    S = max(e[-1] for c in range(case.joint.n_cells) for e in case.joint.exponents(c))
    kmax = max(max(o) for o in case.parent.order)
    h, ks = R.taylor_h(cond, case.parent, ("gamma",), case.child_chart, case.child, S, kmax,
                       method=method, deriv=deriv, **kw)
    L = R.node_matrix(case.parent, case.joint, ("gamma",), h, ks, case.w_parent, case.w_joint)
    m = parent_measure(case.parent, pop, n=n_pop)
    ap = C.pair(case.parent, m, case.w_parent)["M0"]
    a_node = L @ ap
    a_true = R.joint_moments_nested(cond, m, case.parent, case.joint, case.child_chart, case.child,
                                    case.w_joint)
    # per-unit-mass scale: (w_b / w_{b,c}) a_parent[b, 0]
    lookup = {tuple(case.parent.index[b]): b for b in range(case.parent.n_cells)}
    scale = np.zeros(case.joint.n_rows)
    for cj in range(case.joint.n_cells):
        b = lookup[tuple(case.joint.index[cj, :-1])]
        sl = slice(case.joint.offsets[cj], case.joint.offsets[cj + 1])
        scale[sl] = case.w_parent[b] / case.w_joint[cj] * ap[case.parent.offsets[b]]
    err = np.abs(a_node - a_true) / scale
    return dict(a_node=a_node, a_true=a_true, a_parent=ap, err=err, scale=scale, h=h, ks=ks, L=L,
                rows=case.joint.row_exp, measure=m)


def verdict(x: float) -> str:
    return "OK" if x < 1e-3 else ("WARN" if x < 1e-1 else "BAD")


# ------------------------------------------------------------------ a cheap analytic kernel
class ToySynchrotron:
    """Smooth kernel in (gamma, B [uG]) per channel: I = B F(x), x = nu / (4.2 B gamma^2),
    F(x) = x^(1/3) / (1 + x)^2 (power-law tails, so no channel underflows in any E1 cell),
    P = 0.7 I exp(2i lam^2 8). Channels 1e6 to 1e14 Hz. Cheap enough for high orders."""

    def __init__(self, nu=None):
        self.nu = np.geomspace(1e6, 1e14, 17) if nu is None else np.asarray(nu)
        self.lam2 = (2.998e8 / self.nu) ** 2
        self.n_ch = self.nu.size

    def __call__(self, p):
        x = jnp.asarray(self.nu) / (4.2 * p["B"] * p["gamma"] ** 2)
        I = p["B"] * x ** (1.0 / 3.0) / (1.0 + x) ** 2
        return I, 0.7 * I * jnp.exp(2j * jnp.asarray(self.lam2) * 8.0)
