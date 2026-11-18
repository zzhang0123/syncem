"""Closed-form depth columns (Spec Sec. 4.3 bullet 2, Sec. 8.1) and predictions on (gamma, depth) lattices.

Depth cells are affine, u_d = (d - d_c) / h_c. The phase law is exp(i tau d) with
tau = 2 lambda^2, so its Taylor weights in u_d are

    W[c, s](nu) = exp(i tau d_c) (i tau h_c)^s / s!.

For a lattice row (gamma cell b, depth cell c, exponents (r, s)) the columns are

    P[row]_j = sum_t w_jt Kp[b, r](nu_jt) W[c, s](nu_jt)
    I[row]_j = sum_t w_jt Ki[b, r](nu_jt) if s == 0, else 0,

with Kp, Ki the energy columns at the quadrature nodes. The I column does not depend on c,
so one stored spin-0 spectrum per (b, r) serves every depth cell (``spin0_from``).

Two prediction paths are provided. :func:`assemble` writes explicit columns in the row order
of a core ProductTable, for use with core.pair and core.predict. :func:`predict_zones` sums
the same terms for populations that are a sum of zones, each a product of a gamma law and a
depth law, without forming rows. It handles thousands of depth cells.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

import core as C


def phase_weights(tau_jt: np.ndarray, ref: np.ndarray, hw: np.ndarray, s_max: int) -> np.ndarray:
    """W[c, s, j, t] = exp(i tau d_c) (i tau h_c)^s / s!, shape (n_c, s_max + 1, J, T)."""
    base = np.exp(1j * tau_jt[None] * ref[:, None, None])
    step = 1j * tau_jt[None] * hw[:, None, None]
    out = [base]
    for s in range(1, s_max + 1):
        out.append(out[-1] * step / s)
    return np.stack(out, axis=1)


def gamma_row(gamma_table: C.ProductTable, b: int, r: int) -> int:
    """Row of (gamma cell b, power r) in a one-variable gamma table."""
    return int(gamma_table.offsets[b]) + list(gamma_table.exponents(b)).index((r,))


def assemble(ecols: dict, gamma_table: C.ProductTable, table: C.ProductTable, ch: C.Channels,
             tau_jt: np.ndarray) -> dict:
    """Explicit closed-form columns in the rows of ``table`` over ("gamma", "depth").

    Returns {"I", "P", "spin0_from"}: "spin0_from"[row] is the gamma row whose stored spin-0
    spectrum the row repeats, or -1 where the I column is zero (s > 0).
    """
    if table.variables != ("gamma", "depth"):
        raise ValueError("table must be over ('gamma', 'depth')")
    s_max = int(table.row_exp[:, 1].max())
    W = phase_weights(tau_jt, table.reference[:, 1], table.half_width[:, 1], s_max)
    Ichan = np.einsum("jt,rjt->rj", ch.w, ecols["I"])
    rows_I, rows_P, spin0 = [], [], []
    for row, (c, (r, s)) in enumerate(zip(table.row_cell, table.row_exp)):
        g = gamma_row(gamma_table, table.index[c, 0], r)
        rows_P.append(np.einsum("jt,jt,jt->j", ch.w, ecols["P"][g], W[c, s]))
        rows_I.append(Ichan[g] if s == 0 else np.zeros(Ichan.shape[1]))
        spin0.append(g if s == 0 else -1)
    return {"I": np.stack(rows_I), "P": np.stack(rows_P), "spin0_from": np.array(spin0)}


# ------------------------------------------------------------------ zone populations
@dataclass(frozen=True)
class Zone:
    """One zone: gamma measure (normalised to its column) times a depth law.

    ``depth_nodes``/``depth_weights``: quadrature of the depth law (probabilities summing to
    the depth mass inside the lattice). ``char``: callable tau -> exact characteristic
    function E[exp(i tau d)] of the depth law (used by the direct reference).
    """

    gamma: C.Measure
    depth_nodes: np.ndarray
    depth_weights: np.ndarray
    char: object


def zone_measure(z: Zone) -> C.Measure:
    """Joint (gamma, depth) measure of one zone, for core.pair."""
    return C.product(z.gamma, C.Measure({"depth": z.depth_nodes}, z.depth_weights))


def joint_measure(zones) -> C.Measure:
    """Sum of zone measures as one node list."""
    ms = [zone_measure(z) for z in zones]
    return C.Measure({v: np.concatenate([m.nodes[v] for m in ms]) for v in ("gamma", "depth")},
                     np.concatenate([m.weights for m in ms]))


def depth_moments(cells: C.Cells, nodes: np.ndarray, weights: np.ndarray, s_max: int) -> np.ndarray:
    """A[c, s] = sum over nodes in depth cell c of weight u^s (cell weight 1), (n_c, s_max + 1)."""
    lo, hi = cells.lo, cells.hi
    out = np.zeros((cells.n, s_max + 1))
    idx = np.searchsorted(hi, nodes, side="left")  # first cell with hi >= x, closed intervals
    ok = (idx < cells.n)
    ok[ok] &= nodes[ok] >= lo[idx[ok]]
    u = (nodes[ok] - cells.reference[idx[ok]]) / cells.half_width[idx[ok]]
    for s in range(s_max + 1):
        np.add.at(out[:, s], idx[ok], weights[ok] * u**s)
    return out


def predict_zones(ecols: dict, gamma_table: C.ProductTable, w_gamma: np.ndarray, depth_cells: C.Cells,
                  zones, ch: C.Channels, tau_jt: np.ndarray, lower: tuple, azimuth=1.0,
                  chunk=256) -> np.ndarray:
    """(J, 3) prediction sum over (r, s) in ``lower`` of Kp_r(nu) W_s(nu) a_(r,s), zone by zone.

    ``lower`` is the lower set of (r, s) exponents, the same in every product cell.
    Equal to core.predict on the explicit columns of :func:`assemble` with core.pair moments.
    """
    s_max = max(s for _, s in lower)
    r_max = max(r for r, _ in lower)
    J, T = tau_jt.shape
    I = np.zeros(J)
    P = np.zeros((J, T), complex)
    for z in zones:
        a_g = C.pair(gamma_table, z.gamma, w_gamma)["M0"]  # (n_rows_g,), scaled by 1/w_b
        G_I = np.zeros((r_max + 1, J, T))
        G_P = np.zeros((r_max + 1, J, T), complex)
        for b in range(gamma_table.n_cells):
            for r in range(r_max + 1):
                if (r,) in gamma_table.exponents(b):
                    g = gamma_row(gamma_table, b, r)
                    G_I[r] += w_gamma[b] * a_g[g] * ecols["I"][g]
                    G_P[r] += w_gamma[b] * a_g[g] * ecols["P"][g]
        A_d = depth_moments(depth_cells, z.depth_nodes, z.depth_weights, s_max)  # (n_c, s+1)
        D = np.zeros((s_max + 1, J, T), complex)
        for k in range(0, depth_cells.n, chunk):
            sl = slice(k, k + chunk)
            W = phase_weights(tau_jt, depth_cells.reference[sl], depth_cells.half_width[sl], s_max)
            D += np.einsum("cs,csjt->sjt", A_d[sl], W)
        mass0 = A_d[:, 0].sum()
        for r, s in lower:
            P += G_P[r] * D[s]
            if s == 0:
                I += np.einsum("jt,jt->j", ch.w, G_I[r]) * mass0
    P = azimuth * np.einsum("jt,jt->j", ch.w, P)
    return np.stack([I, P.real, P.imag], axis=-1)


def predict_screen(ecols: dict, gamma_table: C.ProductTable, w_gamma: np.ndarray, gamma_measure: C.Measure,
                   ch: C.Channels, char_jt: np.ndarray, azimuth=1.0) -> np.ndarray:
    """Characteristic-function route: depth averaged, P columns carry C_scr(tau) inside the channel sum."""
    a = C.pair(gamma_table, gamma_measure, w_gamma)["M0"]
    wr = w_gamma[gamma_table.row_cell] * a
    I = np.einsum("r,jt,rjt->j", wr, ch.w, ecols["I"])
    P = azimuth * np.einsum("r,jt,rjt,jt->j", wr, ch.w, ecols["P"], char_jt)
    return np.stack([I, P.real, P.imag], axis=-1)


def direct_zones(values: tuple, zones, ch: C.Channels, tau_jt: np.ndarray, azimuth=1.0) -> np.ndarray:
    """Direct integration with the exact Faraday phase. ``values[z]`` = (I, P) at zone z's gamma nodes.

    P_j = c sum_t w_jt sum_z [sum_i w_i Kp(gamma_i)](nu_jt) E_z[exp(i tau d)](nu_jt). No expansion
    in gamma or depth. The depth average is the exact characteristic function of each zone's law.
    """
    J = tau_jt.shape[0]
    I, P = np.zeros(J), np.zeros(J, complex)
    for (vI, vP), z in zip(values, zones):
        kI = np.einsum("i,ijt->jt", z.gamma.weights, vI)
        kP = np.einsum("i,ijt->jt", z.gamma.weights, vP)
        I += np.einsum("jt,jt->j", ch.w, kI) * float(np.sum(z.depth_weights))
        P += np.einsum("jt,jt,jt->j", ch.w, kP, z.char(tau_jt))
    P = azimuth * P
    return np.stack([I, P.real, P.imag], axis=-1)


# ------------------------------------------------------------------ depth laws
def uniform_law(a: float, b: float, cells: C.Cells, n=8, mass=1.0):
    """U[a, b] scaled to ``mass``: GL nodes on panels split at the cell edges, and its characteristic function."""
    br = np.unique(np.r_[a, b, cells.edges[(cells.edges > a) & (cells.edges < b)]])
    m = C.panel_measure("depth", lambda x: mass / (b - a) + 0 * x, br, n=n, chart="affine")

    def char(t):  # mass exp(i t mid) sin(t hw) / (t hw), without the cancellation of the difference form
        mid, hw = 0.5 * (a + b), 0.5 * (b - a)
        return mass * np.exp(1j * t * mid) * np.sinc(t * hw / np.pi)

    return m.nodes["depth"], m.weights, char


def point_law(d: float, mass=1.0):
    return np.array([d]), np.array([mass]), (lambda t: mass * np.exp(1j * t * d))


def gaussian_law(mean: float, sd: float, cells: C.Cells, n=8):
    """N(mean, sd^2) on the cells' range with panels at the cell edges, and the untruncated characteristic function."""
    e = cells.edges
    m = C.panel_measure("depth", lambda x: np.exp(-0.5 * ((x - mean) / sd) ** 2) / (sd * math.sqrt(2 * math.pi)),
                        e, n=n, chart="affine")

    def char(t):
        return np.exp(1j * t * mean - 0.5 * t**2 * sd**2)

    return m.nodes["depth"], m.weights, char


def power_law_gamma(slope: float, lo: float, hi: float, edges: np.ndarray, n=8, column=1.0) -> C.Measure:
    """gamma^-slope on [lo, hi] with unit column times ``column``, panels at the cell edges."""
    k = 1.0 - slope
    norm = column * k / (hi**k - lo**k)
    br = np.unique(np.r_[lo, hi, edges[(edges > lo) & (edges < hi)]])
    return C.panel_measure("gamma", lambda g: norm * g ** (-slope), br, n=n, chart="log")
