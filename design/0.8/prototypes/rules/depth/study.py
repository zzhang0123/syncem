"""Studies (a) to (d) of the depth rule family. Each returns a dict of measured numbers.

(a) Two-zone population, electron index correlated with depth, on gamma cells x depth cells
    with closed-form phase weights, against direct integration with the exact Faraday phase.
(b) Sweep of h and the lowest frequency for the check tau_max h of Spec Sec. 4.6 rule 4.
(c) Independent Gaussian screen: characteristic-function route against resolved depth cells.
(d) A field law with depth as a parent (Spec Sec. 4.3 bullet 3): Cauchy product of the
    phase weights with the depth Taylor coefficients of the phase-free averaged kernel,
    against the full jet and against direct integration.
"""
from __future__ import annotations

import math

import jax
import numpy as np

import core as C
from core import paper as E

from . import cache
from . import check as CK
from . import closed_form as CF
from . import nodes as Nd

AZ = E.AZIMUTH


def gamma_cells(N: int) -> C.Cells:
    return C.Cells.log(E.GAMMA_LO, E.GAMMA_HI, 19, order=N)


def slice_columns(ecols: dict, n_cells: int, N_from: int, N: int) -> dict:
    """Rows r <= N of a one-variable build at order N_from (1-D lower sets nest)."""
    keep = np.array([r <= N for _ in range(n_cells) for r in range(N_from + 1)])
    return {k: v[keep] for k, v in ecols.items()}


def stokes_errors(pred: np.ndarray, ref: np.ndarray) -> dict:
    I = np.abs(ref[:, 0])
    dP = np.abs((pred[:, 1] - ref[:, 1]) + 1j * (pred[:, 2] - ref[:, 2]))
    e = np.max(np.abs(pred - ref) / I[:, None], axis=0)
    return {"I": float(e[0]), "Q": float(e[1]), "U": float(e[2]), "absP_over_I": float(np.max(dP / I)),
            "max_IQU": float(e.max()), "worst_channel": int(np.argmax(dP / I))}


# ------------------------------------------------------------------ (a)
# Two zones: (index, column, depth interval). The lattice depth range is RANGES[name].
ZONES = {
    "rotated": [(2.2, 0.3, (0.93, 1.05)), (3.0, 0.7, (1.12, 1.30))],  # tau d about 26 to 36 rad at 0.08 GHz
    "near_zero": [(2.2, 0.3, (-0.05, 0.01)), (3.0, 0.7, (0.015, 0.06))],  # tau |d| <= 1.7 rad
}
RANGES = {"rotated": (0.9, 1.3), "near_zero": (-0.06, 0.06)}


def two_zones(gedges, dcells, n_g=8, n_d=8, name="rotated"):
    """Electron index correlated with depth: zone A index 2.2, zone B index 3.0, separate depth intervals."""
    return [CF.Zone(CF.power_law_gamma(p, E.GAMMA_LO, E.GAMMA_HI, gedges, n=n_g, column=col),
                    *CF.uniform_law(*d, dcells, n=n_d)) for p, col, d in ZONES[name]]


def part_a(kind="e1", n_depth=(1, 2, 4, 8, 16, 32, 128), lowers=None, ch=None, n_g=8, name="rotated") -> dict:
    ch = ch or C.bump_channels()
    tau, J = Nd.tau(ch), ch.n_ch
    lowers = lowers or {"total2": ((2, 2), 2), "tensor2x2": ((2, 2), 4), "total4": ((4, 4), 4),
                        "gamma2_depth4_total4": ((2, 4), 4)}
    e4 = cache.energy_columns(kind, ch, gamma_cells(4), 4)
    gedges = gamma_cells(4).edges
    out = {"tau_max": Nd.tau_max(ch), "cases": {}}
    lo, hi = RANGES[name]
    zones0 = two_zones(gedges, C.Cells.linear(lo, hi, 4), n_g=n_g, name=name)
    gnodes = zones0[0].gamma.nodes["gamma"]
    assert np.allclose(gnodes, zones0[1].gamma.nodes["gamma"])
    vals = cache.values(kind, ch, gnodes)
    direct = CF.direct_zones([vals, vals], zones0, ch, tau, azimuth=AZ)
    out["P_over_I_max"] = float(np.max(np.abs(direct[:, 1] + 1j * direct[:, 2]) / direct[:, 0]))
    # gamma-only baseline: the same population with depth averaged by its exact characteristic function
    char_pred = {}
    for Ng in (2, 4):
        gt = C.ProductTable.tensor({"gamma": gamma_cells(Ng)}, total=Ng)
        ec = slice_columns(e4, 19, 4, Ng)
        char_pred[Ng] = sum(CF.predict_screen(ec, gt, C.reference_weights(gt), z.gamma, ch, z.char(tau), azimuth=AZ)
                            for z in zones0)
        out["cases"][f"char_gamma{Ng}"] = {"lower": f"gamma{Ng} with exact depth average",
                                           **stokes_errors(char_pred[Ng], direct)}
    for nd in n_depth:
        dcells = C.Cells.linear(lo, hi, nd)
        zones = two_zones(gedges, dcells, n_g=n_g, name=name)
        x = out["tau_max"] * float(dcells.half_width[0])
        for label, (caps, total) in lowers.items():
            Ng = caps[0]
            gt = C.ProductTable.tensor({"gamma": gamma_cells(Ng)}, total=Ng)
            ec = slice_columns(e4, 19, 4, Ng)
            lower = C.lower_set(caps, total)
            pred = CF.predict_zones(ec, gt, C.reference_weights(gt), dcells, zones, ch, tau, lower, azimuth=AZ)
            n_d = CK.effective_depth_order(lower)
            out["cases"][f"nd{nd}_{label}"] = {
                "n_depth_cells": nd, "h": float(dcells.half_width[0]), "x": x, "lower": label,
                "rows": 19 * nd * len(lower), "spin0_spectra": 19 * (Ng + 1),
                "spec_passes": CK.spec_passes(x), "spec_factor_N": CK.spec_factor(x, total),
                "n_d_eff": n_d, "proposed_bound": CK.spec_factor(x, n_d),
                "depth_part_absP_over_I": stokes_errors(pred, char_pred[Ng])["absP_over_I"],
                **stokes_errors(pred, direct)}
    return out


def explicit_vs_factorised(kind="cheap", nd=8) -> dict:
    """core.pair + core.predict on explicit closed-form columns against predict_zones (same model)."""
    ch = C.bump_channels()
    tau, J = Nd.tau(ch), ch.n_ch
    gc = gamma_cells(2)
    gt = C.ProductTable.tensor({"gamma": gc}, total=2)
    ec = cache.energy_columns(kind, ch, gc, 2)
    dcells = C.Cells.linear(0.9, 1.3, nd, order=2)
    tab = C.ProductTable.tensor({"gamma": gc, "depth": dcells}, total=2)
    cols = CF.assemble(ec, gt, tab, ch, tau)
    zones = two_zones(gc.edges, dcells, n_g=6, n_d=6)
    w = C.reference_weights(tab)
    p1 = C.predict(cols, tab, C.pair(tab, CF.joint_measure(zones), w), w, azimuth=AZ)
    p2 = CF.predict_zones(ec, gt, C.reference_weights(gt), dcells, zones, ch, tau, C.lower_set((2, 2), 2), azimuth=AZ)
    s0 = cols["spin0_from"]
    distinct = len(set(s0[s0 >= 0].tolist()))
    return {"rel_diff": float(np.max(np.abs(p1 - p2)) / np.max(np.abs(p1))), "rows": tab.n_rows,
            "I_rows_used": int(np.sum(s0 >= 0)), "stored_spin0_spectra": distinct}


# ------------------------------------------------------------------ (b)
def sweep_channels(nu_c: float) -> C.Channels:
    """Six bump channels from nu_c to 10 nu_c (lowest support 0.8 nu_c), 64 nodes."""
    return C.bump_channels(nu_lo=nu_c, nu_hi=10 * nu_c, n=6)


POPULATIONS = ("edge", "half", "uniform", "uniform3")


def depth_population(pop: str, d0: float, h: float):
    """Depth law relative to a central cell [d0 - h, d0 + h] and cells of half-width h around it."""
    cells = C.Cells.linear(d0 - 3 * h, d0 + 3 * h, 3, order=0)
    if pop == "edge":  # point mass at u = 1 of the central cell (worst case of the remainder)
        return cells, CF.point_law(d0 + h)
    if pop == "half":
        return cells, CF.point_law(d0 + 0.5 * h)
    if pop == "uniform":
        return cells, CF.uniform_law(d0 - h, d0 + h, cells, n=12)
    if pop == "uniform3":  # spans three cells, edges not aligned
        return cells, CF.uniform_law(d0 - 2.7 * h, d0 + 2.2 * h, cells, n=12)
    raise ValueError(pop)


def depth_only_error(Kv: tuple, ch: C.Channels, tau, cells: C.Cells, law, N: int) -> float:
    """|dP| / I for gamma fixed (one kernel value), depth cells with closed-form weights to order N."""
    nodes_, wts, char = law
    kI, kP = Kv  # (J, T)
    A = CF.depth_moments(cells, nodes_, wts, N)
    W = CF.phase_weights(tau, cells.reference, cells.half_width, N)
    Pp = np.einsum("jt,jt,cs,csjt->j", ch.w, kP, A, W)
    Pd = np.einsum("jt,jt,jt->j", ch.w, kP, char(tau))
    I = np.einsum("jt,jt->j", ch.w, kI) * wts.sum()
    return float(np.max(np.abs(Pp - Pd) / I))


def part_b(nu_cs=(1e8, 3e8, 1e9, 1e10), d0s=(0.0, 8.0, 300.0, -300.0), Ns=(0, 1, 2, 3, 4),
           xs=None, kind="cheap", gamma=1e4) -> dict:
    xs = np.geomspace(0.01, 4.0, 28) if xs is None else np.asarray(xs)
    out = {"xs": xs.tolist(), "rows": []}
    for nu_c in nu_cs:
        ch = sweep_channels(nu_c)
        tau = Nd.tau(ch)
        tmax = Nd.tau_max(ch)
        I, P = cache.values(kind, ch, np.array([gamma]))
        Kv = (I[0], P[0])
        for d0 in d0s:
            for pop in POPULATIONS:
                for N in Ns:
                    errs = [depth_only_error(Kv, ch, tau, *depth_population(pop, d0, x / tmax), N) for x in xs]
                    out["rows"].append({"nu_c": nu_c, "nu_min_support": 0.8 * nu_c, "tau_max": tmax, "d0": d0,
                                        "pop": pop, "N": N, "err": errs,
                                        "bound": [CK.spec_factor(x, N) for x in xs]})
    return out


def crossing(xs, errs, level) -> float | None:
    """Smallest x (log-log interpolation) at which errs reaches ``level``."""
    xs, e = np.asarray(xs), np.maximum(np.asarray(errs), 1e-300)
    above = np.flatnonzero(e >= level)
    if above.size == 0:
        return None
    k = int(above[0])
    if k == 0:
        return float(xs[0])
    lx, le = np.log(xs[k - 1:k + 1]), np.log(e[k - 1:k + 1])
    return float(np.exp(lx[0] + (np.log(level) - le[0]) * (lx[1] - lx[0]) / (le[1] - le[0])))


def coupled_worst_case(u_g: float, x: float, lower: tuple, kind="e1", N_g=2) -> float:
    """Depth part of |dP| / I for point masses at energy coordinate u_g in all 19 cells and depth at u_d = 1.

    The reference is the same energy expansion with the exact depth phase (characteristic route),
    so the number isolates the depth truncation, including the cross terms that a total-degree
    lower set drops. Cell masses follow gamma^-2.6. Azimuth factor 1.
    """
    ch = C.bump_channels()
    tau, tmax = Nd.tau(ch), Nd.tau_max(ch)
    gc = gamma_cells(N_g)
    gt = C.ProductTable.tensor({"gamma": gc}, total=N_g)
    ec = slice_columns(cache.energy_columns(kind, ch, gamma_cells(4), 4), 19, 4, N_g)
    wg = C.reference_weights(gt)
    gm = C.Measure({"gamma": gc.reference * np.exp(u_g * gc.half_width)}, wg.copy())
    h = x / tmax
    z = CF.Zone(gm, *CF.point_law(h))
    ref = CF.predict_screen(ec, gt, wg, gm, ch, z.char(tau), azimuth=1.0)
    pred = CF.predict_zones(ec, gt, wg, C.Cells.linear(-h, h, 1), [z], ch, tau, lower, azimuth=1.0)
    return stokes_errors(pred, ref)["absP_over_I"]


# ------------------------------------------------------------------ (c)
def part_c(sds=(0.02, 0.1, 0.5), means=(0.0, 8.0, 300.0), xs=(0.1, 0.3, 1.0, 2.0), kind="e1", N=2,
           lowers=None, k_sigma=6.0, max_cells=6000) -> dict:
    ch = C.bump_channels()
    tau, tmax = Nd.tau(ch), Nd.tau_max(ch)
    lowers = lowers or {"total2": ((2, 2), 2), "tensor2x2": ((2, 2), 4)}
    e = slice_columns(cache.energy_columns(kind, ch, gamma_cells(4), 4), 19, 4, N)
    gc = gamma_cells(N)
    gt = C.ProductTable.tensor({"gamma": gc}, total=N)
    wg = C.reference_weights(gt)
    gm = CF.power_law_gamma(2.6, E.GAMMA_LO, E.GAMMA_HI, gc.edges, n=8)
    vals = cache.values(kind, ch, gm.nodes["gamma"])
    out = {"tau_max": tmax, "rows": []}
    for sd in sds:
        for m in means:
            char = np.exp(1j * tau * m - 0.5 * tau**2 * sd**2)
            p_cf = CF.predict_screen(e, gt, wg, gm, ch, char, azimuth=AZ)
            zone_d = CF.Zone(gm, np.array([m]), np.array([1.0]), lambda t, m=m, sd=sd: np.exp(1j * t * m - 0.5 * t**2 * sd**2))
            p_dir = CF.direct_zones([vals], [zone_d], ch, tau, azimuth=AZ)
            for x in xs:
                h = x / tmax
                nd = int(math.ceil(k_sigma * sd / h))
                if nd > max_cells:
                    continue
                cells = C.Cells.linear(m - nd * h, m + nd * h, nd)
                law = CF.gaussian_law(m, sd, cells, n=8)
                zone = CF.Zone(gm, *law)
                for name, (caps, total) in lowers.items():
                    lower = C.lower_set(caps, total)
                    p_cells = CF.predict_zones(e, gt, wg, cells, [zone], ch, tau, lower, azimuth=AZ)
                    out["rows"].append({
                        "sd": sd, "mean": m, "x": x, "h": h, "n_cells": nd, "lower": name,
                        "proposed_bound": CK.spec_factor(x, CK.effective_depth_order(lower)),
                        "cells_vs_char": stokes_errors(p_cells, p_cf),
                        "cells_vs_direct": stokes_errors(p_cells, p_dir),
                        "char_vs_direct": stokes_errors(p_cf, p_dir),
                        "P_over_I_max": float(np.max(np.abs(p_dir[:, 1] + 1j * p_dir[:, 2]) / p_dir[:, 0]))})
    return out


# ------------------------------------------------------------------ (d)
def depth_parent_field(d_ref=8.0, slope=0.05):
    """B = 5 uG (1 + slope (depth - d_ref)) as a one-node (Delta) conditional on depth."""
    return ("B", lambda p: (5.0 * (1.0 + slope * (p["depth"] - d_ref)).reshape(1), np.ones(1)))


def depth_parent_field_frozen(d_ref=8.0, slope=0.05):
    """The same law with its node frozen (stop_gradient): the frozen-node reading of Spec Sec. 4.3."""
    def rule(p):
        node = jax.lax.stop_gradient(5.0 * (1.0 + slope * (p["depth"] - d_ref)))
        return node.reshape(1), np.ones(1)
    return ("B", rule)


def lognormal_field(width=0.1, d_ref=8.0, slope=0.05, n=12):
    """log B ~ N(log 5 (1 + slope (depth - d_ref)), width^2) by Gauss-Hermite nodes that move with depth."""
    t, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()

    def rule(p):
        med = 5.0 * (1.0 + slope * (p["depth"] - d_ref))
        return med * jax.numpy.exp(width * t), w
    return ("B", rule)


def lognormal_field_reweighted(width=0.1, d_ref=8.0, slope=0.05, n=12, d_cell_ref=8.0):
    """Frozen nodes at the cell reference and weights reweighted by the density ratio (Spec Sec. 8.3 stack mode)."""
    t, w = np.polynomial.hermite_e.hermegauss(n)
    w = w / w.sum()
    med0 = 5.0 * (1.0 + slope * (d_cell_ref - d_ref))
    nodes0 = med0 * np.exp(width * t)

    def rule(p):
        med = 5.0 * (1.0 + slope * (p["depth"] - d_ref))
        lr = lambda m: -0.5 * ((np.log(nodes0) - jax.numpy.log(m)) / width) ** 2
        return nodes0, w * jax.numpy.exp(lr(med) - lr(med0))
    return ("B", rule)


def cauchy_columns(ecols_nodes: dict, table: C.ProductTable, ch: C.Channels, tau_jt) -> dict:
    """Closed-form phase weights times the depth Taylor coefficients of the phase-free kernel.

    ``ecols_nodes``: jet columns of the phase-free averaged kernel on ``table`` at every node,
    (n_rows, J, T). Row (r, s) of cell c gets sum over k <= s of K[(r, s - k)] W[c, k].
    """
    J, T = tau_jt.shape
    s_max = int(table.row_exp[:, 1].max())
    W = CF.phase_weights(tau_jt, table.reference[:, 1], table.half_width[:, 1], s_max)
    P = np.zeros((table.n_rows, J), complex)
    I = np.einsum("jt,rjt->rj", ch.w, ecols_nodes["I"])
    o = table.offsets
    for c in range(table.n_cells):
        ex = list(table.exponents(c))
        for m, (r, s) in enumerate(ex):
            acc = np.zeros((J, T), complex)
            for k in range(s + 1):
                acc += ecols_nodes["P"][o[c] + ex.index((r, s - k))] * W[c, k]
            P[o[c] + m] = np.einsum("jt,jt->j", ch.w, acc)
    return {"I": I, "P": P}


def part_d(gcells=None, dcells=None, total=2, caps=(2, 2), n_ch=6) -> dict:
    ch = C.bump_channels(n=n_ch, nu_lo=1e8, nu_hi=1e10, n_nodes=32)
    tau = Nd.tau(ch)
    gcells = gcells or C.Cells.log(E.GAMMA_LO, E.GAMMA_HI, 19, order=caps[0])
    gcells = C.Cells.from_edges(gcells.edges[[0, 1, -2, -1]], chart="log", order=caps[0])  # first and last cells
    dcells = dcells or C.Cells.from_edges([6.0, 10.0], chart="affine", order=caps[1])
    tab = C.ProductTable.tensor({"gamma": gcells, "depth": dcells}, total=total)
    tab = tab.select([True, False, True]) if tab.n_cells == 3 else tab
    K = C.ContinuumChannelKernel(ch)
    Kn = Nd.PhaseFree(C.ContinuumChannelKernel(Nd.node_channels(ch)))
    eta = ("eta", Nd.eta_rule(4))
    out = {}
    for name, law in (("delta", depth_parent_field()), ("lognormal", lognormal_field())):
        full = C.build_columns(C.AveragedKernel(K, [law, eta]), tab)  # jet differentiates phase and law
        nodes_ = C.build_columns(C.AveragedKernel(Kn, [law, eta]), tab)
        nodes_ = {k: v.reshape(v.shape[0], ch.n_ch, -1) for k, v in nodes_.items()}
        cau = cauchy_columns(nodes_, tab, ch, tau)
        s = tab.row_exp[:, 1]
        r0 = (tab.row_exp[:, 0] == 0)
        i0, i1 = np.abs(full["I"][(s == 0) & r0]), np.abs(full["I"][(s == 1) & r0])
        ok = i0 > 1e-8 * i0.max()
        ratio = i1[ok] / i0[ok]
        out[name] = {"cauchy_vs_jet_P": float(np.max(np.abs(cau["P"] - full["P"])) / np.max(np.abs(full["P"]))),
                     "cauchy_vs_jet_I": float(np.max(np.abs(cau["I"] - full["I"])) / np.max(np.abs(full["I"]))),
                     "I_s1_over_s0_range": [float(ratio.min()), float(ratio.max())]}
        if name == "delta":
            frz = C.build_columns(C.AveragedKernel(K, [depth_parent_field_frozen(), eta]), tab)
            out[name]["frozen_I_s1_max_over_true"] = float(np.max(np.abs(frz["I"][s == 1])) / np.max(np.abs(full["I"][s == 1])))
            out[name]["frozen_vs_jet_I"] = float(np.max(np.abs(frz["I"] - full["I"])) / np.max(np.abs(full["I"])))
        else:
            rw = C.build_columns(C.AveragedKernel(K, [lognormal_field_reweighted(), eta]), tab)
            out[name]["reweighted_vs_jet_I"] = float(np.max(np.abs(rw["I"] - full["I"])) / np.max(np.abs(full["I"])))
            out[name]["reweighted_vs_jet_P"] = float(np.max(np.abs(rw["P"] - full["P"])) / np.max(np.abs(full["P"])))
    # direct integration for the Delta law on channels where tau_max h is small (2.4 to 36 GHz supports)
    ch2 = C.bump_channels(n=n_ch, nu_lo=3e9, nu_hi=3e10, n_nodes=32)
    K = C.ContinuumChannelKernel(ch2)
    avg = C.AveragedKernel(K, [depth_parent_field(), eta])
    full = C.build_columns(avg, tab)
    g = C.panel_measure("gamma", lambda x: x**-2.6, np.unique(np.r_[tab.lo[:, 0], tab.hi[:, 0]]), n=6)
    d = C.panel_measure("depth", lambda x: 1 + 0.2 * (x - 8.0), [6.0, 10.0], n=6, chart="affine")
    m = C.product(g, d)
    w = C.reference_weights(tab)
    pred = C.predict(full, tab, C.pair(tab, m, w), w, azimuth=AZ)
    ref = C.direct(avg, m, tab, azimuth=AZ)
    out["delta_predict_vs_direct"] = stokes_errors(pred, ref)
    out["tau_max_h_columns"] = Nd.tau_max(ch) * 2.0
    out["tau_max_h_direct"] = Nd.tau_max(ch2) * 2.0
    return out
