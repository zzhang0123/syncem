"""R5 cases on the continuum kernel of the paper's E1 configuration.

E1: 41 bump channels 0.1 to 10 GHz, Gaussian screen (8, 0.5) rad m^-2 inside the channel
integral, B ~ U[4, 6] uG, eta on the half range [0, 0.5] (12 nodes), gamma^-2.6 on
[300, 2.9626e6] with 19 log cells. The constant factor 0.4 e^{0.5i} of E1 is replaced by a
factor c that depends on gamma (route ii), B (iii), eta (iii, iv) or depth (iii).

The direct reference integrates the full joint measure: gamma panels (6 nodes per cell),
the parent as a measure variable with more nodes than the basis rule, and phi by the
two-point law given the parents. It never uses c.
"""
from __future__ import annotations

import time

import jax
import jax.numpy as jnp
import numpy as np
from numpy.polynomial.hermite_e import hermegauss
from numpy.polynomial.legendre import leggauss

import core as C
from core import paper as E
from core.kernels import MUG, ContinuumChannelKernel, bump_channels

from . import joint as J
from . import routes as R

LN1E4 = np.log(1e4)


def eta_half_rule(n: int = 12):
    e, we = leggauss(n)
    return C.fixed(0.25 * (e + 1.0), we * 0.5)


def base_kernel(screen=E.SCREEN, n_nodes_F: int = 128) -> ContinuumChannelKernel:
    return ContinuumChannelKernel(bump_channels(n_nodes=64), screen=screen, n_nodes_F=n_nodes_F)


def gamma_measure(table, n: int = 6):
    return E.power_law_measure(table, n=n)


def table(N: int):
    return E.e1_table(N)


def slice_rows(cols: dict, big, N: int):
    """Rows with total degree <= N of a build on ``big`` (the N-table is a row subset)."""
    keep = big.row_exp.sum(axis=1) <= N
    return {k: v[keep] for k, v in cols.items()}


def stokes_P(x: np.ndarray) -> np.ndarray:
    return x[:, 1] + 1j * x[:, 2]


def per_channel(pred: np.ndarray, ref: np.ndarray) -> dict:
    """Errors relative to I and to |P| per channel, summarised at the extremes."""
    eI, eP = R.rel_err_I(pred, ref), R.rel_err_P(pred, ref)
    dP_I = np.abs(stokes_P(pred) - stokes_P(ref)) / np.abs(ref[:, 0])
    return {"max_rel_I": float(eI.max()), "max_dP_over_I": float(dP_I.max()), "max_rel_P": float(eP.max()), "median_rel_P": float(np.median(eP)),
            "ch0_rel_I": float(eI[0]), "ch0_rel_P": float(eP[0]), "ch40_rel_I": float(eI[-1]),
            "ch40_rel_P": float(eP[-1]), "P_over_I_ch0": float(np.abs(stokes_P(ref))[0] / ref[0, 0]),
            "P_over_I_ch40": float(np.abs(stokes_P(ref))[-1] / ref[-1, 0])}


class Timer:
    def __init__(self):
        self.t = {}

    def run(self, name, fn, *a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        self.t[name] = time.perf_counter() - t0
        return out


# ------------------------------------------------------------------ single electron, r3-02
def single_electron_ratio(xs, cB, B_lo=4.0, B_hi=6.0, n_B=48, eta=None, gamma=1e4):
    """|<c K_P> - <c><K_P>| / |<c K_P>| at nu = x nu_c(5 uG, B_perp = 5 uG sqrt(1 - eta^2)).

    ``eta``: a fixed cosine, or None for eta averaged on [0, 0.5] (12 nodes). No channel
    integral and no screen: one frequency per x.
    """
    from syncem.model import ContinuumKernel
    from syncem.constants import C_CGS, E_ESU, M_E

    base = ContinuumKernel(n_nodes_F=128, x_min=1e-12)
    t, wt = leggauss(n_B)
    Bq, wB = 0.5 * (B_hi - B_lo) * t + 0.5 * (B_hi + B_lo), 0.5 * wt
    if eta is None:
        e, we = leggauss(12)
        etas, weta = 0.25 * (e + 1.0), we * 0.5
    else:
        etas, weta = np.array([eta]), np.array([1.0])
    a_B = 3 * E_ESU * (5.0 * MUG) * np.sqrt(1 - (0.0 if eta is None else eta) ** 2) / (4 * np.pi * M_E * C_CGS)
    nu = np.asarray(xs) * a_B * gamma**2
    KP = np.zeros((Bq.size, nu.size))
    for i, B in enumerate(Bq):
        for e_, we_ in zip(etas, weta):
            KP[i] += we_ * np.asarray(base.kernels(jnp.asarray(nu), gamma, B * MUG, e_)[1])
    c = np.asarray([cB(b) for b in Bq], complex)
    good = wB @ (c[:, None] * KP)
    bad = (wB @ c) * (wB @ KP)
    return np.abs(good - bad) / np.abs(good)


# ------------------------------------------------------------------ route (iii), B parent
def field_case(cB_jnp, cB_np, N_list=(2,), B_lo=4.0, B_hi=6.0, n_basis=16, n_direct=24, cache=None, cols_s1=None,
               kernel=None, avg_direct=None, timer=None):
    """Lattice predict (c inside the field average) and the factorised route against direct."""
    timer = timer or Timer()
    K = kernel or base_kernel()
    Nmax = max(N_list)
    big = table(Nmax)
    rule = C.uniform(B_lo, B_hi, n_basis)
    avg_ok = C.AveragedKernel(R.FactorKernel(K, lambda p: cB_jnp(p["B"])), [("B", rule), ("eta", eta_half_rule())])
    cols_ok_big = timer.run(f"cols_ok_N{Nmax}", C.build_columns, avg_ok, big)
    if cols_s1 is None:
        avg_s1 = C.AveragedKernel(K, [("B", rule), ("eta", eta_half_rule())])
        cols_s1 = timer.run(f"cols_s1_N{Nmax}", C.build_columns, avg_s1, big)
    gm = gamma_measure(big)
    m = C.product(gm, C.panel_measure("B", lambda x: 0 * x + 1 / (B_hi - B_lo), [B_lo, B_hi], n=n_direct,
                                      chart="affine"))
    joint = J.with_phi(m, J.factor_law(lambda nd: cB_np(nd["B"])))
    avg_d = avg_direct or C.AveragedKernel(K, [("eta", eta_half_rule())])
    ref = timer.run("direct_joint", J.direct_joint, avg_d, joint, big, cache=cache)
    s1_direct = J.direct_joint(avg_d, m, big, cache=cache)  # <K_P> over the same nodes, no factor
    cbar = J.mean_factor(joint)
    wrong_free = s1_direct.copy()
    wrong_free[:, 1:] = np.stack([(cbar * stokes_P(s1_direct)).real, (cbar * stokes_P(s1_direct)).imag], axis=1)
    dP = np.abs(stokes_P(wrong_free) - stokes_P(ref))
    rel_fact = dP / np.abs(stokes_P(wrong_free))
    out = {"cbar": [cbar.real, cbar.imag], "lattice_free_wrong": per_channel(wrong_free, ref),
           "lattice_free_rel_to_factorised": {"max": float(rel_fact.max()), "min": float(rel_fact.min()),
                                              "median": float(np.median(rel_fact)), "ch0": float(rel_fact[0]),
                                              "ch40": float(rel_fact[-1])}}
    for N in N_list:
        tab = table(N)
        w = E.e1_weights(tab)
        a = C.pair(tab, gm, w)
        c_ok, c_s1 = slice_rows(cols_ok_big, big, N), slice_rows(cols_s1, big, N)
        out[f"N{N}"] = {"correct": per_channel(C.predict(c_ok, tab, a, w), ref),
                        "wrong_mean_c_times_S1": per_channel(R.assemble_constant(c_s1, tab, a, w, cbar), ref),
                        "I_block_equal_S1": float(np.max(np.abs(c_ok["I"] - c_s1["I"])) / np.max(np.abs(c_s1["I"])))}
    return out, cols_s1


# ------------------------------------------------------------------ route (ii), gamma parent
def gamma_case(c_jnp, cols_big, big, N_list, cache, avg_plain, jet_check=None):
    """Cauchy product at assembly vs direct, with the harmonic route and two wrong routes."""
    gm = gamma_measure(big)
    c_np = lambda nd: np.asarray(c_jnp({"gamma": jnp.asarray(nd["gamma"])}), complex)
    joint = J.with_phi(gm, J.factor_law(c_np))
    ref = J.direct_joint(avg_plain, joint, big, cache=cache)
    cbar = J.mean_factor(joint)
    out = {"cbar": [cbar.real, cbar.imag]}
    for N in N_list:
        tab = table(N)
        w = E.e1_weights(tab)
        cols = slice_rows(cols_big, big, N)
        a = C.pair(tab, gm, w)
        ctay = R.factor_taylor(tab, c_jnp)
        P_c = R.cauchy_product(cols["P"], tab, ctay)
        out[f"N{N}"] = {
            "cauchy": per_channel(R.predict_with_P(cols, P_c, tab, a, w), ref),
            "harmonic_M2": per_channel(C.predict(cols, tab, C.pair(tab, joint, w, harmonic=True), w), ref),
            "wrong_global_mean": per_channel(R.assemble_constant(cols, tab, a, w, cbar), ref),
            "wrong_zeroth_order": per_channel(R.predict_with_P(cols, R.zeroth_order_factor(cols["P"], tab, ctay),
                                                               tab, a, w), ref),
        }
        if jet_check is not None and N == jet_check[0]:
            jet = jet_check[1]
            out[f"N{N}"]["cauchy_vs_jet_of_factor_kernel"] = float(np.max(np.abs(P_c - jet["P"])) /
                                                                   np.max(np.abs(jet["P"])))
    return out


# ------------------------------------------------------------------ depth parent (route iii)
def gh_rule(mean, sd, n):
    x, w = hermegauss(n)
    return mean + sd * x, w / np.sqrt(2 * np.pi)
