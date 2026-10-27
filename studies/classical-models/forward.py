"""Forward tests: moment prediction C a_true versus direct quadrature.

For every reference ensemble: max_j |S_direct - C a|/I_j per Stokes component for
N = 0..4; a measured Taylor-remainder envelope per N (sup of the (N+1)-th x_b-derivative
of Kbar on a 65-point grid per bin times the absolute moments; measured, not rigorous);
node-doubling checks of every direct spectrum.  Extras: E1 against the analytic R&L power
law (eta average, finite-support edge) and (p+1)/(p+7/3); KP versus pitch-Legendre order
L = 0..16; E5 joint versus separable azimuth and the de-rotated angle chi(nu); E1M with
the U[4,6] and the Maxwellian basis; JP local spectral index.

Writes forward_results.json and forward_spectra.npz.  Run: python forward.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import sys
import time
from dataclasses import replace
from math import factorial
from pathlib import Path

import numpy as np
from numpy.polynomial.legendre import leggauss
from scipy import integrate
from scipy.interpolate import CubicSpline
from scipy.special import gamma as Gamma
from scipy.special import hyp2f1

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import basis as B  # noqa: E402
import ensembles as E  # noqa: E402
import kernels as K  # noqa: E402
from config import PAPER, PAPER_MAXWELL, Z1, rho1  # noqa: E402
from provenance import provenance, write_json  # noqa: E402

T0 = time.time()
NS = (0, 1, 2, 3, 4)
NMAX = 4
N_GRID = 65
REF = dict(n_chan_nodes=800, n_b=32, n_eta=24)
STOKES = ("I", "Q", "U")
TIMINGS = {}


def tic(label, t):
    TIMINGS[label] = TIMINGS.get(label, 0.0) + time.time() - t


def rel(S, P):
    """max_j |S - P| / I_j per Stokes component."""
    return [float(v) for v in np.max(np.abs(S - P) / S[:, 0:1], axis=0)]


# ------------------------------------------------------------------ basis and envelope
class BasisSet:
    """Separable/joint columns up to N = 4 and derivative grids up to order 5 for one config."""

    def __init__(self, cfg, pitch_l=None):
        t = time.time()
        self.cfg = cfg
        self.DI, self.DQ = B.derivative_spectra(cfg, NMAX, pitch_l=pitch_l)
        tic("basis_columns", t)
        self._grids = None
        self.pitch_l = pitch_l

    def separable(self, N, factor=None):
        return B.separable_columns(self.cfg, self.DI[:, :N + 1], self.DQ[:, :N + 1], factor)

    def joint(self, N):
        return B.joint_columns(self.cfg, self.DI[:, :N + 1], self.DQ[:, :N + 1])

    def grids(self):
        """dI[m], dQ[m]: d^m/dx_b^m Kbar on (n_bins, N_GRID, J), m = 0..NMAX+1."""
        if self._grids is None:
            t = time.time()
            cfg = self.cfg
            x = np.linspace(-1.0, 1.0, N_GRID)
            g = (cfg.gamma_b[:, None] * np.exp(cfg.h * x[None, :])).ravel()
            kI, kQ = B.kbar(cfg, g, order=NMAX + 1, pitch_l=self.pitch_l)
            sh = (cfg.n_bins, N_GRID, -1)
            self._grids = ([cfg.h**m * kI[m].reshape(sh) for m in range(NMAX + 2)],
                           [cfg.h**m * kQ[m].reshape(sh) for m in range(NMAX + 2)])
            tic("derivative_grids", t)
        return self._grids

    def envelope(self, N, zones):
        """Measured Taylor envelope (J, 3); zones = [(factor, abs_moments_{N+1} (n_bins,))]."""
        dI, dQ = self.grids()
        m = N + 1
        supI = np.max(np.abs(dI[m]), axis=1)  # (nb, J)
        env = np.zeros((self.cfg.n_channels, 3))
        for c, am in zones:
            p = c * dQ[m]
            env[:, 0] += am @ supI
            env[:, 1] += am @ np.max(np.abs(p.real), axis=1)
            env[:, 2] += am @ np.max(np.abs(p.imag), axis=1)
        return env / factorial(m)


def forward_table(S, preds, envs=None):
    """Per N: residual metrics and, if envs given, envelope comparison."""
    out = {}
    for N in NS:
        resid = np.abs(S - preds[N])
        row = {"max_resid_over_I_IQU": [float(v) for v in np.max(resid / S[:, 0:1], axis=0)]}
        if envs is not None:
            env = envs[N]
            ok = resid > 1e-13 * S[:, 0:1]
            ratio = np.where(ok, resid / np.where(env > 0, env, np.inf), 0.0)
            row.update({
                "max_envelope_over_I_IQU": [float(v) for v in np.max(env / S[:, 0:1], axis=0)],
                "max_resid_over_envelope": float(np.max(ratio)),
                "cells_resid_exceeds_envelope": int(np.sum(ratio > 1.0)),
                "envelope_over_resid_at_max_resid_IQU": [
                    float(env[np.argmax(resid[:, k] / S[:, 0]), k] / max(resid[:, k].max(), 1e-300)
                          * 1.0) for k in range(3)],
            })
        out[f"N{N}"] = row
    return out


def quadvec_cutoff_check(cfg, smooth_part, gamma_c, alpha, lo, eta=None, measure=None):
    """Independent route for a (1 - gamma/gamma_c)^alpha cut-off: substitute
    1 - gamma/gamma_c = w^5 (smooth integrand for alpha = 0.6) and integrate the 41x3
    channel spectrum adaptively with scipy quad_vec; compare with the Gauss-Jacobi measure.

    smooth_part(gamma) = N(gamma) / (1 - gamma/gamma_c)^alpha.
    """
    s_eta = None if eta is None else [np.sqrt(1 - eta**2)]

    def kern(g):
        if s_eta is None:
            kI, kQ = B.kbar(cfg, [g], order=0)
            return B.to_stokes(kI[0, 0], kQ[0, 0], cfg.azimuth_factor)
        KI, KQ = B.channel_kernels(cfg, [g], eta_s=s_eta, order=0)
        return B.to_stokes(KI[0, 0, 0], KQ[0, 0, 0], cfg.azimuth_factor)

    wmax = (1.0 - lo / gamma_c) ** 0.2

    def f(w):
        g = gamma_c * (1.0 - w**5)
        return (kern(g) * smooth_part(g) * w ** (5 * alpha) * 5.0 * gamma_c * w**4).ravel()

    t = time.time()
    ref, err = integrate.quad_vec(f, 0.0, wmax, epsabs=0, epsrel=1e-11, limit=400)
    ref = ref.reshape(cfg.n_channels, 3)
    if s_eta is None:
        ours = direct_measure(cfg, measure)
    else:
        KI, KQ = B.channel_kernels(cfg, measure.gamma, eta_s=s_eta, order=0)
        ours = B.to_stokes(measure.w @ KI[0, :, 0], measure.w @ KQ[0, :, 0], cfg.azimuth_factor)
    return {"max_abs_diff_over_I_IQU": rel(ref, ours), "quad_vec_err_estimate_2norm_over_min_I": float(
        err / np.min(ref[:, 0])), "seconds": time.time() - t}


def direct_measure(cfg, m):
    return E.direct_from_measure(cfg, m)


# ------------------------------------------------------------------ ensembles
def run_separable(name, ens, bs, cfg):
    t = time.time()
    S = ens.direct(cfg, n=24)
    tic(f"direct_{name}", t)
    t = time.time()
    chk = ens.direct_check(cfg, 24)
    S_ref = ens.direct(replace(cfg, **REF), n=48)
    tic(f"checks_{name}", t)
    m = ens.measure(cfg, 24)
    preds, envs = {}, {}
    for N in NS:
        a = m.moments(cfg, N)
        preds[N] = B.predict(bs.separable(N), a)
        envs[N] = bs.envelope(N, [(cfg.azimuth_factor, m.abs_moments(cfg, N + 1))])
    res = {"direct_check_energy_nodes": chk,
           "direct_check_config_doubling_(chan800,B32,eta24,n48)_max_change_over_I_IQU": rel(S_ref, S),
           "forward": forward_table(S, preds, envs),
           "moments_N2": m.moments(cfg, 2)}
    if isinstance(ens, E.JaffePerola):
        sp = lambda g: ens.norm * (g / E.GAMMA0) ** (-ens.p) / E.GAMMA0  # noqa: E731
        res["independent_quad_vec_check"] = quadvec_cutoff_check(cfg, sp, ens.gamma_c, ens.alpha, ens.lo,
                                                                 measure=m)
    return S, res


def run_kp(ens, bs_iso, cfg):
    t = time.time()
    S = ens.direct(cfg, n=24, n_eta=16)
    tic("direct_E4-KP", t)
    t = time.time()
    chk = ens.direct_check(cfg, 24, 16)
    S_ref = ens.direct(replace(cfg, **REF), n=48, n_eta=32)
    tic("checks_E4-KP", t)
    preds, envs, preds_marg = {}, {}, {}
    leg = ens.legendre_moments(cfg, ells=tuple(range(0, 17, 2)), N=NMAX, n=24, n_mu=24)
    for N in NS:
        a_eff = ens.effective_moments(cfg, N, 24, 16)
        preds[N] = B.predict(bs_iso.separable(N), a_eff)
        preds_marg[N] = B.predict(bs_iso.separable(N), leg[0][:, :N + 1])
        envs[N] = bs_iso.envelope(N, [(cfg.azimuth_factor,
                                       ens.effective_moments(cfg, N, 24, 16, absolute_m=N + 1))])
    # pitch-Legendre columns on a denser eta rule (P_16 weights); checked against 64 nodes
    t = time.time()
    cfg_p = replace(cfg, n_eta=32)
    pitch = {l: BasisSet(cfg_p, pitch_l=l) for l in range(0, 17, 2)}
    chk64 = BasisSet(replace(cfg, n_eta=64), pitch_l=16)
    eta_conv = float(np.max(np.abs(chk64.separable(4) - pitch[16].separable(4))
                            / S[:, 0:1, None]))
    tic("pitch_columns", t)
    legendre = {}
    for N in (2, 4):
        row = {}
        for L in (0, 2, 4, 8, 16):
            pred = sum(B.predict(pitch[l].separable(N), leg[l][:, :N + 1]) for l in range(0, L + 1, 2))
            row[f"L{L}"] = rel(S, pred)
        legendre[f"N{N}"] = row
    a_eff2 = ens.effective_moments(cfg, 2, 24, 16)
    eta_c = 0.25
    gc = float(ens.ageing.gamma_c_kp(eta_c))
    spk = lambda g: 2.0 * ens.norm_unscaled_inv * 0.5 * (g / E.GAMMA0) ** (-ens.p) / E.GAMMA0  # noqa: E731
    qv = quadvec_cutoff_check(cfg, spk, gc, ens.alpha, ens.lo, eta=eta_c,
                              measure=ens.gamma_measure(cfg, eta_c, 24, scale=2.0))
    res = {
        "independent_quad_vec_check_at_eta=0.25": qv,
        "direct_check": chk,
        "direct_check_config_doubling_(chan800,B32,eta24,n48,neta32)_max_change_over_I_IQU": rel(S_ref, S),
        "forward_effective_diagonal_iso_basis": forward_table(S, preds, envs),
        "forward_marginal_moments_iso_basis": forward_table(S, preds_marg),
        "forward_pitch_legendre_vs_L": legendre,
        "pitch_columns_eta32_vs_eta64_max_over_I_(l=16,N=4)": eta_conv,
        "moments_effective_N2": a_eff2,
        "moments_marginal_N2": leg[0][:, :3],
        "moments_legendre_l2_N2": leg[2][:, :3],
        "marginal_minus_effective_max_abs_a0": float(np.max(np.abs(leg[0][:, 0] - a_eff2[:, 0]))),
        "effective_column": float(np.sum(a_eff2[:, 0] * cfg.w_b)),
        "P2_conditional": {"gamma": [3e3, 1e4, 1.5e4, 1.7e4, 2e4, 2.5e4, 3e4],
                           "value": ens.p2_conditional([3e3, 1e4, 1.5e4, 1.7e4, 2e4, 2.5e4, 3e4]).tolist()},
    }
    return S, res


def run_e5(ens, bs, cfg):
    t = time.time()
    fr = ens.fractions(cfg, 48)
    S = ens.direct(cfg, 24, fractions=fr)
    tic("direct_E5", t)
    chk = ens.direct_check(cfg, 24)
    S_ref = ens.direct(replace(cfg, **REF), 48, fractions=fr)
    preds_joint, preds_sep, envs = {}, {}, {}
    ca, cb = ens.factors()
    for N in NS:
        a, M2 = ens.moments(cfg, N, 24, fr)
        preds_joint[N] = B.predict(bs.joint(N), np.concatenate([a.ravel(), M2.real.ravel(), M2.imag.ravel()]))
        preds_sep[N] = B.predict(bs.separable(N), a)
        (fa, Am), (fb, Bm) = ens.zone_abs_moments(cfg, N + 1, 24, fr)
        envs[N] = bs.envelope(N, [(ca, fa * Am), (cb, fb * Bm)])
    chi = E.derotated_angle(cfg, S)
    cfg0 = replace(cfg, screen_mean=0.0, screen_sd=0.0)
    S0 = ens.direct(cfg0, 24, fractions=fr)
    chi0 = 0.5 * np.angle(S0[:, 1] + 1j * S0[:, 2])
    pol = np.hypot(S[:, 1], S[:, 2]) / S[:, 0]
    sel = pol > 0.03
    a2, M22 = ens.moments(cfg, 2, 24, fr)
    res = {
        "fractions_A_B_crossing_channel": list(fr),
        "direct_check": chk,
        "direct_check_config_doubling_max_change_over_I_IQU": rel(S_ref, S),
        "forward_joint_azimuth": forward_table(S, preds_joint, envs),
        "forward_separable_paper_factor_0.4e^{0.5i}": forward_table(S, preds_sep),
        "chi_derotated_deg": np.degrees(chi).tolist(),
        "chi_intrinsic_no_screen_deg": np.degrees(chi0).tolist(),
        "P_over_I": pol.tolist(),
        "chi_derotated_swing_deg_where_P/I>0.03": float(np.degrees(np.ptp(np.unwrap(2 * chi[sel]) / 2))),
        "chi_intrinsic_swing_deg_where_P/I>0.03": float(np.degrees(np.ptp(np.unwrap(2 * chi0[sel]) / 2))),
        "moments_N2": a2, "M2_N2_real": M22.real, "M2_N2_imag": M22.imag,
    }
    return S, res


def run_e1m(ens, bs_u, cfg_m):
    t = time.time()
    S = ens.direct(cfg_m, n=24)
    tic("direct_E1M", t)
    S_ref = ens.direct(replace(cfg_m, n_b=192, n_chan_nodes=800, n_eta=24), n=48)
    bs_m = BasisSet(cfg_m)
    m = ens.measure(cfg_m, 24)
    preds_u, preds_m, envs_m = {}, {}, {}
    for N in NS:
        a = m.moments(cfg_m, N)
        preds_u[N] = B.predict(bs_u.separable(N), a)
        preds_m[N] = B.predict(bs_m.separable(N), a)
        envs_m[N] = bs_m.envelope(N, [(cfg_m.azimuth_factor, m.abs_moments(cfg_m, N + 1))])
    res = {"direct_check_energy_nodes": ens.direct_check(cfg_m, 24),
           "direct_check_(B192,chan800,eta24,n48)_max_change_over_I_IQU": rel(S_ref, S),
           "forward_U46_basis_(misspecified_field)": forward_table(S, preds_u),
           "forward_maxwell_basis": forward_table(S, preds_m, envs_m)}
    return S, res


# ------------------------------------------------------------------ E1 analytic
def _bperp_moment(cfg, s):
    """<B_perp^s> = <B^s>_U[b_lo,b_hi] <(1-eta^2)^{s/2}>_U[-eta_max,eta_max] (analytic)."""
    bmom = (cfg.b_hi ** (s + 1) - cfg.b_lo ** (s + 1)) / ((s + 1) * (cfg.b_hi - cfg.b_lo))
    emom = hyp2f1(0.5, -s / 2, 1.5, cfg.eta_max**2)
    return bmom * emom


def _channel_integral(cfg, j, f):
    """int R_j(nu) f(nu) dnu by adaptive quadrature (independent of the GL channel rule)."""
    nj, hj = cfg.centres[j], cfg.chan_half * cfg.centres[j]
    v, _ = integrate.quad(lambda t: float(rho1(np.array(t))) * f(nj + hj * t), -1, 1,
                          epsabs=0, epsrel=1e-12, limit=400)
    return v / Z1


def _incomplete_mellin(m, which):
    """Tables X -> int_0^X x^m W dx and int_X^inf x^m W dx on ln X in [ln 1e-12, ln 900]."""
    t, w = leggauss(10)
    edges = np.linspace(np.log(1e-12), np.log(900.0), 4001)
    inc = np.zeros(edges.size - 1)
    for i in range(edges.size - 1):
        a, b = edges[i], edges[i + 1]
        u = 0.5 * (b - a) * t + 0.5 * (a + b)
        val = np.exp((m + 1) * u) * K.log_derivatives(np.exp(u), 0, which)[:, 0]
        inc[i] = 0.5 * (b - a) * np.sum(w * val)
    lo = np.concatenate([[0.0], np.cumsum(inc)])
    small = (K.F_SMALL_X_COEFF if which == "F" else 2 ** (-1 / 3) * Gamma(2 / 3)) * 1e-12 ** (m + 4 / 3) / (m + 4 / 3)
    lo = lo + small
    total = K.mellin_F(m) if which == "F" else K.mellin_G(m)
    hi = np.concatenate([np.cumsum(inc[::-1])[::-1], [0.0]])  # int_X^{900}, no cancellation
    return edges, lo, hi, total


def e1_analytic(S_direct, cfg):
    """E1 versus unbounded R&L power law, the leading edge formula and the exact incomplete
    Mellin (finite support) evaluation; intrinsic polarisation fraction."""
    e1 = E.PowerLaw()
    p = e1.p
    alpha, mu = (p - 1) / 2, (p - 3) / 2
    C = e1.norm * E.GAMMA0 ** (p - 1)  # N = C gamma^-p
    s = (p + 1) / 2
    k = (3 * p - 1) / 6
    pref = C * K.A1 / 2
    gmax, gmin = e1.hi, e1.lo
    I_inf, I_lo, Pq_inf = [], [], []
    for j in range(cfg.n_channels):
        c1 = _channel_integral(cfg, j, lambda nu: (nu / K.a1) ** (-alpha))
        c13 = _channel_integral(cfg, j, lambda nu: (nu / K.a1) ** (1 / 3))
        I_inf.append(pref * _bperp_moment(cfg, s) * K.mellin_F(mu) * c1)
        I_lo.append(I_inf[-1] - pref * K.F_SMALL_X_COEFF / k * _bperp_moment(cfg, 2 / 3) * c13 * gmax ** (-2 * k))
        lam = lambda nu: (K.C_SI / nu) ** 2  # noqa: E731
        scr_re = _channel_integral(cfg, j, lambda nu: (nu / K.a1) ** (-alpha) * np.cos(
            2 * lam(nu) * cfg.screen_mean) * np.exp(-2 * lam(nu) ** 2 * cfg.screen_sd**2))
        scr_im = _channel_integral(cfg, j, lambda nu: (nu / K.a1) ** (-alpha) * np.sin(
            2 * lam(nu) * cfg.screen_mean) * np.exp(-2 * lam(nu) ** 2 * cfg.screen_sd**2))
        Pq_inf.append(-pref * _bperp_moment(cfg, s) * K.mellin_G(mu) * (scr_re + 1j * scr_im))
    I_inf, I_lo = np.array(I_inf), np.array(I_lo)
    P_inf = cfg.azimuth_factor * np.array(Pq_inf)
    # exact finite-support route in x-space (incomplete Mellin), cfg B/eta/channel rules
    b, wb = cfg.b_nodes_weights
    e, we = cfg.eta_nodes_weights
    bp = (b[:, None] * np.sqrt(1 - e[None, :] ** 2)).ravel()
    wbp = (wb[:, None] * we[None, :]).ravel()
    exact = {}
    for which in ("F", "G"):
        ed, lo, hi, tot = _incomplete_mellin(mu, which)
        nu = cfg.chan_nu[:, :, None]
        xm = nu / (K.a1 * bp * gmax**2)
        xp = nu / (K.a1 * bp * gmin**2)
        lo_s = CubicSpline(ed, np.log(lo))
        ok = (ed <= np.log(700.0)) & (hi > 1e-290)
        hi_s = CubicSpline(ed[ok], np.log(hi[ok]))
        lxp = np.log(xp)
        hi_v = np.where(lxp < np.log(700.0), np.exp(hi_s(np.minimum(lxp, np.log(700.0)))), 0.0)
        part = tot - np.exp(lo_s(np.log(xm))) - hi_v
        val = pref * np.sum(wbp * bp * (nu / (K.a1 * bp)) ** (-alpha) * part, axis=2)  # (J, T)
        exact[which] = val
    I_ex = exact["F"] @ cfg.chan_w
    Pq_ex = -(exact["G"] * cfg.screen) @ cfg.chan_w
    # intrinsic polarisation fraction (no screen, azimuth factor 1)
    cfg0 = replace(cfg, screen_mean=0.0, screen_sd=0.0, azimuth_amp=2.0, azimuth_phase=0.0)
    S0 = e1.direct(cfg0, n=24)
    Pi_dir = -S0[:, 1] / S0[:, 0]
    Pi_ex = (exact["G"] @ cfg.chan_w) / I_ex
    # R&L 6.36 per steradian, per unit nu, at one (B_perp, nu): identity check of the prefactor
    bt, nut = 5.0 * K.MUG, 1e9
    rl = (np.sqrt(3) * K.E_ESU**3 * C * bt / (4 * np.pi * K.M_E * K.C_CGS**2 * (p + 1))
          * Gamma(p / 4 + 19 / 12) * Gamma(p / 4 - 1 / 12)
          * (2 * np.pi * K.M_E * K.C_CGS * nut / (3 * K.E_ESU * bt)) ** (-(p - 1) / 2))
    mel = pref * 5.0 * K.mellin_F(mu) * (nut / (K.a1 * 5.0)) ** (-alpha)
    sel = [0, 10, 20, 30, 40]
    return {
        "p": p, "C_gamma^-p": C, "<B_perp^((p+1)/2)>_analytic": _bperp_moment(cfg, s),
        "RL636_vs_mellin_form_rel": float(abs(rl / mel - 1)),
        "I_direct_over_I_unbounded_RL": (S_direct[:, 0] / I_inf).tolist(),
        "max_abs(I_direct/I_unbounded - 1)": float(np.max(np.abs(S_direct[:, 0] / I_inf - 1))),
        "max_abs(I_direct/I_edge_leading_order - 1)": float(np.max(np.abs(S_direct[:, 0] / I_lo - 1))),
        "I_direct/I_edge_leading_order_at_0.1_0.316_1_3.16_10GHz": (S_direct[sel, 0] / I_lo[sel]).tolist(),
        "max_abs(I_direct/I_exact_incomplete_mellin - 1)": float(np.max(np.abs(S_direct[:, 0] / I_ex - 1))),
        "max_abs(P_direct - P_exact)/I": float(np.max(np.abs(
            (S_direct[:, 1] + 1j * S_direct[:, 2]) - cfg.azimuth_factor * Pq_ex) / I_ex)),
        "max_abs(P_direct - P_unbounded_RL)/I": float(np.max(np.abs(
            (S_direct[:, 1] + 1j * S_direct[:, 2]) - P_inf) / S_direct[:, 0])),
        "Pi_unbounded_(p+1)/(p+7/3)": float(K.powerlaw_pol_fraction(p)),
        "Pi_direct_no_screen_at_0.1_0.316_1_3.16_10GHz": Pi_dir[sel].tolist(),
        "Pi_exact_incomplete_mellin_at_same": Pi_ex[sel].tolist(),
        "max_abs(Pi_direct - Pi_exact)": float(np.max(np.abs(Pi_dir - Pi_ex))),
        "abs(Pi_direct(0.1GHz) - (p+1)/(p+7/3))": float(abs(Pi_dir[0] - K.powerlaw_pol_fraction(p))),
        "band_slope_lsq_41ch": float(np.polyfit(np.log(cfg.centres), np.log(S_direct[:, 0]), 1)[0]),
        "edge_formula": "1 - I/I_inf ~ F_small x_-^k / (k M_F), k = (3p-1)/6, x_- = nu/(a_B gamma_max^2); "
                        "averaged over B_perp analytically (B_perp^{2/3})",
    }


def local_index(cfg, S):
    """alpha_j = -d ln I / d ln nu by centred differences on the 41 channel centres."""
    return -np.gradient(np.log(S[:, 0]), np.log(cfg.centres))


def family_checks(cfg, ens):
    """a(theta_true) from the vectorised families versus the quadrature truth (N = 4), and timing."""
    out = {}
    for name in ("E1", "E2", "E3", "E4-JP"):
        e = ens[name]
        truth = e.moments(cfg, NMAX, 24)
        fam = E.FAMILIES[name](cfg, e.theta, NMAX)[0]
        th = np.tile(e.theta, (1000, 1)) * (1 + 1e-3 * np.random.default_rng(0).standard_normal((1000, e.theta.size)))
        t = time.time()
        E.FAMILIES[name](cfg, th, 2)
        out[name] = {"theta_true": e.theta.tolist(), "max_abs_diff_N4": float(np.max(np.abs(fam - truth))),
                     "seconds_per_1000_theta_N2": time.time() - t}
    e5 = ens["E5"]
    fr = e5.fractions(cfg, 48)
    za, zb = e5.zones
    th = np.array([fr[0] * za.norm, za.p, fr[1] * zb.norm, zb.p])
    a_f, m_f = E.two_zone_moments(cfg, th, e5.factors(), NMAX)
    a_t, m_t = e5.moments(cfg, NMAX, 24, fr)
    out["E5"] = {"theta_true_(A_a,p_a,A_b,p_b)": th.tolist(),
                 "max_abs_diff_a_N4": float(np.max(np.abs(a_f[0] - a_t))),
                 "max_abs_diff_M2_N4": float(np.max(np.abs(m_f[0] - m_t)))}
    return out


# ------------------------------------------------------------------ main
def main():
    cfg = PAPER
    ens = E.reference_ensembles(cfg)
    bs = BasisSet(cfg)
    results, spectra = {}, {}
    for name in ("E1", "E2", "E3", "E4-JP"):
        S, res = run_separable(name, ens[name], bs, cfg)
        res["params"] = ens[name].describe()
        results[name], spectra[name] = res, S
        print(name, res["forward"]["N2"]["max_resid_over_I_IQU"], flush=True)
    S, res = run_kp(ens["E4-KP"], bs, cfg)
    res["params"] = ens["E4-KP"].describe()
    results["E4-KP"], spectra["E4-KP"] = res, S
    print("E4-KP", res["forward_effective_diagonal_iso_basis"]["N2"]["max_resid_over_I_IQU"], flush=True)
    S, res = run_e5(ens["E5"], bs, cfg)
    res["params"] = ens["E5"].describe()
    results["E5"], spectra["E5"] = res, S
    print("E5", res["forward_joint_azimuth"]["N2"]["max_resid_over_I_IQU"], flush=True)
    S, res = run_e1m(ens["E1M"], bs, PAPER_MAXWELL)
    res["params"] = {**ens["E1M"].describe(), "field": "Maxwellian |B|, B_rms = 5 muG",
                     "maxwell_config": PAPER_MAXWELL.describe()}
    results["E1M"], spectra["E1M"] = res, S
    print("E1M", res["forward_U46_basis_(misspecified_field)"]["N2"]["max_resid_over_I_IQU"], flush=True)
    t = time.time()
    e1a = e1_analytic(spectra["E1"], cfg)
    tic("e1_analytic", t)
    t = time.time()
    fam = family_checks(cfg, ens)
    tic("family_checks", t)
    alpha_jp = local_index(cfg, spectra["E4-JP"])
    alpha_kp = local_index(cfg, spectra["E4-KP"])
    sel = [0, 10, 20, 30, 40]
    jp_info = {"t_Myr": E.AGEING.t_myr, "gamma_c_JP": E.AGEING.gamma_c_jp,
               "nu_c_JP=a1<B_perp>gamma_c^2_Hz": K.a1 * cfg.mean_bperp * E.AGEING.gamma_c_jp**2,
               "local_index_JP_at_0.1_0.316_1_3.16_10GHz": alpha_jp[sel].tolist(),
               "local_index_KP_at_same": alpha_kp[sel].tolist(),
               "injection_index_(p-1)/2": 0.8}
    summary = {}
    for name, res in results.items():
        key = {"E4-KP": "forward_effective_diagonal_iso_basis", "E5": "forward_joint_azimuth",
               "E1M": "forward_maxwell_basis"}.get(name, "forward")
        summary[name] = {f"N{N}": res[key][f"N{N}"]["max_resid_over_I_IQU"] for N in NS}
    summary["E4-KP_marginal(L=0)"] = {f"N{N}": results["E4-KP"]["forward_marginal_moments_iso_basis"][f"N{N}"][
        "max_resid_over_I_IQU"] for N in NS}
    summary["E5_separable_paper_factor"] = {f"N{N}": results["E5"]["forward_separable_paper_factor_0.4e^{0.5i}"][
        f"N{N}"]["max_resid_over_I_IQU"] for N in NS}
    summary["E1M_U46_basis"] = {f"N{N}": results["E1M"]["forward_U46_basis_(misspecified_field)"][f"N{N}"][
        "max_resid_over_I_IQU"] for N in NS}
    out = {
        "description": __doc__,
        "summary_max_resid_over_I_IQU": summary,
        "ensembles": results,
        "direct_spectra_IQU_per_unit_column": {k: v for k, v in spectra.items()},
        "e1_analytic": e1a,
        "family_checks": fam,
        "ageing": jp_info,
        "config": cfg.describe(),
        "envelope_definition": "env_X,j(N) = sum_b sup_{x_b in 65-point grid} |d^{N+1}_x Kbar_X,j| /(N+1)! "
                               "* int_bin |x_b|^{N+1} dN; measured (grid sup), not a rigorous bound",
        "timings_s": TIMINGS,
    }
    out["provenance"] = provenance(__file__, T0)
    write_json(HERE / "forward_results.json", out)
    np.savez(HERE / "forward_spectra.npz", centres=cfg.centres, **{k.replace("-", "_"): v for k, v in spectra.items()})
    print("E1 analytic", {k: v for k, v in e1a.items() if k.startswith("max")})
    print("timings", TIMINGS)


if __name__ == "__main__":
    main()
