"""Core library checks: kernels, random-field kernel, spline tables, stencil, quadrature.

Writes core_checks_results.json.  Run: python core_checks.py
"""
from __future__ import annotations

import os

os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import sys
import time
from dataclasses import replace
from pathlib import Path

import mpmath as mp
import numpy as np
from scipy import integrate

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import basis as B  # noqa: E402
import kernels as K  # noqa: E402
from config import PAPER, PAPER_MAXWELL  # noqa: E402
from provenance import provenance, write_json  # noqa: E402

T0 = time.time()
mp.mp.dps = 40
XS = [1e-4, 1e-2, 0.1, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0]
ORDER = 6


def mp_derivs(x, which):
    """(-d/dt)^r W(e^t) at t = ln x, r = 0..ORDER, by mpmath (independent of the cosh integral)."""
    t0 = mp.log(mp.mpf(x))
    out = []
    if which == "G":
        f = lambda t: mp.exp(t) * mp.besselk(mp.mpf(2) / 3, mp.exp(t))  # noqa: E731
        for r in range(ORDER + 1):
            out.append((-1) ** r * mp.diff(f, t0, r))
        return [float(v) for v in out]
    xx = mp.exp(t0)
    F0 = xx * mp.quad(lambda s: mp.besselk(mp.mpf(5) / 3, s), [xx, xx + 1, xx + 10, mp.inf])
    h = lambda t: mp.exp(2 * t) * mp.besselk(mp.mpf(5) / 3, mp.exp(t))  # noqa: E731
    hd = [(-1) ** m * mp.diff(h, t0, m) for m in range(ORDER)]  # (-d/dt)^m h
    Fr = [F0]
    for r in range(1, ORDER + 1):
        Fr.append(hd[r - 1] - Fr[r - 1])  # (-d/dt) F^{(r-1)} = (-d/dt)^{r-1} h - F^{(r-1)}
    return [float(v) for v in Fr]


def check_kernels():
    res = {}
    for which in ("F", "G"):
        ours = K.log_derivatives(np.array(XS), ORDER, which)
        rows = []
        worst = np.zeros(ORDER + 1)
        for i, x in enumerate(XS):
            ref = np.array(mp_derivs(x, which))
            scale = np.abs(ref[0]) * (1.0 + x) ** np.arange(ORDER + 1)
            err = np.abs(ours[i] - ref) / scale
            worst = np.maximum(worst, err)
            rows.append({"x": x, "err_over_W0(1+x)^r": err.tolist(),
                         "rel_err_own": (np.abs(ours[i] - ref) / np.abs(ref)).tolist()})
        res[which] = {"per_x": rows, "max_err_over_W0(1+x)^r_by_order": worst.tolist()}
    xg = np.geomspace(1e-6, 700.0, 400)
    res["G_vs_scipy_kve_max_rel"] = float(np.max(np.abs(K.G(xg) / K.G_bessel(xg) - 1)))
    return res


def check_random_field():
    xs = np.geomspace(1e-3, 10.0, 61)
    avg = []
    for x in xs:
        f = lambda th: np.sin(th) ** 2 * float(K.F(np.array(x / np.sin(th))))  # noqa: E731
        v, _ = integrate.quad(f, 1e-12, np.pi / 2, epsabs=0, epsrel=1e-13, limit=400)
        avg.append(v)  # (1/2) int_0^pi = int_0^{pi/2} by symmetry
    avg = np.array(avg)
    R = K.random_field_kernel(xs)
    rel = np.abs(R / avg - 1)
    return {"x_range": [1e-3, 10.0], "n_x": len(xs), "max_rel_diff": float(rel.max()),
            "x_at_max": float(xs[rel.argmax()]), "pass_1e-9": bool(rel.max() <= 1e-9)}


def check_mellin():
    out = {}
    for p in (2.2, 2.6, 3.0):
        m = (p - 3) / 2
        for which, ref in (("F", K.mellin_F(m)), ("G", K.mellin_G(m))):
            f = lambda lx: np.exp((m + 1) * lx) * float(  # noqa: E731
                K.log_derivatives(np.array(np.exp(lx)), 0, which)[..., 0])
            v, _ = integrate.quad(f, np.log(1e-14), np.log(800.0), epsabs=0, epsrel=1e-12, limit=800)
            out[f"{which}_p{p}"] = float(abs(v / ref - 1))
        out[f"Pi_p{p}"] = [float(K.mellin_G(m) / K.mellin_F(m)), float(K.powerlaw_pol_fraction(p))]
    return out


def _range_errors(tab, which):
    """Spline errors by y range at off-grid points (0.37 of a cell from the knots)."""
    lny = np.linspace(np.log(1e-5) + 1e-4, np.log(4000.0), 20001) + 0.37 * tab.dx
    ln_t, tau = tab.direct(np.exp(lny), which)
    ev = tab.evaluate(lny, which, tab.max_order)
    T = np.exp(ln_t)
    out = {}
    for lo, hi in ((1e-5, 1e-3), (1e-3, 1.0), (1.0, 30.0), (30.0, 300.0), (300.0, 4000.0)):
        m = (np.exp(lny) >= lo) & (np.exp(lny) < hi) & (ln_t > -650)
        row = {"lnT_max_abs": float(np.max(np.abs(np.log(ev[0][m]) - ln_t[m])))}
        for r in range(1, tab.max_order + 1):
            err = np.abs(ev[r][m] - T[m] * tau[m, r]) / (T[m] * np.maximum(1.0, np.abs(tau[m, r])))
            row[f"r{r}"] = float(err.max())
        out[f"y_{lo:g}_{hi:g}"] = row
    return out


def check_tables():
    out = {}
    for name, cfg in (("uniform_4_6", PAPER), ("maxwell_brms5", PAPER_MAXWELL)):
        tab = B.get_table(cfg)
        out[name] = {w: tab.spline_check(w) for w in ("F", "G")}
        out[name + "_by_y_range"] = {w: _range_errors(tab, w) for w in ("F", "G")}
    return out


def check_stencil():
    """Analytic ln-derivative columns vs the paper's 5-point stencil (and half step).

    Metrics per order r and component (I; complex Q-kernel incl. screen, no azimuth):
      bin_max: max |D_an - D_st| / max_j |D_an| (per bin), all channels;
      significant: same, restricted to channels with |D_{b,0}| >= 1e-3 max_j |D_{b,0}|;
      data_scale: max |w_b (D_an - D_st)| / I_ref_j with I_ref_j = sum_b w_b D^I_{j;b0}
                  (the column change in units of the reference intensity, unit moment).
    """
    DI, DQ = B.derivative_spectra(PAPER, 2)
    iref = np.sum(PAPER.w_b[:, None] * DI[:, 0], axis=0)  # (J,)
    sig = np.abs(DI[:, 0]) >= 1e-3 * np.max(np.abs(DI[:, 0]), axis=1, keepdims=True)
    res = {"definition": check_stencil.__doc__}
    for step in (0.04, 0.02):
        SI, SQ = B.stencil_derivative_spectra(PAPER, step=step)
        per = {}
        for r in range(3):
            for name, a, s in (("I", DI[:, r], SI[:, r]), ("Qc", DQ[:, r], SQ[:, r])):
                d = np.abs(a - s)
                scale = np.max(np.abs(a), axis=1, keepdims=True)
                per[f"{name}_r{r}_bin_max"] = float(np.max(d / scale))
                per[f"{name}_r{r}_significant"] = float(np.max((d / scale)[sig]))
                per[f"{name}_r{r}_data_scale"] = float(np.max(PAPER.w_b[:, None] * d / iref[None, :]))
                per[f"{name}_r{r}_per_bin_max"] = np.max(d / scale, axis=1).tolist()
        res[f"step_{step}"] = per
    r04, r02 = res["step_0.04"], res["step_0.02"]
    res["ratio_step0.04_over_step0.02_data_scale"] = {
        k: r04[k] / r02[k] for k in r04 if k.endswith("data_scale") and r02[k] > 0}
    return res


def check_quadrature():
    """Kbar at bin centres and stencil points: adopted vs doubled channel/B/eta nodes."""
    g = (PAPER.gamma_b[:, None] * np.exp(PAPER.h * np.array([-1.0, 0.0, 1.0]))[None, :]).ravel()
    ref = replace(PAPER, n_chan_nodes=800, n_b=32, n_eta=24)
    out = {}
    for label, cfg in (("adopted(400,16,12)", PAPER), ("chan200", replace(PAPER, n_chan_nodes=200)),
                       ("b8", replace(PAPER, n_b=8)), ("eta6", replace(PAPER, n_eta=6))):
        kI, kQ = B.kbar(cfg, g, order=2)
        rI, rQ = B.kbar(ref, g, order=2)
        scale = np.max(np.abs(rI[0]), axis=1, keepdims=True)
        out[label + "_vs_(800,32,24)"] = {
            f"r{r}_max_diff_over_gamma_max_I": [float(np.max(np.abs(kI[r] - rI[r]) / scale)),
                                                float(np.max(np.abs(kQ[r] - rQ[r]) / scale))]
            for r in range(3)}
    return out


def main():
    t = {}
    results = {}
    for name, fn in (("kernels_vs_mpmath", check_kernels), ("random_field_kernel", check_random_field),
                     ("mellin", check_mellin), ("spline_tables", check_tables),
                     ("stencil_vs_analytic_N<=2", check_stencil), ("kbar_quadrature", check_quadrature)):
        t0 = time.time()
        results[name] = fn()
        t[name] = time.time() - t0
        print(name, "done", round(t[name], 1), "s", flush=True)
    results["timings_s"] = t
    results["config"] = PAPER.describe()
    results["provenance"] = provenance(__file__, T0)
    write_json(HERE / "core_checks_results.json", results)


if __name__ == "__main__":
    main()
