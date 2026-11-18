"""Profile 1: the S1 field-average table (study lib/basis.py KernelTable).

Measures
* the NumPy build against the number of field nodes, split into quadrature (``direct``) and splines,
* where the NumPy time goes (cProfile),
* a jitted JAX port of ``direct`` (same quadrature, vectorised over knots and field nodes),
* an algorithmic alternative: tabulate the single-field primitive once in ln x (S0), then form the
  field average for any node set by spline lookups at ln y - ln B_q (S1). Its accuracy is checked
  against the study quadrature at every knot.

Run: python p1_s1_table.py   (about 1 to 2 minutes)
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np

import _common as U

import syncem  # noqa: F401  (float64)
import jax
import jax.numpy as jnp
from scipy.interpolate import CubicSpline
from scipy.special import logsumexp

import basis as SB
import kernels as SK
from config import PAPER, gl_interval

NB_LIST = (1, 2, 4, 8)
ORDER = SB.TABLE_ORDER


def field_nodes(n_b: int):
    b, w = gl_interval(n_b, 4.0, 6.0)
    return b, w / 2.0


# ----------------------------------------------------------------------------- NumPy study build
def numpy_build(n_b: int) -> dict:
    b, w = field_nodes(n_b)
    tab = SB.KernelTable.__new__(SB.KernelTable)
    tab.max_order, tab.b_nodes, tab.b_weights = ORDER, b, w
    n = int(round((SB.TABLE_LNY[1] - SB.TABLE_LNY[0]) / SB.TABLE_STEP)) + 1
    tab.lny = np.linspace(*SB.TABLE_LNY, n)
    t0 = time.perf_counter()
    raw = {W: tab.direct(np.exp(tab.lny), W) for W in ("F", "G")}
    t1 = time.perf_counter()
    for W, (ln_t, tau) in raw.items():
        [CubicSpline(tab.lny, ln_t).c] + [CubicSpline(tab.lny, tau[:, r]).c for r in range(1, ORDER + 1)]
    t2 = time.perf_counter()
    return {"n_b": n_b, "n_knots": n, "direct_s": t1 - t0, "spline_s": t2 - t1, "total_s": t2 - t0,
            "raw": raw, "lny": tab.lny}


def profile_numpy(n_b: int) -> str:
    pr = cProfile.Profile()
    pr.enable()
    numpy_build(n_b)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
    return s.getvalue()


# ----------------------------------------------------------------------------- JAX port of direct
_POLYS = SK.derivative_polynomials(ORDER)
_PCOEF = np.zeros((ORDER + 1, ORDER + 1))
for _r, _p in enumerate(_POLYS):
    _PCOEF[_r, : _p.size] = _p
_Q, _WQ = np.polynomial.legendre.leggauss(SK.DEFAULT_NODES)


def _jax_scaled(x, which):
    """(ln E0, f (.., ORDER+1)) for x of any shape, as kernels.scaled_log_derivatives."""
    q, wq = jnp.asarray(_Q), jnp.asarray(_WQ)
    xv = x[..., None]
    umax = jnp.arccosh(1.0 + SK.DEFAULT_CUTOFF / xv)
    u = 0.5 * (q + 1.0) * umax
    z = xv * jnp.cosh(u)
    wfun = jnp.cosh(5.0 * u / 3.0) / jnp.cosh(u) if which == "F" else jnp.cosh(2.0 * u / 3.0)
    base = jnp.exp(-2.0 * xv * jnp.sinh(0.5 * u) ** 2) * wfun * wq * (0.5 * umax)
    e0 = base.sum(-1)
    c = jnp.asarray(_PCOEF)
    poly = jnp.zeros(z.shape + (ORDER + 1,)) + c[:, ORDER]
    for k in range(ORDER - 1, -1, -1):
        poly = poly * z[..., None] + c[:, k]
    f = jnp.einsum("...q,...qr->...r", base, poly) / e0[..., None]
    return jnp.log(e0), f


def make_jax_direct(which: str, chunk: int = 512):
    @jax.jit
    def direct(lny, b, w):
        def one(lc):
            y = jnp.exp(lc)
            x = y[:, None] / b[None, :]
            ln_e0, f = _jax_scaled(x, which)
            terms = jnp.log(w)[None, :] - x + ln_e0
            lse = jax.nn.logsumexp(terms, axis=1)
            soft = jnp.exp(terms - lse[:, None])
            return lc + lse, jnp.einsum("yq,yqr->yr", soft, f)

        n = lny.shape[0]
        pad = (-n) % chunk
        lp = jnp.concatenate([lny, jnp.full(pad, lny[-1])]).reshape(-1, chunk)
        a, t = jax.lax.map(one, lp)
        return a.reshape(-1)[:n], t.reshape(-1, ORDER + 1)[:n]

    return direct


# ----------------------------------------------------------------------------- S0 then S1
def s0_table(lnx_lo: float, lnx_hi: float, step: float = SB.TABLE_STEP) -> dict:
    """Splines in ln x of ln E0_W and f_{W,r}, r = 1..ORDER, for W = F, G (one quadrature pass)."""
    n = int(np.ceil((lnx_hi - lnx_lo) / step)) + 1
    lnx = lnx_lo + step * np.arange(n)
    out = {"lnx": lnx}
    for W in ("F", "G"):
        ln_e0, f = SK.scaled_log_derivatives(np.exp(lnx), ORDER, W)
        out[W] = [CubicSpline(lnx, ln_e0)] + [CubicSpline(lnx, f[:, r]) for r in range(1, ORDER + 1)]
    return out


def s1_from_s0(s0: dict, lny: np.ndarray, b, w, which: str):
    """(ln T, tau) at ln y by lookups of S0 at ln y - ln B_q, as KernelTable.direct."""
    lnx = lny[:, None] - np.log(b)[None, :]
    x = np.exp(lnx)
    sp = s0[which]
    ln_e0 = sp[0](lnx)
    f = np.stack([np.ones_like(x)] + [sp[r](lnx) for r in range(1, ORDER + 1)], axis=-1)
    terms = np.log(w)[None, :] - x + ln_e0
    lse = logsumexp(terms, axis=1)
    soft = np.exp(terms - lse[:, None])
    return lny + lse, np.einsum("yq,yqr->yr", soft, f)


def table_error(ln_t, tau, ln_ref, tau_ref) -> dict:
    """Relative error of T^(r) on the knots where T is representable, scaled as spline_check."""
    keep = ln_ref > -650.0
    out = {"lnT_max_abs": float(np.max(np.abs(ln_t[keep] - ln_ref[keep])))}
    for r in range(1, ORDER + 1):
        scale = np.maximum(1.0, np.abs(tau_ref[keep, r]))
        out[f"r{r}"] = float(np.max(np.abs(tau[keep, r] - tau_ref[keep, r]) / scale))
    return out


def main() -> None:
    res = {"numpy_build": [], "jax_direct": [], "s0_s1": {}}
    ref = {}
    for n_b in NB_LIST:
        r = numpy_build(n_b)
        ref[n_b] = r
        res["numpy_build"].append({k: r[k] for k in ("n_b", "n_knots", "direct_s", "spline_s", "total_s")})
        print("numpy", res["numpy_build"][-1], flush=True)
    res["numpy_profile_n_b_2"] = profile_numpy(2)
    print(res["numpy_profile_n_b_2"], flush=True)

    lny = ref[1]["lny"]
    for W in ("F", "G"):
        fn = make_jax_direct(W)
        for n_b in NB_LIST:
            b, w = field_nodes(n_b)
            args = (jnp.asarray(lny), jnp.asarray(b), jnp.asarray(w))
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            first = time.perf_counter() - t0
            (a, t), run, ratio = U.timed(fn, *args, repeat=2)
            ln_ref, tau_ref = ref[n_b]["raw"][W]
            err = table_error(np.asarray(a), np.asarray(t), ln_ref, tau_ref)
            row = {"which": W, "n_b": n_b, "first_call_s": first, "run_s": run, "cpu_over_wall": ratio,
                   "max_err_vs_numpy": max(err.values())}
            res["jax_direct"].append(row)
            print("jax", row, flush=True)

    # S0 once over the ln x range the uniform 4-6 uG field needs, then S1 per node set
    b8, _ = field_nodes(8)
    lo = lny[0] - np.log(6.0) - 0.01
    hi = lny[-1] - np.log(4.0) + 0.01
    t0 = time.perf_counter()
    s0 = s0_table(lo, hi)
    res["s0_s1"]["s0_build_s"] = time.perf_counter() - t0
    res["s0_s1"]["s0_knots"] = int(s0["lnx"].size)
    for n_b in NB_LIST:
        b, w = field_nodes(n_b)
        row = {"n_b": n_b}
        for W in ("F", "G"):
            (a, t), sec, _ = U.timed(s1_from_s0, s0, lny, b, w, W, repeat=2)
            row[f"{W}_s1_s"] = sec
            row[f"{W}_err"] = table_error(a, t, *ref[n_b]["raw"][W])
        res["s0_s1"][f"n_b_{n_b}"] = row
        print("s0/s1", row, flush=True)
    print(U.save("p1_s1_table", res))


if __name__ == "__main__":
    main()
