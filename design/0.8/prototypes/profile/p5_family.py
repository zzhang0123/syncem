"""Profile 5: family fit per spectrum. Bounded least squares with family moments by a 32-node rule.

Family gamma^-p exp(-gamma / gamma_c) with amplitude, theta = (ln A, p, ln gamma_c) (_family.py).
Basis: the study's E1 separable columns at N = 2 on the 21 fitted channels (63 whitened rows).
Truth: the N = 4 basis times the family moments at theta = (0, 2.6, ln 1e5). Noise 1 per cent of I_j.

Variants per spectrum:
(a) scipy least_squares trf, NumPy residual, 2-point finite-difference Jacobian,
(b) the same with a jitted JAX residual and a jitted jacfwd Jacobian,
(c) all-JAX: Levenberg-Marquardt with a fixed iteration count and box clipping, vmapped over pixels.

Run: python p5_family.py   (about 1 minute)
"""
from __future__ import annotations

import time

import numpy as np
from scipy.optimize import least_squares

import _common as U
import _family as FAM

import syncem  # noqa: F401
import jax
import jax.numpy as jnp

import basis as SB
from config import PAPER

LO = np.array([-5.0, 1.5, np.log(1e3)])
HI = np.array([5.0, 4.0, np.log(1e8)])
X0 = np.array([0.3, 2.3, np.log(1e6)])
THETA = np.array([0.0, 2.6, np.log(1e5)])


def setup():
    tab = SB.get_table(PAPER)
    fit = PAPER.fit_idx
    DI4, DQ4 = SB.derivative_spectra(PAPER, 4, table=tab)
    S = np.einsum("jsk,k->js", SB.separable_columns(PAPER, DI4, DQ4), FAM.moments_np(THETA, 4))[fit]
    DI, DQ = SB.derivative_spectra(PAPER, 2, table=tab)
    C = SB.separable_columns(PAPER, DI, DQ)[fit]
    sig = 0.01 * S[:, 0:1] * np.ones((1, 3))
    A = (C / sig[:, :, None]).reshape(-1, C.shape[-1])
    return A, (S / sig).ravel()


def main() -> None:
    A, s_w = setup()
    rng = np.random.default_rng(3)
    D = s_w[None, :] + rng.standard_normal((40, A.shape[0]))
    res = {"rows_cols": list(A.shape)}

    def r_np(th, d):
        return A @ FAM.moments_np(th, 2) - d

    t0 = time.perf_counter()
    sol_a = [least_squares(r_np, X0, args=(d,), bounds=(LO, HI), method="trf", x_scale="jac") for d in D]
    res["a_numpy_fd_ms"] = 1e3 * (time.perf_counter() - t0) / len(D)
    res["a_nfev_median"] = float(np.median([s.nfev for s in sol_a]))
    print(res, flush=True)

    Aj = jnp.asarray(A)
    r_j = jax.jit(lambda th, d: Aj @ FAM.moments_jax(th, 2) - d)
    J_j = jax.jit(jax.jacfwd(lambda th, d: Aj @ FAM.moments_jax(th, 2) - d))
    r_j(X0, D[0]).block_until_ready()
    J_j(X0, D[0]).block_until_ready()
    t0 = time.perf_counter()
    sol_b = [least_squares(lambda th, d=d: np.asarray(r_j(th, d)), X0, jac=lambda th, d=d: np.asarray(J_j(th, d)),
                           bounds=(LO, HI), method="trf", x_scale="jac") for d in D]
    res["b_jax_jit_ms"] = 1e3 * (time.perf_counter() - t0) / len(D)
    res["b_nfev_median"] = float(np.median([s.nfev for s in sol_b]))
    _, sec, _ = U.timed(lambda: [r_j(X0, D[0]).block_until_ready() for _ in range(200)])
    res["jit_residual_call_us"] = 1e6 * sec / 200
    _, sec, _ = U.timed(lambda: [r_np(X0, D[0]) for _ in range(200)])
    res["numpy_residual_call_us"] = 1e6 * sec / 200
    res["ab_theta_max_diff"] = float(np.max([np.abs(a.x - b.x).max() for a, b in zip(sol_a, sol_b)]))
    print(res, flush=True)

    lo, hi = jnp.asarray(LO), jnp.asarray(HI)

    def lm(d, iters=40):
        def body(_, s):
            th, lam = s
            r = Aj @ FAM.moments_jax(th, 2) - d
            Jm = jax.jacfwd(lambda t: Aj @ FAM.moments_jax(t, 2))(th)
            g, H = Jm.T @ r, Jm.T @ Jm
            step = jnp.linalg.solve(H + lam * jnp.diag(jnp.diag(H)), -g)
            tn = jnp.clip(th + step, lo, hi)
            rn = Aj @ FAM.moments_jax(tn, 2) - d
            ok = rn @ rn < r @ r
            return jnp.where(ok, tn, th), jnp.where(ok, lam * 0.3, lam * 10.0)

        return jax.lax.fori_loop(0, iters, body, (jnp.asarray(X0), 1e-2))[0]

    run = jax.jit(jax.vmap(lm))
    Dbig = s_w[None, :] + rng.standard_normal((10_000, A.shape[0]))
    run(Dbig[:10]).block_until_ready()
    th_small = np.asarray(run(jnp.asarray(D)))
    run(Dbig).block_until_ready()
    _, sec, ratio = U.timed(run, Dbig, repeat=2)
    res["c_jax_lm_vmap_us_per_spectrum"] = 1e6 * sec / len(Dbig)
    res["c_cpu_over_wall"] = ratio
    res["c_vs_b_theta_max_diff"] = float(np.max(np.abs(th_small - np.stack([b.x for b in sol_b]))))
    chi_b = np.array([2 * b.cost for b in sol_b])
    chi_c = np.array([np.sum((A @ FAM.moments_np(t, 2) - d) ** 2) for t, d in zip(th_small, D)])
    res["c_minus_b_chi2_max"] = float(np.max(chi_c - chi_b))
    res["projected_1e6_s"] = {"a": res["a_numpy_fd_ms"] * 1e3, "b": res["b_jax_jit_ms"] * 1e3,
                              "a_pool28_ideal": res["a_numpy_fd_ms"] * 1e3 / 28,
                              "c": res["c_jax_lm_vmap_us_per_spectrum"]}
    print(res, flush=True)
    print(U.save("p5_family", res))


if __name__ == "__main__":
    main()
