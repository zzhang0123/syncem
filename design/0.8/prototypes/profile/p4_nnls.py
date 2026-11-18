"""Profile 4: non-negative cells. scipy.optimize.nnls on the 111 x 272 stacked system.

System as studies/classical-models/fits_free_positive.py: 37 channels on 1-145 GHz (111 rows of
whitened I, Q, U), cells 4-19 at N = 2 with 17 Chebyshev-Lobatto atoms per cell (272 columns),
columns scaled to unit norm. Truth: the N = 4 basis on all 19 cells times the moments of
gamma^-2.6 exp(-gamma / 1e5). Noise 1 per cent of I_j.

Measures: serial NNLS per spectrum, a 28-process pool, and a JAX alternative (FISTA projected
gradient under vmap with a fixed iteration count) compared in objective and prediction.

Run: python p4_nnls.py   (about 2 minutes)
"""
from __future__ import annotations

import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace

import numpy as np
from scipy.optimize import nnls

import _common as U
import _family as FAM

N_ATOMS, FIRST_CELL, N = 17, 3, 2
_A = None


def system():
    import basis as SB
    from config import PAPER

    cfg = replace(PAPER, nu_lo=1e9, nu_hi=1.45e11, n_channels=37)
    tab = SB.get_table(PAPER)
    DI4, DQ4 = SB.derivative_spectra(cfg, 4, table=tab)
    S = np.einsum("jsk,k->js", SB.separable_columns(cfg, DI4, DQ4),
                  FAM.moments_np((0.0, 2.6, np.log(1e5)), 4, cfg))
    DI, DQ = SB.derivative_spectra(cfg, N, table=tab)
    cols = SB.separable_columns(cfg, DI, DQ)[:, :, (N + 1) * FIRST_CELL:]
    z = np.cos(np.pi * np.arange(N_ATOMS) / (N_ATOMS - 1))
    M = np.stack([z**r for r in range(N + 1)])
    J, st, n = cols.shape
    atoms = np.einsum("jscm,ma->jsca", cols.reshape(J, st, n // (N + 1), N + 1), M).reshape(J, st, -1)
    sig = 0.01 * S[:, 0:1] * np.ones((1, 3))
    A = (atoms / sig[:, :, None]).reshape(J * 3, -1)
    scale = np.linalg.norm(A, axis=0)
    return A / scale, (S / sig).ravel()


def _init(A):
    global _A
    _A = A


def _solve(d):
    return nnls(_A, d, maxiter=50 * _A.shape[1])


def _solve_chunk(D):
    return [nnls(_A, d, maxiter=50 * _A.shape[1])[1] for d in D]


def fista(A, D, iters: int):
    import syncem  # noqa: F401
    import jax
    import jax.numpy as jnp

    L = float(np.linalg.norm(A, 2) ** 2)
    AtA, At = jnp.asarray(A.T @ A), jnp.asarray(A.T)

    @jax.jit
    def run(D):
        def one(d):
            Atd = At @ d

            def body(_, s):
                x, y, t = s
                xn = jnp.maximum(y - (AtA @ y - Atd) / L, 0.0)
                tn = 0.5 * (1 + jnp.sqrt(1 + 4 * t * t))
                return xn, xn + ((t - 1) / tn) * (xn - x), tn

            z = jnp.zeros(A.shape[1])
            return jax.lax.fori_loop(0, iters, body, (z, z, 1.0))[0]

        return jax.vmap(one)(D)

    return run


def main() -> None:
    A, s_w = system()
    rng = np.random.default_rng(2)
    res = {"shape": list(A.shape), "cond": float(np.linalg.cond(A))}
    D = s_w[None, :] + rng.standard_normal((200, A.shape[0]))
    _init(A)
    t0 = time.perf_counter()
    sols = [_solve(d) for d in D]
    res["serial_ms_per_spectrum"] = 1e3 * (time.perf_counter() - t0) / len(D)
    res["active_atoms_median"] = float(np.median([np.count_nonzero(w) for w, _ in sols]))
    print(res, flush=True)

    Dbig = s_w[None, :] + rng.standard_normal((5600, A.shape[0]))
    with ProcessPoolExecutor(28, initializer=_init, initargs=(A,)) as ex:
        list(ex.map(_solve_chunk, np.array_split(Dbig[:280], 28)))  # warm the workers
        t0 = time.perf_counter()
        list(ex.map(_solve_chunk, np.array_split(Dbig, 28 * 4)))
        res["pool28_ms_per_spectrum"] = 1e3 * (time.perf_counter() - t0) / len(Dbig)
    res["pool_speedup"] = res["serial_ms_per_spectrum"] / res["pool28_ms_per_spectrum"]
    res["projected_1e6_pool28_min"] = res["pool28_ms_per_spectrum"] * 1e6 / 1e3 / 60
    print(res, flush=True)

    import jax

    ref_rn2 = np.array([rn**2 for _, rn in sols])
    ref_w = np.stack([w for w, _ in sols])
    res["fista"] = []
    for iters in (1_000, 10_000, 50_000):
        run = fista(A, D, iters)
        jax.block_until_ready(run(D))
        W, sec, ratio = U.timed(run, D, repeat=1)
        W = np.asarray(W)
        rn2 = np.sum((W @ A.T - D) ** 2, axis=1)
        row = {"iters": iters, "ms_per_spectrum": 1e3 * sec / len(D), "cpu_over_wall": ratio,
               "chi2_excess_median": float(np.median(rn2 - ref_rn2)),
               "chi2_excess_max": float(np.max(rn2 - ref_rn2)),
               "pred_diff_max_over_sigma": float(np.max(np.abs((W - ref_w) @ A.T)))}
        res["fista"].append(row)
        print(row, flush=True)
    print(U.save("p4_nnls", res))


if __name__ == "__main__":
    main()
