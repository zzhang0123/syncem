"""Profile 3: free-moment fit (whitened SVD + UPRE rank) for batches of spectra.

Design: the study's E1 separable columns at N = 2 (57 columns) and the joint columns (171),
on the 21 fitted channels (63 rows). Noise 1 per cent of I_j.
* shared noise: one SVD, then per pixel one projection, a cumulative sum and an argmin,
* per-pixel noise: one SVD per pixel (sigma scaled per pixel by a random factor in [0.7, 1.3]),
  NumPy batched gesdd, the same on a thread pool, and JAX vmap(svd).
The UPRE rule follows studies/classical-models/fits.py LinearFit.scan (without the pair guard).

Run: python p3_free_fit.py   (about 2 minutes)
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import _common as U

import syncem  # noqa: F401
import jax
import jax.numpy as jnp

import basis as SB
from config import PAPER

RNG = np.random.default_rng(1)


def design(kind: str):
    DI, DQ = SB.derivative_spectra(PAPER, 2)
    C = SB.separable_columns(PAPER, DI, DQ) if kind == "sep" else SB.joint_columns(PAPER, DI, DQ)
    fit = PAPER.fit_idx
    a0 = np.zeros(C.shape[-1])
    a0[0:57:3] = 1.0  # unit order-0 energy moments
    S = np.einsum("jsk,k->js", C, a0)
    A = C[fit].reshape(-1, C.shape[-1])  # (63, n)
    s_true = S[fit].reshape(-1)
    sig = np.repeat(0.01 * S[fit, 0], 3)
    return A, s_true, sig


def data(s_true, sig, n_pix, per_pixel: bool):
    scale = RNG.uniform(0.7, 1.3, (n_pix, 1)) if per_pixel else np.ones((n_pix, 1))
    sg = sig[None, :] * scale
    return s_true[None, :] + sg * RNG.standard_normal((n_pix, sig.size)), sg


# ----------------------------------------------------------------------------- shared noise
def shared_numpy(A, sig, D):
    u, s, vt = np.linalg.svd(A / sig[:, None], full_matrices=False)
    Dw = D / sig[None, :]
    proj = Dw @ u
    chi2 = np.sum(Dw**2, 1)[:, None] - np.cumsum(proj**2, 1)
    R = np.arange(1, s.size + 1)
    Rs = R[np.argmin(chi2 + 2 * R, 1)]
    mask = R[None, :] <= Rs[:, None]
    return (proj * mask / s) @ vt, Rs


@jax.jit
def shared_jax(A, sig, D):
    u, s, vt = jnp.linalg.svd(A / sig[:, None], full_matrices=False)
    Dw = D / sig[None, :]
    proj = Dw @ u
    chi2 = jnp.sum(Dw**2, 1)[:, None] - jnp.cumsum(proj**2, 1)
    R = jnp.arange(1, s.size + 1)
    Rs = R[jnp.argmin(chi2 + 2 * R, 1)]
    mask = R[None, :] <= Rs[:, None]
    return (proj * mask / s) @ vt, Rs


# ----------------------------------------------------------------------------- per-pixel noise
def perpix_numpy(A, D, SG):
    Aw = A[None] / SG[:, :, None]
    u, s, vt = np.linalg.svd(Aw, full_matrices=False)
    Dw = D / SG
    proj = np.einsum("pi,pik->pk", Dw, u)
    chi2 = np.sum(Dw**2, 1)[:, None] - np.cumsum(proj**2, 1)
    R = np.arange(1, s.shape[1] + 1)
    Rs = R[np.argmin(chi2 + 2 * R, 1)]
    mask = R[None, :] <= Rs[:, None]
    return np.einsum("pk,pkn->pn", proj * mask / s, vt), Rs


def perpix_threads(A, D, SG, workers=28):
    chunks = np.array_split(np.arange(D.shape[0]), workers)
    with ThreadPoolExecutor(workers) as ex:
        out = list(ex.map(lambda ix: perpix_numpy(A, D[ix], SG[ix]), chunks))
    return np.concatenate([o[0] for o in out]), np.concatenate([o[1] for o in out])


def _one(A, d, sg):
    u, s, vt = jnp.linalg.svd(A / sg[:, None], full_matrices=False)
    dw = d / sg
    proj = dw @ u
    chi2 = jnp.sum(dw**2) - jnp.cumsum(proj**2)
    R = jnp.arange(1, s.size + 1)
    Rs = R[jnp.argmin(chi2 + 2 * R)]
    return ((proj * (R <= Rs) / s) @ vt), Rs


perpix_jax = jax.jit(jax.vmap(_one, in_axes=(None, 0, 0)))


def main() -> None:
    res = {"shared": [], "perpix": []}
    for kind in ("sep", "joint"):
        A, s_true, sig = design(kind)
        m, n = A.shape
        for n_pix in (1_000, 10_000, 100_000):
            D, _ = data(s_true, sig, n_pix, False)
            (a_np, r_np), t_np, c_np = U.timed(shared_numpy, A, sig, D, repeat=3)
            jax.block_until_ready(shared_jax(A, sig, D))
            (a_j, r_j), t_j, c_j = U.timed(shared_jax, A, sig, D, repeat=3)
            row = {"design": f"{m}x{n}", "n_pix": n_pix, "numpy_us_per_pix": 1e6 * t_np / n_pix,
                   "numpy_cpu_over_wall": c_np, "jax_us_per_pix": 1e6 * t_j / n_pix, "jax_cpu_over_wall": c_j,
                   "rank_agree": float(np.mean(np.asarray(r_j) == r_np)),
                   "median_rank": float(np.median(r_np))}
            res["shared"].append(row)
            print("shared", row, flush=True)
        for n_pix in (1_000, 10_000):
            D, SG = data(s_true, sig, n_pix, True)
            (a_np, r_np), t_np, c_np = U.timed(perpix_numpy, A, D, SG, repeat=2)
            (a_th, r_th), t_th, c_th = U.timed(perpix_threads, A, D, SG, repeat=2)
            jax.block_until_ready(perpix_jax(A, D, SG))
            (a_j, r_j), t_j, c_j = U.timed(perpix_jax, A, D, SG, repeat=2)
            row = {"design": f"{m}x{n}", "n_pix": n_pix,
                   "numpy_us_per_pix": 1e6 * t_np / n_pix, "numpy_cpu_over_wall": c_np,
                   "numpy_28threads_us_per_pix": 1e6 * t_th / n_pix, "threads_cpu_over_wall": c_th,
                   "jax_vmap_us_per_pix": 1e6 * t_j / n_pix, "jax_cpu_over_wall": c_j,
                   "rank_agree_jax_numpy": float(np.mean(np.asarray(r_j) == r_np)),
                   "transient_bytes_per_pix_model": 8 * (m * n + m * min(m, n) + min(m, n) * n) * 1.5}
            res["perpix"].append(row)
            print("perpix", row, flush=True)
    print(U.save("p3_free_fit", res))


if __name__ == "__main__":
    main()
