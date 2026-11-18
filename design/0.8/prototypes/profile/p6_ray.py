"""Profile 6: sightlines. Forward prediction of one 300-step ray, and the cost model for 1e5 rays.

Each step k carries B_k [uG], eta_k, phi_k, depth_k [rad m^-2] and a power law A_k gamma^-p_k on
[300, 2.96e6]. The step emission is sum_q w_q n_k(gamma_q) [K_I, K_P exp(2i phi_k)] at the
step's (B, eta, depth), with the 41 paper channels (64 nodes) and no screen.

Routes
(a) direct with the prototype point kernel (syncem F, G by 128-node quadrature), jitted,
(b) direct with F and G tabulated once on a ln x grid (cubic Hermite in ln x, forward values only),
(c) the basis route: moments of the ray on a product table gamma (19 log cells, N = 2) x
    B (4 cells, order 1) x depth (8 cells, order 1), total degree 2, paired in JAX by
    segment_sum and contracted with precomputed columns (random here, same shapes),
(d) the prototype's NumPy pair (core.pair) on the same table for one ray,
(e) one jet cell at d = 3 (gamma, B, depth celled, eta averaged by 12 nodes), compile and run.

Run: python p6_ray.py   (about 3 minutes)
"""
from __future__ import annotations

import time

import numpy as np
from numpy.polynomial.legendre import leggauss

import _common as U

import core as C
import jax
import jax.numpy as jnp
from core import jet as Jt
from core import paper as P

import kernels as SK

N_STEPS = 300
RNG = np.random.default_rng(4)


def ray(n_per_cell: int):
    """Step values and the gamma rule (nodes, ln-gamma weights) with n_per_cell nodes per E1 cell."""
    edges = np.exp(np.linspace(np.log(P.GAMMA_LO), np.log(P.GAMMA_HI), 20))
    t, w = leggauss(n_per_cell)
    lg = np.concatenate([0.5 * (b - a) * t + 0.5 * (b + a) for a, b in zip(np.log(edges[:-1]), np.log(edges[1:]))])
    wl = np.concatenate([0.5 * (b - a) * w for a, b in zip(np.log(edges[:-1]), np.log(edges[1:]))])
    steps = {"B": RNG.uniform(4.2, 5.8, N_STEPS), "eta": RNG.uniform(0.0, 0.5, N_STEPS),
             "phi": RNG.uniform(0, np.pi, N_STEPS), "depth": RNG.uniform(-4.5, 4.5, N_STEPS),
             "p": RNG.uniform(2.4, 2.8, N_STEPS), "A": RNG.uniform(0.5, 1.5, N_STEPS)}
    return steps, np.exp(lg), wl


def point_batches(steps, g, wl):
    """Flattened points (n_steps * n_gamma) and their weights n_k(gamma_q) gamma_q w_q / n_steps."""
    ns, ng = N_STEPS, g.size
    pts = {"gamma": np.tile(g, ns), **{v: np.repeat(steps[v], ng) for v in ("B", "eta", "depth")}}
    wt = (steps["A"][:, None] * g[None, :] ** (1 - steps["p"][:, None]) * wl[None, :] / ns).ravel()
    return pts, wt, np.repeat(steps["phi"], ng)


def direct_fn(K, chunk=32):
    @jax.jit
    def run(pts, wt, phi):
        n = wt.shape[0]
        pad = (-n) % chunk

        def padv(x):
            return jnp.concatenate([x, jnp.full(pad, x[-1])]).reshape(-1, chunk)

        P_ = {v: padv(x) for v, x in pts.items()}
        W = jnp.concatenate([wt, jnp.zeros(pad)]).reshape(-1, chunk)
        E = padv(jnp.exp(2j * phi))

        def body(acc, xs):
            p, w, e = xs
            I, Pc = jax.vmap(lambda q: K(q))(p)
            return (acc[0] + w @ I, acc[1] + (w * e) @ Pc), None

        z = jnp.zeros(K.n_ch)
        return jax.lax.scan(body, (z, z.astype(complex)), (P_, W, E))[0]

    return run


class TabulatedKernel:
    """F and G from a cubic Hermite table in ln x (values and slopes from the study kernel)."""

    def __init__(self, base: C.ContinuumChannelKernel, step=0.01, lnx=(-32.0, 4.5)):
        self.base, self.n_ch = base, base.n_ch
        grid = np.arange(lnx[0], lnx[1] + step, step)
        x = np.exp(grid)
        tabs = {}
        for W in ("F", "G"):
            d = SK.log_derivatives(x, 1, W)  # W and (-x d/dx) W
            tabs[W] = (np.log(d[:, 0]), -d[:, 1] / d[:, 0])  # ln W and d ln W / d ln x
        self.x0, self.h, self.n = grid[0], step, grid.size
        self.tab = {W: tuple(jnp.asarray(a) for a in v) for W, v in tabs.items()}

    def _lnW(self, W, lnx):
        f, df = self.tab[W]
        s = jnp.clip((lnx - self.x0) / self.h, 0.0, self.n - 1.000001)
        i = jnp.floor(s).astype(int)
        t = s - i
        h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
        return h00 * f[i] + h10 * self.h * df[i] + h01 * f[i + 1] + h11 * self.h * df[i + 1]

    def __call__(self, p):
        b = self.base
        geo = b.base  # syncem ContinuumKernel
        positive, amplitude, scale = geo._geometry(p["gamma"], p["B"] * C.kernels.MUG, p["eta"])
        x = geo._argument(b.nu, positive, scale)
        lnx = jnp.log(x)
        hi = lnx > self.x0 + self.h * (self.n - 1)
        F = jnp.where(hi, 0.0, jnp.exp(self._lnW("F", lnx)))
        G = jnp.where(hi, 0.0, jnp.exp(self._lnW("G", lnx)))
        phase = jnp.exp(2j * b.lam2 * p["depth"])
        return jnp.sum(b.w * amplitude * F, -1), jnp.sum(b.w * (-amplitude) * G * phase, -1)


def product_table():
    return C.ProductTable.tensor({"gamma": C.Cells.log(P.GAMMA_LO, P.GAMMA_HI, 19, order=2),
                                  "B": C.Cells.linear(4.0, 6.0, 4, order=1),
                                  "depth": C.Cells.linear(-5.0, 5.0, 8, order=1)}, total=2)


def basis_route(tab, steps, g, wl, n_rays=1000):
    """Per-ray cost of pairing (segment_sum) and prediction with fixed columns."""
    pts, wt, phi = point_batches(steps, g, wl)
    cell = tab.locate(pts)
    exps = np.array(tab.exponents(0))  # same lower set in every cell
    ref, hw = tab.reference[cell], tab.half_width[cell]
    u = np.stack([np.log(pts["gamma"] / ref[:, 0]) / hw[:, 0], (pts["B"] - ref[:, 1]) / hw[:, 1],
                  (pts["depth"] - ref[:, 2]) / hw[:, 2]], 1)
    rows = (tab.offsets[cell][:, None] + np.arange(len(exps))[None, :]).ravel()
    colsI = jnp.asarray(RNG.standard_normal((41, tab.n_rows)))
    colsP = jnp.asarray(RNG.standard_normal((41, tab.n_rows)) + 1j)
    u_j, rows_j, exps_j = jnp.asarray(u), jnp.asarray(rows), jnp.asarray(exps)

    def one(w, ph):
        mono = jnp.prod(u_j[:, None, :] ** exps_j[None, :, :], -1) * w[:, None]  # (M, n_exp)
        m0 = jax.ops.segment_sum(mono.ravel(), rows_j, tab.n_rows)
        m2 = jax.ops.segment_sum((mono * jnp.exp(2j * ph)[:, None]).ravel(), rows_j, tab.n_rows)
        return colsI @ m0, colsP @ m2

    run = jax.jit(jax.vmap(one))
    W = jnp.asarray(wt[None, :] * RNG.uniform(0.9, 1.1, (n_rays, 1)))
    PH = jnp.asarray(np.broadcast_to(phi, W.shape))
    jax.block_until_ready(run(W[:2], PH[:2]))
    _, sec, ratio = U.timed(run, W, PH, repeat=2)
    return {"us_per_ray": 1e6 * sec / n_rays, "cpu_over_wall": ratio, "n_rows": int(tab.n_rows),
            "nodes_per_ray": int(wt.size)}


def jet_d3():
    e, we = leggauss(12)
    K = C.ContinuumChannelKernel(C.bump_channels(n_nodes=64))
    avg = C.AveragedKernel(K, [("eta", C.fixed(0.25 * (e + 1.0), we * 0.5))], outer_batch=1)
    full = P.e1_table(2)
    tab = C.ProductTable.tensor({"gamma": C.Cells.log(float(full.lo[9, 0]), float(full.hi[9, 0]), 1, order=2),
                                 "B": C.Cells.linear(4.0, 6.0, 1, order=1),
                                 "depth": C.Cells.linear(-5.0, 5.0, 1, order=1)}, total=2)
    fn = Jt._cell_routine(avg, tab, tab.signature(0))
    refs, hws = jnp.asarray(tab.reference), jnp.asarray(tab.half_width)
    t0 = time.perf_counter()
    comp = fn.lower(refs, hws).compile()
    t1 = time.perf_counter()
    _, run, ratio = U.timed(comp, refs, hws, repeat=2)
    prim = jax.jit(lambda g: avg({"gamma": g, "B": 5.0, "depth": 0.0}))
    jax.block_until_ready(prim(1e4))
    _, p_sec, _ = U.timed(prim, 1e4, repeat=3)
    return {"rows": int(tab.n_rows), "compile_s": t1 - t0, "run_s": run, "cpu_over_wall": ratio,
            "primal_s": p_sec, "primal_equivalents": run / p_sec}


def main() -> None:
    res = {}
    K = C.ContinuumChannelKernel(C.bump_channels(n_nodes=64))
    T = TabulatedKernel(K)
    for n_per, n_steps in ((6, 30), (6, 300), (24, 300)):
        steps, g, wl = ray(n_per)
        pts, wt, phi = point_batches(steps, g, wl)
        keep = slice(0, n_steps * g.size)  # the first n_steps steps
        pts, wt, phi = {v: x[keep] for v, x in pts.items()}, wt[keep], phi[keep]
        args = ({v: jnp.asarray(x) for v, x in pts.items()}, jnp.asarray(wt), jnp.asarray(phi))
        row = {"n_points": int(wt.size), "n_steps": n_steps}
        out = {}
        for name, ker in (("quad", K), ("table", T)):
            if name == "quad" and n_steps > 30:
                continue  # quadrature kernel timed on 30 steps and scaled by the point count
            fn = direct_fn(ker)
            t0 = time.perf_counter()
            jax.block_until_ready(fn(*args))
            first = time.perf_counter() - t0
            out[name], sec, ratio = U.timed(fn, *args, repeat=2)
            row[name] = {"first_s": first, "run_s": sec, "cpu_over_wall": ratio,
                         "us_per_point": 1e6 * sec / wt.size}
        if "quad" in out:
            I_q, P_q = (np.asarray(a) for a in out["quad"])
            I_t, P_t = (np.asarray(a) for a in out["table"])
            row["table_vs_quad_max_rel"] = max(float(np.max(np.abs(I_t - I_q) / I_q)),
                                               float(np.max(np.abs(P_t - P_q) / I_q)))
        res[f"direct_{n_per}_per_cell_{n_steps}_steps"] = row
        print(row, flush=True)

    tab = product_table()
    steps, g, wl = ray(6)
    res["basis_route"] = basis_route(tab, steps, g, wl)
    print(res["basis_route"], flush=True)
    pts, wt, phi = point_batches(steps, g, wl)
    meas = C.Measure({**pts, "phi": phi}, wt)
    w = C.reference_weights(tab)
    _, sec, _ = U.timed(C.pair, tab, meas, w, harmonic=True)
    res["core_pair_numpy_one_ray_s"] = sec
    print("core.pair", sec, flush=True)
    res["jet_d3_one_cell"] = jet_d3()
    print(res["jet_d3_one_cell"], flush=True)
    print(U.save("p6_ray", res))


if __name__ == "__main__":
    main()
