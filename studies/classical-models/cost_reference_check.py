"""Independent reference for the accuracy column of the kernel-table cost table (cost.py).

cost.py measures each route against route (i), ``direct_from_measure`` on ``E.measure_1d``
with 24 Gauss-Legendre nodes per ln-gamma panel and panels split at the 19 bin edges.  For E2
(no kinks, support [gamma_lo, gamma_hi]) these are the same 456 nodes and weights as the
456-energy table (``fixed_gamma_nodes(cfg, 24)``), so that comparison is circular.  For E6 the
reference and the table share the 16 Gauss-Legendre field nodes on [4, 6] muG.

Here the reference is recomputed on an independent quadrature:
  * energy: 96 Gauss-Legendre nodes per panel (4x), panels offset by half a bin (breakpoints at
    the bin centres instead of the bin edges). For JP the panel that ends at gamma_c keeps
    Gauss-Jacobi with weight (u_c - u)^(p-2), also with 96 nodes.
  * field (E6): 64 Gauss-Legendre nodes on [4, 6] muG (4x, none coincides with the 16 nodes).
A second reference with 8x / offset nodes on a subset of parameter vectors checks that the new
reference itself has converged.  The kernel (B-averaged spline table, channel/eta/screen
averages) is the same in all routes, as in cost.py: this checks the quadrature over the
electron (and field) distribution only.

Errors are max over channels and parameter vectors of |S - S_ref| / I_ref (cost.rel_err).
Writes cost_reference_check_results.json.  Run: python cost_reference_check.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
from numpy.polynomial.legendre import leggauss  # noqa: E402
from scipy.special import roots_jacobi  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))
import basis as B  # noqa: E402
import cost as C  # noqa: E402
import ensembles as E  # noqa: E402
from config import PAPER, gl_interval  # noqa: E402
from provenance import provenance, write_json  # noqa: E402

T0 = time.time()
N_REF = 96          # 4 x the 24 nodes of cost.py
N_REF2 = 192        # 8 x, convergence check of the new reference
NB_REF, NB_REF2 = 64, 96  # field nodes for E6 (cost.py: 16)
N_CHECK = 6         # parameter vectors used for the 8x convergence check (1-D cases)


def offset_panels(cfg, lo, hi):
    """ln-gamma panels on [lo, hi] with breakpoints at the bin centres (half-bin offset)."""
    ulo, uhi = np.log(lo), np.log(hi)
    mids = np.log(cfg.gamma_b)
    pts = np.unique(np.r_[ulo, uhi, mids[(mids > ulo) & (mids < uhi)]])
    return list(zip(pts[:-1], pts[1:]))


def ref_measure(cfg, density, lo, hi, n, cut=None, alpha=None, regular=None):
    """(gamma, w) of N(gamma) dgamma on offset panels, with Gauss-Jacobi on the panel ending at cut."""
    t, wt = leggauss(n)
    gs, ws = [], []
    for ua, ub in offset_panels(cfg, lo, hi):
        if cut is not None and np.isclose(ub, np.log(cut), rtol=0, atol=1e-13):
            tj, wj = roots_jacobi(n, alpha, 0.0)
            g = np.exp(ua + 0.5 * (ub - ua) * (tj + 1.0))
            gs.append(g)
            ws.append((0.5 * (ub - ua)) ** (alpha + 1.0) * wj * regular(g))
        else:
            g = np.exp(0.5 * (ub - ua) * t + 0.5 * (ub + ua))
            gs.append(g)
            ws.append(0.5 * (ub - ua) * wt * g * density(g))
    return np.concatenate(gs), np.concatenate(ws)


def direct(cfg, gw, table):
    g, w = gw
    kI, kQ = B.kbar(cfg, g, order=0, table=table)
    return B.to_stokes(w @ kI[0], w @ kQ[0], cfg.azimuth_factor)


def ref_1d(cfg, table, name, th, n):
    if name == "E2":
        return direct(cfg, ref_measure(cfg, C.e2_density(th), cfg.gamma_lo, cfg.gamma_hi, n), table)
    gc = float(np.exp(th[2]))
    return direct(cfg, ref_measure(cfg, C.jp_density(th), cfg.gamma_lo, gc, n, cut=gc, alpha=th[1] - 2.0,
                                   regular=C.jp_regular(th)), table)


def one_d(name):
    cfg = PAPER
    table = B.get_table(cfg)
    C2 = B.build_basis(cfg, 2, table=table).reshape(-1, 3 * cfg.n_bins)
    thetas = C.theta_list(name, C.N_SLOW, seed=7 if name == "E2" else 8)
    t = time.time()
    old = [C.route_i(cfg, table, name)(th) for th in thetas]
    new = [ref_1d(cfg, table, name, th, N_REF) for th in thetas]
    new2 = [ref_1d(cfg, table, name, th, N_REF2) for th in thetas[:N_CHECK]]
    out = {"n_theta": len(thetas), "ref_nodes_per_eval": int(len(ref_measure(
        cfg, C.e2_density(thetas[0]), cfg.gamma_lo, cfg.gamma_hi, N_REF)[0])) if name == "E2" else None,
        "new_ref_8x_vs_4x_max_rel": max(C.rel_err(a, b) for a, b in zip(new[:N_CHECK], new2)),
        "old_ref_vs_new_ref_max_rel": max(C.rel_err(a, b) for a, b in zip(old, new))}
    rows = {}
    for n_per_bin in (24, 8):
        g, wq = C.fixed_gamma_nodes(cfg, n_per_bin)
        f = C.route_ii(C.build_table(cfg, table, g), g, wq, name)
        rows[f"table_{len(g)}"] = {
            "vs_old_ref": max(C.rel_err(f(th), r) for th, r in zip(thetas, old)),
            "vs_new_ref": max(C.rel_err(f(th), r) for th, r in zip(thetas, new))}
    f3 = C.route_iii(cfg, C2, name)
    rows[f"basis_{3 * cfg.n_bins}"] = {"vs_old_ref": max(C.rel_err(f3(th), r) for th, r in zip(thetas, old)),
                        "vs_new_ref": max(C.rel_err(f3(th), r) for th, r in zip(thetas, new))}
    out["max_rel_err"] = rows
    if name == "E4-JP":
        out["gamma_c_max_over_thetas"] = float(max(np.exp(th[2]) for th in thetas))
        out["gamma_hi_table"] = float(cfg.gamma_hi)
    out["wall_s"] = time.time() - t
    return name, out


def e6():
    """Joint (gamma, B) JP case of cost.joint_case, rebuilt with the same nodes and thetas."""
    cfg = PAPER
    t = time.time()
    p, B0, DB = 2.6, C.B0, C.DB
    bn, bw = gl_interval(16, 4.0, 6.0)
    bw = bw / 2.0
    tab0 = C.fixed_b_table(B0)
    tabs = [C.fixed_b_table(bv) for bv in bn]

    def moms(th_p, th_t, A=1.0):
        gck = np.array([C.gamma_c_local(bv, th_t) for bv in bn])
        th = np.c_[np.full(16, A), np.full(16, th_p), np.log(gck)]
        return E.jp_moments(cfg, th, 2)

    a_k = moms(p, 19.0)
    A_unit = 1.0 / float(np.sum(bw[:, None] * a_k[:, :, 0] * cfg.w_b[None, :]))
    th0 = (A_unit, p, 19.0)
    rng = np.random.default_rng(9)
    thetas = [th0] + [(A_unit * (1 + 0.05 * rng.normal()), p + 0.05 * rng.normal(), 19.0 * (1 + 0.05 * rng.normal()))
                      for _ in range(C.N_SLOW - 1)]
    thetas = thetas[:5]  # cost.py: n_direct = 5

    def old_direct(th):
        A, pp, tm = th
        tot = np.zeros((cfg.n_channels, 3))
        for bv, wv, tb in zip(bn, bw, tabs):
            tot += wv * E.direct_from_measure(cfg, C.jp_measure(cfg, (A, pp, np.log(C.gamma_c_local(bv, tm)))), table=tb)
        return tot

    def new_direct(th, nB, n):
        A, pp, tm = th
        bnr, bwr = gl_interval(nB, 4.0, 6.0)
        tot = np.zeros((cfg.n_channels, 3))
        for bv, wv in zip(bnr, bwr / 2.0):
            gc = float(C.gamma_c_local(bv, tm))
            thb = (A, pp, np.log(gc))
            gw = ref_measure(cfg, C.jp_density(thb), cfg.gamma_lo, gc, n, cut=gc, alpha=pp - 2.0,
                             regular=C.jp_regular(thb))
            tot += wv * direct(cfg, gw, C.fixed_b_table(bv))
        return tot

    old = [old_direct(th) for th in thetas]
    new = [new_direct(th, NB_REF, N_REF) for th in thetas]
    new2 = new_direct(thetas[0], NB_REF2, N_REF2)

    # table route (cost.joint_case)
    g, wq = C.fixed_gamma_nodes(cfg, 24)
    T = np.concatenate([C.build_table(cfg, tb, g) for tb in tabs], axis=1)
    gg, ww, bb = np.tile(g, 16), np.concatenate([wv * wq for wv in bw]), np.repeat(bn, len(g))

    def table_eval(th):
        A, pp, tm = th
        gc = C.gamma_c_local(bb, tm)
        dens = (A / C.GAMMA0) * (gg / C.GAMMA0) ** (-pp) * np.clip(1 - gg / gc, 0, None) ** (pp - 2)
        return (T @ (ww * dens)).reshape(-1, 3)

    C72r = C.joint_columns(cfg, tab0).reshape(-1, 6 * cfg.n_bins)
    xb = (bn - B0) / DB

    def moment_eval(th):
        A, pp, tm = th
        a = moms(pp, tm, A)
        m = np.zeros((cfg.n_bins, 6))
        for j, (r, sB) in enumerate(C.MONOMIALS):
            m[:, j] = np.sum(bw[:, None] * xb[:, None] ** sB * a[:, :, r], axis=0)
        return (C72r @ m.ravel()).reshape(-1, 3)

    out = {"n_theta": len(thetas), "ref_B_nodes": NB_REF, "ref_gamma_nodes_per_panel": N_REF,
           "new_ref_8x_vs_4x_max_rel_theta0": C.rel_err(new[0], new2),
           "old_ref_vs_new_ref_max_rel": max(C.rel_err(a, b) for a, b in zip(old, new)),
           "max_rel_err": {
               f"table_{T.shape[1]}": {"vs_old_ref": max(C.rel_err(table_eval(th), r) for th, r in zip(thetas, old)),
                              "vs_new_ref": max(C.rel_err(table_eval(th), r) for th, r in zip(thetas, new))},
               f"joint_basis_{6 * cfg.n_bins}": {"vs_old_ref": max(C.rel_err(moment_eval(th), r) for th, r in zip(thetas, old)),
                                   "vs_new_ref": max(C.rel_err(moment_eval(th), r) for th, r in zip(thetas, new))}},
           "wall_s": time.time() - t}
    return "E6", out


def _run(job):
    return one_d(job) if job != "E6" else e6()


def main():
    with ProcessPoolExecutor(max_workers=3) as ex:
        res = dict(ex.map(_run, ["E2", "E4-JP", "E6"]))
    res["description"] = __doc__
    res["provenance"] = provenance(__file__, T0, inputs=(HERE / "cost.py",))
    write_json(HERE / "cost_reference_check_results.json", res)
    for k in ("E2", "E4-JP", "E6"):
        print(k, {r: {kk: f"{v:.3g}" for kk, v in d.items()} for r, d in res[k]["max_rel_err"].items()},
              "old-vs-new ref", f"{res[k]['old_ref_vs_new_ref_max_rel']:.3g}")
    print("done", time.time() - T0)


if __name__ == "__main__":
    main()
