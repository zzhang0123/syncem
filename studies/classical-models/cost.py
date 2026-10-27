"""Cost of one parametric evaluation and of building the bases; continuum kernel.

All three routes use the same continuum kernel: the B-averaged spline table of F and G
(basis.KernelTable) and the eta-, channel- and screen-averaging of basis.channel_kernels.
One evaluation returns all 123 rows (41 channels x I, Q, U).
  (i)   recompute: build the energy quadrature of the parametric density and evaluate the
        averaged kernel at every node (ensembles.direct_from_measure);
  (ii)  cached fine gamma table: Kbar at fixed gamma nodes, precomputed once; one evaluation is
        the density at the nodes and a (123 x n_gamma) matrix-vector product;
  (iii) moment basis: family moments a(theta) (12 x 3) and a (123 x 36) matrix-vector product.
Families: E2 smoothly broken power law (no kink) and E4-JP (cut-off that moves with theta).
Timings: one process, BLAS forced single-threaded, each call timed individually
(time.perf_counter_ns), medians of >= 50 calls after warm-up.

Counts: kernel calls and columns to build the continuum basis (analytic derivatives, and the
paper's 5-point stencil for comparison), the harmonic basis (5-point stencil; per-call times
read from the survey memo, not re-run here), and one (gamma, B) joint case: JP ageing with
synchrotron losses in the local field B ~ U[4, 6] muG (gamma_c depends on B), comparing a
(gamma x B) table with a 12-bin x 6-monomial (total degree <= 2 in x_gamma, x_B) basis.

Writes cost_results.json.  Run: python cost.py
"""
from __future__ import annotations

import os

for _v in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import platform  # noqa: E402
import subprocess  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from math import factorial  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
import basis as B  # noqa: E402
import ensembles as E  # noqa: E402
import kernels as K  # noqa: E402
from config import PAPER, gl_interval  # noqa: E402
from provenance import provenance, write_json  # noqa: E402

T0 = time.time()
SURVEY = ("survey memo of the harmonic pipeline, section 3; not distributed with the study "
          "(it records local paths); the values below are copied from it by hand")
GAMMA0 = E.GAMMA0
N_SLOW, N_FAST = 60, 2000
B0, DB = 5.0, 1.0  # joint case: x_B = (B - B0)/DB in [-1, 1]
MONOMIALS = ((0, 0), (1, 0), (2, 0), (0, 1), (1, 1), (0, 2))


# ----------------------------------------------------------------------------- machine
def _sysctl(name):
    try:
        return subprocess.run(["sysctl", "-n", name], capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def machine_info() -> dict:
    cfgd = np.show_config(mode="dicts")
    return {
        "platform": platform.platform(), "machine": platform.machine(),
        "cpu": _sysctl("machdep.cpu.brand_string"), "ncpu": _sysctl("hw.ncpu"),
        "performance_cores": _sysctl("hw.perflevel0.physicalcpu"),
        "efficiency_cores": _sysctl("hw.perflevel1.physicalcpu"),
        "python": platform.python_version(), "numpy": np.__version__, "scipy": scipy.__version__,
        "blas": cfgd["Build Dependencies"]["blas"]["name"],
        "thread_env": {v: os.environ.get(v) for v in ("OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
                                                      "OMP_NUM_THREADS", "MKL_NUM_THREADS")},
        "process": "single Python process, no multiprocessing",
    }


def busy() -> dict:
    """Load averages and the summed %CPU of all other processes (ps), for the timing record."""
    try:
        out = subprocess.run(["ps", "-Ao", "pid=,pcpu="], capture_output=True, text=True, check=True).stdout
        tot = sum(float(c) for pid, c in (ln.split() for ln in out.strip().splitlines()) if int(pid) != os.getpid())
    except (OSError, subprocess.CalledProcessError, ValueError):
        tot = float("nan")
    return {"time": time.strftime("%H:%M:%S"), "loadavg": list(os.getloadavg()), "other_processes_pcpu": tot}


def wait_for_idle(max_wait_s=3600, pcpu_limit=400.0, load_limit=6.0) -> dict:
    """Block until other processes use < pcpu_limit % CPU in total and the 1-min load < load_limit."""
    t0, samples = time.time(), []
    while True:
        b = busy()
        samples.append(b)
        if b["other_processes_pcpu"] < pcpu_limit and b["loadavg"][0] < load_limit:
            break
        if time.time() - t0 > max_wait_s:
            break
        time.sleep(15)
    return {"waited_s": time.time() - t0, "last": samples[-1], "n_samples": len(samples),
            "limits": {"other_processes_pcpu": pcpu_limit, "load_1min": load_limit}}


def timed(fn, args, n, warm=3, keep=False):
    """Time n calls fn(args[i % len(args)]) individually; seconds.  keep: also return outputs."""
    for i in range(warm):
        fn(args[i % len(args)])
    t = np.empty(n)
    outs = []
    for i in range(n):
        a = args[i % len(args)]
        s = time.perf_counter_ns()
        r = fn(a)
        t[i] = (time.perf_counter_ns() - s) * 1e-9
        if keep:
            outs.append(r)
    stats = {"n": n, "median_s": float(np.median(t)), "p16_s": float(np.percentile(t, 16)),
             "p84_s": float(np.percentile(t, 84)), "min_s": float(t.min()), "machine_after": busy()}
    return (stats, outs) if keep else stats


def rel_err(s, ref):
    """max over rows of |s - ref| / I_j(ref), for (41, 3) arrays."""
    return float(np.max(np.abs(s - ref) / ref[:, :1]))


# ----------------------------------------------------------------------------- 1-D families
def e2_density(theta):
    A, p1, p2, lgb, s = theta
    gb = np.exp(lgb)
    return lambda g: (A / GAMMA0) * (g / gb) ** (-p1) * (1 + (g / gb) ** s) ** (-(p2 - p1) / s)


def jp_density(theta):
    A, p, uc = theta
    gc = np.exp(uc)
    return lambda g: (A / GAMMA0) * (g / GAMMA0) ** (-p) * np.clip(1 - g / gc, 0, None) ** (p - 2)


def jp_regular(theta):
    A, p, uc = theta

    def reg(g):
        d = uc - np.log(g)
        ratio = np.where(d > 0, -np.expm1(-d) / np.where(d > 0, d, 1.0), 1.0)
        return g * (A / GAMMA0) * (g / GAMMA0) ** (-p) * ratio ** (p - 2)

    return reg


def jp_measure(cfg, theta, n=24):
    A, p, uc = theta
    gc = float(np.exp(uc))
    return E.measure_1d(cfg, jp_density(theta), cfg.gamma_lo, gc, n, cut=gc, alpha=p - 2.0,
                        regular=jp_regular(theta))


def route_i(cfg, table, name):
    if name == "E2":
        return lambda th: E.direct_from_measure(cfg, E.measure_1d(cfg, e2_density(th), cfg.gamma_lo,
                                                                   cfg.gamma_hi, 24), table=table)
    return lambda th: E.direct_from_measure(cfg, jp_measure(cfg, th), table=table)


def fixed_gamma_nodes(cfg, n_per_bin):
    """GL nodes in ln gamma per bin and weights for dgamma (= gamma du)."""
    u, w = [], []
    for a, b in zip(cfg.log_edges[:-1], cfg.log_edges[1:]):
        uu, ww = gl_interval(n_per_bin, a, b)
        u.append(uu)
        w.append(ww)
    u, w = np.concatenate(u), np.concatenate(w)
    g = np.exp(u)
    return g, w * g


def build_table(cfg, table, g):
    """(123, n_gamma) table of averaged channel kernels (I, Q, U rows, separable azimuth factor)."""
    kI, kQ = B.kbar(cfg, g, order=0, table=table)
    st = B.to_stokes(kI[0], kQ[0], cfg.azimuth_factor)  # (n_g, J, 3)
    return np.transpose(st, (1, 2, 0)).reshape(-1, len(g))


def route_ii(T, g, wq, name):
    dens = e2_density if name == "E2" else jp_density
    return lambda th: (T @ (wq * dens(th)(g))).reshape(-1, 3)


def route_iii(cfg, C2, name):
    fam = E.smooth_broken_moments if name == "E2" else E.jp_moments
    return lambda th: (C2 @ fam(cfg, th, 2)[0].ravel()).reshape(-1, 3)


def theta_list(name, n, seed):
    rng = np.random.default_rng(seed)
    ens = E.reference_ensembles(PAPER)
    if name == "E2":
        base = ens["E2"].theta
        d = np.c_[0.05 * rng.normal(size=n), 0.05 * rng.normal(size=n), 0.05 * rng.normal(size=n),
                  0.1 * rng.normal(size=n), np.zeros(n)]
        th = np.tile(base, (n, 1)) + d
        th[:, 0] = base[0] * (1 + d[:, 0])
    else:
        base = ens["E4-JP"].theta
        d = np.c_[0.05 * rng.normal(size=n), 0.05 * rng.normal(size=n), 0.05 * rng.normal(size=n)]
        th = np.tile(base, (n, 1)) + d
        th[:, 0] = base[0] * (1 + d[:, 0])
    th[0] = base
    return [t for t in th]


def one_d_case(cfg, table, C2, name) -> dict:
    thetas = theta_list(name, N_SLOW, seed=7 if name == "E2" else 8)
    f_i = route_i(cfg, table, name)
    t_i, ref = timed(f_i, thetas, N_SLOW, keep=True)
    out = {"route_i_recompute": {**t_i, "kernel_nodes_per_eval": int(len(
        (E.measure_1d(cfg, e2_density(thetas[0]), cfg.gamma_lo, cfg.gamma_hi, 24) if name == "E2"
         else jp_measure(cfg, thetas[0])).gamma))}}
    for n_per_bin in (24, 8):
        g, wq = fixed_gamma_nodes(cfg, n_per_bin)
        s = time.perf_counter()
        T = build_table(cfg, table, g)
        t_build = time.perf_counter() - s
        f_ii = route_ii(T, g, wq, name)
        err = max(rel_err(f_ii(th), r) for th, r in zip(thetas, ref))
        out[f"route_ii_table_{len(g)}"] = {**timed(f_ii, thetas, N_FAST), "build_s": t_build,
                                           "columns": len(g), "max_rel_err_vs_route_i": err}
    f_iii = route_iii(cfg, C2, name)
    fam = E.smooth_broken_moments if name == "E2" else E.jp_moments
    a0 = fam(cfg, thetas[0], 2)[0].ravel()
    out["route_iii_parts"] = {"family_moments_only": timed(lambda th: fam(cfg, th, 2), thetas, N_FAST),
                              "matvec_123x36_only": timed(lambda a: C2 @ a, [a0], N_FAST)}
    out["route_iii_moment_basis_36"] = {**timed(f_iii, thetas, N_FAST), "columns": 36,
                                        "max_rel_err_vs_route_i": max(rel_err(f_iii(th), r)
                                                                      for th, r in zip(thetas, ref))}
    out["n_theta_cycled"] = len(thetas)
    return out


# ----------------------------------------------------------------------------- build counts
def build_counts(cfg, table) -> dict:
    out = {}
    b, w = cfg.b_nodes_weights
    s = time.perf_counter()
    B.KernelTable(b, w)
    out["kernel_table_build_s_single"] = time.perf_counter() - s
    n_y = len(table.lny)
    out["kernel_table"] = {"y_nodes": n_y, "B_nodes": cfg.n_b, "kernels": 2, "orders": table.max_order + 1,
                           "F_or_G_quadratures": int(2 * n_y * cfg.n_b), "u_nodes_each": K.DEFAULT_NODES}
    per_call = cfg.n_eta * cfg.n_channels * cfg.n_chan_nodes
    out["kbar_call_definition"] = ("one averaged kernel Kbar(gamma): eta (12) x channel (41 x 400 nodes) "
                                   "x screen average of the B-averaged spline, for I, Q, U")
    out["spline_evaluations_per_kbar_call_per_kernel"] = per_call
    gs = [cfg.gamma_b[k] for k in range(cfg.n_bins)]
    out["kbar_order0_single_gamma"] = timed(lambda g: B.kbar(cfg, [g], order=0, table=table), gs, 240)
    out["kbar_order2_single_gamma"] = timed(lambda g: B.kbar(cfg, [g], order=2, table=table), gs, 240)
    out["continuum_analytic_basis"] = {
        "kbar_calls": cfg.n_bins, "orders_per_call": 3, "columns": 36,
        **timed(lambda _: B.build_basis(cfg, 2, table=table), [0], 60)}
    out["continuum_stencil_basis"] = {
        "kbar_calls": 5 * cfg.n_bins, "orders_per_call": 1, "columns": 36,
        **timed(lambda _: B.stencil_derivative_spectra(cfg, table=table), [0], 60)}
    out["harmonic_stencil_basis_read_from_survey"] = {
        "source": SURVEY,
        "kbar_calls": 5 * cfg.n_bins, "half_step_check_calls": 5 * cfg.n_bins, "columns": 36,
        "per_call_s_adopted_settings": {"gamma_382": 1.32, "gamma_6977": 2.26, "gamma_78502": 2.89},
        "per_call_s_coarse_settings": {"gamma_382": 0.50, "gamma_6977": 0.82, "gamma_78502": 1.02},
        "recorded_logs_s": {"basis_refined": 134.7, "basis_halfstep": 49.1, "basis_coarse": 47.9,
                            "truth_refined_96_energy_nodes": 208.7, "truth_coarse": 57.3},
        "label": "read (survey memo section 3, OPENBLAS_NUM_THREADS=1); not re-run here",
        "truth_energy_nodes": 96,
    }
    return out


# ----------------------------------------------------------------------------- joint (gamma, B)
def fixed_b_table(bval):
    return B.KernelTable([bval], [1.0])


def gamma_c_local(bval, t_myr=19.0, u_rad_ev=1.0):
    b_s = K.LOSS_COEFF * (bval * K.MUG) ** 2 / (8 * np.pi)
    b_c = K.LOSS_COEFF * u_rad_ev * K.EV_ERG
    return 1.0 / ((b_s + b_c) * t_myr * K.SEC_PER_MYR)


def joint_columns(cfg, tab0) -> np.ndarray:
    """(41, 3, 72) columns of monomials x_gamma^r x_B^s / (r! s!) derivatives at (gamma_b, B0).

    B-derivatives from the continuum identity Kbar(u, B) = lam G(u + ln(lam)/2), lam = B/B0,
    G(u) = Kbar(u, B0):  d_eps K = G + G'/2,  (1/2) d_eps^2 K = G'/4 + G''/8,  d_u d_eps K = G' + G''/2.
    """
    kI, kQ = B.kbar(cfg, cfg.gamma_b, order=2, table=tab0)  # (3, nb, J): d^r/du^r
    h = cfg.h
    cols = []
    for k in (kI, kQ):
        G0, G1, G2 = k
        e = DB / B0
        c = {(0, 0): G0, (1, 0): h * G1, (2, 0): h**2 / 2 * G2, (0, 1): e * (G0 + 0.5 * G1),
             (1, 1): h * e * (G1 + 0.5 * G2), (0, 2): e**2 * (0.25 * G1 + 0.125 * G2)}
        cols.append(np.stack([c[m] for m in MONOMIALS], axis=1))  # (nb, 6, J)
    DI, DQ = cols
    st = B.to_stokes(DI, DQ, cfg.azimuth_factor) * cfg.w_b[:, None, None, None]
    return np.transpose(st, (2, 3, 0, 1)).reshape(cfg.n_channels, 3, -1)


def joint_columns_stencil_check(cfg, tab0, C72) -> dict:
    """B-derivative columns by the 5-point stencil in x_B (step 0.04) with fixed-B tables."""
    step = 0.04
    offs = (-2, -1, 1, 2)
    tabs = {o: fixed_b_table(B0 + o * step * DB) for o in offs}
    vals = {0: B.kbar(cfg, cfg.gamma_b, order=1, table=tab0)}
    vals.update({o: B.kbar(cfg, cfg.gamma_b, order=1, table=tabs[o]) for o in offs})
    out, cols = {}, {}
    for comp, idx in (("I", 0), ("Q", 1)):
        v = {o: vals[o][idx] for o in vals}  # (2, nb, J)
        d1 = (v[-2] - 8 * v[-1] + 8 * v[1] - v[2]) / (12 * step)
        d2 = (-v[-2] + 16 * v[-1] - 30 * v[0] + 16 * v[1] - v[2]) / (12 * step**2) / 2
        st = {(0, 1): d1[0], (0, 2): d2[0], (1, 1): cfg.h * d1[1]}
        for m, arr in st.items():
            j = MONOMIALS.index(m)
            col = (arr * cfg.w_b[:, None])
            if comp == "I":
                ref = C72[:, 0, j::6].T
                out[f"I_{m}_max_abs_diff_over_col_max"] = float(np.max(np.abs(col - ref)) / np.max(np.abs(ref)))
                cols[m] = {"I": col}
            else:
                p = cfg.azimuth_factor * col
                ref_q = C72[:, 1, j::6].T + 1j * C72[:, 2, j::6].T
                out[f"P_{m}_max_abs_diff_over_col_max"] = float(np.max(np.abs(p - ref_q)) / np.max(np.abs(ref_q)))
                cols[m]["P"] = p
    return out, cols


def stencil_columns_in_gamma_span(cfg, C72, cols, sig) -> dict:
    """Residual of each stencil-built B-derivative column (123 rows, whitened) after
    least-squares projection onto the 36 gamma-derivative columns at B0."""
    gidx = [b * 6 + MONOMIALS.index(m) for b in range(cfg.n_bins) for m in ((0, 0), (1, 0), (2, 0))]
    G = C72.reshape(-1, 72)[:, gidx] / sig[:, None]
    scale = float(np.max(np.linalg.norm(G, axis=0)))
    rel, glob = np.zeros((len(cols), cfg.n_bins)), np.zeros((len(cols), cfg.n_bins))
    for i, (m, c) in enumerate(cols.items()):
        for b in range(cfg.n_bins):
            v = np.stack([c["I"][b], c["P"][b].real, c["P"][b].imag], axis=-1).ravel() / sig
            coef, *_ = np.linalg.lstsq(G, v, rcond=None)
            r = float(np.linalg.norm(G @ coef - v))
            rel[i, b], glob[i, b] = r / np.linalg.norm(v), r / scale
    return {"max_over_own_norm_all_bins": float(rel.max()), "max_over_own_norm_bins_3_11": float(rel[:, 3:].max()),
            "max_over_largest_gamma_column_norm": float(glob.max()),
            "note": "stencil step 0.04 in x_B; residual includes the stencil's own truncation error"}


def joint_case(cfg, table, C2) -> dict:
    out = {"definition": ("N(gamma, B) = P(B) (A/1e4)(gamma/1e4)^-p (1 - gamma/gamma_c(B))^(p-2), "
                          "P(B) = U[4,6] muG, gamma_c(B) = 1/((b_s(B) + b_c) t), b_s(B) from B^2 in the local "
                          "field, b_c Thomson on 1 eV cm^-3, t = 19 Myr, p = 2.6; A sets unit total column"),
           "x_B": "(B - 5 muG)/(1 muG)", "monomials_(r,s)": MONOMIALS}
    s = time.perf_counter()
    tab0 = fixed_b_table(B0)
    out["fixed_B_table_build_s"] = time.perf_counter() - s
    bn, bw = gl_interval(16, 4.0, 6.0)
    bw = bw / 2.0
    s = time.perf_counter()
    tabs = [fixed_b_table(bv) for bv in bn]
    out["fixed_B_tables_16_build_s"] = time.perf_counter() - s
    gcs = np.array([gamma_c_local(bv) for bv in bn])
    out["gamma_c_range"] = [float(gcs.min()), float(gcs.max())]
    p = 2.6

    def moms(th_p, th_t, A=1.0):
        gck = np.array([gamma_c_local(bv, th_t) for bv in bn])
        th = np.c_[np.full(16, A), np.full(16, th_p), np.log(gck)]
        return E.jp_moments(cfg, th, 2)  # (16, 12, 3)

    a_k = moms(p, 19.0)
    A_unit = 1.0 / float(np.sum(bw[:, None] * a_k[:, :, 0] * cfg.w_b[None, :]))
    out["A_unit_column"] = A_unit

    def direct(th, nB=16, n=24, tabs_=None, bn_=None, bw_=None):
        A, pp, tm = th
        tabs_ = tabs if tabs_ is None else tabs_
        bn_, bw_ = (bn, bw) if bn_ is None else (bn_, bw_)
        tot = np.zeros((cfg.n_channels, 3))
        for bv, wv, tb in zip(bn_, bw_, tabs_):
            uc = np.log(gamma_c_local(bv, tm))
            m = jp_measure(cfg, (A, pp, uc), n)
            tot += wv * E.direct_from_measure(cfg, m, table=tb)
        return tot

    th0 = (A_unit, p, 19.0)
    s = time.perf_counter()
    S0 = direct(th0)
    out["direct_one_eval_s"] = time.perf_counter() - s
    # node doubling: B nodes 16 -> 32, gamma nodes 24 -> 48
    bn2, bw2 = gl_interval(32, 4.0, 6.0)
    S2 = direct(th0, n=48, tabs_=[fixed_b_table(bv) for bv in bn2], bn_=bn2, bw_=bw2 / 2.0)
    out["direct_node_doubling_max_rel"] = rel_err(S0, S2)
    # scaling identity check
    errs = []
    for bv, tb in zip(bn[::5], tabs[::5]):
        lam = bv / B0
        kI1, kQ1 = B.kbar(cfg, cfg.gamma_b, table=tb)
        kI2, kQ2 = B.kbar(cfg, cfg.gamma_b * np.sqrt(lam), table=tab0)
        errs.append(float(np.max(np.abs(kI1[0] - lam * kI2[0])) / np.max(kI1[0])))
        errs.append(float(np.max(np.abs(kQ1[0] - lam * kQ2[0])) / np.max(np.abs(kQ1[0]))))
    out["scaling_identity_max_rel"] = max(errs)

    # moment route
    s = time.perf_counter()
    C72 = joint_columns(cfg, tab0)
    out["joint_basis_build_s"] = time.perf_counter() - s
    out["joint_basis_stencil_check"], st_cols = joint_columns_stencil_check(cfg, tab0, C72)
    C72r = C72.reshape(-1, 72)
    xb = (bn - B0) / DB

    def joint_moments(th):
        A, pp, tm = th
        a = moms(pp, tm, A)  # (16, 12, 3)
        m = np.zeros((cfg.n_bins, 6))
        for j, (r, sB) in enumerate(MONOMIALS):
            m[:, j] = np.sum(bw[:, None] * xb[:, None] ** sB * a[:, :, r], axis=0)
        return m, np.sum(bw[:, None, None] * a, axis=0)

    m0, aeff0 = joint_moments(th0)
    # moments check against the direct measures
    mchk = np.zeros((cfg.n_bins, 6))
    for bv, wv in zip(bn, bw):
        mm = jp_measure(cfg, (A_unit, p, np.log(gamma_c_local(bv))))
        for j, (r, sB) in enumerate(MONOMIALS):
            np.add.at(mchk[:, j], mm.b, wv * mm.w * mm.x**r * ((bv - B0) / DB) ** sB)
    mchk /= cfg.w_b[:, None]
    out["joint_moments_vs_measure_max_abs"] = float(np.max(np.abs(mchk - m0)))
    S72 = (C72r @ m0.ravel()).reshape(-1, 3)
    out["moment_route_72_max_rel_err"] = rel_err(S72, S0)
    S36 = (C2 @ aeff0.ravel()).reshape(-1, 3)
    out["preaveraged_36_with_marginal_moments_max_rel_err"] = rel_err(S36, S0)
    # degeneracy: stencil-built (identity-free) B columns lie in the span of the 36 gamma columns
    sig = 0.01 * np.repeat(S0[:, :1], 3, axis=1).ravel()
    out["stencil_B_columns_rel_residual_outside_gamma_span"] = stencil_columns_in_gamma_span(cfg, C72, st_cols, sig)
    # push-forward route: gamma' = gamma sqrt(B/B0) with weight B/B0, fixed-B0 36-column basis
    C36_B0 = B.build_basis(cfg, 2, table=tab0).reshape(-1, 36)
    apf = np.zeros((cfg.n_bins, 3))
    for bv, wv in zip(bn, bw):
        mm = jp_measure(cfg, (A_unit, p, np.log(gamma_c_local(bv))))
        lam = bv / B0
        u = np.log(mm.gamma * np.sqrt(lam))
        bb_ = np.clip(np.searchsorted(cfg.log_edges, u, side="right") - 1, 0, cfg.n_bins - 1)
        xx = (u - np.log(cfg.gamma_b[bb_])) / cfg.h
        for r in range(3):
            np.add.at(apf[:, r], bb_, wv * lam * mm.w * xx**r)
    apf /= cfg.w_b[:, None]
    out["pushforward_36_fixed_B0_max_rel_err"] = rel_err((C36_B0 @ apf.ravel()).reshape(-1, 3), S0)
    out["pushforward_note"] = ("continuum only: K(gamma, B) = (B/B0) K(gamma sqrt(B/B0), B0) maps any (gamma, B) "
                               "ensemble onto gamma-moments at B0; nodes with gamma' outside [300, 1e5] are kept "
                               "in the edge bins (|x| slightly > 1)")

    # table route: fixed gamma nodes (12 x 24) x 16 B nodes
    g, wq = fixed_gamma_nodes(cfg, 24)
    s = time.perf_counter()
    T = np.concatenate([build_table(cfg, tb, g) for tb in tabs], axis=1)  # (123, 16*288)
    out["table_build_s"] = time.perf_counter() - s
    gg = np.tile(g, 16)
    ww = np.concatenate([wv * wq for wv in bw])
    bb = np.repeat(bn, len(g))

    def table_eval(th):
        A, pp, tm = th
        gc = gamma_c_local(bb, tm)
        dens = (A / GAMMA0) * (gg / GAMMA0) ** (-pp) * np.clip(1 - gg / gc, 0, None) ** (pp - 2)
        return (T @ (ww * dens)).reshape(-1, 3)

    def moment_eval(th):
        m, _ = joint_moments(th)
        return (C72r @ m.ravel()).reshape(-1, 3)

    rng = np.random.default_rng(9)
    thetas = [th0] + [(A_unit * (1 + 0.05 * rng.normal()), p + 0.05 * rng.normal(), 19.0 * (1 + 0.05 * rng.normal()))
                      for _ in range(N_SLOW - 1)]
    n_direct = 5
    t_dir, refs = timed(direct, thetas[:n_direct], n_direct, warm=1, keep=True)
    out["route_i_recompute"] = {**t_dir, "kernel_nodes_per_eval": int(16 * len(jp_measure(cfg, (A_unit, p, np.log(gcs[0]))).gamma)),
                                "note": f"{n_direct} calls only (each evaluation recomputes 16 fixed-B channel averages)"}
    out["route_ii_table"] = {**timed(table_eval, thetas, 200), "columns": int(T.shape[1]), "kernel_calls_to_build": int(T.shape[1]),
                             "max_rel_err_vs_route_i": max(rel_err(table_eval(th), r) for th, r in zip(thetas, refs))}
    out["route_iii_joint_moment_basis_72"] = {
        **timed(moment_eval, thetas, 200), "columns": 72, "kernel_calls_to_build": cfg.n_bins,
        "max_rel_err_vs_route_i": max(rel_err(moment_eval(th), r) for th, r in zip(thetas, refs))}
    out["counts"] = {
        "table_nodes_n_gamma_x_n_B": [len(g), 16], "table_columns": int(T.shape[1]),
        "table_entries": int(T.size), "moment_bins_x_monomials": [cfg.n_bins, 6], "moment_columns": 72,
        "moment_entries": int(C72r.size),
        "continuum_kernel_calls_moment": f"{cfg.n_bins} at B0 (orders 0..2 in ln gamma; B derivatives by the scaling identity)",
        "generic_kernel_calls_moment_tensor_5pt_stencil": 25 * cfg.n_bins,
        "generic_kernel_calls_moment_note": ("no scaling identity (e.g. harmonic): 5-point stencils along x_gamma and x_B "
                                             "and their tensor product for the mixed term need 25 evaluations per bin"),
        "table_kernel_calls": int(T.shape[1]),
    }
    return out


# ----------------------------------------------------------------------------- main
def main():
    cfg = PAPER
    idle = wait_for_idle()
    table = B.get_table(cfg)
    C2 = B.build_basis(cfg, 2, table=table).reshape(-1, 36)
    res = {"machine": machine_info(), "wait_for_idle": idle, "machine_start": busy()}
    res["one_d"] = {name: one_d_case(cfg, table, C2, name) for name in ("E2", "E4-JP")}
    res["build"] = build_counts(cfg, table)
    res["joint_gamma_B"] = joint_case(cfg, table, C2)
    res["machine_end"] = busy()
    res["description"] = __doc__
    res["config"] = cfg.describe()
    res["provenance"] = provenance(__file__, T0)
    write_json(HERE / "cost_results.json", res)
    print("done", time.time() - T0)


if __name__ == "__main__":
    main()
