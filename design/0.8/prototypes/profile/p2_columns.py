"""Profile 2: column build on the E1 lattice. Study stack (NumPy) against the prototype jet (JAX).

Stack: lib/basis.py derivative_spectra at N = 2 and 4, with the S1 table from the disk cache,
at 400 channel nodes (the study) and at 64 (the prototype's channel rule).
Jet: core.jet on single cells at N = 2 and 4, compile and run timed apart, and one primal of
the E1 averaged kernel with outer_batch = 1 (lax.map) and None (vmap at every level).

Run: python p2_columns.py   (about 1.5 minutes)
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time
from dataclasses import replace

import numpy as np

import _common as U

import core as C
import jax
import jax.numpy as jnp
from core import jet as J
from core import paper as P

import basis as SB
from config import PAPER


def stack(res: dict) -> None:
    t0 = time.perf_counter()
    tab = SB.get_table(PAPER)
    res["stack_table_load_s"] = time.perf_counter() - t0
    for n_nodes in (400, 64):
        cfg = replace(PAPER, n_chan_nodes=n_nodes)
        for N in (2, 4):
            _, sec, ratio = U.timed(SB.derivative_spectra, cfg, N, table=tab, repeat=3)
            res[f"stack_nodes{n_nodes}_N{N}_s"] = sec
            res[f"stack_nodes{n_nodes}_N{N}_cpu_over_wall"] = ratio
            print(n_nodes, N, sec, ratio, flush=True)
    pr = cProfile.Profile()
    pr.enable()
    SB.derivative_spectra(PAPER, 4, table=tab)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(10)
    res["stack_profile_N4"] = s.getvalue()


def one_cell_table(N: int, cell: int = 9):
    full = P.e1_table(N)
    lo, hi = full.lo[cell, 0], full.hi[cell, 0]
    return C.ProductTable.tensor({"gamma": C.Cells.log(float(lo), float(hi), 1, order=N)}, total=N)


def jet(res: dict) -> None:
    for outer in (1, None):
        avg = P.e1_kernel(outer_batch=outer)
        f = jax.jit(lambda g, b=avg: b({"gamma": g}))
        t0 = time.perf_counter()
        jax.block_until_ready(f(1e4))
        first = time.perf_counter() - t0
        _, sec, ratio = U.timed(f, 1e4, repeat=3)
        res[f"primal_outer{outer}"] = {"first_s": first, "run_s": sec, "cpu_over_wall": ratio}
        print("primal", outer, first, sec, ratio, flush=True)
    avg = P.e1_kernel(outer_batch=1)
    for N in (2, 4):
        tab = one_cell_table(N)
        sig = tab.signature(0)
        fn = J._cell_routine(avg, tab, sig)
        refs, hws = jnp.asarray(tab.reference[[0]]), jnp.asarray(tab.half_width[[0]])
        t0 = time.perf_counter()
        lowered = fn.lower(refs, hws)
        t1 = time.perf_counter()
        comp = lowered.compile()
        t2 = time.perf_counter()
        _, run, ratio = U.timed(comp, refs, hws, repeat=2)
        res[f"jet_N{N}_one_cell"] = {"trace_lower_s": t1 - t0, "compile_s": t2 - t1, "run_s": run,
                                     "cpu_over_wall": ratio,
                                     "primal_equivalents": run / res["primal_outer1"]["run_s"]}
        print("jet", N, res[f"jet_N{N}_one_cell"], flush=True)


def main() -> None:
    res: dict = {}
    stack(res)
    print(res["stack_profile_N4"], flush=True)
    jet(res)
    print(U.save("p2_columns", res))


if __name__ == "__main__":
    main()
