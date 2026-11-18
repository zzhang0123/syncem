"""Profile 4b: NNLS process-pool scaling with single-threaded BLAS in every process.

The pool in p4_nnls.py reached 2.8 times on 28 workers. This run pins BLAS to one thread
(set before NumPy loads) and scans the worker count to separate oversubscription from
the per-task overhead.

Run: python p4b_nnls_pool.py   (about 1 minute)
"""
from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = "1"

import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402

import numpy as np  # noqa: E402

import _common as U  # noqa: E402
import p4_nnls as P4  # noqa: E402


def main() -> None:
    A, s_w = P4.system()
    rng = np.random.default_rng(2)
    D = s_w[None, :] + rng.standard_normal((5600, A.shape[0]))
    P4._init(A)
    _, sec, ratio = U.timed(P4._solve_chunk, D[:300])
    res = {"serial_ms": 1e3 * sec / 300, "serial_cpu_over_wall": ratio, "pool": []}
    print(res, flush=True)
    for workers in (4, 8, 16, 20, 28):
        with ProcessPoolExecutor(workers, initializer=P4._init, initargs=(A,)) as ex:
            list(ex.map(P4._solve_chunk, np.array_split(D[:workers * 10], workers)))
            t0 = time.perf_counter()
            list(ex.map(P4._solve_chunk, np.array_split(D, workers * 4)))
            ms = 1e3 * (time.perf_counter() - t0) / len(D)
        row = {"workers": workers, "ms_per_spectrum": ms, "speedup": res["serial_ms"] / ms,
               "projected_1e6_min": ms * 1e6 / 1e3 / 60}
        res["pool"].append(row)
        print(row, flush=True)
    print(U.save("p4b_nnls_pool", res))


if __name__ == "__main__":
    main()
