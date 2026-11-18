"""Record the agreement values that test_boundaries.py asserts, for REPORT.md.

Run: python boundary_values.py   (about 20 s)
"""
from __future__ import annotations

import math

import numpy as np

import _common as U
import test_boundaries as T

import jax.numpy as jnp


def main() -> None:
    res = {"s1": {}, "table": {}, "harmonic": {}}
    lo = T.SB.TABLE_LNY[0] - np.log(100.0) - 0.01
    hi = np.log(5e4) - np.log(0.1) + 0.01
    s0 = T.P1.s0_table(lo, hi)
    for b_node in (0.1, 1.0, 4.0, 6.0, 50.0, 100.0):
        tab = T.SB.KernelTable.__new__(T.SB.KernelTable)
        tab.max_order, tab.b_nodes, tab.b_weights = T.P1.ORDER, np.array([b_node]), np.array([1.0])
        for W in ("F", "G"):
            ref = tab.direct(np.exp(T.LNY), W)
            got = T.P1.s1_from_s0(s0, T.LNY, np.array([b_node]), np.array([1.0]), W)
            res["s1"][f"{W}_B{b_node}"] = T.P1.table_error(*got, *ref)
    import core as C
    import p6_ray as P6
    from syncem.ultrarel import F, G

    tk = P6.TabulatedKernel(C.ContinuumChannelKernel(C.bump_channels(n=3, n_nodes=8)))
    edge = tk.x0 + tk.h * (tk.n - 1)
    for lnx in [math.log(v) for v in (1e-13, 1e-6, 0.3, 10.0, 40.0)] + [edge - 0.05, edge - 1e-6]:
        for W, fn in (("F", F), ("G", G)):
            q = float(fn(jnp.exp(lnx)))
            t = float(jnp.exp(tk._lnW(W, jnp.asarray(lnx))))
            res["table"][f"{W}_lnx{lnx:.4f}"] = abs(t - q) / max(abs(t), abs(q), 1e-300)
    for gamma in (5.0, 30.0, 300.0, 3000.0):
        for factor in (0.5, 1.0, 2.0):
            rel, scale = T.harmonic_rel(gamma, factor)
            res["harmonic"][f"g{gamma}_x{factor}"] = {"rel": rel, "max_abs": scale}
    print(res)
    print(U.save("boundaries", res))


if __name__ == "__main__":
    main()
