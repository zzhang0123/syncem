"""R5 (iii) with an averaged B parent on the E1 continuum kernel (r3-02), against direct integration.

Correct route: c(B) is a weight inside the spin-2 field average at the field nodes (S2).
Wrong route: <c> times the cached field average S1, which is <c><K_P>. The reference
integrates gamma (6 nodes per cell), B (24 nodes, the basis uses 16) and phi (two-point law
given B) jointly. Also the single-electron numbers of r3-02.

Run from the prototypes directory (about 6 min):
PYTHONPATH=.:../../.. python -m pytest tests/test_R5_e1_field.py -s
"""
from __future__ import annotations

import json
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import core as C
from rules.R5 import e1 as X

RESULTS = Path(__file__).with_name("R5_e1_field_results.json")
CORRECT_TOL = 2e-3  # E1 with a constant factor gives 8.6e-4 of I at N = 2
CACHE: dict = {}
S1: dict = {}
KERNELS: dict = {}


def _record(key, value):
    data = json.loads(RESULTS.read_text()) if RESULTS.exists() else {}
    data[key] = value
    RESULTS.write_text(json.dumps(data, indent=1, default=float))


def _kernels(lo, hi):
    if (lo, hi) not in KERNELS:
        K = X.base_kernel()
        KERNELS[(lo, hi)] = (K, C.AveragedKernel(K, [("eta", X.eta_half_rule())]))
    return KERNELS[(lo, hi)]


# ------------------------------------------------------------------ single electron
@pytest.mark.parametrize("eta", [0.0, None], ids=["eta0", "eta_avg"])
def test_single_electron_r3_02(eta):
    xs = [1e-3, 1e-2, 0.1, 1.0, 3.0, 10.0, 20.0]
    rel_good = X.single_electron_ratio(xs, lambda B: 0.4 * (1 + 0.5 * (B - 5)), eta=eta)
    rel_bad = rel_good / (1 - rel_good)  # |<cK> - <c><K>| / |<c><K>| (same sign, real kernel)
    _record(f"single_electron_{'eta0' if eta == 0.0 else 'eta_avg'}",
            {"x": xs, "rel_to_correct": rel_good.tolist(), "rel_to_factorised": rel_bad.tolist()})
    assert abs(rel_bad[0] - 0.022) < 0.001 and abs(rel_bad[-1] - 0.36) < 0.01  # Spec R5 (iii) numbers


# ------------------------------------------------------------------ lattice cases
VARIANTS = {
    # name: (c in jnp, c in numpy, B range, N list)
    "paper": (lambda B: 0.4 * (1 + 0.5 * (B - 5)), (4.0, 6.0), (0, 2, 4)),
    "zero_mean": (lambda B: 0.4 * (B - 5), (4.0, 6.0), (2,)),
    "near_zero": (lambda B: 1e-3 * (1 + 0.5 * (B - 5)), (4.0, 6.0), (2,)),
    "near_one": (lambda B: 0.995 + 0.005 * (B - 5), (4.0, 6.0), (2,)),
    "strong": (lambda B: 0.5 + 0.5 * (B - 5), (4.0, 6.0), (2,)),
    "phase": (lambda B: 0.6 * jnp.exp(1j * jnp.pi * (B - 5)), (4.0, 6.0), (2,)),
    "range100": (lambda B: (B - 0.5) / 49.5, (0.5, 50.0), (2,)),
}


@pytest.mark.parametrize("name", list(VARIANTS))
def test_field_factor_inside_S2_average(name):
    c, (lo, hi), N_list = VARIANTS[name]
    K, avg_d = _kernels(lo, hi)
    key = (lo, hi, max(N_list))
    timer = X.Timer()
    out, S1[key] = X.field_case(c, lambda b: np.asarray(c(jnp.asarray(b)), complex), N_list, B_lo=lo, B_hi=hi,
                                n_basis=16, n_direct=24 if hi < 10 else 32, cache=CACHE, cols_s1=S1.get(key),
                                kernel=K, avg_direct=avg_d, timer=timer)
    out["timings_s"] = timer.t
    _record(f"field_{name}", out)
    for N in N_list:
        r = out[f"N{N}"]
        assert r["I_block_equal_S1"] < 1e-15
        if N >= 2:
            assert r["correct"]["max_rel_I"] < CORRECT_TOL
            assert r["wrong_mean_c_times_S1"]["max_rel_P"] > 5 * r["correct"]["max_rel_P"] or name == "near_one"
