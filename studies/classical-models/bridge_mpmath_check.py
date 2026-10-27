"""Independent check of the continuum shape functions in bridge_continuum.py.

Compares F(e^u), G(e^u) and their first two u-derivatives (u = ln x) with mpmath
(25 digits; F by quadrature of K_5/3 with breakpoints at x+{1,4,15,40}, derivatives by mp.diff).
Writes bridge_mpmath_check.json. Runtime a few minutes.
Run: python bridge_mpmath_check.py
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
from pathlib import Path  # noqa: E402

import mpmath as mp  # noqa: E402
import numpy as np  # noqa: E402

import bridge_continuum as bc  # noqa: E402

HERE = Path(__file__).resolve().parent
POINTS = (3e-4, 1e-2, 0.3, 1.0, 4.0, 30.0)


def f_mp(u):
    x = mp.e**u
    knots = [x, x + 1, x + 4, x + 15, x + 40, mp.inf]
    return x * mp.quad(lambda t: mp.besselk(mp.mpf(5) / 3, t), knots)


def g_mp(u):
    x = mp.e**u
    return x * mp.besselk(mp.mpf(2) / 3, x)


def main() -> None:
    mp.mp.dps = 25
    rows = []
    for x in POINTS:
        u = mp.log(x)
        fs, gs = bc.shape_functions(np.array([x]), 2)
        ref_f = [f_mp(u), mp.diff(f_mp, u, 1), mp.diff(f_mp, u, 2)]
        ref_g = [g_mp(u), mp.diff(g_mp, u, 1), mp.diff(g_mp, u, 2)]
        rows.append({
            "x": x,
            "F_rel_err_r0_r1_r2": [float(abs(fs[r][0] / ref_f[r] - 1)) for r in range(3)],
            "G_rel_err_r0_r1_r2": [float(abs(gs[r][0] / ref_g[r] - 1)) for r in range(3)],
        })
    out = {
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "bridge_continuum_sha256": hashlib.sha256((HERE / "bridge_continuum.py").read_bytes()).hexdigest(),
        "mpmath": mp.__version__,
        "rows": rows,
        "max_rel_err": max(max(r["F_rel_err_r0_r1_r2"] + r["G_rel_err_r0_r1_r2"]) for r in rows),
    }
    (HERE / "bridge_mpmath_check.json").write_text(json.dumps(out, indent=1) + "\n")
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
