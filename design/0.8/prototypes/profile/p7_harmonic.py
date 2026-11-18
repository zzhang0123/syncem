"""Profile 7: the high-order harmonic route (syncem.model.high_order_harmonic).

high_order_channel_modes at gamma in {20, 100, 300}, B = 5 uG, mu = eta = 0.2 (on the cone), on bump
channels at 0.1, 1 and 10 GHz with 20 per cent half-width (unit peak, as the guide's examples).
Wall time per point, the route per channel, the number of order samples, and a cProfile of the
slowest point split into SciPy special functions, NumPy array work and Python frames.

Run: python p7_harmonic.py   (about 1 minute)
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time

import numpy as np

import _common as U

from syncem.model import Channels
from syncem.model.high_order_harmonic import high_order_channel_modes, high_order_line_powers

CENTRES = np.array([1e8, 1e9, 1e10])
MU, ETA = 0.2, 0.2  # on the emission cone (|mu - eta| < 1/gamma). Off the cone the lines vanish.


def run_point(ch, gamma, **kw):
    return high_order_channel_modes(ch, gamma, 5e-6, MU, ETA, max_quad_order=2048,
                                    allow_unconverged=True, **kw)


def main() -> None:
    ch = Channels.bump(CENTRES, 0.2 * CENTRES)
    res = {"points": []}
    run_point(ch, 300.0)  # warm imports and caches
    for gamma in (20.0, 100.0, 300.0):
        t0 = time.perf_counter()
        r = run_point(ch, gamma)
        sec = time.perf_counter() - t0
        row = {"gamma": gamma, "wall_s": sec, "routes": list(r.routes), "order_samples": int(r.evaluated_orders),
               "active_ranges": [list(map(float, a)) for a in r.active_ranges],
               "us_per_order_sample": 1e6 * sec / max(r.evaluated_orders, 1)}
        res["points"].append(row)
        print(row, flush=True)
    from scipy import special
    from syncem.model.high_order_harmonic import _geometry

    slow_row = max(res["points"], key=lambda p: p["wall_s"])
    slow = slow_row["gamma"]
    a, b = slow_row["active_ranges"][-1]
    m = np.linspace(a, b, 4096)
    _, sec, _ = U.timed(high_order_line_powers, m, slow, 5e-6, MU, ETA, repeat=3)
    res["line_powers_us_per_order"] = 1e6 * sec / m.size
    z = _geometry(slow, MU, ETA)[3]
    _, sec_jv, _ = U.timed(lambda: (special.jv(m - 1.0, m * z), special.jv(m + 1.0, m * z)), repeat=3)
    res["two_jv_us_per_order"] = 1e6 * sec_jv / m.size
    res["jv_share_of_line_powers"] = sec_jv / sec
    res["orders_timed"] = [float(a), float(b)]
    print({k: res[k] for k in ("line_powers_us_per_order", "two_jv_us_per_order", "jv_share_of_line_powers")},
          flush=True)
    pr = cProfile.Profile()
    pr.enable()
    run_point(ch, slow)
    pr.disable()
    st = pstats.Stats(pr)
    total = sum(v[2] for v in st.stats.values())
    special = sum(v[2] for k, v in st.stats.items() if "scipy" in str(k) or "jv" in str(k[2]) or "special" in str(k))
    res["profile_gamma"] = slow
    res["profile_total_tottime_s"] = total
    res["profile_scipy_tottime_s"] = special
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(15)
    res["profile_text"] = s.getvalue()
    print(res["profile_text"], flush=True)
    print(U.save("p7_harmonic", res))


if __name__ == "__main__":
    main()
