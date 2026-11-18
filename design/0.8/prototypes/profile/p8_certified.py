"""Profile 8: interval certification (mpmath) of a dense harmonic channel.

certified_dense_channel_modes on the guide's configuration (0.1 GHz bump, 20 per cent half-width,
gamma = 3000, B = 5 uG, mu = eta = 0, atol = 1e-19), timed three times, and a cProfile split of
the time into mpmath, syncem and other frames.

Run: python p8_certified.py   (under a minute)
"""
from __future__ import annotations

import cProfile
import io
import pstats
import time

import _common as U

from syncem.model import Channels
from syncem.model._certified_dense_harmonic import certified_dense_channel_modes


def main() -> None:
    ch = Channels.bump([1e8], [2e7])
    call = lambda: certified_dense_channel_modes(ch, 3000.0, 5e-6, 0.0, 0.0, atol=1e-19, max_blocks=512)  # noqa: E731
    out = call()
    times = []
    for _ in range(3):
        t0 = time.perf_counter()
        call()
        times.append(time.perf_counter() - t0)
    res = {"wall_s": times, "retained_blocks": [int(b) for b in out.retained_blocks]}
    pr = cProfile.Profile()
    pr.enable()
    call()
    pr.disable()
    st = pstats.Stats(pr)
    tot = sum(v[2] for v in st.stats.values())
    by = {"mpmath": 0.0, "syncem": 0.0, "other": 0.0}
    for k, v in st.stats.items():
        key = "mpmath" if "mpmath" in k[0] else "syncem" if "syncem" in k[0] else "other"
        by[key] += v[2]
    res["profile_tottime_s"] = tot
    res["profile_share"] = {k: v / tot for k, v in by.items()}
    res["profile_calls"] = int(sum(v[1] for v in st.stats.values()))
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(12)
    res["profile_text"] = s.getvalue()
    print(res, flush=True)
    print(U.save("p8_certified", res))


if __name__ == "__main__":
    main()
