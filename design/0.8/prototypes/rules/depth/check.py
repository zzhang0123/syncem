"""Depth-expansion validity check of Spec Sec. 4.6 rule 4, as written and as proposed.

As written: x = tau_max h with tau_max at the lowest channel-support frequency, refuse when
x > 1. The printed factor is x^(N+1) / (N+1)!.

Proposed: every energy power keeps the depth powers s <= N_d (depth outside the total
degree). Then B = x^(N_d+1) / (N_d+1)! bounds the depth part of |dP| / I for any population
inside the depth cells, because |K_P| <= K_I, |azimuth| <= 1, |u_d| <= 1 and
|exp(i a) - sum_{s<=n} (i a)^s / s!| <= |a|^(n+1) / (n+1)! for real a. Refuse when B > tol.
With a total-degree lower set the rows of energy power r keep only s <= N - r, and the
depth error falls like x^1 at fixed energy columns (study.coupled_worst_case), so B is not
a bound there.
"""
from __future__ import annotations

import math


def spec_factor(x: float, N: int) -> float:
    return x ** (N + 1) / math.factorial(N + 1)


def spec_passes(x: float) -> bool:
    """Spec Sec. 4.6 rule 4 as written: DepthExpansionError above x = 1."""
    return x <= 1.0


def depth_power_by_row(lower: tuple) -> dict:
    """s_max(r): the largest depth power that the rows with energy power r keep."""
    by_r: dict = {}
    for r, s in lower:
        by_r[r] = max(by_r.get(r, 0), s)
    return by_r


def effective_depth_order(lower: tuple) -> int:
    """Smallest s_max(r) over the energy powers r. Equals N_d only when depth is outside the total."""
    return min(depth_power_by_row(lower).values())


def proposed_passes(x: float, n_d: int, tol=1e-3) -> bool:
    return spec_factor(x, n_d) <= tol


def x_limit(n_d: int, tol=1e-3) -> float:
    """Largest tau_max h that the proposed rule admits."""
    return (tol * math.factorial(n_d + 1)) ** (1.0 / (n_d + 1))
