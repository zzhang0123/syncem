"""Re-expansions of Taylor columns between coordinates (design/0.8/SPEC.md Sec. 8.1, 8.3).

LABEL ``eq: field scaling``. ``log_from_affine(N, h)`` is the lower-triangular map
``T[n, r] = h^n r! S(n, r) / n!`` (``S`` the Stirling numbers of the second kind) from the coefficients in the
affine coordinate ``z = (x - x_b)/x_b`` to those in the log chart ``u = ln(x / x_b)/h``, because
``z = exp(h u) - 1``.

``field_scaling_columns`` applies the field-scaling relation ``Kbar(gamma, L B) = L Kbar(L^(1/2) gamma, B)``
(PAPER eq: field scaling) to a ``(gamma, B)`` product cell with a log energy chart. With
``g(w) = Kbar(gamma_s e^(h w), B_0)`` at the shifted reference ``gamma_s = gamma_b (B_b / B_0)^(1/2)``, the cell
function is ``f(u, v) = (B(v) / B_0) g(u + t(v))`` with ``t(v) = ln(B(v) / B_b) / (2 h)``. In the log field chart
``B(v) = B_b e^(h_B v)``, in the affine chart ``B(v) = B_b + h_B v``. The coefficient of ``u^r v^s`` needs the
energy coefficients ``g_n`` for ``n <= r + s``.

Units: those of the columns. Shapes: ``T`` has shape ``(N + 1, N + 1)``. Columns are dicts from exponents to channel
arrays. Not certified: the relation holds for kernels that declare ``FieldScalingRelation`` under its
precondition (SPEC Sec. 8.1), which the caller checks.
"""

from __future__ import annotations

from math import comb, factorial
from typing import Any

import numpy as np

__all__ = ["field_scaling_columns", "log_from_affine", "stirling2"]


def stirling2(n_max: int) -> np.ndarray:
    """Stirling numbers of the second kind ``S[n, r]`` for ``n, r <= n_max``."""
    s = np.zeros((n_max + 1, n_max + 1))
    s[0, 0] = 1.0
    for n in range(1, n_max + 1):
        for r in range(1, n + 1):
            s[n, r] = r * s[n - 1, r] + s[n - 1, r - 1]
    return s


def log_from_affine(n_max: int, h: float) -> np.ndarray:
    """``T[n, r]``: log-chart coefficient ``n`` from affine coefficient ``r`` (``z = exp(h u) - 1``)."""
    s = stirling2(n_max)
    return np.array(
        [
            [h**n * factorial(r) * s[n, r] / factorial(n) for r in range(n_max + 1)]
            for n in range(n_max + 1)
        ]
    )


def _series_mul(a: np.ndarray, b: np.ndarray, order: int) -> np.ndarray:
    return np.convolve(a, b)[: order + 1]


def _field_series(
    chart: str, h_b: float, b_ref: float, order: int
) -> tuple[np.ndarray, np.ndarray]:
    """Taylor coefficients in ``v`` of ``B(v) / B_b`` and of ``ln(B(v) / B_b)``."""
    if chart == "log":
        ratio = np.array([h_b**k / factorial(k) for k in range(order + 1)])
        log = np.zeros(order + 1)
        if order >= 1:
            log[1] = h_b
        return ratio, log
    eps = h_b / b_ref
    ratio = np.zeros(order + 1)
    ratio[0] = 1.0
    if order >= 1:
        ratio[1] = eps
    log = np.array([0.0] + [(-1) ** (k + 1) * eps**k / k for k in range(1, order + 1)])
    return ratio, log


def field_scaling_columns(
    energy: dict[int, Any],
    exponents: list[tuple[int, int]],
    *,
    h_gamma: float,
    field_chart: str,
    field_ref: float,
    field_half_width: float,
    field_basis: float,
) -> dict:
    """Columns ``(r, s)`` of a ``(gamma, B)`` cell from the energy columns ``g_n`` at the shifted reference.

    ``energy[n]`` holds ``g_n``, the coefficient of ``w^n`` of ``Kbar(gamma_s e^(h w), B_0)`` with
    ``gamma_s = gamma_b (B_b / B_0)^(1/2)`` and ``B_0 = field_basis``.
    """
    s_max = max(s for _, s in exponents)
    need = max(r + s for r, s in exponents)
    missing = [n for n in range(need + 1) if n not in energy]
    if missing:
        raise ValueError(
            f"energy columns of orders {missing} are needed for the field cell"
        )
    ratio, log = _field_series(field_chart, field_half_width, field_ref, s_max)
    t = log / (2.0 * h_gamma)
    scale = field_ref / field_basis
    powers = [np.eye(1, s_max + 1, 0).ravel()]
    for _ in range(need):
        powers.append(_series_mul(powers[-1], t, s_max))
    out = {}
    for r, s in exponents:
        total = 0.0
        for n in range(r, r + s + 1):
            coef = scale * comb(n, r) * _series_mul(ratio, powers[n - r], s_max)[s]
            if coef != 0.0:
                total = total + coef * energy[n]
        out[(r, s)] = total
    return out
