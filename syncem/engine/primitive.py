"""Primitive table S0: ``F``, ``G`` and their log-derivative stack on a grid in ``ln x`` (SPEC Sec. 8.2, 8.3).

LABEL ``eq: directional continuum``. For ``W(x) = x int_0^inf exp(-x cosh u) w(u) du`` with
``w_F(u) = cosh(5u/3)/cosh u`` (``F``) and ``w_G(u) = cosh(2u/3)`` (``G``), the stack is
``W^(r) = (-x d/dx)^r W = x int exp(-z) P_r(z) w(u) du`` with ``z = x cosh u``, ``P_0 = 1`` and
``P_(r+1) = (z - 1) P_r - z P_r'``. It is held in the log-safe form ``W^(r) = x e^-x E_0(x) f_r(x)`` with
``E_0 = int exp(-x (cosh u - 1)) w(u) du > 0`` and ``f_r = E_r / E_0``. This is a port of
``studies/classical-models/lib/kernels.py`` (Gauss-Legendre on ``u in [0, u_max]``,
``x (cosh u_max - 1) = 70``, 200 nodes).

``PrimitiveTable`` (S0) holds cubic splines of ``ln E_0`` and ``f_r`` for ``r <= order`` on knots
``k * step`` of ``ln x``. The factor ``x e^-x`` is applied exactly at lookup. S1 tables are built from it by
lookups at ``ln y - ln B_q`` (``field_average.FieldAverage.lookups``). SciPy is imported on first use.
``quantised_span`` gives the span of the S0 table that a need in ``ln x`` reads (decision 158): its union with
``_S0_SPAN``, rounded outward to multiples of ``SPAN_QUANTUM`` e-folds on a grid anchored at ``ln x = 0``. The S0
bits, and with them the S1 bits, then depend on the need alone and not on the laws read before.

Units: ``x`` dimensionless. Shapes: ``lookup(lnx)`` returns ``ln E_0`` of the shape of ``lnx`` and ``f`` with a
trailing axis ``order + 1``. Not certified: the quadrature in ``u`` (its truncated tail is below ``e^-70``
times a polynomial) and the spline error, validated by the step-5 tests against the quadrature.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Any

import numpy as np
from numpy.polynomial import polynomial as npoly

__all__ = [
    "DEFAULT_CUTOFF",
    "DEFAULT_NODES",
    "MAX_ORDER",
    "PrimitiveTable",
    "SPAN_QUANTUM",
    "derivative_polynomials",
    "quantised_span",
    "scaled_log_derivatives",
]

MAX_ORDER = 8
DEFAULT_NODES = 200
DEFAULT_CUTOFF = 70.0
# The ends of every S0 span are multiples of SPAN_QUANTUM e-folds from ln x = 0 (decision 158).
SPAN_QUANTUM = 10.0
# Every S0 span covers this ln x range.
_S0_SPAN = (float(np.log(1e-12)), float(np.log(1e7)))
_WHICH = ("F", "G")


def quantised_span(lo: float, hi: float) -> tuple[float, float]:
    """The S0 span of the need ``[lo, hi]`` in ``ln x`` (decision 158).

    The span is the union of the need and ``_S0_SPAN``, rounded outward to multiples of ``SPAN_QUANTUM``. An end at a
    multiple stays. Raises ``ValueError`` for an empty or non-finite need.
    """
    lo, hi = float(lo), float(hi)
    if not (np.isfinite(lo) and np.isfinite(hi) and lo < hi):
        raise ValueError(f"an S0 need takes finite ends with lo < hi, not ({lo}, {hi})")
    lo, hi = min(lo, _S0_SPAN[0]), max(hi, _S0_SPAN[1])
    out = (
        float(SPAN_QUANTUM * np.floor(lo / SPAN_QUANTUM)),
        float(SPAN_QUANTUM * np.ceil(hi / SPAN_QUANTUM)),
    )
    if not (out[0] <= lo and out[1] >= hi):
        raise ValueError(
            f"the need ({lo}, {hi}) is too large to round to multiples of {SPAN_QUANTUM}"
        )
    return out


@lru_cache(maxsize=4)
def derivative_polynomials(order: int) -> tuple[np.ndarray, ...]:
    """Coefficients (ascending powers of ``z``) of ``P_0 .. P_order``."""
    polys = [np.array([1.0])]
    for _ in range(order):
        p = polys[-1]
        polys.append(
            npoly.polysub(
                npoly.polymul(p, [-1.0, 1.0]),
                npoly.polymul(npoly.polyder(p), [0.0, 1.0]),
            )
        )
    return tuple(polys)


@lru_cache(maxsize=4)
def _gauss(nodes: int) -> tuple[np.ndarray, np.ndarray]:
    return np.polynomial.legendre.leggauss(nodes)


def _weight(u: np.ndarray, which: str) -> np.ndarray:
    if which == "F":
        return np.cosh(5.0 * u / 3.0) / np.cosh(u)
    if which == "G":
        return np.cosh(2.0 * u / 3.0)
    raise ValueError(f"which must be one of {_WHICH}, not {which!r}")


def scaled_log_derivatives(
    x: Any,
    order: int = 0,
    which: str = "F",
    nodes: int = DEFAULT_NODES,
    cutoff: float = DEFAULT_CUTOFF,
) -> tuple[np.ndarray, np.ndarray]:
    """``(ln E_0(x), f(x))`` with ``W^(r) = x e^-x E_0 f_r``, ``f`` of shape ``x.shape + (order + 1,)``."""
    xx = np.asarray(x, dtype=float)
    if np.any(~np.isfinite(xx)) or np.any(xx <= 0):
        raise ValueError("kernel arguments must be finite and positive")
    if not 0 <= order <= MAX_ORDER:
        raise ValueError(f"order must lie in 0..{MAX_ORDER}")
    _weight(np.zeros(1), which)
    flat = xx.ravel()
    q, wq = _gauss(nodes)
    polys = derivative_polynomials(order)
    ln_e0 = np.empty(flat.size)
    ratios = np.empty((flat.size, order + 1))
    for s in range(0, flat.size, 2048):
        xv = flat[s : s + 2048, None]
        umax = np.arccosh(1.0 + cutoff / xv)
        u = 0.5 * (q + 1.0) * umax
        z = xv * np.cosh(u)
        base = (
            np.exp(-2.0 * xv * np.sinh(0.5 * u) ** 2)
            * _weight(u, which)
            * wq
            * (0.5 * umax)
        )
        e0 = base.sum(axis=1)
        ln_e0[s : s + 2048] = np.log(e0)
        ratios[s : s + 2048, 0] = 1.0
        for r in range(1, order + 1):
            ratios[s : s + 2048, r] = (base * npoly.polyval(z, polys[r])).sum(
                axis=1
            ) / e0
    return ln_e0.reshape(xx.shape), ratios.reshape(xx.shape + (order + 1,))


@dataclass(frozen=True, eq=False)
class PrimitiveTable:
    """S0: cubic splines of ``ln E_0`` and ``f_r`` in ``ln x`` for ``F`` and ``G``."""

    lnx0: float
    step: float
    n: int
    order: int
    coef: dict

    @classmethod
    def build(
        cls,
        lnx_lo: float = float(np.log(1e-12)),
        lnx_hi: float = float(np.log(750.0)),
        *,
        order: int = 6,
        step: float = 0.0025,
    ) -> "PrimitiveTable":
        """Splines on the knots ``k * step`` that cover ``[lnx_lo, lnx_hi]``."""
        from scipy.interpolate import CubicSpline  # noqa: PLC0415

        if not lnx_lo < lnx_hi or not step > 0:
            raise ValueError("PrimitiveTable needs lnx_lo < lnx_hi and a positive step")
        k_lo, k_hi = int(np.floor(lnx_lo / step)), int(np.ceil(lnx_hi / step))
        lnx = np.arange(k_lo, k_hi + 1) * step
        coef = {}
        for which in _WHICH:
            ln_e0, f = scaled_log_derivatives(np.exp(lnx), order, which)
            rows = [CubicSpline(lnx, ln_e0).c] + [
                CubicSpline(lnx, f[:, r]).c for r in range(1, order + 1)
            ]
            coef[which] = np.stack(rows)
        return cls(float(lnx[0]), float(step), int(lnx.size), int(order), coef)

    @property
    def lnx_hi(self) -> float:
        return self.lnx0 + (self.n - 1) * self.step

    def lookup(self, lnx: Any, which: str, order: int) -> tuple[np.ndarray, np.ndarray]:
        """``(ln E_0, f)`` at ``ln x``. Raises outside the table."""
        if order > self.order:
            raise ValueError(f"order {order} exceeds the table order {self.order}")
        lnx = np.asarray(lnx, dtype=float)
        if np.any(lnx < self.lnx0 - 1e-12) or np.any(lnx > self.lnx_hi + 1e-12):
            raise ValueError(
                f"ln x in [{float(lnx.min()):.4g}, {float(lnx.max()):.4g}] leaves the primitive table "
                f"[{self.lnx0:.4g}, {self.lnx_hi:.4g}]"
            )
        pos = np.clip((lnx - self.lnx0) / self.step, 0.0, self.n - 1 - 1e-9)
        i = pos.astype(np.int64)
        t = (pos - i) * self.step
        c = self.coef[which]

        def horner(k: int) -> np.ndarray:
            ck = c[k]
            return ((ck[0][i] * t + ck[1][i]) * t + ck[2][i]) * t + ck[3][i]

        ln_e0 = horner(0)
        f = np.stack(
            [np.ones_like(ln_e0)] + [horner(r) for r in range(1, order + 1)], axis=-1
        )
        return ln_e0, f
