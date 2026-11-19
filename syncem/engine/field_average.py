"""Field-average table S1 of the continuum stack (design/0.8/SPEC.md Sec. 8.2, 8.3).

LABEL ``eq: directional continuum``. ``T_W^(r)(y) = <B W^(r)(y / B)>_B`` for ``W = F, G`` and ``r = 0..order``, with
``y = nu / (a_1 s gamma^2)``, ``s = sqrt(1 - eta^2)`` and ``B`` in uG, averaged over the field law's nodes
``B_q`` and positive weights ``w_q``. It is stored as cubic splines in ``ln y`` of ``ln T`` and of the ratios
``tau_r = T^(r) / T`` on the study grid ``ln y in [ln 1e-5 - 3685 * 0.0025, ln 1e6]`` (13817 knots), extended
upward in steps of 0.0025 to ``ln(700 B_max)`` when the strongest field node exceeds about 1430 uG, so that ``T`` is
below ``e^-700`` of its scale above the grid for every law. The knot
values come from the quadrature of ``primitive.scaled_log_derivatives`` (``FieldAverage.quadrature``, the
oracle path) or from lookups in the primitive table S0 at ``ln y - ln B_q`` (``FieldAverage.lookups``):
``ln T = ln y + logsumexp_q (ln w_q - x_q + ln E_0(x_q))`` and ``tau_r = sum_q softmax_q f_r(x_q)``. Both port
``studies/classical-models/lib/basis.py`` (``KernelTable``).

``evaluate`` raises ``OutsideTable`` below the grid (an energy too high for the lowest channel) and above the table
order. Units: ``y`` dimensionless, ``B`` in uG inside the table. Shapes: ``evaluate(lny)`` returns ``order + 1``
arrays of the shape of ``lny``. Not certified: the spline error (3.4e-9 in ``ln T`` between knots) and the lookup
error (1.7e-13 in ``ln T`` against the quadrature table), measured in the step-5 tests (decision 140).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .primitive import (
    DEFAULT_CUTOFF,
    DEFAULT_NODES,
    PrimitiveTable,
    scaled_log_derivatives,
)

__all__ = [
    "FieldAverage",
    "OutsideTable",
    "TABLE_LNY",
    "TABLE_ORDER",
    "TABLE_STEP",
    "X_TOP",
]

TABLE_STEP = 0.0025
TABLE_LNY = (float(np.log(1e-5) - 3685 * TABLE_STEP), float(np.log(1e6)))
TABLE_ORDER = 6
_LNY = np.linspace(
    TABLE_LNY[0],
    TABLE_LNY[1],
    int(round((TABLE_LNY[1] - TABLE_LNY[0]) / TABLE_STEP)) + 1,
)
X_TOP = 700.0  # x = y / B at the grid's top for the strongest node: T is below e^-700 of its scale above it


class OutsideTable(ValueError):
    """A request outside the S1 table: ``ln y`` below its grid or an order above its order."""


def _grid(b_max_ug: float) -> np.ndarray:
    """The study grid, extended upward on the same spacing when ``ln(X_TOP b_max)`` lies above ``ln 1e6``."""
    top = float(np.log(X_TOP * b_max_ug))
    if top <= _LNY[-1]:
        return _LNY
    dx = _LNY[1] - _LNY[0]
    extra = int(np.ceil((top - _LNY[-1]) / dx))
    return np.concatenate([_LNY, _LNY[-1] + dx * np.arange(1, extra + 1)])


def _logsumexp(terms: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    top = np.max(terms, axis=-1, keepdims=True)
    lse = top[..., 0] + np.log(np.sum(np.exp(terms - top), axis=-1))
    return lse, np.exp(terms - lse[..., None])


@dataclass(frozen=True, eq=False)
class FieldAverage:
    """S1: splines of ``ln T_W`` and ``tau_{W,r}`` in ``ln y`` for the field law ``(nodes_uG, weights)``."""

    nodes_uG: np.ndarray
    weights: np.ndarray
    order: int
    method: str
    coef: dict
    lny: np.ndarray

    @staticmethod
    def _law(nodes_uG: Any, weights: Any) -> tuple[np.ndarray, np.ndarray]:
        b, w = np.atleast_1d(np.asarray(nodes_uG, float)), np.atleast_1d(
            np.asarray(weights, float)
        )
        if (
            b.shape != w.shape
            or b.ndim != 1
            or np.any(~np.isfinite(w))
            or np.any(w < 0)
        ):
            raise ValueError(
                "the field law needs non-negative weights of the length of its nodes"
            )
        keep = w > 0
        if not np.any(keep):
            raise ValueError("the field law has no positive weight")
        b, w = b[keep], w[keep]
        if np.any(~np.isfinite(b)) or np.any(b <= 0):
            raise ValueError(
                "the field law needs positive finite nodes in uG where its weight is positive"
            )
        return b, w

    @classmethod
    def _from_knots(
        cls,
        nodes: np.ndarray,
        weights: np.ndarray,
        order: int,
        method: str,
        knots: dict,
        lny: np.ndarray,
    ) -> "FieldAverage":
        from scipy.interpolate import CubicSpline  # noqa: PLC0415

        coef = {}
        for which, (ln_t, tau) in knots.items():
            rows = [CubicSpline(lny, ln_t).c] + [
                CubicSpline(lny, tau[:, r]).c for r in range(1, order + 1)
            ]
            coef[which] = np.stack(rows)
        return cls(nodes, weights, int(order), method, coef, lny)

    @classmethod
    def quadrature(
        cls,
        nodes_uG: Any,
        weights: Any,
        *,
        order: int = TABLE_ORDER,
        nodes: int = DEFAULT_NODES,
        cutoff: float = DEFAULT_CUTOFF,
    ) -> "FieldAverage":
        """The oracle path: knot values by quadrature at every field node (about 0.33 s per node)."""
        b, w = cls._law(nodes_uG, weights)
        lny = _grid(float(b.max()))
        knots = {}
        for which in ("F", "G"):
            knots[which] = cls.direct(
                np.exp(lny), b, w, which, order, nodes=nodes, cutoff=cutoff
            )
        return cls._from_knots(b, w, order, "quadrature", knots, lny)

    @classmethod
    def lookups(
        cls,
        nodes_uG: Any,
        weights: Any,
        *,
        order: int = TABLE_ORDER,
        primitive: PrimitiveTable | None = None,
    ) -> "FieldAverage":
        """Knot values by lookups in the primitive table at ``ln y - ln B_q`` (about 6 ms per node).

        A supplied ``primitive`` must cover ``needed_lnx(nodes_uG)`` at both ends and the order.
        """
        b, w = cls._law(nodes_uG, weights)
        lny = _grid(float(b.max()))
        lnx = lny[:, None] - np.log(b)[None, :]
        if primitive is None:
            lo, hi = cls.needed_lnx(b)
            primitive = PrimitiveTable.build(lo, hi, order=order)
        if (
            primitive.order < order
            or primitive.lnx0 > lnx.min() + 1e-12
            or primitive.lnx_hi < lnx.max() - 1e-12
        ):
            raise ValueError(
                "the primitive table does not cover the lookups: build it over needed_lnx(nodes) and the order"
            )
        x = np.exp(lnx)
        knots = {}
        for which in ("F", "G"):
            ln_e0, f = primitive.lookup(lnx, which, order)
            lse, soft = _logsumexp(np.log(w)[None, :] - x + ln_e0)
            knots[which] = (lny + lse, np.einsum("yq,yqr->yr", soft, f))
        return cls._from_knots(b, w, order, "lookup", knots, lny)

    @staticmethod
    def needed_lnx(nodes_uG: Any) -> tuple[float, float]:
        """The ``ln x`` range of S0 that the lookups of a law with these field nodes read, with a 0.01 margin."""
        b = np.atleast_1d(np.asarray(nodes_uG, float))
        if b.size == 0 or np.any(~np.isfinite(b)) or np.any(b <= 0):
            raise ValueError("needed_lnx takes positive finite field nodes in uG")
        lny = _grid(float(b.max()))
        return (
            float(lny[0] - np.log(b.max())) - 0.01,
            float(lny[-1] - np.log(b.min())) + 0.01,
        )

    @staticmethod
    def direct(
        y: Any,
        nodes_uG: np.ndarray,
        weights: np.ndarray,
        which: str,
        order: int,
        *,
        nodes: int = DEFAULT_NODES,
        cutoff: float = DEFAULT_CUTOFF,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Quadrature without splines: ``ln T_W(y)`` and ``tau`` ``(n_y, order + 1)``."""
        y = np.atleast_1d(np.asarray(y, float))
        x = y[:, None] / nodes_uG[None, :]
        ln_e0, f = scaled_log_derivatives(x, order, which, nodes=nodes, cutoff=cutoff)
        lse, soft = _logsumexp(np.log(weights)[None, :] - x + ln_e0)
        return np.log(y) + lse, np.einsum("yq,yqr->yr", soft, f)

    def evaluate(self, lny: Any, which: str, order: int) -> list[np.ndarray]:
        """``T^(r)(y)`` for ``r = 0..order`` at ``ln y``. ``T = 0`` above the table, an error below it."""
        if order > self.order:
            raise OutsideTable(f"order {order} exceeds the S1 table order {self.order}")
        lny = np.asarray(lny, float)
        grid = self.lny
        if np.any(lny < grid[0]):
            raise OutsideTable(
                f"y below the field-average table (ln y = {float(lny.min()):.4g} < {grid[0]:.4g}): the "
                "electron energy is too high for the lowest channel"
            )
        hi = lny > grid[-1]
        x0, dx = grid[0], grid[1] - grid[0]
        pos = np.clip((lny - x0) / dx, 0.0, grid.size - 1 - 1e-9)
        i = pos.astype(np.int64)
        t = (pos - i) * dx
        c = self.coef[which]

        def horner(k: int) -> np.ndarray:
            ck = c[k]
            return ((ck[0][i] * t + ck[1][i]) * t + ck[2][i]) * t + ck[3][i]

        tval = np.exp(horner(0))
        tval[hi] = 0.0
        return [tval] + [tval * horner(r) for r in range(1, order + 1)]
