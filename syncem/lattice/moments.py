"""Moments: an element of the dual of a lattice, in lattice rows (SPEC Sec. 3.3, 4.5).

LABEL ``eq: energy cell response``. ``values`` holds the scaled moments ``a_alpha = (1 / w_c) int psi_alpha dN``
of product cell ``c``, with ``dN`` the column measure. The moments carry the electron column: there is no unit
moment and no separate amplitude. ``index`` and ``key`` are static, so a ``vmap`` over values does not broadcast
cell arrays. ``origin`` and ``masked`` travel with the values (SPEC Sec. 6.3).

Units: cm^-2 for a column measure. Shapes: ``values`` has shape ``(..., n)``. Not certified: the moments are whatever
the caller supplies, and realisability is a separate check.
"""

from __future__ import annotations

import operator
from typing import Any

import equinox as eqx
import numpy as np

from .index import MomentIndex

__all__ = ["ORIGINS", "Moments"]

ORIGINS = ("specified", "family", "minimum_norm", "penalised", "nonnegative_penalised")


class Moments(eqx.Module):
    """Scaled moments in the rows of one lattice, with the lattice key."""

    values: Any
    index: MomentIndex = eqx.field(static=True)
    key: str | None = eqx.field(static=True)
    masked: tuple[int, ...] = eqx.field(static=True)
    origin: str = eqx.field(static=True)
    retained: str | None = eqx.field(static=True)
    component: str | None = eqx.field(static=True)

    def __init__(
        self,
        lattice: Any,
        values: Any,
        *,
        origin: str = "specified",
        masked: tuple[int, ...] = (),
        retained: str | None = None,
        component: str | None = None,
    ):
        if origin not in ORIGINS:
            raise ValueError(f"origin must be one of {ORIGINS}, not {origin!r}")
        if not isinstance(getattr(lattice, "index", None), MomentIndex):
            raise TypeError(f"Moments needs a Lattice, not {type(lattice).__name__}")
        n = lattice.index.n
        if np.shape(values)[-1:] != (n,):
            raise ValueError(
                f"moment values must have last axis {n} (the lattice rows), not shape {np.shape(values)}"
            )
        try:
            if any(isinstance(k, (bool, np.bool_)) for k in masked):
                raise TypeError("bool")
            masked = tuple(operator.index(k) for k in masked)
        except TypeError as err:
            raise ValueError(f"masked rows must be integers, not {masked!r}") from err
        if any(not 0 <= k < n for k in masked):
            raise ValueError("masked rows must be lattice rows")
        self.values = values
        self.index = lattice.index
        self.key = lattice.key
        self.masked = tuple(sorted({int(k) for k in masked}))
        self.origin = origin
        self.retained = retained
        self.component = component

    def table(self, limit: int = 40) -> str:
        """Rows with labels and values (the first vector of a batch). Long tables show both ends."""
        values = np.asarray(self.values).reshape(-1, self.index.n)[0]
        rows = range(self.index.n)
        if self.index.n > limit:
            rows = [*range(limit // 2), *range(self.index.n - limit // 2, self.index.n)]
        lines = [
            f"Moments ({self.origin}), {self.index.n} rows, key {str(self.key)[:12]}"
        ]
        previous = -1
        for k in rows:
            if k != previous + 1:
                lines.append("  ...")
            mark = "  masked" if k in self.masked else ""
            lines.append(f"  {self.index.label(k):<40} {values[k]: .6e}{mark}")
            previous = k
        return "\n".join(lines)
