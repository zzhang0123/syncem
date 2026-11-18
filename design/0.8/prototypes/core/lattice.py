"""Cells on one variable, product-cell tables and the lower-set Taylor index.

Spec Sec. 3.3, 4.1, 4.3, 4.4. A chart maps x to the local coordinate u of a cell:
``log``: u = ln(x / x_b) / h_b, ``affine``: u = (x - x_b) / h_b. Cells and rows are
numbered from zero. Rows of a product cell are its Taylor exponents r with
r_v <= N_v per variable and sum_v r_v <= total, in graded lexicographic order
(higher power of the first variable first within one degree).
"""
from __future__ import annotations

import itertools
from dataclasses import dataclass, replace

import jax.numpy as jnp
import numpy as np

CHARTS = ("log", "affine")


def to_u(chart: str, x, ref, hw):
    """Local coordinate u of x in a cell with reference ref and half-width hw."""
    if chart == "log":
        return jnp.log(x / ref) / hw
    return (x - ref) / hw


def from_u(chart: str, u, ref, hw):
    """Inverse of :func:`to_u`."""
    if chart == "log":
        return ref * jnp.exp(hw * u)
    return ref + hw * u


def lower_set(caps: tuple[int, ...], total: int) -> tuple[tuple[int, ...], ...]:
    """Exponents r with 0 <= r_v <= caps_v and sum r_v <= total, graded lexicographic."""
    ranges = [range(c + 1) for c in caps]
    out = [r for r in itertools.product(*ranges) if sum(r) <= total]
    return tuple(sorted(out, key=lambda r: (sum(r), tuple(-v for v in r))))


@dataclass(frozen=True, eq=False)
class Cells:
    """A partition of one variable's range, each cell with reference, half-width and order."""

    chart: str
    edges: np.ndarray | None  # (n + 1,) or None for one unbounded cell
    reference: np.ndarray  # (n,)
    half_width: np.ndarray  # (n,), in the chart
    order: tuple[int, ...]  # (n,)

    @classmethod
    def from_edges(cls, edges, *, chart="log", order=2, reference=None) -> "Cells":
        if chart not in CHARTS:
            raise ValueError(f"chart must be one of {CHARTS}")
        e = np.asarray(edges, float)
        if e.ndim != 1 or e.size < 2 or np.any(np.diff(e) <= 0):
            raise ValueError("edges must be a strictly increasing 1-D array of length >= 2")
        if chart == "log" and e[0] <= 0:
            raise ValueError("log chart needs positive edges")
        t = np.log(e) if chart == "log" else e
        hw = 0.5 * np.diff(t)
        centre = 0.5 * (t[1:] + t[:-1])
        ref = (np.exp(centre) if chart == "log" else centre) if reference is None else np.asarray(reference, float)
        n = e.size - 1
        orders = (int(order),) * n if np.ndim(order) == 0 else tuple(int(o) for o in order)
        if ref.shape != (n,) or len(orders) != n:
            raise ValueError("reference and order must have one entry per cell")
        return cls(chart, e, ref, hw, orders)

    @classmethod
    def log(cls, lo, hi, n, *, order=2) -> "Cells":
        """n cells equal in ln x on [lo, hi] (edges as the study's config.log_edges)."""
        return cls.from_edges(np.exp(np.linspace(np.log(lo), np.log(hi), n + 1)), chart="log", order=order)

    @classmethod
    def linear(cls, lo, hi, n, *, order=2) -> "Cells":
        return cls.from_edges(np.linspace(lo, hi, n + 1), chart="affine", order=order)

    @classmethod
    def single(cls, reference, scale, *, order=2, chart="affine") -> "Cells":
        """One cell without edges. Its |u| is not bounded by one."""
        return cls(chart, None, np.array([float(reference)]), np.array([float(scale)]), (int(order),))

    @property
    def n(self) -> int:
        return self.reference.size

    @property
    def lo(self) -> np.ndarray:
        if self.edges is None:
            return np.array([0.0 if self.chart == "log" else -np.inf])
        return self.edges[:-1]

    @property
    def hi(self) -> np.ndarray:
        return np.array([np.inf]) if self.edges is None else self.edges[1:]

    def with_order(self, N) -> "Cells":
        orders = (int(N),) * self.n if np.ndim(N) == 0 else tuple(int(o) for o in N)
        return replace(self, order=orders)


@dataclass(frozen=True, eq=False)
class ProductTable:
    """Product cells over the celled variables, one row per product cell.

    Arrays have shape (n_cells, d). ``chart`` and ``order`` are per row and variable.
    ``rows`` are the lattice rows: ``row_cell`` (n_rows,) and ``row_exp`` (n_rows, d).
    """

    variables: tuple[str, ...]
    index: np.ndarray  # (n_cells, d) cell number per variable
    lo: np.ndarray
    hi: np.ndarray
    reference: np.ndarray
    half_width: np.ndarray
    chart: tuple[tuple[str, ...], ...]
    order: tuple[tuple[int, ...], ...]
    total: int

    @classmethod
    def tensor(cls, axes: dict[str, Cells], total: int | None = None) -> "ProductTable":
        names = tuple(axes)
        cells = [axes[v] for v in names]
        total = max(max(c.order) for c in cells) if total is None else int(total)
        combos = list(itertools.product(*[range(c.n) for c in cells]))
        idx = np.array(combos, dtype=int).reshape(len(combos), len(names))

        def col(attr):
            return np.stack([getattr(c, attr)[idx[:, v]] for v, c in enumerate(cells)], axis=1)

        return cls(names, idx, col("lo"), col("hi"), col("reference"), col("half_width"),
                   tuple(tuple(c.chart for c in cells) for _ in combos),
                   tuple(tuple(c.order[i] for c, i in zip(cells, row)) for row in combos), total)

    def select(self, mask) -> "ProductTable":
        """Subset of product cells (a general table)."""
        keep = np.flatnonzero(np.asarray(mask))
        return replace(self, index=self.index[keep], lo=self.lo[keep], hi=self.hi[keep],
                       reference=self.reference[keep], half_width=self.half_width[keep],
                       chart=tuple(self.chart[i] for i in keep), order=tuple(self.order[i] for i in keep))

    @property
    def n_cells(self) -> int:
        return self.index.shape[0]

    def exponents(self, c: int) -> tuple[tuple[int, ...], ...]:
        return lower_set(self.order[c], self.total)

    def signature(self, c: int) -> tuple:
        """Cells with equal signature share one compiled column routine."""
        return (self.chart[c], self.exponents(c))

    @property
    def offsets(self) -> np.ndarray:
        sizes = [len(self.exponents(c)) for c in range(self.n_cells)]
        return np.concatenate([[0], np.cumsum(sizes)]).astype(int)

    @property
    def n_rows(self) -> int:
        return int(self.offsets[-1])

    @property
    def row_cell(self) -> np.ndarray:
        return np.repeat(np.arange(self.n_cells), np.diff(self.offsets))

    @property
    def row_exp(self) -> np.ndarray:
        return np.array([r for c in range(self.n_cells) for r in self.exponents(c)], dtype=int)

    def locate(self, points: dict[str, np.ndarray]) -> np.ndarray:
        """Product cell of each point (first match, closed intervals), -1 outside every cell."""
        x = np.stack([np.asarray(points[v], float) for v in self.variables], axis=-1)  # (M, d)
        inside = np.all((x[:, None, :] >= self.lo[None]) & (x[:, None, :] <= self.hi[None]), axis=-1)
        return np.where(inside.any(axis=1), inside.argmax(axis=1), -1)

    def labels(self) -> tuple[str, ...]:
        out = []
        for c in range(self.n_cells):
            cell = ",".join(f"{v}={i}" for v, i in zip(self.variables, self.index[c]))
            for r in self.exponents(c):
                out.append(f"[{cell} | u^{r}]")
        return tuple(out)
