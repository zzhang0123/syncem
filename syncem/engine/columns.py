"""The column map from lattice rows to basis columns (design/0.8/SPEC.md Sec. 3.4, 4.3, decisions 146 to 148).

LABEL ``eq: angular parity``. ``column_map(lattice, kernel, rules)`` decides which lattice rows have a basis
column, which rows share one column and which stored spectrum each column repeats. The decisions come from
static information only: the kernel's ``Parity`` and ``Beaming`` capabilities, its ``depth``, ``azimuth``
and ``stokes``, the Stokes request and the ``ColumnRules``. Until the plan of step 7, the caller passes the
rules explicitly. ``ColumnRules()`` drops and merges nothing (decision 148). Every dropped or merged row is
listed in ``reasons`` with the pass that decided it.

``ColumnMap`` holds integer NumPy arrays and no per-row objects, so G15 ``lat8`` (2 737 152 rows) maps in
integer arithmetic (decision 147). ``apply`` sums lattice rows into columns (``S a``), ``lift`` forms
``D S`` without ``S``, and ``matrix`` gives ``S`` as a SciPy sparse array, with SciPy imported lazily.
``key`` hashes the index key (no weights), the column capability record, the Stokes request and the rules
record. It is ``None`` when the lattice key is ``None`` (decision 146).

Units: dimensionless integers. Array shapes: ``col_of_row`` ``(n_rows,)``, ``col_start`` ``(n + 1,)``,
``column_block``, ``column_cell``, ``column_base``, ``spin0_from`` and ``spin2_from`` ``(n,)``,
``spin0_base`` ``(n0,)``, ``spin2_base`` ``(n2,)``. ``apply`` maps ``(..., n_rows)`` to ``(..., n)`` and
``lift`` maps ``(..., n)`` to ``(..., n_rows)``. Not certified: whether a rule holds for the averaged laws.
A parity mask is exact only under the reversal precondition of Sec. 4.3, which the plan of step 7 checks.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field, replace
from functools import cached_property
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from ..base._fingerprint import digest, encode
from ..base.capabilities import Beaming, Parity
from ._column_passes import REASONS, STOKES, build, capability, context, stokes_request

if TYPE_CHECKING:
    from ..lattice.lattice import Lattice

__all__ = [
    "REASONS",
    "ColumnMap",
    "ColumnRules",
    "column_capability_record",
    "column_map",
]

MASK_MODES = ("joint", "single")
SYMMETRIES = ("pitch_reversal",)


def _pair(value: Any, what: str) -> tuple[str, str]:
    pair = tuple(value)
    if len(pair) != 2 or not all(isinstance(x, str) for x in pair):
        raise ValueError(f"{what} must be a pair of names, not {value!r}")
    return pair


def _check_stokes(stokes: Any) -> tuple[str, ...] | None:
    if stokes is None:
        return None
    request = tuple(stokes)
    if (
        not request
        or len(set(request)) != len(request)
        or not set(request) <= set(STOKES)
    ):
        raise ValueError(
            f"the Stokes request must name distinct blocks among {STOKES}, not {stokes!r}"
        )
    return request


def _check_masks(masks: Any) -> tuple[tuple[str, str], ...]:
    out = tuple(_pair(m, "a mask") for m in masks)
    for cosine, mode in out:
        if mode not in MASK_MODES:
            raise ValueError(f'mask mode must be "joint" or "single", not {mode!r}')
    names = [c for c, _ in out]
    if len(set(names)) != len(names):
        raise ValueError(f"the masks name a cosine twice: {names}")
    modes = {m for _, m in out}
    if len(modes) > 1:
        raise ValueError(f"the masks mix the modes {sorted(modes)}")
    if modes == {"single"} and len(out) != 1:
        raise ValueError('a "single" mask names one cosine')
    if modes == {"joint"} and len(out) != 2:
        raise ValueError('a "joint" mask names the two cosines of a block')
    return out


@dataclass(frozen=True)
class ColumnRules:
    """Static column rules passed by the caller until the plan of step 7 (decision 148). LABEL ``eq: angular parity``.

    ``stokes``: the requested blocks, ``None`` for the kernel's own. ``masks``: (cosine, "single") for one
    resolved cosine with an averaged partner, or two (cosine, "joint") entries for two resolved cosines.
    ``merge``: the ``Beaming`` pair whose ``(l, k)`` and ``(k, l)`` columns merge. ``depth_free_spin0``: I and
    V do not depend on the kernel's depth variable. ``symmetries``: "pitch_reversal" drops the odd degrees
    of the ``Beaming`` variable. Not certified: the rules are the caller's statement.
    """

    stokes: tuple[str, ...] | None = None
    masks: tuple[tuple[str, str], ...] = ()
    merge: tuple[str, str] | None = None
    depth_free_spin0: bool = False
    symmetries: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "stokes", _check_stokes(self.stokes))
        object.__setattr__(self, "masks", _check_masks(self.masks))
        if self.merge is not None:
            merge = _pair(self.merge, "merge")
            if merge[0] == merge[1]:
                raise ValueError(f"merge needs two distinct variables, not {merge}")
            object.__setattr__(self, "merge", merge)
        if not isinstance(self.depth_free_spin0, bool):
            raise ValueError(
                f"depth_free_spin0 must be a bool, not {self.depth_free_spin0!r}"
            )
        symmetries = tuple(self.symmetries)
        unknown = [s for s in symmetries if s not in SYMMETRIES]
        if unknown:
            raise ValueError(
                f"unknown symmetries {unknown}. The column map reads {SYMMETRIES}"
            )
        object.__setattr__(self, "symmetries", symmetries)

    def record(self) -> bytes:
        """Canonical bytes of the rules, part of ``ColumnMap.key``."""
        fields = (self.stokes, self.masks, self.merge, self.depth_free_spin0)
        return encode(("column rules", *fields, self.symmetries))


class _ColumnRows:
    """Lazy view of ``col_rows``: ``rows[j]`` is the array of lattice rows summed into column ``j``."""

    def __init__(self, start: np.ndarray, rows: np.ndarray) -> None:
        self._start, self._rows = start, rows

    def __len__(self) -> int:
        return self._start.size - 1

    def __getitem__(self, j: int) -> np.ndarray:
        j = int(j) + (len(self) if j < 0 else 0)
        if not 0 <= j < len(self):
            raise IndexError(f"column {j} outside 0..{len(self) - 1}")
        return self._rows[self._start[j] : self._start[j + 1]]


def _array_module(values: Any) -> Any:
    jax = sys.modules.get("jax")
    if jax is not None and isinstance(values, jax.Array):
        import jax.numpy as jnp

        return jnp
    return np


def _frozen(array: Any) -> np.ndarray:
    out = np.array(array, dtype=np.int64)
    out.setflags(write=False)
    return out


@dataclass(frozen=True, eq=False)
class ColumnMap:
    """Exact selection and summation from lattice rows to basis columns (SPEC Sec. 3.4). LABEL ``eq: angular parity``.

    Integer arrays of shape ``(n_rows,)``, ``(n + 1,)``, ``(n,)``, ``(n0,)`` and ``(n2,)``. Equality and hash
    follow ``key`` when both keys are set. Not certified: the rules it was built from.
    """

    col_of_row: np.ndarray
    col_start: np.ndarray
    col_rows: np.ndarray
    column_block: np.ndarray
    column_cell: np.ndarray
    column_base: np.ndarray
    uses: Mapping[str, np.ndarray]
    spin0_from: np.ndarray
    spin2_from: np.ndarray
    spin0_base: np.ndarray
    spin2_base: np.ndarray
    reasons: tuple[tuple[str, np.ndarray], ...]
    key: str | None = None
    stokes: tuple[str, ...] = field(default=STOKES)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ColumnMap):
            return NotImplemented
        if self.key is None or other.key is None:
            return self is other
        return self.key == other.key

    def __hash__(self) -> int:
        return hash(self.key) if self.key is not None else object.__hash__(self)

    @property
    def n(self) -> int:
        return int(self.col_start.size - 1)

    @property
    def n_rows(self) -> int:
        return int(self.col_of_row.size)

    @property
    def n0(self) -> int:
        return int(self.spin0_base.size)

    @property
    def n2(self) -> int:
        return int(self.spin2_base.size)

    @cached_property
    def masked(self) -> np.ndarray:
        """Lattice rows without a column, the ``Moments.masked`` rows."""
        return _frozen(np.flatnonzero(self.col_of_row < 0))

    @property
    def rows(self) -> _ColumnRows:
        return _ColumnRows(self.col_start, self.col_rows)

    @cached_property
    def _extra(self) -> tuple[tuple[np.ndarray, np.ndarray], ...]:
        """(columns, rows) of the second, third, ... row of every column with more than one row."""
        counts = np.diff(self.col_start)
        top = int(counts.max()) if counts.size else 0
        out = []
        for m in range(1, top):
            cols = np.flatnonzero(counts > m)
            out.append((cols, self.col_rows[self.col_start[cols] + m]))
        return tuple(out)

    def _check_last(self, values: Any, size: int, what: str) -> None:
        if values.shape[-1:] != (size,):
            raise ValueError(
                f"{what} needs a last axis of {size}, not shape {tuple(values.shape)}"
            )

    def apply(self, values: Any) -> Any:
        """``S a``: values in lattice rows to column space, ``(..., n_rows) -> (..., n)``."""
        xp = _array_module(values)
        v = xp.asarray(values)
        self._check_last(v, self.n_rows, "apply")
        out = v[..., self.col_rows[self.col_start[:-1]]]
        for cols, rows in self._extra:
            if xp is np:
                out[..., cols] += v[..., rows]
            else:
                out = out.at[..., cols].add(v[..., rows])
        return out

    def lift(self, D: Any) -> Any:
        """``D S`` without forming ``S``, ``(..., n) -> (..., n_rows)``. Rows without a column are zero."""
        xp = _array_module(D)
        d = xp.asarray(D)
        self._check_last(d, self.n, "lift")
        if self.n == 0:
            return xp.zeros(d.shape[:-1] + (self.n_rows,), dtype=d.dtype)
        gathered = d[..., np.maximum(self.col_of_row, 0)]
        return xp.where(self.col_of_row >= 0, gathered, xp.zeros((), dtype=d.dtype))

    def matrix(self) -> Any:
        """``S`` as a SciPy ``csr_array`` of shape ``(n, n_rows)`` with entries 1."""
        from scipy.sparse import csr_array

        data = np.ones(self.col_rows.size)
        return csr_array(
            (data, self.col_rows, self.col_start), shape=(self.n, self.n_rows)
        )

    def labels(self, index: Any) -> tuple[str, ...]:
        """Column labels from the lattice index. A merged column reads "a + b"."""
        if index.n != self.n_rows:
            raise ValueError(
                f"the index has {index.n} rows and the column map {self.n_rows}"
            )
        rows = self.rows
        return tuple(
            " + ".join(index.label(int(k)) for k in rows[j]) for j in range(self.n)
        )


def column_capability_record(kernel: Any) -> bytes:
    """The capabilities that change columns, as data: ``Parity``, ``Beaming``, depth, azimuth, Stokes.

    ``EnergyWindow`` and ``ReferenceDensity`` hold functions and do not change columns, so they are left
    out (decision 146).
    """
    parity, beaming = capability(kernel, Parity), capability(kernel, Beaming)
    return encode(
        (
            "column capabilities",
            None if parity is None else (tuple(parity.even), tuple(parity.odd)),
            None if beaming is None else (beaming.variable, beaming.partner),
            kernel.depth,
            kernel.azimuth,
            tuple(kernel.stokes),
        )
    )


def _key(lattice: Any, kernel: Any, rules: ColumnRules, request: tuple) -> str | None:
    if lattice.key is None:
        return None
    return digest(
        (
            "column map",
            lattice.index.key,
            column_capability_record(kernel),
            request,
            replace(rules, stokes=request).record(),
        )
    )


def column_map(
    lattice: "Lattice",
    kernel: Any,
    rules: ColumnRules = ColumnRules(),
    *,
    max_columns: int = 20_000,
) -> ColumnMap:
    """The column map of ``lattice`` for ``kernel`` under ``rules``. ``SizeError`` above ``max_columns``."""
    if not isinstance(rules, ColumnRules):
        raise TypeError(f"rules must be ColumnRules, not {type(rules).__name__}")
    request = stokes_request(kernel, rules.stokes)
    arrays = build(context(lattice, kernel, rules))
    lattice.check_size(int(arrays["col_start"].size - 1), max_columns=max_columns)
    uses = MappingProxyType({s: _frozen(a) for s, a in arrays.pop("uses").items()})
    reasons = tuple((r, _frozen(rows)) for r, rows in arrays.pop("reasons"))
    return ColumnMap(
        **{name: _frozen(a) for name, a in arrays.items()},
        uses=uses,
        reasons=reasons,
        key=_key(lattice, kernel, rules, request),
        stokes=request,
    )
