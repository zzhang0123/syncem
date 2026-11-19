"""Checks and vectorised passes of the column map (design/0.8/SPEC.md Sec. 4.3, decisions 147 and 148).

LABEL ``eq: angular parity``. Private helpers of ``syncem.engine.columns``. Each pass works on the row arrays
of ``MomentIndex.arrays`` and on a use mask of shape ``(3, n_rows)`` over the Stokes blocks I, P and V, and
returns new arrays. The passes run in this order: feeds (M0 feeds I and V, the spin-2 blocks feed P, and
without ``AzimuthHarmonic`` M0 also feeds P), depth (I and V leave the ``s > 0`` rows), parity masks,
symmetry, merge of ``Beaming`` pairs, and numbering. A row without any use is dropped with the reason of the
pass that removed its last use. Variable names come from the kernel (``kernel.depth``, ``Beaming``,
``Parity``) and from the lattice, never from literals.

Units: dimensionless integers. Array shapes: use masks ``(3, n_rows)``, row arrays ``(n_rows,)``, column arrays
``(n,)``. Not certified: the passes are exact integer arithmetic. Whether a rule holds for the averaged laws
is checked by the plan of step 7, and in step 5 the caller states the rules.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..base.capabilities import Beaming, Parity
from ..lattice.representations import AzimuthHarmonic, Legendre

__all__ = ["REASONS", "STOKES", "Context", "build", "capability", "context"]

STOKES = ("I", "P", "V")
REASONS = ("parity", "beaming", "symmetry", "support", "depth", "unused")
_SPIN0 = np.array([True, False, True])


def capability(kernel: Any, cls: type) -> Any:
    """The kernel's capability of type ``cls``, ``None`` when it declares none. Two of one type raise."""
    found = [c for c in getattr(kernel, "capabilities", ()) if isinstance(c, cls)]
    if len(found) > 1:
        raise ValueError(
            f"kernel {_name(kernel)} declares {len(found)} {cls.__name__} capabilities, and the column "
            "map reads one"
        )
    return found[0] if found else None


def _name(kernel: Any) -> str:
    return str(getattr(kernel, "name", type(kernel).__name__))


def stokes_request(kernel: Any, requested: tuple[str, ...] | None) -> tuple[str, ...]:
    """The requested Stokes blocks in the order I, P, V. Blocks outside ``kernel.stokes`` raise."""
    own = tuple(kernel.stokes)
    wanted = own if requested is None else requested
    outside = [s for s in wanted if s not in own]
    if outside:
        raise ValueError(
            f"kernel {_name(kernel)} has the Stokes blocks {own}, and {', '.join(outside)} is not one of "
            "them"
        )
    return tuple(s for s in STOKES if s in wanted)


@dataclass(frozen=True)
class Context:
    """What the passes read: row arrays, the request, the kernel's names and the rules."""

    index: Any
    table: Any
    arrays: Any
    request: tuple[str, ...]
    harmonic: bool
    legendre: tuple[str, ...]
    parity: Any
    beaming: Any
    depth: str | None
    rules: Any


def _check_lattice(lattice: Any, kernel: Any) -> None:
    names = {v.name for v in kernel.variables}
    reps = dict(lattice.reps)
    outside = sorted((set(lattice.cells.variables) | set(reps)) - names)
    if outside:
        raise ValueError(
            f"the lattice variables {outside} are not variables of the kernel {_name(kernel)}: "
            f"{sorted(names)}"
        )
    for name, rep in reps.items():
        if isinstance(rep, AzimuthHarmonic) and name != kernel.azimuth:
            raise ValueError(
                f"AzimuthHarmonic on {name} resolves the kernel's azimuth, and the azimuth of "
                f"{_name(kernel)} is {kernel.azimuth}"
            )


def _check_masks(lattice: Any, kernel: Any, rules: Any) -> None:
    legendre = {n for n, r in lattice.reps if isinstance(r, Legendre)}
    kinds = {v.name: v.kind for v in kernel.variables}
    for cosine, _ in rules.masks:
        if cosine not in legendre or kinds.get(cosine) != "cosine":
            raise ValueError(
                f"the parity mask on {cosine} needs a Legendre axis of that cosine on the lattice, and the "
                f"Legendre axes are {sorted(legendre)}"
            )
    if rules.masks and capability(kernel, Parity) is None:
        raise ValueError(
            f"parity masks need the Parity capability, and kernel {_name(kernel)} declares none"
        )
    beaming = capability(kernel, Beaming)
    if rules.masks and rules.masks[0][1] == "single" and beaming is not None:
        cosine = rules.masks[0][0]
        pair = {beaming.variable: beaming.partner, beaming.partner: beaming.variable}
        if pair.get(cosine) in legendre:
            raise ValueError(
                f"both cosines {cosine} and {pair[cosine]} are resolved by Legendre degrees, so the mask "
                'is "joint"'
            )


def _check_merge(lattice: Any, kernel: Any, rules: Any) -> None:
    beaming = capability(kernel, Beaming)
    pair = None if beaming is None else (beaming.variable, beaming.partner)
    if pair != rules.merge:
        raise ValueError(
            f"merge {rules.merge} is not the Beaming pair of kernel {_name(kernel)}, which is {pair}"
        )
    degrees = {n: r.degree for n, r in lattice.reps if isinstance(r, Legendre)}
    v, p = rules.merge
    if v not in degrees or p not in degrees or degrees[v] != degrees[p]:
        raise ValueError(
            f"merge needs Legendre axes of equal degree on {v} and {p}, and the lattice has {degrees}"
        )


def _check_rules(lattice: Any, kernel: Any, rules: Any) -> None:
    _check_masks(lattice, kernel, rules)
    if rules.merge is not None:
        _check_merge(lattice, kernel, rules)
    if rules.depth_free_spin0 and kernel.depth is None:
        raise ValueError(
            f"depth_free_spin0 needs the kernel's depth variable, and kernel {_name(kernel)} has none"
        )
    if "pitch_reversal" in rules.symmetries and capability(kernel, Beaming) is None:
        raise ValueError(
            f"pitch_reversal names the pitch cosine through the Beaming capability, and kernel "
            f"{_name(kernel)} declares none"
        )


def context(lattice: Any, kernel: Any, rules: Any) -> Context:
    """Checks of the lattice, the kernel and the rules, and the inputs of the passes."""
    _check_lattice(lattice, kernel)
    request = stokes_request(kernel, rules.stokes)
    _check_rules(lattice, kernel, rules)
    index = lattice.index
    return Context(
        index=index,
        table=lattice.cells,
        arrays=index.arrays,
        request=request,
        harmonic="Re M2" in index.blocks,
        legendre=tuple(n for n, _ in index.legendre),
        parity=capability(kernel, Parity),
        beaming=capability(kernel, Beaming),
        depth=kernel.depth,
        rules=rules,
    )


# ------------------------------------------------------------------ passes on the use mask


def _blocks(names: Any) -> np.ndarray:
    return np.array([s in tuple(names) for s in STOKES])


def _remove(
    uses: np.ndarray, why: np.ndarray, rows: np.ndarray, blocks: np.ndarray, reason: str
) -> tuple[np.ndarray, np.ndarray]:
    """Uses without ``blocks`` on ``rows``. Rows that lose their last use get ``reason``."""
    out = uses & ~(rows[None, :] & blocks[:, None])
    lost = uses.any(0) & ~out.any(0)
    return out, np.where(lost, np.int8(REASONS.index(reason)), why)


def _feeds(ctx: Context) -> tuple[np.ndarray, np.ndarray]:
    spin0 = ctx.arrays.block == ctx.index.blocks.index("M0")
    feeds = {"I": spin0, "P": ~spin0 | (not ctx.harmonic), "V": spin0}
    uses = np.stack([feeds[s] & (s in ctx.request) for s in STOKES])
    why = np.where(uses.any(0), np.int8(-1), np.int8(REASONS.index("unused")))
    return uses, why.astype(np.int8)


def _depth(ctx: Context, uses: np.ndarray, why: np.ndarray) -> tuple:
    if not ctx.rules.depth_free_spin0 or ctx.depth not in ctx.table.variables:
        return uses, why
    s = ctx.arrays.powers[:, ctx.table.variables.index(ctx.depth)]
    return _remove(uses, why, s > 0, _SPIN0, "depth")


def _parity(ctx: Context, uses: np.ndarray, why: np.ndarray) -> tuple:
    if not ctx.rules.masks:
        return uses, why
    axes = [ctx.legendre.index(c) for c, _ in ctx.rules.masks]
    odd = ctx.arrays.degrees[:, axes].sum(axis=1) % 2 == 1
    uses, why = _remove(uses, why, odd, _blocks(ctx.parity.even), "parity")
    return _remove(uses, why, ~odd, _blocks(ctx.parity.odd), "parity")


def _symmetry(ctx: Context, uses: np.ndarray, why: np.ndarray) -> tuple:
    pitch = None if ctx.beaming is None else ctx.beaming.variable
    if "pitch_reversal" not in ctx.rules.symmetries or pitch not in ctx.legendre:
        return uses, why
    odd = ctx.arrays.degrees[:, ctx.legendre.index(pitch)] % 2 == 1
    return _remove(uses, why, odd, _blocks(STOKES), "symmetry")


def _representatives(ctx: Context, kept: np.ndarray) -> np.ndarray:
    """Per row the row whose column it joins: the lower row of a kept ``(l, k)``, ``(k, l)`` pair."""
    rows = np.arange(kept.size, dtype=np.int64)
    if ctx.rules.merge is None:
        return rows
    a = ctx.arrays
    iv, ip = (ctx.legendre.index(n) for n in ctx.rules.merge)
    dims = [d + 1 for _, d in ctx.index.legendre]
    strides = [int(np.prod(dims[i + 1 :], dtype=np.int64)) for i in range(len(dims))]
    dv, dp = a.degrees[:, iv], a.degrees[:, ip]
    shift = (dp - dv) * strides[iv] + (dv - dp) * strides[ip]
    per_cell = np.diff(np.asarray(ctx.index.offsets, dtype=np.int64))[a.cell]
    partner = rows + shift * per_cell
    both = kept & kept[partner]
    return np.where(both, np.minimum(rows, partner), rows)


def _number(ctx: Context, uses: np.ndarray, rep: np.ndarray) -> dict[str, np.ndarray]:
    kept = uses.any(0)
    reps = np.flatnonzero(kept & (rep == np.arange(rep.size)))
    n = reps.size
    of_rep = np.full(rep.size, -1, dtype=np.int64)
    of_rep[reps] = np.arange(n, dtype=np.int64)
    col_of_row = np.where(kept, of_rep[rep], -1)
    members = np.flatnonzero(kept)
    cols = col_of_row[members]
    col_rows = members[np.argsort(cols, kind="stable")]
    counts = np.bincount(cols, minlength=n)
    col_uses = np.stack(
        [np.bincount(cols, weights=u[members], minlength=n) > 0 for u in uses]
    )
    return dict(
        col_of_row=col_of_row,
        col_start=np.concatenate([[0], np.cumsum(counts)]),
        col_rows=col_rows,
        column_block=ctx.arrays.block[reps],
        column_cell=ctx.arrays.cell[reps],
        column_base=reps % ctx.index.block_size,
        merged=members[rep[members] != members],
        col_uses=col_uses.reshape(3, n),
    )


def _groups(keys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Group numbers in order of first appearance, and the position of each group's first member."""
    if keys.shape[0] == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    _, first, inverse = np.unique(keys, axis=0, return_index=True, return_inverse=True)
    order = np.argsort(first, kind="stable")
    relabel = np.empty_like(order)
    relabel[order] = np.arange(order.size)
    return relabel[inverse.reshape(-1)], np.sort(first)


def _cell_groups(table: Any, depth: str) -> np.ndarray:
    """Product cells with equal geometry in every celled variable other than ``depth`` share a number."""
    keep = [i for i, v in enumerate(table.variables) if v != depth]
    if not keep:
        return np.zeros(table.n_cells, dtype=np.int64)
    names = ("lo", "hi", "reference", "half_width")
    floats = [np.asarray(getattr(table, a), dtype=float)[:, keep] + 0.0 for a in names]
    codes: dict[str, int] = {}
    charts = [
        [codes.setdefault(row[i], len(codes)) for i in keep] for row in table.chart
    ]
    key = np.concatenate(floats + [np.array(charts, dtype=float)], axis=1)
    return _groups(key)[0]


def _spin0_keys(ctx: Context, numbered: dict, chosen: np.ndarray) -> np.ndarray:
    base = numbered["column_base"][chosen]
    if not ctx.rules.depth_free_spin0 or ctx.depth not in ctx.table.variables:
        return base
    a, t = ctx.arrays, ctx.table
    keep = [i for i, v in enumerate(t.variables) if v != ctx.depth]
    rows = numbered["col_rows"][numbered["col_start"][:-1][chosen]]
    group = _cell_groups(t, ctx.depth)[a.cell[rows]]
    return np.column_stack([group, a.pair[rows], a.powers[rows][:, keep]])


def _spin(numbered: dict, chosen: np.ndarray, keys: np.ndarray) -> tuple:
    labels, first = _groups(keys)
    spin_from = np.full(numbered["column_base"].size, -1, dtype=np.int64)
    spin_from[chosen] = labels
    return spin_from, numbered["column_base"][chosen[first]]


def build(ctx: Context) -> dict[str, Any]:
    """Every array of the column map, from the passes in order."""
    uses, why = _feeds(ctx)
    for step in (_depth, _parity, _symmetry):
        uses, why = step(ctx, uses, why)
    numbered = _number(ctx, uses, _representatives(ctx, uses.any(0)))
    col_uses = numbered.pop("col_uses")
    chosen0 = np.flatnonzero(col_uses[0] | col_uses[2])
    chosen2 = np.flatnonzero(col_uses[1])
    spin0_from, spin0_base = _spin(
        numbered, chosen0, _spin0_keys(ctx, numbered, chosen0)
    )
    spin2_from, spin2_base = _spin(numbered, chosen2, numbered["column_base"][chosen2])
    merged = numbered.pop("merged")
    groups = [
        (r, merged if r == "beaming" else np.flatnonzero(why == i))
        for i, r in enumerate(REASONS)
    ]
    return dict(
        numbered,
        uses={s: np.flatnonzero(col_uses[b]) for b, s in enumerate(STOKES)},
        spin0_from=spin0_from,
        spin0_base=spin0_base,
        spin2_from=spin2_from,
        spin2_base=spin2_base,
        reasons=tuple((r, rows) for r, rows in groups if rows.size),
    )
