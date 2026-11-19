"""Shared cases of the column-map tests (design/0.8/SPEC.md Sec. 3.4, 4.3, 4.7, decisions 146 to 148).

The lattices of Sec. 4.7, the benchmark stand-in kernel, the cosine kernel of the mask dispatcher, the
Beaming merge case and a brute-force column map built from ``index.row(k)``, one ``Row`` object at a time.
The brute-force map is an independent formulation of the vectorised passes of ``syncem.engine.columns``.
"""

from __future__ import annotations

from pathlib import Path

import jax.numpy as jnp
import numpy as np

from syncem.base.capabilities import Beaming, Parity
from syncem.base.continuum import ContinuumKernel
from syncem.base.user import kernel
from syncem.base.variables import SYNCHROTRON_VARIABLES as SV
from syncem.base.variables import Variable
from syncem.engine.columns import REASONS, ColumnRules
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import AzimuthHarmonic, Legendre

ROOT = Path(__file__).resolve().parents[2]
EPS = np.finfo(float).eps
CANON = ("I", "P", "V")


def energy():
    return Cells.log(300, 2.9626e6, 19)


def _single():
    return dict(
        gamma=Cells.single(1e3, scale=1e2),
        B=Cells.single("5 uG", scale="1 uG"),
        depth=Cells.single(8.0, scale=1.0),
    )


LATTICES = {
    "E1": lambda: Lattice(gamma=energy()),
    "E5": lambda: Lattice(gamma=energy(), phi=AzimuthHarmonic()),
    "E6": lambda: Lattice(
        gamma=energy(), B=Cells.single("5 uG", scale="1 uG"), order=2
    ),
    "E4-KP": lambda: Lattice(gamma=energy(), mu=Legendre(8)),
    "energy-depth": lambda: Lattice(
        gamma=energy(), depth=Cells.linear(6, 10, 3), order=2
    ),
    "benchmark": lambda: Lattice(
        **_single(),
        mu=Legendre(8),
        eta=Legendre(8),
        phi=AzimuthHarmonic(),
        order=2,
        depth_in_total=True,
    ),
    "benchmark-out": lambda: Lattice(
        **_single(), mu=Legendre(8), eta=Legendre(8), phi=AzimuthHarmonic(), order=2
    ),
}


def benchmark_kernel():
    """The named stand-in for the HarmonicKernel of Sec. 4.7: continuum variables, IQUV, Parity, Beaming."""

    @kernel(
        variables=list(SV.values()),
        stokes=("I", "P", "V"),
        azimuth="phi",
        depth="depth",
        capabilities=(Parity(("I", "P"), ("V",)), Beaming("mu", "eta")),
    )
    def benchmark_stand_in(nu, gamma, B, mu, eta):
        one = jnp.ones_like(nu)
        return {"I": one, "P": 0.5 * one, "V": 0.1 * mu * one}

    return benchmark_stand_in


# ------------------------------------------------------------------ brute-force reference from index.row


def _step(uses, why, removed, reason):
    """Remove ``removed`` from ``uses``. A row loses its last use to the pass named ``reason``."""
    left = uses - removed
    if why is None and uses and not left:
        why = reason
    return left, why


def _row_uses(row, kernel_, rules, harmonic, request):
    parity = next(
        (c for c in kernel_.capabilities if isinstance(c, Parity)), Parity((), ())
    )
    beaming = next((c for c in kernel_.capabilities if isinstance(c, Beaming)), None)
    uses = ({"I", "V"} | (set() if harmonic else {"P"})) if row.block == "M0" else {"P"}
    uses &= set(request)
    why = None if uses else "unused"
    if rules.depth_free_spin0 and row.powers.get(kernel_.depth, 0) > 0:
        uses, why = _step(uses, why, {"I", "V"}, "depth")
    if rules.masks:
        total = sum(row.degrees[c] for c, _ in rules.masks)
        drop = set(parity.odd) if total % 2 == 0 else set(parity.even)
        uses, why = _step(uses, why, drop, "parity")
    if (
        "pitch_reversal" in rules.symmetries
        and row.degrees.get(beaming.variable, 0) % 2
    ):
        uses, why = _step(uses, why, set(CANON), "symmetry")
    return uses, why


def _merge_key(row, rules):
    degrees = dict(row.degrees)
    if rules.merge is not None:
        v, p = rules.merge
        degrees[v], degrees[p] = sorted((degrees[v], degrees[p]))
    return (row.block, row.cell, tuple(row.powers.items()), tuple(degrees.items()))


def _shared(keys):
    """Group numbers in order of first appearance and the first member of each group."""
    seen, out, first = {}, [], []
    for j, key in enumerate(keys):
        if key not in seen:
            seen[key] = len(seen)
            first.append(j)
        out.append(seen[key])
    return out, first


def _spin0_key(lattice, kernel_, rules, row, base):
    t = lattice.cells
    depth = kernel_.depth
    if not (rules.depth_free_spin0 and depth in t.variables):
        return base
    geometry = tuple(
        (v, row.intervals[v], float(t.reference[row.cell, i]), t.chart[row.cell][i])
        for i, v in enumerate(t.variables)
        if v != depth
    )
    powers = tuple((v, r) for v, r in row.powers.items() if v != depth)
    return (geometry, powers, tuple(row.degrees.items()))


def _spin_maps(column_uses, keys, wanted, bases):
    chosen = [j for j, u in enumerate(column_uses) if u & wanted]
    groups, first = _shared([keys[j] for j in chosen])
    spin_from = np.full(len(column_uses), -1)
    spin_from[chosen] = groups
    return spin_from, np.array([bases[chosen[f]] for f in first], dtype=np.int64)


def brute_map(lattice, kernel_, rules):
    index = lattice.index
    request = tuple(s for s in CANON if s in (rules.stokes or kernel_.stokes))
    harmonic = "Re M2" in index.blocks
    rows = [index.row(k) for k in range(index.n)]
    found = [_row_uses(r, kernel_, rules, harmonic, request) for r in rows]
    groups: dict = {}
    for k, (uses, _) in enumerate(found):
        if uses:
            groups.setdefault(_merge_key(rows[k], rules), []).append(k)
    members = sorted(groups.values())
    col_of_row = np.full(index.n, -1)
    for j, ks in enumerate(members):
        col_of_row[ks] = j
    reasons = {r: [] for r in REASONS}
    for k, (uses, why) in enumerate(found):
        if not uses:
            reasons[why].append(k)
    for ks in members:
        reasons["beaming"].extend(ks[1:])
    reps = [ks[0] for ks in members]
    bases = [k % index.block_size for k in reps]
    column_uses = [set().union(*(found[k][0] for k in ks)) for ks in members]
    keys0 = [
        _spin0_key(lattice, kernel_, rules, rows[k], b) for k, b in zip(reps, bases)
    ]
    spin0_from, spin0_base = _spin_maps(column_uses, keys0, {"I", "V"}, bases)
    spin2_from, spin2_base = _spin_maps(column_uses, bases, {"P"}, bases)
    return dict(
        col_of_row=col_of_row,
        col_start=np.concatenate([[0], np.cumsum([len(ks) for ks in members])]),
        col_rows=np.array([k for ks in members for k in ks], dtype=np.int64),
        column_block=np.array([index.blocks.index(rows[k].block) for k in reps]),
        column_cell=np.array([rows[k].cell for k in reps]),
        column_base=np.array(bases),
        uses={
            s: np.array([j for j, u in enumerate(column_uses) if s in u]) for s in CANON
        },
        spin0_from=spin0_from,
        spin0_base=spin0_base,
        spin2_from=spin2_from,
        spin2_base=spin2_base,
        reasons=tuple((r, np.array(sorted(v))) for r, v in reasons.items() if v),
        stokes=request,
    )


def assert_matches_brute(cm, lattice, kernel_, rules):
    ref = brute_map(lattice, kernel_, rules)
    for name in (
        "col_of_row",
        "col_start",
        "col_rows",
        "column_block",
        "column_cell",
        "column_base",
        "spin0_from",
        "spin0_base",
        "spin2_from",
        "spin2_base",
    ):
        np.testing.assert_array_equal(getattr(cm, name), ref[name], err_msg=name)
    assert set(cm.uses) == set(CANON)
    for s in CANON:
        np.testing.assert_array_equal(cm.uses[s], ref["uses"][s], err_msg=s)
    assert [r for r, _ in cm.reasons] == [r for r, _ in ref["reasons"]]
    for (_, got), (_, want) in zip(cm.reasons, ref["reasons"]):
        np.testing.assert_array_equal(got, want)
    assert cm.stokes == ref["stokes"]


# ------------------------------------------------------------------ cases shared with the basis tests


def merge_case():
    lat = Lattice(gamma=Cells.log(300, 3e5, 2), mu=Legendre(2), eta=Legendre(2))
    rules = ColumnRules(masks=(("mu", "joint"), ("eta", "joint")), merge=("mu", "eta"))
    return lat, ContinuumKernel(), rules


def cosine_kernel():
    """Even under (mu, eta) -> (-mu, -eta), with a mu eta term (SPEC Sec. 4.3, the case of one cosine resolved)."""

    @kernel(
        variables=[Variable("a", kind="positive"), SV["mu"], SV["eta"]],
        stokes=("I", "P"),
        capabilities=(Parity(("I", "P"), ("V",)),),
    )
    def cosines(nu, a, mu, eta):
        s = (nu / 1e9) ** -0.7
        return {
            "I": a * s * (1 + 0.5 * mu * eta + 0.3 * mu**2),
            "P": 0.4 * a * s * (1 + 0.2 * mu * eta),
        }

    return cosines


N_MU, N_ETA = 16, 8
