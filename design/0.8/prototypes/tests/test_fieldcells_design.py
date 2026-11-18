"""(c) The structural tie of the field-scaling relation in the joint (gamma, B) design (Spec Sec. 5.5).

The design maps lattice rows to [I, Q, U] in 41 channels (123 rows). With the precondition
of Sec. 8.1, every field column (s > 0) of a product cell lies in the span of the energy
columns (s = 0) of that cell, and the vectors of rules/fieldcells/design.predicted_null are
annihilated. An inclination law that depends on B breaks the tie. On log field cells with
h_B = 2 h_gamma, product cells (g, b+1) and (g+1, b) give proportional columns, a tie
between cells that the per-cell relations do not list.

Run from the prototypes directory:
PYTHONPATH=.:../../.. python -m pytest tests/test_fieldcells_design.py
"""
from __future__ import annotations

import numpy as np
import pytest

from core.paper import AZIMUTH
from rules.fieldcells import design as D
from rules.fieldcells import probe as R

GCELLS = {"low": (0, 1, 2, 3), "mid": (5, 6, 7, 8), "high": (15, 16, 17, 18)}


def _summary(tab, cols):
    Dm = D.design(cols, AZIMUTH)
    e = np.flatnonzero(tab.row_exp[:, 1] == 0)
    s = D.singular_values(Dm)
    return Dm, {"rows": tab.n_rows, "structural_rank": D.structural_rank(tab),
                "numerical_rank_joint": D.numerical_rank(Dm), "numerical_rank_energy_only": D.numerical_rank(Dm[:, e]),
                "field_residual": D.field_residual(Dm, tab),
                "null_vectors": int(D.predicted_null(tab).shape[1]),
                "null_residual": D.null_residual(Dm, D.predicted_null(tab)),
                "sigma_min": float(s[-1])}


@pytest.mark.parametrize("blat", ["lin_2_8", "log_ratio100", "lin_ratio1.01", "log_aligned"])
@pytest.mark.parametrize("caps,total", [((2, 1), 2), ((2, 2), 2), ((1, 1), 2), ((3, 3), 3)])
@pytest.mark.parametrize("where", ["low", "mid", "high"])
def test_tie_holds(blat, caps, total, where):
    tab, cols, _ = R.jet("E1", blat, caps, total, GCELLS[where])
    _, out = _summary(tab, cols)
    R.record(f"c_tie[{blat},caps={caps},total={total},{where}]", out)
    assert out["null_residual"] < 1e-12
    assert out["null_vectors"] == tab.n_rows - out["structural_rank"]
    if total <= caps[0]:
        # every field column needs energy orders up to r + s <= total, all of them in the lower set
        assert out["field_residual"] < 1e-8
        assert out["numerical_rank_joint"] <= out["numerical_rank_energy_only"] + 1
    else:
        # caps (1, 1), total 2: the (1, 1) column needs the energy order 2, absent from the set
        assert out["field_residual"] > 1e-6


@pytest.mark.parametrize("blat", ["lin_2_8", "log_ratio100"])
@pytest.mark.parametrize("caps,total", [((2, 1), 2), ((2, 2), 2)])
def test_tie_breaks_with_eta_given_B(blat, caps, total):
    tab, cols, _ = R.jet("etaB", blat, caps, total, GCELLS["mid"])
    _, out = _summary(tab, cols)
    R.record(f"c_tie_eta_given_B[{blat},caps={caps},total={total}]", out)
    assert out["field_residual"] > 1e-3
    assert out["null_residual"] > 1e-4
    assert out["numerical_rank_joint"] > out["numerical_rank_energy_only"]


@pytest.mark.parametrize("caps,total", [((2, 1), 2), ((2, 2), 2)])
def test_cross_cell_tie_aligned(caps, total):
    """h_B = 2 h_gamma: columns of (g, b+1) equal (B_{b+1}/B_b) times columns of (g+1, b)."""
    tab, cols, _ = R.jet("E1", "log_aligned", caps, total, GCELLS["mid"])
    o, worst = tab.offsets, 0.0
    pos = {tuple(ix): c for c, ix in enumerate(tab.index)}
    for (g, b), c in pos.items():
        if (g + 1, b - 1) in pos:
            c2 = pos[(g + 1, b - 1)]
            lam = tab.reference[c, 1] / tab.reference[c2, 1]
            for k in ("I", "P"):
                worst = max(worst, R.rel_err(cols[k][o[c]:o[c + 1]], lam * cols[k][o[c2]:o[c2 + 1]]))
    _, out = _summary(tab, cols)
    _, ref = _summary(*R.jet("E1", "log_ratio100", caps, total, GCELLS["mid"])[:2])
    out["cross_cell_rel_err"] = worst
    out["numerical_rank_joint_log_ratio100"] = ref["numerical_rank_joint"]
    R.record(f"c_cross_cell_aligned[caps={caps},total={total}]", out)
    assert worst < 1e-9


def test_single_cell_singular_values():
    """One product cell (123 x 6): three singular values with the tie, six without."""
    out = {}
    for kind in ("E1", "etaB"):
        tab, cols, _ = R.jet(kind, "lin_2_8", (2, 2), 2, (7,))
        c = int(np.flatnonzero(tab.index[:, 1] == 1)[0])
        rows = slice(tab.offsets[c], tab.offsets[c + 1])
        s = D.singular_values(D.design({"I": cols["I"][rows], "P": cols["P"][rows]}, AZIMUTH))
        out[kind] = [float(v) for v in s]
    R.record("c_single_cell_singular_values[gamma=7,B=1]", out)
    assert out["E1"][3] < 1e-12 and out["etaB"][3] > 1e-8


def test_full_lattice_null_vectors():
    """19 x 3 cells, caps (2, 1), 285 rows: the 114 predicted null vectors."""
    tab, cols, _ = R.jet("E1", "lin_2_8", (2, 1), 2)
    _, out = _summary(tab, cols)
    R.record("c_full_lattice_19x3_caps21", out)
    assert out["null_vectors"] == 114 and out["null_residual"] < 1e-12
