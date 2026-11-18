"""(b) Field-scaling columns against jet columns in B: the dispatch boundary of Spec Sec. 8.1, 8.3 (T5).

Both methods are evaluated directly, bypassing any dispatcher, at the first and last E1 energy
cells, on field lattices with range ratio 1.01 and 100 (affine and log charts) and on the E6
single cell, at orders 2 and 4, with the fixed-field reference B_0 at each cell's own reference,
at 0.5 uG and at 50 uG (energy shifts up to (B_b/B_0)^(1/2) of about 9). rel_err per column is
max over channels |jet - scaling| / max(|jet|, |scaling|).

Run from the prototypes directory:
PYTHONPATH=.:../../.. python -m pytest tests/test_fieldcells_scaling.py
"""
from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

import core as C
from rules.fieldcells import models as M
from rules.fieldcells import probe as R
from rules.fieldcells import scaling as S

FIRST_LAST = (0, 18)
B0_MODES = ("own", 0.5, 50.0)


def _b0(tab, mode):
    return tab.reference[:, 1] if mode == "own" else float(mode)


@pytest.mark.parametrize("N", [2, 4])
@pytest.mark.parametrize("blat", ["lin_ratio1.01", "lin_ratio100", "log_ratio100", "single_5_1", "lin_2_8"])
@pytest.mark.parametrize("b0", B0_MODES)
@pytest.mark.parametrize("gcell", FIRST_LAST)
def test_scaling_equals_jet(blat, N, b0, gcell):
    tab, cols, _ = R.jet("E1", blat, (N, N), N, FIRST_LAST)
    sc = S.scaling_columns(R.avg("E1"), tab, _b0(tab, b0))
    rows = np.flatnonzero(tab.index[tab.row_cell, 0] == gcell)
    errs = {k: R.rel_err(cols[k][rows], sc[k][rows]) for k in ("I", "P")}
    assert all(np.isfinite(cols[k][rows]).all() and np.isfinite(sc[k][rows]).all() for k in ("I", "P"))
    worst = max(errs.values())
    R.record(f"b_scaling_vs_jet[{blat},N={N},B0={b0},gcell={gcell}]",
             {"I": errs["I"], "P": errs["P"], "verdict": R.verdict(worst)})
    assert worst < 1e-8


@pytest.mark.parametrize("blat", ["lin_ratio1.01", "lin_ratio100", "lin_2_8", "single_5_1"])
def test_paper_field_scaling_relation(blat):
    """PAPER eq: field scaling, B d/dB X_r = X_r + (r+1)/(2h) X_{r+1}, on the s = 1 jet columns (affine chart)."""
    tab, cols, _ = R.jet("E1", blat, (4, 4), 4, FIRST_LAST)
    o, worst = tab.offsets, 0.0
    for c in range(tab.n_cells):
        ex = [tuple(e) for e in tab.exponents(c)]
        e_over_b = tab.half_width[c, 1] / tab.reference[c, 1]
        h = tab.half_width[c, 0]
        for k in ("I", "P"):
            X = {e: cols[k][o[c] + i] for i, e in enumerate(ex)}
            for r in range(4):
                want = e_over_b * (X[(r, 0)] + (r + 1) / (2 * h) * X[(r + 1, 0)])
                worst = max(worst, R.rel_err(X[(r, 1)][None], want[None]))
    R.record(f"b_paper_relation_s1[{blat}]", {"rel_err": worst, "verdict": R.verdict(worst)})
    assert worst < 1e-8


@pytest.mark.parametrize("b0", ["own", 5.0])
def test_scaling_fails_with_eta_given_B(b0):
    """Negative control: an inclination law that depends on B breaks the precondition (T5)."""
    tab, cols, _ = R.jet("etaB", "lin_2_8", (2, 2), 2, FIRST_LAST)
    sc = S.scaling_columns(R.avg("etaB"), tab, _b0(tab, b0))
    field = np.flatnonzero(tab.row_exp[:, 1] > 0)
    energy = np.flatnonzero(tab.row_exp[:, 1] == 0)
    e_field = max(R.rel_err(cols[k][field], sc[k][field]) for k in ("I", "P"))
    e_energy = max(R.rel_err(cols[k][energy], sc[k][energy]) for k in ("I", "P"))
    R.record(f"b_negative_eta_given_B[B0={b0}]", {"field_rows": e_field, "energy_rows": e_energy,
                                                   "verdict_field": R.verdict(e_field)})
    assert e_field > 1e-2  # the scaling identity fails, so the jet route is required
    if b0 == "own":
        assert e_energy < 1e-12  # s = 0 rows are the energy jet at the cell itself


class _Unit:
    """I = 1, P = 1 in one channel, so the P columns of a FactorKernel are the Taylor coefficients of c."""

    n_ch = 1

    def __call__(self, p):
        one = 0.0 * p["gamma"] + 1.0
        return one[None], (one + 0j)[None]


def _c(p):
    return 0.4 * jnp.exp(1j * (0.5 + 0.3 * (p["B"] - 5.0) / 5.0 + 0.1 * jnp.log(p["gamma"] / 1e4)))


@pytest.mark.parametrize("blat", ["lin_2_8", "log_ratio100"])
def test_azimuth_factor_at_assembly(blat):
    """A resolved azimuth factor c(gamma, B) enters after the scaling columns by a Cauchy product."""
    tab, _, _ = R.jet("E1", blat, (2, 2), 2, FIRST_LAST)
    fk = C.AveragedKernel(M.FactorKernel(R.kernel(), _c), [("eta", M.eta_rule())])
    want = C.build_columns(fk, tab)["P"]
    coef = C.build_columns(C.AveragedKernel(M.FactorKernel(_Unit(), _c)), tab)["P"][:, 0]
    sc = S.scaling_columns(R.avg("E1"), tab, 5.0)
    got = S.cauchy_factor(sc, tab, coef)
    e = R.rel_err(want, got)
    R.record(f"b_azimuth_factor_cauchy[{blat}]", {"rel_err": e, "verdict": R.verdict(e)})
    assert e < 1e-8
