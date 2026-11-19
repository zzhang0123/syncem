"""S0 per quantised span (design/0.8/SPEC.md decision 158, cache test T14 item 6 of the step-5b plan).

Each S1 table reads the S0 table of its own need: the union of the default span and ``FieldAverage.needed_lnx`` of
the law, rounded outward to multiples of 10 e-folds on a grid anchored at ``ln x = 0``. The S1 bits of a law then
do not depend on the laws read before it in the same memo. ``quantised_span`` chooses between two S0 tables at each
multiple of 10, so its boundaries are tested on both sides, and the S1 tables of a law at a boundary are compared
between the two adjacent spans and with the quadrature oracle (``FieldAverage.quadrature``).
"""

import numpy as np
import pytest

from syncem.base.continuum import ContinuumKernel
from syncem.engine._stack_series import _s1_table, stack_cell
from syncem.engine.average import AveragedKernel, uniform_rule
from syncem.engine.field_average import TABLE_LNY, TABLE_ORDER, FieldAverage
from syncem.engine.primitive import PrimitiveTable, quantised_span
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from tests.v080.test_v080_stack import screen, small_channels

K = ContinuumKernel(x_min=1e-12)
# The default span ln 1e-12 = -27.6 to ln 1e7 = 16.1, rounded outward to multiples of 10.
DEFAULT = (-30.0, 20.0)

# ------------------------------------------------------------------ quantised_span at its boundaries


@pytest.mark.parametrize(
    "need, span",
    [
        ((-22.5, 12.4), DEFAULT),  # inside the default span
        ((-30.0, 20.0), DEFAULT),  # exact multiples are kept
        ((-40.0, 30.0), (-40.0, 30.0)),
        ((np.nextafter(-30.0, 0.0), np.nextafter(20.0, 0.0)), DEFAULT),  # just inside
        ((np.nextafter(-40.0, 0.0), np.nextafter(30.0, 0.0)), (-40.0, 30.0)),
        ((np.nextafter(-30.0, -np.inf), 0.0), (-40.0, 20.0)),  # just outside
        ((0.0, np.nextafter(20.0, np.inf)), (-30.0, 30.0)),
        ((np.nextafter(-40.0, -np.inf), np.nextafter(30.0, np.inf)), (-50.0, 40.0)),
        ((-30.0 - 1e-9, 20.0 + 1e-9), (-40.0, 30.0)),
        ((0.0, 1e-12), DEFAULT),  # a short need
        ((-1234.5, 5678.9), (-1240.0, 5680.0)),
        ((-1e6 - 5.0, 1e6 + 5.0), (-1000010.0, 1000010.0)),  # extreme spans
    ],
)
def test_quantised_span(need, span):
    out = quantised_span(*need)
    assert out == span and all(type(v) is float for v in out)
    assert quantised_span(*out) == out  # a quantised span is its own need


@pytest.mark.parametrize("need", [(-1e15, 1e15), (-1e300, 1e300), (-3e14 - 3.0, -2.0)])
def test_quantised_span_of_extreme_needs_contains_them(need):
    lo, hi = quantised_span(*need)
    assert lo <= min(need[0], DEFAULT[0]) and hi >= max(need[1], DEFAULT[1])
    assert lo == 10.0 * np.floor(lo / 10.0) and hi == 10.0 * np.ceil(hi / 10.0)


@pytest.mark.parametrize(
    "need", [(1.0, 1.0), (2.0, 1.0), (np.nan, 1.0), (-np.inf, 1.0), (0.0, np.inf)]
)
def test_quantised_span_refuses_an_empty_or_non_finite_need(need):
    with pytest.raises(ValueError, match="finite"):
        quantised_span(*need)


# ------------------------------------------------------------------ S1 bits do not depend on the build history

NARROW = (np.linspace(4.0, 6.0, 4), np.full(4, 0.25))  # uG, span (-30, 20)
STRONG = (np.array([1e6]), np.ones(1))  # 1 G, span (-40, 20)
WEAK = (np.array([1e-3]), np.ones(1))  # 1e-3 uG, span (-30, 30)
LAWS = {"narrow": NARROW, "strong": STRONG, "weak": WEAK}


def _bits(table: FieldAverage) -> tuple[bytes, ...]:
    return (table.lny.tobytes(),) + tuple(table.coef[w].tobytes() for w in ("F", "G"))


@pytest.fixture(scope="module")
def alone():
    return {name: _bits(_s1_table({}, "lookup", *law)) for name, law in LAWS.items()}


def test_the_laws_need_three_spans():
    spans = {
        n: quantised_span(*FieldAverage.needed_lnx(law[0])) for n, law in LAWS.items()
    }
    assert spans == {"narrow": DEFAULT, "strong": (-40.0, 20.0), "weak": (-30.0, 30.0)}


@pytest.mark.parametrize(
    "sequence",
    [
        ("strong", "narrow"),
        ("narrow", "strong"),
        ("weak", "narrow"),
        ("narrow", "weak"),
    ],
)
def test_s1_bits_equal_the_table_built_alone(sequence, alone):
    """T14 item 6: narrow then wide, wide then narrow and each alone give the same S1 bits for the same law.

    Probe of the decision-144 rule (one S0 per memo widened to the union, 2026-11-19, 1-minute load 5.1): the
    narrow law read after the 1 G law differed from the narrow law alone by 2.9e-15 of the peak of ``T`` and by up
    to 2.3e-14 of the peak of ``T^(r)``, ``r = 1..6``, at the knot midpoints (plan row 158 measured 1.1e-13 in
    ``ln T`` and 8e-15 of the ``r = 6`` peak for two other ranges). The narrow law read first was bitwise equal.
    """
    memo = {}
    for name in sequence:
        assert _bits(_s1_table(memo, "lookup", *LAWS[name])) == alone[name], name


def test_stack_rows_do_not_depend_on_the_laws_read_before():
    """A cached-S1 cell after a 1 G one-node cell in the same memo equals the cell in a fresh memo, and back."""
    ch, lattice = small_channels(3, 16), Lattice(gamma=Cells.log(300, 3e6, 3, order=2))
    eta = ("eta", uniform_rule(0.0, 0.5, 4))
    narrow = AveragedKernel(
        K, [("B", uniform_rule(4e-6, 6e-6, 4)), eta], phase=screen()
    )
    strong = AveragedKernel(K, [eta], fixed={"B": 1.0}, phase=screen())

    def rows(avg, memo):
        i, p = stack_cell(avg, ch, lattice, 1, memo=memo)
        return np.asarray(i).tobytes(), np.asarray(p).tobytes()

    fresh = {"narrow": rows(narrow, {}), "strong": rows(strong, {})}
    for first, second in (("strong", "narrow"), ("narrow", "strong")):
        memo = {}
        rows({"narrow": narrow, "strong": strong}[first], memo)
        assert rows({"narrow": narrow, "strong": strong}[second], memo) == fresh[second]


# ------------------------------------------------------------------ the two spans at a boundary agree


def _node_at(end: str, edge: float) -> float:
    """The one field node in uG whose ``needed_lnx`` has its ``end`` at ``edge``."""
    if end == "lo":
        return float(np.exp(TABLE_LNY[0] - 0.01 - edge))
    return float(np.exp(TABLE_LNY[1] + 0.01 - edge))


def _errors(table: FieldAverage, ref: FieldAverage) -> tuple[float, float]:
    """``ln T`` difference and the largest ``T^(r)`` difference over ``r = 1..6`` relative to its peak, F and G."""
    lny = np.sort(np.random.default_rng(3).uniform(ref.lny[0], ref.lny[-1], 1500))
    ln_t, peak = 0.0, 0.0
    for which in ("F", "G"):
        a, b = table.evaluate(lny, which, 6), ref.evaluate(lny, which, 6)
        keep = b[0] > np.exp(-650.0)
        ln_t = max(ln_t, float(np.max(np.abs(np.log(a[0][keep]) - np.log(b[0][keep])))))
        for r in range(1, 7):
            scale = np.max(np.abs(b[r][keep]))
            peak = max(peak, float(np.max(np.abs(a[r][keep] - b[r][keep])) / scale))
    return ln_t, peak


@pytest.fixture(scope="module")
def s0_tables():
    return {}


def _s0(cache: dict, span: tuple[float, float]) -> PrimitiveTable:
    if span not in cache:
        cache[span] = PrimitiveTable.build(*span, order=TABLE_ORDER)
    return cache[span]


@pytest.mark.parametrize(
    "end, edge, inner, outer",
    [
        ("lo", -30.0, DEFAULT, (-40.0, 20.0)),  # 1.06e4 uG
        ("lo", -40.0, (-40.0, 20.0), (-50.0, 20.0)),  # 230 G, an extreme field
        ("hi", 20.0, DEFAULT, (-30.0, 30.0)),  # 2.1e-3 uG
        ("hi", 30.0, (-30.0, 30.0), (-30.0, 40.0)),  # 9.5e-8 uG, an extreme field
    ],
)
def test_s1_from_both_spans_at_a_boundary(end, edge, inner, outer, s0_tables):
    """A node just inside a multiple of 10 reads the inner span, and one just outside reads the outer span.

    The inner node's S1 tables from both spans agree, and both equal the quadrature oracle to the tolerances of
    ``test_v080_stack.test_s1_lookup_table_equals_the_quadrature_table`` (SPEC Sec. 8.7: 1.7e-13 in ``ln T`` and
    3.3e-10 of the ``r = 6`` peak). Probe (2026-11-19, 1-minute load 6.7 to 7.1), inner against outer span: at most
    1.1e-13 in ``ln T`` and 5.5e-14 of the peak (0.0 at the upper boundaries), so the bound 1e-12 has a margin of 9.
    Each span against the quadrature: at most 2.3e-13 in ``ln T`` and 3.4e-10 of the peak, under 1e-12 and 1e-9.
    """
    b = _node_at(end, edge)
    inside, outside = (
        (b * (1 - 1e-9), b * (1 + 1e-9))
        if end == "lo"
        else (b * (1 + 1e-9), b * (1 - 1e-9))
    )
    assert quantised_span(*FieldAverage.needed_lnx([inside])) == inner
    assert quantised_span(*FieldAverage.needed_lnx([outside])) == outer
    near = FieldAverage.lookups([inside], [1.0], primitive=_s0(s0_tables, inner))
    far = FieldAverage.lookups([inside], [1.0], primitive=_s0(s0_tables, outer))
    ln_t, peak = _errors(near, far)
    assert ln_t < 1e-12 and peak < 1e-12, (ln_t, peak)
    quad = FieldAverage.quadrature([inside], [1.0])
    for table in (near, far):
        ln_t, peak = _errors(table, quad)
        assert ln_t < 1e-12 and peak < 1e-9, (ln_t, peak)
