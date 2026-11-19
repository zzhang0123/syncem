"""The order branch of ``stack_precondition`` on both sides of the S1 table order (design/0.8/SPEC.md decision 150).

At ``N = 6`` the static check passes and the stack decides. At ``N = 7`` it returns the order reason, and
``routes="auto"`` sends the cell to the jet with that reason without calling the stack. The jet is replaced by a
stub that returns zero rows of the right shape, since only the route is under test here: an order-7 jet run of
this cell took more than 300 s in the review probe at a 1-minute load near 4.
"""

import numpy as np
import pytest

from syncem.engine import _routes
from syncem.engine._routes import route_cells, stack_precondition
from syncem.engine.field_average import TABLE_ORDER
from syncem.records.records import RouteRecord
from tests.v080.test_v080_routes import check_contract, fixed_b_avg, switch_lattice
from tests.v080.test_v080_stack import small_channels

ORDER_REASON = "order 7 exceeds the S1 table order 6: the jet takes the cell"


def test_the_table_order_is_six():
    assert TABLE_ORDER == 6


@pytest.mark.parametrize("order", [TABLE_ORDER, TABLE_ORDER + 1])
def test_order_branch_of_the_precondition(order):
    lattice = switch_lattice(order, 1, 3e7, 1.2e8)
    reason = stack_precondition(fixed_b_avg(), lattice, 0)
    assert reason == (None if order == TABLE_ORDER else ORDER_REASON)


def test_auto_sends_an_order_7_cell_to_the_jet_with_the_order_reason(monkeypatch):
    seen = []

    def stub(averaged, channels, lattice, *, with_v=False, cells=None):
        n = sum(len(lattice.index.cell_rows(c)) for c in cells)
        seen.append(tuple(cells))
        return np.zeros((n, channels.n_ch)), np.zeros((n, channels.n_ch), complex)

    monkeypatch.setattr(_routes, "jet_lattice", stub)
    monkeypatch.setattr(
        _routes, "stack_cell", lambda *a, **k: pytest.fail("stack called")
    )
    avg, lattice = fixed_b_avg(), switch_lattice(TABLE_ORDER + 1, 1, 3e7, 1.2e8)
    _, _, _, records = route_cells(avg, small_channels(), lattice, "auto")
    assert records == (RouteRecord(0, "jet", reason=ORDER_REASON),)
    assert seen == [(0,)]
    check_contract(avg, lattice, records)
