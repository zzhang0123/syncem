"""Declarations replacing unbounded ledger terms (review P1-5 and the declared-sum P2)."""

import numpy as np
import pytest

from syncem.records.ledger import ErrorTerm, Ledger


def _open():
    return Ledger((ErrorTerm.unbounded("no bound", name="source"),))


@pytest.mark.parametrize(
    "term",
    [
        ErrorTerm(0.3, "bound", relative_to="I"),
        ErrorTerm(0.3, "bound", space="retained"),
    ],
)
def test_declaration_in_another_space_keeps_the_stokes_term_open(term):
    ledger = _open()
    assert ledger.envelope() is None
    with pytest.raises(ValueError, match="absolute Stokes"):
        ledger.allow("source", term, note="assumed")
    assert ledger.envelope() is None
    assert ledger.unbounded() == ("source",)


def test_absolute_declaration_closes_the_envelope():
    declared = _open().allow("source", ErrorTerm(0.3, "bound"), note="assumed")
    envelope = declared.envelope()
    np.testing.assert_allclose(envelope.value, 0.3)
    assert envelope.declared == ("source",)


def test_sum_with_a_declared_part_names_the_declaration():
    from syncem.records.ledger import combine_ledgers

    computed = Ledger((ErrorTerm(0.2, "bound", name="source"),))
    declared = _open().allow("source", ErrorTerm(0.3, "bound"), note="assumed")
    total = combine_ledgers([computed, declared])
    envelope = total.envelope()
    np.testing.assert_allclose(envelope.value, 0.5)
    assert envelope.declared == ("source",)
    assert total["source"].note == "sum of computed and user-declared terms"
    both = combine_ledgers([computed, computed])
    assert both.envelope().declared == ()
