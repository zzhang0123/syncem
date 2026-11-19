"""The error ledger of a basis (design/0.8/SPEC.md Sec. 3.4, 7.2, decision 149).

LABEL ``sec: frequency order``. Step 5 has the empty ``Ledger(terms=())``. Step 6 adds kernel-side terms
measured at build, without a change of the field type. ``terms`` is a tuple of ``ErrorTerm`` of
``syncem.model.errors``, each with the five kinds of that module. A term value is in channel Stokes units and
has the shape the term states, usually ``(n_ch, 4)``. ``Ledger`` is an ``equinox.Module``, so term values are
array leaves and the ledger can pass through a transformation.

Not certified: the ledger records the terms it is given and their kinds. It does not verify them.
"""

from __future__ import annotations

from typing import ClassVar

import equinox as eqx

from syncem.model._provenance import strict_json
from syncem.model.errors import ErrorTerm

__all__ = ["Ledger"]


class Ledger(eqx.Module):
    """Tuple of ``ErrorTerm`` measured at build. Empty in step 5 (decision 149)."""

    LABEL: ClassVar[str] = "sec: frequency order"
    terms: tuple

    def __init__(self, terms: tuple = ()):
        if not isinstance(terms, tuple):
            raise ValueError(
                f"ledger terms must be a tuple, got {type(terms).__name__}"
            )
        for term in terms:
            if not isinstance(term, ErrorTerm):
                raise ValueError(f"ledger terms must be ErrorTerm, got {term!r}")
        self.terms = terms

    def to_dict(self) -> dict:
        """Strict-JSON dict ``{"terms": [term dict, ...]}``. Term values must be concrete."""
        return {"terms": [strict_json(t.to_dict()) for t in self.terms]}
