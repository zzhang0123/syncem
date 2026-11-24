"""Tables of fit statistics or differences between spectral predictions.

LABEL [extension].

Comparison rows hold scalar chi-square and related dimensionless summaries
on the shared data shape. Differences between estimators carry no certified
model preference.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Comparison:
    """A labelled comparison. Differences use the second spectrum minus the first."""

    headers: tuple[str, ...]
    rows: tuple[tuple, ...]
    reference: object = None
    candidate: object = None

    @property
    def difference(self):
        if self.reference is not None:
            return self.candidate.stokes - self.reference.stokes
        return None

    def table(self):
        def display(entry):
            if entry is None:
                return "—"
            if isinstance(entry, str):
                return entry
            return np.array2string(np.asarray(entry), precision=6, threshold=8)

        return "\n".join(
            "  ".join(display(entry) for entry in row)
            for row in (self.headers, *self.rows)
        )

    def __str__(self):
        return self.table()
