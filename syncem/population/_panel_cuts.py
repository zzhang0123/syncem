"""Physical quadrature boundaries and the residuals of inverse coordinate maps."""

from typing import NamedTuple

import numpy as np


class PanelCuts(NamedTuple):
    values: np.ndarray
    offsets: np.ndarray


def rounded_cuts(extra):
    for entry in extra:
        if isinstance(entry, PanelCuts):
            yield from entry.values
        else:
            yield entry


def panel_knots(law, lo, hi, extra):
    """Order boundaries by their physical value, including coincident rounded values."""
    cuts = [item for item in extra if isinstance(item, PanelCuts)]
    regular = [item for item in extra if not isinstance(item, PanelCuts)]
    regular.extend(getattr(law, "kinks", ()))
    regular = np.asarray([lo, hi, *(v for v in regular if lo < v < hi)])
    if not cuts:
        return np.unique(regular), None
    values = np.concatenate([regular, *(c.values for c in cuts)])
    offsets = np.concatenate([np.zeros_like(regular), *(c.offsets for c in cuts)])
    keep = ((values > lo) | ((values == lo) & (offsets >= 0))) & (
        (values < hi) | ((values == hi) & (offsets <= 0))
    )
    values, offsets = values[keep], offsets[keep]
    order = np.lexsort((offsets, values))
    values, offsets = values[order], offsets[order]
    unique = np.r_[True, (values[1:] != values[:-1]) | (offsets[1:] != offsets[:-1])]
    return values[unique], offsets[unique]
