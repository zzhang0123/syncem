"""A reported value and its provenance (SPEC Sec. 3.8).

LABEL [extension].

A linear functional returns a scalar or a batch shape in the units carried
by its coefficients. A retained-space estimate has no certified error for
discarded directions.
"""

from typing import Any, NamedTuple


class Observable(NamedTuple):
    """A value, noise-only standard deviation and optional unconstrained share."""

    value: Any
    sigma: Any = None
    unconstrained: Any = None
    origin: str = "specified"
    label: str = ""
