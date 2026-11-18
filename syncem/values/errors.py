"""Error classes of the 0.8 layer (design/0.8/SPEC.md Sec. 2.5 and 3.8).

LABEL ``[extension]``. Every class derives from ``SyncEMError``, a ``ValueError``. Each message names the
remedy and the call that takes it. ``SyncEMError`` and ``UnitError`` are defined in ``syncem.units``,
which comes before this layer and raises ``UnitError`` itself, and are re-exported here.

Units and shapes: none. Not certified: the remedies printed by the raising code are suggestions,
validated by the tests that pin each message.
"""

from __future__ import annotations

from ..units import SyncEMError, UnitError

__all__ = [
    "DataError",
    "DepthExpansionError",
    "ImproperPosterior",
    "NothingToPredict",
    "RegimeError",
    "RoleError",
    "SizeError",
    "SpecifiedMismatch",
    "SyncEMError",
    "UnitError",
]


class RoleError(SyncEMError):
    """A variable's role (resolved, averaged, unknown) has no route for the declared population or lattice."""


class RegimeError(SyncEMError):
    """A cell lies outside the regime of every admissible kernel."""


class SpecifiedMismatch(SyncEMError):
    """A basis built for one set of specified statistics is used with a population that declares others."""


class NothingToPredict(SyncEMError):
    """A prediction from a population whose moments are unknown and not fitted."""


class DataError(SyncEMError):
    """Data that the declared model cannot describe, such as Q and U with an undeclared azimuth."""


class DepthExpansionError(SyncEMError):
    """The depth bound of the Faraday-depth expansion exceeds ``depth_tol`` (SPEC Sec. 4.6, rule 4)."""


class SizeError(SyncEMError):
    """A lattice or basis above ``max_columns`` (SPEC Sec. 4.6)."""


class ImproperPosterior(SyncEMError):
    """``log_prob`` over free moments without a prior, which is not normalisable (SPEC Sec. 6.6)."""
