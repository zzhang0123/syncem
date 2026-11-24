"""Declaration of the single-reference harmonic kernel bridge.

LABEL [extension].

Harmonic inputs use canonical variable units. Spectral modes have shape
(n_ch,) on the supplied responses. A single-reference response has no
certified error for a lattice of harmonic cells.
"""

from __future__ import annotations

from dataclasses import dataclass

from .capabilities import AnalyticTaylor, Lines, Parity
from .variables import SYNCHROTRON_VARIABLES


@dataclass(frozen=True)
class HarmonicKernel:
    """Build the harmonic response through the existing single-reference analytic path."""

    m_max: int | None = None

    name = "harmonic"
    variables = tuple(SYNCHROTRON_VARIABLES.values())
    primary = "gamma"
    azimuth = "phi"
    depth = "depth"
    stokes = ("I", "P", "V")
    traceable = False
    lines = True
    capabilities = (Lines(), AnalyticTaylor(), Parity(("I", "P"), ("V",)))

    def legacy(self, support, channels):
        """Select the harmonic count from declared support when it is omitted."""
        from ..model.harmonic import HarmonicKernel as Native
        from ..model.kernels import required_m_max

        count = required_m_max(support, channels) if self.m_max is None else self.m_max
        return Native(max(1, count))
