"""Typed kernel capabilities (design/0.8/SPEC.md Sec. 3.2, 8.1).

LABEL ``[extension]``. A capability declares a shortcut the engine may take for a kernel. Each has a
precondition that the plan checks before using it (SPEC Sec. 8.1). ``Similarity`` declares that the kernel is
``B_perp W(nu / (a_1 gamma^2 B_perp))`` in its variables. ``FieldScalingRelation`` declares
``Kbar(gamma, L B) = L Kbar(L^(1/2) gamma, B)`` (PAPER eq: field scaling). ``Beaming`` declares that only
``mu = eta`` contributes. ``Parity`` lists the Stokes blocks that are even and odd under the joint reversal of
the cosines. ``EnergyWindow`` carries a function that gives the default energy range. ``ReferenceDensity``
gives the default reference weights on an axis. ``Lines``, ``AnalyticTaylor`` and ``HostOnly`` mark harmonic
lines, an analytic Taylor route and host-only code.

Units and shapes: none of their own. Not certified: a capability is a declaration by the kernel author, and the
test suite checks the shortcuts it enables against the general route.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "AnalyticTaylor",
    "Beaming",
    "EnergyWindow",
    "FieldScalingRelation",
    "HostOnly",
    "Lines",
    "Parity",
    "ReferenceDensity",
    "Similarity",
]


@dataclass(frozen=True)
class Similarity:
    """The kernel is ``B_perp W(nu / (a_1 gamma^2 B_perp))`` in ``variables``: one table serves all cells."""

    variables: tuple[str, ...] = ("gamma",)


@dataclass(frozen=True)
class FieldScalingRelation:
    """``Kbar(gamma, L B) = L Kbar(L^(1/2) gamma, B)``: field columns from energy columns."""

    variable: str = "B"
    energy: str = "gamma"


@dataclass(frozen=True)
class Beaming:
    """Only ``variable = partner`` contributes, ``K_nat -> 2 K_iso(partner) delta(variable - partner)``."""

    variable: str = "mu"
    partner: str = "eta"


@dataclass(frozen=True)
class Parity:
    """Stokes blocks even and odd under the joint reversal of the cosines (PAPER eq: angular parity)."""

    even: tuple[str, ...] = ("I", "P")
    odd: tuple[str, ...] = ("V",)


@dataclass(frozen=True)
class EnergyWindow:
    """``window(averaged, channels, *, density, eps) -> (lo, hi)`` in the kernel's scaled variable."""

    window: Callable[..., tuple[float, float]]


@dataclass(frozen=True)
class ReferenceDensity:
    """Default reference density on one axis (SPEC Sec. 4.5)."""

    variable: str
    density: Callable[[Any], Any]


@dataclass(frozen=True)
class Lines:
    """Dirac lines: channels must be smooth and of finite width."""


@dataclass(frozen=True)
class AnalyticTaylor:
    """The kernel has an analytic Taylor route for its angular projection."""


@dataclass(frozen=True)
class HostOnly:
    """Host code that JAX cannot trace. Columns come through the cell table."""
