"""The kernel protocol of the 0.8 layer (design/0.8/SPEC.md Sec. 3.2, 8.1).

LABEL ``eq: channel kernel``. A kernel names its variables, the variable resolved by default (``primary``), its
Stokes blocks, the variable that enters only through ``exp(i s_X phi)`` (``azimuth``), the variable of the
propagation phase (``depth``), whether JAX can trace it, whether it has Dirac lines, and its capabilities.
``channel(channels, point, *, phase, per_node=False)`` returns the channel-integrated ``Modes`` at one point:
``I`` and ``V`` of shape ``(n_ch,)``, ``P`` of shape ``(n_w, n_ch)`` complex without the azimuth factor, with
the phase inside the channel integral. ``phase=False`` gives the phase-free kernel. ``per_node=True`` returns
the weighted values at the channel quadrature nodes, with a trailing axis ``n_nu``. ``regime(box, channels)``
accepts or refuses one cell box with a reason.

Units: per-electron channel Stokes, erg s^-1 sr^-1 Hz^-1 for unit-area channels. Not certified: the protocol
states a contract, and each kernel states its own physical model and errors.
"""

from __future__ import annotations

from typing import Any, NamedTuple, Protocol, runtime_checkable

from ..model._kernel_helpers import Modes

__all__ = ["Kernel", "Modes", "Regime"]


class Regime(NamedTuple):
    """Outcome of ``Kernel.regime`` for one cell box: accepted, or refused with the reason."""

    ok: bool
    reason: str = ""


@runtime_checkable
class Kernel(Protocol):
    """Contract of a base-model kernel (SPEC Sec. 3.2)."""

    name: str
    variables: tuple
    primary: str | None
    stokes: tuple[str, ...]
    azimuth: str | None
    depth: str | None
    traceable: bool
    lines: bool
    capabilities: tuple

    def channel(
        self, channels: Any, point: dict, *, phase: Any, per_node: bool = False
    ) -> Modes: ...

    def spectral(self, nu: Any, point: dict) -> Modes | None: ...

    def regime(self, box: dict, channels: Any) -> Regime: ...

    def physical_error(self, channels: Any) -> Any: ...

    def describe(self) -> tuple: ...
