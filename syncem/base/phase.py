"""Propagation phase laws ``exp(i tau(nu) depth)`` (design/0.8/SPEC.md Sec. 3.2, 4.3).

LABEL ``eq: channel kernel``. ``PhaseLaw(tau, screen=None)`` multiplies the spin-2 block at every channel node by
``exp(i tau(nu) depth)`` with the point's depth, and by ``screen(tau)`` when an independent screen is given (the
characteristic function of an averaged depth law, or a foreground factor, ``[extension]``).
``FaradayRotation(screen=None)`` is the law with ``tau(nu) = 2 (c / nu)^2`` in m^2, so ``tau depth`` is in radians
for a depth in rad m^-2 (``syncem.model.phase.phase_coordinate``).

``channel_fingerprints(channels)`` gives one sha256 digest per channel of ``tau`` and the screen at that
channel's nodes (design/0.8/SPEC.md decision 159), and ``phase_fingerprints(None, channels)`` gives the
"no phase" tag per channel.

Units: ``nu`` in Hz, ``tau`` in m^2, depth in rad m^-2. Shapes: ``weights(nu, depth)`` has the shape of ``nu``,
and the fingerprints are a tuple of ``n_ch`` strings. Not certified: the screen model, which the caller declares, and the accuracy of large phases.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..model.phase import phase_coordinate
from ._fingerprint import digest

__all__ = ["FaradayRotation", "PhaseLaw", "phase_fingerprints"]

_NO_PHASE = ("phase", "no phase")


class PhaseLaw(eqx.Module):
    """``exp(i tau(nu) depth)`` times an optional independent screen ``screen(tau)``."""

    tau: Callable[[Any], Any] = eqx.field(static=True)
    screen: Any = eqx.field(default=None, kw_only=True)

    def weights(self, nu: Any, depth: Any = 0.0) -> Any:
        """Complex weight at frequencies ``nu`` (Hz) for one depth (rad m^-2)."""
        tau = self.tau(nu)
        out = jnp.exp(1j * tau * depth)
        if self.screen is not None:
            out = out * _screen(self.screen, tau)
        return out

    def channel_fingerprints(self, channels: Any) -> tuple[str, ...]:
        """One digest per channel of ``tau(nodes_j)`` and ``screen(tau(nodes_j))`` as canonical arrays.

        Each channel is evaluated on its own nodes, so a selection of channels keeps its digests. Raises
        ``NotCacheable`` when ``tau`` or the screen is traced.
        """
        nodes = np.asarray(channels.nodes)
        out = []
        for row in nodes:
            tau = self.tau(row)
            screen = None if self.screen is None else _screen(self.screen, tau)
            out.append(digest(("phase", tau, screen)))
        return tuple(out)


def _screen(screen: Any, tau: Any) -> Any:
    """``screen(tau)`` as one complex weight per node (a model screen returns ``(1, *tau.shape)``)."""
    value = screen(tau)
    return value[0] if getattr(value, "ndim", 0) == jnp.ndim(tau) + 1 else value


def phase_fingerprints(phase: Any, channels: Any) -> tuple[str, ...]:
    """Phase fingerprints per channel: those of a ``PhaseLaw``, or the "no phase" tag for ``None`` and ``False``."""
    if phase is None or phase is False:
        return (digest(_NO_PHASE),) * int(np.shape(channels.nodes)[0])
    if not isinstance(phase, PhaseLaw):
        raise TypeError("phase must be a PhaseLaw, False or None")
    return phase.channel_fingerprints(channels)


def FaradayRotation(
    *, screen: Any = None
) -> PhaseLaw:  # noqa: N802  (a constructor in the front door)
    """``tau(nu) = 2 (c / nu)^2`` in m^2, with an optional independent screen."""
    return PhaseLaw(phase_coordinate, screen=screen)
