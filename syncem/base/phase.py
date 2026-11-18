"""Propagation phase laws ``exp(i tau(nu) depth)`` (design/0.8/SPEC.md Sec. 3.2, 4.3).

LABEL ``eq: channel kernel``. ``PhaseLaw(tau, screen=None)`` multiplies the spin-2 block at every channel node by
``exp(i tau(nu) depth)`` with the point's depth, and by ``screen(tau)`` when an independent screen is given (the
characteristic function of an averaged depth law, or a foreground factor, ``[extension]``).
``FaradayRotation(screen=None)`` is the law with ``tau(nu) = 2 (c / nu)^2`` in m^2, so ``tau depth`` is in radians
for a depth in rad m^-2 (``syncem.model.phase.phase_coordinate``).

Units: ``nu`` in Hz, ``tau`` in m^2, depth in rad m^-2. Shapes: ``weights(nu, depth)`` has the shape of ``nu``.
Not certified: the screen model, which the caller declares, and the accuracy of large phases.
"""

from __future__ import annotations

from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp

from ..model.phase import phase_coordinate

__all__ = ["FaradayRotation", "PhaseLaw"]


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


def _screen(screen: Any, tau: Any) -> Any:
    """``screen(tau)`` as one complex weight per node (a model screen returns ``(1, *tau.shape)``)."""
    value = screen(tau)
    return value[0] if getattr(value, "ndim", 0) == jnp.ndim(tau) + 1 else value


def FaradayRotation(
    *, screen: Any = None
) -> PhaseLaw:  # noqa: N802  (a constructor in the front door)
    """``tau(nu) = 2 (c / nu)^2`` in m^2, with an optional independent screen."""
    return PhaseLaw(phase_coordinate, screen=screen)
