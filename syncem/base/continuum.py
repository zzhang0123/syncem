"""The continuum kernel behind the 0.8 kernel protocol (design/0.8/SPEC.md Sec. 3.2, 8.1, 8.3).

LABEL ``eq: directional continuum``. ``ContinuumKernel`` adapts ``syncem.model.ContinuumKernel`` without changing
it: ``K_I = A F(x)``, ``K_Q = -A G(x)``, ``K_V = 0`` with ``x = nu / (a_B gamma^2)`` and ``B_perp = B sqrt(1 -
eta^2)``. The pitch cosine ``mu`` does not enter: the kernel is the isotropic pitch average, and ``Beaming``
declares the limit ``K_nat -> 2 K_iso(eta) delta(mu - eta)``. ``P`` carries the phase law and no azimuth factor.
Its variables are the six synchrotron variables, ``gamma`` is primary, ``phi`` the azimuth and ``depth`` the
depth. Capabilities: ``Similarity``, ``FieldScalingRelation``, ``Beaming``, ``Parity``, ``EnergyWindow``.

``x_min`` guards the small-``x`` end of ``F`` and ``G``. ``x_min=None`` takes the model default 1e-6 when the
kernel is called directly. The engine calls ``resolved(channels, gamma_max, b_perp_max)``, which sets ``x_min``
to half of ``x`` at the lowest channel-support node, the top lattice edge and the largest ``B_perp`` (SPEC Sec.
8.3), so that the last cell is admitted (T3). The default 1e-6 refuses the last paper cell at 0.1 GHz.

Units: per-electron channel Stokes, erg s^-1 sr^-1 Hz^-1 for unit-area channels. Shapes: ``Modes`` with ``I``
``(n_ch,)`` and ``P`` ``(1, n_ch)``, or a trailing ``n_nu`` axis with ``per_node=True``. Not certified: the
continuum limit itself (``physical_error`` is unbounded unless declared), the channel quadrature and the ``F``,
``G`` quadrature (only their tail is bounded).
"""

from __future__ import annotations

from typing import Any, ClassVar

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..model._continuum import ContinuumKernel as _ModelContinuum
from ..model._kernel_helpers import Modes
from .capabilities import (
    Beaming,
    EnergyWindow,
    FieldScalingRelation,
    Parity,
    Similarity,
)
from .kernel import Regime
from .phase import PhaseLaw
from .variables import SYNCHROTRON_VARIABLES
from .window import A1_HZ_PER_G, continuum_window

__all__ = ["ContinuumKernel"]

_DEFAULT_X_MIN = 1e-6


def _response_weights(channels: Any) -> Any:
    if hasattr(channels, "response_weights"):
        return channels.response_weights()
    return channels.weights * channels.node_values()


class ContinuumKernel(eqx.Module):
    """Ultra-relativistic continuum kernel, the isotropic pitch average (PAPER eq: directional continuum)."""

    gamma_min: float = eqx.field(static=True, default=300.0, kw_only=True)
    x_min: float | None = eqx.field(static=True, default=None, kw_only=True)
    n_nodes_F: int = eqx.field(static=True, default=128, kw_only=True)
    tail_cutoff: float = eqx.field(static=True, default=50.0, kw_only=True)

    name: ClassVar[str] = "continuum"
    variables: ClassVar[tuple] = tuple(SYNCHROTRON_VARIABLES.values())
    primary: ClassVar[str] = "gamma"
    stokes: ClassVar[tuple[str, ...]] = ("I", "P")
    azimuth: ClassVar[str] = "phi"
    depth: ClassVar[str] = "depth"
    traceable: ClassVar[bool] = True
    lines: ClassVar[bool] = False
    capabilities: ClassVar[tuple] = (
        Similarity(("gamma",)),
        FieldScalingRelation("B", "gamma"),
        Beaming("mu", "eta"),
        Parity(("I", "P"), ("V",)),
        EnergyWindow(continuum_window),
    )

    def __check_init__(self) -> None:
        if not self.gamma_min > 1.0:
            raise ValueError(f"gamma_min must exceed 1, not {self.gamma_min!r}")
        if self.x_min is not None and not self.x_min > 0:
            raise ValueError(f"x_min must be positive or None, not {self.x_min!r}")

    @property
    def model(self) -> _ModelContinuum:
        """The unchanged kernel of ``syncem.model`` with this ``x_min``."""
        x_min = _DEFAULT_X_MIN if self.x_min is None else self.x_min
        return _ModelContinuum(
            n_nodes_F=self.n_nodes_F, tail_cutoff=self.tail_cutoff, x_min=x_min
        )

    @staticmethod
    def x_min_for(
        channels: Any, gamma_max: float, b_perp_max: float, margin: float = 0.5
    ) -> float:
        """``margin`` times ``x`` at the lowest channel-support node, ``gamma_max`` and ``b_perp_max`` (G).

        The arguments must be concrete numbers: ``x_min`` is a static field.
        """
        nu_lo = float(np.min(np.asarray(channels.nodes)))
        gamma_max, b_perp_max = float(gamma_max), float(b_perp_max)
        if not (gamma_max > 0 and b_perp_max > 0 and nu_lo > 0 and 0 < margin <= 1):
            raise ValueError(
                "x_min_for needs positive gamma_max, b_perp_max, frequencies and 0 < margin <= 1"
            )
        return float(
            margin * nu_lo / (A1_HZ_PER_G * float(b_perp_max) * float(gamma_max) ** 2)
        )

    def resolved(
        self, channels: Any, gamma_max: float, b_perp_max: float
    ) -> "ContinuumKernel":
        """The kernel with ``x_min`` set from the channels, the top lattice edge and the largest ``B_perp``."""
        x_min = self.x_min_for(channels, gamma_max, b_perp_max)
        return ContinuumKernel(
            gamma_min=self.gamma_min,
            x_min=x_min,
            n_nodes_F=self.n_nodes_F,
            tail_cutoff=self.tail_cutoff,
        )

    # ------------------------------------------------------------ evaluation
    def _node_values(self, nu: Any, point: dict, phase: Any) -> tuple[Any, Any]:
        k_i, k_q = self.model.kernels(
            nu, point["gamma"], point["B"], point.get("eta", 0.0)
        )
        k_p = k_q.astype(jnp.result_type(k_q, 1j))
        if phase is not False and phase is not None:
            if not isinstance(phase, PhaseLaw):
                raise TypeError("phase must be a PhaseLaw, False or None")
            k_p = k_p * phase.weights(nu, point.get("depth", 0.0))
        return k_i, k_p

    def channel(
        self, channels: Any, point: dict, *, phase: Any = None, per_node: bool = False
    ) -> Modes:
        """Channel-integrated ``Modes`` at one point (``gamma``, ``B`` in G, optional ``eta`` and ``depth``)."""
        nodes = jnp.asarray(channels.nodes)
        measure = _response_weights(channels)
        k_i, k_p = self._node_values(nodes, point, phase)
        if per_node:
            i = measure * k_i
            return Modes(I=i, V=jnp.zeros_like(i), P=(measure * k_p)[None])
        i = jnp.sum(measure * k_i, axis=-1)
        return Modes(I=i, V=jnp.zeros_like(i), P=jnp.sum(measure * k_p, axis=-1)[None])

    def spectral(self, nu: Any, point: dict, *, phase: Any = None) -> Modes:
        """Kernel values per Hz at frequencies ``nu``, for monochromatic channels."""
        k_i, k_p = self._node_values(jnp.asarray(nu), point, phase)
        return Modes(I=k_i, V=jnp.zeros_like(k_i), P=k_p[None])

    def regime(self, box: dict, channels: Any = None) -> Regime:
        """Accepted when the cell's ``gamma`` interval lies at or above ``gamma_min``."""
        lo, hi = box["gamma"]
        if lo < self.gamma_min * (1 - 1e-12):
            return Regime(
                False,
                f"cell gamma {lo:g} to {hi:g} lies below gamma_min = {self.gamma_min:g} of the "
                "continuum kernel. Start the cells there, or pass ContinuumKernel(gamma_min=...)",
            )
        return Regime(True)

    def physical_error(self, channels: Any) -> Any:
        return self.model.physical_error(channels)

    def describe(self) -> tuple:
        return (
            ("name", self.name),
            ("gamma_min", self.gamma_min),
            ("x_min", self.x_min),
            ("n_nodes_F", self.n_nodes_F),
            ("tail_cutoff", self.tail_cutoff),
        )
