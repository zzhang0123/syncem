"""Representations of a resolved variable other than cells: ``Legendre`` and ``AzimuthHarmonic`` (SPEC Sec. 3.3, 4.2).

LABEL ``eq: legendre spectral response``. ``Legendre(L)`` resolves a cosine by the degrees ``P_l``,
``l = 0..L``, with ``P_l(1) = 1``. The factor ``(2l + 1)/2`` sits in the basis spectra (SPEC Sec. 4.5).
``AzimuthHarmonic()`` resolves the kernel's azimuth by the exact harmonic ``exp(i s_X phi)``: it splits the
lattice rows into a real spin-0 block and a complex spin-2 block and adds no truncation.

Both hold no values of their variable, so they need no unit (``carries_values = False``). Units: none,
the arguments are dimensionless. Shapes: ``Legendre.functions(x)`` returns ``(L + 1, *x.shape)``. Not
certified: the truncation in ``l``, which the basis measures.
"""

from __future__ import annotations

from typing import Any, Hashable, Protocol

import equinox as eqx
import numpy as np

__all__ = ["AzimuthFourier", "AzimuthHarmonic", "Legendre", "Representation"]


class Representation(Protocol):
    """A representation of a resolved variable. Cells with equal signature share one compiled routine."""

    def signature(self) -> Hashable: ...


class Legendre(eqx.Module):
    """Legendre degrees ``l = 0..degree`` on a cosine, ``P_l(1) = 1``."""

    degree: int = eqx.field(static=True)
    given_unit: None = eqx.field(static=True, default=None)
    is_point: bool = eqx.field(static=True, default=False)
    carries_values: bool = eqx.field(static=True, default=False)

    def __check_init__(self) -> None:
        if (
            isinstance(self.degree, bool)
            or not isinstance(self.degree, (int, np.integer))
            or self.degree < 0
        ):
            raise ValueError(
                f"Legendre degree must be a non-negative int, not {self.degree!r}"
            )

    def functions(self, x: Any) -> np.ndarray:
        """``P_l(x)`` for ``l = 0..degree``, shape ``(degree + 1, *x.shape)``."""
        x = np.asarray(x, dtype=float)
        return np.stack(
            [
                np.polynomial.legendre.Legendre.basis(l)(x)
                for l in range(self.degree + 1)
            ]
        )

    def signature(self) -> Hashable:
        return ("Legendre", self.degree)

    def __repr__(self) -> str:
        return f"se.Legendre({self.degree})"


class AzimuthHarmonic(eqx.Module):
    """The exact azimuth harmonic: blocks ``M0``, ``Re M2``, ``Im M2`` over the same rows."""

    given_unit: None = eqx.field(static=True, default=None)
    is_point: bool = eqx.field(static=True, default=False)
    carries_values: bool = eqx.field(static=True, default=False)

    @property
    def modes(self) -> tuple[int, ...]:
        return (2,)

    def signature(self) -> Hashable:
        return ("AzimuthHarmonic",)

    def __repr__(self) -> str:
        return "se.AzimuthHarmonic()"


class AzimuthFourier(eqx.Module):
    """Shared real and imaginary moments with modes 1 through m_max + 2."""

    m_max: int = eqx.field(static=True)
    given_unit: None = eqx.field(static=True, default=None)
    is_point: bool = eqx.field(static=True, default=False)
    carries_values: bool = eqx.field(static=True, default=False)

    def __check_init__(self):
        if (
            isinstance(self.m_max, (bool, np.bool_))
            or not isinstance(self.m_max, (int, np.integer))
            or self.m_max < 0
        ):
            raise ValueError("AzimuthFourier needs a non-negative integer m_max")

    @property
    def modes(self) -> range:
        """Positive moment frequencies, including the spin-two offset."""
        return range(1, self.m_max + 3)

    @property
    def moment_nodes(self) -> int:
        """Minimum Gauss nodes on a panel pairing the highest retained mode."""
        return max(32, 4 * (self.m_max + 2) + 16)

    def signature(self) -> Hashable:
        return ("AzimuthFourier", int(self.m_max))

    def __repr__(self) -> str:
        return f"se.AzimuthFourier({self.m_max})"
