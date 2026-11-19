"""Channels of the 0.8 layer: unit-area responses with unit-aware constructors (design/0.8/SPEC.md Sec. 3.2).

LABEL ``eq: forward response families``. ``Channels`` subclasses ``syncem.model.channels.Channels`` and
adds the unit the frequencies were given in. Its constructors build unit-area responses
(``normalisation="unit_integral"``):

* ``Channels.log(lo, hi, n, *, unit, ...)`` and ``Channels.at(frequencies, *, unit, ...)`` give the bump
  ``R_j = rho[(nu - nu_j)/h_j] / (h_j Z)`` of the manuscript, with ``rho(t) = exp(-t^2/(1-t^2))``. The
  width is ``fractional_width=f``, read as in the manuscript, ``h_j = f nu_j / sigma_rho`` with
  ``sigma_rho^2 = Z^-1 int t^2 rho dt``, so that ``f nu_j`` is the spread of ``R_j``. Or it is
  ``half_width=x``, the support half-width ``h_j = x nu_j`` directly. The paper's examples use
  ``h_j = 0.2 nu_j`` (``half_width=0.2``), which the manuscript labels ``f = 8.0%`` after rounding
  (``0.2 sigma_rho = 0.0795``). With neither, the channels are monochromatic: one node at each centre
  with weight 1, for kernels without lines.
* ``Channels.from_bandpass(tables, *, unit)`` takes one ``(frequencies, response)`` table per channel,
  each falling to zero at both ends, and interpolates linearly. The tables are merged on the union of
  their grids, which is exact for the linear interpolant.

``channels[rows]`` selects rows. ``response_weights()`` returns ``weights * R_j(nodes)``, so that
``sum_t w_jt f(nu_jt)`` is the band average ``int R_j f dnu``. Units: frequencies in Hz inside,
``unit`` records the unit given. Shapes: ``(n_ch, n_nu)`` for nodes and weights. Not certified: the
adequacy of a response as a model of a real bandpass, and the accuracy of the Gauss-Legendre rule
across the kinks of a tabulated bandpass (raise ``n_nu``).
"""

from __future__ import annotations

import dataclasses
import sys
from typing import Any

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..model.channels import Channels as _ModelChannels
from ..units import Quantity, UnitError, dimension_of, parse, to_canonical
from ._fingerprint import NotCacheable, digest

__all__ = ["SIGMA_RHO", "Channels"]


def _sigma_rho() -> float:
    t, w = np.polynomial.legendre.leggauss(512)
    rho = np.exp(-(t**2) / (1 - t**2))
    return float(np.sqrt(np.sum(w * t**2 * rho) / np.sum(w * rho)))


SIGMA_RHO = _sigma_rho()


def _is_pair(value: Any) -> bool:
    return (
        isinstance(value, tuple)
        and len(value) == 2
        and isinstance(value[1], str)
        and not isinstance(value[0], str)
    )


def _astropy(value: Any) -> bool:
    astropy_units = sys.modules.get("astropy.units")
    return astropy_units is not None and isinstance(value, astropy_units.Quantity)


def _carries_unit(value: Any) -> bool:
    return isinstance(value, (str, Quantity)) or _is_pair(value) or _astropy(value)


def _label(value: Any) -> str:
    if isinstance(value, str):
        return parse(value)[1]
    if isinstance(value, Quantity):
        return value.unit
    if _is_pair(value):
        return value[1]
    return value.unit.to_string()


def _in_hz(value: Any) -> np.ndarray:
    return np.atleast_1d(
        np.asarray(to_canonical(value, "frequency", name="frequency"), dtype=float)
    )


def _frequencies(values: Any, unit: str | None, call: str) -> tuple[np.ndarray, str]:
    """Frequencies in Hz and the unit label. Strings, pairs and quantities carry their own unit."""
    listed = isinstance(values, (list, tuple)) and not _is_pair(values)
    carried = _carries_unit(values) or (
        listed and any(_carries_unit(v) for v in values)
    )
    if unit is not None:
        if carried:
            raise UnitError(
                f"{call} gives units in the values and unit={unit!r}. Drop unit=, or pass plain numbers."
            )
        found = dimension_of(unit)
        if found != "frequency":
            raise UnitError(
                f'{call} has unit {unit} ({found or "a product of units"}). Pass a frequency unit such as "GHz".'
            )
        return _in_hz((np.asarray(values, dtype=float), unit)), unit
    if _carries_unit(values):
        return _in_hz(values), _label(values)
    if listed and values and all(_carries_unit(v) for v in values):
        labels = {_label(v) for v in values}
        return np.concatenate([_in_hz(v) for v in values]), (
            labels.pop() if len(labels) == 1 else "Hz"
        )
    raise UnitError(
        f'{call} has no unit. Pass unit="GHz", or give the frequencies as strings such as "1.4 GHz".'
    )


def _concrete(values: Any) -> Any:
    """``values`` unchanged, or ``NotCacheable`` when they are traced."""
    if isinstance(values, jax.core.Tracer):
        raise NotCacheable(f"traced channel array of shape {values.shape} has no value")
    return values


def _positive_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive int, not {value!r}")
    return int(value)


def _half_widths(
    fractional_width: float | None, half_width: float | None
) -> float | None:
    """Support half-width as a fraction of the centre, or ``None`` for monochromatic channels."""
    if fractional_width is not None and half_width is not None:
        raise ValueError("pass one of fractional_width and half_width, not both")
    if half_width is not None:
        if not 0 < half_width < 1:
            raise ValueError(
                f"half_width must lie in (0, 1) so that the support stays above zero, not {half_width!r}"
            )
        return float(half_width)
    if fractional_width is not None:
        if not 0 < fractional_width < SIGMA_RHO:
            raise ValueError(
                f"fractional_width must lie in (0, {SIGMA_RHO:.4f}) so that the support stays above zero, "
                f"not {fractional_width!r}"
            )
        return float(fractional_width) / SIGMA_RHO
    return None


class Channels(_ModelChannels):
    """``syncem.model.Channels`` with unit-aware constructors and the given unit (static)."""

    unit: str = eqx.field(static=True, default="Hz")

    @classmethod
    def log(
        cls,
        lo: Any,
        hi: Any,
        n: int,
        *,
        unit: str | None = None,
        fractional_width: float | None = None,
        half_width: float | None = None,
        n_nu: int = 64,
    ) -> "Channels":
        """``n`` centres spaced logarithmically from ``lo`` to ``hi``, both included."""
        call = f"Channels.log({lo!r}, {hi!r}, {n!r})"
        (lo_hz, hi_hz), label = _frequencies([lo, hi], unit, call)
        n = _positive_int(n, "n")
        if not 0 < lo_hz <= hi_hz or (n > 1) == (lo_hz == hi_hz):
            raise ValueError(f"{call}: need 0 < lo < hi (or lo == hi with n == 1)")
        centres = np.geomspace(lo_hz, hi_hz, n) if n > 1 else np.array([lo_hz])
        return cls._finite(
            centres, label, _half_widths(fractional_width, half_width), n_nu
        )

    @classmethod
    def at(
        cls,
        frequencies: Any,
        *,
        unit: str | None = None,
        fractional_width: float | None = None,
        half_width: float | None = None,
        n_nu: int = 64,
    ) -> "Channels":
        """Channels centred at the given frequencies, in the given order."""
        centres, label = _frequencies(
            frequencies, unit, f"Channels.at({frequencies!r})"
        )
        if (
            centres.ndim != 1
            or centres.size == 0
            or np.any(~np.isfinite(centres))
            or np.any(centres <= 0)
        ):
            raise ValueError(
                "Channels.at needs a non-empty list of positive finite frequencies"
            )
        return cls._finite(
            centres, label, _half_widths(fractional_width, half_width), n_nu
        )

    @classmethod
    def from_bandpass(
        cls, tables: Any, *, unit: str | None = None, n_nu: int = 256
    ) -> "Channels":
        """One ``(frequencies, response)`` table per channel, linear between samples, zero outside."""
        grids, rows, labels = [], [], set()
        for k, (frequencies, response) in enumerate(tables):
            grid, label = _frequencies(
                frequencies, unit, f"Channels.from_bandpass table {k}"
            )
            labels.add(label)
            values = np.asarray(response, dtype=float)
            if (
                grid.shape != values.shape
                or grid.size < 2
                or np.any(np.diff(grid) <= 0)
            ):
                raise ValueError(
                    f"bandpass table {k}: frequencies must increase and match the response in length"
                )
            if (
                np.any(values < 0)
                or values[0] != 0
                or values[-1] != 0
                or not np.any(values > 0)
            ):
                raise ValueError(
                    f"bandpass table {k}: the response must be non-negative, positive somewhere, "
                    "and zero at both ends"
                )
            grids.append(grid)
            rows.append(values)
        if not grids:
            raise ValueError("Channels.from_bandpass needs at least one table")
        union = np.unique(np.concatenate(grids))
        merged = np.stack(
            [np.interp(union, g, r, left=0.0, right=0.0) for g, r in zip(grids, rows)]
        )
        out = cls.from_table(
            union,
            merged,
            smoothness=0,
            normalisation="unit_integral",
            n_nu=_positive_int(n_nu, "n_nu"),
        )
        return dataclasses.replace(out, unit=labels.pop() if len(labels) == 1 else "Hz")

    @classmethod
    def _finite(
        cls, centres: np.ndarray, label: str, half: float | None, n_nu: int
    ) -> "Channels":
        if half is None:
            c = jnp.asarray(centres)
            return cls(
                family="monochromatic",
                centres_hz=c,
                widths_hz=jnp.zeros_like(c),
                support=jnp.stack([c, c], axis=-1),
                nodes=c[:, None],
                weights=jnp.ones_like(c)[:, None],
                normalisation="unit_integral",
                smoothness=0,
                n_nu=1,
                unit=label,
            )
        out = cls.bump(
            centres,
            half * centres,
            normalisation="unit_integral",
            n_nu=_positive_int(n_nu, "n_nu"),
        )
        return dataclasses.replace(out, unit=label)

    # ------------------------------------------------------------ evaluation
    def __call__(self, nu_hz: Any) -> Any:
        if self.family == "monochromatic":
            raise ValueError(
                "monochromatic channels have no response function. Kernels evaluate them at the centres"
            )
        return super().__call__(nu_hz)

    @classmethod
    def from_basis(
        cls, reference: Any, coefficients: Any, *, n_nu: int | None = None
    ) -> "Channels":
        """``R_j = sum_k c_jk phi_k`` over the responses of ``reference``, which keeps its unit."""
        if getattr(reference, "family", None) == "monochromatic":
            raise ValueError(
                "monochromatic channels have no response to combine. Use channels of finite width"
            )
        out = super().from_basis(reference, coefficients, n_nu=n_nu)
        return dataclasses.replace(out, unit=getattr(reference, "unit", "Hz"))

    def peak_scale(self) -> Any:
        if self.family == "monochromatic":
            return jnp.ones_like(self.centres_hz)
        return super().peak_scale()

    def node_values(self) -> Any:
        if self.family == "monochromatic":
            return jnp.ones_like(self.weights)
        return super().node_values()

    def response_weights(self) -> Any:
        """``(n_ch, n_nu)`` weights of the band average: ``weights * R_j(nodes)``."""
        return self.weights * self.node_values()

    def row_keys(self) -> tuple[str, ...]:
        """One sha256 digest per channel (decision 153), for keying stored rows channel by channel.

        Channel ``j`` hashes ``nodes[j]``, ``response_weights()[j]`` and the ``family``, ``normalisation``
        and ``smoothness`` labels. The ``unit`` label is not hashed, so channels given in GHz and in MHz
        at the same frequencies share keys. ``channels[rows].row_keys()`` equals the selected keys when
        the selection leaves the arrays of each row unchanged. Raises ``NotCacheable`` under a trace.
        """
        nodes = np.asarray(_concrete(self.nodes))
        weights = np.asarray(_concrete(self.response_weights()))
        labels = ("channel row", self.family, self.normalisation, self.smoothness)
        return tuple(
            digest((*labels, nodes[j], weights[j])) for j in range(nodes.shape[0])
        )

    def __getitem__(self, rows: Any) -> "Channels":
        index = np.arange(self.n_ch)[rows]
        index = np.atleast_1d(index)
        if index.size == 0:
            raise ValueError("channel selection is empty")

        def pick(values: Any) -> Any:
            return None if values is None else tuple(values[i] for i in index)

        table = None if self.table is None else (self.table[0], pick(self.table[1]))
        bounds = self.derivative_bound
        return dataclasses.replace(
            self,
            centres_hz=self.centres_hz[index],
            widths_hz=self.widths_hz[index],
            support=self.support[index],
            nodes=self.nodes[index],
            weights=self.weights[index],
            table=table,
            coefficients=(
                None if self.coefficients is None else self.coefficients[index]
            ),
            response_fn=pick(self.response_fn),
            derivative_bound=(
                None
                if bounds is None
                else tuple((order, pick(values)) for order, values in bounds)
            ),
        )
