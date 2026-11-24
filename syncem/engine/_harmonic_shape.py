"""Finite Bessel-amplitude tables for the harmonic shape basis."""

from dataclasses import dataclass
from numbers import Integral

import numpy as np

from ..constants import C_CGS, E_ESU, M_E
from ..model.high_order_harmonic import (
    _active_range,
    _geometry,
    _prepare_host_responses,
    _scipy_special,
    high_order_line_powers,
)
from ..values.errors import RegimeError, SizeError

_POWER = 2 * np.pi * E_ESU**2 / C_CGS


def _ordered_bounds(values, size=None):
    result = np.asarray(values)
    if (
        result.ndim != 1
        or result.size < 2
        or (size is not None and result.size != size)
        or np.iscomplexobj(result)
    ):
        raise ValueError("expected an increasing real one-dimensional set of bounds")
    result = result.astype(float)
    if np.any(~np.isfinite(result)) or np.any(result[1:] <= result[:-1]):
        raise ValueError("bounds must be finite and strictly increasing")
    return result


def _positive_int(value, name):
    if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
        raise ValueError(f"{name} must be a positive integer")
    return int(value)


def _channel_orders(supports, spacing, max_orders):
    ranges = [
        (
            _active_range((lo, hi), spacing[1])[0],
            _active_range((lo, hi), spacing[0])[1],
        )
        for lo, hi in supports
    ]
    merged = []
    for lo, hi in sorted(ranges):
        if merged and lo <= merged[-1][1] + 1:
            merged[-1] = (merged[-1][0], max(merged[-1][1], hi))
        else:
            merged.append((lo, hi))
    count = sum(hi - lo + 1 for lo, hi in merged)
    if count > max_orders:
        raise SizeError(
            f"spacing interval and channels require {count} line orders including "
            f"edge padding. Narrow spacing or increase max_orders={max_orders}."
        )
    orders = np.concatenate([np.arange(lo, hi + 1) for lo, hi in merged])
    selections = tuple(
        slice(
            int(np.searchsorted(orders, lo)),
            int(np.searchsorted(orders, hi, side="right")),
        )
        for lo, hi in ranges
    )
    return orders, selections


def _amplitudes(orders, z):
    """Return the two Bessel amplitudes with the common integer-order factor."""
    jv = _scipy_special().jv
    x = orders * z
    lower = jv(orders - 1, x)
    upper = jv(orders + 1, x)
    h = orders * (0.5 * (lower + upper))
    t = orders * (0.5 * (lower - upper))
    amplitudes = np.stack((h, t), axis=-1)
    if np.any(~np.isfinite(amplitudes)):
        raise ArithmeticError(
            "nonfinite Bessel amplitudes for the harmonic shape basis"
        )
    return amplitudes


@dataclass(frozen=True)
class ShapeTable:
    spacing: tuple
    edges: tuple
    orders: np.ndarray
    selections: tuple
    responses: tuple
    supports: tuple
    nodes: np.ndarray
    barycentric: np.ndarray
    amplitudes: np.ndarray

    @classmethod
    def build(cls, channels, spacing, edges, order, max_orders):
        spacing = _ordered_bounds(spacing, size=2)
        edges = _ordered_bounds(edges)
        order = _positive_int(order, "order")
        max_orders = _positive_int(max_orders, "max_orders")
        if spacing[0] <= 0:
            raise ValueError("spacing bounds must be positive in Hz")
        if edges[0] < 0 or edges[-1] > 1:
            raise ValueError("z_edges must lie in [0, 1]")
        response = _prepare_host_responses(channels, derivatives=False)
        if not response.supports or any(
            not (0 < lo < hi < np.inf) for lo, hi in response.supports
        ):
            raise ValueError("harmonic channels need finite positive-width passbands")
        orders, selections = _channel_orders(response.supports, spacing, max_orders)
        index = np.arange(order + 1)
        unit = (1 - np.cos(np.pi * index / order)) / 2
        nodes = edges[:-1, None] + np.diff(edges)[:, None] * unit
        nodes[:, -1] = edges[1:]
        if np.any(np.diff(nodes, axis=1) <= 0):
            raise ValueError("z_edges and order must give distinct float64 nodes")
        barycentric = (-1.0) ** index
        barycentric[[0, -1]] *= 0.5
        return cls(
            tuple(spacing),
            tuple(edges),
            orders,
            selections,
            response.functions,
            response.supports,
            nodes,
            barycentric,
            _amplitudes(orders, nodes[..., None]),
        )

    def interpolation(self, z):
        """Return one cell and its cardinal weights at the argument ratio."""
        if not self.edges[0] <= z <= self.edges[-1]:
            raise RegimeError(f"argument ratio {z:.17g} is outside z_edges")
        cell = min(
            np.searchsorted(self.edges, z, side="right") - 1, len(self.nodes) - 1
        )
        distance = z - self.nodes[cell]
        nearest = np.argmin(np.abs(distance))
        if distance[nearest] == 0:
            weights = np.zeros_like(self.barycentric)
            weights[nearest] = 1
        else:
            weights = self.barycentric * (distance[nearest] / distance)
            weights /= np.sum(weights)
        return cell, weights


def _channel_stokes(weight, polarised, i, q, v):
    linear = np.sum(polarised * q)
    return np.array([np.sum(weight * i), linear.real, linear.imag, np.sum(weight * v)])


def _line_stokes(amplitudes, s):
    h, t = amplitudes.T
    parallel = s * h
    par, perp = parallel**2, t**2
    return par + perp, par - perp, 2 * parallel * t


def _weights(response, frequency, mass, spin, transmission):
    band = np.asarray(response(frequency)).reshape(frequency.shape)
    polarised = spin * band
    active = polarised != 0
    if transmission is not None and np.any(active):
        polarised[active] *= transmission(frequency[active])
    return mass * band, polarised


def shape_coordinates(gamma, B, mu, eta):
    """Return the three shape coordinates and the cgs line-power scale."""
    if not (
        np.isfinite(gamma)
        and gamma >= 1
        and np.isfinite(B)
        and B >= 0
        and abs(mu) <= 1
        and abs(eta) <= 1
    ):
        raise ValueError("require finite gamma >= 1, B >= 0 and |mu|, |eta| <= 1")
    if B == 0 or gamma == 1 or abs(mu) == 1:
        return None

    _, perpendicular, doppler, z, delta = _geometry(gamma, mu, eta)
    gyro = E_ESU * B / (2 * np.pi * gamma * M_E * C_CGS)
    spacing = gyro / doppler
    power = (_POWER**0.5 * spacing * perpendicular / doppler**0.5) ** 2
    if not (np.isfinite(spacing) and spacing > 0 and np.isfinite(power)):
        raise ArithmeticError(
            "harmonic spacing or line-power scale is not representable"
        )
    s = eta if abs(eta) == 1 else np.clip(delta / doppler, -1.0, 1.0)
    return (spacing, z, s), power


def point_stokes(
    table, coordinates, weight, spin, transmission, *, check, physical=None
):
    """Contract one reduced atom and optional direct references."""
    values = np.zeros((1 + check + (physical is not None), len(table.responses), 4))
    if weight == 0 and physical is None:
        return values
    spacing, z, s = coordinates
    if not table.spacing[0] <= spacing <= table.spacing[1]:
        raise RegimeError(f"harmonic spacing {spacing:.17g} Hz is outside spacing")
    axis = z == 0 and abs(s) == 1
    if axis:
        spin = 0j
    else:
        cell, cardinal = table.interpolation(z)
    for j, (use, response, support) in enumerate(
        zip(table.selections, table.responses, table.supports, strict=True)
    ):
        if axis:
            if not support[0] <= spacing <= support[1]:
                continue
            orders = np.ones(1, dtype=int)
            amplitudes = np.full((1, 2), 0.5)
        else:
            lo, hi = _active_range(support, spacing)
            # The table's orders are contiguous within each channel selection.
            start = use.start + lo - table.orders[use.start]
            take = slice(start, start + hi - lo + 1)
            orders = table.orders[take]
            amplitudes = np.einsum(
                "k,kms->ms", cardinal, table.amplitudes[cell, :, take]
            )
        frequency = orders * spacing
        weighted, polarised = _weights(response, frequency, weight, spin, transmission)
        values[0, j] = _channel_stokes(
            weighted, polarised, *_line_stokes(amplitudes, s)
        )
        if check:
            direct = amplitudes if axis else _amplitudes(orders, z)
            values[1, j] = _channel_stokes(
                weighted, polarised, *_line_stokes(direct, s)
            )
        if physical is not None:
            geometry, original_mass, original_spin = physical
            i, q, v, physical_nu = high_order_line_powers(orders, *geometry)
            physical_weight, physical_spin = _weights(
                response,
                physical_nu,
                original_mass,
                0j if axis else original_spin,
                transmission,
            )
            values[2, j] = _channel_stokes(physical_weight, physical_spin, i, q, v)
    if np.any(~np.isfinite(values)):
        raise ArithmeticError("nonfinite harmonic shape spectrum")
    return values
