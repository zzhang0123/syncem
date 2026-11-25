"""Transverse field and critical frequency at continuous quadrature nodes."""

import numpy as np

from ..base.window import A1_HZ_PER_G
from ..population._node_offsets import sum_with_residual as _sum
from ..population._panel_cuts import PanelCuts


def transverse(field, eta):
    return field * np.sqrt(np.maximum(0, (1 - eta) * (1 + eta)))


def _split(fraction):
    # Dekker's split of a binary64 mantissa into 26 and 27 significant bits.
    scaled = (2**27 + 1) * fraction
    high = scaled - (scaled - fraction)
    return high, fraction - high


def _product(left, right):
    """Multiply two values with residuals, retaining the product rounding."""
    a, da = left
    b, db = right
    # The split operates on mantissas so its scaling cannot overflow.
    fa, ea = np.frexp(a)
    fb, eb = np.frexp(b)
    ah, al = _split(fa)
    bh, bl = _split(fb)
    p = fa * fb
    error = ((ah * bh - p) + ah * bl + al * bh) + al * bl
    return _sum(a * b, np.ldexp(error, ea + eb) + (a * db + da * b) + da * db)


def _sqrt(pair):
    square, ds = pair
    sine = np.sqrt(np.maximum(square, 0.0))
    rounded, residual = _product((sine, 0.0), (sine, 0.0))
    error = np.divide(
        (square - rounded) + (ds - residual),
        2 * sine,
        where=sine > 0,
        out=np.zeros_like(sine),
    )
    return _sum(sine, error)


def _sine(eta, offset):
    # Form each distance to a pole before applying the node displacement.
    lower, dl = _sum(1.0, -eta)
    upper, du = _sum(1.0, eta)
    return _sqrt(_product(_sum(lower, dl - offset), _sum(upper, du + offset)))


def _ratio(numerator, denominator):
    leading = numerator / denominator[0]
    rounded, error = _product((leading, 0.0), denominator)
    return _sum(leading, ((numerator - rounded) - error) / denominator[0])


def inverse_frequency_faces(axis, driver, point, faces):
    """Invert the same physical map used for ownership, retaining fractional cuts."""
    constant = (A1_HZ_PER_G, 0.0)

    def coordinate(name):
        return point[name], point.get((name,), 0.0)

    if axis == "gamma":
        scale = _product(
            constant, _product(coordinate("B"), _sine(*coordinate(driver)))
        )
        if scale[0] <= 0:
            return ()
        return (PanelCuts(*_sqrt(_ratio(faces, scale))),)
    energy = coordinate("gamma")
    scale = _product(constant, _product(energy, energy))
    if axis == "B":
        scale = _product(scale, _sine(*coordinate(driver)))
        return (PanelCuts(*_ratio(faces, scale)),) if scale[0] > 0 else ()
    scale = _product(scale, coordinate("B"))
    if scale[0] <= 0:
        return ()
    inside = (faces < scale[0]) | ((faces == scale[0]) & (scale[1] >= 0))
    magnitude, shift = _sine(*_ratio(faces[inside & (faces >= 0)], scale))
    return (PanelCuts(np.r_[-magnitude, magnitude], np.r_[-shift, shift]),)


def projected_frequency(point, shifts):
    """Push the physical node and its integration displacement through νc."""
    field, eta, gamma = (point[name] for name in ("B", "eta", "gamma"))
    if not any(name in shifts for name in ("B", "eta", "gamma")):
        # Atomic coordinates use the public point-kernel multiplication order.
        sine = transverse(1.0, eta)
        bp = field * sine
        return sine, bp, A1_HZ_PER_G * bp * gamma**2, None
    # A continuous node can lie on either side of its rounded representation.
    # Carry that displacement through the same map as its quadrature weight.
    sine = _sine(eta, shifts.get("eta", 0.0))
    bp = _product((field, shifts.get("B", 0.0)), sine)
    energy = (gamma, shifts.get("gamma", 0.0))
    frequency, residual = _product(
        _product((A1_HZ_PER_G, 0.0), bp), _product(energy, energy)
    )
    return sine[0], bp[0], frequency, residual
