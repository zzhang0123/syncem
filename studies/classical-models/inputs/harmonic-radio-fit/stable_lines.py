"""Local manuscript-only stabilized integer-harmonic line values.

The fast large-order branch uses the cubic-contour approximation already in
syncmoments.model._verified_airy, retaining the finite-gamma geometry and
harmonic frequencies. It is a line approximation, not a continuum radiation
kernel. Its retained integer-line error can be checked with the package's
integer_bessel_airy_enclosure; no uniform propagated error certificate is
claimed by this floating-point vectorized helper. Continuous orders are used
only by local channel quadrature and are not certified by integer enclosures.

Large-order lines beyond Airy argument 10 are omitted; their I, |Q| and |V|
have an analytical DLMF envelope returned by omitted_tail_envelope. Numerical
integration of that envelope is still a population quadrature estimate.
Sources: https://dlmf.nist.gov/10.9.E2 ; https://dlmf.nist.gov/9.5.E1 ;
https://dlmf.nist.gov/10.14.E5 ; https://dlmf.nist.gov/10.14.E6 .
"""

from __future__ import annotations

import math
import numpy as np
from scipy.special import airy
from syncmoments.constants import C_CGS, E_ESU, M_E
from syncmoments.model.high_order_harmonic import (
    _geometry,
    high_order_line_powers as _scipy_line_powers,
)

AIRY_ORDER_MIN = 1e6
AIRY_ARGUMENT_MAX = 10.0


def _setup(m, gamma, B, mu, eta):
    orders = np.asarray(m, dtype=float)
    if np.any(~np.isfinite(orders)) or np.any(orders < 1) or np.any(orders > 2**53 - 2):
        raise ValueError("orders must be finite in [1, 2**53-2]")
    _, perpendicular, D, z, delta = _geometry(gamma, mu, eta)
    defect = (delta / D) ** 2 + (
        (1 / gamma) * math.sqrt((1 - eta) * (1 + eta)) / D
    ) ** 2
    # Avoid losing the small turning-point coordinate through 1 - z.
    one_minus_z = defect / (1 + z) if z >= 0.5 else 1 - z
    spacing = E_ESU * B / (2 * math.pi * gamma * M_E * C_CGS * D)
    if z > 0:
        scale = np.cbrt(2 / (orders * z))
        argument = orders * one_minus_z * scale
    else:
        scale = np.zeros_like(orders)
        argument = np.full_like(orders, np.inf)
    prefactor = (
        E_ESU**2
        * (E_ESU * B / (gamma * M_E * C_CGS)) ** 2
        / (2 * math.pi * C_CGS * D**3)
    )
    return (
        orders,
        perpendicular,
        D,
        z,
        delta,
        defect,
        spacing,
        scale,
        argument,
        prefactor,
    )


def stable_high_order_line_powers(m, gamma, B, mu, eta):
    """Return (I,Q,V,nu); scalar geometry, arbitrary real-order array shape.

    Large orders with 0.9<=z<=1 and Airy argument<=10 use cubic Airy line
    values. Other large-order lines are omitted with omitted_tail_envelope.
    Small orders retain the package SciPy evaluator. No certificate is
    attached to the value-only tuple.
    """
    setup = _setup(m, gamma, B, mu, eta)
    orders, perpendicular, D, z, delta, _, spacing, scale, argument, prefactor = setup
    intensity, stokes_q, stokes_v = (np.zeros_like(orders) for _ in range(3))
    small = orders < AIRY_ORDER_MIN
    if np.any(small):
        intensity[small], stokes_q[small], stokes_v[small], _ = _scipy_line_powers(
            orders[small], gamma, B, mu, eta
        )
    selected = (~small) & (argument <= AIRY_ARGUMENT_MAX) & (z >= 0.9)
    if np.any(selected):
        ai, aip, _, _ = airy(argument[selected])
        J = scale[selected] * ai
        Jp = -scale[selected] ** 2 * aip
        parallel = delta * perpendicular * J / (D * z)
        transverse = perpendicular * Jp
        pref = prefactor * orders[selected] ** 2
        intensity[selected] = pref * (parallel**2 + transverse**2)
        stokes_q[selected] = pref * (parallel**2 - transverse**2)
        stokes_v[selected] = 2 * pref * parallel * transverse
    return intensity, stokes_q, stokes_v, orders * spacing


def omitted_tail_envelope(m, gamma, B, mu, eta):
    """DLMF envelope on omitted-line I, |Q| and |V| (float evaluation).

    For r=sqrt(1-z^2), xi=atanh(r)-r >= r^3/3. Using this lower bound in
    DLMF10.14.5--6 avoids subtracting nearby values in the exponential.
    This analytical inequality is not an outward-rounded certificate of this
    float implementation, and excludes the retained cubic-line remainder.
    """
    setup = _setup(m, gamma, B, mu, eta)
    orders, perpendicular, D, z, delta, defect, _, _, argument, prefactor = setup
    bound = np.zeros_like(orders)
    selected = (orders >= AIRY_ORDER_MIN) & ((argument > AIRY_ARGUMENT_MAX) | (z < 0.9))
    if z == 0 or perpendicular == 0 or B == 0 or not np.any(selected):
        return bound
    n = orders[selected]
    parallel_coefficient = delta * perpendicular / (D * z)
    derivative_coefficient_sq = math.sqrt(1 + z * z) / (z * z * 2 * math.pi)
    factor = prefactor * (
        parallel_coefficient**2 * n * n
        + perpendicular**2 * derivative_coefficient_sq * n
    )
    exponent = -2 * n * max(0, defect) ** 1.5 / 3
    bound[selected] = factor * np.exp(exponent)
    return bound


# Drop-in name for local forward scripts; package functions are not patched.
high_order_line_powers = stable_high_order_line_powers
