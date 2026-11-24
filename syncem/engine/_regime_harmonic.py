"""Eager I and P for the finite-channel harmonic branch.

The integer sum uses line powers at the actual line frequencies. A dense
smooth bump response uses frequency quadrature of the real-order summand.
The dense sum approximation and numerical line powers have unbounded error.
The angular and frequency counts can be refined independently.
"""

from dataclasses import dataclass, replace
from numbers import Integral

import numpy as np
from numpy.polynomial.legendre import legvander

from ..constants import C_CGS, E_ESU, M_E
from ..model.high_order_harmonic import (
    _MAX_EXACT_FLOAT_INT,
    _active_range,
    _prepare_host_responses,
    _scipy_special,
    high_order_line_powers,
)
from .cache import default_cache

_DIRECT_LIMIT = 4096
_DEFAULT_FREQUENCY_NODES = 64
_BESSEL_UNDERFLOW_EXPONENT = 800
_LINE_BLOCK = 2**17
_GYRO_HZ_PER_G = E_ESU / (2 * np.pi * M_E * C_CGS)


def _rule(nodes):
    cache = default_cache()
    key = f"regime-legendre:{nodes}"
    rule = cache.get("memo", key)
    if rule is None:
        rule = cache.put("memo", key, _scipy_special().roots_legendre(nodes))
    return rule


def _check_nodes(nodes, name):
    if (
        isinstance(nodes, bool)
        or not isinstance(nodes, Integral)
        or not 1 <= nodes <= 8192
    ):
        raise ValueError(f"{name} must be an integer in [1, 8192]")


def _phase_values(phase, nu):
    if phase is None:
        return np.ones_like(nu)
    result = np.broadcast_to(np.asarray(phase(nu)), nu.shape)
    if not np.all(np.isfinite(result)):
        raise ArithmeticError("harmonic phase weights are not finite")
    return result


@dataclass(frozen=True)
class _PreparedChannels:
    source: object
    phase: object
    frequency_nodes: int
    response: object
    support: np.ndarray
    frequency: np.ndarray
    weight: np.ndarray
    phased_weight: np.ndarray
    phase_at_pitch: object = None


def prepare_harmonic_channels(channels, *, phase=None, frequency_nodes=None):
    """Prepare the responses and dense frequency weights once per build."""
    count = _DEFAULT_FREQUENCY_NODES if frequency_nodes is None else frequency_nodes
    _check_nodes(count, "frequency_nodes")
    support = np.asarray(channels.support)
    if (
        not np.all(np.isfinite(support))
        or np.any(support[:, 0] <= 0)
        or np.any(support[:, 1] <= support[:, 0])
    ):
        raise ValueError(
            "harmonic channels need finite positive support with nonzero width"
        )
    response = _prepare_host_responses(channels, derivatives=False)
    if channels.family == "bump":
        nodes, weights = _rule(count)
        half = (support[:, 1] - support[:, 0]) / 2
        nu = support[:, :1] + half[:, None] * (1 + nodes)
        weight = np.stack(
            [
                radius * weights * function(frequencies)
                for radius, function, frequencies in zip(
                    half, response.functions, nu, strict=True
                )
            ]
        )
        phased = weight * _phase_values(phase, nu)
    else:
        nu = weight = phased = np.empty((channels.n_ch, 0))
    return _PreparedChannels(
        channels, phase, count, response, support, nu, weight, phased
    )


def _geometry(gamma, mu, eta):
    """Vectorised geometry with the finite-gamma defect near the ridge."""
    inv = 1 / gamma
    beta = np.sqrt(1 - inv**2)
    one_minus_beta = inv**2 / (1 + beta)
    sin_eta = np.sqrt((1 - eta) * (1 + eta))
    cos_defect = np.where(
        (mu >= 0) & (eta >= 0),
        (1 - mu) + mu * (1 - eta),
        np.where((mu <= 0) & (eta <= 0), (1 + mu) - mu * (1 + eta), 1 - mu * eta),
    )
    doppler = one_minus_beta + beta * cos_defect
    if np.any(doppler <= 0) or np.any(~np.isfinite(doppler)):
        raise ValueError("harmonic Doppler denominator is not representable")
    b_perp = beta * np.sqrt((1 - mu) * (1 + mu))
    delta = (eta - mu) + mu * one_minus_beta
    direct_z = b_perp * sin_eta / doppler
    defect = (delta / doppler) ** 2 + (inv * sin_eta / doppler) ** 2
    z = np.where(direct_z < 0.5, direct_z, np.sqrt(np.maximum(0, 1 - defect)))
    return b_perp, doppler, delta, z


def _axis_rule(gamma, B, eta, support, breaks, frequencies, count):
    """Include the two pitch cosines where the fundamental crosses a band edge."""
    gyro = _GYRO_HZ_PER_G * (B / gamma)
    beta = np.sqrt(1 - gamma**-2)
    band_edges = np.clip((1 - gyro / frequencies) / (eta * beta), *support)
    fixed_edges = np.broadcast_to(
        np.asarray([support[0], *breaks, support[1]]),
        (len(frequencies), 2 + len(breaks)),
    )
    edges = np.sort(np.concatenate((fixed_edges, band_edges), axis=1), axis=1)
    nodes, weights = _rule(count)
    half = (edges[:, 1:] - edges[:, :-1]) / 2
    points = edges[:, :-1, None] + half[..., None] * (1 + nodes)
    return points.reshape(len(frequencies), -1), (half[..., None] * weights).reshape(
        len(frequencies), -1
    )


def _pitch_rule(gamma, B, eta, support, breaks, floors, count):
    """Resolve the ridge on the full interval with panels in asinh coordinates."""
    beta = np.sqrt(1 - gamma**-2)
    centre = beta * eta
    sin_eta = np.sqrt((1 - eta) * (1 + eta))
    # D on the ridge. Use a positive form near the observing axis.
    doppler = sin_eta**2 + (eta / gamma) ** 2
    spacing = _GYRO_HZ_PER_G * (B / gamma) / doppler
    first_order = np.maximum(1, floors / spacing)
    critical = 1.5 * _GYRO_HZ_PER_G * B * gamma**2 * sin_eta
    x = floors / max(critical, np.finfo(float).tiny)
    h = np.maximum(sin_eta / gamma, doppler * first_order ** (-1 / 3)) / np.sqrt(1 + x)
    if not np.all(np.isfinite(h) & (h > 0)):
        raise ValueError("harmonic pitch ridge scale is not representable")
    boundaries = np.arcsinh(
        (np.asarray([support[0], *breaks, support[1]]) - centre) / h[:, None]
    )
    lo, hi = boundaries[:, :1], boundaries[:, -1:]
    panels = np.sort(
        np.concatenate([boundaries, np.clip(np.array([-3, 0, 3]), lo, hi)], axis=1),
        axis=1,
    )
    nodes, weights = _rule(count)
    half = (panels[:, 1:] - panels[:, :-1]) / 2
    t = panels[:, :-1, None] + half[..., None] * (1 + nodes)
    pitch = np.clip(centre + h[:, None, None] * np.sinh(t), *support)
    weight = (half[..., None] * weights) * (h[:, None, None] * np.cosh(t))
    return pitch.reshape(len(floors), -1), weight.reshape(len(floors), -1)


def _integer_sum(prepared, j, gamma, B, mu, eta, spacing):
    if abs(eta) == 1:
        orders = np.ones(1)
    else:
        lo, hi = _active_range(prepared.support[j], spacing)
        orders = np.arange(lo, hi + 1, dtype=float)
    i, q, _, frequency = high_order_line_powers(orders, gamma, B, mu, eta)
    response = np.asarray(prepared.response.functions[j](frequency)).reshape(-1)
    active = response != 0
    intensity = np.sum(response * i)
    polarized = 0j
    if abs(eta) < 1 and np.any(active):
        polarized = np.sum(
            response[active]
            * q[active]
            * _phase_values(prepared.phase, frequency[active])
            * _conditional_weights(prepared, frequency[active], mu)
        )
    if not (np.isfinite(intensity) and np.isfinite(polarized)):
        raise ArithmeticError("nonfinite harmonic integer sum")
    return intensity, polarized


def _conditional_weights(prepared, frequency, pitch):
    if prepared.phase_at_pitch is None:
        return 1.0
    return _phase_values(lambda nu: prepared.phase_at_pitch(nu, pitch), frequency)


def _dense_sum(
    prepared, channel, gamma, B, spacing, b_perp, doppler, delta, z, *, pitch=None
):
    """Integrate the real-order summand with d m = d nu / spacing."""
    m = prepared.frequency[channel] / spacing[:, None]
    x = m * z[:, None]
    order = m - 1
    ratio = np.divide(x, order, out=np.ones_like(x), where=order >= 1)
    gap = np.maximum(0, 1 - ratio) * (1 + ratio)
    # For x < order this is a lower bound on the exponential decay of both
    # Bessel terms. See 0.9 decision 11. The discarded terms round to zero.
    evaluate = order * gap**1.5 <= 3 * _BESSEL_UNDERFLOW_EXPONENT
    jv = _scipy_special().jv
    lower = np.zeros_like(x)
    upper = np.zeros_like(x)
    kept_order = m[evaluate]
    kept_x = x[evaluate]
    lower[evaluate] = jv(kept_order - 1, kept_x)
    upper[evaluate] = jv(kept_order + 1, kept_x)
    a_par = (delta * b_perp / (2 * doppler))[:, None] * (lower + upper)
    a_perp = (0.5 * b_perp)[:, None] * (lower - upper)
    omega = 2 * np.pi * _GYRO_HZ_PER_G * B / gamma
    pref = E_ESU**2 * omega**2 / (2 * np.pi * C_CGS)
    weighted_pref = (pref / (doppler**3 * spacing))[:, None] * m**2
    parallel = weighted_pref * a_par**2
    transverse = weighted_pref * a_perp**2
    phased_weight = prepared.phased_weight[channel]
    if pitch is not None and prepared.phase_at_pitch is not None:
        phased_weight = phased_weight * _conditional_weights(
            prepared, prepared.frequency[channel], pitch
        )
    return np.column_stack(
        [
            np.sum((parallel + transverse) * prepared.weight[channel], axis=1),
            np.sum((parallel - transverse) * phased_weight, axis=1),
        ]
    )


def _channel_values(prepared, gamma, B, mu, eta, used):
    values = np.zeros(mu.shape + (2,), dtype=complex)
    ch, point = np.nonzero(used & (np.abs(mu) < 1))
    if ch.size == 0:
        return values
    pitch = mu[ch, point]
    b_perp, doppler, delta, z = _geometry(gamma, pitch, eta)
    spacing = (_GYRO_HZ_PER_G * (B / gamma)) / doppler
    if np.any(spacing <= 0) or np.any(~np.isfinite(spacing)):
        raise ValueError("harmonic frequency spacing is not finite positive")
    if abs(eta) == 1:
        for ix in np.flatnonzero(
            (prepared.support[ch, 0] <= spacing) & (spacing <= prepared.support[ch, 1])
        ):
            values[ch[ix], point[ix]] = _integer_sum(
                prepared, ch[ix], gamma, B, pitch[ix], eta, spacing[ix]
            )
        return values
    low, high = (prepared.support[ch] / spacing[:, None]).T
    if np.any(~np.isfinite(high)) or np.any(np.ceil(high) + 2 > _MAX_EXACT_FLOAT_INT):
        raise ValueError("active harmonic order exceeds exact unit spacing of float64")
    count = np.ceil(high) - np.floor(low) - 1
    dense = np.flatnonzero(count > _DIRECT_LIMIT)
    if dense.size:
        if not all(prepared.response.smooth_edges[j] for j in np.unique(ch[dense])):
            raise ValueError(
                "dense harmonic channels require a bump with smooth support edges"
            )
        if np.any(high[dense] > 16 * low[dense]):
            raise ValueError(
                "dense harmonic channel spans more than a factor of 16 in order"
            )
    for ix in np.flatnonzero(count <= _DIRECT_LIMIT):
        values[ch[ix], point[ix]] = _integer_sum(
            prepared, ch[ix], gamma, B, pitch[ix], eta, spacing[ix]
        )
    batch = max(1, _LINE_BLOCK // prepared.frequency_nodes)
    for start in range(0, len(dense), batch):
        ix = dense[start : start + batch]
        values[ch[ix], point[ix]] = _dense_sum(
            prepared,
            ch[ix],
            gamma,
            B,
            spacing[ix],
            b_perp[ix],
            doppler[ix],
            delta[ix],
            z[ix],
            pitch=pitch[ix],
        )
    if not np.all(np.isfinite(values)):
        raise ArithmeticError("nonfinite harmonic channel result")
    return values


def harmonic_channel(
    channels,
    gamma,
    B,
    eta,
    pitch,
    *,
    phase=None,
    angular_nodes=8,
    frequency_nodes=None,
    degrees=None,
    pitch_support=(-1.0, 1.0),
    pitch_breaks=(),
    prepared=None,
    phase_at_pitch=None,
):
    """Return natural I and phase-rotated P at one physical gamma, B and eta.

    Pitch is a scalar cosine or a probability density evaluated inside its
    support. With degrees supplied, use pitch=None and return Legendre
    projections on the full domain, with leading axis indexed by degree.
    Use different angular and frequency counts to measure refinement.
    `phase_at_pitch` returns transmission at a scalar or at batched pitch cosines.
    Each frequency row in a batch belongs to the corresponding cosine.
    """
    if any(np.ndim(x) != 0 or np.iscomplexobj(x) for x in (gamma, B, eta)):
        raise ValueError("gamma, B and eta must be real scalars")
    gamma, B, eta = map(float, (gamma, B, eta))
    if not (
        np.isfinite(gamma)
        and gamma >= 1
        and np.isfinite(B)
        and B >= 0
        and abs(eta) <= 1
    ):
        raise ValueError(
            "harmonic points require finite gamma >= 1, B >= 0 and |eta| <= 1"
        )
    _check_nodes(angular_nodes, "angular_nodes")
    if degrees is not None:
        degrees = tuple(degrees)
        if (
            pitch is not None
            or tuple(pitch_support) != (-1, 1)
            or not degrees
            or any(
                isinstance(l, bool) or not isinstance(l, Integral) or l < 0
                for l in degrees
            )
        ):
            raise ValueError(
                "Legendre projection needs degrees >= 0, pitch=None and full support"
            )
        angular_nodes = max(angular_nodes, (max(degrees) + 2) // 2)
        _check_nodes(angular_nodes, "Legendre angular node count")
    if prepared is None:
        prepared = prepare_harmonic_channels(
            channels, phase=phase, frequency_nodes=frequency_nodes
        )
    elif (
        prepared.source is not channels
        or (phase is not None and phase is not prepared.phase)
        or (frequency_nodes is not None and frequency_nodes != prepared.frequency_nodes)
    ):
        raise ValueError(
            "prepared harmonic channels do not match the requested evaluation"
        )
    if phase_at_pitch is not None:
        prepared = replace(prepared, phase_at_pitch=phase_at_pitch)
    shape = (channels.n_ch, 2)
    if gamma == 1 or B == 0:
        return np.zeros(
            shape if degrees is None else (len(degrees), *shape), dtype=complex
        )
    if degrees is None and not callable(pitch):
        if np.ndim(pitch) != 0 or np.iscomplexobj(pitch):
            raise ValueError("pitch cosine must be a real scalar")
        mu = float(pitch)
        if not abs(mu) <= 1:
            raise ValueError("pitch cosine must lie in [-1, 1]")
        points = np.full((channels.n_ch, 1), mu)
        return _channel_values(
            prepared, gamma, B, points, eta, np.ones_like(points, dtype=bool)
        )[:, 0]
    lo, hi = map(float, pitch_support)
    if not -1 <= lo < hi <= 1:
        raise ValueError("pitch density needs a positive-width interval in [-1, 1]")
    breaks = tuple(sorted({x for x in pitch_breaks if lo < x < hi}))
    if abs(eta) == 1:
        mu, weight = _axis_rule(
            gamma, B, eta, (lo, hi), breaks, prepared.support, angular_nodes
        )
    else:
        mu, weight = _pitch_rule(
            gamma, B, eta, (lo, hi), breaks, prepared.support[:, 0], angular_nodes
        )
    if degrees is None:
        density = np.broadcast_to(np.asarray(pitch(mu.ravel())), (mu.size,)).reshape(
            mu.shape
        )
        if (
            np.iscomplexobj(density)
            or np.any(~np.isfinite(density))
            or np.any(density < 0)
        ):
            raise ValueError("pitch density values must be finite and nonnegative")
        weight = weight * density
    values = _channel_values(prepared, gamma, B, mu, eta, weight != 0)
    if degrees is None:
        return np.einsum("cn,cns->cs", weight, values)
    degree_array = np.asarray(degrees)
    polys = legvander(mu, int(degree_array.max()))[:, :, degree_array]
    return (degree_array + 0.5)[:, None, None] * np.einsum(
        "cn,cnl,cns->lcs", weight, polys, values
    )
