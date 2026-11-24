"""Extend the paper's physical energy distributions for the two-regime runs."""

from __future__ import annotations

from math import isfinite

import equinox as eqx

from .. import paper
from ..distributions import Conditional
from ..distributions.electrons import ElectronFamily
from ..population import Population

_PRESETS = {
    "E1": "e1",
    "E1M": "e1m",
    "E2": "e2",
    "E3": "e3",
    "E4": "e4",
    "E4-JP": "e4",
    "E4-KP": "e4kp",
    "E5": "e5",
    "E6": "e6",
}


def _with_lower_limit(law: ElectronFamily, gamma_min: float) -> ElectronFamily:
    fields = {"lo": gamma_min}
    if law.column is not None:
        fields.update(column=None, amplitude=law._coefficient())
    return law._at(fields)


def _kp(mu, gamma_min) -> ElectronFamily:
    return _with_lower_limit(paper._kp(mu), gamma_min)


def _local_ageing(B, gamma_min) -> ElectronFamily:
    return _with_lower_limit(paper._local_field_ageing(B), gamma_min)


_CONDITIONAL = {"E4-KP": _kp, "E6": _local_ageing}


def population(name: str, *, gamma_min: float = 100.0) -> Population:
    """Return a study ensemble at the original density on its extended support."""
    if isinstance(gamma_min, bool):
        raise ValueError("gamma_min must be a Lorentz factor between 1 and 300")
    gamma_min = float(gamma_min)
    if not isfinite(gamma_min) or not 1 <= gamma_min <= paper._GAMMA_LO:
        raise ValueError("gamma_min must be a Lorentz factor between 1 and 300")
    family = name.upper()
    if family not in _PRESETS:
        raise ValueError(f"unknown paper ensemble {name!r}")
    original = getattr(paper, _PRESETS[family])()
    if gamma_min == paper._GAMMA_LO:
        return original
    if family in _CONDITIONAL:
        base = original.factors["gamma"]
        gamma = Conditional(
            _CONDITIONAL[family],
            params={**base.params, "gamma_min": gamma_min},
            units=dict(base.units),
            parity=dict(base.parity),
            name=base.name,
        )
        return original.replace(gamma=gamma)
    if original.components:
        extended = tuple(
            part.replace(gamma=_with_lower_limit(part.factors["gamma"], gamma_min))
            for part in original.components
        )
        return eqx.tree_at(lambda p: p.components, original, extended)
    return original.replace(
        gamma=_with_lower_limit(original.factors["gamma"], gamma_min)
    )
