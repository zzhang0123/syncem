"""User density families (SPEC Sec. 3.5, G14).

LABEL [extension].

The declared variable supplies the unit of the input samples. A density
returns values of the same shape and its rule returns (n_node,) arrays of
nodes and weights. Normalization is by quadrature with no certified
enclosure.
"""

from __future__ import annotations

import inspect
from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp

from ..base._fingerprint import NotCacheable, digest
from ..base._function_fingerprint import function_fingerprint
from ..units import CANONICAL, convert, dimension_of
from ._common import (
    Law,
    concrete,
    count,
    interval,
    log_interval,
    node_count,
    pair_values,
    probabilities,
    require,
    resolved,
    scalar,
)
from .parameters import Free


class UserFamily(Law):
    """A normalised density from a function in the declared unit."""

    fn: Callable = eqx.field(static=True)
    variables: tuple[str, ...] = eqx.field(static=True)
    lo: float = eqx.field(static=True)
    hi: float = eqx.field(static=True)
    given_unit: str | None = eqx.field(static=True)
    cutoff_fn: Any = eqx.field(static=True)
    linear: tuple[str, ...] = eqx.field(static=True)
    nodes: int = eqx.field(static=True)
    params: dict

    def __init__(self, fn, variable, support, unit, cutoff, linear, nodes, params):
        self.fn = fn
        self.variables = (variable,)
        self.lo, self.hi = support
        self.given_unit = unit
        self.cutoff_fn = cutoff
        self.linear = linear
        self.nodes = count(nodes)
        self.params = params

    def _unit(self, x, reverse=False):
        unit = self.given_unit
        if unit is None:
            return x
        canonical = CANONICAL[dimension_of(unit)]
        return convert(x, unit, canonical) if reverse else convert(x, canonical, unit)

    @property
    def cutoff(self):
        if self.cutoff_fn is None:
            return None
        resolved(*self.params.values())
        if callable(self.cutoff_fn):
            signature = inspect.signature(self.cutoff_fn)
            params = {
                name: value
                for name, value in self.params.items()
                if name in signature.parameters
            }
            upper = self.cutoff_fn(**params)
        else:
            upper = self.cutoff_fn
        value, exponent = upper if isinstance(upper, tuple) else (upper, 0.0)
        value = self._unit(scalar(value, "family cutoff"), reverse=True)
        exponent = scalar(exponent, "family cutoff exponent")
        if concrete(exponent):
            require(exponent > -1, "family cutoff exponent must exceed -1")
        return value, exponent

    @property
    def fixed_nodes(self):
        return not callable(self.cutoff_fn)

    @property
    def support(self):
        cutoff = self.cutoff
        if cutoff is None:
            return self.lo, self.hi
        upper = jnp.minimum(self.hi, cutoff[0])
        require(upper > self.lo, "family cutoff must exceed the lower support")
        return self.lo, upper

    def _raw(self, x, params=None):
        params = self.params if params is None else params
        resolved(*params.values())
        y = jnp.asarray(self.fn(self._unit(x), **params), dtype=float)
        return jnp.broadcast_to(y, jnp.shape(x))

    def _panel(self, lo, hi, n):
        if self.lo > 0:
            return log_interval(lo, hi, n)
        return interval(lo, hi, n)

    def _weights(self, lo, hi, n):
        cutoff = self.cutoff
        if cutoff is None:
            x, w = self._panel(lo, hi, n)
        else:
            # Cancel an algebraic endpoint by a power map on the entire panel.
            # With a moving endpoint this keeps the weights and nodes traceable.
            upper, exponent = cutoff
            t, wt = interval(0.0, 1.0, n)
            # Take the derivative only through positive distances. At hi == upper,
            # the transformed lower edge is identically zero even when upper moves.
            distance = upper - hi
            left = jnp.where(
                distance > 0,
                jnp.where(distance > 0, distance, 1.0) ** (exponent + 1),
                0.0,
            )
            right = (upper - lo) ** (exponent + 1)
            z = left + (right - left) * t[::-1]
            d = jnp.maximum(z, jnp.finfo(float).tiny) ** (1 / (exponent + 1))
            x = upper - d
            w = wt[::-1] * (right - left) / ((exponent + 1) * d**exponent)
        y = self._raw(x)
        require(
            jnp.all(jnp.isfinite(y) & (y >= 0)),
            "family density must be finite and nonnegative on its nodes",
        )
        return x, w * y

    def rule(self, n=None, *, parents=None):
        return probabilities(*self._weights(*self.support, node_count(n, self.nodes)))

    def _mass(self):
        _, w = self._weights(*self.support, self.nodes)
        mass = jnp.sum(w)
        require(
            jnp.isfinite(mass) & (mass > 0),
            "family density must have positive finite mass",
        )
        return mass

    def _raw_panel(self, lo, hi, n=None):
        support_lo, support_hi = self.support
        lo = jnp.maximum(lo, support_lo)
        hi = jnp.minimum(hi, support_hi)
        valid = hi > lo
        a = jnp.where(valid, lo, support_lo)
        b = jnp.where(valid, hi, support_hi)
        x, w = self._weights(a, b, node_count(n, self.nodes))
        return x, jnp.where(valid, w, 0.0)

    def panel_rule(self, lo, hi, n=None, *, fitted=False):
        """Quadrature on a cell with the normalisation of the full law."""
        x, w = self._raw_panel(lo, hi, n)
        return x, w / self._mass()

    def density(self, x, *, parents=None):
        lo, hi = self.support
        norm = self._mass()
        safe = (lo + hi) / 2
        x = jnp.asarray(x)
        inside = (x >= lo) & (x < hi)
        return jnp.where(inside, self._raw(jnp.where(inside, x, safe)) / norm, 0.0)

    def linear_density(self, x):
        """Raw coefficient effects for the parameters named in linear=."""
        if not self.linear:
            raise ValueError("the family declares no linear parameters")
        zero = {**self.params, **dict.fromkeys(self.linear, 0.0)}
        zero_family = eqx.tree_at(lambda law: law.params, self, zero)
        lo, hi = zero_family.support
        x = jnp.asarray(x)
        inside = (x >= lo) & (x < hi)
        safe = jnp.where(inside, x, (lo + hi) / 2)
        base = self._raw(safe, zero)
        effects = [self._raw(safe, {**zero, name: 1.0}) - base for name in self.linear]
        return jnp.where(inside[..., None], jnp.stack(effects, axis=-1), 0.0)

    def record(self):
        # Functions get content fingerprints instead of textual reprs.
        data = {
            "family": getattr(self.fn, "__name__", "user family"),
            "variables": list(self.variables),
            "given_unit": self.given_unit,
            "support": [self.lo, self.hi],
            "parameters": {
                name: value.record() if isinstance(value, Free) else float(value)
                for name, value in self.params.items()
            },
            "linear": list(self.linear),
            "nodes": self.nodes,
        }
        try:
            cut = (
                function_fingerprint(self.cutoff_fn)
                if callable(self.cutoff_fn)
                else self.cutoff_fn
            )
            fingerprint = digest((function_fingerprint(self.fn), cut, data))
        except NotCacheable:
            fingerprint = None
        return {**data, "fingerprint": fingerprint}


def family(*, variable, support, unit=None, cutoff=None, linear=()):
    """Decorate an unnormalised scalar density fn(x, **params).

    The density function takes coordinates in unit. An optional cutoff is
    an upper endpoint or an endpoint and algebraic exponent in that unit.
    It can also be a callable with named family parameters.
    """
    lo, hi, given_unit = pair_values(*support, unit=unit)
    require(hi > lo, "family support must consist of two increasing finite edges")
    canonical_support = (float(lo), float(hi))
    linear = tuple(linear)

    def decorate(fn):
        signature = inspect.signature(fn)
        parameters = tuple(signature.parameters.values())
        if not parameters or parameters[0].kind not in (
            inspect.Parameter.POSITIONAL_ONLY,
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
        ):
            raise TypeError("family density needs a positional coordinate")
        keys = {parameter.name for parameter in parameters[1:]}
        if len(set(linear)) != len(linear) or set(linear) - keys:
            raise ValueError("linear= must name distinct parameters of the family")
        if "nodes" in keys:
            raise ValueError("family reserves the name nodes for its quadrature")
        if any(
            p.kind not in (p.POSITIONAL_OR_KEYWORD, p.KEYWORD_ONLY)
            for p in parameters[1:]
        ):
            raise TypeError("family density needs named scalar parameters")

        def declare(*, nodes=128, **params):
            for name in linear:
                if (
                    name not in params
                    and signature.parameters[name].default is inspect.Parameter.empty
                ):
                    params[name] = Free()
            bound = signature.bind(0.0, **params)
            bound.apply_defaults()
            values = {name: scalar(bound.arguments[name], name) for name in keys}
            return UserFamily(
                fn,
                variable,
                canonical_support,
                given_unit,
                cutoff,
                linear,
                nodes,
                values,
            )

        return declare

    return decorate
