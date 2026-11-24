"""Scoped rules for the 0.8 Taylor jet (SPEC decision 172).

JAX 0.10 has no jet rules for scan and conditional branches. Keep
branch selection and runtime checks on the primal and propagate the
series through the branch selected there. JAX supplies the rules for
ordinary arithmetic.
"""

from __future__ import annotations

import threading
from contextlib import contextmanager
from contextvars import ContextVar
from functools import partial

import jax.numpy as jnp
from equinox._unvmap import unvmap_all_p, unvmap_any_p, unvmap_max_p
from jax import lax
from jax._src import core, pjit
from jax.experimental import jet

_LOCK = threading.RLock()
_ACTIVE = ContextVar("syncem_taylor_rules", default=False)


def _zero(primitive, primals, series, **params):
    return primitive.bind(*primals, **params), jet.zero_series


def _minimum(primals, series):
    x, y = jnp.broadcast_arrays(*primals)
    less, equal = x < y, x == y

    def select(dx, dy):
        dx, dy, _ = jnp.broadcast_arrays(dx, dy, x)
        return lax.select(equal, (dx + dy) / 2, lax.select(less, dx, dy))

    return lax.min(x, y), [select(*terms) for terms in zip(*series)]


def _body(jaxpr, primals, series):
    """Trace the body with Taylor terms only for floating and complex operands."""
    evaluate = core.jaxpr_as_fun(jaxpr)
    varying = tuple(
        i for i, p in enumerate(primals) if jnp.issubdtype(p.dtype, jnp.inexact)
    )
    if not varying:
        values = evaluate(*primals)
        return values, [tuple(jnp.zeros_like(v) for _ in series[0]) for v in values]

    def call(*args):
        actual = list(primals)
        for index, arg in zip(varying, args):
            actual[index] = arg
        return evaluate(*actual)

    values, terms = jet.jet2(
        call,
        tuple(primals[i] for i in varying),
        tuple(series[i] for i in varying),
    )
    return list(values), [tuple(term) for term in terms]


def _scan(
    primals, series, *, jaxpr, num_consts, num_carry, length, reverse, unroll, **_
):
    end = num_consts + num_carry
    const, carry, xs = primals[:num_consts], primals[num_consts:end], primals[end:]
    const_s, carry_s, xs_s = (
        series[:num_consts],
        series[num_consts:end],
        series[end:],
    )

    def step(state, x):
        c, c_s = state
        v, v_s = x
        ys, ys_s = _body(jaxpr, (*const, *c, *v), (*const_s, *c_s, *v_s))
        return (ys[:num_carry], ys_s[:num_carry]), (ys[num_carry:], ys_s[num_carry:])

    (out, out_s), (ys, ys_s) = lax.scan(
        step,
        (list(carry), list(map(tuple, carry_s))),
        (list(xs), list(map(tuple, xs_s))),
        length=length,
        reverse=reverse,
        unroll=unroll,
    )
    return out + ys, out_s + ys_s


def _branch(branch, inputs):
    return _body(branch, *inputs)


def _cond(primals, series, *, branches, **_):
    return lax.switch(
        primals[0],
        tuple(partial(_branch, branch) for branch in branches),
        (primals[1:], series[1:]),
    )


def _guard_terms(jaxpr, primals, series, params):
    """The Equinox error guard has the identity JVP and a checked primal."""
    graph = jaxpr.jaxpr
    checked = next(
        (e for e in graph.eqns if e.primitive.name == "custom_jvp_call"), None
    )
    if checked is None:
        terms = [
            (
                jet.zero_series
                if isinstance(v, core.Literal)
                else series[graph.invars.index(v)]
            )
            for v in graph.outvars
        ]
        return pjit.jit_p.bind(*primals, **params), terms
    if checked.outvars != graph.outvars[1:]:
        raise TypeError("unsupported Equinox error guard in Taylor mode")
    positions = [graph.invars.index(v) for v in checked.invars[: len(checked.outvars)]]
    values = pjit.jit_p.bind(*primals, **params)
    # Keep the checked primal live even if the request has no order-zero row.
    # In EQX_ON_ERROR=nan the failed check propagates its NaN into every term.
    terms = [jet.zero_series]
    for position, value in zip(positions, values[1:]):
        if jnp.issubdtype(value.dtype, jnp.inexact):
            terms.append(
                tuple(jnp.where(jnp.isnan(value), value, t) for t in series[position])
            )
        else:
            terms.append(jet.zero_series)
    return values, terms


def is_equinox_guard(params):
    return (
        params.get("name") == "branched_error_if_impl"
        and "equinox/_jit.py" in params["jaxpr"].jaxpr.debug_info.func_src_info
    )


def _jit(base_rule, primals, series, **params):
    jaxpr = params["jaxpr"]
    if is_equinox_guard(params):
        return _guard_terms(jaxpr, primals, series, params)
    # A separate ClosedJaxpr keeps JAX's cached nested transforms scoped
    # to the registry active for this call.
    return base_rule(
        primals,
        series,
        **{**params, "jaxpr": core.ClosedJaxpr(jaxpr.jaxpr, jaxpr.consts)},
    )


def _local_rule(primitive, rule, fallback):
    def apply(*args, **kwargs):
        if _ACTIVE.get():
            return rule(*args, **kwargs)
        if fallback is None:
            raise KeyError(primitive)
        return fallback(*args, **kwargs)

    return apply


@contextmanager
def taylor_rules():
    """Extend the registry for this tracing context and restore it on exit."""
    if _ACTIVE.get():
        yield
        return
    rules = {
        lax.min_p: _minimum,
        lax.scan_p: _scan,
        lax.cond_p: _cond,
        **{
            primitive: partial(_zero, primitive)
            for primitive in (
                lax.reduce_and_p,
                lax.reduce_or_p,
                unvmap_all_p,
                unvmap_any_p,
                unvmap_max_p,
            )
        },
    }
    with _LOCK:
        rules[pjit.jit_p] = partial(_jit, jet.jet_rules[pjit.jit_p])
        previous = {p: jet.jet_rules.get(p) for p in rules}
        for primitive, rule in rules.items():
            jet.jet_rules[primitive] = _local_rule(primitive, rule, previous[primitive])
        token = _ACTIVE.set(True)
        try:
            yield
        finally:
            _ACTIVE.reset(token)
            for primitive, rule in previous.items():
                if rule is None:
                    jet.jet_rules.pop(primitive)
                else:
                    jet.jet_rules[primitive] = rule
