"""Refusal checks and frozen-node weights of the stack walk (design/0.8/SPEC.md Sec. 8.3, decisions 139, 144, 145).

LABEL ``eq: illustrated derivative spectra``. ``_weights`` gives the nodes and the weight series of one averaged
level. A rule whose parents include the celled ``gamma`` keeps its nodes at the cell's reference point and reweights
them, ``w_q(u) = w_q(ref) rho(x_q | gamma(u)) / rho(x_q | ref)``. ``_check_frozen`` compares the reweighted weights
with a jet of the rule itself (decision 139), and ``_check_undeclared`` refuses a law that reads a celled variable it
does not declare (an order-1 probe per rule and cell). ``_check_lattice`` checks the celled set and the Legendre axes
against the averaged kernel. Each refusal raises ``StackRefused`` with the reason, and the caller takes the jet.
Units: those of the law's nodes, and dimensionless weights. Shapes: nodes ``(n_q,)``, weight series
``(n_q, order + 1)``. Not certified: the checks bound moments of the law, not the integrand, so a law whose nodes do
not resolve the kernel gives a quadrature error that the jet shares.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import jax.numpy as jnp
import numpy as np

from .average import AveragedKernel, NodeRule
from .provider_jet import taylor_coefficients

if TYPE_CHECKING:
    from ._stack_series import _Context

__all__ = ["StackRefused"]


class StackRefused(ValueError):
    """The stack provider does not apply to this product cell. The reason names the condition."""


def _check_lattice(averaged: AveragedKernel, lattice: Any) -> None:
    """The celled set the stack takes, and a lattice consistent with the averaged kernel."""
    names = lattice.cells.variables
    if "gamma" not in names or not set(names) <= {"gamma", "B"}:
        raise StackRefused(
            f"celled variables {names}: the stack takes gamma and optionally B. The jet takes the cell"
        )
    taken = {n for n, _ in averaged.averaged} | {n for n, _ in averaged.fixed}
    if set(names) & taken:
        raise ValueError(
            f"{sorted(set(names) & taken)} are celled and also averaged or fixed in the averaged kernel"
        )
    if lattice.index.legendre != tuple((n, L) for n, L, _ in averaged.legendre):
        raise ValueError(
            "the lattice's Legendre axes differ from the averaged kernel's"
        )


def _call(rule: NodeRule, name: str, point: dict) -> tuple[np.ndarray, np.ndarray]:
    """``(nodes, probs)`` of a rule at a concrete point. A read of a variable the point lacks refuses."""
    try:
        nodes, probs = rule(point)
    except KeyError as err:
        raise StackRefused(
            f"the law of {name} reads {err} outside its declared parents {rule.parents}: the jet takes the cell"
        ) from err
    return np.asarray(nodes, float), np.asarray(probs, float)


def _check_undeclared(rule: NodeRule, name: str, point: dict, ctx: _Context) -> None:
    """A law must declare every celled variable it reads. Probed once per rule and cell by an order-1 jet."""
    if id(rule) in ctx.probed:
        return
    ctx.probed.add(id(rule))
    for var in ctx.probe:
        if var in rule.parents or var not in point:
            continue

        def moved(z: Any, var: str = var) -> Any:
            x, p = rule({**point, var: point[var] * jnp.exp(z[0])})
            return {"x": jnp.ravel(jnp.asarray(x)), "p": jnp.ravel(jnp.asarray(p))}

        try:
            jet = taylor_coefficients(moved, 1, [(0,), (1,)])
        except TypeError as err:  # a read through Python control flow or float()
            raise StackRefused(
                f"the law of {name} cannot be traced in {var}, which it does not declare: the jet takes the cell"
            ) from err
        for key in ("x", "p"):
            a = np.asarray(jet[key])
            scale = max(float(np.max(np.abs(a[0]), initial=0.0)), 1e-300)
            if not np.all(np.isfinite(a)) or np.any(np.abs(a[1]) > 1e-10 * scale):
                raise StackRefused(
                    f"the law of {name} depends on {var} but declares parents {rule.parents}: the jet takes the cell"
                )


_MAX_AMPLIFICATION = 1e4  # decision 139: the cancellation error is 1.5e-14 to 5.4e-14 of the amplification


def _weights(
    rule: NodeRule, name: str, point: dict, ctx: _Context
) -> tuple[Any, np.ndarray]:
    """Nodes and weight series ``(n_q, order + 1)`` of one averaged level."""
    if rule.parents is None:
        raise StackRefused(
            "a law with undeclared parents: only the jet differentiates it"
        )
    _check_undeclared(rule, name, point, ctx)
    nodes, probs = _call(rule, name, point)
    series = np.zeros((nodes.size, ctx.order + 1))
    series[:, 0] = probs
    if "gamma" not in rule.parents:
        return nodes, series
    if not ctx.reweight:
        return nodes, series  # the naive frozen weights, kept for the regression test
    if rule.density is None:
        raise StackRefused(
            "a law conditional on the celled gamma without a density: frozen-node reweighting needs it"
        )

    def ratio(z: Any) -> Any:
        moved = {**point, "gamma": ctx.gamma_at(z[0])}
        return rule.density(jnp.asarray(nodes), moved) / rule.density(
            jnp.asarray(nodes), point
        )

    try:
        dens = np.asarray(rule.density(jnp.asarray(nodes), point), float)
        if np.any((probs > 0) & ~(np.isfinite(dens) & (dens > 0))):
            raise StackRefused(
                f"the density of a law conditional on gamma is not positive and finite at its own nodes {nodes}: "
                "the jet takes the cell"
            )
        coefs = np.asarray(
            taylor_coefficients(ratio, 1, [(n,) for n in range(ctx.order + 1)])
        )  # (order+1, n_q)
    except TypeError as err:  # a density that JAX cannot trace in gamma
        raise StackRefused(
            f"the density of the law of {name} cannot be differentiated in gamma: {err}"
        ) from err
    series = probs[:, None] * coefs.T
    _check_frozen(rule, point, ctx, nodes, series)
    amplification = float(
        np.max(np.sum(np.abs(series), axis=0)) / np.sum(np.abs(probs))
    )
    if amplification > _MAX_AMPLIFICATION:
        raise StackRefused(
            f"frozen-node reweighting amplifies the weights by {amplification:.3g} at {point}: the law moves by "
            "many widths across the cell and the columns lose digits by cancellation. The jet takes the cell"
        )
    return nodes, series


_TEST_POWERS = 3


def _check_frozen(
    rule: NodeRule,
    point: dict,
    ctx: _Context,
    nodes: np.ndarray,
    series: np.ndarray,
    tol: float = 1e-8,
) -> None:
    """Decision 139: the reweighted weights reproduce the jet of the rule itself.

    With nodes that do not move with ``gamma`` the reference is the rule's own weight jet, node by node (an exact
    identity for a density that matches the rule). With moving nodes it is the jet of ``sum_q p_q(gamma)
    t(x_q(gamma))^m``, ``m = 0..3``, ``t = (x - centre) / spread`` from the reference nodes, against
    ``sum_q w_q^(k) t_q^m``. They agree to ``tol`` of ``max(1, sum_q |w_q^(k)| |t_q|^m)``. A support that moves with
    ``gamma``, a density whose ``gamma`` dependence differs from the rule's in these moments, or non-finite weights
    fail. The check bounds moments, not the integrand: a law whose nodes do not resolve the kernel is a quadrature
    error that the jet shares (decision 139). The columns are Taylor coefficients at the reference point, so node
    shifts across the cell do not enter.
    """
    order = series.shape[1] - 1
    probs = series[:, 0]
    centre = float(np.sum(probs * nodes))
    spread = float(np.sqrt(np.sum(probs * (nodes - centre) ** 2))) or 1.0
    powers = np.arange(_TEST_POWERS + 1)

    def moments(z: Any) -> Any:
        x, p = rule({**point, "gamma": ctx.gamma_at(z[0])})
        x, p = jnp.asarray(x), jnp.asarray(p)
        t = (x - centre) / spread
        return {"x": x, "p": p, "m": jnp.stack([jnp.sum(p * t**m) for m in powers])}

    try:
        jet = taylor_coefficients(moments, 1, [(n,) for n in range(order + 1)])
    except TypeError as err:  # a rule that JAX cannot trace in gamma
        raise StackRefused(
            f"the rule of a law conditional on gamma cannot be differentiated: {err}"
        ) from err
    jx, jp, jm = (np.asarray(jet[k]) for k in ("x", "p", "m"))
    if np.all(np.abs(jx[1:]) <= 1e-12 * (float(np.max(np.abs(nodes))) or 1.0)):
        scale = np.maximum(1.0, np.sum(np.abs(series), axis=0))[None, :]
        bad = (
            ~np.isfinite(series)
            | ~np.isfinite(jp.T)
            | (np.abs(series - jp.T) > tol * scale)
        )
        if np.any(bad):
            q, k = (int(i) for i in np.argwhere(bad)[0])
            raise StackRefused(
                f"frozen-node reweighting does not reproduce the rule's weights at its fixed nodes at {point}: node "
                f"{q}, order {k}, {series[q, k]:.6g} against {jp[k, q]:.6g}. The density does not match the rule. "
                "The jet takes the cell"
            )
        return
    t = (nodes - centre) / spread
    frozen = series.T @ (t[:, None] ** powers)  # (order + 1, m)
    scale = np.maximum(1.0, np.abs(series).T @ (np.abs(t)[:, None] ** powers))
    bad = ~np.isfinite(frozen) | ~np.isfinite(jm) | (np.abs(frozen - jm) > tol * scale)
    if np.any(bad):
        k, m = (int(i) for i in np.argwhere(bad)[0])
        raise StackRefused(
            f"frozen-node reweighting does not reproduce the moving nodes at {point}: order {k}, test power {m}, "
            f"{frozen[k, m]:.6g} against {jm[k, m]:.6g}. The support moves with gamma, the density does not "
            "match the rule, or the nodes do not resolve these moments. The jet takes the cell"
        )
