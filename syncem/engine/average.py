"""The averaged kernel: the function that every column provider differentiates (design/0.8/SPEC.md Sec. 8.1).

LABEL ``eq: illustrated derivative spectra``. ``AveragedKernel(kernel, averaged, legendre=..., fixed=...,
phase=...)`` maps the values of the resolved variables to the channel ``Modes`` averaged over the other
variables. Each averaged variable has a quadrature rule, a function of the point that returns ``(nodes,
probabilities)``, so a law may depend on resolved variables and on averaged variables listed before it (nested
quadrature). A ``Legendre`` variable is projected on ``(2l + 1)/2 P_l(x)`` by Gauss-Legendre nodes on
``[-1, 1]`` and adds a trailing axis per variable. The outermost average runs as ``lax.map`` in batches of
``outer_batch``, and the inner levels are vmapped: vmapping every level ran out of memory at ``N = 4`` with 32
``eta`` and 16 ``B`` nodes (prototype beaming).

The distributions layer supplies rules from its laws. ``uniform_rule`` and ``fixed_rule`` cover the paper's
configurations. ``core_adapter`` exposes the averaged kernel through the ``angular_projection`` contract of
``syncem.model``, so that the unchanged ``_build_core`` can differentiate it.

Units: those of the kernel. Shapes: ``I``, ``V`` ``(n_ch, *L)``, ``P`` ``(n_w, n_ch, *L)`` with one axis per
Legendre variable. Not certified: the quadrature error of the averages, which the basis measures against
direct integration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..base.capabilities import Beaming
from ..model._kernel_helpers import Modes, ProjectedModes
from ._beaming_diagonal import (
    diagonal_weight,
    pair_weights,
    point_weights,
    project_pair,
    project_point,
)

__all__ = ["AveragedKernel", "NodeRule", "core_adapter", "fixed_rule", "uniform_rule"]

Rule = Callable[[dict], tuple[Any, Any]]


@dataclass(frozen=True)
class NodeRule:
    """A quadrature rule ``point -> (nodes, probabilities)`` with what the stack provider needs to know.

    ``parents`` names the variables the law depends on (``None``: unknown, so only the jet may use it).
    ``density(x, point)`` is the law's density, needed for frozen-node reweighting when a parent is celled. A rule
    whose support moves with a celled parent has no valid reweighting and goes to the jet.
    """

    fn: Callable[[dict], tuple[Any, Any]]
    parents: tuple[str, ...] | None = ()
    density: Callable[[Any, dict], Any] | None = None

    def __call__(self, point: dict) -> tuple[Any, Any]:
        return self.fn(point)


def fixed_rule(nodes: Any, probs: Any) -> NodeRule:
    """A rule that ignores the point: fixed nodes and probabilities."""
    nodes, probs = jnp.asarray(nodes, dtype=float), jnp.asarray(probs, dtype=float)
    return NodeRule(lambda point: (nodes, probs), parents=())


def uniform_rule(
    lo: Any, hi: Any, n: int, *, parents: tuple[str, ...] | None = None
) -> NodeRule:
    """Gauss-Legendre nodes of a uniform law on ``[lo, hi]``. ``lo`` and ``hi`` may be functions of the point.

    With constant ends the law has no parents. With moving ends ``parents`` names them (default unknown).
    """
    t, w = (jnp.asarray(a) for a in np.polynomial.legendre.leggauss(int(n)))
    moving = callable(lo) or callable(hi)

    def rule(point: dict) -> tuple[Any, Any]:
        a = lo(point) if callable(lo) else lo
        b = hi(point) if callable(hi) else hi
        return 0.5 * (b - a) * t + 0.5 * (a + b), 0.5 * w

    def density(x: Any, point: dict) -> Any:
        a = lo(point) if callable(lo) else lo
        b = hi(point) if callable(hi) else hi
        return jnp.where((x >= a) & (x <= b), 1.0 / (b - a), 0.0)

    return NodeRule(
        rule, parents=parents if moving else (), density=None if moving else density
    )


class AveragedKernel(eqx.Module):
    """``point -> Modes`` averaged over the averaged variables and projected on the Legendre axes."""

    kernel: Any
    averaged: tuple[tuple[str, Any], ...] = eqx.field(static=True)
    legendre: tuple[tuple[str, int, int], ...] = eqx.field(static=True)
    fixed: tuple[tuple[str, float], ...] = eqx.field(static=True)
    ties: tuple = eqx.field(static=True)
    leaf_factors: tuple = eqx.field(static=True)
    emission_ties: tuple = eqx.field(static=True)
    legendre_panels: tuple = eqx.field(static=True)
    diagonal: tuple[str, ...] = eqx.field(static=True)
    phase: Any = None
    outer_batch: int | None = eqx.field(static=True, default=1)

    def __init__(
        self,
        kernel: Any,
        averaged: Any = (),
        *,
        legendre: dict | None = None,
        fixed: dict | None = None,
        phase: Any = None,
        outer_batch: int | None = 1,
        ties: tuple = (),
        leaf_factors: tuple = (),
        legendre_panels: tuple = (),
        diagonal: tuple[str, ...] = (),
        emission_ties: tuple = (),
    ):
        names = [n for n, _ in averaged] + list(legendre or {}) + list(fixed or {})
        if len(names) != len(set(names)):
            raise ValueError(f"each variable takes one role: {names}")
        variables = {v.name: v for v in getattr(kernel, "variables", ())}
        unknown = set(names) - set(variables)
        if unknown:
            raise ValueError(
                f"{sorted(unknown)} are not variables of the kernel {getattr(kernel, 'name', kernel)}"
            )
        for name in legendre or {}:
            if variables[name].kind != "cosine":
                raise ValueError(
                    f"Legendre degrees resolve a cosine, and {name} is the {variables[name].meaning}"
                )
        if outer_batch is not None and (
            isinstance(outer_batch, bool)
            or not isinstance(outer_batch, (int, np.integer))
            or outer_batch < 1
        ):
            raise ValueError(
                f"outer_batch must be None or a positive int, not {outer_batch!r}"
            )
        self.kernel = kernel
        self.averaged = tuple((str(n), r) for n, r in averaged)
        self.legendre = tuple(
            (str(n), int(L), int(nodes)) for n, (L, nodes) in (legendre or {}).items()
        )
        self.fixed = tuple((str(n), float(v)) for n, v in (fixed or {}).items())
        self.phase = phase
        self.outer_batch = None if outer_batch is None else int(outer_batch)
        self.ties = tuple(ties)
        self.leaf_factors = tuple(leaf_factors)
        self.emission_ties = tuple(emission_ties)
        if diagonal and (
            len(diagonal) != 2
            or len(set(diagonal)) != 2
            or not any(
                isinstance(cap, Beaming)
                and {cap.variable, cap.partner} == set(diagonal)
                for cap in getattr(kernel, "capabilities", ())
            )
            or not set(diagonal).intersection(legendre or {})
            or any(
                name not in variables or variables[name].kind != "cosine"
                for name in diagonal
            )
            or any(name in dict(self.averaged) for name in diagonal)
        ):
            raise ValueError(
                "a Beaming diagonal needs a Legendre axis and a resolved or fixed partner"
            )
        self.diagonal = tuple(diagonal)
        if self.diagonal and ("all", diagonal_weight) not in self.leaf_factors:
            self.leaf_factors = (*self.leaf_factors, ("all", diagonal_weight))
        self.legendre_panels = tuple(legendre_panels)

    def __call__(self, channels: Any, point: dict) -> Modes:
        """Averaged and projected ``Modes`` at the values of the resolved variables in ``point``."""
        taken = (
            {n for n, _ in self.averaged}
            | {n for n, _, _ in self.legendre}
            | {n for n, _ in self.fixed}
        )
        clash = set(point) & taken
        if clash:
            raise ValueError(
                f"{sorted(clash)} are averaged, projected or fixed in this kernel and take no point value"
            )
        variables = {v.name for v in getattr(self.kernel, "variables", ())}
        unknown = set(point) - variables
        if unknown:
            raise ValueError(f"{sorted(unknown)} are not variables of the kernel")
        full = {**dict(self.fixed), **point}
        i, v, p = self._legendre(channels, full, 0)
        return Modes(I=i, V=v, P=p)

    def _legendre(self, channels: Any, point: dict, level: int) -> tuple[Any, Any, Any]:
        if level == len(self.legendre):
            return self._average(channels, point, 0)
        name, degree, n = self.legendre[level]
        if name in self.diagonal:
            return self._diagonal_projection(channels, point, level)
        x, w = np.polynomial.legendre.leggauss(n)
        panels = dict(self.legendre_panels).get(name)
        if isinstance(panels, NodeRule):
            # Integrate this projection once its external parents have been
            # drawn. Place its degree back at its declared axis position.
            i, v, p = self._legendre(channels, point, level + 1)
            return (
                jnp.moveaxis(i, -1, 1),
                jnp.moveaxis(v, -1, 1),
                jnp.moveaxis(p, -1, 2),
            )
        if panels is not None:
            x = np.concatenate([a + (b - a) * (x + 1) / 2 for a, b in panels])
            w = np.concatenate([w * (b - a) / 2 for a, b in panels])
        table = np.stack(
            [np.polynomial.legendre.Legendre.basis(l)(x) for l in range(degree + 1)],
            axis=1,
        )
        weights = jnp.asarray(
            w[:, None] * table * (2 * np.arange(degree + 1) + 1) / 2.0
        )
        i, v, p = jax.vmap(
            lambda x_: self._legendre(channels, {**point, name: x_}, level + 1)
        )(jnp.asarray(x))
        # Degree axis right after the channel axes, so the Legendre axes follow their declared order.
        spin0 = "nc...,nl->cl..."
        return (
            jnp.einsum(spin0, i, weights),
            jnp.einsum(spin0, v, weights),
            jnp.einsum("nwc...,nl->wcl...", p, weights),
        )

    def _diagonal_projection(self, channels, point, level):
        name, degree, n = self.legendre[level]
        other = next(axis for axis in self.diagonal if axis != name)
        locations = {axis: i for i, (axis, _, _) in enumerate(self.legendre)}
        other_level = locations.get(other)
        if other_level is None:
            coordinate = point[other]
            modes = self._legendre(channels, {**point, name: coordinate}, level + 1)
            return project_point(modes, point_weights(coordinate, degree))
        if other_level < level:
            # This degree is inserted by the earlier, joint quadrature.
            return self._legendre(channels, point, level + 1)
        _, other_degree, other_n = self.legendre[other_level]
        x, w = pair_weights(degree, other_degree, max(n, other_n))

        def one(coordinate):
            tied = {**point, name: coordinate, other: coordinate}
            return self._legendre(channels, tied, level + 1)

        modes = jax.vmap(one)(x)
        return project_pair(modes, w, other_level - level)

    def tied_point(self, point):
        """Supply folded coordinates whose physical partner already has a value."""
        if not self.ties:
            return point
        return {
            **point,
            **{child: point[parent] for child, parent in self.ties if parent in point},
        }

    def physical_point(self, point):
        """Use the sightline for known physical laws while retaining raw sample coordinates for weights."""
        if not self.emission_ties:
            return point
        return {**point, **{a: point[b] for a, b in self.emission_ties if b in point}}

    def _average(self, channels: Any, point: dict, level: int) -> tuple[Any, Any, Any]:
        point = self.tied_point(point)
        for name, panel in self.legendre_panels:
            if (
                isinstance(panel, NodeRule)
                and name not in point
                and set(panel.parents).issubset(point)
            ):
                x, w = panel(point)
                degree = next(
                    degree for axis, degree, _ in self.legendre if axis == name
                )
                weights = w[:, None] * jax.vmap(lambda q: point_weights(q, degree))(x)
                values = jax.vmap(
                    lambda q: self._average(channels, {**point, name: q}, level)
                )(x)
                return tuple(
                    jnp.tensordot(value, weights, axes=(0, 0)) for value in values
                )
        if level == len(self.averaged):
            physical = self.physical_point(point)
            modes = self._leaf_modes(channels, physical)
            i = jnp.asarray(modes.I)
            v = jnp.zeros_like(i) if modes.V is None else jnp.asarray(modes.V)
            if v.shape != i.shape:
                raise ValueError(
                    f"kernel {getattr(self.kernel, 'name', self.kernel)} returned V of shape {v.shape}, and "
                    f"the protocol needs the shape of I, {i.shape}"
                )
            p = modes.P
            for block, factor in self.leaf_factors:
                if block == "depth":
                    continue
                scale = factor(point if block == "sample" else physical)
                p = scale * p
                if block in ("all", "sample"):
                    i, v = scale * i, scale * v
            return i, v, p
        name, rule = self.averaged[level]
        nodes, probs = rule(self.physical_point(point))

        def one(value: Any) -> tuple[Any, Any, Any]:
            return self._average(channels, {**point, name: value}, level + 1)

        if level == 0 and self.outer_batch is not None:
            values = jax.lax.map(one, nodes, batch_size=self.outer_batch)
        else:
            values = jax.vmap(one)(nodes)
        return tuple(jnp.tensordot(probs, a, axes=1) for a in values)

    def _leaf_modes(self, channels, physical):
        for block, factor in self.leaf_factors:
            if block == "depth":
                rule = factor(physical)
                if callable(rule):
                    phase = self.phase.with_screen(rule)
                    return self.kernel.channel(channels, physical, phase=phase)
                depths, weights = rule
                name = self.kernel.depth

                def one(depth):
                    return self.kernel.channel(
                        channels, {**physical, name: depth}, phase=self.phase
                    )

                values = jax.vmap(one)(depths)
                return jax.tree.map(
                    lambda value: jnp.tensordot(weights, value, axes=1), values
                )
        return self.kernel.channel(channels, physical, phase=self.phase)


class _CoreAdapter(eqx.Module):
    """The averaged kernel through the ``angular_projection`` contract of ``syncem.model`` (one pair ``(0, 0)``)."""

    averaged: AveragedKernel
    name: ClassVar[str] = "averaged"
    LABEL: ClassVar[tuple[str, ...]] = ("[extension]",)
    components: ClassVar[tuple[str, ...]] = ("I", "Q")
    required_closures: ClassVar[tuple[str, ...]] = ("uniform_mu",)

    def angular_projection(
        self,
        channels: Any,
        gamma: Any,
        B: Any,
        *,
        phase: Any = None,
        depth_ref: Any = 0.0,
        s_depth: Any = 1.0,
        truncation: Any = None,
        quadrature: Any = None,
    ) -> ProjectedModes:
        names = {n for n, _ in self.averaged.averaged} | {
            n for n, _ in self.averaged.fixed
        }
        point = {"gamma": gamma} if "B" in names else {"gamma": gamma, "B": B}
        modes = self.averaged(channels, point)
        return ProjectedModes(
            I=modes.I[:, None], V=modes.V[:, None], P=modes.P[:, :, None]
        )


def core_adapter(averaged: AveragedKernel) -> _CoreAdapter:
    """The adapter for ``syncem.model.basis._build_core``. Legendre axes are not supported on this route."""
    if averaged.legendre:
        raise ValueError("the core route takes no Legendre axes. Use the jet provider")
    return _CoreAdapter(averaged)
