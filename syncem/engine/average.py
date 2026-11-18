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

from typing import Any, Callable, ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..model._kernel_helpers import Modes, ProjectedModes

__all__ = ["AveragedKernel", "core_adapter", "fixed_rule", "uniform_rule"]

Rule = Callable[[dict], tuple[Any, Any]]


def fixed_rule(nodes: Any, probs: Any) -> Rule:
    """A rule that ignores the point: fixed nodes and probabilities."""
    nodes, probs = jnp.asarray(nodes, dtype=float), jnp.asarray(probs, dtype=float)
    return lambda point: (nodes, probs)


def uniform_rule(lo: Any, hi: Any, n: int) -> Rule:
    """Gauss-Legendre nodes of a uniform law on ``[lo, hi]``. ``lo`` and ``hi`` may be functions of the point."""
    t, w = (jnp.asarray(a) for a in np.polynomial.legendre.leggauss(int(n)))

    def rule(point: dict) -> tuple[Any, Any]:
        a = lo(point) if callable(lo) else lo
        b = hi(point) if callable(hi) else hi
        return 0.5 * (b - a) * t + 0.5 * (a + b), 0.5 * w

    return rule


class AveragedKernel(eqx.Module):
    """``point -> Modes`` averaged over the averaged variables and projected on the Legendre axes."""

    kernel: Any
    averaged: tuple[tuple[str, Any], ...] = eqx.field(static=True)
    legendre: tuple[tuple[str, int, int], ...] = eqx.field(static=True)
    fixed: tuple[tuple[str, float], ...] = eqx.field(static=True)
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
        i, p = self._legendre(channels, full, 0)
        return Modes(I=i, V=jnp.zeros_like(i), P=p)

    def _legendre(self, channels: Any, point: dict, level: int) -> tuple[Any, Any]:
        if level == len(self.legendre):
            return self._average(channels, point, 0)
        name, degree, n = self.legendre[level]
        x, w = np.polynomial.legendre.leggauss(n)
        table = np.stack(
            [np.polynomial.legendre.Legendre.basis(l)(x) for l in range(degree + 1)],
            axis=1,
        )
        weights = jnp.asarray(
            w[:, None] * table * (2 * np.arange(degree + 1) + 1) / 2.0
        )
        i, p = jax.vmap(
            lambda v: self._legendre(channels, {**point, name: v}, level + 1)
        )(jnp.asarray(x))
        # Degree axis right after the channel axes, so the Legendre axes follow their declared order.
        return jnp.einsum("nc...,nl->cl...", i, weights), jnp.einsum(
            "nwc...,nl->wcl...", p, weights
        )

    def _average(self, channels: Any, point: dict, level: int) -> tuple[Any, Any]:
        if level == len(self.averaged):
            modes = self.kernel.channel(channels, point, phase=self.phase)
            return modes.I, modes.P
        name, rule = self.averaged[level]
        nodes, probs = rule(point)

        def one(value: Any) -> tuple[Any, Any]:
            return self._average(channels, {**point, name: value}, level + 1)

        if level == 0 and self.outer_batch is not None:
            i, p = jax.lax.map(one, nodes, batch_size=self.outer_batch)
        else:
            i, p = jax.vmap(one)(nodes)
        return jnp.tensordot(probs, i, axes=1), jnp.tensordot(probs, p, axes=1)


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
