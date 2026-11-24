"""Eager assignment of compatible kernels to product cells (SPEC Sec. 8.4).

LABEL [extension].

Each product cell has a scalar child index. The child kernels share
canonical input units and mode shapes. Agreement at a switch samples the
matching faces and has no certified extension away from them.
"""

from __future__ import annotations

from collections.abc import Mapping
from copy import copy
from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from ..model.errors import ErrorTerm, WEAKEST_ORDER
from ..values.errors import RegimeError, RoleError
from .kernel import Kernel, Modes, Regime

__all__ = ["HybridKernel"]


@dataclass(frozen=True, eq=False, init=False)
class HybridKernel:
    """An eager per-cell assignment and a shared traceable kernel protocol."""

    name: ClassVar[str] = "hybrid"
    traceable: ClassVar[bool] = True
    lines: ClassVar[bool] = False
    kernels: tuple
    variables: tuple
    stokes: tuple[str, ...]
    primary: str | None
    azimuth: str | None
    depth: str | None
    capabilities: tuple
    switch: tuple[str, float] | None
    switch_tol: float
    assign: object
    geometry: tuple
    assignment: tuple[int, ...]
    _by_box: Mapping

    def __init__(self, *kernels, switch=None, assign=None, switch_tol=1e-3):
        if len(kernels) < 2:
            raise RoleError("HybridKernel needs at least two kernels")
        if any(
            not isinstance(k, Kernel)
            or not k.traceable
            or k.lines
            or isinstance(k, HybridKernel)
            for k in kernels
        ):
            raise RoleError(
                "HybridKernel entries must be traceable non-harmonic kernels without lines"
            )
        if (switch is None) == (assign is None):
            raise RoleError("supply one of switch= or assign= to HybridKernel")
        if assign is not None and not callable(assign):
            raise TypeError("assign must be a callable taking one cell box")
        first = kernels[0]
        signature = tuple(
            getattr(first, name)
            for name in ("variables", "primary", "azimuth", "depth")
        )
        if any(
            tuple(getattr(k, n) for n in ("variables", "primary", "azimuth", "depth"))
            != signature
            for k in kernels[1:]
        ):
            raise RoleError(
                "hybrid kernels must have identical variables and primary, azimuth and depth roles"
            )
        common = tuple(s for s in first.stokes if all(s in k.stokes for k in kernels))
        if "I" not in common:
            raise RoleError(
                "hybrid kernels need common Stokes I to compare switch errors"
            )
        threshold = None
        if switch is not None:
            if len(kernels) != 2 or not isinstance(switch, Mapping) or len(switch) != 1:
                raise RoleError(
                    "switch={variable: edge} needs two kernels and one threshold"
                )
            name, value = next(iter(switch.items()))
            variables = {v.name: v for v in first.variables}
            if name not in variables:
                raise RoleError(
                    f"hybrid switch variable {name!r} is absent from the kernels"
                )
            edge = variables[name].place(value)
            if not isinstance(edge, Real) or not np.isfinite(edge):
                raise RoleError("the hybrid switch edge must be a finite scalar")
            threshold = name, float(edge)
        if (
            isinstance(switch_tol, (bool, np.bool_))
            or not isinstance(switch_tol, Real)
            or not np.isfinite(switch_tol)
            or switch_tol <= 0
        ):
            raise ValueError("switch_tol must be positive and finite")
        shared = tuple(
            c
            for c in first.capabilities
            if all(c in k.capabilities for k in kernels[1:])
        )
        for name, value in (
            ("kernels", kernels),
            *zip(("variables", "primary", "azimuth", "depth"), signature),
            ("stokes", common),
            ("capabilities", shared),
            ("switch", threshold),
            ("assign", assign),
            ("switch_tol", float(switch_tol)),
            ("geometry", ()),
            ("assignment", ()),
            ("_by_box", MappingProxyType({})),
        ):
            object.__setattr__(self, name, value)

    def cell_index(self, box):
        """Return the declared kernel index for an interval mapping."""
        if self.geometry:
            names = tuple(item[0] for item in self.geometry[0])
            if all(name in box for name in names):
                signature = tuple((name, *box[name]) for name in names)
                if signature in self._by_box:
                    return self._by_box[signature]
        if self.switch is not None:
            name, edge = self.switch
            if name not in box:
                raise RoleError(
                    f"resolve the hybrid switch variable {name!r} on the lattice"
                )
            lo, hi = box[name]
            if hi <= edge:
                return 0
            if lo >= edge:
                return 1
            raise RoleError(f"split the cell on {name} at the hybrid switch {edge:g}")
        selected = self.assign(box)
        if (
            isinstance(selected, (bool, np.bool_))
            or not isinstance(selected, Integral)
            or not 0 <= selected < len(self.kernels)
        ):
            raise RoleError(
                f"hybrid assign must return an index in 0..{len(self.kernels) - 1}, got {selected!r}"
            )
        return int(selected)

    def bind(self, table):
        """Freeze the host assignment in the given cell order."""
        if not table.variables:
            raise RoleError("HybridKernel needs a lattice with celled variables")
        boxes = tuple(
            tuple(
                (name, float(a), float(b))
                for name, a, b in zip(table.variables, lo, hi)
            )
            for lo, hi in zip(table.lo, table.hi)
        )
        if boxes == self.geometry:
            return self
        assignment = tuple(
            self.cell_index({name: (lo, hi) for name, lo, hi in cell}) for cell in boxes
        )
        bound = copy(self)
        object.__setattr__(bound, "geometry", boxes)
        object.__setattr__(bound, "assignment", assignment)
        object.__setattr__(
            bound, "_by_box", MappingProxyType(dict(zip(boxes, assignment)))
        )
        return bound

    def regime(self, box, channels):
        label = self.cell_index(box)
        try:
            return self.kernels[label].regime(box, channels)
        except KeyError as exc:
            return Regime(
                False, f"hybrid regime check needs a support interval for {exc.args[0]}"
            )

    def _point_index(self, point):
        if not self.geometry and self.switch is None:
            raise RoleError(
                "bind the hybrid assignment to a lattice before point evaluation"
            )
        if self.switch is None:
            selected = jnp.asarray(-1)
        else:
            name, edge = self.switch
            selected = jnp.asarray(jnp.asarray(point[name]) > edge, dtype=int)
        if self.geometry:
            names = tuple(entry[0] for entry in self.geometry[0])
            bounds = np.asarray(
                [[(lo, hi) for _, lo, hi in box] for box in self.geometry]
            )
            coordinates = jnp.stack([jnp.asarray(point[name]) for name in names])
            internal = jnp.all(
                (bounds[..., 0] <= coordinates) & (coordinates <= bounds[..., 1]),
                axis=-1,
            )
            selected = jnp.where(
                jnp.any(internal),
                jnp.asarray(self.assignment)[jnp.argmax(internal)],
                selected,
            )
        return eqx.error_if(
            selected,
            selected < 0,
            "hybrid assign= has no kernel outside its cell union. Integrate within='cells' or use switch=.",
        )

    def _modes(self, modes):
        i = jnp.asarray(modes.I, dtype=float)
        p = (
            jnp.asarray(modes.P, dtype=complex)
            if "P" in self.stokes
            else jnp.zeros((1, *i.shape), complex)
        )
        v = (
            jnp.asarray(modes.V, dtype=float)
            if "V" in self.stokes
            else jnp.zeros_like(i)
        )
        if p.shape != (1, *i.shape) or v.shape != i.shape:
            raise RoleError(
                "hybrid kernels must return one spin-two mode with the common Stokes geometry"
            )
        return Modes(I=i, P=p, V=v)

    def channel(self, channels, point, *, phase=None, per_node=False):
        """Evaluate direct integration points in the bound geometry."""
        branches = tuple(
            lambda k=k: self._modes(
                k.channel(channels, point, phase=phase, per_node=per_node)
            )
            for k in self.kernels
        )
        return jax.lax.switch(self._point_index(point), branches)

    def spectral(self, nu, point):
        branches = tuple(
            lambda k=k: self._modes(k.spectral(nu, point)) for k in self.kernels
        )
        return jax.lax.switch(self._point_index(point), branches)

    def physical_error(self, channels):
        terms = tuple(k.physical_error(channels) for k in self.kernels)
        if any(term.kind == "unbounded" for term in terms):
            return ErrorTerm.unbounded(
                "one or more hybrid child kernels have unbounded physical error",
                "E_phys",
            )
        valued = tuple(term for term in terms if term.value is not None)
        if not valued:
            return ErrorTerm.not_applicable(
                "no hybrid child has an applicable physical error term"
            )
        values = jnp.broadcast_arrays(*(t.value for t in valued))
        kind = max((t.kind for t in valued), key=WEAKEST_ORDER.index)
        return ErrorTerm(
            jnp.max(jnp.stack(values), axis=0),
            kind,
            "maximum declared physical envelope across hybrid children",
            "E_phys",
        )

    def describe(self):
        children = tuple(tuple(k.describe()) for k in self.kernels)
        reasons = tuple(
            reason
            for entries in children
            for reason in dict(entries).get("not_cached", ())
        )
        return (
            ("name", self.name),
            (
                "children",
                tuple(
                    (type(k).__module__, type(k).__qualname__, entries)
                    for k, entries in zip(self.kernels, children)
                ),
            ),
            ("switch", self.switch),
            ("assignment", self.assignment),
            ("not_cached", reasons),
        )

    def check_regime(self, boxes, channels):
        for cell, (box, branch) in enumerate(zip(boxes, self.assignment, strict=True)):
            try:
                status = self.kernels[branch].regime(box, channels)
            except KeyError as exc:
                raise RegimeError(
                    f"cell {cell}: hybrid child needs a support interval for {exc.args[0]}"
                ) from exc
            if not status.ok:
                raise RegimeError(
                    f"cell {cell}, hybrid kernel {branch}: {status.reason}"
                )
