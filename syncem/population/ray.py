"""Piecewise constant line-of-sight profiles (SPEC Sec. 3.5 and 5.1, R8).

LABEL [extension]. The shapes of step lengths and densities
are broadcastable to ``(..., n_steps)``. The field vector has shape
``(..., n_steps, 3)``, in Gauss after placement, and its last component
points towards the observer. Lengths use cm and densities use cm^-3.
Depths use rad m^-2. Profile discretisation is not certified.
"""

from __future__ import annotations

from typing import Any, NamedTuple

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..model._provenance import strict_json
from ..rm import RM_PER_UNIT
from ..units import Quantity, convert, to_canonical
from .blocks import _checked, _finite, _real, _record_hint

__all__ = ["Ray"]


class _Step(NamedTuple):
    factors: dict
    electrons: Any
    column: Any


def _profile(value, dimension, name):
    return _finite(_real(to_canonical(value, dimension, name=name), name), name)


def _law_profile(value, shape, name, *, optional=False):
    if value is None and optional:
        return None
    if hasattr(value, "rule"):
        return value
    if not isinstance(value, (list, tuple, np.ndarray)):
        raise TypeError(f"{name} needs a law or a profile of laws")
    try:
        profile = np.broadcast_to(np.asarray(value, dtype=object), shape)
    except ValueError as error:
        raise ValueError(
            f"{name} laws must broadcast to ray profile {shape}"
        ) from error
    if not all(hasattr(law, "rule") for law in profile.flat):
        raise TypeError(f"each entry of the {name} profile must supply rule()")
    return tuple(profile.flat)


class Ray(eqx.Module):
    """Profile of emitting steps, with one component per step on each sightline."""

    step: Any
    B_vector: Any
    thermal_density: Any
    electron_density: Any
    electrons: Any
    pitch: Any

    def __init__(
        self,
        *,
        step,
        B_vector,
        thermal_density,
        electrons,
        electron_density,
        pitch=None,
    ):
        step = _profile(step, "length", "step")
        field = _profile(B_vector, "field", "B_vector")
        thermal = _profile(thermal_density, "density", "thermal_density")
        emit = _profile(electron_density, "density", "electron_density")
        if field.ndim < 2 or field.shape[-1] != 3 or field.shape[-2] == 0:
            raise ValueError(
                "B_vector needs shape (..., n_steps, 3) with at least one step"
            )
        try:
            shape = np.broadcast_shapes(
                field.shape[:-1], step.shape, thermal.shape, emit.shape
            )
        except ValueError as error:
            raise ValueError(
                "lengths, field and densities need compatible ray profile shapes"
            ) from error
        if any(size == 0 for size in shape):
            raise ValueError("a ray profile must be nonempty")
        step = _checked(step, step <= 0, "step lengths must be positive")
        thermal = _checked(thermal, thermal < 0, "thermal_density must be nonnegative")
        emit = _checked(emit, emit < 0, "electron_density must be nonnegative")
        self.step = jnp.broadcast_to(step, shape)
        self.B_vector = jnp.broadcast_to(field, (*shape, 3))
        self.thermal_density = jnp.broadcast_to(thermal, shape)
        self.electron_density = jnp.broadcast_to(emit, shape)
        self.electrons = _law_profile(electrons, shape, "electrons")
        self.pitch = _law_profile(pitch, shape, "pitch", optional=True)

    @property
    def shape(self) -> tuple[int, ...]:
        """Leading sightline axes."""
        return self.step.shape[:-1]

    @property
    def n_steps(self) -> int:
        """Number of steps along a sightline."""
        return self.step.shape[-1]

    @property
    def variables(self) -> tuple[str, ...]:
        """Synchrotron variables supplied by the ray."""
        return ("gamma", "B", "eta", "phi", "depth") + (
            () if self.pitch is None else ("mu",)
        )

    def __getitem__(self, indices):
        """Select on the sightline axes, retaining all steps."""
        if not self.shape:
            raise IndexError("this Ray is one sightline")
        mapping = np.arange(self.step.size).reshape(self.step.shape)
        leading = indices if isinstance(indices, tuple) else (indices,)
        # The final step axis is retained even with slices or a leading ellipsis.
        selected = mapping[(*leading, slice(None))]
        if selected.shape[-1:] != (self.n_steps,):
            raise IndexError("Ray indices select sightlines, not steps")
        profiles = {
            name: (
                tuple(value[i] for i in selected.ravel())
                if isinstance(value, tuple)
                else value
            )
            for name, value in (("electrons", self.electrons), ("pitch", self.pitch))
        }
        at = (*leading, slice(None))
        result = object.__new__(type(self))
        state = {
            "step": self.step[at],
            "B_vector": self.B_vector[(*at, slice(None))],
            "thermal_density": self.thermal_density[at],
            "electron_density": self.electron_density[at],
            **profiles,
        }
        for name, value in state.items():
            object.__setattr__(result, name, value)
        return result

    def coordinates(self) -> dict:
        """Canonical field, inclination, azimuth and depth at each emitting midpoint."""
        bx, by, bz = jnp.moveaxis(self.B_vector, -1, 0)
        field = jnp.hypot(jnp.hypot(bx, by), bz)
        eta = bz / jnp.where(field > 0, field, 1.0)
        signed = (
            RM_PER_UNIT
            * self.thermal_density
            * convert(bz, "G", "uG")
            * convert(self.step, "cm", "pc")
        )
        # Profiles run toward the observer at the end of the final step.
        depth = jnp.cumsum(signed[..., ::-1], axis=-1)[..., ::-1] - 0.5 * signed
        return {
            "B": _finite(field, "ray field magnitude"),
            "eta": eta,
            "phi": jnp.arctan2(by, bx),
            "depth": _finite(depth, "ray depth"),
        }

    @property
    def columns(self):
        """Physical emitting column of each step in cm^-2."""
        return _finite(self.step * self.electron_density, "ray column")

    def iter_steps(self):
        """Yield fixed-point factors, the electron shape law and column for one sightline.

        The caller uses the returned electron law at unit column before
        applying the supplied physical column. A zero field or zero column
        gives no emission and contributes no component.
        """
        if self.shape:
            raise ValueError("select one Ray sightline before iterating over its steps")
        points = self.coordinates()
        column = self.columns
        for i in range(self.n_steps):
            if not bool((column[i] > 0) & (points["B"][i] > 0)):
                continue
            factors = {
                "B": Quantity(points["B"][i], "G"),
                "eta": points["eta"][i],
                "phi": Quantity(points["phi"][i], "rad"),
                "depth": points["depth"][i],
            }
            if self.pitch is not None:
                factors["mu"] = (
                    self.pitch[i] if isinstance(self.pitch, tuple) else self.pitch
                )
            electrons = (
                self.electrons[i]
                if isinstance(self.electrons, tuple)
                else self.electrons
            )
            yield _Step(factors, electrons, column[i])

    def record(self) -> dict:
        """Strict JSON with profile units and the electron and pitch declarations."""
        return strict_json(
            {
                "family": "Ray",
                "variables": self.variables,
                "step": self.step,
                "B_vector": self.B_vector,
                "thermal_density": self.thermal_density,
                "electron_density": self.electron_density,
                "electrons": _record_hint(self.electrons),
                "pitch": _record_hint(self.pitch),
                "units": {
                    "step": "cm",
                    "B_vector": "G",
                    "thermal_density": "cm^-3",
                    "electron_density": "cm^-3",
                },
            }
        )
