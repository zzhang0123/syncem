"""Fixed-field reference interpolation on scaled energy."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from ..base.capabilities import FieldScalingRelation
from ..base.kernel import Modes
from .average import AveragedKernel, NodeRule

if TYPE_CHECKING:
    from .provider_table import CellKernelTable


class TableRefused(ValueError):
    """The cell table or scaling reference precondition does not hold."""


@dataclass(frozen=True)
class ScalingReference:
    """A fixed-field energy table for field-scaling at the joint-law nodes."""

    tables: tuple[CellKernelTable, ...]
    relation: FieldScalingRelation
    reference_field: float

    def evaluate(self, points: Mapping[str, Any]) -> Modes:
        """Evaluate at paired energy and field nodes inside the scaled span."""
        if set(points) != {self.relation.energy, self.relation.variable}:
            raise ValueError(
                "field scaling requires only paired energy and field nodes"
            )
        energy = np.asarray(points[self.relation.energy], dtype=float)
        field = np.asarray(points[self.relation.variable], dtype=float)
        if (
            energy.ndim != 1
            or energy.shape != field.shape
            or np.any(~np.isfinite(energy))
            or np.any(energy <= 0)
            or np.any(~np.isfinite(field))
            or np.any(field < 0)
        ):
            raise ValueError(
                "field scaling requires matching positive energies and nonnegative fields"
            )
        ratio = field / self.reference_field
        scaled = energy * np.sqrt(ratio)
        remaining = ratio > 0
        first = self.tables[0]
        result = Modes(
            *(
                np.zeros((energy.size, *a.shape[1:]), dtype=a.dtype)
                for a in first.coefficients
            )
        )
        for table in self.tables:
            axis = table.axes[0]
            use = remaining & (scaled >= axis.lo) & (scaled <= axis.hi)
            if not np.any(use):
                continue
            values = table.evaluate({axis.name: scaled[use]})
            for dst, value in zip(result, values, strict=True):
                dst[use] = value * ratio[use].reshape((-1,) + (1,) * (value.ndim - 1))
            remaining &= ~use
        if np.any(remaining):
            raise TableRefused(
                "scaled energies leave the fixed-field table's covered span"
            )
        return result


def scaling_reference(
    averaged: AveragedKernel,
    tables: Iterable[CellKernelTable],
    *,
    azimuth_parents: Iterable[str] = (),
) -> ScalingReference:
    """Use checked one-axis energy tables at fixed field when the laws permit."""
    relations = [
        c for c in averaged.kernel.capabilities if isinstance(c, FieldScalingRelation)
    ]
    if len(relations) != 1:
        raise TableRefused("the kernel needs one declared FieldScalingRelation")
    relation = relations[0]
    names = {relation.energy, relation.variable}
    field = dict(averaged.fixed).get(relation.variable)
    if field is None or not np.isfinite(field) or field <= 0:
        raise TableRefused(
            "field scaling requires the reference field to be positive and fixed"
        )
    if names.intersection(azimuth_parents) or any(
        not isinstance(rule, NodeRule)
        or rule.parents is None
        or names.intersection(rule.parents)
        for _, rule in averaged.averaged
    ):
        raise TableRefused(
            "field scaling requires pitch, inclination, depth and azimuth laws independent of energy and field"
        )
    tables = tuple(tables)
    if not tables or any(
        tuple(a.name for a in t.axes) != (relation.energy,) for t in tables
    ):
        raise TableRefused("field scaling needs one-axis tables of the declared energy")
    channel_keys = tables[0].channel_keys
    if any(
        t.averaged is not averaged or t.channel_keys != channel_keys for t in tables
    ):
        raise TableRefused(
            "field scaling needs tables of the same fixed-field kernel and channels"
        )
    return ScalingReference(tables, relation, field)
