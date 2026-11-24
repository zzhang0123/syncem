"""Structural forward and inverse planning records (SPEC Sec. 3.6).

LABEL [extension].

A plan records the selected lattice shape and unit-bearing inputs per
component. It has no certified error on an observed or generated spectrum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

from .roles import CompiledRoles


@dataclass(frozen=True, eq=False)
class PlanRecord:
    """Serializable facts and defaults of an eager role assignment."""

    layout: CompiledRoles
    kernel: str
    lattice_key: str | None
    index: Any
    columns: Any
    kind: str
    fourier_order: int | None = None
    depth_truncation: float | None = None
    build: Any = None
    kernel_assignment: tuple[int, ...] = ()

    @property
    def roles(self):
        return self.layout.components[0].roles

    @property
    def assumptions(self):
        return self.layout.assumptions

    @property
    def caveats(self):
        return self.layout.caveats

    @property
    def defaults(self):
        return self.layout.components[0].defaults

    @property
    def notes(self):
        return self.caveats + (() if self.build is None else self.build.notes)

    def to_dict(self):
        """Strict JSON of the structural plan without materialized rows."""
        return {
            "roles": {name: asdict(role) for name, role in self.roles.items()},
            "kernel": self.kernel,
            "kernel_assignment": list(self.kernel_assignment),
            "lattice_key": self.lattice_key,
            "rows": 0 if self.index is None else self.index.n,
            "columns": 0 if self.columns is None else self.columns.n,
            "kind": self.kind,
            "depth_truncation": self.depth_truncation,
            **(
                {"azimuth_fourier_order": self.fourier_order}
                if self.fourier_order is not None
                else {}
            ),
            "defaults": [list(item) for item in self.defaults],
            "assumptions": list(self.assumptions),
            "caveats": list(self.caveats),
            "basis_groups": [list(group) for group in self.layout.basis_groups],
            "build": None if self.build is None else self.build.to_dict(),
        }


@dataclass(frozen=True, eq=False)
class Plan:
    """The role assignment and geometry, before running any spectral provider."""

    record: PlanRecord
    lattice: Any

    def __getattr__(self, name):
        return getattr(self.record, name)

    def __str__(self):
        line = f"{self.kind} plan with {self.kernel}"
        if self.index is not None and self.columns is not None:
            line += f" · {self.index.n} rows · {self.columns.n} columns · {self.index.n * 8:,} dense moment bytes per sightline"
        items = [line]
        for name, role in self.roles.items():
            default = " [default]" if role.default else ""
            items.append(f"{name}: {role.role}, {role.how} ({role.rule}){default}")
        if self.fourier_order is not None:
            items.append(
                f"AzimuthFourier(m_max={self.fourier_order}): azimuth_truncation measured by a direct check."
            )
        if self.depth_truncation is not None:
            items.append(f"depth phase remainder bound {self.depth_truncation:.4g}")
        if self.kernel_assignment:
            groups = sorted(set(self.kernel_assignment))
            items.append(
                "Hybrid kernels per cell  "
                + ", ".join(
                    f"{k} ({self.kernel_assignment.count(k)} cells)" for k in groups
                )
            )
        items.extend(self.caveats)
        return "\n".join(items)
