"""Import the finite response of a frozen single-reference basis (SPEC Secs. 3.4 and 4.2).

LABEL ``eq: channel basis expansion``. A stored Taylor coefficient uses the
same affine chart in both bases. Built-in unit-peak band responses are divided
by the band area in Hz. Pair a source vector with a separately supplied
physical column in cm^-2 to obtain moments in the one-cell lattice.

Shapes: the imported design is ``(n_ch, n_stokes, n_columns)``.
Not certified: the source's kernel accuracy and its Taylor remainder.
"""

from __future__ import annotations

import dataclasses
from typing import Any, ClassVar

import equinox as eqx
import numpy as np

from ..base.capabilities import Parity
from ..base.channels import Channels
from ..base.variables import SYNCHROTRON_VARIABLES
from ..lattice.cells import Cells
from ..lattice.lattice import Lattice
from ..lattice.moments import Moments
from ..lattice.representations import AzimuthHarmonic, Legendre
from ..model._basis_checks import UNITS as LEGACY_UNITS
from ..model.basis import SpectralBasis as LegacyBasis
from ..model.channels import Channels as LegacyChannels
from ..model.errors import ErrorTerm as LegacyErrorTerm
from ..model.moments import JointMoments
from ..records.ledger import ErrorTerm, Ledger
from ..units import to_canonical
from ..values.errors import RoleError, UnitError
from ._basis_update import build_record
from .average import AveragedKernel
from .basis import SpectralBasis
from .columns import ColumnRules, column_map

__all__ = ["SingleReferenceBasis", "from_single_reference", "same_line_response"]

_AXES = ("gamma", "B", "depth")
_REBUILD = (
    "a stored single-reference basis has no kernel callable. "
    "Build a new basis with the original kernel to"
)


def _line_keys(channels):
    from ._regime_columns import regime_channel_keys

    return regime_channel_keys(channels)


def same_line_response(stored, wanted) -> bool:
    """Whether two channel sets have the same line transmission and quadrature rows.

    A harmonic response reads the transmission at discrete line frequencies, so
    equal quadrature rows do not imply equal responses. Opaque responses compare
    equal only as identical declarations.
    """
    keys = _line_keys(wanted)
    if None in keys:
        return bool(eqx.tree_equal(stored, wanted))
    return keys == _line_keys(stored)


@dataclasses.dataclass(frozen=True)
class _StoredKernel:
    """Structural declarations for the stored response."""

    stokes: tuple[str, ...]
    capabilities: tuple[Parity, ...]
    name: ClassVar[str] = "stored single-reference response"
    primary: ClassVar[str] = "gamma"
    azimuth: ClassVar[str] = "phi"
    depth: ClassVar[str] = "depth"
    variables: ClassVar[tuple] = tuple(SYNCHROTRON_VARIABLES.values())

    def channel(self, *args, **kwargs):
        raise RoleError(f"{_REBUILD} evaluate additional kernel values")

    def physical_error(self, channels):
        """The imported coefficients do not declare a source discrepancy."""
        return LegacyErrorTerm.unbounded(
            "the stored basis has no physical discrepancy bound"
        )


class SingleReferenceBasis(SpectralBasis):
    """A one-cell basis containing an imported response and its source record."""

    legacy_provenance: Any = eqx.field(static=True, kw_only=True)
    legacy_reference: tuple = eqx.field(static=True, kw_only=True)
    legacy_kernel: Any = eqx.field(default=None, kw_only=True)
    source_kernel: Any = eqx.field(default=None, kw_only=True)

    def _match_channel_rows(self, channels, keys):
        from ._regime_columns import regime_channel_indices

        found = regime_channel_indices(self.channels, channels)
        if np.any(found < 0):
            if eqx.tree_equal(self.channels, channels):
                # Opaque responses can still reuse the identical ordered declaration.
                return np.arange(channels.n_ch), {}
            if not set(keys).issubset(self.channels.row_keys()):
                raise RoleError(f"{_REBUILD} evaluate new channel rows")
            raise RoleError(
                f"{_REBUILD} evaluate a new or unidentified line transmission "
                "on the stored quadrature rows"
            )
        return found, {}

    def with_order(self, order):
        """Restrict the imported response to a lower total order."""
        valid = isinstance(order, (int, np.integer)) and not isinstance(order, bool)
        if valid and order > self.lattice.order:
            raise RoleError(f"{_REBUILD} extend the Taylor order")
        return super().with_order(order)

    def direct(self, population, *, within="support"):
        """Direct integration requires the source kernel."""
        raise RoleError(f"{_REBUILD} integrate a population directly")

    def moments(self, population, *, allow_mismatch=False):
        """Check supplied lattice moments against the imported lattice."""
        if not isinstance(population, Moments):
            raise RoleError(
                "supply Moments on the imported lattice, or use "
                "from_legacy_moments(legacy_moments, column=...)"
            )
        return super().moments(population, allow_mismatch=allow_mismatch)

    def predict(self, population_or_moments, *, check=True, allow_mismatch=False):
        """Contract the stored finite response with supplied lattice moments."""
        moments = self.moments(population_or_moments, allow_mismatch=allow_mismatch)
        return super().predict(moments, check=check, allow_mismatch=allow_mismatch)

    def from_legacy_moments(self, moments, *, column):
        """Convert normalised source moments with an explicit column in cm^-2."""
        if not isinstance(moments, JointMoments):
            raise TypeError("moments must be syncem.model.JointMoments")
        if moments.reference.describe() != self.legacy_reference:
            raise RoleError(
                "legacy moments must carry the imported reference and scales"
            )
        mass = to_canonical(column, "column", name="single-reference column")
        mass = np.asarray(mass)
        if mass.ndim or np.iscomplexobj(mass) or not np.isfinite(mass) or mass < 0:
            raise UnitError(
                "single-reference column must be a finite non-negative scalar"
            )
        vectors = (np.asarray(moments.m0), moments.m2.real, moments.m2.imag)
        if (
            not all(np.all(np.isfinite(a)) for a in vectors)
            or abs(vectors[0][0] - 1.0) > 1e-12
        ):
            raise ValueError(
                "legacy moments must be finite and normalised to M0[0] = 1"
            )
        source0 = {row: j for j, row in enumerate(moments.index.h0)}
        source2 = {row: j for j, row in enumerate(moments.index.h2)}
        rows = _row_keys(self.lattice)
        blocks = self.lattice.index.arrays.block
        values = np.zeros(self.lattice.index.n)
        for k in np.flatnonzero(self.columns.col_of_row >= 0):
            block = int(blocks[k])
            table = source0 if block == 0 else source2
            j = table.get(tuple(rows[k]))
            if j is None:
                raise RoleError(
                    "legacy moments omit a row used by the imported response"
                )
            values[k] = vectors[block][j]
        return Moments(self.lattice, mass * values, masked=tuple(self.columns.masked))


def _channels(source):
    if type(source) not in (LegacyChannels, Channels):
        raise UnitError("import a legacy Channels with a declared band response")
    normalisation = source.normalisation
    if normalisation not in ("unit_peak", "unit_integral"):
        raise UnitError("legacy channels must declare unit_peak or unit_integral")
    if source.family == "callable":
        raise UnitError(
            "legacy callable channels declare response units without a band area. "
            "Import a basis built on built-in or tabulated bands."
        )
    fields = {
        f.name: getattr(source, f.name) for f in dataclasses.fields(LegacyChannels)
    }
    fields["normalisation"] = "unit_integral"
    unit = getattr(source, "unit", "Hz")
    if source.family == "basis":
        reference, ref_scale = _channels(source.reference_channels)
        weighted = np.asarray(source.coefficients) / ref_scale[None, :]
        area = weighted.sum(axis=1)
        if np.any(~np.isfinite(area) | (area <= 0)):
            raise UnitError("legacy composed channel areas must be finite and positive")
        fields["reference_channels"] = reference
        fields["coefficients"] = weighted / area[:, None]
        return Channels(**fields, unit=unit), 1.0 / area
    channels = Channels(**fields, unit=unit)
    multiplier = (
        np.asarray(channels.peak_scale())
        if normalisation == "unit_peak"
        else np.ones(channels.n_ch)
    )
    if np.any(~np.isfinite(multiplier) | (multiplier <= 0)):
        raise UnitError("legacy channel areas must be finite and positive")
    return channels, multiplier


def _lattice(source):
    tr = source.truncation
    reference = source.reference
    refs = (reference.gamma0, reference.B0, reference.depth_ref)
    units = ("", "G", "rad m^-2")
    axes = {}
    depth_in_total = tr.depth_degree is None
    for name, ref, scale, cap, unit in zip(
        _AXES, refs, reference.scales, tr.caps(), units
    ):
        lo, hi = (float(value) for value in getattr(source.support, name))
        if name == "depth" and lo == hi and (not source.index.n2 or cap == 0):
            continue
        if not lo < hi or not lo <= float(ref) <= hi:
            raise RoleError(
                f"legacy {name} support must have positive width and contain its reference "
                "to make a one-cell lattice"
            )
        axes[name] = Cells.single(
            float(ref), float(scale), order=cap, unit=unit, support=(lo, hi)
        )
    in_total = sum(
        cap
        for name, cap in zip(_AXES, tr.caps())
        if name in axes and (name != "depth" or depth_in_total)
    )
    reps = {"mu": Legendre(tr.L_mu), "eta": Legendre(tr.L_eta)}
    if source.index.n2:
        reps["phi"] = AzimuthHarmonic()
    return Lattice(
        weights=(1.0,),
        order=min(tr.N, in_total),
        depth_in_total=depth_in_total,
        **axes,
        **reps,
    )


def _row_keys(lattice):
    """Legacy (l, k, r, s, b) keys in lattice row order."""
    arrays = lattice.index.arrays
    powers = np.zeros((lattice.index.n, len(_AXES)), dtype=np.int64)
    for k, name in enumerate(lattice.cells.variables):
        powers[:, _AXES.index(name)] = arrays.powers[:, k]
    return np.column_stack((arrays.degrees, powers))


def _row_indices(rows, source_rows):
    lookup = {row: k for k, row in enumerate(source_rows)}
    try:
        return np.array([lookup[tuple(row)] for row in rows], dtype=np.int64)
    except KeyError as exc:
        raise RoleError(
            "legacy moment index does not match the one-cell lattice"
        ) from exc


def _spectra(source, lattice, columns, scale):
    rows = _row_keys(lattice)
    spin0 = _row_indices(rows[columns.spin0_base], source.index.h0)
    spin2 = _row_indices(rows[columns.spin2_base], source.index.h2)
    i, v, p = (np.asarray(a) for a in (source.I_basis, source.V_basis, source.P_basis))
    arrays = {"I": (i, source.index.parity_I()), "V": (v, source.index.parity_V())}
    active0 = []
    for name, (arr, parity) in arrays.items():
        permitted = np.asarray(parity, dtype=bool) & (name in columns.stokes)
        if np.any(arr[:, ~permitted] != 0):
            raise RoleError(
                f"legacy {name} coefficients conflict with the stored parity or Stokes declaration"
            )
        if name in columns.stokes:
            active0.append(arr[:, spin0])
    s0 = np.stack(active0, axis=1) * scale[:, None, None]
    s2 = p[:, spin2] * scale[:, None] if "P" in columns.stokes else None
    return s0, s2


def from_single_reference(old_basis) -> SingleReferenceBasis:
    """Import a frozen ``syncem.model.SpectralBasis`` as a one-cell basis."""
    if not isinstance(old_basis, LegacyBasis):
        raise TypeError("old_basis must be a syncem.model.SpectralBasis")
    provenance = old_basis.provenance
    if getattr(provenance, "units", None) != LEGACY_UNITS:
        raise UnitError(
            "legacy basis must declare the frozen model's per-electron Stokes and affine coordinate units"
        )
    declared = dict(provenance.channels).get("normalisation")
    if declared != old_basis.channels.normalisation:
        raise UnitError(
            "legacy basis provenance must declare the same channel normalisation as its Channels"
        )

    channels, multiplier = _channels(old_basis.channels)
    lattice = _lattice(old_basis)
    stokes = ("I",) + (("P",) if old_basis.index.n2 else ())
    if "V" in old_basis.index.components:
        stokes += ("V",)
    parity = Parity(even=("I", "P") if old_basis.index.parity else ("I",))
    kernel = _StoredKernel(stokes, (parity,))
    rules = ColumnRules(
        masks=(("mu", "joint"), ("eta", "joint")), depth_free_spin0=True
    )
    max_columns = max(20_000, old_basis.index.n_real)
    columns = column_map(lattice, kernel, rules, max_columns=max_columns)
    s0, s2 = _spectra(old_basis, lattice, columns, multiplier)
    record = build_record(
        (),
        columns.stokes,
        "quadrature",
        columns,
        lattice,
        defaults={"max_columns": max_columns, "legacy_normalisation": declared},
        notes=(
            "Imported the stored single-reference Taylor response.",
            "No S1 or column provider ran during import.",
            "Legacy normalised moments need an explicit physical column.",
        ),
    )
    return SingleReferenceBasis(
        lattice=lattice,
        channels=channels,
        columns=columns,
        spectra0=s0,
        spectra2=s2,
        azimuth_factor=None,
        specified=(),
        record=record,
        ledger=Ledger(
            (
                ErrorTerm.unbounded(
                    "The legacy per-electron error budget needs a legacy prediction and its amplitude.",
                    name="legacy_error",
                ),
            ),
            assumptions=provenance.assumptions,
        ),
        key=None,
        averaged=AveragedKernel(kernel),
        rules=rules,
        legacy_provenance=provenance,
        legacy_reference=old_basis.reference.describe(),
    )
