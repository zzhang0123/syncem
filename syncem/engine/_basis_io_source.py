"""Source compatibility and stored-only basis behavior for the basis archive."""

from __future__ import annotations

from collections.abc import Mapping
import functools

import equinox as eqx

from ..base._fingerprint import NotCacheable, digest
from ..base._function_fingerprint import function_fingerprint
from ..base.capabilities import Parity
from ..base.hybrid import HybridKernel
from ..lattice.moments import Moments
from ..model.errors import ErrorTerm as ModelErrorTerm
from ..values.errors import DataError
from ._keys import kernel_record
from .average import AveragedKernel
from .basis import SpectralBasis
from .single_reference import _StoredKernel as _LegacyKernel

_REBUILD = (
    "this loaded basis has only stored spectra. Supply its compatible AveragedKernel "
    "as averaged= to SpectralBasis.load, or register it with registry=."
)
# Standard field names on a Kernel whose spectra are already evaluated.
_KERNEL_FIELDS = ("name", "variables", "primary", "stokes", "azimuth", "depth", "lines")


class _StoredKernel(eqx.Module):
    """A kernel descriptor for stored spectra, with no live forward functions."""

    name: str = eqx.field(static=True)
    variables: tuple = eqx.field(static=True)
    primary: str | None = eqx.field(static=True)
    stokes: tuple = eqx.field(static=True)
    azimuth: str | None = eqx.field(static=True)
    depth: str | None = eqx.field(static=True)
    lines: bool = eqx.field(static=True)
    source_key: str | None = eqx.field(static=True)
    stored_only = True
    traceable = False
    capabilities = ()

    def channel(self, *args, **kwargs):
        """Require a live source declaration to evaluate a new channel."""
        raise DataError(_REBUILD)

    spectral = channel
    regime = channel

    def physical_error(self, channels):
        """Record the absence of a usable kernel-side bound for new channels."""
        return ModelErrorTerm.unbounded(_REBUILD)

    def describe(self):
        """The saved source key is provenance for this stored descriptor."""
        return (("name", self.name), ("stored source", self.source_key))


def _identity(value):
    return value


def hybrid_child_keys(kernel):
    """Fingerprint all child declarations, including numerical kernel state."""
    keys = []
    for child in kernel.kernels:
        record = kernel_record(child)
        try:
            state = function_fingerprint(functools.partial(_identity, child))
        except NotCacheable:
            state = None
        keys.append(
            None if record is None or state is None else digest((record, state))
        )
    return tuple(keys)


def basis_source_key(averaged):
    """Fingerprint a live kernel and all averaged rule state when possible."""
    if not isinstance(averaged, AveragedKernel):
        raise TypeError("a basis source must be an AveragedKernel.")
    kernel = averaged.kernel
    if isinstance(kernel, _StoredKernel):
        return kernel.source_key
    if isinstance(kernel, _LegacyKernel):
        return None
    try:
        record = kernel_record(kernel)
        if record is None:
            return None
        if isinstance(kernel, HybridKernel):
            # The bound assignment identifies the selected kernels. Its
            # originating classifier need not be serializable.
            child_keys = hybrid_child_keys(kernel)
            if not kernel.assignment or None in child_keys:
                return None
            average_state = eqx.tree_at(lambda a: a.kernel, averaged, None)
            state = function_fingerprint(functools.partial(_identity, average_state))
            return digest(
                ("hybrid basis source", record, child_keys, kernel.switch_tol, state)
            )
        state = function_fingerprint(functools.partial(_identity, averaged))
        return digest(("basis source", record, state))
    except NotCacheable:
        return None


def require_rebuild(basis):
    """Refuse a route that needs live declarations on a stored-only basis."""
    if isinstance(basis.averaged.kernel, (_StoredKernel, _LegacyKernel)):
        raise DataError(_REBUILD)


class ArchivedBasis(SpectralBasis):
    """Loaded columns. Population replay needs the original forward declaration."""

    reference_names: tuple[str, ...] = eqx.field(static=True, kw_only=True)

    def at(self, channels):
        """Select saved rows, or use a verified live source for new rows."""
        if isinstance(self.averaged.kernel, _StoredKernel):
            known = set(self.channels.row_keys())
            if any(key not in known for key in channels.row_keys()):
                require_rebuild(self)
        return super().at(channels)

    def with_order(self, order):
        """Change Taylor order with the live source and its reference weights."""
        require_rebuild(self)
        if set(self.reference_names) != {name for name, _ in self.lattice._references}:
            raise DataError(
                "changing order needs the original lattice reference density declarations."
            )
        return super().with_order(order)

    def moments(self, population, *, allow_mismatch=False):
        """Check stored-coordinate Moments against the archived lattice."""
        if not isinstance(population, Moments):
            raise DataError(
                "predict a loaded basis from supplied Moments. "
                "Build from the original population and phase declaration to pair a population."
            )
        return super().moments(population, allow_mismatch=allow_mismatch)

    def direct(self, population, *, within="support"):
        """Use live forward declarations for a separate direct quadrature."""
        raise DataError(
            "direct integration needs the original population and phase declaration. "
            "Rebuild the basis from those inputs."
        )


def _source(body, averaged, registry):
    saved = body["source_key"]
    if registry is not None:
        if not isinstance(registry, Mapping):
            raise TypeError(
                "registry must map source keys to AveragedKernel instances."
            )
        if averaged is not None:
            raise TypeError("pass either averaged= or registry=.")
        averaged = None if saved is None else registry.get(saved)
    legacy = body["legacy"]
    if legacy is not None:
        if averaged is not None:
            raise DataError(
                "an imported single-reference basis supports only its stored response."
            )
        parity = tuple(Parity(*p) for p in legacy["parity"])
        return AveragedKernel(_LegacyKernel(body["kernel"]["stokes"], parity))
    if averaged is None:
        return AveragedKernel(_StoredKernel(**body["kernel"], source_key=saved))
    hybrid = body.get("hybrid")
    if hybrid is not None and isinstance(averaged, AveragedKernel):
        kernel = averaged.kernel
        if isinstance(kernel, HybridKernel):
            if (
                kernel.switch != hybrid["switch"]
                or kernel.switch_tol != hybrid["switch_tol"]
                or hybrid_child_keys(kernel) != hybrid["child_keys"]
            ):
                raise DataError(
                    "the supplied hybrid assignment or child sources are not compatible with the archive."
                )
            kernel = kernel.bind(body["lattice"]["cells"])
            if kernel.assignment != hybrid["assignment"]:
                raise DataError(
                    "the supplied hybrid assignment is not compatible with the archive."
                )
            averaged = eqx.tree_at(lambda a: a.kernel, averaged, kernel)
    if (
        not isinstance(averaged, AveragedKernel)
        or isinstance(averaged.kernel, _StoredKernel)
        or saved is None
        or basis_source_key(averaged) != saved
    ):
        raise DataError(
            "the supplied averaged kernel is not compatible with the archived basis source."
        )
    return averaged
