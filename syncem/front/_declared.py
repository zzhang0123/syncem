"""Explicit kernel and phase declarations against a supplied basis (SPEC Sec. 3.4)."""

from __future__ import annotations

import equinox as eqx
import numpy as np

from ..base._fingerprint import digest
from ..base.harmonic import HarmonicKernel
from ..base.hybrid import HybridKernel
from ..base.phase import FaradayRotation
from ..engine._keys import kernel_record
from ..engine.basis_sum import BasisSum
from ..values.errors import RoleError

__all__ = ["check_azimuth_declared", "check_declared", "effective_phase", "same_kernel"]


def same_kernel(left, right) -> bool:
    """Whether two declarations describe the same kernel on the basis cells."""
    if left is right:
        return True
    if type(left) is not type(right):
        return False
    if type(left) is HarmonicKernel:
        return left == right
    if type(left) is HybridKernel:
        return (
            left.switch == right.switch
            and left.switch_tol == right.switch_tol
            and left.geometry == right.geometry
            and left.assignment == right.assignment
            and len(left.kernels) == len(right.kernels)
            and all(same_kernel(a, b) for a, b in zip(left.kernels, right.kernels))
        )
    if not hasattr(left, "describe"):
        return False
    first, second = kernel_record(left), kernel_record(right)
    return first is not None and second is not None and digest(first) == digest(second)


def effective_phase(phase, kernel):
    """Resolve the default Faraday law of a kernel with a depth variable."""
    if phase is None and getattr(kernel, "depth", None) is not None:
        return FaradayRotation()
    return phase


def _joint(part):
    return getattr(part, "joint", part)


def check_declared(basis, *, kernel=None, phase=None) -> None:
    """Refuse a kernel or phase that differs from the one the basis was built with.

    A basis built in this process records its phase. A loaded or imported basis
    does not, so an explicit phase cannot be verified and is refused.
    """
    if kernel is None and phase is None:
        return
    parts = basis.parts if isinstance(basis, BasisSum) else (basis,)
    for part in map(_joint, parts):
        model = getattr(part, "source_kernel", None)
        model = part.averaged.kernel if model is None else model
        requested = kernel
        if isinstance(model, HybridKernel) and isinstance(kernel, HybridKernel):
            requested = kernel.bind(part.lattice.cells)
        if requested is not None and not same_kernel(model, requested):
            raise RoleError(
                "kernel= differs from the kernel of the supplied basis. "
                "Rebuild the basis with this kernel, or omit kernel=."
            )
        if phase is None:
            continue
        if getattr(part, "forward_population", None) is None:
            raise RoleError(
                "the supplied basis does not record the phase it was built with. "
                "Rebuild the basis with phase=, or omit phase=."
            )
        stored = effective_phase(part.forward_phase, model)
        if not eqx.tree_equal(stored, effective_phase(phase, model), typematch=True):
            raise RoleError(
                "phase= differs from the phase of the supplied basis. "
                "Rebuild the basis with this phase, or omit phase=."
            )


def _observes_polarisation(data) -> bool:
    linear = [name in ("Q", "U") for name in data.components]
    if not any(linear):
        return False
    model_rows = np.tile(linear, data.channels.n_ch)
    if data.response is None:
        polarised = model_rows
    else:
        polarised = np.any(np.asarray(data.response)[:, model_rows] != 0, axis=1)
    return bool(np.any(np.asarray(data.mask) & polarised))


def _declares(population, name) -> bool:
    if name in population.factors:
        return True
    return any(name in getattr(block, "variables", ()) for block in population.blocks)


def check_azimuth_declared(population, data, kernel) -> None:
    """Refuse a fit of Q or U data whose population leaves the azimuth undeclared.

    The forward default (a uniform azimuth, Q = U = 0) reads as a result in a
    polarisation fit (SPEC decision 7).
    """
    azimuth = getattr(kernel, "azimuth", None)
    if azimuth is None or not _observes_polarisation(data):
        return
    parts = population.parts if population.components else (population,)
    for part in parts:
        if not _declares(part, azimuth):
            raise RoleError(
                f"a fit of Q or U data needs the azimuth {azimuth!r} declared, "
                f"for example {azimuth}=se.UniformAzimuth() or its law "
                "(SPEC decision 7)"
            )
