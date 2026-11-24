"""Per-channel records for a basis and a reselected response grid."""

from dataclasses import replace

import numpy as np

from ..base.continuum import ContinuumKernel
from ..base.depth import depth_bound
from ..base.window import continuum_reference_outside
from ..lattice._lattice_parts import REFERENCE_SLOPE
from ..records.ledger import ErrorTerm, Ledger


def depth_term(lattice, kernel, phase, channels, depth_tol):
    bound = depth_bound(lattice, kernel.depth, phase, channels, depth_tol)
    if bound is None:
        return None
    value, support = bound
    return ErrorTerm(
        value,
        "bound",
        f"affine depth phase remainder over the {support}",
        name="depth_truncation",
        relative_to="band-integrated polarized amplitude before the depth phase",
    )


def channel_ledger(basis, channels, rows):
    """Gather terms over stored rows and recompute model terms for new rows."""
    stored = np.all(rows >= 0)
    replacement = None
    if not stored and basis.forward_depth_tol is not None:
        replacement = depth_term(
            basis.lattice,
            basis.averaged.kernel,
            basis.averaged.phase,
            channels,
            basis.forward_depth_tol,
        )
    terms = []
    for term in basis.ledger.terms:
        if not stored and term.name == "outside_lattice_reference":
            continue
        current = term
        value = term.value
        if term.name == "depth_truncation" and replacement is not None:
            current = replacement
        elif term.name == "physical_kernel" and not stored:
            current = ErrorTerm.from_model(
                term.name, basis.averaged.kernel.physical_error(channels)
            )
        elif (
            value is not None
            and value.ndim >= 2
            and value.shape[-2] == basis.channels.n_ch
        ):
            current = (
                replace(term, value=np.take(np.asarray(value), rows, axis=-2))
                if stored
                else ErrorTerm.unbounded(
                    "reevaluation needs the original declaration", name=term.name
                )
            )
        terms.append(current)
    if not stored:
        tail = _reference_tail(basis, channels)
        if tail is not None:
            terms.append(tail)
    return Ledger(tuple(terms), assumptions=basis.ledger.assumptions)


def coverage_ledger(basis):
    """Recompute the coverage estimate of ``basis`` on its own lattice and channels."""
    name = "outside_lattice_reference"
    if all(term.name != name for term in basis.ledger.terms):
        return basis.ledger
    terms = tuple(term for term in basis.ledger.terms if term.name != name)
    tail = _reference_tail(basis, basis.channels)
    if tail is None:
        tail = ErrorTerm.unbounded(
            "the coverage estimate was not recomputed for the selected lattice",
            name=name,
        )
    return Ledger((*terms, tail), assumptions=basis.ledger.assumptions)


def _reference_tail(basis, channels):
    averaged, lattice = basis.averaged, basis.lattice
    kernel = averaged.kernel
    if (
        type(kernel) is not ContinuumKernel
        or lattice.cells.variables != ("gamma",)
        or averaged.ties
        or averaged.leaf_factors
        or averaged.legendre
        or "I" not in basis.record.stokes
    ):
        return None
    fixed, rules = dict(averaged.fixed), dict(averaged.averaged)

    def rule(name):
        if name in fixed:
            return np.atleast_1d(fixed[name]), np.ones(1)
        if name in rules and rules[name].parents == ():
            return tuple(np.asarray(x) for x in rules[name]({}))
        return None

    field, eta = rule("B"), rule("eta")
    if field is None or eta is None:
        return None
    mask = np.abs(eta[0]) < 1
    weights = eta[1][mask]
    if not np.any(mask) or weights.sum() <= 0:
        return None
    cells = tuple(zip(lattice.cells.lo[:, 0], lattice.cells.hi[:, 0]))
    share = continuum_reference_outside(
        channels,
        cells,
        kernel.gamma_min,
        field,
        (eta[0][mask], weights / weights.sum()),
        density=REFERENCE_SLOPE,
    )
    if share is None:
        return None
    return ErrorTerm(
        share[:, None],
        "estimate",
        "estimate for the reference density. Use predict= or a wider lattice when refitting",
        name="outside_lattice_reference",
        relative_to="reference power-law Stokes I",
    )
