"""Compare an explicitly marginalised population with its original joint law."""

from dataclasses import replace

from ..records.ledger import ErrorTerm, Ledger
from ..values.errors import RoleError, SizeError
from ._accuracy import check_policy, estimate_direct
from ._planning import plan
from ._direct import direct
from ._independence_scaling import scaling_difference


def _ordinary_difference(result, basis, original, name, entries, budget):
    residual = tuple(
        entries.get(name) for name in ("representation", "outside_lattice")
    )
    if any(term is None or term.value is None for term in residual):
        return None
    parent = name.split("|", 1)[-1]
    kernel, channels = basis.averaged.kernel, basis.channels
    try:
        layout = plan(
            original,
            channels,
            kernel=kernel,
            phase=basis.forward_phase,
            resolve=(parent,),
        )
        reference = replace(basis, lattice=layout.lattice)
        if estimate_direct(reference) + estimate_direct(basis) > budget:
            return None
        truth = direct(reference, original).stokes
    except (RoleError, SizeError):
        return None
    difference = result.stokes - sum(term.value for term in residual) - truth
    for term in entries.values():
        if (
            term.name == "azimuth_truncation" or term.name.startswith("specified:")
        ) and term.signed:
            difference -= term.value
    return difference


def independence_check(result, basis, original, names, check):
    """Measure a single dropped edge when both direct comparisons are available."""
    ledger = result.ledger.assume_independence(names)
    policy = check_policy(check)
    if policy is None or len(names) != 1:
        return replace(result, ledger=ledger)
    name = names[0]
    entries = {term.name: term for term in ledger.terms}
    difference = _ordinary_difference(
        result, basis, original, name, entries, policy.max_seconds
    )
    note = "direct marginalised spectrum minus direct joint spectrum"
    if difference is None:
        difference = scaling_difference(basis, original, name, policy.max_seconds)
        note += " through the checked fixed-field scaling table"
    if difference is None:
        return replace(result, ledger=ledger)
    measured = ErrorTerm(
        difference,
        "measured",
        note,
        name=name,
        signed=True,
    )
    terms = tuple(
        measured if term.name == measured.name else term for term in ledger.terms
    )
    return replace(result, ledger=Ledger(terms, assumptions=ledger.assumptions))
