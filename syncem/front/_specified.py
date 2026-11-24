"""Ordered direct comparisons of a basis's assumed population statistics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Number

import jax
import numpy as np

from ..population.declaration import Population, ordered
from ..records.ledger import ErrorTerm, Ledger
from ..units import Quantity

__all__ = ["comparison_plan", "replace_terms", "specified_terms"]

_DEFAULT = "The basis assumes this statistic and its declared parent dependence."
_MEASURED = (
    "Direct Stokes shift from replacing this statistic in dependency order. "
    "The remaining shifts account for the other replacements."
)


def _names(basis):
    lattice = basis.lattice
    resolved = set(lattice.cells.variables) | set(lattice.representations)
    source = basis.forward_population
    declared = (
        set(source.factors)
        if source is not None
        else {record.variable for record in basis.specified}
    )
    return tuple(
        variable.name
        for variable in basis.averaged.kernel.variables
        if variable.name in declared - resolved
    )


def _terms(names, *, measured=None, note=_DEFAULT):
    measured = {} if measured is None else measured
    return tuple(
        ErrorTerm(
            measured[name],
            "measured",
            _MEASURED,
            name=f"specified:{name}",
            signed=True,
        )
        if name in measured
        else ErrorTerm.unbounded(note, name=f"specified:{name}")
        for name in names
    )


def specified_terms(basis):
    """The assumed-statistic terms when no replacement has been evaluated."""
    return _terms(_names(basis))


def replace_terms(ledger, terms):
    """Replace existing managed terms without duplicate ledger names."""
    replacement = {term.name: term for term in terms}
    kept = tuple(term for term in ledger.terms if term.name not in replacement)
    return Ledger((*kept, *replacement.values()), assumptions=ledger.assumptions)


@dataclass(frozen=True, eq=False)
class _PartComparison:
    baseline: Population
    replacements: tuple[tuple[str, Population], ...]


@dataclass(frozen=True, eq=False)
class _Comparison:
    names: tuple[str, ...]
    parts: tuple[_PartComparison, ...]
    unavailable: str | None = None

    @property
    def baseline_parts(self):
        return tuple(part.baseline for part in self.parts)

    @property
    def changed(self):
        return tuple(
            dict.fromkeys(name for part in self.parts for name, _ in part.replacements)
        )

    def evaluation_count(self, *, projected=False):
        """Count both disjoint baseline panels, the ladder and any projection."""
        return (2 + int(projected)) * len(self.parts) + sum(
            len(part.replacements) for part in self.parts
        )

    def unbounded(self, note=None):
        if self.unavailable or note is None:
            return _terms(self.names, note=self.unavailable or _DEFAULT)
        changed = set(self.changed)
        return tuple(
            ErrorTerm.unbounded(
                note if name in changed else _DEFAULT, name=f"specified:{name}"
            )
            for name in self.names
        )

    def measure(self, baseline_support, evaluate_support):
        """Measure per-statistic shifts from the physical in-cell plus outside totals.

        The callback evaluates the supplied leaf on its full support using
        the same direct kernel. The baseline totals use the already computed
        covered and uncovered panels so the signed partition telescopes.
        """
        if self.unavailable is not None:
            return self.unbounded()
        changes = {}
        for part, previous in zip(self.parts, baseline_support, strict=True):
            for name, candidate in part.replacements:
                current = evaluate_support(candidate)
                changes[name] = changes.get(name, 0) + (previous - current)
                previous = current
        return _terms(self.names, measured=changes)


def _argument(declaration, variable):
    # A stored point is already canonical. Restore its unit before placement.
    if variable.unit and isinstance(declaration, (Number, np.ndarray, jax.Array)):
        return Quantity(declaration, variable.unit)
    return declaration


def _same(left, right):
    return bool(np.all(left == right))


def _ladder(part, supplied, source, names, definitions, record):
    order = ordered(
        supplied.factors,
        available=tuple(name for block in supplied.blocks for name in block.variables),
    )
    mismatches = tuple(
        name
        for name, target in order
        if name in names and not _same(record(source.factors[name]), record(target))
    )
    basis_statistics = {
        name: _argument(source.factors[name], definitions[name]) for name in mismatches
    }
    baseline = part.replace(**basis_statistics) if mismatches else part
    current = baseline
    replacements = []
    for name in mismatches:
        declaration = (
            part.factors[name]
            if name in part.factors
            else _argument(supplied.factors[name], definitions[name])
        )
        current = current.replace(**{name: declaration})
        replacements.append((name, current))
    return _PartComparison(baseline, tuple(replacements))


def comparison_plan(basis, population, *, place, record):
    """Keep resolved declarations while substituting the averaged basis statistics.

    The target dependency order avoids introducing a cycle when dependencies
    change. The plan retains each supplied population's joint blocks and
    physical scale.
    """
    names = _names(basis)
    if any(
        isinstance(leaf, jax.core.Tracer)
        for leaf in jax.tree.leaves((population, basis.forward_population))
    ):
        return _Comparison(names, (), "unmeasured (traced)")
    source = basis.forward_population
    if not isinstance(population, Population):
        return _Comparison(names, (), "unmeasured (requires a specified population)")
    if not names:
        return _Comparison(
            names, tuple(_PartComparison(part, ()) for part in population.parts)
        )
    if source is None:
        return _Comparison(names, (), "unmeasured (basis declarations are unavailable)")
    kernel = basis.averaged.kernel
    definitions = {variable.name: variable for variable in kernel.variables}
    compared = tuple((part, place(part, kernel)[0]) for part in population.parts)
    if any(set(names) - set(supplied.factors) for _, supplied in compared):
        return _Comparison(
            names, (), "unmeasured (an averaged statistic is part of a joint block)"
        )
    parts = tuple(
        _ladder(part, supplied, source, names, definitions, record)
        for part, supplied in compared
    )
    return _Comparison(names, parts)
