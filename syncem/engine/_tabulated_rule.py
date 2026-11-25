"""The trace identity of an averaged rule and the fingerprint of a fixed value (design/0.8/SPEC.md decision 160).

LABEL ``[extension]``. ``TabulatedRule(rule, fingerprint)`` wraps a rule for the jet. With a fingerprint,
equality and hash are on the function identity (the ``NodeRule`` fields, or the callable object) and the
fingerprint, so ``eqx.filter_jit`` reuses a trace only for an equal chain. Without one, the wrapper carries a
fresh token and equals only itself. A ``TabulatedRule`` is not a ``NodeRule``, so ``with_tabulated_rules``
wraps the kernel passed to the jet routines after the stack routing. ``fixed_fingerprint(value)`` digests the
family "fixed" and the canonical float of a fixed value.

Units: those of the wrapped rule. Shapes: those of the wrapped rule. Not certified: the wrapper changes no
value and only the identity of a trace.
"""

from __future__ import annotations

import dataclasses
from typing import Any

from ..base._fingerprint import digest
from .average import AveragedKernel, NodeRule

__all__ = ["TabulatedRule", "fixed_fingerprint", "with_tabulated_rules"]


def fixed_fingerprint(value: Any) -> str:
    """Digest of the family "fixed" and the canonical float of a fixed value."""
    return digest(("fixed", float(value)))


@dataclasses.dataclass(frozen=True, eq=False)
class TabulatedRule:
    """A rule for the jet, equal to another by (function identity, fingerprint), or only to itself."""

    rule: Any
    fingerprint: str | None
    token: object = dataclasses.field(default_factory=object, repr=False)

    def __call__(self, point: dict) -> tuple[Any, Any]:
        return self.rule(point)

    @property
    def parents(self) -> tuple[str, ...] | None:
        return getattr(self.rule, "parents", None)

    @property
    def density(self) -> Any:
        return getattr(self.rule, "density", None)

    def _identity(self) -> Any:
        return self.rule if isinstance(self.rule, NodeRule) else id(self.rule)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, TabulatedRule):
            return NotImplemented
        if self.fingerprint is None or other.fingerprint is None:
            return self is other
        same = self._identity() == other._identity()
        return same and self.fingerprint == other.fingerprint

    def __hash__(self) -> int:
        if self.fingerprint is None:
            return hash(self.token)
        return hash((self.fingerprint, self._identity()))


def with_tabulated_rules(
    averaged: AveragedKernel, fingerprints: dict
) -> AveragedKernel:
    """``averaged`` with each rule wrapped as ``TabulatedRule(rule, fingerprints.get(name))``."""
    unwrap = [
        (n, r.rule if isinstance(r, TabulatedRule) else r) for n, r in averaged.averaged
    ]
    return AveragedKernel(
        averaged.kernel,
        tuple((n, TabulatedRule(r, fingerprints.get(n))) for n, r in unwrap),
        legendre={n: (degree, k) for n, degree, k in averaged.legendre},
        fixed=dict(averaged.fixed),
        phase=averaged.phase,
        outer_batch=averaged.outer_batch,
        ties=averaged.ties,
        leaf_factors=averaged.leaf_factors,
        legendre_panels=averaged.legendre_panels,
        diagonal=averaged.diagonal,
        emission_ties=averaged.emission_ties,
    )
