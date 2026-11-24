"""The 0.8 error ledger (SPEC Sec. 7.2 and decisions 149 and 175).

LABEL [extension]. A term's value is in channel Stokes
units, usually ``(..., n_ch, n_stokes)`` or broadcastable to that layout.
Measured terms may hold signed differences. Envelopes use their absolute
values and distinguish computed estimates from declared inputs.

Not certified. The ledger records the terms it is given and their kinds.

The Stokes envelope has shape (..., n_ch, n_stokes) with broadcastable
valued terms.
"""

from __future__ import annotations

from dataclasses import replace
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp

from syncem.model._provenance import AssumptionRecord, strict_json
from syncem.model.errors import KINDS, WEAKEST_ORDER
from syncem.model.errors import ErrorTerm as ModelErrorTerm

__all__ = ["Envelope", "ErrorTerm", "Ledger"]
_VALUELESS = ("unbounded", "not_applicable")
_PARTITION = ("representation", "outside_lattice")


class ErrorTerm(eqx.Module):
    """A named error term with an optional signed channel-space value."""

    LABEL: ClassVar[str] = "[extension] channel error ledger"
    value: jax.Array | None
    kind: str = eqx.field(static=True)
    note: str = eqx.field(static=True)
    manuscript_term: str = eqx.field(static=True)
    name: str = eqx.field(static=True)
    declared: bool = eqx.field(static=True)
    signed: bool = eqx.field(static=True)
    relative_to: str | None = eqx.field(static=True)
    space: str = eqx.field(static=True)

    def __init__(
        self,
        value,
        kind,
        note="",
        manuscript_term="",
        *,
        name="",
        declared=False,
        signed=None,
        relative_to=None,
        space="stokes",
    ):
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        if not all(isinstance(v, str) for v in (name, note, manuscript_term)):
            raise ValueError("error name, note and manuscript_term must be strings")
        if not isinstance(declared, bool):
            raise ValueError("declared must be a bool")
        if relative_to is not None and (
            not isinstance(relative_to, str) or not relative_to.strip()
        ):
            raise ValueError("relative_to must name the reference quantity")
        if space not in ("stokes", "retained"):
            raise ValueError("space must be stokes or retained")
        if signed is None:
            signed = kind == "measured" and not declared and name in _PARTITION
        if not isinstance(signed, bool) or (
            signed and (kind != "measured" or declared)
        ):
            raise ValueError("only computed measurements can hold signed values")
        if kind in _VALUELESS:
            if value is not None:
                raise ValueError(f"kind {kind!r} carries no value")
            if declared:
                raise ValueError(f"kind {kind!r} cannot be declared")
        else:
            if value is None:
                raise ValueError(f"kind {kind!r} requires a value")
            value = jnp.asarray(value)
            if jnp.iscomplexobj(value):
                raise ValueError("error values must be real")
            value = value.astype(jnp.result_type(value, 1.0))
            value = eqx.error_if(
                value, jnp.any(~jnp.isfinite(value)), "error values must be finite"
            )
            if not signed:
                value = eqx.error_if(
                    value, jnp.any(value < 0), "error magnitudes must be nonnegative"
                )
        self.value, self.kind, self.note = value, kind, note
        self.manuscript_term, self.name = manuscript_term, name
        self.declared, self.signed = declared, signed
        self.relative_to = relative_to
        self.space = space

    @classmethod
    def from_model(cls, name: str, term: ModelErrorTerm) -> ErrorTerm:
        """Record a kernel term using its existing magnitude and kind."""
        return cls(
            term.value,
            term.kind,
            term.note,
            term.manuscript_term,
            name=name,
            signed=False,
        )

    @classmethod
    def unbounded(cls, note="", manuscript_term="", *, name="") -> ErrorTerm:
        """Record missing information under the supplied name."""
        return cls(None, "unbounded", note, manuscript_term, name=name)

    @classmethod
    def not_applicable(cls, note="", *, name="") -> ErrorTerm:
        """Record a term that does not arise for the route."""
        return cls(None, "not_applicable", note, name=name)

    @classmethod
    def declared_zero(cls, reason, shape=(1, 4), *, name="") -> ErrorTerm:
        """Record a zero magnitude supplied by the user."""
        if not isinstance(reason, str) or not reason.strip():
            raise ValueError("a declared zero error requires a reason")
        return cls(jnp.zeros(shape), "bound", reason, name=name, declared=True)

    def scaled(self, factor) -> ErrorTerm:
        """Rescale a value by a nonnegative factor in the same channel layout."""
        if self.value is None or self.relative_to is not None or self.space != "stokes":
            return self
        factor = jnp.asarray(factor)
        factor = eqx.error_if(
            factor, jnp.any(factor < 0), "error scale must be nonnegative"
        )
        return replace(self, value=self.value * factor)

    def to_dict(self) -> dict:
        """Strict-JSON representation of a concrete term."""
        result = {
            "value": self.value,
            "kind": self.kind,
            "note": self.note,
            "manuscript_term": self.manuscript_term,
        }
        if self.name or self.declared or self.signed:
            result.update(name=self.name, declared=self.declared, signed=self.signed)
        if self.relative_to is not None:
            result["relative_to"] = self.relative_to
        if self.space != "stokes":
            result["space"] = self.space
        return strict_json(result)


class Envelope(eqx.Module):
    """Channel-wise sum of magnitudes and its weakest kind."""

    value: jax.Array
    kind: str = eqx.field(static=True)
    declared: tuple[str, ...] = eqx.field(static=True)

    @property
    def note(self) -> str:
        """Name the declarations on which this envelope depends."""
        return (
            "conditional on declared: " + ", ".join(self.declared)
            if self.declared
            else ""
        )


def _stokes_term(term):
    return term.relative_to is None and term.space == "stokes"


def _sum_magnitudes(terms) -> jax.Array:
    return sum(
        (jnp.abs(t.value) for t in terms if t.value is not None and _stokes_term(t)),
        jnp.asarray(0.0),
    )


class Ledger(eqx.Module):
    """Tuple of terms with unique names and the assumptions recorded by declarations."""

    LABEL: ClassVar[str] = "[extension] channel error ledger"
    terms: tuple[ErrorTerm, ...]
    assumptions: tuple[AssumptionRecord, ...] = eqx.field(static=True)

    def __init__(self, terms: tuple = (), *, assumptions: tuple = ()):
        if not isinstance(terms, tuple):
            raise ValueError(
                f"ledger terms must be a tuple, got {type(terms).__name__}"
            )
        converted = []
        names = set()
        for term in terms:
            if isinstance(term, ModelErrorTerm):
                # Step 5 accepted unnamed model terms. New terms can carry
                # unique names without giving those inputs invented names.
                term = ErrorTerm.from_model("", term)
            if not isinstance(term, ErrorTerm):
                raise ValueError(f"ledger terms must be ErrorTerm, got {term!r}")
            if term.name and term.name in names:
                raise ValueError(f"duplicate error term {term.name!r}")
            names.add(term.name)
            converted.append(term)
        self.terms = tuple(converted)
        if not isinstance(assumptions, tuple) or not all(
            isinstance(a, AssumptionRecord) for a in assumptions
        ):
            raise ValueError("ledger assumptions must be a tuple of AssumptionRecord")
        self.assumptions = assumptions

    def __getitem__(self, name: str) -> ErrorTerm:
        """Return the term under the supplied name."""
        for term in self.terms:
            if term.name and term.name == name:
                return term
        raise KeyError(name)

    def by_kind(self, kind: str) -> tuple[ErrorTerm, ...]:
        """Return computed terms with the supplied kind."""
        if kind not in KINDS:
            raise ValueError(f"kind must be one of {KINDS}, got {kind!r}")
        return tuple(t for t in self.terms if t.kind == kind and not t.declared)

    def sum(self, kind: str) -> jax.Array | None:
        """Sum magnitudes in a computed kind or return None for a valueless kind."""
        terms = self.by_kind(kind)
        return None if kind in _VALUELESS else _sum_magnitudes(terms)

    def unbounded(self) -> tuple[str, ...]:
        """Names of terms without a finite envelope."""
        return tuple(t.name for t in self.by_kind("unbounded"))

    def declared(self) -> tuple[ErrorTerm, ...]:
        """Terms supplied by the user."""
        return tuple(t for t in self.terms if t.declared)

    def envelope(self) -> Envelope | None:
        """Sum all magnitudes with their weakest kind and any declarations."""
        if any(_stokes_term(t) for t in self.by_kind("unbounded")):
            return None
        valued = tuple(t for t in self.terms if t.value is not None and _stokes_term(t))
        kind = max((t.kind for t in valued), key=WEAKEST_ORDER.index, default="bound")
        names = tuple(t.name for t in self.declared() if _stokes_term(t))
        return Envelope(_sum_magnitudes(valued), kind, names)

    def allow(
        self, name: str, term: ErrorTerm | ModelErrorTerm, *, note: str
    ) -> Ledger:
        """Replace a named unbounded term with a declaration and record its reason."""
        if not isinstance(note, str) or not note.strip():
            raise ValueError("a declared error requires a nonempty note")
        if self[name].kind != "unbounded":
            raise ValueError(f"only an unbounded term can be declared, got {name!r}")
        if isinstance(term, ModelErrorTerm):
            term = ErrorTerm.from_model(name, term)
        if not isinstance(term, ErrorTerm) or term.value is None:
            raise ValueError("a declaration requires a valued ErrorTerm")
        old = self[name]
        if (term.space, term.relative_to) != (old.space, old.relative_to):
            where = (
                "absolute Stokes"
                if _stokes_term(old)
                else f"{old.space} space relative to {old.relative_to}"
            )
            raise ValueError(
                f"{name!r} is an error in {where} units. Declare it in the same "
                "space and reference, or convert a relative error with the "
                "reference intensity first."
            )
        replacement = replace(term, name=name, note=note, declared=True, signed=False)
        record = AssumptionRecord(name, note, (), "declared error", (), 0, term.kind)
        terms = tuple(replacement if t.name == name else t for t in self.terms)
        return Ledger(terms, assumptions=(*self.assumptions, record))

    def assume_independence(self, names):
        """Record dropped conditional edges without assigning a source error."""
        existing = {term.name for term in self.terms}
        names = tuple(name for name in dict.fromkeys(names) if name not in existing)
        note = "dependency replaced by the specified parent marginal"
        terms = tuple(ErrorTerm.unbounded(note, name=name) for name in names)
        provenance = tuple(
            AssumptionRecord(name, note, (), "dropped dependency", (), 0, "unbounded")
            for name in names
        )
        return Ledger(
            (*self.terms, *terms), assumptions=(*self.assumptions, *provenance)
        )

    def scaled(self, factor) -> Ledger:
        """Apply a unit conversion to every valued term, preserving their signs."""
        return Ledger(
            tuple(t.scaled(factor) for t in self.terms), assumptions=self.assumptions
        )

    def through(self, response, source_shape, target_shape) -> Ledger:
        """Propagate signed differences by R and magnitudes by its absolute value."""
        abs_response = jnp.abs(response)

        def apply(term):
            if term.value is None or term.space != "stokes":
                return term
            if term.relative_to is not None:
                return ErrorTerm.unbounded(
                    f"response changes the reference for {term.relative_to}",
                    name=term.name,
                )
            shape = jnp.broadcast_shapes(term.value.shape, source_shape)
            if shape[-2:] != source_shape:
                raise ValueError(
                    f"error {term.name!r} does not match the input Stokes layout"
                )
            value = jnp.broadcast_to(term.value, shape)
            flat = value.reshape(*shape[:-2], response.shape[-1])
            operator = response if term.signed else abs_response
            projected = (flat @ operator.T).reshape(*flat.shape[:-1], *target_shape)
            return replace(term, value=projected)

        return Ledger(tuple(apply(t) for t in self.terms), assumptions=self.assumptions)

    def to_dict(self) -> dict:
        """Strict-JSON representation of concrete terms and any new assumptions."""
        result = {"terms": [t.to_dict() for t in self.terms]}
        if self.assumptions:
            result["assumptions"] = [r.to_dict() for r in self.assumptions]
        return result


def combine_ledgers(ledgers):
    """Combine terms for additive spectra and preserve every declared assumption."""
    ledgers = tuple(ledgers)
    if len(ledgers) == 1:
        return ledgers[0]
    parts = tuple({t.name: t for t in ledger.terms} for ledger in ledgers)
    terms = []
    for name in dict.fromkeys(key for part in parts for key in part):
        entries = [part.get(name) for part in parts]
        applicable = [t for t in entries if t is None or t.kind != "not_applicable"]
        if not applicable:
            terms.append(entries[0])
            continue
        if any(t is None or t.value is None for t in applicable):
            terms.append(
                ErrorTerm.unbounded("a component lacks a finite term", name=name)
            )
            continue
        weakest = max(applicable, key=lambda t: WEAKEST_ORDER.index(t.kind))
        relative = weakest.relative_to
        if any(t.space != "stokes" or t.relative_to != relative for t in applicable):
            terms.append(
                ErrorTerm.unbounded("components use different references", name=name)
            )
            continue
        signed = all(t.signed for t in applicable)
        values = [t.value if signed else jnp.abs(t.value) for t in applicable]
        try:
            if relative is None:
                value = sum(values)
            else:
                value = jnp.max(jnp.stack(jnp.broadcast_arrays(*values)), axis=0)
                relative = f"sum of component magnitudes of {relative}"
        except (TypeError, ValueError):
            terms.append(
                ErrorTerm.unbounded(
                    "components use different Stokes layouts", name=name
                )
            )
            continue
        declared = [t.declared for t in applicable]
        note = weakest.note
        if any(declared) and not all(declared):
            note = "sum of computed and user-declared terms"
        terms.append(
            replace(
                weakest,
                value=value,
                signed=signed,
                relative_to=relative,
                # A total that includes a declaration depends on user input.
                declared=any(declared),
                note=note,
            )
        )
    assumptions = tuple(
        dict.fromkeys(a for ledger in ledgers for a in ledger.assumptions)
    )
    return Ledger(tuple(terms), assumptions=assumptions)
