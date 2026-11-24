"""Keys, memo entries and S2 entries of one basis build (design/0.8/SPEC.md decisions 157 to 162).

LABEL ``eq: channel basis expansion``. ``BuildRun.open`` prepares one build of ``build_from_averaged``. With
``cache=False`` it computes no key, gives the stack a ``local_memo()`` (decision 167) and the jet fresh
routines, and stores nothing in the process cache. Otherwise it computes the kernel record, the averaged-kernel record, the phase fingerprints,
the channel row keys and the rule chains ``d_0 .. d_N`` of every averaged rule on every cell, with ``N`` the
largest total degree of the cell's exponents. The basis memo key (``_keys.memo_key``) hashes them with the
column-map key, the provider settings and the engine source digest. Each reason without a fingerprint gives
the note "not cached: <reason>" and the key ``None``, and such a build reads and stores nothing in the cache. Its
stack also reads through a ``local_memo()``. ``finish`` adds the note of decision 168 when one-node tables were
not kept.

A cacheable build reads the memo first. On a miss the stack reads its S0 and S1 tables through a ``StageMemo``
and the jet reads the rules wrapped as ``TabulatedRule`` with the fingerprint of each rule, the digest of its
``d_N`` on every cell. Its routines come from ``Cache.routine`` with a token of the kernel record, the phase
fingerprints and the wrapped rules. ``BuildRun.reuse`` hands the cache, the keys, the stack memo and the jet
to ``_s2_columns.reuse_cells``, which reads and stores the S2 entries column by column. The memo entry holds
``spectra0``, ``spectra2``, the column map, the specified records, the build record and the build keys.
``derived_reuse`` gives ``at`` the S2 store of a cached basis on other channels.

Units: those of the kernel per unit scaled moment. Shapes: S2 values ``(n_ch, n_cell_rows)`` per (cell,
block). Not certified: a key is as complete as the fingerprints it hashes, and a rule chain identifies a rule
by finitely many values and coefficients (decision 157).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from ..base._fingerprint import NotCacheable, digest
from ..base.phase import phase_fingerprints
from ._jet_routine import jet_max_order
from ._keys import ENGINE_SOURCE, averaged_record, kernel_record_with_reason, memo_key
from ._keys import design_key as _design_key
from ._routes import JetSetup
from ._rule_fingerprint import chain_with_reason, with_tabulated_rules
from ._s2_columns import Reuse
from ._stage_memo import StageMemo, local_memo
from .cache import Cache, default_cache
from .field_average import TABLE_ORDER
from .provider_jet import make_routine

__all__ = [
    "HIT_EVENTS",
    "BuildKeys",
    "BuildRun",
    "build_keys",
    "cell_orders",
    "derived_reuse",
    "fresh_jet",
]

_NO_KEY = "not cached: lattice key is None"
_OFF = "not cached: cache=False"
HIT_EVENTS = {
    "memo hits": 1,
    "memo misses": 0,
    "S2 columns computed": 0,
    "S2 columns reused": 0,
    "S2 columns stored": 0,
    "channel rows computed": 0,
}


@dataclasses.dataclass(frozen=True)
class BuildKeys:
    """The fingerprints and the memo key of one build. ``memo`` is ``None`` when ``reasons`` is not empty."""

    memo: str | None
    reasons: tuple[str, ...]
    kernel: tuple | None
    averaged: tuple
    phase: tuple[str, ...] | None
    row_keys: tuple[str, ...]
    chains: dict
    fingerprints: dict
    geometry: tuple[str, ...]
    rules: bytes
    orders: tuple[int, ...]


def cell_orders(lattice: Any) -> tuple[int, ...]:
    """The largest total degree of the exponents of each cell, over every celled variable."""
    index = lattice.index
    return tuple(
        max(sum(e) for e in index.patterns[int(index.cell_pattern[c])])
        for c in range(lattice.n_cells)
    )


def _chains(averaged: Any, lattice: Any, orders: tuple) -> tuple[dict, dict, list]:
    """Chains per rule and cell, the fingerprint of each rule (or ``None``) and the reasons."""
    chains, prints, reasons = {}, {}, []
    max_order = max(jet_max_order(), TABLE_ORDER)
    if averaged.averaged and max(orders, default=0) > max_order:
        return (
            chains,
            prints,
            ["not cached: rule order exceeds the jet and stack ceilings"],
        )
    for name, rule in averaged.averaged:
        found, why = [], None
        for cell, order in enumerate(orders):
            chain, why = chain_with_reason(rule, name, averaged, lattice, cell, order)
            if chain is None:
                break
            found.append(chain)
        if why is not None:
            reasons.append(why)
            chains[name], prints[name] = None, None
            continue
        chains[name] = tuple(found)
        prints[name] = digest(("rule", name, tuple(c[-1] for c in found)))
    return chains, prints, reasons


def _kernel(kernel: Any) -> tuple[tuple | None, list[str]]:
    record, why = kernel_record_with_reason(kernel)
    if record is not None:
        return record, []
    return None, list(why) or ["not cached: the kernel record has no encoding"]


def _phase(averaged: Any, channels: Any) -> tuple[tuple | None, list[str]]:
    try:
        return phase_fingerprints(averaged.phase, channels), []
    except NotCacheable as err:
        return None, [f"not cached: the phase law has no fingerprint: {err.reason}"]


def build_keys(
    averaged: Any, channels: Any, lattice: Any, columns: Any, rules: Any, provider: Any
) -> BuildKeys:
    """The fingerprints of a build and its memo key, ``None`` with the reasons when one is missing."""
    reasons = [] if columns.key is not None else [_NO_KEY]
    kernel, why_kernel = _kernel(averaged.kernel)
    phase, why_phase = _phase(averaged, channels)
    orders = cell_orders(lattice)
    chains, prints, why_rules = _chains(averaged, lattice, orders)
    reasons += why_kernel + why_phase + why_rules
    record, row_keys = averaged_record(averaged), channels.row_keys()
    memo = None
    if not reasons:
        try:
            memo = memo_key(
                column_map=columns.key,
                kernel=kernel,
                averaged=record,
                phase=phase,
                rules=prints,
                row_keys=row_keys,
                provider=provider,
                engine=ENGINE_SOURCE,
            )
        except NotCacheable as err:
            reasons.append(
                f"not cached: the provider settings have no encoding: {err.reason}"
            )
    geometry = lattice.cells.geometry_digests()
    parts = (
        kernel,
        record,
        phase,
        row_keys,
        chains,
        prints,
        geometry,
        rules.record(),
        orders,
    )
    return BuildKeys(memo, tuple(reasons), *parts)


def _routines(cache: Cache | None, token: Any) -> Any:
    """Routines from ``cache`` for ``token``, or fresh routines of this build when ``cache`` is ``None``."""
    if cache is not None:
        return lambda n, c, e: cache.routine(n, c, e, token, make=make_routine)
    made: dict = {}

    def fresh(n: tuple, c: tuple, e: tuple) -> Any:
        if (n, c, e) not in made:
            made[(n, c, e)] = make_routine(n, c, e)
        return made[(n, c, e)]

    return fresh


def fresh_jet(averaged: Any, fingerprints: dict) -> JetSetup:
    """The jet with the rules wrapped as ``TabulatedRule`` and routines of its own, shared with no other build."""
    return JetSetup(with_tabulated_rules(averaged, fingerprints), _routines(None, None))


def _jet_setup(averaged: Any, keys: BuildKeys, cache: Cache) -> JetSetup:
    """The wrapped rules and the routines of ``cache`` under the trace token of the kernel, phase and rules."""
    wrapped = with_tabulated_rules(averaged, keys.fingerprints)
    token = (digest(("trace", keys.kernel, keys.phase)), wrapped.averaged)
    return JetSetup(wrapped, _routines(cache, token))


@dataclasses.dataclass
class BuildRun:
    """One build: its cache (``None`` when nothing is read or stored), keys, stack memo, jet setup and notes."""

    cache: Cache | None
    keys: BuildKeys | None
    memo: Any
    jet: JetSetup
    notes: tuple[str, ...]

    @classmethod
    def open(
        cls, averaged, channels, lattice, columns, rules, routes, s1, cache
    ) -> "BuildRun":
        if not cache:
            return cls(None, None, local_memo(), fresh_jet(averaged, {}), (_OFF,))
        provider = (("routes", routes), ("s1", s1), ("jet max order", jet_max_order()))
        keys = build_keys(averaged, channels, lattice, columns, rules, provider)
        if keys.memo is None:
            jet = fresh_jet(averaged, keys.fingerprints)
            return cls(None, keys, local_memo(), jet, keys.reasons)
        store = default_cache()
        jet = _jet_setup(averaged, keys, store)
        return cls(store, keys, StageMemo(store), jet, ())

    @property
    def memo_key(self) -> str | None:
        return None if self.keys is None else self.keys.memo

    @property
    def fingerprints(self) -> dict:
        return {} if self.keys is None else self.keys.fingerprints

    @property
    def reuse(self) -> Reuse | None:
        """The S2 store of this build, or ``None`` when it reads and stores nothing."""
        if self.cache is None:
            return None
        return Reuse(self.cache, self.keys, self.memo, self.jet)

    def lookup(self) -> Any:
        """The memo entry of this build, or ``None``."""
        return None if self.cache is None else self.cache.get("memo", self.keys.memo)

    def design_key(self, lattice: Any, factor: Any) -> str | None:
        return _design_key(self.memo_key, lattice.key, factor)

    def finish(self) -> dict:
        """Close the stack memo. The memo and cache events of the build, and its note on one-node tables."""
        memo = self.memo
        counts = {"memo misses": int(self.cache is not None), **self.close()}
        note = memo.note() if isinstance(memo, StageMemo) else None
        if note is not None:
            self.notes = (*self.notes, note)
        return counts

    def close(self) -> dict:
        """Close the stack memo once. Its event counts, or none for a dict memo."""
        memo, self.memo = self.memo, {}
        return memo.close() if isinstance(memo, StageMemo) else {}

    def store(self, basis: Any) -> None:
        """Put the memo entry of ``basis`` with the build keys, which ``at`` reads."""
        if self.cache is None:
            return
        entry = {
            "spectra0": np.asarray(basis.spectra0),
            "spectra2": None if basis.spectra2 is None else np.asarray(basis.spectra2),
            "columns": basis.columns,
            "specified": basis.specified,
            "record": basis.record,
            "keys": self.keys,
        }
        self.cache.put("memo", self.keys.memo, entry)


def derived_reuse(basis: Any, channels: Any) -> tuple[Reuse, StageMemo] | None:
    """The S2 store for new rows of ``basis`` on ``channels``, or ``None`` when ``basis`` is not cached.

    The keys are those of the memo entry, or of ``build_keys`` when the entry is gone, with the row keys and
    phase fingerprints of ``channels``. The caller closes the returned ``StageMemo``.
    """
    defaults = dict(basis.record.defaults)
    if not defaults.get("cache", False):
        return None
    store, memo = default_cache(), dict(basis.record.keys).get("memo")
    entry = store.get("memo", memo) if memo is not None else None
    keys = None if entry is None else entry.get("keys")
    if keys is None:
        provider = (("s1", basis.record.s1),)
        args = (basis.averaged, channels, basis.lattice, basis.columns, basis.rules)
        keys = build_keys(*args, provider)
        if keys.reasons:
            return None
    phase, why = _phase(basis.averaged, channels)
    if why:
        return None
    keys = dataclasses.replace(keys, row_keys=channels.row_keys(), phase=phase)
    stage = StageMemo(store)
    return Reuse(store, keys, stage, _jet_setup(basis.averaged, keys, store)), stage
