"""Named assumptions and the ``assumptions=`` keyword (``syncem.model.assumptions``).

Private helper of :mod:`syncem.model.assumptions`, which re-exports every
public name here; split out to keep that file near 400 lines. It imports the
classes of ``assumptions.py``, which imports this module last.

LABEL: ``eq: uncoupled moments`` (``angles_independent``, ``known_angles``),
``eq: conditional legendre coefficients`` (the ``a_lk`` of
``angular_table``); ``uniform_field_direction``, ``uniform_azimuth`` and
``known_field`` are ``[extension]``.

In the coupled case of ``eq: conditional legendre coefficients`` the
coefficients ``a_lk(gamma, B, phi, depth)`` vary with the conditioning
variables and every retained moment is free (``no_assumption``). In the
uncoupled case they are constants and ``eq: uncoupled moments`` factorises
every moment into ``a_lk / ((2l+1)(2k+1))`` times a non-angular moment: the
partition ``{mu, eta} | {gamma, B, phi, depth}``. With unknown ``a_lk`` the
map is bilinear (``angles_independent``: 33 free parameters at
``(2, 2, 2)``, 8 angular and 25 non-angular; not affine, so linear fits
refuse it). With known ``a_lk`` the angular group is a ``fixed_table`` and
the map is affine in the 25 non-angular moments (``known_angles``).

``declare(index, assumptions)`` builds a ``ParameterMap`` from a name, an
``Assume(name, **params)`` or a tuple of them; a tuple is combined with
``ParameterMap.assume`` in the order of ``ASSUMPTIONS`` (not the order given),
so the record name ``"a+b"``, the key of ``assumption_allowances``, does not
depend on the order of the tuple. A combined map keeps the rule of
``assume``: no discrepancy unless ``declare(..., discrepancy=)`` supplies
one for the combination, so its budget term is ``unbounded``.

Units: dimensionless (Legendre and displacement moments); ``known_field``
takes ``B`` in Gauss. Shapes: ``angular_table`` returns ``(n_ang,)``, one
entry per nonconstant ``(l, k)`` projection of the rows of ``m`` (the
``tables_spec`` of the ``('mu', 'eta')`` group); ``a_lk`` is ``(L_mu + 1,
L_eta + 1)``. Not certified: that a population satisfies a declared
assumption (the discrepancy is an input, measured on supplied samples with
``JointMoments.from_samples(..., discrepancy="measured")``, or ``unbounded``).
"""

from __future__ import annotations

from types import MappingProxyType
from typing import ClassVar

import jax
import jax.numpy as jnp
import numpy as np

from . import _factor as _f
from .assumptions import (
    VARS,
    Closure,
    Factorisation,
    ParameterMap,
    azimuth_separable,
    field_independent,
    field_reversal_symmetric,
    fully_independent,
    gaussian_screen,
    independent_screen,
    isotropic_pitch,
    no_assumption,
    pitch_symmetric,
)

ANGLES = ("mu", "eta")
NON_ANGULAR = ("gamma", "B", "phi", "depth")
_UNCOUPLED = "eq: uncoupled moments"


def _rest(variable):
    return tuple(v for v in VARS if v != variable)


def _concrete(value):
    return not isinstance(value, jax.core.Tracer)


def angles_independent(index) -> ParameterMap:
    """``{mu, eta} | {gamma, B, phi, depth}`` with a free angular table: the
    uncoupled case of ``eq: uncoupled moments``, constant unknown ``a_lk``.

    Bilinear (two free groups), so ``fit_linear`` and ``fit_combinations``
    refuse it; fit it with ``fit_bfgs`` on ``LogDensity(basis, data,
    assumptions="angles_independent")`` or declare the coefficients with
    :func:`known_angles`. Declares the independence; nothing verifies it.
    """
    factorisation = Factorisation((ANGLES, NON_ANGULAR), "angles_independent")
    return ParameterMap.build(index, factorisation, label=_UNCOUPLED)


def _angular_spec(index):
    groups = _f.canonical_partition((ANGLES, NON_ANGULAR))
    return _f.build_tables(index, groups, ())[0][groups.index(ANGLES)]


def angular_table(index, a_lk=None) -> jax.Array:
    """Fixed ``('mu', 'eta')`` table ``<P_l(mu) P_k(eta)> = a_lk / ((2l+1)(2k+1))``.

    ``a_lk``: ``None`` (every entry zero: isotropic pitch and uniform
    ``eta``; the azimuth is not constrained, so the ``h = 2`` moments stay
    free and an isotropic field direction also needs :func:`uniform_azimuth`)
    or an ``(L_mu + 1, L_eta + 1)`` array of the coefficients of
    ``eq: conditional legendre coefficients`` with ``a_00 = 1`` (checked when
    concrete). Returns ``(n_ang,)`` in the ``tables_spec`` order of the
    angular group (the input of :func:`known_angles`); traceable in ``a_lk``.
    Not checked: that the coefficients describe a nonnegative density.
    """
    spec = _angular_spec(index)
    if a_lk is None:
        return jnp.zeros(len(spec))
    a = jnp.asarray(a_lk, dtype=float)
    t = index.truncation
    if a.shape != (t.L_mu + 1, t.L_eta + 1):
        raise ValueError(f"a_lk must have shape ({t.L_mu + 1}, {t.L_eta + 1})")
    if _concrete(a):
        if not np.all(np.isfinite(np.asarray(a))):
            raise ValueError("a_lk must be finite")
        if abs(float(a[0, 0]) - 1.0) > 1e-12:
            raise ValueError("a_lk[0, 0] must equal 1 (normalised angular density)")
    if not spec:
        return jnp.zeros(0)
    l, k = (np.asarray([p[i] for p in spec]) for i in (0, 1))
    return a[l, k] / ((2 * l + 1) * (2 * k + 1))


def known_angles(index, table=None, *, a_lk=None) -> ParameterMap:
    """``{mu, eta} | rest`` with a ``fixed_table`` angular marginal: the
    uncoupled case with known ``a_lk`` (``eq: uncoupled moments``), affine in
    the non-angular moments.

    ``table``: the ``(n_ang,)`` output of :func:`angular_table`; or ``a_lk``
    (passed to it); neither gives the zero table, which has the moments of
    ``isotropic_pitch.assume(uniform_field_direction)`` (the azimuth stays
    free). Passing both raises ``ValueError``. Declares the coefficients;
    nothing verifies them.
    """
    if table is not None and a_lk is not None:
        raise ValueError("pass table or a_lk to known_angles, not both")
    if table is None:
        table = angular_table(index, a_lk)
    closure = Closure(ANGLES, "fixed_table", (jnp.asarray(table, dtype=float),))
    factorisation = Factorisation((ANGLES, NON_ANGULAR), "known_angles")
    return ParameterMap.build(index, factorisation, (closure,), label=_UNCOUPLED)


def uniform_field_direction(index) -> ParameterMap:
    """``{eta} | rest`` with ``uniform_eta``: rows with ``k >= 1`` are zero and
    dropped (``a_lk = 0`` for ``k >= 1`` in ``eq: conditional legendre
    coefficients``) ``[extension]``.

    Uniform ``eta = cos(theta)``, independent of the other variables; the sky
    azimuth is declared separately (:func:`uniform_azimuth`), and an
    isotropic field direction is both. Declares it; nothing verifies it.
    """
    factorisation = Factorisation((_rest("eta"), ("eta",)), "uniform_field_direction")
    closure = Closure(("eta",), "uniform_eta")
    return ParameterMap.build(index, factorisation, (closure,))


def uniform_azimuth(index) -> ParameterMap:
    """``{phi} | rest`` with ``uniform_phi``: every ``h = 2`` row
    (``<e^{2i phi} ...>``) is zero and dropped, so the finite ``Q, U``
    response vanishes ``[extension]``. Declares it; nothing verifies it."""
    factorisation = Factorisation((_rest("phi"), ("phi",)), "uniform_azimuth")
    closure = Closure(("phi",), "uniform_phi")
    return ParameterMap.build(index, factorisation, (closure,))


def known_field(index, B) -> ParameterMap:
    """``{B} | rest`` with a ``delta`` closure at the field strength ``B``
    (Gauss): every electron sees one field strength ``[extension]``.

    ``B`` is a scalar leaf (traceable; checked positive when concrete). The
    map needs a reference for ``affine_pieces``. Declares the value; nothing
    verifies it.
    """
    value = jnp.asarray(B, dtype=float)
    if value.ndim != 0:
        raise ValueError("known_field needs a scalar B (Gauss)")
    if _concrete(value) and not (np.isfinite(float(value)) and float(value) > 0.0):
        raise ValueError("known_field needs a finite B > 0 (Gauss)")
    factorisation = Factorisation((_rest("B"), ("B",)), "known_field")
    closure = Closure(("B",), "delta", (value,))
    return ParameterMap.build(index, factorisation, (closure,))


ASSUMPTIONS = MappingProxyType(
    {
        "no_assumption": no_assumption,
        "isotropic_pitch": isotropic_pitch,
        "independent_screen": independent_screen,
        "gaussian_screen": gaussian_screen,
        "field_independent": field_independent,
        "azimuth_separable": azimuth_separable,
        "fully_independent": fully_independent,
        "pitch_symmetric": pitch_symmetric,
        "field_reversal_symmetric": field_reversal_symmetric,
        "angles_independent": angles_independent,
        "known_angles": known_angles,
        "uniform_field_direction": uniform_field_direction,
        "uniform_azimuth": uniform_azimuth,
        "known_field": known_field,
    }
)


class Assume:
    """One named assumption with its builder parameters ``[extension]``.

    ``Assume(name, **params)``: ``name`` a key of ``ASSUMPTIONS``; ``params``
    are passed to the builder as keywords, e.g. ``Assume("known_field",
    B=5e-6)`` or ``Assume("known_angles", a_lk=a)``. A declaration only;
    nothing verifies that the population satisfies it.
    """

    LABEL: ClassVar[str] = "[extension] named assumption"
    __slots__ = ("name", "params")

    def __init__(self, name: str, **params):
        if name not in ASSUMPTIONS:
            raise ValueError(
                f"unknown assumption {name!r}; the names are {tuple(ASSUMPTIONS)}"
            )
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "params", tuple(params.items()))

    def __setattr__(self, key, value):
        raise AttributeError("Assume is immutable")

    def __repr__(self) -> str:
        args = "".join(f", {k}={v!r}" for k, v in self.params)
        return f"Assume({self.name!r}{args})"

    def build(self, index) -> ParameterMap:
        """The builder's ``ParameterMap`` on ``index``; ``ValueError`` when
        the builder's parameters are missing or unknown (module docstring)."""
        try:
            return ASSUMPTIONS[self.name](index, **dict(self.params))
        except TypeError as error:
            raise ValueError(f"{self!r}: {error}") from None


def _specs(assumptions):
    items = assumptions if isinstance(assumptions, (tuple, list)) else (assumptions,)
    specs = []
    for item in items:
        if isinstance(item, str):
            item = Assume(item)
        if not isinstance(item, Assume):
            raise ValueError(
                "assumptions must be a name, an Assume or a tuple of them, got "
                f"{type(item).__name__}"
            )
        specs.append(item)
    if not specs:
        raise ValueError("assumptions is empty; use 'no_assumption' to declare none")
    names = [s.name for s in specs]
    repeated = sorted({n for n in names if names.count(n) > 1})
    if repeated:
        raise ValueError(f"assumptions {repeated} are declared more than once")
    order = tuple(ASSUMPTIONS)
    return sorted(specs, key=lambda s: order.index(s.name))


def _odd_under(kind, group, entry):
    """True when the symmetry ``kind`` flips the sign of the table ``entry``
    of ``group`` (odd total exponent on its flipped variables)."""
    flipped = _f.FLIPPED[kind]
    return sum(e for v, e in zip(group, entry) if v in flipped) % 2 == 1


def _check_symmetric_tables(pm) -> None:
    """``ValueError`` naming the first nonzero entry of a concrete
    ``fixed_table`` that a symmetry closure of ``pm`` sets to zero.

    ``ParameterMap.build`` and ``assume`` take such a table as its
    symmetrised marginal and ignore the entry (Section 7.4 of the design
    document); a declaration by name that states both the entry and the
    symmetry contradicts itself, so ``declare`` refuses it. Entries the
    symmetry keeps, and traced tables, are not checked.
    """
    if not any(pm.symmetrised):
        return
    kinds = [c.kind for c in pm.closures if c.kind in _f.SYMMETRY_KINDS]
    groups = pm.factorisation.groups
    plain = tuple(c for c in pm.closures if c.kind not in _f.SYMMETRY_KINDS)
    full = _f.build_tables(pm.index, groups, plain)[0]
    for closure, trim in zip(pm.closures, pm.symmetrised):
        if trim is None or not _concrete(closure.hyper[0]):
            continue
        group = tuple(closure.group)
        values = np.asarray(closure.hyper[0])
        for entry, value in zip(full[groups.index(group)], values):
            odd = [kind for kind in kinds if _odd_under(kind, group, entry)]
            if odd and value != 0:
                raise ValueError(
                    f"fixed_table on {group} sets {_f.entry_label(group, entry)} "
                    f"= {value:.6g}, which {odd[0]} sets to zero; zero that "
                    "coefficient or drop the symmetry"
                )


def declare(index, assumptions, *, discrepancy=None) -> ParameterMap:
    """``ParameterMap`` of named assumptions (module docstring).

    ``assumptions``: a name of ``ASSUMPTIONS``, an ``Assume`` or a tuple of
    them; several are combined with ``ParameterMap.assume`` in registry
    order (``ValueError`` on a conflict, a repeated name or an unknown
    name, and on a concrete known table with a nonzero entry that a
    declared symmetry sets to zero, see :func:`_check_symmetric_tables`).
    ``discrepancy``: an optional moment-level ``ErrorTerm`` ``(n_real,)``
    for the resulting map (otherwise ``None``, so its record is
    ``unbounded``). Not certified: that the population satisfies them.
    """
    maps = [spec.build(index) for spec in _specs(assumptions)]
    pm = maps[0] if len(maps) == 1 else maps[0].assume(*maps[1:])
    _check_symmetric_tables(pm)
    if discrepancy is None:
        return pm
    return ParameterMap.build(
        pm.index,
        pm.factorisation,
        pm.closures,
        discrepancy=discrepancy,
        name=pm.name,
        label=pm.LABEL,
        free_hyper=pm.free_hyper,
    )


def resolve_map(index, parameter_map, assumptions):
    """``parameter_map``, or ``declare(index, assumptions)``; ``ValueError``
    when both are given (the ``assumptions=`` keyword of the fits and of
    ``JointMoments.from_samples``)."""
    if assumptions is None:
        return parameter_map
    if parameter_map is not None:
        raise ValueError("pass parameter_map= or assumptions=, not both")
    return declare(index, assumptions)


def _bilinear_angles(pm):
    """True when closing the angular group with :func:`known_angles` makes
    ``pm`` affine: exactly two free groups, one of them ``('mu', 'eta')``
    (groups are in canonical ``VARS`` order), and no fitted hyper-parameters
    (a nodal map has one group)."""
    free = pm.free_groups()
    groups = pm.factorisation.groups
    angular = any(tuple(groups[g]) == ANGLES for g in free)
    return len(free) == 2 and angular and not pm.free_hyper


def require_affine(pm, route) -> None:
    """``ValueError`` for a non-affine map on the linear ``route``; the
    message points to ``fit_bfgs`` and, when :func:`known_angles` would make
    the map affine (the bilinear uncoupled case), to that and to
    ``LogDensity(..., assumptions=...)``."""
    if pm.is_affine():
        return
    message = (
        f"parameter map '{pm.name}' is not affine (several free groups, fitted "
        f"hyper-parameters or nodal), so {route} cannot use it; use fit_bfgs or "
        "fit_nodal"
    )
    if _bilinear_angles(pm):
        message += (
            "; its angular table and its non-angular moments are both free, so "
            "the moments are bilinear (eq: uncoupled moments): declare the "
            "angular coefficients with known_angles(index, a_lk=...) "
            "(assumptions=Assume('known_angles', a_lk=...)) for a linear fit, or "
            "fit both factors with fit_bfgs on LogDensity(basis, data, "
            "assumptions=...)"
        )
    raise ValueError(message)


__all__ = [
    "ASSUMPTIONS",
    "Assume",
    "declare",
    "angles_independent",
    "angular_table",
    "known_angles",
    "uniform_field_direction",
    "uniform_azimuth",
    "known_field",
]
