"""User kernels: a traceable function of frequency and point values becomes a ``Kernel`` (design/0.8/SPEC.md Sec. 3.2).

LABEL ``[extension]``. ``@se.kernel(variables=..., stokes=..., azimuth=None, depth=None, capabilities=())`` turns
``fn(nu, **point) -> {"I": ..., "P": ..., "V": ...}`` into a kernel that meets the protocol of ``base.kernel``.
``fn`` takes frequencies in Hz and the point values in the canonical units of its variables, without the azimuth
and the depth, and returns spectra per Hz. The depth enters ``P`` only through the phase law, so ``I`` and ``V``
do not depend on it (SPEC Sec. 4.3). ``channel`` integrates them over the channel responses, applies a phase law to ``P``, and returns the
weighted node values with ``per_node=True``. Cells on any of its variables go to the jet provider.

Units: those of ``fn`` per Hz, integrated over unit-area channels. Shapes: ``I``, ``V`` ``(n_ch,)``, ``P``
``(1, n_ch)``, with a trailing ``n_nu`` axis per node. Not certified: the physical model of ``fn``, for which
``physical_error`` is unbounded.
"""

from __future__ import annotations

import hashlib
import inspect
from typing import Any, Callable

import equinox as eqx
import jax.numpy as jnp
import numpy as np

from ..model.errors import ErrorTerm
from .kernel import Modes, Regime
from .phase import PhaseLaw
from .variables import Variable

__all__ = ["UserKernel", "kernel"]

_STOKES = ("I", "P", "V")


def _digest_value(value: Any, depth: int = 0) -> str:
    """Stable digest of a closure value: arrays by bytes, functions by their code, others by repr."""
    if hasattr(value, "__code__") and depth < 3:
        return (
            _digest_code(value.__code__)
            + "|"
            + "|".join(
                _digest_value(c.cell_contents, depth + 1)
                for c in (value.__closure__ or ())
            )
        )
    try:
        array = np.asarray(value)
        if array.dtype != object:
            return (
                f"{array.dtype}{array.shape}"
                + hashlib.sha256(np.ascontiguousarray(array).tobytes()).hexdigest()
            )
    except (TypeError, ValueError):
        pass
    return repr(value)


def _digest_code(code: Any) -> str:
    """Digest of a code object: bytecode, names and constants, nested code objects by their own digest."""
    consts = tuple(
        _digest_code(c) if hasattr(c, "co_code") else repr(c) for c in code.co_consts
    )
    return hashlib.sha256(
        repr((code.co_code, code.co_names, consts)).encode()
    ).hexdigest()


def _identity(fn: Callable) -> tuple:
    """Module, qualified name and a hash of the code, names, constants and closure values.

    Globals that the function reads and functions it calls by name are not hashed. A changed global keeps the
    record, so a cache keyed on it must be cleared by hand.
    """
    code = getattr(fn, "__code__", None)
    if code is None:
        return (
            getattr(fn, "__module__", ""),
            getattr(fn, "__qualname__", ""),
            repr(fn),
        )
    cells = tuple(_digest_value(c.cell_contents) for c in (fn.__closure__ or ()))
    payload = (_digest_code(code) + "|" + "|".join(cells)).encode()
    return (
        getattr(fn, "__module__", ""),
        getattr(fn, "__qualname__", ""),
        hashlib.sha256(payload).hexdigest(),
    )


def _measure(channels: Any) -> Any:
    if hasattr(channels, "response_weights"):
        return channels.response_weights()
    return channels.weights * channels.node_values()


class UserKernel(eqx.Module):
    """A kernel from a traceable function ``fn(nu, **point) -> {"I": ...}``."""

    fn: Callable = eqx.field(static=True)
    variables: tuple[Variable, ...] = eqx.field(static=True)
    stokes: tuple[str, ...] = eqx.field(static=True)
    azimuth: str | None = eqx.field(static=True, default=None)
    depth: str | None = eqx.field(static=True, default=None)
    capabilities: tuple = eqx.field(static=True, default=())
    traceable: bool = eqx.field(static=True, default=True)
    lines: bool = eqx.field(static=True, default=False)

    def __check_init__(self) -> None:
        names = [v.name for v in self.variables]
        if (
            not names
            or len(names) != len(set(names))
            or not all(isinstance(v, Variable) for v in self.variables)
        ):
            raise ValueError(
                "a kernel needs a non-empty list of Variable objects with distinct names"
            )
        if not self.stokes or not set(self.stokes) <= set(_STOKES):
            raise ValueError(
                f"stokes must be a non-empty subset of {_STOKES}, not {self.stokes!r}"
            )
        for role in (self.azimuth, self.depth):
            if role is not None and role not in names:
                raise ValueError(
                    f"{role!r} is not one of the kernel's variables {names}"
                )

    @property
    def name(self) -> str:
        return getattr(self.fn, "__name__", "kernel")

    @property
    def primary(self) -> str:
        return self.variables[0].name

    def _values(self, nu: Any, point: dict, phase: Any) -> tuple[Any, Any, Any]:
        names = {v.name for v in self.variables}
        unknown = set(point) - names
        if unknown:
            raise ValueError(f"{sorted(unknown)} are not variables of {self.name}")
        missing = names - set(point) - {self.azimuth, self.depth}
        if missing:
            raise ValueError(f"{self.name} needs values for {sorted(missing)}")
        args = {k: v for k, v in point.items() if k not in (self.azimuth, self.depth)}
        out = self.fn(nu, **args)
        if not isinstance(out, dict) or not set(out) <= set(self.stokes) or not out:
            raise ValueError(
                f"{self.name} must return a dict with keys among {self.stokes}"
            )
        zero = jnp.zeros(jnp.shape(nu))
        i = jnp.asarray(out.get("I", zero))
        v = jnp.asarray(out.get("V", zero))
        p = jnp.asarray(out.get("P", zero)).astype(jnp.result_type(i, 1j))
        if phase is not False and phase is not None:
            if not isinstance(phase, PhaseLaw):
                raise TypeError("phase must be a PhaseLaw, False or None")
            p = p * phase.weights(nu, point.get(self.depth, 0.0) if self.depth else 0.0)
        return i, v, p

    def channel(
        self, channels: Any, point: dict, *, phase: Any = None, per_node: bool = False
    ) -> Modes:
        nodes = jnp.asarray(channels.nodes)
        measure = _measure(channels)
        i, v, p = self._values(nodes, point, phase)
        if per_node:
            return Modes(I=measure * i, V=measure * v, P=(measure * p)[None])
        return Modes(
            I=jnp.sum(measure * i, -1),
            V=jnp.sum(measure * v, -1),
            P=jnp.sum(measure * p, -1)[None],
        )

    def spectral(self, nu: Any, point: dict, *, phase: Any = None) -> Modes:
        i, v, p = self._values(jnp.asarray(nu), point, phase)
        return Modes(I=i, V=v, P=p[None])

    def regime(self, box: dict, channels: Any = None) -> Regime:
        return Regime(True)

    def physical_error(self, channels: Any) -> ErrorTerm:
        return ErrorTerm.unbounded(
            f"user kernel {self.name}: the physical model is the user's", "E_phys"
        )

    def describe(self) -> tuple:
        """Kernel record: the function's identity and a content hash, the variables and the roles."""
        return (
            ("name", self.name),
            ("function", _identity(self.fn)),
            ("variables", tuple((v.name, v.kind, v.unit) for v in self.variables)),
            ("stokes", self.stokes),
            ("azimuth", self.azimuth),
            ("depth", self.depth),
            ("capabilities", tuple(map(repr, self.capabilities))),
        )


def kernel(
    *,
    variables: Any,
    stokes: Any,
    azimuth: str | None = None,
    depth: str | None = None,
    capabilities: Any = (),
) -> Callable[[Callable], UserKernel]:
    """Decorator: ``fn(nu, **point) -> {"I": ...}`` becomes a traceable ``Kernel``."""

    def wrap(fn: Callable) -> UserKernel:
        try:
            params = set(inspect.signature(fn).parameters)
        except (TypeError, ValueError):
            params = set()
        for role in (azimuth, depth):
            if role is not None and role in params:
                raise ValueError(
                    f"{getattr(fn, '__name__', fn)} takes {role!r} as an argument, and the {role} role enters "
                    "only through the azimuth factor or the phase law. Remove it from the function"
                )
        return UserKernel(
            fn, tuple(variables), tuple(stokes), azimuth, depth, tuple(capabilities)
        )

    return wrap
