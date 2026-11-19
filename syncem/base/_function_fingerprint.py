"""Function fingerprints of design/0.8/SPEC.md decision 156.

``function_fingerprint(fn)`` is the sha256 hex digest of a format tag, ``sys.version_info[:2]`` and the
bytes below. No ``repr``, ``hash()`` or ``id()`` enters.

Library code is the standard library, the package ``syncem`` and any package whose files lie under a
site-packages directory. All other code is user code, a versioned package outside site-packages (an
editable install, a source tree on ``sys.path``) and ``__main__`` included.

* A user function hashes its code object (bytecode, names, argument counts, flags, the exception table and
  the constants, nested code objects recursively, frozenset constants sorted by their encoding), its
  defaults and keyword defaults, its closure cells and the globals that its bytecode loads. Line numbers,
  the file name and the qualified name are not hashed, so a function defined twice from the same text has
  one fingerprint.
* A value in a closure cell, a default or a global is written as by ``encode``, with functions, classes
  and library modules admitted inside tuples, lists, dicts, sets and dataclass fields. A global user
  function is hashed recursively, and a function already on the hashing stack is written as a back
  reference to its stack position.
* A user module named as a global enters through the attribute paths that the bytecode reads from it
  (``LOAD_GLOBAL`` followed by ``LOAD_ATTR``), and the value at each path is hashed by these rules. A user
  module stored, passed, or held in a closure cell, a default or a container raises ``NotCacheable``.
* A library module, and a library function, class or other callable, is written by its module, qualified
  name and installed version, and is not traced into. A syncem function with closure cells is traced.
* A user function wrapped by ``jax.jit`` or ``functools.wraps`` is hashed through ``__wrapped__``. A
  ``functools.partial`` is hashed by its function, arguments and keywords, and a bound method by its
  function and its instance.
* A user class, and any other value without an encoding, raises ``NotCacheable`` with a reason.

A dataclass value is written by its class name and fields, so an edit to a method of a user dataclass does
not change the fingerprint. This follows the dataclass rule of ``encode``. A read through ``getattr`` with
a computed name, ``importlib`` or another indirection is not seen. The module imports NumPy, the standard
library and ``_fingerprint``.
"""

from __future__ import annotations

import dataclasses
import dis
import functools
import hashlib
import importlib.metadata
import itertools
import os
import site
import sys
import sysconfig
import types
from typing import Any

import numpy as np

from ._fingerprint import NotCacheable, _length, _text, _unordered, encode

__all__ = ["function_fingerprint"]

_FUNCTION_FORMAT = b"syncem-function-1"
_MISSING = object()
_GLOBAL_OPS = frozenset({"LOAD_GLOBAL", "LOAD_NAME", "STORE_GLOBAL", "DELETE_GLOBAL"})
_READ_OPS = frozenset({"LOAD_GLOBAL", "LOAD_NAME"})
_ATTR_OPS = frozenset({"LOAD_ATTR", "LOAD_METHOD"})
_OWN_PACKAGES = frozenset({"syncem"})  # keyed by version, its closures traced


def _site_roots() -> tuple[str, ...]:
    paths = {sysconfig.get_paths().get(k) for k in ("purelib", "platlib")}
    try:
        paths.update(site.getsitepackages())
        paths.add(site.getusersitepackages())
    except (
        AttributeError
    ):  # a virtual environment whose site module lacks these functions
        pass
    return tuple(sorted({os.path.realpath(p) for p in paths if p}))


_SITE_ROOTS = _site_roots()


def _installed_version(module_name: Any) -> str | None:
    """Version of the package that holds ``module_name``: "" for the standard library, ``None`` without one."""
    if not isinstance(module_name, str) or not module_name:
        return None
    top = module_name.split(".")[0]
    if top in sys.stdlib_module_names:
        return ""
    version = getattr(sys.modules.get(top), "__version__", None)
    if isinstance(version, str):
        return version
    if top == "__main__":
        return None
    try:
        return importlib.metadata.version(top)
    except (importlib.metadata.PackageNotFoundError, ValueError):
        return None


def _under_site(top: str) -> bool:
    """True when the files of the imported package ``top`` lie under a site-packages directory."""
    module = sys.modules.get(top)
    files = [getattr(module, "__file__", None), *getattr(module, "__path__", ())]
    files = [os.path.realpath(f) for f in files if isinstance(f, str)]
    if module is not None and not files:
        return True  # a module built into the interpreter
    roots = [r + os.sep for r in _SITE_ROOTS]
    return bool(files) and all(any(f.startswith(r) for r in roots) for f in files)


def _library_version(module_name: Any) -> str | None:
    """ "" for the standard library, the version of syncem or of a package under site-packages, else ``None``."""
    version = _installed_version(module_name)
    if not version:
        return version
    top = module_name.split(".")[0]
    return version if top in _OWN_PACKAGES or _under_site(top) else None


def _named(tag: bytes, module: str, name: str, version: str) -> bytes:
    return tag + b"".join(_text(b"", s.encode()) for s in (module, name, version))


def _const(value: Any) -> bytes:
    """A code constant: nested code objects recursively, tuples and frozensets entry by entry."""
    if isinstance(value, types.CodeType):
        return b"c" + _code(value)
    if isinstance(value, tuple):
        return _length(len(value)) + b"T" + b"".join(_const(v) for v in value)
    if isinstance(value, frozenset):
        return _unordered(b"Z", [_const(v) for v in value])
    return b"." if value is Ellipsis else encode(value)


def _code(code: types.CodeType) -> bytes:
    """Bytecode, names, counts, flags, exception table and constants. No line numbers or file name."""
    ints = (
        code.co_argcount,
        code.co_posonlyargcount,
        code.co_kwonlyargcount,
        code.co_flags,
    )
    names = (
        code.co_name,
        code.co_names,
        code.co_varnames,
        code.co_freevars,
        code.co_cellvars,
    )
    table = getattr(code, "co_exceptiontable", b"")
    return (
        encode(ints)
        + encode(names)
        + _text(b"", code.co_code)
        + _text(b"", table)
        + _const(code.co_consts)
    )


def _nested(code: types.CodeType) -> list[types.CodeType]:
    """``code`` and the code objects nested in its constants."""
    found = [code]
    for const in code.co_consts:
        if isinstance(const, types.CodeType):
            found += _nested(const)
    return found


def _global_names(code: types.CodeType) -> set[str]:
    """Names that the bytecode of ``code`` and of its nested code objects loads or stores as globals."""
    ops = (i for c in _nested(code) for i in dis.get_instructions(c))
    return {i.argval for i in ops if i.opname in _GLOBAL_OPS}


def _attribute_paths(code: types.CodeType, name: str) -> set[tuple[str, ...]]:
    """Attribute paths that the bytecode reads from the global module ``name``. Any other use raises."""
    paths = set()
    for c in _nested(code):
        ops = list(dis.get_instructions(c))
        for k, op in enumerate(ops):
            if op.opname not in _GLOBAL_OPS or op.argval != name:
                continue
            path = tuple(
                itertools.takewhile(lambda o: o.opname in _ATTR_OPS, ops[k + 1 :])
            )
            if op.opname not in _READ_OPS or not path:
                raise NotCacheable("it is used other than by attribute reads")
            paths.add(tuple(o.argval for o in path))
    return paths


def _context(where: str, run: Any) -> bytes:
    try:
        return run()
    except NotCacheable as error:
        raise NotCacheable(f"{where}: {error.reason}") from None


def _user_module(value: Any) -> bool:
    return (
        isinstance(value, types.ModuleType) and _library_version(value.__name__) is None
    )


def _module_reads(fn: types.FunctionType, name: str, stack: list) -> bytes:
    """The values at the attribute paths that ``fn`` reads from the user module bound to ``name``."""
    parts = []
    for path in sorted(_attribute_paths(fn.__code__, name)):
        value = fn.__globals__[name]
        for attr in path:
            if not _user_module(value):
                break
            value = getattr(value, attr, _MISSING)
        body = b"u" if value is _MISSING else _value(value, stack)
        parts.append(_text(b"", ".".join(path).encode()) + body)
    return b"O" + _length(len(parts)) + b"".join(parts)


def _globals(fn: types.FunctionType, stack: list) -> bytes:
    namespace = fn.__globals__
    builtins = fn.__builtins__ if isinstance(fn.__builtins__, dict) else {}
    parts = []
    for name in sorted(_global_names(fn.__code__)):
        if name not in namespace:
            body = b"b" if name in builtins else b"u"
        elif _user_module(namespace[name]):
            where = f"user module {namespace[name].__name__}"
            body = _context(where, lambda: _module_reads(fn, name, stack))
        else:
            body = _context(f"global {name!r}", lambda: _value(namespace[name], stack))
        parts.append(_text(b"", name.encode()) + body)
    return _length(len(parts)) + b"".join(parts)


def _cells(fn: types.FunctionType, stack: list) -> bytes:
    parts = []
    for name, cell in zip(fn.__code__.co_freevars, fn.__closure__ or ()):
        try:
            contents = cell.cell_contents
        except ValueError:
            parts.append(b"e")
            continue
        parts.append(
            _context(f"closure value {name!r}", lambda: _value(contents, stack))
        )
    return _length(len(parts)) + b"".join(parts)


def _function(fn: types.FunctionType, stack: list) -> bytes:
    """A user function: code, defaults, keyword defaults, closure cells and named globals."""
    for depth, seen in enumerate(stack):
        if seen is fn:
            return b"R" + _length(depth)
    stack.append(fn)
    try:
        where = getattr(fn, "__qualname__", "function")
        defaults = (fn.__defaults__ or (), fn.__kwdefaults__ or {})
        return b"U" + _context(
            where,
            lambda: _code(fn.__code__)
            + _value(defaults, stack)
            + _cells(fn, stack)
            + _globals(fn, stack),
        )
    finally:
        stack.pop()


def _library(value: Any) -> bytes | None:
    """Module, name and version of a callable of library code. A syncem function with closure cells is traced."""
    if isinstance(value, types.FunctionType):
        module = value.__module__
        if value.__closure__ and str(module).split(".")[0] in _OWN_PACKAGES:
            return None
    else:
        module = getattr(value, "__module__", None) or type(value).__module__
    version = _library_version(module)
    name = getattr(value, "__qualname__", None) or getattr(value, "__name__", None)
    if version is None or not isinstance(name, str):
        return None
    return _named(b"G", module, name, version)


def _callable(value: Any, stack: list) -> bytes:
    if isinstance(value, functools.partial):
        parts = (value.args, value.keywords)
        return b"P" + _callable(value.func, stack) + _value(parts, stack)
    if isinstance(value, types.MethodType):
        return b"m" + _callable(value.__func__, stack) + _value(value.__self__, stack)
    named = _library(value)
    if named is not None:
        owner = getattr(value, "__self__", None)
        if owner is None or isinstance(owner, (types.ModuleType, type)):
            return named
        return named + _value(owner, stack)
    if isinstance(value, type):
        raise NotCacheable(
            f"class {value.__qualname__} of user code has no fingerprint"
        )
    if isinstance(value, types.FunctionType):
        return _function(value, stack)
    wrapped = getattr(value, "__wrapped__", None)
    if callable(wrapped):
        kind = type(value).__qualname__.encode()
        return _text(b"W", kind) + _callable(wrapped, stack)
    raise NotCacheable(
        f"callable of type {type(value).__qualname__} has no fingerprint"
    )


def _value(value: Any, stack: list) -> bytes:
    """``encode(value)`` with functions, classes and library modules admitted inside containers and dataclasses."""
    if isinstance(value, (tuple, list)):
        tag = b"T" if isinstance(value, tuple) else b"L"
        return tag + _length(len(value)) + b"".join(_value(v, stack) for v in value)
    if isinstance(value, dict):
        parts = [encode(k) + _value(v, stack) for k, v in value.items()]
        return _unordered(b"D", parts)
    if isinstance(value, (set, frozenset)):
        tag = b"Z" if isinstance(value, frozenset) else b"E"
        return _unordered(tag, [_value(v, stack) for v in value])
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        cls = type(value)
        fields = [
            _text(b"", f.name.encode()) + _value(getattr(value, f.name), stack)
            for f in dataclasses.fields(value)
        ]
        name = f"{cls.__module__}.{cls.__qualname__}".encode()
        return _text(b"K", name) + _length(len(fields)) + b"".join(fields)
    if isinstance(value, types.ModuleType):
        version = _library_version(value.__name__)
        if version is None:
            raise NotCacheable(
                f"user module {value.__name__} held as a value has no fingerprint"
            )
        return _named(b"M", value.__name__, "", version)
    if callable(value) and not isinstance(value, np.ndarray):
        return _callable(value, stack)
    return encode(value)


def function_fingerprint(fn: Any) -> str:
    """sha256 hex fingerprint of a callable by the rules of the module docstring.

    Raises ``NotCacheable`` with a reason for a value without a fingerprint.
    """
    if not callable(fn):
        raise NotCacheable(f"{type(fn).__qualname__} is not callable")
    python = encode(tuple(sys.version_info[:2]))
    return hashlib.sha256(_FUNCTION_FORMAT + python + _callable(fn, [])).hexdigest()
