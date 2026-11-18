"""Import-graph checker for the 0.8 layers of design/0.8/SPEC.md Sec. 3.1.

Each module of ``syncem`` belongs to one layer. A module-level import may name only the layers that
``ALLOWED`` lists for its own layer (some only through named modules). Imports under
``if TYPE_CHECKING:`` are exempt. A function-local import of a layer that the module may not import
at module level must sit in a method listed in ``DELEGATIONS``, the table of SPEC Sec. 3.1.

Modules present at v0.7.10 (``guard_v0_7_10_hashes.json``) form the layer ``existing``. They may
import only each other and ``__version__`` from the package root. Every new layer may reuse
``existing`` (SPEC Sec. 9.2) except ``units``, ``values`` and ``lattice``.
"""

from __future__ import annotations

import ast
import json
from dataclasses import dataclass
from pathlib import Path

HASHES = Path(__file__).resolve().parent / "guard_v0_7_10_hashes.json"

NEW_LAYERS = (
    "units",
    "values",
    "base",
    "distributions",
    "lattice",
    "records",
    "engine",
    "population",
    "planning",
    "results",
    "fitting",
)
FRONT_MODULES = ("syncem", "syncem.front", "syncem.paper")
ROOT_ATTRIBUTES = ("syncem.__version__",)

_ALL = None  # every module of the target layer
_BASE = ("units", "values")
ALLOWED: dict[str, dict[str, tuple[str, ...] | None]] = {
    "units": {},
    "values": {"units": _ALL},
    "base": {"units": _ALL, "values": _ALL, "existing": _ALL},
    "distributions": {"units": _ALL, "values": _ALL, "base": _ALL, "existing": _ALL},
    "lattice": {
        "units": _ALL,
        "values": _ALL,
        "distributions": _ALL,
        "base": ("syncem.base.variables",),
    },
    "records": {
        k: _ALL for k in (*_BASE, "base", "distributions", "lattice", "existing")
    },
    "engine": {
        k: _ALL
        for k in (*_BASE, "base", "distributions", "lattice", "records", "existing")
    },
    "population": {
        **{
            k: _ALL for k in (*_BASE, "distributions", "lattice", "records", "existing")
        },
        "base": ("syncem.base.variables",),
    },
    "planning": {
        **{
            k: _ALL
            for k in (
                *_BASE,
                "distributions",
                "lattice",
                "records",
                "population",
                "existing",
            )
        },
        "base": (
            "syncem.base.variables",
            "syncem.base.kernel",
            "syncem.base.capabilities",
            "syncem.base.channels",
        ),
        "engine": ("syncem.engine.columns",),
    },
    "results": {
        k: _ALL
        for k in (
            *_BASE,
            "base",
            "distributions",
            "lattice",
            "records",
            "engine",
            "population",
            "planning",
            "existing",
        )
    },
    "fitting": {
        k: _ALL
        for k in (
            *_BASE,
            "base",
            "distributions",
            "lattice",
            "records",
            "engine",
            "population",
            "planning",
            "results",
            "existing",
        )
    },
    "front": {k: _ALL for k in (*NEW_LAYERS, "existing", "front")},
    "existing": {"existing": _ALL},
}

# (layer of the module, class, method) -> layers it may import function-locally. SPEC Sec. 3.1.
DELEGATIONS: dict[tuple[str, str, str], frozenset[str]] = {
    ("engine", "SpectralBasis", "moments"): frozenset({"population"}),
    ("engine", "SpectralBasis", "predict"): frozenset({"front"}),
    ("engine", "SpectralBasis", "direct"): frozenset({"population"}),
    ("engine", "SpectralBasis", "fit"): frozenset({"fitting"}),
    ("engine", "SpectralBasis", "build"): frozenset({"population"}),
    ("engine", "SpectralBasis", "at"): frozenset({"population"}),
    ("lattice", "Lattice", "default"): frozenset({"planning"}),
    ("lattice", "Lattice", "single"): frozenset({"population"}),
}


def _existing_modules() -> frozenset[str]:
    names = set()
    for relpath in json.loads(HASHES.read_text())["sha256"]:
        parts = (
            relpath[: -len(".py")].split("/")
            if relpath.endswith(".py")
            else relpath.split("/")
        )
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.add(".".join(parts))
    return frozenset(names)


EXISTING = _existing_modules()


def layer_of(module: str) -> str | None:
    """Layer of a module name, ``None`` outside the package. Unknown package modules give ``None`` too."""
    if module in FRONT_MODULES or module.startswith("syncem.front."):
        return "front"
    if module in ROOT_ATTRIBUTES:
        return "root"
    if module in EXISTING or any(
        module.startswith(e + ".") for e in EXISTING if e != "syncem"
    ):
        return "existing"
    parts = module.split(".")
    if parts[0] == "syncem" and len(parts) > 1 and parts[1] in NEW_LAYERS:
        return parts[1]
    return None


def _in_package(module: str) -> bool:
    return module == "syncem" or module.startswith("syncem.")


def _permitted(source_layer: str, target: str, target_layer: str) -> bool:
    if target_layer in (source_layer, "root"):
        return True
    rule = ALLOWED[source_layer]
    if target_layer not in rule:
        return False
    modules = rule[target_layer]
    return modules is None or any(
        target == m or target.startswith(m + ".") for m in modules
    )


COMPUTED = "<computed name>"
_TYPING_MODULES = ("typing", "typing_extensions")


@dataclass(frozen=True)
class _Context:
    cls: str | None
    func: str | None
    type_checking: bool


@dataclass(frozen=True)
class _Guards:
    """Names that spell ``TYPE_CHECKING`` in one module: aliases from typing, and typing's aliases."""

    names: frozenset[str]
    modules: frozenset[str]

    @classmethod
    def of(cls, tree: ast.AST) -> "_Guards":
        names, modules = set(), set()
        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module in _TYPING_MODULES:
                names.update(
                    a.asname or a.name for a in node.names if a.name == "TYPE_CHECKING"
                )
            elif isinstance(node, ast.Import):
                modules.update(
                    a.asname or a.name for a in node.names if a.name in _TYPING_MODULES
                )
        return cls(frozenset(names), frozenset(modules))

    def match(self, test: ast.expr) -> bool:
        if isinstance(test, ast.Name):
            return test.id in self.names
        return (
            isinstance(test, ast.Attribute)
            and test.attr == "TYPE_CHECKING"
            and isinstance(test.value, ast.Name)
            and test.value.id in self.modules
        )


def _is_dynamic_import(node: ast.Call) -> bool:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id in ("__import__", "import_module")
    return isinstance(func, ast.Attribute) and func.attr == "import_module"


def _root_target(name: str) -> str:
    """``from syncem import name``: a module or layer if one has that name, else the front door."""
    candidate = f"syncem.{name}"
    return candidate if layer_of(candidate) is not None else "syncem"


def _targets(node: ast.AST, module: str, is_package: bool) -> list[str]:
    if isinstance(node, ast.Call):
        arg = node.args[0] if node.args else None
        if isinstance(arg, ast.Constant) and isinstance(arg.value, str):
            return [arg.value]
        return [COMPUTED]
    if isinstance(node, ast.Import):
        return [alias.name for alias in node.names]
    if node.level:
        base = module.split(".") if is_package else module.split(".")[:-1]
        base = base[: len(base) - (node.level - 1)]
        origin = ".".join(base + ([node.module] if node.module else []))
    else:
        origin = node.module or ""
    names = [a.name for a in node.names if a.name != "*"]
    if origin == "syncem":
        return [_root_target(n) for n in names] or [origin]
    if origin.startswith("syncem.") and origin.split(".")[-1] in NEW_LAYERS:
        return [f"{origin}.{n}" for n in names] or [origin]
    return [origin]


def _walk(tree: ast.AST, ctx: _Context, guards: _Guards):
    for child in ast.iter_child_nodes(tree):
        if isinstance(child, ast.ClassDef):
            yield from _walk(
                child, _Context(child.name, None, ctx.type_checking), guards
            )
        elif isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # A def nested in a function loses the class: only the listed method itself may delegate.
            owner = ctx.cls if ctx.func is None else None
            yield from _walk(
                child, _Context(owner, child.name, ctx.type_checking), guards
            )
        elif isinstance(child, ast.If) and guards.match(child.test):
            exempt = _Context(ctx.cls, ctx.func, True)
            yield from _walk(
                ast.Module(body=child.body, type_ignores=[]), exempt, guards
            )
            yield from _walk(
                ast.Module(body=child.orelse, type_ignores=[]), ctx, guards
            )
        elif isinstance(child, (ast.Import, ast.ImportFrom)):
            yield child, ctx
        elif isinstance(child, ast.Call) and _is_dynamic_import(child):
            yield child, ctx
            yield from _walk(child, ctx, guards)
        else:
            yield from _walk(child, ctx, guards)


def _violation(layer: str, ctx: _Context, where: str, target: str) -> str | None:
    if target == COMPUTED:
        return f"{where}: dynamic import with a computed name, which the layer check cannot follow"
    if not _in_package(target):
        return None
    target_layer = layer_of(target)
    if target_layer is None:
        return f"{where}: imports {target}, which is in no layer"
    if _permitted(layer, target, target_layer):
        return None
    if ctx.func is None:
        return f"{where}: module-level import of {target} ({target_layer}) from layer {layer}"
    if target_layer in DELEGATIONS.get((layer, ctx.cls or "", ctx.func), frozenset()):
        return None
    owner = f"{ctx.cls}.{ctx.func}" if ctx.cls else ctx.func
    return (
        f"{where}: function-local import of {target} ({target_layer}) in {owner}, "
        "which is not a listed delegation"
    )


def check_source(module: str, source: str, is_package: bool | None = None) -> list[str]:
    """Violations of the layer rules by one module's source, as readable strings.

    Imports from a package ``__init__`` of a restricted layer (``from syncem.base import Kernel``)
    are read as the submodule of that name and flagged if no such module is allowed. Import the
    defining module instead.
    """
    layer = layer_of(module)
    if layer is None or layer == "root":
        return [f"{module}: module is in no layer of SPEC Sec. 3.1"]
    if is_package is None:
        is_package = module == "syncem"
    tree = ast.parse(source)
    found = []
    for node, ctx in _walk(tree, _Context(None, None, False), _Guards.of(tree)):
        if ctx.type_checking:
            continue
        for target in _targets(node, module, is_package):
            message = _violation(layer, ctx, f"{module}:{node.lineno}", target)
            if message:
                found.append(message)
    return found


def check_package(package: Path) -> list[str]:
    """Violations of every module under the package directory ``package`` (named ``syncem``)."""
    found = []
    for path in sorted(package.rglob("*.py")):
        if "__pycache__" in path.parts:
            continue
        rel = path.relative_to(package.parent).with_suffix("")
        parts = list(rel.parts)
        is_package = parts[-1] == "__init__"
        if is_package:
            parts = parts[:-1]
        found.extend(
            check_source(".".join(parts), path.read_text(), is_package=is_package)
        )
    return found
