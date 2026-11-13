"""Run the harmonic-pipeline modules in inputs/harmonic-radio-fit against syncem.

radio_numerics.py and stable_lines.py are copies of the harmonic pipeline that produced the
frozen harmonic matrix (inputs/README.md). They import the package by its earlier name,
`syncmoments`, and call syncem.constants and syncem.model.high_order_harmonic. The harmonic
pipeline pinned SyncMoments d51f9be. use_syncem() checks that the code they call is unchanged
since that commit apart from the rename to syncem: the whole of syncem.constants, and in
syncem.model.high_order_harmonic the functions _geometry and high_order_line_powers with every
module-level name they reach. The check compares SHA-256 digests of the abstract syntax tree
with docstrings removed, so edits to comments and docstrings (as in SyncEM 0.6.0 and later) pass,
while any change to the executed code fails.

use_syncem() imports syncem from the package root of this repository (two levels above the
study directory), checks that it resolved there, registers syncem and its loaded submodules
under the earlier name, and puts inputs/harmonic-radio-fit on sys.path. Call it before
importing radio_numerics or stable_lines, in every process that imports them.
"""
from __future__ import annotations

import ast
import hashlib
import sys
from pathlib import Path

STUDY = Path(__file__).resolve().parents[1]
PACKAGE_ROOT = STUDY.parents[1]
HARMONIC_INPUTS = STUDY / "inputs" / "harmonic-radio-fit"
EARLIER_NAME = "syncmoments"  # package name before SyncEM 0.6.0, used by the frozen modules
D51F9BE = "d51f9bef4fa744a536a60c3da821b1033818a232"
# Names called by the frozen modules (None: the whole module), and the SHA-256 of their
# docstring-free syntax tree at SyncMoments d51f9be.
D51F9BE_CODE = {
    "constants.py": (None, "97a24f1ff6cc6aaf0367298fc78923a4df74744f36341ade115f30dcd606cb17"),
    "model/high_order_harmonic.py": (("_geometry", "high_order_line_powers"), "09454647a7e5ee92699ded4b530ccd53d247af06b4cfa88841368c2c27fc9b6f"),
}


def _strip_docstrings(tree: ast.AST) -> ast.AST:
    for node in ast.walk(tree):
        if isinstance(node, (ast.Module, ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            body = node.body
            if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant) \
                    and isinstance(body[0].value.value, str):
                node.body = body[1:] or [ast.Pass()]
    return tree


def _top_level_names(tree: ast.Module) -> dict:
    names = {}
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            names[node.name] = node
        elif isinstance(node, (ast.Assign, ast.AnnAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            names.update({t.id: node for t in targets if isinstance(t, ast.Name)})
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            names.update({(a.asname or a.name).split(".")[0]: node for a in node.names})
    return names


def code_digest(source: str, roots: tuple | None) -> str:
    """SHA-256 of the docstring-free syntax tree of `source`, or of `roots` and every name they reach."""
    tree = _strip_docstrings(ast.parse(source))
    if roots is None:
        return hashlib.sha256(ast.dump(tree).encode()).hexdigest()
    names, seen, stack = _top_level_names(tree), set(), list(roots)
    while stack:
        name = stack.pop()
        if name in seen or name not in names:
            continue
        seen.add(name)
        stack.extend(n.id for n in ast.walk(names[name]) if isinstance(n, ast.Name) and n.id in names)
    dumped = "\n".join(f"{name}: {ast.dump(names[name])}" for name in sorted(seen))
    return hashlib.sha256(dumped.encode()).hexdigest()


def source_before_rename(relpath: str) -> str:
    """syncem/<relpath> with the package name mapped back to the earlier name."""
    source = (PACKAGE_ROOT / "syncem" / relpath).read_text()
    return source.replace("syncem", EARLIER_NAME).replace("SyncEM", "SyncMoments")


def use_syncem():
    """Import syncem from PACKAGE_ROOT and register it under the earlier package name."""
    root = str(PACKAGE_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    import syncem  # noqa: PLC0415
    import syncem.constants  # noqa: F401, PLC0415
    import syncem.model.high_order_harmonic  # noqa: F401, PLC0415

    if Path(syncem.__file__).resolve().parents[1] != PACKAGE_ROOT:
        raise RuntimeError("syncem did not resolve to the package root of this repository")
    for relpath, (roots, expected) in D51F9BE_CODE.items():
        if code_digest(source_before_rename(relpath), roots) != expected:
            raise RuntimeError(f"code that the frozen modules call in syncem/{relpath} differs from "
                               "SyncMoments d51f9be beyond the rename")
    for name, module in list(sys.modules.items()):
        if name == "syncem" or name.startswith("syncem."):
            sys.modules[EARLIER_NAME + name[len("syncem"):]] = module
    path = str(HARMONIC_INPUTS)
    if path not in sys.path:
        sys.path.insert(0, path)
    return syncem


def check_record() -> dict:
    """What use_syncem() checked, for a results JSON."""
    import syncem  # noqa: PLC0415

    return {
        "package": "syncem",
        "version": syncem.__version__,
        "resolved_to_package_root_of_this_repository": True,
        "code_equal_to_SyncMoments_d51f9be_up_to_the_rename_and_docstrings": {
            rel: "whole module" if roots is None else list(roots) for rel, (roots, _) in D51F9BE_CODE.items()},
        "SyncMoments_pin": D51F9BE,
    }
