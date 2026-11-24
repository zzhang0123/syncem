"""Docstring contract of ``syncem.model`` (documentation audit, 2026-09-24).

Every public module of ``syncem.model`` and ``syncem.model.fit`` states its
manuscript labels (``LABEL``), its units (or that it is dimensionless), its
shapes, and what it does not certify. Every public class or function has its
own docstring; every public class names a manuscript label (a ``LABEL``
attribute or a label in its docstring, ``[extension]`` for additions); every
public docstring states what it does not certify, either itself or by
pointing to the module docstring that does.

The last three tests check ``README.md`` and ``docs/DESIGN.md`` against the
code: DESIGN names every keyword of the documented signatures, neither file
repeats a statement superseded by the round-2 fixes, and the README shows
``assumption_allowances`` with ``screen_factorisation_bound``.

The checks are textual. They pin the presence of the statements, not their
correctness; the numerical content is tested by the other files here.
"""

from __future__ import annotations

import ast
import importlib
import inspect
import json
import pkgutil
import re
from pathlib import Path

import pytest

import syncem.model
import syncem.model.fit

LABEL_RX = re.compile(r"LABEL|eq:|\[extension\]|app:|sec:|tab:")
UNITS_RX = re.compile(
    r"\bunits?\b|Gauss|\bHz\b|rad/m\^?2|erg|metres|dimensionless|Dimensionless",
    re.I,
)
SHAPE_RX = re.compile(
    r"\(n_\w+|\(S,|\(n0|\(n2|\(4 n_ch|shape|\(n, n\)|tuples? of", re.I
)
NOT_CERTIFIED_RX = re.compile(
    r"not\s+(be\s+)?(certif|verif|checked|bounded|sufficient|a\s+bound"
    r"|an?\s+(accuracy\s+|package\s+)?certificate)"
    r"|certif\w*\s+nothing|nothing\s+[\w\s]{0,24}?(certif|verif)"
    r"|never\s+a\s+(package\s+)?certificate|neither\s+is\s+verified"
    r"|does\s+not\s+(certify|verify|bound)|assumed\s+or\s+certified"
    r"|no\s+certifi\w*|certifies\s+only",
    re.IGNORECASE,
)
MODULE_POINTER_RX = re.compile(r"module\s+docstring|see\s+:(class|func|mod):", re.I)


def _public_modules():
    names = []
    for package in (syncem.model, syncem.model.fit):
        for info in pkgutil.iter_modules(package.__path__):
            if info.name.startswith("_") or info.name == "fit":
                continue
            names.append(f"{package.__name__}.{info.name}")
    return tuple(sorted(names))


def _new_layer_modules():
    """Public modules of the 0.8 layers (design/0.8/SPEC.md Sec. 9.7): ``syncem.units`` and the subpackages."""
    names = ["syncem.units", "syncem.paper"]
    for layer in NEW_LAYERS:
        try:
            package = importlib.import_module(f"syncem.{layer}")
        except ModuleNotFoundError:
            continue
        names += [f"syncem.{layer}"] + [
            f"syncem.{layer}.{i.name}"
            for i in pkgutil.iter_modules(package.__path__)
            if not i.name.startswith("_")
        ]
    return tuple(sorted(names))


NEW_LAYERS = (
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
    "front",
)
MODEL_MODULES = _public_modules()
MODULES = MODEL_MODULES + _new_layer_modules()


def _public_objects(module):
    names = getattr(module, "__all__", None)
    if names is None:
        names = [n for n in vars(module) if not n.startswith("_")]
    for name in names:
        obj = getattr(module, name, None)
        if not (inspect.isclass(obj) or inspect.isfunction(obj)):
            continue
        if not getattr(obj, "__module__", "").startswith("syncem.model"):
            continue
        yield name, obj


OBJECTS = tuple(
    (module_name, name)
    for module_name in MODULES
    for name, _ in _public_objects(importlib.import_module(module_name))
)


def test_public_surface_is_nonempty():
    assert len(MODULES) >= 17
    assert len(OBJECTS) >= 80


@pytest.mark.parametrize("module_name", MODULES)
def test_module_docstring_states_label_units_shapes_and_limits(module_name):
    doc = importlib.import_module(module_name).__doc__ or ""
    missing = [
        what
        for what, rx in (
            ("LABEL", re.compile(r"LABEL")),
            ("units", UNITS_RX),
            ("shapes", SHAPE_RX),
            ("not certified", NOT_CERTIFIED_RX),
        )
        if not rx.search(doc)
    ]
    assert not missing, f"{module_name} module docstring lacks {missing}"


@pytest.mark.parametrize(("module_name", "name"), OBJECTS)
def test_public_object_docstring(module_name, name):
    module = importlib.import_module(module_name)
    obj = getattr(module, name)
    own = obj.__dict__.get("__doc__") if inspect.isclass(obj) else obj.__doc__
    assert own and own.strip(), f"{module_name}.{name} has no docstring"
    doc = inspect.getdoc(obj)
    if inspect.isclass(obj) and not hasattr(obj, "LABEL"):
        assert LABEL_RX.search(doc), f"{module_name}.{name}: no manuscript label"
    limits = NOT_CERTIFIED_RX.search(doc) or (
        MODULE_POINTER_RX.search(doc) and NOT_CERTIFIED_RX.search(module.__doc__)
    )
    assert limits, f"{module_name}.{name}: no 'not certified' statement"


# -- README / DESIGN against the current signatures (round-2 documentation) ---------

ROOT = Path(__file__).resolve().parents[2]


def test_new_layer_labels_match_the_manuscript_snapshot():
    snapshot = json.loads(
        (ROOT / "tests" / "v080" / "data" / "manuscript_labels.json").read_text()
    )
    labels = set(snapshot["labels"])
    unknown = []
    for name in _new_layer_modules():
        mod = importlib.import_module(name)
        source = Path(mod.__file__).read_text()
        docs = mod.__doc__ or ""
        assert ";" not in docs, name
        cited = re.findall(r"``((?:eq|sec|app|tab|fig):[^\x60]+)``", docs)
        cited.extend(re.findall(r"^LABEL ((?:eq|sec|app|tab|fig):[^.\n]+)", docs, re.M))
        for node in ast.walk(ast.parse(source)):
            if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
                if node.target.id == "LABEL" and isinstance(node.value, ast.Constant):
                    cited.append(node.value.value)
        unknown.extend(
            (name, label)
            for label in cited
            if label not in labels and not label.startswith("[extension]")
        )
    assert not unknown, unknown


DOCUMENTED = (
    ("syncem.model.predict", "predict"),
    ("syncem.model.predict", "direct_channel_average"),
    ("syncem.model.basis", "build_basis"),
    ("syncem.model.basis", "basis_convergence"),
    ("syncem.model.fit.linear", "fit_linear"),
    ("syncem.model.fit.nonlinear", "fit_bfgs"),
    ("syncem.model.fit.nonlinear", "fit_nodal"),
    ("syncem.model.fit.reduction", "reduce_response"),
    ("syncem.model.fit.combinations", "fit_combinations"),
)
STALE = (
    "record(theta",  # ParameterMap.record() is theta-independent
    "record(*, theta",
    "Neither helper has a `predict` input",  # assumption_allowances exists
    "no `predict` input for an allowance",
    "does not\nre-check a basis-level zero",  # predict(samples=...) runs the check
    "does not re-check a basis-level zero",
    "BFGS and nodal fits report the\n  noise term only",
    "BFGS and nodal fits report the noise term only",
    "BFGS and\nnodal fits have no bias bound",
)


def _doc(name):
    return (ROOT / name).read_text(encoding="utf-8")


@pytest.mark.parametrize(("module_name", "name"), DOCUMENTED)
def test_design_names_every_keyword_of_the_documented_signatures(module_name, name):
    fn = getattr(importlib.import_module(module_name), name)
    design = _doc("docs/DESIGN.md")
    keywords = [
        p.name
        for p in inspect.signature(fn).parameters.values()
        if p.kind is inspect.Parameter.KEYWORD_ONLY
    ]
    missing = [k for k in keywords if f"{k}=" not in design and f"`{k}`" not in design]
    assert not missing, f"docs/DESIGN.md does not name {name}({missing})"


@pytest.mark.parametrize("doc", ["README.md", "docs/DESIGN.md"])
def test_docs_drop_statements_superseded_by_round_two(doc):
    text = _doc(doc)
    found = [phrase for phrase in STALE if phrase in text]
    assert not found, f"{doc} still says {found}"


def test_readme_shows_assumption_allowances_with_the_screen_bound():
    readme = _doc("README.md")
    assert "assumption_allowances=" in readme
    assert "screen_factorisation_bound(" in readme


def test_package_docstring_lists_every_public_model_module():
    """``syncem.__doc__`` names each public ``model`` and ``model.fit`` module."""
    import syncem

    doc = " ".join((syncem.__doc__ or "").split())
    missing = []
    for module_name in MODEL_MODULES:
        short = module_name.removeprefix("syncem.model.")
        if not re.search(rf"(?<![\w.]){re.escape(short)}\b", doc):
            missing.append(short)
    assert not missing, f"syncem module docstring omits {missing}"
