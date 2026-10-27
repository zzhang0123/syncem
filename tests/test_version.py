"""The package version has one source and reaches every provenance record."""

import re
from pathlib import Path

import syncem
from syncem.model import _basis_core

ROOT = Path(__file__).resolve().parents[1]


def test_version_matches_pyproject():
    text = (ROOT / "pyproject.toml").read_text()
    declared = re.search(r'^version = "([^"]+)"', text, re.M).group(1)
    assert syncem.__version__ == declared


def test_provenance_version_is_the_package_version_not_installed_metadata():
    # importlib.metadata reports whatever distribution is installed, which can
    # be a stale build or, under the old name, an unrelated PyPI project;
    # provenance must use the version of the imported package.
    assert _basis_core.package_version() == syncem.__version__
