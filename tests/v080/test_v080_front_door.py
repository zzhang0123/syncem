"""The lazy front door of ``syncem`` (design/0.8/SPEC.md Sec. 9.1): new names load their layer on first use."""

import subprocess
import sys

import pytest

import syncem

NAMES = {
    "Channels": "syncem.base.channels",
    "Variable": "syncem.base.variables",
    "Quantity": "syncem.units",
    "Cells": "syncem.lattice.cells",
    "CellTable": "syncem.lattice.table",
    "Legendre": "syncem.lattice.representations",
    "AzimuthHarmonic": "syncem.lattice.representations",
    "Lattice": "syncem.lattice.lattice",
    "MomentIndex": "syncem.lattice.index",
    "Moments": "syncem.lattice.moments",
    "ContinuumKernel": "syncem.base.continuum",
    "kernel": "syncem.base.user",
    "UserKernel": "syncem.base.user",
    "Kernel": "syncem.base.kernel",
    "Modes": "syncem.base.kernel",
    "PhaseLaw": "syncem.base.phase",
    "FaradayRotation": "syncem.base.phase",
    "Similarity": "syncem.base.capabilities",
    "Beaming": "syncem.base.capabilities",
    "UnitError": "syncem.values.errors",
    "RoleError": "syncem.values.errors",
    "RegimeError": "syncem.values.errors",
    "SpecifiedMismatch": "syncem.values.errors",
    "NothingToPredict": "syncem.values.errors",
    "DataError": "syncem.values.errors",
    "DepthExpansionError": "syncem.values.errors",
    "SizeError": "syncem.values.errors",
    "ImproperPosterior": "syncem.values.errors",
}


@pytest.mark.parametrize("name", sorted(NAMES))
def test_front_door_names(name):
    module = __import__(NAMES[name], fromlist=[name])
    assert getattr(syncem, name) is getattr(module, name)
    assert name in dir(syncem)


def test_unknown_name_raises_attribute_error():
    with pytest.raises(AttributeError, match="no attribute 'Population'"):
        syncem.Population  # noqa: B018  (arrives with the population step)


def test_import_syncem_loads_neither_the_model_nor_scipy():
    code = (
        "import sys, syncem\n"
        "assert 'syncem.model' not in sys.modules, 'model'\n"
        "assert 'scipy' not in sys.modules, 'scipy'\n"
        "syncem.Quantity, syncem.UnitError\n"
        "assert 'syncem.model' not in sys.modules, 'model after Quantity'\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True)
