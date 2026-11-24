"""Weighted single-reference geometry for joint samples.

LABEL [extension].

Single-reference lattice rows put the physical coordinates in the kernel's
canonical units. The resulting moment vector has shape (n_row,). The
coordinate conversion carries no certified accuracy of the single-reference
approximation.
"""

from __future__ import annotations

import numpy as np

from ..base.variables import SYNCHROTRON_VARIABLES
from ..lattice.cells import Cells
from ..lattice.lattice import Lattice
from ..lattice.representations import AzimuthHarmonic, Legendre
from ..values.errors import RoleError
from .blocks import Samples


def single_lattice(*, about, order=2, degree=8):
    """Use sample means and ranges as the references and physical panels."""
    if not isinstance(about, Samples):
        raise TypeError("Lattice.single needs a Samples declaration")
    known = {
        name: SYNCHROTRON_VARIABLES[name]
        for name in about.variables
        if name in SYNCHROTRON_VARIABLES
    }
    if set(known) != set(about.variables):
        raise RoleError("non-synchrotron samples need an explicit one-cell lattice")
    placed = about.placed(known)
    points, probability = placed.rule()
    axes = {}
    for name, values in points.items():
        variable = known[name]
        if variable.kind == "cosine":
            axes[name] = Legendre(degree)
            continue
        if variable.kind == "angle":
            axes[name] = AzimuthHarmonic()
            continue
        reference = float(np.dot(values, probability))
        lo, hi = (float(a) for a in placed.support[name])
        scale = max(reference - lo, hi - reference)
        if scale == 0:
            scale = abs(reference) * 0.01 if reference != 0 else 0.01
            lo, hi = reference - scale, reference + scale
            if variable.kind == "positive":
                lo = max(reference / 2, lo)
        axes[name] = Cells.single(
            reference, scale, support=(lo, hi), order=order, unit=variable.unit or None
        )
    return Lattice(order=order, depth_in_total=True, weights=(1.0,), **axes)
