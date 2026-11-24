"""Reference integral with only the azimuth Fourier truncation (SPEC Sec. 7.2).

The quadrature is over the supplied physical measure, including all spatial
variation. The periodic projection shares the grid of the column provider.
Its difference from the full direct integral measures omitted angular
orders on those panels. Units and mode shapes are those of direct integration.
Not certified: the physical kernel and quadrature convergence.
"""

from __future__ import annotations

import numpy as np

from ..base.kernel import Modes
from .direct import _continued, _evaluate, _runner, _zeros, sum_modes
from .provider_fourier import sample_count


def projected_integral(averaged, channels, panels, m_max, *, b_perp_max=None):
    """Integrate the angularly projected kernel over the physical panels."""
    averaged = _continued(averaged, channels, b_perp_max)
    run = _runner(averaged, channels, 4)
    n = sample_count(m_max)
    name = averaged.kernel.azimuth
    total = _zeros(averaged, channels)
    for points, weights in panels:
        phi = points[name]
        spin = np.exp(2j * phi)
        count = phi.size
        for angle in 2 * np.pi * np.arange(n) / n:
            difference = np.remainder(phi - angle + np.pi, 2 * np.pi) - np.pi
            turns = difference / (2 * np.pi)
            window = (2 * m_max + 1) * np.sinc((2 * m_max + 1) * turns) / np.sinc(turns)
            fixed = {**points, name: np.full(count, angle)}
            nodes = _evaluate(averaged, channels, run, fixed, count, None, False)
            phase = spin.reshape((count,) + (1,) * (np.ndim(nodes.P) - 1))
            natural = Modes(nodes.I, nodes.V, nodes.P * phase)
            part = sum_modes(natural, weights * window / n)
            total = Modes(*(a + b for a, b in zip(total, part, strict=True)))
    return total
