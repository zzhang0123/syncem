"""Stored jet rows before and after the one-channel calls of design/0.8/SPEC.md decision 165 (step 5b-3).

``data/jet_5b2_reference.npz`` (sha256 b9f128cc3ba2cc30b91d164225c0b9d0fc3705b052d2322a931ec8264c990760) holds the
rows of ``jet_lattice`` at commit fcfe5cf, where one call evaluated every channel and every channel node.
``data/jet_5b3_reference.npz`` (sha256 f4bc743b8bf2d94af0d832f0c72310eb189ba54687544bcdca05b877f49582d5) holds
the rows after the change, with one call per (cell, channel, node chunk) on a ``ResponseView`` whose response
weights are evaluated on the host (decision 171). The cases:

* E1 (``B`` on 16 and ``eta`` on 12 nodes of the paper configuration, ``FaradayRotation(GaussianScreen(8,
  0.5))``), cells 0, 9 and 18 of 19 at orders 0, 2 and 4, on 5 channels of 32 nodes,
* E1 at order 2 on 3 channels of 65 nodes (two chunks of 33 after the change),
* a user kernel in three celled variables at total order 2, every cell,
* a user kernel with ``Legendre(4)`` on a cosine and the V block, every cell.

The current rows equal the 5b3 file bit for bit. They lie within ``REFERENCE_TOL`` of the 5b2 rows over the
largest value of each channel. Measured gaps to 5b2 (2026-11-19, jax 0.10.0 on CPU, 1-minute load 5.3): E1 at
orders 0, 2 and 4 1.5e-16 in I and 2.5e-15 in P, E1 on 65 nodes 3.4e-16 in I and 1.7e-15 in P, the
three-variable kernel 1.6e-16 and the Legendre kernel 2.8e-16. The response weights evaluated inside the program
(the first 5b3 file, sha256 e6c7b8a7...) gave E1 bit for bit at orders 0 and 2 against 5b2.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

import _study
from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.phase import FaradayRotation
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine.average import AveragedKernel, fixed_rule
from syncem.engine.provider_jet import jet_lattice
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import Legendre
from syncem.model.phase import GaussianScreen

DATA = Path(__file__).resolve().parent / "data"
REFERENCE_5B2 = DATA / "jet_5b2_reference.npz"
REFERENCE_5B3 = DATA / "jet_5b3_reference.npz"
SHA_5B2 = "b9f128cc3ba2cc30b91d164225c0b9d0fc3705b052d2322a931ec8264c990760"
SHA_5B3 = "f4bc743b8bf2d94af0d832f0c72310eb189ba54687544bcdca05b877f49582d5"
REFERENCE_TOL = 1e-13  # decision 165, over the channel maximum


def e1_kernel() -> AveragedKernel:
    cfg = _study.paper()
    b, wb = cfg.b_nodes_weights
    screen = GaussianScreen(cfg.screen_mean, cfg.screen_sd)
    return AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        [("B", fixed_rule(b * 1e-6, wb)), ("eta", fixed_rule(*cfg.eta_nodes_weights))],
        phase=FaradayRotation(screen=screen),
    )


def e1_lattice(order: int) -> Lattice:
    return Lattice(gamma=Cells.log(300, _study.paper().gamma_hi, 19, order=order))


def chans(n: int, n_nu: int, lo: float = 0.1, hi: float = 10.0) -> Channels:
    return Channels.log(lo, hi, n, unit="GHz", half_width=0.2, n_nu=n_nu)


def three_kernel() -> AveragedKernel:
    @kernel(
        variables=[
            Variable("a", kind="positive"),
            Variable("b", kind="positive"),
            Variable("c", kind="real"),
        ],
        stokes=("I", "P"),
    )
    def three(nu, a, b, c):
        x = nu / 1e9
        i = a**1.5 * b**-0.5 * x**0.3 * jnp.exp(-x * (0.5 + c) / (1.0 + a))
        return {"I": i, "P": (0.3 + 0.1j * c) * i}

    return AveragedKernel(three)


def three_lattice() -> Lattice:
    return Lattice(
        a=Cells.log(1.0, 4.0, 2),
        b=Cells.log(0.5, 2.0, 2),
        c=Cells.linear(0.1, 0.9, 2),
        order=2,
    )


def cosine_kernel() -> AveragedKernel:
    @kernel(
        variables=[Variable("g", kind="positive"), Variable("mu", kind="cosine")],
        stokes=("I", "P", "V"),
    )
    def beamed(nu, g, mu):
        x = nu / 1e9
        i = g**-1.2 * (1.0 + 0.3 * mu**2) * x**-0.7 * jnp.exp(-x / (5.0 * g))
        return {"I": i, "P": 0.5 * (1.0 - mu**2) * i, "V": 0.1 * mu * g**-0.5 * i}

    return AveragedKernel(beamed, legendre={"mu": (4, 16)})


def cosine_lattice() -> Lattice:
    return Lattice(g=Cells.log(1.0, 100.0, 3, order=2), mu=Legendre(4))


def rows() -> dict[str, np.ndarray]:
    """The jet rows of every case, by name."""
    out: dict[str, np.ndarray] = {}
    avg = e1_kernel()
    for order in (0, 2, 4):
        i, p = jet_lattice(avg, chans(5, 32), e1_lattice(order), cells=[0, 9, 18])
        out[f"e1_N{order}_I"], out[f"e1_N{order}_P"] = np.asarray(i), np.asarray(p)
    i, p = jet_lattice(avg, chans(3, 65), e1_lattice(2), cells=[0, 9, 18])
    out["e1_65_I"], out["e1_65_P"] = np.asarray(i), np.asarray(p)
    i, p = jet_lattice(three_kernel(), chans(5, 32), three_lattice())
    out["three_I"], out["three_P"] = np.asarray(i), np.asarray(p)
    i, p, v = jet_lattice(cosine_kernel(), chans(5, 32), cosine_lattice(), with_v=True)
    out["cosine_I"], out["cosine_P"], out["cosine_V"] = (
        np.asarray(a) for a in (i, p, v)
    )
    return out


def channel_gap(new: np.ndarray, old: np.ndarray) -> float:
    """Largest difference over the largest value of each channel (axis 1) of ``old``."""
    scale = np.abs(old).max(axis=0)
    diff = np.abs(new - old).max(axis=0)
    assert np.all(diff[scale == 0] == 0)
    return float(np.max(diff[scale > 0] / scale[scale > 0], initial=0.0))


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.fixture(scope="module")
def current() -> dict[str, np.ndarray]:
    return rows()


def test_the_stored_references_have_their_pinned_digests():
    assert _sha256(REFERENCE_5B2) == SHA_5B2
    assert _sha256(REFERENCE_5B3) == SHA_5B3


def test_the_rows_equal_the_5b3_reference_bit_for_bit(current):
    stored = np.load(REFERENCE_5B3)
    assert sorted(stored.files) == sorted(current)
    for name, value in current.items():
        assert value.dtype == stored[name].dtype and value.shape == stored[name].shape
        assert value.tobytes() == stored[name].tobytes(), name


def test_the_rows_lie_within_the_bound_of_the_5b2_reference(current):
    stored = np.load(REFERENCE_5B2)
    assert sorted(stored.files) == sorted(current)
    for name, value in current.items():
        assert value.shape == stored[name].shape, name
        assert np.all(np.isfinite(value)), name
        assert channel_gap(value, stored[name]) < REFERENCE_TOL, name
