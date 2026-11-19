"""Step 5b-0 of design/0.8/SPEC.md (decision 145): the stack split, the hoisted channel arrays and the mode report.

The reference ``data/stack_5a_reference.npz`` (sha256
a62c6b22783d5fd5934bef3b9f139eb3e4ed9ca46317ceb919f68f5045cf322e) holds the ``(I, P)`` rows that ``stack_cell``
gave at commit 2cc657f (step 5a), before the refactor, on ``small_channels()`` (5 channels of 32 nodes). The cases
are cells 0, 9 and 18 at order 4 of ``Cells.log(300, 3e6, 19)`` in per-node mode (``B`` on 16 nodes given ``eta``
on 12 nodes) and in cached-S1 mode (``B`` on 16 Gauss-Legendre nodes on [4, 6] uG, ``eta`` on 12 nodes on [0, 0.5],
``FaradayRotation(GaussianScreen(8, 0.5))``), and the first, middle and last ``(gamma, B)`` cells of a 3 by 3
field-scaling lattice at order 4. Each case uses one memo across its cells, in that order. The refactor moves code
and computes the channel arrays once per walk and depth value, so the rows are equal bit for bit.

The reference ``data/stack_5a_depth_reference.npz`` (sha256
ada832335ccdaa9052fdd46478a078aa2d679782845a81b906916b2ae04ce8ac) holds the rows of the same commit for depth on 4
nodes given ``B`` on 3 nodes, ``eta`` on 3 nodes and ``FaradayRotation()``, cells 0, 1 and 2 of
``Cells.log(300, 3e6, 3)`` at order 2 in per-node mode (``test_v080_stack.test_depth_given_averaged_b``). Its depth
varies across the field nodes, so it checks the values of the per-depth channel arrays.
"""

import hashlib
from pathlib import Path

import numpy as np
import pytest

import syncem.engine._stack_series as series_module
import syncem.engine.provider_stack as provider_module
from syncem.base.continuum import ContinuumKernel
from syncem.base.phase import FaradayRotation
from syncem.engine._stack_series import StackRefused, stack_cell
from syncem.engine.average import AveragedKernel, uniform_rule
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from tests.v080.test_v080_stack import screen, small_channels

REFERENCE = Path(__file__).resolve().parent / "data" / "stack_5a_reference.npz"
REFERENCE_SHA256 = "a62c6b22783d5fd5934bef3b9f139eb3e4ed9ca46317ceb919f68f5045cf322e"
DEPTH_REFERENCE = REFERENCE.with_name("stack_5a_depth_reference.npz")
DEPTH_REFERENCE_SHA256 = (
    "ada832335ccdaa9052fdd46478a078aa2d679782845a81b906916b2ae04ce8ac"
)
K = ContinuumKernel(x_min=1e-12)
CELLS = (0, 9, 18)


def per_node_case():
    b = uniform_rule(lambda p: 4e-6 * (1 + p["eta"]), 6e-6, 16, parents=("eta",))
    avg = AveragedKernel(
        K, [("eta", uniform_rule(0.0, 0.5, 12)), ("B", b)], phase=screen()
    )
    return avg, Lattice(gamma=Cells.log(300, 3e6, 19, order=4)), CELLS


def cached_case():
    avg = AveragedKernel(
        K,
        [("B", uniform_rule(4e-6, 6e-6, 16)), ("eta", uniform_rule(0.0, 0.5, 12))],
        phase=screen(),
    )
    return avg, Lattice(gamma=Cells.log(300, 3e6, 19, order=4)), CELLS


def field_case():
    avg = AveragedKernel(K, [("eta", uniform_rule(0.0, 0.5, 6))], phase=screen())
    lattice = Lattice(
        gamma=Cells.log(300, 3e6, 3, order=4),
        B=Cells.log("2 uG", "8 uG", 3, order=4),
        order=4,
    )
    return avg, lattice, (0, lattice.n_cells // 2, lattice.n_cells - 1)


def depth_case():
    depth = uniform_rule(
        lambda p: 6.0 + 4e5 * p["B"], lambda p: 10.0 + 4e5 * p["B"], 4, parents=("B",)
    )
    avg = AveragedKernel(
        K,
        [
            ("B", uniform_rule(4e-6, 6e-6, 3)),
            ("eta", uniform_rule(0.0, 0.5, 3)),
            ("depth", depth),
        ],
        phase=FaradayRotation(),
    )
    lattice = Lattice(gamma=Cells.log(300, 3e6, 3))
    return avg, lattice, tuple(range(lattice.n_cells))


CASES = {
    "per_node": (per_node_case, "per-node", REFERENCE),
    "cached": (cached_case, "cached-S1", REFERENCE),
    "field": (field_case, "field-scaling", REFERENCE),
    "depth": (depth_case, "per-node", DEPTH_REFERENCE),
}


@pytest.mark.parametrize(
    "path, digest",
    [(REFERENCE, REFERENCE_SHA256), (DEPTH_REFERENCE, DEPTH_REFERENCE_SHA256)],
)
def test_reference_file_is_the_step_5a_output(path, digest):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("name", list(CASES))
def test_rows_equal_the_step_5a_output_bit_for_bit(name):
    build, mode, path = CASES[name]
    avg, lattice, cells = build()
    ch, memo, record = small_channels(), {}, []
    with np.load(path) as ref:
        for c in cells:
            i, p = stack_cell(avg, ch, lattice, c, memo=memo, record=record)
            assert np.array_equal(np.asarray(i), ref[f"{name}_{c}_I"]), (name, c)
            assert np.array_equal(np.asarray(p), ref[f"{name}_{c}_P"]), (name, c)
    assert record == [mode] * len(cells)


def _small(levels, phase=None):
    avg = AveragedKernel(K, levels, phase=screen() if phase is None else phase)
    return avg, Lattice(gamma=Cells.log(300, 3e6, 3))


def test_record_receives_the_mode():
    ch, record = small_channels(3, 16), []
    b = uniform_rule(lambda p: 4e-6 * (1 + p["eta"]), 6e-6, 4, parents=("eta",))
    avg, lattice = _small([("eta", uniform_rule(0.0, 0.5, 4)), ("B", b)])
    stack_cell(avg, ch, lattice, 0, record=record)
    avg, lattice = _small(
        [("B", uniform_rule(4e-6, 6e-6, 4)), ("eta", uniform_rule(0.0, 0.5, 4))]
    )
    stack_cell(avg, ch, lattice, 0, record=record)
    avg, lattice, _ = field_case()
    stack_cell(avg, ch, lattice, 0, record=record)
    assert record == ["per-node", "cached-S1", "field-scaling"]


def test_a_refused_cell_records_nothing():
    record = []
    bare = AveragedKernel(K, [("B", lambda p: (np.ones(1), np.ones(1)))])
    with pytest.raises(StackRefused, match="bare function"):
        stack_cell(
            bare,
            small_channels(3, 16),
            Lattice(gamma=Cells.log(300, 3e6, 2)),
            0,
            record=record,
        )
    assert record == []


def _count_arrays(monkeypatch):
    """Count ``_channel_arrays`` by depth through both names, the walk's import and the one inside the provider."""
    calls = []
    original = provider_module._channel_arrays

    def counted(channels, phase, depth=0.0):
        calls.append(float(depth))
        return original(channels, phase, depth)

    monkeypatch.setattr(series_module, "_channel_arrays", counted)
    monkeypatch.setattr(provider_module, "_channel_arrays", counted)
    return calls


def test_channel_arrays_once_per_walk_in_per_node_mode(monkeypatch):
    """``B`` on 4 nodes given ``eta`` on 4 nodes: 16 leaves at depth 0, one computation per cell."""
    calls = _count_arrays(monkeypatch)
    b = uniform_rule(lambda p: 4e-6 * (1 + p["eta"]), 6e-6, 4, parents=("eta",))
    avg, lattice = _small([("eta", uniform_rule(0.0, 0.5, 4)), ("B", b)])
    memo = {}
    for c in range(lattice.n_cells):
        stack_cell(avg, small_channels(3, 16), lattice, c, memo=memo)
    assert calls == [0.0] * lattice.n_cells


def test_channel_arrays_once_per_depth_value(monkeypatch):
    """Depth given an averaged ``B``: 3 field nodes by 4 depth nodes give 12 depth values, each computed once."""
    calls = _count_arrays(monkeypatch)
    depth = uniform_rule(
        lambda p: 6.0 + 4e5 * p["B"], lambda p: 10.0 + 4e5 * p["B"], 4, parents=("B",)
    )
    avg, lattice = _small(
        [
            ("B", uniform_rule(4e-6, 6e-6, 3)),
            ("eta", uniform_rule(0.0, 0.5, 3)),
            ("depth", depth),
        ],
        FaradayRotation(),
    )
    stack_cell(avg, small_channels(3, 16), lattice, 1)
    assert len(calls) == 12 and len(set(calls)) == 12
