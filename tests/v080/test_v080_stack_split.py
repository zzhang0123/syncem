"""Step 5b-0 of design/0.8/SPEC.md (decision 145): the stack split, the hoisted channel arrays and the mode report.

The reference ``data/stack_5a_reference.npz`` (sha256
547abefd03ec1024e66d6e20c6499087179381a5ff80f4e97c542229e52be738) holds the ``(I, P)`` rows that ``stack_cell``
gives with the S0 span (-30, 20) of decision 158 on ``small_channels()`` (5 channels of 32 nodes). It was written
in step 5b-2. The cases are cells 0, 9 and 18 at order 4 of ``Cells.log(300, 3e6, 19)`` in per-node mode (``B`` on
16 nodes given ``eta`` on 12 nodes) and in cached-S1 mode (``B`` on 16 Gauss-Legendre nodes on [4, 6] uG, ``eta``
on 12 nodes on [0, 0.5], ``FaradayRotation(GaussianScreen(8, 0.5))``), and the first, middle and last ``(gamma,
B)`` cells of a 3 by 3 field-scaling lattice at order 4. Each case uses one memo across its cells, in that order.
The rows of the current code equal it bit for bit.

The reference ``data/stack_5a_depth_reference.npz`` (sha256
f679748eb1477002d83c8405043e6bd34aaab4592517aeeb0e55ab5014c4b296) holds the rows for depth on 4 nodes given ``B``
on 3 nodes, ``eta`` on 3 nodes and ``FaradayRotation()``, cells 0, 1 and 2 of ``Cells.log(300, 3e6, 3)`` at order 2
in per-node mode (``test_v080_stack.test_depth_given_averaged_b``). Its depth varies across the field nodes, so it
checks the values of the per-depth channel arrays.

The files ``data/stack_5a_reference_step5a.npz`` (sha256
a62c6b22783d5fd5934bef3b9f139eb3e4ed9ca46317ceb919f68f5045cf322e) and ``data/stack_5a_depth_reference_step5a.npz``
(sha256 ada832335ccdaa9052fdd46478a078aa2d679782845a81b906916b2ae04ce8ac) are the rows of commit 2cc657f (step 5a),
with the S0 span from ``ln 1e-12`` to ``ln 1e7``. The step-5b-0 refactor reproduced them bit for bit. The rows of
the S0 span of decision 158 moved by at most 6.2e-15 of the largest value of their row (measured 6.216e-15 and
9.08e-16, 2026-11-19). ``STEP_5A_TOL = 1e-14`` is a margin of 1.6 on that measurement. Their distance to the rows
with S1 by quadrature changed by at most 5.3e-16 of the row maximum (2026-11-19, 1-minute load 20.6 to 24.9).
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
REFERENCE_SHA256 = "547abefd03ec1024e66d6e20c6499087179381a5ff80f4e97c542229e52be738"
DEPTH_REFERENCE = REFERENCE.with_name("stack_5a_depth_reference.npz")
DEPTH_REFERENCE_SHA256 = (
    "f679748eb1477002d83c8405043e6bd34aaab4592517aeeb0e55ab5014c4b296"
)
STEP_5A = {
    REFERENCE: (
        REFERENCE.with_name("stack_5a_reference_step5a.npz"),
        "a62c6b22783d5fd5934bef3b9f139eb3e4ed9ca46317ceb919f68f5045cf322e",
    ),
    DEPTH_REFERENCE: (
        REFERENCE.with_name("stack_5a_depth_reference_step5a.npz"),
        "ada832335ccdaa9052fdd46478a078aa2d679782845a81b906916b2ae04ce8ac",
    ),
}
STEP_5A_TOL = 1e-14
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
def test_reference_file_has_its_digest(path, digest):
    assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


@pytest.mark.parametrize("path", [REFERENCE, DEPTH_REFERENCE])
def test_references_are_within_the_measured_move_of_the_step_5a_rows(path):
    old_path, digest = STEP_5A[path]
    assert hashlib.sha256(old_path.read_bytes()).hexdigest() == digest
    with np.load(path) as new, np.load(old_path) as old:
        assert sorted(new.files) == sorted(old.files)
        for key in new.files:
            a, b = new[key], old[key]
            scale = np.abs(a).max(axis=tuple(range(1, a.ndim)), keepdims=True)
            assert a.shape == b.shape and np.max(np.abs(a - b) / scale) < STEP_5A_TOL


@pytest.mark.parametrize("name", list(CASES))
def test_rows_equal_the_reference_bit_for_bit(name):
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
