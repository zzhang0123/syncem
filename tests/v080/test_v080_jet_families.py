"""Jet programs on every channel family (design/0.8/SPEC.md decision 171, step 5b-3 review).

``Channels.from_bandpass``, ``from_table`` and ``from_callable`` hold per-channel static fields (``table``,
``response_fn``, ``derivative_bound``). A one-channel slice of them has a static part of its own, so before
decision 171 every channel compiled its own program and the 4 trace slots missed on every call. The chunk
views are now ``ResponseView`` objects: nodes, weights and the host response weights as array leaves, and the
chunk size as the only static field. The tests check one compile per chunk size on each family, no compile on
a repeated call, bitwise channel-subset independence, one program shared across families, and the fallback
for channel objects without response weights.
"""

from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.engine._jet_routine import ResponseView, _key, chunk_views
from syncem.engine.average import AveragedKernel, fixed_rule
from syncem.engine.provider_jet import jet_lattice, make_routine
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice

N = 6
CENTRES = np.geomspace(1.0, 10.0, N)  # GHz


def averaged() -> AveragedKernel:
    return AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        [
            ("B", fixed_rule(np.array([1e-5]), np.array([1.0]))),
            ("eta", fixed_rule(np.array([0.5]), np.array([1.0]))),
        ],
    )


def lattice(order: int = 1) -> Lattice:
    return Lattice(gamma=Cells.log(300, 3e4, 3, order=order))


def bandpass(n_nu: int = 16) -> Channels:
    tables = [
        (np.array([c * 0.8, c * 0.9, c * 1.1, c * 1.2]), np.array([0.0, 1.0, 1.0, 0.0]))
        for c in CENTRES
    ]
    return Channels.from_bandpass(tables, unit="GHz", n_nu=n_nu)


def table(n_nu: int = 16) -> Channels:
    grid = np.geomspace(0.5e9, 20e9, 200)
    resp = np.array(
        [np.clip(1 - np.abs(np.log(grid / (c * 1e9))) / 0.2, 0, None) for c in CENTRES]
    )
    return Channels.from_table(grid, resp, smoothness=0, n_nu=n_nu)


def callable_(n_nu: int = 16) -> Channels:
    fns = tuple(
        (lambda nu, c=c: jnp.exp(-(((nu - c * 1e9) / (0.1 * c * 1e9)) ** 2)))
        for c in CENTRES
    )
    support = np.stack([CENTRES * 0.7e9, CENTRES * 1.3e9], axis=1)
    return Channels.from_callable(fns, support, 0, n_nu=n_nu)


def log(n_nu: int = 16) -> Channels:
    return Channels.log(1.0, 10.0, N, unit="GHz", half_width=0.2, n_nu=n_nu)


FAMILIES = {"log": log, "bandpass": bandpass, "table": table, "callable": callable_}


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("n_nu", [16, 70])
def test_each_family_compiles_one_program_per_chunk_size(family, n_nu):
    ch, avg, lat = FAMILIES[family](n_nu), averaged(), lattice()
    routine = make_routine(("gamma",), ("log",), ((0,), (1,)))
    first, again = {}, {}
    jet_lattice(avg, ch, lat, events=first, routine=lambda *a: routine)
    jet_lattice(avg, ch, lat, events=again, routine=lambda *a: routine)
    n_chunks = 1 if n_nu <= 64 else 2
    assert first["jet calls"] == 3 * N * n_chunks
    assert first["jet compiles"] == 1 and again["jet compiles"] == 0
    assert routine.programs == 1


@pytest.mark.parametrize("family", sorted(FAMILIES))
@pytest.mark.parametrize("rows", [[N - 1], [N - 1, 0, 2], [3, 3, 1]])
def test_family_columns_do_not_depend_on_the_channel_set(family, rows):
    ch, avg, lat = FAMILIES[family](), averaged(), lattice(2)
    full = jet_lattice(avg, ch, lat)
    sub = jet_lattice(avg, ch[rows], lat)
    for a, b in zip(full, sub):
        assert np.asarray(a)[:, rows].tobytes() == np.asarray(b).tobytes()


def test_views_of_every_family_share_one_key():
    probe = (None, None, jnp.asarray([1.0]), jnp.asarray([1.0]))
    keys = {
        _key((probe[0], view, *probe[2:]))
        for make in FAMILIES.values()
        for j in range(N)
        for view in chunk_views(make(), j)
    }
    assert len(keys) == 1


def test_a_view_holds_the_response_weights_of_its_row():
    ch = table(70)
    views = chunk_views(ch, 4)
    assert all(isinstance(v, ResponseView) and v.n_ch == 1 for v in views)
    measure = np.concatenate([np.asarray(v.response_weights())[0] for v in views])
    assert measure[:70].tobytes() == np.asarray(ch[[4]].response_weights())[0].tobytes()
    assert np.all(measure[70:] == 0.0)


@dataclasses.dataclass(frozen=True)
class _Plain:
    """A channel object with nodes and weights only."""

    nodes: object
    weights: object
    n_nu: int

    @property
    def n_ch(self) -> int:
        return int(np.shape(self.nodes)[0])

    def __getitem__(self, rows):
        return dataclasses.replace(
            self, nodes=self.nodes[rows], weights=self.weights[rows]
        )


def test_a_channel_object_without_response_weights_gives_slices():
    nodes = jnp.asarray(np.linspace(1e9, 2e9, 140).reshape(2, 70))
    plain = _Plain(nodes, jnp.full((2, 70), 1e7), 70)
    views = chunk_views(plain, 1)
    assert len(views) == 2 and all(isinstance(v, _Plain) for v in views)
    assert all(v.nodes.shape == (1, 35) and v.n_nu == 35 for v in views)
    assert np.asarray(views[0].nodes)[0, 0] == np.asarray(nodes)[1, 0]
