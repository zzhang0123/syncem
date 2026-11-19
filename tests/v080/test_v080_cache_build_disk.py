"""Build-level cases of the disk cache and the jet routines (design/0.8/SPEC.md decisions 160 and 161).

``cache=False`` writes nothing and compiles fresh routines, and ``cache=True`` shares one routine between
builds with equal kernel, phase and rule fingerprints. A fresh process with another ``PYTHONHASHSEED`` reads the
S0 and S1 tables of the first from disk, computes equal keys and gives spectra equal by ``tobytes()``. The slow
matrix runs ``PYTHONHASHSEED`` 1 to 4 on a kernel with frozenset constants. Every assertion is an exact identity.
Subprocesses set ``SYNCEM_CACHE`` to the test's ``tmp_path``. T14 item 7: a build without a fingerprint has the
key ``None``, a "not cached: <reason>" note and stores nothing in memory or on disk, and its cacheable twin
stores its memo entry and its S0 table.
"""

import json
import os
import subprocess
import sys
import textwrap
from pathlib import Path

import jax.numpy as jnp
import numpy as np
import pytest

from syncem.engine import provider_jet
from syncem.engine.average import AveragedKernel, fixed_rule
from syncem.engine.basis import build_from_averaged
from syncem.engine.cache import default_cache
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from tests.v080.test_v080_cache import (
    A2,
    GAMMA2,
    _global_rule,
    _stack,
    ab_kernel,
    build,
    held_kernel,
    memo_key,
    paper_case,
    paper_channels,
)
from tests.v080.test_v080_stack import small_channels

REPO = Path(__file__).resolve().parents[2]


def _avg():
    return AveragedKernel(ab_kernel, [("b", fixed_rule([1.0, 2.0], [0.5, 0.5]))])


def _files(root):
    return sorted(p for p in Path(root).rglob("*") if p.is_file())


@pytest.fixture
def traces(monkeypatch):
    """One entry per trace of a jet routine (``taylor_coefficients`` runs once per trace)."""
    seen = []
    original = provider_jet.taylor_coefficients

    def counting(*args, **kwargs):
        seen.append(1)
        return original(*args, **kwargs)

    monkeypatch.setattr(provider_jet, "taylor_coefficients", counting)
    return seen


@pytest.mark.parametrize("cache", [False, True])
def test_routines_are_fresh_without_the_cache_and_shared_with_it(
    cache, traces, tmp_path
):
    """Two channel sets of one shape miss the memo. With the cache they share one routine and its trace."""
    avg, lattice = _avg(), Lattice(a=Cells.log(1.0, 30.0, 2, order=2))
    for ch in (small_channels(3, 8), small_channels(3, 8)[::-1]):
        build_from_averaged(avg, ch, lattice, cache=cache)
    assert len(traces) == (1 if cache else 2)
    if not cache:
        assert _files(tmp_path / "syncem_cache") == []
        assert default_cache().events["routines"] == 0 and default_cache().nbytes == 0


def test_cache_false_on_the_stack_writes_nothing(tmp_path):
    avg, lattice, factor = paper_case("E1")
    basis = build_from_averaged(
        avg, paper_channels(4), lattice, azimuth_factor=factor, cache=False
    )
    assert basis.key is None and _files(tmp_path / "syncem_cache") == []
    assert default_cache().nbytes == 0


CHILD = """
import json, sys
import numpy as np
sys.path.insert(0, "tests/v080")
from syncem.engine.basis import build_from_averaged
from tests.v080.test_v080_cache import paper_case, paper_channels
avg, lattice, factor = paper_case("E1")
basis = build_from_averaged(avg, paper_channels(8), lattice, azimuth_factor=factor)
np.save(sys.argv[1], np.asarray(basis.spectra0))
np.save(sys.argv[2], np.asarray(basis.spectra2))
print(json.dumps({"keys": dict(basis.record.keys), "events": dict(basis.record.events)}))
"""


def _child(script, args, env_extra, tmp_path):
    env = {**os.environ, "SYNCEM_CACHE": str(tmp_path / "syncem_cache"), **env_extra}
    env["PYTHONPATH"] = os.pathsep.join([str(REPO), str(REPO / "tests" / "v080")])
    run = subprocess.run(
        [sys.executable, "-c", textwrap.dedent(script), *args],
        cwd=REPO,
        env=env,
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert run.returncode == 0, run.stderr[-4000:]
    return json.loads(run.stdout.strip().splitlines()[-1])


def test_a_fresh_process_under_another_hash_seed_reads_the_disk_and_agrees(tmp_path):
    avg, lattice, factor = paper_case("E1")
    here = build_from_averaged(avg, paper_channels(8), lattice, azimuth_factor=factor)
    assert {p.parent.name for p in _files(tmp_path / "syncem_cache")} == {"s0", "s1"}
    s0, s2 = tmp_path / "s0.npy", tmp_path / "s2.npy"
    seed = "7" if os.environ.get("PYTHONHASHSEED") != "7" else "11"
    there = _child(CHILD, [str(s0), str(s2)], {"PYTHONHASHSEED": seed}, tmp_path)
    assert there["keys"] == dict(here.record.keys)
    assert there["events"]["S1 tables read"] >= 1
    assert there["events"]["S0 tables built"] == there["events"]["S1 tables built"] == 0
    assert np.load(s0).tobytes() == np.asarray(here.spectra0).tobytes()
    assert np.load(s2).tobytes() == np.asarray(here.spectra2).tobytes()


SEEDED = """
import json
import jax.numpy as jnp
from syncem.base.user import kernel
from syncem.base.variables import Variable
from syncem.engine.average import AveragedKernel, fixed_rule
from syncem.engine.basis import build_from_averaged
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from tests.v080.test_v080_stack import small_channels

@kernel(variables=[Variable("a", kind="positive"), Variable("b", kind="positive")], stokes=("I",))
def tagged(nu, a, b):
    scale = 2.0 if "beta" in {"alpha", "beta", "gamma", "delta"} else 1.0
    return {"I": scale * a * b * jnp.ones_like(nu)}

avg = AveragedKernel(tagged, [("b", fixed_rule([1.0, 2.0], [0.5, 0.5]))])
lattice = Lattice(a=Cells.log(1.0, 30.0, 2, order=2))
basis = build_from_averaged(avg, small_channels(3, 8), lattice)
print(json.dumps({"keys": dict(basis.record.keys), "bits": basis.spectra0.tobytes().hex()}))
"""


@pytest.mark.slow
def test_frozenset_constants_give_equal_keys_under_hash_seeds_1_to_4(tmp_path):
    runs = []
    for seed in "1234":
        where = tmp_path / seed
        where.mkdir()
        runs.append(_child(SEEDED, [], {"PYTHONHASHSEED": seed}, where))
    assert runs[0]["keys"]["memo"] is not None
    assert all(r == runs[0] for r in runs[1:])


# ------------------------------------------------------------------ 7. not cacheable


def _bare_rule(point):
    return jnp.asarray([1.0, 2.0]), jnp.asarray([0.5, 0.5])


NOT_CACHEABLE = {
    "user class global": lambda: (
        AveragedKernel(held_kernel, [("b", _global_rule())]),
        Lattice(a=A2),
        None,
    ),
    "bare callable": lambda: (
        AveragedKernel(ab_kernel, [("b", _bare_rule)]),
        Lattice(a=A2),
        None,
    ),
    "lattice key None": lambda: (
        _stack()[0],
        Lattice(gamma=GAMMA2).with_key(None),
        0.4,
    ),
}
CACHEABLE = {
    "jet": lambda: (
        AveragedKernel(ab_kernel, [("b", _global_rule())]),
        Lattice(a=A2),
        None,
    ),
    "stack": _stack,
}


@pytest.mark.parametrize("name", list(NOT_CACHEABLE))
def test_not_cacheable_gives_key_none_a_reason_and_stores_nothing(name, tmp_path):
    basis = build(NOT_CACHEABLE[name](), small_channels(3, 8))
    assert basis.key is None and memo_key(basis) is None
    assert [n for n in basis.record.notes if n.startswith("not cached: ")]
    assert default_cache().nbytes == 0 and _files(tmp_path / "syncem_cache") == []


@pytest.mark.parametrize("name", list(CACHEABLE))
def test_the_cacheable_twin_stores_its_memo_and_tables(name, tmp_path):
    basis = build(CACHEABLE[name](), small_channels(3, 8))
    assert basis.key is not None and not [
        n for n in basis.record.notes if "not cached" in n
    ]
    assert default_cache().peek("memo", memo_key(basis)) == "memory"
    stages = {p.parent.name for p in _files(tmp_path / "syncem_cache")}
    assert stages == ({"s0"} if name == "stack" else set())
