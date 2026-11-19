"""The study-oracle helper of decision 154 (design/0.8/SPEC.md Sec. 9.3 and Sec. 13).

The helper ``_study`` reads the study sources of ``studies/classical-models`` in place. It checks
them against pinned SHA-256 digests, imports them under a temporary ``sys.path`` entry, removes
their generic module names afterwards and builds kernel tables in memory. These tests check the
hash refusal, the interpreter state after a load, that no file under ``studies/classical-models/cache``
changes, and the copy of ``forward_spectra.npz`` under ``tests/acceptance/data``.
"""

import os
import shutil
import subprocess
import sys
import types
from dataclasses import replace

import numpy as np
import pytest

import _study

TABLE_CACHE = _study.STUDY / "cache"


def _cache_snapshot():
    if not TABLE_CACHE.exists():
        return None
    return sorted(
        (p.name, p.stat().st_size, p.stat().st_mtime_ns) for p in TABLE_CACHE.iterdir()
    )


def _interpreter_state():
    return (
        list(sys.path),
        {name: sys.modules.get(name) for name in _study.MODULE_NAMES},
        {var: os.environ.get(var) for var in _study.THREAD_VARS},
        sys.dont_write_bytecode,
    )


def _copy_sources(dst):
    for rel in _study.SOURCES:
        target = dst / rel
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(_study.STUDY / rel, target)


def _small_config(paper):
    """Two field nodes keep the table build under a second. The pins do not depend on it."""
    return replace(paper, n_b=2)


# ------------------------------------------------------------------ hash check


def test_sources_match_their_pins():
    _study.check_sources()


def test_pins_equal_the_v0_7_10_release():
    """The pinned digests are those of the files at the tag v0.7.10, which the manuscript cites."""
    root = _study.ROOT
    try:
        subprocess.run(
            ["git", "-C", str(root), "rev-parse", "--verify", "v0.7.10^{commit}"],
            capture_output=True,
            check=True,
        )
    except (OSError, subprocess.CalledProcessError):
        pytest.skip("git or the tag v0.7.10 is not available in this checkout")
    for rel, digest in _study.SOURCES.items():
        blob = subprocess.run(
            ["git", "-C", str(root), "show", f"v0.7.10:studies/classical-models/{rel}"],
            capture_output=True,
            check=True,
        ).stdout
        assert _study.sha256_bytes(blob) == digest, rel


def test_an_edited_copy_is_refused(tmp_path):
    _copy_sources(tmp_path)
    with open(tmp_path / "lib" / "basis.py", "a") as f:
        f.write("\n# edited\n")
    before = _interpreter_state()
    with pytest.raises(_study.StudyHashError, match="lib/basis.py"):
        _study.load(tmp_path)
    assert _interpreter_state() == before


def test_a_missing_source_is_refused(tmp_path):
    _copy_sources(tmp_path)
    (tmp_path / "ensemble_e6.py").unlink()
    with pytest.raises(_study.StudyHashError, match="ensemble_e6.py"):
        _study.load(tmp_path)


def test_an_unedited_copy_passes_the_check(tmp_path):
    _copy_sources(tmp_path)
    _study.check_sources(tmp_path)


# ------------------------------------------------------------------ isolation


def test_load_leaves_sys_modules_sys_path_and_environment_unchanged():
    fakes = {name: types.ModuleType(name) for name in ("basis", "config", "kernels")}
    saved = {name: sys.modules.get(name) for name in fakes}
    sys.modules.update(fakes)
    try:
        before = _interpreter_state()
        study = _study.load()
        assert _interpreter_state() == before
        for name in fakes:
            assert sys.modules[name] is fakes[name]
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
    lib = _study.STUDY / "lib"
    assert study.basis.__file__ == str(lib / "basis.py")
    assert study.basis.K is study.kernels
    assert study.ensembles.B is study.basis
    assert study.cost.B is study.basis
    assert study.e6.C is study.cost


def test_study_is_loaded_once():
    assert _study.study() is _study.study()


def test_paper_configurations():
    paper, maxwell = _study.paper(), _study.paper_maxwell()
    assert (paper.b_kind, paper.n_b, paper.n_channels, paper.n_bins) == (
        "uniform",
        16,
        41,
        19,
    )
    assert (maxwell.b_kind, maxwell.n_b, maxwell.b_rms) == ("maxwell", 96, 5.0)
    assert paper.azimuth_factor == pytest.approx(0.4 * np.exp(0.5j), rel=1e-15)


# ------------------------------------------------------------------ in-memory tables


def test_tables_are_built_in_memory_and_nothing_is_written():
    before = _cache_snapshot()
    cfg = _small_config(_study.paper())
    table = _study.kernel_table(cfg, max_order=2)
    assert _study.kernel_table(cfg, max_order=2) is table
    di, dq = _study.derivative_spectra(cfg, 2, table)
    assert di.shape == (19, 3, 41) and dq.shape == (19, 3, 41)
    assert np.all(np.isfinite(di)) and np.all(np.isfinite(dq))
    sep = _study.separable_columns(cfg, di, dq)
    joint = _study.joint_columns(cfg, di, dq)
    assert sep.shape == (41, 3, 57) and joint.shape == (41, 3, 171)
    assert _study.study().basis._cached_table.cache_info().currsize == 0
    assert _cache_snapshot() == before


def test_wrappers_refuse_a_missing_table():
    with pytest.raises(TypeError, match="table"):
        _study.derivative_spectra(_study.paper(), 2, None)
    with pytest.raises(TypeError, match="table"):
        _study.build_basis(_study.paper(), 2, "separable", None)


def test_e6_joint_columns_reproduce_the_cost_columns():
    """ensemble_e6.joint_columns at N = 2 equals cost.joint_columns bit for bit (ensemble_e6.py:7-9)."""
    before = _cache_snapshot()
    paper = _study.paper()
    cols = _study.e6_joint_columns(paper, 2)
    assert cols.shape == (41, 3, 114)
    reference = _study.study().cost.joint_columns(paper, _study.e6_table())
    assert np.array_equal(cols, reference)
    assert _cache_snapshot() == before


@pytest.mark.slow
def test_in_memory_paper_table_equals_the_study_cache_file():
    """The PAPER table built in memory equals the study's own cached table file bit for bit.

    The file is read, never written. The test skips when the gitignored cache does not hold it.
    """
    study = _study.study()
    paper = _study.paper()
    path = _study.study_cache_path(paper)
    if not path.exists():
        pytest.skip(f"{path.name} is not in studies/classical-models/cache")
    before = _cache_snapshot()
    table = _study.kernel_table(paper)
    with np.load(path) as stored:
        assert np.array_equal(stored["lny"], table.lny)
        assert np.array_equal(stored["coefF"], table.coef["F"])
        assert np.array_equal(stored["coefG"], table.coef["G"])
        assert int(stored["max_order"]) == study.basis.TABLE_ORDER
    assert _cache_snapshot() == before


# ------------------------------------------------------------------ acceptance data


def test_forward_spectra_copy_matches_its_hash():
    copied = _study.verify_sums(_study.ACCEPTANCE_DATA / "SHA256SUMS")
    assert set(copied) == {"forward_spectra.npz"}
    study_sums = _study.read_sums(_study.STUDY / "SHA256SUMS")
    assert copied["forward_spectra.npz"] == study_sums["forward_spectra.npz"]


def test_verify_sums_refuses_an_edited_file(tmp_path):
    shutil.copyfile(_study.ACCEPTANCE_DATA / "SHA256SUMS", tmp_path / "SHA256SUMS")
    data = (_study.ACCEPTANCE_DATA / "forward_spectra.npz").read_bytes()
    (tmp_path / "forward_spectra.npz").write_bytes(data + b"\0")
    with pytest.raises(_study.StudyHashError, match="forward_spectra.npz"):
        _study.verify_sums(tmp_path / "SHA256SUMS")
