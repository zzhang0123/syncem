"""Study oracles of ``studies/classical-models``, read in place (design/0.8/SPEC.md decision 154).

The study sources import each other as top-level modules with generic names (``config``,
``kernels``, ``basis``). ``load`` checks every source it executes against the SHA-256 digests in
``SOURCES``, imports the modules under a temporary ``sys.path`` entry, and afterwards restores
``sys.path``, the ``sys.modules`` entries of ``MODULE_NAMES``, the BLAS thread variables that
``cost.py`` sets at import, and ``sys.dont_write_bytecode``. The loaded modules stay reachable
through the returned ``Study`` and import nothing at call time on the paths used here.

``studies/classical-models/SHA256SUMS`` lists the study outputs and inputs, not ``lib/`` or the
scripts. The digests in ``SOURCES`` are those of the files at the tag v0.7.10, which the
manuscript cites. A test compares them with the tag when git and the tag are available.

Kernel tables are built in memory by ``kernel_table`` and passed as ``table=``. The study's own
``basis.get_table`` writes ``studies/classical-models/cache`` on a miss, so every wrapper here
refuses a missing table. ``e6_joint_columns`` calls ``ensemble_e6.joint_columns``, which builds its
one-node table at B0 = 5 muG in memory through ``cost.fixed_b_table``.

Units are those of the study: frequencies in Hz, field strengths in muG, spectra in
erg s^-1 Hz^-1 sr^-1 per electron. The column arrays have the shape (J, 3, n) with Stokes
[I, Q, U], and the derivative spectra the shape (n_bins, N + 1, J). These oracles are study code
and are not certified by syncem.
"""

from __future__ import annotations

import hashlib
import importlib
import os
import sys
from collections.abc import Mapping
from dataclasses import dataclass
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
STUDY = ROOT / "studies" / "classical-models"
ACCEPTANCE_DATA = ROOT / "tests" / "acceptance" / "data"

# sha256 of each study source executed by ``load``, at the tag v0.7.10.
SOURCES: Mapping[str, str] = {
    "lib/kernels.py": "b0983233f8fee93b12b10c299c0ea0d0011d72a106884029ec46107ba4121570",
    "lib/config.py": "78aaca6292cd15db1d820d25651c288ed47d14213b8da3d55440b7d4546e887c",
    "lib/basis.py": "438afd89638545dbddfd9acf51cbef67d67d8484a24dcd28d2a1755b6697eace",
    "lib/ensembles.py": "608dc23a9f795bd7cf4af10b84bfb618c41d3d0d5d858e878aacb410fcbc7c60",
    "lib/provenance.py": "32dc912a4bc75a4460d6264b1ffcf8e278e7da1c6db8e0a43e810fa31f737590",
    "cost.py": "cee5bde0846fc3bbf7edc5385b2613c6b37aa92bc8589f5b8aaf4e275b1916fb",
    "ensemble_e6.py": "5f90b22c7e0461e46c5b7418cc1f4b0961ebb52fc7dc457d5a99ff4a878d3285",
}

# Import order. Each module imports only modules before it.
MODULE_NAMES = (
    "kernels",
    "config",
    "basis",
    "ensembles",
    "provenance",
    "cost",
    "ensemble_e6",
)
THREAD_VARS = (
    "OPENBLAS_NUM_THREADS",
    "OMP_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
    "MKL_NUM_THREADS",
)


class StudyHashError(RuntimeError):
    """A study file is missing or differs from its pinned digest."""


# ------------------------------------------------------------------ digests


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: Path) -> str:
    return sha256_bytes(Path(path).read_bytes())


def _mismatches(directory: Path, expected: Mapping[str, str]) -> list[str]:
    bad = []
    for rel, digest in expected.items():
        path = Path(directory) / rel
        if not path.is_file():
            bad.append(f"{rel} (missing)")
        elif sha256_file(path) != digest:
            bad.append(f"{rel} (sha256 differs)")
    return bad


def check_sources(study: Path = STUDY, sources: Mapping[str, str] = SOURCES) -> None:
    """Raise ``StudyHashError`` unless every source under ``study`` matches its pinned digest."""
    bad = _mismatches(study, sources)
    if bad:
        raise StudyHashError(
            f"study sources under {study} do not match their pins: {', '.join(bad)}"
        )


def read_sums(path: Path) -> dict[str, str]:
    """Parse a sha256sum file into {relative path: digest}."""
    out = {}
    for line in Path(path).read_text().splitlines():
        if line.strip():
            digest, rel = line.split(maxsplit=1)
            out[rel.strip().lstrip("*")] = digest
    return out


def verify_sums(path: Path) -> dict[str, str]:
    """Check every file listed in the sha256sum file ``path``, relative to its directory."""
    sums = read_sums(path)
    bad = _mismatches(Path(path).parent, sums)
    if bad:
        raise StudyHashError(f"files listed in {path} do not match: {', '.join(bad)}")
    return sums


# ------------------------------------------------------------------ isolated import


@dataclass(frozen=True)
class Study:
    """The loaded study modules. Their globals refer to each other, not to ``sys.modules``."""

    kernels: ModuleType
    config: ModuleType
    basis: ModuleType
    ensembles: ModuleType
    provenance: ModuleType
    cost: ModuleType
    e6: ModuleType


def _import_all(study: Path) -> dict[str, ModuleType]:
    sys.path[:0] = [str(study / "lib"), str(study)]
    importlib.invalidate_caches()
    return {name: importlib.import_module(name) for name in MODULE_NAMES}


def load(study: Path = STUDY, sources: Mapping[str, str] = SOURCES) -> Study:
    """Check and import the study sources without leaving state in the interpreter."""
    check_sources(study, sources)
    saved_path = list(sys.path)
    saved_modules = {
        name: sys.modules.pop(name) for name in MODULE_NAMES if name in sys.modules
    }
    saved_env = {var: os.environ.get(var) for var in THREAD_VARS}
    saved_bytecode = sys.dont_write_bytecode
    sys.dont_write_bytecode = True
    try:
        modules = _import_all(Path(study))
    finally:
        sys.path[:] = saved_path
        for name in MODULE_NAMES:
            sys.modules.pop(name, None)
        sys.modules.update(saved_modules)
        for var, value in saved_env.items():
            if value is None:
                os.environ.pop(var, None)
            else:
                os.environ[var] = value
        sys.dont_write_bytecode = saved_bytecode
    return Study(**{("e6" if k == "ensemble_e6" else k): v for k, v in modules.items()})


@cache
def study() -> Study:
    """The study at ``STUDY``, loaded once per process."""
    return load()


def paper() -> Any:
    """``config.PAPER``: 41 channels over 0.1 to 10 GHz, B ~ U[4, 6] muG with 16 nodes, 19 cells."""
    return study().config.PAPER


def paper_maxwell() -> Any:
    """``config.PAPER_MAXWELL``: PAPER with a Maxwellian field strength, b_rms 5 muG, 96 nodes."""
    return study().config.PAPER_MAXWELL


# ------------------------------------------------------------------ in-memory tables


def _table_key(cfg: Any, max_order: int | None) -> tuple:
    basis = study().basis
    order = basis.TABLE_ORDER if max_order is None else int(max_order)
    return basis._table_key(cfg, order)


@cache
def _table_for_key(key: tuple) -> Any:
    b_kind, b_lo, b_hi, b_rms, n_b, cut, max_order = key
    s = study()
    cfg = s.config.Config(
        b_kind=b_kind, b_lo=b_lo, b_hi=b_hi, b_rms=b_rms, n_b=n_b, maxwell_cut=cut
    )
    b, w = cfg.b_nodes_weights
    return s.basis.KernelTable(b, w, max_order=max_order)


def kernel_table(cfg: Any, max_order: int | None = None) -> Any:
    """``basis.KernelTable`` for the field-strength law of ``cfg``, built in memory once per key.

    The key is the study's own ``basis._table_key``, so two configurations that share a table in
    the study share it here. ``max_order`` defaults to ``basis.TABLE_ORDER`` (6).
    """
    return _table_for_key(_table_key(cfg, max_order))


def study_cache_path(cfg: Any, max_order: int | None = None) -> Path:
    """The file name that ``basis.get_table`` would read or write for ``cfg``. Nothing is opened."""
    key = _table_key(cfg, max_order)
    lib = STUDY / "lib"
    src = sha256_bytes(
        (lib / "kernels.py").read_bytes() + (lib / "basis.py").read_bytes()
    )[:16]
    name = "table_" + sha256_bytes(repr(key).encode())[:16] + f"_{src}.npz"
    return STUDY / "cache" / name


def _require(table: Any) -> Any:
    if table is None:
        raise TypeError(
            "table is None. Pass kernel_table(cfg). The study default writes studies/classical-models/cache."
        )
    return table


# ------------------------------------------------------------------ oracles


def derivative_spectra(
    cfg: Any, N: int, table: Any, pitch_l: int | None = None
) -> tuple:
    """``basis.derivative_spectra``: D^I (n_bins, N + 1, J) and D^Q complex, no azimuth factor."""
    return study().basis.derivative_spectra(
        cfg, N, pitch_l=pitch_l, table=_require(table)
    )


def separable_columns(cfg: Any, DI: Any, DQ: Any, factor: complex | None = None) -> Any:
    """``basis.separable_columns``: (J, 3, n_bins (N + 1)), P = factor D^Q (default azimuth factor)."""
    return study().basis.separable_columns(cfg, DI, DQ, factor=factor)


def joint_columns(cfg: Any, DI: Any, DQ: Any) -> Any:
    """``basis.joint_columns``: (J, 3, 3 n) for [a_br, Re M2_br, Im M2_br], n = n_bins (N + 1)."""
    return study().basis.joint_columns(cfg, DI, DQ)


def build_basis(
    cfg: Any, N: int, variant: str, table: Any, pitch_l: int | None = None
) -> Any:
    """``basis.build_basis`` for variant "separable", "joint" or "q", with an in-memory table."""
    return study().basis.build_basis(
        cfg, N, variant, pitch_l=pitch_l, table=_require(table)
    )


def e6_table() -> Any:
    """The one-node table at B0 = 5 muG of ``ensemble_e6._tab0``, built in memory by ``cost``."""
    return study().e6._tab0()


def e6_joint_columns(cfg: Any, N: int = 2) -> Any:
    """``ensemble_e6.joint_columns``: (J, 3, n_monomials n_bins), column n_monomials cell + monomial.

    The monomials are x_gamma^r x_B^s with r + s <= N, ordered by s then r
    (``ensemble_e6.monomials``), and x_B = (B - 5 muG)/(1 muG).
    """
    return study().e6.joint_columns(cfg, N)


def e6_monomials(N: int = 2) -> tuple:
    """``ensemble_e6.monomials``: the (r, s) exponents of the E6 columns."""
    return study().e6.monomials(N)
