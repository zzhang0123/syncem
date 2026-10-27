"""Run every script of the study in dependency order, then write SHA256SUMS.

Run: python run_all.py
     python run_all.py --checksums-only   (only rewrite SHA256SUMS from the files on disk)

The producing scripts run first, then the verify scripts, then make_figures.py, which comes
last because it reads verify_fits_results.json. Each script runs in its own process with this
interpreter, from this directory, with its default arguments. The run stops at the first script
that exits with a non-zero status. The wall time of each script is printed. kp_bridge_tables.py
only prints the tables of kp_bridge_results.json and is not run.

SOURCE_DATE_EPOCH is set to 2026-10-27 00:00 UTC unless it is already set, so that matplotlib
writes a fixed creation date into the PDF figures. PYTHONDONTWRITEBYTECODE is set to 1 unless it
is already set, so that importing the modules in inputs/ writes no bytecode there.

SHA256SUMS is in sha256sum format, with paths relative to this directory. It lists every file
in inputs/, every results JSON, every table in tables/, every npz product and every figure in
figures/. Bytecode caches (__pycache__/, *.pyc) and *.tmp files are skipped. Check it with
`sha256sum -c SHA256SUMS` (or `shasum -a 256 -c SHA256SUMS`); in a git checkout without the release
archive of npz products and figures, add `--ignore-missing`.
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE_DATE_EPOCH = "1793059200"  # 2026-10-27T00:00:00Z

# (script, outputs of earlier scripts that it reads)
PRODUCERS = (
    ("core_checks.py", ()),
    ("regression.py", ()),
    ("forward.py", ()),
    ("fits.py", ("forward_spectra.npz", "regression_results.json")),
    ("fits_tails.py", ("forward_spectra.npz",)),
    ("params.py", ("forward_spectra.npz",)),
    ("kp_pseudotruth.py", ("forward_spectra.npz", "fits_results.json", "params_results.json")),
    ("baselines.py", ("forward_spectra.npz",)),
    ("cost.py", ()),
    ("kp_bridge.py", ()),
    ("harmonic_correction.py", ("forward_spectra.npz", "kp_bridge_results.json")),
    ("bridge_mpmath_check.py", ()),
    ("bridge_harmonic_probe.py", ()),
    ("bridge.py", ("bridge_harmonic_probe.npz", "bridge_harmonic_probe.json",
                   "bridge_mpmath_check.json")),
)
VERIFIERS = (
    ("verify_core.py", ("forward_results.json", "regression_results.json")),
    ("verify_fits.py", ("fits_results.json", "fits_products.npz")),
    ("verify_params.py", ("params_results.json", "params_realisations.npz")),
    ("verify_baselines.py", ("baselines_results.json", "forward_spectra.npz")),
    ("verify_kpbridge.py", ("kp_bridge_results.json",)),
    ("verify_bridge.py", ("bridge_results.json", "bridge_harmonic_probe.npz")),
)
FIGURE_INPUTS = ("fits_results.json", "fits_products.npz", "forward_results.json",
                 "forward_spectra.npz", "params_results.json", "baselines_results.json",
                 "baselines_plotdata.npz", "cost_results.json", "kp_pseudotruth_results.json",
                 "kp_pseudotruth.npz", "verify_fits_results.json")
FIGURES = (("make_figures.py", FIGURE_INPUTS),)
STAGES = (("producing scripts", PRODUCERS), ("verify scripts", VERIFIERS), ("figures", FIGURES))

CHECKSUM_FILE = HERE / "SHA256SUMS"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


SKIPPED_SUFFIXES = (".pyc", ".tmp")


def is_checksummed(path: Path) -> bool:
    """False for directories, bytecode caches and temporary files."""
    if not path.is_file() or path.suffix in SKIPPED_SUFFIXES:
        return False
    return "__pycache__" not in path.relative_to(HERE).parts


def checksummed_files() -> list[Path]:
    """inputs/, results JSONs, tables, npz products and figures, sorted by relative path."""
    files = list((HERE / "inputs").rglob("*"))
    files += HERE.glob("*.json")
    files += HERE.glob("*.npz")
    files += (HERE / "tables").glob("*")
    files += (HERE / "figures").glob("*")
    return sorted((p for p in files if is_checksummed(p)), key=lambda p: p.relative_to(HERE).as_posix())


def write_checksums() -> int:
    files = checksummed_files()
    lines = [f"{sha256(p)}  {p.relative_to(HERE).as_posix()}\n" for p in files]
    CHECKSUM_FILE.write_text("".join(lines))
    print(f"wrote {CHECKSUM_FILE.name}: {len(lines)} files", flush=True)
    return len(lines)


def run_script(script: str, needs: tuple, env: dict) -> float:
    missing = [name for name in needs if not (HERE / name).exists()]
    if missing:
        raise SystemExit(f"{script}: missing input(s) {', '.join(missing)}")
    print(f"\n=== {script}", flush=True)
    t0 = time.perf_counter()
    result = subprocess.run([sys.executable, script], cwd=HERE, env=env, check=False)
    wall = time.perf_counter() - t0
    if result.returncode != 0:
        print(f"=== {script} failed with exit status {result.returncode} after {wall:.1f} s",
              flush=True)
        raise SystemExit(result.returncode)
    print(f"=== {script}: {wall:.1f} s", flush=True)
    return wall


def main(argv: list[str]) -> int:
    if argv[1:] == ["--checksums-only"]:
        write_checksums()
        return 0
    if argv[1:]:
        raise SystemExit("usage: python run_all.py [--checksums-only]")
    env = dict(os.environ)
    env.setdefault("SOURCE_DATE_EPOCH", SOURCE_DATE_EPOCH)
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    t0 = time.perf_counter()
    walls = []
    for stage, scripts in STAGES:
        print(f"\n##### {stage}", flush=True)
        for script, needs in scripts:
            walls.append((script, run_script(script, needs, env)))
    print("\nwall time per script:")
    for script, wall in walls:
        print(f"  {script:28s} {wall:9.1f} s")
    print(f"  {'total':28s} {time.perf_counter() - t0:9.1f} s", flush=True)
    write_checksums()
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv))
