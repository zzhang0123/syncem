"""Evaluate the frozen harmonic channel kernel on a gamma grid for the bridge study.

Runs radio_numerics.kernel of the harmonic pipeline (inputs/harmonic-radio-fit, read only,
no bytecode written) with syncem from the package root of this repository. The pipeline
pinned SyncMoments d51f9be; lib/syncem_compat.py checks that the syncem modules it calls
equal that commit up to the rename. The gamma grid gamma_k = 10^(2.5 + 0.1 k), k = 0..25,
shifts x = nu/(a_B gamma^2) by exactly four channels per step, so equal-x pairs at different
gamma test the gamma^-2 law.

Outputs (this directory): bridge_harmonic_probe.npz, bridge_harmonic_probe.json.
Run: python bridge_harmonic_probe.py [n_processes]
"""

from __future__ import annotations

import sys

sys.dont_write_bytecode = True

import hashlib  # noqa: E402
import json  # noqa: E402
import os  # noqa: E402
import platform  # noqa: E402
import time  # noqa: E402
from multiprocessing import get_context  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import scipy  # noqa: E402

HERE = Path(__file__).resolve().parent
LIB = HERE / "lib"
RADIO = HERE / "inputs" / "harmonic-radio-fit"
ADOPTED = dict(field=8, eta=7, pitch=18, channel=64)
COARSE = dict(field=6, eta=5, pitch=14, channel=48)
# Same B and eta nodes as ADOPTED; denser pitch and order quadrature only.
REFINED = dict(field=8, eta=7, pitch=26, channel=96)
REFINED2 = dict(field=8, eta=7, pitch=36, channel=128)  # convergence check, even k only
GRID = 10.0 ** (2.5 + 0.1 * np.arange(26))
CHECK_BINS = (0, 6, 11)
STENCIL_BINS = (8, 9, 10, 11)  # refined-quadrature 5-point stencil, step 0.04
STEP = 0.04


def sha(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def rel(path: Path) -> str:
    return Path(path).resolve().relative_to(HERE).as_posix()


def syncem_digest(syncem) -> str:
    """radio_numerics.package_digest() method applied to the imported syncem package."""
    package = Path(syncem.__file__).resolve().parent
    digest = hashlib.sha256()
    for path in sorted(package.rglob("*.py")):
        digest.update(path.relative_to(package).as_posix().encode())
        digest.update(b"\0")
        digest.update(path.read_bytes())
    return digest.hexdigest()


def _numerics():
    if str(LIB) not in sys.path:
        sys.path.insert(0, str(LIB))
    from syncem_compat import use_syncem  # noqa: PLC0415

    use_syncem()
    import radio_numerics as n  # noqa: PLC0415

    return n


def _evaluate(task):
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    sys.dont_write_bytecode = True
    n = _numerics()
    label, gamma, settings = task
    t0 = time.perf_counter()
    values, tail = n.kernel(float(gamma), settings)
    return label, float(gamma), values, tail, time.perf_counter() - t0


def main(processes: int) -> None:
    n = _numerics()
    import syncem  # noqa: PLC0415
    from syncem_compat import check_record  # noqa: PLC0415

    tasks = [("adopted", g, ADOPTED) for g in GRID]
    tasks += [("coarse", g, COARSE) for g in GRID]
    tasks += [("refined", g, REFINED) for g in GRID]
    tasks += [("refined2", g, REFINED2) for g in GRID[::2]]
    tasks += [(f"bin{b}", float(n.GAMMA_MIDS[b]), ADOPTED) for b in CHECK_BINS]
    tasks += [(f"stencil{b}_{o}", float(np.exp(n.LOG_MIDS[b] + n.LOG_HALF[b] * STEP * o)), REFINED)
              for b in STENCIL_BINS for o in range(-2, 3)]
    t0 = time.perf_counter()
    with get_context("spawn").Pool(processes) as pool:
        results = pool.map(_evaluate, tasks, chunksize=1)
    wall = time.perf_counter() - t0
    by = {}
    for label, gamma, values, tail, seconds in results:
        by.setdefault(label, []).append((gamma, values, tail, seconds))
    cache = np.load(RADIO / "trusted_basis_refined.npz")["pure_coefficients"]
    repro = {}
    for b in CHECK_BINS:
        (_, values, _, _), = by[f"bin{b}"]
        repro[str(b)] = float(np.max(np.abs(values - cache[b, 0])) / np.max(np.abs(cache[b, 0])))
    adopted = np.stack([v for _, v, _, _ in by["adopted"]])
    coarse = np.stack([v for _, v, _, _ in by["coarse"]])
    refined = np.stack([v for _, v, _, _ in by["refined"]])
    refined2 = np.stack([v for _, v, _, _ in by["refined2"]])
    stencil = np.zeros((len(STENCIL_BINS), 3, len(n.CENTRES), 3))
    for i, b in enumerate(STENCIL_BINS):
        v = [by[f"stencil{b}_{o}"][0][1] for o in range(-2, 3)]
        stencil[i, 0] = v[2]
        stencil[i, 1] = (v[0] - 8 * v[1] + 8 * v[3] - v[4]) / (12 * STEP)
        stencil[i, 2] = (-v[0] + 16 * v[1] - 30 * v[2] + 16 * v[3] - v[4]) / (24 * STEP**2)
    np.savez(
        HERE / "bridge_harmonic_probe.npz",
        stencil_refined_pure=stencil,
        stencil_bins=np.asarray(STENCIL_BINS),
        gamma=GRID,
        adopted=adopted,
        coarse=coarse,
        refined=refined,
        refined2=refined2,
        refined2_gamma=GRID[::2],
        adopted_tail=np.stack([t for _, _, t, _ in by["adopted"]]),
        coarse_tail=np.stack([t for _, _, t, _ in by["coarse"]]),
    )
    from syncem import constants as k  # noqa: PLC0415

    record = {
        "script": rel(Path(__file__)),
        "script_sha256": sha(Path(__file__)),
        "radio_numerics_sha256": sha(RADIO / "radio_numerics.py"),
        "stable_lines_sha256": sha(RADIO / "stable_lines.py"),
        "syncem_compat_sha256": sha(LIB / "syncem_compat.py"),
        "syncem_check": check_record(),
        "syncem_package_digest": syncem_digest(syncem),
        "pinned_package_sha256": n.PACKAGE_SHA,
        "pinned_package_note": "digest of SyncMoments d51f9be recorded by radio_numerics.py; syncem differs by the rename",
        "companion_pin": n.PIN,
        "constants": {"E_ESU": k.E_ESU, "M_E": k.M_E, "C_CGS": k.C_CGS, "C_SI_M": k.C_SI_M},
        "settings": {"adopted": ADOPTED, "coarse": COARSE, "refined": REFINED,
                     "refined2_even_k_only": REFINED2},
        "gamma_grid": GRID.tolist(),
        "bin_centre_reproduction_max_abs_over_max": repro,
        "seconds_per_kernel": {lab: [s for *_, s in v] for lab, v in by.items()},
        "wall_seconds": wall,
        "processes": processes,
        "versions": {"python": platform.python_version(), "numpy": np.__version__,
                     "scipy": scipy.__version__, "jax": getattr(sys.modules.get("jax"), "__version__", "not imported"),
                     "syncem": syncem.__version__},
        "output_npz_sha256": sha(HERE / "bridge_harmonic_probe.npz"),
    }
    (HERE / "bridge_harmonic_probe.json").write_text(json.dumps(record, indent=1) + "\n")
    print(json.dumps({k_: record[k_] for k_ in ("bin_centre_reproduction_max_abs_over_max",
                                                 "wall_seconds")}, indent=1))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 14)
