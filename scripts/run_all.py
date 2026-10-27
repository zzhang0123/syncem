"""Run every validation, design and figure script, reporting process completion.

Usage: PYTHONPATH=. python3 scripts/run_all.py [--no-figures]

The figure scripts are included by default so that figures/*.pdf cannot drift
out of step with the physics modules; pass --no-figures to skip them.
"""

from __future__ import annotations

import subprocess
import sys
import os

SCRIPTS = [
    "validate_model.py",  # emissivity side (7 checks)
    "validate_transfer.py",  # RT transfer (5 checks)
    "validate_kirchhoff.py",  # absorption / Kirchhoff (6 checks)
    "validate_rt_moments.py",  # Magnus / conversion / cross-cumulants (4 checks)
    "validate_sed.py",  # SED / spectral index / LOS cumulants (6 checks)
    "validate_cumulants.py",  # finite Bell contractions and scalar Gaussian example
    "test_design.py",  # JAX/Equinox jit + differentiability
]

# Companion demonstration figures; the manuscript uses separate independent
# generators. Regenerate these whenever their implementation or inputs change.
FIGURES = [
    # figures/derivative_spectra.pdf (``sec: derivative spectra``, Section 4.3)
    "plot_derivative_spectra.py",
    # figures/moment_vs_cumulant.pdf ([extension] Gamma-distribution MGF
    # truncation comparison)
    "compare_moment_cumulant.py",
    # figures/foreground_demo.pdf ([extension] illustrative global 21-cm
    # foreground)
    "application_foreground.py",
    # figures/transfer.pdf ([extension] polarised radiative transfer)
    "plot_transfer.py",
    # figures/sed_moment_reconstruction.{pdf,png} and its JSON/NPZ/CSV data
    # ([extension] continuum SED example, 72 channels over 0.4-3 GHz, from the
    # 2026-09-26 manuscript draft): public-API reduce_response +
    # fit_combinations example
    "sed_reconstruction_example.py",
]

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def main():
    # Diagnostics that print numbers are not scientific pass/fail assertions.
    # Run the actual regression suite before interpreting script completion.
    checked = subprocess.run(
        [sys.executable, "-m", "pytest", "tests", "-q"],
        cwd=ROOT,
        env={**os.environ, "PYTHONPATH": ROOT},
    )
    if checked.returncode:
        sys.exit(checked.returncode)
    figures = "--no-figures" not in sys.argv
    names = SCRIPTS + (FIGURES if figures else [])
    failures = []
    for name in names:
        path = os.path.join(ROOT, "scripts", name)
        r = subprocess.run(
            [sys.executable, path],
            cwd=ROOT,
            capture_output=True,
            text=True,
            env={**os.environ, "PYTHONPATH": ROOT},
        )
        status = "COMPLETED" if r.returncode == 0 else "FAIL"
        print(f"[{status}] {name}")
        if r.returncode != 0:
            failures.append(name)
            print(
                "    "
                + (
                    r.stderr.strip().splitlines()[-1]
                    if r.stderr.strip()
                    else "(no stderr)"
                )
            )
    print()
    if failures:
        print(f"{len(failures)} script(s) failed: {', '.join(failures)}")
        sys.exit(1)
    print(
        f"all {len(names)} scripts completed (scientific assertions live in pytest)"
        + ("" if figures else "  (figures skipped: --no-figures)")
    )


if __name__ == "__main__":
    main()
