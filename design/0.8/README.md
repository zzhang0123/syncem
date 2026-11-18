# SyncEM 0.8 design

Design of the 0.8 top-level layer: a base model, a lattice of cells over its
variables with per-cell expansions, and population declarations that compile
to moments. Release 0.8.0 is additive and leaves `syncem.model` unchanged.

| File | Content |
|---|---|
| `TOPLEVEL.md` | Overview, about 270 lines. Start here |
| `SPEC.md` | Specification, version 6. The authority when files disagree |
| `OWNER_DECISIONS.md` | Answers to the 20 open questions of SPEC Sec. 14 (2026-11-17) |
| `NATIVE.md` | Native-code decision: no Rust or other new native code in 0.8 and 0.9 |
| `prototypes/` | JAX prototype of the core and six rule families, with tests and profiles. `prototypes/RESULTS.md` reports the results and the corrections applied in SPEC version 6 |

Labels in the documents:

* `PAPER:<line>` is a line of the companion manuscript (`main.tex`, not in this repository).
* `PKG/` is this repository's `syncem/`, and `ST/` is `studies/classical-models/`.
* `J1` to `J3`, `P1` to `P5`, `CTX` and `r1-`, `r2-`, `r3-` items are judge reports, proposals,
  context notes and review findings from the design process. They are kept outside the repository.

The implementation plan is SPEC Sec. 10.2 (steps 0 to 13, about 76 working days).
Each step has acceptance tests; the boundary-validation tests of SPEC Sec. 8.8
compare both methods at every dispatch threshold and at extreme parameters.

Prototype tests, from `design/0.8/prototypes`:

```sh
PYTHONPATH=.:rules:../../.. python -m pytest -q tests/test_core.py -k "not A2 and not A3"   # about 30 s
PYTHONPATH=.:rules:../../.. python -m pytest -q profile/test_boundaries.py                  # about 15 s
PYTHONPATH=.:rules:../../.. python -m pytest -q tests                                       # about 42 min
```

The prototypes are outside `testpaths`, so the package test suite does not collect them.
