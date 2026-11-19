# SyncEM native-code decision record, 0.8 and 0.9

Date: 2026-11-18. Status: proposed for the owner. Inputs: `SPEC.md` (version 5, whose cited sections keep their numbers in version 6), `OWNER_DECISIONS.md`, `prototypes/RESULTS.md` Sec. 6 and the profile files in `prototypes/profile/results/`. Every number below was re-derived from those files. Where RESULTS disagrees with them, Sec. 3.1 gives the corrected value.

## 1. Recommendation

SyncEM 0.8 and 0.9 ship no Rust and no other new native code. The package stays in JAX, Equinox, NumPy and SciPy (a base dependency by owner decision Q4, imported lazily). Three seams are kept so that native code can replace a host path later without a public API change: host-only kernels behind the `HostOnly` capability, the certification arithmetic backend, and the harmonic host driver (Sec. 6).

Evidence. Eight hot paths were profiled on the paper configurations (Sec. 3). None is interpreter-bound in a way that a Rust rewrite fixes better than an algorithmic change or batching in the current stack. The largest measured factors are algorithmic or come from batching:

* S1 by an S0 table in `ln x` plus lookups: 53 to 77 times per field node (prototype profile p1).
* Tabulated F, G for forward kernels: 45 times, 1.1e-11 of I (prototype profile p6).
* Basis-route pairing for rays: 690 times against tabulated direct integration, 2.7e4 times against quadrature direct integration (prototype profile p6).
* Batched JAX Levenberg-Marquardt for family fits: 40 times against SciPy `trf` (prototype profile p5).
* Per-pixel SVD on all cores: 13 times against one core (prototype profile p3).

The jet compile cost (RESULTS Sec. 4, item 10) is a JAX compile cost, and a Rust port would not change it.

Uncertainty. All profile files were written between 22:29 and 22:40 on 2026-11-17. RESULTS reports load averages 60 to 108 on 28 cores after 22:33, and the files that record the load show 65 to 108, so absolute times are upper bounds (RESULTS Sec. 7.4). Ratios timed back to back hold. The `gmpy2` (2 to 5 times) and `python-flint`/Arb (10 to 100 times) speed-ups are estimates. A jet over a smooth tabulated F, G, a JAX port of the stack provider, sharding of the jet across cells and the fit-level effect of the S0/S1 split were not measured.

## 2. Workload and constraints

* Users and interfaces. A Python library used from scripts, notebooks and batch jobs through the front door `se.*` (SPEC Sec. 2). No other language consumes it today.
* Core value. `basis.predict`, family moments, free-moment fits and `problem.log_prob` are `jit`, `vmap` and `grad` compatible, and the jet provider differentiates in reference points and in specified-statistic parameters (SPEC Sec. 8.6). Native code is invisible to `jax.grad` and `jax.jit`. It enters a traced path only as a custom primitive with its own JVP, transpose and batching rules, and a host callback breaks `grad` and `vmap`. Any native kernel on a traced path therefore costs derivative rules to the order the jet needs (builds to order 6 were run).
* Numerics. CPU, float64 throughout (`import syncem` enables it). Provider checks fail above 1e-10 of the channel column maximum (RESULTS 3.5.6), and the profiled certification call uses `atol = 1e-19`.
* Sizes. 41 channels, 19 energy cells, 57 to 513 lattice rows on the paper configurations, 63 x 57 and 63 x 171 designs, maps of 1e6 pixels, rays of 300 steps and 1e5 rays.
* Machines. Laptops to clusters. The profile machine is an M3 Ultra with 28 cores. No Rust toolchain is installed (`cargo` and `rustc` not found), and neither `gmpy2` nor `python-flint` is installed. mpmath runs on its pure-Python backend (`mpmath.libmp.BACKEND == 'python'`).
* Maintenance. One maintainer. A native extension adds a wheel matrix (macOS arm64, Linux x86_64 and aarch64) and a second language to review.

## 3. Measured hot paths

| # | Path | Measured | Bound by | Native candidate |
|---|---|---|---|---|
| 1 | S1 field-average table | NumPy 0.32 to 0.37 s per field node, 98.3 to 99.8 per cent in the quadrature. JAX port of the same quadrature 36 ms (1 node), 247 ms (8 nodes). S0 once 0.38 s, then S1 by lookups 4.5 ms (1 node) and about 6 ms per node at 2 to 8 nodes. Lookup against study quadrature: at most 3.7e-8 at `r <= 6` for single field nodes, 1.8e-7 at `r = 6` for averages over 2 to 8 nodes (`p1_s1_table.json`, `boundaries.json`). These are pointwise errors near sign changes. SPEC decision 140 gives the table error as 3.3e-10 of the peak at `r = 6` | the quadrature redone per field node, compiled NumPy on one core | No. The lookup removes the per-node work in NumPy |
| 2 | Column build, stack against jet | quiet rerun (load 2.5): stack for 19 cells 0.15 s (`N = 2`) and 0.21 s (`N = 4`) at 400 channel nodes, 0.026 and 0.037 s at 64. Jet, one cell: 0.35 s plus 0.26 s compile (`N = 2`, 1.8 primals), 3.0 s plus 2.4 s compile (`N = 4`, 15.8 primals), cpu/wall 12.5 and 15.4. E1 averaged-kernel primal 0.19 s. Jet 260 and 1570 times the 64-node stack (`p2_columns.json`, the loaded run kept as `p2_columns_loaded.json`) | re-evaluation of F, G by quadrature at every node, and JAX compile | Unlikely. A smooth tabulated F, G in JAX is the larger lever |
| 3 | Free-moment fit | shared noise, 1e5 pixels: JAX 0.21 us (63 x 57) and 0.15 us (63 x 171), NumPy one core 0.83 and 0.76 us. Per-pixel noise: one core 177 and 611 us, NumPy 28 threads 13.6 and 39 us, JAX `vmap` 16.7 and 51 us. Ranks agree (`p3_free_fit.json`) | LAPACK on small matrices | No |
| 4 | NNLS, 111 x 272, cond 1.5e18 | serial 2.96 ms, median 13 active atoms. Pool of 28 with default BLAS threads 1.05 ms. Pool with BLAS pinned to one thread 0.51 ms (4 and 8 workers) to 0.71 ms (28 workers) per spectrum. JAX FISTA 6.5 to 73 ms with `chi2` excess up to 3.8 (`p4_nnls.json`, `p4b_nnls_pool.json`) | compiled active set, thread oversubscription | No |
| 5 | Family fit, 63 x 57 | `trf` 2.39 ms with 12 evaluations and a NumPy residual of 19 us. JAX Levenberg-Marquardt under `vmap`, 40 iterations: 60 us per spectrum, parameters within 4.3e-7 of `trf`, `chi2` difference 2.3e-13 (`p5_family.json`) | Python overhead inside `trf` | No |
| 6 | Ray, 300 steps | basis route 3.7 ms per ray (`vmap` over 1000 rays, 4864 rows), plus columns once (608 cells x 0.245 s). Direct with tabulated F, G 2.55 s. Direct with quadrature F, G 9.86 s at 30 steps, 98.6 s extrapolated to 300. NumPy `core.pair` 2.05 s per ray (`p6_ray.json`) | scatter-adds and complex exponentials | No |
| 7 | High-order harmonic host route | 2.2 to 2.9 ms per point for 3 channels, 0.75 to 0.96 ms per channel point. Line powers 0.28 us per order sample, 88 per cent of it in two SciPy `jv` calls. Line-power work is 4 to 8 per cent of the wall time (`p7_harmonic.json`) | per-call overhead, 92 to 96 per cent | Plausible, but NumPy vectorisation reaches the Bessel floor first |
| 8 | Interval certification | 1.30 to 1.43 s per call (117 retained blocks), 73 per cent in mpmath, 4.04 million calls (`p8_certified.json`) | pure-Python multiprecision | Yes if routine, through `gmpy2` or `python-flint`/Arb before any new Rust crate |

### 3.1 Corrections to RESULTS

1. Sec. 5 S1 row and Sec. 6 row 1. NumPy per field node is 0.32 to 0.37 s (0.37 s at 4 nodes), not 0.32 to 0.34 s. The quadrature share is 98.3 to 99.8 per cent, and 99.6 per cent holds at 4 nodes only. The JAX port at 8 nodes is 247 ms, not 248 ms.
2. Sec. 6 row 1 and the first recommendation bullet. S0 plus lookups is 53 to 77 times faster per field node, not 60 to 150 times. The value 150 follows from the F-only lookup (2.2 ms) against the quadrature of F and G.
3. Sec. 5 S1 row and Sec. 6 boundary paragraph. The bound 3.7e-8 at `r = 6` holds for single field nodes (`boundaries.json`). For field averages over 2 to 8 nodes, `p1_s1_table.json` gives up to 1.0e-8 at `r = 4`, 1.2e-7 at `r = 5` and 1.8e-7 at `r = 6`. All are inside the test thresholds of `profile/test_boundaries.py` (1e-6 at `r = 6`).
4. Sec. 6 row 2. `p2_columns.json` gives the stack at 0.25 and 0.40 s (400 nodes), not 0.17 and 0.25 s, and records the jet at cpu/wall 4.6 and 8.3, not 15 to 18 cores. The 19-cell jet figures 10.4 and 70.7 s and the ratios 360 and 1600 came from a `tests/core_results.json` that the full-suite run overwrote (it now holds 0.35 and 2.94 s per cell). From the files on disk, the jet costs 140 to 520 times the 64-node stack per cell at `N = 2` and 590 to 1070 times at `N = 4`. A quiet rerun of p2 on 2026-11-18 (load 2.5) settles it: stack 0.15 and 0.21 s at 400 nodes, 0.026 and 0.037 s at 64. Jet 0.35 and 3.0 s per cell (6.7 and 58 s for 19 cells, cpu/wall 12.5 and 15.4). The jet costs 260 times the 64-node stack at `N = 2` and 1570 times at `N = 4`. SPEC and TOPLEVEL use these values.
5. Sec. 6 recommendation, ray bullet. 2.55 s against 3.7 ms is 690 times, not 670. The quadrature direct time is extrapolated from 30 steps, and 98.6 s against 3.7 ms is 2.7e4 times, not 2.6e4.
6. Sec. 6 row 7. Per-call overhead is 92 to 96 per cent and SciPy `jv` 3 to 7 per cent of the wall time, not about 90 and 10 per cent.
7. Sec. 5 NNLS row. The serial run of `p4b_nnls_pool.json` (7.4 ms at cpu/wall 0.40) was slowed by load. Against the 2.96 ms serial of `p4_nnls.json`, the pinned pool gives 4.2 to 5.8 times, not the logged 10 to 15 times.

## 4. Options per path

| Path | Algorithmic change | Batching in JAX | Existing native library | New Rust | Chosen, and why it has lower regret |
|---|---|---|---|---|---|
| S1 table | S0 in `ln x` plus lookups, 53 to 77 times | JAX port of the quadrature, about 10 times | SciPy splines, already used | a port of the quadrature keeps the per-node redo | algorithmic. It removes the work in NumPy and leaves the 0.9 JAX port of the stack unchanged |
| Column build | stack by default where its precondition holds. Smooth tabulated F, G (piecewise Chebyshev in `ln x`) as the candidate for the jet primal | padded `lax.map` over cells (SPEC Sec. 8.5). Sharding not measured | none | a native F, G is invisible to the jet unless it carries derivative rules to order 6 | stack and jet in JAX, tabulated F, G as the next measurement. The jet is the general route and the oracle of every shortcut, so it must stay traceable |
| Free-moment fit | none needed | `vmap` with chunks from `chunk_bytes` | LAPACK through JAX and NumPy | no gain on a LAPACK-bound path | JAX `vmap` |
| NNLS | the active set is already compiled | FISTA is slower and inaccurate at cond 1.5e18 | SciPy `nnls` in a host process pool, BLAS pinned to one thread | would reimplement a mature kernel that has no defect | SciPy with the pinned pool |
| Family fit | none | jitted `vmap` LM, 40 times | SciPy `trf` kept as the test oracle | removes overhead that batching already removes | JAX LM. It stays differentiable and batches over pixels |
| Ray | basis route as the production path, 690 times | `segment_sum` pairing, 3.7 ms against 2.05 s for NumPy `core.pair` | none | none needed | basis route, with tabulated F, G for the direct reference |
| Harmonic host route | already switches to the corrected integral | not traceable at these orders (`required_m_max` 4.1e8, SPEC Sec. 8.4) | NumPy vectorisation over points and order samples with SciPy `jv` | would cut the residual overhead | NumPy vectorisation first. Rust only past the trigger of Sec. 7 |
| Certification | none measured | interval arithmetic does not map to JAX | `gmpy2` as mpmath's backend (no code change), `python-flint`/Arb (a port of the enclosures) | a correctness-critical reimplementation of interval arithmetic | an existing library in a `certification` extra |

## 5. Execution choices

| Choice | Decision | Reason |
|---|---|---|
| Rust or other new native code | defer, with the seams of Sec. 6 | no path where it beats the algorithmic or batching fix. It is invisible to autodiff, no toolchain is installed, and it adds a wheel matrix |
| CPU parallelism | implement now | `vmap` for free-moment fits and ray pairing, and for the batched family LM when it lands in 0.9 (SPEC Sec. 10.1). Per-pixel batching in chunks sized by `chunk_bytes`. A host process pool for NNLS and the other host fits, with BLAS pinned to one thread in each worker (`OMP_NUM_THREADS=1`, `VECLIB_MAXIMUM_THREADS=1`). The default BLAS threads oversubscribe (1.05 ms against 0.51 ms) |
| GPU | retain the seam | JAX code stays device-agnostic: no host transfers inside jitted functions, no CPU-only primitives on traced paths. No GPU-specific work. GPU execution is untested and is not claimed. The host paths (stack, table, NNLS, `trf`, harmonic driver, certification) stay on the CPU |
| Multi-node | defer | a map of 1e6 pixels is projected at 8.5 min of NNLS in one pinned pool (`p4b_nnls_pool.json`) and seconds of free fits. Pixel chunks share no mutable state, so a job array split by pixel ranges needs no code change |

Backend choice is an implementation detail for the pool and for `vmap` (same results). The certification backend changes enclosure widths and not the contract, so it is an implementation detail too. The smooth tabulated F, G changes the kernel values and is scientific semantics: it is recorded in `KernelRecord` and checked against quadrature F, G before it becomes a default.

## 6. Seams

**Host-only kernels (`HostOnly` capability, SPEC Sec. 3.2 and 8.1).**
* Boundary. `Kernel.channel(channels, point, *, phase) -> Modes` with `traceable = False`, called only by the table provider (17 Chebyshev-Lobatto nodes per cell), by `basis.direct` and by the accuracy check, all on the host. `HighOrderHarmonicKernel` [0.9] is the first such kernel.
* Inputs. `point` as `{name: float}` in canonical units for the names of `Kernel.variables`, `channels` a `Channels` object, `phase` the phase law. Outputs. `Modes` with I, V of shape `(n_ch,)` and P of shape `(n_w, n_ch)` complex, float64 NumPy arrays, new on every call.
* Ownership. The engine owns the inputs and the cell table. The kernel holds only immutable configuration, keeps no references to its inputs or outputs, and is picklable so that a process pool can call it.
* Errors. Python exceptions of the `syncem` error types (refused regime, unconverged result). No NaN encodes an error. The table provider records the error per cell.
* Slot-in rule. A native kernel implements the same protocol and declares `HostOnly`. A private batched entry point over the 17 table nodes per cell, defaulting to a loop over `channel`, removes the per-call overhead without a public change.

**Certification arithmetic backend.**
* Boundary. The interval context used by the certified routines of `syncem.model` (`_certified_dense_harmonic.certified_dense_channel_modes`, `_verified_bessel`), today `mpmath.ctx_iv.MPIntervalContext()` created per call with `dps = decimal_digits` (default 70, at least 30).
* Inputs. Point values as binary floats, channels, `atol` in output units, `max_blocks`. Outputs. Enclosures whose whole-channel componentwise absolute error meets `atol`.
* Ownership. One context per call. No module-level state, so calls run in parallel processes.
* Errors. The routine raises when `atol` is not met or `max_blocks` is reached, and an `ImportError` names the missing package. Every backend keeps these semantics.
* Slot-in rule. `gmpy2` speeds mpmath's own arithmetic with no code change. `python-flint`/Arb needs the enclosure functions ported behind the same context interface. Both go in a `certification` extra with mpmath, which no extra declares today.

**Harmonic host driver.**
* Boundary. `syncem.model.high_order_harmonic.high_order_channel_modes(channels, gamma, B, mu, eta, *, phase, direct_limit, quad_order, max_quad_order, rtol, atol, allow_unconverged)`, AST-pinned and unchanged in 0.8, behind the `HighOrderHarmonicKernel` adapter [0.9].
* Inputs. Bump channels and one point in G and cosines. Outputs. Channel modes with the route per channel (direct line sum or corrected integral) and the order samples used, as p7 records them.
* Ownership. A vectorised NumPy driver over points and order samples is a new module behind the adapter. The pinned module stays the oracle.
* Errors. An unconverged result raises unless `allow_unconverged=True`, which records it. The vectorised driver and any later native driver keep this rule.

## 7. Acceptance criteria and revisit triggers

Acceptance of the choices made now (thresholds set before comparison):
* S0 plus S1 lookups: `ln T` within 1e-10, `r = 2` within 1e-9 and `r = 6` within 1e-6 of the study quadrature at single field nodes from 0.1 to 100 uG and at the field rules of SPEC Sec. 2.3 (the thresholds of `profile/test_boundaries.py`).
* Batched LM: parameters within 1e-6 of `trf` and `chi2` within 1e-10 on the p5 set, with `trf` kept as the test oracle.
* NNLS pool: the same active sets and solutions as the serial run to 1e-12, and at least 4 times the serial throughput at 4 workers on a quiet machine.
* Tabulated F, G for the ray direct reference: within 1e-10 of quadrature F, G on both sides of the table end.
* Smooth tabulated F, G for the jet: jet columns within 1e-10 of the channel column maximum of the quadrature jet at `r <= 4`. The measured primal gain decides adoption.
* Vectorised harmonic driver: equal to the pinned module to 1e-12 relative on the boundary grid (`gamma` 5 to 3000, candidate counts 0.5, 1 and 2 times the direct limit, on the cone).

Revisit triggers:
* The harmonic driver exceeds about 0.3 ms per channel point on the p7 configuration after NumPy vectorisation. The floor of the line powers there is 27 to 75 us per channel point.
* Certification is used routinely: it enters a default path (plan, build or accuracy check) or a documented workflow makes more than 100 calls per run (2.2 to 2.4 min at 1.30 to 1.43 s per call). Then measure `gmpy2`, then `python-flint`/Arb, and consider Rust only if both miss the need.
* A profile shows a host path above half the wall time of an end-to-end fit or build on a representative configuration, after the changes of Sec. 4 are in place.
* A non-JAX consumer appears, for example a C, Fortran or service caller that needs the kernels without Python. A native core with a Python binding, and JAX primitives with derivative rules, then become a design question.

## 8. Risks

* Scientific equivalence. The fit-level effect of the S0/S1 split is unmeasured (RESULTS Sec. 7.2). The cubic Hermite F, G table is adequate for forward values only, not for jet derivatives beyond second order. The smooth table is a candidate and is not measured.
* Process pools. Forking a process that has initialised JAX can deadlock, so the pool uses the `spawn` start method. A library must not reset the caller's thread policy, so the BLAS limit applies in the worker processes only. Whether the environment variables or `threadpoolctl` limit Accelerate and OpenBLAS after import is to be verified on macOS and Linux.
* Timing. Absolute times are upper bounds under load, and the stack-to-jet ratio is not settled (Sec. 3.1, item 4).
* Later native code. A native kernel on a traced path needs JVP, transpose and batching rules and its own equivalence tests against the JAX oracle. `gmpy2` and `python-flint` wheels for the supported platforms are to be verified before the extra is declared.

## 9. Checks performed and proposed

Performed for this record: every number of Sec. 3 re-derived from `prototypes/profile/results/*.json` and `*.log`, with the corrections of Sec. 3.1. In the profiling interpreter, mpmath runs on its Python backend, and `gmpy2`, `python-flint`, `cargo` and `rustc` are absent. The kernel protocol (SPEC Sec. 3.2, 8.1) and the signatures of `certified_dense_channel_modes` and `high_order_channel_modes` were read in the source. The 42 profile boundary tests passed in the profile step (RESULTS Sec. 6) and were not rerun here.

Performed after the first draft: a quiet rerun of p2 (item 4 of Sec. 3.1). Proposed: p8 with `gmpy2` installed, a vectorised harmonic driver timed on the p7 configuration, a jet over a smooth tabulated F, G, a JAX port of the stack provider (0.9), the fit-level effect of the S0/S1 split, and the pinned process pool on Linux with OpenBLAS.
