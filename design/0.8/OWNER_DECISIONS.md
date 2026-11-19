# Owner decisions on SPEC Sec. 14 (2026-11-17)

Decided by Zheng in conversation. Numbers are the question numbers of SPEC
Sec. 14. "Rec." means the recommended option was chosen.

| # | Question | Decision |
|---|---|---|
| 16 | Size of 0.8.0 | (a) everything in 0.8.0, about 72 days (Rec.), plus 4 days for Q19 |
| 1 | `Population` keywords | (a) `gamma, B, mu, eta, phi, depth` (Rec.) |
| 14 | Default reference weights | (a) `gamma^-2.6` with a printed line (Rec.) |
| 15 | Amplitude convention | (a) `dN/dgamma` at the pivot, `paper_coefficient` converts. No paper change (Rec.) |
| 11 | E4-KP normalisation in `se.paper` | follow the study code, state it in the docstring (Rec.) |
| 2 | Undeclared field direction | (a) isotropic, `Q = U = 0` stated (Rec.) |
| 3 | Undeclared electron normalisation | (a) amplitude 1 at `gamma = 1e4`, free in a fit (Rec.) |
| 7 | Rank rule for maps | (a) UPRE per pixel with a histogram, cap 14 documented for intensity-only maps (Rec.) |
| 5 | Source of continuum basis spectra | (a) derivative columns, table as second product (Rec.) |
| 13 | Paper and API link | (a) paper keeps citing v0.7.10, acceptance tests are the link (Rec.) |
| 12 | Two conventions until 1.0 | (a) accept, README pointer (Rec.) |
| 4 | SciPy | **(b) base dependency** (not the recommended extra) |
| 8 | Disk cache | (a) on, 2 GB, least recently used removed first (Rec.) |
| 9 | Cell numbering | (a) from zero (Rec.) |
| 6 | Basis class name | (a) `SpectralBasis` (Rec.) |
| 10 | `PointMasses` | (a) 0.9, tagged `[extension]` (Rec.) |
| 17 | Density jump inside a `mu` cell under `Beaming` | (a) `RoleError` naming the edges (Rec.) |
| 19 | Laws conditional on the azimuth | **(b) `AzimuthFourier(m_max)` in 0.8.0**, about 4 more days (not the recommended deferral) |
| 20 | `max_columns` | (a) 20 000 with a refusal that prints sizes (Rec.) |
| 18 | Chain-rule columns in `eta` | (a) a `Similarity` extension in 0.9 only if the jet is slow on `gamma x eta` lattices (Rec.) |

Consequences, applied in SPEC version 6 and TOPLEVEL on 2026-11-18:
* Q4: `scipy` moves from extras into `dependencies` in the package layout and
  the infrastructure constraints. The lazy-import design stays, since import
  time matters. The publish smoke test installs SciPy.
* Q19: `AzimuthFourier(m_max)` becomes a 0.8.0 representation for the azimuth
  with its truncation term and acceptance tests in step 7. R4a (b) routes
  instead of raising. The effort total becomes about 76 days.

# Owner decisions on the step-5b plan (2026-11-19)

Decided by Zheng in conversation. Numbers are the question numbers of Sec. 6
of the step-5b plan (revision 2, task T-001). "Rec." means the proposed option
was chosen.

| # | Question | Decision |
|---|---|---|
| 1 | Memory of one-node S1 tables | the proposal of SPEC decision 162 (Rec.): a byte-bounded in-memory store with default 1e9 bytes, one-node tables rebuilt on eviction, batching of per-node leaves from S0 deferred |
| 2 | `SYNCEM_CACHE` inside a synced folder | a warning, not an error, for a directory inside `~/Documents`, `~/Desktop`, `~/Library/Mobile Documents` or `~/Library/CloudStorage` (Rec.). The default is the platform cache directory (SPEC decision 161) |
| 163 (8) | Builds with S2 column reuse | "只要不失科学性和正确性,按你推荐的来" (follow the recommendation as long as scientific soundness and correctness are kept). T14 states a bound of 1e-13 of the channel maximum for builds with S2 column reuse to order 5 on the jet and order 6 on the stack and names its source, the batch dependence of an uncached build (SPEC decision 164) (Rec.) |

Question 3 (study oracles read in place or copied) is open. SPEC decision 154
stands as committed until it is answered. Row 163 (8) is item 8 of SPEC
decision 163, an open point of the step-5b-2 review and not a plan question.
The owner delegated it on 2026-11-19, and SPEC decision 164 settles it.
