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
