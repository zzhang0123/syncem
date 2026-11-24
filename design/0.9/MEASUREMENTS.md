# Kernel switch measurements

The reference model is the vacuum finite-gamma helical-orbit response.
These measurements use Python 3.12, NumPy 2.3.5 and SciPy 1.16.3.
Refinement measures finite-rule changes. The special-function and dense
line-sum errors are unbounded.

## Fixed pitch distribution

The switch comparison uses an isotropic pitch distribution at fixed
inclination. The reference integrates the finite-gamma natural I and Q
over the pitch cone using `studies/classical-models/harmonic_cells.py`.
The refined cone width and Gauss order were 60 and 64, compared with 40
and 32. The largest pairwise refinement at gamma 300 was (5.0\times10^{-10})
of the local reference. Differences below are finite-gamma divided by
leading continuum, minus one.

| Lorentz factor | Inclination cosine | Frequency / critical frequency | I | Q |
| --- | --- | --- | --- | --- |
| 100 | 0 | 1 | -1.20e-4 | -9.87e-5 |
| 300 | 0 | 1 | -1.33e-5 | -1.10e-5 |
| 300 | 0.5 | 1 | -8.98e-6 | -1.10e-5 |
| 300 | 0.9 | 0.01 | +4.15e-4 | -1.93e-4 |
| 300 | 0 | 35 | -2.34e-4 | -2.34e-4 |
| 300 | 0 | 266 | -1.77e-3 | -1.77e-3 |

For 41 unit-area bump channels with half-width 0.2, gamma 300, B = 5 uG,
and inclination cosine 0 or 0.5, the channel-level discrepancy is at
most 2.92e-5 in I and 1.53e-5 in Q relative to peak continuum I for centres
from 0.1 to 3 critical frequencies. Doubling the pitch order from 32 to 64
and the frequency order from 128 to 256 together changes these maxima by
at most 5.2e-9 of peak I. The same settings at 0.1–10 GHz give differences
of 3.32e-4 in I and 3.31e-4 in Q relative to the faint low-energy spectrum's
own peak. These measurements support 300 as a starting switch for the
paper geometry. They give no uniform error over angle or frequency.

## Harmonic cell refinement

The paper preset uses 16 log-energy cells of order four, 32 pitch nodes
per panel and 64 channel nodes. At fixed B = 5 uG, eta = 0.3, isotropic
pitch and zero depth, its checked build gives local I errors of +1.36e-9
and +3.89e-6 at one and five critical frequencies for a point mass just
below the switch. The reference integrates the dense real-order spectrum
directly. The corresponding below-to-above I changes are -1.48e-5 and
-4.70e-5. Across the five tested centres from 0.1 to 5 critical
frequencies, the largest absolute I and Q errors are 1.01e-7 and 9.48e-8
of peak I. With the base rule, seven points across the last cell give
local I errors through five critical frequencies below 7.3e-6.

At 0.1 GHz the fixed-geometry point-mass I is only 1.65e-49 cgs. Its
checked basis is about 22% different from the direct harmonic reference
there. Both branch representations have unbounded errors in the deep
tail. In the separate few-line example at gamma 10, B = 10 G and eta =
0.45, summing orders 1 and 2 against an adaptive pitch integral gives
I differences of 7.24e-5 and 3.00e-6 with 32 and 64 pitch nodes,
respectively.

For E1 under the paper's field, inclination and screen laws at 0.1 GHz,
the base rule gives a harmonic intensity fraction of 3.48e-19. The
checked build gives 3.79e-19. Doubling the energy cells with the refined
quadrature gives 3.89e-19. The largest Stokes changes relative to total I
are 3.09e-20 under build-rule refinement and 9.50e-21 under the combined
cell-width and interpolation-rule change. The maximum unweighted column
change is 2.97e-48 in cgs spectral units. The loaded three-build probe took
793 s and the one-minute load at completion was 19.30.

For E1M at the same channel, the three harmonic intensity fractions are
8.65e-11, 1.57e-11 and 1.58e-11. The maximum Stokes changes relative to
total I are 7.08e-11 and 3.93e-15, respectively. Here the joint refinement
of field, pitch, inclination, frequency and interpolation rules is much
larger than narrowing the energy cells. The independent integration below
gives a branch fraction of 2.91e-11. These measurements leave the fractional
error of the E1M harmonic tail unbounded. The loaded three-build probe
took 804 s and the one-minute load at completion was 3.24.

The comparison can be regenerated from the repository root.

```sh
PYTHONPATH=. python3 design/0.9/regime_switch_measurements.py
PYTHONPATH=. python3 design/0.9/regime_switch_measurements.py --section paper
```

## Routing and cost

At gamma 100, 4 uG and both cosines zero, two bump responses contain 4096
candidate lines and the third contains 4098. The integer and forced dense
routes differ by at most 3.98e-13 of the peak I. The first channel has
I = 4.29719182e-28 and Q = -4.29719182e-28 in cgs spectral units. The
routing tests cover this threshold and the finite-width and inclination
extremes. The continuum branch requires at least 32 harmonic spacings
in each finite support on the paired physical nodes. This is a regime
criterion. It supplies no error bound.

On a loaded machine at one-minute load 3.42, the frozen host fixed-point
route took 0.25–0.60 ms per channel for the 0.1, 1 and 10 GHz bands at
gamma 100 and 300. The active orders reached 2.57e11. For the new batched
route over all 41 paper channels, one field and inclination pair took
0.033–0.067 s with 64 frequency and 8 pitch nodes, or 0.123–0.249 s with
128 and 16 nodes. The one-minute load was 3.50–3.62. This implies about
172 s for the harmonic columns of E1 with two quadrature levels, before
building the continuum columns or pairing the population.

## Low-energy contribution at 0.1 GHz

An independent dense harmonic integration over gamma 100–300 with the
same electron density coefficient as above the switch gives intensity
fractions of 3.89e-19 for E1 and 2.91e-11 for E1M. These calculations use
an isotropic pitch law, an inclination cosine uniform on [-0.5, 0.5], a
bump half-width of 0.2 and the paper field distributions. E1M samples
fields above the 4–6 uG range of E1. The result is an estimate for this
channel and source. The regression test with a 5 mG field instead puts
48–53% of the intensity below the switch. Its summed I and complex P
differ from independently integrated branch spectra by at most 9.4e-8
of channel I, with a Gaussian screen applied at the emission frequency.

To run the threshold tests and the full two-branch regression from the
repository root:

```sh
python3 -m pytest -q tests/v090/test_harmonic_routing.py tests/v090/test_v090_two_branch_reference.py
```

## Initial end-to-end replay

The 41-channel, 400-node forward replay compared each rebuilt I, Q, U
spectrum to the archived direct continuum spectrum with the same original
density amplitudes. The largest discrepancy, scaled by I in each channel,
was 7.81e-7 across the eight ensembles. The E6 reference was rebuilt with
the frozen study driver. The source columns above the switch were one to
the precision of the numerical pairing. The E1 full build with refined
harmonic columns took 184 s, with 188 s through prediction. A replay of the
eight cases using the initial harmonic column rules and refined population
panels took 72.4 s, with 1.54 GB peak resident memory at initial one-minute
load 2.76. The initial low branch's fractional accuracy is poorer in the deep
radio tail. For E1 the two-cell low branch gives a maximum
contribution of 2.4e-18 of the total channel I, compared to the independent
0.1 GHz estimate at the scale of 3.9e-19. Finite-cell representation error
of that branch remains unbounded.

All eight seeded radio fits retained their original ranks. The fitted
chi-squares increased by 0.045 to 0.200 with the physical projection.
The effective diagonal E4-KP gamma moments on the original 19 cells agree
with the frozen reference to 3.5e-15 in the scaled moment coordinates.
The empirical-law comparisons run on the rebuilt synthetic spectra.

For E2 over 1–100 GHz and 500 draws per noise level, the initial 64-cell
median absolute relative I error at 300 GHz is listed below.

| Fractional noise | Method | Two-part replay | Archived |
| --- | --- | --- | --- |
| 0.01% | Free moments | 1.50% | 1.58% |
| 0.01% | Ridge | 1.67% | 1.66% |
| 0.01% | Nonnegative ridge | 1.27% | 1.26% |
| 0.01% | Nonnegative | 4.28% | 4.37% |
| 1% | Free moments | 16.79% | 15.74% |
| 1% | Ridge | 15.89% | 15.84% |
| 1% | Nonnegative ridge | 15.06% | 14.91% |
| 1% | Nonnegative | 25.71% | 25.60% |

These three workflows are covered by `tests/v090/test_paper_replay.py`.
The checks allow the physical projection's effect on the inverse response.
The tests use both branch responses even when the radio-band harmonic
term is faint. Run them with `python3 -m pytest -q -m slow tests/v090/test_paper_replay.py`.

## Physical inverse resolution

The first physical projection used interpolation of the effective-cell
response. At 0.01% observational noise that choice changed the fitted
chi-square in the broad radio bands. The log-energy derivative map removes
the interpolation across effective-cell faces for the specified-geometry
energy fit. At 64 effective cells, increasing the basis order from four
to eight reduced the maximum 300 GHz ratio error in four high-precision
E2/E3 seed cases from 7.77e-4 to 5.35e-7. The order-eight replay completed
160 family fits, including deliberate wrong-family fits. Across the 20
archived endpoint checks its maximum absolute ratio change was 8.69e-7.

The first E3 1–40 GHz draw at 0.01% noise was sensitive to cell width.
With 64 cells it selected UPRE rank 16 and gave a 300 GHz intensity ratio
of 0.286. The archived fit selected rank 12 and gave 0.8377326310595562.
With 128, 256 and 512 cells the rank was 12. The absolute ratio errors were
1.05e-7, 5.78e-10 and 4.9e-13, respectively. The same 512-cell resolution
reduces the first E2 1–100 GHz high-precision chi-square difference to
3.33e-8. It fits below the 5e-7 first-draw regression tolerance. The
512-cell probe peaked at 1.86 GB, with one-minute load 4.73 at start and
3.90 at its last sample. This fixes the two targeted cases without changing
the physical fit cells or their weights.

## Final 512-cell verification

The full paper replay passed at the final effective resolution, including
all eight forward spectra and seeded fits, the synthetic empirical-law
comparisons, 500 E2 extrapolations per noise level and the sensitive E3
seed. The checked E6 field projection uses 11 extra points per axis.
On the archived observations it selects rank 9 and reproduces the
chi-square to 9.4e-8.

For E2 and E3, the final 300 GHz median intensity errors are below. Each
entry gives per cent at 0.01% and 1% noise, respectively, from 500 shared
draws per noise level. All 32 entries round to the published values.
The largest absolute change from the archive is 1.71e-11 in fractional
units. The eight published E2 and E3 linear-polarisation entries also
round to the original values, using the complex P error divided by I.

| Population and fit band | Free | Ridge | Non-negative ridge | Non-negative |
| --- | ---: | ---: | ---: | ---: |
| E2, 1–40 GHz | 19.2718 / 44.9485 | 10.9252 / 46.8866 | 8.8950 / 45.8056 | 69.2520 / 95.2730 |
| E2, 1–100 GHz | 1.5808 / 15.7384 | 1.6612 / 15.8389 | 1.2588 / 14.9149 | 4.3742 / 25.6011 |
| E3, 1–40 GHz | 16.6901 / 64.7209 | 17.5800 / 61.2299 | 15.4060 / 58.9129 | 40.9597 / 84.7106 |
| E3, 1–100 GHz | 2.2972 / 25.6351 | 2.1921 / 25.4312 | 1.8303 / 24.2976 | 4.3269 / 16.8566 |

The 121-channel intensity replay used the current harmonic preset and
500 draws for each band, noise level and ensemble. It checked free and
non-negative ridge fits for E2, E3 and E4-JP. The largest change in the
16th, 50th or 84th percentile curve was 3.29e-7 of the reference power
law. The largest change in the median error curve was 7.62e-8 in the
figure's fractional error units. In the 1–100 GHz, 0.01% panels, the
largest change in a percentile curve was 1.10e-7 of the power law.
All individual non-negative ridge curves were within 6.15e-9 of the
larger of the archived prediction magnitude and the reference power law.
Of the 6,000 free-moment curves, 31 exceeded 1e-4 on that scale and the
largest change was 0.0220. All 6,000 UPRE ranks matched. A repeat with
archived observations retained the largest E4-JP outlier at the same
rank. The retained condition number on that draw is 5.22e12. The loaded
replay took 228 s. The one-minute load was 12.79 after
the basis build and 12.99 at completion.

Projecting the archived fitted moment and cumulant families through the
current 121-channel response gives a largest normalized draw change of
7.35e-10 across 24 combinations of ensemble, band, noise and family. The
largest envelope change was 1.52e-7 of the reference power law. The
largest median error curve change was 1.51e-7 in the figure's fractional
units. This calculation fixes the archived fit parameters. The endpoint
replay independently reruns the family fits.

The E4-JP unpenalised profile was replayed for the first 100 draws at
1% noise in each fit band. Its median 300 GHz intensity upper estimate is
0.2141743 of the power law for 1–40 GHz and 1.7492573e-5 for 1–100 GHz.
The corresponding archive medians are 0.2134922 and 1.7492566e-5. Both
results round to the reported values. All recovered weights were
non-negative. The largest recomputed wide-band chi-square increase was
1 + 6.56e-11. Recovery with a checked dual gap is specified in decision
192 of the [0.8 specification](../0.8/SPEC.md).

The endpoint moment and cumulant refits used 500 draws in each of 24
groups for E2, E3 and E4-JP, with the checked 16-cell harmonic and 512-cell
continuum response. All 12 printed family polarisation medians matched.
The largest absolute difference in these fractional error medians was
6.33e-9. Together with the linear and non-negative fits, all 24 printed
polarisation entries matched. The loaded family refits took 1,032 s at
one-minute loads 12.25 to 14.69.

```sh
OPENBLAS_NUM_THREADS=1 PYTHONPATH=. python3 design/0.9/replay_extrapolation.py /tmp/syncem-curves.json \
    --curves --no-refine --profile-draws 0 --methods free posridge
```

All five archived family-extrapolation campaigns passed at 500 draws
each, with the original 300 GHz endpoint, chi-square and median error
checks. The 4-draw checks passed as well. The run used the original stored
observations and noise draws and took 157 s.

The fit-summary replay compares true-moment prediction errors at orders
two and four and seeded fit statistics on regenerated radio spectra. It
includes the shared E5 azimuth, the E6 joint energy-field fit and the E6
field marginal. The empirical intensity replay uses the original five
spectral laws, both noise levels and 500 shared draws. It also checks the
noise-free rank-four fit and the capped-UPRE errors and ranks. GALPROP
retains its monochromatic continuum response.

The three intensity-only electron-family rows are regenerated with the
original least-squares procedure applied to the two-branch physical
columns. The current preset regenerated the spectra before fitting.
All 36 synthetic medians matched the archive to two significant figures.
The maximum absolute change in fractional RMS was 3.54e-10. The loaded
replay took 733 s at one-minute loads 14.12 to 7.40.

The parameter replay ran 1,000 draws each for E1, E2 with fixed
smoothness, E3 and E4-JP. All 4,000 UPRE ranks and the 24,000 one-standard-
deviation coverage classifications across the full and compressed fits
matched the archive. The largest displacement of a fitted parameter
was 9.42e-5 of its archived standard error. The largest fractional change
of a standard error was 2.22e-5. Runtime was 369 s, peak RSS was 1.09 GB and
the one-minute load was 2.95 at start and 3.93 at completion.

The default paper fit has 21 physical energy cells. The two cells below
gamma 300 extend the original 19-cell fit while preserving its
high-energy reference weights. At fixed density amplitude the E1 total
column becomes 5.799548 times the original unit column. In E5 the steep
zone fraction is 0.9378848 above the switch and 0.9783999 over the full
extended interval. The comparison of rebuilt and archived forward
spectra measures a change between implementations. The original
quadrature convergence check measured refinement within the continuum
reference model.

The final regression commands are:

```sh
python3 -m pytest -q tests/v090
python3 -m pytest -q -m slow tests/v090/test_paper_replay.py
python3 -m pytest -q -m slow tests/v090/test_paper_family_replay.py
python3 -m pytest -q -m slow tests/v090/test_paper_parameter_replay.py
OPENBLAS_NUM_THREADS=1 python3 -m tests.v090.replay_paper_baseline_families --workers 4
```

The final paper preset passed all six radio, empirical and E2
extrapolation tests. The separate family and parameter regression run
passed all 18 slow cases. The non-slow 0.8 and existing acceptance
suites passed all 1,558 selected tests. The 0.9.0 source distribution and
wheel passed strict Twine validation. An installation of the wheel
reported version 0.9.0 and imported the regime and profile interfaces.
