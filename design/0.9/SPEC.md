# SyncEM 0.9 paper response

This extension implements the population response in `eq: reduced general`,
`eq: reduced continuum`, `eq: continuum joint response` and `eq: cell sum`.
The scope is the existing forward, inference, comparison and extrapolation
analyses of the manuscript, using intensity and linear polarisation. The two
regimes share physical population declarations and observing channels.

## Partition and normalisation

Partition the physical electron measure at `gamma_switch`. The harmonic part
has `gamma < gamma_switch` and the continuum part owns point masses at the
switch. Integration over either part retains the density or column weights
of the original measure. A population with conditional laws is partitioned
at its physical energy node, after conditioning on the relevant parents.

The starting configuration for the paper is `gamma_min = 100` and
`gamma_switch = 300`. The paper populations keep their existing density
amplitudes and high-energy cutoffs. The extra interval increases their total
column. The original unit-column convention still identifies the column above
300. Record the actual mass of each branch separately from Stokes flux.

## Harmonic basis

Keep physical `(gamma, B, depth)` cells and project the natural I and Q kernels
on Legendre degrees of the pitch and inclination cosines when resolved. The
coefficient of degree `l` includes `(2l+1)/2` in the angular integral when the
moment contains the unnormalised Legendre polynomial. Specified laws can be
averaged into columns and evaluated at the moving parent nodes. Accumulate the
azimuth as `exp(2i phi)` and apply the depth phase at each emitted frequency.

Use the host real-order Bessel line powers. Where the channel contains few
lines, sum positive integer orders. For dense smooth channels integrate over
order using the existing Euler–Maclaurin route or its change of variable
`m = nu / spacing`. The integral uses `dm = dnu / spacing`. State the support
conditions of the dense route. Reused rows must have the same response on
the channel support. Callable or custom responses are rebuilt when this
identity is unavailable. A ridge coordinate around `mu = eta` resolves
the pitch integral. The value and cell interpolants have unbounded numerical
error. Record Stokes or column changes under refinement of angular, channel
and cell interpolation rules as estimates.

## Continuum basis

Above the switch, first put `mu = eta` in the joint law and apply the factor
of two from the beaming delta function. Define
`nu_c = A1_HZ_PER_G * B * sqrt(1 - eta**2) * gamma**2`. A reference field `B0`
sets the basis amplitude. The pushed-forward measure carries
`2 p_mu(eta | parents) * B_perp / B0` when integrating over `eta`. If the relation
is evaluated in reverse, the inclination density supplies the diagonal factor.
Carry `exp(2i phi)` in the spin-2 moment and retain depth or average its independent
phase with its declared characteristic function. A phase average conditional on
an emitting variable must use that variable's current node. The beaming route
needs a density in at least one cosine. Where both orders of integration
are admissible, use the first continuous cosine law at each set of parent
values. A point conditional can drive the integral against the other cosine's
density if the order satisfies the dependencies. The sum separates its moments in cells
of `ln(nu_c)` and depth from the harmonic moments. Use physical cell faces as
quadrature boundaries where a change of variables yields them. Otherwise
check the projected moment integration by refining its physical panels.

For free fits the two sets of columns are concatenated on the same
observations. An analysis with stipulated physical moments or a family uses
the map from those variables into the two responses, including physical
column weights, screen and azimuth. Non-negative physical cell masses use
the same map in their constraints. An independent linear mechanism may use
its own coordinates in the same observation likelihood (0.8 decision 188).

The continuum approximation also requires many lines across each channel.
In the radiating cone the order spacing is approximately
`e * B / (2*pi*gamma*m_e*c*(1 - beta*mu*eta))`. For projected fits this
spacing is checked over the energy and field cell intervals and on the
fixed-geometry quadrature used by the projection. This check also applies
when a fit is evaluated on new channels. It does not certify unsampled
fixed geometry. Record any regime restriction when this scale cannot be
checked for the population. A monochromatic
empirical comparison, without a physical harmonic population and a finite
response, keeps the continuum interpretation of the supplied data.

## Acceptance

Compare both kernels at the switch, around the emitting band (`x = nu/nu_c`
near unity) and in the bright and faint extremes. Compare the dense and
integer harmonic routes at their count threshold. Pin the integrated
reference and refine against both branch-specific direct integrals. Include
physical populations crossing the switch for which both branches contribute
measurably. Use the same I, Q and U analysis settings as the study. The
acceptance campaign covers the synthetic forward figures, seeded fits,
empirical comparisons, positivity, extrapolation, family fits and parameter
coverage. Compare absolute Stokes at preserved electron amplitudes and
check the inferred quantities in their original physical conventions.
Document changes in numerical results from the new coordinates.

## Decision log

| # | Decision | Basis |
|---|---|---|
| 1 | Use disjoint harmonic and continuum bases on their own coordinates. Include only I and P. | T-001 r27 and r28. |
| 2 | Evaluate the harmonic branch with host real-order Bessel line powers and record refinement without a strict bound. | T-001 r27. |
| 3 | Extend the study's physical energy support while preserving its density amplitudes. | T-001 r27 and r28 require the existing analyses. Renormalising the longer range would change the in-band intensity even in the limit of zero harmonic emission. |
| 4 | Retain the proposed switch and floor. Require 32 spacings on the sampled continuum measure and record the switch discrepancy with its channel, field and angular conditions. | T-001 r27 delegates the numerical choice. See [the numerical comparisons](MEASUREMENTS.md). |
| 5 | Use the normalized physical-cell Taylor view for the study inverse problems. Transfer by local interpolation of unit-atom responses. Preserve physical minimum-norm weights and fit cuts. | The free critical-frequency moments alone do not enforce the study’s field, azimuth or physical energy constraints. |
| 6 | On log-energy cells with specified geometry and an independent screen, differentiate the continuum moment map at the physical reference. Keep local interpolation for the harmonic branch and the other physical views. | In each effective cell, the local log-frequency coordinate is affine in log energy with slope two before scaling. Differentiating through samples in different effective cells can mix their finite-cell truncations. |
| 7 | Use order eight on the default effective cells for the paper workflow. Keep the physical fit order and scale separate from this provider order. | A first-draw comparison for E2 and E3 in the two high-precision extrapolation bands reduced the maximum absolute 300 GHz intensity-ratio discrepancy from 7.77e-4 at order four to 5.35e-7 at order eight. |
| 8 | Use 512 log critical-frequency cells in the paper preset. | At order eight and 64 cells, a near-tied UPRE selection changed the first E3 high-precision 1–40 GHz rank from 12 to 16. Both 256 and 512 cells recovered rank 12. At 512 cells the first E2 high-precision chi-square difference was 3.3e-8, within the archived first-draw tolerance. |
| 9 | In the paper view with both energy and field cells, use 11 interpolation points beyond each axis degree. | The E6 seeded rank remains 9, and the chi-square difference on the archived observations falls from 0.2002 with three extra points to 9.4e-8 with eleven. |
| 10 | Store reusable regime quadrature and interpolation rules in the shared byte-limited cache. Use its S0 stage for the critical-frequency primitive tables. | The same memory budget covers the 0.8 and 0.9 tables. See 0.8 decision 160 and T-001 r35. |
| 11 | Skip dense-route Bessel evaluations whose absolute upper bound rounds to zero in float64. Keep the integer route and dense support checks. | Far from the emitting pitch cone the real-order Bessel powers are below the representable range. The cutoff lies below the smallest float64 subnormal. |
| 12 | Use 16 log-energy cells of order four with 32 pitch nodes per panel for the paper harmonic response. Keep order two in the harmonic field axis. Use two low-energy cells in the default physical fit. | The sampled last-cell representation error through five critical frequencies falls below 8e-6 of local harmonic intensity. The physical fit coordinates and weights stay independent of both provider meshes. |
| 13 | Reuse harmonic rows only when the response at line frequencies is identified. Test the 32-spacing continuum condition throughout the physical fit cells at the sampled fixed geometry. Determine the beaming diagonal from each evaluated conditional law when either orientation is possible. Report lattice coverage from the pairing used in the returned spectrum. | Package review T-001 r41. A continuum quadrature fingerprint cannot distinguish all channel responses between its nodes. The source used to build a projected basis can have different support from the fit. A conditional can return a point law on part of its parent support. |
| 14 | Allow an unresolved leaf depth law conditioned on physical gamma, field strength, pitch or viewing cosine. Evaluate its characteristic on harmonic emission frequencies and continuum band nodes before reducing the physical variables. Fit using physical-cell columns and keep the characteristic truncation term unbounded. | T-001 r41 package review. A critical-frequency moment alone cannot determine a phase law that varies along its physical preimage. |
| 15 | Offer an opt-in forward harmonic basis in the three spectral-shape variables. Keep an exact channel response in line spacing, interpolate the Bessel amplitudes in argument ratio and retain the signed polarisation ratio algebraically. | T-002. The channel response and Faraday phase can vary on the scale of individual lines. The three shape variables omit the absolute electron power. Forming Stokes powers from interpolated amplitudes preserves the single-line polarisation relation. |
| 16 | Retain a reusable weighted joint rule with shape coordinates, Stokes weights and depth marks. Use the stored float64 line spacing for both response and phase. Preserve the physical reference frequencies in the coordinate comparison. | T-002 r5 independent acceptance. The joint push-forward must be reusable without the original geometry. At open channel edges the physical multiplication order can select different lines for identical reduced coordinates. |

For decision 11, let `n = m - 1 >= 1`, `z = x/n < 1` and
`t = sqrt(1 - z^2)`. The Bessel equation in `log(z)` gives
`d^2 J_n(nz) / d log(z)^2 = n^2 (1-z^2) J_n(nz)`.
The logarithmic derivative starts above `n sqrt(1-z^2)` at small `z` and
cannot cross it from above because the latter decreases while the former
has zero derivative at equality. Integrating up to `z = 1` and using
`|J_n(n)| <= 2` gives `|J_n(x)| <= 2 exp[-n(atanh(t)-t)]`.
The same upper bound applies to order `m+1` since the exponent increases
with order at fixed argument. Use the weaker lower bound `n t^3/3 > 800`
on the exponent. The margin allows evaluation of the cutoff in float64.

## Package entry points

`paper.population(name)` supplies the extended source declaration. The
existing presets `paper.e1()` through `paper.e6()` supply the 0.8 reference
configuration. `paper.regime_basis(name, channels)` builds both branches.
`RegimeBasis.build` accepts custom physical and critical-frequency lattices.
The public `RegimeBasis.predict` returns I, Q and U with a refinement ledger.
Its plan records the physical low- and high-energy columns separately when
pairing is checked. The upper reduced zeroth moment carries a field-weighted
column and cannot be read as the original source column.

```python
import syncem as se

source = se.paper.population("E2")
channels = se.paper.channels()
response = se.paper.regime_basis("E2", channels)
truth = response.predict(source)
physical = se.paper.energy_basis(response, source)
noise = 0.01 * truth.I[:, None]
data = se.StokesData(channels, stokes=truth.stokes, sigma=noise, unit="cgs")
fit = se.fit(source.replace(gamma=se.FreeMoments()), data, basis=physical)
```

The projected design can use the same physical reference weights, physical
cell selections and inference methods as the study. Its scaled physical
moments map to both reduced lattices. The log-energy derivatives use the
local polynomial in the effective cell containing each reference point.
Jump terms from changes in cell ownership are outside that derivative
approximation. The harmonic and general product-cell maps use local
interpolation of the response to physical mass points. E5 resolves azimuth and E6 resolves
field strength on this fit view. The separable KP fit uses the JP basis
statistics and `paper.effective_kp_moments` gives its diagonal comparison
moments. The harmonic forward KP response uses its pitch Legendre rows.

`RegimeBasis.fit` directly fits independent moments in both reduced lattices.
Its retained likelihood can also be used by `RegimeBasis.fit_family` with
`keep_directions=True`. Family parameters remain in physical coordinates.
Use the projected fit view for the paper's physical regularization and
positive-cell constraints. For a projected fit cut, select the physical
cells and retain their weights before passing the basis to `se.fit`.
A fitted atom at the switch belongs to the continuum side of that cut.

The dense harmonic driver is limited to smooth bump channels. Finite
non-bump passbands are supported on the integer route. Custom depth phases
without a support envelope carry an unbounded harmonic depth term. The
paper entry point uses 16 harmonic energy cells of order four and 512
critical-frequency cells of order eight by default. The default fit uses
two cells below the switch and the original 19 above it. Branch column
and panel refinements measure numerical changes. Direct population
comparisons are required to assess finite-cell truncation.

## Optional harmonic spectral-shape basis

`HarmonicShapeBasis.build(channels, spacing=(lo, hi), z_edges=..., order=...)`
builds a reusable finite basis for specified physical populations. Use
`basis.predict(population)` or `se.predict(population, channels, basis=basis)`
for I, Q, U and signed V. The existing physical-variable construction is
the default.

At each physical node let `D = 1 - beta_parallel * eta`,
`nu1 = omega_B / (2*pi*D)`, `z = beta_perp * sqrt(1-eta**2) / D`,
and `r = (eta-beta_parallel) / (beta_perp*sqrt(1-eta**2))`.
In the off-axis emitting domain use the equivalent bounded coordinate
`s = z*r = (eta-beta_parallel)/D`. The physical mass is multiplied by
`A = (2*pi*e**2/c) * nu1**2 * beta_perp**2/D` in Gaussian cgs.
Use the physical joint rule and its weights after evaluating any conditional
laws. Different physical nodes with the same shape can have different
`A`, azimuth or depth.

`basis.moments(population)` returns `HarmonicShapeStatistics`, the
empirical joint push-forward on `(nu1, z, s)`. Each atom carries its
power-weighted mass for I and V and its power-weighted spin-two mass for
Q and U. The complex mass carries a depth value or a resolved depth-law
characteristic. A depth law that is not a parent of another variable
contributes a characteristic only to the complex mass. Use its
characteristic when available or normalise its finite probability rule.
Identical shape coordinates can have several atoms.
Evaluate the characteristic against the phase at the actual line
frequency when applying the basis. The statistics have no channel or
foreground phase dependence. `basis.predict(stats)` attaches the statistics
to the returned spectrum. Prediction from a physical population streams
the same push-forward. The materialised rule has a configurable atom budget.

The statistics archive stores numeric arrays, finite depth rules and the
parameters of the built-in depth characteristics. An opaque custom
characteristic can be used in memory and saving it raises. Loading verifies
the archive and does not load executable objects.

For each positive integer order `m` define `H = (J_{m-1}(m*z) +
J_{m+1}(m*z))/2` and `T = (J_{m-1}(m*z) - J_{m+1}(m*z))/2`.
Use the natural Stokes line powers
`A*m**2*(s**2*H**2 + T**2, s**2*H**2 - T**2, 2*s*H*T)`
for I, Q and V. Tabulate `m*H` and `m*T` at Chebyshev–Lobatto nodes
in each declared `z` cell and interpolate the amplitudes with cardinal
polynomials. Products of these polynomials give the finite Stokes basis
with twice the amplitude degree. Evaluate both the channel response and
the propagation phase at the float64 product `m*nu1` of the stored spacing.
Channel support edges are open. This sets the line frequency convention
of the reduced rule, including for atoms that lie on edges after rounding.
Multiply the Q moment weights by
`exp(2i*phi)` and the phase at the actual line frequency, including any
declared screen. Depth and azimuth can also be physical nodes. Use a leaf
law's characteristic function when it is supplied.

Zero field or perpendicular velocity gives zero emission. Use
`Delta(0, unit="G")` for a zero-field point law. At
`eta = +/-1` with nonzero perpendicular velocity use the physical
axis limit. Only the fundamental emits, with `I=A/2`, `V=eta*I`
and `Q=U=0`. Emitting nodes need a finite positive spacing in the
declared interval in Hz. Off-axis emitting nodes also need a ratio in
the declared `z` cells. The initial route uses finite positive passbands
and a budget on the distinct integer orders across the spacing interval.
A request beyond that budget raises before constructing the table. It
uses neither a dense real-order integral nor a harmonic-order truncation.

With checking enabled, compare the finite prediction with a direct sum
using the same transformed nodes. A prediction from the physical
population also compares that sum with the existing physical harmonic
line powers and their own line frequencies on the same physical rule.
Record both signed differences where available. For a standalone rule
the coordinate term remains unbounded because the original geometry
is absent. Special-function and roundoff errors and the physical
population quadrature error remain unbounded. Check absolute channel IQUV
for representative and limiting geometry, including both signs of V,
correlated physical nodes, a frequency-dependent Faraday phase, sharp
passbands and the spacing and ratio boundaries. Check finite-cell refinement
against the transformed direct sum. Check standalone replay on other
channels and with another phase, archive round-trip, coincident shapes
with different depth marks, and exact and adjacent floats at sharp edges.
