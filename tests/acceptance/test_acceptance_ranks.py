"""Shapes and numerical ranks of the E1 and E5 designs (design/0.8/SPEC.md Sec. 9.6 row 1, decision 154).

The designs come from ``engine.basis.build_from_averaged`` on the PAPER configuration of ``studies/classical-models``
(19 energy cells at N = 2, 41 channels over 0.1 to 10 GHz, B ~ U[4, 6] uG and eta on the study's nodes, the
Gaussian Faraday screen). E1 is the separable design with the azimuth factor 0.4 e^{0.5i}, E5 the joint design
with ``AzimuthHarmonic``. The noise is sigma = 0.01 I_j of the direct spectrum of each ensemble in
``tests/acceptance/data/forward_spectra.npz`` (``ST/lib/config.py:185-188``). Rows are in (channel, Stokes) order,
and the rank counts the singular values with ``s > 1e-13 s_max`` (``ST/fits.py:74-91``). Fitted rows are the
channels ``arange(0, 41, 2)``.

The ranks are exact (SPEC). The margins pin the last kept and the first dropped ratio ``s / s_max`` per S1 path to
2 per cent, the plan's tolerance (step 5b plan Sec. 4). The RED probe ran on 2026-11-19 before ``basis_sum.py``
existed, at a 1-minute load of 7.7 to 8.0 (script ``probe_ranks.py`` in the session scratchpad). It measured, with
the study oracle columns (``lib/basis.py``) for comparison:

=================  ============  ============  ============
case               quadrature    lookups       oracle
=================  ============  ============  ============
E1 fit last kept   1.271729e-12  1.271728e-12  1.271730e-12
E1 fit first drop  5.867392e-14  5.843465e-14  5.867784e-14
E1 all last kept   4.329405e-13  4.329342e-13  4.329583e-13
E1 all first drop  6.898220e-15  7.072992e-15  6.895323e-15
E5 fit last kept   5.365906e-11  5.365906e-11  5.365906e-11
E5 fit first drop  3.396943e-17  3.220960e-17  3.135759e-17
E5 all last kept   3.416656e-13  3.416656e-13  3.416655e-13
E5 all first drop  4.614592e-15  4.611034e-15  4.601337e-15
=================  ============  ============  ============

The narrowest margin is the E1 first dropped value on the fitted rows, a factor 1.7 below the cut on both paths.
The E1 values on all rows differ by 2.5 per cent between the paths, so each path has its own pin. The E5 first
dropped values lie more than a factor 20 below the cut and sit at the rounding floor of the SVD, so they are held
below 5e-14 only. ``ST/fits_results.json`` stores the last kept ratios on the fitted rows (``checks.
eps_outside_column_span_lstsq``): 1.271730482692782e-12 for E1 and 5.365906393423703e-11 for E5, both within
2 per cent of the measured values on both paths. Builds took 10.6 s with the quadrature S1 and 1.4 s with lookups
for E1 and E5 together. Against a copy of HEAD 0df9b8d, which has no ``syncem.engine.basis``, the test fails at
collection with ``ModuleNotFoundError: No module named 'syncem.engine.basis'`` ("1 error in 0.52s", load 2.5).
"""

import numpy as np
import pytest

from syncem.base.channels import Channels
from syncem.base.continuum import ContinuumKernel
from syncem.base.phase import FaradayRotation
from syncem.engine.average import AveragedKernel, fixed_rule
from syncem.engine.basis import build_from_averaged
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import AzimuthHarmonic
from syncem.model.phase import GaussianScreen
from tests.v080 import _study

DATA = _study.ACCEPTANCE_DATA
FORWARD_SHA256 = (
    "9b9469c65fc5df489fc65f4d058decba15a48efe2ecb6e03ff47990f0f00841c"  # ST/SHA256SUMS
)
CUT = 1e-13  # ST/fits.py svd_signed rtol
NOISE = 0.01  # sigma = 0.01 I_j, ST/lib/config.py noise_frac
RTOL = 0.02  # plan Sec. 4: margins to 2 per cent
FLOOR = 5e-14  # first dropped values at the rounding floor are held below this only
RANKS = {("E1", "fit"): 26, ("E1", "all"): 28, ("E5", "fit"): 61, ("E5", "all"): 77}
STORED = {"E1": 1.271730482692782e-12, "E5": 5.365906393423703e-11}  # fits_results.json
LAST_KEPT = {
    ("E1", "fit"): {"quadrature": 1.271729e-12, "lookup": 1.271728e-12},
    ("E1", "all"): {"quadrature": 4.329405e-13, "lookup": 4.329342e-13},
    ("E5", "fit"): {"quadrature": 5.365906e-11, "lookup": 5.365906e-11},
    ("E5", "all"): {"quadrature": 3.416656e-13, "lookup": 3.416656e-13},
}
FIRST_DROPPED = {
    ("E1", "fit"): {"quadrature": 5.867392e-14, "lookup": 5.843465e-14},
    ("E1", "all"): {"quadrature": 6.898220e-15, "lookup": 7.072992e-15},
}
CASES = [(e, r) for e in ("E1", "E5") for r in ("fit", "all")]


def build(cfg, lattice, s1, factor):
    """The PAPER basis on ``lattice``: the study's B and eta nodes as fixed rules, B in gauss."""
    b, wb = cfg.b_nodes_weights
    levels = [
        ("B", fixed_rule(b * 1e-6, wb)),
        ("eta", fixed_rule(*cfg.eta_nodes_weights)),
    ]
    avg = AveragedKernel(
        ContinuumKernel(x_min=1e-12),
        levels,
        phase=FaradayRotation(screen=GaussianScreen(cfg.screen_mean, cfg.screen_sd)),
    )
    ch = Channels.log(0.1, 10, cfg.n_channels, unit="GHz", half_width=0.2, n_nu=400)
    return build_from_averaged(avg, ch, lattice, s1=s1, azimuth_factor=factor)


@pytest.fixture(scope="module")
def forward():
    sums = _study.verify_sums(DATA / "SHA256SUMS")
    assert sums["forward_spectra.npz"] == FORWARD_SHA256
    with np.load(DATA / "forward_spectra.npz") as data:
        return {name: np.asarray(data[name]) for name in ("E1", "E5")}


@pytest.fixture(scope="module", params=["quadrature", "lookup"])
def designs(request):
    cfg = _study.paper()
    gamma = Cells.log(300, cfg.gamma_hi, cfg.n_bins, order=2)
    e1 = build(cfg, Lattice(gamma=gamma), request.param, cfg.azimuth_factor)
    e5 = build(cfg, Lattice(gamma=gamma, phi=AzimuthHarmonic()), request.param, None)
    return request.param, {"E1": np.asarray(e1.design()), "E5": np.asarray(e5.design())}


def ratios(design, direct, rows):
    """``s / s_max`` of the whitened design on ``rows``, rows in (channel, Stokes) order."""
    sigma = NOISE * direct[:, 0:1] * np.ones((1, 3))
    a = (design[rows] / sigma[rows][:, :, None]).reshape(-1, design.shape[-1])
    s = np.linalg.svd(a, compute_uv=False)
    return s / s[0]


def rows_of(which):
    cfg = _study.paper()
    return cfg.fit_idx if which == "fit" else np.arange(cfg.n_channels)


def test_configuration_matches_the_study():
    cfg = _study.paper()
    assert cfg.noise_frac == NOISE and cfg.n_channels == 41 and cfg.n_bins == 19
    np.testing.assert_array_equal(cfg.fit_idx, np.arange(0, 41, 2))


def test_design_shapes(designs):
    _, d = designs
    assert d["E1"].shape == (41, 3, 57)
    assert d["E5"].shape == (41, 3, 171)


@pytest.mark.parametrize(("ensemble", "which"), CASES)
def test_numerical_ranks_and_margins(designs, forward, ensemble, which):
    s1, d = designs
    r = ratios(d[ensemble], forward[ensemble], rows_of(which))
    rank = int(np.sum(r > CUT))
    assert rank == RANKS[ensemble, which]
    assert r[rank - 1] == pytest.approx(LAST_KEPT[ensemble, which][s1], rel=RTOL)
    pinned = FIRST_DROPPED.get((ensemble, which))
    if pinned is None:
        assert r[rank] < FLOOR
    else:
        assert r[rank] == pytest.approx(pinned[s1], rel=RTOL)


@pytest.mark.parametrize("ensemble", ["E1", "E5"])
def test_last_kept_on_fitted_rows_equals_the_stored_study_value(
    designs, forward, ensemble
):
    _, d = designs
    r = ratios(d[ensemble], forward[ensemble], rows_of("fit"))
    assert r[RANKS[ensemble, "fit"] - 1] == pytest.approx(STORED[ensemble], rel=RTOL)
