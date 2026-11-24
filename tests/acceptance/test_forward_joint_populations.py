"""Paper E4-KP, E5 and E6 forward errors and E6 seeded UPRE (SPEC Sec. 9.6)."""

import numpy as np
import pytest

import syncem as se
from syncem.fitting.free_fit import fit_free
from tests.v080 import _study

pytestmark = pytest.mark.slow


def _max_error(out, truth):
    return np.max(np.abs(out.stokes - truth) / truth[:, :1])


def _build(population, channels, lattice, **options):
    return se.spectral_basis(
        population, channels, lattice=lattice, check=False, **options
    )


def test_e4_kp_pitch_projection_at_study_resolution():
    _study.verify_sums(_study.ACCEPTANCE_DATA / "SHA256SUMS")
    with np.load(_study.ACCEPTANCE_DATA / "forward_spectra.npz") as archived:
        truth = archived["E4_KP"]
    source, channels = se.paper.e4kp(), se.paper.channels()
    lattice = se.Lattice(gamma=se.paper.cells(order=4), mu=se.Legendre(8))
    basis = _build(source, channels, lattice)
    lower = basis.with_order(2)
    assert lower.columns.n == 285
    for model, error in ((lower, 8.8e-3), (basis, 6.6e-4)):
        assert _max_error(model.predict(source, check=False), truth) == pytest.approx(
            error, rel=0.02
        )
    assert _max_error(basis.direct(source), truth) <= 1e-10


def test_e5_joint_and_one_factor():
    _study.verify_sums(_study.ACCEPTANCE_DATA / "SHA256SUMS")
    with np.load(_study.ACCEPTANCE_DATA / "forward_spectra.npz") as archived:
        truth = archived["E5"]
    channels = se.paper.channels()
    gamma = se.paper.cells(order=4)
    pop = se.paper.e5()
    lattice = se.Lattice(gamma=gamma, phi=se.AzimuthHarmonic())
    basis = _build(pop, channels, lattice)
    for order, maximum in ((2, 1.76e-3), (4, 1.39e-4)):
        out = basis.with_order(order).predict(pop, check=False)
        assert _max_error(out, truth) == pytest.approx(maximum, rel=0.01)

    factor = se.AzimuthFactor(_study.paper().azimuth_factor)
    first, second = (part.replace(phi=factor) for part in pop.parts)
    independent = first + second
    separate = _build(independent, channels, se.Lattice(gamma=se.paper.cells(order=2)))
    assert _max_error(
        separate.predict(independent, check=False), truth
    ) == pytest.approx(2.90e-1, rel=0.01)


@pytest.fixture(scope="module")
def e6():
    study = _study.study()
    cfg = study.config.PAPER
    channels = se.paper.channels()
    pop = se.paper.e6()
    gamma = se.paper.cells(order=2)
    field = se.Cells.linear(4, 6, 1, unit="uG", order=2)
    lattice = se.Lattice(gamma=gamma, B=field, order=2)
    basis = _build(pop, channels, lattice)
    return study, cfg, basis, pop, gamma, study.e6.direct(cfg)


def test_e6_joint_and_field_averaged(e6):
    _, _, basis, pop, gamma, truth = e6
    assert basis.design().shape == (41, 3, 114)
    assert _max_error(basis.predict(pop, check=False), truth) == pytest.approx(
        7.8e-3, rel=0.01
    )
    independent = _build(pop, basis.channels, se.Lattice(gamma=gamma), average="B")
    assert _max_error(independent.predict(pop, check=False), truth) == pytest.approx(
        8.4e-2, rel=0.01
    )


def test_e6_independence_term_uses_the_checked_scaling_reference(e6):
    _, _, joint, pop, gamma, truth = e6
    basis = _build(pop, joint.channels, se.Lattice(gamma=gamma), average="B")
    spectrum = basis.predict(pop)
    term = spectrum.ledger["independence:gamma|B"]
    assert term.kind == "measured" and term.signed
    fraction = np.max(abs(np.asarray(term.value)) / truth[:, :1], axis=0)
    np.testing.assert_allclose(
        fraction, [0.07961845, 0.02173290, 0.01229480], atol=5e-8
    )


def test_e6_seeded_fit_and_rank_boundary(e6):
    study, cfg, basis, _, _, truth = e6
    rows = cfg.fit_idx
    sigma = 0.01 * truth[:, :1]
    observed = truth.copy()
    observed[rows] += sigma[rows] * cfg.draw_noise(len(rows))
    data = se.StokesData(basis.channels, stokes=observed, sigma=sigma, unit="cgs")
    with pytest.warns(RuntimeWarning, match="numerical rank cut"):
        fit = fit_free(basis, data.withhold(np.setdiff1d(np.arange(41), rows)))

    design = (study.e6.joint_columns(cfg)[rows] / sigma[rows, :, None]).reshape(63, -1)
    left, singular, _ = np.linalg.svd(design, full_matrices=False)
    numerical = int(np.count_nonzero(singular > 1e-13 * singular[0]))
    assert numerical == 25
    assert singular[numerical] / singular[0] == pytest.approx(9.94e-14, rel=0.002)
    whitened = (observed[rows] / sigma[rows]).ravel()
    projection = left[:, :numerical].T @ whitened
    scores = 2 * np.arange(1, numerical + 1) - np.cumsum(projection**2)
    rank = int(np.argmin(np.r_[0, scores]))
    expected = np.dot(whitened, whitened) - np.dot(projection[:rank], projection[:rank])
    assert fit.rank == rank == 9
    assert fit.dof == 54
    assert fit.chi2 == pytest.approx(expected, abs=1e-6)
