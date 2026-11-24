"""Rank, singular pairs and the whitened estimator from SPEC Sec. 6.2."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from syncem.fitting.rank import (
    FullRank,
    Rank,
    UPRE,
    rank_cut,
    reduce_svd,
    solve_svd,
    warn_rank_margin,
)
from tests.acceptance.test_acceptance_ranks import build
from tests.v080 import _study
from syncem.lattice.cells import Cells
from syncem.lattice.lattice import Lattice
from syncem.lattice.representations import AzimuthHarmonic


def test_strict_rank_cut_and_signed_nearest_margin():
    singular = jnp.array([[1.0, 2e-13, 0.9945e-13], [2.0, 2e-13, 1e-13]])
    cut = jax.jit(rank_cut)(singular)
    np.testing.assert_array_equal(cut.numerical_rank, [2, 1])
    np.testing.assert_allclose(cut.margin, [-0.0055, 0.0], atol=2e-16)
    np.testing.assert_array_equal(cut.index, [2, 1])
    with pytest.warns(RuntimeWarning, match="nearby singular value"):
        warn_rank_margin(cut)
    zeros = rank_cut(jnp.zeros(2))
    assert int(zeros.numerical_rank) == 0 and not bool(zeros.near)


def test_complex_pair_rotation_and_whole_pair_rank():
    # Real form of the one-dimensional complex linear map z -> (2 + i) z.
    design = jnp.array([[2.0, -1.0], [1.0, 2.0]])
    reduction = jax.jit(lambda d: reduce_svd(d, complex_columns=((0,), (1,))))(design)
    assert reduction.paired.tolist() == [True]
    np.testing.assert_allclose(reduction.directions, np.eye(2), atol=2e-16)
    np.testing.assert_allclose(
        (reduction.left * reduction.singular) @ reduction.directions, design, atol=1e-15
    )
    out = solve_svd(reduction, jnp.array([7.0, 1.0]), UPRE())
    assert out.rank == 2
    np.testing.assert_allclose(out.moments, np.linalg.solve(design, [7.0, 1.0]))
    assert solve_svd(reduction, jnp.array([7.0, 1.0]), UPRE(cap=1)).rank == 0
    with pytest.raises(eqx.EquinoxRuntimeError, match="splits a singular pair"):
        solve_svd(reduction, jnp.array([7.0, 1.0]), Rank(1))
    real = reduce_svd(jnp.eye(2))
    assert solve_svd(real, jnp.array([7.0, 1.0]), Rank(1)).rank == 1


def test_shared_noise_batch_upre_matches_vmap_and_lstsq():
    design = jnp.array([[6.0, 0.0, 0.0], [0.0, 3.0, 0.0], [0.0, 0.0, 1.0]])
    data = jnp.array([[8.0, 0.5, 0.1], [4.0, 3.0, 0.1], [4.0, 3.0, 4.0]])
    reduction = reduce_svd(design)
    batch = jax.jit(solve_svd)(reduction, data)
    pointwise = jax.vmap(lambda d: solve_svd(reduction, d))(data)
    for a, b in zip(batch, pointwise):
        np.testing.assert_allclose(a, b, atol=5e-15)
    np.testing.assert_array_equal(batch.rank, [1, 2, 3])
    np.testing.assert_allclose(batch.chi2, [0.26, 0.01, 0.0], atol=1e-15)
    full = solve_svd(reduction, data, FullRank())
    np.testing.assert_allclose(full.moments, np.linalg.solve(design, data.T).T)
    np.testing.assert_allclose(
        full.sigma, np.broadcast_to(1 / np.diag(design), data.shape)
    )


@pytest.mark.parametrize(
    "ensemble, expected_rank, expected_chi2",
    [("E1", 7, 52.13245518691838), ("E5", 17, 43.16593433381058)],
)
def test_seeded_study_reduction_and_rounding_at_e5_pair_57(
    ensemble, expected_rank, expected_chi2
):
    cfg = _study.paper()
    gamma = Cells.log(300, cfg.gamma_hi, cfg.n_bins, order=2)
    if ensemble == "E1":
        lattice = Lattice(gamma=gamma)
        factor = cfg.azimuth_factor
        pair_columns = None
    else:
        lattice = Lattice(gamma=gamma, phi=AzimuthHarmonic())
        factor = None
        # The joint design columns are in lattice label order, with I before real and imaginary.
        pair_columns = (tuple(range(57, 114)), tuple(range(114, 171)))
    basis = build(cfg, lattice, "lookup", factor)
    design = np.asarray(basis.design())[cfg.fit_idx]
    with np.load(_study.ACCEPTANCE_DATA / "forward_spectra.npz") as spectra:
        truth = spectra[ensemble][cfg.fit_idx]
    sigma = 0.01 * truth[:, :1]
    data = (truth / sigma + cfg.draw_noise(len(cfg.fit_idx))).reshape(-1)
    W = (design / sigma[..., None]).reshape(len(data), -1)
    reduction = reduce_svd(W, complex_columns=pair_columns)
    solution = solve_svd(reduction, data)
    assert int(solution.rank) == expected_rank
    assert float(solution.chi2) == pytest.approx(expected_chi2, abs=1e-6)
    if ensemble == "E5":
        assert bool(reduction.pair_rounding_limited[57])
        assert not bool(reduction.paired[57])


def test_e6_actual_fit_margin_is_below_the_cut():
    cfg = _study.paper()
    truth = _study.study().e6.direct(cfg)
    design = _study.e6_joint_columns(cfg, 2)
    fitted = cfg.fit_idx
    white = (design[fitted] / (0.01 * truth[fitted, :1, None])).reshape(
        3 * len(fitted), -1
    )
    # The stored probe addresses the cut on the study's fitted rows. Use its
    # singular-value-only path to pin this margin independently of vectors.
    values = np.linalg.svd(white, compute_uv=False)
    cut = rank_cut(values)
    assert bool(cut.near)
    assert float(cut.margin) == pytest.approx(-0.0055, abs=0.003)
