"""Expansion order, reference point and prescribed scalar absorption, continuum kernel.

Part A (Sec. 5.2): for E1, E2 and E4-JP, Taylor orders N = 0..4 and reference points moved
from each energy-cell centre by q h_b with q in {-1/2, 0, +1/2}:
  * forward error max_j |(C a)_j - I_j| / I_j over the 41 channels, with the true moments
    about the moved reference point (intensity only)
  * TSVD fits of the 21 fitted intensities at UPRE rank (R <= 14), one noise-free fit and
    200 realisations of 1% noise, and the rms error in the 20 withheld channels
    (16th, 50th and 84th percentiles over the realisations)
The direct spectrum uses 48 channel nodes and is checked against a refined quadrature
(96 channel nodes, 24 eta nodes, 32 field-strength nodes, 48 energy nodes per panel).

Part B (Sec. 5.4): E1 seen through a passive homogeneous slab with
tau_nu = tau_0.1 (nu / 0.1 GHz)^-2.1 for tau_0.1 in {0, 0.1, 1, 10}, the attenuation applied
before channel integration and no noise added. Two emission-only fits of the 21 fitted
intensities are compared on the withheld channels:
  * the free-coefficient N = 2 basis at the centred reference points (TSVD, UPRE rank)
  * nonnegative least squares on direct channel kernels at 192 and at 384 log-spaced
    energies on [300, 2.96e6], with a free normalisation per energy column. This tests a finite
    nonnegative grid, not all nonnegative electron distributions.

All errors are fractions of the noise-free channel intensity. Changing the reference point
also changes the TSVD metric, so Part A fits compare bases as well as orders.

Writes sensitivity_results.json.  Run: python sensitivity.py  (about 15 s, 1.2 GB peak memory)
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from math import comb, factorial
from pathlib import Path

import numpy as np
from scipy.optimize import nnls

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "lib"))
sys.path.insert(0, str(HERE))
import baseline_models as BM  # noqa: E402
import basis as B  # noqa: E402
import ensembles as E  # noqa: E402
import kernels as K  # noqa: E402
from config import PAPER  # noqa: E402
from provenance import provenance, write_json  # noqa: E402

T0 = time.time()
OUT = HERE / "sensitivity_results.json"
SEED, NREAL = 20261102, 200
OFFSETS = (-0.5, 0.0, 0.5)
ORDERS = tuple(range(5))
NAMES = ("E1", "E2", "E4-JP")
TAUS = (0.0, 0.1, 1.0, 10.0)
SLAB_ENSEMBLE = "E1"
NNLS_GRIDS = (192, 384)
REFINEMENT_TOL = 1e-6


def intensity_nodes(cfg, gamma, order=0):
    """Angle- and field-averaged continuum I and its first `order` ln-gamma derivatives.

    Values at each channel quadrature node, with shape (order + 1, n_gamma, J, T), before
    channel or electron integration. Only F is evaluated, so no polarisation or screen terms.
    """
    gamma = np.atleast_1d(gamma)
    table = B.get_table(cfg)
    out = np.empty((order + 1, len(gamma), cfg.n_channels, cfg.n_chan_nodes))
    eta_s, we = cfg.eta_s, cfg.eta_nodes_weights[1]
    per = max(1, 1_000_000 // (len(eta_s) * cfg.chan_nu.size))
    for start in range(0, len(gamma), per):
        g = gamma[start:start + per]
        ly = (np.log(cfg.chan_nu / K.a1)[None, None]
              - 2 * np.log(g)[:, None, None, None]
              - np.log(eta_s)[None, :, None, None])
        for r, values in enumerate(table.evaluate(ly, "F", order)):
            out[r, start:start + per] = (2**r * K.A1
                * np.einsum("gejt,e->gjt", values, eta_s * we))
    return out


def slab(tau):
    """Passive homogeneous slab attenuation (1 - e^-tau) / tau, equal to 1 at tau = 0."""
    tau = np.asarray(tau, float)
    result = np.ones_like(tau)
    np.divide(-np.expm1(-tau), tau, out=result, where=tau != 0)
    return result


def direct(cfg, ensemble, n_energy=24, tau=0.0):
    """Direct channel intensities, with the slab attenuation applied at each channel node."""
    measure = ensemble.measure(cfg, n_energy)
    nodes = intensity_nodes(cfg, measure.gamma)[0]
    monochromatic = np.einsum("g,gjt->jt", measure.w, nodes)
    attenuation = slab(tau * (cfg.chan_nu / 1e8)**-2.1)
    return np.einsum("jt,t->j", monochromatic * attenuation, cfg.chan_w)


def columns(cfg, order, offset):
    """Intensity basis columns with reference points gamma_b exp(offset h)."""
    pivots = cfg.gamma_b * np.exp(offset * cfg.h)
    values = intensity_nodes(cfg, pivots, order)
    averaged = np.einsum("rgjt,t->rgj", values, cfg.chan_w)
    factor = np.array([cfg.h**r / factorial(r) for r in range(order + 1)])
    return (averaged.transpose(2, 1, 0) * cfg.w_b[None, :, None]
            * factor[None, None, :]).reshape(cfg.n_channels, -1)


def shifted_moments(cfg, ensemble, order, offset):
    """True moments about the moved reference points, checked against the binomial shift."""
    measure = ensemble.measure(cfg)
    x = (np.log(measure.gamma / cfg.gamma_b[measure.b]) / cfg.h) - offset
    moments = np.zeros((cfg.n_bins, order + 1))
    for r in range(order + 1):
        np.add.at(moments[:, r], measure.b, measure.w * x**r)
    moments /= cfg.w_b[:, None]
    original = measure.moments(cfg, order)
    shifted = np.stack([sum(comb(r, k) * (-offset)**(r-k) * original[:, k]
                            for k in range(r + 1)) for r in range(order + 1)], axis=1)
    np.testing.assert_allclose(moments, shifted, rtol=3e-12, atol=3e-13)
    return moments.ravel()


def fit_predictions(matrix, truth, cfg, noise):
    """TSVD fits to the rows cfg.fit_idx only. The withheld rows never enter rank selection."""
    fi = cfg.fit_idx
    sigma = cfg.noise_frac * truth
    svd = BM.tsvd_setup(matrix[fi], sigma[fi])
    numerical_rank = int(np.sum(svd["s"] > svd["s"][0] * 1e-12))
    cap = min(BM.R_CAP, numerical_rank)
    predictions, ranks = [], []
    for z in noise:
        tf = BM.tsvd_fit(svd, truth[fi] + sigma[fi] * z[fi], sigma[fi], r_cap=cap)
        rank = tf["R_star"]
        predictions.append(matrix @ BM.tsvd_moments(svd, tf, rank))
        ranks.append(rank)
    return np.asarray(predictions), np.asarray(ranks), numerical_rank


def rms_fraction(prediction, truth, indices):
    return np.sqrt(np.mean(((prediction[..., indices] - truth[indices])
                            / truth[indices])**2, axis=-1))


def nonnegative_mixture(cfg, truth, n_grid):
    """Nonnegative mixture of direct channel kernels at n_grid log-spaced energies.

    Each energy column has a free normalisation, and there is no Taylor truncation. A poor
    fit is a property of this finite grid, not a proof that no nonnegative density fits.
    """
    gamma = np.geomspace(cfg.gamma_lo, cfg.gamma_hi, n_grid)
    matrix = np.einsum("gjt,t->jg", intensity_nodes(cfg, gamma)[0], cfg.chan_w)
    sigma = cfg.noise_frac * truth
    weighted = matrix[cfg.fit_idx] / sigma[cfg.fit_idx, None]
    scale = np.linalg.norm(weighted, axis=0)
    keep = scale > 1e-100
    weights, _ = nnls(weighted[:, keep] / scale[keep], truth[cfg.fit_idx] / sigma[cfg.fit_idx],
                      maxiter=20000)
    return matrix[:, keep] @ (weights / scale[keep])


def order_pivot_scan(cfg, ensembles, truth, noise):
    records = []
    for order in ORDERS:
        for offset in OFFSETS:
            matrix = columns(cfg, order, offset)
            for name in NAMES:
                forward = matrix @ shifted_moments(cfg, ensembles[name], order, offset)
                pred, ranks, nrank = fit_predictions(matrix, truth[name], cfg, noise)
                error = rms_fraction(pred, truth[name], cfg.held_idx)
                records.append(dict(name=name, order=order, offset=offset,
                    forward_max=float(np.max(abs(forward / truth[name] - 1))),
                    noiseless_held_rms=float(error[0]), numerical_rank=nrank,
                    noisy_held_rms_percentiles=np.percentile(error[1:], [16, 50, 84]).tolist(),
                    median_rank=float(np.median(ranks[1:]))))
        print(f"order {order} done", flush=True)
    return records


def absorption_test(cfg, refined, ensemble, checks):
    matrix = columns(cfg, 2, 0.0)
    records, transparent = [], None
    for tau in TAUS:
        target = direct(cfg, ensemble, tau=tau)
        fine = direct(refined, ensemble, n_energy=48, tau=tau)
        checks[f"slab_tau{tau}_refinement"] = float(np.max(abs(target / fine - 1)))
        if tau == 0.0:
            transparent = target
        pred, ranks, _ = fit_predictions(matrix, target, cfg, np.zeros((1, cfg.n_channels)))
        record = dict(tau=tau, rank=int(ranks[0]),
            free_train_rms=float(rms_fraction(pred[0], target, cfg.fit_idx)),
            free_held_rms=float(rms_fraction(pred[0], target, cfg.held_idx)))
        for n_grid in NNLS_GRIDS:
            mix = nonnegative_mixture(cfg, target, n_grid)
            record[f"nnls{n_grid}_train_rms"] = float(rms_fraction(mix, target, cfg.fit_idx))
            record[f"nnls{n_grid}_held_rms"] = float(rms_fraction(mix, target, cfg.held_idx))
        records.append(record)
        print(f"slab tau={tau}: {record}", flush=True)
    return records, transparent


def main():
    cfg = replace(PAPER, n_chan_nodes=48)
    refined = replace(cfg, n_chan_nodes=96, n_eta=24, n_b=32)
    ensembles = E.reference_ensembles(cfg)
    noise = np.random.default_rng(SEED).normal(size=(NREAL, cfg.n_channels))
    # Row 0 is the noise-free fit, kept apart from the NREAL realisations.
    all_noise = np.vstack([np.zeros(cfg.n_channels), noise])
    checks, truth = {}, {}
    for name in NAMES:
        truth[name] = direct(cfg, ensembles[name])
        fine = direct(refined, ensembles[name], n_energy=48)
        checks[f"{name}_refinement_max_fraction"] = float(np.max(abs(fine / truth[name] - 1)))
        print(f"direct {name}: refinement {checks[f'{name}_refinement_max_fraction']:.3g}", flush=True)
    records = order_pivot_scan(cfg, ensembles, truth, all_noise)
    # Independent check of the intensity-only integration against basis.kbar.
    kI, _ = B.kbar(cfg, cfg.gamma_b, order=2)
    own = np.einsum("rgjt,t->rgj", intensity_nodes(cfg, cfg.gamma_b, 2), cfg.chan_w)
    checks["intensity_vs_existing_kbar_scaled"] = float(np.max(abs(own - kI)) / np.max(abs(kI)))
    np.testing.assert_allclose(own, kI, rtol=3e-11, atol=1e-290)
    checks["transparent_limit"] = float(np.max(abs(slab(np.zeros(4)) - 1)))
    assert checks["transparent_limit"] == 0
    slab_records, transparent = absorption_test(cfg, refined, ensembles[SLAB_ENSEMBLE], checks)
    np.testing.assert_array_equal(transparent, truth[SLAB_ENSEMBLE])
    worst = max(v for k, v in checks.items() if "refinement" in k)
    if worst >= REFINEMENT_TOL:
        raise AssertionError(f"direct spectrum not converged: {worst:.3g} >= {REFINEMENT_TOL}")
    write_json(OUT, dict(
        description="Expansion order and reference-point scan (Sec. 5.2) and prescribed "
                    "scalar-absorption slab (Sec. 5.4), continuum kernel, intensity only.",
        seed=SEED, nreal=NREAL, config=cfg.describe(), refined_config=refined.describe(),
        offsets=OFFSETS, orders=ORDERS, slab_ensemble=SLAB_ENSEMBLE, nnls_grids=NNLS_GRIDS,
        errors="fractions of the noise-free channel intensity, no extrapolation outside the band",
        limitations="finite continuum tests, a moved reference point also changes the TSVD "
                    "metric, passive scalar slab, finite nonnegative grids only",
        checks=checks, sensitivity=records, absorption=slab_records,
        provenance=provenance(__file__, T0, inputs=(HERE / "baseline_models.py",))))
    print(f"wrote {OUT.name} in {time.time() - T0:.1f} s")


if __name__ == "__main__":
    main()
