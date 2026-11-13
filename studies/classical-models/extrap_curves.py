"""Dense 1-300 GHz model curves of one realisation of the extrapolation test (for SED-fit panels).

Uses the fits_extrap.py setup and its first noise realisation (seed 20261112, 1% noise): fits on the
21 channels over 1-40 GHz exactly as fits_extrap.py does (basis TSVD/UPRE, electron families through
the basis, spectral laws with P = c <I_law screen>), then evaluates every fitted model and the truth
on 121 log-spaced channels over 1-300 GHz with the same channel response.

The dense grid shares its first channel (1 GHz) with the fitted grid and its last channel (300 GHz)
with the prediction grid, so verify_extrap.py compares the curves there with fits_extrap_products.npz.

Reads fits_extrap_products.npz (python fits_extrap.py).
Writes extrap_curves.npz: nu_dense (Hz), nu_fit, data_fit_<E> (21,3), sigma_fit_<E> (21,),
truth_<E> (121,3), <E>_<model> (121,3), R_star_<E>; and extrap_curves.json with the fitted
parameters and a provenance block.  Run: python extrap_curves.py
"""
from __future__ import annotations

import sys
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent / "lib"))
import baseline_models as BM  # noqa: E402
import fits as F  # noqa: E402
import fits_extrap as X  # noqa: E402
import spectral_laws as SL  # noqa: E402
from config import PAPER  # noqa: E402

HERE = Path(__file__).resolve().parent
PRODUCTS = HERE / "fits_extrap_products.npz"
DENSE_CFG = replace(PAPER, nu_lo=1e9, nu_hi=3e11, n_channels=121)
FRAC = 0.01
T0 = time.time()


def main() -> None:
    ens = F.E.reference_ensembles(PAPER)
    cf, full_f = X.columns(X.FIT_CFG)
    cp, _ = X.columns(X.PRED_CFG)
    cd, full_d = X.columns(DENSE_CFG)
    Sf, af, _ = X.truth(X.FIT_CFG, ens)
    Sp, _, _ = X.truth(X.PRED_CFG, ens)
    Sd, _, _ = X.truth(DENSE_CFG, ens)
    nf, npred = X.FIT_CFG.n_channels, X.PRED_CFG.n_channels
    F.FIT, F.HELD = np.arange(nf), np.arange(nf, nf + npred)
    F.CFG = replace(PAPER, noise_frac=FRAC)
    z0 = np.random.default_rng(X.SEED).standard_normal((X.N_REAL, nf, 3))[0]
    g_fit = SL.channel_grid(X.FIT_CFG.chan_nu, X.FIT_CFG.chan_w)
    g_dense = SL.channel_grid(DENSE_CFG.chan_nu, DENSE_CFG.chan_w)
    out = {"nu_dense": DENSE_CFG.centres, "nu_fit": X.FIT_CFG.centres, "nu_pred": X.PRED_CFG.centres}
    params = {}
    ref = np.load(PRODUCTS)
    for name in X.ROWS:
        key = "joint" if name == "E5" else "sep"
        d = Sf[name] + FRAC * Sf[name][:, :1] * z0
        sig = FRAC * Sf[name][:, 0]
        out[f"data_fit_{name}"], out[f"sigma_fit_{name}"], out[f"truth_{name}"] = d, sig, Sd[name]
        C = np.concatenate([cf[key], cp[key]], axis=0)
        nk = X.KEEP.size
        cplx = (np.arange(nk, 2 * nk), np.arange(2 * nk, 3 * nk)) if name == "E5" else None
        lf = F.LinearFit(name, C, np.concatenate([Sf[name], Sp[name]]), af[name], cplx=cplx)
        sc = lf.scan((lf.Sf + z0.ravel())[None], lf.Sh[None])
        R = int(sc["R_star"][0])
        out[f"R_star_{name}"] = R
        out[f"{name}_basis"] = F.B.predict(cd[key], lf.a_hat(sc["proj"][0], R))
        if name != "E5":
            for fam in X.FAMILIES:
                res = BM.fit_family(fam, full_f.reshape(-1, full_f.shape[-1]), d.ravel(), np.repeat(sig, 3))
                out[f"{name}_{fam}"] = F.B.predict(full_d, res["params"][0] * BM.family_moments(fam, res["params"][1:]).ravel())
                params[f"{name}_{fam}"] = {"params": res["params"].tolist(), "chi2": res["chi2"],
                                           "dof": int(3 * nf - BM.family_n_params(fam))}
        for law in X.LAWS:
            res = SL.fit(law, g_fit, d[:, 0], sig)
            shape_fit = X.screened(law, res["params"], X.FIT_CFG)
            w = 1.0 / sig**2
            c = np.sum(w * np.conj(shape_fit) * (d[:, 1] + 1j * d[:, 2])) / np.sum(w * np.abs(shape_fit) ** 2)
            P = c * X.screened(law, res["params"], DENSE_CFG)
            out[f"{name}_{law}"] = np.stack([SL.predict(law, res["params"], g_dense), P.real, P.imag], axis=-1)
            params[f"{name}_{law}"] = {"params": res["params"].tolist(), "chi2_I": res["chi2"]}
        # consistency with fits_extrap.py: the dense curves at the prediction centres reproduce seed-0 predictions
        jj = [int(np.argmin(np.abs(DENSE_CFG.centres - n))) for n in X.PRED_CFG.centres]
        params[f"{name}_check_note"] = "dense grid nodes nearest to prediction centres (not identical channels)"
        params[f"{name}_basis_pred_seed0_max_rel_diff_nearest"] = float(
            np.max(np.abs(out[f"{name}_basis"][jj, 0] - ref[f"{name}_0.01_basis_pred_seed0"][:, 0]) / Sp[name][:, 0]))
    np.savez(HERE / "extrap_curves.npz", **out)
    F.write_json(HERE / "extrap_curves.json", {"description": __doc__, "noise_frac": FRAC, "fits": params,
                                               "provenance": F.provenance(__file__, T0, inputs=(PRODUCTS, HERE / "fits_extrap.py"))})
    print({k: v for k, v in params.items() if "max_rel_diff" in k})
    print("done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
