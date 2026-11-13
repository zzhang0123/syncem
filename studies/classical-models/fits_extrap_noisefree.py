"""Noise-free minimum-norm extrapolation of the free-moment basis in the extrapolation test.

For E2, E3 and E5 at the median R* of each noise level in fits_extrap_results.json, with the setup
of fits_extrap.py (fit 1-40 GHz, predict 40-300 GHz, cells 4-19):
  * the largest noise-free minimum-norm prediction error |dI|/I over 40-300 GHz;
  * the share of the true 300 GHz intensity carried by the directions outside span(V_R);
  * the norm fraction of the 300 GHz intensity row outside span(V_R).

Reads fits_extrap_results.json (python fits_extrap.py).
Writes fits_extrap_noisefree_results.json.  Run: python fits_extrap_noisefree.py
"""
from __future__ import annotations

import json
import time
from dataclasses import replace
from pathlib import Path

import numpy as np

import fits_extrap as X

F = X.F
HERE = Path(__file__).resolve().parent
RESULTS = HERE / "fits_extrap_results.json"
OUT = HERE / "fits_extrap_noisefree_results.json"
T0 = time.time()


def one_row(name: str, R: int, cf: dict, cp: dict, Sf: dict, Sp: dict, af: dict) -> dict:
    """Noise-free diagnostics of one ensemble at rank R (F.CFG sets the noise level)."""
    key = "joint" if name == "E5" else "sep"
    cpn = cp[key]
    C = np.concatenate([cf[key], cpn], axis=0)
    S = np.concatenate([Sf[name], Sp[name]], axis=0)
    nk = X.KEEP.size
    cplx = (np.arange(nk, 2 * nk), np.arange(2 * nk, 3 * nk)) if name == "E5" else None
    lf = F.LinearFit(name, C, S, af[name], cplx=cplx)
    pred = F.B.predict(cpn, lf.a_hat(lf.Sf @ lf.u, R))
    err = np.abs(pred[:, 0] - Sp[name][:, 0]) / Sp[name][:, 0]
    V = lf.vt[:R].T
    a_proj = V @ (V.T @ lf.a_true)  # true moments projected on the retained directions
    c = cpn[-1, 0, :]  # 300 GHz intensity row
    return {"R": R, "noisefree_max_dI_over_I": float(err.max()),
            "share_300GHz_discarded": float((c @ (lf.a_true - a_proj)) / (c @ lf.a_true)),
            "row300_norm_fraction_outside_VR": float(np.linalg.norm(c - V @ (V.T @ c)) / np.linalg.norm(c))}


def main() -> None:
    summary = json.loads(RESULTS.read_text())["summary"]
    ens = F.E.reference_ensembles(X.PAPER)
    cf, _ = X.columns(X.FIT_CFG)
    cp, _ = X.columns(X.PRED_CFG)
    Sf, af, _ = X.truth(X.FIT_CFG, ens)
    Sp, _, _ = X.truth(X.PRED_CFG, ens)
    nf, npred = X.FIT_CFG.n_channels, X.PRED_CFG.n_channels
    F.FIT, F.HELD = np.arange(nf), np.arange(nf, nf + npred)
    out = {}
    for frac in X.NOISE:
        tag = f"{frac:g}"
        F.CFG = replace(X.PAPER, noise_frac=frac)
        for name in X.ROWS:
            R = int(round(summary[tag][name]["basis"]["R_star_median"]))
            out.setdefault(tag, {})[name] = rec = one_row(name, R, cf, cp, Sf, Sp, af)
            print(f"{tag} {name} R={R} noise-free max error {100 * rec['noisefree_max_dI_over_I']:.1f}%, "
                  f"300 GHz row outside V_R {rec['row300_norm_fraction_outside_VR']:.3f}")
    F.CFG = X.PAPER
    F.write_json(OUT, {"description": __doc__, "results": out,
                       "provenance": F.provenance(__file__, T0, inputs=(RESULTS, HERE / "fits_extrap.py",
                                                                        HERE / "fits.py"))})
    print("done", round(time.time() - T0, 1), "s")


if __name__ == "__main__":
    main()
