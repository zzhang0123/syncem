"""How the reference weights w_b (the metric of the free-moment fit) change in-band fits and 300 GHz extrapolation.

Same data, same basis spectra, same estimators (TSVD at the UPRE rank, Tikhonov at the UPRE alpha) as
fits_positive_ridge.py. Only the column scaling changes: a_br = M_br / w_b with w_b from different reference laws.
"""
import sys, os
ST = str(__import__("pathlib").Path(__file__).resolve().parents[4] / "studies/classical-models")
sys.path.insert(0, ST); os.chdir(ST)
import numpy as np
import fits as F
import extrap_curves as EC
import fits_cmb_bands as CB
import fits_free_positive as P
import fits_mixture_forms as FM
import fits_positive_ridge as PR
from config import PAPER

N_REAL = int(sys.argv[1]) if len(sys.argv) > 1 else 200
cfg = PAPER
first = 3  # cells 4-19
edges = cfg.edges[first:]
def law_mass(p):
    if p == 1.0:
        m = np.diff(np.log(edges))
    else:
        k = 1.0 - p
        m = np.diff(edges ** k) / k
    return m / m.sum()
w_paper = cfg.w_b[first:] / cfg.w_b[first:].sum()
assert np.allclose(w_paper, law_mass(2.6))

ens = F.E.reference_ensembles(PAPER)
Z = CB.shared_noise()
nu = EC.DENSE_CFG.centres / 1e9
cases = {
    "paper gamma^-2.6": (law_mass(2.6), 1.0),
    "gamma^-2.0": (law_mass(2.0), 1.0),
    "gamma^-3.2": (law_mass(3.2), 1.0),
    "equal per cell (w=1)": (np.ones_like(w_paper) / w_paper.size, 1.0),
    "gamma^-2.6, u^r scaled 0.3^r": (law_mass(2.6), 0.3),
}
print(f"n_real={N_REAL}. Entries: median |I_fit/I - 1| at 300 GHz in per cent (E4-JP: in units of I_PL), "
      "and the median UPRE rank / ridge n_eff")
for name in ("E2", "E3", "E4-JP"):
    S_full, S_dense = FM.truth(ens, name, CB.FULL_CFG), FM.truth(ens, name, EC.DENSE_CFG)
    t_ipl = S_dense[:, 0] / FM.reference_law(nu, S_dense[:, 0])(nu)
    key = "dI_over_IPL_300GHz_median_p16_p84" if name == "E4-JP" else "dI_over_I_300GHz_median_p16_p84"
    cf0 = P.columns(name, CB.FULL_CFG, 2)
    cd0 = P.columns(name, EC.DENSE_CFG, 2)
    for band, n_fit in (("1-40", 21), ("1-100", 26)):
        for frac in (1e-2, 1e-4):
            row = []
            for label, (w, ord_scale) in cases.items():
                # columns carry w_paper; rescale each cell to w, and order r by ord_scale^r
                f = np.repeat(w / w_paper, 3) * np.tile(ord_scale ** np.arange(3), w.size)
                cf, cd = cf0[:n_fit] * f, cd0 * f
                S = S_full[:n_fit]
                sig = frac * np.repeat(S[:, 0], 3)
                A = cf.reshape(-1, cf.shape[-1]) / sig[:, None]
                D = np.stack([(S + (frac * S[:, :1]) * Z[i, :n_fit]).ravel() for i in range(N_REAL)]) / sig[None, :]
                R = PR.upre_rank(A, D)
                a_free = np.stack([PR.tsvd(A, D[i:i + 1], int(R[i]))[0] for i in range(N_REAL)])
                s_free, _ = PR.summarise(a_free, A, D, cd, S_dense, t_ipl)
                a_r, _, neff = PR.ridge(A, D)
                s_ridge, _ = PR.summarise(a_r, A, D, cd, S_dense, t_ipl)
                row.append((label, 100 * s_free[key][0], np.median(R), 100 * s_ridge[key][0], np.median(neff),
                            s_free["chi2_median"]))
            print(f"\n{name} {band} GHz noise {frac:g}")
            for label, ef, r, er, ne, c2 in row:
                print(f"  {label:32s} free {ef:9.3f} (R*={r:4.1f}, chi2 {c2:6.1f})   ridge {er:9.3f} (n_eff {ne:5.1f})")
