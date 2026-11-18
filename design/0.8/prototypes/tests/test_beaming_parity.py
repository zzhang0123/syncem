"""(d) Half range and parity masks under Beaming (Spec Sec. 4.3 checks (a) to (d), r3-04).

Kernel: E1 continuum kernel at gamma = 1e4, B ~ U[4, 6] uG, bump channels, Gaussian screen.
mu is resolved by Legendre(L) (eta weights (2l+1) P_l(eta)) or averaged in the beaming limit.

Two references.
* ``direct``: the beaming integral int q(eta) 2 p(eta) Kbar(eta) deta on a graded rule (panels
  refined geometrically towards |eta| = 1, where Kbar has an end-point singularity through
  sqrt(1 - eta^2)). It measures the law's quadrature error as well.
* ``full``: the same column construction with the full-range rule whose nodes are the mirror of
  the half-range nodes, all degrees kept. A shortcut is exact when it equals this to rounding.
Run: PYTHONPATH=.:../../.. python -m pytest tests/test_beaming_parity.py
"""
from __future__ import annotations

import numpy as np
import pytest

from rules.beaming import beaming as BM
from rules.beaming import cases as CS
from rules.beaming.record import record

LAWS = {  # name -> intervals of a uniform eta law
    "even_U(-0.5,0.5)": [(-0.5, 0.5)],
    "even_isotropic": [(-1.0, 1.0)],
    "even_near0_U(-0.02,0.02)": [(-0.02, 0.02)],
    "even_near1_|eta|>0.95": [(-1.0, -0.95), (0.95, 1.0)],
    "asym_U(0,0.5)": [(0.0, 0.5)],
    "asym_near1_U(0.95,1)": [(0.95, 1.0)],
}
EVEN = {k for k in LAWS if k.startswith("even")}
LS = (3, 8, 16)
N_RULE = 12   # nodes per half-range interval, the inclination default of Spec Sec. 2.3
_K: dict = {}


def kernel(nodes):
    key = tuple(np.round(np.asarray(nodes, float), 15))
    if key not in _K:
        _K[key] = CS.kernel_at_eta(np.asarray(nodes, float))
    return _K[key]


def graded_breaks(a, b, levels=10):
    """Breaks on [a, b] refined geometrically towards an end at |eta| = 1."""
    pts = {a, b, 0.5 * (a + b)}
    for end, inner in ((b, a), (a, b)):
        if abs(abs(end) - 1.0) < 1e-15:
            d = abs(end - inner)
            pts |= {end - np.sign(end) * d * 0.5**k for k in range(1, levels)}
    return sorted(pts)


def reference_rule(intervals, n=32):
    xs, ws = [], []
    for a, b in intervals:
        x, w = BM.gl_panels(graded_breaks(a, b), n)
        xs.append(x)
        ws.append(w)
    x, w = np.concatenate(xs), np.concatenate(ws)
    return x, w / w.sum()


def half_rule(intervals, n=N_RULE):
    pos = [(max(a, 0.0), b) for a, b in intervals if b > 0]
    return BM.union_law(pos, n)


def plain_rule(intervals, n=N_RULE):
    return BM.union_law(intervals, n)


def pitch(name, L):
    if name == "exp_k2":
        dens = CS.exp_pitch(2.0)
        return dens, CS.legendre_moments_numeric(dens, L)
    if name == "even_poly":
        dens, mom = CS.poly_pitch(CS.EVEN)
        return dens, mom(L)
    dens, mom = CS.poly_pitch(CS.ASYM)
    return dens, mom(L)


def beaming_sum(eta, q, dens):
    I, P = kernel(eta)
    w = q * 2.0 * dens(eta)
    return CS.stokes(I @ w, P @ w)


def legendre_predict(eta, probs, moments, L, mask_odd=False, factor=2.0):
    I, P = kernel(eta)
    W = BM.pitch_weights(eta, probs, L, factor, mask_odd)
    return CS.stokes(I @ W @ moments, P @ W @ moments)


def rel(pred, ref):
    return float(np.max(np.abs(pred - ref) / np.abs(ref[:, :1])))


def tagged(out):
    return {k: [v, BM.verdict(v)] for k, v in out.items()}


# ------------------------------------------------------------------ averaged eta, mu on Legendre(L)
@pytest.mark.parametrize("L", LS)
@pytest.mark.parametrize("pname", ["poly_odd", "exp_k2"])
@pytest.mark.parametrize("law", list(LAWS))
def test_half_range_and_mask_with_averaged_eta(law, pname, L):
    iv = LAWS[law]
    dens, mom = pitch(pname, L)
    direct = beaming_sum(*reference_rule(iv), dens)
    if law in EVEN:
        half = half_rule(iv)
        full = BM.mirror(*half)
        base = legendre_predict(*full, mom, L)
        preds = {"full_mask_odd": legendre_predict(*full, mom, L, mask_odd=True),
                 "half_mask_odd": legendre_predict(*half, mom, L, mask_odd=True),
                 "half_all_l": legendre_predict(*half, mom, L)}
    else:
        full = plain_rule(iv)
        base = legendre_predict(*full, mom, L)
        a, b = iv[0]  # the half-range shortcut read off the support: the even law on +-[a, b]
        sym = BM.mirror(*full) if a >= 0 else full
        preds = {"full_mask_odd": legendre_predict(*full, mom, L, mask_odd=True),
                 "as_even_law_mask_odd": legendre_predict(*sym, mom, L, mask_odd=True)}
    out = {"full_all_l_vs_direct": rel(base, direct)}
    out.update({f"{k}_vs_full_all_l": rel(v, base) for k, v in preds.items()})
    out.update({f"{k}_vs_direct": rel(v, direct) for k, v in preds.items()})
    out = tagged(out)
    record(f"d_averaged_eta|{law}|{pname}|L{L}", out)
    tol = 1e-12
    if law in EVEN:
        assert out["full_mask_odd_vs_full_all_l"][0] < tol   # odd columns vanish for an even law
        assert out["half_mask_odd_vs_full_all_l"][0] < tol   # half range exact with the mask
        assert out["half_all_l_vs_full_all_l"][0] > 1e-3     # half range without the mask fails
    else:
        assert out["full_mask_odd_vs_full_all_l"][0] > 1e-3  # the mask drops real emission
        assert out["as_even_law_mask_odd_vs_full_all_l"][0] > 1e-3


@pytest.mark.parametrize("law", list(LAWS))
def test_law_quadrature_against_graded_reference(law):
    """The 12-node rule of each law (24 for a mirrored even law) against the graded reference, isotropic pitch."""
    iv = LAWS[law]
    dens = CS.poly_pitch({0: 1.0})[0]
    direct = beaming_sum(*reference_rule(iv), dens)
    out = {}
    for n in (12, 24, 48):
        rule = BM.mirror(*half_rule(iv, n)) if law in EVEN else plain_rule(iv, n)
        out[f"n{n}"] = rel(beaming_sum(*rule, dens), direct)
    fine = beaming_sum(*reference_rule(iv, 64), dens)
    out["reference_32_vs_64"] = rel(direct, fine)
    out = tagged(out)
    record(f"d_law_quadrature|{law}", out)
    assert out["reference_32_vs_64"][0] < 1e-9


def test_odd_column_ratio_for_asymmetric_law():
    """J2 finding 1: for eta ~ U[0, 1/2] the l = 1, 3 columns relative to l = 0 (continuum kernel here)."""
    eta, q = plain_rule([(0.0, 0.5)], 24)
    I, _ = kernel(eta)
    cols = I @ BM.pitch_weights(eta, q, 3)
    r1, r3 = cols[:, 1] / cols[:, 0], cols[:, 3] / cols[:, 0]
    out = {"l1_over_l0_I_range": [float(r1.min()), float(r1.max())],
           "l3_over_l0_I_range": [float(r3.min()), float(r3.max())]}
    record("d_odd_columns_U(0,0.5)", out)
    assert np.all(np.abs(r1) > 0.5) and np.all(np.abs(r3) > 1.0)


@pytest.mark.parametrize("law", sorted(EVEN))
def test_odd_columns_vanish_on_full_range_of_even_law(law):
    half = half_rule(LAWS[law])
    eta, q = BM.mirror(*half)
    I, P = kernel(eta)
    W = BM.pitch_weights(eta, q, 16)
    cI, cP = I @ W, P @ W
    odd = max(float(np.max(np.abs(cI[:, 1::2])) / np.max(np.abs(cI[:, 0]))),
              float(np.max(np.abs(cP[:, 1::2])) / np.max(np.abs(cP[:, 0]))))
    hI = kernel(half[0])[0] @ BM.pitch_weights(*half, 16)
    record(f"d_odd_columns_full_range|{law}", {"max_odd_over_l0_L16": odd,
           "half_range_l1_over_l0_I_max": float(np.max(np.abs(hI[:, 1] / hI[:, 0])))})
    assert odd < 1e-12


# ------------------------------------------------------------------ pointwise partner
POINTWISE = {
    "Delta(0)": ([0.0], [1.0]),
    "Delta(0.3)": ([0.3], [1.0]),
    "Delta(0.999)": ([0.999], [1.0]),
    "samples_asym": ([0.1, 0.3, 0.45], [0.2, 0.5, 0.3]),
    "samples_symmetric": ([-0.3, 0.3], [0.5, 0.5]),
}


@pytest.mark.parametrize("L", LS)
@pytest.mark.parametrize("name", list(POINTWISE))
def test_pointwise_partner(name, L):
    """eta pointwise (Delta or samples), mu on Legendre(L): the mask is exact only for an even point set."""
    eta, q = (np.array(v, float) for v in POINTWISE[name])
    dens, mom = pitch("poly_odd", L)
    ref = beaming_sum(eta, q, dens)
    out = tagged({"no_mask": rel(legendre_predict(eta, q, mom, L), ref),
                  "mask_odd": rel(legendre_predict(eta, q, mom, L, mask_odd=True), ref)})
    record(f"d_pointwise_partner|{name}|L{L}", out)
    assert out["no_mask"][0] < 1e-12
    if name in ("Delta(0)", "samples_symmetric"):
        assert out["mask_odd"][0] < 1e-12
    else:
        assert out["mask_odd"][0] > 1e-3


# ------------------------------------------------------------------ mu averaged in the beaming limit
@pytest.mark.parametrize("pname", ["even_poly", "poly_odd", "exp_k2"])
@pytest.mark.parametrize("law", sorted(EVEN))
def test_averaged_mu_half_range_needs_even_pitch_density(law, pname):
    """mu averaged (pitch law in the basis), eta averaged on the half range: check (c) of Sec. 4.3."""
    dens = pitch(pname, 3)[0]
    half = half_rule(LAWS[law])
    full = BM.mirror(*half)
    base = beaming_sum(*full, dens)
    out = tagged({"half_vs_full": rel(beaming_sum(*half, dens), base),
                  "full_vs_direct": rel(base, beaming_sum(*reference_rule(LAWS[law]), dens))})
    record(f"d_averaged_mu|{law}|{pname}", out)
    if pname == "even_poly":
        assert out["half_vs_full"][0] < 1e-12
    else:
        assert out["half_vs_full"][0] > 1e-3


@pytest.mark.parametrize("L", (3, 16))
@pytest.mark.parametrize("pname", ["even_poly", "poly_odd"])
def test_averaged_mu_eta_legendre_mask(pname, L):
    """mu averaged with pitch density p, eta on Legendre(L): column k = int (2k+1)/2 P_k(x) 2 p(x) K(x) dx.

    The odd-k mask on eta is exact only when p is even (check (c) applies to an averaged pitch).
    The eta population n(eta) = (1 + 0.5 eta + 0.2 P_2(eta) + 0.1 P_3(eta))/2 has odd moments.
    """
    dens = pitch(pname, 3)[0]
    x, wx = reference_rule([(-1.0, 1.0)])
    I, P = kernel(x)
    m = np.zeros(L + 1)
    m[:4] = [1.0, 0.5 / 3, 0.2 / 5, 0.1 / 7]
    n_eta = 0.5 * (1 + 0.5 * x + 0.2 * 0.5 * (3 * x**2 - 1) + 0.1 * 0.5 * (5 * x**3 - 3 * x))
    ref_w = wx * 2.0 * n_eta * 2.0 * dens(x)       # wx are probabilities of U[-1, 1]: dx = 2 wx
    ref = CS.stokes(I @ ref_w, P @ ref_w)
    ell = np.arange(L + 1)
    W = (2.0 * wx * 2.0 * dens(x))[:, None] * BM.legendre_matrix(x, L) * (2 * ell + 1) / 2.0
    full = CS.stokes(I @ W @ m, P @ W @ m)
    Wm = W * (ell % 2 == 0)
    masked = CS.stokes(I @ Wm @ m, P @ Wm @ m)
    out = tagged({"all_k": rel(full, ref), "mask_odd_k": rel(masked, ref)})
    record(f"d_averaged_mu_eta_legendre|{pname}|L{L}", out)
    assert out["all_k"][0] < 1e-12
    if pname == "even_poly":
        assert out["mask_odd_k"][0] < 1e-12
    else:
        assert out["mask_odd_k"][0] > 1e-3
