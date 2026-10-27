"""Print the KP-bridge tables from kp_bridge_results.json (no new numbers).

Run: python kp_bridge_tables.py
"""
from __future__ import annotations

import json
from pathlib import Path

R = json.loads((Path(__file__).resolve().parent / "kp_bridge_results.json").read_text())
P = R["B_points"]


def pick(gamma, x, label, pdf, bessel="olver"):
    for p in P:
        if (p["gamma"] == gamma and p["x"] == x and p["label"] == label and p["pdf"][0] == pdf
                and p["bessel"] == bessel):
            return p
    return None


def table_far():
    print("\n# gamma^2 d_rel_I at fixed eta (KP edge mu_c=0.25 | iso | kp_smooth)")
    for x in (0.1, 1.0, 3.0):
        for lab in ("eta=0.5", "eta=0.35"):
            row = []
            for g in (20, 40, 80, 160, 320, 1000):
                kp, iso = pick(g, x, lab, "kp_edge"), pick(g, x, lab, "iso")
                sm = pick(g, x, lab, "kp_smooth")
                row.append(f"{g}: {kp['g2_d_rel_I']:+.3f}|{iso['g2_d_rel_I']:+.3f}"
                           + (f"|{sm['g2_d_rel_I']:+.3f}" if sm else ""))
            print(f"x={x:g} {lab}: " + "  ".join(row))


def table_edge(S="I"):
    print(f"\n# near-edge d_rel_{S} (k>0) and e_harm (k<=0) vs gamma, with prediction")
    comp = R["D_profile"]["harm_vs_pred"]
    for x in (0.1, 1.0, 3.0):
        for k in (-3.0, -1.5, -0.5, 0.0, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0):
            cells, pred = [], None
            for g in (20, 40, 80, 160, 320, 1000):
                c = [r for r in comp if r["S"] == S and r["gamma"] == g and r["x"] == x
                     and abs(r["k"] - k) < 1e-9 and r["bessel"] == "olver"]
                if not c:
                    continue
                c = c[0]
                if k > 0:
                    cells.append(f"{c['d_harm']:+.3e}")
                    pred = c["d_pred"]
                else:
                    cells.append(f"{c['e_harm']:.4e}")
                    pred = c["e_pred"]
            print(f"x={x:g} k={k:+g}: " + " ".join(cells) + f"  pred {pred:+.3e}")


def table_resid():
    print("\n# gamma x (d_harm - d_pred), I, k>0")
    comp = R["D_profile"]["harm_vs_pred"]
    for x in (0.1, 1.0, 3.0):
        for k in (0.5, 1.0, 2.0, 4.0):
            cells = []
            for g in (20, 40, 80, 160, 320, 1000):
                c = [r for r in comp if r["S"] == "I" and r["gamma"] == g and r["x"] == x
                     and abs(r["k"] - k) < 1e-9 and r["bessel"] == "olver"]
                if c:
                    cells.append(f"{c[0]['gamma_x_resid']:+.3f}")
            print(f"x={x:g} k={k:g}: " + " ".join(cells))


def table_iso_equiv():
    print("\n# isotropic-equivalent error d_iso_I at k=1 and k=0, times gamma^0.6")
    for x in (0.1, 1.0, 3.0):
        for lab in ("k=0", "k=1", "k=2"):
            cells = []
            for g in (20, 40, 80, 160, 320, 1000):
                p = pick(g, x, lab, "kp_edge")
                cells.append(f"{p['d_iso_I'] * g**0.6:+.4f}")
            print(f"x={x:g} {lab}: " + " ".join(cells))


def misc():
    A = R["A_checks"]
    print("\n# checks")
    print("olver vs mpmath:", [(r["m"], r["z"], f"{r['relerr_J']:.1e}", f"{r['relerr_Jp']:.1e}")
                              for r in A["olver_vs_mpmath"]])
    print("b0c0:", A["b0c0_fit"])
    print("regression:", A["regression_check_beaming"])
    print("window:", A["window"])
    print("scipy vs olver:", A["scipy_vs_olver_max_g2_abs_diff_d_iso_I"])
    spreads = [(p["gamma"], p["x"], p["label"], p.get("phase_spread_I")) for p in P
               if p["pdf"][0] == "kp_edge" and p.get("phase_spread_I") is not None
               and p["n_phases"] > 1]
    for g in (20, 40, 80, 160, 320):
        s = [v for gg, _, _, v in spreads if gg == g]
        print(f"max phase spread d_rel_I gamma={g}: {max(s):.2e}")
    print("\n# C smooth fits")
    for k, v in R["C_smooth_fit"]["fits"].items():
        print(k, "c0,c1,c2 =", [round(c, 3) for c in v["c0_c1_c2"]], "resid",
              [round(r, 3) for r in v["fit_resid"]], "c2 profile", round(v["c2_profile_pred"], 3))
        print("    c1 profile part", round(v["c1_profile_part"], 3), "KP:",
              {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v["kp"].items()})
    print("\n# D thresholds k(|f|<eps)", R["D_profile"]["k_threshold_I"])
    for key, v in R["D_profile"]["prediction"].items():
        print(key, "u2=", round(v["u2"], 4), [(k, None if f is None else round(f, 5))
                                              for k, f in zip(v["k"], v["f_pred"])])
    print("\n# E eta-integrated")
    for r in R["E_eta_integrated"]:
        print(r["gamma"], r["x_at_eta_half"], r["pdf"], r["q"], r["panel_width_gamma"],
              f"g2relI={r['g2_rel_I']:+.4f} g2relQ={r['g2_rel_Q']:+.4f} L1={r['L1_rel_I_node_quadrature']:.3e}")
    print("\n# F gamma-integrated")
    for r in R["F_gamma_integrated"]:
        print(r["Gamma"], r["x_at_gamma_c_eta"], r["kind"], r["panel_width"],
              f"G2relI={r['G2_rel_I']:+.4f} G2relQ={r['G2_rel_Q']:+.4f}")
    print("\n# G cubic Airy")
    for r in R["G_cubic_airy"]["lines"]:
        print(r["dk"], [f"{v:.2e}" for v in r["stable_over_olver_minus1"]],
              [f"{v:.1e}" for v in r["stable_over_mycubic_minus1"]])
    for r in R["G_cubic_airy"]["sums"]:
        print(r["x"], r["label"], r["pdf"], f"g2(cubic/olver-1) I={r['g2_cubic_over_olver_minus1_I']:+.3f}"
              f" Q={r['g2_cubic_over_olver_minus1_Q']:+.3f}  g2 d_olver I={r['g2_olver_d_rel_I']}")
    print("\n# H", json.dumps(R["H_physical"], indent=0)[:1500])
    print("\nwall", R["provenance"]["wall_clock_s"])


if __name__ == "__main__":
    table_far()
    table_edge("I")
    table_edge("Q")
    table_resid()
    table_iso_equiv()
    misc()


def resid_bounds():
    comp = R["D_profile"]["harm_vs_pred"]
    for S in ("I", "Q"):
        rows = [r for r in comp if r["S"] == S and r["bessel"] == "olver" and 0 < r["k"] <= 4.0 + 1e-9]
        worst = max(rows, key=lambda r: abs(r["gamma_x_resid"]))
        w160 = max((r for r in rows if r["gamma"] >= 160), key=lambda r: abs(r["gamma_x_resid"]))
        print(f"max |gamma (d-f)| {S}, 0<k<=4: {worst['gamma_x_resid']:+.3f} at gamma={worst['gamma']} "
              f"x={worst['x']} k={worst['k']:.2f}; gamma>=160: {w160['gamma_x_resid']:+.3f}")


if __name__ == "__main__":
    resid_bounds()
