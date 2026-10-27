"""LaTeX (booktabs) table snippets for the classical-model section.

Every number is read from the loaded results JSONs (or from figure_data's pair-invariant
summaries of fits_products.npz); only prose and formatting are written here.
"""
from __future__ import annotations

import math

import numpy as np

ENS_ORDER = ("E1", "E2", "E3", "E4-JP", "E4-KP", "E5", "E1M")


# ----------------------------------------------------------------------------- formatting
def num(x: float, sig: int = 3) -> str:
    """Significant-figure formatting; powers of ten for very large or small magnitudes."""
    x = float(x)
    if x == 0 or not math.isfinite(x):
        return "0" if x == 0 else "--"
    e = math.floor(math.log10(abs(x)))
    if e >= 5 or e <= -4:
        m = round(x / 10**e, sig - 1)
        if abs(m) >= 10:
            m, e = m / 10, e + 1
        return rf"{m:.{sig - 1}f}\times10^{{{e}}}"
    dec = max(0, sig - 1 - e)
    return f"{round(x, dec):.{dec}f}"


def decimals_for(sd: float, sig: int = 2) -> int:
    return max(0, sig - 1 - math.floor(math.log10(abs(sd))))


def pm(x: float, sd: float) -> str:
    d = decimals_for(sd)
    return rf"{x:.{d}f}\pm{sd:.{d}f}"


def signed(x: float, dec: int = 2) -> str:
    """Explicit sign, and plain zero when the value rounds to zero."""
    return f"{0.0:.{dec}f}" if round(x, dec) == 0 else f"{x:+.{dec}f}"


def m(s: str) -> str:
    return f"${s}$"


def sci(x: float, sig: int = 2) -> str:
    """Always a x 10^b (error columns)."""
    x = float(x)
    e = math.floor(math.log10(abs(x)))
    mant = round(x / 10**e, sig - 1)
    if abs(mant) >= 10:
        mant, e = mant / 10, e + 1
    return rf"{mant:.{sig - 1}f}\times10^{{{e}}}"


def table(caption: str, label: str, colspec: str, header: list[str], rows: list[str], notes: str = "",
          tabcolsep: str = "", tabularx: bool = False) -> str:
    body = "\n".join(rows)
    env, width = ("tabularx", "{\\textwidth}") if tabularx else ("tabular", "")
    sep = f"\\setlength{{\\tabcolsep}}{{{tabcolsep}}}\n" if tabcolsep else ""
    out = (f"\\begin{{table}}[tbp]\n\\centering\n\\small\n{sep}\\caption{{{caption}}}\n\\label{{{label}}}\n"
           f"\\begin{{{env}}}{width}{{@{{}}{colspec}@{{}}}}\n\\toprule\n{' & '.join(header)} \\\\\n\\midrule\n"
           f"{body}\n\\bottomrule\n\\end{{{env}}}\n")
    if notes:
        out += f"\\par\\smallskip\\parbox{{\\linewidth}}{{\\footnotesize {notes}}}\n"
    return out + "\\end{table}\n"


# ----------------------------------------------------------------------------- tables
def ensembles_table(data: dict) -> str:
    fwd = data["forward"]
    ens, cfg, ag = fwd["ensembles"], fwd["config"], fwd["ageing"]
    p1, p2, p3, kp, e5 = (ens[k]["params"] for k in ("E1", "E2", "E3", "E4-KP", "E5"))
    fA = ens["E5"]["fractions_A_B_crossing_channel"][0]
    dphi = math.degrees(e5["phi_b"] - e5["phi_a"])
    sup = rf"[{num(p1['lo'])},\,10^{{{round(math.log10(p1['hi']))}}}]"
    rows = [
        rf"E1 & power law & $p={num(p1['p'], 2)}$ on ${sup}$ & analytic limit, edge, $\Pi$ & Rybicki \& Lightman (1979) \\",
        rf"E2 & smoothly broken power law & $p_1={num(p2['p1'], 2)}$, $p_2={num(p2['p2'], 2)}$, $s={num(p2['s'], 1)}$, "
        rf"$\gamma_b={num(p2['gamma_br'], 2)}$ & break in band; parameters; $B$--$\gamma$ ridge & Strong et al. (2011); Orlando (2018) \\",
        rf"E3 & index mixture & $\bar p={num(p3['pbar'], 2)}$, $\sigma_p={num(p3['sigma_p'], 1)}$, "
        rf"$\gamma_0=10^{{{round(math.log10(p3['gamma_0']))}}}$ & curvature from mixing & Chluba et al. (2017) \\",
        rf"E4-JP & ageing, isotropised pitch & $p={num(kp['p'], 2)}$, $t={num(ag['t_Myr'], 2)}$\,Myr, "
        rf"$\gamma_c={num(ag['gamma_c_JP'], 3)}$ & steepening in band & Jaffe \& Perola (1973) \\",
        rf"E4-KP & ageing, fixed pitch & same injection and $t$; $\gamma_c(\mu)={num(kp['gamma_c_KP(mu=0)'], 3)}$--"
        rf"${num(kp['gamma_c_KP(mu=1)'], 3)}$ & energy--pitch correlation & Kardashev (1962); Pacholczyk (1970) \\",
        rf"E5 & two power-law zones & $p_A={num(e5['p_a'], 2)}$, $p_B={num(e5['p_b'], 2)}$, "
        rf"$\Delta\phi={dphi:.0f}^\circ$, $f_A={num(fA, 3)}$ & energy--azimuth correlation & -- \\",
        rf"E1M & E1, Maxwellian $|\boldsymbol B|$ & $B_{{\rm rms}}={num(ens['E1M']['params']['maxwell_config']['b_rms'], 1)}\,\mu$G "
        rf"& misspecified field & -- \\",
    ]
    caption = (
        "Reference ensembles of the classical-model tests (unit electron column). Unless stated, "
        rf"$B\sim U[{num(cfg['b_lo'], 1)},{num(cfg['b_hi'], 1)}]\,\mu$G, $\eta\sim U[-{num(cfg['eta_max'], 1)},{num(cfg['eta_max'], 1)}]$, "
        rf"isotropic pitch angles, azimuth factor $\langle e^{{2i\phi}}\rangle={num(cfg['azimuth_amp'] / 2, 1)}"
        rf"e^{{{num(2 * cfg['azimuth_phase'], 1)}i}}$ and a Gaussian screen of mean ${num(cfg['screen_mean'], 1)}$ and "
        rf"standard deviation ${num(cfg['screen_sd'], 1)}\,\mathrm{{rad\,m^{{-2}}}}$. E4 losses: synchrotron at "
        rf"$\langle B^2\rangle={num(cfg['mean_b2_muG2'], 3)}\,\mu\mathrm G^2$ and Thomson inverse Compton on "
        rf"${num(kp['u_rad_eV_cm3'], 1)}\,\mathrm{{eV\,cm^{{-3}}}}$; the JP break frequency is "
        rf"${num(ag['nu_c_JP=a1<B_perp>gamma_c^2_Hz'] / 1e9, 3)}\,$GHz. E5 zones cross in intensity at "
        rf"${num(e5['nu_cross_Hz'] / 1e9, 1)}\,$GHz; $f_A$ is the zone-A electron fraction. E1M shares E1's electron "
        "spectrum and is fitted with the uniform-field basis.")
    rl = r">{\raggedright\arraybackslash}"
    colspec = f"l{rl}p{{2.3cm}}{rl}X{rl}p{{3.0cm}}{rl}p{{2.9cm}}"
    return table(caption, "tab: classical ensembles", colspec,
                 ["Name", "Family", "Parameters", "Tests", "Motivation"], rows, tabularx=True)


def fit_summary_table(data: dict, modes: dict) -> str:
    fits, ver = data["fits"], data["verify_fits"]["ensembles"]
    kp = data["kp_pseudo"]["fit_summary"]
    kp_ps, kp_spec = kp["pseudo_truth_JP_t_eff"], kp["spec_truth_effective_diagonal"]
    S = fits["summary"]
    rows = []
    for k in ENS_ORDER:
        s = S[k]
        cov = fits["ensembles"][k]["realisations"]["coverage_fixed_R"]
        cov_fixed, cov_post = cov["joint_ellipsoid_analytic"], s["coverage_joint_post"]
        if k == "E4-KP":  # against the JP(t_eff) pseudo-truth (kp_pseudotruth.py)
            cov_fixed = kp_ps["coverage_fixed_R"]["joint_ellipsoid_analytic"]
            cov_post = kp_ps["coverage_post_selection_joint"]
        e_held = ver["E5_joint" if k == "E5" else k]["E_chi2_held_pred_including_mean_residual"]
        name = {"E5": "E5, joint $(a,M_2)$", "E4-KP": r"E4-KP$^\dagger$"}.get(k, k)
        rows.append(
            f"{name} & {s['R_star']} & {num(s['chi2_fit'], 3)}/{s['dof_fit']} & "
            f"{num(s['chi2_held_predictive'], 3)} ({num(e_held, 3)}) & {num(100 * s['max_err_over_I'], 2)} & "
            f"{num(modes[k]['max_abs_bias_invariant'], 3 if modes[k]['max_abs_bias_invariant'] >= 10 else 2)} & "
            f"{cov_fixed:.3f} & "
            f"{cov_post:.3f} \\\\")
    sf = fits["E5"]["separable_free_factor"]["seeded"]
    rows.append(
        f"E5, one free $c$ & {sf['R_star']}+2 & {num(sf['chi2_fit'], 4)}/{sf['dof_fit']} & "
        f"{num(sf['chi2_held_predictive_fixed_c'], 4)} & {num(100 * sf['max_fit_minus_direct_over_I']['value'], 2)} "
        "& -- & -- & -- \\\\")
    st = fits["settings"]
    n_held = st["n_held_data"]
    pt = data["kp_pseudo"]["pseudo_truth"]
    fwd = data["kp_pseudo"]["forward"]
    caption = (
        rf"Seeded UPRE fits with the $N={st['N']}$ basis ({st['n_fit_data']} fitted and {n_held} withheld data). "
        r"$\chi^2_{\rm held}$ is the predictive form $r^{\rm T}(I+H_RH_R^{\rm T})^{-1}r$; in brackets its "
        rf"expectation at fixed $R^*$, ${n_held}+\mu^{{\rm T}}(I+H_RH_R^{{\rm T}})^{{-1}}\mu$, where $\mu$ is the "
        "mean withheld residual left by truncation and representation error. The maximum error is "
        r"$\max|{\rm fit}-{\rm direct}|/I_j$ over all channels and $I,Q,U$. $|K\varepsilon|/\sigma$ is the "
        "largest representation bias of a retained combination; for E5 each exactly degenerate singular pair "
        "enters through its two-dimensional norm. Coverage refers to the joint 68.3\\% ellipsoid of the retained "
        rf"combinations: analytic at fixed $R^*$ and Monte Carlo with $R^*$ reselected in each of "
        rf"{st['n_realisations']} realisations. Truths are the moments of the generating spectrum; E1M uses the E1 "
        r"moments. $^\dagger$E4-KP is compared with the moments of the JP spectrum with "
        rf"$t_{{\rm eff}}/t={num(pt['t_eff_over_t'], 5)}$, which reproduces the KP spectrum to "
        rf"${sci(fwd['max_abs_S_JP_minus_S_KP_over_I_IQU'][0])}\,I_j$; against the effective diagonal moments of "
        rf"$N_{{\rm eff}}$ the bias is ${num(kp_spec['max_abs_rep_bias_over_sigma'], 3)}\sigma$ and the coverage "
        rf"{kp_spec['coverage_fixed_R']['joint_ellipsoid_analytic']:.3f} (fixed $R^*$) and "
        rf"{kp_spec['coverage_post_selection_joint']:.3f} (reselected).")
    header = [r"Ensemble", r"$R^*$", r"$\chi^2_{\rm fit}$/dof", r"$\chi^2_{\rm held}$ (exp.)",
              r"max err.\ [\%]", r"$|K\varepsilon|/\sigma$", r"cov.\ $R^*$", r"cov.\ post"]
    return table(caption, "tab: classical fit summary", "lccccccc", header, rows, tabcolsep="4pt")


PARAM_TEX = {"lnA": r"$\ln A$", "p": r"$p$", "p1": r"$p_1$", "p2": r"$p_2$", "ln_gamma_b": r"$\ln\gamma_b$",
             "pbar": r"$\bar p$", "kappa=sigma_p^2": r"$\sigma_p^2$", "ln_nu_b": r"$\ln(\nu_b/{\rm Hz})$"}
TASKS = (("E1", "E1"), ("E2_s2", "E2"), ("E3", "E3"), ("E4-JP", "E4-JP"),
         ("E1M_U46basis", r"E1M$^\dagger$"), ("E4-KP_JPfamily", r"E4-KP$^\ddagger$"))


def parameter_table(data: dict) -> str:
    tasks = data["params"]["tasks"]
    rows = []
    for key, label in TASKS:
        t = tasks[key]
        ref = np.asarray(t["reference_point"]["phi"])
        comp, full = t["seeded_comp"], t["seeded_full"]
        ratio = t["fisher_sd_comp_over_full_at_R_seed"]
        bias = t["noisefree"]["comp"]["shift_over_fisher_sd"]
        for i, name in enumerate(t["coords"]):
            d = decimals_for(comp["sd"][i])
            head = f"{label} ({t['R_star_seeded']})" if i == 0 else ""
            rows.append(f"{head} & {PARAM_TEX[name]} & {m(f'{ref[i]:.{d}f}')} & {m(pm(comp['phi'][i], comp['sd'][i]))} & "
                        f"{m(pm(full['phi'][i], full['sd'][i]))} & {ratio[i]:.4f} & {m(signed(bias[i]))} \\\\")
        if key != TASKS[-1][0]:
            rows.append(r"\addlinespace")
    kind = tasks["E4-KP_JPfamily"]["reference_point"]["kind"]
    caption = (
        r"Physical parameters from the retained combinations (compressed) and from the full linear likelihood, "
        r"seeded data. The number after each task is $R^*$. $\sigma_{\rm c}/\sigma_{\rm f}$ is the Gauss--Newton "
        r"ratio at $R^*$; the bias is the shift of a noise-free compressed fit in units of $\sigma$, the cost of the "
        r"$N=2$ expansion. Fit coordinates: $A$ is the amplitude at $\gamma=10^4$ and $\nu_b=a_1\langle B_\perp\rangle"
        rf"\gamma_c^2$; E2 is fitted with $s={num(data['forward']['ensembles']['E2']['params']['s'], 1)}$ fixed. "
        r"$^\dagger$Maxwellian-field data fitted with the uniform-field basis; reference is the E1 truth. "
        rf"$^\ddagger$KP data fitted with the JP family; reference is the {kind}.")
    header = ["Task ($R^*$)", "Parameter", "Truth/ref.", "Compressed", "Full",
              r"$\sigma_{\rm c}/\sigma_{\rm f}$", r"Bias [$\sigma$]"]
    return table(caption, "tab: classical parameters", "llccccc", header, rows, tabcolsep="4pt")


BASE_ROWS = (("s1", "PySM s1"), ("s3", "PySM s3"), ("bmom3", r"$\beta$-mom., $\omega_1=0$"),
             ("logcubic", "log-cubic"), ("bmom4", r"$\beta$-mom., $\omega_1$ free"),
             ("ePL", "electron PL"), ("eLP", "electron log-par."), ("eSBPL", "electron broken PL"))
BASE_COLS = ("E1", "E2", "E3", "E4-JP", "E4-KP", "E5", "GALPROP")


def baselines_table(data: dict) -> str:
    bl = data["baselines"]
    res, models = bl["results"], {**bl["models"]["laws"], **bl["models"]["families"]}

    def mc_r(col):
        return int(res[col]["monte_carlo"]["summary"]["moment_UPRE"]["R_star"]["median"])

    def nf_moment(col):
        if col == "GALPROP":
            return res[col]["noise_free_moment_at_MC_median_Rstar"]["rms_res_held"]
        return res[col]["noise_free_tsvd_curve"][mc_r(col) - 1]["rms_res_held"]

    def cells(fn):
        return " & ".join(num(100 * fn(c), 2) for c in BASE_COLS)

    rows = [r"\multicolumn{9}{@{}l}{\emph{Noise-free data: withheld rms residual [\%]}} \\",
            r"moments, $R=\tilde R^*$ & $\tilde R^*$ & " + cells(nf_moment) + r" \\",
            r"moments, $R=4$ & 4 & " + cells(lambda c: res[c]["noise_free"]["moment_R4"]["rms_res_held"]) + r" \\"]
    for key, label in BASE_ROWS:
        rows.append(f"{label} & {models[key]} & " + cells(lambda c, k=key: res[c]["noise_free"][k]["rms_res_held"]) + r" \\")
    n_real = bl["noise"]["n_real"]
    rows += [r"\addlinespace", rf"\multicolumn{{9}}{{@{{}}l}}{{\emph{{1\% noise, median (95th percentile) of "
                                 rf"{n_real} realisations [\%]}}}} \\"]
    summ = lambda c, k: res[c]["monte_carlo"]["summary"][k]["rms_res_held"]  # noqa: E731

    def mp(c, k):
        v = summ(c, k)
        return f"{num(100 * v['median'], 2)}\\,({num(100 * v['p95'], 2)})"

    def cells_mp(k):
        return " & ".join(mp(c, k) for c in BASE_COLS)
    cap = res["E1"]["monte_carlo"]["summary"]["moment_UPRE"]["r_cap"]
    rows.append(r"moments, UPRE & $R^*$ & " + cells_mp("moment_UPRE") + r" \\")
    rows.append(r"moments, $R=4$ & 4 & " + cells_mp("moment_R4") + r" \\")
    for key, label in BASE_ROWS:
        rows.append(f"{label} & {models[key]} & " + cells_mp(key) + r" \\")
    rows.append(r"\addlinespace")
    rows.append(r"$\tilde R^*$ (median) & & " + " & ".join(str(mc_r(c)) for c in BASE_COLS) + r" \\")

    def tail(c):
        return res[c]["monte_carlo"]["summary"]["moment_UPRE"]["R_star_uncapped_frac_ge_15"]
    rows.append(r"$R^*\geq15$, no limit & & " + " & ".join(num(tail(c), 2) if tail(c) else "0"
                                                                for c in BASE_COLS) + r" \\")
    nh_syn = res["E1"]["monte_carlo"]["n_held"]
    nh_gal = res["GALPROP"]["monte_carlo"]["n_held"]
    syn = [c for c in BASE_COLS if c != "GALPROP"]
    p99c = [100 * res[c]["monte_carlo"]["summary"]["moment_UPRE"]["rms_res_held"]["p99"] for c in syn]
    p99u = [100 * res[c]["monte_carlo"]["summary"]["moment_UPRE_uncapped"]["rms_res_held"]["p99"] for c in syn]
    ab = bl["absorption_caveat"]
    att = ab["slab_attenuation_at_lowest_node_min_max"]
    ref = ab["refit_withheld"]
    dmax = max(abs(ref["de_absorbed_central_slab"][k][0] - ref["published"][k][0]) for k in ref["published"])
    caption = (
        r"Intensity-only baselines: rms of $({\rm fit}-{\rm direct})/I_j$ over the withheld channels "
        rf"({nh_syn} for E1--E5; {nh_gal} native nodes for the \textsc{{GALPROP}} $|b|\geq60^\circ$ spectrum). "
        r"The second column gives the number of fitted parameters. The moment basis is the TSVD of the $N=2$ "
        r"intensity columns; $\tilde R^*$ is its median UPRE rank, and $R=4$ matches the largest law. "
        rf"The UPRE search is limited to $R\leq{cap}$, the range of the Picard scan. Without the limit, $R^*\geq15$ "
        rf"occurs in the fraction of realisations given in the last row, and the 99th percentile of the moment-basis "
        rf"rms on E1--E5 is {num(min(p99u), 2)}--{num(max(p99u), 2)}\% instead of "
        rf"{num(min(p99c), 2)}--{num(max(p99c), 2)}\%. "
        r"\textsc{GALPROP} has no noise; its weights use $\sigma=1\%$ of the published value. "
        rf"The published \textsc{{GALPROP}} spectrum turns over near {num(ab['turnover']['peak_node_GHz'] * 1e3, 2)}\,MHz; "
        rf"a mixed-slab fit with $\tau\propto\nu^{{-2.1}}$ to this turnover gives "
        rf"{num(100 * att[0], 2)}--{num(100 * att[1], 2)}\% attenuation at "
        rf"{num(ab['lowest_node_GHz'], 4)}\,GHz, and dividing it out changes the noise-free withheld rms by at most "
        rf"{num(dmax, 1)} percentage points. "
        r"The $\beta$-moment row with $\omega_1$ free is a generic 4-parameter family: with $\bar\beta$ also free, "
        r"$\bar\beta$ and $\omega_1$ are degenerate.")
    header = ["Model", "$k$"] + [r"\textsc{GALPROP}" if c == "GALPROP" else c for c in BASE_COLS]
    return table(caption, "tab: classical baselines", "lc" + "c" * len(BASE_COLS), header, rows, tabcolsep="2.5pt")


def _t(seconds: float) -> str:
    if seconds >= 0.1:
        return f"{num(seconds, 3)}\\,s"
    if seconds >= 1e-3:
        return f"{num(seconds * 1e3, 3)}\\,ms"
    return f"{num(seconds * 1e6, 3)}\\,$\\mu$s"


def cost_table(data: dict) -> str:
    c = data["cost"]
    rows = []
    for case in ("E2", "E4-JP"):
        o = c["one_d"][case]
        ri, rii, riii = o["route_i_recompute"], o["route_ii_table_288"], o["route_iii_moment_basis_36"]
        rows += [
            rf"{case} & (i) recompute & {ri['kernel_nodes_per_eval']} nodes/eval & -- & {_t(ri['median_s'])} & -- \\",
            rf" & (ii) $\gamma$ table & {rii['columns']} & {_t(rii['build_s'])} & {_t(rii['median_s'])} & "
            rf"${sci(rii['max_rel_err_vs_route_i'])}$ \\",
            rf" & (iii) moment basis & {riii['columns']} & {_t(c['build']['continuum_analytic_basis']['median_s'])} & "
            rf"{_t(riii['median_s'])} & ${sci(riii['max_rel_err_vs_route_i'])}$ \\",
            r"\addlinespace"]
    j = c["joint_gamma_B"]
    n36 = data["fits"]["ensembles"]["E1"]["seeded"]["n_columns"]
    rows += [
        rf"joint $(\gamma,B)$ & (i) recompute & {j['route_i_recompute']['kernel_nodes_per_eval']} nodes/eval & -- & "
        rf"{_t(j['route_i_recompute']['median_s'])} & -- \\",
        rf" & (ii) $\gamma\times B$ table & {j['route_ii_table']['columns']} & {_t(j['table_build_s'])} & "
        rf"{_t(j['route_ii_table']['median_s'])} & ${sci(j['route_ii_table']['max_rel_err_vs_route_i'])}$ \\",
        rf" & (iii) joint moments & {j['route_iii_joint_moment_basis_72']['columns']} & {_t(j['joint_basis_build_s'])} & "
        rf"{_t(j['route_iii_joint_moment_basis_72']['median_s'])} & "
        rf"${sci(j['route_iii_joint_moment_basis_72']['max_rel_err_vs_route_i'])}$ \\",
        rf" & $B$-avg., marginal moments & {n36} & -- & -- & ${sci(j['preaveraged_36_with_marginal_moments_max_rel_err'])}$ \\",
        rf" & push-forward to $B_0$ & {n36} & -- & -- & ${sci(j['pushforward_36_fixed_B0_max_rel_err'])}$ \\"]
    b = c["build"]
    h = b["harmonic_stencil_basis_read_from_survey"]
    per = h["per_call_s_adopted_settings"]
    mach = c["machine"]
    caption = (
        rf"Cost of one parametric evaluation of the {data['fits']['settings']['n_fit_data'] + data['fits']['settings']['n_held_data']} "
        r"data rows with the same continuum kernel (median wall-clock, one thread, "
        rf"{mach['cpu']}, numpy {mach['numpy']}). Errors are the largest relative difference from route (i). "
        r"Route (ii) for E2 reuses the nodes of route (i), so its error is zero by construction. Building the "
        rf"$N=2$ continuum basis takes {b['continuum_analytic_basis']['kbar_calls']} averaged-kernel calls with "
        rf"analytic derivatives ({_t(b['continuum_analytic_basis']['median_s'])}) or "
        rf"{b['continuum_stencil_basis']['kbar_calls']} with the 5-point stencil "
        rf"({_t(b['continuum_stencil_basis']['median_s'])}); the harmonic stencil basis needs {h['kbar_calls']} calls "
        rf"of {num(min(per.values()), 2)}--{num(max(per.values()), 2)}\,s each (timings of the harmonic pipeline, not re-run). "
        rf"In the joint case the {j['route_iii_joint_moment_basis_72']['columns']} joint columns need "
        rf"{j['route_iii_joint_moment_basis_72']['kernel_calls_to_build']} kernel calls because the field scaling "
        r"identity supplies the $B$ derivatives.")
    header = ["Case", "Route", "Columns", "Build", "Per evaluation", "Max rel.\\ error"]
    return table(caption, "tab: classical cost", "llcccc", header, rows, tabcolsep="4pt")


# ----------------------------------------------------------------------------- compact tables
# The layout of Tables 4 and 5 of the paper, to cross-check their entries: the fit summary with an N=2
# forward-error column (Table 4), and the baselines as one block of "noise-free / 1% median" cells
# (Table 5).  Notation follows the paper ($R_\star$).

# forward_results.json summary key of the basis used in each fit (E1M is fitted with the uniform-field basis)
FWD_KEY = {"E1": "E1", "E2": "E2", "E3": "E3", "E4-JP": "E4-JP", "E5": "E5", "E1M": "E1M_U46_basis"}
FWD_KEY_ONE_FACTOR = "E5_separable_paper_factor"
FIT_NAME = {"E4-KP": r"E4-KP$^\dagger$", "E5": "E5, joint moments"}


def _forward_n2(data: dict, key: str) -> float:
    """max over channels and I, Q, U of |S_direct - C a|/I_j at N = 2 (forward.py)."""
    return float(max(data["forward"]["summary_max_resid_over_I_IQU"][key]["N2"]))


def _forward_n2_kp_pseudo(data: dict) -> float:
    """E4-KP direct spectrum against the basis at the JP(t_eff) moments, N = 2 (kp_pseudotruth.py)."""
    return float(max(data["kp_pseudo"]["forward"]["N2"]["max_abs_S_KP_minus_C_a_over_I_IQU"]))


def fit_summary_compact_table(data: dict, modes: dict) -> str:
    fits, kp = data["fits"], data["kp_pseudo"]
    S, st = fits["summary"], fits["settings"]
    rows = []
    for k in ENS_ORDER:
        s = S[k]
        fwd = _forward_n2_kp_pseudo(data) if k == "E4-KP" else _forward_n2(data, FWD_KEY[k])
        b = modes[k]["max_abs_bias_invariant"]
        rows.append(f"{FIT_NAME.get(k, k)} & {s['R_star']} & {num(s['chi2_fit'], 3)}/{s['dof_fit']} & "
                    f"{num(100 * fwd, 2)} & {num(100 * s['max_err_over_I'], 2)} & {num(b, 3 if b >= 10 else 2)} \\\\")
    sf = fits["E5"]["separable_free_factor"]["seeded"]
    rows.append(f"E5, one azimuth factor & {sf['R_star']}+2 & {num(sf['chi2_fit'], 4)}/{sf['dof_fit']} & "
                f"{num(100 * _forward_n2(data, FWD_KEY_ONE_FACTOR), 2)} & "
                f"{num(100 * sf['max_fit_minus_direct_over_I']['value'], 2)} & -- \\\\")
    kp_eta = kp["fit_summary"]["spec_truth_effective_diagonal"]
    caption = (
        r"Forward errors and fits with the $N=2$ basis. Fwd.\ is "
        r"$100\max_{j,X}|S^{\rm direct}_{X,j}-(\boldsymbol C\boldsymbol a)_{X,j}|/I_j$ over the "
        rf"{data['forward']['config']['n_channels']} channels and $X\in\{{I,Q,U\}}$ at the reference moments, with "
        r"the basis of the fit (E1M: uniform-field basis; E5, one azimuth factor: the mean azimuth factor shared by "
        rf"all bins). The fits use one noise realisation ({st['n_fit_data']} fitted and {st['n_held_data']} withheld "
        r"data), with $R_\star$ chosen by UPRE. The maximum error is $\max|{\rm fit}-{\rm direct}|/I_j$ over all "
        r"channels and $I,Q,U$. Bias is the largest $|\boldsymbol V_R^{\mathsf T}\boldsymbol K\boldsymbol\epsilon|$ "
        r"of a retained combination in units of its noise standard deviation; for E5 each exactly degenerate pair "
        r"of singular values enters through its two-dimensional norm. $^\dagger$Relative to the moments of the JP "
        rf"spectrum with $t_{{\rm eff}}/t={num(kp['pseudo_truth']['t_eff_over_t'], 5)}$; relative to the moments of "
        rf"the $\eta$-averaged KP density the forward error is ${num(100 * _forward_n2(data, 'E4-KP'), 2)}$ and the "
        rf"bias ${num(kp_eta['max_abs_rep_bias_over_sigma'], 2)}$.")
    header = ["Ensemble", r"$R_\star$", r"$\chi^2_{\rm fit}$/dof", r"Fwd.\ [\%]", r"Max.\ error [\%]",
              r"Bias [$\sigma$]"]
    return table(caption, "tab: classical fit summary", "lccccc", header, rows, tabcolsep="4pt")


BASE_LABEL_COMPACT = {**dict(BASE_ROWS), "eLP": "electron log-parabola"}


def ceil_sig(x: float, sig: int = 1) -> str:
    """Round |x| up to `sig` significant figures (for 'at most' statements)."""
    e = math.floor(math.log10(abs(x))) - sig + 1
    return num(math.ceil(abs(x) / 10**e) * 10**e, sig)


def _and_list(names: list[str]) -> str:
    return names[0] if len(names) == 1 else ", ".join(names[:-1]) + " and " + names[-1]


def baselines_compact_table(data: dict) -> str:
    bl, fwd_cfg = data["baselines"], data["forward"]["config"]
    res, models = bl["results"], {**bl["models"]["laws"], **bl["models"]["families"]}
    rank = {c: int(res[c]["monte_carlo"]["summary"]["moment_UPRE"]["R_star"]["median"]) for c in BASE_COLS}

    def median(c, k):
        return res[c]["monte_carlo"]["summary"][k]["rms_res_held"]["median"]

    def nf_moment(c):  # noise-free withheld rms at the median UPRE rank (same source as baselines_table)
        if c == "GALPROP":
            return res[c]["noise_free_moment_at_MC_median_Rstar"]["rms_res_held"]
        return res[c]["noise_free_tsvd_curve"][rank[c] - 1]["rms_res_held"]

    def pair(nf, mc):
        return f"{num(100 * nf, 2)}/{num(100 * mc, 2)}"

    r4 = {c: res[c]["noise_free"]["moment_R4"]["rms_res_held"] for c in BASE_COLS}
    rows = [rf"moments, UPRE & {min(rank.values())}--{max(rank.values())} & "
            + " & ".join(pair(nf_moment(c), median(c, "moment_UPRE")) for c in BASE_COLS) + r" \\",
            r"moments, $R=4$ & 4 & " + " & ".join(num(100 * r4[c], 2) for c in BASE_COLS) + r" \\"]
    for key, _ in BASE_ROWS:
        rows.append(f"{BASE_LABEL_COMPACT[key]} & {models[key]} & "
                    + " & ".join(pair(res[c]["noise_free"][key]["rms_res_held"], median(c, key)) for c in BASE_COLS)
                    + r" \\")
    tex_name = {c: r"\textsc{GALPROP}" if c == "GALPROP" else c for c in BASE_COLS}
    by_rank = {r: [tex_name[c] for c in BASE_COLS if rank[c] == r] for r in sorted(set(rank.values()))}
    rank_text = "; ".join(f"{r} for {_and_list(cs)}" for r, cs in by_rank.items())
    d_r4 = max(abs(median(c, "moment_R4") - r4[c]) for c in BASE_COLS)
    nh_syn, nh_gal = res["E1"]["monte_carlo"]["n_held"], res["GALPROP"]["monte_carlo"]["n_held"]
    cap = res["E1"]["monte_carlo"]["summary"]["moment_UPRE"]["r_cap"]
    caption = (
        r"Intensity-only comparison: rms of $({\rm fit}-{\rm direct})/I_j$ over the withheld channels "
        rf"({nh_syn} for E1--E5; {nh_gal} native nodes for the \textsc{{GALPROP}} $|b|\geq60^\circ$ spectrum), "
        rf"without noise and, after the slash, the median over {bl['noise']['n_real']} realisations with $1\%$ "
        r"noise. The second column gives the number of fitted parameters. The moment rows use the truncated "
        r"estimator on the $N=2$ intensity columns. In the UPRE row the noise-free values are at the median UPRE "
        rf"rank ({rank_text}), and the $1\%$-noise values have $R_\star$ reselected in each realisation. The $R=4$ "
        r"row matches the largest law; its $1\%$-noise medians differ from the noise-free values by at most "
        rf"${ceil_sig(100 * d_r4)}$ percentage points. The UPRE search is limited to $R\leq{cap}$ for the "
        rf"{fwd_cfg['n_channels'] - nh_syn} fitted intensities. The \textsc{{GALPROP}} spectrum has no noise; its "
        r"weights use $\sigma=1\%$ of the published value. With $\bar\beta$ also free, $\bar\beta$ and $\omega_1$ "
        r"are nearly degenerate, so the $\beta$-moment row with $\omega_1$ free acts as a generic four-parameter "
        r"family.")
    header = ["Model", "$k$"] + [tex_name[c] for c in BASE_COLS]
    return table(caption, "tab: classical baselines", "lc" + "c" * len(BASE_COLS), header, rows, tabcolsep="2.2pt")


def all_tables(data: dict, modes: dict) -> dict[str, str]:
    return {"ensembles": ensembles_table(data), "fit_summary": fit_summary_table(data, modes),
            "parameter_recovery": parameter_table(data), "baselines": baselines_table(data),
            "cost": cost_table(data), "fit_summary_compact": fit_summary_compact_table(data, modes),
            "baselines_compact": baselines_compact_table(data)}
