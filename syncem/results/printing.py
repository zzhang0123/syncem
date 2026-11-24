"""Present fit diagnostics without materializing the moment or data arrays.

LABEL [extension].

Fit summaries render scalar observations and rank counts over the pixel
shape. Units and assumptions come from the result. The printout has no
certified statistical interpretation beyond the estimator's records.
"""

import numpy as np


def fit_summary(fit):
    """Print the estimator, goodness of fit and the recorded assumptions."""
    lines = [f"{fit.kind} fit", fit.estimator]
    data = getattr(fit, "data", None)
    if data is None:
        data = getattr(getattr(fit, "problem", None), "data", None)
    if data is not None:
        axes = " ".join(getattr(data, "components", ()))
        lines.append(
            f"Observation units  {data.unit}" + (f"    Stokes {axes}" if axes else "")
        )
    ranks = np.asarray(fit.rank)
    if ranks.ndim:
        unique, counts = np.unique(ranks, return_counts=True)
        histogram = ", ".join(
            f"{rank} ({count} sightlines)" for rank, count in zip(unique, counts)
        )
        lines.append(f"Retained ranks  {histogram}")
    else:
        chi2, dof, probability = map(float, (fit.chi2, fit.dof, fit.p_value))
        cut = getattr(fit, "rank_cut", None)
        numerical = (
            ""
            if cut is None or np.ndim(cut.numerical_rank)
            else f" of numerical rank {int(cut.numerical_rank)}"
        )
        lines.append(
            f"Rank {int(ranks)}{numerical}    χ² {chi2:.6g}    dof {dof:.6g}    p {probability:.4g}"
        )
    effective = getattr(fit, "effective_parameters", None)
    if effective is not None and np.ndim(effective) == 0:
        lines.append(f"Effective parameters  {float(effective):.6g}")
    if fit.withheld is not None and np.ndim(fit.withheld.chi2) == 0:
        lines.append(
            f"Withheld χ²  {float(fit.withheld.chi2):.6g} on {int(fit.withheld.n)} rows"
        )
    plan = fit.plan
    layout = getattr(plan, "layout", None)
    if layout is not None:
        for i, component in enumerate(layout.components, 1):
            if len(layout.components) > 1:
                lines.append(f"Population component {i}")
            lines.extend(
                f"{name}  {role.role}, {role.how}"
                + (" [default]" if role.default else "")
                for name, role in component.roles.items()
            )
            lines.extend(
                f"{name}  {setting} [default]" for name, setting in component.defaults
            )
    lines.extend(getattr(plan, "notes", ()))
    if fit.ledger.unbounded():
        lines.append("Unbounded terms  " + ", ".join(fit.ledger.unbounded()))
    return "\n".join(lines)
